# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import os
import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.deepspeed_utils import apply_deepspeed_patches
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_mcore_adapter_available, is_ray_available
from ..hparams import get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from ..database import load_supabase_keys, register_trained_model
from .callbacks.qat import get_qat_callback
from .callbacks_module import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import get_ray_trainer, get_swanlab_callback


if is_ray_available():
    import ray
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    # Apply DeepSpeed patches early to handle pin memory issues
    apply_deepspeed_patches()

    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    # Add QAT callback if enabled
    qat_callback = get_qat_callback(model_args)
    if qat_callback is not None:
        callbacks.append(qat_callback)

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    def _is_rank_zero() -> bool:
        if dist.is_available() and dist.is_initialized():
            try:
                return dist.get_rank() == 0
            except Exception:
                return True
        return True

    required_supabase_keys = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    supabase_ready = False
    if _is_rank_zero():
        if all(os.environ.get(key) for key in required_supabase_keys):
            supabase_ready = True
        else:
            supabase_ready = load_supabase_keys()
            supabase_ready = supabase_ready and all(os.environ.get(key) for key in required_supabase_keys)

    training_start = datetime.now(timezone.utc)

    # FSDP runtime status logging removed to avoid premature accelerator initialization

    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        else:  # dpo
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and ray.is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")

    if not _is_rank_zero():
        return

    if not supabase_ready:
        return

    if not training_args.push_to_hub:
        return

    hub_repo_id = getattr(training_args, "hub_model_id", None)
    if not hub_repo_id:
        return

    def _first_str(value: Any) -> Optional[str]:
        if isinstance(value, (list, tuple)):
            return _first_str(value[0] if value else None)
        if isinstance(value, set):
            return _first_str(next(iter(value)) if value else None)
        return str(value) if value is not None else None

    dataset_name = _first_str(getattr(data_args, "dataset", None))
    if not dataset_name:
        dataset_name = _first_str(getattr(data_args, "dataset_dir", None))
    if not dataset_name:
        logger.warning_rank0("Supabase registration skipped: dataset name is unavailable.")
        return

    def _to_jsonable(obj: Any) -> Any:
        try:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
        except Exception:
            pass
        try:
            if hasattr(obj, "to_json_string"):
                return json.loads(obj.to_json_string())
        except Exception:
            pass
        if dataclasses.is_dataclass(obj):
            try:
                return dataclasses.asdict(obj)
            except Exception:
                pass
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)

    training_type = (finetuning_args.stage or "").lower()
    if training_type in {"ppo", "dpo", "kto", "rm"}:
        training_type = "RL"
    else:
        training_type = "SFT"

    created_by = ""
    if "/" in hub_repo_id:
        created_by = hub_repo_id.split("/", 1)[0]
    created_by = created_by or os.environ.get("HF_USERNAME") or os.environ.get("JOB_CREATOR", "")

    agent_name = (
        os.environ.get("TRAINING_AGENT_NAME")
        or os.environ.get("DC_AGENT_NAME")
        or finetuning_args.finetuning_type
        or "llama-factory"
    )

    training_parameters = {
        "model_args": _to_jsonable(model_args),
        "data_args": _to_jsonable(data_args),
        "training_args": _to_jsonable(training_args),
        "finetuning_args": _to_jsonable(finetuning_args),
        "generating_args": _to_jsonable(generating_args),
    }

    wandb_link = None
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            wandb_link = wandb.run.url
    except Exception:
        wandb_link = None

    training_end = datetime.now(timezone.utc)

    record = {
        "agent_name": agent_name,
        "training_start": training_start.isoformat(),
        "training_end": training_end.isoformat(),
        "created_by": created_by,
        "base_model_name": getattr(model_args, "model_name_or_path", None),
        "dataset_name": dataset_name,
        "training_type": training_type,
        "training_parameters": training_parameters,
        "wandb_link": wandb_link,
        "traces_location_s3": os.environ.get("TRACE_S3_PATH"),
        "model_name": hub_repo_id,
    }

    if not record["base_model_name"]:
        logger.warning_rank0("Supabase registration skipped: base_model_name missing.")
        return

    result = register_trained_model(record)
    if result.get("success"):
        logger.info_rank0("Registered trained model metadata with Supabase.")
    else:
        logger.warning_rank0(
            "Supabase registration failed: %s",
            result.get("error", "unknown error"),
        )


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")
