# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial

import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils

from ...extras.packages import is_transformers_version_greater_than


try:
    from ring_flash_attn import zigzag_ring_flash_attn_func

    RING_FLASH_ATTN_AVAILABLE = True
except ImportError:
    RING_FLASH_ATTN_AVAILABLE = False
    zigzag_ring_flash_attn_func = None

# Try native Ulysses implementation first (preferred)
try:
    from .ulysses import UlyssesAttention as NativeUlyssesAttention

    NATIVE_ULYSSES_AVAILABLE = True
except ImportError:
    NATIVE_ULYSSES_AVAILABLE = False
    NativeUlyssesAttention = None

# Fallback to external yunchang package
try:
    from yunchang import UlyssesAttention as YunchangUlyssesAttention
    from yunchang.kernels import AttnType

    YUNCHANG_AVAILABLE = True
except ImportError:
    YUNCHANG_AVAILABLE = False
    YunchangUlyssesAttention = None
    AttnType = None


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="zigzag-ring",
    position_ids=None,
    softmax_scale=None,
    softcap=0.0,
    **kwargs,
):
    if mode == "zigzag-ring":
        if not RING_FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "ring-flash-attn is required for zigzag-ring mode. Please install it with: pip install ring-flash-attn flash-attn"
            )
        attn_output = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, deterministic=deterministic, causal=is_causal, group=group
        )
    elif mode == "ulysses":
        # Use native implementation if available, otherwise fall back to yunchang
        if NATIVE_ULYSSES_AVAILABLE:
            # Import flash attention function for native implementation
            from flash_attn import flash_attn_func

            # Adjust query_length for sequence parallelism
            world_size = dist.get_world_size(group) if group is not None else 1
            adjusted_q_len = q_len * world_size

            dist_attn = NativeUlyssesAttention(
                sequence_process_group=group,
                attn_fn=flash_attn_func
            )
            attn_output = dist_attn(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                query_length=adjusted_q_len,
                dropout_p=dropout,
                causal=is_causal,
                deterministic=deterministic,
                position_ids=position_ids,
                softmax_scale=softmax_scale,
                softcap=softcap
            )
        elif YUNCHANG_AVAILABLE:
            dist_attn = YunchangUlyssesAttention(sequence_process_group=group, attn_type=AttnType.FA)
            attn_output = dist_attn(
                query_states, key_states, value_states, deterministic=deterministic, dropout_p=dropout, causal=is_causal
            )
        else:
            raise ImportError(
                "Neither native Ulysses nor yunchang is available for ulysses mode. "
                "Please ensure the native implementation is properly installed or install yunchang: pip install yunchang"
            )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def apply_sequence_parallel(model_args, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # Check dependencies based on mode
    if model_args.sequence_parallel_mode == "zigzag-ring" and not RING_FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "ring-flash-attn is required for zigzag-ring mode. Please install it with: pip install ring-flash-attn flash-attn"
        )
    elif model_args.sequence_parallel_mode == "ulysses":
        if not NATIVE_ULYSSES_AVAILABLE and not YUNCHANG_AVAILABLE:
            raise ImportError(
                "Either native Ulysses implementation or yunchang is required for ulysses mode. "
                "Please ensure the native implementation is properly installed or install yunchang: pip install yunchang"
            )

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)

    try:
        new_flash_attention_forward = partial(
            new_flash_attn_forward,
            group=group_this,
            mode=model_args.sequence_parallel_mode,
            deterministic=full_determinism,
        )

        # Support for newer transformers versions with AttentionInterface
        if is_transformers_version_greater_than("4.51.0"):
            try:
                from transformers.models.llama import modeling_llama
                from transformers.models.mistral import modeling_mistral
                from transformers.models.phi3 import modeling_phi3
                from transformers.models.qwen2 import modeling_qwen2

                # Register attention interface for different model types
                def register_attention_interface(module):
                    if hasattr(module, "register_attention_implementation"):
                        module.register_attention_implementation(
                            "sequence_parallel_attention",
                            new_flash_attention_forward
                        )

                # Register for common model architectures
                for model_module in [modeling_qwen2, modeling_llama, modeling_phi3, modeling_mistral]:
                    try:
                        register_attention_interface(model_module)
                    except (AttributeError, ImportError):
                        # Skip if module doesn't support attention interface or isn't available
                        continue

            except ImportError:
                # Fallback to older method for transformer versions that don't support AttentionInterface
                pass

        # monkey patching for older transformer versions
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward

    except Exception as e:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please pip install transformers within the versions that llama-factory requires. "
            f"Error details: {str(e)}"
        )

    return group_this
