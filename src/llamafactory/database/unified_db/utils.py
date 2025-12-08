#!/usr/bin/env python3
"""
Dataset Registration Utilities for DC-Agents

This module provides utility functions for dataset registration using Supabase
with support for both HuggingFace and local parquet file datasets.
"""

import logging
import os
import json
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import Client

from .config import get_default_client, get_admin_client
from .models import clean_dataset_metadata, clean_model_metadata, clean_agent_metadata, clean_benchmark_metadata

logger = logging.getLogger(__name__)

try:
    from harbor.utils.traces_utils import (
        convert_openai_to_sharegpt,
        export_traces,
        rows_to_dataset,
        push_dataset,
    )
except ImportError:  # pragma: no cover - Harbor optional in some environments

    def _missing_harbor(*_args, **_kwargs):
        raise ImportError(
            "Harbor is required for trace export utilities. Install harbor or add it to PYTHONPATH."
        )

    def convert_openai_to_sharegpt(*args, **kwargs):
        _missing_harbor()

    def export_traces(*args, **kwargs):
        _missing_harbor()

    def rows_to_dataset(*args, **kwargs):
        _missing_harbor()

    def push_dataset(*args, **kwargs):
        _missing_harbor()


def load_supabase_keys() -> bool:
    """Load Supabase credentials from KEYS env var if available."""
    keys_env = os.environ.get("KEYS")
    if not keys_env:
        warnings.warn(
            "Supabase credentials not loaded: set KEYS env variable to a secrets file "
            "to enable database registration."
        )
        return False

    keys_path = os.path.expandvars(keys_env)
    if not os.path.isfile(keys_path):
        warnings.warn(
            f"Supabase credentials file not found at '{keys_path}'. "
            "Model uploads will not be registered in the database until KEYS points to a valid file."
        )
        return False

    try:
        with open(keys_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = os.path.expandvars(value)
    except Exception as exc:
        warnings.warn(
            f"Failed to load Supabase credentials from '{keys_path}': {exc!r}. "
            "Database registration will be skipped."
        )
        return False

    required = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        warnings.warn(
            "Missing Supabase settings "
            f"{', '.join(missing)} after loading KEYS file. "
            "Model uploads will not be registered; ensure the KEYS file exports these values."
        )
        return False

    return True


def get_supabase_client(use_admin: bool = False) -> Client:
    """Get Supabase client for database operations."""
    if use_admin:
        return get_admin_client()
    return get_default_client()


# ==================== DATASET UTILITIES ====================

def get_dataset_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a dataset from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('datasets').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving dataset by name {name}: {e}")
        return None


def create_dataset(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new dataset in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('datasets').insert(dataset_data).execute()

        if not response.data:
            raise ValueError("Failed to create dataset")

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


def update_dataset(dataset_id: str, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing dataset in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('datasets').update(dataset_data).eq('id', dataset_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update dataset with ID {dataset_id}")

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating dataset {dataset_id}: {e}")
        raise


# ==================== DATASET REGISTRATION FUNCTIONS ====================

def register_hf_dataset(
    repo_name: str,
    dataset_type: str,
    name: Optional[str] = None,
    created_by: Optional[str] = None,
    data_generation_hash: Optional[str] = None,
    generation_start: Optional[datetime] = None,
    generation_end: Optional[datetime] = None,
    generation_parameters: Optional[Dict] = None,
    forced_update: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Register a HuggingFace dataset with comprehensive auto-filling.
    """
    try:
        try:
            from datasets import load_dataset
            from huggingface_hub import dataset_info
        except ImportError:
            raise ImportError("datasets and huggingface_hub libraries required for HF dataset registration.")

        logger.info(f"Registering HuggingFace dataset: {repo_name}")

        dataset_name = name or repo_name
        existing = get_dataset_by_name(dataset_name)
        if existing and not forced_update:
            logger.info(f"Dataset {dataset_name} already exists")
            return {"success": True, "dataset": existing, "exists": True}

        try:
            hf_info = dataset_info(repo_name)
            logger.info(f"Retrieved HF metadata for {repo_name}")
        except Exception as e:
            logger.error(f"Failed to get HF dataset info for {repo_name}: {e}")
            return {"success": False, "error": f"Could not access HuggingFace dataset {repo_name}: {e}"}

        num_tasks = None
        try:
            dataset = load_dataset(repo_name)['train']
            if hasattr(dataset, '__len__'):
                num_tasks = len(dataset)
            elif hasattr(dataset, 'num_rows'):
                num_tasks = dataset.num_rows
            logger.info(f"Dataset size: {num_tasks} rows")
        except Exception as e:
            logger.warning(f"Could not determine dataset size for {repo_name}: {e}")
            num_tasks = None

        if not created_by:
            if '/' in repo_name:
                created_by = repo_name.split('/')[0]
            else:
                created_by = "hf-uploader"

        auto_params = {
            "hf_repo": repo_name,
            "source": "huggingface_hub",
            "access_method": "datasets_library",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "hf_metadata": {
                "fingerprint": getattr(dataset, '_fingerprint', None) if 'dataset' in locals() else None,
                "commit_hash": getattr(hf_info, 'sha', None),
                "tags": getattr(hf_info, 'tags', []),
                "description": getattr(hf_info, 'description', ''),
            }
        }

        if generation_parameters:
            auto_params.update(generation_parameters)

        now = datetime.now(timezone.utc)
        dataset_data = {
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),
            "name": dataset_name,
            "created_by": created_by,
            "data_location": f"https://huggingface.co/datasets/{repo_name}",
            "creation_location": "HuggingFace",
            "generation_status": "completed",
            "generation_parameters": auto_params,
            "generation_start": generation_start.isoformat() if generation_start else None,
            "generation_end": generation_end.isoformat() if generation_end else None,
            "hf_fingerprint": getattr(dataset, '_fingerprint', None) if 'dataset' in locals() else None,
            "hf_commit_hash": getattr(hf_info, 'sha', None),
            "num_tasks": num_tasks,
            "dataset_type": dataset_type,
            "data_generation_hash": data_generation_hash,
        }

        if kwargs:
            dataset_data.update(kwargs)

        if existing:
            updated = update_dataset(existing['id'], dataset_data)
            return {"success": True, "dataset": updated, "updated": True}
        else:
            created = create_dataset(dataset_data)
            return {"success": True, "dataset": created}

    except Exception as e:
        logger.error(f"Failed to register HuggingFace dataset {repo_name}: {e}")
        return {"success": False, "error": str(e)}


def register_local_parquet(
    dataset_path: str,
    dataset_type: str,
    name: Optional[str] = None,
    created_by: Optional[str] = None,
    data_generation_hash: Optional[str] = None,
    generation_parameters: Optional[Dict] = None,
    forced_update: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Register a local parquet dataset."""
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        import pyarrow.parquet as pq

        dataset_name = name or os.path.basename(dataset_path.rstrip("/"))
        existing = get_dataset_by_name(dataset_name)
        if existing and not forced_update:
            return {"success": True, "dataset": existing, "exists": True}

        num_tasks = None
        try:
            parquet_files = []
            if os.path.isdir(dataset_path):
                for root, _, files in os.walk(dataset_path):
                    parquet_files.extend(os.path.join(root, f) for f in files if f.endswith(".parquet"))
            elif dataset_path.endswith(".parquet"):
                parquet_files = [dataset_path]

            num_tasks = 0
            for parquet_file in parquet_files:
                table = pq.read_table(parquet_file)
                num_tasks += table.num_rows
        except Exception as e:
            logger.warning(f"Could not determine dataset size for {dataset_path}: {e}")
            num_tasks = None

        if not created_by:
            created_by = "local-uploader"

        auto_params = {
            "source": "local_parquet",
            "path": dataset_path,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if generation_parameters:
            auto_params.update(generation_parameters)

        now = datetime.now(timezone.utc)
        dataset_data = {
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),
            "name": dataset_name,
            "created_by": created_by,
            "data_location": dataset_path,
            "creation_location": "local",
            "generation_status": "completed",
            "generation_parameters": auto_params,
            "generation_start": None,
            "generation_end": None,
            "hf_fingerprint": None,
            "hf_commit_hash": None,
            "num_tasks": num_tasks,
            "dataset_type": dataset_type,
            "data_generation_hash": data_generation_hash,
        }

        if kwargs:
            dataset_data.update(kwargs)

        if existing:
            updated = update_dataset(existing['id'], dataset_data)
            return {"success": True, "dataset": updated, "updated": True}
        else:
            created = create_dataset(dataset_data)
            return {"success": True, "dataset": created}

    except Exception as e:
        logger.error(f"Failed to register local parquet dataset {dataset_path}: {e}")
        return {"success": False, "error": str(e)}


def delete_dataset_by_id(dataset_id: str) -> Dict[str, Any]:
    """Delete dataset from database by ID."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('datasets').delete().eq('id', dataset_id).execute()

        if response.data:
            logger.info(f"Successfully deleted dataset: {dataset_id}")
            return {"success": True, "message": f"Dataset with ID {dataset_id} deleted successfully", "deleted_id": dataset_id}
        else:
            return {"success": False, "error": f"Failed to delete dataset with ID {dataset_id}"}

    except Exception as e:
        logger.error(f"Error deleting dataset by ID {dataset_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_dataset_by_name(name: str) -> Dict[str, Any]:
    """Delete dataset from database by name."""
    try:
        dataset = get_dataset_by_name(name)
        if not dataset:
            return {"success": False, "error": f"Dataset with name '{name}' not found"}
        return delete_dataset_by_id(dataset['id'])

    except Exception as e:
        logger.error(f"Error deleting dataset by name {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== MODEL UTILITIES ====================

def get_model_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a model from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('models').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving model by name {name}: {e}")
        return None


def create_model(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new model in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('models').insert(model_data).execute()

        if not response.data:
            raise ValueError("Failed to create model")

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


def update_model(model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing model in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('models').update(model_data).eq('id', model_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update model with ID {model_id}")

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        raise


def delete_model_by_id(model_id: str) -> Dict[str, Any]:
    """Delete model from database by ID."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('models').delete().eq('id', model_id).execute()

        if response.data:
            logger.info(f"Successfully deleted model: {model_id}")
            return {"success": True, "message": f"Model with ID {model_id} deleted successfully", "deleted_id": model_id}
        else:
            return {"success": False, "error": f"Failed to delete model with ID {model_id}"}

    except Exception as e:
        logger.error(f"Error deleting model by ID {model_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_model_by_name(name: str) -> Dict[str, Any]:
    """Delete model from database by name."""
    try:
        model = get_model_by_name(name)
        if not model:
            return {"success": False, "error": f"Model with name '{name}' not found"}
        return delete_model_by_id(model['id'])

    except Exception as e:
        logger.error(f"Error deleting model by name {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== AGENT UTILITIES ====================

def get_agent_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve an agent from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('agents').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving agent by name {name}: {e}")
        return None


def create_agent(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new agent in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('agents').insert(agent_data).execute()

        if not response.data:
            raise ValueError("Failed to create agent")

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


def update_agent(agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing agent in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('agents').update(agent_data).eq('id', agent_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update agent with ID {agent_id}")

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise


def register_agent(name: str, agent_version_hash: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
    """Register or update an agent in the database."""
    try:
        existing = get_agent_by_name(name)
        now = datetime.now(timezone.utc).isoformat()
        agent_data = {
            "name": name,
            "agent_version_hash": agent_version_hash,
            "description": description,
            "updated_at": now,
        }

        if existing:
            updated = update_agent(existing['id'], agent_data)
            return {"success": True, "agent": updated, "updated": True}
        else:
            created = create_agent(agent_data)
            return {"success": True, "agent": created}

    except Exception as e:
        logger.error(f"Failed to register agent {name}: {e}")
        return {"success": False, "error": str(e)}


def delete_agent_by_id(agent_id: str) -> Dict[str, Any]:
    """Delete agent from database by ID."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('agents').delete().eq('id', agent_id).execute()

        if response.data:
            logger.info(f"Successfully deleted agent: {agent_id}")
            return {"success": True, "message": f"Agent with ID {agent_id} deleted successfully", "deleted_id": agent_id}
        else:
            return {"success": False, "error": f"Failed to delete agent with ID {agent_id}"}

    except Exception as e:
        logger.error(f"Error deleting agent by ID {agent_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_agent_by_name(name: str) -> Dict[str, Any]:
    """Delete agent from database by name."""
    try:
        agent = get_agent_by_name(name)
        if not agent:
            return {"success": False, "error": f"Agent with name '{name}' not found"}
        return delete_agent_by_id(agent['id'])

    except Exception as e:
        logger.error(f"Error deleting agent by name {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== BENCHMARK UTILITIES ====================

def get_benchmark_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a benchmark from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('benchmarks').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving benchmark by name {name}: {e}")
        return None


def create_benchmark(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new benchmark in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('benchmarks').insert(benchmark_data).execute()

        if not response.data:
            raise ValueError("Failed to create benchmark")

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating benchmark: {e}")
        raise


def update_benchmark(benchmark_id: str, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing benchmark in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('benchmarks').update(benchmark_data).eq('id', benchmark_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update benchmark with ID {benchmark_id}")

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating benchmark {benchmark_id}: {e}")
        raise


def register_benchmark(
    name: str,
    benchmark_version_hash: Optional[str] = None,
    is_external: bool = False,
    external_link: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Register or update a benchmark in the database."""
    try:
        existing = get_benchmark_by_name(name)
        now = datetime.now(timezone.utc).isoformat()
        benchmark_data = {
            "name": name,
            "benchmark_version_hash": benchmark_version_hash,
            "is_external": is_external,
            "external_link": external_link,
            "description": description,
            "updated_at": now,
        }

        if existing:
            updated = update_benchmark(existing['id'], benchmark_data)
            return {"success": True, "benchmark": updated, "updated": True}
        else:
            created = create_benchmark(benchmark_data)
            return {"success": True, "benchmark": created}

    except Exception as e:
        logger.error(f"Failed to register benchmark {name}: {e}")
        return {"success": False, "error": str(e)}


def delete_benchmark_by_id(benchmark_id: str) -> Dict[str, Any]:
    """Delete benchmark from database by ID."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('benchmarks').delete().eq('id', benchmark_id).execute()

        if response.data:
            logger.info(f"Successfully deleted benchmark: {benchmark_id}")
            return {"success": True, "message": f"Benchmark with ID {benchmark_id} deleted successfully", "deleted_id": benchmark_id}
        else:
            return {"success": False, "error": f"Failed to delete benchmark with ID {benchmark_id}"}

    except Exception as e:
        logger.error(f"Error deleting benchmark by ID {benchmark_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_benchmark_by_name(name: str) -> Dict[str, Any]:
    """Delete benchmark from database by name."""
    try:
        benchmark = get_benchmark_by_name(name)
        if not benchmark:
            return {"success": False, "error": f"Benchmark with name '{name}' not found"}
        return delete_benchmark_by_id(benchmark['id'])

    except Exception as e:
        logger.error(f"Error deleting benchmark by name {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== TRAINED MODEL REGISTRATION ====================

def register_trained_model(
    training_record: Dict[str, Any],
    forced_update: bool = False
) -> Dict[str, Any]:
    """
    Register a newly trained model (SFT/RL)
    """
    try:
        def _unwrap(value):
            if isinstance(value, (list, tuple, set)):
                if not value:
                    return None
                try:
                    first = value[0]
                except TypeError:
                    first = next(iter(value))
                return _unwrap(first)
            return value

        agent_name = _unwrap(training_record.get('agent_name'))
        base_model_name = _unwrap(training_record.get('base_model_name'))
        training_type = _unwrap(training_record.get('training_type'))
        if not agent_name:
            return {"success": False, "error": "agent_name is required"}
        if not base_model_name:
            return {"success": False, "error": "base_model_name is required"}
        if training_type not in ('SFT', 'RL'):
            return {"success": False, "error": "training_type must be 'SFT' or 'RL'"}

        def _normalize_dataset_list(raw: Any) -> List[str]:
            if raw is None:
                return []
            if isinstance(raw, str):
                parts = raw.split(',')
            elif isinstance(raw, (list, tuple, set)):
                parts = list(raw)
            else:
                parts = [raw]
            normalized: List[str] = []
            for item in parts:
                name = str(item).strip()
                if name and name not in normalized:
                    normalized.append(name)
            return normalized

        dataset_list = _normalize_dataset_list(training_record.get('dataset_names'))
        if not dataset_list:
            dataset_list = _normalize_dataset_list(training_record.get('dataset_name'))
        if not dataset_list:
            return {"success": False, "error": "dataset_name is required"}

        def _parse_ts(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace('Z', '+00:00')) if val.endswith('Z') else datetime.fromisoformat(val)
            raise ValueError("timestamp must be datetime or ISO string")

        raw_start = training_record.get('training_start')
        if not raw_start:
            return {"success": False, "error": "training_start is required"}
        training_start_dt = _parse_ts(raw_start)
        training_end_dt = _parse_ts(training_record.get('training_end'))

        training_params = training_record.get('training_parameters')
        if training_params is None:
            training_params = {}
        elif isinstance(training_params, str):
            try:
                training_params = json.loads(training_params)
            except Exception:
                training_params = {"raw": training_params}
        elif isinstance(training_params, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in training_params.items():
                try:
                    json.dumps(value)
                    cleaned[key] = value
                except TypeError:
                    cleaned[key] = str(value)
            training_params = cleaned
        else:
            try:
                json.dumps(training_params)
                training_params = {"value": training_params}
            except TypeError:
                training_params = {"raw": str(training_params)}

        created_by = training_record.get('created_by') or ''
        wandb_link = training_record.get('wandb_link') or ''
        traces_location_s3 = training_record.get('traces_location_s3') or ''
        explicit_name = training_record.get('model_name') or ''

        agent_res = register_agent(name=agent_name)
        if not agent_res.get('success'):
            return agent_res
        agent = agent_res['agent']
        agent_id = agent['id']

        dataset_id: Optional[str] = None
        dataset_names_csv: Optional[str] = None
        if len(dataset_list) == 1:
            dataset_name_single = dataset_list[0]
            ds = get_dataset_by_name(dataset_name_single)
            if not ds:
                ds_res = register_hf_dataset(
                    repo_name=dataset_name_single,
                    dataset_type=training_type,
                    name=dataset_name_single,
                    created_by=created_by,
                )
                if not ds_res.get('success'):
                    return {"success": False, "error": ds_res.get('error', 'Dataset registration failed')}
                ds = ds_res['dataset']
            dataset_id = ds['id']
        else:
            dataset_names_csv = ",".join(dataset_list)
            for name in dataset_list:
                ds = get_dataset_by_name(name)
                if not ds:
                    ds_res = register_hf_dataset(
                        repo_name=name,
                        dataset_type=training_type,
                        name=name,
                        created_by=created_by,
                    )
                    if not ds_res.get('success'):
                        return {"success": False, "error": ds_res.get('error', 'Dataset registration failed')}

        base_m = get_model_by_name(base_model_name)
        if not base_m:
            now_dt = datetime.now(timezone.utc)
            now_ts = now_dt.isoformat()
            base_training_start = training_start_dt or now_dt
            base_training_end = training_end_dt or base_training_start
            base_payload = {
                "name": base_model_name,
                "created_by": (created_by or (base_model_name.split('/')[0] if '/' in base_model_name else "hf-uploader")),
                "creation_location": "HuggingFace",
                "creation_time": now_ts,
                "updated_at": now_ts,
                "is_external": True,
                "weights_location": f"https://huggingface.co/{base_model_name}",
                "training_status": "completed",
                "training_parameters": {
                    "source": "huggingface_hub",
                    "registered_at": now_ts,
                },
                "agent_id": agent_id,
                "training_type": training_type,
                "training_start": base_training_start.isoformat(),
                "training_end": base_training_end.isoformat(),
            }
            base_m = create_model(base_payload)
        base_model_id = base_m['id']

        if explicit_name:
            model_name = explicit_name
        else:
            dataset_name_for_default = dataset_list[0]
            date_str = (training_end_dt or training_start_dt).strftime('%Y%m%d')
            model_name = f"{dataset_name_for_default}_{date_str}"

        existing = get_model_by_name(model_name)
        now_ts = datetime.now(timezone.utc).isoformat()
        weights_location = f"https://huggingface.co/{model_name}"
        training_status = 'completed' if training_end_dt else 'in_progress'

        model_data = {
            "name": model_name,
            "created_by": created_by,
            "creation_location": "HuggingFace",
            "creation_time": now_ts,
            "updated_at": now_ts,
            "is_external": True,
            "weights_location": weights_location,
            "training_status": training_status,
            "training_parameters": training_params,
            "description": None,
            "agent_id": agent_id,
            "base_model_id": base_model_id,
            "dataset_id": dataset_id,
            "dataset_names": dataset_names_csv,
            "training_type": training_type,
            "training_start": training_start_dt.isoformat(),
            "training_end": training_end_dt.isoformat() if training_end_dt else None,
            "wandb_link": wandb_link,
            "traces_location_s3": traces_location_s3,
        }

        if existing and not forced_update:
            return {"success": True, "model": existing, "exists": True}
        if existing and forced_update:
            updated = update_model(existing['id'], model_data)
            return {"success": True, "model": updated, "updated": True}
        created = create_model(model_data)
        return {"success": True, "model": created}

    except Exception as e:
        logger.error(f"Failed to register trained model: {e}")
        return {"success": False, "error": str(e)}
