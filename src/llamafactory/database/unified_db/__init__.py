"""
DC-Agents Dataset and Model Registration Package

A comprehensive registration system for managing datasets and models
with support for HuggingFace and local files.
"""

# Import main functions for easy access
from .utils import (
    register_hf_dataset,
    register_local_parquet,
    get_dataset_by_name,
    delete_dataset_by_id,
    delete_dataset_by_name,
    get_model_by_name,
    delete_model_by_id,
    delete_model_by_name,
    register_agent,
    get_agent_by_name,
    delete_agent_by_id,
    delete_agent_by_name,
    register_benchmark,
    get_benchmark_by_name,
    delete_benchmark_by_id,
    delete_benchmark_by_name,
    register_trained_model,
    load_supabase_keys,
)
from .models import DatasetModel, ModelModel, AgentModel, BenchmarkModel

__all__ = [
    "register_hf_dataset",
    "register_local_parquet",
    "get_dataset_by_name",
    "delete_dataset_by_id",
    "delete_dataset_by_name",
    "get_model_by_name",
    "delete_model_by_id",
    "delete_model_by_name",
    "register_agent",
    "get_agent_by_name",
    "delete_agent_by_id",
    "delete_agent_by_name",
    "register_benchmark",
    "get_benchmark_by_name",
    "delete_benchmark_by_id",
    "delete_benchmark_by_name",
    "register_trained_model",
    "load_supabase_keys",
    "DatasetModel",
    "ModelModel",
    "AgentModel",
    "BenchmarkModel",
]
