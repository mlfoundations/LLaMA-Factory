#!/usr/bin/env python3
"""
Models for DC-Agents Dataset and Model Registration

This module provides DatasetModel and ModelModel for data validation and serialization
supporting both HuggingFace and local file registration.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_serializer, field_validator


class DatasetModel(BaseModel):
    """Dataset model for database operations."""
    id: Optional[UUID] = Field(default_factory=uuid4)
    name: str
    created_by: str
    creation_location: str
    creation_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    generation_start: Optional[datetime] = None
    generation_end: Optional[datetime] = None
    data_location: str
    generation_parameters: Dict[str, Any]
    generation_status: Optional[str] = None
    dataset_type: str
    data_generation_hash: Optional[str] = None
    hf_fingerprint: Optional[str] = None
    hf_commit_hash: Optional[str] = None
    num_tasks: Optional[int] = None
    last_modified: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator('dataset_type')
    @classmethod
    def validate_dataset_type(cls, v: str) -> str:
        if v not in ["SFT", "RL"]:
            raise ValueError("dataset_type must be either 'SFT' or 'RL'")
        return v

    @field_serializer('id')
    def serialize_id(self, value: Optional[UUID]) -> Optional[str]:
        return str(value) if value else None

    @field_serializer('creation_time')
    def serialize_creation_time(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @field_serializer('generation_start')
    def serialize_generation_start(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @field_serializer('generation_end')
    def serialize_generation_end(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @field_serializer('updated_at')
    def serialize_updated_at(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @field_serializer('last_modified')
    def serialize_last_modified(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None


def clean_dataset_metadata(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean dataset metadata for API responses."""
    if not dataset_data:
        return {}

    cleaned = {
        'id': str(dataset_data.get('id')),
        'name': dataset_data.get('name'),
        'created_by': dataset_data.get('created_by'),
        'creation_location': dataset_data.get('creation_location'),
        'creation_time': dataset_data.get('creation_time'),
        'generation_start': dataset_data.get('generation_start'),
        'generation_end': dataset_data.get('generation_end'),
        'data_location': dataset_data.get('data_location'),
        'generation_parameters': dataset_data.get('generation_parameters', {}),
        'generation_status': dataset_data.get('generation_status'),
        'dataset_type': dataset_data.get('dataset_type'),
        'data_generation_hash': dataset_data.get('data_generation_hash'),
        'hf_fingerprint': dataset_data.get('hf_fingerprint'),
        'hf_commit_hash': dataset_data.get('hf_commit_hash'),
        'num_tasks': dataset_data.get('num_tasks'),
        'last_modified': dataset_data.get('last_modified'),
        'updated_at': dataset_data.get('updated_at')
    }

    return {k: v for k, v in cleaned.items() if v is not None}


class ModelModel(BaseModel):
    """Model model for ML model registration."""
    id: Optional[UUID] = Field(default_factory=uuid4)
    name: str
    base_model_id: Optional[UUID] = None
    created_by: str
    creation_location: str
    creation_time: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_id: Optional[UUID] = None
    is_external: bool = False
    weights_location: str
    wandb_link: Optional[str] = None
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Training result related fields
    training_start: datetime
    training_end: Optional[datetime] = None
    training_parameters: Dict[str, Any]
    training_status: Optional[str] = None
    agent_id: UUID
    training_type: Optional[str] = None
    traces_location_s3: Optional[str] = None
    description: Optional[str] = None

    @field_validator('training_type')
    @classmethod
    def validate_training_type(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in ["SFT", "RL"]:
            raise ValueError("training_type must be either 'SFT' or 'RL'")
        return v

    @field_serializer('id')
    def serialize_id(self, value: Optional[UUID]) -> Optional[str]:
        return str(value) if value else None

    @field_serializer('base_model_id')
    def serialize_base_model_id(self, value: Optional[UUID]) -> Optional[str]:
        return str(value) if value else None

    @field_serializer('dataset_id')
    def serialize_dataset_id(self, value: Optional[UUID]) -> Optional[str]:
        return str(value) if value else None

    @field_serializer('agent_id')
    def serialize_agent_id(self, value: UUID) -> str:
        return str(value)

    @field_serializer('creation_time')
    def serialize_creation_time(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @field_serializer('updated_at')
    def serialize_updated_at(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @field_serializer('training_start')
    def serialize_training_start(self, value: datetime) -> str:
        return value.isoformat()

    @field_serializer('training_end')
    def serialize_training_end(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None


def clean_model_metadata(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean model metadata for API responses."""
    if not model_data:
        return {}

    cleaned = {
        'id': str(model_data.get('id')) if model_data.get('id') else None,
        'name': model_data.get('name'),
        'base_model_id': str(model_data.get('base_model_id')) if model_data.get('base_model_id') else None,
        'created_by': model_data.get('created_by'),
        'creation_location': model_data.get('creation_location'),
        'creation_time': model_data.get('creation_time'),
        'dataset_id': str(model_data.get('dataset_id')) if model_data.get('dataset_id') else None,
        'is_external': model_data.get('is_external'),
        'weights_location': model_data.get('weights_location'),
        'wandb_link': model_data.get('wandb_link'),
        'updated_at': model_data.get('updated_at'),
        'training_start': model_data.get('training_start'),
        'training_end': model_data.get('training_end'),
        'training_parameters': model_data.get('training_parameters', {}),
        'training_status': model_data.get('training_status'),
        'agent_id': str(model_data.get('agent_id')) if model_data.get('agent_id') else None,
        'training_type': model_data.get('training_type'),
        'traces_location_s3': model_data.get('traces_location_s3'),
        'description': model_data.get('description')
    }

    return {k: v for k, v in cleaned.items() if v is not None}


class AgentModel(BaseModel):
    """Agent model for evaluation agents."""
    id: Optional[UUID] = Field(default_factory=uuid4)
    name: str
    agent_version_hash: Optional[str] = Field(None, max_length=64)
    description: Optional[str] = None
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('agent_version_hash')
    @classmethod
    def validate_agent_version_hash(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) != 64:
            raise ValueError("agent_version_hash must be exactly 64 characters (SHA-256 hash)")
        return v

    @field_serializer('id')
    def serialize_id(self, value: Optional[UUID]) -> Optional[str]:
        return str(value) if value else None

    @field_serializer('updated_at')
    def serialize_updated_at(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None


def clean_agent_metadata(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean agent metadata for API responses."""
    if not agent_data:
        return {}

    cleaned = {
        'id': str(agent_data.get('id')) if agent_data.get('id') else None,
        'name': agent_data.get('name'),
        'agent_version_hash': agent_data.get('agent_version_hash'),
        'description': agent_data.get('description'),
        'updated_at': agent_data.get('updated_at')
    }

    return {k: v for k, v in cleaned.items() if v is not None}


class BenchmarkModel(BaseModel):
    """Benchmark model for evaluation benchmarks."""
    id: Optional[UUID] = Field(default_factory=uuid4)
    name: str
    benchmark_version_hash: Optional[str] = Field(None, max_length=64)
    is_external: bool = False
    external_link: Optional[str] = None
    description: Optional[str] = None
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('benchmark_version_hash')
    @classmethod
    def validate_benchmark_version_hash(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) != 64:
            raise ValueError("benchmark_version_hash must be exactly 64 characters (SHA-256 hash)")
        return v

    @field_serializer('id')
    def serialize_id(self, value: Optional[UUID]) -> Optional[str]:
        return str(value) if value else None

    @field_serializer('updated_at')
    def serialize_updated_at(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None


def clean_benchmark_metadata(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean benchmark metadata for API responses."""
    if not benchmark_data:
        return {}

    cleaned = {
        'id': str(benchmark_data.get('id')) if benchmark_data.get('id') else None,
        'name': benchmark_data.get('name'),
        'benchmark_version_hash': benchmark_data.get('benchmark_version_hash'),
        'is_external': benchmark_data.get('is_external'),
        'external_link': benchmark_data.get('external_link'),
        'description': benchmark_data.get('description'),
        'updated_at': benchmark_data.get('updated_at')
    }

    return {k: v for k, v in cleaned.items() if v is not None}
