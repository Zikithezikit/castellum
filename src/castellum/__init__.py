"""
Castellum — AI pipeline orchestration library.

Public re-exports. Everything a user needs should be importable from `castellum` directly.
"""

from castellum.core.task import Task, TaskKind, ai_task
from castellum.core.pipeline import Pipeline, pipeline
from castellum.runtime.runtime import Runtime
from castellum.backends.base import LLMClient
from castellum.backends.local.base import LocalModelBackend
from castellum.backends.vector.base import VectorStoreBackend

__all__ = [
    "ai_task",
    "Task",
    "TaskKind",
    "pipeline",
    "Pipeline",
    "Runtime",
    "LLMClient",
    "LocalModelBackend",
    "VectorStoreBackend",
]

__version__ = "0.3.0"
