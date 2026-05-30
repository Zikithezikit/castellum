from __future__ import annotations
import contextvars
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from castellum.runtime.scheduler import Scheduler


@dataclass
class PipelineContext:
    scheduler: "Scheduler"
    run_id: str
    metadata: dict = field(default_factory=dict)


_current_context: contextvars.ContextVar[PipelineContext | None] = \
    contextvars.ContextVar("castellum_context", default=None)


def get_current_context() -> PipelineContext | None:
    return _current_context.get()


def set_current_context(ctx: PipelineContext | None) -> None:
    _current_context.set(ctx)
