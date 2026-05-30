from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator


class MetricsBackend(Enum):
    NONE = "none"
    PROMETHEUS = "prometheus"
    OTEL = "otel"


@dataclass
class TaskStats:
    calls: int = 0
    failures: int = 0
    retries: int = 0
    total_duration_s: float = 0.0

    @property
    def avg_duration_s(self) -> float:
        return self.total_duration_s / self.calls if self.calls else 0.0


class MetricsCollector:
    def __init__(
        self,
        *,
        backend: str | MetricsBackend = MetricsBackend.NONE,
        namespace: str = "castellum",
        enable_traces: bool = False,
    ) -> None:
        self._namespace = namespace
        self._enable_traces = enable_traces
        if isinstance(backend, str):
            try:
                backend = MetricsBackend(backend)
            except ValueError:
                backend = MetricsBackend.NONE
        self._backend = backend
        self._stats: dict[str, TaskStats] = {}
        self._exporter = self._build_exporter(backend, namespace)

    def _build_exporter(self, backend: MetricsBackend, namespace: str) -> Any:
        if backend == MetricsBackend.PROMETHEUS:
            from castellum.metrics.prometheus import PrometheusExporter

            return PrometheusExporter(namespace=namespace)
        if backend == MetricsBackend.OTEL:
            from castellum.metrics.otel import OtelExporter

            return OtelExporter(namespace=namespace)
        return None

    @contextmanager
    def task_timer(self, task_name: str) -> Iterator[None]:
        start = time.perf_counter()
        stats = self._stats.setdefault(task_name, TaskStats())
        try:
            yield
            duration = time.perf_counter() - start
            stats.calls += 1
            stats.total_duration_s += duration
            if self._exporter:
                self._exporter.record_call(task_name, duration=duration)
        except Exception:
            stats.failures += 1
            if self._exporter:
                self._exporter.record_failure(task_name)
            raise

    def record_retry(self, task_name: str) -> None:
        self._stats.setdefault(task_name, TaskStats()).retries += 1

    def snapshot(self) -> dict[str, TaskStats]:
        return dict(self._stats)

    def current_span(self) -> _NoopSpan:
        return _NoopSpan()


class _NoopSpan:
    def set_tag(self, key: str, value: Any) -> None:
        pass
