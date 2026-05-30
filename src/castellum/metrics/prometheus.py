from __future__ import annotations


class PrometheusExporter:
    def __init__(self, *, namespace: str) -> None:
        try:
            from prometheus_client import Counter, Histogram  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Install 'castellum[prometheus]' for Prometheus support.") from e

        self._calls = Counter(f"{namespace}_task_calls_total", "Total task calls", ["task"])
        self._duration = Histogram(f"{namespace}_task_duration_seconds", "Task duration", ["task"])
        self._failures = Counter(f"{namespace}_task_failures_total", "Task failures", ["task"])

    def record_call(self, task_name: str, *, duration: float) -> None:
        self._calls.labels(task=task_name).inc()
        self._duration.labels(task=task_name).observe(duration)

    def record_failure(self, task_name: str) -> None:
        self._failures.labels(task=task_name).inc()
