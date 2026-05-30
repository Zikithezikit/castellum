from __future__ import annotations


class OtelExporter:
    def __init__(self, *, namespace: str) -> None:
        try:
            from opentelemetry import metrics as otel_metrics  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Install 'castellum[otel]' for OpenTelemetry support.") from e

        meter = otel_metrics.get_meter(namespace)
        self._call_counter = meter.create_counter(f"{namespace}.task.calls")
        self._duration_histogram = meter.create_histogram(f"{namespace}.task.duration_seconds")

    def record_call(self, task_name: str, *, duration: float) -> None:
        attrs = {"task": task_name}
        self._call_counter.add(1, attrs)
        self._duration_histogram.record(duration, attrs)
