"""Prometheus metrics exporter for AgentProbe.

Exports agent reliability metrics to Prometheus, enabling Grafana dashboards
for production monitoring of agentic LLM systems.

Metrics exported:
  agentprobe_tool_call_total          — Counter of tool calls by framework/task/result
  agentprobe_tool_failure_total       — Counter of tool failures by type
  agentprobe_latency_seconds          — Histogram of agent turn latencies
  agentprobe_mttr_seconds             — Gauge of mean-time-to-recovery
  agentprobe_failure_rate             — Gauge of current failure rate [0, 1]
  agentprobe_task_success_total       — Counter of tasks completed successfully
  agentprobe_task_failure_total       — Counter of tasks that failed

Usage::

    from agentprobe.observer.dashboard import PrometheusExporter

    exporter = PrometheusExporter(port=8000)
    exporter.start()     # starts /metrics HTTP server on :8000

    # Record metrics manually
    exporter.record_tool_call(framework="langgraph", task="rag", success=False,
                              failure_type="timeout")
    exporter.record_latency(framework="langgraph", latency_seconds=2.4)
    exporter.set_failure_rate(framework="langgraph", rate=0.12)

    # Or auto-record from benchmark results
    exporter.record_benchmark_run(results)
"""

from __future__ import annotations

import threading
from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# Default latency histogram buckets (seconds) covering agent response times
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf"))


class PrometheusExporter:
    """Prometheus metrics exporter for AgentProbe benchmark results.

    Falls back to a no-op implementation if prometheus_client is not installed,
    so benchmarks can run without a Prometheus server.

    Args:
        port: HTTP port for the /metrics endpoint (default: 8000).
        prefix: Metric name prefix (default: "agentprobe").
    """

    def __init__(self, port: int = 8000, prefix: str = "agentprobe") -> None:
        self.port = port
        self.prefix = prefix
        self._server_started = False
        self._lock = threading.Lock()

        if not _PROMETHEUS_AVAILABLE:
            import warnings

            warnings.warn(
                "prometheus_client not installed. Metrics export is disabled. "
                "Install with: pip install prometheus-client",
                stacklevel=2,
            )
            self._available = False
            return

        self._available = True
        p = prefix

        self._tool_call_total = Counter(
            f"{p}_tool_call_total",
            "Total tool calls",
            ["framework", "task", "result"],
        )
        self._tool_failure_total = Counter(
            f"{p}_tool_failure_total",
            "Total tool failures by failure type",
            ["framework", "task", "failure_type"],
        )
        self._latency_histogram = Histogram(
            f"{p}_latency_seconds",
            "Agent turn latency in seconds",
            ["framework", "task"],
            buckets=_LATENCY_BUCKETS,
        )
        self._mttr_gauge = Gauge(
            f"{p}_mttr_seconds",
            "Mean time to recovery in seconds",
            ["framework"],
        )
        self._failure_rate_gauge = Gauge(
            f"{p}_failure_rate",
            "Current failure rate [0, 1]",
            ["framework"],
        )
        self._task_success_total = Counter(
            f"{p}_task_success_total",
            "Total tasks completed successfully",
            ["framework", "task"],
        )
        self._task_failure_total = Counter(
            f"{p}_task_failure_total",
            "Total tasks that failed",
            ["framework", "task"],
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Prometheus /metrics HTTP server (non-blocking)."""
        if not self._available:
            return
        with self._lock:
            if not self._server_started:
                start_http_server(self.port)
                self._server_started = True

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_tool_call(
        self,
        framework: str,
        task: str,
        success: bool,
        failure_type: str = "none",
    ) -> None:
        if not self._available:
            return
        result = "success" if success else "failure"
        self._tool_call_total.labels(framework=framework, task=task, result=result).inc()
        if not success:
            self._tool_failure_total.labels(
                framework=framework, task=task, failure_type=failure_type
            ).inc()

    def record_latency(
        self,
        framework: str,
        latency_seconds: float,
        task: str = "unknown",
    ) -> None:
        if not self._available:
            return
        self._latency_histogram.labels(framework=framework, task=task).observe(latency_seconds)

    def set_mttr(self, framework: str, mttr_seconds: float) -> None:
        if not self._available:
            return
        self._mttr_gauge.labels(framework=framework).set(mttr_seconds)

    def set_failure_rate(self, framework: str, rate: float) -> None:
        if not self._available:
            return
        self._failure_rate_gauge.labels(framework=framework).set(rate)

    def record_task_result(
        self, framework: str, task: str, success: bool
    ) -> None:
        if not self._available:
            return
        if success:
            self._task_success_total.labels(framework=framework, task=task).inc()
        else:
            self._task_failure_total.labels(framework=framework, task=task).inc()

    def record_benchmark_run(self, results: dict[str, Any]) -> None:
        """Record a complete benchmark run result dict (as produced by BenchmarkRunner).

        Expected keys: framework, task, runs (list of RunResult dicts).
        """
        if not self._available:
            return
        framework = results.get("framework", "unknown")
        task = results.get("task", "unknown")
        runs: list[dict[str, Any]] = results.get("runs", [])

        for run in runs:
            success = run.get("success", False)
            latency = run.get("latency_ms", 0.0) / 1000.0
            failure_type = run.get("failure_type", "none")

            self.record_tool_call(framework, task, success, failure_type)
            self.record_latency(framework, latency, task)
            self.record_task_result(framework, task, success)

        # Compute aggregate metrics
        total = len(runs)
        if total > 0:
            failures = sum(1 for r in runs if not r.get("success", True))
            self.set_failure_rate(framework, failures / total)

            # MTTR: average latency of failed runs that eventually recovered
            recovery_latencies = [
                r.get("recovery_latency_ms", 0.0) / 1000.0
                for r in runs
                if not r.get("success", True) and r.get("recovered", False)
            ]
            if recovery_latencies:
                self.set_mttr(framework, sum(recovery_latencies) / len(recovery_latencies))
