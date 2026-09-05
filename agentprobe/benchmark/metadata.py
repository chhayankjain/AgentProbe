"""Experiment metadata capture for reproducibility.

Records hardware, software, and configuration details alongside benchmark
results so that reviewers can assess the experimental context.

Usage::

    from agentprobe.benchmark.metadata import capture_metadata

    meta = capture_metadata(
        model="llama3.1",
        llm_backend="ollama",
        frameworks=["langgraph", "langchain", "autogen"],
    )
    meta.to_dict()  # JSON-serializable
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentMetadata:
    """Immutable record of experiment environment and configuration."""

    # Hardware
    cpu_model: str
    cpu_count: int
    ram_gb: float
    os_name: str
    os_version: str
    arch: str

    # Software
    python_version: str
    agentprobe_version: str
    git_hash: str
    git_branch: str
    git_dirty: bool

    # LLM configuration
    llm_backend: str
    model_name: str
    ollama_version: str

    # Experiment configuration
    frameworks: tuple[str, ...]
    n_runs_per_cell: int
    seed: int
    timestamp_utc: str

    # Dependency versions
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["frameworks"] = list(d["frameworks"])
        return d


def _get_git_info() -> tuple[str, str, bool]:
    """Return (commit_hash, branch, is_dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit, branch, len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "unknown", False


def _get_ollama_version() -> str:
    """Return Ollama version string, or 'not available'."""
    try:
        out = subprocess.check_output(
            ["ollama", "--version"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out.split()[-1] if out else "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not available"


def _get_ram_gb() -> float:
    """Return total system RAM in GB."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return round(int(out) / (1024**3), 1)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        return 0.0
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def _get_cpu_model() -> str:
    """Return CPU model string."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        return platform.processor() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return platform.processor() or "unknown"


def _get_dependency_versions() -> dict[str, str]:
    """Return versions of key dependencies."""
    deps: dict[str, str] = {}
    for pkg in [
        "langgraph", "langchain", "langchain_core", "langchain_community",
        "opentelemetry", "pydantic", "numpy", "pandas", "scipy",
    ]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = "not installed"

    # pyautogen uses a different import name
    try:
        import autogen  # type: ignore[import-untyped]
        deps["pyautogen"] = getattr(autogen, "__version__", "unknown")
    except ImportError:
        deps["pyautogen"] = "not installed"

    return deps


def capture_metadata(
    model: str,
    llm_backend: str = "ollama",
    frameworks: list[str] | None = None,
    n_runs_per_cell: int = 10,
    seed: int = 42,
) -> ExperimentMetadata:
    """Capture full experiment metadata snapshot.

    Call this once at experiment start and save alongside results.
    """
    git_hash, git_branch, git_dirty = _get_git_info()

    try:
        from agentprobe import __version__ as ap_version
    except ImportError:
        ap_version = "unknown"

    return ExperimentMetadata(
        cpu_model=_get_cpu_model(),
        cpu_count=os.cpu_count() or 0,
        ram_gb=_get_ram_gb(),
        os_name=platform.system(),
        os_version=platform.release(),
        arch=platform.machine(),
        python_version=sys.version.split()[0],
        agentprobe_version=ap_version,
        git_hash=git_hash,
        git_branch=git_branch,
        git_dirty=git_dirty,
        llm_backend=llm_backend,
        model_name=model,
        ollama_version=_get_ollama_version() if llm_backend == "ollama" else "n/a",
        frameworks=tuple(frameworks or ["langgraph", "langchain", "autogen"]),
        n_runs_per_cell=n_runs_per_cell,
        seed=seed,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dependency_versions=_get_dependency_versions(),
    )
