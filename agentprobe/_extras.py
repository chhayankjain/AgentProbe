"""Helpful errors when an optional dependency is missing."""

from __future__ import annotations

#: Third-party packages and the extra that installs each one.
_EXTRA_FOR = {
    "langchain": "langchain",
    "langchain_core": "langchain",
    "langgraph": "langchain",
    "opentelemetry": "observability",
    "prometheus_client": "observability",
    "grpc": "observability",
    "pydantic": "research",
    "rich": "research",
    "numpy": "research",
    "scipy": "research",
}


def missing_extra(
    exc: ModuleNotFoundError, feature: str, extra: str | None = None
) -> ImportError | ModuleNotFoundError:
    """Turn "no module named X" into an error that names the extra to install.

    *extra* is the extra that covers the whole feature (``"research"`` for the
    benchmark, which needs several packages); without it, the extra that provides
    the missing package is named. Returns *exc* unchanged when the missing module
    isn't an optional dependency, so a genuine bug is never disguised as an
    installation problem.
    """
    root = (exc.name or "").split(".")[0]
    if root not in _EXTRA_FOR:
        return exc
    extra = extra or _EXTRA_FOR[root]
    return ImportError(
        f"{feature} needs the optional dependency {root!r}. "
        f'Install it with: pip install "agentprobe-toolkit[{extra}]"'
    )
