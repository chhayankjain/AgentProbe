"""Centralized configuration for AgentProbe loaded from environment variables.

Usage::

    from agentprobe.config import settings

    # Access a setting
    url = settings.ollama_base_url

    # Fail fast when a key is required
    key = settings.require_openai()
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Immutable configuration derived from environment variables.

    All fields have safe defaults; optional keys are ``str | None``.
    Use :meth:`require_openai` / :meth:`require_anthropic` at call-sites
    that actually need a key so the error surface is clear and local.
    """

    openai_api_key: str | None
    anthropic_api_key: str | None
    ollama_base_url: str
    otlp_endpoint: str | None
    no_console_trace: bool
    prometheus_port: int

    @classmethod
    def from_env(cls) -> Settings:
        """Build a :class:`Settings` instance from the current environment."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            otlp_endpoint=os.getenv("AGENTPROBE_OTLP_ENDPOINT"),
            no_console_trace=os.getenv("AGENTPROBE_NO_CONSOLE_TRACE") == "1",
            prometheus_port=int(os.getenv("AGENTPROBE_PROMETHEUS_PORT", "8000")),
        )

    def require_openai(self) -> str:
        """Return the OpenAI API key or raise :exc:`RuntimeError`."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required but not set. "
                "Add it to your environment or .env file."
            )
        return self.openai_api_key

    def require_anthropic(self) -> str:
        """Return the Anthropic API key or raise :exc:`RuntimeError`."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required but not set. "
                "Add it to your environment or .env file."
            )
        return self.anthropic_api_key


# Module-level singleton — import this rather than constructing Settings manually.
settings: Settings = Settings.from_env()
