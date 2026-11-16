"""Provider-agnostic LLM construction for experiments and adapters.

Every experiment script used to hardcode ``ChatOllama`` and a local AutoGen
``api_base``. This module centralises that choice behind an :class:`LLMSpec`
so one ``--provider/--model/--base-url/--api-key-env`` set of flags covers:

* **ollama** (default, used for every result in the paper) — local, no key.
* Any **OpenAI-compatible** endpoint, with presets for the common free and
  hosted tiers: ``openai``, ``groq``, ``openrouter``, ``cerebras``,
  ``mistral``, ``github`` (GitHub Models) and ``gemini`` (Google AI Studio's
  OpenAI-compatible endpoint). Use ``openai-compatible`` with an explicit
  ``--base-url`` for anything else (vLLM, LM Studio, llama.cpp server, ...).
* **anthropic** via ``langchain-anthropic`` (LangGraph/LangChain adapters only).

Two things matter when a hosted backend is used in a fault-injection study:

1. **Backend errors are not agent failures.** A provider-side 429 or 5xx is an
   infrastructure event and must not be counted alongside the *injected* tool
   faults. :func:`is_llm_backend_error` lets the runner tell them apart; the
   chat model is also built with client-side retries and an optional
   requests-per-minute limiter so such errors are rare in the first place.
2. **Hosted models change silently.** :meth:`LLMSpec.describe` records the
   provider, resolved base URL, exact model id and the *name* of the API-key
   environment variable (never its value) so a batch's metadata says which
   backend produced it.

Usage::

    from agentprobe.llm import LLMSpec, build_chat_model, autogen_llm_config

    spec = LLMSpec.from_provider("groq", model="llama-3.3-70b-versatile", requests_per_minute=25)
    llm = build_chat_model(spec)              # LangChain chat model for LangGraph/LangChain
    cfg = autogen_llm_config(spec)            # llm_config dict for AutoGenToolAgent

Install the matching extra: ``pip install "agentprobe-toolkit[openai]"`` for any
OpenAI-compatible provider, ``[anthropic]`` for Anthropic, ``[ollama]`` for Ollama.
Note that ``[openai]`` (``openai>=1``) cannot coexist with ``[autogen]``
(``openai<1``); run hosted LangGraph/LangChain batches in a separate environment.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

OLLAMA_OPENAI_COMPAT_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str | None
    api_key_env: str | None
    openai_compatible: bool = True
    note: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset(
        base_url=OLLAMA_OPENAI_COMPAT_URL,
        api_key_env=None,
        note="Local Ollama. LangChain adapters use langchain-ollama natively; "
        "AutoGen uses the OpenAI-compatible endpoint.",
    ),
    "openai-compatible": ProviderPreset(
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        note="Any OpenAI-compatible server; --base-url is required.",
    ),
    "openai": ProviderPreset(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "groq": ProviderPreset(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        note="Free tier is token-per-day limited; use --rpm to stay under it.",
    ),
    "openrouter": ProviderPreset(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        note="':free' models share a low daily request cap.",
    ),
    "cerebras": ProviderPreset(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
    ),
    "mistral": ProviderPreset(
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
    ),
    "github": ProviderPreset(
        base_url="https://models.github.ai/inference",
        api_key_env="GITHUB_TOKEN",
        note="GitHub Models. A fine-grained PAT with 'models' permission.",
    ),
    "gemini": ProviderPreset(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        note="Google AI Studio's OpenAI-compatible endpoint.",
    ),
    "anthropic": ProviderPreset(
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
        openai_compatible=False,
        note="Uses langchain-anthropic. Not available for the AutoGen adapter.",
    ),
}

PROVIDER_NAMES: tuple[str, ...] = tuple(PROVIDER_PRESETS)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMSpec:
    """Everything needed to construct a chat model for one experiment.

    Attributes:
        provider: One of :data:`PROVIDER_NAMES`.
        model: Exact model id as the provider expects it (recorded verbatim in metadata).
        base_url: Endpoint; defaults to the provider preset. Required for ``openai-compatible``.
        api_key_env: Environment variable holding the key; defaults to the preset.
            The key itself is read lazily and never stored on the spec.
        temperature: Sampling temperature. The paper used ``0.0`` throughout.
        max_retries: Client-side retries for transient provider errors (429/5xx/timeouts).
        timeout_seconds: Per-request timeout for hosted providers.
        requests_per_minute: Optional client-side throttle (``None`` = unlimited).
    """

    provider: str = "ollama"
    model: str = "llama3.1"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_retries: int = 6
    timeout_seconds: float | None = 120.0
    requests_per_minute: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown provider {self.provider!r}; expected one of {', '.join(PROVIDER_NAMES)}"
            )
        if self.provider == "openai-compatible" and not self.base_url:
            raise ValueError("provider 'openai-compatible' requires base_url")
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_provider(cls, provider: str, model: str, **overrides: Any) -> LLMSpec:
        """Build a spec from a provider name, filling base_url/api_key_env from the preset."""
        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            raise ValueError(
                f"Unknown provider {provider!r}; expected one of {', '.join(PROVIDER_NAMES)}"
            )
        kwargs: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "base_url": preset.base_url,
            "api_key_env": preset.api_key_env,
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)

    # -- derived properties --------------------------------------------------

    @property
    def preset(self) -> ProviderPreset:
        return PROVIDER_PRESETS[self.provider]

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    @property
    def openai_compatible(self) -> bool:
        return self.preset.openai_compatible

    @property
    def resolved_base_url(self) -> str | None:
        return self.base_url or self.preset.base_url

    def resolve_api_key(self) -> str:
        """Return the API key from the environment, or a placeholder for local Ollama.

        Raises:
            RuntimeError: when a hosted provider's key variable is unset.
        """
        if self.is_local and not self.api_key_env:
            return "ollama"
        env = self.api_key_env or self.preset.api_key_env
        if not env:
            return "ollama"
        key = os.getenv(env)
        if not key:
            raise RuntimeError(
                f"Provider {self.provider!r} needs an API key in ${env}, which is not set. "
                f"Export it or pass --api-key-env to name a different variable."
            )
        return key

    def describe(self) -> dict[str, Any]:
        """Metadata-safe description. Never includes the key value."""
        return {
            "llm_provider": self.provider,
            "model_id": self.model,
            "llm_base_url": self.resolved_base_url,
            "llm_api_key_env": self.api_key_env or self.preset.api_key_env,
            "llm_temperature": self.temperature,
            "llm_max_retries": self.max_retries,
            "llm_timeout_seconds": self.timeout_seconds,
            "llm_requests_per_minute": self.requests_per_minute,
            "llm_is_local": self.is_local,
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _rate_limiter(spec: LLMSpec) -> Any:
    if spec.requests_per_minute is None:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    rps = spec.requests_per_minute / 60.0
    return InMemoryRateLimiter(
        requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=1
    )


def build_chat_model(spec: LLMSpec) -> Any:
    """Return a LangChain chat model for *spec* (for the LangGraph/LangChain adapters)."""
    if spec.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "langchain-ollama is required for provider 'ollama': "
                'pip install "agentprobe-toolkit[ollama]"'
            ) from e
        kwargs: dict[str, Any] = {"model": spec.model, "temperature": spec.temperature}
        # Keep the paper's exact construction (no base_url) unless one was given explicitly.
        if spec.base_url and spec.base_url != OLLAMA_OPENAI_COMPAT_URL:
            kwargs["base_url"] = spec.base_url.removesuffix("/v1")
        limiter = _rate_limiter(spec)
        if limiter is not None:
            kwargs["rate_limiter"] = limiter
        return ChatOllama(**kwargs)

    if spec.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "langchain-anthropic is required for provider 'anthropic': "
                'pip install "agentprobe-toolkit[anthropic]"'
            ) from e
        kwargs = {
            "model": spec.model,
            "temperature": spec.temperature,
            "api_key": spec.resolve_api_key(),
            "max_retries": spec.max_retries,
        }
        if spec.timeout_seconds is not None:
            kwargs["timeout"] = spec.timeout_seconds
        limiter = _rate_limiter(spec)
        if limiter is not None:
            kwargs["rate_limiter"] = limiter
        return ChatAnthropic(**kwargs)

    # Every other provider is OpenAI-compatible.
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"langchain-openai is required for provider {spec.provider!r}: "
            'pip install "agentprobe-toolkit[openai]" (in an environment without the '
            "[autogen] extra, which pins openai<1)"
        ) from e
    kwargs = {
        "model": spec.model,
        "temperature": spec.temperature,
        "base_url": spec.resolved_base_url,
        "api_key": spec.resolve_api_key(),
        "max_retries": spec.max_retries,
    }
    if spec.timeout_seconds is not None:
        kwargs["timeout"] = spec.timeout_seconds
    limiter = _rate_limiter(spec)
    if limiter is not None:
        kwargs["rate_limiter"] = limiter
    kwargs.update(spec.extra)
    return ChatOpenAI(**kwargs)


def autogen_llm_config(spec: LLMSpec) -> dict[str, Any]:
    """Return an ``llm_config`` for :class:`~agentprobe.agents.autogen_agent.AutoGenToolAgent`.

    pyautogen 0.1.14 speaks the legacy ``openai<1`` client, which honours
    ``api_base``/``api_key`` for any OpenAI-compatible endpoint, so every
    preset except ``anthropic`` works here.
    """
    if not spec.openai_compatible:
        raise ValueError(
            f"Provider {spec.provider!r} is not OpenAI-compatible and cannot be used "
            "with the AutoGen adapter"
        )
    entry: dict[str, Any] = {
        "model": spec.model,
        "api_base": spec.resolved_base_url,
        "api_key": spec.resolve_api_key(),
    }
    cfg: dict[str, Any] = {"config_list": [entry]}
    if spec.timeout_seconds is not None:
        cfg["request_timeout"] = spec.timeout_seconds
    return cfg


# ---------------------------------------------------------------------------
# Backend-error detection
# ---------------------------------------------------------------------------

#: Module prefixes of exceptions raised by LLM clients / transports.
_BACKEND_MODULE_PREFIXES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google.",
    "httpx",
    "httpcore",
    "ollama",
    "langchain_ollama",
    "langchain_openai",
    "langchain_anthropic",
    "requests.",
    "urllib3",
    "aiohttp",
)

#: Exception class names that are backend errors regardless of module.
_BACKEND_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIStatusError",
        "InternalServerError",
        "ServiceUnavailableError",
        "AuthenticationError",
        "PermissionDeniedError",
        "ResponseError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    }
)


def is_llm_backend_error(exc: BaseException) -> bool:
    """True if *exc* came from the LLM provider or its transport, not from the agent.

    Injected tool faults (``agentprobe.injector.tool_failure.ToolCallError`` and
    subclasses) and the harness's own timeout are never backend errors. The check
    walks ``__cause__``/``__context__`` so a framework that wraps the provider
    exception is still recognised.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        mod = type(cur).__module__ or ""
        name = type(cur).__name__
        if mod.startswith("agentprobe."):
            return False
        if mod.startswith(_BACKEND_MODULE_PREFIXES) or name in _BACKEND_EXCEPTION_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# ---------------------------------------------------------------------------
# argparse helpers shared by the experiment scripts
# ---------------------------------------------------------------------------


def add_llm_args(parser: argparse.ArgumentParser, default_model: str = "llama3.1") -> None:
    """Add ``--provider/--model/--base-url/--api-key-env/--rpm/--max-retries/--llm-timeout``."""
    g = parser.add_argument_group("LLM backend")
    g.add_argument(
        "--provider",
        default="ollama",
        choices=PROVIDER_NAMES,
        help="LLM provider (default: ollama, as used in the paper). "
        "Hosted presets set base URL and key variable automatically.",
    )
    g.add_argument("--model", default=default_model, help="Exact model id for the provider")
    g.add_argument(
        "--base-url",
        default=None,
        help="Override the provider's endpoint (required for --provider openai-compatible)",
    )
    g.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable holding the API key (default: provider preset)",
    )
    g.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Client-side requests-per-minute throttle for hosted tiers",
    )
    g.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Client retries for transient provider errors (429/5xx/timeout)",
    )
    g.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds for hosted providers",
    )


def spec_from_args(args: argparse.Namespace) -> LLMSpec:
    """Build an :class:`LLMSpec` from a namespace produced by :func:`add_llm_args`."""
    return LLMSpec.from_provider(
        args.provider,
        model=args.model,
        base_url=getattr(args, "base_url", None),
        api_key_env=getattr(args, "api_key_env", None),
        requests_per_minute=getattr(args, "rpm", None),
        max_retries=getattr(args, "max_retries", None),
        timeout_seconds=getattr(args, "llm_timeout", None),
    )


__all__ = [
    "LLMSpec",
    "ProviderPreset",
    "PROVIDER_PRESETS",
    "PROVIDER_NAMES",
    "OLLAMA_OPENAI_COMPAT_URL",
    "build_chat_model",
    "autogen_llm_config",
    "is_llm_backend_error",
    "add_llm_args",
    "spec_from_args",
]
