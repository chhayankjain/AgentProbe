"""Tests for agentprobe.llm — provider-agnostic LLM construction."""

from __future__ import annotations

import argparse

import pytest

from agentprobe.injector.tool_failure import ToolCallRateLimited, ToolCallTimeout
from agentprobe.llm import (
    PROVIDER_NAMES,
    PROVIDER_PRESETS,
    LLMSpec,
    add_llm_args,
    autogen_agent_kwargs,
    autogen_llm_config,
    build_chat_model,
    is_llm_backend_error,
    llm_call_boundary,
    looks_like_provider_error,
    mark_llm_call_error,
    spec_from_args,
)

# ---------------------------------------------------------------------------
# LLMSpec
# ---------------------------------------------------------------------------


class TestLLMSpec:
    def test_default_is_local_ollama_like_the_paper(self):
        spec = LLMSpec()
        assert spec.provider == "ollama" and spec.model == "llama3.1"
        assert spec.is_local and spec.temperature == 0.0
        assert spec.resolve_api_key() == "ollama"  # no env var needed

    @pytest.mark.parametrize("provider", [p for p in PROVIDER_NAMES if p != "openai-compatible"])
    def test_presets_fill_base_url_and_key_env(self, provider: str):
        spec = LLMSpec.from_provider(provider, model="m")
        preset = PROVIDER_PRESETS[provider]
        assert spec.resolved_base_url == preset.base_url
        assert (spec.api_key_env or None) == preset.api_key_env

    def test_openai_compatible_requires_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            LLMSpec.from_provider("openai-compatible", model="m")
        spec = LLMSpec.from_provider("openai-compatible", model="m", base_url="http://x:8000/v1")
        assert spec.resolved_base_url == "http://x:8000/v1"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMSpec.from_provider("nope", model="m")

    def test_missing_key_names_the_variable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        spec = LLMSpec.from_provider("groq", model="m")
        with pytest.raises(RuntimeError, match=r"\$GROQ_API_KEY"):
            spec.resolve_api_key()

    def test_key_read_from_custom_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_KEY", "sk-test")
        spec = LLMSpec.from_provider("groq", model="m", api_key_env="MY_KEY")
        assert spec.resolve_api_key() == "sk-test"

    def test_describe_never_leaks_the_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-value")
        spec = LLMSpec.from_provider(
            "openrouter", model="meta-llama/llama-3.3-70b", requests_per_minute=20
        )
        d = spec.describe()
        assert "sk-secret-value" not in repr(d)
        assert d["llm_provider"] == "openrouter"
        assert d["model_id"] == "meta-llama/llama-3.3-70b"
        assert d["llm_api_key_env"] == "OPENROUTER_API_KEY"
        assert d["llm_requests_per_minute"] == 20
        assert d["llm_is_local"] is False

    def test_trailing_slash_stripped_from_base_url(self):
        # Clients append "/chat/completions"; "…/openai/" would become "…/openai//chat/completions" (404).
        spec = LLMSpec.from_provider("openai-compatible", model="m", base_url="http://h/v1/")
        assert spec.resolved_base_url == "http://h/v1"
        assert not LLMSpec.from_provider("gemini", model="m").resolved_base_url.endswith("/")

    def test_rpm_must_be_positive(self):
        with pytest.raises(ValueError):
            LLMSpec.from_provider("groq", model="m", requests_per_minute=0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_autogen_config_uses_api_base_for_legacy_client(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        cfg = autogen_llm_config(LLMSpec.from_provider("groq", model="m", timeout_seconds=30))
        entry = cfg["config_list"][0]
        assert entry == {"model": "m", "api_base": "https://api.groq.com/openai/v1", "api_key": "k"}
        assert cfg["request_timeout"] == 30

    def test_autogen_config_ollama_matches_paper(self):
        cfg = autogen_llm_config(LLMSpec.from_provider("ollama", model="llama3.1"))
        assert cfg["config_list"][0]["api_base"] == "http://localhost:11434/v1"
        assert cfg["config_list"][0]["api_key"] == "ollama"

    def test_autogen_rejects_non_openai_compatible(self):
        with pytest.raises(ValueError, match="not OpenAI-compatible"):
            autogen_llm_config(LLMSpec.from_provider("anthropic", model="claude"))

    def test_autogen_agent_kwargs(self):
        gem = autogen_agent_kwargs(
            LLMSpec.from_provider("gemini", model="m", requests_per_minute=10)
        )
        assert gem == {"requests_per_minute": 10, "max_retries": 6, "stop_on_final_answer": True}
        assert autogen_agent_kwargs(LLMSpec.from_provider("ollama", model="m")) == {
            "requests_per_minute": None,
            "max_retries": 6,
        }

    def test_openai_compatible_chat_model(self, monkeypatch: pytest.MonkeyPatch):
        pytest.importorskip("langchain_openai")
        from langchain_openai import ChatOpenAI

        monkeypatch.setenv("GROQ_API_KEY", "k")
        spec = LLMSpec.from_provider(
            "groq", model="llama-3.3-70b-versatile", requests_per_minute=30, max_retries=4
        )
        llm = build_chat_model(spec)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "llama-3.3-70b-versatile"
        assert llm.openai_api_base == "https://api.groq.com/openai/v1"
        assert llm.max_retries == 4
        assert llm.temperature == 0.0
        assert llm.rate_limiter is not None

    def test_ollama_chat_model_matches_paper_construction(self):
        pytest.importorskip("langchain_ollama")
        from langchain_ollama import ChatOllama

        llm = build_chat_model(LLMSpec.from_provider("ollama", model="qwen2.5:7b"))
        assert isinstance(llm, ChatOllama)
        assert llm.model == "qwen2.5:7b" and llm.temperature == 0.0


# ---------------------------------------------------------------------------
# Backend-error detection
# ---------------------------------------------------------------------------


def _exc_from_module(name: str, module: str) -> Exception:
    cls = type(name, (Exception,), {"__module__": module})
    return cls("boom")


class TestIsLLMBackendError:
    @pytest.mark.parametrize(
        "name,module",
        [
            ("RateLimitError", "openai"),
            ("RateLimitError", "openai.error"),
            ("APIConnectionError", "openai"),
            ("InternalServerError", "openai"),
            ("ConnectError", "httpx"),
            ("ReadTimeout", "httpx"),
            ("ResponseError", "ollama._types"),
            ("OverloadedError", "anthropic"),
            ("Weird", "google.api_core.exceptions"),
        ],
    )
    def test_provider_errors_from_a_model_call_are_backend_errors(self, name, module):
        exc = _exc_from_module(name, module)
        assert looks_like_provider_error(exc)
        assert not is_llm_backend_error(exc)  # unmarked: origin unknown
        assert is_llm_backend_error(mark_llm_call_error(exc))

    def test_boundary_context_manager_marks(self):
        exc = _exc_from_module("RateLimitError", "openai")
        with pytest.raises(Exception) as info:
            with llm_call_boundary():
                raise exc
        assert is_llm_backend_error(info.value)

    def test_tool_transport_error_is_not_a_backend_error(self):
        """A real tool failing with httpx.ReadTimeout must stay an agent-visible failure."""
        assert not is_llm_backend_error(_exc_from_module("ReadTimeout", "httpx"))

    def test_marked_non_provider_error_is_not_backend(self):
        assert not is_llm_backend_error(mark_llm_call_error(ValueError("bad output parse")))

    def test_injected_tool_faults_are_never_backend_errors(self):
        assert not is_llm_backend_error(mark_llm_call_error(ToolCallRateLimited()))
        assert not is_llm_backend_error(ToolCallTimeout("t"))

    def test_wrapped_marked_cause_is_followed(self):
        inner = mark_llm_call_error(_exc_from_module("RateLimitError", "openai"))
        outer = RuntimeError("framework wrapper")
        outer.__cause__ = inner
        assert is_llm_backend_error(outer)

    def test_injected_fault_wrapping_backend_error_stays_injected(self):
        outer = ToolCallTimeout("injected")
        outer.__cause__ = mark_llm_call_error(_exc_from_module("ReadTimeout", "httpx"))
        assert not is_llm_backend_error(outer)


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------


class TestArgs:
    def test_defaults_reproduce_paper_backend(self):
        p = argparse.ArgumentParser()
        add_llm_args(p)
        spec = spec_from_args(p.parse_args([]))
        assert spec == LLMSpec.from_provider(
            "ollama", model="llama3.1", max_retries=6, timeout_seconds=120.0
        )

    def test_hosted_flags(self):
        p = argparse.ArgumentParser()
        add_llm_args(p)
        ns = p.parse_args(
            [
                "--provider",
                "openrouter",
                "--model",
                "x/y",
                "--rpm",
                "20",
                "--api-key-env",
                "K",
                "--max-retries",
                "2",
            ]
        )
        spec = spec_from_args(ns)
        assert spec.provider == "openrouter" and spec.model == "x/y"
        assert spec.requests_per_minute == 20 and spec.api_key_env == "K" and spec.max_retries == 2

    def test_openai_compatible_needs_base_url_flag(self):
        p = argparse.ArgumentParser()
        add_llm_args(p)
        with pytest.raises(ValueError):
            spec_from_args(p.parse_args(["--provider", "openai-compatible", "--model", "m"]))
        spec = spec_from_args(
            p.parse_args(
                ["--provider", "openai-compatible", "--model", "m", "--base-url", "http://h/v1"]
            )
        )
        assert spec.resolved_base_url == "http://h/v1"
