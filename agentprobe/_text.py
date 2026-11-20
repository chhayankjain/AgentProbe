"""Text helpers with no dependencies (used by the probe and the adapters)."""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    """Return the text of a LangChain message regardless of content format.

    Ollama and OpenAI-compatible models return ``content`` as a plain string.
    Gemini (via ``langchain-google-genai``) and Anthropic return a list of
    content blocks such as ``[{"type": "text", "text": ...}]``, sometimes with
    provider extras (thought signatures) attached. Adapters must not leak that
    structure into ``output``, or answer-presence grading would see ``"[{'type'..."``.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type", "text") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)
