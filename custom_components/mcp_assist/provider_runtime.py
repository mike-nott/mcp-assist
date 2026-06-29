"""Shared provider runtime helpers for MCP Assist model-backed features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .const import (
    ANTHROPIC_BASE_URL,
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_SERVER_TYPE,
    CONF_TIMEOUT,
    DEFAULT_API_KEY,
    DEFAULT_LLAMACPP_URL,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SERVER_TYPE,
    DEFAULT_TIMEOUT,
    DEFAULT_VLLM_URL,
    GEMINI_BASE_URL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_VLLM,
)


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    """Resolved provider settings for a conversation profile."""

    server_type: str
    model_name: str
    api_key: str
    timeout: int
    base_url: str


def _get_entry_value(entry: Any, key: str, default: Any) -> Any:
    """Read a config-entry value from options first, then data."""
    if entry is None:
        return default

    value = entry.options.get(key, entry.data.get(key))
    return default if value is None else value


def _get_configured_base_url(entry: Any) -> str | None:
    """Return the explicitly configured provider URL, if one is set."""
    if entry is None:
        return None

    for source in (entry.options, entry.data):
        if CONF_LMSTUDIO_URL not in source:
            continue

        value = source.get(CONF_LMSTUDIO_URL)
        if value is None:
            continue

        url = str(value).strip().rstrip("/")
        if url:
            return url

    return None


def resolve_provider_runtime_config(entry: Any) -> ProviderRuntimeConfig:
    """Resolve the current provider/runtime settings for a profile entry."""
    server_type = str(_get_entry_value(entry, CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE))
    model_name = str(_get_entry_value(entry, CONF_MODEL_NAME, DEFAULT_MODEL_NAME))
    api_key = str(_get_entry_value(entry, CONF_API_KEY, DEFAULT_API_KEY) or "")
    timeout = int(_get_entry_value(entry, CONF_TIMEOUT, DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
    configured_url = _get_configured_base_url(entry)

    if server_type == SERVER_TYPE_OPENAI:
        base_url = configured_url or OPENAI_BASE_URL
    elif server_type == SERVER_TYPE_GEMINI:
        base_url = GEMINI_BASE_URL.rstrip("/")
    elif server_type == SERVER_TYPE_ANTHROPIC:
        base_url = ANTHROPIC_BASE_URL.rstrip("/")
    elif server_type == SERVER_TYPE_OPENROUTER:
        base_url = OPENROUTER_BASE_URL.rstrip("/")
    elif server_type == SERVER_TYPE_VLLM:
        base_url = configured_url or DEFAULT_VLLM_URL
    elif server_type == SERVER_TYPE_OLLAMA:
        base_url = configured_url or DEFAULT_OLLAMA_URL
    elif server_type == SERVER_TYPE_LLAMACPP:
        base_url = configured_url or DEFAULT_LLAMACPP_URL
    else:
        base_url = configured_url or DEFAULT_LMSTUDIO_URL

    return ProviderRuntimeConfig(
        server_type=server_type,
        model_name=model_name,
        api_key=api_key,
        timeout=timeout,
        base_url=base_url.rstrip("/"),
    )


def build_provider_auth_headers(server_type: str, api_key: str) -> dict[str, str]:
    """Build provider auth headers shared by the agent and MCP media tools."""
    normalized_key = str(api_key or "")
    if server_type == SERVER_TYPE_OPENAI:
        if (
            normalized_key
            and len(normalized_key) > 5
            and normalized_key.lower() not in {"none", "null", "fake", "na", "n/a"}
        ):
            return {"Authorization": f"Bearer {normalized_key}"}
        return {}

    if server_type in {
        SERVER_TYPE_GEMINI,
        SERVER_TYPE_ANTHROPIC,
    }:
        return {"Authorization": f"Bearer {normalized_key}"}

    if server_type == SERVER_TYPE_OPENROUTER:
        return {
            "Authorization": f"Bearer {normalized_key}",
            "HTTP-Referer": "https://github.com/mike-nott/mcp-assist",
            "X-Title": "MCP Assist for Home Assistant",
        }

    return {}
