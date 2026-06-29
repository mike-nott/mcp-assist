"""Tests for shared provider runtime helpers."""

from __future__ import annotations

from custom_components.mcp_assist.const import (
    CONF_LMSTUDIO_URL,
    CONF_SERVER_TYPE,
    DEFAULT_OLLAMA_URL,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
)
from custom_components.mcp_assist.provider_runtime import (
    build_provider_auth_headers,
    resolve_provider_runtime_config,
)


def test_resolve_provider_runtime_config_prefers_custom_openai_base_url(
    profile_entry_factory,
) -> None:
    """OpenAI profiles should reuse the configured compatible base URL."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: "https://openai-proxy.example.com/v1/",
        }
    )

    runtime_config = resolve_provider_runtime_config(entry)

    assert runtime_config.server_type == SERVER_TYPE_OPENAI
    assert runtime_config.base_url == "https://openai-proxy.example.com/v1"


def test_resolve_provider_runtime_config_keeps_default_local_ollama_url(
    profile_entry_factory,
) -> None:
    """Local Ollama profiles should keep the default base URL when unset."""
    entry = profile_entry_factory(
        data={CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA, CONF_LMSTUDIO_URL: ""}
    )

    runtime_config = resolve_provider_runtime_config(entry)

    assert runtime_config.server_type == SERVER_TYPE_OLLAMA
    assert runtime_config.base_url == DEFAULT_OLLAMA_URL


def test_build_provider_auth_headers_supports_openrouter_and_local_profiles() -> None:
    """Shared auth header logic should match the cloud/local provider rules."""
    openrouter_headers = build_provider_auth_headers(
        SERVER_TYPE_OPENROUTER,
        "openrouter-key",
    )
    local_headers = build_provider_auth_headers(SERVER_TYPE_OLLAMA, "")

    assert openrouter_headers["Authorization"] == "Bearer openrouter-key"
    assert openrouter_headers["HTTP-Referer"] == "https://github.com/mike-nott/mcp-assist"
    assert openrouter_headers["X-Title"] == "MCP Assist for Home Assistant"
    assert local_headers == {}


def test_build_provider_auth_headers_skips_placeholder_openai_keys() -> None:
    """OpenAI-compatible providers should not send obviously fake placeholder keys."""
    placeholder_headers = build_provider_auth_headers(SERVER_TYPE_OPENAI, "fake")
    valid_headers = build_provider_auth_headers(
        SERVER_TYPE_OPENAI,
        "sk-live-value",
    )

    assert placeholder_headers == {}
    assert valid_headers == {"Authorization": "Bearer sk-live-value"}
