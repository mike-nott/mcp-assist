"""Shared test fixtures for MCP Assist."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mcp_assist.const import (
    CONF_ALLOWED_IPS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_LMSTUDIO_URL,
    CONF_MCP_PORT,
    CONF_MODEL_NAME,
    CONF_PROFILE_NAME,
    CONF_SEARCH_PROVIDER,
    CONF_SERVER_TYPE,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_ENABLE_ASSIST_BRIDGE,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_ENABLE_GAP_FILLING,
    DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    DEFAULT_ENABLE_RECORDER_TOOLS,
    DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_MCP_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_SERVER_TYPE,
    DOMAIN,
    SYSTEM_ENTRY_UNIQUE_ID,
)


@pytest.fixture
def system_entry_factory(hass) -> Callable[..., MockConfigEntry]:
    """Create and add a shared system entry."""

    def _factory(
        *,
        data: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> MockConfigEntry:
        entry = MockConfigEntry(
            domain=DOMAIN,
            title="Shared MCP Server Settings",
            unique_id=SYSTEM_ENTRY_UNIQUE_ID,
            source="system",
            data={
                CONF_MCP_PORT: DEFAULT_MCP_PORT,
                CONF_SEARCH_PROVIDER: DEFAULT_SEARCH_PROVIDER,
                CONF_ALLOWED_IPS: DEFAULT_ALLOWED_IPS,
                CONF_ENABLE_GAP_FILLING: DEFAULT_ENABLE_GAP_FILLING,
                CONF_ENABLE_ASSIST_BRIDGE: DEFAULT_ENABLE_ASSIST_BRIDGE,
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS: DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
                CONF_ENABLE_RECORDER_TOOLS: DEFAULT_ENABLE_RECORDER_TOOLS,
                CONF_ENABLE_CALCULATOR_TOOLS: DEFAULT_ENABLE_CALCULATOR_TOOLS,
                CONF_ENABLE_DEVICE_TOOLS: DEFAULT_ENABLE_DEVICE_TOOLS,
                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                **(data or {}),
            },
            options=options or {},
        )
        entry.add_to_hass(hass)
        return entry

    return _factory


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations) -> None:
    """Allow Home Assistant's flow manager to discover this custom integration."""
    return None


@pytest.fixture
def profile_entry_factory(hass) -> Callable[..., MockConfigEntry]:
    """Create and add a profile entry."""

    def _factory(
        *,
        title: str = "Ollama - Test Profile",
        unique_id: str = f"{DOMAIN}_ollama_test_profile",
        data: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> MockConfigEntry:
        entry = MockConfigEntry(
            domain=DOMAIN,
            title=title,
            unique_id=unique_id,
            source="user",
            data={
                CONF_PROFILE_NAME: "Test Profile",
                CONF_SERVER_TYPE: DEFAULT_SERVER_TYPE,
                CONF_LMSTUDIO_URL: DEFAULT_LMSTUDIO_URL,
                CONF_MODEL_NAME: DEFAULT_MODEL_NAME,
                **(data or {}),
            },
            options=options or {},
        )
        entry.add_to_hass(hass)
        return entry

    return _factory
