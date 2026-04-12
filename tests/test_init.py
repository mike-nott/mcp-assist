"""Tests for integration setup helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mcp_assist import (
    _migrate_brave_search_tool_name,
    async_setup_entry,
    async_unload_entry,
    ensure_system_entry,
)
from custom_components.mcp_assist.const import (
    CONF_ALLOWED_IPS,
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MCP_PORT,
    CONF_PROFILE_NAME,
    CONF_SEARCH_PROVIDER,
    CONF_TECHNICAL_PROMPT,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_MCP_PORT,
    DOMAIN,
    SYSTEM_ENTRY_UNIQUE_ID,
)


def _mock_system_entry_init(hass, data: dict) -> MockConfigEntry:
    """Simulate creation of the shared system entry without full HA dependency setup."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Shared MCP Server Settings",
        unique_id=SYSTEM_ENTRY_UNIQUE_ID,
        source="system",
        data=data,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_migrate_brave_search_tool_name_updates_legacy_prompt(
    hass, profile_entry_factory
) -> None:
    """Legacy brave_search references should be migrated in-place."""
    entry = profile_entry_factory(
        options={CONF_TECHNICAL_PROMPT: "Use brave_search for current events."}
    )

    await _migrate_brave_search_tool_name(hass, entry)

    assert entry.options[CONF_TECHNICAL_PROMPT] == "Use search for current events."


@pytest.mark.asyncio
async def test_ensure_system_entry_copies_shared_settings_from_first_profile(
    hass, profile_entry_factory
) -> None:
    """System entry creation should copy the shared MCP settings from the first profile."""
    profile_entry_factory(
        data={
            CONF_MCP_PORT: 1883,
            CONF_SEARCH_PROVIDER: "none",
            CONF_BRAVE_API_KEY: "",
            CONF_ALLOWED_IPS: "",
            CONF_ENABLE_GAP_FILLING: True,
        },
        options={
            CONF_MCP_PORT: 8124,
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "duckduckgo",
            CONF_BRAVE_API_KEY: "abc123",
            CONF_ALLOWED_IPS: "10.0.0.0/24",
            CONF_INCLUDE_CURRENT_USER: False,
            CONF_INCLUDE_HOME_LOCATION: False,
            CONF_ENABLE_GAP_FILLING: False,
            CONF_ENABLE_ASSIST_BRIDGE: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_MEMORY_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
            CONF_ENABLE_DEVICE_TOOLS: False,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: True,
            CONF_MEMORY_DEFAULT_TTL_DAYS: 14,
            CONF_MEMORY_MAX_TTL_DAYS: 90,
            CONF_MEMORY_MAX_ITEMS: 250,
        },
    )

    async_init = AsyncMock(
        side_effect=lambda domain, context, data: _mock_system_entry_init(hass, data)
    )
    with patch.object(hass.config_entries.flow, "async_init", async_init):
        system_entry = await ensure_system_entry(hass)

    assert system_entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID
    assert system_entry.data[CONF_MCP_PORT] == 8124
    assert system_entry.data[CONF_ENABLE_WEB_SEARCH] is True
    assert system_entry.data[CONF_SEARCH_PROVIDER] == "duckduckgo"
    assert system_entry.data[CONF_BRAVE_API_KEY] == "abc123"
    assert system_entry.data[CONF_ALLOWED_IPS] == "10.0.0.0/24"
    assert system_entry.data[CONF_INCLUDE_CURRENT_USER] is False
    assert system_entry.data[CONF_INCLUDE_HOME_LOCATION] is False
    assert system_entry.data[CONF_ENABLE_GAP_FILLING] is False
    assert system_entry.data[CONF_ENABLE_DEVICE_TOOLS] is False
    assert system_entry.data[CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT] is True
    assert system_entry.data[CONF_ENABLE_WEATHER_FORECAST_TOOL] is False
    assert system_entry.data[CONF_ENABLE_MEMORY_TOOLS] is True
    assert system_entry.data[CONF_ENABLE_UNIT_CONVERSION_TOOLS] is True
    assert system_entry.data[CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS] is True
    assert system_entry.data[CONF_MEMORY_DEFAULT_TTL_DAYS] == 14
    assert system_entry.data[CONF_MEMORY_MAX_TTL_DAYS] == 90
    assert system_entry.data[CONF_MEMORY_MAX_ITEMS] == 250


@pytest.mark.asyncio
async def test_ensure_system_entry_uses_defaults_without_profiles(hass) -> None:
    """System entry creation should fall back to defaults when no profiles exist yet."""
    async_init = AsyncMock(
        side_effect=lambda domain, context, data: _mock_system_entry_init(hass, data)
    )
    with patch.object(hass.config_entries.flow, "async_init", async_init):
        system_entry = await ensure_system_entry(hass)

    assert system_entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID
    assert system_entry.data[CONF_MCP_PORT] == DEFAULT_MCP_PORT
    assert CONF_ENABLE_WEB_SEARCH in system_entry.data
    assert CONF_INCLUDE_CURRENT_USER in system_entry.data
    assert CONF_INCLUDE_HOME_LOCATION in system_entry.data
    assert system_entry.data[CONF_ENABLE_DEVICE_TOOLS] == DEFAULT_ENABLE_DEVICE_TOOLS
    assert CONF_ENABLE_WEATHER_FORECAST_TOOL in system_entry.data
    assert CONF_ENABLE_MEMORY_TOOLS in system_entry.data
    assert CONF_ENABLE_UNIT_CONVERSION_TOOLS in system_entry.data
    assert CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS in system_entry.data
    assert CONF_MEMORY_DEFAULT_TTL_DAYS in system_entry.data
    assert CONF_MEMORY_MAX_TTL_DAYS in system_entry.data
    assert CONF_MEMORY_MAX_ITEMS in system_entry.data


@pytest.mark.asyncio
async def test_async_setup_and_unload_reuse_shared_runtime_objects(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Profiles should share the MCP server and index manager until the last unload."""
    system_entry_factory()
    entry_one = profile_entry_factory(title="Ollama - One", unique_id=f"{DOMAIN}_one", data={CONF_PROFILE_NAME: "One"})
    entry_two = profile_entry_factory(title="Ollama - Two", unique_id=f"{DOMAIN}_two", data={CONF_PROFILE_NAME: "Two"})

    index_manager = SimpleNamespace(start=AsyncMock())
    mcp_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())

    with (
        patch("custom_components.mcp_assist.IndexManager", return_value=index_manager) as index_cls,
        patch("custom_components.mcp_assist.MCPServer", return_value=mcp_server) as server_cls,
        patch.object(hass.config_entries, "async_forward_entry_setups", AsyncMock(return_value=True)),
        patch.object(hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)),
    ):
        assert await async_setup_entry(hass, entry_one) is True
        assert await async_setup_entry(hass, entry_two) is True
        assert hass.data[DOMAIN]["mcp_refcount"] == 2
        assert index_cls.call_count == 1
        assert server_cls.call_count == 1

        assert await async_unload_entry(hass, entry_one) is True
        assert hass.data[DOMAIN]["mcp_refcount"] == 1
        mcp_server.stop.assert_not_called()

        assert await async_unload_entry(hass, entry_two) is True
        mcp_server.stop.assert_awaited_once()
        assert "shared_mcp_server" not in hass.data[DOMAIN]
