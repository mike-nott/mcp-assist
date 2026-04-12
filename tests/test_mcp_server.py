"""Tests for MCP server configuration-sensitive behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist.const import (
    CONF_ALLOWED_IPS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_LMSTUDIO_URL,
)
from custom_components.mcp_assist.mcp_server import MCPServer


def test_server_collects_allowed_ips_from_url_and_shared_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The MCP server should whitelist localhost, LM Studio host, and configured networks."""
    system_entry_factory(data={CONF_ALLOWED_IPS: "10.0.0.0/24,192.168.1.25"})
    entry = profile_entry_factory(
        data={CONF_LMSTUDIO_URL: "http://192.168.50.12:11434"}
    )

    server = MCPServer(hass, 8099, entry)

    assert "127.0.0.1" in server.allowed_ips
    assert "::1" in server.allowed_ips
    assert "192.168.50.12" in server.allowed_ips
    assert "10.0.0.0/24" in server.allowed_ips
    assert "192.168.1.25" in server.allowed_ips


def test_tool_enablement_follows_shared_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Optional tool families should respect their shared enable/disable toggles."""
    system_entry_factory(
        data={
            CONF_ENABLE_DEVICE_TOOLS: False,
            CONF_ENABLE_ASSIST_BRIDGE: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("discover_entities") is True
    assert server._is_tool_enabled("discover_devices") is False
    assert server._is_tool_enabled("list_assist_tools") is False
    assert server._is_tool_enabled("call_service_with_response") is False
    assert server._is_tool_enabled("analyze_entity_history") is False
    assert server._is_tool_enabled("add") is False
    assert server._is_tool_enabled("play_music_assistant") is False


@pytest.mark.asyncio
async def test_handle_tools_list_filters_disabled_tool_families(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Disabled optional tools should not be advertised in the MCP tool list."""
    system_entry_factory(
        data={
            CONF_ENABLE_DEVICE_TOOLS: False,
            CONF_ENABLE_ASSIST_BRIDGE: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.custom_tools = SimpleNamespace(
        get_tool_definitions=lambda: [{"name": "add", "description": "calc", "inputSchema": {}}],
        is_custom_tool=lambda tool_name: tool_name == "add",
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "discover_entities" in tool_names
    assert "discover_devices" not in tool_names
    assert "list_assist_tools" not in tool_names
    assert "call_service_with_response" not in tool_names
    assert "get_entity_history" not in tool_names
    assert "play_music_assistant" not in tool_names
    assert "add" not in tool_names


@pytest.mark.asyncio
async def test_default_tool_list_stays_streamlined(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Compatibility aliases and optional bridge tools should stay out of the default list."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "get_entity_history" in tool_names
    assert "get_last_entity_event" not in tool_names
    assert "list_assist_tools" not in tool_names


@pytest.mark.asyncio
async def test_handle_tool_call_rejects_disabled_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Disabled tools should fail closed even if called directly."""
    system_entry_factory(data={CONF_ENABLE_DEVICE_TOOLS: False})
    server = MCPServer(hass, 8099, profile_entry_factory())

    with pytest.raises(ValueError, match="disabled"):
        await server.handle_tool_call({"name": "discover_devices", "arguments": {}})


def test_validate_service_blocks_music_assistant_when_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Generic service actions should also respect the Music Assistant toggle."""
    system_entry_factory(data={CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False})
    server = MCPServer(hass, 8099, profile_entry_factory())

    with pytest.raises(ValueError, match="Music Assistant support is disabled"):
        server.validate_service("music_assistant", "play_media")


@pytest.mark.asyncio
async def test_list_domains_hides_disabled_music_assistant_domain(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Domain summaries should not advertise disabled optional domains."""
    system_entry_factory(data={CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False})
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.list_domains = AsyncMock(
        return_value=[
            {"domain": "light", "count": 1},
            {"domain": "music_assistant", "count": 2},
        ]
    )

    result = await server.tool_list_domains()
    text = result["content"][0]["text"]

    assert "light: 1 entities" in text
    assert "music_assistant" not in text
