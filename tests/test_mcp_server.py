"""Tests for MCP server configuration-sensitive behavior."""

from __future__ import annotations

import base64
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession
from homeassistant.components.weather import WeatherEntityFeature
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_socket import disable_socket, enable_socket

from custom_components.mcp_assist.custom_tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.const import (
    CONF_ALLOWED_IPS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_LMSTUDIO_URL,
)
from custom_components.mcp_assist.mcp_server import MCPServer

BUILTIN_SPECS = load_builtin_tool_toggle_specs()


def _builtin_spec(tool_name: str):
    """Return the built-in packaged-tool spec for a tool name."""
    for spec in BUILTIN_SPECS:
        if tool_name in spec.tool_names:
            return spec
    raise AssertionError(f"Missing built-in spec for {tool_name}")


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


@pytest.mark.asyncio
async def test_server_start_serves_health_endpoint(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Starting the MCP server should expose a working health endpoint."""
    system_entry_factory()
    server = MCPServer(hass, 0, profile_entry_factory())

    enable_socket()
    await server.start()
    try:
        assert server.site is not None
        sockets = server.site._server.sockets  # type: ignore[attr-defined]
        assert sockets
        bound_port = sockets[0].getsockname()[1]

        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{bound_port}/health") as response:
                assert response.status == 200
                payload = await response.json()

        assert payload["status"] == "healthy"
        assert payload["server"] == "ha-entity-discovery"
        assert payload["tools_available"] > 0
    finally:
        await server.stop()
        disable_socket()


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
            CONF_ENABLE_MEMORY_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: False,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("discover_entities") is True
    assert server._is_tool_enabled("discover_devices") is False
    assert server._is_tool_enabled("list_assist_tools") is False
    assert server._is_tool_enabled("get_calendar_events") is False
    assert server._is_tool_enabled("call_service_with_response") is False
    assert server._is_tool_enabled("get_weather_forecast") is False
    assert server._is_tool_enabled("analyze_entity_history") is False
    assert server._is_tool_enabled("remember_memory") is False
    assert server._is_tool_enabled("add") is False
    assert server._is_tool_enabled("convert_unit") is False
    assert server._is_tool_enabled("play_music_assistant") is False


def test_unit_conversion_can_stay_enabled_when_calculator_is_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Unit conversion should be independently gateable from calculator math tools."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("add") is False
    assert server._is_tool_enabled("convert_unit") is True


def test_weather_forecast_tool_and_weather_services_can_be_disabled_independently(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather forecast capability should be hideable without disabling all response services."""
    system_entry_factory(
        data={
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: True,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("call_service_with_response") is True
    assert server._is_tool_enabled("get_weather_forecast") is False
    assert "Weather forecast support is disabled" in (
        server._get_domain_capability_error("weather") or ""
    )


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
            CONF_ENABLE_MEMORY_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: False,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.custom_tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {"name": "add", "description": "calc", "inputSchema": {}},
            {"name": "convert_unit", "description": "convert", "inputSchema": {}},
        ],
        is_custom_tool=lambda tool_name: tool_name in {"add", "convert_unit"},
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "discover_entities" in tool_names
    assert "discover_devices" not in tool_names
    assert "list_assist_tools" not in tool_names
    assert "get_calendar_events" not in tool_names
    assert "call_service_with_response" not in tool_names
    assert "get_weather_forecast" not in tool_names
    assert "get_entity_history" not in tool_names
    assert "remember_memory" not in tool_names
    assert "play_music_assistant" not in tool_names
    assert "add" not in tool_names
    assert "convert_unit" not in tool_names


@pytest.mark.asyncio
async def test_handle_tools_list_can_keep_unit_conversion_when_calculator_math_is_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Package-based built-ins should still honor the separate global math/unit toggles."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.custom_tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {"name": "add", "description": "calc", "inputSchema": {}},
            {"name": "convert_unit", "description": "convert", "inputSchema": {}},
        ],
        is_custom_tool=lambda tool_name: tool_name in {"add", "convert_unit"},
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "add" not in tool_names
    assert "convert_unit" in tool_names


@pytest.mark.asyncio
async def test_handle_tools_list_hides_web_search_tools_when_shared_toggle_is_off(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The shared web-search toggle should still hide search and read_url together."""
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.custom_tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {"name": "search", "description": "search", "inputSchema": {}},
            {"name": "read_url", "description": "read", "inputSchema": {}},
        ],
        is_custom_tool=lambda tool_name: tool_name in {"search", "read_url"},
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "search" not in tool_names
    assert "read_url" not in tool_names


def test_package_based_search_and_read_url_can_be_toggled_independently(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared built-in package toggles should not force search and read_url together."""
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": False,
            CONF_ENABLE_WEB_SEARCH: False,
            "search_provider": "brave",
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.custom_tools = SimpleNamespace(
        get_builtin_toggle_spec=lambda name: _builtin_spec(name),
        get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
    )

    assert server._is_tool_enabled("search") is True
    assert server._is_tool_enabled("read_url") is False


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
    assert "get_calendar_events" in tool_names
    assert "remember_memory" not in tool_names
    assert "get_last_entity_event" not in tool_names
    assert "list_assist_tools" not in tool_names


@pytest.mark.asyncio
async def test_handle_tools_list_uses_cache_for_stable_signature(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Repeated tools/list requests should reuse the cached tool surface when settings are unchanged."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    custom_tools = SimpleNamespace(
        tools={},
        get_cache_signature=lambda: ("stable",),
        get_tool_definitions=lambda: [
            {"name": "search", "description": "search", "inputSchema": {"type": "object", "properties": {}}},
        ],
    )
    server.custom_tools = custom_tools

    result_one = await server.handle_tools_list()
    custom_tools.get_tool_definitions = lambda: (_ for _ in ()).throw(
        AssertionError("tools/list should have been served from cache")
    )
    result_two = await server.handle_tools_list()

    assert result_one == result_two


@pytest.mark.asyncio
async def test_handle_tools_list_invalidates_cache_when_custom_tool_surface_changes(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A changed external custom-tool surface should invalidate the cached tools/list response."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    state = {
        "signature": ("v1",),
        "definitions": [
            {
                "name": "sample_tool_status",
                "description": "sample status",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
    }
    server.custom_tools = SimpleNamespace(
        tools={},
        get_cache_signature=lambda: state["signature"],
        get_tool_definitions=lambda: state["definitions"],
    )

    first = await server.handle_tools_list()

    state["signature"] = ("v2",)
    state["definitions"] = [
        {
            "name": "sample_tool_history",
            "description": "sample history",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]

    second = await server.handle_tools_list()

    first_names = {tool["name"] for tool in first["tools"]}
    second_names = {tool["name"] for tool in second["tools"]}

    assert "sample_tool_status" in first_names
    assert "sample_tool_history" not in first_names
    assert "sample_tool_history" in second_names
    assert "sample_tool_status" not in second_names


@pytest.mark.asyncio
async def test_handle_tool_call_rejects_disabled_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Disabled tools should fail closed even if called directly."""
    system_entry_factory(data={CONF_ENABLE_DEVICE_TOOLS: False})
    server = MCPServer(hass, 8099, profile_entry_factory())

    with pytest.raises(ValueError, match="disabled"):
        await server.handle_tool_call({"name": "discover_devices", "arguments": {}})


@pytest.mark.asyncio
async def test_memory_tools_store_recall_and_forget(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Memory tools should store, search, and delete persisted memories."""
    system_entry_factory(data={CONF_ENABLE_MEMORY_TOOLS: True})
    server = MCPServer(hass, 8099, profile_entry_factory())
    await server.memory_manager.async_initialize()

    remembered = await server.tool_remember_memory(
        {"memory": "Front door code is changing next week", "category": "household"}
    )
    recalled = await server.tool_recall_memories({"query": "front door code"})
    memory_id = recalled["memories"][0]["id"]
    forgotten = await server.tool_forget_memory({"memory_id": memory_id})
    recalled_again = await server.tool_recall_memories({"query": "front door code"})

    assert "Stored memory" in remembered["content"][0]["text"]
    assert recalled["result_count"] == 1
    assert forgotten["deleted_count"] == 1
    assert recalled_again["result_count"] == 0


@pytest.mark.asyncio
async def test_tool_discover_entities_reports_paging_metadata(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Entity discovery responses should tell callers when more results are available."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.discover_entities_page = AsyncMock(
        return_value={
            "items": [
                {
                    "entity_id": "light.kitchen",
                    "name": "Kitchen Light",
                    "state": "on",
                    "area": "Kitchen",
                },
                {
                    "entity_id": "light.pantry",
                    "name": "Pantry Light",
                    "state": "on",
                    "area": "Kitchen",
                },
            ],
            "total_found": 5,
            "returned_count": 2,
            "remaining_count": 3,
            "offset": 0,
            "limit": 2,
            "has_more": True,
            "next_offset": 2,
        }
    )

    result = await server.tool_discover_entities({"domain": "light", "limit": 2})

    assert "Showing 1-2 of 5 entities; 3 more available (next_offset=2):" in (
        result["content"][0]["text"]
    )
    assert result["pagination"]["next_offset"] == 2
    assert len(result["entities"]) == 2


@pytest.mark.asyncio
async def test_tool_discover_devices_reports_paging_metadata(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Device discovery responses should include paging metadata too."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.discover_devices_page = AsyncMock(
        return_value={
            "items": [
                {
                    "device_id": "abc123",
                    "name": "Kitchen Speaker",
                    "entity_count": 3,
                    "domains": ["media_player"],
                    "entities_preview": [{"entity_id": "media_player.kitchen"}],
                }
            ],
            "total_found": 4,
            "returned_count": 1,
            "remaining_count": 3,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "next_offset": 2,
        }
    )

    result = await server.tool_discover_devices({"domain": "media_player", "limit": 1, "offset": 1})

    assert "Showing 2-2 of 4 devices; 2 more available (next_offset=2)" in (
        result["content"][0]["text"]
    )
    assert result["pagination"]["offset"] == 1
    assert result["pagination"]["next_offset"] == 2


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


def test_general_discovery_results_group_by_area_and_sort_names(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """General discovery output should be grouped by area and sorted predictably."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    text = server._format_general_discovery_results(
        [
            {
                "entity_id": "light.primary_bedroom_bedside_right",
                "name": "Primary Bedroom Bedside Lamp: Right",
                "state": "on",
                "area": "Primary Bedroom",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.office_desk_strip",
                "name": "Office Desk Light Strip",
                "state": "on",
                "area": "Office",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.primary_bedroom_bedside_left",
                "name": "Primary Bedroom Bedside Lamp: Left",
                "state": "on",
                "area": "Primary Bedroom",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.office_cans",
                "name": "Office Cans",
                "state": "on",
                "area": "Office",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.primary_bathroom_vanity",
                "name": "Primary Bathroom Vanity",
                "state": "on",
                "area": "Primary Bathroom",
                "floor": "Downstairs",
            },
        ]
    )

    assert "Found 5 entities across 3 groups:" in text
    assert text.index("Office (2):") < text.index("Primary Bathroom (1):")
    assert text.index("Primary Bathroom (1):") < text.index("Primary Bedroom (2):")
    assert text.index("Office Cans") < text.index("Office Desk Light Strip")
    assert text.index("Primary Bedroom Bedside Lamp: Left") < text.index(
        "Primary Bedroom Bedside Lamp: Right"
    )


def test_general_discovery_results_keep_no_area_bucket_last(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Ungrouped entities should appear after named rooms."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    text = server._format_general_discovery_results(
        [
            {
                "entity_id": "light.loft_lamp",
                "name": "Loft Lamp",
                "state": "on",
                "floor": "Upstairs",
            },
            {
                "entity_id": "light.kitchen_pendants",
                "name": "Kitchen Pendants",
                "state": "on",
                "area": "Kitchen",
                "floor": "Downstairs",
            },
        ]
    )

    assert text.index("Kitchen (Downstairs) (1):") < text.index(
        "No area (Upstairs) (1):"
    )


def test_prepare_response_service_data_uses_supported_weather_forecast_type(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather response calls should default to a supported forecast type."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "supported_features": int(
                WeatherEntityFeature.FORECAST_HOURLY
                | WeatherEntityFeature.FORECAST_TWICE_DAILY
            )
        },
    )

    prepared = server._prepare_response_service_data(
        "weather",
        "get_forecasts",
        {},
        resolved_target={"entity_id": ["weather.home"]},
    )

    assert prepared["type"] == "twice_daily"


def test_prepare_response_service_data_adjusts_unsupported_weather_forecast_type(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather response calls should correct unsupported forecast types."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "supported_features": int(
                WeatherEntityFeature.FORECAST_HOURLY
                | WeatherEntityFeature.FORECAST_TWICE_DAILY
            )
        },
    )

    prepared = server._prepare_response_service_data(
        "weather",
        "get_forecasts",
        {"type": "daily"},
        resolved_target={"entity_id": ["weather.home"]},
    )

    assert prepared["type"] == "twice_daily"


@pytest.mark.asyncio
async def test_call_service_with_response_uses_supported_weather_forecast_type(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Weather response-service calls should use the entity's supported type."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "supported_features": int(
                WeatherEntityFeature.FORECAST_HOURLY
                | WeatherEntityFeature.FORECAST_TWICE_DAILY
            )
        },
    )

    server._get_response_service_info = AsyncMock(
        return_value=(
            {
                "fields": {"type": {"required": True}},
                "target": {"entity": {"domain": "weather"}},
            },
            None,
        )
    )
    server.resolve_target = AsyncMock(return_value={"entity_id": ["weather.home"]})
    async_call_mock = AsyncMock(
        return_value={
            "weather.home": {
                "forecast": [
                    {
                        "datetime": "2026-04-13T08:00:00-07:00",
                        "condition": "sunny",
                        "temperature": 72,
                    }
                ]
            }
        }
    )
    monkeypatch.setattr(type(hass.services), "async_call", async_call_mock)

    result = await server.tool_call_service_with_response(
        {
            "domain": "weather",
            "service": "get_forecasts",
            "target": {"entity_id": "weather.home"},
            "data": {},
        }
    )

    async_call_mock.assert_awaited_once()
    service_data = async_call_mock.await_args.kwargs["service_data"]
    assert service_data["type"] == "twice_daily"
    assert "Forecast type used: twice_daily." in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_weather_forecast_discovers_entity_and_summarizes_tomorrow(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather forecast helper should use an HA weather entity instead of acting source-less."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "friendly_name": "Weather",
            "supported_features": int(WeatherEntityFeature.FORECAST_TWICE_DAILY),
        },
    )
    server.discovery.discover_entities = AsyncMock(
        return_value=[
            {
                "entity_id": "weather.home",
                "name": "Weather",
                "forecast_service_supported": True,
            }
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "weather.home": {
                    "forecast": [
                        {
                            "datetime": "2026-04-13T09:00:00-07:00",
                            "condition": "sunny",
                            "temperature": 72,
                            "is_daytime": True,
                        },
                        {
                            "datetime": "2026-04-13T21:00:00-07:00",
                            "condition": "cloudy",
                            "temperature": 61,
                            "is_daytime": False,
                        },
                    ]
                }
            },
        }
    )

    result = await server.tool_get_weather_forecast({"when": "tomorrow"})

    server.tool_call_service_with_response.assert_awaited_once()
    request_args = server.tool_call_service_with_response.await_args.args[0]
    assert request_args["domain"] == "weather"
    assert request_args["service"] == "get_forecasts"
    assert request_args["target"] == {"entity_id": ["weather.home"]}
    assert request_args["data"]["type"] == "twice_daily"
    assert "Tomorrow for Weather:" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_calendar_events_discovers_calendar_and_summarizes_next_event(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Calendar helper should discover a relevant calendar and summarize the next event."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "calendar.mariners_baseball",
        "off",
        {"friendly_name": "Mariners Baseball"},
    )
    server.discovery.discover_entities = AsyncMock(
        return_value=[
            {
                "entity_id": "calendar.mariners_baseball",
                "name": "Mariners Baseball",
            }
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "calendar.mariners_baseball": {
                    "events": [
                        {
                            "summary": "Rangers at Mariners",
                            "start": "2026-04-14T18:40:00-07:00",
                            "end": "2026-04-14T21:40:00-07:00",
                            "location": "T-Mobile Park",
                        }
                    ]
                }
            },
        }
    )

    result = await server.tool_get_calendar_events({"query": "Mariners", "limit": 1})

    server.tool_call_service_with_response.assert_awaited_once()
    request_args = server.tool_call_service_with_response.await_args.args[0]
    assert request_args["domain"] == "calendar"
    assert request_args["service"] == "get_events"
    assert request_args["target"] == {"entity_id": ["calendar.mariners_baseball"]}
    assert "start_date_time" in request_args["data"]
    assert "end_date_time" in request_args["data"]
    assert "Next matching calendar event:" in result["content"][0]["text"]
    assert "Rangers at Mariners" in result["content"][0]["text"]
    assert result["structuredContent"]["selected_calendars"] == [
        {
            "entity_id": "calendar.mariners_baseball",
            "name": "Mariners Baseball",
        }
    ]


@pytest.mark.asyncio
async def test_get_calendar_events_falls_back_to_event_text_search(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """General query text should fall back to event-text matching across calendars."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "calendar.ash_jason",
        "off",
        {"friendly_name": "Ash & Jason"},
    )
    server.discovery.discover_entities = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "entity_id": "calendar.ash_jason",
                    "name": "Ash & Jason",
                }
            ],
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "calendar.ash_jason": {
                    "events": [
                        {
                            "summary": "Dentist cleaning",
                            "description": "Downtown appointment",
                            "start": "2026-04-18T09:00:00-07:00",
                            "end": "2026-04-18T10:00:00-07:00",
                        },
                        {
                            "summary": "Birthday party",
                            "start": "2026-04-19T14:00:00-07:00",
                            "end": "2026-04-19T16:00:00-07:00",
                        },
                    ]
                }
            },
        }
    )

    result = await server.tool_get_calendar_events({"query": "dentist", "limit": 1})

    assert server.discovery.discover_entities.await_count == 2
    first_call = server.discovery.discover_entities.await_args_list[0]
    second_call = server.discovery.discover_entities.await_args_list[1]
    assert first_call.kwargs["name_contains"] == "dentist"
    assert "name_contains" not in second_call.kwargs or second_call.kwargs["name_contains"] is None
    assert "Dentist cleaning" in result["content"][0]["text"]
    assert result["structuredContent"]["event_text"] == "dentist"


@pytest.mark.asyncio
async def test_get_calendar_events_prefers_named_sports_calendar_over_personal_event_match(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Generic words like 'game' should not force a fallback to personal calendar events."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "calendar.mariners_baseball",
        "off",
        {"friendly_name": "Mariners Baseball"},
    )
    hass.states.async_set(
        "calendar.ash_jason",
        "off",
        {"friendly_name": "Ash & Jason"},
    )
    server.discovery.discover_entities = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "entity_id": "calendar.mariners_baseball",
                    "name": "Mariners Baseball",
                }
            ],
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "calendar.mariners_baseball": {
                    "events": [
                        {
                            "summary": "Astros @ Mariners",
                            "start": "2026-04-12T13:10:00-07:00",
                            "end": "2026-04-12T16:10:00-07:00",
                            "location": "T-Mobile Park",
                        }
                    ]
                },
                "calendar.ash_jason": {
                    "events": [
                        {
                            "summary": "Mariners Game <3",
                            "start": "2026-04-18T15:00:00-07:00",
                            "end": "2026-04-18T20:00:00-07:00",
                        }
                    ]
                },
            },
        }
    )

    result = await server.tool_get_calendar_events({"query": "Mariners game", "limit": 1})

    assert server.discovery.discover_entities.await_count == 2
    first_call = server.discovery.discover_entities.await_args_list[0]
    second_call = server.discovery.discover_entities.await_args_list[1]
    assert first_call.kwargs["name_contains"] == "Mariners game"
    assert second_call.kwargs["name_contains"] == "mariners"

    request_args = server.tool_call_service_with_response.await_args.args[0]
    assert request_args["target"] == {"entity_id": ["calendar.mariners_baseball"]}
    assert "Astros @ Mariners" in result["content"][0]["text"]
    assert "Mariners Game <3" not in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolve_weather_forecast_target_uses_generic_ranking(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather target resolution should not rely on install-specific entity names."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.discover_entities = AsyncMock(
        return_value=[
            {
                "entity_id": "weather.weather",
                "name": "Weather",
                "forecast_service_supported": False,
                "forecast_available": False,
            },
            {
                "entity_id": "weather.acme_sky",
                "name": "Acme Sky",
                "forecast_service_supported": True,
                "forecast_available": True,
            },
            {
                "entity_id": "weather.zeta_forecast",
                "name": "Zeta Forecast",
                "forecast_service_supported": True,
                "forecast_available": True,
            },
        ]
    )

    resolved_target, entity_info = await server._resolve_weather_forecast_target()

    assert resolved_target == {"entity_id": ["weather.acme_sky"]}
    assert entity_info == {
        "entity_id": "weather.acme_sky",
        "name": "Acme Sky",
    }


@pytest.mark.asyncio
async def test_observe_action_outcome_confirms_expected_lock_state(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Mechanical actions should confirm completion once the expected state is reached."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "lock.front_door_deadbolt",
        "locked",
        {"friendly_name": "Front Door Deadbolt"},
    )

    result = await server._observe_action_outcome(
        domain="lock",
        service="lock",
        entity_ids=["lock.front_door_deadbolt"],
    )

    assert result["status"] == "confirmed"
    assert result["progress_phrase"] == "locking"
    assert result["state_lines"] == ["  • Front Door Deadbolt: locked"]


@pytest.mark.asyncio
async def test_tool_perform_action_reports_pending_lock_transition(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Slow lock transitions should be reported as pending, not as failures."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.resolve_target = AsyncMock(
        return_value={"entity_id": ["lock.front_door_deadbolt"]}
    )
    server._observe_action_outcome = AsyncMock(
        return_value={
            "status": "pending",
            "progress_phrase": "locking",
            "state_lines": ["  • Front Door Deadbolt: unlocked"],
        }
    )
    async_call_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(type(hass.services), "async_call", async_call_mock)

    result = await server.tool_perform_action(
        {
            "domain": "lock",
            "action": "lock",
            "target": {"entity_id": "lock.front_door_deadbolt"},
            "data": {},
        }
    )

    text = result["content"][0]["text"]
    async_call_mock.assert_awaited_once()
    assert text.startswith("✅ Sent lock.lock")
    assert "may still be locking" in text
    assert "Current states right now:" in text
    assert "Front Door Deadbolt: unlocked" in text
    assert "Successfully executed lock.lock" not in text


def test_history_resolution_prefers_related_contact_sensor_for_open_requests(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Open-history requests should prefer a strongly matching door sensor over a lock."""
    system_entry_factory()
    entry = profile_entry_factory()
    server = MCPServer(hass, 8099, entry)

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    lock_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_lock")},
        name="Front Door Deadbolt",
    )
    contact_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_contact")},
        name="Front Door",
    )

    entity_registry.async_get_or_create(
        "lock",
        "test",
        "front_door_deadbolt",
        suggested_object_id="front_door_deadbolt",
        device_id=lock_device.id,
    )
    entity_registry.async_get_or_create(
        "binary_sensor",
        "test",
        "front_door_contact",
        suggested_object_id="front_door",
        device_id=contact_device.id,
    )

    hass.states.async_set(
        "lock.front_door_deadbolt",
        "locked",
        {"friendly_name": "Front Door Deadbolt"},
    )
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"friendly_name": "Front Door", "device_class": "door"},
    )

    with patch(
        "custom_components.mcp_assist.mcp_server.async_should_expose",
        return_value=True,
    ):
        history_entity_id, resolution_note = server._resolve_history_entity_for_request(
            "lock.front_door_deadbolt",
            None,
            "opened",
        )

    assert history_entity_id == "binary_sensor.front_door"
    assert resolution_note is not None
    assert "binary_sensor.front_door" in resolution_note


@pytest.mark.asyncio
async def test_analyze_history_falls_back_to_related_contact_sensor_when_primary_has_no_matches(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Count analysis should try strong related entities before returning zero matches."""
    system_entry_factory()
    entry = profile_entry_factory()
    server = MCPServer(hass, 8099, entry)

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    lock_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_lock_count")},
        name="Front Door Deadbolt",
    )
    contact_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_contact_count")},
        name="Front Door",
    )

    entity_registry.async_get_or_create(
        "lock",
        "test",
        "front_door_deadbolt_count",
        suggested_object_id="front_door_deadbolt",
        device_id=lock_device.id,
    )
    entity_registry.async_get_or_create(
        "binary_sensor",
        "test",
        "front_door_contact_count",
        suggested_object_id="front_door",
        device_id=contact_device.id,
    )

    hass.states.async_set(
        "lock.front_door_deadbolt",
        "locked",
        {"friendly_name": "Front Door Deadbolt"},
    )
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"friendly_name": "Front Door", "device_class": "door"},
    )

    now = dt_util.utcnow()

    async def fake_fetch(entity_id, *args, **kwargs):
        if entity_id == "lock.front_door_deadbolt":
            return [
                SimpleNamespace(
                    state="locked",
                    last_changed=now - timedelta(hours=12),
                    last_updated=now - timedelta(hours=12),
                )
            ]
        if entity_id == "binary_sensor.front_door":
            return [
                SimpleNamespace(
                    state="off",
                    last_changed=now - timedelta(hours=24),
                    last_updated=now - timedelta(hours=24),
                ),
                SimpleNamespace(
                    state="on",
                    last_changed=now - timedelta(hours=18),
                    last_updated=now - timedelta(hours=18),
                ),
                SimpleNamespace(
                    state="off",
                    last_changed=now - timedelta(hours=17, minutes=55),
                    last_updated=now - timedelta(hours=17, minutes=55),
                ),
            ]
        raise AssertionError(f"Unexpected entity history fetch for {entity_id}")

    server._fetch_entity_history_states = AsyncMock(side_effect=fake_fetch)

    with patch(
        "custom_components.mcp_assist.mcp_server.async_should_expose",
        return_value=True,
    ):
        result = await server.tool_analyze_entity_history(
            {
                "entity_id": "lock.front_door_deadbolt",
                "event": "opened",
                "analysis": "count",
                "hours": 24,
            }
        )

    text = result["content"][0]["text"]

    assert "Using related entity binary_sensor.front_door" in text
    assert "Front Door (binary_sensor.front_door)" in text
    assert "Recorded opened event in the last 24 hours: 1" in text


@pytest.mark.asyncio
async def test_handle_tools_list_includes_media_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The core MCP tool list should include the generic image/media helpers."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}
    tool_map = {tool["name"]: tool for tool in result["tools"]}

    assert "analyze_image" in tool_names
    assert "get_image" in tool_names
    assert "generate_image" in tool_names
    assert tool_map["analyze_image"]["llmDescription"] == (
        "Analyze an image or camera snapshot with the active multimodal model."
    )


@pytest.mark.asyncio
async def test_reload_external_custom_tools_clears_cache_and_notifies_clients(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Reloading external tools should invalidate the cached surface and notify clients."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server._cached_tools_list = [{"name": "stale"}]
    server._cached_tools_signature = ("stale",)
    server.custom_tools = SimpleNamespace(
        reload_tool_packages=AsyncMock(
            return_value={
                "external_custom_tools_enabled": True,
                "external_packages": [],
                "built_in_packages": [],
            }
        )
    )
    server.broadcast_notification = AsyncMock()

    diagnostics = await server.reload_external_custom_tools()

    assert diagnostics["external_custom_tools_enabled"] is True
    assert server._cached_tools_list is None
    assert server._cached_tools_signature is None
    server.broadcast_notification.assert_awaited_once_with(
        "notifications/tools/list_changed"
    )


@pytest.mark.asyncio
async def test_tool_get_image_returns_image_block_from_local_file(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """get_image should return an MCP image block for local image files."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    image_path = tmp_path / "guest_wifi.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsample")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    result = await server.tool_get_image({"image_path": "guest_wifi.png"})

    assert result["isError"] is False
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["mimeType"] == "image/png"
    assert base64.b64decode(result["content"][1]["data"]) == b"\x89PNG\r\n\x1a\nsample"


@pytest.mark.asyncio
async def test_tool_analyze_image_returns_answer_and_optional_image_block(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """analyze_image should return the model answer and optional image content."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    image_path = tmp_path / "driveway.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\ndriveway")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    monkeypatch.setattr(
        server,
        "_analyze_image_with_provider",
        AsyncMock(return_value="A white SUV is in the driveway."),
    )

    result = await server.tool_analyze_image(
        {"image_path": "driveway.png", "include_image": True}
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "A white SUV is in the driveway."
    assert result["content"][1]["type"] == "image"
    assert result["structuredContent"]["source"]["type"] == "image_path"
