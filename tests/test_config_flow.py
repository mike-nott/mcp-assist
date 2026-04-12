"""Tests for MCP Assist config flow helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import voluptuous as vol
from homeassistant.data_entry_flow import FlowResultType, section

from custom_components.mcp_assist.config_flow import (
    ADVANCED_SECTION_KEY,
    CONVERSATION_SECTION_KEY,
    CONTEXT_SECTION_KEY,
    DISCOVERY_SECTION_KEY,
    DISABLE_ASSIST_BRIDGE_FIELD,
    DISABLE_CALCULATOR_FIELD,
    DISABLE_DEVICE_FIELD,
    DISABLE_MUSIC_ASSISTANT_FIELD,
    DISABLE_RECORDER_FIELD,
    DISABLE_RESPONSE_SERVICE_FIELD,
    DISABLE_UNIT_CONVERSION_FIELD,
    DISABLE_WEB_SEARCH_FIELD,
    DISABLE_WEATHER_FORECAST_FIELD,
    MCPAssistConfigFlow,
    MCPAssistOptionsFlow,
    MODEL_SECTION_KEY,
    PERFORMANCE_SECTION_KEY,
    PROFILE_SECTION_KEY,
    PROMPTS_SECTION_KEY,
    TOOLS_SECTION_KEY,
    _apply_profile_tool_disables,
    _infer_prompt_mode,
    _needs_prompt_followup,
    _normalize_prompt_inputs,
    validate_allowed_ips,
)
from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_MAX_ENTITIES_PER_DISCOVERY,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_MCP_PORT,
    CONF_LMSTUDIO_URL,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_SEARCH_PROVIDER,
    CONF_SERVER_TYPE,
    CONF_SYSTEM_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    DEFAULT_TECHNICAL_PROMPT,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_DEFAULT,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_MOLTBOT,
)


def test_infer_prompt_mode_defaults_when_prompt_matches_builtin() -> None:
    """Legacy prompt storage should infer default mode when the prompt matches the built-in text."""
    assert _infer_prompt_mode(None, "builtin prompt", "builtin prompt") == PROMPT_MODE_DEFAULT
    assert _infer_prompt_mode(None, "custom prompt", "builtin prompt") == PROMPT_MODE_CUSTOM


def test_normalize_prompt_inputs_drops_default_prompt_text() -> None:
    """Blank prompt overrides should fall back to built-in defaults."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT: "",
            CONF_TECHNICAL_PROMPT: "   ",
        },
        server_type="ollama",
        default_system_prompt="builtin system",
    )

    assert CONF_SYSTEM_PROMPT not in normalized
    assert CONF_TECHNICAL_PROMPT not in normalized
    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert normalized[CONF_TECHNICAL_PROMPT_MODE] == PROMPT_MODE_DEFAULT


def test_normalize_prompt_inputs_marks_nonblank_prompts_as_custom() -> None:
    """Nonblank prompt overrides should be stored as custom values."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT: "Be formal",
            CONF_TECHNICAL_PROMPT: "Always inspect attributes",
        },
        server_type="ollama",
        default_system_prompt="builtin system",
    )

    assert normalized[CONF_SYSTEM_PROMPT] == "Be formal"
    assert normalized[CONF_TECHNICAL_PROMPT] == "Always inspect attributes"
    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_CUSTOM
    assert normalized[CONF_TECHNICAL_PROMPT_MODE] == PROMPT_MODE_CUSTOM


def test_normalize_prompt_inputs_for_moltbot_forces_default_system_prompt() -> None:
    """Moltbot should always ignore custom system prompts."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_SYSTEM_PROMPT: "custom",
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: "keep this one",
        },
        server_type=SERVER_TYPE_MOLTBOT,
        default_system_prompt="builtin system",
    )

    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert CONF_SYSTEM_PROMPT not in normalized
    assert normalized[CONF_TECHNICAL_PROMPT] == "keep this one"


def test_normalize_prompt_inputs_treats_builtin_prompt_text_as_default() -> None:
    """Prompt text matching the built-in default should not be stored as custom."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT: "builtin system",
            CONF_TECHNICAL_PROMPT: DEFAULT_TECHNICAL_PROMPT,
        },
        server_type="ollama",
        default_system_prompt="builtin system",
    )

    assert CONF_SYSTEM_PROMPT not in normalized
    assert CONF_TECHNICAL_PROMPT not in normalized
    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert normalized[CONF_TECHNICAL_PROMPT_MODE] == PROMPT_MODE_DEFAULT


def test_needs_prompt_followup_when_switching_prompt_visibility() -> None:
    """Prompt followup is disabled because the fields are always visible."""
    assert _needs_prompt_followup({}, server_type="ollama") is False


def test_validate_allowed_ips_accepts_ips_and_cidr_ranges() -> None:
    """Allowed IP parsing should accept both addresses and CIDR entries."""
    assert validate_allowed_ips("192.168.1.10,10.0.0.0/24") == (True, "")


def test_validate_allowed_ips_rejects_invalid_values() -> None:
    """Allowed IP parsing should reject malformed values."""
    is_valid, message = validate_allowed_ips("192.168.1.10,not-an-ip")

    assert is_valid is False
    assert "not-an-ip" in message


async def test_system_flow_creates_shared_config_entry(hass) -> None:
    """The system-source config flow should create the shared config entry without UI steps."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "system"}

    result = await flow.async_step_system({CONF_MCP_PORT: 7788})

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Shared MCP Server Settings"
    assert result["data"][CONF_MCP_PORT] == 7788


async def test_model_step_always_shows_prompt_fields_without_mode_dropdowns(hass) -> None:
    """The model step should always expose prompt textareas directly."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}
    flow.step2_data = {CONF_LMSTUDIO_URL: "http://localhost:11434"}

    with patch(
        "custom_components.mcp_assist.config_flow.fetch_models_from_lmstudio",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_model()

    assert MODEL_SECTION_KEY in result["data_schema"].schema
    assert PROMPTS_SECTION_KEY in result["data_schema"].schema

    prompts_section = result["data_schema"].schema[PROMPTS_SECTION_KEY]
    assert isinstance(prompts_section, section)

    prompt_keys = {
        getattr(key, "schema", key) for key in prompts_section.schema.schema.keys()
    }
    assert CONF_SYSTEM_PROMPT in prompt_keys
    assert CONF_TECHNICAL_PROMPT in prompt_keys


async def test_model_step_prompt_overrides_are_optional(hass) -> None:
    """Prompt fields should be optional and prefilled with the effective prompts."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}
    flow.step2_data = {CONF_LMSTUDIO_URL: "http://localhost:11434"}

    with patch(
        "custom_components.mcp_assist.config_flow.fetch_models_from_lmstudio",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_model()

    prompts_section = result["data_schema"].schema[PROMPTS_SECTION_KEY]
    marker_by_key = {
        getattr(marker, "schema", marker): marker
        for marker in prompts_section.schema.schema.keys()
    }

    assert isinstance(marker_by_key[CONF_SYSTEM_PROMPT], vol.Optional)
    assert isinstance(marker_by_key[CONF_TECHNICAL_PROMPT], vol.Optional)
    assert marker_by_key[CONF_SYSTEM_PROMPT].description["suggested_value"]
    assert marker_by_key[CONF_TECHNICAL_PROMPT].description["suggested_value"]


def test_apply_profile_tool_disables_marks_checked_tools_disabled() -> None:
    """Checked profile tool toggles should store disabled flags."""
    normalized = _apply_profile_tool_disables(
        {
            DISABLE_DEVICE_FIELD: True,
            DISABLE_ASSIST_BRIDGE_FIELD: True,
        }
    )

    assert normalized[CONF_PROFILE_ENABLE_DEVICE_TOOLS] is False
    assert normalized[CONF_PROFILE_ENABLE_ASSIST_BRIDGE] is False


def test_apply_profile_tool_disables_leaves_unchecked_tools_inherited() -> None:
    """Unchecked profile tool toggles should fall back to shared settings."""
    normalized = _apply_profile_tool_disables(
        {
            DISABLE_DEVICE_FIELD: False,
            DISABLE_ASSIST_BRIDGE_FIELD: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: False,
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
        }
    )

    assert CONF_PROFILE_ENABLE_DEVICE_TOOLS not in normalized
    assert CONF_PROFILE_ENABLE_ASSIST_BRIDGE not in normalized


async def test_advanced_step_groups_profile_tools_into_checkbox_section(hass) -> None:
    """Advanced settings should expose per-profile tool disable checkboxes."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}

    result = await flow.async_step_advanced()

    assert CONVERSATION_SECTION_KEY in result["data_schema"].schema
    assert PERFORMANCE_SECTION_KEY in result["data_schema"].schema
    tools_section = result["data_schema"].schema[TOOLS_SECTION_KEY]
    assert isinstance(tools_section, section)

    section_keys = [
        getattr(key, "schema", key) for key in tools_section.schema.schema.keys()
    ]
    assert section_keys == [
        DISABLE_ASSIST_BRIDGE_FIELD,
        DISABLE_CALCULATOR_FIELD,
        DISABLE_DEVICE_FIELD,
        DISABLE_MUSIC_ASSISTANT_FIELD,
        DISABLE_RECORDER_FIELD,
        DISABLE_RESPONSE_SERVICE_FIELD,
        DISABLE_UNIT_CONVERSION_FIELD,
        DISABLE_WEATHER_FORECAST_FIELD,
        DISABLE_WEB_SEARCH_FIELD,
    ]
    assert all(value is bool for value in tools_section.schema.schema.values())


async def test_shared_mcp_step_groups_context_discovery_and_tools(hass) -> None:
    """Shared MCP settings should group context, discovery, and tools fields into sections."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}

    result = await flow.async_step_mcp_server()

    context_section = result["data_schema"].schema[CONTEXT_SECTION_KEY]
    discovery_section = result["data_schema"].schema[DISCOVERY_SECTION_KEY]
    tools_section = result["data_schema"].schema[TOOLS_SECTION_KEY]

    assert isinstance(context_section, section)
    assert isinstance(discovery_section, section)
    assert isinstance(tools_section, section)

    context_keys = {
        getattr(key, "schema", key) for key in context_section.schema.schema.keys()
    }
    discovery_keys = {
        getattr(key, "schema", key) for key in discovery_section.schema.schema.keys()
    }
    tool_keys = [
        getattr(key, "schema", key) for key in tools_section.schema.schema.keys()
    ]

    assert context_keys == {
        CONF_INCLUDE_CURRENT_USER,
        CONF_INCLUDE_HOME_LOCATION,
    }
    assert discovery_keys == {
        CONF_ENABLE_GAP_FILLING,
        CONF_MAX_ENTITIES_PER_DISCOVERY,
    }
    assert tool_keys == [
        CONF_ENABLE_ASSIST_BRIDGE,
        CONF_ENABLE_CALCULATOR_TOOLS,
        CONF_ENABLE_DEVICE_TOOLS,
        CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
        CONF_ENABLE_RECORDER_TOOLS,
        CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
        CONF_ENABLE_UNIT_CONVERSION_TOOLS,
        CONF_ENABLE_WEATHER_FORECAST_TOOL,
        CONF_ENABLE_WEB_SEARCH,
        CONF_SEARCH_PROVIDER,
        CONF_BRAVE_API_KEY,
    ]


async def test_options_step_groups_profile_settings_into_sections(
    hass, profile_entry_factory
) -> None:
    """Options flow should organize profile settings into clear sections."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id

    with patch(
        "custom_components.mcp_assist.config_flow.fetch_models_from_lmstudio",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_init()

    top_level_keys = set(result["data_schema"].schema.keys())
    assert PROFILE_SECTION_KEY in top_level_keys
    assert MODEL_SECTION_KEY in top_level_keys
    assert PROMPTS_SECTION_KEY in top_level_keys
    assert CONVERSATION_SECTION_KEY in top_level_keys
    assert ADVANCED_SECTION_KEY in top_level_keys
    assert TOOLS_SECTION_KEY in top_level_keys
