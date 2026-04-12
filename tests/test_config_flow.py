"""Tests for MCP Assist config flow helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.data_entry_flow import FlowResultType, section

from custom_components.mcp_assist.config_flow import (
    ENABLED_TOOLS_FIELD,
    MCPAssistConfigFlow,
    TOOLS_SECTION_KEY,
    _apply_tool_family_selection,
    _infer_prompt_mode,
    _needs_prompt_followup,
    _normalize_prompt_inputs,
    validate_allowed_ips,
)
from custom_components.mcp_assist.const import (
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_MCP_PORT,
    CONF_LMSTUDIO_URL,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_SERVER_TYPE,
    CONF_SYSTEM_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    DOMAIN,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_DEFAULT,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_MOLTBOT,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
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
    )

    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert CONF_SYSTEM_PROMPT not in normalized
    assert normalized[CONF_TECHNICAL_PROMPT] == "keep this one"


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

    schema_keys = {
        getattr(key, "schema", key) for key in result["data_schema"].schema.keys()
    }

    assert CONF_SYSTEM_PROMPT in schema_keys
    assert CONF_TECHNICAL_PROMPT in schema_keys
    assert CONF_SYSTEM_PROMPT_MODE not in schema_keys
    assert CONF_TECHNICAL_PROMPT_MODE not in schema_keys


def test_apply_tool_family_selection_expands_profile_multiselect() -> None:
    """The tools multiselect should expand into stored per-profile booleans."""
    normalized = _apply_tool_family_selection(
        {ENABLED_TOOLS_FIELD: ["device"]},
        TOOL_FAMILY_PROFILE_SETTINGS,
    )

    assert normalized[CONF_PROFILE_ENABLE_DEVICE_TOOLS] is True
    assert normalized[CONF_PROFILE_ENABLE_ASSIST_BRIDGE] is False


def test_apply_tool_family_selection_expands_shared_multiselect() -> None:
    """The shared tools multiselect should expand into shared booleans."""
    normalized = _apply_tool_family_selection(
        {ENABLED_TOOLS_FIELD: ["assist_bridge"]},
        TOOL_FAMILY_SHARED_SETTINGS,
    )

    assert normalized[CONF_ENABLE_ASSIST_BRIDGE] is True


async def test_advanced_step_groups_profile_tools_into_multiselect_section(hass) -> None:
    """Advanced settings should group optional tools into one multiselect section."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}

    result = await flow.async_step_advanced()

    tools_section = result["data_schema"].schema[TOOLS_SECTION_KEY]
    assert isinstance(tools_section, section)

    section_keys = {
        getattr(key, "schema", key) for key in tools_section.schema.schema.keys()
    }
    assert ENABLED_TOOLS_FIELD in section_keys

    selector = next(iter(tools_section.schema.schema.values()))
    assert selector.config["multiple"] is True
