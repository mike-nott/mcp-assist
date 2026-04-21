"""Tests for MCP Assist config flow helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import voluptuous as vol
import voluptuous_serialize
from homeassistant.data_entry_flow import FlowResultType, section

from custom_components.mcp_assist.config_flow import (
    ADVANCED_SECTION_KEY,
    CONVERSATION_SECTION_KEY,
    CONTEXT_SECTION_KEY,
    DISCOVERY_SECTION_KEY,
    DISABLE_ASSIST_BRIDGE_FIELD,
    DISABLE_CUSTOM_TOOLS_FIELD,
    DISABLE_DEVICE_FIELD,
    DISABLE_MEMORY_FIELD,
    DISABLE_MUSIC_ASSISTANT_FIELD,
    DISABLE_RECORDER_FIELD,
    DISABLE_RESPONSE_SERVICE_FIELD,
    DISABLE_WEATHER_FORECAST_FIELD,
    MEMORY_SECTION_KEY,
    MCPAssistConfigFlow,
    MCPAssistOptionsFlow,
    MODEL_SECTION_KEY,
    PERFORMANCE_SECTION_KEY,
    PROFILE_SECTION_KEY,
    PROMPTS_SECTION_KEY,
    PROVIDER_SECTION_KEY,
    PROFILE_DISABLE_FIELD_BY_FAMILY,
    STATIC_TOOL_FAMILY_ALPHABETICAL,
    TOOLS_SECTION_KEY,
    _build_profile_tools_section,
    _build_shared_tools_section,
    _apply_profile_tool_disables,
    _infer_prompt_mode,
    _needs_prompt_followup,
    _normalize_shared_tool_inputs,
    _normalize_prompt_inputs,
    validate_allowed_ips,
)
from custom_components.mcp_assist.custom_tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_MAX_ENTITIES_PER_DISCOVERY,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MCP_PORT,
    CONF_LMSTUDIO_URL,
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    CONF_OPENCLAW_SESSION_KEY,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_SEARCH_PROVIDER,
    CONF_SERVER_TYPE,
    CONF_SYSTEM_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    DEFAULT_MEMORY_MAX_TTL_DAYS,
    DEFAULT_TECHNICAL_PROMPT,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_DEFAULT,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENCLAW,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
)

BUILTIN_SPECS = load_builtin_tool_toggle_specs()
BUILTIN_SPEC_BY_PACKAGE = {spec.package_id: spec for spec in BUILTIN_SPECS}
PROFILE_BUILTIN_ORDER = [
    spec.profile_disable_label
    for spec in sorted(BUILTIN_SPECS, key=lambda item: item.profile_disable_label.casefold())
]
SHARED_BUILTIN_ORDER = [
    spec.shared_label
    for spec in sorted(BUILTIN_SPECS, key=lambda item: item.shared_label.casefold())
]


def _builtin_shared_key(package_id: str) -> str:
    """Return the shared form key for a built-in packaged tool."""
    return BUILTIN_SPEC_BY_PACKAGE[package_id].shared_setting_key


def _builtin_profile_key(package_id: str) -> str:
    """Return the profile-disable form key for a built-in packaged tool."""
    return f"disable_{package_id}"


PROFILE_TOOL_ORDER = [
    DISABLE_ASSIST_BRIDGE_FIELD,
    _builtin_profile_key("calculator"),
    DISABLE_CUSTOM_TOOLS_FIELD,
    DISABLE_DEVICE_FIELD,
    DISABLE_MEMORY_FIELD,
    DISABLE_MUSIC_ASSISTANT_FIELD,
    _builtin_profile_key("read_url"),
    DISABLE_RECORDER_FIELD,
    DISABLE_RESPONSE_SERVICE_FIELD,
    _builtin_profile_key("search"),
    _builtin_profile_key("unit_conversion"),
    DISABLE_WEATHER_FORECAST_FIELD,
]

SHARED_TOOL_ORDER = [
    CONF_ENABLE_ASSIST_BRIDGE,
    _builtin_shared_key("calculator"),
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    _builtin_shared_key("read_url"),
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    _builtin_shared_key("search"),
    _builtin_shared_key("unit_conversion"),
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
]


def _section_field_names(form_section: section) -> set[str]:
    """Return normalized field names from a flow section."""
    return {
        getattr(marker, "schema", marker)
        for marker in form_section.schema.schema.keys()
    }


def test_builtin_tool_toggle_specs_include_ui_descriptions() -> None:
    """Built-in packaged-tool metadata should carry UI subtitles for both forms."""
    for spec in BUILTIN_SPECS:
        assert spec.shared_description
        assert spec.profile_disable_description


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


def test_normalize_prompt_inputs_for_openclaw_forces_default_system_prompt() -> None:
    """OpenClaw should always ignore custom system prompts."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_SYSTEM_PROMPT: "custom",
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: "keep this one",
        },
        server_type=SERVER_TYPE_OPENCLAW,
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
            DISABLE_CUSTOM_TOOLS_FIELD: True,
            _builtin_profile_key("calculator"): True,
            _builtin_profile_key("search"): True,
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_PROFILE_ENABLE_DEVICE_TOOLS] is False
    assert normalized[CONF_PROFILE_ENABLE_ASSIST_BRIDGE] is False
    assert normalized[CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS] is False
    assert normalized["profile_enable_calculator_tools"] is False
    assert normalized["profile_enable_search_tool"] is False


def test_apply_profile_tool_disables_leaves_unchecked_tools_inherited() -> None:
    """Unchecked profile tool toggles should fall back to shared settings."""
    normalized = _apply_profile_tool_disables(
        {
            DISABLE_DEVICE_FIELD: False,
            DISABLE_ASSIST_BRIDGE_FIELD: False,
            DISABLE_CUSTOM_TOOLS_FIELD: False,
            _builtin_profile_key("calculator"): False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: False,
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            "profile_enable_calculator_tools": False,
        },
        BUILTIN_SPECS,
    )

    assert CONF_PROFILE_ENABLE_DEVICE_TOOLS not in normalized
    assert CONF_PROFILE_ENABLE_ASSIST_BRIDGE not in normalized
    assert CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS not in normalized
    assert "profile_enable_calculator_tools" not in normalized


def test_normalize_shared_tool_inputs_maps_built_in_fields_to_setting_keys() -> None:
    """Built-in toggle fields should store real shared setting keys."""
    normalized = _normalize_shared_tool_inputs(
        {
            _builtin_shared_key("search"): True,
            _builtin_shared_key("read_url"): True,
            CONF_SEARCH_PROVIDER: "",
        },
        BUILTIN_SPECS,
    )

    assert normalized["enable_search_tool"] is True
    assert normalized["enable_read_url_tool"] is True
    assert normalized[CONF_SEARCH_PROVIDER] == "duckduckgo"


def test_normalize_shared_tool_inputs_accepts_legacy_built_in_labels() -> None:
    """Older in-flight form data using built-in labels should still normalize."""
    normalized = _normalize_shared_tool_inputs(
        {
            BUILTIN_SPEC_BY_PACKAGE["search"].shared_label: True,
            BUILTIN_SPEC_BY_PACKAGE["read_url"].shared_label: True,
            CONF_SEARCH_PROVIDER: "",
        },
        BUILTIN_SPECS,
    )

    assert normalized["enable_search_tool"] is True
    assert normalized["enable_read_url_tool"] is True
    assert BUILTIN_SPEC_BY_PACKAGE["search"].shared_label not in normalized
    assert BUILTIN_SPEC_BY_PACKAGE["read_url"].shared_label not in normalized


def test_normalize_shared_tool_inputs_infers_provider_from_legacy_web_search() -> None:
    """Legacy web-search enablement should still infer a real provider."""
    normalized = _normalize_shared_tool_inputs(
        {
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "",
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_SEARCH_PROVIDER] == "duckduckgo"


def test_normalize_shared_tool_inputs_clamps_memory_ttls() -> None:
    """Shared memory TTL settings should be coerced into a safe valid range."""
    normalized = _normalize_shared_tool_inputs(
        {
            CONF_MEMORY_MAX_TTL_DAYS: 3,
            CONF_MEMORY_DEFAULT_TTL_DAYS: 99,
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_MEMORY_MAX_TTL_DAYS] == 3
    assert normalized[CONF_MEMORY_DEFAULT_TTL_DAYS] == 3

    fallback = _normalize_shared_tool_inputs(
        {
            CONF_MEMORY_MAX_TTL_DAYS: "oops",
            CONF_MEMORY_DEFAULT_TTL_DAYS: "nope",
        },
        BUILTIN_SPECS,
    )

    assert fallback[CONF_MEMORY_MAX_TTL_DAYS] == DEFAULT_MEMORY_MAX_TTL_DAYS
    assert fallback[CONF_MEMORY_DEFAULT_TTL_DAYS] == DEFAULT_MEMORY_DEFAULT_TTL_DAYS


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
    assert section_keys == PROFILE_TOOL_ORDER
    assert all(value is bool for value in tools_section.schema.schema.values())
    marker_by_key = {
        getattr(marker, "schema", marker): marker
        for marker in tools_section.schema.schema.keys()
    }
    assert marker_by_key[_builtin_profile_key("calculator")].description is None
    assert marker_by_key[_builtin_profile_key("search")].description is None


async def test_shared_mcp_step_groups_context_discovery_and_tools(hass) -> None:
    """Shared MCP settings should group context, discovery, and tools fields into sections."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}

    result = await flow.async_step_mcp_server()

    context_section = result["data_schema"].schema[CONTEXT_SECTION_KEY]
    discovery_section = result["data_schema"].schema[DISCOVERY_SECTION_KEY]
    memory_section = result["data_schema"].schema[MEMORY_SECTION_KEY]
    tools_section = result["data_schema"].schema[TOOLS_SECTION_KEY]

    assert isinstance(context_section, section)
    assert isinstance(discovery_section, section)
    assert isinstance(memory_section, section)
    assert isinstance(tools_section, section)

    context_keys = {
        getattr(key, "schema", key) for key in context_section.schema.schema.keys()
    }
    discovery_keys = {
        getattr(key, "schema", key) for key in discovery_section.schema.schema.keys()
    }
    memory_keys = {
        getattr(key, "schema", key) for key in memory_section.schema.schema.keys()
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
    assert memory_keys == {
        CONF_MEMORY_DEFAULT_TTL_DAYS,
        CONF_MEMORY_MAX_TTL_DAYS,
        CONF_MEMORY_MAX_ITEMS,
    }
    assert tool_keys == [
        *SHARED_TOOL_ORDER,
        CONF_SEARCH_PROVIDER,
        CONF_BRAVE_API_KEY,
    ]
    tool_markers = {
        getattr(marker, "schema", marker): marker
        for marker in tools_section.schema.schema.keys()
    }
    external_default = tool_markers[CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS].default
    assert external_default() is False if callable(external_default) else external_default is False
    assert tool_markers[_builtin_shared_key("calculator")].description is None
    assert tool_markers[_builtin_shared_key("read_url")].description is None


def test_built_in_tool_checkboxes_rely_on_translation_subtitles() -> None:
    """Built-in packaged tool checkboxes should not override translated subtitles inline."""
    shared_section = _build_shared_tools_section({}, BUILTIN_SPECS)
    profile_section = _build_profile_tools_section({}, BUILTIN_SPECS, {}, {})

    shared_names = {
        _builtin_shared_key("calculator"),
        _builtin_shared_key("read_url"),
        _builtin_shared_key("search"),
        _builtin_shared_key("unit_conversion"),
    }
    profile_names = {
        _builtin_profile_key("calculator"),
        _builtin_profile_key("read_url"),
        _builtin_profile_key("search"),
        _builtin_profile_key("unit_conversion"),
    }

    shared_checkbox_fields = {
        marker: value
        for marker, value in shared_section.schema.schema.items()
        if getattr(marker, "schema", marker) in shared_names
    }
    profile_checkbox_fields = {
        marker: value
        for marker, value in profile_section.schema.schema.items()
        if getattr(marker, "schema", marker) in profile_names
    }

    shared_serialized = voluptuous_serialize.convert(vol.Schema(shared_checkbox_fields))
    profile_serialized = voluptuous_serialize.convert(vol.Schema(profile_checkbox_fields))

    shared_by_name = {item["name"]: item for item in shared_serialized}
    profile_by_name = {item["name"]: item for item in profile_serialized}

    for name in shared_names:
        assert "description" not in shared_by_name[name]

    for name in profile_names:
        assert "description" not in profile_by_name[name]


def test_optional_tool_family_checkbox_sets_stay_in_sync() -> None:
    """Every optional tool family should exist in both shared and profile checkbox builders."""
    assert set(STATIC_TOOL_FAMILY_ALPHABETICAL) <= set(TOOL_FAMILY_PROFILE_SETTINGS)
    assert set(STATIC_TOOL_FAMILY_ALPHABETICAL) <= set(TOOL_FAMILY_SHARED_SETTINGS)
    assert set(STATIC_TOOL_FAMILY_ALPHABETICAL) == set(PROFILE_DISABLE_FIELD_BY_FAMILY)


def test_tool_checkbox_translations_cover_all_declared_tool_fields() -> None:
    """Static shared/profile tool checkbox fields should still have labels and descriptions."""
    strings = json.loads(
        Path("custom_components/mcp_assist/strings.json").read_text(encoding="utf-8")
    )
    for root, advanced_step in (("config", "advanced"), ("options", "init")):
        advanced_tools = strings[root]["step"][advanced_step]["sections"][TOOLS_SECTION_KEY]
        shared_tools = strings[root]["step"]["mcp_server"]["sections"][TOOLS_SECTION_KEY]

        expected_profile_fields = {
            PROFILE_DISABLE_FIELD_BY_FAMILY[family]
            for family in STATIC_TOOL_FAMILY_ALPHABETICAL
        }
        expected_shared_fields = {
            TOOL_FAMILY_SHARED_SETTINGS[family][0]
            for family in STATIC_TOOL_FAMILY_ALPHABETICAL
        }
        expected_profile_fields.update(
            _builtin_profile_key(spec.package_id) for spec in BUILTIN_SPECS
        )
        expected_shared_fields.update(
            _builtin_shared_key(spec.package_id) for spec in BUILTIN_SPECS
        )

        assert expected_profile_fields <= set(advanced_tools["data"])
        assert expected_profile_fields <= set(advanced_tools["data_description"])
        assert expected_shared_fields <= set(shared_tools["data"])
        assert expected_shared_fields <= set(shared_tools["data_description"])


def test_provider_section_translations_cover_provider_specific_fields() -> None:
    """Provider-only settings should have section translations in both config and options flows."""
    strings = json.loads(
        Path("custom_components/mcp_assist/strings.json").read_text(encoding="utf-8")
    )

    expected_provider_fields = {
        CONF_OLLAMA_KEEP_ALIVE,
        CONF_OLLAMA_NUM_CTX,
        CONF_OPENCLAW_SESSION_KEY,
    }

    for root, step in (("config", "advanced"), ("options", "init")):
        provider_section = strings[root]["step"][step]["sections"][PROVIDER_SECTION_KEY]
        assert expected_provider_fields <= set(provider_section["data"])
        assert expected_provider_fields <= set(provider_section["data_description"])


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


async def test_options_step_for_ollama_keeps_provider_fields_in_provider_section(
    hass, profile_entry_factory
) -> None:
    """Ollama provider-only settings should live in the provider section, not advanced."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
            CONF_LMSTUDIO_URL: "http://localhost:11434",
        }
    )
    flow.handler = entry.entry_id

    with patch(
        "custom_components.mcp_assist.config_flow.fetch_models_from_lmstudio",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_init()

    provider_section = result["data_schema"].schema[PROVIDER_SECTION_KEY]
    advanced_section = result["data_schema"].schema[ADVANCED_SECTION_KEY]

    assert isinstance(provider_section, section)
    assert _section_field_names(provider_section) == {
        CONF_OLLAMA_NUM_CTX,
        CONF_OLLAMA_KEEP_ALIVE,
    }
    assert CONF_OLLAMA_NUM_CTX not in _section_field_names(advanced_section)
    assert CONF_OLLAMA_KEEP_ALIVE not in _section_field_names(advanced_section)


async def test_options_step_for_openclaw_hides_model_prompts_and_uses_provider_section(
    hass, profile_entry_factory
) -> None:
    """OpenClaw options should hide model/prompts and keep the session key in provider settings."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        title="OpenClaw - Test Profile",
        unique_id="mcp_assist_openclaw_test_profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENCLAW},
    )
    flow.handler = entry.entry_id

    result = await flow.async_step_init()

    top_level_keys = set(result["data_schema"].schema.keys())
    provider_section = result["data_schema"].schema[PROVIDER_SECTION_KEY]
    advanced_section = result["data_schema"].schema[ADVANCED_SECTION_KEY]

    assert MODEL_SECTION_KEY not in top_level_keys
    assert PROMPTS_SECTION_KEY not in top_level_keys
    assert PROVIDER_SECTION_KEY in top_level_keys
    assert isinstance(provider_section, section)
    assert _section_field_names(provider_section) == {CONF_OPENCLAW_SESSION_KEY}
    assert CONF_OPENCLAW_SESSION_KEY not in _section_field_names(advanced_section)
