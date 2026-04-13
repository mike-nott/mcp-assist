"""LM Studio MCP conversation agent."""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation import chat_log
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    intent,
    area_registry as ar,
    device_registry as dr,
    llm,
    chat_session,
)
from homeassistant.util import dt as dt_util

from .localization import get_language_instruction
from .const import (
    DOMAIN,
    CONF_PROFILE_NAME,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_DEBUG_MODE,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_FOLLOW_UP_MODE,
    CONF_RESPONSE_MODE,
    CONF_MAX_HISTORY,
    CONF_SERVER_TYPE,
    CONF_API_KEY,
    CONF_CONTROL_HA,
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
    CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_FOLLOW_UP_PHRASES,
    CONF_END_WORDS,
    CONF_CLEAN_RESPONSES,
    CONF_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    PROMPT_MODE_DEFAULT,
    PROMPT_MODE_CUSTOM,
    DEFAULT_DEBUG_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MCP_PORT,
    DEFAULT_SERVER_TYPE,
    DEFAULT_API_KEY,
    DEFAULT_CONTROL_HA,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_FOLLOW_UP_PHRASES,
    DEFAULT_END_WORDS,
    DEFAULT_CLEAN_RESPONSES,
    DEFAULT_TIMEOUT,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_INCLUDE_CURRENT_USER,
    DEFAULT_INCLUDE_HOME_LOCATION,
    DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS,
    RESPONSE_MODE_INSTRUCTIONS,
    DEVICE_TECHNICAL_INSTRUCTIONS,
    RESPONSE_SERVICE_TECHNICAL_INSTRUCTIONS,
    RECORDER_ANALYSIS_TECHNICAL_INSTRUCTIONS,
    MEMORY_TECHNICAL_INSTRUCTIONS,
    ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS,
    CALCULATOR_TECHNICAL_INSTRUCTIONS,
    UNIT_CONVERSION_TECHNICAL_INSTRUCTIONS,
    MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_MOLTBOT,
    SERVER_TYPE_VLLM,
    OPENAI_BASE_URL,
    GEMINI_BASE_URL,
    ANTHROPIC_BASE_URL,
    OPENROUTER_BASE_URL,
    TOOL_FAMILY_EXTERNAL_CUSTOM,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
    get_optional_tool_family,
)
from .conversation_history import ConversationHistory

_LOGGER = logging.getLogger(__name__)

MCP_TOOL_CACHE_TTL_SECONDS = 30.0
MAX_TOOL_RESULT_CHARS = 8000
MAX_TOOL_RESULT_LINES = 120


class MCPAssistConversationEntity(ConversationEntity):
    """MCP Assist conversation entity with multi-provider support."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the MCP Assist conversation entity."""
        super().__init__()

        self.hass = hass
        self.entry = entry
        self.history = ConversationHistory()
        self._current_chat_log = None  # ChatLog for debug view tracking
        self._cached_llm_tools: list[dict[str, Any]] | None = None
        self._cached_llm_tools_key: tuple[Any, ...] | None = None
        self._cached_llm_tools_fetched_at = 0.0

        # Entity attributes
        profile_name = entry.data.get("profile_name", "MCP Assist")

        # Static configuration (doesn't change)
        data = entry.data
        self.server_type = data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        # Server type display names
        server_display_names = {
            SERVER_TYPE_LMSTUDIO: "LM Studio",
            SERVER_TYPE_LLAMACPP: "llama.cpp",
            SERVER_TYPE_OLLAMA: "Ollama",
            SERVER_TYPE_OPENAI: "OpenAI",
            SERVER_TYPE_GEMINI: "Gemini",
            SERVER_TYPE_ANTHROPIC: "Claude",
            SERVER_TYPE_OPENROUTER: "OpenRouter",
            SERVER_TYPE_MOLTBOT: "Moltbot",
            SERVER_TYPE_VLLM: "vLLM",
        }
        server_display_name = server_display_names.get(
            self.server_type, self.server_type
        )

        # Set entity attributes
        self._attr_unique_id = entry.entry_id
        self._attr_name = f"{server_display_name} - {profile_name}"
        self._attr_suggested_object_id = (
            f"{self.server_type}_{profile_name.lower().replace(' ', '_')}"
        )

        # Device info
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"{server_display_name} - {profile_name}",
            manufacturer="MCP Assist",
            model=server_display_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        # Set base URL based on server type
        # OpenAI now reads from config (like local servers) instead of static constant
        if self.server_type == SERVER_TYPE_OPENAI:
            # Read URL from config (defaults to official OpenAI URL if not set)
            # Uses same CONF_LMSTUDIO_URL field as local servers
            url = self.entry.options.get(
                CONF_LMSTUDIO_URL,
                self.entry.data.get(CONF_LMSTUDIO_URL, OPENAI_BASE_URL)
            ).rstrip("/")
            self.base_url = url
            _LOGGER.info("🌐 AGENT: Using OpenAI-compatible URL: %s", self.base_url)
        elif self.server_type == SERVER_TYPE_GEMINI:
            self.base_url = GEMINI_BASE_URL
        elif self.server_type == SERVER_TYPE_ANTHROPIC:
            self.base_url = ANTHROPIC_BASE_URL
        elif self.server_type == SERVER_TYPE_OPENROUTER:
            self.base_url = OPENROUTER_BASE_URL
        else:
            # LM Studio or Ollama - URL can change, so make it a property below
            pass

        # All other config values are now dynamic properties (see @property methods below)

        # Log the actual configuration being used
        if self.debug_mode:
            _LOGGER.debug(f"🔍 Server Type: {self.server_type}")
            _LOGGER.debug(f"🔍 Base URL: {self.base_url_dynamic}")
            _LOGGER.debug("🔍 Debug mode: ON")
            _LOGGER.debug(f"🔍 Max iterations: {self.max_iterations}")

        _LOGGER.info(
            "MCP Assist Agent initialized - Server: %s, Model: %s, MCP Port: %d, URL: %s",
            self.server_type,
            self.model_name,
            self.mcp_port,
            self.base_url_dynamic,
        )

    def _get_shared_setting(self, key: str, default: Any) -> Any:
        """Get a shared setting from system entry with fallback to profile entry."""
        # Import here to avoid circular dependency
        from . import get_system_entry

        # Try to get from system entry first
        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        # Fallback to profile entry for backward compatibility
        value = self.entry.options.get(key, self.entry.data.get(key))
        if value is not None:
            return value

        # Return default
        return default

    def _get_profile_setting(self, key: str, default: Any) -> Any:
        """Get a profile-specific setting from this conversation profile."""
        value = self.entry.options.get(key, self.entry.data.get(key))
        if value is not None:
            return value
        return default

    def _is_optional_tool_family_enabled(self, family: str) -> bool:
        """Return whether an optional tool family is enabled for this profile."""
        if family == "unit_conversion":
            shared_enabled = self._get_shared_setting(
                CONF_ENABLE_UNIT_CONVERSION_TOOLS,
                None,
            )
            if shared_enabled is None:
                shared_enabled = self._get_shared_setting(
                    CONF_ENABLE_CALCULATOR_TOOLS,
                    DEFAULT_ENABLE_CALCULATOR_TOOLS,
                )

            profile_enabled = self._get_profile_setting(
                CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
                None,
            )
            if profile_enabled is None:
                profile_enabled = self._get_profile_setting(
                    CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
                    DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS,
                )

            return bool(shared_enabled and profile_enabled)

        shared_key, shared_default = TOOL_FAMILY_SHARED_SETTINGS[family]
        profile_key, profile_default = TOOL_FAMILY_PROFILE_SETTINGS[family]
        return bool(
            self._get_shared_setting(shared_key, shared_default)
            and self._get_profile_setting(profile_key, profile_default)
        )

    def _is_tool_enabled_for_profile(self, tool_name: str) -> bool:
        """Return whether a tool should be visible to this profile."""
        family = get_optional_tool_family(tool_name)
        if family is not None:
            return self._is_optional_tool_family_enabled(family)
        if self._is_external_custom_tool(tool_name):
            return self.external_custom_tools_enabled
        return True

    # Dynamic configuration properties - read from entry.options/data each time
    @property
    def base_url_dynamic(self) -> str:
        """Get base URL (dynamic for local servers)."""
        if self.server_type in [
            SERVER_TYPE_OPENAI,
            SERVER_TYPE_GEMINI,
            SERVER_TYPE_ANTHROPIC,
            SERVER_TYPE_OPENROUTER,
        ]:
            return self.base_url  # Static cloud URLs
        else:
            # LM Studio, Ollama, llamacpp, Moltbot, vLLM - read dynamically
            return self.entry.options.get(
                CONF_LMSTUDIO_URL, self.entry.data.get(CONF_LMSTUDIO_URL, "")
            ).rstrip("/")

    @property
    def api_key(self) -> str:
        """Get API key (dynamic)."""
        return self.entry.options.get(
            CONF_API_KEY, self.entry.data.get(CONF_API_KEY, DEFAULT_API_KEY)
        )

    @property
    def model_name(self) -> str:
        """Get model name (dynamic)."""
        base_model = self.entry.options.get(
            CONF_MODEL_NAME, self.entry.data.get(CONF_MODEL_NAME, "")
        )
        # Format for provider-specific requirements
        if self.server_type == SERVER_TYPE_MOLTBOT:
            if not base_model.startswith("moltbot:"):
                return f"moltbot:{base_model}"
        return base_model

    @property
    def mcp_port(self) -> int:
        """Get MCP port (shared setting)."""
        return self._get_shared_setting(CONF_MCP_PORT, DEFAULT_MCP_PORT)

    @property
    def debug_mode(self) -> bool:
        """Get debug mode (dynamic)."""
        return self.entry.options.get(
            CONF_DEBUG_MODE, self.entry.data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE)
        )

    @property
    def clean_responses(self) -> bool:
        """Get clean responses setting (dynamic)."""
        return self.entry.options.get(
            CONF_CLEAN_RESPONSES,
            self.entry.data.get(CONF_CLEAN_RESPONSES, DEFAULT_CLEAN_RESPONSES),
        )

    @property
    def max_iterations(self) -> int:
        """Get max iterations (dynamic)."""
        return self.entry.options.get(
            CONF_MAX_ITERATIONS,
            self.entry.data.get(CONF_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS),
        )

    @property
    def max_history(self) -> int:
        """Get max history messages/turns (dynamic)."""
        return self.entry.options.get(
            CONF_MAX_HISTORY, self.entry.data.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
        )

    @property
    def max_tokens(self) -> int:
        """Get max tokens (dynamic)."""
        return self.entry.options.get(
            CONF_MAX_TOKENS, self.entry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        )

    @property
    def temperature(self) -> float:
        """Get temperature (dynamic)."""
        return self.entry.options.get(
            CONF_TEMPERATURE, self.entry.data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        )

    @property
    def follow_up_mode(self) -> str:
        """Get response mode (dynamic, with backward compatibility)."""
        return self.entry.options.get(
            CONF_RESPONSE_MODE,
            self.entry.data.get(
                CONF_RESPONSE_MODE,
                self.entry.options.get(
                    CONF_FOLLOW_UP_MODE,
                    self.entry.data.get(CONF_FOLLOW_UP_MODE, DEFAULT_RESPONSE_MODE),
                ),
            ),
        )

    @property
    def ollama_keep_alive(self) -> str:
        """Get Ollama keep_alive parameter."""
        return self.entry.options.get(
            CONF_OLLAMA_KEEP_ALIVE,
            self.entry.data.get(CONF_OLLAMA_KEEP_ALIVE, DEFAULT_OLLAMA_KEEP_ALIVE),
        )

    @property
    def ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter."""
        return self.entry.options.get(
            CONF_OLLAMA_NUM_CTX,
            self.entry.data.get(CONF_OLLAMA_NUM_CTX, DEFAULT_OLLAMA_NUM_CTX),
        )

    @property
    def search_provider(self) -> str:
        """Get search provider (shared setting) with backward compatibility."""
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER, None)

        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    @property
    def music_assistant_support_enabled(self) -> bool:
        """Get effective Music Assistant support setting for this profile."""
        return self._is_optional_tool_family_enabled("music_assistant")

    @property
    def assist_bridge_enabled(self) -> bool:
        """Get effective Assist bridge setting for this profile."""
        return self._is_optional_tool_family_enabled("assist_bridge")

    @property
    def native_response_service_tools_enabled(self) -> bool:
        """Get effective response-service tool setting for this profile."""
        return self._is_optional_tool_family_enabled("response_service")

    @property
    def weather_forecast_tools_enabled(self) -> bool:
        """Get effective weather forecast helper setting for this profile."""
        return (
            self.native_response_service_tools_enabled
            and self._is_optional_tool_family_enabled("weather_forecast")
        )

    @property
    def external_custom_tools_enabled(self) -> bool:
        """Get effective external custom tool setting for this profile."""
        return self._is_optional_tool_family_enabled(TOOL_FAMILY_EXTERNAL_CUSTOM)

    @property
    def recorder_tools_enabled(self) -> bool:
        """Get effective recorder tool setting for this profile."""
        return self._is_optional_tool_family_enabled("recorder")

    @property
    def memory_tools_enabled(self) -> bool:
        """Get effective persisted memory tool setting for this profile."""
        return self._is_optional_tool_family_enabled("memory")

    @property
    def calculator_tools_enabled(self) -> bool:
        """Get effective calculator tool setting for this profile."""
        return self._is_optional_tool_family_enabled("calculator")

    @property
    def unit_conversion_tools_enabled(self) -> bool:
        """Get effective unit-conversion tool setting for this profile."""
        return self._is_optional_tool_family_enabled("unit_conversion")

    @property
    def device_tools_enabled(self) -> bool:
        """Get effective device tool setting for this profile."""
        return self._is_optional_tool_family_enabled("device")

    @property
    def web_search_tools_enabled(self) -> bool:
        """Get effective web-search tool setting for this profile."""
        return self._is_optional_tool_family_enabled("web_search")

    def _get_shared_custom_tools_loader(self) -> Any | None:
        """Return the shared custom tool loader, if available."""
        server = self.hass.data.get(DOMAIN, {}).get("shared_mcp_server")
        return getattr(server, "custom_tools", None) if server else None

    def _is_external_custom_tool(self, tool_name: str) -> bool:
        """Return whether a tool name comes from an external custom tool package."""
        custom_tools = self._get_shared_custom_tools_loader()
        if custom_tools is None:
            return False

        checker = getattr(custom_tools, "is_external_custom_tool", None)
        if not callable(checker):
            return False

        try:
            return bool(checker(tool_name))
        except Exception as err:
            _LOGGER.debug(
                "Unable to classify external custom tool %s: %s", tool_name, err
            )
            return False

    def _build_disabled_tool_family_instructions(self) -> str:
        """Build prompt instructions for disabled optional tool families."""
        lines: list[str] = []

        if not self.device_tools_enabled:
            lines.append(
                "- Device tools are disabled. Do not call discover_devices or get_device_details. Use discover_entities and get_entity_details instead."
            )

        if not self.web_search_tools_enabled:
            lines.append(
                "- Web search tools are disabled. Do not call search or read_url."
            )

        if not self.assist_bridge_enabled:
            lines.append(
                "- Native Assist bridge tools are disabled. Do not call list_assist_tools, call_assist_tool, get_assist_prompt, or get_assist_context_snapshot."
            )

        if not self.native_response_service_tools_enabled:
            lines.append(
                "- Native response-service tools are disabled. Do not call list_response_services or call_service_with_response. Use entity details or other MCP tools instead."
            )
        elif not self.weather_forecast_tools_enabled:
            lines.append(
                "- Weather forecast tools are disabled. Do not call get_weather_forecast, and do not use call_service_with_response for weather forecasts."
            )

        if not self.recorder_tools_enabled:
            lines.append(
                "- Recorder history analysis tools are disabled. Do not call analyze_entity_history or get_entity_state_at_time."
            )

        if not self.memory_tools_enabled:
            lines.append(
                "- Memory tools are disabled. Do not call remember_memory, recall_memories, or forget_memory."
            )

        if not self.calculator_tools_enabled:
            lines.append(
                "- Calculator tools are disabled. Do not call arithmetic or expression-evaluation tools."
            )

        if not self.external_custom_tools_enabled:
            lines.append(
                "- External custom tools are disabled. Do not call tools provided by user-defined packages."
            )

        if not self.unit_conversion_tools_enabled:
            lines.append(
                "- Unit-conversion tools are disabled. Do not call convert_unit."
            )

        if not lines:
            return ""

        return "## Disabled Optional Tool Families\n" + "\n".join(lines)

    def _build_optional_technical_instructions(self, current_area: str) -> str:
        """Build optional prompt sections for enabled capability families."""
        sections: list[str] = []

        if self.device_tools_enabled:
            sections.append(DEVICE_TECHNICAL_INSTRUCTIONS.strip())

        if self.native_response_service_tools_enabled:
            sections.append(RESPONSE_SERVICE_TECHNICAL_INSTRUCTIONS.strip())

        if self.recorder_tools_enabled:
            sections.append(RECORDER_ANALYSIS_TECHNICAL_INSTRUCTIONS.strip())

        if self.memory_tools_enabled:
            sections.append(MEMORY_TECHNICAL_INSTRUCTIONS.strip())

        if self.assist_bridge_enabled:
            sections.append(ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS.strip())

        if self.calculator_tools_enabled:
            sections.append(CALCULATOR_TECHNICAL_INSTRUCTIONS.strip())

        if self.unit_conversion_tools_enabled:
            sections.append(UNIT_CONVERSION_TECHNICAL_INSTRUCTIONS.strip())

        if self.music_assistant_support_enabled:
            sections.append(
                MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS.replace(
                    "{current_area}", current_area
                ).strip()
            )

        if self.external_custom_tools_enabled:
            external_custom_tool_instructions = (
                self._get_external_custom_tool_instructions()
            )
            if external_custom_tool_instructions:
                sections.append(external_custom_tool_instructions)

        return "\n\n".join(section for section in sections if section)

    def _get_external_custom_tool_instructions(self) -> str:
        """Return prompt additions from loaded external custom tool packages."""
        if not self.external_custom_tools_enabled:
            return ""

        custom_tools = self._get_shared_custom_tools_loader()
        if custom_tools is None:
            return ""

        try:
            return str(custom_tools.get_external_prompt_instructions() or "").strip()
        except Exception as err:
            _LOGGER.debug(
                "Unable to read external custom tool prompt instructions: %s", err
            )
            return ""

    @staticmethod
    def _compact_text(text: str, *, max_len: int = 160) -> str:
        """Compact instructional text for lower token usage."""
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            return ""

        for separator in (". ", "\n", "; "):
            if separator in normalized:
                normalized = normalized.split(separator, 1)[0].strip()
                break

        if len(normalized) <= max_len:
            return normalized

        truncated = normalized[: max_len - 1].rstrip()
        last_space = truncated.rfind(" ")
        if last_space > 40:
            truncated = truncated[:last_space]
        return truncated.rstrip(" ,;:.") + "."

    def _compact_schema_for_llm(self, schema: Any, *, keep_description: bool = False) -> Any:
        """Strip nonessential JSON-schema verbosity before sending tools to the LLM."""
        if isinstance(schema, list):
            compacted_list = [
                self._compact_schema_for_llm(item, keep_description=keep_description)
                for item in schema
            ]
            return [item for item in compacted_list if item not in (None, {}, [])]

        if not isinstance(schema, dict):
            return schema

        compacted: dict[str, Any] = {}

        for key, value in schema.items():
            if key in {"$schema", "title", "default", "examples", "example"}:
                continue

            if key == "description":
                if keep_description:
                    compact_description = self._compact_text(str(value), max_len=120)
                    if compact_description:
                        compacted[key] = compact_description
                continue

            if key == "properties":
                properties: dict[str, Any] = {}
                for prop_name, prop_schema in value.items():
                    compact_prop = self._compact_schema_for_llm(prop_schema)
                    if compact_prop:
                        properties[prop_name] = compact_prop
                if properties:
                    compacted[key] = properties
                continue

            if key == "required":
                if value:
                    compacted[key] = value
                continue

            if key == "additionalProperties":
                continue

            compact_value = self._compact_schema_for_llm(
                value, keep_description=keep_description
            )
            if compact_value not in (None, {}, []):
                compacted[key] = compact_value

        return compacted

    def _convert_mcp_tools_to_llm_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tools to a compact OpenAI-style tool schema."""
        openai_tools = []

        for tool in tools:
            parameters = self._compact_schema_for_llm(
                tool.get("inputSchema", {}), keep_description=False
            )
            if not parameters:
                parameters = {"type": "object", "properties": {}}
            elif parameters.get("type") == "object" and "properties" not in parameters:
                parameters["properties"] = {}

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": self._compact_text(
                            tool.get("description", ""), max_len=180
                        ),
                        "parameters": parameters,
                    },
                }
            )

        return openai_tools

    def _build_mcp_tool_cache_key(self) -> tuple[Any, ...]:
        """Build a cache key for the current profile-visible MCP tool surface."""
        return (
            self.mcp_port,
            self.search_provider,
            self.assist_bridge_enabled,
            self.native_response_service_tools_enabled,
            self.weather_forecast_tools_enabled,
            self.recorder_tools_enabled,
            self.memory_tools_enabled,
            self.calculator_tools_enabled,
            self.external_custom_tools_enabled,
            self.unit_conversion_tools_enabled,
            self.device_tools_enabled,
            self.music_assistant_support_enabled,
            self.web_search_tools_enabled,
            self._get_external_custom_tool_cache_signature(),
        )

    def _get_external_custom_tool_cache_signature(self) -> tuple[Any, ...]:
        """Return a cache signature for loaded external custom tools."""
        server = self.hass.data.get(DOMAIN, {}).get("shared_mcp_server")
        custom_tools = getattr(server, "custom_tools", None) if server else None
        if custom_tools is None:
            return ()

        get_cache_signature = getattr(custom_tools, "get_cache_signature", None)
        if callable(get_cache_signature):
            try:
                raw_signature = get_cache_signature()
                if isinstance(raw_signature, tuple):
                    return raw_signature
                return (raw_signature,)
            except Exception as err:
                _LOGGER.debug(
                    "Unable to read external custom tool cache signature: %s", err
                )

        get_external_prompt_instructions = getattr(
            custom_tools, "get_external_prompt_instructions", None
        )
        if callable(get_external_prompt_instructions):
            try:
                return (str(get_external_prompt_instructions() or "").strip(),)
            except Exception as err:
                _LOGGER.debug(
                    "Unable to read external custom tool prompt instructions: %s", err
                )

        return ()

    def _compact_tool_result_for_llm(self, tool_name: str, content: Any) -> str:
        """Keep tool results useful while avoiding oversized follow-up payloads."""
        text = "" if content is None else str(content)
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return ""

        original_length = len(text)
        original_lines = text.count("\n") + 1

        if original_length <= MAX_TOOL_RESULT_CHARS and original_lines <= MAX_TOOL_RESULT_LINES:
            return text

        lines = text.splitlines()
        truncated_lines = lines[:MAX_TOOL_RESULT_LINES]
        compacted = "\n".join(truncated_lines).strip()

        if len(compacted) > MAX_TOOL_RESULT_CHARS:
            compacted = compacted[:MAX_TOOL_RESULT_CHARS].rstrip()
            last_break = max(compacted.rfind("\n"), compacted.rfind(" "))
            if last_break > int(MAX_TOOL_RESULT_CHARS * 0.7):
                compacted = compacted[:last_break].rstrip()

        omitted_lines = max(0, original_lines - len(truncated_lines))
        omitted_chars = max(0, original_length - len(compacted))
        hint = (
            "Use narrower filters, paging, or a more specific follow-up tool call if you need the omitted detail."
        )
        if tool_name in {"discover_entities", "discover_devices"}:
            hint = (
                "Use limit/offset paging or narrower filters if you need more of the result set."
            )
        elif tool_name in {"get_entity_details", "get_device_details", "get_index"}:
            hint = (
                "Call again with a narrower target if you need more of this structured detail."
            )

        summary_parts: list[str] = []
        if omitted_lines:
            summary_parts.append(f"{omitted_lines} more lines")
        if omitted_chars:
            summary_parts.append(f"{omitted_chars} more chars")
        summary = ", ".join(summary_parts) or "additional content omitted"

        return (
            f"{compacted}\n\n"
            f"[Tool result truncated for model context: {summary}. {hint}]"
        )

    @property
    def attribution(self) -> str:
        """Return attribution."""
        server_name = {
            SERVER_TYPE_LMSTUDIO: "LM Studio",
            SERVER_TYPE_LLAMACPP: "llama.cpp",
            SERVER_TYPE_OLLAMA: "Ollama",
            SERVER_TYPE_OPENAI: "OpenAI",
            SERVER_TYPE_GEMINI: "Gemini",
            SERVER_TYPE_ANTHROPIC: "Claude",
            SERVER_TYPE_OPENROUTER: "OpenRouter",
            SERVER_TYPE_MOLTBOT: "Moltbot",
            SERVER_TYPE_VLLM: "vLLM",
        }.get(self.server_type, "LLM")
        return f"Powered by {server_name} with MCP entity discovery"

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"  # Support all languages

    @property
    def supported_features(self) -> int:
        """Return supported features."""
        features = ConversationEntityFeature(0)

        # Check if home control is enabled in config
        control_enabled = self.entry.options.get(
            CONF_CONTROL_HA, self.entry.data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA)
        )

        if control_enabled:
            features |= ConversationEntityFeature.CONTROL

        return features

    @property
    def follow_up_phrases(self) -> str:
        """Return follow-up phrases for pattern detection."""
        return self.entry.options.get(
            CONF_FOLLOW_UP_PHRASES,
            self.entry.data.get(CONF_FOLLOW_UP_PHRASES, DEFAULT_FOLLOW_UP_PHRASES),
        )

    @property
    def end_words(self) -> str:
        """Return end conversation words for user ending detection."""
        return self.entry.options.get(
            CONF_END_WORDS, self.entry.data.get(CONF_END_WORDS, DEFAULT_END_WORDS)
        )

    @property
    def profile_name(self) -> str:
        """Return profile name."""
        return self.entry.data.get(CONF_PROFILE_NAME, "MCP Assist")

    @property
    def timeout(self) -> int:
        """Get request timeout in seconds (dynamic)."""
        return self.entry.options.get(
            CONF_TIMEOUT, self.entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        )

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

        # Store entity reference for index manager to access
        if self.entry.entry_id in self.hass.data[DOMAIN]:
            self.hass.data[DOMAIN][self.entry.entry_id]["agent"] = self

        _LOGGER.info("Conversation entity registered: %s", self._attr_name)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)

        # Remove entity reference
        if self.entry.entry_id in self.hass.data.get(DOMAIN, {}):
            self.hass.data[DOMAIN][self.entry.entry_id].pop("agent", None)

        await super().async_will_remove_from_hass()
        _LOGGER.info("Conversation entity unregistered: %s", self._attr_name)

    def _get_server_display_name(self) -> str:
        """Get friendly display name for the server type."""
        return {
            SERVER_TYPE_LMSTUDIO: "LM Studio",
            SERVER_TYPE_LLAMACPP: "llama.cpp",
            SERVER_TYPE_OLLAMA: "Ollama",
            SERVER_TYPE_OPENAI: "OpenAI",
            SERVER_TYPE_GEMINI: "Gemini",
            SERVER_TYPE_ANTHROPIC: "Claude",
            SERVER_TYPE_OPENROUTER: "OpenRouter",
            SERVER_TYPE_MOLTBOT: "Moltbot",
            SERVER_TYPE_VLLM: "vLLM",
        }.get(self.server_type, "the LLM server")

    def _get_friendly_error_message(self, error: Exception) -> str:
        """Convert technical errors to user-friendly TTS messages."""
        error_str = str(error).lower()
        error_full = str(error)  # Keep original case for extracting details

        # Category A: Connection/Network Errors
        if any(
            x in error_str
            for x in [
                "connection",
                "refused",
                "cannot connect",
                "no route",
                "unreachable",
            ]
        ):
            if self.server_type in [
                SERVER_TYPE_OPENAI,
                SERVER_TYPE_GEMINI,
                SERVER_TYPE_ANTHROPIC,
            ]:
                return f"I couldn't reach {self._get_server_display_name()}'s API servers. Please check your internet connection and try again."
            else:
                return f"I couldn't connect to {self._get_server_display_name()} at {self.base_url_dynamic}. Please check that the server is running and the address is correct in your integration settings."

        if "timeout" in error_str or "timed out" in error_str:
            return f"The {self._get_server_display_name()} server took too long to respond. This might be because the model is slow or busy. Try again or consider using a faster model."

        # Category B: Authentication
        if any(
            x in error_str
            for x in [
                "401",
                "403",
                "unauthorized",
                "invalid_api_key",
                "invalid api key",
            ]
        ):
            return f"Your {self._get_server_display_name()} API key is invalid or missing. Please check your API key in the integration settings."

        if "insufficient_quota" in error_str or "permission denied" in error_str:
            return f"Your {self._get_server_display_name()} account doesn't have permission for this operation. Check your account status and billing."

        # Category C: Resource Limits
        if (
            "maximum context length" in error_str
            or "context_length_exceeded" in error_str
            or "too many tokens" in error_str
        ):
            # Try to extract token limit if present
            token_match = re.search(r"(\d+)\s*tokens?", error_str)
            if token_match:
                return f"The conversation has exceeded the model's {token_match.group(1)} token limit. Start a new conversation or reduce the history limit in Advanced Settings."
            return "The conversation has exceeded the model's token limit. Start a new conversation or reduce the history limit in Advanced Settings."

        if (
            "rate limit" in error_str
            or "429" in error_str
            or "too many requests" in error_str
        ):
            return f"You've hit {self._get_server_display_name()}'s rate limit. Wait a minute and try again, or upgrade your plan for higher limits."

        if "quota exceeded" in error_str or "insufficient credits" in error_str:
            return f"Your {self._get_server_display_name()} account has run out of credits or quota. Check your billing and add credits to continue."

        # Category D: Model Errors
        if "404" in error_str or ("model" in error_str and "not found" in error_str):
            return f"The model '{self.model_name}' wasn't found on {self._get_server_display_name()}. Check that the model name is correct in your integration settings."

        if self.server_type == SERVER_TYPE_OLLAMA and (
            "model not loaded" in error_str or "pull the model" in error_str
        ):
            return f"The model '{self.model_name}' isn't loaded in Ollama. Run 'ollama pull {self.model_name}' to download it first."

        # Category E: MCP Errors
        if (
            f"localhost:{self.mcp_port}" in error_str
            or f"127.0.0.1:{self.mcp_port}" in error_str
        ):
            return f"I couldn't connect to the MCP server on port {self.mcp_port}. The integration may not have initialized correctly. Try restarting Home Assistant."

        # Category F: Response Errors
        if "empty response" in error_str or "no response" in error_str:
            return f"The {self._get_server_display_name()} server returned an empty response. This sometimes happens with certain models. Try rephrasing your request."

        if "json" in error_str and (
            "parse" in error_str or "decode" in error_str or "malformed" in error_str
        ):
            return f"I received a malformed response from {self._get_server_display_name()}. This might be a temporary server issue. Please try again."

        # Category G: Generic fallback
        # Extract first meaningful part of error (up to 100 chars, stop at newline)
        error_snippet = error_full.split("\n")[0][:100]
        return f"An unexpected error occurred while talking to {self._get_server_display_name()}. The error was: {error_snippet}. Check the Home Assistant logs for more details."

    def _record_tool_calls_to_chatlog(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Record tool calls to ChatLog for debug view."""
        if not self._current_chat_log:
            return

        try:
            # Convert tool calls to llm.ToolInput format
            llm_tool_calls = []
            for tc in tool_calls:
                tool_input = llm.ToolInput(
                    id=tc.get("id", str(uuid.uuid4())),
                    tool_name=tc.get("function", {}).get("name", "unknown"),
                    tool_args=self._parse_tool_arguments(
                        tc.get("function", {}).get("arguments")
                    ),
                    external=True,  # MCP tools are executed externally, not by ChatLog
                )
                llm_tool_calls.append(tool_input)

            # Add assistant content with tool calls
            assistant_content = chat_log.AssistantContent(
                agent_id=self.entity_id, tool_calls=llm_tool_calls
            )
            self._current_chat_log.async_add_assistant_content_without_tools(
                assistant_content
            )

            if self.debug_mode:
                _LOGGER.debug(f"📊 Recorded {len(tool_calls)} tool calls to ChatLog")
        except Exception as e:
            _LOGGER.error(f"Error recording tool calls to ChatLog: {e}")

    def _stringify_tool_arguments(self, arguments: Any) -> str:
        """Normalize tool arguments to a JSON string."""
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        """Parse tool arguments whether they arrive as a dict or JSON string."""
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            if not arguments.strip():
                return {}
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _normalize_tool_call_arguments(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize tool_call function.arguments for internal/provider use."""
        normalized = []
        for tool_call in tool_calls:
            normalized_call = dict(tool_call)
            function = dict(normalized_call.get("function", {}))
            if function:
                function["arguments"] = self._stringify_tool_arguments(
                    function.get("arguments")
                )
                normalized_call["function"] = function
            normalized.append(normalized_call)
        return normalized

    def _format_tool_calls_for_ollama(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert tool calls to Ollama's native argument shape."""
        formatted = []
        for tool_call in tool_calls:
            formatted_call = dict(tool_call)
            function = dict(formatted_call.get("function", {}))
            if function:
                function["arguments"] = self._parse_tool_arguments(
                    function.get("arguments")
                )
                formatted_call["function"] = function
            formatted.append(formatted_call)
        return formatted

    def _record_tool_result_to_chatlog(
        self, tool_call_id: str, tool_name: str, tool_result: Dict[str, Any]
    ) -> None:
        """Record a single tool result to ChatLog for debug view."""
        if not self._current_chat_log:
            return

        try:
            result_content = chat_log.ToolResultContent(
                agent_id=self.entity_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_result=tool_result,
            )
            # Use callback method to add tool result
            self._current_chat_log.async_add_assistant_content_without_tools(
                result_content
            )

            if self.debug_mode:
                _LOGGER.debug(f"📊 Recorded tool result for {tool_name} to ChatLog")
        except Exception as e:
            _LOGGER.error(f"Error recording tool result to ChatLog: {e}")

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process user input and return response."""
        _LOGGER.info("🎤 Voice request started - Processing: %s", user_input.text)

        # Create ChatLog for debug view
        with chat_session.async_get_chat_session(
            self.hass, user_input.conversation_id
        ) as session:
            with chat_log.async_get_chat_log(
                self.hass, session, user_input  # Automatically adds user content
            ) as log:
                # Store ChatLog for tool execution methods to access
                self._current_chat_log = log

                try:
                    return await self._async_process_with_chatlog(
                        user_input, session.conversation_id
                    )
                finally:
                    # Clean up
                    self._current_chat_log = None

    async def _async_process_with_chatlog(
        self, user_input: ConversationInput, conversation_id: str
    ) -> ConversationResult:
        """Process user input with ChatLog tracking."""
        try:
            _LOGGER.debug("Conversation ID: %s", conversation_id)

            # Get conversation history
            history = self.history.get_history(conversation_id)
            _LOGGER.debug("History retrieved: %d turns", len(history))

            # Build system prompt with context
            system_prompt = await self._build_system_prompt_with_context(user_input)
            if self.debug_mode:
                _LOGGER.info(
                    f"📝 System prompt built, length: {len(system_prompt)} chars"
                )
                _LOGGER.info(f"📝 System prompt preview: {system_prompt[:200]}...")

            # Build conversation messages
            messages = self._build_messages(system_prompt, user_input.text, history)

            # Store conversation_id for Moltbot session management
            self._current_conversation_id = conversation_id

            if self.debug_mode:
                _LOGGER.info(f"📨 Messages built: {len(messages)} messages")
                for i, msg in enumerate(messages):
                    role = msg.get("role")
                    content_len = (
                        len(msg.get("content", "")) if msg.get("content") else 0
                    )
                    _LOGGER.info(
                        f"  Message {i}: role={role}, content_length={content_len}"
                    )

            # Call LLM API
            _LOGGER.info(f"📡 Calling {self.server_type} API...")
            response_text = await self._call_llm(messages)
            _LOGGER.info(
                f"✅ {self.server_type} response received, length: %d",
                len(response_text),
            )

            # Strip thinking tags from reasoning models (e.g., Qwen3, DeepSeek R1, GPT-OSS)
            response_text, thinking_content = self._strip_thinking_tags(response_text)
            if thinking_content and self.debug_mode:
                _LOGGER.info(
                    f"🧠 Thinking content (stripped): {thinking_content[:500]}..."
                )

            if self.debug_mode:
                # Use repr() to show newlines and hidden characters
                _LOGGER.info(f"💬 Full response (repr): {repr(response_text)}")
            else:
                # For non-debug, just show first 500 chars
                preview = (
                    response_text[:500] if len(response_text) > 500 else response_text
                )
                _LOGGER.info(f"💬 Full response preview: {preview}")

            # Parse response and execute any Home Assistant actions
            actions_taken = await self._execute_actions(response_text, user_input)

            # Add final assistant response to ChatLog
            if self._current_chat_log:
                final_content = chat_log.AssistantContent(
                    agent_id=self.entity_id, content=response_text
                )
                self._current_chat_log.async_add_assistant_content_without_tools(
                    final_content
                )

            # Store in conversation history
            self.history.add_turn(
                conversation_id, user_input.text, response_text, actions=actions_taken
            )

            # Create intent response
            intent_response = intent.IntentResponse(language=user_input.language)
            # Clean response for TTS (character normalization always, aggressive cleaning if enabled)
            cleaned_text = self._clean_text_for_tts(response_text)
            intent_response.async_set_speech(cleaned_text)

            # Note: Card data removed as it was causing JSON serialization errors
            # Actions are already executed via MCP tools, so card isn't needed

            # Check if user wants to end (stopwords+1 algorithm)
            user_wants_to_end = False
            if self.follow_up_mode in ["default", "always"]:
                user_wants_to_end = self._detect_user_ending_intent(user_input.text)
                if user_wants_to_end and self.debug_mode:
                    _LOGGER.info("🎯 User ending intent detected (stopwords+1)")

            # Determine follow-up mode
            if user_wants_to_end:
                # User explicitly wants to end
                continue_conversation = False
            elif self.follow_up_mode == "always":
                # Always continue regardless of tool
                continue_conversation = True
            elif self.follow_up_mode == "none":
                # Never continue regardless of tool
                continue_conversation = False
            else:  # "default" - smart mode
                # Use the LLM's indication if it called the tool
                if hasattr(self, "_expecting_response"):
                    continue_conversation = self._expecting_response
                    # Clear for next conversation
                    delattr(self, "_expecting_response")
                    if self.debug_mode:
                        _LOGGER.info("🎯 Using LLM's set_conversation_state indication")
                else:
                    # LLM didn't indicate, use pattern detection as fallback
                    continue_conversation = self._detect_follow_up_patterns(
                        response_text
                    )
                    if self.debug_mode:
                        if continue_conversation:
                            _LOGGER.info("🎯 Pattern detection triggered continuation")
                        else:
                            _LOGGER.info("🎯 No patterns detected, closing conversation")

            if self.debug_mode:
                _LOGGER.info(
                    f"🎯 Follow-up mode: {self.follow_up_mode}, Continue: {continue_conversation}"
                )

            return ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
                continue_conversation=continue_conversation,
            )

        except Exception as err:
            _LOGGER.exception("Error processing conversation")

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                self._get_friendly_error_message(err),
            )

            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,  # Don't continue on errors
            )

    def _detect_user_ending_intent(self, text: str) -> bool:
        """Detect if user wants to end conversation using stopwords+1 algorithm.

        Handles both single words and multi-word phrases.

        Returns True if:
        - User message contains at least one stop word/phrase, AND
        - User message has ≤1 non-stop word (excluding agent name and matched phrases)

        Examples:
        - "stop" → True (0 non-stop words)
        - "no thanks" → True (both are stop words)
        - "no thank you" → True ("thank you" is a stop phrase)
        - "bye Jarvis" → True (Jarvis removed, 0 non-stop)
        - "ok please" → True (1 non-stop word)
        - "no turn on light" → False (3 non-stop words)
        """
        if not text:
            return False

        # Parse end words from config
        end_words_raw = [
            word.strip().lower() for word in self.end_words.split(",") if word.strip()
        ]
        if not end_words_raw:
            return False

        # Separate multi-word phrases from single words
        multi_word_phrases = [phrase for phrase in end_words_raw if " " in phrase]
        single_words = [word for word in end_words_raw if " " not in word]

        # Normalize text
        text_lower = text.lower().strip()

        # Check if any multi-word phrases are present and remove them
        has_stop_word = False
        remaining_text = text_lower

        for phrase in multi_word_phrases:
            if phrase in remaining_text:
                has_stop_word = True
                # Replace matched phrase with spaces to preserve word boundaries
                remaining_text = remaining_text.replace(phrase, " ")

        # Split remaining text into words
        words = remaining_text.split()

        # Remove agent name
        profile_name_lower = self.profile_name.lower()
        words = [word for word in words if word != profile_name_lower]

        # Check if any single-word stop words are present
        for word in words:
            if word in single_words:
                has_stop_word = True

        if not has_stop_word:
            return False

        # Count non-stop words (words not in single_words list)
        non_stop_words = [
            word for word in words if word not in single_words and word.strip()
        ]

        # End if ≤1 non-stop word
        return len(non_stop_words) <= 1

    def _detect_follow_up_patterns(self, text: str) -> bool:
        """Detect if the response expects a follow-up based on patterns."""
        if not text:
            return False

        # Debug logging to see what we're checking
        if self.debug_mode:
            _LOGGER.info(
                f"🔍 Pattern detection - Full response length: {len(text)} chars"
            )
            _LOGGER.info(f"🔍 Pattern detection - Last 200 chars: {text[-200:]}")

        # Check last 200 characters for efficiency
        check_text = text[-200:].lower()

        # Pattern 1: Ends with a question mark
        if check_text.rstrip().endswith("?"):
            if self.debug_mode:
                _LOGGER.info("📊 Question detected: phrase ends with question mark")
            return True

        # Pattern 2: Question phrases (user-configurable)
        question_phrases = [
            phrase.strip().lower()
            for phrase in self.follow_up_phrases.split(",")
            if phrase.strip()
        ]

        for phrase in question_phrases:
            if phrase in check_text:
                if self.debug_mode:
                    _LOGGER.info(f"📊 Follow-up phrase detected: '{phrase}'")
                return True

        return False

    async def _get_current_area(self, user_input: ConversationInput) -> str:
        """Get the area of the satellite/device making the request."""
        try:
            # Try to get device_id from context
            device_id = (
                user_input.device_id if hasattr(user_input, "device_id") else None
            )

            if not device_id:
                _LOGGER.debug("No device_id in conversation input")
                return "Unknown"

            # Get device registry and look up device
            device_reg = dr.async_get(self.hass)
            device_entry = device_reg.async_get(device_id)

            if not device_entry:
                _LOGGER.debug("No device found for device_id: %s", device_id)
                return "Unknown"

            # Get area from device
            area_id = device_entry.area_id
            if not area_id:
                _LOGGER.debug("Device %s has no assigned area", device_id)
                return "Unknown"

            # Get area registry and look up area name
            area_reg = ar.async_get(self.hass)
            area_entry = area_reg.async_get_area(area_id)

            if not area_entry:
                _LOGGER.debug("Area ID %s not found in registry", area_id)
                return "Unknown"

            area_name = area_entry.name
            _LOGGER.info(
                "📍 Current area detected: %s (from device %s)", area_name, device_id
            )
            return area_name

        except Exception as e:
            _LOGGER.warning("Error getting current area: %s", e)
            return "Unknown"

    def _get_home_location(self) -> str:
        """Return a compact home-location summary for prompt context."""
        if not self._get_shared_setting(
            CONF_INCLUDE_HOME_LOCATION, DEFAULT_INCLUDE_HOME_LOCATION
        ):
            return ""

        location_name = str(
            getattr(self.hass.config, "location_name", "") or ""
        ).strip()
        latitude = getattr(self.hass.config, "latitude", None)
        longitude = getattr(self.hass.config, "longitude", None)

        coordinates = ""
        try:
            if latitude is not None and longitude is not None:
                coordinates = f"{float(latitude):.4f}, {float(longitude):.4f}"
        except (TypeError, ValueError):
            coordinates = ""

        if location_name and coordinates:
            return f"{location_name} ({coordinates})"
        if location_name:
            return location_name
        return coordinates

    async def _get_current_user_name(self, user_input: ConversationInput) -> str:
        """Return the current Home Assistant user name for prompt context."""
        if not self._get_shared_setting(
            CONF_INCLUDE_CURRENT_USER, DEFAULT_INCLUDE_CURRENT_USER
        ):
            return ""

        try:
            user_id = getattr(getattr(user_input, "context", None), "user_id", None)
            if not user_id:
                return ""

            user = await self.hass.auth.async_get_user(user_id)
            if not user:
                return ""

            return str(getattr(user, "name", "") or "").strip()
        except Exception as err:
            _LOGGER.debug("Unable to resolve current HA user: %s", err)
            return ""

    def _get_default_system_prompt(self) -> str:
        """Get the built-in localized default system prompt."""
        return (
            get_language_instruction(self.hass.config.language)
            or DEFAULT_SYSTEM_PROMPT
        )

    def _resolve_prompt_setting(
        self, *, prompt_key: str, mode_key: str, default_prompt: str
    ) -> str:
        """Resolve a prompt using Default/Custom mode with backward compatibility."""
        options = self.entry.options
        data = self.entry.data

        explicit_mode = options.get(mode_key, data.get(mode_key))
        stored_prompt = options.get(prompt_key, data.get(prompt_key))

        if explicit_mode == PROMPT_MODE_CUSTOM:
            return "" if stored_prompt is None else str(stored_prompt)

        if explicit_mode == PROMPT_MODE_DEFAULT:
            return default_prompt

        if stored_prompt in (None, "", default_prompt):
            return default_prompt

        return str(stored_prompt)

    async def _build_system_prompt_with_context(
        self, user_input: ConversationInput
    ) -> str:
        """Build the compact system prompt used for model calls."""
        try:
            now = dt_util.now()
            if self.entry.data.get(CONF_SERVER_TYPE) == SERVER_TYPE_MOLTBOT:
                system_prompt = ""
            else:
                system_prompt = self._resolve_prompt_setting(
                    prompt_key=CONF_SYSTEM_PROMPT,
                    mode_key=CONF_SYSTEM_PROMPT_MODE,
                    default_prompt=self._get_default_system_prompt(),
                )
            technical_prompt = self._resolve_prompt_setting(
                prompt_key=CONF_TECHNICAL_PROMPT,
                    mode_key=CONF_TECHNICAL_PROMPT_MODE,
                    default_prompt=DEFAULT_TECHNICAL_PROMPT,
            )

            current_area = "Unknown"
            needs_current_area = "{current_area}" in technical_prompt or (
                self.music_assistant_support_enabled
            )
            if needs_current_area:
                current_area = await self._get_current_area(user_input)
            if "{current_area}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{current_area}", current_area
                )

            if "{time}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{time}", now.strftime("%H:%M:%S")
                )

            if "{date}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{date}", now.strftime("%Y-%m-%d")
                )

            if (
                "{current_user}" in technical_prompt
                or "{current_user_context}" in technical_prompt
            ):
                current_user = await self._get_current_user_name(user_input)
                if "{current_user}" in technical_prompt:
                    technical_prompt = technical_prompt.replace(
                        "{current_user}", current_user
                    )
                if "{current_user_context}" in technical_prompt:
                    technical_prompt = technical_prompt.replace(
                        "{current_user_context}",
                        f"Current user: {current_user}" if current_user else "",
                    )

            if (
                "{home_location}" in technical_prompt
                or "{home_location_context}" in technical_prompt
            ):
                home_location = self._get_home_location()
                if "{home_location}" in technical_prompt:
                    technical_prompt = technical_prompt.replace(
                        "{home_location}", home_location
                    )
                if "{home_location_context}" in technical_prompt:
                    technical_prompt = technical_prompt.replace(
                        "{home_location_context}",
                        f"Home location: {home_location}" if home_location else "",
                    )

            if "{response_mode}" in technical_prompt:
                mode_instructions = RESPONSE_MODE_INSTRUCTIONS.get(
                    self.follow_up_mode, RESPONSE_MODE_INSTRUCTIONS["default"]
                )
                technical_prompt = technical_prompt.replace(
                    "{response_mode}", mode_instructions
                )

            if "{index}" in technical_prompt:
                index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")
                if index_manager:
                    index = await index_manager.get_index()
                    index_json = json.dumps(index, separators=(",", ":"))
                else:
                    index_json = "{}"
                    _LOGGER.warning("IndexManager not available, using empty index")
                technical_prompt = technical_prompt.replace("{index}", index_json)

            technical_prompt = re.sub(r"\n{3,}", "\n\n", technical_prompt).strip()

            optional_instructions = self._build_optional_technical_instructions(
                current_area
            )
            if optional_instructions:
                technical_prompt = (
                    f"{technical_prompt.rstrip()}\n\n{optional_instructions}"
                )

            if system_prompt:
                return f"{system_prompt}\n\n{technical_prompt}"
            return technical_prompt

        except Exception as e:
            _LOGGER.error("Error building system prompt: %s", e)
            return "You are a Home Assistant voice assistant. Use MCP tools to control devices."

    async def _get_home_context(self) -> str:
        """Get lightweight home context (areas and domains) to help LLM with discovery."""
        try:
            # Fetch areas
            areas_result = await self._call_mcp_tool("list_areas", {})
            areas_text = ""
            if "result" in areas_result:
                areas_text = areas_result["result"]

            # Fetch domains
            domains_result = await self._call_mcp_tool("list_domains", {})
            domains_text = ""
            if "result" in domains_result:
                domains_text = domains_result["result"]

            # Format context section
            context = "# Your Home Configuration\n\n"
            if areas_text:
                context += f"{areas_text}\n\n"
            if domains_text:
                context += f"{domains_text}\n"

            _LOGGER.debug("Home context added: %d characters", len(context))
            return context

        except Exception as e:
            _LOGGER.warning("Could not fetch home context: %s", e)
            return ""

    def _build_system_prompt(self) -> str:
        """Build system prompt (legacy sync version - note: cannot include index without async)."""
        try:
            now = dt_util.now()
            if self.entry.data.get(CONF_SERVER_TYPE) == SERVER_TYPE_MOLTBOT:
                system_prompt = ""
            else:
                system_prompt = self._resolve_prompt_setting(
                    prompt_key=CONF_SYSTEM_PROMPT,
                    mode_key=CONF_SYSTEM_PROMPT_MODE,
                    default_prompt=self._get_default_system_prompt(),
                )
            technical_prompt = self._resolve_prompt_setting(
                prompt_key=CONF_TECHNICAL_PROMPT,
                mode_key=CONF_TECHNICAL_PROMPT_MODE,
                default_prompt=DEFAULT_TECHNICAL_PROMPT,
            )

            if "{time}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{time}", now.strftime("%H:%M:%S")
                )
            if "{date}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{date}", now.strftime("%Y-%m-%d")
                )
            if "{current_area}" in technical_prompt:
                technical_prompt = technical_prompt.replace("{current_area}", "Unknown")
            if "{current_user}" in technical_prompt:
                technical_prompt = technical_prompt.replace("{current_user}", "")
            if "{current_user_context}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{current_user_context}", ""
                )
            if "{home_location}" in technical_prompt:
                technical_prompt = technical_prompt.replace(
                    "{home_location}", self._get_home_location()
                )
            if "{home_location_context}" in technical_prompt:
                home_location = self._get_home_location()
                technical_prompt = technical_prompt.replace(
                    "{home_location_context}",
                    f"Home location: {home_location}" if home_location else "",
                )
            if "{index}" in technical_prompt:
                technical_prompt = technical_prompt.replace("{index}", "{}")
            if "{response_mode}" in technical_prompt:
                mode_instructions = RESPONSE_MODE_INSTRUCTIONS.get(
                    self.follow_up_mode, RESPONSE_MODE_INSTRUCTIONS["default"]
                )
                technical_prompt = technical_prompt.replace(
                    "{response_mode}", mode_instructions
                )

            technical_prompt = re.sub(r"\n{3,}", "\n\n", technical_prompt).strip()

            optional_instructions = self._build_optional_technical_instructions(
                "Unknown"
            )
            if optional_instructions:
                technical_prompt = (
                    f"{technical_prompt.rstrip()}\n\n{optional_instructions}"
                )

            if system_prompt:
                return f"{system_prompt}\n\n{technical_prompt}"
            return technical_prompt

        except Exception as e:
            _LOGGER.error("Error building system prompt: %s", e)
            # Return a basic prompt as fallback
            return "You are a Home Assistant voice assistant. Use MCP tools to control devices."

    def _build_messages(
        self, system_prompt: str, user_text: str, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build message list for LM Studio."""
        messages = [{"role": "system", "content": system_prompt}]

        # For Moltbot, skip history - server manages context via user field
        if self.server_type != SERVER_TYPE_MOLTBOT:
            # Add conversation history using the configured limit
            history_limit = max(0, self.max_history)
            if history_limit > 0:
                for turn in history[-history_limit:]:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    async def _fetch_mcp_tools_from_server(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch and convert available tools from the MCP server."""
        try:
            mcp_url = f"http://localhost:{self.mcp_port}"

            # Get tools list from MCP server
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{mcp_url}/",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": 1,
                    },
                ) as response:
                    if response.status != 200:
                        _LOGGER.warning("Failed to get MCP tools: %d", response.status)
                        return None

                    data = await response.json()
                    if "result" in data and "tools" in data["result"]:
                        tools = self._filter_mcp_tools_for_profile(
                            data["result"]["tools"]
                        )
                        _LOGGER.info(
                            "Retrieved %d MCP tools after profile filtering",
                            len(tools),
                        )

                        tool_names = []
                        for tool in tools:
                            tool_names.append(tool["name"])

                        _LOGGER.info("MCP tools available: %s", ", ".join(tool_names))
                        if "perform_action" in tool_names:
                            _LOGGER.info("✅ perform_action tool is available")
                        else:
                            _LOGGER.warning("⚠️ perform_action tool NOT found!")

                        return self._convert_mcp_tools_to_llm_tools(tools)
                    return None

        except Exception as err:
            _LOGGER.error("Failed to get MCP tools: %s", err)
            return None

    async def _get_mcp_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Return available LLM-facing MCP tools, using a short-lived cache."""
        cache_key = self._build_mcp_tool_cache_key()
        now = time.monotonic()

        if (
            self._cached_llm_tools is not None
            and self._cached_llm_tools_key == cache_key
            and (now - self._cached_llm_tools_fetched_at) < MCP_TOOL_CACHE_TTL_SECONDS
        ):
            _LOGGER.debug("Using cached MCP tool schema for profile")
            return list(self._cached_llm_tools)

        tools = await self._fetch_mcp_tools_from_server()
        if tools is not None:
            self._cached_llm_tools = list(tools)
            self._cached_llm_tools_key = cache_key
            self._cached_llm_tools_fetched_at = now
            return list(tools)

        if self._cached_llm_tools is not None and self._cached_llm_tools_key == cache_key:
            _LOGGER.warning("Using stale cached MCP tools after refresh failure")
            return list(self._cached_llm_tools)

        return None

    def _filter_mcp_tools_for_profile(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter shared MCP tools down to the subset enabled for this profile."""
        filtered_tools = [
            tool
            for tool in tools
            if self._is_tool_enabled_for_profile(tool.get("name", ""))
        ]

        filtered_names = [tool.get("name", "") for tool in filtered_tools]
        _LOGGER.info("Profile-visible MCP tools: %s", ", ".join(filtered_names))
        return filtered_tools

    async def _call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single MCP tool and return the result."""
        _LOGGER.info(f"🔧 Executing MCP tool: {tool_name} with args: {arguments}")

        if not self._is_tool_enabled_for_profile(tool_name):
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Tool '{tool_name}' is disabled for this profile. "
                            "Use this profile's enabled tools instead."
                        ),
                    }
                ],
            }

        if (
            tool_name == "call_service_with_response"
            and str(arguments.get("domain") or "").strip().lower() == "weather"
            and not self.weather_forecast_tools_enabled
        ):
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Weather forecast tools are disabled for this profile. "
                            "Use this profile's enabled tools instead."
                        ),
                    }
                ],
            }

        try:
            mcp_url = f"http://localhost:{self.mcp_port}"

            # Create JSON-RPC request for tool execution
            request_id = f"tool_{uuid.uuid4().hex[:8]}"
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": request_id,
            }

            _LOGGER.debug(f"MCP request: {json.dumps(payload, indent=2)}")

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{mcp_url}/", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            f"MCP tool call failed: {response.status} - {error_text}"
                        )
                        return {"error": f"Tool execution failed: {error_text}"}

                    data = await response.json()
                    _LOGGER.debug(f"MCP response: {json.dumps(data, indent=2)}")

                    if "result" in data and "content" in data["result"]:
                        # Extract the text content from the MCP response
                        content = data["result"]["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text_result = content[0].get("text", "")
                            if self.debug_mode:
                                _LOGGER.info(
                                    f"🔍 MCP tool '{tool_name}' returned {len(text_result)} chars"
                                )
                                _LOGGER.info(
                                    f"🔍 Full result (repr): {repr(text_result)}"
                                )
                                # Also log each line separately for readability
                                for i, line in enumerate(text_result.split("\n")):
                                    _LOGGER.info(f"  Line {i}: {line}")
                            return {"result": text_result}
                        return {"result": str(content)}
                    elif "error" in data:
                        return {"error": data["error"]}
                    else:
                        return {"result": str(data.get("result", ""))}

        except Exception as e:
            _LOGGER.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    def _strip_thinking_tags(self, text: str) -> tuple[str, str]:
        """Strip thinking/reasoning tags from model output.

        Reasoning models like Qwen3, DeepSeek R1, and GPT-OSS output their
        chain-of-thought reasoning in <think>...</think> tags. This content
        should not be shown to users or spoken via TTS.

        Returns:
            Tuple of (cleaned_text, thinking_content)
            - cleaned_text: Response with thinking tags removed
            - thinking_content: The extracted thinking content (for debug logs)
        """
        import re

        # Match <think>...</think> tags (case insensitive, multiline)
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if not matches:
            return text, ""

        # Extract all thinking content
        thinking_content = "\n".join(matches)

        # Remove all <think>...</think> blocks from the text
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace that might be left
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)  # Multiple newlines
        cleaned_text = cleaned_text.strip()

        return cleaned_text, thinking_content

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS to handle special characters properly."""
        import re

        # ALWAYS run character normalization (existing fixes)
        # Replace ALL apostrophe variants with standard apostrophe
        text = text.replace(
            """, "'")  # U+2019 RIGHT SINGLE QUOTATION MARK
        text = text.replace(""",
            "'",
        )  # U+2018 LEFT SINGLE QUOTATION MARK
        text = text.replace("´", "'")  # U+00B4 ACUTE ACCENT
        text = text.replace("`", "'")  # U+0060 GRAVE ACCENT
        text = text.replace("′", "'")  # U+2032 PRIME
        text = text.replace("‛", "'")  # U+201B SINGLE HIGH-REVERSED-9 QUOTATION MARK
        text = text.replace("ʻ", "'")  # U+02BB MODIFIER LETTER TURNED COMMA
        text = text.replace("ʼ", "'")  # U+02BC MODIFIER LETTER APOSTROPHE
        text = text.replace("ˈ", "'")  # U+02C8 MODIFIER LETTER VERTICAL LINE
        text = text.replace("ˊ", "'")  # U+02CA MODIFIER LETTER ACUTE ACCENT
        text = text.replace("ˋ", "'")  # U+02CB MODIFIER LETTER GRAVE ACCENT

        # Replace smart quotes
        text = text.replace('"', '"')  # U+201C LEFT DOUBLE QUOTATION MARK
        text = text.replace('"', '"')  # U+201D RIGHT DOUBLE QUOTATION MARK
        text = text.replace("„", '"')  # U+201E DOUBLE LOW-9 QUOTATION MARK
        text = text.replace("‟", '"')  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK

        # Replace dashes with commas for pauses
        text = text.replace("—", ", ")  # U+2014 EM DASH
        text = text.replace("–", ", ")  # U+2013 EN DASH
        text = text.replace("‒", ", ")  # U+2012 FIGURE DASH
        text = text.replace("―", ", ")  # U+2015 HORIZONTAL BAR

        # Other fixes
        text = text.replace("…", "...")  # U+2026 HORIZONTAL ELLIPSIS
        text = text.replace("•", "-")  # U+2022 BULLET

        # Normalize stray spaces before punctuation so spoken/displayed text
        # does not come out as "it , the weather entity is available."
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)

        # ONLY apply aggressive cleaning if clean_responses enabled
        if not self.clean_responses:
            return text

        # 1. Strip emojis
        text = re.sub(
            r"[\U00010000-\U0010ffff]", "", text
        )  # Supplementary planes (most emojis)
        text = re.sub(
            r"[\u2600-\u26FF\u2700-\u27BF]", "", text
        )  # Misc symbols & dingbats
        text = re.sub(r"[\uE000-\uF8FF]", "", text)  # Private use area

        # 2. Remove markdown (order matters - bold before italic)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** → bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # *italic* → italic
        text = re.sub(r"__(.+?)__", r"\1", text)  # __bold__ → bold
        text = re.sub(r"_(.+?)_", r"\1", text)  # _italic_ → italic
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # [text](url) → text
        text = re.sub(r"`([^`]+)`", r"\1", text)  # `code` → code
        text = re.sub(r"```[\s\S]+?```", "", text)  # ```code block``` → removed
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # # Header → Header

        # 3. Convert symbols to words
        SYMBOL_MAP = {
            "°C": " degrees celsius",
            "°F": " degrees fahrenheit",
            "°": " degrees",
            "%": " percent",
            "€": " euros",
            "£": " pounds",
            "$": " dollars",
            "&": " and",
            "+": " plus",
            "=": " equals",
            "<": " less than",
            ">": " greater than",
            "@": " at",
            "#": " number",
            "×": " times",
            "÷": " divided by",
        }
        for symbol, word in SYMBOL_MAP.items():
            text = text.replace(symbol, word)

        # 4. Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    async def _trigger_tts(self, text: str):
        """Streaming interim TTS is intentionally disabled.

        Home Assistant already handles speaking the final response, and the old
        hardcoded media_player target was both environment-specific and added
        avoidable work during each request.
        """
        del text
        return

    async def _execute_single_tool_call(
        self, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single tool call and return an OpenAI/Ollama tool message."""
        tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        arguments_str = function.get("arguments")

        _LOGGER.info(f"📞 Processing tool call {tool_call_id}: {tool_name}")

        try:
            arguments = self._parse_tool_arguments(arguments_str)

            # Execute the tool
            result = await self._call_mcp_tool(tool_name, arguments)

            # Format result for LLM consumption
            if "error" in result:
                error_data = result["error"]
                if isinstance(error_data, dict):
                    error_msg = error_data.get("message", str(error_data))
                else:
                    error_msg = str(error_data)
                content = f"ERROR: {error_msg}"
            else:
                content = result.get("result", "")

            content = self._compact_tool_result_for_llm(tool_name, content)

            if tool_name == "set_conversation_state" and content:
                if "conversation_state:true" in content.lower():
                    self._expecting_response = True
                    _LOGGER.debug(
                        "🔄 Conversation will continue - expecting response"
                    )
                elif "conversation_state:false" in content.lower():
                    self._expecting_response = False
                    _LOGGER.debug(
                        "🔄 Conversation will close - not expecting response"
                    )

            if self.server_type == SERVER_TYPE_OLLAMA:
                return {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": content if content is not None else "",
                }

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content if content is not None else "",
            }

        except Exception as e:
            _LOGGER.error(f"Error executing tool {tool_name}: {e}")
            error_content = json.dumps({"error": str(e)})
            if self.server_type == SERVER_TYPE_OLLAMA:
                return {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": error_content,
                }

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": error_content,
            }

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute a list of tool calls and return results in OpenAI format."""
        if not tool_calls:
            return []

        return list(await asyncio.gather(
            *(self._execute_single_tool_call(tool_call) for tool_call in tool_calls)
        ))

    async def _test_streaming_basic(self) -> bool:
        """Test basic streaming without tools to isolate connection issues."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 10,
        }

        _LOGGER.info(
            f"🧪 Testing basic streaming to: {self.base_url_dynamic}/v1/chat/completions"
        )
        _LOGGER.info(f"🧪 Model: {self.model_name}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url_dynamic}/v1/chat/completions"
                headers = self._get_auth_headers()
                async with session.post(url, headers=headers, json=payload) as response:
                    _LOGGER.info(
                        f"✅ Basic streaming connected! Status: {response.status}"
                    )
                    _LOGGER.info(f"📋 Headers: {dict(response.headers)}")

                    # Try to read first few lines
                    line_count = 0
                    async for line in response.content:
                        line_str = line.decode("utf-8").strip()
                        _LOGGER.info(f"📨 Line {line_count}: {line_str[:100]}")
                        line_count += 1
                        if line_count >= 3:
                            break

                    _LOGGER.info(
                        f"✅ Basic streaming works! Received {line_count} lines"
                    )
                    return True

        except aiohttp.ClientConnectionError as e:
            _LOGGER.error(f"❌ Connection error: {e}")
            return False
        except Exception as e:
            _LOGGER.error(f"❌ Basic streaming failed: {type(e).__name__}: {e}")
            import traceback

            _LOGGER.error(traceback.format_exc())
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on server type."""
        if self.server_type == SERVER_TYPE_OPENAI:
            # OpenAI uses Bearer token
            # For custom OpenAI-compatible URLs, only send auth if key looks valid
            if self.api_key and len(self.api_key) > 5 and self.api_key.lower() not in ["none", "null", "fake", "na", "n/a"]:
                return {"Authorization": f"Bearer {self.api_key}"}
            else:
                return {}  # No auth for custom services that don't require it
        elif self.server_type == SERVER_TYPE_GEMINI:
            # Gemini OpenAI-compatible endpoint uses Bearer token like OpenAI
            return {"Authorization": f"Bearer {self.api_key}"}
        elif self.server_type == SERVER_TYPE_ANTHROPIC:
            # Anthropic OpenAI-compatible endpoint uses Bearer token
            return {"Authorization": f"Bearer {self.api_key}"}
        elif self.server_type == SERVER_TYPE_OPENROUTER:
            # OpenRouter uses Bearer token with optional HTTP-Referer header
            return {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/mike-nott/mcp-assist",
                "X-Title": "MCP Assist for Home Assistant",
            }
        elif self.server_type == SERVER_TYPE_MOLTBOT:
            # Moltbot uses Bearer token
            return {"Authorization": f"Bearer {self.api_key}"}
        else:
            # Local servers (LM Studio, Ollama, llamacpp, vLLM) don't need auth
            return {}

    def _build_openai_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """Build OpenAI-compatible payload for LM Studio, OpenAI, Gemini, Anthropic, Moltbot, vLLM."""
        payload = {"model": self.model_name, "messages": messages, "stream": stream}

        # Temperature (skip for GPT-5+/o1 models)
        if not (
            self.model_name.startswith("gpt-5") or self.model_name.startswith("o1")
        ):
            payload["temperature"] = self.temperature

        # Token limits
        if self.max_tokens > 0:
            if self.model_name.startswith("gpt-5") or self.model_name.startswith("o1"):
                payload["max_completion_tokens"] = self.max_tokens
            else:
                payload["max_tokens"] = self.max_tokens

        # Tools
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Moltbot: Add session management via user field
        if self.server_type == SERVER_TYPE_MOLTBOT and hasattr(
            self, "_current_conversation_id"
        ):
            payload["user"] = self._current_conversation_id

        return payload

    def _build_ollama_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """Build Ollama native API payload."""
        # Convert tool messages (Ollama doesn't use tool_call_id)
        ollama_messages = []
        for msg in messages:
            if msg.get("role") == "tool":
                tool_msg = {"role": "tool", "content": msg.get("content", "")}
                if msg.get("tool_name"):
                    tool_msg["tool_name"] = msg["tool_name"]
                ollama_messages.append(tool_msg)
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_msg = dict(msg)
                assistant_msg["tool_calls"] = self._format_tool_calls_for_ollama(
                    msg["tool_calls"]
                )
                ollama_messages.append(assistant_msg)
            else:
                ollama_messages.append(msg)

        # Parse keep_alive - can be int (seconds/-1) or string (duration like "5m")
        keep_alive_value = self.ollama_keep_alive
        try:
            # Try to parse as integer (for -1, 0, or seconds)
            keep_alive_value = int(keep_alive_value)
        except (ValueError, TypeError):
            # Keep as string for duration format like "5m", "24h", "-1m"
            pass

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": stream,
            "keep_alive": keep_alive_value,
            "options": {},
        }

        # Temperature
        if self.temperature is not None:
            payload["options"]["temperature"] = self.temperature

        # Token limits
        if self.max_tokens > 0:
            payload["options"]["num_predict"] = self.max_tokens

        # Context window (if configured)
        if self.ollama_num_ctx > 0:
            payload["options"]["num_ctx"] = self.ollama_num_ctx

        # Tools (same format as OpenAI)
        if tools:
            payload["tools"] = tools

        return payload

    async def _call_llm_streaming(self, messages: List[Dict[str, Any]]) -> str:
        """Stream LLM responses with immediate TTS feedback."""
        _LOGGER.info(f"🚀 Starting streaming {self.server_type} conversation")

        # Test streaming once and cache result
        if not hasattr(self, "_streaming_available"):
            self._streaming_available = await self._test_streaming_basic()

        if not self._streaming_available:
            _LOGGER.warning("Streaming not available, falling back to HTTP")
            raise Exception("Streaming not available")

        # Get MCP tools once
        tools = await self._get_mcp_tools()
        conversation_messages = list(messages)

        # Buffers for streaming
        tool_arg_buffers = {}  # index -> partial JSON string
        tool_names = {}  # index -> tool name
        tool_ids = {}  # index -> tool_call_id
        response_text = ""
        sentence_buffer = ""
        completed_tools = set()

        for iteration in range(self.max_iterations):
            _LOGGER.info(f"🔄 Stream iteration {iteration + 1}")
            if self.debug_mode and iteration == 0:
                _LOGGER.info(f"🎯 Using model: {self.model_name}")

            # Debug logging for iteration 2+ if enabled
            if self.debug_mode and iteration >= 1:
                _LOGGER.info(
                    f"🔄 Iteration {iteration + 1}: {len(conversation_messages)} messages to send"
                )
                for i, msg in enumerate(conversation_messages):
                    role = msg.get("role")
                    has_tool_calls = "tool_calls" in msg
                    tool_call_id = msg.get("tool_call_id", "")
                    content_preview = (
                        str(msg.get("content", ""))[:100] if msg.get("content") else ""
                    )
                    _LOGGER.info(
                        f"  Msg {i}: {role}, tool_calls={has_tool_calls}, tool_call_id={tool_call_id}, content={content_preview}"
                    )

            # Clean messages for streaming compatibility
            cleaned_messages = []
            for i, msg in enumerate(conversation_messages):
                # Clean the message for streaming
                cleaned_msg = msg.copy()

                # Fix None content
                if cleaned_msg.get("content") is None:
                    cleaned_msg["content"] = ""

                # Assistant messages with tool_calls must have NO content field at all
                if cleaned_msg.get("role") == "assistant" and cleaned_msg.get(
                    "tool_calls"
                ):
                    cleaned_msg.pop("content", None)  # Remove the field entirely

                cleaned_messages.append(cleaned_msg)

            # Build payload using appropriate method based on server type
            if self.server_type == SERVER_TYPE_OLLAMA:
                payload = self._build_ollama_payload(
                    cleaned_messages, tools, stream=True
                )
            else:
                payload = self._build_openai_payload(
                    cleaned_messages, tools, stream=True
                )

            # Debug: Log actual cleaned payload being sent in iteration 2+
            if self.debug_mode and iteration >= 1:
                _LOGGER.info(
                    f"📤 Sending {len(cleaned_messages)} messages to LLM (iteration {iteration + 1}):"
                )
                _LOGGER.info(f"📤 Model: {self.model_name}")
                _LOGGER.info(f"📤 Temperature: {payload.get('temperature', 'default')}")
                _LOGGER.info(
                    f"📤 Max tokens: {payload.get('max_tokens', payload.get('max_completion_tokens', 'default'))}"
                )
                for i, msg in enumerate(cleaned_messages):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    content_len = len(str(content)) if content else 0
                    if role == "tool":
                        # Show first 200 chars of tool responses
                        preview = str(content)[:200] if content else ""
                        _LOGGER.info(
                            f"  [{i}] {role}: {content_len} chars - {preview}..."
                        )
                    else:
                        _LOGGER.info(f"  [{i}] {role}: {content_len} chars")

            # Only clean if needed (performance optimization)
            clean_payload = payload
            # Quick check if cleaning is needed
            for msg in payload.get("messages", []):
                if (
                    msg.get("role") == "assistant"
                    and "tool_calls" in msg
                    and "content" in msg
                ):
                    # Need to clean - remove content from assistant messages with tool_calls
                    def clean_for_json(obj):
                        """Remove keys with None values recursively."""
                        if isinstance(obj, dict):
                            return {
                                k: clean_for_json(v)
                                for k, v in obj.items()
                                if v is not None
                            }
                        elif isinstance(obj, list):
                            return [clean_for_json(v) for v in obj]
                        return obj

                    clean_payload = clean_for_json(payload)
                    break

            has_tool_calls = False
            current_tool_calls = []
            current_thought_signature = None  # Track Gemini 3 thought signatures

            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Use appropriate endpoint based on server type
                    if self.server_type == SERVER_TYPE_OLLAMA:
                        url = f"{self.base_url_dynamic}/api/chat"
                    else:
                        url = f"{self.base_url_dynamic}/v1/chat/completions"
                    headers = self._get_auth_headers()

                    _LOGGER.info(f"📡 Streaming to: {url}")
                    if self.debug_mode:
                        _LOGGER.debug(
                            f"📦 Payload size: {len(json.dumps(clean_payload))} bytes"
                        )
                        _LOGGER.debug(f"🔧 Using model: {self.model_name}")

                    # Use clean_payload instead of payload
                    async with session.post(
                        url, headers=headers, json=clean_payload
                    ) as response:
                        _LOGGER.info(
                            f"🔌 Connection established, status: {response.status}"
                        )
                        if self.debug_mode:
                            _LOGGER.debug(
                                f"📋 Response headers: {dict(response.headers)}"
                            )

                        if response.status != 200:
                            try:
                                error_data = await response.json()
                                error_text = json.dumps(error_data, indent=2)
                            except Exception:
                                error_text = await response.text()
                            # Fallback to non-streaming
                            _LOGGER.error(
                                f"❌ Streaming failed with status {response.status}"
                            )
                            _LOGGER.error(f"❌ Full error response: {error_text}")
                            raise Exception(
                                f"Streaming failed: {error_text}"
                            )  # Raise to trigger fallback

                        if self.debug_mode:
                            _LOGGER.debug("📖 Starting to read stream...")

                        async for line in response.content:
                            if not line:
                                continue

                            line_str = line.decode("utf-8").strip()

                            try:
                                if self.server_type == SERVER_TYPE_OLLAMA:
                                    # Ollama: Each line is complete JSON
                                    if not line_str:
                                        continue

                                    data = json.loads(line_str)

                                    # Check for completion
                                    if data.get("done"):
                                        break

                                    # Extract message
                                    message = data.get("message", {})
                                    delta = {}

                                    if "content" in message and message["content"]:
                                        delta["content"] = message["content"]

                                    if "tool_calls" in message:
                                        delta["tool_calls"] = message["tool_calls"]

                                else:
                                    # OpenAI: SSE format with "data: " prefix
                                    if not line_str.startswith("data: "):
                                        continue
                                    if line_str == "data: [DONE]":
                                        break

                                    data = json.loads(line_str[6:])
                                    choice = data["choices"][0]
                                    delta = choice.get("delta", {})

                                    # Capture thought_signature from tool_calls (it's inside the first tool_call, not at choice/delta level)
                                    if (
                                        "tool_calls" in delta
                                        and current_thought_signature is None
                                    ):
                                        for tc_delta in delta["tool_calls"]:
                                            if "extra_content" in tc_delta:
                                                google_data = tc_delta.get(
                                                    "extra_content", {}
                                                ).get("google", {})
                                                if "thought_signature" in google_data:
                                                    current_thought_signature = (
                                                        google_data["thought_signature"]
                                                    )
                                                    _LOGGER.info(
                                                        f"🧠 Captured thought_signature: {current_thought_signature[:50]}..."
                                                    )
                                                    break  # Only in first tool_call

                                # Handle streamed content
                                if "content" in delta and delta["content"]:
                                    chunk = delta["content"]
                                    response_text += chunk
                                    sentence_buffer += chunk

                                    # Trigger TTS on complete sentence
                                    if any(
                                        sentence_buffer.endswith(p)
                                        for p in [". ", "! ", "? ", ".\n", "!\n", "?\n"]
                                    ):
                                        await self._trigger_tts(sentence_buffer.strip())
                                        sentence_buffer = ""

                                # Handle streamed tool calls
                                if "tool_calls" in delta:
                                    has_tool_calls = True
                                    for tc in delta["tool_calls"]:
                                        idx = tc.get("index", 0)

                                        # Initialize tool call if new
                                        if idx >= len(current_tool_calls):
                                            current_tool_calls.append({})

                                        if "id" in tc:
                                            tool_ids[idx] = tc["id"]
                                            current_tool_calls[idx]["id"] = tc["id"]
                                            # Add the required type field
                                            current_tool_calls[idx]["type"] = "function"

                                        if "function" in tc:
                                            func = tc["function"]
                                            if "name" in func:
                                                tool_names[idx] = func["name"]
                                                if (
                                                    "function"
                                                    not in current_tool_calls[idx]
                                                ):
                                                    current_tool_calls[idx][
                                                        "function"
                                                    ] = {}
                                                current_tool_calls[idx]["function"][
                                                    "name"
                                                ] = func["name"]
                                                _LOGGER.info(
                                                    f"🔧 Tool streaming: {func['name']}"
                                                )

                                            if "arguments" in func:
                                                if (
                                                    "function"
                                                    not in current_tool_calls[idx]
                                                ):
                                                    current_tool_calls[idx][
                                                        "function"
                                                    ] = {}

                                                raw_arguments = func["arguments"]
                                                if isinstance(raw_arguments, dict):
                                                    current_tool_calls[idx]["function"][
                                                        "arguments"
                                                    ] = self._stringify_tool_arguments(
                                                        raw_arguments
                                                    )

                                                    tool_name = tool_names.get(idx)
                                                    if (
                                                        tool_name
                                                        and idx not in completed_tools
                                                    ):
                                                        completed_tools.add(idx)
                                                        if (
                                                            tool_name
                                                            == "discover_entities"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Looking for devices..."
                                                            )
                                                        elif (
                                                            tool_name
                                                            == "perform_action"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Controlling the device..."
                                                            )
                                                    continue

                                                if idx not in tool_arg_buffers:
                                                    tool_arg_buffers[idx] = ""
                                                tool_arg_buffers[idx] += (
                                                    self._stringify_tool_arguments(
                                                        raw_arguments
                                                    )
                                                )

                                                # Try to parse arguments
                                                try:
                                                    json.loads(
                                                        tool_arg_buffers[idx]
                                                    )
                                                    # Valid JSON - save it
                                                    if (
                                                        "function"
                                                        not in current_tool_calls[idx]
                                                    ):
                                                        current_tool_calls[idx][
                                                            "function"
                                                        ] = {}
                                                    current_tool_calls[idx]["function"][
                                                        "arguments"
                                                    ] = tool_arg_buffers[idx]

                                                    # Quick feedback for tool execution
                                                    tool_name = tool_names.get(idx)
                                                    if (
                                                        tool_name
                                                        and idx not in completed_tools
                                                    ):
                                                        completed_tools.add(idx)
                                                        if (
                                                            tool_name
                                                            == "discover_entities"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Looking for devices..."
                                                            )
                                                        elif (
                                                            tool_name
                                                            == "perform_action"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Controlling the device..."
                                                            )

                                                except json.JSONDecodeError:
                                                    # Still accumulating arguments
                                                    pass

                            except Exception as e:
                                _LOGGER.debug(f"Stream parsing: {e}")

            except Exception as stream_error:
                _LOGGER.error(
                    f"❌ Streaming iteration {iteration + 1} failed: {stream_error}"
                )
                if iteration == 0:
                    # First iteration failed, try fallback
                    raise stream_error
                else:
                    # Later iteration failed, return what we have
                    break

            # Handle any remaining sentence
            if sentence_buffer.strip():
                await self._trigger_tts(sentence_buffer.strip())
                sentence_buffer = ""

            # If we got tool calls, execute them
            if has_tool_calls and current_tool_calls:
                _LOGGER.info(
                    f"⚡ Executing {len(current_tool_calls)} streamed tool calls"
                )
                if self.debug_mode:
                    _LOGGER.debug(
                        f"📝 Discarding intermediate narration: {len(response_text)} chars"
                    )
                    _LOGGER.debug(
                        f"📊 Tool calls structure: {json.dumps(current_tool_calls, indent=2)}"
                    )

                # Add assistant message with tool calls
                # LM Studio streaming requires NO content field at all when tool_calls exist
                # Gemini 3: thought_signature goes INSIDE each tool_call, not at message level
                if current_thought_signature is not None:
                    for tool_call in current_tool_calls:
                        tool_call["extra_content"] = {
                            "google": {"thought_signature": current_thought_signature}
                        }
                    _LOGGER.info(
                        f"🧠 Added thought_signature to {len(current_tool_calls)} tool calls"
                    )
                elif self.server_type == SERVER_TYPE_GEMINI:
                    # Only warn for Gemini - other providers don't use thought_signature
                    _LOGGER.warning(
                        "⚠️ No thought_signature captured for Gemini 3 (this will cause 400 error on next turn)"
                    )

                assistant_msg = {
                    "role": "assistant",
                    "tool_calls": current_tool_calls
                    # NO content field - must be completely absent
                }
                if self.server_type == SERVER_TYPE_OLLAMA:
                    assistant_msg["tool_calls"] = self._format_tool_calls_for_ollama(
                        current_tool_calls
                    )
                    if response_text:
                        assistant_msg["content"] = response_text

                conversation_messages.append(assistant_msg)

                # Record tool calls to ChatLog for debug view
                self._record_tool_calls_to_chatlog(current_tool_calls)

                # Execute tools
                tool_results = await self._execute_tool_calls(current_tool_calls)

                # Record tool results to ChatLog for debug view
                for idx, result in enumerate(tool_results):
                    if idx < len(current_tool_calls):
                        tc = current_tool_calls[idx]
                        tool_call_id = result.get(
                            "tool_call_id", tc.get("id", "unknown")
                        )
                        tool_name = tc.get("function", {}).get("name", "unknown")
                        # Parse content as JSON if possible, otherwise use as-is
                        try:
                            tool_result_data = json.loads(result.get("content", "{}"))
                        except Exception:
                            tool_result_data = {"result": result.get("content", "")}
                        self._record_tool_result_to_chatlog(
                            tool_call_id, tool_name, tool_result_data
                        )

                conversation_messages.extend(tool_results)

                # Reset for next iteration - we don't want intermediate narration in final response
                response_text = (
                    ""  # Clear accumulated text since it was just pre-tool narration
                )
                tool_arg_buffers.clear()
                tool_names.clear()
                tool_ids.clear()
                completed_tools.clear()

                # Continue to get next response after tools
                continue
            else:
                # No tool calls, return the response
                if response_text:
                    return response_text
                else:
                    # No content and no tools, might need another iteration
                    _LOGGER.warning("Empty response from streaming, retrying...")

        # Hit max iterations
        if response_text:
            return response_text
        else:
            return f"I reached the maximum of {self.max_iterations} tool calls while processing your request. Try simplifying your request, or increase the limit in Advanced Settings if you have a complex automation need."

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Call LLM API with MCP tools and handle tool execution loop."""
        # Try streaming first, fallback to HTTP if needed
        try:
            return await self._call_llm_streaming(messages)
        except Exception as e:
            _LOGGER.warning(f"Streaming failed ({e}), using HTTP fallback")
            return await self._call_llm_http(messages)

    async def _call_llm_http(self, messages: List[Dict[str, Any]]) -> str:
        """Original HTTP-based LLM call (fallback)."""
        _LOGGER.info(f"🚀 Using HTTP fallback for {self.server_type}")

        # Get MCP tools once
        tools = await self._get_mcp_tools()
        if not tools:
            _LOGGER.warning("No MCP tools available - proceeding without tools")

        # Keep a mutable copy of messages for the conversation
        conversation_messages = list(messages)

        # Tool execution loop
        for iteration in range(self.max_iterations):
            _LOGGER.info(
                f"🔄 HTTP Iteration {iteration + 1}: Calling {self.server_type} with {len(conversation_messages)} messages"
            )

            # Build payload using appropriate method based on server type
            if self.server_type == SERVER_TYPE_OLLAMA:
                payload = self._build_ollama_payload(
                    conversation_messages, tools, stream=False
                )
            else:
                payload = self._build_openai_payload(
                    conversation_messages, tools, stream=False
                )

            # Clean payload to remove None values and ensure no content in assistant+tool_calls
            def clean_for_json_http(obj):
                """Remove keys with None values recursively."""
                if isinstance(obj, dict):
                    cleaned = {}
                    for k, v in obj.items():
                        if v is not None:
                            # Special handling for messages
                            if k == "messages" and isinstance(v, list):
                                cleaned_messages = []
                                for msg in v:
                                    cleaned_msg = clean_for_json_http(msg)
                                    # Ensure assistant+tool_calls has no content field
                                    if (
                                        cleaned_msg.get("role") == "assistant"
                                        and "tool_calls" in cleaned_msg
                                    ):
                                        cleaned_msg.pop("content", None)
                                    cleaned_messages.append(cleaned_msg)
                                cleaned[k] = cleaned_messages
                            else:
                                cleaned[k] = clean_for_json_http(v)
                    return cleaned
                elif isinstance(obj, list):
                    return [clean_for_json_http(v) for v in obj]
                return obj

            clean_payload = clean_for_json_http(payload)

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Use appropriate endpoint based on server type
                if self.server_type == SERVER_TYPE_OLLAMA:
                    url = f"{self.base_url_dynamic}/api/chat"
                else:
                    url = f"{self.base_url_dynamic}/v1/chat/completions"
                headers = self._get_auth_headers()

                async with session.post(
                    url, headers=headers, json=clean_payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"{self.server_type} API error {response.status}: {error_text}"
                        )

                    data = await response.json()

                    # Parse response based on server type
                    thought_signature = None  # Track for Gemini 3
                    if self.server_type == SERVER_TYPE_OLLAMA:
                        # Ollama: Direct message field
                        message = data.get("message", {})
                    else:
                        # OpenAI: Wrapped in choices array
                        if "choices" not in data or not data["choices"]:
                            raise Exception(f"No response from {self.server_type}")
                        choice = data["choices"][0]
                        message = choice.get("message", {})

                    # Check if there are tool calls to execute
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_calls = (
                            self._format_tool_calls_for_ollama(message["tool_calls"])
                            if self.server_type == SERVER_TYPE_OLLAMA
                            else self._normalize_tool_call_arguments(
                                message["tool_calls"]
                            )
                        )
                        _LOGGER.info(
                            f"🛠️ {self.server_type} requested {len(tool_calls)} tool calls"
                        )

                        # Capture thought_signature from first tool_call (Gemini 3)
                        if tool_calls and "extra_content" in tool_calls[0]:
                            google_data = (
                                tool_calls[0].get("extra_content", {}).get("google", {})
                            )
                            if "thought_signature" in google_data:
                                thought_signature = google_data["thought_signature"]
                                _LOGGER.info(
                                    f"🧠 Captured thought_signature: {thought_signature[:50]}..."
                                )

                        # Ensure each tool_call has the required type field
                        for tc in tool_calls:
                            if "type" not in tc:
                                tc["type"] = "function"
                            if "function" in tc:
                                _LOGGER.info(
                                    f"  - {tc['function'].get('name')}: {tc['function'].get('arguments')}"
                                )

                        # Preserve thought_signature in tool_calls for Gemini 3
                        # It should already be there from the response, just keep it

                        assistant_msg = {
                            "role": "assistant",
                            "tool_calls": tool_calls
                            # NO content field - must be completely absent
                        }
                        if (
                            self.server_type == SERVER_TYPE_OLLAMA
                            and message.get("content")
                        ):
                            assistant_msg["content"] = message.get("content")

                        conversation_messages.append(assistant_msg)

                        # Record tool calls to ChatLog for debug view
                        self._record_tool_calls_to_chatlog(tool_calls)

                        # Execute the tool calls
                        _LOGGER.info("⚡ Executing tool calls against MCP server...")
                        tool_results = await self._execute_tool_calls(tool_calls)

                        # Record tool results to ChatLog for debug view
                        for idx, result in enumerate(tool_results):
                            if idx < len(tool_calls):
                                tc = tool_calls[idx]
                                tool_call_id = result.get(
                                    "tool_call_id", tc.get("id", "unknown")
                                )
                                tool_name = tc.get("function", {}).get(
                                    "name", "unknown"
                                )
                                # Parse content as JSON if possible, otherwise use as-is
                                try:
                                    tool_result_data = json.loads(
                                        result.get("content", "{}")
                                    )
                                except Exception:
                                    tool_result_data = {
                                        "result": result.get("content", "")
                                    }
                                self._record_tool_result_to_chatlog(
                                    tool_call_id, tool_name, tool_result_data
                                )

                        # Add tool results to conversation
                        conversation_messages.extend(tool_results)

                        _LOGGER.info(
                            f"📊 Added {len(tool_results)} tool results to conversation"
                        )

                        # Continue the loop to get next response
                        continue

                    else:
                        # No more tool calls, we have the final response
                        final_content = message.get("content", "").strip()
                        _LOGGER.info(
                            f"💬 Final response received (length: {len(final_content)})"
                        )
                        _LOGGER.info(f"💬 Full response: {final_content}")
                        return final_content

        # If we hit max iterations, return what we have
        _LOGGER.warning(
            f"⚠️ Hit maximum iterations ({self.max_iterations}) in tool execution loop"
        )
        return f"I reached the maximum of {self.max_iterations} tool calls while processing your request. Try simplifying your request, or increase the limit in Advanced Settings if you have a complex automation need."

    async def _execute_actions(
        self, response_text: str, user_input: ConversationInput
    ) -> List[Dict[str, Any]]:
        """Parse response for any action information.

        NOTE: With MCP tools, LM Studio executes actions directly via the MCP server.
        We don't need to parse intents or execute them - just return info about what happened.
        """
        actions_taken = []

        # MCP tools are executed by LM Studio directly, so we just log what was mentioned
        # The actual actions have already been performed via MCP's perform_action tool

        _LOGGER.info(
            "MCP-enabled response completed. Actions were executed via MCP tools if needed."
        )

        # We could parse the response to extract what was done for logging purposes
        # but the actual execution happens through MCP, not here

        if (
            "turned on" in response_text.lower()
            or "turning on" in response_text.lower()
        ):
            actions_taken.append(
                {"type": "mcp_action", "description": "Turned on devices via MCP"}
            )
        elif (
            "turned off" in response_text.lower()
            or "turning off" in response_text.lower()
        ):
            actions_taken.append(
                {"type": "mcp_action", "description": "Turned off devices via MCP"}
            )
        elif "toggled" in response_text.lower():
            actions_taken.append(
                {"type": "mcp_action", "description": "Toggled devices via MCP"}
            )

        return actions_taken
