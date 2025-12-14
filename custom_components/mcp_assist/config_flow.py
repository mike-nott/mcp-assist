"""Config flow for MCP Assist integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    BooleanSelector,
)

from .const import (
    DOMAIN,
    CONF_PROFILE_NAME,
    CONF_SERVER_TYPE,
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_AUTO_START,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
    CONF_CONTROL_HA,
    CONF_FOLLOW_UP_MODE,
    CONF_RESPONSE_MODE,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    CONF_MAX_HISTORY,
    CONF_MAX_ITERATIONS,
    CONF_DEBUG_MODE,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_BRAVE_API_KEY,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_GEMINI,
    DEFAULT_SERVER_TYPE,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MCP_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    DEFAULT_CONTROL_HA,
    DEFAULT_FOLLOW_UP_MODE,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_DEBUG_MODE,
    DEFAULT_ENABLE_CUSTOM_TOOLS,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_API_KEY,
    OPENAI_BASE_URL,
    GEMINI_BASE_URL,
    OPENAI_MODELS,
    GEMINI_MODELS,
)

_LOGGER = logging.getLogger(__name__)


async def fetch_models_from_lmstudio(hass: HomeAssistant, url: str) -> list[str]:
    """Fetch available models from inference server."""
    _LOGGER.info("🌐 FETCH: Starting model fetch from %s", url)
    try:
        # Small delay to ensure server is ready
        await asyncio.sleep(0.5)

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            _LOGGER.info("📡 FETCH: Sending request to %s/v1/models", url)
            async with session.get(f"{url}/v1/models") as resp:
                _LOGGER.info("📥 FETCH: Got response with status %d", resp.status)
                if resp.status != 200:
                    _LOGGER.warning("⚠️ FETCH: Non-200 status, returning empty list")
                    return []

                models = await resp.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]
                sorted_models = sorted(model_ids) if model_ids else []
                _LOGGER.info("✨ FETCH: Returning %d sorted models: %s", len(sorted_models), sorted_models)
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: Exception during fetch: %s", err, exc_info=True)
        return []


STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Required(CONF_PROFILE_NAME): str,
    vol.Required(CONF_SERVER_TYPE, default=DEFAULT_SERVER_TYPE): SelectSelector(
        SelectSelectorConfig(
            options=[
                {"value": "lmstudio", "label": "LM Studio"},
                {"value": "ollama", "label": "Ollama"},
                {"value": "openai", "label": "OpenAI"},
                {"value": "gemini", "label": "Google Gemini"},
            ],
            mode=SelectSelectorMode.LIST,
        )
    ),
})

STEP_MCP_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MCP_PORT, default=DEFAULT_MCP_PORT): vol.Coerce(int),
        vol.Required(CONF_AUTO_START, default=True): bool,
    }
)


async def validate_lmstudio_connection(
    hass: HomeAssistant, data: dict[str, Any]
) -> dict[str, Any]:
    """Validate LM Studio connection."""
    url = data[CONF_LMSTUDIO_URL].rstrip("/")
    model_name = data[CONF_MODEL_NAME]

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test models endpoint
            async with session.get(f"{url}/v1/models") as resp:
                if resp.status != 200:
                    raise CannotConnect(f"LM Studio not responding (status {resp.status})")

                models = await resp.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]

                if not model_ids:
                    raise NoModelsLoaded("No models loaded in LM Studio")

                # Check if specified model exists
                if model_name not in model_ids:
                    _LOGGER.warning(
                        "Model '%s' not found. Available models: %s",
                        model_name,
                        model_ids
                    )

            # Test chat completions endpoint
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False
            }

            async with session.post(f"{url}/v1/chat/completions", json=test_payload) as resp:
                if resp.status != 200:
                    raise InvalidModel(f"Model '{model_name}' not working (status {resp.status})")

    except aiohttp.ClientError as err:
        raise CannotConnect(f"Failed to connect to LM Studio: {err}") from err

    return {"title": f"LM Studio ({model_name})"}


class MCPAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for MCP Assist."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize."""
        self.step1_data: dict[str, Any] = {}
        self.step2_data: dict[str, Any] = {}
        self.step3_data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 1 - profile name and server type."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate profile name is not empty
            profile_name = user_input.get(CONF_PROFILE_NAME, "").strip()
            if not profile_name:
                errors[CONF_PROFILE_NAME] = "profile_name_required"
            else:
                # Store data and move to step 2
                self.step1_data = user_input
                return await self.async_step_server()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_server(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 2 - server configuration (URL for local, API key for cloud)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate MCP port
            mcp_port = user_input[CONF_MCP_PORT]
            if not 1024 <= mcp_port <= 65535:
                errors[CONF_MCP_PORT] = "invalid_port"

            if not errors:
                # Store data and move to step 3
                self.step2_data = user_input
                return await self.async_step_model()

        # Get server type from step 1 to build dynamic schema
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        # Build schema based on server type
        if server_type in [SERVER_TYPE_LMSTUDIO, SERVER_TYPE_OLLAMA]:
            # Local servers - show URL field
            default_url = DEFAULT_OLLAMA_URL if server_type == SERVER_TYPE_OLLAMA else DEFAULT_LMSTUDIO_URL
            server_schema = vol.Schema({
                vol.Required(CONF_LMSTUDIO_URL, default=default_url): str,
                vol.Required(CONF_MCP_PORT, default=DEFAULT_MCP_PORT): vol.Coerce(int),
            })
        else:
            # Cloud providers - show API key field
            server_schema = vol.Schema({
                vol.Required(CONF_API_KEY): str,
                vol.Required(CONF_MCP_PORT, default=DEFAULT_MCP_PORT): vol.Coerce(int),
            })

        return self.async_show_form(
            step_id="server",
            data_schema=server_schema,
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 3 - model selection and prompts."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Store data and move to step 4 (advanced)
            self.step3_data = user_input
            return await self.async_step_advanced()

        # Get server URL from step 2 to fetch models
        server_url = self.step2_data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL).rstrip("/")

        # Try to fetch models from the user-provided URL
        _LOGGER.debug("Attempting to fetch models from %s", server_url)
        models = await fetch_models_from_lmstudio(self.hass, server_url)
        _LOGGER.debug("Fetched %d models: %s", len(models), models)

        # Build dynamic schema based on whether models were fetched
        if models:
            # Show dropdown with available models
            _LOGGER.info("Showing model dropdown with %d models", len(models))
            model_schema = vol.Schema({
                vol.Required(CONF_MODEL_NAME): SelectSelector(
                    SelectSelectorConfig(
                        options=models,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
                vol.Required(CONF_TECHNICAL_PROMPT, default=DEFAULT_TECHNICAL_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
            })
        else:
            # Show text input as fallback
            _LOGGER.info("No models fetched, showing text input")
            model_schema = vol.Schema({
                vol.Required(CONF_MODEL_NAME, default=DEFAULT_MODEL_NAME): str,
                vol.Required(CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
                vol.Required(CONF_TECHNICAL_PROMPT, default=DEFAULT_TECHNICAL_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
            })

        return self.async_show_form(
            step_id="model",
            data_schema=model_schema,
            errors=errors,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 4 - advanced settings."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Combine data from all 4 steps
            combined_data = {
                **self.step1_data,
                **self.step2_data,
                **self.step3_data,
                **user_input
            }

            # Create unique ID based on profile name
            profile_name = combined_data[CONF_PROFILE_NAME]
            server_type = combined_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
            server_display = "Ollama" if server_type == "ollama" else "LM Studio"

            unique_id = f"{DOMAIN}_{profile_name.lower().replace(' ', '_')}"
            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title=f"{server_display} - {profile_name}",
                data=combined_data,
            )

        advanced_schema = vol.Schema({
            vol.Required(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            ),
            vol.Required(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): vol.Coerce(int),
            vol.Required(CONF_RESPONSE_MODE, default=DEFAULT_RESPONSE_MODE): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "default", "label": "Smart"},
                        {"value": "always", "label": "Always"},
                        {"value": "none", "label": "None"},
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY): vol.Coerce(int),
            vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
            vol.Required(CONF_MAX_ITERATIONS, default=DEFAULT_MAX_ITERATIONS): vol.Coerce(int),
            vol.Optional(CONF_BRAVE_API_KEY, default=DEFAULT_BRAVE_API_KEY): str,
            vol.Required(CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE): bool,
        })

        return self.async_show_form(
            step_id="advanced",
            data_schema=advanced_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow for this handler."""
        return MCPAssistOptionsFlow()


class MCPAssistOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for MCP Assist integration."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            # Support both old and new config keys
            if CONF_FOLLOW_UP_MODE in user_input and CONF_RESPONSE_MODE not in user_input:
                user_input[CONF_RESPONSE_MODE] = user_input[CONF_FOLLOW_UP_MODE]
                del user_input[CONF_FOLLOW_UP_MODE]

            # Update entry title if profile name changed
            new_profile_name = user_input.get(CONF_PROFILE_NAME)
            if new_profile_name:
                # Get server type from original data (can't be changed in options)
                server_type = self.config_entry.data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
                server_display = "Ollama" if server_type == "ollama" else "LM Studio"
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    title=f"{server_display} - {new_profile_name}"
                )

            return self.async_create_entry(title="", data=user_input)

        # Get current values from options, then data, then defaults
        options = self.config_entry.options
        data = self.config_entry.data

        # Handle backward compatibility
        response_mode_value = options.get(CONF_RESPONSE_MODE,
                                         options.get(CONF_FOLLOW_UP_MODE,
                                         DEFAULT_RESPONSE_MODE))

        # Fetch models from server
        lmstudio_url = options.get(CONF_LMSTUDIO_URL, data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL)).rstrip("/")
        _LOGGER.info("🔍 OPTIONS: Attempting to fetch models from %s", lmstudio_url)
        try:
            models = await fetch_models_from_lmstudio(self.hass, lmstudio_url)
            _LOGGER.info("✅ OPTIONS: Successfully fetched %d models: %s", len(models), models)
        except Exception as err:
            _LOGGER.error("❌ OPTIONS: Failed to fetch models: %s", err)
            models = []
        current_model = options.get(CONF_MODEL_NAME, data.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME))

        # Build model selector based on whether models were fetched
        if models:
            # Show dropdown with available models
            model_selector = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        else:
            # Show text input as fallback
            model_selector = str

        options_schema = vol.Schema(
            {
                # 1. Profile Name
                vol.Required(
                    CONF_PROFILE_NAME,
                    default=options.get(CONF_PROFILE_NAME, data.get(CONF_PROFILE_NAME, "Default"))
                ): str,

                # 2. Server URL
                vol.Required(
                    CONF_LMSTUDIO_URL,
                    default=lmstudio_url
                ): str,

                # 3. Model Name (dynamic: dropdown if models found, text input if not)
                vol.Required(
                    CONF_MODEL_NAME,
                    default=current_model
                ): model_selector,

                # 4. System Prompt
                vol.Required(
                    CONF_SYSTEM_PROMPT,
                    default=options.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)),

                # 5. Technical Instructions
                vol.Required(
                    CONF_TECHNICAL_PROMPT,
                    default=options.get(CONF_TECHNICAL_PROMPT, DEFAULT_TECHNICAL_PROMPT)
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)),

                # 6. Temperature
                vol.Required(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),

                # 7. Max Response Tokens
                vol.Required(
                    CONF_MAX_TOKENS,
                    default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
                ): vol.Coerce(int),

                # 8. Response Mode
                vol.Required(
                    CONF_RESPONSE_MODE,
                    default=response_mode_value
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": "default", "label": "Smart"},
                            {"value": "always", "label": "Always"},
                            {"value": "none", "label": "None"},
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),

                # 9. Max History Messages
                vol.Required(
                    CONF_MAX_HISTORY,
                    default=options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
                ): vol.Coerce(int),

                # 10. Control Home Assistant
                vol.Required(
                    CONF_CONTROL_HA,
                    default=options.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA)
                ): bool,

                # 11. Max Tool Iterations
                vol.Required(
                    CONF_MAX_ITERATIONS,
                    default=options.get(CONF_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS)
                ): vol.Coerce(int),

                # 12. Enable Custom Tools
                vol.Optional(
                    CONF_ENABLE_CUSTOM_TOOLS,
                    default=options.get(CONF_ENABLE_CUSTOM_TOOLS, DEFAULT_ENABLE_CUSTOM_TOOLS)
                ): bool,

                # 13. Brave API Key
                vol.Optional(
                    CONF_BRAVE_API_KEY,
                    default=options.get(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY)
                ): str,

                # 14. MCP Server Port
                vol.Required(
                    CONF_MCP_PORT,
                    default=options.get(CONF_MCP_PORT, data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT))
                ): vol.Coerce(int),

                # 15. Debug Mode
                vol.Required(
                    CONF_DEBUG_MODE,
                    default=options.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE)
                ): bool,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            description_placeholders={
                "lmstudio_url": data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL),
            },
        )


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class NoModelsLoaded(HomeAssistantError):
    """Error to indicate no models are loaded."""


class InvalidModel(HomeAssistantError):
    """Error to indicate the model is invalid."""