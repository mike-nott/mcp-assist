"""LM Studio MCP conversation agent."""

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation import chat_log
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    intent,
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    llm,
    chat_session,
)
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_PROFILE_NAME,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
    CONF_DEBUG_MODE,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_FOLLOW_UP_MODE,
    CONF_RESPONSE_MODE,
    CONF_SERVER_TYPE,
    CONF_API_KEY,
    CONF_CONTROL_HA,
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    CONF_SEARCH_PROVIDER,
    CONF_BRAVE_API_KEY,
    CONF_ALLOWED_IPS,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_FOLLOW_UP_PHRASES,
    CONF_END_WORDS,
    CONF_CLEAN_RESPONSES,
    CONF_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    DEFAULT_DEBUG_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_FOLLOW_UP_MODE,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_MCP_PORT,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_ENABLE_GAP_FILLING,
    DEFAULT_SERVER_TYPE,
    DEFAULT_API_KEY,
    DEFAULT_CONTROL_HA,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_FOLLOW_UP_PHRASES,
    DEFAULT_END_WORDS,
    DEFAULT_CLEAN_RESPONSES,
    DEFAULT_TIMEOUT,
    RESPONSE_MODE_INSTRUCTIONS,
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
)
from .conversation_history import ConversationHistory

_LOGGER = logging.getLogger(__name__)


class MCPAssistConversationEntity(ConversationEntity):
    """MCP Assist conversation entity with multi-provider support."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the MCP Assist conversation entity."""
        super().__init__()

        self.hass = hass
        self.entry = entry
        self.history = ConversationHistory()
        self._current_chat_log = None  # ChatLog for debug view tracking

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
            _LOGGER.debug(f"🔍 Debug mode: ON")
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
                    tool_args=json.loads(tc.get("function", {}).get("arguments", "{}")),
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

    async def _build_system_prompt_with_context(
        self, user_input: ConversationInput
    ) -> str:
        """Build system prompt with Smart Entity Index."""
        try:
            # Get base prompts (check options first, then data, then defaults)
            system_prompt = self.entry.options.get(
                CONF_SYSTEM_PROMPT,
                self.entry.data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
            )
            technical_prompt = self.entry.options.get(
                CONF_TECHNICAL_PROMPT,
                self.entry.data.get(CONF_TECHNICAL_PROMPT, DEFAULT_TECHNICAL_PROMPT),
            )

            # Format time and date variables
            current_time = dt_util.now().strftime("%H:%M:%S")
            current_date = dt_util.now().strftime("%Y-%m-%d")
            technical_prompt = technical_prompt.replace("{time}", current_time)
            technical_prompt = technical_prompt.replace("{date}", current_date)

            # Get current area from satellite (if available)
            current_area = await self._get_current_area(user_input)
            technical_prompt = technical_prompt.replace("{current_area}", current_area)

            # Inject mode-specific instructions
            mode_instructions = RESPONSE_MODE_INSTRUCTIONS.get(
                self.follow_up_mode, RESPONSE_MODE_INSTRUCTIONS["default"]
            )
            technical_prompt = technical_prompt.replace(
                "{response_mode}", mode_instructions
            )

            # Get Smart Entity Index from IndexManager
            index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")
            if index_manager:
                index = await index_manager.get_index()
                index_json = json.dumps(index, indent=2)
            else:
                index_json = "{}"
                _LOGGER.warning("IndexManager not available, using empty index")

            # Replace {index} placeholder
            technical_prompt = technical_prompt.replace("{index}", index_json)

            # Combine: system prompt + technical prompt
            return f"{system_prompt}\n\n{technical_prompt}"

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
            # Get prompts (check options first, then data, then defaults)
            system_prompt = self.entry.options.get(
                CONF_SYSTEM_PROMPT,
                self.entry.data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
            )
            technical_prompt = self.entry.options.get(
                CONF_TECHNICAL_PROMPT,
                self.entry.data.get(CONF_TECHNICAL_PROMPT, DEFAULT_TECHNICAL_PROMPT),
            )

            # Format time and date variables
            current_time = dt_util.now().strftime("%H:%M:%S")
            current_date = dt_util.now().strftime("%Y-%m-%d")

            # Replace placeholders in technical prompt
            technical_prompt = technical_prompt.replace("{time}", current_time)
            technical_prompt = technical_prompt.replace("{date}", current_date)
            technical_prompt = technical_prompt.replace("{current_area}", "Unknown")
            technical_prompt = technical_prompt.replace("{index}", "{}")

            # Combine prompts
            return f"{system_prompt}\n\n{technical_prompt}"

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
            # Add conversation history (last 5 turns)
            for turn in history[-5:]:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    async def _get_mcp_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch available tools from MCP server."""
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
                        tools = data["result"]["tools"]
                        _LOGGER.info("Retrieved %d MCP tools", len(tools))

                        # Convert to OpenAI format for LM Studio
                        openai_tools = []
                        tool_names = []
                        for tool in tools:
                            openai_tools.append(
                                {
                                    "type": "function",
                                    "function": {
                                        "name": tool["name"],
                                        "description": tool["description"],
                                        "parameters": tool.get("inputSchema", {}),
                                    },
                                }
                            )
                            tool_names.append(tool["name"])

                        _LOGGER.info("MCP tools available: %s", ", ".join(tool_names))
                        if "perform_action" in tool_names:
                            _LOGGER.info("✅ perform_action tool is available")
                        else:
                            _LOGGER.warning("⚠️ perform_action tool NOT found!")

                        return openai_tools
                    return None

        except Exception as err:
            _LOGGER.error("Failed to get MCP tools: %s", err)
            return None

    async def _call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single MCP tool and return the result."""
        _LOGGER.info(f"🔧 Executing MCP tool: {tool_name} with args: {arguments}")

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
        """Send text to TTS for immediate feedback."""
        if not text or len(text) < 3:  # Skip very short fragments
            return

        _LOGGER.info(f"🔊 TTS: {text[:50]}...")

        # Use HA's TTS service for immediate feedback
        try:
            # Get the default TTS service
            await self.hass.services.async_call(
                "tts",
                "speak",
                {
                    "message": self._clean_text_for_tts(text),
                    "entity_id": "media_player.default",  # Adjust to your setup
                    "cache": True,  # Cache for faster response
                },
                blocking=False,  # Don't wait for TTS to complete
            )
        except Exception as e:
            _LOGGER.debug(f"TTS not available or failed: {e}")
            # Don't fail the whole request if TTS fails

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute a list of tool calls and return results in OpenAI format."""
        results = []

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            arguments_str = function.get("arguments", "{}")

            _LOGGER.info(f"📞 Processing tool call {tool_call_id}: {tool_name}")

            try:
                # Parse arguments based on server type
                if self.server_type == SERVER_TYPE_OLLAMA:
                    # Ollama: Arguments are already parsed objects
                    arguments = (
                        arguments_str
                        if isinstance(arguments_str, dict)
                        else json.loads(arguments_str)
                        if arguments_str
                        else {}
                    )
                else:
                    # OpenAI: Arguments are JSON strings
                    arguments = json.loads(arguments_str) if arguments_str else {}

                # Execute the tool
                result = await self._call_mcp_tool(tool_name, arguments)

                # Format result for OpenAI
                if "error" in result:
                    # Extract error message as plain text so LLM actually reads it
                    error_data = result["error"]
                    if isinstance(error_data, dict):
                        error_msg = error_data.get("message", str(error_data))
                    else:
                        error_msg = str(error_data)
                    content = f"ERROR: {error_msg}"
                else:
                    content = result.get("result", "")

                # Check if this is the conversation state tool
                if tool_name == "set_conversation_state" and content:
                    # Parse the expecting_response value from the result
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

                # Format result based on server type
                if self.server_type == SERVER_TYPE_OLLAMA:
                    # Ollama doesn't use tool_call_id
                    results.append(
                        {
                            "role": "tool",
                            "content": content if content is not None else "",
                        }
                    )
                else:
                    # OpenAI format with tool_call_id
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content if content is not None else "",
                        }
                    )

                _LOGGER.info(f"✅ Tool {tool_name} executed successfully")

            except Exception as e:
                _LOGGER.error(f"Error executing tool {tool_name}: {e}")
                # Format error result based on server type
                if self.server_type == SERVER_TYPE_OLLAMA:
                    # Ollama doesn't use tool_call_id
                    results.append(
                        {"role": "tool", "content": json.dumps({"error": str(e)})}
                    )
                else:
                    # OpenAI format with tool_call_id
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"error": str(e)}),
                        }
                    )

        return results

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
            return {"Authorization": f"Bearer {self.api_key}"}
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
                ollama_messages.append(
                    {"role": "tool", "content": msg.get("content", "")}
                )
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
                            except:
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
                                                if idx not in tool_arg_buffers:
                                                    tool_arg_buffers[idx] = ""
                                                tool_arg_buffers[idx] += func[
                                                    "arguments"
                                                ]

                                                # Try to parse arguments
                                                try:
                                                    args_json = json.loads(
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
                        except:
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
                        tool_calls = message["tool_calls"]
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
                                except:
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
