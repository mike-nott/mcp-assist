"""LM Studio MCP conversation agent."""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Literal

import aiohttp

from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent, area_registry as ar, device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
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
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    DEFAULT_DEBUG_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_FOLLOW_UP_MODE,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_SERVER_TYPE,
    DEFAULT_API_KEY,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_ANTHROPIC,
    OPENAI_BASE_URL,
    GEMINI_BASE_URL,
    ANTHROPIC_BASE_URL,
)
from .conversation_history import ConversationHistory

_LOGGER = logging.getLogger(__name__)


class MCPAssistAgent(AbstractConversationAgent):
    """MCP Assist conversation agent with multi-provider support."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history = ConversationHistory()

        # Static configuration (doesn't change)
        data = entry.data
        self.server_type = data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        # Set base URL based on server type (static)
        if self.server_type == SERVER_TYPE_OPENAI:
            self.base_url = OPENAI_BASE_URL
        elif self.server_type == SERVER_TYPE_GEMINI:
            self.base_url = GEMINI_BASE_URL
        elif self.server_type == SERVER_TYPE_ANTHROPIC:
            self.base_url = ANTHROPIC_BASE_URL
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
            self.base_url_dynamic
        )

    # Dynamic configuration properties - read from entry.options/data each time
    @property
    def base_url_dynamic(self) -> str:
        """Get base URL (dynamic for local servers)."""
        if self.server_type in [SERVER_TYPE_OPENAI, SERVER_TYPE_GEMINI, SERVER_TYPE_ANTHROPIC]:
            return self.base_url  # Static
        else:
            # LM Studio/Ollama - read dynamically
            return self.entry.options.get(
                CONF_LMSTUDIO_URL,
                self.entry.data.get(CONF_LMSTUDIO_URL, "")
            ).rstrip("/")

    @property
    def api_key(self) -> str:
        """Get API key (dynamic)."""
        return self.entry.options.get(CONF_API_KEY, self.entry.data.get(CONF_API_KEY, DEFAULT_API_KEY))

    @property
    def model_name(self) -> str:
        """Get model name (dynamic)."""
        return self.entry.options.get(CONF_MODEL_NAME, self.entry.data.get(CONF_MODEL_NAME, ""))

    @property
    def mcp_port(self) -> int:
        """Get MCP port (dynamic)."""
        return self.entry.options.get(CONF_MCP_PORT, self.entry.data.get(CONF_MCP_PORT, 0))

    @property
    def debug_mode(self) -> bool:
        """Get debug mode (dynamic)."""
        return self.entry.options.get(CONF_DEBUG_MODE, self.entry.data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE))

    @property
    def max_iterations(self) -> int:
        """Get max iterations (dynamic)."""
        return self.entry.options.get(CONF_MAX_ITERATIONS, self.entry.data.get(CONF_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS))

    @property
    def max_tokens(self) -> int:
        """Get max tokens (dynamic)."""
        return self.entry.options.get(CONF_MAX_TOKENS, self.entry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))

    @property
    def temperature(self) -> float:
        """Get temperature (dynamic)."""
        return self.entry.options.get(CONF_TEMPERATURE, self.entry.data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))

    @property
    def follow_up_mode(self) -> str:
        """Get response mode (dynamic, with backward compatibility)."""
        return self.entry.options.get(
            CONF_RESPONSE_MODE,
            self.entry.data.get(
                CONF_RESPONSE_MODE,
                self.entry.options.get(
                    CONF_FOLLOW_UP_MODE,
                    self.entry.data.get(CONF_FOLLOW_UP_MODE, DEFAULT_RESPONSE_MODE)
                )
            )
        )

    @property
    def attribution(self) -> str:
        """Return attribution."""
        server_name = {
            SERVER_TYPE_LMSTUDIO: "LM Studio",
            SERVER_TYPE_OLLAMA: "Ollama",
            SERVER_TYPE_OPENAI: "OpenAI",
            SERVER_TYPE_GEMINI: "Google Gemini",
            SERVER_TYPE_ANTHROPIC: "Anthropic Claude",
        }.get(self.server_type, "LLM")
        return f"Powered by {server_name} with MCP entity discovery"

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"  # Support all languages

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process user input and return response."""
        _LOGGER.info("🎤 Voice request started - Processing: %s", user_input.text)

        try:
            _LOGGER.debug("Getting conversation ID...")
            # Get conversation history
            conversation_id = user_input.conversation_id or "default"
            _LOGGER.debug("Conversation ID: %s", conversation_id)

            _LOGGER.debug("Getting history...")
            history = self.history.get_history(conversation_id)
            _LOGGER.debug("History retrieved: %d turns", len(history))

            # Build system prompt with context
            system_prompt = await self._build_system_prompt_with_context(user_input)
            if self.debug_mode:
                _LOGGER.info(f"📝 System prompt built, length: {len(system_prompt)} chars")
                _LOGGER.info(f"📝 System prompt preview: {system_prompt[:200]}...")

            # Build conversation messages
            messages = self._build_messages(system_prompt, user_input.text, history)
            if self.debug_mode:
                _LOGGER.info(f"📨 Messages built: {len(messages)} messages")
                for i, msg in enumerate(messages):
                    role = msg.get('role')
                    content_len = len(msg.get('content', '')) if msg.get('content') else 0
                    _LOGGER.info(f"  Message {i}: role={role}, content_length={content_len}")

            # Call LLM API
            _LOGGER.info(f"📡 Calling {self.server_type} API...")
            response_text = await self._call_llm(messages)
            _LOGGER.info(f"✅ {self.server_type} response received, length: %d", len(response_text))

            # Parse response and execute any Home Assistant actions
            actions_taken = await self._execute_actions(response_text, user_input)

            # Store in conversation history
            self.history.add_turn(
                conversation_id,
                user_input.text,
                response_text,
                actions=actions_taken
            )

            # Create intent response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response_text)

            # Note: Card data removed as it was causing JSON serialization errors
            # Actions are already executed via MCP tools, so card isn't needed

            # Determine follow-up mode
            if self.follow_up_mode == "always":
                # Always continue regardless of tool
                continue_conversation = True
            elif self.follow_up_mode == "none":
                # Never continue regardless of tool
                continue_conversation = False
            else:  # "default" - smart mode
                # Use the LLM's indication if it called the tool
                if hasattr(self, '_expecting_response'):
                    continue_conversation = self._expecting_response
                    # Clear for next conversation
                    delattr(self, '_expecting_response')
                    _LOGGER.debug("🎯 Using LLM's set_conversation_state indication")
                else:
                    # LLM didn't indicate, use pattern detection as fallback
                    continue_conversation = self._detect_follow_up_patterns(response_text)
                    if continue_conversation:
                        _LOGGER.debug("🎯 Pattern detection triggered continuation")
                    else:
                        _LOGGER.debug("🎯 No patterns detected, closing conversation")

            _LOGGER.debug(f"🎯 Follow-up mode: {self.follow_up_mode}, Continue: {continue_conversation}")

            return ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
                continue_conversation=continue_conversation
            )

        except Exception as err:
            _LOGGER.exception("Error processing conversation")

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I encountered an error: {err}"
            )

            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False  # Don't continue on errors
            )

    def _detect_follow_up_patterns(self, text: str) -> bool:
        """Detect if the response expects a follow-up based on patterns."""
        if not text:
            return False

        # Check last 200 characters for efficiency
        check_text = text[-200:].lower()

        # Pattern 1: Ends with a question mark
        if check_text.rstrip().endswith('?'):
            _LOGGER.debug("📊 Pattern detected: ends with question mark")
            return True

        # Pattern 2: Question phrases
        question_phrases = [
            "which one", "would you like", "would you prefer", "should i",
            "do you want", "how about", "shall i", "can i", "may i",
            "want me to", "need anything else", "anything else"
        ]

        for phrase in question_phrases:
            if phrase in check_text:
                _LOGGER.debug(f"📊 Pattern detected: question phrase '{phrase}'")
                return True

        # Pattern 3: Offering alternatives
        alternative_patterns = [
            " or ", "instead", "alternatively"
        ]

        # Only check for alternatives if there's also a question indicator
        if any(pattern in check_text for pattern in alternative_patterns):
            # Check if it's actually offering a choice (has question indicators)
            if any(phrase in check_text for phrase in ["would you", "do you", "should", "?"]):
                _LOGGER.debug("📊 Pattern detected: offering alternatives")
                return True

        return False

    async def _get_current_area(self, user_input: ConversationInput) -> str:
        """Get the area of the satellite/device making the request."""
        try:
            # Try to get device_id from context
            device_id = user_input.device_id if hasattr(user_input, 'device_id') else None

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
            _LOGGER.info("📍 Current area detected: %s (from device %s)", area_name, device_id)
            return area_name

        except Exception as e:
            _LOGGER.warning("Error getting current area: %s", e)
            return "Unknown"

    async def _build_system_prompt_with_context(self, user_input: ConversationInput) -> str:
        """Build system prompt with Smart Entity Index."""
        try:
            # Get base prompts (check options first, then data, then defaults)
            system_prompt = self.entry.options.get(CONF_SYSTEM_PROMPT,
                                                    self.entry.data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT))
            technical_prompt = self.entry.options.get(CONF_TECHNICAL_PROMPT,
                                                       self.entry.data.get(CONF_TECHNICAL_PROMPT, DEFAULT_TECHNICAL_PROMPT))

            # Format time and date variables
            current_time = dt_util.now().strftime('%H:%M:%S')
            current_date = dt_util.now().strftime('%Y-%m-%d')
            technical_prompt = technical_prompt.replace('{time}', current_time)
            technical_prompt = technical_prompt.replace('{date}', current_date)

            # Get current area from satellite (if available)
            current_area = await self._get_current_area(user_input)
            technical_prompt = technical_prompt.replace('{current_area}', current_area)

            # Get Smart Entity Index from IndexManager
            index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")
            if index_manager:
                index = await index_manager.get_index()
                index_json = json.dumps(index, indent=2)
            else:
                index_json = "{}"
                _LOGGER.warning("IndexManager not available, using empty index")

            # Replace {index} placeholder
            technical_prompt = technical_prompt.replace('{index}', index_json)

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
            system_prompt = self.entry.options.get(CONF_SYSTEM_PROMPT,
                                                    self.entry.data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT))
            technical_prompt = self.entry.options.get(CONF_TECHNICAL_PROMPT,
                                                       self.entry.data.get(CONF_TECHNICAL_PROMPT, DEFAULT_TECHNICAL_PROMPT))

            # Format time and date variables
            current_time = dt_util.now().strftime('%H:%M:%S')
            current_date = dt_util.now().strftime('%Y-%m-%d')

            # Replace placeholders in technical prompt
            technical_prompt = technical_prompt.replace('{time}', current_time)
            technical_prompt = technical_prompt.replace('{date}', current_date)
            technical_prompt = technical_prompt.replace('{current_area}', 'Unknown')
            technical_prompt = technical_prompt.replace('{index}', '{}')

            # Combine prompts
            return f"{system_prompt}\n\n{technical_prompt}"

        except Exception as e:
            _LOGGER.error("Error building system prompt: %s", e)
            # Return a basic prompt as fallback
            return "You are a Home Assistant voice assistant. Use MCP tools to control devices."

    def _build_messages(
        self,
        system_prompt: str,
        user_text: str,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build message list for LM Studio."""
        messages = [{"role": "system", "content": system_prompt}]

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
                        "id": 1
                    }
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
                            openai_tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "description": tool["description"],
                                    "parameters": tool.get("inputSchema", {})
                                }
                            })
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

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single MCP tool and return the result."""
        _LOGGER.info(f"🔧 Executing MCP tool: {tool_name} with args: {arguments}")

        try:
            mcp_url = f"http://localhost:{self.mcp_port}"

            # Create JSON-RPC request for tool execution
            request_id = f"tool_{uuid.uuid4().hex[:8]}"
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": request_id
            }

            _LOGGER.debug(f"MCP request: {json.dumps(payload, indent=2)}")

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{mcp_url}/", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(f"MCP tool call failed: {response.status} - {error_text}")
                        return {"error": f"Tool execution failed: {error_text}"}

                    data = await response.json()
                    _LOGGER.debug(f"MCP response: {json.dumps(data, indent=2)}")

                    if "result" in data and "content" in data["result"]:
                        # Extract the text content from the MCP response
                        content = data["result"]["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text_result = content[0].get("text", "")
                            if self.debug_mode:
                                _LOGGER.info(f"🔍 MCP tool '{tool_name}' returned {len(text_result)} chars")
                                _LOGGER.info(f"🔍 Full result (repr): {repr(text_result)}")
                                # Also log each line separately for readability
                                for i, line in enumerate(text_result.split('\n')):
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

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS to handle special characters properly."""
        # Replace ALL apostrophe variants with standard apostrophe
        text = text.replace(''', "'")  # U+2019 RIGHT SINGLE QUOTATION MARK
        text = text.replace(''', "'")  # U+2018 LEFT SINGLE QUOTATION MARK
        text = text.replace('´', "'")  # U+00B4 ACUTE ACCENT
        text = text.replace('`', "'")  # U+0060 GRAVE ACCENT
        text = text.replace('′', "'")  # U+2032 PRIME
        text = text.replace('‛', "'")  # U+201B SINGLE HIGH-REVERSED-9 QUOTATION MARK
        text = text.replace('ʻ', "'")  # U+02BB MODIFIER LETTER TURNED COMMA
        text = text.replace('ʼ', "'")  # U+02BC MODIFIER LETTER APOSTROPHE
        text = text.replace('ˈ', "'")  # U+02C8 MODIFIER LETTER VERTICAL LINE
        text = text.replace('ˊ', "'")  # U+02CA MODIFIER LETTER ACUTE ACCENT
        text = text.replace('ˋ', "'")  # U+02CB MODIFIER LETTER GRAVE ACCENT

        # Replace smart quotes
        text = text.replace('"', '"')  # U+201C LEFT DOUBLE QUOTATION MARK
        text = text.replace('"', '"')  # U+201D RIGHT DOUBLE QUOTATION MARK
        text = text.replace('„', '"')  # U+201E DOUBLE LOW-9 QUOTATION MARK
        text = text.replace('‟', '"')  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK

        # Replace dashes with commas for pauses
        text = text.replace('—', ', ')  # U+2014 EM DASH
        text = text.replace('–', ', ')  # U+2013 EN DASH
        text = text.replace('‒', ', ')  # U+2012 FIGURE DASH
        text = text.replace('―', ', ')  # U+2015 HORIZONTAL BAR

        # Other fixes
        text = text.replace('…', '...')  # U+2026 HORIZONTAL ELLIPSIS
        text = text.replace('•', '-')    # U+2022 BULLET

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
                'tts',
                'speak',
                {
                    'message': self._clean_text_for_tts(text),
                    'entity_id': 'media_player.default',  # Adjust to your setup
                    'cache': True  # Cache for faster response
                },
                blocking=False  # Don't wait for TTS to complete
            )
        except Exception as e:
            _LOGGER.debug(f"TTS not available or failed: {e}")
            # Don't fail the whole request if TTS fails

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a list of tool calls and return results in OpenAI format."""
        results = []

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            arguments_str = function.get("arguments", "{}")

            _LOGGER.info(f"📞 Processing tool call {tool_call_id}: {tool_name}")

            try:
                # Parse arguments from JSON string
                arguments = json.loads(arguments_str) if arguments_str else {}

                # Execute the tool
                result = await self._call_mcp_tool(tool_name, arguments)

                # Format result for OpenAI
                if "error" in result:
                    content = json.dumps({"error": result["error"]})
                else:
                    content = result.get("result", "")

                # Check if this is the conversation state tool
                if tool_name == "set_conversation_state" and content:
                    # Parse the expecting_response value from the result
                    if "conversation_state:true" in content.lower():
                        self._expecting_response = True
                        _LOGGER.debug("🔄 Conversation will continue - expecting response")
                    elif "conversation_state:false" in content.lower():
                        self._expecting_response = False
                        _LOGGER.debug("🔄 Conversation will close - not expecting response")

                # Add tool result to conversation (required for OpenAI strict message format)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content if content is not None else ""
                })

                _LOGGER.info(f"✅ Tool {tool_name} executed successfully")

            except Exception as e:
                _LOGGER.error(f"Error executing tool {tool_name}: {e}")
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": str(e)})
                })

        return results

    async def _test_streaming_basic(self) -> bool:
        """Test basic streaming without tools to isolate connection issues."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 10
        }

        _LOGGER.info(f"🧪 Testing basic streaming to: {self.base_url_dynamic}/v1/chat/completions")
        _LOGGER.info(f"🧪 Model: {self.model_name}")

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url_dynamic}/v1/chat/completions"
                headers = self._get_auth_headers()
                async with session.post(url, headers=headers, json=payload) as response:
                    _LOGGER.info(f"✅ Basic streaming connected! Status: {response.status}")
                    _LOGGER.info(f"📋 Headers: {dict(response.headers)}")

                    # Try to read first few lines
                    line_count = 0
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        _LOGGER.info(f"📨 Line {line_count}: {line_str[:100]}")
                        line_count += 1
                        if line_count >= 3:
                            break

                    _LOGGER.info(f"✅ Basic streaming works! Received {line_count} lines")
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
        else:
            # Local servers (LM Studio, Ollama) don't need auth
            return {}

    async def _call_llm_streaming(self, messages: List[Dict[str, Any]]) -> str:
        """Stream LLM responses with immediate TTS feedback."""
        _LOGGER.info(f"🚀 Starting streaming {self.server_type} conversation")

        # Test streaming once and cache result
        if not hasattr(self, '_streaming_available'):
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
                _LOGGER.info(f"🔄 Iteration {iteration + 1}: {len(conversation_messages)} messages to send")
                for i, msg in enumerate(conversation_messages):
                    role = msg.get('role')
                    has_tool_calls = 'tool_calls' in msg
                    tool_call_id = msg.get('tool_call_id', '')
                    content_preview = str(msg.get('content', ''))[:100] if msg.get('content') else ''
                    _LOGGER.info(f"  Msg {i}: {role}, tool_calls={has_tool_calls}, tool_call_id={tool_call_id}, content={content_preview}")

            # Clean messages for streaming compatibility
            cleaned_messages = []
            for i, msg in enumerate(conversation_messages):

                # Clean the message for streaming
                cleaned_msg = msg.copy()

                # Fix None content
                if cleaned_msg.get('content') is None:
                    cleaned_msg['content'] = ""

                # Assistant messages with tool_calls must have NO content field at all
                if cleaned_msg.get('role') == 'assistant' and cleaned_msg.get('tool_calls'):
                    cleaned_msg.pop('content', None)  # Remove the field entirely

                cleaned_messages.append(cleaned_msg)


            payload = {
                "model": self.model_name,
                "messages": cleaned_messages,  # Use cleaned messages
                "stream": True  # Enable streaming
            }

            # GPT-5+ and o1 models don't support custom temperature (only default of 1)
            if not (self.model_name.startswith("gpt-5") or self.model_name.startswith("o1")):
                payload["temperature"] = self.temperature

            # Add token limit parameter - GPT-5+ uses max_completion_tokens, older models use max_tokens
            if self.max_tokens > 0:
                if self.model_name.startswith("gpt-5") or self.model_name.startswith("o1"):
                    payload["max_completion_tokens"] = self.max_tokens
                else:
                    payload["max_tokens"] = self.max_tokens

            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            # Debug: Log actual cleaned payload being sent in iteration 2+
            if self.debug_mode and iteration >= 1:
                _LOGGER.info(f"📤 Sending {len(cleaned_messages)} messages to LLM (iteration {iteration + 1}):")
                _LOGGER.info(f"📤 Model: {self.model_name}")
                _LOGGER.info(f"📤 Temperature: {payload.get('temperature', 'default')}")
                _LOGGER.info(f"📤 Max tokens: {payload.get('max_tokens', payload.get('max_completion_tokens', 'default'))}")
                for i, msg in enumerate(cleaned_messages):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    content_len = len(str(content)) if content else 0
                    if role == "tool":
                        # Show first 200 chars of tool responses
                        preview = str(content)[:200] if content else ""
                        _LOGGER.info(f"  [{i}] {role}: {content_len} chars - {preview}...")
                    else:
                        _LOGGER.info(f"  [{i}] {role}: {content_len} chars")


            # Only clean if needed (performance optimization)
            clean_payload = payload
            # Quick check if cleaning is needed
            for msg in payload.get('messages', []):
                if msg.get('role') == 'assistant' and 'tool_calls' in msg and 'content' in msg:
                    # Need to clean - remove content from assistant messages with tool_calls
                    def clean_for_json(obj):
                        """Remove keys with None values recursively."""
                        if isinstance(obj, dict):
                            return {k: clean_for_json(v) for k, v in obj.items() if v is not None}
                        elif isinstance(obj, list):
                            return [clean_for_json(v) for v in obj]
                        return obj
                    clean_payload = clean_for_json(payload)
                    break

            has_tool_calls = False
            current_tool_calls = []

            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    url = f"{self.base_url_dynamic}/v1/chat/completions"
                    headers = self._get_auth_headers()

                    _LOGGER.info(f"📡 Streaming to: {url}")
                    if self.debug_mode:
                        _LOGGER.debug(f"📦 Payload size: {len(json.dumps(clean_payload))} bytes")
                        _LOGGER.debug(f"🔧 Using model: {self.model_name}")

                    # Use clean_payload instead of payload
                    async with session.post(url, headers=headers, json=clean_payload) as response:
                        _LOGGER.info(f"🔌 Connection established, status: {response.status}")
                        if self.debug_mode:
                            _LOGGER.debug(f"📋 Response headers: {dict(response.headers)}")

                        if response.status != 200:
                            error_text = await response.text()
                            # Fallback to non-streaming
                            _LOGGER.error(f"❌ Streaming failed with status {response.status}: {error_text[:500]}")
                            raise Exception(f"Streaming failed: {error_text}")  # Raise to trigger fallback

                        if self.debug_mode:
                            _LOGGER.debug("📖 Starting to read stream...")

                        async for line in response.content:

                            if not line:
                                continue

                            line_str = line.decode('utf-8').strip()
                            if not line_str.startswith('data: '):
                                continue

                            if line_str == 'data: [DONE]':
                                break

                            try:
                                data = json.loads(line_str[6:])
                                delta = data['choices'][0].get('delta', {})

                                # Handle streamed content
                                if 'content' in delta and delta['content']:
                                    chunk = delta['content']
                                    response_text += chunk
                                    sentence_buffer += chunk

                                    # Trigger TTS on complete sentence
                                    if any(sentence_buffer.endswith(p) for p in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                                        await self._trigger_tts(sentence_buffer.strip())
                                        sentence_buffer = ""

                                # Handle streamed tool calls
                                if 'tool_calls' in delta:
                                    has_tool_calls = True
                                    for tc in delta['tool_calls']:
                                        idx = tc.get('index', 0)

                                        # Initialize tool call if new
                                        if idx >= len(current_tool_calls):
                                            current_tool_calls.append({})

                                        if 'id' in tc:
                                            tool_ids[idx] = tc['id']
                                            current_tool_calls[idx]['id'] = tc['id']
                                            # Add the required type field
                                            current_tool_calls[idx]['type'] = 'function'

                                        if 'function' in tc:
                                            func = tc['function']
                                            if 'name' in func:
                                                tool_names[idx] = func['name']
                                                if 'function' not in current_tool_calls[idx]:
                                                    current_tool_calls[idx]['function'] = {}
                                                current_tool_calls[idx]['function']['name'] = func['name']
                                                _LOGGER.info(f"🔧 Tool streaming: {func['name']}")

                                            if 'arguments' in func:
                                                if idx not in tool_arg_buffers:
                                                    tool_arg_buffers[idx] = ""
                                                tool_arg_buffers[idx] += func['arguments']

                                                # Try to parse arguments
                                                try:
                                                    args_json = json.loads(tool_arg_buffers[idx])
                                                    # Valid JSON - save it
                                                    if 'function' not in current_tool_calls[idx]:
                                                        current_tool_calls[idx]['function'] = {}
                                                    current_tool_calls[idx]['function']['arguments'] = tool_arg_buffers[idx]

                                                    # Quick feedback for tool execution
                                                    tool_name = tool_names.get(idx)
                                                    if tool_name and idx not in completed_tools:
                                                        completed_tools.add(idx)
                                                        if tool_name == "discover_entities":
                                                            await self._trigger_tts("Looking for devices...")
                                                        elif tool_name == "perform_action":
                                                            await self._trigger_tts("Controlling the device...")

                                                except json.JSONDecodeError:
                                                    # Still accumulating arguments
                                                    pass

                            except Exception as e:
                                _LOGGER.debug(f"Stream parsing: {e}")

            except Exception as stream_error:
                _LOGGER.error(f"❌ Streaming iteration {iteration + 1} failed: {stream_error}")
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
                _LOGGER.info(f"⚡ Executing {len(current_tool_calls)} streamed tool calls")
                if self.debug_mode:
                    _LOGGER.debug(f"📝 Discarding intermediate narration: {len(response_text)} chars")
                    _LOGGER.debug(f"📊 Tool calls structure: {json.dumps(current_tool_calls, indent=2)}")

                # Add assistant message with tool calls
                # LM Studio streaming requires NO content field at all when tool_calls exist
                assistant_msg = {
                    "role": "assistant",
                    "tool_calls": current_tool_calls
                    # NO content field - must be completely absent
                }
                conversation_messages.append(assistant_msg)

                # Execute tools
                tool_results = await self._execute_tool_calls(current_tool_calls)
                conversation_messages.extend(tool_results)

                # Reset for next iteration - we don't want intermediate narration in final response
                response_text = ""  # Clear accumulated text since it was just pre-tool narration
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

        return response_text if response_text else "I'm processing your request."

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
            _LOGGER.info(f"🔄 HTTP Iteration {iteration + 1}: Calling {self.server_type} with {len(conversation_messages)} messages")

            payload = {
                "model": self.model_name,
                "messages": conversation_messages,
                "stream": False
            }

            # GPT-5+ and o1 models don't support custom temperature (only default of 1)
            if not (self.model_name.startswith("gpt-5") or self.model_name.startswith("o1")):
                payload["temperature"] = self.temperature

            # Add token limit parameter - GPT-5+ uses max_completion_tokens, older models use max_tokens
            if self.max_tokens > 0:
                if self.model_name.startswith("gpt-5") or self.model_name.startswith("o1"):
                    payload["max_completion_tokens"] = self.max_tokens
                else:
                    payload["max_tokens"] = self.max_tokens

            # Add tools if available
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            # Clean payload to remove None values and ensure no content in assistant+tool_calls
            def clean_for_json_http(obj):
                """Remove keys with None values recursively."""
                if isinstance(obj, dict):
                    cleaned = {}
                    for k, v in obj.items():
                        if v is not None:
                            # Special handling for messages
                            if k == 'messages' and isinstance(v, list):
                                cleaned_messages = []
                                for msg in v:
                                    cleaned_msg = clean_for_json_http(msg)
                                    # Ensure assistant+tool_calls has no content field
                                    if cleaned_msg.get('role') == 'assistant' and 'tool_calls' in cleaned_msg:
                                        cleaned_msg.pop('content', None)
                                    cleaned_messages.append(cleaned_msg)
                                cleaned[k] = cleaned_messages
                            else:
                                cleaned[k] = clean_for_json_http(v)
                    return cleaned
                elif isinstance(obj, list):
                    return [clean_for_json_http(v) for v in obj]
                return obj

            clean_payload = clean_for_json_http(payload)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url_dynamic}/v1/chat/completions"
                headers = self._get_auth_headers()

                async with session.post(url, headers=headers, json=clean_payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"{self.server_type} API error {response.status}: {error_text}")

                    data = await response.json()

                    if "choices" not in data or not data["choices"]:
                        raise Exception(f"No response from {self.server_type}")

                    choice = data["choices"][0]
                    message = choice.get("message", {})

                    # Check if there are tool calls to execute
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_calls = message["tool_calls"]
                        _LOGGER.info(f"🛠️ {self.server_type} requested {len(tool_calls)} tool calls")

                        # Ensure each tool_call has the required type field
                        for tc in tool_calls:
                            if 'type' not in tc:
                                tc['type'] = 'function'
                            if "function" in tc:
                                _LOGGER.info(f"  - {tc['function'].get('name')}: {tc['function'].get('arguments')}")

                        # Add assistant message with tool calls to conversation
                        # LM Studio requires NO content field at all when tool_calls exist
                        assistant_msg = {
                            "role": "assistant",
                            "tool_calls": tool_calls
                            # NO content field - must be completely absent
                        }
                        conversation_messages.append(assistant_msg)

                        # Execute the tool calls
                        _LOGGER.info("⚡ Executing tool calls against MCP server...")
                        tool_results = await self._execute_tool_calls(tool_calls)

                        # Add tool results to conversation
                        conversation_messages.extend(tool_results)

                        _LOGGER.info(f"📊 Added {len(tool_results)} tool results to conversation")

                        # Continue the loop to get next response
                        continue

                    else:
                        # No more tool calls, we have the final response
                        final_content = message.get("content", "").strip()
                        _LOGGER.info(f"💬 Final response received (length: {len(final_content)})")
                        return final_content

        # If we hit max iterations, return what we have
        _LOGGER.warning("⚠️ Hit maximum iterations (5) in tool execution loop")
        return "I'm still processing your request. Please try again."

    async def _execute_actions(
        self,
        response_text: str,
        user_input: ConversationInput
    ) -> List[Dict[str, Any]]:
        """Parse response for any action information.

        NOTE: With MCP tools, LM Studio executes actions directly via the MCP server.
        We don't need to parse intents or execute them - just return info about what happened.
        """
        actions_taken = []

        # MCP tools are executed by LM Studio directly, so we just log what was mentioned
        # The actual actions have already been performed via MCP's perform_action tool

        _LOGGER.info("MCP-enabled response completed. Actions were executed via MCP tools if needed.")

        # We could parse the response to extract what was done for logging purposes
        # but the actual execution happens through MCP, not here

        if "turned on" in response_text.lower() or "turning on" in response_text.lower():
            actions_taken.append({"type": "mcp_action", "description": "Turned on devices via MCP"})
        elif "turned off" in response_text.lower() or "turning off" in response_text.lower():
            actions_taken.append({"type": "mcp_action", "description": "Turned off devices via MCP"})
        elif "toggled" in response_text.lower():
            actions_taken.append({"type": "mcp_action", "description": "Toggled devices via MCP"})

        return actions_taken