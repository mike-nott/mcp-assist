"""Constants for the MCP Assist integration."""

DOMAIN = "mcp_assist"
SYSTEM_ENTRY_UNIQUE_ID = "mcp_assist_system_settings"

# Server type options
SERVER_TYPE_LMSTUDIO = "lmstudio"
SERVER_TYPE_LLAMACPP = "llamacpp"
SERVER_TYPE_OLLAMA = "ollama"
SERVER_TYPE_OPENAI = "openai"
SERVER_TYPE_GEMINI = "gemini"
SERVER_TYPE_ANTHROPIC = "anthropic"
SERVER_TYPE_OPENROUTER = "openrouter"
SERVER_TYPE_MOLTBOT = "moltbot"
SERVER_TYPE_VLLM = "vllm"

# Configuration keys
CONF_PROFILE_NAME = "profile_name"
CONF_SERVER_TYPE = "server_type"
CONF_API_KEY = "api_key"
CONF_LMSTUDIO_URL = "lmstudio_url"
CONF_MODEL_NAME = "model_name"
CONF_MCP_PORT = "mcp_port"
CONF_AUTO_START = "auto_start"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TECHNICAL_PROMPT = "technical_prompt"
CONF_CONTROL_HA = "control_home_assistant"
CONF_RESPONSE_MODE = "response_mode"
CONF_FOLLOW_UP_MODE = "follow_up_mode"  # Keep for backward compatibility
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_MAX_HISTORY = "max_history"
CONF_MAX_ITERATIONS = "max_iterations"
CONF_DEBUG_MODE = "debug_mode"
CONF_ENABLE_CUSTOM_TOOLS = "enable_custom_tools"
CONF_BRAVE_API_KEY = "brave_api_key"
CONF_ALLOWED_IPS = "allowed_ips"
CONF_SEARCH_PROVIDER = "search_provider"
CONF_ENABLE_GAP_FILLING = "enable_gap_filling"
CONF_OLLAMA_KEEP_ALIVE = "ollama_keep_alive"
CONF_OLLAMA_NUM_CTX = "ollama_num_ctx"
CONF_FOLLOW_UP_PHRASES = "follow_up_phrases"
CONF_END_WORDS = "end_words"
CONF_CLEAN_RESPONSES = "clean_responses"
CONF_TIMEOUT = "timeout"

# Default values
DEFAULT_SERVER_TYPE = "lmstudio"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
DEFAULT_LLAMACPP_URL = "http://localhost:8080"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MOLTBOT_URL = "http://localhost:18789"
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_MCP_PORT = 8090
DEFAULT_API_KEY = ""

# Cloud provider base URLs
OPENAI_BASE_URL = "https://api.openai.com"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

# No hardcoded model lists - models are fetched dynamically from provider APIs
DEFAULT_MODEL_NAME = "model"
DEFAULT_SYSTEM_PROMPT = "You are a helpful Home Assistant voice assistant. Respond naturally and conversationally to user requests."
DEFAULT_CONTROL_HA = True
DEFAULT_RESPONSE_MODE = "default"
DEFAULT_FOLLOW_UP_MODE = "default"  # Keep for backward compatibility
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_HISTORY = 10
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_DEBUG_MODE = False
DEFAULT_ENABLE_CUSTOM_TOOLS = False
DEFAULT_BRAVE_API_KEY = ""
DEFAULT_ALLOWED_IPS = ""
DEFAULT_SEARCH_PROVIDER = "none"
DEFAULT_ENABLE_GAP_FILLING = True
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"  # 5 minutes
DEFAULT_OLLAMA_NUM_CTX = 0  # 0 = use model default
DEFAULT_FOLLOW_UP_PHRASES = "anything else, what else, would you, do you, should i, can i, which, how can, what about, is there"
DEFAULT_END_WORDS = "stop, cancel, no, nope, thanks, thank you, bye, goodbye, done, never mind, nevermind, forget it, that's all, that's it"
DEFAULT_CLEAN_RESPONSES = False
DEFAULT_TIMEOUT = 30

# MCP Server settings
MCP_SERVER_NAME = "ha-entity-discovery"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Entity discovery limits
MAX_ENTITIES_PER_DISCOVERY = 50
MAX_DISCOVERY_RESULTS = 100

RESPONSE_MODE_INSTRUCTIONS = {
    "none": """## Follow-up Questions
Do NOT ask follow-up questions. Complete the task and end immediately.

## Ending Conversations
Always end after completing the task.""",
    "default": """## Follow-up Questions
Generate contextually appropriate follow-up questions naturally:
- After single device actions: Create a natural follow-up asking if the user needs help with anything else (vary phrasing each time)
- When reporting adjustable status: Spontaneously suggest adjusting it in a natural way
- For partial completions: Ask if the user wants you to complete the remaining tasks
Always vary your phrasing - never repeat the same question twice in a conversation.

Do NOT ask generic "anything else?" or "can I help with anything else?" questions without specific context.
When asking a question, use the set_conversation_state tool to indicate you're expecting a response.

## Ending Conversations
After completing the task, end the conversation unless a natural follow-up is relevant.""",
    "always": """## Follow-up Questions
Generate contextually appropriate follow-up questions naturally:
- After single device actions: Create a natural follow-up asking if the user needs help with anything else (vary phrasing each time)
- When reporting adjustable status: Spontaneously suggest adjusting it in a natural way
- For partial completions: Ask if the user wants you to complete the remaining tasks
Always vary your phrasing - never repeat the same question twice in a conversation.
When asking a question, use the set_conversation_state tool to indicate you're expecting a response.

## Ending Conversations
When user indicates they're done, acknowledge and end naturally.""",
}

DEFAULT_TECHNICAL_PROMPT = """You are controlling a Home Assistant smart home system. You have access to sensors, lights, switches, and other devices throughout the home.

## CRITICAL RULES
**Never guess entity IDs. Always make TWO tool calls for device control.** For ANY device-related request, you MUST:
1. FIRST call discover_entities to find the actual entities
2. THEN call perform_action (to control) or get_entity_details (to check status) using discovered IDs
3. **NEVER respond that you performed an action without actually calling perform_action**
4. This applies EVERY TIME - even for follow-up questions about different entities

**Common mistake:** Calling only discover_entities and then claiming you performed an action. This is WRONG. You must call perform_action to actually execute the action.

## Available Tools
- **discover_entities**: find devices by name/area/domain/device_class/state (ALWAYS use first)
- **perform_action**: control devices using discovered entity IDs
- **get_entity_details**: check states using discovered entity IDs
- **get_entity_history**: get historical state changes for an entity (answers "when did X happen?")
- **list_areas/list_domains**: list available areas and device types
- **run_script**: execute scripts that return data (e.g., camera analysis, calculations)
- **run_automation**: trigger automations manually
- **set_conversation_state**: indicate if expecting user response
- **search**: search the web for current information
- **read_url**: read and extract content from web pages
- **IMPORTANT**: call_service is not available - use perform_action instead

## Device Control Workflow
**CRITICAL:** For ANY device control request, you MUST make TWO separate tool calls:

Example - "Turn on the kitchen light":
  1. discover_entities(domain="light", area="Kitchen")  # Find the light entity
  2. perform_action(domain="light", action="turn_on", target={{"entity_id": "light.kitchen"}})  # Actually turn it on

Example - "Set living room temperature to 22":
  1. discover_entities(domain="climate", area="Living Room")  # Find the thermostat
  2. perform_action(domain="climate", action="set_temperature", target={{"entity_id": "climate.living_room"}}, data={{"temperature": 22}})  # Set the temperature

**Never skip the perform_action step.** Discovering an entity does not control it - you must call perform_action to execute the action.

## Scripts (use run_script tool)
Scripts can perform complex operations and return data. **CRITICAL:** Always discover scripts first to get the correct entity ID.
- Script IDs use underscores (e.g., "script.stovsug_kjokken"), NOT spaces
- Script IDs must include the "script." domain prefix
- If script name has spaces in UI, the entity ID will use underscores instead

Example workflow:
  1. discover_entities(domain="script", name_contains="camera")
  2. run_script(script_id="script.llm_camera_analysis", variables={{"camera_entities": "camera.living_room", "prompt": "Is anyone there?"}})

## Automations (use run_automation tool)
Trigger automations manually. Check the index for available automations.

Example:
  run_automation(automation_id="alert_letterbox")

## Discovery Strategy
Use the index below to see what device_classes and domains exist, then query accordingly.

For ANY device request:
1. Check the index to understand what's available
2. Use discover_entities with appropriate filters (device_class, area, domain, name_contains, state)
3. If no results, try broader search

## Response Rules
- Short, concise replies in plain text only
- Use Friendly Names (e.g., "Living Room Light"), never entity IDs
- Use natural language for states ("on" → "turned on", "home" → "at home")

{response_mode}

## Index
{index}

Current area: {current_area}
Current time: {time}
Current date: {date}"""
