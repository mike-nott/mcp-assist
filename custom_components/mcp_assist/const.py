"""Constants for the MCP Assist integration."""

DOMAIN = "mcp_assist"

# Server type options
SERVER_TYPE_LMSTUDIO = "lmstudio"
SERVER_TYPE_OLLAMA = "ollama"
SERVER_TYPE_OPENAI = "openai"
SERVER_TYPE_GEMINI = "gemini"
SERVER_TYPE_ANTHROPIC = "anthropic"

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
CONF_ENABLE_GAP_FILLING = "enable_gap_filling"
CONF_OLLAMA_KEEP_ALIVE = "ollama_keep_alive"
CONF_OLLAMA_NUM_CTX = "ollama_num_ctx"

# Default values
DEFAULT_SERVER_TYPE = "lmstudio"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MCP_PORT = 8090
DEFAULT_API_KEY = ""

# Cloud provider base URLs
OPENAI_BASE_URL = "https://api.openai.com"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# No hardcoded model lists - models are fetched dynamically from provider APIs
DEFAULT_MODEL_NAME = "model"
DEFAULT_SYSTEM_PROMPT = "You are a helpful Home Assistant voice assistant. Respond naturally and conversationally to user requests."
DEFAULT_CONTROL_HA = True
DEFAULT_RESPONSE_MODE = "default"
DEFAULT_FOLLOW_UP_MODE = "default"  # Keep for backward compatibility
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_HISTORY = 10
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_DEBUG_MODE = False
DEFAULT_ENABLE_CUSTOM_TOOLS = False
DEFAULT_BRAVE_API_KEY = ""
DEFAULT_ALLOWED_IPS = ""
DEFAULT_ENABLE_GAP_FILLING = True
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"  # 5 minutes
DEFAULT_OLLAMA_NUM_CTX = 0  # 0 = use model default

# MCP Server settings
MCP_SERVER_NAME = "ha-entity-discovery"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Entity discovery limits
MAX_ENTITIES_PER_DISCOVERY = 50
MAX_DISCOVERY_RESULTS = 100

DEFAULT_TECHNICAL_PROMPT = """You are controlling a Home Assistant smart home system. You have access to sensors, lights, switches, and other devices throughout the home.

## CRITICAL RULES
**Never guess entity IDs.** For ANY device-related request, you MUST:
- FIRST call discover_entities to find the actual entities
- THEN call perform_action or get_entity_details using discovered IDs
- This applies EVERY TIME - even for follow-up questions about different entities

## Available Tools
- **discover_entities**: find devices by name/area/domain/device_class/state (ALWAYS use first)
- **perform_action**: control devices using discovered entity IDs
- **get_entity_details**: check states using discovered entity IDs
- **list_areas/list_domains**: list available areas and device types
- **run_script**: execute scripts that return data (e.g., camera analysis, calculations)
- **run_automation**: trigger automations manually
- **set_conversation_state**: indicate if expecting user response
- **brave_search**: search the web for current information
- **read_url**: read and extract content from web pages

## Scripts (use run_script tool)
Scripts can perform complex operations and return data. Check the index for available scripts with their parameters.

Example - Camera analysis:
  run_script(script_id="llm_camera_analysis", variables={{"camera_entities": "camera.living_room", "prompt": "Is anyone there?"}})

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
- Short, concise replies in plain text only (no *, **, markup, or URLs)
- Use Friendly Names (e.g., "Living Room Light"), never entity IDs
- Use natural language for states ("on" → "turned on", "home" → "at home")

## Follow-up Questions
- After single device actions: "Anything else?"
- When reporting adjustable status: "Want me to change it?"
- For partial completions: "Should I handle the others?"

## Ending Conversations
When user indicates they're done - do not respond further.

## Index
{index}

Current area: {current_area}
Current time: {time}
Current date: {date}"""