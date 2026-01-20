# MCP Assist for Home Assistant

A Home Assistant conversation agent that uses MCP (Model Context Protocol) for efficient entity discovery, achieving **95% token reduction** compared to traditional methods. Works with LM Studio, Ollama, OpenAI, Google Gemini, and Anthropic Claude.

## Key Features

- ✅ **95% Token Reduction**: Uses MCP tools for dynamic entity discovery instead of sending all entities
- ✅ **No Entity Dumps**: Never sends 12,000+ token entity lists to the LLM
- ✅ **Smart Entity Index**: Pre-generated system structure index (~400-800 tokens) for context-aware queries
- ✅ **Multi-Platform Support**: Works with LM Studio, Ollama, OpenAI, Google Gemini, and Anthropic Claude
- ✅ **Multi-turn Conversations**: Maintains conversation context and history
- ✅ **Dynamic Discovery**: Finds entities by area, type, device_class, state, or name on-demand
- ✅ **Web Search Tools**: Optional Brave Search integration for current information
- ✅ **Works with 1000+ Entities**: Efficient even with large Home Assistant installations
- ✅ **Multi-Profile Support**: Run multiple conversation agents with different models

## The Problem MCP Assist Solves

Traditional voice assistants send your **entire entity list** (lights, switches, sensors, etc.) to the LLM with every request. For a typical home with 200+ devices, this means:
- **12,000+ tokens** sent every time
- Expensive API costs (cloud LLMs)
- Slow response times
- Context window limitations
- Poor performance with large homes

## How MCP Assist Works

Instead of dumping all entities, MCP Assist:

1. **Starts an MCP Server** on Home Assistant that exposes entity discovery tools
2. **Your LLM connects** to the MCP server and gets access to these tools:
   - `get_index` - Get system structure index (areas, domains, device_classes, people, etc.)
   - `discover_entities` - Find entities by type, area, domain, device_class, or state
   - `get_entity_details` - Get current state and attributes
   - `perform_action` - Control devices
   - `run_script` - Execute scripts and return response data
   - `run_automation` - Trigger automations manually
   - `list_areas` - List all areas in your home
   - `list_domains` - List all entity types
   - `set_conversation_state` - Smart follow-up handling
3. **LLM uses the index for smart queries** - Understands what exists without full context dump
4. **LLM discovers on-demand** - Only fetches the entities it needs for each request
5. **Token usage drops** from 12,000+ to ~400 tokens per request

## Token Usage Comparison

| Method | Token Usage | Description |
|--------|-------------|-------------|
| **Traditional** | 12,000+ tokens | Sends all entity states |
| **MCP Assist** | ~400 tokens | Uses MCP tools for discovery |
| **Reduction** | **95%** | Massive efficiency gain |

## Smart Entity Index (v0.5.0+)

The Smart Entity Index provides a lightweight (~400-800 tokens) snapshot of your Home Assistant system structure, enabling context-aware queries without full entity dumps. The index includes areas, domains, device classes, people, calendars, zones, automations, and scripts. For entities without standardized device_class attributes (like custom integrations), LLM-powered gap-filling automatically infers semantic categories from naming patterns. This results in faster, more accurate queries that use ~95% fewer tokens compared to traditional entity dumps.

## Requirements

- Home Assistant 2024.1+
- One of:
  - **Local LLMs**: LM Studio v0.3.17+ or Ollama
  - **Cloud LLMs**: OpenAI, Google Gemini, or Anthropic Claude (API key required)
- Python 3.11+

## Installation

### Add to HACS

[![Open your Home Assistant instance and add this repository to HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mike-nott&repository=mcp-assist&category=integration)

### Option A: HACS (Recommended)
1. Click the badge above to add this repository to HACS, or manually add it as a custom repository
2. Install "MCP Assist" from HACS
3. Restart Home Assistant

### Option B: Manual Installation
1. Copy the `custom_components/mcp_assist` folder to your Home Assistant `custom_components` directory
2. Restart Home Assistant

## Configuration

### 1. Add the Integration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "MCP Assist" and select it

### 2. Setup Flow

**Step 1 - Profile & Server Type:**
- Profile Name: Give your assistant a name (e.g., "Living Room Assistant")
- Server Type: Choose your LLM provider
  - **LM Studio** - Local, free, runs on your machine
  - **Ollama** - Local, free, command-line based
  - **OpenAI** - Cloud, paid, GPT-5.2 series
  - **Google Gemini** - Cloud, paid/free tier, Gemini 3.0 series

**Step 2 - Server Configuration:**

*For Local Servers (LM Studio / Ollama):*
- Server URL: Where your LLM server is running
  - LM Studio: `http://localhost:1234` (default)
  - Ollama: `http://localhost:11434` (default)
- MCP Server Port: Port for the MCP server (default: 8090)

*For Cloud Providers (OpenAI / Gemini / Anthropic):*
- API Key: Your provider API key (see below for setup)
- MCP Server Port: Port for the MCP server (default: 8090)

**Step 3 - Model & Prompts:**
- Model Name: Select from auto-loaded models or enter manually
- System Prompt: Customize the assistant's personality
- Technical Instructions: Advanced prompt for tool usage (pre-configured)

**Step 4 - Advanced Settings:**
- Temperature: Response randomness (0.0-1.0)
- Max Response Tokens: Maximum length of responses
- Response Mode: Smart / Always / None (follow-up question behavior)
- Control Home Assistant: Enable/disable device control
- Max Tool Iterations: How many tool calls allowed per request
- Enable Web Search: Turn on Brave Search integration
- Brave Search API Key: Your API key (if using web search)
- Debug Mode: Extra logging for troubleshooting
- **Ollama Keep Alive** (Ollama only): Control how long models stay loaded in memory
  - `-1` = Keep loaded indefinitely
  - `0` = Unload immediately after response
  - `"5m"` = Keep for 5 minutes (default)
  - Duration strings like `"24h"`, `"168h"` also supported
- **Ollama Context Window** (Ollama only): Custom context window size (0 = use model default)

### 3. Set as Voice Assistant

1. In Home Assistant, go to **Settings** → **Voice Assistants**
2. Set your preferred assistant to your MCP Assist profile name
3. Test with commands!

## Usage Examples

### Basic Commands
- "Turn on the kitchen lights"
- "Turn off all the lights in the bedroom"
- "What's the temperature in the living room?"

### Multi-Turn Conversations
- **User**: "What lights are on?"
- **Assistant**: "The kitchen and living room lights are on."
- **User**: "Turn off the kitchen one"
- **Assistant**: "I've turned off the kitchen light."

### Complex Query Example

**User**: "Do we have a leak?"

**Behind the scenes:**
```
1. LLM calls get_index → Sees moisture sensors and water flow monitors exist in system
2. LLM calls discover_entities(device_class="moisture")
   → Returns: binary_sensor.bathroom_leak, binary_sensor.kitchen_sink_leak, binary_sensor.laundry_leak
3. LLM calls discover_entities(name_contains="water flow")
   → Returns: sensor.water_flow_rate
4. LLM calls get_entity_details for each sensor
   → bathroom leak: "off", kitchen leak: "off", laundry leak: "on", water flow: "2.5 gpm"
5. LLM synthesizes response
```

**Assistant**: "Yes, the laundry room leak sensor is detecting water and water is flowing at 2.5 gallons per minute. The bathroom and kitchen sensors are dry."

**Follow-up User**: "Turn off the water main"

**Behind the scenes:**
```
1. LLM calls discover_entities(name_contains="water main")
   → Returns: switch.water_main_shutoff
2. LLM calls perform_action(entity_id="switch.water_main_shutoff", action="turn_off")
   → Success
```

**Assistant**: "I've shut off the main water valve."

### Web Search (if enabled)
- "What's the weather forecast for tomorrow?"
- "Search for the latest Home Assistant updates"
- "What time does the store close?"

## Configuration Options

### Profile Settings
- **Profile Name**: Unique name for this assistant
- **Server Type**: LM Studio, Ollama (more coming)
- **Server URL**: Where your LLM is running
- **Model Name**: Which model to use

### Prompts
- **System Prompt**: Sets the assistant's personality and behavior
- **Technical Instructions**: Low-level instructions for tool usage (usually leave as default)

### Advanced Settings
- **Temperature**: 0.0 (deterministic) to 1.0 (creative)
- **Max Response Tokens**: Limit response length (default: 500)
- **Max History Messages**: How many conversation turns to remember (default: 10)
- **Max Tool Iterations**: Prevent infinite loops (default: 5)
- **Response Mode**:
  - **Smart** (default): LLM decides if follow-up needed
  - **Always**: Always wait for follow-up
  - **None**: Never wait for follow-up
- **Enable Smart Entity Index**: Context-aware entity discovery with automatic gap-filling for uncommon devices (default: enabled)

### MCP Server Settings
- **MCP Server Port**: Default 8090 (change if port conflict)
- **Additional Allowed IPs/Ranges**: Whitelist Docker containers (e.g., `172.30.0.0/16`) or specific IPs for external MCP clients like Claude Code add-on

### Web Search
- **Enable Web Search**: Turn on Brave Search & URL reading tools
- **Brave Search API Key**: Get one from https://brave.com/search/api/

## Model Compatibility Guide

Not all LLM models support tool calling (function calling) equally well. **Model performance varies significantly between inference engines** (LM Studio vs Ollama), so testing is recommended.

### Understanding Tool Calling Requirements

Tool calling (function calling) requires the model to:
1. Understand the user's request
2. Decide which tool to call
3. Format the tool arguments correctly as JSON
4. Interpret the tool results
5. Generate a natural response

**Factors affecting tool calling success**:
- **Model size**: Larger models (8B+) generally handle tool calling better
- **Model architecture**: Vision-Language (VL) models behave differently than standard models
- **Inference engine**: LM Studio and Ollama optimize models differently
- **Quantization level**: Q4 vs Q8 can affect instruction following

### Instruct vs Thinking/Reasoning Models

**Instruct Models** (e.g., `qwen3-8b-instruct`):
- Fast response times
- Best for simple, single-action requests ("turn on the kitchen lights")
- May struggle with complex queries requiring multiple tool calls
- Good for basic voice commands

**Thinking/Reasoning Models** (e.g., `qwen3-8b-thinking`):
- Slower response times (more deliberate reasoning)
- **Much better at complex requests** requiring multiple tool calls and context
- Handles multi-step queries reliably ("check all rooms for open windows, then turn off lights in those rooms")
- **Recommended for Home Assistant** where queries often involve discovery + action combinations

Choose the model type that best fits your use case. Thinking/reasoning models offer better reliability with complex multi-tool queries, while instruct models provide faster responses for simple commands.

### Recommended Models

**Consistently Reliable**:
- ✅ **Qwen3 VL 32B Instruct** - Excellent tool calling
- ✅ **Qwen3 30B A3B Instruct** - Very good tool calling
- ✅ **Qwen3 8B Instruct** - Good balance, works reliably
- ✅ **Anthropic Opus 4.5** - The absolute best at tool calling (cloud)
- ✅ **OpenAI GPT-5.2 series** - Very good (cloud)
- ✅ **Google Gemini 3 Pro** - Very Good (cloud)

### Testing Your Model

When tool calling **doesn't work**, you'll see:
- Model claims "I turned on the lights" but nothing happens
- No `perform_action` tool calls in the logs
- Actions don't execute, only narration

When tool calling **works correctly**, you'll see in logs:
- `discover_entities` called to find devices
- `perform_action` called to control them
- "✅ Successfully executed" messages
- Devices actually change state

### General Guidelines

**Start with larger models** (30B) if your hardware supports it - they work consistently across platforms.

**If using smaller models** (4B-8B), test thoroughly:
- Try a simple command like "turn on the kitchen lights"
- Check logs to verify tools are being called
- Confirm the device actually changes state
- If it doesn't work, try the same model on a different platform (LM Studio vs Ollama)

**Vision-Language (VL) models** are optimized for multimodal tasks and may have different tool calling behavior than standard models.

### Dynamic Model Switching

One of MCP Assist's features is **dynamic model switching** - you can change models in the configuration UI and it takes effect immediately without restarting Home Assistant. This makes it easy to:
- Test different models
- Switch between fast (Q4) and quality (Q8) quantizations
- Try new models as they're released

## Troubleshooting

### Integration Won't Start
- Check that the MCP port isn't already in use
- Verify Home Assistant has permission to bind to the port
- Check the Home Assistant logs for specific error messages

### LM Studio Can't Connect to MCP
- Ensure the MCP server is running (check integration status)
- Verify the MCP configuration in LM Studio is correct
- Check that the URL in LM Studio matches your MCP port (default: 8090)
- Restart LM Studio after changing MCP configuration

### Ollama Connection Issues
- Verify Ollama is running: `ollama list`
- Check the URL matches where Ollama is running (default: `http://localhost:11434`)
- Ensure the model is loaded in Ollama

### Cloud Provider Connection Issues
- Verify your API key is valid with your provider
- Check you have sufficient credits/quota remaining
- Check for rate limit errors in Home Assistant logs
- Try regenerating your API key if authentication fails
- Verify your internet connection is working
- Check Home Assistant logs for specific error codes

### No Response from Assistant
- Verify your LLM has a model loaded
- Check that the model name in the integration matches exactly
- Ensure your entities are exposed to the conversation assistant
- Check Home Assistant logs for API errors
- Enable Debug Mode for more detailed logging

### Poor Response Quality
- Try a different/larger model
- Adjust the temperature setting (lower = more focused)
- Ensure the model supports tool calling/function calling
- Check that Technical Instructions are not modified

### Tools Not Working
- Verify "Control Home Assistant" is enabled
- Check that entities are exposed (Settings → Voice Assistants → Expose)
- Look for MCP server errors in logs
- Ensure Max Tool Iterations isn't set too low

## Entity Exposure

The integration only discovers entities that are exposed to the "conversation" assistant. To expose entities:

1. Go to **Settings** → **Voice Assistants** → **Expose**
2. Select entities you want the assistant to control
3. The integration will automatically discover these when needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mike-nott/mcp-assist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mike-nott/mcp-assist/discussions)
- **Home Assistant Community**: [Community Forum](https://community.home-assistant.io/)
