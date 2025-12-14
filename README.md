# MCP Assist for Home Assistant

A Home Assistant conversation agent that uses MCP (Model Context Protocol) for efficient entity discovery, achieving **95% token reduction** compared to traditional methods. Works with LM Studio, Ollama, and soon OpenAI/Gemini.

## Key Features

- ✅ **95% Token Reduction**: Uses MCP tools for dynamic entity discovery instead of sending all entities
- ✅ **No Entity Dumps**: Never sends 12,000+ token entity lists to the LLM
- ✅ **Multi-Platform Support**: Works with LM Studio, Ollama (OpenAI & Gemini coming soon)
- ✅ **Multi-turn Conversations**: Maintains conversation context and history
- ✅ **Dynamic Discovery**: Finds entities by area, type, state, or name on-demand
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
   - `discover_entities` - Find entities by type, area, domain, or state
   - `get_entity_details` - Get current state and attributes
   - `perform_action` - Control devices
   - `list_areas` - List all areas in your home
   - `list_domains` - List all entity types
   - `set_conversation_state` - Smart follow-up handling
3. **LLM discovers on-demand** - Only fetches the entities it needs for each request
4. **Token usage drops** from 12,000+ to ~400 tokens per request

## Token Usage Comparison

| Method | Token Usage | Description |
|--------|-------------|-------------|
| **Traditional** | 12,000+ tokens | Sends all entity states |
| **MCP Assist** | ~400 tokens | Uses MCP tools for discovery |
| **Reduction** | **95%** | Massive efficiency gain |

## Requirements

- Home Assistant 2024.1+
- One of:
  - LM Studio v0.3.17+ (with MCP support)
  - Ollama
  - OpenAI (coming soon)
  - Google Gemini (coming soon)
- Python 3.11+

## Installation

### Option A: HACS (Recommended)
1. Add this repository to HACS as a custom repository
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
- Server Type: Choose LM Studio or Ollama

**Step 2 - Server Configuration:**
- Server URL: Where your LLM server is running
  - LM Studio: `http://localhost:1234` (default)
  - Ollama: `http://localhost:11434` (default)
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

### 3. Set as Voice Assistant

1. In Home Assistant, go to **Settings** → **Voice Assistants**
2. Set your preferred assistant to your MCP Assist profile name
3. Test with commands!

## Usage Examples

### Basic Commands
- "Turn on the kitchen lights"
- "Turn off all the lights in the bedroom"
- "What's the temperature in the living room?"

### Area-Based Control
- "Turn on the lights upstairs"
- "Close all the blinds on the main floor"
- "What sensors are active in the basement?"

### State Queries
- "Which lights are currently on?"
- "Show me all the open doors"
- "Are there any motion sensors triggered?"

### Multi-Turn Conversations
- **User**: "What lights are on?"
- **Assistant**: "The kitchen and living room lights are on."
- **User**: "Turn off the kitchen one"
- **Assistant**: "I've turned off the kitchen light."

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

### MCP Server Settings
- **MCP Server Port**: Default 8090 (change if port conflict)

### Web Search
- **Enable Web Search**: Turn on Brave Search & URL reading tools
- **Brave Search API Key**: Get one from https://brave.com/search/api/

## Multi-Profile Support

You can create multiple MCP Assist profiles for different use cases:

- **Kitchen Assistant**: Connected to Ollama, casual tone
- **Bedroom Assistant**: Connected to LM Studio, quiet and efficient
- **Main Assistant**: Cloud-based (OpenAI), most capable

Each profile runs independently and connects to the same shared MCP server.

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

## Development

This integration is open source. Contributions are welcome!

### MCP Tools Available

The integration exposes these MCP tools to your LLM:

- **discover_entities**: Find entities by name, area, domain, or state
- **get_entity_details**: Get detailed state and attributes for specific entities
- **perform_action**: Control devices (turn_on, turn_off, set_temperature, etc.)
- **list_areas**: List all areas with entity counts
- **list_domains**: List all entity types available
- **set_conversation_state**: Signal if expecting user response (for smart follow-ups)

### Optional Web Search Tools

When "Enable Web Search" is on:
- **brave_search**: Search the web for current information
- **read_url**: Extract content from web pages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mike-nott/mcp-assist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mike-nott/mcp-assist/discussions)
- **Home Assistant Community**: [Community Forum](https://community.home-assistant.io/)
