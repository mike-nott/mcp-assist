# Changelog

All notable changes to MCP Assist will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-12-14

### Fixed
- **JSON Serialization Error**: Removed malformed card data structure that prevented voice responses from being delivered to users
- **OpenAI/Gemini Configuration Bugs**: Fixed server type preservation and data fallback patterns in config flow
  - Server type now correctly preserved when editing profiles
  - Data fallback pattern fixed: `options → data → defaults`
  - Profile type no longer switches unexpectedly when editing
- **Configuration Caching**: Converted all configuration values to dynamic `@property` methods
  - Model changes now take effect immediately without restarting Home Assistant
  - All parameters (temperature, max_tokens, etc.) apply instantly
  - Significantly improved UX for testing and configuration changes

### Changed
- **Model Support**: Updated to latest GPT-5.2 and Gemini 3.0 model names
  - GPT-5 models now use `max_completion_tokens` instead of `max_tokens`
  - Temperature parameter excluded for GPT-5+ and o1 models (they only support default of 1.0)

### Added
- **Model Compatibility Guide**: Comprehensive documentation on model performance
  - Explains inference engine differences (LM Studio vs Ollama)
  - Documents platform-specific model behavior (e.g., Qwen3 4B works on Ollama but not LM Studio)
  - Provides testing guidance to verify tool calling works correctly
  - Includes recommendations for different use cases (local vs cloud, hardware constraints)
- **Enhanced Debug Logging**: Added comprehensive logging for troubleshooting
  - System prompt length and preview
  - Message counts and content lengths
  - Tool response details with line-by-line output
  - Exact payload details (model, temperature, max_tokens)

### Improved
- **Code Refactoring**: Cleaned up obsolete code and simplified CustomToolsLoader
- **Provider-Agnostic Naming**: Further refinements to multi-provider architecture

## [0.3.0] - 2025-12-14

### Added
- **OpenAI Support**: Full integration with OpenAI's GPT-5.2 series
  - gpt-5.2-2025-12-11 (standard flagship model)
  - gpt-5.2-instant (faster for chat/info seeking)
  - gpt-5.2-thinking (best for coding/planning)
  - gpt-5.2-pro-2025-12-11 (pro - highest accuracy)
  - Legacy support for gpt-4o and gpt-4-turbo
- **Google Gemini Support**: Full integration with Gemini 3.0 series
  - gemini-3-pro-preview-11-2025 (latest reasoning-first)
  - gemini-3-pro-preview-11-2025-thinking (enhanced reasoning)
  - gemini-3-pro-preview (global endpoints)
  - Legacy support for gemini-2.0-flash-exp and gemini-1.5-pro
- **Dynamic Configuration Flow**: Different setup steps based on server type
  - Local servers (LM Studio/Ollama): Server URL configuration
  - Cloud providers (OpenAI/Gemini): API key configuration
- **Provider-Specific Authentication**: Automatic header management
  - OpenAI: `Authorization: Bearer {api_key}`
  - Gemini: `x-goog-api-key: {api_key}`
  - Local servers: No authentication required
- **Comprehensive Documentation**:
  - Cloud provider setup guides with step-by-step instructions
  - API key acquisition and security best practices
  - Cost breakdowns and billing information for cloud providers
  - Cloud vs Local LLM comparison table
  - Provider-specific troubleshooting sections

### Changed
- **Renamed Agent Class**: `LMStudioMCPAgent` → `MCPAssistAgent` for multi-provider clarity
- **Renamed Methods**: All internal methods renamed for provider-agnostic naming
  - `_call_lmstudio()` → `_call_llm()`
  - `_call_lmstudio_http()` → `_call_llm_http()`
  - `_call_lmstudio_streaming()` → `_call_llm_streaming()`
- **Updated Logging**: All log messages now use dynamic `server_type` instead of hardcoded "LM Studio"
- **Enhanced Error Messages**: Provider-specific error messages for better debugging
- **Base URL Handling**: Unified base URL logic for all providers
  - Cloud providers use fixed base URLs
  - Local servers use user-provided URLs

### Fixed
- **Security**: Removed hardcoded Brave Search API key (GitGuardian alert)
  - Changed to require user-provided API key via config
  - All API keys now properly secured in Home Assistant config storage

### Deprecated
- None

### Removed
- Hardcoded API credentials from codebase
- "Prefer Streaming" option (now always enabled with auto-fallback)

### Security
- API keys stored securely in Home Assistant's encrypted config storage
- Added security warnings and best practices to documentation
- No credentials hardcoded in source code
- Git history cleaned of exposed API keys

## [0.2.0] - 2025-12-XX

### Added
- Initial public release
- MCP (Model Context Protocol) integration for efficient entity discovery
- Support for LM Studio and Ollama
- 95% token reduction compared to traditional methods
- Multi-turn conversation support
- Dynamic entity discovery tools
- Optional Brave Search integration
- Multi-profile support
- Streaming support with automatic HTTP fallback

### Features
- MCP server for Home Assistant entity discovery
- Tools: discover_entities, perform_action, get_entity_details, list_areas, list_domains
- Smart follow-up handling
- Configurable temperature, tokens, and iterations
- Debug mode for troubleshooting

---

## Version Comparison

| Version | Local Support | Cloud Support | Models | Key Features |
|---------|--------------|---------------|--------|--------------|
| **0.3.1** | LM Studio, Ollama | OpenAI, Gemini | GPT-5.2, Gemini 3.0 | Bug fixes, dynamic config |
| **0.3.0** | LM Studio, Ollama | OpenAI, Gemini | GPT-5.2, Gemini 3.0 | Multi-provider, cloud auth |
| **0.2.0** | LM Studio, Ollama | None | Any local model | Initial MCP release |

---

## Upgrade Notes

### Upgrading to 0.3.1

**Backward Compatibility**: ✅ Fully compatible
- All existing configurations continue working
- No breaking changes
- Drop-in replacement for 0.3.0

**Key Improvements**:
- **Voice responses now work correctly** - JSON serialization bug fixed
- **Model changes take effect immediately** - No need to restart Home Assistant
- **Configuration editing is more reliable** - Server type preservation fixed

**Recommended Actions**:
- After upgrading, you can now change models in the config UI without restarting HA
- If you experienced "no response" issues in 0.3.0, this release fixes it
- Review the Model Compatibility Guide in the README for optimal model selection

**Breaking Changes**: None

### Upgrading to 0.3.0

**Backward Compatibility**: ✅ Fully compatible
- Existing LM Studio and Ollama configurations will continue working
- No configuration changes required for local servers
- Agent class renamed internally but Home Assistant integration unchanged

**New Users**:
- Can now choose between local (LM Studio, Ollama) and cloud (OpenAI, Gemini) providers
- Cloud providers require API keys (costs apply)
- See README for detailed setup instructions

**Breaking Changes**: None

---

## Links
- [GitHub Repository](https://github.com/mike-nott/mcp-assist)
- [Issues](https://github.com/mike-nott/mcp-assist/issues)
- [Documentation](https://github.com/mike-nott/mcp-assist/blob/main/README.md)
