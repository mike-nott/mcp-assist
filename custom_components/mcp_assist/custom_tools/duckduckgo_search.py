"""DuckDuckGo Search custom tool for MCP Assist."""
import logging
from typing import Dict, Any, List
from duckduckgo_search import DDGS

_LOGGER = logging.getLogger(__name__)

class DuckDuckGoSearchTool:
    """DuckDuckGo Search tool (no API key required)."""

    def __init__(self, hass):
        """Initialize DuckDuckGo Search tool."""
        self.hass = hass

    async def initialize(self):
        """Initialize the tool."""
        pass  # No initialization needed

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name == "search"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definition for DuckDuckGo Search."""
        return [{
            "name": "search",
            "description": "Search the web for current information using DuckDuckGo",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "count": {
                        "type": "number",
                        "description": "Number of results to return (default 5, max 20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }]

    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DuckDuckGo Search."""
        query = arguments.get("query")
        count = min(arguments.get("count", 5), 20)  # Enforce max limit

        _LOGGER.debug(f"DuckDuckGo Search: '{query}' (count: {count})")

        try:
            # Run synchronous DDGS().text() in thread pool
            results = await self.hass.async_add_executor_job(
                self._search_sync, query, count
            )

            # Format results for LLM
            text_results = f"🔍 Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                text_results += f"{i}. **{result['title']}**\n"
                text_results += f"   {result['href']}\n"
                text_results += f"   {result['body']}\n\n"

            return {
                "content": [{
                    "type": "text",
                    "text": text_results
                }]
            }

        except Exception as e:
            _LOGGER.error(f"DuckDuckGo Search exception: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Search error: {str(e)}"
                }]
            }

    def _search_sync(self, query: str, count: int) -> List[Dict[str, str]]:
        """Synchronous search wrapper for thread pool execution."""
        try:
            raw_results = DDGS().text(
                keywords=query,
                max_results=count,
                region="us-en",
                safesearch="moderate",
                backend="auto"
            )

            # Normalize format to match Brave Search return structure
            return [
                {
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),  # DDG uses "href", we keep it
                    "body": r.get("body", "")   # DDG uses "body", we keep it
                }
                for r in raw_results
            ]
        except Exception as e:
            _LOGGER.error(f"DDG sync search failed: {e}")
            raise
