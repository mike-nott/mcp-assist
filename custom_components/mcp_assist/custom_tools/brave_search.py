"""Brave Search custom tool for MCP Assist."""
import aiohttp
import asyncio
import logging
from typing import Dict, Any, List

_LOGGER = logging.getLogger(__name__)

NEWS_QUERY_HINTS = (
    "news",
    "headline",
    "headlines",
    "latest",
    "today",
    "current",
    "right now",
    "breaking",
    "what's happening",
    "what is happening",
)

class BraveSearchTool:
    """Brave Search API tool."""

    def __init__(self, hass, api_key=None):
        """Initialize Brave Search tool."""
        self.hass = hass
        # Use provided API key - no fallback for security
        self.api_key = api_key or ""
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    async def initialize(self):
        """Initialize the tool."""
        pass  # No logging needed

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name == "search"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definition for Brave Search."""
        return [{
            "name": "search",
            "description": (
                "Search the web for up-to-date information, including current events and live news, "
                "using Brave Search."
            ),
            "llmDescription": "Search the web or news for current information.",
            "keywords": ["news", "latest", "current", "today", "right now", "web"],
            "example_queries": [
                "What's happening right now in Iran?",
                "Latest Seahawks news today",
            ],
            "preferred_when": (
                "Use for web, internet, current-events, breaking-news, and latest-information questions."
            ),
            "returns": (
                "Search results with titles, URLs, snippets, and structured result metadata."
            ),
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
                    },
                    "mode": {
                        "type": "string",
                        "description": "Search mode: auto treats current-events style queries as news-oriented.",
                        "enum": ["auto", "web", "news"],
                        "default": "auto",
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }]

    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Brave Search."""
        query = arguments.get("query")
        count = min(arguments.get("count", 5), 20)  # Enforce max limit
        mode = self._normalize_mode(arguments.get("mode"), query)
        query_to_send = self._query_for_mode(query, mode)

        _LOGGER.debug("Brave Search: '%s' (count: %s, mode: %s)", query_to_send, count, mode)

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }

        # Fix: Convert all values to strings for URL parameters
        params = {
            "q": query_to_send,
            "count": str(count),  # Convert to string
            "text_decorations": "false",  # String not boolean
            "search_lang": "en",
            "country": "us"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error(f"Brave Search error {response.status}: {error}")
                        return {
                            "content": [{
                                "type": "text",
                                "text": f"❌ Search failed (HTTP {response.status}): {error[:200]}"
                            }]
                        }

                    data = await response.json()

                    # Format results for LLM
                    results = []
                    for item in data.get("web", {}).get("results", [])[:count]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", "")
                        })

                    # Format as text for the LLM
                    heading = "News results" if mode == "news" else "Search results"
                    text_results = f"🔍 {heading} for '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        text_results += f"{i}. **{result['title']}**\n"
                        text_results += f"   {result['url']}\n"
                        text_results += f"   {result['description']}\n\n"

                    return {
                        "content": [{
                            "type": "text",
                            "text": text_results
                        }],
                        "structuredContent": {
                            "query": query,
                            "provider_query": query_to_send,
                            "mode": mode,
                            "count": len(results),
                            "results": results,
                        },
                    }

        except asyncio.TimeoutError:  # Fix: Correct exception
            _LOGGER.error("Brave Search timeout")
            return {
                "content": [{
                    "type": "text",
                    "text": "❌ Search timeout - please try again"
                }]
            }
        except Exception as e:
            _LOGGER.error(f"Brave Search exception: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Search error: {str(e)}"
                }]
            }

    def _normalize_mode(self, raw_mode: Any, query: Any) -> str:
        """Normalize requested search mode."""
        normalized = str(raw_mode or "auto").strip().lower()
        if normalized not in {"auto", "web", "news"}:
            normalized = "auto"
        if normalized == "auto":
            lowered_query = str(query or "").strip().lower()
            if any(hint in lowered_query for hint in NEWS_QUERY_HINTS):
                return "news"
            return "web"
        return normalized

    def _query_for_mode(self, query: Any, mode: str) -> str:
        """Slightly bias news queries toward fresher web results."""
        normalized_query = str(query or "").strip()
        if mode != "news":
            return normalized_query
        lowered_query = normalized_query.lower()
        if any(hint in lowered_query for hint in NEWS_QUERY_HINTS):
            return normalized_query
        return f"{normalized_query} latest news"
