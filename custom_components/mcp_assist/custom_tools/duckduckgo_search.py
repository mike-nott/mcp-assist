"""DuckDuckGo/DDGS Search custom tool for MCP Assist."""

import logging
from typing import Dict, Any, List

try:  # Prefer the renamed package when available.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - compatibility for older installs
    from duckduckgo_search import DDGS

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
            "description": (
                "Search the web for up-to-date information, including live news and current events, "
                "using DuckDuckGo/DDGS."
            ),
            "keywords": ["news", "latest", "current", "today", "right now", "web"],
            "example_queries": [
                "What's happening right now in Iran?",
                "Latest Mariners news today",
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
                        "description": "Search mode: auto picks news mode for current-events style queries.",
                        "enum": ["auto", "web", "news"],
                        "default": "auto",
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
        mode = self._normalize_mode(arguments.get("mode"), query)

        _LOGGER.debug("DuckDuckGo Search: '%s' (count: %s, mode: %s)", query, count, mode)

        try:
            # Run synchronous DDGS search in thread pool
            results = await self.hass.async_add_executor_job(
                self._search_sync, query, count, mode
            )

            # Format results for LLM
            heading = "News results" if mode == "news" else "Search results"
            text_results = f"🔍 {heading} for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                text_results += f"{i}. **{result['title']}**\n"
                if result.get("source") or result.get("date"):
                    details = " | ".join(
                        part
                        for part in [result.get("source", ""), result.get("date", "")]
                        if part
                    )
                    if details:
                        text_results += f"   {details}\n"
                text_results += f"   {result['url']}\n"
                text_results += f"   {result['snippet']}\n\n"

            return {
                "content": [{
                    "type": "text",
                    "text": text_results
                }],
                "structuredContent": {
                    "query": query,
                    "mode": mode,
                    "count": len(results),
                    "results": results,
                },
            }

        except Exception as e:
            _LOGGER.error(f"DuckDuckGo Search exception: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Search error: {str(e)}"
                }]
            }

    def _search_sync(self, query: str, count: int, mode: str = "auto") -> List[Dict[str, str]]:
        """Synchronous search wrapper for thread pool execution."""
        try:
            normalized_mode = self._normalize_mode(mode, query)
            client = DDGS()
            if normalized_mode == "news":
                try:
                    raw_results = client.news(
                        keywords=query,
                        max_results=count,
                    )
                except Exception as err:
                    _LOGGER.warning(
                        "DDGS news search failed for %r, falling back to web search: %s",
                        query,
                        err,
                    )
                    raw_results = client.text(
                        keywords=query,
                        max_results=count,
                        region="us-en",
                        safesearch="moderate",
                        backend="auto",
                    )
            else:
                raw_results = client.text(
                    keywords=query,
                    max_results=count,
                    region="us-en",
                    safesearch="moderate",
                    backend="auto",
                )

            return [self._normalize_result(r) for r in raw_results]
        except Exception as e:
            _LOGGER.error(f"DDG sync search failed: {e}")
            raise

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

    @staticmethod
    def _normalize_result(raw_result: Dict[str, Any]) -> Dict[str, str]:
        """Normalize DDGS web/news results into one structure."""
        return {
            "title": str(raw_result.get("title", "") or ""),
            "url": str(raw_result.get("url") or raw_result.get("href") or ""),
            "snippet": str(raw_result.get("body") or raw_result.get("snippet") or ""),
            "source": str(raw_result.get("source", "") or ""),
            "date": str(raw_result.get("date", "") or ""),
        }
