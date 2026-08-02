"""Read URL custom tool for ha-lmstudio-mcp."""
import aiohttp
import asyncio
import ipaddress
import logging
import socket
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urljoin

_LOGGER = logging.getLogger(__name__)

MAX_REDIRECTS = 5


async def _validate_url(url: str) -> Optional[str]:
    """SSRF guard: validate scheme and reject URLs resolving to non-public IPs.

    Returns an error message string, or None if the URL is safe to fetch.

    Note: we validate at DNS-resolve time only. A malicious server could
    DNS-rebind between this check and aiohttp's own resolution; pinning the
    connection to the resolved IP is out of scope for this minimal guard.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ['http', 'https']:
        return f"❌ Unsupported URL scheme: {parsed.scheme or '(none)'}"
    if not parsed.netloc or not parsed.hostname:
        return "❌ Invalid URL format"

    hostname = parsed.hostname
    try:
        # Literal IP hostname - check directly without DNS
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        # Resolve via the event loop's executor so we never block the loop
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except (socket.gaierror, OSError):
            return f"❌ Could not resolve hostname: {hostname}"
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]

    for addr in addresses:
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved or addr.is_unspecified):
            _LOGGER.debug(f"Blocked URL {url}: resolves to non-public address {addr}")
            return f"❌ URL not allowed: resolves to a non-public address ({addr})"

    return None

class ReadUrlTool:
    """Tool to read and extract content from URLs."""

    def __init__(self, hass):
        """Initialize Read URL tool."""
        self.hass = hass
        self.max_content_length = 50000  # Max characters to return

    async def initialize(self):
        """Initialize the tool."""
        pass  # No logging needed

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name == "read_url"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definition for Read URL."""
        return [{
            "name": "read_url",
            "description": "Read and extract text content from a webpage URL",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to read"
                    },
                    "summary": {
                        "type": "boolean",
                        "description": "Return a summary instead of full content (default false)",
                        "default": False
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }]

    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read and extract content from URL."""
        url = arguments.get("url")
        summary_only = arguments.get("summary", False)

        _LOGGER.debug(f"Reading URL: {url}")

        # Validate URL (scheme + SSRF guard)
        error = await _validate_url(url)
        if error:
            return {
                "content": [{
                    "type": "text",
                    "text": error
                }]
            }
        parsed = urlparse(url)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ha-lmstudio-mcp/1.0)"
            }

            async with aiohttp.ClientSession() as session:
                # Follow redirects manually (max 5 hops), re-validating each
                # Location target to prevent redirect-based SSRF
                for _ in range(MAX_REDIRECTS + 1):
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15),
                        allow_redirects=False
                    ) as response:
                        if response.status in (301, 302, 303, 307, 308):
                            location = response.headers.get('Location')
                            if not location:
                                return {
                                    "content": [{
                                        "type": "text",
                                        "text": "❌ Redirect response with no Location header"
                                    }]
                                }
                            url = urljoin(url, location)
                            _LOGGER.debug(f"Following redirect to: {url}")
                            error = await _validate_url(url)
                            if error:
                                return {
                                    "content": [{
                                        "type": "text",
                                        "text": error
                                    }]
                                }
                            parsed = urlparse(url)
                            continue

                        if response.status != 200:
                            return {
                                "content": [{
                                    "type": "text",
                                    "text": f"❌ HTTP {response.status}: Failed to fetch URL"
                                }]
                            }

                        # Check content type
                        content_type = response.headers.get('Content-Type', '')
                        if 'text/html' not in content_type and 'text/plain' not in content_type:
                            return {
                                "content": [{
                                    "type": "text",
                                    "text": f"❌ Unsupported content type: {content_type}"
                                }]
                            }

                        html = await response.text()

                        # Simple text extraction without BeautifulSoup dependency
                        text = await self._extract_text(html, content_type)

                        # Get title from HTML
                        title = parsed.netloc
                        if '<title>' in html and '</title>' in html:
                            title_start = html.index('<title>') + 7
                            title_end = html.index('</title>')
                            title = html[title_start:title_end].strip()

                        # Truncate if needed
                        truncated = False
                        if len(text) > self.max_content_length:
                            text = text[:self.max_content_length] + "..."
                            truncated = True

                        # Create summary if requested
                        if summary_only and len(text) > 1000:
                            # Take first 1000 chars as simple summary
                            text = text[:1000] + "..."

                        result_text = f"📖 **{title}**\n"
                        result_text += f"URL: {url}\n"
                        result_text += f"Length: {len(text)} chars"
                        if truncated:
                            result_text += " (truncated)"
                        result_text += f"\n\n{text}"

                        return {
                            "content": [{
                                "type": "text",
                                "text": result_text
                            }]
                        }

                return {
                    "content": [{
                        "type": "text",
                        "text": "❌ Too many redirects"
                    }]
                }

        except aiohttp.ClientTimeout:
            return {
                "content": [{
                    "type": "text",
                    "text": "❌ Timeout reading URL"
                }]
            }
        except Exception as e:
            _LOGGER.error(f"Read URL exception: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error reading URL: {str(e)}"
                }]
            }

    async def _extract_text(self, html: str, content_type: str) -> str:
        """Extract text from HTML without BeautifulSoup."""
        if 'text/plain' in content_type:
            return html

        # Basic HTML tag removal (simplified without BeautifulSoup)
        import re

        # Remove script and style blocks
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

        # Remove HTML tags
        html = re.sub(r'<[^>]+>', ' ', html)

        # Decode HTML entities
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")
        html = html.replace('&nbsp;', ' ')

        # Clean up whitespace
        lines = html.split('\n')
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        text = '\n'.join(lines)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()