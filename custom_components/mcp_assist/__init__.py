"""The MCP Assist integration."""

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.components import conversation
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, CONF_MCP_PORT, DEFAULT_MCP_PORT
from .mcp_server import MCPServer
from .index_manager import IndexManager

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up MCP Assist from a config entry."""
    profile_name = entry.data.get("profile_name", "Default")
    _LOGGER.info("Setting up MCP Assist integration - Profile: %s", profile_name)

    hass.data.setdefault(DOMAIN, {})

    try:
        # Handle shared MCP server and index manager
        if "shared_mcp_server" not in hass.data[DOMAIN]:
            # First entry - create shared MCP server and index manager
            mcp_port = entry.data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            _LOGGER.info("Creating shared MCP server on port %d", mcp_port)

            # Create and start index manager
            index_manager = IndexManager(hass)
            await index_manager.start()
            hass.data[DOMAIN]["index_manager"] = index_manager

            # Create and start MCP server
            mcp_server = MCPServer(hass, mcp_port, entry)
            await mcp_server.start()

            hass.data[DOMAIN]["shared_mcp_server"] = mcp_server
            hass.data[DOMAIN]["mcp_refcount"] = 0
            hass.data[DOMAIN]["mcp_port"] = mcp_port

            _LOGGER.info("✅ Shared MCP server and index manager created successfully")
        else:
            # Reuse existing MCP server
            mcp_port = hass.data[DOMAIN]["mcp_port"]
            _LOGGER.info("Reusing existing shared MCP server on port %d", mcp_port)

            # Warn if user tried to configure different port
            requested_port = entry.data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            if requested_port != mcp_port:
                _LOGGER.warning(
                    "Profile '%s' requested port %d, but using shared port %d. "
                    "All profiles share the same MCP server.",
                    profile_name, requested_port, mcp_port
                )

        # Increment reference count
        hass.data[DOMAIN]["mcp_refcount"] += 1
        _LOGGER.debug("MCP server refcount: %d", hass.data[DOMAIN]["mcp_refcount"])

        # Store metadata (per entry)
        hass.data[DOMAIN][entry.entry_id] = {
            "profile_name": profile_name,
            "mcp_port": mcp_port
        }

        # Forward to platform to create conversation entity
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        _LOGGER.info("✅ Profile '%s' setup complete, Entry ID: %s", profile_name, entry.entry_id)

        return True

    except Exception as err:
        _LOGGER.error("Failed to setup MCP Assist profile '%s': %s", profile_name, err)
        raise ConfigEntryNotReady(f"Setup failed: {err}") from err


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    profile_name = entry.data.get("profile_name", "Default")
    _LOGGER.info("Unloading MCP Assist profile '%s'", profile_name)

    # Unload platforms (this will unregister conversation entity)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if not unload_ok:
        return False

    # Remove entry data
    hass.data[DOMAIN].pop(entry.entry_id, None)

    # Decrement MCP server reference count
    if "mcp_refcount" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["mcp_refcount"] -= 1
        refcount = hass.data[DOMAIN]["mcp_refcount"]
        _LOGGER.debug("MCP server refcount after unload: %d", refcount)

        # Only stop MCP server and index manager when last profile is removed
        if refcount <= 0:
            _LOGGER.info("Last profile removed - stopping shared MCP server and index manager")
            mcp_server = hass.data[DOMAIN].pop("shared_mcp_server", None)
            if mcp_server:
                await mcp_server.stop()
            hass.data[DOMAIN].pop("index_manager", None)
            hass.data[DOMAIN].pop("mcp_port", None)
            hass.data[DOMAIN].pop("mcp_refcount", None)
        else:
            _LOGGER.info("Shared MCP server still in use by %d profile(s)", refcount)

    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)