"""The MCP Assist integration."""

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.components import conversation
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    DOMAIN,
    SYSTEM_ENTRY_UNIQUE_ID,
    CONF_MCP_PORT,
    DEFAULT_MCP_PORT,
    CONF_TECHNICAL_PROMPT,
    CONF_PROFILE_NAME,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_BRAVE_API_KEY,
    CONF_ALLOWED_IPS,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_GAP_FILLING,
    DEFAULT_ENABLE_CUSTOM_TOOLS,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_ENABLE_GAP_FILLING,
)
from .mcp_server import MCPServer
from .index_manager import IndexManager

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION]


async def _migrate_brave_search_tool_name(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """One-time migration: Replace 'brave_search' with 'search' in Technical Instructions."""
    options = entry.options
    data = entry.data

    technical_prompt = options.get(CONF_TECHNICAL_PROMPT, data.get(CONF_TECHNICAL_PROMPT, ""))

    if "brave_search" in technical_prompt:
        updated_prompt = technical_prompt.replace("brave_search", "search")
        new_options = {**options, CONF_TECHNICAL_PROMPT: updated_prompt}
        hass.config_entries.async_update_entry(entry, options=new_options)

        _LOGGER.info(
            "Profile '%s': Migrated Technical Instructions from 'brave_search' to 'search'",
            entry.data.get(CONF_PROFILE_NAME, "Default")
        )


def get_system_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the system config entry that stores shared MCP settings."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
            return entry
    return None


async def ensure_system_entry(hass: HomeAssistant) -> ConfigEntry:
    """Ensure system entry exists, create from first profile if not (self-healing)."""
    system_entry = get_system_entry(hass)

    if system_entry is None:
        _LOGGER.info("System entry not found, creating from first profile's settings (self-healing)")

        # Find first profile entry to copy shared settings from
        first_profile = None
        for entry in hass.config_entries.async_entries(DOMAIN):
            # Skip if this IS the system entry (shouldn't happen but be safe)
            if entry.unique_id != SYSTEM_ENTRY_UNIQUE_ID:
                first_profile = entry
                break

        # Extract shared settings from first profile (with fallback to defaults)
        if first_profile:
            _LOGGER.info("Copying shared settings from profile: %s",
                        first_profile.data.get(CONF_PROFILE_NAME, "Unknown"))

            # Get search provider with backward compatibility for enable_custom_tools
            search_provider = first_profile.options.get(
                CONF_SEARCH_PROVIDER,
                first_profile.data.get(CONF_SEARCH_PROVIDER)
            )
            # Backward compat: if search_provider not set but enable_custom_tools was True, use "brave"
            if not search_provider:
                if first_profile.options.get(CONF_ENABLE_CUSTOM_TOOLS, first_profile.data.get(CONF_ENABLE_CUSTOM_TOOLS, False)):
                    search_provider = "brave"
                else:
                    search_provider = DEFAULT_SEARCH_PROVIDER

            shared_settings = {
                CONF_MCP_PORT: first_profile.options.get(
                    CONF_MCP_PORT,
                    first_profile.data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
                ),
                CONF_SEARCH_PROVIDER: search_provider,
                CONF_BRAVE_API_KEY: first_profile.options.get(
                    CONF_BRAVE_API_KEY,
                    first_profile.data.get(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY)
                ),
                CONF_ALLOWED_IPS: first_profile.options.get(
                    CONF_ALLOWED_IPS,
                    first_profile.data.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
                ),
                CONF_ENABLE_GAP_FILLING: first_profile.options.get(
                    CONF_ENABLE_GAP_FILLING,
                    first_profile.data.get(CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING)
                ),
            }
        else:
            # No profiles exist yet (shouldn't happen in normal flow), use defaults
            _LOGGER.info("No existing profiles found, using default shared settings")
            shared_settings = {
                CONF_MCP_PORT: DEFAULT_MCP_PORT,
                CONF_SEARCH_PROVIDER: DEFAULT_SEARCH_PROVIDER,
                CONF_BRAVE_API_KEY: DEFAULT_BRAVE_API_KEY,
                CONF_ALLOWED_IPS: DEFAULT_ALLOWED_IPS,
                CONF_ENABLE_GAP_FILLING: DEFAULT_ENABLE_GAP_FILLING,
            }

        # Create system entry with extracted/default settings
        await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": "system"},
            data=shared_settings
        )

        # Get the created entry
        system_entry = get_system_entry(hass)

        if system_entry is None:
            raise ConfigEntryNotReady("Failed to create system entry")

        _LOGGER.info("✅ System entry created successfully with settings: mcp_port=%s, search_provider=%s",
                    shared_settings.get(CONF_MCP_PORT),
                    shared_settings.get(CONF_SEARCH_PROVIDER))

    return system_entry


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up MCP Assist from a config entry."""
    # Skip setup for system entry (it only stores config, doesn't create entities)
    if entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
        _LOGGER.debug("Skipping setup for system entry (config only)")
        return True

    profile_name = entry.data.get("profile_name", "Default")
    _LOGGER.info("Setting up MCP Assist integration - Profile: %s", profile_name)

    # Migrate legacy "brave_search" tool name to "search" in Technical Instructions
    await _migrate_brave_search_tool_name(hass, entry)

    hass.data.setdefault(DOMAIN, {})

    # Ensure system entry exists (creates with defaults if not)
    await ensure_system_entry(hass)

    # Create lock for server initialization if it doesn't exist
    if "server_init_lock" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["server_init_lock"] = asyncio.Lock()

    try:
        # Use lock to prevent race condition when multiple profiles load simultaneously
        async with hass.data[DOMAIN]["server_init_lock"]:
            # Handle shared MCP server and index manager
            if "shared_mcp_server" not in hass.data[DOMAIN]:
                # First entry - create shared MCP server and index manager
                # Read MCP port from system entry
                system_entry = get_system_entry(hass)
                mcp_port = system_entry.data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
                _LOGGER.info("Creating shared MCP server on port %d (from system entry)", mcp_port)

                # Create and start index manager
                index_manager = IndexManager(hass)
                await index_manager.start()
                hass.data[DOMAIN]["index_manager"] = index_manager

                # Create and start MCP server
                mcp_server = MCPServer(hass, mcp_port, entry)
                try:
                    await mcp_server.start()
                except OSError as e:
                    if "Address already in use" in str(e):
                        _LOGGER.error(
                            "Port %d is already in use. Either change CONF_MCP_PORT in your "
                            "configuration or stop the service using port %d.",
                            mcp_port, mcp_port
                        )
                        raise ConfigEntryNotReady(f"Port {mcp_port} already in use") from e
                    raise

                hass.data[DOMAIN]["shared_mcp_server"] = mcp_server
                hass.data[DOMAIN]["mcp_refcount"] = 0
                hass.data[DOMAIN]["mcp_port"] = mcp_port

                _LOGGER.info("✅ Shared MCP server and index manager created successfully")
            else:
                # Reuse existing MCP server
                mcp_port = hass.data[DOMAIN]["mcp_port"]
                _LOGGER.info("Reusing existing shared MCP server on port %d", mcp_port)

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

        # Register update listener for option changes
        entry.async_on_unload(entry.add_update_listener(async_update_options))

        _LOGGER.info("✅ Profile '%s' setup complete, Entry ID: %s", profile_name, entry.entry_id)

        return True

    except Exception as err:
        _LOGGER.error("Failed to setup MCP Assist profile '%s': %s", profile_name, err)
        raise ConfigEntryNotReady(f"Setup failed: {err}") from err


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    _LOGGER.debug("Options updated for entry %s, reloading...", entry.entry_id)
    # Reload the integration to apply new options (including search provider changes)
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # System entry doesn't need unloading (no platforms)
    if entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
        _LOGGER.debug("System entry unloaded (no cleanup needed)")
        return True

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