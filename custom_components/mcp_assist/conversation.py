"""Conversation platform for MCP Assist."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .agent import MCPAssistConversationEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up MCP Assist conversation entity."""
    async_add_entities([MCPAssistConversationEntity(hass, entry)])
