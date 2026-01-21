"""Fast Path Processor for handling simple commands without LLM."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .loader import KeywordLoader

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


@dataclass
class FastPathResult:
    """Result of a Fast Path processing attempt."""

    success: bool
    """Whether the command was successfully processed."""

    handled: bool
    """Whether Fast Path attempted to handle this (even if it failed)."""

    response: str
    """The response message to send to the user."""

    action: str | None = None
    """The detected action (e.g., 'turn_on', 'set_brightness')."""

    entity_ids: list[str] = field(default_factory=list)
    """List of entity IDs that were affected."""

    value: Any = None
    """The extracted value (e.g., brightness percentage)."""

    error: str | None = None
    """Error message if something went wrong."""


# Action mapping to Home Assistant services
ACTION_SERVICE_MAP: dict[str, tuple[str, str]] = {
    "turn_on": ("homeassistant", "turn_on"),
    "turn_off": ("homeassistant", "turn_off"),
    "toggle": ("homeassistant", "toggle"),
    "open": ("cover", "open_cover"),
    "close": ("cover", "close_cover"),
    "lock": ("lock", "lock"),
    "unlock": ("lock", "unlock"),
}

# Domain-specific mappings for brightness, position, temperature
DOMAIN_VALUE_SERVICES: dict[str, dict[str, tuple[str, str, str]]] = {
    "light": {
        "brightness": ("light", "turn_on", "brightness_pct"),
        "color_temp": ("light", "turn_on", "color_temp_kelvin"),
    },
    "cover": {
        "position": ("cover", "set_cover_position", "position"),
    },
    "climate": {
        "temperature": ("climate", "set_temperature", "temperature"),
    },
    "fan": {
        "brightness": ("fan", "set_percentage", "percentage"),  # brightness -> percentage for fans
    },
}


def is_fast_path_candidate(text: str, loader: KeywordLoader) -> bool:
    """Quick check if text might be a Fast Path candidate.
    
    This is a lightweight check to avoid full processing for complex queries.
    
    Args:
        text: The user input text
        loader: The KeywordLoader instance
        
    Returns:
        True if the text might be handled by Fast Path
    """
    text_lower = text.lower()
    
    # Skip if text contains question words (likely a query, not a command)
    question_indicators = [
        "?", "was ", "wer ", "wie ", "wo ", "wann ", "warum ", "welche",
        "what ", "who ", "how ", "where ", "when ", "why ", "which",
        "combien", "pourquoi", "comment", "quand", "où",
        "qué", "quién", "cómo", "cuándo", "dónde", "por qué",
        "wat ", "wie ", "waar ", "wanneer ", "waarom",
    ]
    
    for indicator in question_indicators:
        if indicator in text_lower:
            return False
    
    # Skip if text is too long (likely a complex request)
    if len(text.split()) > 12:
        return False
    
    # Check if any action keyword is present
    all_keywords = loader.get_all_action_keywords()
    for action, keywords in all_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True
    
    # Check for value patterns that indicate a command
    # e.g., "50%", "22 grad"
    if re.search(r'\d+\s*(%|°|grad|degrees?|prozent|percent)', text_lower):
        return True
    
    return False


class FastPathProcessor:
    """Processor for handling simple commands without LLM involvement."""

    def __init__(
        self,
        hass: HomeAssistant,
        language: str = "en",
        entity_names: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the Fast Path Processor.
        
        Args:
            hass: Home Assistant instance
            language: Language code for keywords and responses
            entity_names: Mapping of entity_id to list of names/aliases
        """
        self._hass = hass
        self._loader = KeywordLoader(language)
        self._entity_names = entity_names or {}
        self._language = language

    def set_entity_names(self, entity_names: dict[str, list[str]]) -> None:
        """Update the entity names mapping.
        
        Args:
            entity_names: Mapping of entity_id to list of names/aliases
        """
        self._entity_names = entity_names

    def set_language(self, language: str) -> None:
        """Change the language for keywords and responses.
        
        Args:
            language: The new language code
        """
        if language != self._language:
            self._language = language
            self._loader.language = language
            self._loader.reload()

    async def process(self, text: str) -> FastPathResult:
        """Process a text command through the Fast Path.
        
        Args:
            text: The user's command text
            
        Returns:
            FastPathResult with the outcome
        """
        text_lower = text.lower()
        
        # Step 1: Detect action
        action, action_keyword = self._detect_action(text_lower)
        
        if not action:
            return FastPathResult(
                success=False,
                handled=False,
                response="",
            )
        
        _LOGGER.debug("Fast Path detected action: %s (keyword: '%s')", action, action_keyword)
        
        # Step 2: Extract value (if any)
        value, value_type = self._extract_value(text_lower)
        
        _LOGGER.debug("Fast Path extracted value: %s (type: %s)", value, value_type)
        
        # Step 3: Find matching entities
        # Remove the action keyword from text for better entity matching
        text_for_entity = text_lower.replace(action_keyword.lower(), " ").strip()
        entity_ids = self._find_entities(text_for_entity)
        
        if not entity_ids:
            _LOGGER.debug("Fast Path found no matching entities in: %s", text_for_entity)
            return FastPathResult(
                success=False,
                handled=True,
                response=self._loader.get_error_response("no_entity"),
                action=action,
                error="No matching entity found",
            )
        
        _LOGGER.debug("Fast Path found entities: %s", entity_ids)
        
        # Step 4: Determine the final action and service
        if value is not None and value_type:
            final_action = f"set_{value_type}"
        elif action in ("increase", "decrease"):
            # For now, increase/decrease without value are not fully supported
            # We would need to know the current state and adjust
            final_action = action
        else:
            final_action = action
        
        # Step 5: Execute the action
        try:
            await self._execute_action(entity_ids, final_action, value, value_type)
            
            # Build response
            entity_name = self._get_friendly_name(entity_ids[0]) if entity_ids else "Entity"
            response = self._loader.get_response(final_action, len(entity_ids))
            response = response.format(
                entity=entity_name,
                count=len(entity_ids),
                value=value if value is not None else "",
            )
            
            return FastPathResult(
                success=True,
                handled=True,
                response=response,
                action=final_action,
                entity_ids=entity_ids,
                value=value,
            )
            
        except Exception as e:
            _LOGGER.error("Fast Path execution failed: %s", e)
            error_response = self._loader.get_error_response("action_failed")
            return FastPathResult(
                success=False,
                handled=True,
                response=error_response.format(error=str(e)),
                action=final_action,
                entity_ids=entity_ids,
                error=str(e),
            )

    def _detect_action(self, text: str) -> tuple[str | None, str]:
        """Detect the action from the text.
        
        Args:
            text: Lowercase user input
            
        Returns:
            Tuple of (action_name, matched_keyword) or (None, "")
        """
        all_keywords = self._loader.get_all_action_keywords()
        
        best_match: tuple[str | None, str] = (None, "")
        
        for action, keywords in all_keywords.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in text:
                    # Prefer longer matches (more specific)
                    if len(keyword_lower) > len(best_match[1]):
                        best_match = (action, keyword_lower)
        
        return best_match

    def _extract_value(self, text: str) -> tuple[Any, str | None]:
        """Extract a value from the text.
        
        Args:
            text: Lowercase user input
            
        Returns:
            Tuple of (value, value_type) or (None, None)
        """
        # Check for brightness (percentage)
        for pattern in self._loader.get_value_patterns("brightness"):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1)), "brightness"
                except (ValueError, IndexError):
                    pass
        
        # Check for temperature
        for pattern in self._loader.get_value_patterns("temperature"):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Handle both . and , as decimal separator
                    temp_str = match.group(1).replace(",", ".")
                    return float(temp_str), "temperature"
                except (ValueError, IndexError):
                    pass
        
        # Check for position (cover)
        for pattern in self._loader.get_value_patterns("position"):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1)), "position"
                except (ValueError, IndexError):
                    pass
        
        # Check for color temperature keywords
        for color_kw in self._loader.get_color_temp_keywords():
            keyword = color_kw.get("keyword", "").lower()
            aliases = [a.lower() for a in color_kw.get("aliases", [])]
            value = color_kw.get("value")
            
            if keyword in text or any(alias in text for alias in aliases):
                return value, "color_temp"
        
        return None, None

    def _find_entities(self, text: str) -> list[str]:
        """Find entities mentioned in the text.
        
        Args:
            text: The text to search for entity names
            
        Returns:
            List of matching entity IDs
        """
        matched_entities: list[tuple[str, int]] = []
        text_normalized = self._normalize_for_matching(text)
        
        for entity_id, names in self._entity_names.items():
            for name in names:
                name_normalized = self._normalize_for_matching(name)
                
                # Direct substring match
                if name_normalized in text_normalized:
                    matched_entities.append((entity_id, len(name_normalized)))
                    break
        
        # Sort by match length (longer = more specific) and return entity IDs
        matched_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for entity_id, _ in matched_entities:
            if entity_id not in seen:
                seen.add(entity_id)
                result.append(entity_id)
        
        return result

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for entity matching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Replace umlauts and special chars
        replacements = {
            "ä": "a", "ö": "o", "ü": "u", "ß": "ss",
            "é": "e", "è": "e", "ê": "e", "ë": "e",
            "á": "a", "à": "a", "â": "a",
            "í": "i", "ì": "i", "î": "i",
            "ó": "o", "ò": "o", "ô": "o",
            "ú": "u", "ù": "u", "û": "u",
            "ñ": "n", "ç": "c",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text

    def _get_friendly_name(self, entity_id: str) -> str:
        """Get the friendly name for an entity.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            The friendly name or entity_id if not found
        """
        names = self._entity_names.get(entity_id, [])
        if names:
            return names[0]
        
        # Fallback: get from Home Assistant state
        state = self._hass.states.get(entity_id)
        if state and state.attributes.get("friendly_name"):
            return state.attributes["friendly_name"]
        
        return entity_id

    async def _execute_action(
        self,
        entity_ids: list[str],
        action: str,
        value: Any,
        value_type: str | None,
    ) -> None:
        """Execute the action on the entities.
        
        Args:
            entity_ids: List of entity IDs to act on
            action: The action to perform
            value: The value to set (if applicable)
            value_type: The type of value
        """
        for entity_id in entity_ids:
            domain = entity_id.split(".")[0]
            
            if value is not None and value_type:
                # Value-based action (set brightness, temperature, etc.)
                domain_services = DOMAIN_VALUE_SERVICES.get(domain, {})
                service_info = domain_services.get(value_type)
                
                if service_info:
                    service_domain, service_name, attr_name = service_info
                    await self._hass.services.async_call(
                        service_domain,
                        service_name,
                        {
                            "entity_id": entity_id,
                            attr_name: value,
                        },
                        blocking=True,
                    )
                else:
                    _LOGGER.warning(
                        "No service mapping for domain=%s, value_type=%s",
                        domain, value_type
                    )
            else:
                # Simple action (turn_on, turn_off, etc.)
                base_action = action.replace("set_", "")
                service_info = ACTION_SERVICE_MAP.get(base_action)
                
                if service_info:
                    service_domain, service_name = service_info
                    
                    # Use domain-specific service if available
                    if domain == "cover" and base_action in ("turn_on", "toggle"):
                        service_domain = "cover"
                        service_name = "open_cover" if base_action == "turn_on" else "toggle"
                    elif domain == "cover" and base_action == "turn_off":
                        service_domain = "cover"
                        service_name = "close_cover"
                    
                    await self._hass.services.async_call(
                        service_domain,
                        service_name,
                        {"entity_id": entity_id},
                        blocking=True,
                    )
                else:
                    _LOGGER.warning("No service mapping for action: %s", action)
