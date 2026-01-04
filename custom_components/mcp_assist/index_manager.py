"""Smart Index Manager for context-aware entity discovery.

This module generates and maintains a lightweight index of Home Assistant's
system structure, enabling LLMs to make smart, targeted entity queries without
requiring full context dumps.

The index includes:
- Areas with entity counts
- Domains with counts
- Device classes (grouped by domain) with counts
- Inferred entity types (via LLM gap-filling)
- People, pets, calendars, zones, automations, scripts, input helpers
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback, Event, Context
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    device_registry as dr,
)
from homeassistant.helpers.entity_registry import EVENT_ENTITY_REGISTRY_UPDATED
from homeassistant.components.homeassistant import async_should_expose

_LOGGER = logging.getLogger(__name__)


class IndexManager:
    """Manages the system structure index for smart entity discovery."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize index manager."""
        self.hass = hass
        self._index: Optional[Dict[str, Any]] = None
        self._last_updated: Optional[datetime] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_debounce_seconds = 60

    async def start(self) -> None:
        """Start index manager and set up event listeners."""
        _LOGGER.info("Starting Smart Index Manager")

        # Listen for entity registry changes
        @callback
        def entity_registry_changed(event: Event) -> None:
            """Schedule index refresh on entity changes."""
            _LOGGER.debug("Entity registry changed, scheduling index refresh")
            self._schedule_refresh()

        self.hass.bus.async_listen(
            EVENT_ENTITY_REGISTRY_UPDATED,
            entity_registry_changed
        )

        _LOGGER.info("✅ Smart Index Manager started successfully")
        _LOGGER.debug("Index will be generated lazily on first request")

    def _schedule_refresh(self) -> None:
        """Schedule index refresh with debouncing."""
        # Cancel pending refresh if exists
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

        # Schedule new refresh after debounce period
        self._refresh_task = asyncio.create_task(self._delayed_refresh())

    async def _delayed_refresh(self) -> None:
        """Refresh index after debounce delay."""
        try:
            await asyncio.sleep(self._refresh_debounce_seconds)
            await self.refresh_index()
        except asyncio.CancelledError:
            _LOGGER.debug("Index refresh cancelled (newer change detected)")

    async def refresh_index(self) -> None:
        """Generate fresh index from current system state."""
        start_time = datetime.now()
        _LOGGER.info("Generating system structure index...")

        try:
            self._index = await self.generate_index()
            self._last_updated = datetime.now()

            # Calculate index size estimate
            index_str = str(self._index)
            char_count = len(index_str)
            token_estimate = char_count // 4  # Rough estimate: 4 chars per token

            elapsed = (datetime.now() - start_time).total_seconds()
            _LOGGER.info(
                "✅ Index generated in %.2fs - Size: %d chars (~%d tokens)",
                elapsed,
                char_count,
                token_estimate
            )

        except Exception as err:
            _LOGGER.error("Failed to generate index: %s", err, exc_info=True)

    async def get_index(self) -> Dict[str, Any]:
        """Get the current index, generating if needed."""
        if self._index is None:
            await self.refresh_index()

        return self._index or {}

    async def generate_index(self) -> Dict[str, Any]:
        """Generate the system structure index.

        Returns a comprehensive index including areas, domains, device_classes,
        people, pets, calendars, zones, automations, scripts, and input helpers.
        """
        # Query all index components in parallel for speed
        results = await asyncio.gather(
            self._get_areas(),
            self._get_domains(),
            self._get_device_classes(),
            self._get_entities_without_device_class(),
            self._get_people(),
            self._get_calendars(),
            self._get_zones(),
            self._get_automations(),
            self._get_scripts(),
            self._get_input_helpers(),
            return_exceptions=True
        )

        # Extract results and handle any errors
        (
            areas,
            domains,
            device_classes,
            entities_without_device_class,
            people,
            calendars,
            zones,
            automations,
            scripts,
            input_helpers,
        ) = results

        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _LOGGER.error("Error generating index component %d: %s", i, result)

        # Perform LLM gap-filling for entities without device_class (if enabled)
        inferred_types = {}
        if not isinstance(entities_without_device_class, Exception) and entities_without_device_class:
            # Check if gap-filling is enabled in config
            gap_filling_enabled = await self._is_gap_filling_enabled()
            if gap_filling_enabled:
                inferred_types = await self._infer_entity_types(entities_without_device_class)
            else:
                _LOGGER.debug("Gap-filling disabled in config, skipping entity type inference")

        # Build index dict
        index = {
            "areas": areas if not isinstance(areas, Exception) else [],
            "domains": domains if not isinstance(domains, Exception) else {},
            "device_classes": device_classes if not isinstance(device_classes, Exception) else {},
            "inferred_types": inferred_types,
            "people": people if not isinstance(people, Exception) else [],
            "pets": [],  # Could be extracted from inferred_types or person entities
            "calendars": calendars if not isinstance(calendars, Exception) else [],
            "zones": zones if not isinstance(zones, Exception) else [],
            "automations": automations if not isinstance(automations, Exception) else [],
            "scripts": scripts if not isinstance(scripts, Exception) else [],
        }

        # Add input helpers
        if not isinstance(input_helpers, Exception):
            index.update(input_helpers)

        return index

    async def _get_areas(self) -> List[Dict[str, Any]]:
        """Get all areas with entity counts."""
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)

        # Count entities per area
        area_counts = defaultdict(int)
        for entity in entity_reg.entities.values():
            if async_should_expose(self.hass, "conversation", entity.entity_id):
                if entity.area_id:
                    area_counts[entity.area_id] += 1
                # Also check device area
                elif entity.device_id:
                    device_reg = dr.async_get(self.hass)
                    device = device_reg.async_get(entity.device_id)
                    if device and device.area_id:
                        area_counts[device.area_id] += 1

        # Build area list with counts
        areas = []
        for area in area_reg.async_list_areas():
            count = area_counts.get(area.id, 0)
            if count > 0:  # Only include areas with entities
                areas.append({
                    "name": area.name,
                    "entity_count": count
                })

        # Sort by name
        areas.sort(key=lambda x: x["name"])

        _LOGGER.debug("Found %d areas with entities", len(areas))
        return areas

    async def _get_domains(self) -> Dict[str, int]:
        """Get all domains with entity counts."""
        entity_reg = er.async_get(self.hass)

        domain_counts = defaultdict(int)
        for entity in entity_reg.entities.values():
            if async_should_expose(self.hass, "conversation", entity.entity_id):
                domain = entity.entity_id.split('.')[0]
                domain_counts[domain] += 1

        # Convert to regular dict and sort by count (descending)
        domains = dict(sorted(
            domain_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        _LOGGER.debug("Found %d domains with %d total entities",
                     len(domains), sum(domains.values()))
        return domains

    async def _get_device_classes(self) -> Dict[str, Dict[str, int]]:
        """Get device classes grouped by domain with counts."""
        entity_reg = er.async_get(self.hass)

        # Structure: {domain: {device_class: count}}
        device_classes = defaultdict(lambda: defaultdict(int))

        for entity in entity_reg.entities.values():
            if not async_should_expose(self.hass, "conversation", entity.entity_id):
                continue

            domain = entity.entity_id.split('.')[0]

            # Get device_class from entity attributes
            state_obj = self.hass.states.get(entity.entity_id)
            if state_obj and state_obj.attributes:
                device_class = state_obj.attributes.get('device_class')
                if device_class:
                    device_classes[domain][device_class] += 1

        # Convert to regular dicts and sort
        result = {}
        for domain in sorted(device_classes.keys()):
            # Sort by count descending
            result[domain] = dict(sorted(
                device_classes[domain].items(),
                key=lambda x: x[1],
                reverse=True
            ))

        total_classes = sum(len(classes) for classes in result.values())
        _LOGGER.debug("Found %d device classes across %d domains",
                     total_classes, len(result))
        return result

    async def _get_people(self) -> List[Dict[str, Any]]:
        """Get all person entities with names."""
        entity_reg = er.async_get(self.hass)

        people = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('person.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    # Get friendly name from state
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]

                    people.append({
                        "name": name,
                        # Aliases could be populated via LLM or config
                        # For now, leave empty
                    })

        people.sort(key=lambda x: x["name"])
        _LOGGER.debug("Found %d people", len(people))
        return people

    async def _get_calendars(self) -> List[str]:
        """Get all calendar entities."""
        entity_reg = er.async_get(self.hass)

        calendars = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('calendar.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                    calendars.append(name)

        calendars.sort()
        _LOGGER.debug("Found %d calendars", len(calendars))
        return calendars

    async def _get_zones(self) -> List[str]:
        """Get all zone entities."""
        entity_reg = er.async_get(self.hass)

        zones = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('zone.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                    # Exclude 'Home' zone as it's the default
                    if name.lower() != 'home':
                        zones.append(name)

        zones.sort()
        _LOGGER.debug("Found %d zones", len(zones))
        return zones

    async def _get_automations(self) -> List[str]:
        """Get all automation entities."""
        entity_reg = er.async_get(self.hass)

        automations = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('automation.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                    automations.append(name)

        automations.sort()
        _LOGGER.debug("Found %d automations", len(automations))
        return automations

    async def _get_scripts(self) -> List[str]:
        """Get all script entities."""
        entity_reg = er.async_get(self.hass)

        scripts = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('script.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                    scripts.append(name)

        scripts.sort()
        _LOGGER.debug("Found %d scripts", len(scripts))
        return scripts

    async def _get_input_helpers(self) -> Dict[str, List[str]]:
        """Get all input helper entities grouped by type."""
        entity_reg = er.async_get(self.hass)

        # Input helper prefixes
        input_types = [
            'input_boolean',
            'input_text',
            'input_number',
            'input_select',
            'input_datetime',
            'input_button',
        ]

        helpers = {input_type: [] for input_type in input_types}

        for entity in entity_reg.entities.values():
            for input_type in input_types:
                if entity.entity_id.startswith(f'{input_type}.'):
                    if async_should_expose(self.hass, "conversation", entity.entity_id):
                        state_obj = self.hass.states.get(entity.entity_id)
                        name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                        helpers[input_type].append(name)
                        break

        # Sort each list and convert to plural keys
        result = {}
        for input_type, names in helpers.items():
            if names:  # Only include types with entities
                names.sort()
                # Convert key: input_boolean → input_booleans
                plural_key = f"{input_type}s"
                result[plural_key] = names

        total_helpers = sum(len(names) for names in result.values())
        _LOGGER.debug("Found %d input helpers across %d types",
                     total_helpers, len(result))
        return result

    async def _get_entities_without_device_class(self) -> List[str]:
        """Get entity IDs that don't have a device_class attribute."""
        entity_reg = er.async_get(self.hass)

        entities_without_device_class = []

        for entity in entity_reg.entities.values():
            if not async_should_expose(self.hass, "conversation", entity.entity_id):
                continue

            # Get state object to check for device_class
            state_obj = self.hass.states.get(entity.entity_id)
            if not state_obj:
                continue

            # Check if entity has device_class attribute
            device_class = state_obj.attributes.get('device_class')
            if not device_class:
                entities_without_device_class.append(entity.entity_id)

        _LOGGER.debug("Found %d entities without device_class",
                     len(entities_without_device_class))
        return entities_without_device_class

    def _extract_patterns(self, entity_ids: List[str]) -> Dict[str, List[str]]:
        """Extract common naming patterns from entity IDs.

        Groups entities by their naming patterns for LLM analysis.

        Args:
            entity_ids: List of entity IDs to analyze

        Returns:
            Dict mapping patterns to lists of matching entity IDs
        """
        patterns = defaultdict(list)

        for entity_id in entity_ids:
            # Split into domain and name
            if '.' not in entity_id:
                continue

            domain, name = entity_id.split('.', 1)

            # Look for common suffixes/patterns
            # Examples: *_person_detected, *_ble_area, *_battery_level

            # Check for common suffixes
            common_suffixes = [
                '_person_detected',
                '_vehicle_detected',
                '_animal_detected',
                '_package_detected',
                '_ble_area',
                '_ble_room_presence',
                '_battery_level',
                '_battery',
                '_rssi',
                '_signal_strength',
                '_linkquality',
            ]

            matched = False
            for suffix in common_suffixes:
                if name.endswith(suffix):
                    pattern = f"{domain}.*{suffix}"
                    patterns[pattern].append(entity_id)
                    matched = True
                    break

            # If no suffix match, try to find common word patterns
            if not matched:
                # Look for entities with similar structure
                # e.g., sensor.bedroom_temp, sensor.kitchen_temp → sensor.*_temp
                words = name.split('_')
                if len(words) >= 2:
                    # Use last word as potential pattern
                    last_word = words[-1]
                    if len(last_word) > 2:  # Ignore very short words
                        pattern = f"{domain}.*_{last_word}"
                        patterns[pattern].append(entity_id)

        # Filter out patterns with too few entities (noise)
        filtered_patterns = {
            pattern: entities
            for pattern, entities in patterns.items()
            if len(entities) >= 2  # At least 2 entities to be considered a pattern
        }

        _LOGGER.debug("Extracted %d patterns from %d entities",
                     len(filtered_patterns), len(entity_ids))
        return filtered_patterns

    async def _infer_entity_types(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Use LLM to infer semantic types for entities without device_class.

        Args:
            entity_ids: List of entity IDs without device_class

        Returns:
            Dict mapping inferred type names to pattern and count info
        """
        if not entity_ids:
            return {}

        # Extract patterns first
        patterns = self._extract_patterns(entity_ids)

        if not patterns:
            _LOGGER.debug("No patterns found for LLM gap-filling")
            return {}

        # Build prompt for LLM
        pattern_list = "\n".join([
            f"- {pattern}: {len(entities)} entities (examples: {', '.join(entities[:3])})"
            for pattern, entities in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
        ])

        prompt = f"""Analyze these Home Assistant entity patterns and categorize them into semantic types.

Entity patterns found:
{pattern_list}

For each pattern, determine what type of sensor/detection it represents. Use descriptive category names like "person_detection", "location_tracking", "battery_monitoring", etc.

Respond with ONLY a JSON object in this exact format:
{{
  "category_name": {{
    "pattern": "entity_pattern",
    "count": number_of_entities,
    "description": "brief description"
  }},
  ...
}}

Example:
{{
  "person_detection": {{
    "pattern": "binary_sensor.*_person_detected",
    "count": 6,
    "description": "Detects presence of people"
  }},
  "location_tracking": {{
    "pattern": "sensor.*_ble_area",
    "count": 9,
    "description": "BLE-based room location tracking"
  }}
}}

Focus on meaningful categories that would help discover relevant entities for user queries."""

        try:
            # Call LLM via conversation agent
            inferred = await self._call_llm_for_inference(prompt)
            _LOGGER.info("LLM gap-filling completed: found %d inferred types", len(inferred))
            return inferred
        except Exception as err:
            _LOGGER.warning("LLM gap-filling failed: %s. Index will not include inferred types.", err)
            return {}

    async def _call_llm_for_inference(self, prompt: str) -> Dict[str, Any]:
        """Call the user's configured LLM to perform inference.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Parsed inference results as dict

        Raises:
            Exception: If LLM call fails or response cannot be parsed
        """
        from homeassistant.components import conversation
        from .const import DOMAIN

        # Get any conversation agent entry (they all share the same LLM config)
        entries = [entry for entry in self.hass.config_entries.async_entries(DOMAIN)]
        if not entries:
            raise ValueError("No MCP Assist config entries found")

        entry = entries[0]

        # Get the agent
        agent_data = self.hass.data.get(DOMAIN, {}).get(entry.entry_id)
        if not agent_data:
            raise ValueError("No agent found for LLM inference")

        agent = agent_data.get("agent")
        if not agent:
            raise ValueError("Agent not available for LLM inference")

        # Create a minimal conversation input with all required fields
        conversation_input = conversation.ConversationInput(
            text=prompt,
            context=Context(),  # Minimal context
            conversation_id=None,  # One-shot query
            device_id=None,  # Not from a specific device
            satellite_id=None,  # Not from a satellite
            language="en",
            agent_id=entry.entry_id,  # Use the entry ID as agent ID
        )

        # Call the agent
        _LOGGER.debug("Calling LLM for entity type inference...")
        response = await agent.async_process(conversation_input)

        if not response or not response.response:
            raise ValueError("Empty response from LLM")

        # Parse the response
        response_text = response.response.speech.get("plain", {}).get("speech", "")
        if not response_text:
            raise ValueError("No speech in LLM response")

        # Parse JSON from response
        return self._parse_inferred_types(response_text)

    def _parse_inferred_types(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured inferred types.

        Args:
            response_text: Raw text response from LLM

        Returns:
            Dict of inferred types with pattern, count, description

        Raises:
            ValueError: If response cannot be parsed
        """
        import json

        # Try to extract JSON from response
        # LLM might wrap it in markdown code blocks
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            # Remove first line (```json or similar) and last line (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = '\n'.join(lines)

        # Try to parse as JSON
        try:
            parsed = json.loads(response_text)
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")

            # Validate structure
            for category, data in parsed.items():
                if not isinstance(data, dict):
                    raise ValueError(f"Category {category} is not a dict")
                if "pattern" not in data or "count" not in data:
                    raise ValueError(f"Category {category} missing required fields")

            _LOGGER.debug("Successfully parsed %d inferred types from LLM", len(parsed))
            return parsed

        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to parse LLM response as JSON: %s", err)
            _LOGGER.debug("LLM response was: %s", response_text[:500])
            raise ValueError(f"Invalid JSON in LLM response: {err}")
        except ValueError as err:
            _LOGGER.error("Invalid structure in LLM response: %s", err)
            raise

    async def _is_gap_filling_enabled(self) -> bool:
        """Check if gap-filling is enabled in config.

        Returns:
            True if gap-filling is enabled, False otherwise
        """
        from .const import DOMAIN, CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING

        # Get any config entry to check the setting
        entries = [entry for entry in self.hass.config_entries.async_entries(DOMAIN)]
        if not entries:
            _LOGGER.debug("No config entries found, using default gap-filling setting: %s",
                         DEFAULT_ENABLE_GAP_FILLING)
            return DEFAULT_ENABLE_GAP_FILLING

        entry = entries[0]  # All entries share the same MCP server/index

        # Check options first, then data, then default
        enabled = entry.options.get(
            CONF_ENABLE_GAP_FILLING,
            entry.data.get(CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING)
        )

        _LOGGER.debug("Gap-filling enabled in config: %s", enabled)
        return enabled
