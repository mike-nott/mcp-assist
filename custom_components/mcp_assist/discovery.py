"""Smart Entity Discovery System for MCP Server.

This module provides intelligent entity discovery with relationship understanding,
pattern recognition, and structured results optimized for LLM interaction.
"""

import asyncio
import fnmatch
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er, device_registry as dr
from homeassistant.components.homeassistant import async_should_expose

from .const import MAX_ENTITIES_PER_DISCOVERY

_LOGGER = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of discovery queries."""
    PERSON = "person"
    PET = "pet"
    DEVICE = "device"
    AREA = "area"
    STATE = "state"
    AGGREGATE = "aggregate"
    GENERAL = "general"


class EntityPattern:
    """Entity pattern definitions for relationship detection."""

    # Person-related patterns
    PERSON_PATTERNS = [
        (r"^person\.{name}$", "primary", "Person entity"),
        (r"^device_tracker\..*{name}.*", "device_tracking", "Device tracker"),
        (r"^sensor\..*{name}.*_ble_area$", "ble_tracking", "BLE area sensor"),
        (r"^sensor\..*{name}.*_ble_room_presence$", "ble_tracking", "BLE room presence"),
        (r"^input_text\.room_{name}$", "room_tracking", "Room tracker"),
        (r"^input_text\.{name}_room$", "room_tracking", "Room tracker"),
        (r"^input_boolean\.{name}_home$", "presence", "Home presence"),
        (r"^input_boolean\.{name}_inside$", "presence", "Inside tracker"),
        (r"^binary_sensor\.{name}_home$", "presence", "Home sensor"),
    ]

    # Pet-related patterns (similar to person but without device_tracker)
    PET_PATTERNS = [
        (r"^person\.{name}$", "primary", "Pet as person entity"),
        (r"^binary_sensor\.{name}$", "primary", "Pet sensor"),
        (r"^sensor\..*{name}.*_ble_area$", "ble_tracking", "BLE area sensor"),
        (r"^input_text\.room_{name}$", "room_tracking", "Room tracker"),
        (r"^input_text\.{name}_room$", "room_tracking", "Room tracker"),
        (r"^input_boolean\.{name}_inside$", "presence", "Inside tracker"),
        (r"^input_boolean\.{name}_home$", "presence", "Home presence"),
    ]


class SmartDiscovery:
    """Smart entity discovery with relationship understanding."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize smart discovery."""
        self.hass = hass
        self._entity_cache = None
        self._cache_time = None

    async def discover_entities(
        self,
        entity_type: Optional[str] = None,
        area: Optional[str] = None,
        domain: Optional[str] = None,
        state: Optional[str] = None,
        name_contains: Optional[str] = None,
        limit: int = 20,
        device_class: Optional[Union[str, List[str]]] = None,
        name_pattern: Optional[str] = None,
        inferred_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Smart entity discovery with relationship understanding.

        This is the main entry point that routes to appropriate discovery strategy.

        Args:
            entity_type: Entity type (deprecated, use domain)
            area: Area name to filter by
            domain: Domain to filter by (e.g., 'sensor', 'light')
            state: State to filter by (e.g., 'on', 'off')
            name_contains: Substring to search for in entity names
            limit: Maximum number of entities to return
            device_class: Device class to filter by (e.g., 'temperature', 'motion')
                         Can be a single string or list of strings for OR logic
            name_pattern: Wildcard pattern to match entity IDs (e.g., '*_person_detected')
            inferred_type: Inferred entity type from the index (e.g., 'person_detection')
                          Looks up the pattern from the index and applies it
        """
        # Detect query type and intent
        query_type = self._detect_query_type(
            entity_type, area, domain, state, name_contains
        )

        # Log the detected query type
        _LOGGER.debug(f"Detected query type: {query_type}, name_contains: {name_contains}")

        # Route to appropriate discovery method
        if query_type == QueryType.PERSON and name_contains and not device_class and not name_pattern and not inferred_type:
            return await self._discover_person_entities(name_contains, limit)
        elif query_type == QueryType.PET and name_contains and not device_class and not name_pattern and not inferred_type:
            return await self._discover_pet_entities(name_contains, limit)
        elif query_type == QueryType.AGGREGATE and not device_class and not name_pattern and not inferred_type:
            return await self._discover_aggregate_entities(domain, state, limit)
        elif area and not device_class and not name_pattern and not inferred_type:
            return await self._discover_area_entities(area, domain, state, limit)
        else:
            # Fall back to general discovery (handles device_class, name_pattern, and inferred_type)
            return await self._discover_general_entities(
                entity_type, area, domain, state, name_contains, limit, device_class, name_pattern, inferred_type
            )

    def _detect_query_type(
        self,
        entity_type: Optional[str],
        area: Optional[str],
        domain: Optional[str],
        state: Optional[str],
        name_contains: Optional[str]
    ) -> QueryType:
        """Detect the type of query based on parameters."""
        # Check for person/pet queries
        if name_contains:
            name_lower = name_contains.lower()

            # Check if it's likely a person name
            if self._is_likely_person_name(name_lower):
                # Check if it's actually a pet
                if self._is_likely_pet_name(name_lower):
                    return QueryType.PET
                return QueryType.PERSON

            # Check for aggregate queries
            if any(word in name_lower for word in ["who", "anyone", "everyone", "all"]):
                return QueryType.AGGREGATE

        # Area-based query
        if area and not name_contains:
            return QueryType.AREA

        # State-based query
        if state and not name_contains and not area:
            return QueryType.STATE

        # Device type query
        if domain and not name_contains:
            return QueryType.DEVICE

        return QueryType.GENERAL

    def _is_likely_person_name(self, name: str) -> bool:
        """Check if the name is likely a person's name."""
        # Check if we have a person entity with this name
        person_entity = f"person.{name}"
        if self.hass.states.get(person_entity):
            return True

        # Check for device_tracker patterns (often indicates a person)
        for entity_id in self.hass.states.async_entity_ids():
            if f"device_tracker" in entity_id and name in entity_id.lower():
                return True

        # Check for common person-related patterns
        for entity_id in self.hass.states.async_entity_ids():
            if re.match(rf"input_text\.room_{name}", entity_id):
                return True
            if re.match(rf"sensor\..*{name}.*_ble_area", entity_id):
                return True

        return False

    def _is_likely_pet_name(self, name: str) -> bool:
        """Check if the name is likely a pet's name.

        Detection logic:
        - Pets typically have tracking entities but no device_tracker
        - Pets often have 'inside' or room tracking but no person entity
        - Generic keywords like 'cat', 'dog', 'pet' indicate pet queries
        """
        name_lower = name.lower()

        # Check for generic pet keywords
        if name_lower in ["cat", "dog", "pet", "puppy", "kitten", "bird", "fish"]:
            return True

        # Check if we have pet-specific entities without person entities
        has_pet_entities = False
        has_person_entity = self.hass.states.get(f"person.{name_lower}") is not None
        has_device_tracker = False

        # Look for pet-specific patterns
        for entity_id in self.hass.states.async_entity_ids():
            entity_lower = entity_id.lower()
            if name_lower in entity_lower:
                # Pet indicators
                if "inside" in entity_lower or "room" in entity_lower:
                    has_pet_entities = True
                # Person indicators
                if "device_tracker" in entity_lower:
                    has_device_tracker = True

        # It's likely a pet if:
        # - Has tracking entities but no device_tracker (pets don't have phones)
        # - Has no person entity
        return has_pet_entities and not has_person_entity and not has_device_tracker

    async def _discover_person_entities(
        self, name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities related to a person with smart grouping."""
        name_lower = name.lower()
        results = {
            "query": name,
            "query_type": "person",
            "primary_entities": [],
            "related_entities": {
                "device_tracking": [],
                "ble_tracking": [],
                "room_tracking": [],
                "presence": [],
                "other": []
            }
        }

        # Find all entities related to this person
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            entity_id_lower = entity_id.lower()

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Check against patterns
            matched = False
            for pattern, category, description in EntityPattern.PERSON_PATTERNS:
                pattern_regex = pattern.replace("{name}", name_lower)
                if re.match(pattern_regex, entity_id_lower):
                    entity_info = self._create_entity_info(state_obj, description)

                    if category == "primary":
                        results["primary_entities"].append(entity_info)
                    else:
                        results["related_entities"][category].append(entity_info)
                    matched = True
                    break

            # Also check friendly name and aliases
            if not matched:
                # Check friendly name
                if name_lower in state_obj.name.lower():
                    entity_info = self._create_entity_info(state_obj)
                    results["related_entities"]["other"].append(entity_info)
                else:
                    # Check entity aliases
                    entity_registry = er.async_get(self.hass)
                    entity_entry = entity_registry.async_get(entity_id)
                    if entity_entry and entity_entry.aliases:
                        for alias in entity_entry.aliases:
                            if name_lower in alias.lower():
                                entity_info = self._create_entity_info(state_obj)
                                entity_info["matched_alias"] = alias
                                results["related_entities"]["other"].append(entity_info)
                                break

        # Format results for return
        return self._format_smart_results(results, limit)

    async def _discover_pet_entities(
        self, name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities related to a pet with smart grouping."""
        name_lower = name.lower()
        results = {
            "query": name,
            "query_type": "pet",
            "primary_entities": [],
            "related_entities": {
                "ble_tracking": [],
                "room_tracking": [],
                "presence": [],
                "other": []
            }
        }

        # Find all entities related to this pet
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            entity_id_lower = entity_id.lower()

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Check against patterns
            matched = False
            for pattern, category, description in EntityPattern.PET_PATTERNS:
                pattern_regex = pattern.replace("{name}", name_lower)
                if re.match(pattern_regex, entity_id_lower):
                    entity_info = self._create_entity_info(state_obj, description)

                    if category == "primary":
                        results["primary_entities"].append(entity_info)
                    else:
                        results["related_entities"][category].append(entity_info)
                    matched = True
                    break

            # Also check friendly name and aliases
            if not matched:
                # Check friendly name
                if name_lower in state_obj.name.lower():
                    entity_info = self._create_entity_info(state_obj)
                    results["related_entities"]["other"].append(entity_info)
                else:
                    # Check entity aliases
                    entity_registry = er.async_get(self.hass)
                    entity_entry = entity_registry.async_get(entity_id)
                    if entity_entry and entity_entry.aliases:
                        for alias in entity_entry.aliases:
                            if name_lower in alias.lower():
                                entity_info = self._create_entity_info(state_obj)
                                entity_info["matched_alias"] = alias
                                results["related_entities"]["other"].append(entity_info)
                                break

        # Format results for return
        return self._format_smart_results(results, limit)

    async def _discover_aggregate_entities(
        self, domain: Optional[str], state: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities for aggregate queries like 'who is home'."""
        results = {
            "query": f"{'all ' + domain if domain else 'entities'} {'with state ' + state if state else ''}",
            "query_type": "aggregate",
            "primary_entities": [],
            "related_entities": {}
        }

        # Focus on person and presence entities for "who is home" type queries
        target_domains = ["person", "device_tracker", "binary_sensor"]
        if domain:
            target_domains = [domain]

        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            entity_domain = entity_id.split(".")[0]

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Filter by domain
            if entity_domain not in target_domains:
                continue

            # Filter by state if specified
            if state and state_obj.state.lower() != state.lower():
                continue

            # Add to appropriate category
            entity_info = self._create_entity_info(state_obj)

            if entity_domain == "person":
                results["primary_entities"].append(entity_info)
            else:
                category = entity_domain.replace("_", " ").title()
                results["related_entities"].setdefault(category, []).append(entity_info)

        return self._format_smart_results(results, limit)

    async def _discover_area_entities(
        self, area: str, domain: Optional[str], state: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities in a specific area with smart grouping."""
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        # Handle if area is passed as list (defensive)
        if isinstance(area, list):
            if not area:
                return []
            area = area[0]  # Use first area

        # Find area by name
        area_id = None
        area_entry = None
        for a in area_registry.areas.values():
            if a.name.lower() == area.lower():
                area_entry = a
                area_id = a.id
                break

        if not area_id:
            return []

        results = {
            "query": f"{area} area",
            "query_type": "area",
            "area_name": area_entry.name,
            "primary_entities": [],
            "related_entities": {}
        }

        # Group entities by domain
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Get entity's area
            entity_entry = entity_registry.async_get(entity_id)
            entity_area_id = None

            if entity_entry:
                if entity_entry.area_id:
                    entity_area_id = entity_entry.area_id
                elif entity_entry.device_id:
                    device_entry = device_registry.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id:
                        entity_area_id = device_entry.area_id

            # Check if entity is in the target area
            if entity_area_id != area_id:
                continue

            # Apply domain filter
            entity_domain = entity_id.split(".")[0]
            if domain and entity_domain != domain:
                continue

            # Apply state filter
            if state and state_obj.state.lower() != state.lower():
                continue

            # Add to results grouped by domain
            entity_info = self._create_entity_info(state_obj)
            domain_key = entity_domain.replace("_", " ").title()
            results["related_entities"].setdefault(domain_key, []).append(entity_info)

        return self._format_smart_results(results, limit)

    async def _discover_general_entities(
        self,
        entity_type: Optional[str],
        area: Optional[str],
        domain: Optional[str],
        state: Optional[str],
        name_contains: Optional[str],
        limit: int,
        device_class: Optional[Union[str, List[str]]] = None,
        name_pattern: Optional[str] = None,
        inferred_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """General entity discovery with enhanced search."""
        # Handle inferred_type by looking up pattern from index
        if inferred_type and not name_pattern:
            from .const import DOMAIN
            index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")
            if index_manager:
                index = await index_manager.get_index()
                inferred_types = index.get("inferred_types", {})
                if inferred_type in inferred_types:
                    pattern_data = inferred_types[inferred_type]
                    name_pattern = pattern_data.get("pattern")
                    _LOGGER.debug("Inferred type '%s' mapped to pattern '%s'",
                                 inferred_type, name_pattern)
                else:
                    _LOGGER.warning("Inferred type '%s' not found in index", inferred_type)
            else:
                _LOGGER.warning("Index manager not available for inferred_type lookup")

        entities = []
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        # Get area ID if area name provided
        area_id = None
        if area:
            for a in area_registry.areas.values():
                if a.name.lower() == area.lower():
                    area_id = a.id
                    break

        # Limit to prevent excessive results
        limit = min(limit, MAX_ENTITIES_PER_DISCOVERY)

        # Search through all entities
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id

            # Check if entity is exposed to conversation
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Apply filters
            entity_domain = entity_id.split(".")[0]

            # Domain filter
            if domain and entity_domain != domain:
                continue

            # Entity type filter (same as domain for backward compatibility)
            if entity_type and entity_domain != entity_type:
                continue

            # State filter
            if state and state_obj.state.lower() != state.lower():
                continue

            # Get entity registry entry for aliases and area info
            entity_entry = entity_registry.async_get(entity_id)

            # Enhanced name search - search entity_id, friendly name, AND aliases
            if name_contains:
                search_term = name_contains.lower()

                # Check entity_id and friendly names
                found_match = (
                    search_term in entity_id.lower() or
                    search_term in state_obj.name.lower() or
                    search_term in state_obj.attributes.get("friendly_name", "").lower()
                )

                # Also check entity aliases if available
                if not found_match and entity_entry and entity_entry.aliases:
                    for alias in entity_entry.aliases:
                        if search_term in alias.lower():
                            found_match = True
                            break

                if not found_match:
                    continue

            # Device class filter
            if device_class:
                entity_device_class = state_obj.attributes.get('device_class')

                # Convert single device_class to list for uniform handling
                device_class_list = [device_class] if isinstance(device_class, str) else device_class

                # Check if entity's device_class matches any in the list (OR logic)
                if entity_device_class not in device_class_list:
                    continue

            # Name pattern filter (wildcard matching)
            if name_pattern:
                if not fnmatch.fnmatch(entity_id, name_pattern):
                    continue

            # Get area information
            entity_area_id = None
            entity_area_name = None

            if entity_entry:
                if entity_entry.area_id:
                    entity_area_id = entity_entry.area_id
                elif entity_entry.device_id:
                    device_entry = device_registry.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id:
                        entity_area_id = device_entry.area_id

                if entity_area_id:
                    area_entry = area_registry.async_get_area(entity_area_id)
                    if area_entry:
                        entity_area_name = area_entry.name

            # Apply area filter
            if area_id and entity_area_id != area_id:
                continue

            # Create entity info
            entity_info = self._create_entity_info(state_obj)
            entity_info["area"] = entity_area_name
            entity_info["area_id"] = entity_area_id

            entities.append(entity_info)

            # Stop if we hit the limit
            if len(entities) >= limit:
                break

        _LOGGER.debug(
            f"General discovery found {len(entities)} entities with filters: "
            f"type={entity_type}, area={area}, domain={domain}, "
            f"state={state}, name_contains={name_contains}"
        )

        return entities

    def _create_entity_info(
        self, state_obj: Any, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create entity information dictionary."""
        entity_info = {
            "entity_id": state_obj.entity_id,
            "name": state_obj.name,
            "domain": state_obj.domain,
            "state": state_obj.state,
        }

        if description:
            entity_info["type"] = description

        # Add area information if available
        entity_registry = er.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        entity_entry = entity_registry.async_get(state_obj.entity_id)
        if entity_entry:
            # Add area information
            area_id = entity_entry.area_id
            if not area_id and entity_entry.device_id:
                device_entry = device_registry.async_get(entity_entry.device_id)
                if device_entry:
                    area_id = device_entry.area_id

            if area_id:
                area_entry = area_registry.async_get_area(area_id)
                if area_entry:
                    entity_info["area"] = area_entry.name

            # Add aliases if present
            if entity_entry.aliases:
                entity_info["aliases"] = list(entity_entry.aliases)

        # Add useful attributes
        if state_obj.attributes:
            useful_attrs = {}
            for attr in ["brightness", "temperature", "humidity", "unit_of_measurement",
                        "device_class", "friendly_name"]:
                if attr in state_obj.attributes:
                    useful_attrs[attr] = state_obj.attributes[attr]
            if useful_attrs:
                entity_info["attributes"] = useful_attrs

        return entity_info

    def _format_smart_results(
        self, results: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Format smart discovery results for the LLM."""
        # Flatten the results into a list while preserving structure information
        formatted = []

        # Add primary entities first
        for entity in results.get("primary_entities", [])[:limit]:
            entity["relationship"] = "primary"
            formatted.append(entity)

        # Add related entities with their categories
        remaining_limit = limit - len(formatted)
        for category, entities in results.get("related_entities", {}).items():
            if remaining_limit <= 0:
                break
            for entity in entities[:remaining_limit]:
                entity["relationship"] = category
                formatted.append(entity)
                remaining_limit -= 1
                if remaining_limit <= 0:
                    break

        # Add metadata about the query
        if formatted:
            # Add a summary as the first item
            summary = {
                "entity_id": "_summary",
                "query_type": results.get("query_type", "general"),
                "query": results.get("query", ""),
                "total_found": len(results.get("primary_entities", [])) +
                              sum(len(v) for v in results.get("related_entities", {}).values()),
                "primary_count": len(results.get("primary_entities", [])),
                "related_count": sum(len(v) for v in results.get("related_entities", {}).values()),
            }
            formatted.insert(0, summary)

        return formatted

    @staticmethod
    def _serialize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize entity attributes, converting datetime objects to ISO strings."""
        serialized = {}
        for key, value in attributes.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = EntityDiscovery._serialize_attributes(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    item.isoformat() if isinstance(item, datetime) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    # Legacy methods for backward compatibility
    async def get_entity_details(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Get detailed information about specific entities."""
        details = {}
        entity_registry = er.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        for entity_id in entity_ids:
            # Check if entity is exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                details[entity_id] = {"error": "Entity not exposed to conversation"}
                continue

            state_obj = self.hass.states.get(entity_id)
            if not state_obj:
                details[entity_id] = {"error": "Entity not found"}
                continue

            entity_entry = entity_registry.async_get(entity_id)
            area_name = None
            device_name = None

            if entity_entry:
                # Get area info
                area_id = entity_entry.area_id
                if not area_id and entity_entry.device_id:
                    device_entry = device_registry.async_get(entity_entry.device_id)
                    if device_entry:
                        area_id = device_entry.area_id
                        device_name = device_entry.name

                if area_id:
                    area_entry = area_registry.async_get_area(area_id)
                    if area_entry:
                        area_name = area_entry.name

            entity_details = {
                "entity_id": entity_id,
                "name": state_obj.name,
                "domain": state_obj.domain,
                "state": state_obj.state,
                "attributes": self._serialize_attributes(dict(state_obj.attributes)),
                "area": area_name,
                "device": device_name,
                "last_changed": state_obj.last_changed.isoformat(),
                "last_updated": state_obj.last_updated.isoformat(),
            }

            if entity_entry:
                entity_details.update({
                    "unique_id": entity_entry.unique_id,
                    "entity_category": entity_entry.entity_category,
                    "disabled": entity_entry.disabled_by is not None,
                })

            details[entity_id] = entity_details

        return details

    async def list_areas(self) -> List[Dict[str, Any]]:
        """List all areas with entity counts."""
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        areas = []
        for area_entry in area_registry.areas.values():
            # Count entities in this area
            entity_count = 0

            # Count entities directly assigned to area
            for entity_entry in entity_registry.entities.values():
                if entity_entry.area_id == area_entry.id:
                    if async_should_expose(self.hass, "conversation", entity_entry.entity_id):
                        entity_count += 1

            # Count entities via devices in this area
            for device_entry in device_registry.devices.values():
                if device_entry.area_id == area_entry.id:
                    for entity_entry in entity_registry.entities.values():
                        if (entity_entry.device_id == device_entry.id
                            and not entity_entry.area_id  # Not already counted
                            and async_should_expose(self.hass, "conversation", entity_entry.entity_id)):
                            entity_count += 1

            areas.append({
                "id": area_entry.id,
                "name": area_entry.name,
                "entity_count": entity_count
            })

        # Sort by name
        areas.sort(key=lambda x: x["name"])
        return areas

    async def list_domains(self) -> List[Dict[str, Any]]:
        """List all domains with entity counts."""
        domain_counts = {}

        for state_obj in self.hass.states.async_all():
            if async_should_expose(self.hass, "conversation", state_obj.entity_id):
                domain = state_obj.domain
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        domains = [
            {"domain": domain, "count": count}
            for domain, count in domain_counts.items()
        ]

        # Sort by count (descending) then by name
        domains.sort(key=lambda x: (-x["count"], x["domain"]))
        return domains

    async def get_entities_by_area(self, area_id: str) -> List[Dict[str, Any]]:
        """Get all entities in a specific area by area ID."""
        entities = []
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        for state in self.hass.states.async_all():
            entity_entry = entity_registry.async_get(state.entity_id)
            entity_area_id = None

            if entity_entry:
                if entity_entry.area_id:
                    entity_area_id = entity_entry.area_id
                elif entity_entry.device_id:
                    device_entry = device_registry.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id:
                        entity_area_id = device_entry.area_id

            if entity_area_id == area_id:
                # Only include entities that should be exposed
                if async_should_expose(self.hass, "conversation", state.entity_id):
                    entities.append({
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                        "domain": state.entity_id.split(".")[0]
                    })

        _LOGGER.debug(f"Found {len(entities)} entities in area '{area_id}'")
        return entities


# For backward compatibility, keep the old class name as an alias
EntityDiscovery = SmartDiscovery