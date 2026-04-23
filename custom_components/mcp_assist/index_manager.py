"""Smart Index Manager for context-aware entity discovery.

This module generates and maintains a lightweight index of Home Assistant's
system structure, enabling LLMs to make smart, targeted entity queries without
requiring full context dumps.

The index includes:
- Areas with entity counts
- Floors with area/entity counts
- Labels with entity coverage
- Devices summary
- Domains with counts
- Device classes (grouped by domain) with counts
- Inferred entity types (via LLM gap-filling)
- People, pets, calendars, zones, automations, scripts, input helpers
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime

from homeassistant.core import HomeAssistant, callback, Event, Context
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    device_registry as dr,
)
from homeassistant.helpers.entity_registry import EVENT_ENTITY_REGISTRY_UPDATED
from homeassistant.components.homeassistant import async_should_expose

try:
    from homeassistant.helpers import floor_registry as fr
except ImportError:  # pragma: no cover - older Home Assistant versions
    fr = None

try:
    from homeassistant.helpers import label_registry as lr
except ImportError:  # pragma: no cover - older Home Assistant versions
    lr = None

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
        self._gap_filling_in_progress = False  # Re-entrancy guard for gap-filling
        self._first_index_generated = False  # Skip gap-filling on first index (startup)

    async def start(self) -> None:
        """Start index manager and set up event listeners."""
        _LOGGER.info("Starting Smart Index Manager")

        @callback
        def registry_changed(event: Event) -> None:
            """Schedule index refresh on registry changes."""
            _LOGGER.debug("%s changed, scheduling index refresh", event.event_type)
            self._schedule_refresh()

        registry_events = [
            EVENT_ENTITY_REGISTRY_UPDATED,
            ar.EVENT_AREA_REGISTRY_UPDATED,
            dr.EVENT_DEVICE_REGISTRY_UPDATED,
        ]
        if fr is not None:
            registry_events.append(fr.EVENT_FLOOR_REGISTRY_UPDATED)
        if lr is not None:
            registry_events.append(lr.EVENT_LABEL_REGISTRY_UPDATED)

        for event_type in registry_events:
            self.hass.bus.async_listen(event_type, registry_changed)

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
        floors, labels, people, pets, calendars, zones, automations, scripts,
        and input helpers.
        """
        # Query all index components in parallel for speed
        results = await asyncio.gather(
            self._get_areas(),
            self._get_floors(),
            self._get_labels(),
            self._get_devices(),
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
            floors,
            labels,
            devices,
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
            # Skip gap-filling on first index generation (LLM may not be ready at startup)
            if not self._first_index_generated:
                _LOGGER.info("Skipping gap-filling on first index generation (startup)")
            # Check if gap-filling is already in progress (prevent recursion)
            elif self._gap_filling_in_progress:
                _LOGGER.debug("Gap-filling already in progress, skipping to prevent recursion")
            else:
                # Check if gap-filling is enabled in config
                gap_filling_enabled = await self._is_gap_filling_enabled()
                if gap_filling_enabled:
                    inferred_types = await self._infer_entity_types(entities_without_device_class)
                else:
                    _LOGGER.debug("Gap-filling disabled in config, skipping entity type inference")

        # Mark first index as generated
        if not self._first_index_generated:
            self._first_index_generated = True

        # Build index dict
        index = {
            "areas": areas if not isinstance(areas, Exception) else [],
            "floors": floors if not isinstance(floors, Exception) else [],
            "labels": labels if not isinstance(labels, Exception) else [],
            "devices": devices if not isinstance(devices, Exception) else {},
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

    def _get_label_names(self, label_ids: Set[str], label_registry: Any) -> List[str]:
        """Resolve label IDs to label names."""
        if not label_ids:
            return []

        label_names = set()
        for label_id in label_ids:
            label_name = label_id
            if label_registry is not None:
                label_entry = label_registry.async_get_label(label_id)
                if label_entry:
                    label_name = label_entry.name
            label_names.add(label_name)

        return sorted(label_names, key=str.casefold)

    def _get_entity_aliases(self, entity_entry: Any) -> List[str]:
        """Resolve aliases for an entity registry entry."""
        if entity_entry is None:
            return []

        get_entity_aliases = getattr(er, "async_get_entity_aliases", None)
        if get_entity_aliases is not None:
            try:
                aliases = get_entity_aliases(self.hass, entity_entry)
            except Exception:  # pragma: no cover - defensive fallback
                _LOGGER.debug("Falling back to raw entity aliases in index", exc_info=True)
                aliases = getattr(entity_entry, "aliases", []) or []
        else:
            aliases = getattr(entity_entry, "aliases", []) or []

        return sorted(
            {alias for alias in aliases if isinstance(alias, str)},
            key=str.casefold,
        )

    def _build_named_index_entry(
        self,
        name: str,
        aliases: List[str],
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build a compact index entry for a named object and its aliases."""
        normalized_name = name.casefold()
        filtered_aliases = sorted(
            {
                alias
                for alias in aliases
                if alias and alias.casefold() != normalized_name
            },
            key=str.casefold,
        )

        item: Dict[str, Any] = {"name": name}
        if filtered_aliases:
            item["aliases"] = filtered_aliases

        for key, value in extra.items():
            if value not in (None, "", [], {}):
                item[key] = value

        return item

    async def _get_areas(self) -> List[Dict[str, Any]]:
        """Get all areas with entity counts."""
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)
        device_reg = dr.async_get(self.hass)
        floor_reg = fr.async_get(self.hass) if fr else None
        label_reg = lr.async_get(self.hass) if lr else None

        # Count entities per area
        area_counts = defaultdict(int)
        exposed_device_ids = set()
        for entity in entity_reg.entities.values():
            if async_should_expose(self.hass, "conversation", entity.entity_id):
                if entity.device_id:
                    exposed_device_ids.add(entity.device_id)
                if entity.area_id:
                    area_counts[entity.area_id] += 1
                # Also check device area
                elif entity.device_id:
                    device = device_reg.async_get(entity.device_id)
                    if device and device.area_id:
                        area_counts[device.area_id] += 1

        area_device_counts = defaultdict(int)
        for device_id in exposed_device_ids:
            device_entry = device_reg.async_get(device_id)
            if device_entry and device_entry.area_id:
                area_device_counts[device_entry.area_id] += 1

        # Build area list with counts
        areas = []
        for area in area_reg.async_list_areas():
            count = area_counts.get(area.id, 0)
            if count > 0:  # Only include areas with entities
                floor_id = getattr(area, "floor_id", None)
                floor_name = None
                floor_aliases: List[str] = []
                if floor_id and floor_reg is not None:
                    floor_entry = floor_reg.async_get_floor(floor_id)
                    if floor_entry:
                        floor_name = floor_entry.name
                        floor_aliases = sorted(
                            getattr(floor_entry, "aliases", set()) or set(),
                            key=str.casefold,
                        )

                label_ids = set(getattr(area, "labels", set()) or set())
                areas.append({
                    "name": area.name,
                    "aliases": sorted(getattr(area, "aliases", set()) or set(), key=str.casefold),
                    "entity_count": count,
                    "device_count": area_device_counts.get(area.id, 0),
                    "floor": floor_name,
                    "floor_id": floor_id,
                    "floor_aliases": floor_aliases,
                    "labels": self._get_label_names(label_ids, label_reg),
                    "label_ids": sorted(label_ids),
                })

        # Sort by name
        areas.sort(key=lambda x: x["name"])

        _LOGGER.debug("Found %d areas with entities", len(areas))
        return areas

    async def _get_floors(self) -> List[Dict[str, Any]]:
        """Get all floors with area/entity coverage."""
        if fr is None:
            return []

        floor_reg = fr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)
        device_reg = dr.async_get(self.hass)

        floor_area_names = defaultdict(list)
        area_floor_ids = {}
        for area_entry in area_reg.async_list_areas():
            if getattr(area_entry, "floor_id", None):
                floor_area_names[area_entry.floor_id].append(area_entry.name)
                area_floor_ids[area_entry.id] = area_entry.floor_id

        floor_entities = defaultdict(set)
        exposed_device_ids = set()
        for entity in entity_reg.entities.values():
            if not async_should_expose(self.hass, "conversation", entity.entity_id):
                continue

            area_id = entity.area_id
            if not area_id and entity.device_id:
                device_entry = device_reg.async_get(entity.device_id)
                if device_entry:
                    area_id = device_entry.area_id
                    exposed_device_ids.add(device_entry.id)
            elif entity.device_id:
                exposed_device_ids.add(entity.device_id)

            if not area_id:
                continue

            area_entry = area_reg.async_get_area(area_id)
            floor_id = getattr(area_entry, "floor_id", None) if area_entry else None
            if floor_id:
                floor_entities[floor_id].add(entity.entity_id)

        floor_device_counts = defaultdict(int)
        for device_id in exposed_device_ids:
            device_entry = device_reg.async_get(device_id)
            if not device_entry or not device_entry.area_id:
                continue

            floor_id = area_floor_ids.get(device_entry.area_id)
            if floor_id:
                floor_device_counts[floor_id] += 1

        floors = []
        for floor_entry in floor_reg.async_list_floors():
            areas = sorted(floor_area_names.get(floor_entry.floor_id, []), key=str.casefold)
            floors.append({
                "name": floor_entry.name,
                "aliases": sorted(getattr(floor_entry, "aliases", set()) or set(), key=str.casefold),
                "area_count": len(areas),
                "entity_count": len(floor_entities.get(floor_entry.floor_id, set())),
                "device_count": floor_device_counts.get(floor_entry.floor_id, 0),
                "areas": areas,
            })

        floors.sort(key=lambda x: x["name"])
        return floors

    async def _get_labels(self) -> List[Dict[str, Any]]:
        """Get all labels with area/device/entity coverage."""
        if lr is None:
            return []

        label_reg = lr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)
        device_reg = dr.async_get(self.hass)

        label_areas = defaultdict(set)
        for area_entry in area_reg.async_list_areas():
            for label_id in getattr(area_entry, "labels", set()) or set():
                label_areas[label_id].add(area_entry.id)

        label_devices = defaultdict(set)
        for device_entry in device_reg.devices.values():
            for label_id in getattr(device_entry, "labels", set()) or set():
                label_devices[label_id].add(device_entry.id)

        label_entities = defaultdict(set)
        for entity_entry in entity_reg.entities.values():
            if not async_should_expose(self.hass, "conversation", entity_entry.entity_id):
                continue

            device_entry = (
                device_reg.async_get(entity_entry.device_id)
                if entity_entry.device_id
                else None
            )
            area_id = entity_entry.area_id
            if not area_id and device_entry:
                area_id = device_entry.area_id
            area_entry = area_reg.async_get_area(area_id) if area_id else None

            combined_labels = set(getattr(entity_entry, "labels", set()) or set())
            if device_entry:
                combined_labels.update(getattr(device_entry, "labels", set()) or set())
            if area_entry:
                combined_labels.update(getattr(area_entry, "labels", set()) or set())

            for label_id in combined_labels:
                label_entities[label_id].add(entity_entry.entity_id)

        labels = []
        for label_entry in label_reg.async_list_labels():
            labels.append({
                "name": label_entry.name,
                "entity_count": len(label_entities.get(label_entry.label_id, set())),
                "area_count": len(label_areas.get(label_entry.label_id, set())),
                "device_count": len(label_devices.get(label_entry.label_id, set())),
            })

        labels.sort(key=lambda x: x["name"])
        return labels

    async def _get_devices(self) -> Dict[str, Any]:
        """Get a compact device summary for the index."""
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)
        device_reg = dr.async_get(self.hass)
        floor_reg = fr.async_get(self.hass) if fr else None

        device_domains = defaultdict(set)
        for entity in entity_reg.entities.values():
            if not async_should_expose(self.hass, "conversation", entity.entity_id):
                continue
            if entity.device_id and device_reg.async_get(entity.device_id):
                device_domains[entity.device_id].add(entity.entity_id.split(".")[0])

        by_domain = defaultdict(int)
        by_area = defaultdict(int)
        by_floor = defaultdict(int)
        manufacturers = defaultdict(int)

        for device_entry in device_reg.devices.values():
            if device_entry.id not in device_domains:
                continue

            manufacturers[getattr(device_entry, "manufacturer", None) or "Unknown"] += 1

            for domain in device_domains[device_entry.id]:
                by_domain[domain] += 1

            if device_entry.area_id:
                area_entry = area_reg.async_get_area(device_entry.area_id)
                if area_entry:
                    by_area[area_entry.name] += 1
                    if floor_reg is not None and getattr(area_entry, "floor_id", None):
                        floor_entry = floor_reg.async_get_floor(area_entry.floor_id)
                        if floor_entry:
                            by_floor[floor_entry.name] += 1

        return {
            "count": len(device_domains),
            "by_domain": dict(sorted(by_domain.items(), key=lambda item: (-item[1], item[0]))),
            "by_area": dict(sorted(by_area.items(), key=lambda item: (-item[1], item[0]))),
            "by_floor": dict(sorted(by_floor.items(), key=lambda item: (-item[1], item[0]))),
            "manufacturers": dict(sorted(manufacturers.items(), key=lambda item: (-item[1], item[0]))),
        }

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

                    people.append(
                        self._build_named_index_entry(
                            name,
                            self._get_entity_aliases(entity),
                        )
                    )

        people.sort(key=lambda x: x["name"].casefold())
        _LOGGER.debug("Found %d people", len(people))
        return people

    async def _get_calendars(self) -> List[Dict[str, Any]]:
        """Get all calendar entities."""
        entity_reg = er.async_get(self.hass)

        calendars = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('calendar.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                    calendars.append(
                        self._build_named_index_entry(
                            name,
                            self._get_entity_aliases(entity),
                        )
                    )

        calendars.sort(key=lambda x: x["name"].casefold())
        _LOGGER.debug("Found %d calendars", len(calendars))
        return calendars

    async def _get_zones(self) -> List[Dict[str, Any]]:
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
                        zones.append(
                            self._build_named_index_entry(
                                name,
                                self._get_entity_aliases(entity),
                            )
                        )

        zones.sort(key=lambda x: x["name"].casefold())
        _LOGGER.debug("Found %d zones", len(zones))
        return zones

    async def _get_automations(self) -> List[Dict[str, Any]]:
        """Get all automation entities."""
        entity_reg = er.async_get(self.hass)

        automations = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('automation.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else entity.entity_id.split('.')[1]
                    automations.append(
                        self._build_named_index_entry(
                            name,
                            self._get_entity_aliases(entity),
                        )
                    )

        automations.sort(key=lambda x: x["name"].casefold())
        _LOGGER.debug("Found %d automations", len(automations))
        return automations

    async def _get_scripts(self) -> List[Dict[str, Any]]:
        """Get script entities with field definitions."""
        entity_reg = er.async_get(self.hass)

        # Access script component to get field definitions
        script_component = self.hass.data.get("script")

        scripts = []
        for entity in entity_reg.entities.values():
            if entity.entity_id.startswith('script.'):
                if async_should_expose(self.hass, "conversation", entity.entity_id):
                    script_id = entity.entity_id.split('.')[1]
                    state_obj = self.hass.states.get(entity.entity_id)
                    name = state_obj.name if state_obj else script_id
                    script_info = self._build_named_index_entry(
                        name,
                        self._get_entity_aliases(entity),
                        id=script_id,
                    )

                    # Get fields from the script entity
                    fields = None
                    if script_component:
                        script_entity = script_component.get_entity(entity.entity_id)
                        if script_entity and hasattr(script_entity, 'fields') and script_entity.fields:
                            fields = script_entity.fields

                    if fields:
                        script_info["fields"] = {}
                        # Extract field descriptions
                        for field_name, field_data in fields.items():
                            desc = field_data.get('description', '') if isinstance(field_data, dict) else ''
                            script_info["fields"][field_name] = {"description": desc} if desc else {}

                    scripts.append(script_info)

        scripts.sort(key=lambda x: x["name"].casefold())
        _LOGGER.debug("Found %d scripts (%d with fields)",
                     len(scripts),
                     sum(1 for s in scripts if s.get("fields")))
        return scripts

    async def _get_input_helpers(self) -> Dict[str, List[Dict[str, Any]]]:
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
                        helpers[input_type].append(
                            self._build_named_index_entry(
                                name,
                                self._get_entity_aliases(entity),
                            )
                        )
                        break

        # Sort each list and convert to plural keys
        result = {}
        for input_type, names in helpers.items():
            if names:  # Only include types with entities
                names.sort(key=lambda x: x["name"].casefold())
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

        # Set re-entrancy guard to prevent recursion
        # (gap-filling calls agent → agent builds prompt with index → index calls gap-filling)
        self._gap_filling_in_progress = True

        try:
            # Call LLM via conversation agent
            # TODO: Future improvement - use dedicated async_call_llm_direct() method
            # instead of full async_process() to avoid conversation context overhead
            inferred = await self._call_llm_for_inference(prompt)
            _LOGGER.info("LLM gap-filling completed: found %d inferred types", len(inferred))
            return inferred
        except Exception as err:
            _LOGGER.debug("LLM gap-filling failed: %s. Index will not include inferred types.", err)
            return {}
        finally:
            # Always clear the flag, even if exception occurred
            self._gap_filling_in_progress = False

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

        # Call the agent (uses configured max_tokens - user should set 2000+ for gap-filling)
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
        except json.JSONDecodeError as e:
            # JSON parsing failed - try to repair incomplete JSON
            _LOGGER.warning("Gap-filling JSON parsing failed at position %d: %s", e.pos, e.msg)
            _LOGGER.debug("Response text (first 500 chars): %s", response_text[:500])

            # Attempt to repair by closing incomplete JSON
            repaired_text = response_text
            # Count open braces
            open_braces = repaired_text.count('{')
            close_braces = repaired_text.count('}')

            if open_braces > close_braces:
                _LOGGER.info("Attempting to repair incomplete JSON (adding %d closing braces)", open_braces - close_braces)
                repaired_text += '}' * (open_braces - close_braces)

                try:
                    parsed = json.loads(repaired_text)
                    _LOGGER.info("✅ JSON repair successful - recovered %d categories", len(parsed))
                    return parsed
                except json.JSONDecodeError:
                    _LOGGER.error("JSON repair failed, gap-filling will be skipped this cycle")

            raise ValueError(f"Invalid JSON in LLM response: {e.msg}")

        _LOGGER.debug("Successfully parsed %d inferred types from LLM", len(parsed))
        return parsed

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
