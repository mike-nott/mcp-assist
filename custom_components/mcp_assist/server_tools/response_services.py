"""Response-service MCP server tools."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time
import json
import logging
from typing import Any, Dict, List

from homeassistant.core import SupportsResponse
from homeassistant.helpers import service as service_helper
from homeassistant.util import dt as dt_util

from ..custom_tools.music_assistant import summarize_music_assistant_response
from ..domain_registry import get_domain_info

_LOGGER = logging.getLogger(__name__)


class ResponseServicesMixin:
    """Response-service MCP server tool implementations."""

    async def tool_list_response_services(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List dynamically available HA services that support response data."""
        domain_filter = str(args.get("domain") or "").strip().casefold()
        query = str(args.get("query") or "").strip().casefold()
        limit = self._coerce_int_arg(
            args.get("limit"), default=50, minimum=1, maximum=200
        )

        catalog = await self._get_response_service_catalog()
        rows: List[tuple[str, str, Dict[str, Any]]] = []

        for domain, services in catalog.items():
            if domain_filter and domain.casefold() != domain_filter:
                continue

            for service_name, description in services.items():
                haystacks = [
                    domain,
                    service_name,
                    str(description.get("name") or ""),
                    str(description.get("description") or ""),
                ]
                if query and not any(query in text.casefold() for text in haystacks):
                    continue

                rows.append((domain, service_name, description))

        rows.sort(key=lambda item: (item[0], item[1]))
        rows = rows[:limit]

        if not rows:
            filters = []
            if domain_filter:
                filters.append(f"domain='{domain_filter}'")
            if query:
                filters.append(f"query='{query}'")
            filter_text = f" for {', '.join(filters)}" if filters else ""
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No response-capable services found{filter_text}.",
                    }
                ]
            }

        text_parts = [
            f"Found {len(rows)} response-capable Home Assistant services:"
        ]
        for domain, service_name, description in rows:
            detail_parts = [
                f"response: {description.get('supports_response', 'optional')}"
            ]
            target_domains = self._extract_service_target_domains(description)
            if target_domains:
                detail_parts.append(f"target domains: {', '.join(target_domains)}")
            required_fields = self._get_required_service_fields(description)
            if required_fields:
                detail_parts.append(f"required: {', '.join(required_fields)}")
            field_names = self._get_service_field_names(description)
            if field_names:
                preview_fields = field_names[:6]
                detail_parts.append(
                    "fields: "
                    + ", ".join(preview_fields)
                    + ("..." if len(field_names) > len(preview_fields) else "")
                )
            if description.get("description"):
                detail_parts.append(str(description["description"]))

            text_parts.append(
                f"- {domain}.{service_name} ({'; '.join(detail_parts)})"
            )

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_call_service_with_response(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a native HA service that returns structured response data."""
        domain = str(args.get("domain") or "").strip().lower()
        service = str(args.get("service") or "").strip().lower()
        target = args.get("target")
        data = args.get("data", {}) or {}
        timeout = self._coerce_int_arg(
            args.get("timeout"), default=60, minimum=1, maximum=300
        )

        if not domain:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'domain'.",
                    }
                ]
            }

        if not service:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'service'.",
                    }
                ]
            }

        if target is None:
            target = {}
        elif not isinstance(target, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: 'target' must be an object with entity_id, area_id, floor_id, label_id, or device_id.",
                    }
                ]
            }

        if not isinstance(data, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: 'data' must be an object of service parameters.",
                    }
                ]
            }

        _LOGGER.info(
            "📖 Calling response service: %s.%s on %s with data %s",
            domain,
            service,
            target,
            data,
        )

        service_description, validation_error = await self._get_response_service_info(
            domain, service
        )
        if validation_error:
            return {
                "content": [
                    {"type": "text", "text": f"❌ Error: {validation_error}"}
                ]
            }

        resolved_target = {}
        try:
            if target:
                resolved_target = await self.resolve_target(target)
                resolved_target = self._restrict_resolved_target_for_service(
                    resolved_target,
                    service_description=service_description,
                )
        except Exception as err:
            error_msg = f"Failed to resolve target: {err}"
            _LOGGER.error(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        prepared_data = self._prepare_response_service_data(
            domain,
            service,
            data,
            resolved_target=resolved_target,
        )
        valid_params, validation_msg = self._validate_response_service_parameters(
            service_description, prepared_data
        )
        if not valid_params:
            return {
                "content": [{"type": "text", "text": f"❌ Error: {validation_msg}"}]
            }

        self.publish_progress(
            "tool_start",
            f"Calling response service: {domain}.{service}",
            tool="call_service_with_response",
            domain=domain,
            service=service,
        )

        try:
            service_data = {**resolved_target, **prepared_data}
            response = await asyncio.wait_for(
                self.hass.services.async_call(
                    domain=domain,
                    service=service,
                    service_data=service_data,
                    blocking=True,
                    return_response=True,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"Service-response call timed out after {timeout} seconds: "
                f"{domain}.{service}"
            )
            _LOGGER.error("❌ %s", error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}
        except Exception as err:
            error_msg = f"Service-response call failed: {err}"
            _LOGGER.exception("❌ %s", error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        self.publish_progress(
            "tool_complete",
            f"Response service completed: {domain}.{service}",
            tool="call_service_with_response",
            success=True,
        )

        serialized_response = self._serialize_service_response_value(response)
        result_text = self._format_service_response_result(
            domain,
            service,
            resolved_target,
            serialized_response,
            request_data=prepared_data,
        )
        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": result_text}]
        }
        if serialized_response is not None:
            result["response"] = serialized_response

        return result

    def _prepare_response_service_data(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any] | None,
        *,
        resolved_target: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Normalize service-response data and fill safe defaults."""
        prepared = dict(data or {})

        if (
            domain == "weather"
            and service in {"get_forecast", "get_forecasts"}
        ):
            requested_type = str(prepared.get("type") or "").strip().lower() or None
            supported_types = self._get_weather_target_forecast_types(
                resolved_target
            )

            chosen_type = requested_type
            if supported_types:
                if requested_type not in supported_types:
                    chosen_type = self._select_preferred_weather_forecast_type(
                        supported_types
                    )
                    if requested_type and chosen_type and chosen_type != requested_type:
                        entity_ids = self._normalize_target_values(
                            (resolved_target or {}).get("entity_id")
                        )
                        _LOGGER.info(
                            "Weather forecast type '%s' is not supported by target %s; "
                            "using '%s' instead from supported types %s",
                            requested_type,
                            entity_ids or "entities",
                            chosen_type,
                            supported_types,
                        )

            if chosen_type:
                prepared["type"] = chosen_type

        return prepared

    async def _get_response_service_catalog(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build a catalog of HA services that support response data."""
        descriptions = await service_helper.async_get_all_descriptions(self.hass)
        catalog: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for domain, services in self.hass.services.async_services().items():
            if self._get_domain_capability_error(domain):
                continue

            for service_name in services:
                supports_response = self.hass.services.supports_response(
                    domain, service_name
                )
                if supports_response == SupportsResponse.NONE:
                    continue

                description = dict(descriptions.get(domain, {}).get(service_name, {}))
                description["supports_response"] = (
                    "optional"
                    if supports_response == SupportsResponse.OPTIONAL
                    else "only"
                )
                catalog.setdefault(domain, {})[service_name] = description

        return catalog

    async def _get_response_service_info(
        self, domain: str, service: str
    ) -> tuple[Dict[str, Any], str | None]:
        """Get dynamic metadata for a response-capable HA service."""
        capability_error = self._get_domain_capability_error(domain)
        if capability_error:
            return {}, capability_error

        if not self.hass.services.has_service(domain, service):
            domain_info = get_domain_info(domain)
            if domain_info is None:
                return {}, (
                    f"Domain '{domain}' is not registered in Home Assistant right now. "
                    "Use list_response_services() to inspect available response-capable services."
                )
            return {}, (
                f"Service '{domain}.{service}' is not registered in Home Assistant right now. "
                "Use list_response_services() to inspect available response-capable services."
            )

        supports_response = self.hass.services.supports_response(domain, service)
        if supports_response == SupportsResponse.NONE:
            return {}, (
                f"Service '{domain}.{service}' does not support native response data. "
                "Use list_response_services() to find services that do."
            )

        catalog = await self._get_response_service_catalog()
        description = dict(catalog.get(domain, {}).get(service, {}))
        description["supports_response"] = (
            "optional"
            if supports_response == SupportsResponse.OPTIONAL
            else "only"
        )
        return description, None

    def _validate_response_service_parameters(
        self, description: Dict[str, Any], provided_params: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate required service parameters from HA's native service description."""
        required_fields = self._get_required_service_fields(description)
        missing = [field for field in required_fields if field not in provided_params]
        if missing:
            return (
                False,
                "Missing required parameters: " + ", ".join(missing),
            )

        return True, "Parameters valid"

    def _get_required_service_fields(self, description: Dict[str, Any]) -> List[str]:
        """Extract required field names from a service description."""
        fields = description.get("fields", {})
        if not isinstance(fields, dict):
            return []

        required_fields = []
        for field_name, metadata in fields.items():
            if isinstance(metadata, dict) and metadata.get("required") is True:
                required_fields.append(str(field_name))

        return sorted(required_fields)

    def _get_service_field_names(self, description: Dict[str, Any]) -> List[str]:
        """Extract service field names from a service description."""
        fields = description.get("fields", {})
        if not isinstance(fields, dict):
            return []

        return sorted(str(field_name) for field_name in fields.keys())

    def _extract_service_target_domains(
        self, description: Dict[str, Any]
    ) -> List[str]:
        """Extract allowed target entity domains from a service description."""
        target = description.get("target")
        if not isinstance(target, dict):
            return []

        entity_target = target.get("entity")
        if not isinstance(entity_target, dict):
            return []

        domain_value = entity_target.get("domain")
        if domain_value is None:
            return []
        if isinstance(domain_value, str):
            return [domain_value]
        if isinstance(domain_value, (list, tuple, set)):
            return [str(item) for item in domain_value if item]
        return [str(domain_value)]

    def _restrict_resolved_target_for_service(
        self,
        resolved_target: Dict[str, Any],
        *,
        default_domain: str | None = None,
        service_description: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Restrict resolved entity targets using known service target metadata."""
        allowed_domains = []
        if service_description:
            allowed_domains = self._extract_service_target_domains(service_description)
        if not allowed_domains and default_domain:
            allowed_domains = [default_domain]

        entity_ids = self._normalize_target_values(resolved_target.get("entity_id"))
        if not allowed_domains:
            resolved_domains = {entity_id.split(".", 1)[0] for entity_id in entity_ids}
            if len(resolved_domains) > 1:
                raise ValueError(
                    "Resolved target spans multiple domains, and this service does not "
                    "publish target-domain metadata. Use explicit entity_id values from discovery."
                )
            return {"entity_id": entity_ids}

        filtered_entity_ids = [
            entity_id
            for entity_id in entity_ids
            if entity_id.split(".", 1)[0] in allowed_domains
        ]

        if not filtered_entity_ids:
            raise ValueError(
                "Resolved target did not include any exposed entities accepted by this service."
            )

        return {"entity_id": filtered_entity_ids}

    def _restrict_resolved_target_to_domain(
        self, resolved_target: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Restrict resolved entity targets to the requested domain."""
        return self._restrict_resolved_target_for_service(
            resolved_target, default_domain=domain
        )

    def _serialize_service_response_value(self, value: Any) -> Any:
        """Serialize HA service response data to JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): self._serialize_service_response_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_service_response_value(item) for item in value]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return str(value)

    def _format_service_response_result(
        self,
        domain: str,
        service: str,
        resolved_target: Dict[str, Any],
        response: Any,
        *,
        request_data: Dict[str, Any] | None = None,
    ) -> str:
        """Format a response-returning service call for the LLM."""
        entity_ids = self._normalize_target_values(resolved_target.get("entity_id"))
        if entity_ids:
            target_count = len(entity_ids)
            target_label = "entity" if target_count == 1 else "entities"
            header = (
                f"✅ Retrieved response from {domain}.{service} for "
                f"{target_count} target {target_label}."
            )
        else:
            header = f"✅ Retrieved response from {domain}.{service}."

        text_parts = [header]

        if domain == "weather" and service in {"get_forecast", "get_forecasts"}:
            forecast_type = str((request_data or {}).get("type") or "").strip()
            if forecast_type:
                text_parts.append("")
                text_parts.append(f"Forecast type used: {forecast_type}.")

        summary_lines = self._build_service_response_summary(domain, service, response)
        if summary_lines:
            text_parts.append("")
            text_parts.extend(summary_lines)

        if response is None:
            text_parts.append("")
            text_parts.append("No response data was returned.")
        else:
            text_parts.append("")
            text_parts.append("Response:")
            text_parts.append(json.dumps(response, indent=2))

        return "\n".join(text_parts)

    def _build_service_response_summary(
        self, domain: str, service: str, response: Any
    ) -> List[str]:
        """Build a concise summary for structured service responses."""
        if domain == "music_assistant":
            return summarize_music_assistant_response(service, response)
        if domain == "weather" and service in {"get_forecast", "get_forecasts"}:
            return self._summarize_weather_response(response)
        if domain == "calendar" and service == "get_events":
            return self._summarize_calendar_response(response)

        if isinstance(response, dict):
            return [f"Summary: {len(response)} top-level response entries."]
        if isinstance(response, list):
            return [f"Summary: {len(response)} response items."]
        return []

    def _format_service_response_datetime(self, value: Any) -> str | None:
        """Format a service-response date/time value in local time when possible."""
        if value is None:
            return None

        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            return value.isoformat()
        else:
            parsed = dt_util.parse_datetime(str(value))
            if parsed is None:
                return str(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
            )

        return self._format_absolute_time(dt_util.as_utc(parsed))
