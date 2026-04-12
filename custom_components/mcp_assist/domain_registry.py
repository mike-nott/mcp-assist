"""Domain Registry for Home Assistant MCP Integration.

This module contains all domain service mappings and validation logic.
Centralizes domain definitions to keep the main MCP server clean and maintainable.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging

_LOGGER = logging.getLogger(__name__)

# Priority levels for domain implementation
PRIORITY_ESSENTIAL = 1  # Must have for basic HA control
PRIORITY_COMMON = 2  # Found in most homes
PRIORITY_STANDARD = 3  # Standard helpers and inputs
PRIORITY_EXTENDED = 4  # Extended functionality
PRIORITY_SPECIALIZED = 5  # Specialized use cases

# Domain types
TYPE_CONTROLLABLE = "controllable"
TYPE_READ_ONLY = "read_only"
TYPE_SERVICE_ONLY = "service_only"

# Complete domain registry with all services and parameters
DOMAIN_REGISTRY = {
    # ========== P1: Essential Core Domains ==========
    "light": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": ["turn_on", "turn_off", "toggle"],
        "parameters": {
            "turn_on": {
                "optional": [
                    "transition",
                    "brightness",
                    "brightness_pct",
                    "brightness_step",
                    "brightness_step_pct",
                    "rgb_color",
                    "rgbw_color",
                    "rgbww_color",
                    "color_name",
                    "hs_color",
                    "xy_color",
                    "color_temp_kelvin",
                    "white",
                    "profile",
                    "flash",
                    "effect",
                ]
            },
            "turn_off": {"optional": ["transition", "flash"]},
            "toggle": {
                "optional": [
                    "transition",
                    "brightness",
                    "brightness_pct",
                    "rgb_color",
                    "color_temp_kelvin",
                    "effect",
                ]
            },
        },
        "description": "Control lights with brightness, color, and effects",
    },
    "switch": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": ["turn_on", "turn_off", "toggle"],
        "parameters": {},
        "description": "Control binary switches",
    },
    "cover": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": [
            "open_cover",
            "close_cover",
            "stop_cover",
            "toggle",
            "set_cover_position",
            "open_cover_tilt",
            "close_cover_tilt",
            "stop_cover_tilt",
            "set_cover_tilt_position",
        ],
        "parameters": {
            "set_cover_position": {"required": ["position"]},
            "set_cover_tilt_position": {"required": ["tilt_position"]},
        },
        "description": "Control covers, blinds, and garage doors",
    },
    "climate": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": [
            "set_temperature",
            "set_hvac_mode",
            "set_preset_mode",
            "set_fan_mode",
            "set_humidity",
            "set_swing_mode",
            "set_swing_horizontal_mode",
            "turn_on",
            "turn_off",
            "toggle",
        ],
        "parameters": {
            "set_temperature": {
                "optional": [
                    "temperature",
                    "target_temp_high",
                    "target_temp_low",
                    "hvac_mode",
                ]
            },
            "set_hvac_mode": {"required": ["hvac_mode"]},
            "set_preset_mode": {"required": ["preset_mode"]},
            "set_fan_mode": {"required": ["fan_mode"]},
            "set_humidity": {"required": ["humidity"]},
            "set_swing_mode": {"required": ["swing_mode"]},
            "set_swing_horizontal_mode": {"required": ["swing_horizontal_mode"]},
        },
        "description": "Control thermostats and HVAC systems",
    },
    "lock": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": ["lock", "unlock", "open"],
        "parameters": {
            "lock": {"optional": ["code"]},
            "unlock": {"optional": ["code"]},
            "open": {"optional": ["code"]},
        },
        "description": "Control door locks",
    },
    "scene": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": ["turn_on", "reload", "apply", "create", "delete"],
        "parameters": {
            "turn_on": {"optional": ["transition"]},
            "apply": {"required": ["entities"], "optional": ["transition"]},
            "create": {
                "required": ["scene_id"],
                "optional": ["entities", "snapshot_entities"],
            },
        },
        "description": "Activate and manage scenes",
    },
    "script": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": ["turn_on", "turn_off", "toggle", "reload"],
        "parameters": {},
        "description": "Execute and manage scripts",
    },
    "automation": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_ESSENTIAL,
        "services": ["trigger", "turn_on", "turn_off", "toggle", "reload"],
        "parameters": {
            "trigger": {"optional": ["skip_condition"]},
            "turn_off": {"optional": ["stop_actions"]},
        },
        "description": "Control and trigger automations",
    },
    # ========== P2: Common Control Domains ==========
    "fan": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_COMMON,
        "services": [
            "turn_on",
            "turn_off",
            "toggle",
            "set_percentage",
            "set_preset_mode",
            "oscillate",
            "set_direction",
            "increase_speed",
            "decrease_speed",
        ],
        "parameters": {
            "turn_on": {"optional": ["percentage", "preset_mode"]},
            "set_percentage": {"required": ["percentage"]},
            "set_preset_mode": {"required": ["preset_mode"]},
            "oscillate": {"required": ["oscillating"]},
            "set_direction": {"required": ["direction"]},
            "increase_speed": {"optional": ["percentage_step"]},
            "decrease_speed": {"optional": ["percentage_step"]},
        },
        "description": "Control fans with speed and oscillation",
    },
    "media_player": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_COMMON,
        "services": [
            "turn_on",
            "turn_off",
            "toggle",
            "volume_up",
            "volume_down",
            "volume_set",
            "volume_mute",
            "media_play",
            "media_pause",
            "media_stop",
            "media_play_pause",
            "media_next_track",
            "media_previous_track",
            "media_seek",
            "play_media",
            "select_source",
            "select_sound_mode",
            "clear_playlist",
            "shuffle_set",
            "repeat_set",
            "join",
            "unjoin",
            "browse_media",
            "search_media",
        ],
        "parameters": {
            "volume_set": {"required": ["volume_level"]},
            "volume_mute": {"required": ["is_volume_muted"]},
            "media_seek": {"required": ["seek_position"]},
            "play_media": {"required": ["media"], "optional": ["enqueue", "announce"]},
            "select_source": {"required": ["source"]},
            "select_sound_mode": {"optional": ["sound_mode"]},
            "shuffle_set": {"required": ["shuffle"]},
            "repeat_set": {"required": ["repeat"]},
            "join": {"required": ["group_members"]},
            "browse_media": {"optional": ["media_content_type", "media_content_id"]},
            "search_media": {
                "required": ["search_query"],
                "optional": [
                    "media_content_type",
                    "media_content_id",
                    "media_filter_classes",
                ],
            },
        },
        "response_services": ["browse_media", "search_media"],
        "description": "Control media players and streaming devices",
    },
    "vacuum": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_COMMON,
        "services": [
            "turn_on",
            "turn_off",
            "toggle",
            "start",
            "stop",
            "pause",
            "start_pause",
            "return_to_base",
            "clean_spot",
            "locate",
            "send_command",
            "set_fan_speed",
        ],
        "parameters": {
            "send_command": {"required": ["command"], "optional": ["params"]},
            "set_fan_speed": {"required": ["fan_speed"]},
        },
        "description": "Control robot vacuums",
    },
    "alarm_control_panel": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_COMMON,
        "services": [
            "alarm_disarm",
            "alarm_arm_home",
            "alarm_arm_away",
            "alarm_arm_night",
            "alarm_arm_vacation",
            "alarm_arm_custom_bypass",
            "alarm_trigger",
        ],
        "parameters": {
            "alarm_disarm": {"optional": ["code"]},
            "alarm_arm_home": {"optional": ["code"]},
            "alarm_arm_away": {"optional": ["code"]},
            "alarm_arm_night": {"optional": ["code"]},
            "alarm_arm_vacation": {"optional": ["code"]},
            "alarm_arm_custom_bypass": {"optional": ["code"]},
            "alarm_trigger": {"optional": ["code"]},
        },
        "description": "Control security alarm systems",
    },
    "camera": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_COMMON,
        "services": [
            "turn_on",
            "turn_off",
            "enable_motion_detection",
            "disable_motion_detection",
            "snapshot",
            "record",
            "play_stream",
        ],
        "parameters": {
            "snapshot": {"required": ["filename"]},
            "record": {"required": ["filename"], "optional": ["duration", "lookback"]},
            "play_stream": {"required": ["media_player"], "optional": ["format"]},
        },
        "description": "Control cameras and capture media",
    },
    "sensor": {
        "type": TYPE_READ_ONLY,
        "priority": PRIORITY_COMMON,
        "services": [],
        "parameters": {},
        "error_message": "Sensors are read-only. Use 'get_entity_details' to read sensor values.",
        "description": "Read-only numeric and string sensors",
    },
    "binary_sensor": {
        "type": TYPE_READ_ONLY,
        "priority": PRIORITY_COMMON,
        "services": [],
        "parameters": {},
        "error_message": "Binary sensors are read-only. Use 'get_entity_details' to read sensor state.",
        "description": "Read-only on/off state sensors",
    },
    "device_tracker": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_COMMON,
        "services": ["see"],
        "parameters": {
            "see": {
                "optional": [
                    "mac",
                    "dev_id",
                    "host_name",
                    "location_name",
                    "gps",
                    "gps_accuracy",
                    "battery",
                ]
            }
        },
        "description": "Track device locations",
    },
    # ========== P3: Standard Helper Domains ==========
    "input_boolean": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["turn_on", "turn_off", "toggle", "reload"],
        "parameters": {},
        "description": "Boolean input helpers",
    },
    "input_number": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["set_value", "increment", "decrement", "reload"],
        "parameters": {"set_value": {"required": ["value"]}},
        "description": "Numeric input helpers",
    },
    "input_text": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["set_value", "reload"],
        "parameters": {"set_value": {"required": ["value"]}},
        "description": "Text input helpers",
    },
    "input_select": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": [
            "select_option",
            "select_next",
            "select_previous",
            "select_first",
            "select_last",
            "set_options",
            "reload",
        ],
        "parameters": {
            "select_option": {"required": ["option"]},
            "select_next": {"optional": ["cycle"]},
            "select_previous": {"optional": ["cycle"]},
            "set_options": {"required": ["options"]},
        },
        "description": "Dropdown selection helpers",
    },
    "input_datetime": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["set_datetime", "reload"],
        "parameters": {
            "set_datetime": {"optional": ["date", "time", "datetime", "timestamp"]}
        },
        "description": "Date and time input helpers",
    },
    "input_button": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["press", "reload"],
        "parameters": {},
        "description": "Button input helpers",
    },
    "timer": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["start", "pause", "cancel", "finish", "change", "reload"],
        "parameters": {
            "start": {"optional": ["duration"]},
            "change": {"required": ["duration"]},
        },
        "description": "Timer helpers",
    },
    "counter": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["increment", "decrement", "reset", "set_value"],
        "parameters": {"set_value": {"required": ["value"]}},
        "description": "Counter helpers",
    },
    "person": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["reload"],
        "parameters": {},
        "description": "Person entities",
    },
    "zone": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["reload"],
        "parameters": {},
        "description": "Geographic zones",
    },
    "group": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_STANDARD,
        "services": ["reload", "set", "remove"],
        "parameters": {
            "set": {
                "required": ["object_id"],
                "optional": [
                    "name",
                    "icon",
                    "entities",
                    "add_entities",
                    "remove_entities",
                    "all",
                ],
            },
            "remove": {"required": ["object_id"]},
        },
        "description": "Entity groups",
    },
    # ========== P4: Modern Platform Domains ==========
    "select": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": [
            "select_option",
            "select_first",
            "select_last",
            "select_next",
            "select_previous",
        ],
        "parameters": {
            "select_option": {"required": ["option"]},
            "select_next": {"optional": ["cycle"]},
            "select_previous": {"optional": ["cycle"]},
        },
        "description": "Selection entities",
    },
    "number": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["set_value"],
        "parameters": {"set_value": {"required": ["value"]}},
        "description": "Numeric control entities",
    },
    "button": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["press"],
        "parameters": {},
        "description": "Button entities",
    },
    "update": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["install", "skip", "clear_skipped"],
        "parameters": {"install": {"optional": ["version", "backup"]}},
        "description": "Update entities for firmware and software",
    },
    "text": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["set_value"],
        "parameters": {"set_value": {"required": ["value"]}},
        "description": "Text control entities",
    },
    "date": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["set_value"],
        "parameters": {"set_value": {"required": ["date"]}},
        "description": "Date control entities",
    },
    "time": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["set_value"],
        "parameters": {"set_value": {"required": ["time"]}},
        "description": "Time control entities",
    },
    "datetime": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_EXTENDED,
        "services": ["set_value"],
        "parameters": {"set_value": {"required": ["datetime"]}},
        "description": "Date and time control entities",
    },
    # ========== P5: Specialized Domains ==========
    "water_heater": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": [
            "set_temperature",
            "set_operation_mode",
            "set_away_mode",
            "turn_on",
            "turn_off",
        ],
        "parameters": {
            "set_temperature": {
                "required": ["temperature"],
                "optional": ["operation_mode"],
            },
            "set_operation_mode": {"required": ["operation_mode"]},
            "set_away_mode": {"required": ["away_mode"]},
        },
        "description": "Control water heaters",
    },
    "humidifier": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["turn_on", "turn_off", "toggle", "set_mode", "set_humidity"],
        "parameters": {
            "set_mode": {"required": ["mode"]},
            "set_humidity": {"required": ["humidity"]},
        },
        "description": "Control humidifiers and dehumidifiers",
    },
    "siren": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["turn_on", "turn_off", "toggle"],
        "parameters": {"turn_on": {"optional": ["tone", "duration", "volume_level"]}},
        "description": "Control alarm sirens",
    },
    "valve": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": [
            "open_valve",
            "close_valve",
            "set_valve_position",
            "stop_valve",
            "toggle",
        ],
        "parameters": {"set_valve_position": {"required": ["position"]}},
        "description": "Control water and gas valves",
    },
    "lawn_mower": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["start_mowing", "pause", "dock"],
        "parameters": {},
        "description": "Control robotic lawn mowers",
    },
    "weather": {
        "type": TYPE_READ_ONLY,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["get_forecast", "get_forecasts"],
        "parameters": {
            "get_forecast": {"required": ["type"]},
            "get_forecasts": {"required": ["type"]},
        },
        "response_services": ["get_forecast", "get_forecasts"],
        "description": "Weather information and forecasts",
    },
    "calendar": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["create_event"],
        "parameters": {
            "create_event": {
                "required": ["summary"],
                "optional": [
                    "description",
                    "location",
                    "start_date",
                    "end_date",
                    "start_date_time",
                    "end_date_time",
                    "in",
                ]
            }
        },
        "response_services": ["get_events"],
        "description": "Read calendar events and create new calendar events",
    },
    "todo": {
        "type": TYPE_CONTROLLABLE,
        "priority": PRIORITY_SPECIALIZED,
        "services": [
            "add_item",
            "update_item",
            "remove_item",
            "remove_completed_items",
        ],
        "parameters": {
            "add_item": {
                "required": ["item"],
                "optional": ["due_date", "due_datetime", "description"],
            },
            "update_item": {
                "required": ["item"],
                "optional": [
                    "rename",
                    "status",
                    "due_date",
                    "due_datetime",
                    "description",
                ],
            },
            "remove_item": {"required": ["item"]},
            "remove_completed_items": {"required": []},
        },
        "response_services": ["get_items"],
        "description": "Manage to-do lists and query their items",
    },
    "sun": {
        "type": TYPE_READ_ONLY,
        "priority": PRIORITY_SPECIALIZED,
        "services": [],
        "parameters": {},
        "error_message": "Sun is a system entity providing sunrise/sunset times. Use 'get_entity_details' to read values.",
        "description": "Solar position and sunrise/sunset times",
    },
    # ========== P6: Service-Only Domains ==========
    "notify": {
        "type": TYPE_SERVICE_ONLY,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["notify", "send_message", "persistent_notification"],
        "parameters": {
            "notify": {
                "required": ["message"],
                "optional": ["title", "target", "data"],
            },
            "send_message": {"required": ["message"], "optional": ["title"]},
            "persistent_notification": {
                "required": ["message"],
                "optional": ["title", "data"],
            },
        },
        "description": "Send notifications to devices and services",
    },
    "tts": {
        "type": TYPE_SERVICE_ONLY,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["speak", "say", "clear_cache"],
        "parameters": {
            "speak": {
                "required": ["media_player_entity_id", "message"],
                "optional": ["cache", "language", "options"],
            },
            "say": {
                "required": ["entity_id", "message"],
                "optional": ["cache", "language", "options"],
            },
        },
        "description": "Text-to-speech services",
    },
    "persistent_notification": {
        "type": TYPE_SERVICE_ONLY,
        "priority": PRIORITY_SPECIALIZED,
        "services": ["create", "dismiss", "mark_read"],
        "parameters": {
            "create": {
                "required": ["message"],
                "optional": ["title", "notification_id"],
            },
            "dismiss": {"required": ["notification_id"]},
            "mark_read": {"required": ["notification_id"]},
        },
        "description": "Create persistent UI notifications",
    },
}

# Common action aliases that map to standard services
ACTION_ALIASES = {
    # Common aliases for turn_on/turn_off
    "activate": "turn_on",
    "deactivate": "turn_off",
    "enable": "turn_on",
    "disable": "turn_off",
    "start": "turn_on",
    "stop": "turn_off",
    # Lock-specific aliases
    "secure": "lock",
    "unsecure": "unlock",
    # Cover-specific aliases
    "open": "open_cover",
    "close": "close_cover",
    "raise": "open_cover",
    "lower": "close_cover",
    # Media player aliases
    "play": "media_play",
    "pause": "media_pause",
    "next": "media_next_track",
    "previous": "media_previous_track",
    "mute": "volume_mute",
    # Climate aliases
    "heat": "set_hvac_mode",
    "cool": "set_hvac_mode",
    # Vacuum aliases
    "clean": "start",
    "dock": "return_to_base",
    "home": "return_to_base",
}


def get_domain_info(domain: str) -> Optional[Dict[str, Any]]:
    """Get domain configuration from registry.

    Args:
        domain: The domain name to look up

    Returns:
        Domain configuration dict or None if not found
    """
    return DOMAIN_REGISTRY.get(domain)


def get_supported_domains(priority: Optional[int] = None) -> List[str]:
    """Get list of supported domains, optionally filtered by priority.

    Args:
        priority: Optional priority level to filter by

    Returns:
        List of domain names
    """
    if priority is None:
        return list(DOMAIN_REGISTRY.keys())

    return [
        domain
        for domain, info in DOMAIN_REGISTRY.items()
        if info.get("priority") == priority
    ]


def map_action_to_service(domain: str, action: str) -> str:
    """Map common action names to domain-specific service names.

    Args:
        domain: The target domain
        action: The action requested (might be an alias)

    Returns:
        The actual service name to call
    """
    # First check if it's already a valid service name
    domain_info = get_domain_info(domain)
    if domain_info and action in domain_info.get("services", []):
        return action

    # Check common aliases
    if action in ACTION_ALIASES:
        return ACTION_ALIASES[action]

    # Domain-specific mappings
    if domain == "cover":
        if action in ["raise", "lift"]:
            return "open_cover"
        elif action in ["lower", "drop"]:
            return "close_cover"
    elif domain == "calendar":
        if action in ["create", "add", "new", "schedule"]:
            return "create_event"
    elif domain == "lock":
        if action == "secure":
            return "lock"
        elif action == "unsecure":
            return "unlock"
    elif domain == "todo":
        if action in ["create", "add", "new"]:
            return "add_item"
        if action in ["delete", "remove"]:
            return "remove_item"
        if action in ["clear_completed", "remove_completed", "cleanup"]:
            return "remove_completed_items"
    elif domain == "vacuum":
        if action == "clean":
            return "start"
        elif action in ["dock", "home"]:
            return "return_to_base"

    # Default: return the action as-is
    return action


def validate_domain_action(domain: str, action: str) -> Tuple[bool, str]:
    """Validate if an action is valid for a domain.

    Args:
        domain: The target domain
        action: The action to validate

    Returns:
        Tuple of (is_valid, service_name_or_error_message)
    """
    # Get domain info
    domain_info = get_domain_info(domain)

    # Check if domain exists
    if not domain_info:
        # List similar domains if available
        similar = [d for d in DOMAIN_REGISTRY.keys() if domain in d or d in domain]
        if similar:
            return (
                False,
                f"Domain '{domain}' not supported. Did you mean: {', '.join(similar[:3])}?",
            )
        return (
            False,
            f"Domain '{domain}' not supported. Use 'list_domains' to see available domains.",
        )

    # Check if domain is read-only
    if domain_info["type"] == TYPE_READ_ONLY:
        return False, domain_info.get(
            "error_message",
            f"Domain '{domain}' is read-only. Use 'get_entity_details' to read values.",
        )

    # Map the action to a service name
    service = map_action_to_service(domain, action)

    # Check if service is valid for this domain
    if service in domain_info.get("services", []):
        return True, service

    # Provide helpful error with available services
    available_services = domain_info.get("services", [])
    if available_services:
        return (
            False,
            f"Action '{action}' not valid for {domain}. Available: {', '.join(available_services[:5])}",
        )

    return False, f"Domain '{domain}' has no available services"


def get_service_parameters(domain: str, service: str) -> Dict[str, List[str]]:
    """Get required and optional parameters for a service.

    Args:
        domain: The target domain
        service: The service name

    Returns:
        Dict with 'required' and 'optional' parameter lists
    """
    domain_info = get_domain_info(domain)
    if not domain_info:
        return {"required": [], "optional": []}

    params = domain_info.get("parameters", {}).get(service, {})
    return {
        "required": params.get("required", []),
        "optional": params.get("optional", []),
    }


def validate_service_parameters(
    domain: str, service: str, provided_params: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate that required parameters are provided for a service.

    Args:
        domain: The target domain
        service: The service name
        provided_params: Parameters provided by the user

    Returns:
        Tuple of (is_valid, error_message_or_success)
    """
    params_info = get_service_parameters(domain, service)
    required_params = params_info.get("required", [])

    # Check for missing required parameters
    missing = [p for p in required_params if p not in provided_params]
    if missing:
        return (
            False,
            f"Missing required parameters for {domain}.{service}: {', '.join(missing)}",
        )

    if domain == "calendar" and service == "create_event":
        has_date_range = "start_date" in provided_params or "end_date" in provided_params
        has_datetime_range = (
            "start_date_time" in provided_params or "end_date_time" in provided_params
        )
        has_relative_time = "in" in provided_params

        mode_count = sum((has_date_range, has_datetime_range, has_relative_time))
        if mode_count == 0:
            return (
                False,
                "Missing calendar time fields for calendar.create_event. Provide either "
                "start_date/end_date, start_date_time/end_date_time, or 'in'.",
            )
        if mode_count > 1:
            return (
                False,
                "calendar.create_event accepts only one scheduling mode: either "
                "start_date/end_date, start_date_time/end_date_time, or 'in'.",
            )
        if has_date_range and not {
            "start_date",
            "end_date",
        }.issubset(provided_params):
            return (
                False,
                "calendar.create_event requires both start_date and end_date when using all-day dates.",
            )
        if has_datetime_range and not {
            "start_date_time",
            "end_date_time",
        }.issubset(provided_params):
            return (
                False,
                "calendar.create_event requires both start_date_time and end_date_time when using timed events.",
            )

    if domain == "todo":
        if "due_date" in provided_params and "due_datetime" in provided_params:
            return (
                False,
                f"{domain}.{service} accepts only one of due_date or due_datetime.",
            )

        if service == "update_item" and not any(
            key in provided_params
            for key in (
                "rename",
                "status",
                "due_date",
                "due_datetime",
                "description",
            )
        ):
            return (
                False,
                "todo.update_item requires at least one update field such as rename, status, due_date, due_datetime, or description.",
            )

    return True, "Parameters valid"


def get_response_services(domain: str) -> List[str]:
    """Get response-returning services for a domain."""
    domain_info = get_domain_info(domain)
    if not domain_info:
        return []

    response_services = domain_info.get("response_services")
    if response_services is not None:
        return list(response_services)

    if domain_info.get("type") == TYPE_READ_ONLY:
        return list(domain_info.get("services", []))

    return []


def validate_response_service(domain: str, service: str) -> Tuple[bool, str]:
    """Validate if a service is supported for response-returning reads."""
    domain_info = get_domain_info(domain)

    if not domain_info:
        similar = [d for d in DOMAIN_REGISTRY.keys() if domain in d or d in domain]
        if similar:
            return (
                False,
                f"Domain '{domain}' not supported. Did you mean: {', '.join(similar[:3])}?",
            )
        return (
            False,
            f"Domain '{domain}' not supported. Use 'list_domains' to see available domains.",
        )

    available_services = get_response_services(domain)
    if service in available_services:
        return True, service

    if available_services:
        return (
            False,
            f"Service '{service}' is not supported for response reads on {domain}. "
            f"Available: {', '.join(available_services[:5])}",
        )

    return (
        False,
        f"Domain '{domain}' has no supported response-returning read services.",
    )


def get_domains_by_type(domain_type: str) -> List[str]:
    """Get all domains of a specific type.

    Args:
        domain_type: One of TYPE_CONTROLLABLE, TYPE_READ_ONLY, TYPE_SERVICE_ONLY

    Returns:
        List of domain names
    """
    return [
        domain
        for domain, info in DOMAIN_REGISTRY.items()
        if info.get("type") == domain_type
    ]


def get_domain_statistics() -> Dict[str, int]:
    """Get statistics about registered domains.

    Returns:
        Dict with counts by type and priority
    """
    stats = {
        "total": len(DOMAIN_REGISTRY),
        "controllable": len(get_domains_by_type(TYPE_CONTROLLABLE)),
        "read_only": len(get_domains_by_type(TYPE_READ_ONLY)),
        "service_only": len(get_domains_by_type(TYPE_SERVICE_ONLY)),
    }

    # Add priority counts
    for priority in range(1, 7):
        priority_domains = get_supported_domains(priority)
        if priority_domains:
            stats[f"priority_{priority}"] = len(priority_domains)

    return stats


# Export the main functions
__all__ = [
    "DOMAIN_REGISTRY",
    "get_domain_info",
    "get_supported_domains",
    "validate_domain_action",
    "validate_response_service",
    "map_action_to_service",
    "get_service_parameters",
    "validate_service_parameters",
    "get_response_services",
    "get_domains_by_type",
    "get_domain_statistics",
    "PRIORITY_ESSENTIAL",
    "PRIORITY_COMMON",
    "PRIORITY_STANDARD",
    "PRIORITY_EXTENDED",
    "PRIORITY_SPECIALIZED",
    "TYPE_CONTROLLABLE",
    "TYPE_READ_ONLY",
    "TYPE_SERVICE_ONLY",
]
