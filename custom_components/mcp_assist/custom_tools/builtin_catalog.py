"""Catalog helpers for built-in manifest-based MCP Assist tool packages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..const import CUSTOM_TOOL_MANIFEST_FILENAME


@dataclass(frozen=True)
class BuiltInToolToggleSpec:
    """Enable/disable metadata declared by a built-in tool package."""

    package_id: str
    package_name: str
    package_description: str
    tool_names: tuple[str, ...]
    shared_setting_key: str
    profile_setting_key: str
    shared_default: bool
    profile_default: bool
    shared_label: str
    shared_description: str
    profile_disable_label: str
    profile_disable_description: str
    legacy_shared_setting_keys: tuple[str, ...] = ()
    legacy_profile_setting_keys: tuple[str, ...] = ()
    requires_search_provider: bool = False


def get_builtin_packages_root() -> Path:
    """Return the built-in package directory."""
    return Path(__file__).resolve().parent / "packages"


def load_builtin_tool_toggle_specs(
    packages_root: Path | None = None,
) -> tuple[BuiltInToolToggleSpec, ...]:
    """Load built-in tool toggle metadata from package manifests."""
    root = packages_root or get_builtin_packages_root()
    if not root.is_dir():
        return ()

    specs: list[BuiltInToolToggleSpec] = []
    for package_dir in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not package_dir.is_dir() or package_dir.name.startswith((".", "__")):
            continue

        manifest_path = package_dir / CUSTOM_TOOL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue

        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            continue

        built_in = raw_manifest.get("built_in_toggle")
        if not isinstance(built_in, dict):
            continue

        package_id = str(raw_manifest.get("id") or package_dir.name).strip()
        package_name = str(raw_manifest.get("name") or package_id).strip() or package_id
        package_description = str(raw_manifest.get("description") or "").strip()
        tool_names_value = built_in.get("tool_names") or ()
        if not isinstance(tool_names_value, (list, tuple)):
            raise ValueError(
                f"{manifest_path} built_in_toggle.tool_names must be a list of strings"
            )
        tool_names = tuple(
            str(tool_name).strip()
            for tool_name in tool_names_value
            if str(tool_name).strip()
        )
        if not tool_names:
            raise ValueError(f"{manifest_path} must declare built_in_toggle.tool_names")

        shared_setting_key = str(built_in.get("shared_setting_key") or "").strip()
        profile_setting_key = str(built_in.get("profile_setting_key") or "").strip()
        if not shared_setting_key or not profile_setting_key:
            raise ValueError(
                f"{manifest_path} must declare built_in_toggle shared/profile setting keys"
            )

        shared_label = (
            str(built_in.get("shared_label") or "").strip() or package_name
        )
        profile_disable_label = (
            str(built_in.get("profile_disable_label") or "").strip()
            or f"Disable {package_name}"
        )

        shared_description = str(built_in.get("shared_description") or "").strip()
        profile_disable_description = str(
            built_in.get("profile_disable_description") or ""
        ).strip()

        legacy_shared_setting_keys = tuple(
            str(key).strip()
            for key in tuple(built_in.get("legacy_shared_setting_keys") or ())
            if str(key).strip()
        )
        legacy_profile_setting_keys = tuple(
            str(key).strip()
            for key in tuple(built_in.get("legacy_profile_setting_keys") or ())
            if str(key).strip()
        )

        specs.append(
            BuiltInToolToggleSpec(
                package_id=package_id,
                package_name=package_name,
                package_description=package_description,
                tool_names=tool_names,
                shared_setting_key=shared_setting_key,
                profile_setting_key=profile_setting_key,
                shared_default=bool(built_in.get("shared_default", False)),
                profile_default=bool(built_in.get("profile_default", True)),
                shared_label=shared_label,
                shared_description=shared_description,
                profile_disable_label=profile_disable_label,
                profile_disable_description=profile_disable_description,
                legacy_shared_setting_keys=legacy_shared_setting_keys,
                legacy_profile_setting_keys=legacy_profile_setting_keys,
                requires_search_provider=bool(
                    built_in.get("requires_search_provider", False)
                ),
            )
        )

    return tuple(specs)


def get_builtin_toggle_spec_by_tool_name(
    tool_name: str,
    specs: tuple[BuiltInToolToggleSpec, ...],
) -> BuiltInToolToggleSpec | None:
    """Return the built-in toggle metadata for a tool name, if any."""
    normalized_tool_name = str(tool_name or "").strip()
    if not normalized_tool_name:
        return None
    for spec in specs:
        if normalized_tool_name in spec.tool_names:
            return spec
    return None


def get_builtin_toggle_spec_by_package_id(
    package_id: str,
    specs: tuple[BuiltInToolToggleSpec, ...],
) -> BuiltInToolToggleSpec | None:
    """Return the built-in toggle metadata for a package id, if any."""
    normalized_package_id = str(package_id or "").strip()
    if not normalized_package_id:
        return None
    for spec in specs:
        if spec.package_id == normalized_package_id:
            return spec
    return None


def get_builtin_shared_setting_value(
    spec: BuiltInToolToggleSpec,
    get_setting: Any,
) -> Any:
    """Resolve a built-in package shared setting with legacy fallback keys."""
    value = get_setting(spec.shared_setting_key, None)
    if value is not None:
        return value

    for legacy_key in spec.legacy_shared_setting_keys:
        value = get_setting(legacy_key, None)
        if value is not None:
            return value

    return spec.shared_default


def get_builtin_profile_setting_value(
    spec: BuiltInToolToggleSpec,
    get_profile_setting: Any,
) -> Any:
    """Resolve a built-in package profile setting with legacy fallback keys."""
    value = get_profile_setting(spec.profile_setting_key, None)
    if value is not None:
        return value

    for legacy_key in spec.legacy_profile_setting_keys:
        value = get_profile_setting(legacy_key, None)
        if value is not None:
            return value

    return spec.profile_default


def is_builtin_package_enabled_for_shared_settings(
    spec: BuiltInToolToggleSpec,
    get_setting: Any,
    *,
    search_provider: str = "none",
) -> bool:
    """Return whether a built-in package should be enabled by shared settings."""
    if not bool(get_builtin_shared_setting_value(spec, get_setting)):
        return False

    if spec.requires_search_provider and search_provider not in {
        "brave",
        "duckduckgo",
    }:
        return False

    return True


def is_builtin_package_enabled_for_profile(
    spec: BuiltInToolToggleSpec,
    get_shared_setting: Any,
    get_profile_setting: Any,
    *,
    search_provider: str = "none",
) -> bool:
    """Return whether a built-in package is enabled for a specific profile."""
    return bool(
        is_builtin_package_enabled_for_shared_settings(
            spec,
            get_shared_setting,
            search_provider=search_provider,
        )
        and get_builtin_profile_setting_value(spec, get_profile_setting)
    )
