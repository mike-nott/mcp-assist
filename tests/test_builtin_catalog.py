"""Tests for built-in packaged-tool toggle metadata."""

from __future__ import annotations

from custom_components.mcp_assist.custom_tools.builtin_catalog import (
    BuiltInToolToggleSpec,
    get_builtin_profile_setting_value,
    get_builtin_shared_setting_value,
    get_builtin_toggle_spec_by_package_id,
    get_builtin_toggle_spec_by_tool_name,
    is_builtin_package_enabled_for_profile,
    is_builtin_package_enabled_for_shared_settings,
    load_builtin_tool_toggle_specs,
)


BUILTIN_SPECS = load_builtin_tool_toggle_specs()


def _spec(package_id: str) -> BuiltInToolToggleSpec:
    """Return a built-in spec by package id."""
    spec = get_builtin_toggle_spec_by_package_id(package_id, BUILTIN_SPECS)
    assert spec is not None
    return spec


def test_builtin_catalog_loads_expected_manifest_packages() -> None:
    """The manifest catalog should discover the built-in package toggle metadata."""
    package_ids = {spec.package_id for spec in BUILTIN_SPECS}

    assert {"calculator", "unit_conversion", "search", "read_url"} <= package_ids


def test_builtin_catalog_looks_up_specs_by_package_id_and_tool_name() -> None:
    """Package and tool-name lookups should resolve to the same toggle metadata."""
    assert get_builtin_toggle_spec_by_package_id("read_url", BUILTIN_SPECS) == _spec(
        "read_url"
    )
    assert get_builtin_toggle_spec_by_tool_name("add", BUILTIN_SPECS) == _spec(
        "calculator"
    )
    assert get_builtin_toggle_spec_by_tool_name(
        "convert_unit", BUILTIN_SPECS
    ) == _spec("unit_conversion")
    assert get_builtin_toggle_spec_by_tool_name("missing_tool", BUILTIN_SPECS) is None


def test_builtin_shared_setting_value_prefers_explicit_key_over_legacy_keys() -> None:
    """Explicit shared settings should override any legacy fallback keys."""
    spec = _spec("search")

    value = get_builtin_shared_setting_value(
        spec,
        lambda key, default=None: {
            spec.shared_setting_key: False,
            "enable_web_search": True,
        }.get(key, default),
    )

    assert value is False


def test_builtin_shared_setting_value_falls_back_to_legacy_keys() -> None:
    """Legacy shared settings should still enable migrated built-in packages."""
    spec = _spec("read_url")

    value = get_builtin_shared_setting_value(
        spec,
        lambda key, default=None: {"enable_web_search": True}.get(key, default),
    )

    assert value is True


def test_builtin_profile_setting_value_falls_back_to_legacy_keys() -> None:
    """Legacy profile settings should still feed the built-in package toggles."""
    spec = _spec("unit_conversion")

    value = get_builtin_profile_setting_value(
        spec,
        lambda key, default=None: {"profile_enable_calculator_tools": False}.get(
            key, default
        ),
    )

    assert value is False


def test_builtin_package_enabled_for_shared_settings_requires_supported_provider() -> None:
    """Search-backed packages should stay disabled until a search provider is selected."""
    spec = _spec("search")

    def get_setting(key, default=None):
        return {spec.shared_setting_key: True}.get(key, default)

    assert (
        is_builtin_package_enabled_for_shared_settings(
            spec,
            get_setting,
            search_provider="none",
        )
        is False
    )
    assert (
        is_builtin_package_enabled_for_shared_settings(
            spec,
            get_setting,
            search_provider="duckduckgo",
        )
        is True
    )


def test_builtin_package_enabled_for_profile_combines_shared_and_profile_state() -> None:
    """Profile enablement should require both shared and profile toggles."""
    spec = _spec("read_url")
    shared_settings = {spec.shared_setting_key: True}

    assert (
        is_builtin_package_enabled_for_profile(
            spec,
            lambda key, default=None: shared_settings.get(key, default),
            lambda key, default=None: {spec.profile_setting_key: True}.get(key, default),
            search_provider="duckduckgo",
        )
        is True
    )
    assert (
        is_builtin_package_enabled_for_profile(
            spec,
            lambda key, default=None: shared_settings.get(key, default),
            lambda key, default=None: {spec.profile_setting_key: False}.get(
                key, default
            ),
            search_provider="duckduckgo",
        )
        is False
    )
