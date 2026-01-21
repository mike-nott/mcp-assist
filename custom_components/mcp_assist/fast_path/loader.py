"""YAML loader for Fast Path keywords and responses."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)

# Path to the YAML files
YAML_DIR = Path(__file__).parent


def get_available_languages() -> list[str]:
    """Get list of available language codes from keyword files."""
    keywords_dir = YAML_DIR / "keywords"
    languages = []
    
    if keywords_dir.exists():
        for file_path in keywords_dir.glob("*.yaml"):
            if file_path.stem != "custom":
                languages.append(file_path.stem)
    
    return sorted(languages)


class KeywordLoader:
    """Loader for Fast Path keywords and responses."""

    def __init__(self, language: str = "en") -> None:
        """Initialize the keyword loader.
        
        Args:
            language: The language code (e.g., 'de', 'en', 'fr')
        """
        self._language = language
        self._keywords: dict[str, Any] = {}
        self._responses: dict[str, Any] = {}
        self._custom_keywords: dict[str, Any] = {}
        self._loaded = False

    @property
    def language(self) -> str:
        """Return current language."""
        return self._language

    @language.setter
    def language(self, value: str) -> None:
        """Set language and reload if changed."""
        if value != self._language:
            self._language = value
            self._loaded = False

    def load(self) -> None:
        """Load keywords and responses from YAML files."""
        self._keywords = self._load_yaml("keywords", self._language)
        self._responses = self._load_yaml("responses", self._language)
        self._custom_keywords = self._load_custom_keywords()
        self._merge_custom_keywords()
        self._loaded = True
        
        _LOGGER.debug(
            "Loaded Fast Path keywords for language '%s': %d actions",
            self._language,
            len(self._keywords.get("actions", {}))
        )

    def _load_yaml(self, folder: str, filename: str) -> dict[str, Any]:
        """Load a YAML file from the specified folder.
        
        Args:
            folder: The folder name ('keywords' or 'responses')
            filename: The filename without extension
            
        Returns:
            The parsed YAML content or empty dict on error
        """
        file_path = YAML_DIR / folder / f"{filename}.yaml"
        
        if not file_path.exists():
            _LOGGER.warning("YAML file not found: %s", file_path)
            # Fallback to English
            if filename != "en":
                return self._load_yaml(folder, "en")
            return {}
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content else {}
        except yaml.YAMLError as e:
            _LOGGER.error("Error parsing YAML file %s: %s", file_path, e)
            return {}
        except OSError as e:
            _LOGGER.error("Error reading YAML file %s: %s", file_path, e)
            return {}

    def _load_custom_keywords(self) -> dict[str, Any]:
        """Load custom keywords if the file exists."""
        custom_path = YAML_DIR / "keywords" / "custom.yaml"
        
        if not custom_path.exists():
            return {}
        
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content else {}
        except (yaml.YAMLError, OSError) as e:
            _LOGGER.warning("Error loading custom keywords: %s", e)
            return {}

    def _merge_custom_keywords(self) -> None:
        """Merge custom keywords into the main keywords dict."""
        if not self._custom_keywords:
            return
        
        custom_actions = self._custom_keywords.get("actions", {})
        
        for action_name, lang_keywords in custom_actions.items():
            if self._language in lang_keywords:
                if action_name not in self._keywords.get("actions", {}):
                    if "actions" not in self._keywords:
                        self._keywords["actions"] = {}
                    self._keywords["actions"][action_name] = []
                
                # Add custom keywords to existing list
                existing = self._keywords["actions"].get(action_name, [])
                custom = lang_keywords[self._language]
                self._keywords["actions"][action_name] = list(set(existing + custom))
        
        # Merge custom value keywords (e.g., color_temp)
        custom_values = self._custom_keywords.get("values", {})
        if self._language in custom_values.get("color_temp", {}):
            if "values" not in self._keywords:
                self._keywords["values"] = {}
            if "color_temp" not in self._keywords["values"]:
                self._keywords["values"]["color_temp"] = {"keywords": []}
            
            existing_keywords = self._keywords["values"]["color_temp"].get("keywords", [])
            custom_color_keywords = custom_values["color_temp"][self._language]
            self._keywords["values"]["color_temp"]["keywords"] = existing_keywords + custom_color_keywords

    def get_action_keywords(self, action: str) -> list[str]:
        """Get keywords for a specific action.
        
        Args:
            action: The action name (e.g., 'turn_on', 'turn_off')
            
        Returns:
            List of keywords for this action
        """
        if not self._loaded:
            self.load()
        
        return self._keywords.get("actions", {}).get(action, [])

    def get_all_action_keywords(self) -> dict[str, list[str]]:
        """Get all action keywords.
        
        Returns:
            Dict mapping action names to keyword lists
        """
        if not self._loaded:
            self.load()
        
        return self._keywords.get("actions", {})

    def get_value_patterns(self, value_type: str) -> list[str]:
        """Get regex patterns for a specific value type.
        
        Args:
            value_type: The value type (e.g., 'brightness', 'temperature')
            
        Returns:
            List of regex patterns
        """
        if not self._loaded:
            self.load()
        
        return self._keywords.get("values", {}).get(value_type, {}).get("patterns", [])

    def get_color_temp_keywords(self) -> list[dict[str, Any]]:
        """Get color temperature keywords with their values.
        
        Returns:
            List of keyword dicts with 'keyword', 'aliases', and 'value'
        """
        if not self._loaded:
            self.load()
        
        return self._keywords.get("values", {}).get("color_temp", {}).get("keywords", [])

    def get_response(self, action: str, count: int = 1) -> str:
        """Get the response template for an action.
        
        Args:
            action: The action name
            count: Number of entities affected
            
        Returns:
            The response template string
        """
        if not self._loaded:
            self.load()
        
        responses = self._responses.get("responses", {}).get(action, {})
        
        if count == 1:
            return responses.get("single", "Done.")
        else:
            return responses.get("multiple", "Done.")

    def get_error_response(self, error_type: str) -> str:
        """Get an error response template.
        
        Args:
            error_type: The error type (e.g., 'no_entity', 'action_failed')
            
        Returns:
            The error response template
        """
        if not self._loaded:
            self.load()
        
        return self._responses.get("errors", {}).get(error_type, "An error occurred.")

    def reload(self) -> None:
        """Force reload of all YAML files."""
        self._loaded = False
        self.load()
