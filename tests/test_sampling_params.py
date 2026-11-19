"""Payload tests for models that reject sampling parameters.

Home Assistant is stubbed so the test runs without installing it; only the
payload-building methods of the conversation entity are exercised.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _HomeAssistantStubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve any homeassistant.* import as a stub module."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "homeassistant" or fullname.startswith("homeassistant."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _load_agent():
    sys.meta_path.insert(0, _HomeAssistantStubs())
    conversation = importlib.import_module("homeassistant.components.conversation")
    conversation.ConversationEntity = type("ConversationEntity", (), {})
    return importlib.import_module("custom_components.mcp_assist.agent")


agent_module = _load_agent()
const = importlib.import_module("custom_components.mcp_assist.const")


def _payload(server_type, model, temperature=0.5):
    entity = object.__new__(agent_module.MCPAssistConversationEntity)
    entity.server_type = server_type
    entity.base_url = "https://example.invalid"
    entity.entry = SimpleNamespace(
        options={
            const.CONF_MODEL_NAME: model,
            const.CONF_TEMPERATURE: temperature,
        },
        data={},
    )
    return entity._build_openai_payload([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-opus-5-5-20261101",
        "claude-opus-4-7",
        "claude-opus-4.8",
        "claude-opus-4-8-20260801",
        "Claude-Opus-5-5",
    ],
)
def test_anthropic_opus_47_plus_omits_temperature(model):
    assert "temperature" not in _payload(const.SERVER_TYPE_ANTHROPIC, model)


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-3-opus-20240229",
    ],
)
def test_other_anthropic_models_keep_temperature(model):
    assert _payload(const.SERVER_TYPE_ANTHROPIC, model, 0.3)["temperature"] == 0.3


def test_same_model_name_on_other_providers_is_unchanged():
    payload = _payload(const.SERVER_TYPE_LMSTUDIO, "claude-opus-5-5")
    assert payload["temperature"] == 0.5


def test_gemini_still_omits_temperature():
    assert "temperature" not in _payload(const.SERVER_TYPE_GEMINI, "gemini-3.6-pro")
