"""Tests for persisted MCP Assist memory storage."""

from __future__ import annotations

from datetime import timedelta

import pytest

from homeassistant.util import dt as dt_util

from custom_components.mcp_assist.memory_manager import MemoryManager


@pytest.mark.asyncio
async def test_memory_manager_remember_and_recall(hass) -> None:
    """Memories should persist in memory manager state and be searchable."""
    manager = MemoryManager(hass)
    await manager.async_initialize()

    stored = await manager.remember(
        "Jason prefers pour-over coffee",
        default_ttl_days=30,
        max_ttl_days=365,
        category="preference",
        max_items=100,
    )
    recalled = await manager.recall(query="coffee", limit=5)

    assert stored["id"]
    assert recalled["total_found"] == 1
    assert recalled["items"][0]["text"] == "Jason prefers pour-over coffee"


@pytest.mark.asyncio
async def test_memory_manager_purges_expired_memories(hass) -> None:
    """Expired memories should be removed during initialization/recall."""
    manager = MemoryManager(hass)
    expired_time = (dt_util.utcnow() - timedelta(days=1)).isoformat()
    manager._loaded = True
    manager._memories = [
        {
            "id": "expired123",
            "text": "old memory",
            "category": None,
            "created_at": (dt_util.utcnow() - timedelta(days=2)).isoformat(),
            "expires_at": expired_time,
            "ttl_days": 1,
        }
    ]

    recalled = await manager.recall(limit=5)

    assert recalled["total_found"] == 0
    assert manager._memories == []


@pytest.mark.asyncio
async def test_memory_manager_forget_by_query(hass) -> None:
    """Forget should delete the best matching memory when no id is given."""
    manager = MemoryManager(hass)
    await manager.async_initialize()
    await manager.remember(
        "The guest room thermostat should stay at 68",
        default_ttl_days=30,
        max_ttl_days=365,
        category="preference",
        max_items=100,
    )

    deleted = await manager.forget(query="guest room thermostat")
    recalled = await manager.recall(query="guest room thermostat", limit=5)

    assert deleted["deleted_count"] == 1
    assert recalled["total_found"] == 0


@pytest.mark.asyncio
async def test_memory_manager_prunes_to_max_items(hass) -> None:
    """Only the newest configured number of memories should be retained."""
    manager = MemoryManager(hass)
    await manager.async_initialize()

    await manager.remember(
        "first",
        default_ttl_days=30,
        max_ttl_days=365,
        max_items=2,
    )
    await manager.remember(
        "second",
        default_ttl_days=30,
        max_ttl_days=365,
        max_items=2,
    )
    third = await manager.remember(
        "third",
        default_ttl_days=30,
        max_ttl_days=365,
        max_items=2,
    )

    items = await manager.list_all()

    assert third["pruned_count"] == 1
    assert [item["text"] for item in items] == ["third", "second"]
