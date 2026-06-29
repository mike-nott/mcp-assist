"""Persistent memory storage for MCP Assist."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import Any
from uuid import uuid4

from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}_memory_store"


class MemoryManager:
    """Manage shared persisted memories for MCP Assist."""

    def __init__(self, hass) -> None:
        """Initialize the memory manager."""
        self.hass = hass
        self._store: Store[dict[str, Any]] = Store(
            hass,
            _STORAGE_VERSION,
            _STORAGE_KEY,
        )
        self._lock = asyncio.Lock()
        self._loaded = False
        self._memories: list[dict[str, Any]] = []

    async def async_initialize(self) -> None:
        """Load persisted memories and purge expired entries."""
        async with self._lock:
            await self._ensure_loaded_locked()
            changed = self._purge_expired_locked()
            if changed:
                await self._save_locked()

    async def async_shutdown(self) -> None:
        """Flush memory state if needed."""
        return None

    async def remember(
        self,
        text: str,
        *,
        default_ttl_days: int,
        max_ttl_days: int,
        ttl_days: int | None = None,
        category: str | None = None,
        max_items: int = 500,
    ) -> dict[str, Any]:
        """Store a memory with TTL and pruning."""
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise ValueError("memory text is required")

        normalized_category = self._normalize_text(category)
        effective_max_ttl = max(1, int(max_ttl_days))
        effective_default_ttl = max(1, min(int(default_ttl_days), effective_max_ttl))
        effective_ttl = effective_default_ttl if ttl_days is None else int(ttl_days)
        effective_ttl = max(1, min(effective_ttl, effective_max_ttl))
        effective_max_items = max(1, int(max_items))

        async with self._lock:
            await self._ensure_loaded_locked()
            self._purge_expired_locked()

            now = dt_util.utcnow()
            expires_at = now + timedelta(days=effective_ttl)
            memory = {
                "id": uuid4().hex[:12],
                "text": normalized_text,
                "category": normalized_category,
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "ttl_days": effective_ttl,
            }
            self._memories.append(memory)
            pruned_count = self._prune_locked(max_items=effective_max_items)
            await self._save_locked()

        return {
            **memory,
            "pruned_count": pruned_count,
        }

    async def recall(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Recall active memories by query or category."""
        normalized_query = self._normalize_text(query)
        normalized_category = self._normalize_text(category)
        effective_limit = max(1, int(limit))

        async with self._lock:
            await self._ensure_loaded_locked()
            changed = self._purge_expired_locked()
            memories = list(self._memories)
            if changed:
                await self._save_locked()

        ranked = self._rank_memories(
            memories,
            query=normalized_query,
            category=normalized_category,
        )
        items = ranked[:effective_limit]
        return {
            "items": items,
            "total_found": len(ranked),
            "returned_count": len(items),
            "remaining_count": max(0, len(ranked) - len(items)),
        }

    async def forget(
        self,
        *,
        memory_id: str | None = None,
        query: str | None = None,
        category: str | None = None,
        delete_all_matches: bool = False,
    ) -> dict[str, Any]:
        """Forget one or more memories by id or search criteria."""
        normalized_id = self._normalize_text(memory_id)
        normalized_query = self._normalize_text(query)
        normalized_category = self._normalize_text(category)
        if not normalized_id and not normalized_query and not normalized_category:
            raise ValueError("memory_id or query/category is required")

        async with self._lock:
            await self._ensure_loaded_locked()
            self._purge_expired_locked()
            target_ids: set[str]
            deleted: list[dict[str, Any]] = []

            if normalized_id:
                target_ids = {normalized_id}
            else:
                ranked = self._rank_memories(
                    list(self._memories),
                    query=normalized_query,
                    category=normalized_category,
                )
                if not ranked:
                    return {
                        "deleted": [],
                        "deleted_count": 0,
                    }
                if delete_all_matches:
                    target_ids = {item["id"] for item in ranked}
                else:
                    target_ids = {ranked[0]["id"]}

            kept: list[dict[str, Any]] = []
            for memory in self._memories:
                if memory["id"] in target_ids:
                    deleted.append(dict(memory))
                else:
                    kept.append(memory)

            self._memories = kept
            if deleted:
                await self._save_locked()

        return {
            "deleted": deleted,
            "deleted_count": len(deleted),
        }

    async def list_all(self) -> list[dict[str, Any]]:
        """Return all active memories, newest first."""
        async with self._lock:
            await self._ensure_loaded_locked()
            changed = self._purge_expired_locked()
            items = sorted(
                (dict(item) for item in self._memories),
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
            if changed:
                await self._save_locked()
            return items

    async def _ensure_loaded_locked(self) -> None:
        """Load persisted storage on first use."""
        if self._loaded:
            return

        stored = await self._store.async_load()
        raw_items = stored.get("items", []) if isinstance(stored, dict) else []
        self._memories = [
            normalized
            for item in raw_items
            if (normalized := self._normalize_loaded_memory(item)) is not None
        ]
        self._loaded = True

    async def _save_locked(self) -> None:
        """Persist the current memory list."""
        await self._store.async_save({"items": list(self._memories)})

    def _normalize_loaded_memory(self, item: Any) -> dict[str, Any] | None:
        """Normalize a loaded memory entry from storage."""
        if not isinstance(item, dict):
            return None

        memory_id = self._normalize_text(item.get("id"))
        text = self._normalize_text(item.get("text"))
        created_at = self._normalize_text(item.get("created_at"))
        expires_at = self._normalize_text(item.get("expires_at"))
        if not memory_id or not text or not created_at or not expires_at:
            return None

        return {
            "id": memory_id,
            "text": text,
            "category": self._normalize_text(item.get("category")),
            "created_at": created_at,
            "expires_at": expires_at,
            "ttl_days": int(item.get("ttl_days") or 0),
        }

    def _purge_expired_locked(self) -> bool:
        """Remove expired memories and report whether anything changed."""
        now = dt_util.utcnow()
        before = len(self._memories)
        kept: list[dict[str, Any]] = []
        for memory in self._memories:
            expires_at = dt_util.parse_datetime(str(memory.get("expires_at") or ""))
            if expires_at is None:
                continue
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=dt_util.UTC)
            if expires_at > now:
                kept.append(memory)
        self._memories = kept
        return len(self._memories) != before

    def _prune_locked(self, *, max_items: int) -> int:
        """Trim the memory store to the newest items."""
        if len(self._memories) <= max_items:
            return 0

        ordered = sorted(
            self._memories,
            key=lambda item: item.get("created_at", ""),
            reverse=True,
        )
        self._memories = ordered[:max_items]
        return max(0, len(ordered) - len(self._memories))

    def _rank_memories(
        self,
        memories: list[dict[str, Any]],
        *,
        query: str | None,
        category: str | None,
    ) -> list[dict[str, Any]]:
        """Rank memories by relevance and recency."""
        if not memories:
            return []

        filtered = memories
        if category:
            filtered = [
                item
                for item in filtered
                if self._normalize_text(item.get("category")) == category
            ]

        if not query:
            return sorted(
                (dict(item) for item in filtered),
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )

        query_lower = query.casefold()
        terms = [term for term in re.findall(r"[a-z0-9]+", query_lower) if term]
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for memory in filtered:
            haystack_parts = [
                str(memory.get("text") or ""),
                str(memory.get("category") or ""),
            ]
            haystack = " ".join(haystack_parts).casefold()
            score = 0
            if query_lower in haystack:
                score += 100
            for term in terms:
                if term in haystack:
                    score += 10
            if terms and all(term in haystack for term in terms):
                score += 25
            if not score:
                continue
            ranked.append((score, str(memory.get("created_at") or ""), dict(memory)))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked]

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        """Normalize a text value for storage and search."""
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None
