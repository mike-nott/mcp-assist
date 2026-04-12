"""Tests for conversation history management."""

from __future__ import annotations

from datetime import datetime, timedelta

from custom_components.mcp_assist.conversation_history import ConversationHistory


def test_recent_context_includes_action_summaries() -> None:
    """Recent context should include summarized tool activity."""
    history = ConversationHistory()
    history.add_turn(
        "abc",
        "Turn on the porch light",
        "Done.",
        actions=[
            {
                "type": "intent_executed",
                "intent": "HassTurnOn",
                "entity_ids": ["light.porch"],
            }
        ],
    )

    context = history.get_recent_context("abc")

    assert "User: Turn on the porch light" in context
    assert "Assistant: Done." in context
    assert "Executed HassTurnOn on light.porch" in context


def test_cleanup_limits_turns_and_removes_old_entries() -> None:
    """Cleanup should enforce max turns and max age."""
    history = ConversationHistory(max_history_age_hours=1, max_turns_per_conversation=2)
    history.add_turn("abc", "one", "one")
    history.add_turn("abc", "two", "two")
    history.add_turn("abc", "three", "three")

    assert [turn["user"] for turn in history.get_history("abc")] == ["two", "three"]

    history._conversations["abc"][0]["timestamp"] = datetime.now() - timedelta(hours=2)
    assert [turn["user"] for turn in history.get_history("abc")] == ["three"]


def test_get_stats_reports_counts_and_timestamps() -> None:
    """Stats should reflect stored conversation state."""
    history = ConversationHistory()
    history.add_turn("first", "hi", "hello")
    history.add_turn("second", "status", "all good")

    stats = history.get_stats()

    assert stats["total_conversations"] == 2
    assert stats["total_turns"] == 2
    assert stats["average_turns_per_conversation"] == 1.0
    assert stats["oldest_turn"] is not None
    assert stats["newest_turn"] is not None
