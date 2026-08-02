"""Conversation history management for LM Studio MCP."""

import logging
from typing import Any, Dict, List, Optional
from datetime import timedelta

from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class ConversationHistory:
    """Manage conversation history for multi-turn conversations."""

    def __init__(
        self,
        max_history_age_hours: int = 24,
        max_turns_per_conversation: int = 20,
        max_conversations: int = 200,
    ) -> None:
        """Initialize conversation history manager."""
        self.max_history_age = timedelta(hours=max_history_age_hours)
        self.max_turns = max_turns_per_conversation
        self.max_conversations = max_conversations
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}

    def add_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        actions: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add a conversation turn."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        turn = {
            "timestamp": dt_util.utcnow(),
            "user": user_message,
            "assistant": assistant_message,
            "actions": actions or []
        }

        self._conversations[conversation_id].append(turn)

        # Cleanup this conversation, then sweep abandoned ones so a stream of
        # one-shot voice sessions (each a fresh conversation id) can't grow the
        # store without bound.
        self._cleanup_conversation(conversation_id)
        self._prune_all_conversations()

        _LOGGER.debug(
            "Added turn to conversation %s (now %d turns)",
            conversation_id,
            len(self._conversations[conversation_id])
        )

    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a conversation ID."""
        if conversation_id not in self._conversations:
            return []

        # Cleanup old conversations first
        self._cleanup_conversation(conversation_id)

        return self._conversations[conversation_id].copy()

    def get_recent_context(self, conversation_id: str, max_turns: int = 3) -> str:
        """Get recent conversation context as formatted string."""
        history = self.get_history(conversation_id)

        if not history:
            return ""

        # Get last few turns
        recent_turns = history[-max_turns:] if len(history) > max_turns else history

        context_parts = ["Recent conversation:"]
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")

            # Add actions taken if any
            if turn.get("actions"):
                actions_summary = []
                for action in turn["actions"]:
                    if action.get("type") == "intent_executed":
                        actions_summary.append(f"Executed {action['intent']} on {', '.join(action.get('entity_ids', []))}")
                    elif action.get("type") == "entities_mentioned":
                        actions_summary.append(f"Referenced entities: {', '.join(action.get('entity_ids', []))}")

                if actions_summary:
                    context_parts.append(f"Actions: {'; '.join(actions_summary)}")

        return "\n".join(context_parts)

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear history for a specific conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            _LOGGER.debug("Cleared conversation history for %s", conversation_id)

    def clear_all(self) -> None:
        """Clear all conversation history."""
        self._conversations.clear()
        _LOGGER.debug("Cleared all conversation history")

    def _cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up old turns from a conversation."""
        if conversation_id not in self._conversations:
            return

        conversation = self._conversations[conversation_id]
        cutoff_time = dt_util.utcnow() - self.max_history_age

        # Remove old turns
        conversation[:] = [
            turn for turn in conversation
            if turn["timestamp"] > cutoff_time
        ]

        # Limit number of turns
        if len(conversation) > self.max_turns:
            conversation[:] = conversation[-self.max_turns:]

        # Remove conversation if empty
        if not conversation:
            del self._conversations[conversation_id]

    def _prune_all_conversations(self) -> None:
        """Drop expired turns from every conversation and cap the total count.

        _cleanup_conversation only ever touches the conversation being read or
        written, so abandoned conversation ids (Home Assistant assigns a fresh
        one per voice session) would otherwise linger in memory until a manual
        clear. This sweeps all of them and enforces a hard ceiling.
        """
        cutoff_time = dt_util.utcnow() - self.max_history_age

        for conversation_id in list(self._conversations):
            conversation = self._conversations[conversation_id]
            conversation[:] = [
                turn for turn in conversation if turn["timestamp"] > cutoff_time
            ]
            if len(conversation) > self.max_turns:
                conversation[:] = conversation[-self.max_turns:]
            if not conversation:
                del self._conversations[conversation_id]

        # Hard cap: evict the least-recently-active conversations.
        overflow = len(self._conversations) - self.max_conversations
        if overflow > 0:
            ordered = sorted(
                self._conversations.items(),
                key=lambda item: item[1][-1]["timestamp"],
            )
            for conversation_id, _turns in ordered[:overflow]:
                del self._conversations[conversation_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation history."""
        total_conversations = len(self._conversations)
        total_turns = sum(len(turns) for turns in self._conversations.values())

        # Calculate average turns per conversation
        avg_turns = total_turns / total_conversations if total_conversations > 0 else 0

        # Find oldest and newest turns
        all_timestamps = []
        for conversation in self._conversations.values():
            all_timestamps.extend(turn["timestamp"] for turn in conversation)

        oldest = min(all_timestamps) if all_timestamps else None
        newest = max(all_timestamps) if all_timestamps else None

        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "average_turns_per_conversation": round(avg_turns, 1),
            "oldest_turn": oldest.isoformat() if oldest else None,
            "newest_turn": newest.isoformat() if newest else None,
        }