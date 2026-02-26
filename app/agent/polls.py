"""
Polls — Cross-channel poll creation and management.

Supports creating polls with multiple options across channels
that support them (Telegram, Discord, WhatsApp, Slack, Teams).
Tracks votes and results in a unified model.

Usage:
    from app.agent.polls import get_poll_manager

    mgr = get_poll_manager()
    poll = mgr.create_poll(
        question="Which framework?",
        options=["FastAPI", "Flask", "Django"],
        channel_type="telegram",
        channel_id="123",
    )
    mgr.cast_vote(poll.poll_id, voter_id="user1", option_index=0)
    results = mgr.get_results(poll.poll_id)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PollType(str, Enum):
    SINGLE_CHOICE = "single"
    MULTIPLE_CHOICE = "multiple"
    QUIZ = "quiz"  # One correct answer


class PollState(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    DELETED = "deleted"


@dataclass
class PollOption:
    """A single poll option."""
    index: int
    text: str
    voter_ids: Set[str] = field(default_factory=set)
    is_correct: bool = False  # For quiz polls

    @property
    def vote_count(self) -> int:
        return len(self.voter_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "vote_count": self.vote_count,
        }


@dataclass
class Poll:
    """A poll with options and votes."""
    poll_id: str
    question: str
    options: List[PollOption]
    channel_type: str
    channel_id: str
    poll_type: PollType = PollType.SINGLE_CHOICE
    state: PollState = PollState.ACTIVE
    creator_id: str = ""
    anonymous: bool = True
    created_at: float = 0.0
    closed_at: Optional[float] = None
    external_id: Optional[str] = None  # Channel-native poll ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def total_votes(self) -> int:
        voters = set()
        for opt in self.options:
            voters.update(opt.voter_ids)
        return len(voters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poll_id": self.poll_id,
            "question": self.question,
            "options": [o.to_dict() for o in self.options],
            "poll_type": self.poll_type.value,
            "state": self.state.value,
            "channel_type": self.channel_type,
            "total_votes": self.total_votes,
            "anonymous": self.anonymous,
        }

    def results(self) -> Dict[str, Any]:
        """Get poll results with percentages."""
        total = self.total_votes
        option_results = []
        for opt in self.options:
            pct = round(opt.vote_count / total * 100, 1) if total else 0.0
            option_results.append({
                "index": opt.index,
                "text": opt.text,
                "votes": opt.vote_count,
                "percentage": pct,
            })
        # Sort by votes descending
        option_results.sort(key=lambda x: x["votes"], reverse=True)
        return {
            "poll_id": self.poll_id,
            "question": self.question,
            "total_votes": total,
            "options": option_results,
            "state": self.state.value,
        }


class PollManager:
    """
    Manages polls across all channels.

    Handles creation, voting, closing, and result aggregation.
    """

    def __init__(self):
        self._polls: Dict[str, Poll] = {}
        self._counter: int = 0

    def create_poll(
        self,
        question: str,
        options: List[str],
        channel_type: str,
        channel_id: str,
        *,
        poll_type: PollType = PollType.SINGLE_CHOICE,
        creator_id: str = "",
        anonymous: bool = True,
        correct_option: Optional[int] = None,
    ) -> Poll:
        """
        Create a new poll.

        Args:
            question: The poll question.
            options: List of option texts.
            channel_type: Channel where poll is posted.
            channel_id: Channel ID.
            poll_type: single, multiple, or quiz.
            correct_option: Index of correct answer (for quiz).
        """
        if len(options) < 2:
            raise ValueError("Poll must have at least 2 options")
        if len(options) > 10:
            raise ValueError("Poll can have at most 10 options")

        self._counter += 1
        poll_id = f"poll_{self._counter}"

        poll_options = []
        for i, text in enumerate(options):
            is_correct = (i == correct_option) if correct_option is not None else False
            poll_options.append(PollOption(index=i, text=text, is_correct=is_correct))

        poll = Poll(
            poll_id=poll_id,
            question=question,
            options=poll_options,
            channel_type=channel_type,
            channel_id=channel_id,
            poll_type=poll_type,
            creator_id=creator_id,
            anonymous=anonymous,
        )

        self._polls[poll_id] = poll
        logger.info(f"[POLL] Created {poll_id}: {question} ({len(options)} options)")
        return poll

    def cast_vote(
        self,
        poll_id: str,
        voter_id: str,
        option_index: int,
    ) -> bool:
        """
        Cast a vote on a poll.

        Returns True if vote was accepted.
        """
        poll = self._polls.get(poll_id)
        if not poll:
            raise ValueError(f"Poll {poll_id} not found")

        if poll.state != PollState.ACTIVE:
            raise ValueError(f"Poll {poll_id} is {poll.state.value}")

        if option_index < 0 or option_index >= len(poll.options):
            raise ValueError(f"Invalid option index {option_index}")

        # For single choice, remove any previous vote
        if poll.poll_type == PollType.SINGLE_CHOICE:
            for opt in poll.options:
                opt.voter_ids.discard(voter_id)

        poll.options[option_index].voter_ids.add(voter_id)
        return True

    def retract_vote(
        self,
        poll_id: str,
        voter_id: str,
        option_index: Optional[int] = None,
    ) -> bool:
        """Retract a vote. If option_index is None, retract all votes."""
        poll = self._polls.get(poll_id)
        if not poll:
            return False

        if option_index is not None:
            poll.options[option_index].voter_ids.discard(voter_id)
        else:
            for opt in poll.options:
                opt.voter_ids.discard(voter_id)
        return True

    def close_poll(self, poll_id: str) -> Optional[Dict[str, Any]]:
        """Close a poll and return final results."""
        poll = self._polls.get(poll_id)
        if not poll:
            return None

        poll.state = PollState.CLOSED
        poll.closed_at = time.time()
        return poll.results()

    def delete_poll(self, poll_id: str) -> bool:
        """Delete a poll."""
        poll = self._polls.pop(poll_id, None)
        return poll is not None

    def get_poll(self, poll_id: str) -> Optional[Poll]:
        """Get a poll by ID."""
        return self._polls.get(poll_id)

    def get_results(self, poll_id: str) -> Optional[Dict[str, Any]]:
        """Get current poll results."""
        poll = self._polls.get(poll_id)
        if not poll:
            return None
        return poll.results()

    def list_polls(
        self,
        channel_type: Optional[str] = None,
        channel_id: Optional[str] = None,
        *,
        state: Optional[PollState] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List polls with optional filters."""
        polls = list(self._polls.values())

        if channel_type:
            polls = [p for p in polls if p.channel_type == channel_type]
        if channel_id:
            polls = [p for p in polls if p.channel_id == channel_id]
        if state:
            polls = [p for p in polls if p.state == state]

        polls.sort(key=lambda p: p.created_at, reverse=True)
        return [p.to_dict() for p in polls[:limit]]

    def stats(self) -> Dict[str, Any]:
        """Get poll statistics."""
        active = sum(1 for p in self._polls.values() if p.state == PollState.ACTIVE)
        closed = sum(1 for p in self._polls.values() if p.state == PollState.CLOSED)
        total_votes = sum(p.total_votes for p in self._polls.values())

        return {
            "total_polls": len(self._polls),
            "active": active,
            "closed": closed,
            "total_votes": total_votes,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[PollManager] = None


def get_poll_manager() -> PollManager:
    """Get the global poll manager."""
    global _manager
    if _manager is None:
        _manager = PollManager()
    return _manager
