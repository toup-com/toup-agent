"""One namespace for every image in a conversation, and one rule for what "it"
means.

Round 16 shipped an edit that composited the user's own face into a science
fiction laboratory. The user had asked for a picture of two cartoon characters,
got one, and then said "make morty playing with the portal machine". The edit
went to a selfie they had uploaded days earlier, in a different chat.

Nothing malfunctioned. Every step did what it was written to do:

* `edit_image`'s source resolver read ``Message.attachments`` filtered to
  ``role == "user"``. A generated image is persisted on the **assistant**
  message (``agent_runner._save_session`` sets ``attachments`` on the assistant
  row), so the picture the agent had just made was structurally invisible to
  the only lookup that could have found it.
* That same query joined ``Conversation`` **only to filter by user_id** and
  then ordered by ``Message.created_at`` across every conversation the user
  has ever had. It was not a "current image" pointer; it was a global "newest
  photo this human ever uploaded, anywhere" pointer.
* The tool's own description told the model not to pass an explicit source:
  "just CALL this tool — it automatically finds their most recently uploaded
  image. Do NOT ask them to re-upload first."

So the model was steered away from being specific, and the fallback it was
steered into could only ever return an upload. There was no path by which the
just-generated image could win. The user's face was not an accident of ranking;
it was the only candidate in the set.

What this module changes
------------------------
* **Both roles.** An image is an image. Generated, edited and uploaded ones
  live in the same namespace and rank against each other by time.
* **Thread-scoped, never global.** Candidates come from the conversation the
  turn is running in — the same ``conversation_id == session_id`` scope
  ``AgentRunner._load_history`` uses, so what resolves as "it" is drawn from
  exactly the transcript the model is reasoning over. A photo in another chat
  is not a candidate, and when the thread has no image the caller asks rather
  than reaching for one.
* **Stable ids.** ``Attachment.id`` (hex32, minted in ``doc_generators._persist``)
  already identified every one of these files; nothing surfaced it to the model.
  ``generate_image``/``edit_image`` now return it and ``edit_image`` accepts it,
  so a follow-up edit can be exact instead of inferential.
* **Origin, in words.** Every resolution reports what it picked and where it
  came from, so a wrong pick is visible in the turn instead of arriving as a
  surprise in the picture.

The registry is the message log itself. There is no second store to fall out of
sync with it — see `thread_images`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Origin labels. ``uploaded`` is the user's file; ``generated`` and ``edited``
#: are ours, split because the distinction is what the phrase "the one you just
#: made" is pointing at.
ORIGIN_UPLOADED = "uploaded"
ORIGIN_GENERATED = "generated"
ORIGIN_EDITED = "edited"
ORIGIN_EXPLICIT = "explicit"

#: `_tool_edit_image` names its outputs `edited_<hex8>.png`. Everything else an
#: assistant message carries is something we generated. A user-supplied
#: `filename` on the tool call can defeat this, which costs a slightly wrong
#: word in one sentence and nothing else.
_EDITED_NAME_RE = re.compile(r"^edited[-_]", re.IGNORECASE)

#: Bound on the history scan. A conversation with more images than this has an
#: older one that nobody means by "it".
_SCAN_LIMIT = 60


def _is_image(att: Any) -> bool:
    return (
        isinstance(att, dict)
        and str(att.get("mime_type", "")).startswith("image/")
        and bool(att.get("storage_path"))
    )


@dataclass(frozen=True)
class ImageArtifact:
    """One image, plus where it came from.

    ``attachment`` is the persisted dict shape
    (``{id, filename, mime_type, storage_path, ...}``) that both the inbound
    upload path and ``doc_generators._persist`` produce, so a caller can load
    the bytes the same way whichever end it came from.
    """

    attachment: Dict[str, Any]
    origin: str
    role: str = ""
    created_at: Optional[datetime] = None
    turn_scope: str = "history"  # "history" | "this_turn"

    @property
    def id(self) -> str:
        return str(self.attachment.get("id") or "")

    @property
    def filename(self) -> str:
        return str(self.attachment.get("filename") or "image.png")

    @property
    def storage_path(self) -> str:
        return str(self.attachment.get("storage_path") or "")

    @property
    def mime_type(self) -> str:
        return str(self.attachment.get("mime_type") or "image/png")

    def describe(self) -> str:
        """One clause naming this image, for the model-facing tool result.

        Written to be pasteable into a sentence the agent says to the user, so
        a wrong resolution reads wrong out loud rather than hiding in an id.
        """
        when = "earlier in this turn" if self.turn_scope == "this_turn" else "earlier in this conversation"
        if self.origin == ORIGIN_EXPLICIT:
            return f"the image you named explicitly ({self.filename})"
        if self.origin == ORIGIN_GENERATED:
            return f"the picture you generated {when} ({self.filename})"
        if self.origin == ORIGIN_EDITED:
            return f"the edited picture from {when} ({self.filename})"
        if self.turn_scope == "this_turn":
            return f"the photo the user attached to this message ({self.filename})"
        return f"the photo the user uploaded {when} ({self.filename})"


def origin_for(att: Dict[str, Any], role: str) -> str:
    """Origin of a persisted attachment, from the role of the message it rode in on."""
    if role == "user":
        return ORIGIN_UPLOADED
    name = str(att.get("filename") or "")
    return ORIGIN_EDITED if _EDITED_NAME_RE.match(name) else ORIGIN_GENERATED


def turn_artifacts(
    pending_attachments: Sequence[Any],
    inbound_media: Sequence[Any],
) -> List[ImageArtifact]:
    """Images belonging to the CURRENT turn, newest-intent first.

    Nothing from this turn is in the database yet — the assistant message is
    written once, at the end (`agent_runner._save_session`) — so a turn that
    generates a picture and then edits it in the same response would find an
    empty thread if this did not exist.

    Order is deliberate and is not chronological. A file the user attached to
    *this* message outranks one we produced a moment ago while answering it:
    "here's my photo, put me on a beach" is unambiguous, and it is the case
    where guessing wrong is worst. Below that, our own newest output wins.
    """
    out: List[ImageArtifact] = []
    for att in reversed(list(inbound_media or ())):
        if _is_image(att):
            out.append(ImageArtifact(
                attachment=dict(att), origin=ORIGIN_UPLOADED,
                role="user", turn_scope="this_turn",
            ))
    for att in reversed(list(pending_attachments or ())):
        if _is_image(att):
            out.append(ImageArtifact(
                attachment=dict(att), origin=origin_for(att, "assistant"),
                role="assistant", turn_scope="this_turn",
            ))
    return out


async def thread_images(
    *,
    conversation_id: Optional[str],
    user_id: Optional[str],
    limit: int = _SCAN_LIMIT,
) -> List[ImageArtifact]:
    """Every image in THIS conversation, newest first, both roles.

    Scoped to ``Message.conversation_id`` — the same predicate
    ``AgentRunner._load_history`` uses — so the candidate set and the
    transcript the model is reasoning over cannot disagree about what "this
    conversation" contains. ``user_id`` is an extra belt-and-braces filter for
    any shared-DB deployment; the agent DB is per-tenant.

    Returns ``[]`` — never raises. A lookup failure must degrade into "ask the
    user which image" and never into "silently use a different one".
    """
    if not conversation_id:
        return []
    try:
        from sqlalchemy import and_, select

        from app.db.database import async_session_maker
        from app.db.models import Conversation, Message

        stmt = select(Message).where(
            and_(
                Message.conversation_id == conversation_id,
                Message.attachments.isnot(None),
            )
        )
        if user_id:
            stmt = stmt.join(
                Conversation, Message.conversation_id == Conversation.id
            ).where(Conversation.user_id == user_id)
        stmt = stmt.order_by(Message.created_at.desc()).limit(limit)
        async with async_session_maker() as db:
            rows = (await db.execute(stmt)).scalars().all()
    except Exception:
        logger.exception("image_artifacts: thread scan failed")
        return []

    out: List[ImageArtifact] = []
    for msg in rows:  # newest message first
        role = str(getattr(msg, "role", "") or "")
        if role not in ("user", "assistant"):
            continue
        imgs = [a for a in (getattr(msg, "attachments", None) or []) if _is_image(a)]
        # Within one message the LAST attachment is the most recent, so walk it
        # backwards to keep the whole list in newest-first order.
        for att in reversed(imgs):
            out.append(ImageArtifact(
                attachment=dict(att),
                origin=origin_for(att, role),
                role=role,
                created_at=getattr(msg, "created_at", None),
            ))
    return out


async def resolve_implicit(
    *,
    conversation_id: Optional[str],
    user_id: Optional[str],
    pending_attachments: Sequence[Any],
    inbound_media: Sequence[Any],
) -> Optional[ImageArtifact]:
    """What "it" / "the image" / "that pic" points at, with no id given.

    This turn first (nothing of it is persisted yet), then the newest image in
    this conversation regardless of who put it there. Returns None when the
    conversation contains no image at all — the caller must then ask, because
    the alternative is the Round 16 bug: reaching outside the thread and
    editing a face nobody mentioned.
    """
    for art in turn_artifacts(pending_attachments, inbound_media):
        return art
    for art in await thread_images(conversation_id=conversation_id, user_id=user_id):
        return art
    return None


async def resolve_by_id(
    image_id: str,
    *,
    conversation_id: Optional[str],
    user_id: Optional[str],
    pending_attachments: Sequence[Any],
    inbound_media: Sequence[Any],
) -> Optional[ImageArtifact]:
    """The artifact with this id, or None. Case-insensitive on the hex.

    Scoped identically to `resolve_implicit`: an id is a handle for something
    in this conversation, not a key into every file the tenant owns. A caller
    that gets None must surface an error — falling back to "the newest image"
    would turn a typo into a silently different picture.
    """
    wanted = (image_id or "").strip().lower()
    if not wanted:
        return None
    for art in turn_artifacts(pending_attachments, inbound_media):
        if art.id.lower() == wanted:
            return art
    for art in await thread_images(conversation_id=conversation_id, user_id=user_id):
        if art.id.lower() == wanted:
            return art
    return None


def choices_hint(artifacts: Sequence[ImageArtifact], cap: int = 4) -> str:
    """The images available, for an error the model has to relay.

    Named, not enumerated as ids alone: the agent has to turn this into a
    question a person can answer.
    """
    if not artifacts:
        return ""
    lines = [
        f"  - {a.describe()} — image_id {a.id}"
        for a in artifacts[:cap]
    ]
    return "Images in this conversation, newest first:\n" + "\n".join(lines)
