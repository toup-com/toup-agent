"""Socket-scoped Realtime ordering; execution durability belongs to the job service.

The provider reader only registers calls and records lifecycle events. A single
worker executes functions, and this coordinator owns all provider writes. Never
wait for a function from the provider reader. Realtime's function-call response
and the eventual spoken answer are separate responses:
https://developers.openai.com/api/docs/guides/realtime-conversations
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
SendEvent = Callable[[dict[str, Any]], Awaitable[None]]
RunFunction = Callable[[], Awaitable[str]]
ResponseRejected = Callable[[dict[str, Any]], Awaitable[None]]
ACTIVE_RESPONSE_RETRY_LIMIT = 3


class LifecycleCapacityError(RuntimeError):
    """The session must end rather than forget deduplication history."""


@dataclass
class _Call:
    response_id: str
    result: str | None = None
    sent: bool = False


@dataclass
class _Response:
    epoch: int
    input_revision: int = 0
    created: bool = False
    done: bool = False
    status: str = "in_progress"
    calls: set[str] = field(default_factory=set)
    followed_up: bool = False


@dataclass
class _Request:
    epoch: int
    response: dict[str, Any]
    automatic: bool = True
    event_id: str = ""
    input_revision: int = 0
    active_conflicts: int = 0


class ResponseCoordinator:
    """Coordinate one default conversation on one Realtime WebSocket.

    All calls run on the same asyncio event loop. Pass every provider write
    through ``send_event``. Feed response IDs (and response.created metadata)
    back into this instance; filter streamed audio/text with ``accepts_response``.
    Speech interruption advances an epoch, preserving function outputs while
    suppressing automatic speech from the interrupted generation.

    The fixed session limits bound both memory and deduplication history. At a
    limit, close/reconnect the socket; never evict a call ID and execute it again.
    """

    def __init__(
        self,
        send: SendEvent,
        *,
        max_calls: int = 1024,
        max_responses: int = 4096,
    ) -> None:
        self._send = send
        self._lock = asyncio.Lock()
        self._calls: dict[str, _Call] = {}
        self._responses: dict[str, _Response] = {}
        self._active: set[str] = set()
        self._max_calls = max_calls
        self._max_responses = max_responses
        self._epoch = 0
        self._input_revision = 0
        self._speaking = False
        self._awaiting_user_response = False
        self._speech_paused = False
        self._desired: _Request | None = None
        self._deferred_automatic: _Request | None = None
        self._retry_after_active: deque[_Request] = deque()
        self._creating: _Request | None = None
        self._closed = False
        self._flush_task: asyncio.Task[None] | None = None
        self._dirty = False
        self._response_rejected_handler: ResponseRejected | None = None

    def set_response_rejected_handler(
        self, handler: ResponseRejected | None,
    ) -> None:
        self._response_rejected_handler = handler

    async def _notify_response_rejected(self, request: _Request | None) -> None:
        if request is None or self._response_rejected_handler is None:
            return
        try:
            await self._response_rejected_handler(dict(request.response))
        except Exception:
            logger.exception("Realtime rejected-response handler failed")

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def closed(self) -> bool:
        return self._closed

    def _response(self, response_id: str) -> _Response:
        if not response_id:
            raise ValueError("A provider response_id is required")
        if response_id not in self._responses:
            if len(self._responses) >= self._max_responses:
                raise LifecycleCapacityError("Realtime response history is full")
            self._responses[response_id] = _Response(
                epoch=self._epoch, input_revision=self._input_revision,
            )
        return self._responses[response_id]

    def register_call(self, call_id: str, response_id: str) -> bool:
        """Reserve a call synchronously before scheduling any side effects."""
        if self._closed or call_id in self._calls:
            return False
        if not call_id:
            raise ValueError("A provider call_id is required")
        if len(self._calls) >= self._max_calls:
            raise LifecycleCapacityError("Realtime function history is full")
        response = self._response(response_id)
        self._calls[call_id] = _Call(response_id=response_id)
        response.calls.add(call_id)
        return True

    def set_call_result(self, call_id: str, result: str) -> None:
        """Store once and arrange delivery without blocking the socket reader."""
        call = self._calls[call_id]
        if self._closed or call.result is not None:
            return
        call.result = result
        self._dirty = True
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._background_flush())

    async def _background_flush(self) -> None:
        try:
            while self._dirty and not self._closed:
                self._dirty = False
                await self._flush()
        except Exception:
            # A write failure has an unknown delivery outcome. Do not replay
            # the write, and especially never rerun the function to recover it.
            self._closed = True
            logger.exception("Realtime result delivery failed; socket is unusable")

    async def response_created(
        self, response_id: str, metadata: dict[str, Any] | None = None,
    ) -> None:
        response = self._response(response_id)
        if response.done or response.created:
            return  # duplicate/late created must never resurrect a response
        request_id = (metadata or {}).get("voice_response_request")
        # None supports callers that do not expose metadata. An actual empty
        # metadata dict identifies a provider automatic response, not our ACK.
        if self._creating and (
            request_id == self._creating.event_id or metadata is None
        ):
            response.epoch = self._creating.epoch
            response.input_revision = self._creating.input_revision
            self._creating = None
        response.created = True
        self._active.add(response_id)
        if response.epoch == self._epoch and not self._speaking:
            self._awaiting_user_response = False
        await self._flush()

    async def response_done(
        self,
        response_id: str,
        call_ids: Iterable[str] = (),
        *,
        status: str = "completed",
    ) -> None:
        """Pass IDs from response.output, including calls not yet submitted.

        The full list fences a batch even if output_item.done is delivered late.
        Duplicate response.done events are harmless and cannot clear another
        active response or a response.create awaiting acknowledgement.
        """
        response = self._response(response_id)
        response.calls.update(cid for cid in call_ids if cid)
        if not response.done:
            response.done = True
            response.status = status
        self._active.discard(response_id)
        if response.epoch == self._epoch and not self._speaking:
            self._awaiting_user_response = False
        await self._flush()

    def accepts_response(self, response_id: str | None) -> bool:
        """Whether an audio/text frame still belongs to the audible generation."""
        if self._closed or self._speaking or self._speech_paused:
            return False
        response = self._responses.get(response_id or "")
        return response is not None and not response.done and response.epoch == self._epoch

    async def speech_started(self) -> None:
        dropped: list[_Request | None] = []
        if not self._speaking:
            dropped = [
                self._desired, self._deferred_automatic,
                *self._retry_after_active,
            ]
            self._epoch += 1
            self._desired = None
            self._deferred_automatic = None
            self._retry_after_active.clear()
        self._speaking = True
        self._speech_paused = False
        self._awaiting_user_response = True
        for request in dropped:
            await self._notify_response_rejected(request)
        await self._flush()

    async def speech_stopped(self, *, expect_user_response: bool = True) -> None:
        self._speaking = False
        if not expect_user_response or any(
            self._responses[rid].epoch == self._epoch for rid in self._active
        ):
            self._awaiting_user_response = False
        await self._flush()

    async def interrupt_speech(self, *, expect_user_response: bool = False) -> None:
        """Stop-speaking is not work cancellation; mute until a new user turn."""
        dropped = [
            self._desired, self._deferred_automatic,
            *self._retry_after_active,
        ]
        self._epoch += 1
        self._desired = None
        self._deferred_automatic = None
        self._retry_after_active.clear()
        self._speech_paused = not expect_user_response
        self._awaiting_user_response = expect_user_response
        for request in dropped:
            await self._notify_response_rejected(request)
        await self._flush()

    def _queue_response(self, request: _Request) -> bool:
        if request.epoch != self._epoch:
            return False
        request.input_revision = self._input_revision
        # Several producers can request a reply to the same conversation
        # snapshot. Coalesce those requests, but retain a follow-up if new
        # context/results arrived after the current generation started.
        if request.automatic and not request.response:
            if self._creating and (
                self._creating.epoch == request.epoch
                and self._creating.input_revision == request.input_revision
            ):
                return False
            if any(
                self._responses[rid].epoch == request.epoch
                and self._responses[rid].input_revision == request.input_revision
                for rid in self._active
            ):
                return False
        # Explicit user requests have priority over automatic tool follow-ups.
        if self._desired is None:
            self._desired = request
            return True
        if not request.automatic:
            if self._desired.automatic:
                if self._deferred_automatic is not None:
                    # Keep the older displaced automatic intent owned before
                    # replacing the single fast-path slot.
                    self._retry_after_active.append(self._deferred_automatic)
                self._deferred_automatic = self._desired
            self._desired = request
            return True
        return False

    async def request_response(
        self,
        response: dict[str, Any] | None = None,
        *,
        automatic: bool = True,
        epoch: int | None = None,
    ) -> bool:
        if epoch is not None and epoch != self._epoch:
            return False
        if response is not None and not isinstance(response, dict):
            raise ValueError("Realtime response options must be an object")
        response_payload = dict(response or {})
        raw_metadata = response_payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, dict):
            raise ValueError("Realtime response metadata must be an object")
        metadata = dict(raw_metadata or {})
        effective_metadata_count = len(metadata) + int(
            "voice_response_request" not in metadata
        )
        if effective_metadata_count > 16 or any(
            not isinstance(key, str) or not isinstance(value, str)
            or len(key) > 64 or len(value) > 512
            for key, value in metadata.items()
        ):
            raise ValueError(
                "Realtime response metadata must contain at most 16 bounded string pairs"
            )
        if not automatic:
            self._speech_paused = False
            self._awaiting_user_response = False
        queued = self._queue_response(_Request(
            epoch=self._epoch if epoch is None else epoch,
            response=response_payload,
            automatic=automatic,
        ))
        await self._flush()
        return queued

    async def send_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "response.create":
            await self.request_response(event.get("response"))
            return
        async with self._lock:
            if not self._closed:
                await self._write(event)

    async def response_error(self, event_id: str, *, retry_on_active: bool = False) -> bool:
        """Release one rejected create; retry only a proven active-response race."""
        if not self._creating or self._creating.event_id != event_id:
            return False
        rejected = self._creating
        self._creating = None
        notify_rejected = not retry_on_active
        if retry_on_active:
            # The provider's error proves another response is active even when
            # its response.created event has not reached our reader yet. Park
            # the exact intent until that response is observed and completes.
            self._awaiting_user_response = True
            rejected.active_conflicts += 1
            if rejected.active_conflicts > ACTIVE_RESPONSE_RETRY_LIMIT:
                notify_rejected = True
            elif not self._queue_response(rejected):
                if rejected.epoch == self._epoch:
                    # An explicit request may already own _desired. Preserve
                    # the rejected automatic create behind it rather than
                    # orphaning its relay token. There is only one _creating
                    # request, but a deque keeps repeated provider conflicts
                    # lossless and maintains their order.
                    self._retry_after_active.append(rejected)
                else:
                    notify_rejected = True
        await self._flush()
        if notify_rejected:
            await self._notify_response_rejected(rejected)
        return True

    async def _write(self, event: dict[str, Any]) -> None:
        if event.get("type") in {
            "conversation.item.create", "conversation.item.delete", "conversation.item.truncate",
        }:
            self._input_revision += 1
        try:
            await self._send(event)
        except BaseException:
            self._closed = True
            raise

    async def _flush(self) -> None:
        async with self._lock:
            if self._closed:
                return
            # Lifecycle callbacks may add responses while a socket send yields.
            for response in tuple(self._responses.values()):
                if not response.done or not response.calls or response.followed_up:
                    continue
                calls = [self._calls.get(cid) for cid in sorted(response.calls)]
                if any(call is None or call.result is None for call in calls):
                    continue
                for call_id in sorted(response.calls):
                    call = self._calls[call_id]
                    if call.sent:
                        continue
                    # Sending may have succeeded before a transport failure;
                    # mark first and never retry an ambiguous provider write.
                    call.sent = True
                    await self._write({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": call.result,
                        },
                    })
                response.followed_up = True
                if response.status == "completed":
                    self._queue_response(_Request(epoch=response.epoch, response={}))

            if self._desired is None:
                while self._retry_after_active:
                    retry = self._retry_after_active.popleft()
                    if retry.epoch == self._epoch:
                        self._desired = retry
                        break

            if self._desired is None and self._deferred_automatic is not None:
                deferred, self._deferred_automatic = self._deferred_automatic, None
                if deferred.epoch == self._epoch:
                    self._desired = deferred

            if (
                not self._desired or self._active or self._creating
                or self._speaking or self._awaiting_user_response
                or (self._speech_paused and self._desired.automatic)
            ):
                return
            request, self._desired = self._desired, None
            if request.epoch != self._epoch:
                return
            request.event_id = f"voice_response_{uuid.uuid4().hex}"
            self._creating = request  # reserve before awaiting send/ACK
            payload = dict(request.response)
            metadata = dict(payload.get("metadata") or {})
            metadata["voice_response_request"] = request.event_id
            if len(metadata) > 16 or any(
                not isinstance(key, str) or not isinstance(value, str)
                or len(key) > 64 or len(value) > 512
                for key, value in metadata.items()
            ):
                self._creating = None
                raise ValueError(
                    "Realtime response metadata must contain at most 16 bounded string pairs"
                )
            payload["metadata"] = metadata
            await self._write({
                "type": "response.create",
                "event_id": request.event_id,
                "response": payload,
            })

    async def close(self) -> None:
        self._closed = True
        self._desired = None
        self._deferred_automatic = None
        self._retry_after_active.clear()
        self._response_rejected_handler = None
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)


class FunctionCallSupervisor:
    """Bounded serial function execution, independent of response generation.

    A single worker prevents two native mutations from executing concurrently.
    ``submit`` is synchronous, including overflow handling, so no tool can stall
    the provider reader. Durable work must be detached by the callback's job
    service; ``close`` cancels only this socket's worker and callback awaiter.
    """

    def __init__(self, coordinator: ResponseCoordinator, *, max_pending: int = 8) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._coordinator = coordinator
        self._max_pending = max_pending
        self._queue: deque[tuple[str, RunFunction]] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._running: asyncio.Task[str] | None = None
        self._running_id: str | None = None
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def pending_count(self) -> int:
        return len(self._queue) + int(self._running_id is not None)

    def submit(self, *, call_id: str, response_id: str, run: RunFunction) -> bool:
        if self._closed or not self._coordinator.register_call(call_id, response_id):
            return False
        if self.pending_count >= self._max_pending:
            self._coordinator.set_call_result(
                call_id,
                "ERROR: Voice tool queue is full. This function was not executed. "
                "Wait for existing work; do not automatically repeat the request.",
            )
            return False
        self._idle.clear()
        self._queue.append((call_id, run))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work())
        return True

    async def _work(self) -> None:
        try:
            while self._queue and not self._closed and not self._coordinator.closed:
                call_id, run = self._queue.popleft()
                self._running_id = call_id
                try:
                    self._running = asyncio.create_task(run())
                    result = await self._running
                    if not isinstance(result, str):
                        raise TypeError("Realtime function callbacks must return a string")
                except asyncio.CancelledError:
                    if self._closed:
                        raise
                    result = (
                        "ERROR: This function's local wait was cancelled. External work "
                        "may already have started; inspect its task status before retrying."
                    )
                except Exception:
                    logger.exception("Realtime function failed (call_id=%s)", call_id)
                    result = (
                        "ERROR: Function execution failed; its external effects are unknown. "
                        "Check existing task status before retrying. Do not claim completion."
                    )
                finally:
                    self._running = None
                    self._running_id = None
                self._coordinator.set_call_result(call_id, result)
        finally:
            self._idle.set()

    async def wait_idle(self, *, timeout: float) -> bool:
        """Give already submitted callbacks a bounded shutdown grace."""
        if self.pending_count == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return self.pending_count == 0

    def cancel(self, call_id: str) -> bool:
        """Explicit socket-task cancellation; never call this for a barge-in."""
        if call_id == self._running_id and self._running:
            self._running.cancel()
            return True
        for queued in self._queue:
            if queued[0] == call_id:
                self._queue.remove(queued)
                self._coordinator.set_call_result(
                    call_id, "ERROR: Function cancelled before execution; no work was started.",
                )
                if not self._queue and self._running_id is None:
                    self._idle.set()
                return True
        return False

    async def close(self) -> None:
        self._closed = True
        self._queue.clear()
        if self._worker and not self._worker.done():
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        self._idle.set()
