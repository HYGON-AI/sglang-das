"""Cancellable admission before submitting Encoder native transfer work.

Only admitted work is submitted to the executor. Retirement may discard a
waiter, but never interrupts native work which can still access its buffers.
All queue state belongs to the asyncio event loop.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class TransferQueueTimeout(Exception):
    pass


class TransferRequestReleased(Exception):
    pass


class EncoderAbortRegistry:
    """Bounded tombstones for aborts racing ahead of encode dispatch."""

    def __init__(self, ttl: float, max_entries: int = 65536):
        self.ttl = max(1.0, ttl)
        self.max_entries = max_entries
        self._entries = OrderedDict()

    def _expire(self, now: float) -> None:
        while self._entries and next(iter(self._entries.values())) <= now:
            self._entries.popitem(last=False)

    def remember(self, req_id: str) -> bool:
        now = time.monotonic()
        self._expire(now)
        if req_id in self._entries:
            return False
        self._entries[req_id] = now + self.ttl
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return True

    def __contains__(self, req_id: str) -> bool:
        self._expire(time.monotonic())
        return req_id in self._entries

    def forget(self, req_id: str) -> None:
        self._entries.pop(req_id, None)


@dataclass(eq=False)
class TransferTicket:
    req_id: str
    queued_at: float
    admitted_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class EncoderTransferQueue:
    def __init__(self, workers: int, *, phase: str = "rdma", trace: bool = False):
        if workers < 1:
            raise ValueError("Encoder transfer workers must be positive")
        self._slots = asyncio.Semaphore(workers)
        self._queued = set()
        self._running = set()
        self.trace = trace
        self.phase = phase
        self.expired = 0
        self.abandoned = 0

    def _trace(self, event: str, ticket: TransferTicket, **details) -> None:
        if self.trace:
            logger.info(
                "Encoder transfer lifecycle phase=%s event=%s req_id=%s "
                "age_s=%.3f details=%s",
                self.phase,
                event,
                ticket.req_id,
                time.monotonic() - ticket.queued_at,
                details,
            )

    def mark_started(self, ticket: TransferTicket, started_at: float) -> None:
        # Invoked with loop.call_soon_threadsafe by the executor worker.
        if ticket in self._running:
            ticket.started_at = started_at
            self._trace(
                "native_start", ticket, queue_wait_s=started_at - ticket.queued_at
            )

    def mark_finished(self, ticket: TransferTicket, finished_at: float) -> None:
        if ticket in self._running:
            ticket.finished_at = finished_at
            self._trace(
                "native_end",
                ticket,
                native_s=(
                    finished_at - ticket.started_at
                    if ticket.started_at is not None
                    else None
                ),
            )

    def snapshot(self) -> dict:
        now = time.monotonic()
        return {
            "queued": len(self._queued),
            "admitted": len(self._running),
            "running": sum(
                t.started_at is not None and t.finished_at is None
                for t in self._running
            ),
            "oldest_queued_s": max(
                (now - t.queued_at for t in self._queued), default=0.0
            ),
            "oldest_running_s": max(
                (
                    now - t.started_at
                    for t in self._running
                    if t.started_at is not None and t.finished_at is None
                ),
                default=0.0,
            ),
            "expired": self.expired,
            "abandoned": self.abandoned,
        }

    @asynccontextmanager
    async def slot(self, req_id: str, released: asyncio.Event, timeout: float):
        ticket = TransferTicket(req_id, time.monotonic())
        self._queued.add(ticket)
        self._trace("queue_enter", ticket)
        acquire = asyncio.create_task(self._slots.acquire())
        retired = asyncio.create_task(released.wait())
        admitted = False
        try:
            try:
                done, _ = await asyncio.wait(
                    (acquire, retired),
                    timeout=max(0.0, timeout),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Retirement wins even if a slot became available in this turn.
                if released.is_set():
                    self.abandoned += 1
                    self._trace("retired_before_start", ticket)
                    raise TransferRequestReleased(req_id)
                if acquire not in done:
                    self.expired += 1
                    self._trace("queue_timeout", ticket)
                    raise TransferQueueTimeout(req_id)
                acquire.result()
                admitted = True
            finally:
                self._queued.discard(ticket)
                retired.cancel()
                if not admitted:
                    if not acquire.done():
                        # Semaphore.acquire refunds a permit if cancelled after
                        # its waiter was woken but before it actually returns.
                        acquire.cancel()
                    elif not acquire.cancelled() and acquire.exception() is None:
                        self._slots.release()
                await asyncio.gather(acquire, retired, return_exceptions=True)

            # Draining the helper tasks above yields to the loop; a release
            # can arrive after acquisition but before executor submission.
            if released.is_set():
                self.abandoned += 1
                self._trace("retired_before_start", ticket)
                raise TransferRequestReleased(req_id)
            ticket.admitted_at = time.monotonic()
            self._running.add(ticket)
            self._trace("admitted", ticket)
            yield ticket
        finally:
            if admitted:
                self._running.discard(ticket)
                self._slots.release()
                self._trace("slot_released", ticket)
            else:
                self._trace("queue_exit", ticket)
