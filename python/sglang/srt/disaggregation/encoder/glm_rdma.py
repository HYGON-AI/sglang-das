"""GLM encoder transport buffers, using the source branch's admission policy."""

import logging
import os
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _rdma_pool_max_bytes() -> int:
    return envs.SGLANG_MC_RDMA_POOL_MAX_MB.get() * 1024 * 1024


def _rdma_pool_acquire_timeout_secs() -> float:
    """Max seconds ``RdmaBufferPool.acquire`` blocks waiting for budget before
    it allocates over the limit anyway. This is the admission control that
    turns SGLANG_MC_RDMA_POOL_MAX_MB into a real cap on peak
    live buffers (previously it only bounded the idle free list, so a burst
    of concurrent large multimodal receives could allocate unbounded and OOM
    the prefill pod). 0 disables the wait (legacy unbounded behavior)."""
    return envs.SGLANG_MC_RDMA_POOL_ACQUIRE_TIMEOUT_SECS.get()


def rdma_pool_enabled() -> bool:
    """Whether to use the Mooncake RDMA registered-buffer pools.

    Controlled by the byte cap that sizes the pool:
      * SGLANG_MC_RDMA_POOL_MAX_MB (default 0)
    > 0 ENABLES the pool: the receiver and the sender both use
    RdmaBufferPool. 0 (the default) DISABLES it -- the receiver and sender
    fall back to the original per-request register + deregister logic.
    """
    return _rdma_pool_max_bytes() > 0


def _encode_drain_timeout_s() -> float:
    try:
        # float() accepts "30" and "30.5" alike; int() would silently reject
        # float strings and fall back, mis-aligning the bound with the
        # encoder's actual write timeout.
        mc_timeout = int(float(os.environ.get("MC_TRANSFER_TIMEOUT", "30")))
    except (TypeError, ValueError):
        logger.warning(
            "invalid MC_TRANSFER_TIMEOUT=%r (not a number); falling back to 30s "
            "for the encode drain bound. The bound can expire while an "
            "encoder's RDMA write is still in flight and reintroduce the "
            "deregister-while-writing race.",
            os.environ.get("MC_TRANSFER_TIMEOUT"),
        )
        mc_timeout = 30
    return max(5, mc_timeout) + 5.0


class RdmaBufferPool:
    """Pool of long-lived, RDMA-registered CPU buffers.

    The receiver uses these buffers as RDMA-write targets. The encoder sender
    copies embeddings into the same kind of registered-once buffers and uses
    them as RDMA-write sources, avoiding per-transfer registration churn.

    The byte cap (SGLANG_MC_RDMA_POOL_MAX_MB) acts on two sides.
    release() bounds only the idle reuse cache -- in-flight and quiesced
    buffers stay registered until their transfers are safe to release, so a
    traffic or failure spike does not trigger deregister/registration churn.
    acquire() additionally admission-controls the total live working set
    against the same cap: it first evicts idle buffers of smaller size
    classes to make room, and only when none are left blocks until a
    release/discard frees budget instead of allocating unbounded (which
    would OOM the pod). That wait is bounded by
    SGLANG_MC_RDMA_POOL_ACQUIRE_TIMEOUT_SECS, and a request larger
    than the whole cap is always admitted, so the working set can still
    briefly exceed the cap.
    """

    def __init__(self, engine):
        self._engine = engine
        self._lock = threading.Lock()
        # Signalled whenever budget frees up (release/discard) so acquire()
        # waiters can retry instead of allocating over budget.
        self._cond = threading.Condition(self._lock)
        self._free = {}
        self._floor = 1 * 1024 * 1024
        self._max_total_bytes = _rdma_pool_max_bytes()
        self._total_bytes = 0
        self._total_count = 0
        self._free_bytes = 0
        self._free_count = 0
        self._warned_free_over = False
        self._acquire_timeout = _rdma_pool_acquire_timeout_secs()

    def _size_class(self, nbytes: int) -> int:
        nbytes = max(int(nbytes), self._floor)
        return 1 << (nbytes - 1).bit_length()

    def acquire(self, nbytes: int) -> torch.Tensor:
        class_bytes = self._size_class(nbytes)
        deadline = None
        evicted = []
        reused = None
        with self._lock:
            while True:
                free = self._free.get(class_bytes)
                if free:
                    reused = free.pop()
                    self._free_bytes -= reused.numel()
                    self._free_count -= 1
                    break

                best_key = None
                for size, buffers in self._free.items():
                    if (
                        size > class_bytes
                        and buffers
                        and (best_key is None or size < best_key)
                    ):
                        best_key = size
                if best_key is not None:
                    reused = self._free[best_key].pop()
                    self._free_bytes -= reused.numel()
                    self._free_count -= 1
                    break

                # No reusable buffer: we must allocate a new one. Admission
                # control -- block until releases free budget instead of
                # allocating unbounded (a burst of concurrent large multimodal
                # receives otherwise blows past the cap and OOMs the pod).
                would_exceed = self._total_bytes + class_bytes > self._max_total_bytes
                # Proceed anyway when within budget, when nothing is
                # outstanding to free up (a single request larger than the
                # whole cap must still run; also avoids deadlock), or when the
                # wait is disabled.
                if (
                    not would_exceed
                    or self._total_count == 0
                    or self._acquire_timeout <= 0
                ):
                    break
                # GLM NOTE: any idle buffer left here is smaller than
                # class_bytes (a larger one would have been reused above).
                # Evict idle buffers for budget instead of waiting -- a free
                # cache saturated with small size classes must not starve
                # larger allocations into the timeout path (that wedge held
                # prefill pods in permanent 60s-wait + register churn).
                if self._free_count > 0:
                    while (
                        self._free_count > 0
                        and self._total_bytes + class_bytes > self._max_total_bytes
                    ):
                        smallest = min(
                            size for size, buffers in self._free.items() if buffers
                        )
                        buffer = self._free[smallest].pop()
                        self._free_bytes -= buffer.numel()
                        self._free_count -= 1
                        self._total_bytes -= buffer.numel()
                        self._total_count -= 1
                        evicted.append(buffer)
                    continue
                if deadline is None:
                    deadline = time.monotonic() + self._acquire_timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "mooncake RDMA pool: acquire waited %.0fs for %d bytes "
                        "but pool still at bytes=%d/%d count=%d; allocating "
                        "over budget",
                        self._acquire_timeout,
                        class_bytes,
                        self._total_bytes,
                        self._max_total_bytes,
                        self._total_count,
                    )
                    break
                self._cond.wait(timeout=remaining)

            if reused is None:
                # Reserve the budget under the lock so concurrent acquirers
                # observe it and cannot all race past the check together.
                self._total_bytes += class_bytes
                self._total_count += 1

        # Deregister evicted idle buffers outside the lock (engine calls are
        # slow). Safe: they sat in the free cache, so no in-flight write
        # references their MRs.
        for buffer in evicted:
            self._engine.deregister(buffer.data_ptr())
        if reused is not None:
            return reused

        try:
            buffer = torch.empty(class_bytes, dtype=torch.uint8)
            # The transfer-engine wrapper raises on failure and returns None.
            self._engine.register(buffer.data_ptr(), buffer.nbytes)
        except BaseException:
            # Roll back the reservation so a failed allocation does not leak
            # budget (which would wedge every future acquire).
            with self._lock:
                self._total_bytes -= class_bytes
                self._total_count -= 1
                self._cond.notify_all()
            raise
        return buffer

    def release(self, buffer: torch.Tensor) -> None:
        if buffer is None:
            return
        buffer_bytes = buffer.numel()
        with self._lock:
            over_budget = self._free_bytes + buffer_bytes > self._max_total_bytes
            if not over_budget:
                self._free.setdefault(buffer_bytes, []).append(buffer)
                self._free_bytes += buffer_bytes
                self._free_count += 1
                self._cond.notify_all()
                return
            if not self._warned_free_over:
                self._warned_free_over = True
                logger.warning(
                    "mooncake RDMA free buffer cache at budget "
                    "(cached bytes=%d/%d, count=%d; returned bytes=%d; "
                    "registered bytes=%d count=%d); deregistering returned "
                    "buffer. Increase SGLANG_MC_RDMA_POOL_MAX_MB to retain a "
                    "larger working set.",
                    self._free_bytes,
                    self._max_total_bytes,
                    self._free_count,
                    buffer_bytes,
                    self._total_bytes,
                    self._total_count,
                )
            self._total_bytes -= buffer_bytes
            self._total_count -= 1
            self._cond.notify_all()
        self._engine.deregister(buffer.data_ptr())

    def discard(self, buffer: torch.Tensor) -> None:
        """Deregister an aborted request's buffer instead of reusing it."""
        if buffer is None:
            return
        with self._lock:
            self._total_bytes -= buffer.numel()
            self._total_count -= 1
            remaining_bytes = self._total_bytes
            remaining_count = self._total_count
            self._cond.notify_all()
        logger.warning(
            "mooncake RDMA pool: discarding buffer (bytes=%d) on abort/timeout; "
            "pool now bytes=%d count=%d",
            buffer.numel(),
            remaining_bytes,
            remaining_count,
        )
        self._engine.deregister(buffer.data_ptr())


class RdmaRegRefcount:
    """Reference-count registrations of a shared Mooncake source tensor."""

    def __init__(self, engine):
        self._engine = engine
        self._lock = threading.Lock()
        self._refcounts = {}
        self._pinned_tensors = {}

    def acquire(self, tensor: torch.Tensor) -> int:
        addr = tensor.data_ptr()
        with self._lock:
            refcount = self._refcounts.get(addr, 0)
            if refcount == 0:
                # The wrapper already checks the native registration status.
                self._engine.register(addr, tensor.nbytes)
                self._pinned_tensors[addr] = tensor
            self._refcounts[addr] = refcount + 1
        return addr

    def release(self, addr: int) -> None:
        with self._lock:
            refcount = self._refcounts.get(addr, 0)
            if refcount <= 1:
                self._refcounts.pop(addr, None)
                self._pinned_tensors.pop(addr, None)
                self._engine.deregister(addr)
            else:
                self._refcounts[addr] = refcount - 1


def defer_rdma_release(release, *owners):
    """Pin failed transfers for the source branch's write timeout window.

    The timer owns the buffers independently of an HTTP task's cancellation.
    E and P must use the same MC_TRANSFER_TIMEOUT, as on sglang_glm_dev.
    """

    def quiesced():
        try:
            release(*owners)
        except Exception:
            logger.exception("GLM encoder failed to release quiesced RDMA buffers")

    timer = threading.Timer(_encode_drain_timeout_s(), quiesced)
    timer.daemon = True
    timer.start()


class GlmSourceBuffers:
    def __init__(self, engine):
        self.engine = engine
        self.pool = RdmaBufferPool(engine) if rdma_pool_enabled() else None
        self.registrations = RdmaRegRefcount(engine)

    def transfer(self, embedding, session_id, destination):
        # Called in the transfer worker: pool admission and the synchronous copy
        # cannot block the event loop that processes /send and releases buffers.
        if not embedding.nbytes:
            return 0
        embedding = embedding.contiguous()
        if self.pool is not None:
            buffer = self.pool.acquire(embedding.nbytes)
            try:
                buffer[: embedding.nbytes].copy_(embedding.view(-1).view(torch.uint8))
            except BaseException:
                self.pool.release(buffer)
                raise
            release = self.pool.release
        else:
            buffer = embedding
            self.registrations.acquire(buffer)

            def release(tensor):
                self.registrations.release(tensor.data_ptr())

        success = False
        try:
            ret = self.engine.transfer_sync(
                session_id, buffer.data_ptr(), destination, embedding.nbytes
            )
            success = ret == 0
            return ret
        finally:
            if success:
                release(buffer)
            else:
                defer_rdma_release(release, buffer)
