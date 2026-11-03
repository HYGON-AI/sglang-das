"""Direct GLM RDMA sources and heuristic failed-transfer retention.

A native sync timeout does not cancel RDMA. Delayed release below is not a
completion fence; bounded safe retirement still requires native cancellation
and drain, plus confirmation before the receiver reuses its destination.
"""

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)


def _encode_drain_timeout_s() -> float:
    try:
        # float() accepts "30" and "30.5" alike; int() would silently reject
        # float strings and fall back, mis-aligning the bound with the
        # encoder's actual write timeout.
        # The deployed SHCA build (11d962) defaults to 60s; upstream defaults
        # to 30s. Keep the retention heuristic conservative for this build.
        mc_timeout = int(float(os.environ.get("MC_TRANSFER_TIMEOUT", "60")))
    except (TypeError, ValueError):
        logger.warning(
            "invalid MC_TRANSFER_TIMEOUT=%r (not a number); falling back to 60s "
            "for failed-transfer retention. This delay does not prove that "
            "the encoder's RDMA writes have stopped.",
            os.environ.get("MC_TRANSFER_TIMEOUT"),
        )
        mc_timeout = 60
    return max(5, mc_timeout) + 5.0


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
    """Retain failed transfers for a heuristic window, without a DMA fence.

    The timer owns the buffers independently of an HTTP task's cancellation.
    E and P must use the same MC_TRANSFER_TIMEOUT. Even matching settings do
    not guarantee quiescence: the native sync deadline can return while work
    remains queued or posted. This is not safe recovery from a native hang.
    """

    def release_after_delay():
        try:
            release(*owners)
        except Exception:
            logger.exception("GLM encoder failed to release delayed RDMA buffers")

    timer = threading.Timer(_encode_drain_timeout_s(), release_after_delay)
    timer.daemon = True
    timer.start()


class GlmSourceBuffers:
    """Transfer directly from the embedding; never stage it through host memory."""

    def __init__(self, engine):
        self.engine = engine
        self.registrations = RdmaRegRefcount(engine)

    def transfer(self, embedding, session_id, destination):
        # Registration runs in the transfer worker, after cancellable admission.
        if not embedding.nbytes:
            return 0
        buffer = embedding.contiguous()
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
