"""One pre-capture candidate stream per worker, shared by HCU indexers."""

from typing import Optional

import torch

_CANDIDATE_STREAM: Optional[torch.cuda.Stream] = None


def init_candidate_stream() -> None:
    """Create the shared candidate stream for this worker's device.

    Call from backend init, before capture. Idempotent, so the eager and the
    capture paths both observe the same stream object.
    """
    global _CANDIDATE_STREAM
    if _CANDIDATE_STREAM is None:
        # Each SGLang worker owns one device, so one process-wide stream is
        # enough for every candidate indexer instance in that worker.
        _CANDIDATE_STREAM = torch.cuda.Stream()


def get_candidate_stream() -> torch.cuda.Stream:
    """The shared candidate stream, creating it on first use.

    Callers that run inside capture must not reach the creating path; the
    backend calls ``init_candidate_stream`` before capture so this only ever
    returns the stream that was created at init.
    """
    if _CANDIDATE_STREAM is None:
        init_candidate_stream()
    return _CANDIDATE_STREAM


def get_candidate_stream_if_initialized() -> Optional[torch.cuda.Stream]:
    """The shared candidate stream, or None when it was never created.

    Lets the scheduler avoid allocating a colliding side stream without
    forcing the candidate stream into existence on workers that never use it.
    """
    return _CANDIDATE_STREAM
