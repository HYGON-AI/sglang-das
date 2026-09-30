"""Request-owned progress for abort cleanup's resource substeps.

Opaque allocator calls cannot be rolled back. A failure with unknown side effects
is quarantined; only explicitly side-effect-free failures may be retried.
"""

from typing import Callable, TypeVar

T = TypeVar("T")


class RetryableCleanupError(RuntimeError):
    """The failing action guarantees it has made no resource ownership changes."""


class CleanupOutcomeUnknown(RuntimeError):
    """An earlier attempt may have freed resources; repeating it is unsafe."""


def run_cleanup_step(req, name: str, action: Callable[[], T]) -> T:
    if not getattr(req, "external_kv_abort_requested", False):
        return action()
    steps = req.external_kv_cleanup_steps
    state, result = steps.get(name, (None, None))
    if state == "done":
        return result
    if state == "unknown":
        raise CleanupOutcomeUnknown(
            f"Request {req.rid}: cleanup step {name} has unknown side effects; "
            "retaining resources instead of repeating a possible free"
        )
    # Record before entering the opaque operation, including exceptional exits.
    steps[name] = ("unknown", None)
    try:
        result = action()
    except RetryableCleanupError:
        steps.pop(name, None)
        raise
    steps[name] = ("done", result)
    return result
