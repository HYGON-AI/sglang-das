import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


UNKNOWN = "unknown"
AVAILABLE = "available"
UNAVAILABLE = "unavailable"


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _config_or_env(
    config: dict[str, int], key: str, env_name: str, default: int
) -> int:
    if key in config:
        return int(config[key])
    return _positive_env_int(env_name, default)


@dataclass(frozen=True)
class DfsHealthSnapshot:
    read_state: str
    write_state: str
    generation: int
    read_failures: int
    write_failures: int
    read_successes: int
    write_successes: int
    read_error: int
    write_error: int


@dataclass
class _NativeProbeCall:
    capability: str
    done: threading.Event
    started_at: float
    thread: threading.Thread | None = None
    result: int | Exception | None = None
    timed_out: bool = False


class MooncakeDfsHealthManager:
    """Owns rank-local DFS health and the native client's effective policy."""

    def __init__(
        self,
        store: Any,
        config: dict[str, int] | None = None,
        *,
        defer_policy_application: bool = False,
    ):
        self._store = store
        self._defer_policy_application = defer_policy_application
        config = config or {}
        self._interval_s = _config_or_env(
            config,
            "probe_interval_s",
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S",
            10,
        )
        self._failure_threshold = _config_or_env(
            config,
            "failure_threshold",
            "SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD",
            3,
        )
        self._success_threshold = _config_or_env(
            config,
            "success_threshold",
            "SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD",
            3,
        )
        self._probe_timeout_s = _config_or_env(
            config,
            "probe_timeout_s",
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S",
            2,
        )
        for name, value in (
            ("probe_interval_s", self._interval_s),
            ("failure_threshold", self._failure_threshold),
            ("success_threshold", self._success_threshold),
            ("probe_timeout_s", self._probe_timeout_s),
        ):
            if value <= 0:
                raise ValueError(f"Mooncake DFS health {name} must be positive")
        self._lock = threading.Lock()
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._probe_lock = threading.Lock()
        self._probe_calls: dict[str, _NativeProbeCall | None] = {
            "read": None,
            "write": None,
        }
        self._read_state = UNKNOWN
        self._write_state = UNKNOWN
        self._generation = 0
        self._read_failures = 0
        self._write_failures = 0
        self._read_successes = 0
        self._write_successes = 0
        self._read_error = 0
        self._write_error = 0
        self._effective_policy = (False, False)

        for method in (
            "probe_dfs_read",
            "probe_dfs_write",
            "set_dfs_access_policy",
        ):
            if not hasattr(store, method):
                raise RuntimeError(
                    f"The installed Mooncake package lacks {method}(); "
                    "rebuild/install Mooncake with DFS health support."
                )

        # UNKNOWN is fail-closed. This call happens before SGLang warmup.
        self._apply_policy(False, False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="mooncake-dfs-health", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, float(self._interval_s) + 1.0))
        self._thread = None

    def snapshot(self) -> DfsHealthSnapshot:
        with self._lock:
            return DfsHealthSnapshot(
                read_state=self._read_state,
                write_state=self._write_state,
                generation=self._generation,
                read_failures=self._read_failures,
                write_failures=self._write_failures,
                read_successes=self._read_successes,
                write_successes=self._write_successes,
                read_error=self._read_error,
                write_error=self._write_error,
            )

    def local_policy(self) -> tuple[bool, bool]:
        """Return this rank's fail-closed policy candidate."""
        with self._lock:
            return (
                self._read_state == AVAILABLE,
                self._write_state == AVAILABLE,
            )

    def apply_effective_policy(self, allow_read: bool, allow_write: bool) -> None:
        """Apply a scheduler-coordinated policy to this rank's native client."""
        policy = (bool(allow_read), bool(allow_write))
        with self._policy_lock:
            if policy == self._effective_policy:
                return
            old_policy = self._effective_policy
            self._apply_policy(*policy)
            self._effective_policy = policy
        logger.warning(
            "Mooncake DFS effective policy transition: "
            "old_read=%s old_write=%s new_read=%s new_write=%s",
            old_policy[0],
            old_policy[1],
            policy[0],
            policy[1],
        )

    def probe_once(self) -> DfsHealthSnapshot:
        # Launch both operations before waiting for either one. Each operation has
        # its own deadline and in-flight slot, so a stuck read cannot delay or
        # change the write result (and vice versa).
        read_call = self._start_native_probe("read", "probe_dfs_read")
        write_call = self._start_native_probe("write", "probe_dfs_write")
        read_error = self._collect_native_probe(read_call)
        write_error = self._collect_native_probe(write_call)
        read_ok = read_error == 0
        write_ok = write_error == 0

        with self._lock:
            old_read = self._read_state
            old_write = self._write_state
            self._read_state, self._read_successes, self._read_failures = (
                self._next_state(
                    self._read_state,
                    read_ok,
                    self._read_successes,
                    self._read_failures,
                )
            )
            self._write_state, self._write_successes, self._write_failures = (
                self._next_state(
                    self._write_state,
                    write_ok,
                    self._write_successes,
                    self._write_failures,
                )
            )
            self._read_error = read_error
            self._write_error = write_error
            read_changed = old_read != self._read_state
            write_changed = old_write != self._write_state
            self._generation += int(read_changed) + int(write_changed)
            allow_read = self._read_state == AVAILABLE
            allow_write = self._write_state == AVAILABLE
            snapshot = self.snapshot_unlocked()

        if (read_changed or write_changed) and not self._defer_policy_application:
            self.apply_effective_policy(allow_read, allow_write)
        if read_changed:
            logger.warning(
                "Mooncake DFS local health transition: capability=read "
                "old_state=%s new_state=%s error=%s generation=%s",
                old_read,
                snapshot.read_state,
                read_error,
                snapshot.generation,
            )
        if write_changed:
            logger.warning(
                "Mooncake DFS local health transition: capability=write "
                "old_state=%s new_state=%s error=%s generation=%s",
                old_write,
                snapshot.write_state,
                write_error,
                snapshot.generation,
            )
        return snapshot

    def _start_native_probe(
        self, capability: str, method_name: str
    ) -> _NativeProbeCall:
        with self._probe_lock:
            existing = self._probe_calls[capability]
            if existing is not None:
                return existing
            call = _NativeProbeCall(
                capability=capability,
                done=threading.Event(),
                started_at=time.monotonic(),
            )

            def invoke() -> None:
                try:
                    call.result = int(getattr(self._store, method_name)())
                except Exception as exc:
                    call.result = exc
                finally:
                    call.done.set()
                    # A timed-out result is stale: release its in-flight slot,
                    # but never let a later probe cycle count it as success.
                    with self._probe_lock:
                        if (
                            call.timed_out
                            and self._probe_calls[call.capability] is call
                        ):
                            self._probe_calls[call.capability] = None

            call.thread = threading.Thread(
                target=invoke,
                name=f"mooncake-dfs-{capability}-probe",
                daemon=True,
            )
            self._probe_calls[capability] = call
            call.thread.start()
            return call

    def _collect_native_probe(self, call: _NativeProbeCall) -> int:
        remaining = max(
            0.0,
            self._probe_timeout_s - (time.monotonic() - call.started_at),
        )
        if not call.done.wait(remaining):
            with self._probe_lock:
                call.timed_out = True
                # The native call may have finished between wait() timing out
                # and acquiring the lock. Clear it here in that case; otherwise
                # invoke() will clear it when the call eventually returns.
                if call.done.is_set() and self._probe_calls[call.capability] is call:
                    self._probe_calls[call.capability] = None
            logger.warning(
                "Mooncake native DFS %s probe timed out after %ss",
                call.capability,
                self._probe_timeout_s,
            )
            return -1

        with self._probe_lock:
            result = call.result
            timed_out = call.timed_out
            if self._probe_calls[call.capability] is call:
                self._probe_calls[call.capability] = None
        if timed_out:
            return -1
        if isinstance(result, Exception):
            logger.error(
                "Mooncake native DFS %s probe raised",
                call.capability,
                exc_info=(type(result), result, result.__traceback__),
            )
            return -1
        if result is None:
            logger.error(
                "Mooncake native DFS %s probe returned no result",
                call.capability,
            )
            return -1
        return int(result)

    def snapshot_unlocked(self) -> DfsHealthSnapshot:
        return DfsHealthSnapshot(
            self._read_state,
            self._write_state,
            self._generation,
            self._read_failures,
            self._write_failures,
            self._read_successes,
            self._write_successes,
            self._read_error,
            self._write_error,
        )

    def _next_state(
        self, state: str, success: bool, successes: int, failures: int
    ) -> tuple[str, int, int]:
        if success:
            successes += 1
            failures = 0
            if successes >= self._success_threshold:
                state = AVAILABLE
        else:
            failures += 1
            successes = 0
            if failures >= self._failure_threshold:
                state = UNAVAILABLE
        return state, successes, failures

    def _apply_policy(self, allow_read: bool, allow_write: bool) -> None:
        rc = int(self._store.set_dfs_access_policy(allow_read, allow_write))
        if rc != 0:
            raise RuntimeError(
                "Mooncake set_dfs_access_policy failed: "
                f"read={allow_read}, write={allow_write}, error={rc}"
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.probe_once()
            except Exception:
                logger.exception("Mooncake DFS health probe failed")
            if self._stop.wait(self._interval_s):
                break
