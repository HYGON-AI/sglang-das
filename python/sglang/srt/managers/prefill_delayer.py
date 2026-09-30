import dataclasses
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


class RecentPrefillBatchSizeTracker:
    """Track the largest of the latest non-empty prefill attempts.

    The default window keeps 16 attempts. Successful admissions use their
    actual batch size; rejected attempts use a conservative local estimate.
    Decode-only and idle scheduler passes do not age the high-watermark.
    """

    def __init__(self, window_size: int = 16):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self._recent_attempt_sizes = deque(maxlen=window_size)

    @property
    def max_prefill_bs(self) -> int:
        return max(self._recent_attempt_sizes, default=0)

    def observe_attempt(self, attempted_prefill_bs: int) -> int:
        if attempted_prefill_bs <= 0:
            raise ValueError(
                "attempted_prefill_bs must be positive for a non-empty attempt, "
                f"got {attempted_prefill_bs}"
            )
        self._recent_attempt_sizes.append(attempted_prefill_bs)
        return self.max_prefill_bs


@dataclass(frozen=True)
class _State:
    delayed_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    max_waiting_queue_len: int = 0
    last_queue_growth_time: float = field(default_factory=time.perf_counter)

    def bump_delayed_count(self) -> "_State":
        return dataclasses.replace(self, delayed_count=self.delayed_count + 1)

    def observe_waiting_queue_len(self, waiting_queue_len: int, now: float) -> "_State":
        if waiting_queue_len <= self.max_waiting_queue_len:
            return self
        return dataclasses.replace(
            self,
            max_waiting_queue_len=waiting_queue_len,
            last_queue_growth_time=now,
        )


class _NegotiateOutput(NamedTuple):
    next_state: Optional[_State]
    input_estimation: str
    output_allow: bool
    output_reason: str
    num_prefillable: int
    num_token_watermark_force_allow: int
    # Accumulated wait of the prefill being released on this pass. Carried
    # explicitly because `next_state` is None on every release path and thus
    # cannot convey it to the metrics observation.
    wait_forward_passes: int = 0
    wait_seconds: float = 0.0


class PrefillDelayer:
    def __init__(
        self,
        dp_size: int,
        attn_tp_size: int,
        attn_cp_size: int,
        cpu_group,
        server_args,
        max_delay_passes: int,
        token_usage_low_watermark: Optional[float],
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
        device: Optional["torch.device"] = "cpu",
        device_group=None,
        debug_log_enabled: bool = True,
    ):
        self._max_delay_passes = max_delay_passes
        self._token_usage_low_watermark = token_usage_low_watermark
        self._debug_log_enabled = _DEBUG_LOG and debug_log_enabled
        # Queue-based trigger is opt-in: activates only when queue_min_ratio
        # is explicitly set. Additive with the slot-based trigger.
        self._queue_min_ratio = server_args.prefill_delayer_queue_min_ratio
        # Fall back to 5000ms if unset; this is a local safety cap, not a
        # semantic default, so we don't surface it via ServerArgs.
        self._max_delay_ms = server_args.prefill_delayer_max_delay_ms
        if self._max_delay_ms is None:
            self._max_delay_ms = 5000.0
        self._enable_idle_coalescing = getattr(
            server_args, "enable_prefill_idle_coalescing", False
        )
        self._idle_coalesce_max_delay_ms = getattr(
            server_args, "prefill_idle_coalesce_max_delay_ms", 50.0
        )
        self._idle_coalesce_settle_ms = getattr(
            server_args, "prefill_idle_coalesce_settle_ms", 10.0
        )
        self._idle_coalesce_burst_max_delay_ms = getattr(
            server_args, "prefill_idle_coalesce_burst_max_delay_ms", 500.0
        )
        self._idle_coalesce_max_batch_size = getattr(
            server_args, "prefill_idle_coalesce_max_batch_size", 32
        )
        if self._idle_coalesce_max_delay_ms <= 0:
            raise ValueError(
                "prefill_idle_coalesce_max_delay_ms must be positive, got "
                f"{self._idle_coalesce_max_delay_ms}"
            )
        if self._idle_coalesce_settle_ms <= 0:
            raise ValueError(
                "prefill_idle_coalesce_settle_ms must be positive, got "
                f"{self._idle_coalesce_settle_ms}"
            )
        if self._idle_coalesce_burst_max_delay_ms <= 0:
            raise ValueError(
                "prefill_idle_coalesce_burst_max_delay_ms must be positive, got "
                f"{self._idle_coalesce_burst_max_delay_ms}"
            )
        if self._idle_coalesce_max_batch_size < 2:
            raise ValueError(
                "prefill_idle_coalesce_max_batch_size must be at least 2, got "
                f"{self._idle_coalesce_max_batch_size}"
            )
        self._queue_trigger_enabled = self._queue_min_ratio is not None
        self._prefill_max_requests = server_args.prefill_max_requests
        logger.info(
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark} "
            f"queue_min_ratio={self._queue_min_ratio} "
            f"max_delay_ms={self._max_delay_ms} "
            f"idle_coalescing={self._enable_idle_coalescing} "
            f"idle_coalesce_max_delay_ms={self._idle_coalesce_max_delay_ms} "
            f"idle_coalesce_settle_ms={self._idle_coalesce_settle_ms} "
            f"idle_coalesce_burst_max_delay_ms="
            f"{self._idle_coalesce_burst_max_delay_ms} "
            f"idle_coalesce_max_batch_size={self._idle_coalesce_max_batch_size} "
            f"queue_trigger_enabled={self._queue_trigger_enabled}"
        )
        self.dp_size = dp_size
        self.enable_dp_attention = get_parallel().enable_dp_attention
        dp_size_dim = dp_size if self.enable_dp_attention else 1

        # Mirror scheduler_dp_attn_mixin's NCCL all-gather path: when the
        # env flag is on (or overlap scheduling is disabled), ride the NCCL
        # device group on `device` instead of gloo on CPU.
        use_nccl = (
            get_schedule().disable_overlap_schedule
            or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
        )
        if use_nccl:
            assert (
                device_group is not None
            ), "device_group is required when using NCCL for PrefillDelayer all-gather"
            self._gather_group = device_group
            self._gather_device = device
        else:
            self._gather_group = cpu_group
            self._gather_device = "cpu"

        # Fields packed per rank into the all-gather tensor: prefillable,
        # token_watermark_force_allow, running_batch, max_prefill_bs,
        # waiting_queue_len, wait_elapsed_us, queue_quiet_elapsed_us.
        # The gather group contains every attention TP *and* CP rank for each
        # DP replica.  Keep those dimensions folded together and select one
        # representative below, matching MLPSyncBatchInfo's layout.  Omitting
        # CP here makes all_gather_into_tensor undersized for DP+CP topologies.
        self._global_info_buffer = torch.empty(
            (dp_size_dim, attn_tp_size * attn_cp_size, 7),
            dtype=torch.int64,
            device=self._gather_device,
        )

        self._metrics_collector = metrics_collector

        self._curr_state: Optional[_State] = None
        self.skip_first_delayer = True

        assert (
            not get_schedule().disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

    def _negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        token_usage: float,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> _NegotiateOutput:
        out = self._negotiate_should_allow_prefill_pure(
            prev_state=self._curr_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
            running_batch=running_batch,
            max_prefill_bs=max_prefill_bs,
            max_running_requests=max_running_requests,
            waiting_queue_len=waiting_queue_len,
        )
        self._curr_state = out.next_state
        return out

    # (Almost) pure function, do not modify self state
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: Optional[_State],
        local_prefillable: bool,
        token_usage: float,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> _NegotiateOutput:
        # Compute local states
        local_token_watermark_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        # Gather global states
        # Never use each worker's local clock directly for a scheduling
        # decision.  Around the deadline one CP/TP worker can cross the
        # threshold one scheduler pass before another; allowing on only a
        # subset of ranks forks collective ordering and deadlocks the model.
        # Publish the local observation in the existing all-gather and make
        # every worker consume the same representative-rank decision below.
        now = time.perf_counter()
        local_observed_state = prev_state
        if local_observed_state is not None:
            local_observed_state = local_observed_state.observe_waiting_queue_len(
                waiting_queue_len, now
            )
        local_wait_elapsed_us = int(
            (now - local_observed_state.start_time) * 1_000_000
            if local_observed_state is not None
            else 0
        )
        local_queue_quiet_elapsed_us = int(
            (now - local_observed_state.last_queue_growth_time) * 1_000_000
            if local_observed_state is not None
            else 0
        )
        tp0_info = self._gather_info(
            local_prefillable=local_prefillable,
            local_token_watermark_force_allow=local_token_watermark_force_allow,
            running_batch=running_batch,
            max_prefill_bs=max_prefill_bs,
            waiting_queue_len=waiting_queue_len,
            wait_elapsed_us=local_wait_elapsed_us,
            queue_quiet_elapsed_us=local_queue_quiet_elapsed_us,
        )
        global_prefillable = tp0_info[:, 0]
        global_token_watermark_force_allow = tp0_info[:, 1]
        global_running_batch = tp0_info[:, 2]
        global_max_prefill_bs = tp0_info[:, 3]
        global_waiting_queue_len = tp0_info[:, 4]
        # Wait for every DP representative to reach its deadline.  All
        # TP/CP workers see the same gathered tensor and therefore take the
        # same release path.  For the common DP=1 case this is rank 0's clock.
        global_wait_elapsed_ms = tp0_info[:, 5].min().item() / 1000.0
        global_queue_quiet_elapsed_ms = tp0_info[:, 6].min().item() / 1000.0

        # Compute derived global states
        if global_prefillable.min().item() > 0:
            prefillable_status = "all"
        elif global_prefillable.max().item() == 0:
            prefillable_status = "none"
        else:
            prefillable_status = "mixed"
        global_exists_token_watermark_force_allow = (
            global_token_watermark_force_allow.max().item() > 0
        )
        debug_info = dict(
            input_estimation=prefillable_status,
            num_prefillable=global_prefillable.sum().item(),
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),
        )

        # Wait accumulated so far, taken from prev_state. Release paths attach
        # this so the wait histograms observe the real value; delay paths leave
        # the defaults (0) since the wait isn't finished and isn't observed.
        wait_info = dict(
            wait_forward_passes=prev_state.delayed_count if prev_state else 0,
            wait_seconds=(
                (time.perf_counter() - prev_state.start_time) if prev_state else 0.0
            ),
        )

        # Compute outputs
        if prefillable_status == "all":
            # Safety valve: low KV usage means GPU is underutilized, skip
            # delay. Mirrors the check in the "mixed" branch.
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                    **wait_info,
                )

            if not self.enable_dp_attention:
                max_running_requests = (
                    max_running_requests + self.dp_size - 1
                ) // self.dp_size

            global_running_batch_max = int(global_running_batch.max().item())
            global_max_prefill_bs_max = int(global_max_prefill_bs.max().item())
            global_waiting_queue_max = int(global_waiting_queue_len.max().item())

            # Keep every rank's local wait state synchronized to the same
            # global queue high-watermark. If any DP replica observed growth,
            # all ranks reset their quiet window on this scheduler pass.
            idle_state = local_observed_state
            if idle_state is None:
                idle_state = _State(
                    start_time=now,
                    max_waiting_queue_len=global_waiting_queue_max,
                    last_queue_growth_time=now,
                )
            elif global_waiting_queue_max > idle_state.max_waiting_queue_len:
                idle_state = idle_state.observe_waiting_queue_len(
                    global_waiting_queue_max, now
                )
                global_queue_quiet_elapsed_ms = 0.0

            # Queue-based trigger: delay prefill until the waiting queue
            # reaches queue_min = min(running_req * ratio, max_prefill_bs),
            # capped by a wall-clock timeout to bound worst-case TTFT.
            # Targets workloads where decode requests finish one-at-a-time
            # and fragment prefill into many tiny batches.
            queue_condition = False
            if self._queue_trigger_enabled and global_running_batch_max > 0:
                queue_capacity = (
                    self._prefill_max_requests
                    if self._prefill_max_requests is not None
                    else global_max_prefill_bs_max
                )
                queue_min_effective = min(
                    int(global_running_batch_max * self._queue_min_ratio),
                    queue_capacity,
                )
                queue_condition = (
                    queue_min_effective > 0
                    and global_waiting_queue_max < queue_min_effective
                )
                if queue_condition and global_wait_elapsed_ms >= self._max_delay_ms:
                    queue_condition = False

            # A deliberately narrow idle-only coalescing hook. It applies only
            # to the first batch on an idle engine. The first request waits up
            # to max_delay_ms. Once two or more requests exist, queue growth
            # resets a short settle window, with a separate larger burst cap.
            # This allows a concurrent burst to naturally form BS4/8/16/32
            # without making C1 pay the burst-scale timeout.
            # Reaching max_batch_size releases immediately; actual token/KV
            # admission remains the ordinary scheduler's responsibility.
            idle_coalesce_condition = (
                self._enable_idle_coalescing
                and global_running_batch_max == 0
                and global_waiting_queue_max > 0
                and global_waiting_queue_max < self._idle_coalesce_max_batch_size
                and (
                    (
                        global_waiting_queue_max == 1
                        and global_wait_elapsed_ms < self._idle_coalesce_max_delay_ms
                    )
                    or (
                        global_waiting_queue_max >= 2
                        and global_wait_elapsed_ms
                        < self._idle_coalesce_burst_max_delay_ms
                        and global_queue_quiet_elapsed_ms
                        < self._idle_coalesce_settle_ms
                    )
                )
            )

            slot_condition = (
                max_running_requests - global_running_batch_max
                < global_max_prefill_bs_max
            )

            if slot_condition or queue_condition or idle_coalesce_condition:
                # When the "max_decode_bs - running_bs < max_prefill_bs" condition is met,
                # the first merge_batch causes the decoding to fail to reach the maximum batch size.
                if self.skip_first_delayer and not idle_coalesce_condition:
                    self.skip_first_delayer = False
                    pass
                else:
                    # Bound the wait like the "mixed" branch: on a saturated
                    # engine slot_condition may never turn false, so cap the
                    # delay by max_delay_passes.
                    prev_delayed_count = prev_state.delayed_count if prev_state else 0
                    if prev_delayed_count < self._max_delay_passes - 1:
                        next_state = (
                            idle_state
                            if idle_coalesce_condition
                            else (local_observed_state or _State())
                        )
                        next_state = next_state.bump_delayed_count()
                        return _NegotiateOutput(
                            next_state=next_state,
                            output_allow=False,
                            output_reason="delay",
                            **debug_info,
                        )
                    return _NegotiateOutput(
                        next_state=None,
                        output_allow=True,
                        output_reason="wait_timeout",
                        **debug_info,
                        **wait_info,
                    )
            exist_previous_wait = prev_state is not None
            return _NegotiateOutput(
                next_state=None,
                output_allow=True,
                output_reason="wait_success" if exist_previous_wait else "no_wait",
                **debug_info,
                **wait_info,
            )
        elif prefillable_status == "none":
            return _NegotiateOutput(
                next_state=None,
                # It does not matter whether we allow or not, thus we allow for simplicity
                output_allow=True,
                output_reason="",
                **debug_info,
                **wait_info,
            )
        elif prefillable_status == "mixed":
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                    **wait_info,
                )

            prev_delayed_count = prev_state.delayed_count if prev_state else 0
            if prev_delayed_count < self._max_delay_passes - 1:
                next_state = prev_state or _State()
                next_state = next_state.bump_delayed_count()
                return _NegotiateOutput(
                    next_state=next_state,
                    output_allow=False,
                    output_reason="delay",
                    **debug_info,
                )
            else:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="wait_timeout",
                    **debug_info,
                    **wait_info,
                )
        else:
            raise NotImplementedError

    def _gather_info(
        self,
        local_prefillable: bool,
        local_token_watermark_force_allow: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        waiting_queue_len: int = 0,
        wait_elapsed_us: int = 0,
        queue_quiet_elapsed_us: int = 0,
    ):
        local_info = torch.tensor(
            [
                int(local_prefillable),
                int(local_token_watermark_force_allow),
                running_batch,
                max_prefill_bs,
                waiting_queue_len,
                wait_elapsed_us,
                queue_quiet_elapsed_us,
            ],
            device=self._gather_device,
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._gather_group,
        )
        tp0_info = self._global_info_buffer[:, 0, :]
        return tp0_info


class PrefillDelayerSinglePassExecutor:
    def __init__(self, prefill_delayer: PrefillDelayer, token_usage: float):
        self._prefill_delayer = prefill_delayer
        self._token_usage = token_usage
        self._result: Optional[_NegotiateOutput] = None
        self._attempted_prefill_bs = 0

    @property
    def _called(self) -> bool:
        return self._result is not None

    def finalize(self, *, actual_prefill_bs: int) -> int:
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        _record_single_pass_result(
            actual_execution=actual_prefill_bs > 0,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
            debug_log_enabled=self._prefill_delayer._debug_log_enabled,
        )
        return actual_prefill_bs or self._attempted_prefill_bs

    def _estimate_attempted_prefill_bs(
        self,
        *,
        running_batch: int,
        max_running_requests: int,
        waiting_queue_len: int,
    ) -> int:
        local_max_running_requests = max_running_requests
        if not self._prefill_delayer.enable_dp_attention:
            local_max_running_requests = (
                max_running_requests + self._prefill_delayer.dp_size - 1
            ) // self._prefill_delayer.dp_size

        # The delayer negotiates before PrefillAdder materializes can_run_list,
        # so a rejected pass has no exact batch size. This upper bound is exact
        # when the waiting queue is the limiter (for example, two queued
        # requests after a cached BS=10 spike), and it never exceeds the local
        # request slots available to the candidate batch.
        free_slots = max(local_max_running_requests - running_batch, 1)
        non_empty_queue_len = max(waiting_queue_len, 1)
        return min(non_empty_queue_len, free_slots)

    def negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> bool:
        if local_prefillable:
            self._attempted_prefill_bs = max(
                self._attempted_prefill_bs,
                self._estimate_attempted_prefill_bs(
                    running_batch=running_batch,
                    max_running_requests=max_running_requests,
                    waiting_queue_len=waiting_queue_len,
                ),
            )
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
                running_batch=running_batch,
                max_prefill_bs=max_prefill_bs,
                max_running_requests=max_running_requests,
                waiting_queue_len=waiting_queue_len,
            )
        return self._result.output_allow


def _record_single_pass_result(
    actual_execution: bool,
    output: _NegotiateOutput,
    metrics_collector: Optional["SchedulerMetricsCollector"],
    *,
    debug_log_enabled: bool,
) -> None:
    if debug_log_enabled:
        if output.output_allow and (output.output_reason == "wait_timeout"):
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={output.num_prefillable}, "
                f"actual_execution={actual_execution})"
            )
        elif output.output_allow and (output.output_reason == "token_watermark"):
            logger.info(
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={output.num_prefillable}, "
                f"num_token_watermark_force_allow={output.num_token_watermark_force_allow}, "
                f"actual_execution={actual_execution})"
            )
        else:
            assert output.output_reason in {
                "",
                "wait_success",
                "no_wait",
                "delay",
            }

    if metrics_collector is not None:
        metrics_collector.observe_prefill_delayer_outcome(
            forward_passes=output.wait_forward_passes,
            wait_seconds=output.wait_seconds,
            input_estimation=output.input_estimation,
            output_allow=output.output_allow,
            output_reason=output.output_reason,
            actual_execution=actual_execution,
        )
