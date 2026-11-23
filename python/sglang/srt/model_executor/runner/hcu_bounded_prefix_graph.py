"""Breakable CUDA graphs for CP8 pre-tail and bounded SWA tail on HCU DSV4.1."""

import logging
import os
from typing import Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.runtime_context import get_forward, get_parallel

logger = logging.getLogger(__name__)

# Dense row buckets tuned for the launch-bound / collective-skew regime on HCU
# CP8: small/medium steps eliminate launch bubbles and cross-rank collective
# jitter, while large (>99% GPU-saturated) compute-bound steps fall back to
# eager execution with zero padding overhead.
_PRE_TAIL_MIN_ROWS: int = 32
_TAIL_MIN_ROWS: int = 16
_PRE_TAIL_BUCKETS: Tuple[int, ...] = (
    32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
)
_TAIL_BUCKETS: Tuple[int, ...] = (
    16, 32, 48, 64, 80, 96, 112, 128, 160, 192,
)
_PRE_TAIL_MAX_ROWS: int = int(
    os.environ.get("SGLANG_HCU_BOUNDED_PREFIX_BCG_MAX_ROWS", "512")
)
_TAIL_MAX_ROWS: int = int(
    os.environ.get("SGLANG_HCU_BOUNDED_TAIL_BCG_MAX_ROWS", "192")
)


def _pick_row_bucket(
    live_rows: int,
    buckets: Tuple[int, ...],
    max_rows: Optional[int] = None,
    min_rows: int = 1,
) -> Optional[int]:
    if live_rows < min_rows:
        return None
    if max_rows is not None and live_rows > max_rows:
        return None
    for b in buckets:
        if b >= live_rows and (max_rows is None or b <= max_rows):
            return b
    return None


def _admit_prefix_bucket(
    live_rows: int,
    cached_lens: Optional[Sequence[int]],
    buckets: Tuple[int, ...],
    *,
    min_rows: int,
    max_rows: Optional[int],
) -> Optional[int]:
    if not cached_lens or not any(p > 0 for p in cached_lens):
        return None
    return _pick_row_bucket(live_rows, buckets, max_rows=max_rows, min_rows=min_rows)


def _engram_output(runner, engram, hidden):
    return torch.empty_like(hidden)


@eager_on_graph(capture_stub=_engram_output)
def _engram(runner, engram, hidden):
    live_rows = getattr(runner, "live_rows", None)
    if live_rows is not None and live_rows < hidden.shape[0]:
        y_live = engram(
            hidden[:live_rows],
            runner.hash_ids[:, engram.layer_hash_index],
            runner.forward_batch,
            cp_all_tokens=True,
        )
        y = hidden.clone()
        y[:live_rows].copy_(y_live)
        return y
    return engram(
        hidden,
        runner.hash_ids[:, engram.layer_hash_index],
        runner.forward_batch,
        cp_all_tokens=True,
    )


def _layers_support_hcu_mega_bcg(layers) -> bool:
    return all(
        layer.self_attn.is_dsv41
        and getattr(layer.mlp.experts, "_mega_moe_hcu_w8a8_weights", False)
        and getattr(layer.mlp.experts, "_mega_moe_hcu_runtime", None) == "deep_gemm"
        for layer in layers
    )


class HcuBoundedPrefixGraph:
    """Run CP8 pre-tail layers (0..late_layer_start) via BCG and return full-row state."""

    # MegaMoE's expert path is captured; its router reads live request IDs.
    capture_experts = True

    def __init__(self, model):
        self.model = model
        parallel = get_parallel()
        self.supported = (
            model.start_layer == 0
            and model.late_layer_start > 0
            and parallel.attn_cp_size == parallel.tp_size == parallel.moe_ep_size == 8
            and parallel.attn_tp_size == 1
            and _layers_support_hcu_mega_bcg(
                model.layers[model.start_layer : model.late_layer_start]
            )
        )
        self.warmed = set()
        self.entries = {}
        self.forward_batch = self.global_ids = self.hash_ids = None
        self.live_rows: Optional[int] = None
        self.log_replays = envs.SGLANG_HCU_BOUNDED_PREFIX_BCG_LOG_REPLAYS.get()

    def run(self, hidden, positions, ids, global_ids, hashes, batch):
        if not self.supported or torch.cuda.is_current_stream_capturing():
            return None
        live_rows = hidden.shape[0]
        if positions.shape[0] != live_rows:
            return None

        bucket_rows = _admit_prefix_bucket(
            live_rows,
            batch.extend_prefix_lens_cpu,
            _PRE_TAIL_BUCKETS,
            min_rows=_PRE_TAIL_MIN_ROWS,
            max_rows=_PRE_TAIL_MAX_ROWS,
        )
        if bucket_rows is None:
            return None
        key = bucket_rows
        if key not in self.entries and len(self.entries) >= len(_PRE_TAIL_BUCKETS):
            return None
        tokens = bucket_rows * 8

        # Every rank sees the same global batch shape and warms the bucket first.
        if key not in self.warmed:
            self.warmed.add(key)
            return None

        self.forward_batch, self.global_ids, self.hash_ids = batch, global_ids, hashes
        try:
            with get_forward().scoped(dsv41_tail_graph_runner=self):
                if key not in self.entries:
                    static_hidden = torch.zeros(
                        (bucket_rows, *hidden.shape[1:]),
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    static_pos = torch.zeros(
                        (bucket_rows,),
                        dtype=positions.dtype,
                        device=positions.device,
                    )
                    static_hidden[:live_rows].copy_(hidden)
                    static_pos[:live_rows].copy_(positions)
                    self.live_rows = bucket_rows
                    graph = BreakableCUDAGraph()
                    torch.cuda.synchronize()
                    group = get_parallel().tp_group
                    group.barrier()
                    with graph_pool_capture_scope(), BreakableCUDAGraphCapture(
                        graph,
                        pool=torch.cuda.graph_pool_handle(),
                        stream=torch.cuda.current_stream(),
                        barrier_fn=group.barrier,
                    ):
                        result = self._forward(
                            static_hidden, static_pos, ids, global_ids, batch
                        )
                    self.entries[key] = (
                        graph,
                        static_hidden,
                        static_pos,
                        result.residual,
                        result.pre,
                    )
                    if get_parallel().attn_cp_rank == 0:
                        logger.info(
                            "bounded prefix capture bucket_rows=%d live_rows=%d tokens=%d layers=%d segments=%d breaks=%d",
                            bucket_rows,
                            live_rows,
                            tokens,
                            self.model.late_layer_start,
                            len(graph._segments),
                            len(graph._break_fns),
                        )
                graph, static_hidden, static_pos, residual, pre = self.entries[key]
                static_hidden[:live_rows].copy_(hidden)
                static_pos[:live_rows].copy_(positions)
                if live_rows < bucket_rows:
                    static_hidden[live_rows:].zero_()
                    static_pos[live_rows:].zero_()
                self.live_rows = live_rows
                with graph_pool_replay_scope():
                    graph.replay()
                if self.log_replays and get_parallel().attn_cp_rank == 0:
                    logger.info(
                        "bounded prefix replay bucket_rows=%d live_rows=%d tail_start=%d",
                        bucket_rows,
                        live_rows,
                        self.model.late_layer_start,
                    )
                if live_rows < bucket_rows:
                    return mhc.HcState(
                        residual[:live_rows],
                        pre[:live_rows] if pre is not None else None,
                    )
                return mhc.HcState(residual, pre)
        finally:
            self.forward_batch = self.global_ids = self.hash_ids = None
            self.live_rows = None

    def _forward(self, hidden, positions, ids, global_ids, batch):
        state = mhc.HcState(hidden)
        for i in range(self.model.start_layer, self.model.late_layer_start):
            layer = self.model.layers[i]
            if layer.engram is not None:
                state = state.with_residual(_engram(self, layer.engram, state.residual))
            state = layer.forward_hc_pre_from_prev(
                positions, state, ids, batch, global_ids, seam_open=False,
            )
        return state.materialized(layer.hc_cfg)


class HcuBoundedTailMegaGraph:
    """Run CP8 bounded SWA tail layers (late_layer_start..end_layer) via BCG for MegaMoE."""

    capture_experts = True

    def __init__(self, model):
        self.model = model
        parallel = get_parallel()
        self.supported = (
            model.late_layer_start is not None
            and model.late_layer_start < model.end_layer
            and parallel.attn_cp_size == parallel.tp_size == parallel.moe_ep_size == 8
            and parallel.attn_tp_size == 1
            and _layers_support_hcu_mega_bcg(
                model.layers[model.late_layer_start : model.end_layer]
            )
        )
        self.warmed = set()
        self.entries = {}
        self.forward_batch = self.global_ids = self.hash_ids = None
        self.live_rows: Optional[int] = None
        self.log_replays = envs.SGLANG_HCU_BOUNDED_PREFIX_BCG_LOG_REPLAYS.get()

    def run(self, state, positions, ids, global_ids, hashes, batch, tail):
        if not self.supported or torch.cuda.is_current_stream_capturing():
            return None
        if not isinstance(state.streams, torch.Tensor) or state.input is not None:
            return None
        cp = getattr(tail, "cp_metadata", None)
        if cp is None or len(set(cp.per_rank_actual_token)) != 1:
            return None
        live_rows = state.residual.shape[0]
        if positions.shape[0] != live_rows:
            return None
        bucket_rows = _admit_prefix_bucket(
            live_rows,
            batch.extend_prefix_lens_cpu,
            _TAIL_BUCKETS,
            min_rows=_TAIL_MIN_ROWS,
            max_rows=_TAIL_MAX_ROWS,
        )
        if bucket_rows is None:
            return None
        has_pre = state.pre is not None
        key = (bucket_rows, has_pre)
        if key not in self.entries and len(self.entries) >= len(_TAIL_BUCKETS) * 2:
            return None
        if key not in self.warmed:
            self.warmed.add(key)
            return None

        self.forward_batch, self.global_ids, self.hash_ids = batch, global_ids, hashes
        try:
            with get_forward().scoped(dsv41_tail_graph_runner=self):
                if key not in self.entries:
                    static_res = torch.zeros(
                        (bucket_rows, *state.residual.shape[1:]),
                        dtype=state.residual.dtype,
                        device=state.residual.device,
                    )
                    static_res[:live_rows].copy_(state.residual)
                    static_pre = None
                    if has_pre:
                        static_pre = torch.zeros(
                            (bucket_rows, *state.pre.shape[1:]),
                            dtype=state.pre.dtype,
                            device=state.pre.device,
                        )
                        static_pre[:live_rows].copy_(state.pre)
                    static_pos = torch.zeros(
                        (bucket_rows,),
                        dtype=positions.dtype,
                        device=positions.device,
                    )
                    static_pos[:live_rows].copy_(positions)
                    self.live_rows = bucket_rows
                    graph = BreakableCUDAGraph()
                    torch.cuda.synchronize()
                    group = get_parallel().tp_group
                    group.barrier()
                    with graph_pool_capture_scope(), BreakableCUDAGraphCapture(
                        graph,
                        pool=torch.cuda.graph_pool_handle(),
                        stream=torch.cuda.current_stream(),
                        barrier_fn=group.barrier,
                    ):
                        in_state = mhc.HcState(static_res, static_pre)
                        result = self._forward(
                            in_state, static_pos, ids, global_ids, batch
                        )
                    self.entries[key] = (
                        graph,
                        static_res,
                        static_pre,
                        static_pos,
                        result.residual,
                        result.pre,
                    )
                    if get_parallel().attn_cp_rank == 0:
                        logger.info(
                            "bounded tail mega capture bucket_rows=%d live_rows=%d layers=%d..%d segments=%d breaks=%d",
                            bucket_rows,
                            live_rows,
                            self.model.late_layer_start,
                            self.model.end_layer - 1,
                            len(graph._segments),
                            len(graph._break_fns),
                        )
                graph, static_res, static_pre, static_pos, residual, pre = self.entries[
                    key
                ]
                static_res[:live_rows].copy_(state.residual)
                if static_pre is not None:
                    static_pre[:live_rows].copy_(state.pre)
                static_pos[:live_rows].copy_(positions)
                if live_rows < bucket_rows:
                    static_res[live_rows:].zero_()
                    if static_pre is not None:
                        static_pre[live_rows:].zero_()
                    static_pos[live_rows:].zero_()
                self.live_rows = live_rows
                with graph_pool_replay_scope():
                    graph.replay()
                if self.log_replays and get_parallel().attn_cp_rank == 0:
                    logger.info(
                        "bounded tail mega replay bucket_rows=%d live_rows=%d",
                        bucket_rows,
                        live_rows,
                    )
                if live_rows < bucket_rows:
                    return mhc.HcState(
                        residual[:live_rows],
                        pre[:live_rows] if pre is not None else None,
                    )
                return mhc.HcState(residual, pre)
        finally:
            self.forward_batch = self.global_ids = self.hash_ids = None
            self.live_rows = None

    def _forward(self, state, positions, ids, global_ids, batch):
        for i in range(self.model.late_layer_start, self.model.end_layer):
            layer = self.model.layers[i]
            if layer.engram is not None:
                state = state.with_residual(_engram(self, layer.engram, state.residual))
            state = layer.forward_hc_pre_from_prev(
                positions, state, ids, batch, global_ids, seam_open=False,
            )
        return state.materialized(layer.hc_cfg)
