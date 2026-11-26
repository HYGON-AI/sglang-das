import json
import logging
from copy import copy
from typing import Any, Optional

import msgspec
import torch


class FoldedExtend(msgspec.Struct, frozen=True):
    """An extend batch whose prefix hits start at their replay start."""

    batch: Any  # ScheduleBatch copy fed to ForwardBatch.init_new
    keep_rows: torch.Tensor  # [original rows] int64, folded-row index of each
    scheduled_row: torch.Tensor  # [folded rows] int64, original row, -1 if replayed
    row_floor: torch.Tensor  # [folded rows] int64
    replay_lens: list[int]  # [bs] leading replay rows per request
    num_rows: int


def supports_encoder_swa_replay_images() -> bool:
    """Image replay needs the CUDA fold or the HCU synchronized replay path."""
    from sglang.srt.runtime_context import get_platform
    from sglang.srt.utils import is_hcu

    return get_platform().is_cuda or is_hcu()


def fold_encoder_swa_replay(worker, batch) -> Optional[FoldedExtend]:
    """Prepend each hit's <=128 replay tokens to its extend instead of replaying."""
    runner = worker.model_runner
    window = runner.token_to_kv_pool.request_window
    if window is None or not batch.forward_mode.is_extend_without_speculative():
        return None
    rows = _reset_windows_and_collect_hits(batch=batch, window=window)
    if not rows:
        return None
    return _fold_batch(batch=batch, runner=runner, rows=rows)


def _hcu_fold_supported(runner, batch, parallel) -> bool:
    """A bounded eager CP8 domain whose decoder tail contains only new rows."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.layers.cp.utils import enable_cp_v2
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_helpers import (
        is_cp_cache_layer_split_deepseek_v4_pool,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.runtime_context import (
        get_disagg,
        get_resources,
        get_schedule,
        max_prefill_buffer_tokens,
    )
    from sglang.srt.utils.common import ceil_align

    resources = get_resources()
    if (
        batch.forward_mode != ForwardMode.EXTEND
        or not batch.reqs
        or parallel.attn_cp_size != 8
        or parallel.attn_dp_size != 1
        or parallel.pp_size != 1
        or parallel.cp_strategy != "interleave"
        or enable_cp_v2()
        or get_disagg().disaggregation_mode != "prefill"
        or not getattr(runner.attn_backend, "is_dsv41", False)
        or not runner.spec_algorithm.is_dspark()
        or not is_cp_cache_layer_split_deepseek_v4_pool(runner.token_to_kv_pool)
        or (
            getattr(runner, "prefill_cuda_graph_runner", None) is not None
            and not isinstance(runner.prefill_cuda_graph_runner, EagerRunner)
        )
        or getattr(batch, "return_logprob", False)
        or getattr(batch, "spec_info", None) is not None
        or getattr(batch, "has_grammar", False)
        or getattr(batch, "input_embeds", None) is not None
        # Capture writes back by cache location, including any replay rows.
        or resources.experts_capturer is not None
        or resources.indexer_capturer is not None
    ):
        return False
    resets = getattr(batch, "encoder_swa_reset", None)
    if resets is None:
        resets = [True] * len(batch.reqs)
    total_replay = 0
    replay_starts: list[Optional[int]] = []
    for reset, prefix, extend in zip(
        resets, batch.prefix_lens, batch.extend_lens, strict=True
    ):
        if reset and prefix:
            # 128 is divisible by CP8: prepending it preserves every new row's owner.
            # The DSpark tail is at most 128 rows, including with FULL capture.
            # At least 128 new rows therefore keep replay rows out of that tail.
            if prefix < 128 or prefix % 2 or extend < 128:
                return False
            replay = replay_len(prefix)
            total_replay += replay
            replay_starts.append(prefix - replay)
        else:
            replay_starts.append(None)
    if total_replay == 0:
        return False
    execution_rows = ceil_align(
        sum(batch.extend_lens) + total_replay, parallel.attn_tp_size
    )
    execution_rows = ceil_align(execution_rows, get_cp_padding_align_size())
    execution_limit = min(
        get_schedule().chunked_prefill_size or 0, max_prefill_buffer_tokens()
    )
    if get_moe_a2a_backend().is_megamoe():
        # MegaMoE receives the interleaved CP shard, including padding rows.
        per_rank_limit = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        execution_limit = min(execution_limit, per_rank_limit * parallel.attn_cp_size)
    if execution_rows > execution_limit:
        return False
    if not _hcu_fold_images_are_before_replay(batch, replay_starts):
        return False
    counts = batch.global_num_tokens
    return counts is None or (
        len(counts) == 1
        and counts[0] == batch.extend_num_tokens
        and batch.extend_logprob_start_lens is not None
    )


def _hcu_fold_images_are_before_replay(
    batch, replay_starts: list[Optional[int]]
) -> bool:
    """Keep image features on the existing path when a forward touches them."""
    raw_inputs = batch.multimodal_inputs or []
    if not any(x is not None for x in raw_inputs):
        return True
    from sglang.srt.managers.schedule_batch import Modality

    for inputs_for_request, replay_start in zip(raw_inputs, replay_starts):
        if inputs_for_request is None:
            continue
        if replay_start is None:
            return False
        items = getattr(inputs_for_request, "mm_items", ())
        if not items:
            return False
        for item in items:
            if item.modality != Modality.IMAGE or not item.offsets:
                return False
            # The processor reports inclusive token ranges. Retain MM metadata
            # for the forward, but only fold when there are no image rows to embed.
            if any(
                not 0 <= first <= last < replay_start for first, last in item.offsets
            ):
                return False
    return True


def try_fold_hcu_encoder_swa_replay(worker, batch) -> Optional[FoldedExtend]:
    """Keep the general HCU guard; fold only the checked eager CP domain.

    All other HCU cases still use the synchronized separate replay. CUDA's
    existing fold selection and the scheduler's global declaration are unchanged.
    """
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import is_hcu

    runner = worker.model_runner
    if (
        not is_hcu()
        or getattr(runner.token_to_kv_pool, "request_window", None) is None
        or not batch.forward_mode.is_extend_without_speculative()
        or not any(
            reset and prefix
            for reset, prefix in zip(batch.encoder_swa_reset, batch.prefix_lens)
        )
    ):
        return None
    parallel = get_parallel()
    if not _hcu_fold_supported(runner, batch, parallel):
        return None
    # This is the first replay collective on either path. A negative extend
    # length distinguishes a fold vote from a separate replay's prefix plan.
    plan = tuple(
        (req.rid, bool(reset), int(prefix))
        for req, reset, prefix in zip(
            batch.reqs, batch.encoder_swa_reset, batch.prefix_lens, strict=True
        )
    ) + (("hcu-eager-fold-domain", True, -int(batch.extend_num_tokens)),)
    _check_hcu_cp_replay_plan(
        plan, parallel.attn_cp_size, parallel.attn_cp_group.cpu_group
    )
    folded = fold_encoder_swa_replay(worker, batch)
    assert folded is not None
    if batch.engram_history is None:
        # HCU's host-table path may defer history creation. The folded input
        # starts 128 rows earlier than the scheduler's original extend.
        folded.batch.engram_history = _engram_history(
            reqs=batch.reqs, starts=folded.batch.prefix_lens, runner=runner
        )
    # DP1's scheduler counted the original extend because the general HCU fold
    # property stays false. Only this copied execution batch gains replay rows.
    _set_execution_counts(batch=folded.batch, scheduled=batch)
    folded.batch.global_cp_num_tokens = None  # legacy interleave, never CP-v2
    if parallel.attn_cp_rank == 0:
        if len(batch.reqs) == 1:
            prefix = batch.prefix_lens[0]
            logger.info(
                "Encoder SWA folded replay: rid=%s prefix=%d start=%d "
                "replay_tokens=128 new_tokens=%d execution_tokens=%d cp_size=8",
                batch.reqs[0].rid,
                prefix,
                prefix - 128,
                batch.extend_num_tokens,
                folded.num_rows,
            )
        else:
            replay_reqs = sum(1 for r in folded.replay_lens if r)
            replay_tokens = sum(folded.replay_lens)
            logger.info(
                "Encoder SWA folded replay: num_reqs=%d replay_reqs=%d "
                "replay_tokens=%d new_tokens=%d execution_tokens=%d cp_size=8",
                len(batch.reqs),
                replay_reqs,
                replay_tokens,
                batch.extend_num_tokens,
                folded.num_rows,
            )
    return folded


def resets_swa_window(req) -> bool:
    """Whether this step starts the request's SWA window: a new request slot or a
    retraction. The prefill budget and `encoder_swa_reset` both read this."""
    return req.kv.req_pool_idx is None or req.is_retracted


def replay_len(prefix_len: int) -> int:
    """Cached tokens replayed in front of a hit's extend: the last SWA window.
    The prefill budget, the MLP-sync row count and the fold all read this."""
    return min(prefix_len, 128)


def folded_replay_rows(batch) -> int:
    if not batch.forward_mode.is_extend_without_speculative():
        return 0
    return sum(
        replay_len(end)
        for reset, end in zip(batch.encoder_swa_reset, batch.prefix_lens)
        if reset
    )


def apply_folded_extend(folded: FoldedExtend, forward_batch) -> None:
    forward_batch.encoder_swa_row_floor = folded.row_floor
    forward_batch.encoder_swa_compress_rows = folded.keep_rows


def drop_folded_rows(
    *, logits_output, folded: FoldedExtend, tail_lens: Optional[list[int]] = None
) -> None:
    """Hand downstream consumers (the DSpark draft) only the original extend rows."""
    hidden = logits_output.hidden_states
    token_indices = logits_output.hidden_states_token_indices
    if token_indices is not None:
        # Decoder-SWA tail rows index the folded extend; keep those on original rows.
        keep = _tail_rows_on_original(
            folded=folded, tail_lens=tail_lens, num_tail_rows=token_indices.shape[0]
        ).to(token_indices.device, non_blocking=True)
        logits_output.hidden_states_token_indices = folded.scheduled_row[
            token_indices[keep]
        ]
        if hidden is not None:
            logits_output.hidden_states = hidden[keep]
    elif hidden is not None and hidden.shape[0] >= folded.num_rows:
        # MLP-sync padding rows, if any, follow the real rows and are dropped too.
        logits_output.hidden_states = hidden[folded.keep_rows]


def _tail_rows_on_original(
    *, folded: FoldedExtend, tail_lens: list[int], num_tail_rows: int
):
    keep, off = [], 0
    for n, r, tail in zip(folded.batch.extend_lens, folded.replay_lens, tail_lens):
        keep.append(torch.arange(off + max(0, tail - (n - r)), off + tail))
        off += tail
    assert off == num_tail_rows, (off, num_tail_rows)
    return torch.cat(keep)


def run_encoder_swa_replay(worker, batch):
    from sglang.srt.utils import is_hcu

    if is_hcu():
        return _run_hcu_encoder_swa_replay(worker, batch)

    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )

    runner = worker.model_runner
    window = runner.token_to_kv_pool.request_window
    if window is None or not batch.forward_mode.is_extend_without_speculative():
        return
    rows = _reset_windows_and_collect_hits(batch=batch, window=window)
    if not rows:
        return
    # One forward replays every hitting request; each request's window is
    # floored at its own start, so the result matches per-request replays.
    replay = _build_replay_batch(batch=batch, runner=runner, rows=rows)
    fb = ForwardBatch.init_new(
        replay,
        runner,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    fb.encoder_swa_replay = True
    runner.forward(fb)


def _reset_windows_and_collect_hits(*, batch, window):
    rows = []
    for i, reset in enumerate(batch.encoder_swa_reset):
        if not reset:
            continue
        window.reset(batch.req_pool_indices[i : i + 1])
        end = batch.prefix_lens[i]
        if not end:
            continue
        if end % 2:
            raise ValueError(
                "encoder SWA replay requires an even cached-prefix boundary"
            )
        rows.append((i, end - replay_len(end), end))
    return rows


def _build_replay_batch(*, batch, runner, rows):
    idx = [i for i, _, _ in rows]
    starts = [s for _, s, _ in rows]
    ends = [e for _, _, e in rows]
    lens = [e - s for s, e in zip(starts, ends)]
    reqs = [batch.reqs[i] for i in idx]
    slots = batch.req_pool_indices[idx]
    req_to_token = runner.req_to_token_pool.req_to_token

    replay = copy(batch)
    replay.reqs = reqs
    replay.input_ids = torch.tensor(
        [
            t
            for r, s, e in zip(reqs, starts, ends)
            for t in r.full_untruncated_fill_ids[s:e]
        ],
        dtype=torch.int64,
        device=runner.device,
    )
    replay.prefill_input_ids_cpu = None
    replay.req_pool_indices = slots
    replay.req_pool_indices_cpu = batch.req_pool_indices_cpu[idx]
    replay.prefix_lens = starts
    replay.extend_lens = lens
    replay.extend_num_tokens = sum(lens)
    replay.seq_lens = torch.tensor(ends, dtype=torch.int64, device=runner.device)
    replay.seq_lens_cpu = torch.tensor(ends, dtype=torch.int64)
    replay.seq_lens_sum = sum(ends)
    replay.orig_seq_lens = replay.seq_lens
    replay.out_cache_loc = torch.cat(
        [req_to_token[slots[k], s:e] for k, (s, e) in enumerate(zip(starts, ends))]
    ).long()
    replay.return_logprob = False
    replay.top_logprobs_nums = None
    replay.token_ids_logprobs = None
    replay.extend_logprob_start_lens = lens
    replay.extend_input_logprob_token_ids = None
    replay.is_prefill_only = True
    replay.spec_info = None
    replay.sampling_info = None
    replay.has_grammar = False
    replay.multimodal_inputs = [None] * len(reqs)
    replay.engram_history = _engram_history(reqs=reqs, starts=starts, runner=runner)
    _set_execution_counts(batch=replay, scheduled=batch)
    return replay


def _engram_history(*, reqs, starts, runner):
    hasher = runner.model.model.engram_hasher
    if hasher is None:
        return None
    n = hasher.max_ngram_size - 1
    hist = []
    for r, s in zip(reqs, starts):
        ids = r.full_untruncated_fill_ids[max(0, s - n) : s]
        hist.append(hasher.history_row(ids))
    return torch.tensor(hist, dtype=torch.int32, device=runner.device)


def _fold_batch(*, batch, runner, rows) -> FoldedExtend:
    bs = len(batch.reqs)
    device = runner.device
    replay = [0] * bs
    for i, s, e in rows:
        replay[i] = e - s
    ext = list(batch.extend_lens)
    pre = [p - r for p, r in zip(batch.prefix_lens, replay)]
    new_ext = [x + r for x, r in zip(ext, replay)]
    req_to_token = runner.req_to_token_pool.req_to_token
    ids, locs, keep, cpu_ids = [], [], [], []
    off = fold_off = 0
    for i, req in enumerate(batch.reqs):
        r, n = replay[i], ext[i]
        if r:
            head = list(req.full_untruncated_fill_ids[pre[i] : pre[i] + r])
            ids.append(torch.tensor(head, dtype=batch.input_ids.dtype, device=device))
            locs.append(
                req_to_token[batch.req_pool_indices[i], pre[i] : pre[i] + r].long()
            )
            cpu_ids.append(torch.tensor(head, dtype=torch.int64))
        ids.append(batch.input_ids[off : off + n])
        locs.append(batch.out_cache_loc[off : off + n].long())
        if batch.prefill_input_ids_cpu is not None:
            cpu_ids.append(batch.prefill_input_ids_cpu[off : off + n].to(torch.int64))
        keep.append(torch.arange(fold_off + r, fold_off + r + n))
        off += n
        fold_off += r + n

    folded = copy(batch)
    folded.input_ids = torch.cat(ids)
    folded.out_cache_loc = torch.cat(locs)
    folded.prefill_input_ids_cpu = (
        torch.cat(cpu_ids) if batch.prefill_input_ids_cpu is not None else None
    )
    folded.prefix_lens = pre
    folded.extend_lens = new_ext
    folded.extend_num_tokens = fold_off
    if batch.extend_logprob_start_lens is not None:
        folded.extend_logprob_start_lens = [
            x + r for x, r in zip(batch.extend_logprob_start_lens, replay)
        ]
    if batch.engram_history is not None:
        folded.engram_history = _engram_history(
            reqs=batch.reqs, starts=pre, runner=runner
        )
    floor = torch.tensor(
        [p if r else 0 for p, r in zip(pre, replay)], dtype=torch.int64
    )
    row_floor = torch.repeat_interleave(
        floor, torch.tensor(new_ext), output_size=fold_off
    )
    keep_rows = torch.cat(keep)
    scheduled_row = torch.full((fold_off,), -1, dtype=torch.int64)
    scheduled_row[keep_rows] = torch.arange(keep_rows.shape[0])
    return FoldedExtend(
        batch=folded,
        keep_rows=keep_rows.to(device, non_blocking=True),
        scheduled_row=scheduled_row.to(device, non_blocking=True),
        row_floor=row_floor.to(device, non_blocking=True),
        replay_lens=replay,
        num_rows=fold_off,
    )


def _set_execution_counts(*, batch, scheduled) -> None:
    """MLP-sync counts of a batch copied from `scheduled` with different rows."""
    if scheduled.global_num_tokens is None:
        return
    # Attention DP is admitted only with the fold, so the gather holds one count.
    if (
        len(scheduled.global_num_tokens) != 1
        or scheduled.global_num_tokens[0] != scheduled.extend_num_tokens
    ):
        raise ValueError(
            "encoder SWA replay expects one MLP-sync count equal to the scheduled "
            f"extend ({scheduled.extend_num_tokens}), got {scheduled.global_num_tokens}"
        )
    batch.global_num_tokens = [batch.extend_num_tokens]
    # The scheduler's formula; the draft counts stay with the scheduled batch.
    batch.global_num_tokens_for_logprob = [
        sum(
            max(n - start, 1)
            for n, start in zip(batch.extend_lens, batch.extend_logprob_start_lens)
        )
    ]


logger = logging.getLogger(__name__)


_HCU_REPLAY_PLAN_BYTES = 4096


def _check_hcu_cp_replay_plan(plan, cp_size, cpu_group):
    """Check the complete usual plan in one CPU collective, without hashing.

    Arbitrary-size/nonstandard plans and differing frames take the original
    object comparison. All ranks choose that fallback from the gathered headers,
    so a long request ID on just one rank cannot change collective ordering.
    """
    ordinary = type(plan) is tuple and all(
        type(row) is tuple
        and len(row) == 3
        and type(row[0]) is str
        and type(row[1]) is bool
        and type(row[2]) is int
        for row in plan
    )
    data = (
        json.dumps(plan, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        if ordinary
        else b""
    )
    length = len(data) if ordinary else _HCU_REPLAY_PLAN_BYTES + 1
    storage = bytearray(8 + _HCU_REPLAY_PLAN_BYTES)
    storage[:8] = length.to_bytes(8, "little")
    size = min(len(data), _HCU_REPLAY_PLAN_BYTES)
    storage[8 : 8 + size] = data[:size]
    frame = torch.frombuffer(storage, dtype=torch.uint8)
    frames = [torch.empty_like(frame) for _ in range(cp_size)]
    torch.distributed.all_gather(frames, frame, group=cpu_group)
    received = [other.numpy().tobytes() for other in frames]
    expected = bytes(storage)
    fallback = any(
        int.from_bytes(other[:8], "little") > _HCU_REPLAY_PLAN_BYTES
        or other != expected
        for other in received
    )
    if fallback:
        plans = [None] * cp_size
        torch.distributed.all_gather_object(plans, plan, group=cpu_group)
        if any(other != plan for other in plans):
            raise RuntimeError("Encoder SWA replay plan differs across CP ranks")


def _run_hcu_encoder_swa_replay(worker, batch):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )

    runner = worker.model_runner
    window = runner.token_to_kv_pool.request_window
    if window is None or not batch.forward_mode.is_extend_without_speculative():
        return
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.attn_cp_size > 1:
        # Every CP rank must enter the same replay forwards and collectives,
        # even when its layer-sharded SWA state contains different layers.
        plan = tuple(
            (req.rid, bool(reset), int(prefix))
            for req, reset, prefix in zip(
                batch.reqs, batch.encoder_swa_reset, batch.prefix_lens, strict=True
            )
        )
        _check_hcu_cp_replay_plan(
            plan, parallel.attn_cp_size, parallel.attn_cp_group.cpu_group
        )
    for i, reset in enumerate(batch.encoder_swa_reset):
        if not reset:
            continue
        slot = batch.req_pool_indices[i : i + 1]
        window.reset(slot)
        end = batch.prefix_lens[i]
        if not end:
            continue
        if end % 2:
            raise ValueError(
                "encoder SWA replay requires an even cached-prefix boundary"
            )
        start = max(0, end - 128)
        req = batch.reqs[i]
        if parallel.attn_cp_rank == 0:
            logger.info(
                "Encoder SWA replay: rid=%s prefix=%d start=%d replay_tokens=%d cp_size=%d",
                req.rid,
                end,
                start,
                end - start,
                parallel.attn_cp_size,
            )
        replay = copy(batch)
        replay.reqs = [req]
        replay.input_ids = torch.tensor(
            list(req.full_untruncated_fill_ids[start:end]),
            dtype=torch.int64,
            device=runner.device,
        )
        replay.prefill_input_ids_cpu = None
        replay.mix_running_indices = None
        replay.mix_running_indices_cpu = None
        replay.req_pool_indices = slot
        replay.req_pool_indices_cpu = batch.req_pool_indices_cpu[i : i + 1]
        replay.prefix_lens = [start]
        replay.extend_lens = [end - start]
        replay.extend_num_tokens = end - start
        # HCU replay is admitted only for interleave CP; the parent
        # CP-v2 token counts do not describe this new replay forward.
        replay.global_cp_num_tokens = None
        if batch.global_num_tokens is not None:
            # Replay is a separate single-request forward. Parent-batch counts
            # can be smaller than its input (negative padding) or much larger.
            replay.global_num_tokens = [end - start] * len(batch.global_num_tokens)
            replay.global_num_tokens_for_logprob = [1] * len(batch.global_num_tokens)
        replay.seq_lens = torch.tensor([end], dtype=torch.int64, device=runner.device)
        replay.seq_lens_cpu = torch.tensor([end], dtype=torch.int64)
        replay.seq_lens_sum = end
        replay.orig_seq_lens = replay.seq_lens
        replay.orig_seq_lens_cpu = [end]
        replay.encoder_swa_reset = [False]
        replay.out_cache_loc = runner.req_to_token_pool.req_to_token[
            slot[0], start:end
        ].long()
        replay.out_cache_loc_dsv4 = None
        replay.return_logprob = False
        replay.top_logprobs_nums = None
        replay.token_ids_logprobs = None
        replay.extend_logprob_start_lens = [end - start]
        replay.extend_input_logprob_token_ids = None
        replay.is_prefill_only = True
        replay.spec_info = None
        replay.sampling_info = None
        replay.has_grammar = False
        # Keep the selected request's absolute image spans and cached features.
        # The model scatters only [start, end) before CP shards its input rows;
        # scheduler hash IDs remain intact, as on the CUDA folded path.
        mm_inputs = getattr(batch, "multimodal_inputs", None)
        replay.multimodal_inputs = [mm_inputs[i]] if mm_inputs is not None else [None]
        replay.input_embeds = None
        replay.replace_embeds = None
        replay.replace_positions = None
        replay.engram_history = None
        hasher = runner.model.model.engram_hasher
        if hasher is not None:
            # The shared CUDA helper normalizes large image hash IDs before
            # materializing the int32 history, preserving image-token masking.
            replay.engram_history = _engram_history(
                reqs=[req], starts=[start], runner=runner
            )
        fb = ForwardBatch.init_new(
            replay,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
            encoder_swa_replay=True,
        )
        runner.forward(fb)
