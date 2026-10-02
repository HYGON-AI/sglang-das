"""LightOp stand-ins for the DeepGEMM kernels of the DSV4.1 paged fp4 indexer on
HCU (gfx928/936/938).

The decode/verify indexer keeps DeepGEMM's contract end to end: fp4 queries
``(q_fp4 [rows, 1, H, 64] int8, q_sf [rows, 1, H] int32)``, the fused index-K
pool ``[pages, 64, 1, 68]`` uint8, fp32 head weights, int32 ``context_lens`` and
an int32 block table, plus a split-KV schedule planned once per forward. Only
the kernels change:

* ``fp8_fp4_paged_mqa_logits``           -> ``paged_mqa_logits_fp4`` (block 64)
* ``get_paged_mqa_logits_metadata``      -> LightOp's schedule planner
* ``fp8_fp4_paged_sparse_mqa_logits``    -> ``paged_mqa_logits_fp4`` at block 8,
  reading the published candidate blocks as its block table
* ``amax8_varlen`` + block top-k + sort  -> ``dsv41_candidate_block_table``

Every call takes device tensors only and allocates nothing data dependent, so
the path is graph capturable.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hcu

logger = logging.getLogger(__name__)

_is_hcu = is_hcu()

# dsv41_candidate_block_table is compiled for 8-position blocks only.
SUPPORTED_CANDIDATE_BLOCK_SIZES = (8,)
DEFAULT_CANDIDATE_BLOCK_SIZE = 8
# LightOp paged_mqa_logits_fp4 ABI carrying (q_fp4, q_sf) + block_table.
_REQUIRED_ABI = 2
# First ABI taking rows_per_request (verify rows scored in pairs, one KV read).
_PAIRED_ROWS_ABI = 3


@functools.cache
def _lightop_modules():
    try:
        from lightop import attention as lightop_attention
        from lightop import op as lightop_op
    except (ImportError, OSError) as exc:
        logger.warning("LightOp unavailable for the DSV4.1 paged indexer: %s", exc)
        return None
    return lightop_op, lightop_attention


@functools.cache
def lightop_paged_indexer_available() -> bool:
    """True when the paged fp4 decode indexer can run on LightOp: HCU, the flag
    on, and a LightOp build providing the paged ABI and the candidate table."""
    if not (_is_hcu and envs.SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER.get()):
        return False
    mods = _lightop_modules()
    if mods is None:
        return False
    op, attention = mods
    abi = getattr(attention, "PAGED_MQA_LOGITS_FP4_ABI", 1)
    missing = [
        name
        for name in (
            "paged_mqa_logits_fp4",
            "get_paged_mqa_logits_metadata",
            "dsv41_candidate_block_table",
        )
        if not hasattr(op, name)
    ]
    if abi < _REQUIRED_ABI or missing:
        logger.warning(
            "LightOp lacks the paged fp4 indexer (abi=%s, missing=%s); the DSV4.1 "
            "decode indexer keeps the slot-based path",
            abi,
            missing,
        )
        return False
    logger.info("Using LightOp for the DSV4.1 paged fp4 decode indexer")
    return True


@functools.cache
def lightop_paired_rows_available() -> bool:
    """True when paged_mqa_logits_fp4 accepts rows_per_request > 1."""
    if not lightop_paged_indexer_available():
        return False
    _, attention = _lightop_modules()
    return attention.PAGED_MQA_LOGITS_FP4_ABI >= _PAIRED_ROWS_ABI


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _int32_lens(lens: torch.Tensor) -> torch.Tensor:
    """A contiguous int32 row vector, the only shape the operators accept."""
    lens = lens.reshape(-1)
    if lens.dtype != torch.int32:
        lens = lens.to(torch.int32)
    return lens.contiguous()


# The planner splits rows every 4 * block_kv positions; paged_mqa_logits_fp4
# scores 256-position splits whatever the pool page or table block size.
_SCHEDULE_BLOCK_KV = 64


def get_paged_mqa_logits_schedule(
    context_lens: torch.Tensor, rows_per_request: int = 1
) -> torch.Tensor:
    """Split-KV schedule ``[num_sms + 1, 2]`` int32 of (row, split) starts for the
    rows' ``context_lens`` (``[rows]`` or ``[rows, 1]``), split 256 positions.
    With ``rows_per_request > 1`` it is planned on the kernel's row pairs instead,
    and only valid for paged_mqa_logits_fp4 called with the same value."""
    op, attention = _lightop_modules()
    lens = _int32_lens(context_lens)
    if rows_per_request > 1:
        lens = attention.paged_mqa_logits_route_lens(lens, rows_per_request)
    return op.get_paged_mqa_logits_metadata(
        lens, _SCHEDULE_BLOCK_KV, _num_cus(lens.device.index)
    )


# The planner kernel covers up to this many rows (aligned-batch dispatch table).
MAX_SCHEDULE_ROWS = 8192


def paged_mqa_logits_fp4(
    q: Tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule: Optional[torch.Tensor],
    max_context_len: int,
    table_block_size: int,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """fp32 logits ``[rows, max_context_len]`` (row stride padded); column ``j``
    of row ``b`` scores pool slot ``block_table[b, j // table_block_size] *
    table_block_size + j % table_block_size``. Columns at or past
    ``context_lens[b]`` are left unwritten, like DeepGEMM with
    ``clean_logits=False``: every consumer reads a row up to its length only.

    ``rows_per_request > 1`` (block 64, lightop_paired_rows_available()) declares
    every that many rows one verify request, all reading the block table row of
    the first; the kernel then reads each KV tile once per pair of rows, with
    bit-identical logits. ``schedule`` must be planned with the same value."""
    op, _ = _lightop_modules()
    q_fp4, q_sf = q
    # Older builds lack the trailing argument; pass it only when pairing.
    paired = (rows_per_request,) if rows_per_request > 1 else ()
    return op.paged_mqa_logits_fp4(
        q_fp4,
        q_sf,
        k_cache,
        weights,
        _int32_lens(context_lens),
        block_table,
        schedule,
        max_context_len,
        table_block_size,
        None,
        *paired,
    )


def candidate_block_table(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    topk_blocks: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Level one of the two-level indexer in one pass over the dense logits.

    Per row, of the ``ceil(seq_lens[b] / block_size)`` blocks, keeps the
    ``topk_blocks`` with the largest block maximum, the newest block always
    kept, and returns them ascending as pool blocks (``phys_blocks``
    ``[rows, topk_blocks]`` int32, padded past the kept count) together with
    the length of the resulting sparse row (``valid_lens`` ``[rows]`` int32,
    the newest block possibly partial). A row with at most ``topk_blocks``
    blocks keeps all of them, so its sparse row equals its dense row. The
    operator rejects a ``block_size`` outside SUPPORTED_CANDIDATE_BLOCK_SIZES."""
    op, _ = _lightop_modules()
    rows = logits.shape[0]
    phys_blocks = torch.empty(
        rows, topk_blocks, dtype=torch.int32, device=logits.device
    )
    valid_lens = torch.empty(rows, dtype=torch.int32, device=logits.device)
    op.dsv41_candidate_block_table(
        logits,
        _int32_lens(seq_lens),
        page_table,
        page_size,
        block_size,
        phys_blocks,
        valid_lens,
    )
    return phys_blocks, valid_lens
