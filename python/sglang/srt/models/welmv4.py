# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import (
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
)
from sglang.srt.layers.rotary_embedding.yarn import yarn_get_mscale
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.layers.welmv4_op import (
    WelmV4FusedRMSNorm,
    WelmV4InplaceRotaryEmbedding,
    inplace_sigmoid_mul,
    mmq_style_expert_bias_topk,
    mmq_style_k_rms_norm,
    mmq_style_norm_after_attn,
    mmq_style_router_linear,
)
from sglang.srt.layers.welmv4_fast_op import (
    WELM_FAST_ROUTER_MAX_M,
    welm_fast_router_topk,
    welm_qk_norm_rope,
    sigmoid_mul_strided_,
)
import sglang.srt.layers.attention.flashattention_interface as _fa_iface
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args

# from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, make_layers

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
WELM_KV_MIRROR_PP_KEY_PREFIX = "welm_kv_mirror"
WELM_KV_MIRROR_STATES_KEY = "welm_kv_mirror_states"
_WELM_FAST_ROUTER = get_bool_env_var("WELM_FAST_ROUTER", "true")
_WELM_FAST_QK = get_bool_env_var("WELM_FAST_QK", "true")
# Decode: write k/v into the paged KV cache from the qk-norm/rope kernel.
_WELM_FUSED_KV_STORE = get_bool_env_var("WELM_FUSED_KV_STORE", "true")


def _welm_fused_kv_store_buffers(forward_batch, layer_id, dtype):
    """(k_cache, v_cache) of the HCU legacy-layout pool for a direct decode
    write, or None when the pool uses any other layout / dtype."""
    from sglang.srt.mem_cache import memory_pool as _mp

    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        pool = getattr(get_attn_backend(), "token_to_kv_pool", None)
    except Exception:
        pool = None
    if (
        pool is None
        or not _mp._kv_layout_hcu_fa
        or getattr(pool, "use_hnd", True)
        or not hasattr(pool, "k_buffer")
        or getattr(pool, "dtype", None) != dtype
        or getattr(pool, "store_dtype", None) != dtype
        or getattr(pool, "start_layer", None) is None
    ):
        return None
    idx = layer_id - pool.start_layer
    if not 0 <= idx < len(pool.k_buffer):
        return None
    k_buf, v_buf = pool.k_buffer[idx], pool.v_buffer[idx]
    if k_buf.dim() != 4 or v_buf.dim() != 4:
        return None
    return k_buf, v_buf
# Run the shared expert as an extra (always selected, weight 1) expert inside
# the routed fused MoE instead of three separate GEMM/activation kernels.
_WELM_FUSE_SHARED_EXPERT = get_bool_env_var("WELM_FUSE_SHARED_EXPERT", "true")
# Append the head-wise attention gate rows to the qkv projection (one GEMM).
_WELM_MERGE_GATE_PROJ = get_bool_env_var("WELM_MERGE_GATE_PROJ", "true")
_WELM_DUMP_PROCESS_DIR = None
_WELM_DUMP_BASE_DIR = None
_WELM_DUMP_PASS_ID = -1
def get_attention_tp_rank() -> int:
    return get_parallel().attn_tp_rank


def get_attention_tp_size() -> int:
    return get_parallel().attn_tp_size


class WelmV4CommunicatorRMSNorm(nn.Module):
    """Adapt WeLM fused RMSNorm to LayerCommunicator's return-value contract."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = WelmV4FusedRMSNorm(hidden_size, eps=eps)
        self.weight = self.norm.weight
        self.eps = self.norm.eps
        self.variance_epsilon = self.norm.eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if post_residual_addition is not None and residual is not None:
            residual = residual + post_residual_addition
        output = self.norm(x, residual, **kwargs)
        if not kwargs and residual is None and isinstance(output, tuple):
            return output[0]
        return output


def _welm_get_tp_group():
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


def _welm_dump_enabled() -> bool:
    return os.getenv("SGLANG_DUMP_ACTIVATIONS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _welm_should_dump_layer(layer_idx: int) -> bool:
    if not _welm_dump_enabled():
        return False
    layer_idxs = os.getenv("SGLANG_DUMP_ACTIVATIONS_LAYER_IDXS")
    if not layer_idxs:
        return True
    return str(layer_idx) in {x.strip() for x in layer_idxs.split(",") if x.strip()}


def _welm_dump_tensor(name: str, tensor: torch.Tensor) -> None:
    global _WELM_DUMP_PROCESS_DIR
    if not isinstance(tensor, torch.Tensor):
        return
    if _WELM_DUMP_PROCESS_DIR is None:
        process_dir = os.getenv("SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR")
        if process_dir:
            _WELM_DUMP_PROCESS_DIR = Path(process_dir)
        else:
            base_dir = Path(
                os.getenv("SGLANG_DUMP_ACTIVATIONS_DIR", "/tmp/sglang_welm_dump")
            )
            _WELM_DUMP_PROCESS_DIR = (
                base_dir / f"TP0_PP0_Rank0_pid{os.getpid()}" / "Pass00000"
            )
            os.environ["SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR"] = str(
                _WELM_DUMP_PROCESS_DIR
            )
        _WELM_DUMP_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), _WELM_DUMP_PROCESS_DIR / f"{name}.pt")


def _welm_start_dump_pass() -> None:
    global _WELM_DUMP_BASE_DIR, _WELM_DUMP_PASS_ID, _WELM_DUMP_PROCESS_DIR
    if not _welm_dump_enabled():
        return
    if _WELM_DUMP_BASE_DIR is None:
        _WELM_DUMP_BASE_DIR = (
            Path(os.getenv("SGLANG_DUMP_ACTIVATIONS_DIR", "/tmp/sglang_welm_dump"))
            / f"TP0_PP0_Rank0_pid{os.getpid()}"
        )
    _WELM_DUMP_PASS_ID += 1
    _WELM_DUMP_PROCESS_DIR = _WELM_DUMP_BASE_DIR / f"Pass{_WELM_DUMP_PASS_ID:05d}"
    os.environ["SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR"] = str(_WELM_DUMP_PROCESS_DIR)
    _WELM_DUMP_PROCESS_DIR.mkdir(parents=True, exist_ok=True)


_WELM_FUSED_EMBED_AR = get_bool_env_var("WELM_FUSED_EMBED_AR", "true")


def _welm_trace_target_logits(forward_batch, input_ids, positions, logits_output):
    """Optional low-volume target-logit/state trace for MTP diagnosis."""
    path = os.getenv("WELM_TRACE_TARGET_LOGITS")
    if not path or get_attention_tp_rank() != 0:
        return
    try:
        states = getattr(forward_batch, "model_specific_states", None) or {}
        mirror = states.get(WELM_KV_MIRROR_STATES_KEY, {})
        trace = {
            "mode": str(forward_batch.forward_mode),
            "input_shape": tuple(input_ids.shape),
            "position_shape": tuple(positions.shape),
            "input_head": input_ids[:8].detach().cpu().tolist(),
            "position_head": positions[:8].detach().cpu().tolist(),
            "state_keys": sorted(str(k) for k in mirror),
            "state_shapes": {
                str(k): (tuple(v[0].shape), tuple(v[1].shape))
                for k, v in mirror.items()
            },
        }
        logits = getattr(logits_output, "next_token_logits", None)
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            rows = [0] if logits.shape[0] == 1 else [0, logits.shape[0] - 1]
            trace["logit_rows"] = []
            for row in rows:
                values, ids = torch.topk(logits[row].float(), k=5)
                trace["logit_rows"].append(
                    {
                        "row": row,
                        "ids": ids.detach().cpu().tolist(),
                        "values": values.detach().cpu().tolist(),
                    }
                )
        with open(path, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(trace, sort_keys=True) + "\n")
    except Exception:
        logger.exception("WeLM target trace failed")


def _oe_hashed_ids(input_ids, n_gram_input_ids, vocab_size, oe_grams, oe_vocab_sizes):
    """Hashed n-gram ids for each OE table (same integer ops as the reference)."""
    input_ids_ngram = []
    tmp = input_ids
    for g in range(1, max(oe_grams)):
        gram = n_gram_input_ids.get_gram(g + 1) if n_gram_input_ids is not None else None
        if gram is not None:
            tmp = tmp + gram * (vocab_size**g)
        input_ids_ngram.append(hash_input_ids_vectorized(tmp))
    return [input_ids_ngram[oe_grams[i] - 2] % vs for i, vs in enumerate(oe_vocab_sizes)]


def _can_fuse_embed_ar(embed_tokens, oe_embed):
    if not _WELM_FUSED_EMBED_AR or _welm_dump_enabled():
        return False
    mods = [embed_tokens, *oe_embed]
    for m in mods:
        if not isinstance(m, VocabParallelEmbedding) or getattr(m, "use_attn_tp_group", False):
            return False
        if m.tp_size <= 1 or m.output_dtype is not None:
            return False
    from sglang.srt.layers.communicator import get_attn_tp_context

    return not get_attn_tp_context().input_scattered


def welm_embed_with_oe(
    embed_tokens, oe_embed, oe_up_proj, input_ids, n_gram_input_ids,
    vocab_size, oe_grams, oe_vocab_sizes,
):
    """(embed(ids) + oe_up_proj(cat(oe_embed_i(hash_i)))) / 2 with ONE all-reduce.

    Every vocab-parallel lookup is non-zero on exactly one rank, so summing the
    concatenated local shards is bit-identical to reducing each separately.
    """
    hashed = _oe_hashed_ids(input_ids, n_gram_input_ids, vocab_size, oe_grams, oe_vocab_sizes)
    parts = [m._embed_local_shard(h) for m, h in zip(oe_embed, hashed)]
    parts.append(embed_tokens._embed_local_shard(input_ids))
    buf = tensor_model_parallel_all_reduce(torch.cat(parts, dim=-1))
    oe_dim = sum(p.shape[-1] for p in parts[:-1])
    oe, base = buf.split([oe_dim, buf.shape[-1] - oe_dim], dim=-1)
    emb_new, _ = oe_up_proj(oe)
    return (base + emb_new) / 2.0


def _welm_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.welmv4_fast_op import skinny_gemm, skinny_gemm_supported

    if x.dim() == 2 and skinny_gemm_supported(x, w):
        return skinny_gemm(x, w)
    return F.linear(x, w)


def hash_input_ids_vectorized(input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.to(torch.int64)
    result = ids * 2654435761
    result = result & 0xFFFFFFFF
    return result.to(input_ids.dtype)


def _welm_kv_mirror_pp_key(layer_idx: int, suffix: str) -> str:
    return f"{WELM_KV_MIRROR_PP_KEY_PREFIX}.{layer_idx}.{suffix}"


def _empty_welm_kv_mirror_states() -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    return {}


def _get_welm_kv_mirror_states(
    forward_batch: ForwardBatch,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Get a private shallow copy of this forward's mirror activations.

    Keys are physical mirror-layer ids rather than imitated-layer ids.  This
    is intentional: the checkpoint maps NextN layers 48, 49 and 50 all to
    target layer 0, so a single ``{0: (K, V)}`` entry cannot represent three
    consumers safely.
    """
    model_specific_states = forward_batch.model_specific_states or {}
    return dict(
        model_specific_states.get(
            WELM_KV_MIRROR_STATES_KEY, _empty_welm_kv_mirror_states()
        )
    )


def _set_welm_kv_mirror_states(
    forward_batch: ForwardBatch,
    kv_mirror_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Store or clear mirror state without touching other model state."""
    model_specific_states = dict(forward_batch.model_specific_states or {})
    if kv_mirror_states:
        model_specific_states[WELM_KV_MIRROR_STATES_KEY] = dict(kv_mirror_states)
    else:
        model_specific_states.pop(WELM_KV_MIRROR_STATES_KEY, None)
    forward_batch.model_specific_states = model_specific_states or None


def _pack_welm_kv_mirror_pp_tensor(
    proxy_tensors: Dict[str, Any], key: str, tensor: torch.Tensor
) -> None:
    flat = tensor.contiguous().view(-1)
    try:
        tp_size = get_attention_tp_size()
    except (AssertionError, RuntimeError):
        tp_size = 1
    pad_len = 0
    if tp_size > 1:
        while (flat.numel() + pad_len) % tp_size != 0:
            pad_len += 1
    proxy_tensors[key] = (
        torch.cat([flat, flat.new_empty(pad_len)]) if pad_len else flat
    )
    proxy_tensors[f"{key}.shape"] = tuple(tensor.shape)


def _unpack_welm_kv_mirror_pp_tensor(
    pp_proxy_tensors: PPProxyTensors, key: str, tensor: torch.Tensor
) -> torch.Tensor:
    shape = pp_proxy_tensors.tensors.get(f"{key}.shape")
    if shape is None:
        return tensor
    return tensor[: math.prod(shape)].view(shape)


def _pack_welm_kv_mirror_states(
    proxy_tensors: Dict[str, Any],
    kv_mirror_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    if not kv_mirror_states:
        return
    for layer_idx, (mirror_k, mirror_v) in kv_mirror_states.items():
        _pack_welm_kv_mirror_pp_tensor(
            proxy_tensors, _welm_kv_mirror_pp_key(layer_idx, "k"), mirror_k
        )
        _pack_welm_kv_mirror_pp_tensor(
            proxy_tensors, _welm_kv_mirror_pp_key(layer_idx, "v"), mirror_v
        )


def _unpack_welm_kv_mirror_states(
    pp_proxy_tensors: PPProxyTensors,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    states = _empty_welm_kv_mirror_states()
    for key, tensor in pp_proxy_tensors.tensors.items():
        if not key.startswith(f"{WELM_KV_MIRROR_PP_KEY_PREFIX}.") or key.endswith(".shape"):
            continue
        _, layer_idx, suffix = key.rsplit(".", 2)
        tensor = _unpack_welm_kv_mirror_pp_tensor(pp_proxy_tensors, key, tensor)
        mirror_k, mirror_v = states.get(int(layer_idx), (None, None))
        if suffix == "k":
            mirror_k = tensor
        elif suffix == "v":
            mirror_v = tensor
        states[int(layer_idx)] = (mirror_k, mirror_v)
    return {
        layer_idx: (mirror_k, mirror_v)
        for layer_idx, (mirror_k, mirror_v) in states.items()
        if mirror_k is not None and mirror_v is not None
    }


def _welm_mirror_ids_for_imitated(attn, imitated_layer_idx: int) -> List[int]:
    return [
        mirror
        for mirror, imitated in zip(
            getattr(attn, "kv_mirror_layers", []),
            getattr(attn, "kv_mirror_imitated_layers", []),
        )
        if imitated == imitated_layer_idx
    ]


class WelmSpecNgramRing:
    """Per-request ring of recent token ids for speculative decoding.

    Over-encoding needs the 1/2/3 tokens preceding every input token. Outside
    plain prefill (target verify, draft decode, draft extend) the scheduler
    cannot provide them, because they include draft tokens that only exist on
    the device. Each request keeps its last ``SLOTS`` tokens, indexed by
    ``target_position % SLOTS``; a forward first writes its own tokens and
    then reads the predecessors, so chains inside one forward resolve to the
    drafted tokens and everything before the segment resolves to committed
    tokens. Draft-model inputs are shifted by one (the draft at position p
    consumes token p + 1), so they use ``target_position = position + 1``.

    Request pool slot 0 is reserved for padding, so padded graph rows only
    ever touch row 0.
    """

    SLOTS = 8
    ROWS = 1 << 16
    buf: Optional[torch.Tensor] = None

    @classmethod
    def ensure(cls, device) -> torch.Tensor:
        if cls.buf is None:
            cls.buf = torch.zeros((cls.ROWS, cls.SLOTS), dtype=torch.int64, device=device)
        return cls.buf


def _welm_read_ring(ring, rows, tpos):
    grams = []
    for back in (1, 2, 3):
        p = tpos - back
        val = ring[rows, torch.remainder(p, ring.shape[1])]
        grams.append(torch.where(p >= 0, val, torch.zeros_like(val)))
    return grams


def welm_use_device_ngram() -> bool:
    """Whether n-gram ids come from the on-device ring instead of the scheduler.

    Always for speculative decoding. With the overlap scheduler the CPU-side
    decode n-grams lag one step behind (the previous token is still on the
    device), so non-speculative decode uses the ring as well.
    """
    args = get_global_server_args()
    if args.speculative_algorithm is not None:
        return True
    return not args.disable_overlap_schedule and get_bool_env_var(
        "WELM_DEVICE_NGRAM", "true"
    )


def welm_spec_n_gram_input_ids(
    input_ids, positions, forward_batch, is_draft: bool, draft_shift: int = 1
):
    """Return the NGramInputIds to use for this forward (see WelmSpecNgramRing).

    ``draft_shift`` is how many tokens a draft input runs ahead of its position:
    1 for a single NextN layer, ``1 + k`` for multi-layer EAGLE step ``k``.
    """
    from sglang.srt.managers.schedule_batch import NGramInputIds

    current = forward_batch.n_gram_input_ids
    if not welm_use_device_ngram():
        return current
    mode = forward_batch.forward_mode
    num_tokens = input_ids.shape[0]
    if num_tokens == 0 or mode.is_idle():
        return current
    if (
        current is not None
        and len(current.input_ids_grams) > 0
        and 0 < current.input_ids_grams[0].shape[0] < num_tokens
    ):
        # DP attention pads the batch to a multiple of the attention-TP size;
        # pad the scheduler's grams the same way (padding rows are discarded).
        pad = num_tokens - current.input_ids_grams[0].shape[0]
        current.input_ids_grams = [
            torch.cat([g, g.new_zeros((pad, *g.shape[1:]))]) for g in current.input_ids_grams
        ]
    ring = WelmSpecNgramRing.ensure(input_ids.device)
    slots = ring.shape[1]
    ids = input_ids.to(torch.int64)
    shift = draft_shift if is_draft else 0
    tpos = positions.to(torch.int64) + shift

    if mode.is_extend_without_speculative():
        # Prefill: per-token request rows and in-segment offsets.  Map tokens
        # to segments with searchsorted rather than repeat_interleave: under a
        # prefill CUDA graph ``num_tokens`` is the padded bucket size while
        # the segment lengths sum to the real token count, and
        # repeat_interleave(output_size=...) would read out of bounds.
        seg_lens = forward_batch.extend_seq_lens.to(torch.int64)
        bs = seg_lens.shape[0]
        seg_end = torch.cumsum(seg_lens, 0)
        tok = torch.arange(num_tokens, device=ids.device)
        seg = torch.searchsorted(seg_end, tok, right=True)
        valid = seg < bs
        seg = torch.clamp(seg, max=bs - 1)
        rows = forward_batch.req_pool_indices.to(torch.int64)[seg]
        idx = tok - (seg_end - seg_lens)[seg]
        lens = seg_lens[seg]
        # Keep only the tail of each segment so no ring slot is written twice.
        # Masked-out writes go to the reserved padding row 0 instead of being
        # filtered with a boolean index, which would need a host sync and
        # break prefill CUDA graph capture.
        if (
            not is_draft
            and current is not None
            and len(current.input_ids_grams) > 0
            and current.input_ids_grams[0].shape[0] == num_tokens
        ):
            # Mixed batches mark decode rows' grams with -1: their
            # predecessors come from the ring, like a plain decode step.
            # In place: callers (and prefill-graph capture) keep these
            # tensors as the batch's n-gram buffers.
            ring_grams = _welm_read_ring(ring, rows, tpos)
            for g, rg in zip(current.input_ids_grams, ring_grams):
                g.copy_(torch.where(g < 0, rg.to(g.dtype), g))
        tail = valid & (idx >= lens - slots)
        ring[torch.where(tail, rows, 0), torch.remainder(tpos, slots)] = ids
        if not is_draft:
            # Target prefill uses the exact scheduler grams; also record the
            # predecessors of short segments (prefix-cache hits, chunking).
            first = valid & (idx == 0)
            for back, gram in zip((1, 2, 3), current.input_ids_grams):
                p = tpos - back
                sel = first & (p >= 0) & (lens + back <= slots)
                ring[torch.where(sel, rows, 0), torch.remainder(p, slots)] = gram.to(
                    torch.int64
                )
            return current
        # Draft prefill: input j is target token j + shift of the segment, so
        # its grams are the target grams of token j + shift; the last ``shift``
        # inputs (newly sampled / proposed tokens) read predecessors from the
        # ring.  Positions near a segment end clamp but take the ring branch.
        from_ring = idx >= lens - shift
        nxt = torch.clamp(
            torch.arange(num_tokens, device=ids.device) + shift, max=num_tokens - 1
        )
        ring_grams = _welm_read_ring(ring, rows, tpos)
        grams = [
            torch.where(from_ring, rg, g.to(torch.int64)[nxt])
            for rg, g in zip(ring_grams, current.input_ids_grams)
        ]
        return NGramInputIds(input_ids_grams=grams)

    # Verify / draft decode / draft extend: uniform tokens per request.
    bs = forward_batch.req_pool_indices.shape[0]
    per_req = num_tokens // bs
    rows = torch.repeat_interleave(forward_batch.req_pool_indices.to(torch.int64), per_req)
    ring[rows, torch.remainder(tpos, slots)] = ids
    out = NGramInputIds(input_ids_grams=_welm_read_ring(ring, rows, tpos))
    return out


def _welm_mirror_kv_from_projection(
    attn,
    hidden_states: torch.Tensor,
    qkv: Optional[torch.Tensor] = None,
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """Project every physical mirror layer owned by one target layer.

    Three-layer checkpoints can map several physical NextN layers to the same
    target layer.  Their K/V projections are distinct (for example, layer 48
    is not necessarily identical to layers 49/50), so one five-way QKV
    projection cannot represent all of them.  The post-load path stores the
    mirror K/V weights as one concatenated projection on the imitated target
    attention; keep the fallback for the original single-mirror layout.
    """
    mirror_ids = getattr(attn, "welm_kv_mirror_source_ids", None)
    mirror_weight = getattr(attn, "welm_kv_mirror_proj_weight", None)
    if mirror_ids and mirror_weight is not None:
        mirror_bias = getattr(attn, "welm_kv_mirror_proj_bias", None)
        mirror_qkv = F.linear(hidden_states, mirror_weight, mirror_bias)
        kv_width = attn.kv_size
        result = []
        for i, mirror_id in enumerate(mirror_ids):
            offset = i * 2 * kv_width
            mirror_k, mirror_v = mirror_qkv[..., offset : offset + kv_width], mirror_qkv[
                ..., offset + kv_width : offset + 2 * kv_width
            ]
            result.append((mirror_id, mirror_k, mirror_v))
        return result

    if qkv is None:
        qkv, _ = attn.qkv_proj(hidden_states)
    expected_width = attn.q_size + 4 * attn.kv_size
    if qkv.shape[-1] < expected_width:
        return []
    _, _, _, mirror_k, mirror_v = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.kv_size, attn.kv_size],
        dim=-1,
    )
    mirror_ids = _welm_mirror_ids_for_imitated(attn, attn.kv_mirror_layer_idx)
    return [(mirror_id, mirror_k, mirror_v) for mirror_id in mirror_ids]


def welm_mtp_mirror_kv(input_ids, positions, forward_batch, embed_tokens, oe_fn):
    """Set the NextN layer's KV-mirror activation for a non-prefill draft forward.

    In draft prefill the NextN layer (a KV-mirror layer imitating layer 0)
    projects target layer 0's attention input of token p with its own K/V
    weights. Reproduce that for draft decode / draft extend: token p is the
    predecessor of the draft input, and layer 0's input is
    input_layernorm(OE-embedding(token p)).
    """
    from sglang.srt.managers.schedule_batch import NGramInputIds

    ring = WelmSpecNgramRing.ensure(input_ids.device)
    bs = forward_batch.req_pool_indices.shape[0]
    rows = torch.repeat_interleave(
        forward_batch.req_pool_indices.to(torch.int64), input_ids.shape[0] // bs
    )
    tpos = positions.to(torch.int64)
    tok = ring[rows, torch.remainder(tpos, ring.shape[1])]
    grams = NGramInputIds(input_ids_grams=_welm_read_ring(ring, rows, tpos))
    h = oe_fn(tok, grams, embed_tokens(tok))
    layer0 = LayerManager.decoder_layer[0]
    h0 = layer0.input_layernorm(h)
    if isinstance(h0, tuple):
        h0 = h0[0]
    attn0 = layer0.self_attn
    qkv, _ = attn0.qkv_proj(h0)
    mirror_pairs = _welm_mirror_kv_from_projection(attn0, h0, qkv=qkv)
    # Draft decode/extend has no target-layer forward in the same call.
    # Reconstruct target layer-0's mirror K/V and publish one entry per
    # physical consumer (48/49/50) on this ForwardBatch.
    if not mirror_pairs:
        mirror_ids = _welm_mirror_ids_for_imitated(attn0, 0)
        if not mirror_ids:
            mirror_ids = list(
                range(
                    LayerManager.num_target_layers,
                    LayerManager.num_target_layers
                    + LayerManager.num_nextn_predict_layers,
                )
            )
        if qkv.shape[-1] >= attn0.q_size + 4 * attn0.kv_size:
            _, _, _, mirror_k, mirror_v = qkv.split(
                [
                    attn0.q_size,
                    attn0.kv_size,
                    attn0.kv_size,
                    attn0.kv_size,
                    attn0.kv_size,
                ],
                dim=-1,
            )
            mirror_pairs = [
                (mirror_id, mirror_k, mirror_v) for mirror_id in mirror_ids
            ]
    states = _get_welm_kv_mirror_states(forward_batch)
    for mirror_id, mirror_k, mirror_v in mirror_pairs:
        states[mirror_id] = (mirror_k, mirror_v)
    _set_welm_kv_mirror_states(forward_batch, states)


class LayerManager:
    decoder_layer = dict()
    num_nextn_predict_layers: int = 0
    num_target_layers: int = 0
    num_nextn_predict_layer_idx: List[int] = []

    @staticmethod
    def set_decoder_layer(layer_idx, decoder_layer):
        LayerManager.decoder_layer[layer_idx] = decoder_layer

    @staticmethod
    def post_init(kv_mirror_layers, kv_mirror_imitated_layers, is_nextn=False):

        if is_nextn:
            LayerManager.num_nextn_predict_layer_idx = kv_mirror_layers
        for mirror_layer_id in kv_mirror_layers:
            imitated_layer_id = kv_mirror_imitated_layers[
                kv_mirror_layers.index(mirror_layer_id)
            ]
            # ``decoder_layer`` is keyed by global layer id and may be sparse
            # (per-step MTP draft models register only their own layer).
            if (
                mirror_layer_id not in LayerManager.decoder_layer
                or imitated_layer_id not in LayerManager.decoder_layer
            ):
                continue
            mirror_layer_attn = LayerManager.decoder_layer[mirror_layer_id].self_attn
            imitated_layer_attn = LayerManager.decoder_layer[
                imitated_layer_id
            ].self_attn

            mirror_qkv_proj_weight = mirror_layer_attn.qkv_proj.weight
            mirror_qkv_proj_bias = getattr(mirror_layer_attn.qkv_proj, "bias", None)
            imitated_qkv_proj_weight = imitated_layer_attn.qkv_proj.weight
            imitated_qkv_proj_bias = getattr(imitated_layer_attn.qkv_proj, "bias", None)
            assert (mirror_qkv_proj_bias is not None) == (
                imitated_qkv_proj_bias is not None
            )

            # A mirror consumer only needs the physical layer's Q projection.
            # Keep this compact q-only view for the consumer path; the target
            # imitated layer retains its normal Q/K/V projection and receives
            # each mirror layer's K/V through the concatenated source below.
            mirror_q_weight = mirror_qkv_proj_weight[: mirror_layer_attn.q_size]
            mirror_q_bias = (
                None
                if mirror_qkv_proj_bias is None
                else mirror_qkv_proj_bias[: mirror_layer_attn.q_size]
            )
            if hasattr(mirror_layer_attn, "qkv_proj_weight"):
                mirror_layer_attn.qkv_proj_weight.copy_(mirror_q_weight)
            else:
                mirror_layer_attn.qkv_proj_weight = mirror_q_weight.clone()
            if mirror_q_bias is None:
                mirror_layer_attn.qkv_proj_bias = None
            elif (
                hasattr(mirror_layer_attn, "qkv_proj_bias")
                and mirror_layer_attn.qkv_proj_bias is not None
            ):
                mirror_layer_attn.qkv_proj_bias.copy_(mirror_q_bias)
            else:
                mirror_layer_attn.qkv_proj_bias = mirror_q_bias.clone()

            # Several physical mirror layers may imitate one target layer.
            # Accumulate their K/V projections instead of overwriting the
            # target's projection on every iteration.  The checkpoint's K/V
            # weights are not guaranteed to be identical across 48/49/50.
            source_ids = list(
                getattr(imitated_layer_attn, "welm_kv_mirror_source_ids", [])
            )
            source_weights = list(
                getattr(imitated_layer_attn, "welm_kv_mirror_source_weights", [])
            )
            source_biases = list(
                getattr(imitated_layer_attn, "welm_kv_mirror_source_biases", [])
            )
            mirror_kv_weight = mirror_qkv_proj_weight[
                mirror_layer_attn.q_size : mirror_layer_attn.q_size
                + 2 * mirror_layer_attn.kv_size
            ]
            mirror_kv_bias = (
                None
                if mirror_qkv_proj_bias is None
                else mirror_qkv_proj_bias[
                    mirror_layer_attn.q_size : mirror_layer_attn.q_size
                    + 2 * mirror_layer_attn.kv_size
                ]
            )
            # post_init runs before and after weight loading; re-registering a
            # mirror layer must replace its entry, not project it twice.
            if mirror_layer_id in source_ids:
                slot = source_ids.index(mirror_layer_id)
                source_weights[slot] = mirror_kv_weight
                source_biases[slot] = mirror_kv_bias
            else:
                source_ids.append(mirror_layer_id)
                source_weights.append(mirror_kv_weight)
                source_biases.append(mirror_kv_bias)
            imitated_layer_attn.welm_kv_mirror_source_ids = source_ids
            imitated_layer_attn.welm_kv_mirror_source_weights = source_weights
            imitated_layer_attn.welm_kv_mirror_source_biases = source_biases
            imitated_layer_attn.welm_kv_mirror_proj_weight = torch.cat(
                source_weights, dim=0
            ).contiguous()
            if any(bias is not None for bias in source_biases):
                imitated_layer_attn.welm_kv_mirror_proj_bias = torch.cat(
                    [bias for bias in source_biases if bias is not None], dim=0
                ).contiguous()
            else:
                imitated_layer_attn.welm_kv_mirror_proj_bias = None
            # Keep the fused [qkv | gate | mirror] weight in sync with the new
            # mirror sources (only once the layer has been merged before).
            if getattr(imitated_layer_attn, "qkvg_weight", None) is not None:
                imitated_layer_attn.build_qkvg_weight()

        torch.cuda.empty_cache()


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        swiglu_clamp_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.swiglu_clamp_limit = swiglu_clamp_limit

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        if self.swiglu_clamp_limit is not None and self.swiglu_clamp_limit > 0:
            d = gate_up.shape[-1] // 2
            gate = F.silu(gate_up[..., :d]).clamp_(max=self.swiglu_clamp_limit)
            up = gate_up[..., d:].clamp(min=-self.swiglu_clamp_limit, max=self.swiglu_clamp_limit)
            x = gate * up
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


def expert_bias_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    expert_bias: torch.Tensor,
    renormalize: bool = False,
    score_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    if score_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1).type_as(gating_output)
    else:
        scores = torch.sigmoid(gating_output).type_as(gating_output)

    if (
        scores.is_cuda
        and scores.dtype == torch.float32
        and expert_bias.dtype == torch.float32
    ):
        topk_scores, indices = mmq_style_expert_bias_topk(scores, expert_bias, topk)
    else:
        scores_for_routing = scores + expert_bias
        _, indices = torch.topk(scores_for_routing, topk, dim=-1)
        topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)

    return topk_scores, indices


def sigmoid_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if softmax, then use qwen3 moe's routing function
    scores = torch.sigmoid(gating_output).type_as(gating_output)
    scores_for_routing = scores
    if correction_bias is not None:
        scores += correction_bias
    _, indices = torch.topk(scores, topk, dim=-1)
    topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)
    return topk_scores, indices


class Qwen2MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.expert_bias = torch.nn.Parameter(
            torch.zeros((config.num_experts), dtype=torch.float32)
        )
        self.layer_id = layer_id
        self.num_hidden_layers = config.num_hidden_layers
        self.last_final_experts_output: Optional[torch.Tensor] = None
        self.last_final_shared_output: Optional[torch.Tensor] = None
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        moe_clamp_limits = getattr(config, "moe_expert_swiglu_clamp_limit_layerwise", [])
        moe_clamp_limit = (
            moe_clamp_limits[layer_id]
            if layer_id < len(moe_clamp_limits) and moe_clamp_limits[layer_id] > 0
            else None
        )
        shared_clamp_limits = getattr(config, "shared_expert_swiglu_clamp_limit_layerwise", [])
        shared_clamp_limit = (
            shared_clamp_limits[layer_id]
            if layer_id < len(shared_clamp_limits) and shared_clamp_limits[layer_id] > 0
            else None
        )

        self.router_score_func = (
            config.router_score_func
            if hasattr(config, "router_score_func")
            else "softmax"
        )
        if config.moe_routing_type == "expert_bias":
            from functools import partial

            custom_routing_function = partial(
                expert_bias_routing,
                expert_bias=self.expert_bias,
                score_func=self.router_score_func,
            )
            self._ref_routing_function = custom_routing_function
            self.custom_routing_function = self._routing_with_precomputed
        else:
            if self.router_score_func == "softmax":
                self.custom_routing_function = None
            elif self.router_score_func == "sigmoid":
                self.custom_routing_function = sigmoid_routing_function
            else:
                raise ValueError(f"Unknown router_score_func: {self.router_score_func}")

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            custom_routing_function=self.custom_routing_function,
        )

        self.num_experts = config.num_experts
        self.fuse_shared_expert = (
            _WELM_FUSE_SHARED_EXPERT
            and quant_config is None
            and config.moe_routing_type == "expert_bias"
            and config.shared_expert_intermediate_size == config.moe_intermediate_size
            and shared_clamp_limit == moe_clamp_limit
            and not getattr(config, "has_shared_expert_gate", True)
            and get_parallel().moe_ep_size == 1
        )
        self.num_fused_shared = 1 if self.fuse_shared_expert else 0
        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok + self.num_fused_shared,
            num_experts=config.num_experts + self.num_fused_shared,
            num_fused_shared_experts=self.num_fused_shared,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            # The installed FusedMoE API names this post-gate activation
            # clamp ``swiglu_limit`` (the official WeLM tree calls it
            # ``swiglu_clamp_limit``).
            swiglu_limit=moe_clamp_limit,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.weight.data = self.gate.weight.to(torch.float32)
        # With an a2a backend (DeepEP) every rank holds its own tokens and the
        # combine returns complete per-token outputs: the shared expert is
        # replicated and no TP all-reduce follows the MoE.
        from sglang.srt.layers.moe import get_moe_a2a_backend

        self.use_a2a = not get_moe_a2a_backend().is_none()
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                swiglu_clamp_limit=shared_clamp_limit,
                **({"tp_rank": 0, "tp_size": 1} if self.use_a2a else {}),
            )
        else:
            self.shared_expert = None

        self.shared_expert_gate = None
        has_shared_expert_gate = getattr(
            config, "has_shared_expert_gate", True
        )  # default to true since qwen2_moe always has it
        if has_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        self._precomputed_topk = None
        self.gate_weight_bf16: Optional[torch.Tensor] = None

    def _routing_with_precomputed(self, hidden_states, gating_output, topk, renormalize):
        pre = self._precomputed_topk
        if pre is not None:
            self._precomputed_topk = None
            return pre
        topk_weights, topk_ids = self._ref_routing_function(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        if self.fuse_shared_expert:
            m = topk_ids.shape[0]
            topk_ids = torch.cat(
                [topk_ids, topk_ids.new_full((m, 1), self.num_experts)], dim=1
            )
            topk_weights = torch.cat(
                [topk_weights, topk_weights.new_ones((m, 1))], dim=1
            )
        return topk_weights, topk_ids

    def post_load_weights(self):
        self.gate_weight_bf16 = self.gate.weight.data.to(torch.bfloat16).contiguous()
        if self.fuse_shared_expert:
            # FusedMoE keeps the fused shared slot after the routed experts.
            w13 = self.experts.w13_weight
            w2 = self.experts.w2_weight
            gate_up = self.shared_expert.gate_up_proj.weight
            down = self.shared_expert.down_proj.weight
            assert w13.shape[1:] == gate_up.shape, (w13.shape, gate_up.shape)
            assert w2.shape[1:] == down.shape, (w2.shape, down.shape)
            w13.data[self.num_experts].copy_(gate_up.data)
            w2.data[self.num_experts].copy_(down.data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        return_components: bool = False,
        skip_component_output: bool = False,
    ) -> torch.Tensor:
        dump_this_layer = _welm_should_dump_layer(self.layer_id)
        dump_prefix = f"model.layers.{self.layer_id}.mlp"
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.router.input", hidden_states)
            _welm_dump_tensor(f"{dump_prefix}.router.input_fp32", hidden_states_fp32)
        shared_output = None
        if self.shared_expert is not None and not self.fuse_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                )
        use_fast_router = (
            _WELM_FAST_ROUTER
            and not dump_this_layer
            and self.gate_weight_bf16 is not None
            and self.router_score_func == "sigmoid"
            and getattr(self, "_ref_routing_function", None) is not None
            and 0 < num_tokens <= WELM_FAST_ROUTER_MAX_M
            and hidden_states.dtype == torch.bfloat16
        )
        if use_fast_router:
            topk_weights, topk_ids, router_logits = welm_fast_router_topk(
                hidden_states,
                self.gate_weight_bf16,
                self.expert_bias,
                self.topk.topk_config.top_k,
                write_scores=True,
                shared_expert_id=self.num_experts if self.fuse_shared_expert else -1,
            )
            self._precomputed_topk = (topk_weights, topk_ids)
        else:
            router_logits = mmq_style_router_linear(hidden_states, self.gate.weight)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.router.logits", router_logits)
            router_scores = (
                torch.softmax(router_logits, dim=-1).type_as(router_logits)
                if self.router_score_func == "softmax"
                else torch.sigmoid(router_logits).type_as(router_logits)
            )
            _welm_dump_tensor(f"{dump_prefix}.router.scores", router_scores)
        topk_output = self.topk(hidden_states, router_logits)
        self._precomputed_topk = None
        if dump_this_layer and hasattr(topk_output, "topk_weights"):
            _welm_dump_tensor(f"{dump_prefix}.router.topk_scores", topk_output.topk_weights)
            _welm_dump_tensor(f"{dump_prefix}.router.topk_ids", topk_output.topk_ids)
        experts_output = self.experts(hidden_states, topk_output)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.experts_output", experts_output)
        if return_components and skip_component_output:
            return (
                experts_output.view(num_tokens, hidden_dim),
                experts_output.view(num_tokens, hidden_dim),
                shared_output.view(num_tokens, hidden_dim)
                if shared_output is not None
                else None,
            )
        final_hidden_states = experts_output
        self.last_final_experts_output = None
        self.last_final_shared_output = None
        if shared_output is not None:
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.shared_output", shared_output)
            if (
                self.layer_id == self.num_hidden_layers - 1
                and self.tp_size == 1
                and not use_reduce_scatter
            ):
                self.last_final_experts_output = experts_output
                self.last_final_shared_output = shared_output
            final_hidden_states = final_hidden_states + shared_output
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.output", final_hidden_states)
        if self.tp_size > 1 and not use_reduce_scatter and not self.use_a2a:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        if return_components:
            return (
                final_hidden_states,
                experts_output.view(num_tokens, hidden_dim),
                shared_output.view(num_tokens, hidden_dim)
                if shared_output is not None
                else None,
            )
        return final_hidden_states


class LinearScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


# WelmV4InplaceRotaryEmbedding
class Qwen2MoeYarnScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        compress: float = 0,
        max_position: int = 40 * 4096,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.compress = compress
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.max_position = max_position
        inv_freq_extra = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        inv_freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq_extra", inv_freq_extra, persistent=False)
        self.register_buffer("inv_freq_inter", inv_freq_inter, persistent=False)

        self.cos_sin_cache = self._update_cos_sin_cache(self.max_position)

    def _update_cos_sin_cache(self, seqlen: int):
        """Update cos/sin cache with YaRN scaling"""
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float32
        ).to(device=self.inv_freq_inter.device)

        inv_freq = (
            self.inv_freq_inter * (1 - inv_freq_mask)
            + self.inv_freq_extra * inv_freq_mask
        )

        seq = (
            torch.arange(seqlen, device=self.inv_freq_extra.device, dtype=torch.float32)
            * self.compress
        )

        freqs = torch.outer(seq, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        _cos_cached = (torch.cos(freqs) * _mscale).to(torch.float32)
        _sin_cached = (torch.sin(freqs) * _mscale).to(torch.float32)
        cache = torch.cat((_cos_cached, _sin_cached), dim=-1)
        return cache


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    compress: float = 1.0,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        raise ValueError(f"Please set RoPE scaling")
    else:
        scaling_type = rope_scaling["type"]

        if scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )

        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            base_max_position = int(original_max_position * scaling_factor)
            if max_position < base_max_position:
                raise ValueError(
                    f"max_position ({max_position}) < original_max_position "
                    f"({original_max_position}) * scaling_factor ({scaling_factor})"
                )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            rotary_emb = Qwen2MoeYarnScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
                compress=compress,
                max_position=max_position,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


class Qwen2MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        compress: float = 1.0,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        out_bias: int = False,
        qk_norm: bool = False,
        k_norm: bool = False,
        qk_rope_head_dim: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
        kv_mirror_layers=[],
        kv_mirror_imitated_layers=[],
        sliding_window_size_layerwise=[],
        enable_attn_sink_layerwise=[],
        layer_idx: Optional[int] = None,
        o_norm=False,
        rms_norm_eps: float = 1e-5,
        total_layer_num: int = 1,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = attn_tp_rank

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.compress = compress
        self.max_position_embeddings = max_position_embeddings
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_norm = qk_norm
        self.only_k_norm = k_norm
        self.use_o_norm = o_norm
        self.total_layer_num = total_layer_num
        self.o_norm = (
            WelmV4FusedRMSNorm(self.hidden_size, eps=rms_norm_eps)
            if self.use_o_norm
            else None
        )

        self.q_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=rms_norm_eps)
            if self.qk_norm
            else None
        )
        self.k_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=rms_norm_eps)
            if self.qk_norm or self.only_k_norm
            else None
        )

        self.kv_mirror_layers = kv_mirror_layers
        self.kv_mirror_imitated_layers = kv_mirror_imitated_layers
        self.layer_idx = layer_idx
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.kv_mirror_layers:",
            self.kv_mirror_layers,
            "self.kv_mirror_imitated_layers:",
            self.kv_mirror_imitated_layers,
            flush=True,
        )
        # Layer-wise config lists are indexed by the global layer id; the NextN
        # (MTP) layer is local layer 0 but sits after the target layers.
        cfg_layer_idx = layer_idx + total_layer_num if is_nextn else layer_idx
        if len(sliding_window_size_layerwise) > cfg_layer_idx:
            window = sliding_window_size_layerwise[cfg_layer_idx]
            # Match the HF reference: a window equal to max_position_embeddings
            # means full attention, and a window of N keeps the query plus the
            # previous N - 1 keys (FlashAttention's inclusive left bound).
            # Passing the raw value would mark full layers as SWA layers.
            if window <= 0 or window >= max_position_embeddings:
                self.sliding_window_size = -1
            else:
                self.sliding_window_size = window - 1
        else:
            self.sliding_window_size = -1
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.sliding_window_size:",
            self.sliding_window_size,
            flush=True,
        )
        if len(enable_attn_sink_layerwise) > cfg_layer_idx:
            self.enable_attention_sink = enable_attn_sink_layerwise[cfg_layer_idx]
        else:
            self.enable_attention_sink = False
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.enable_attention_sink:",
            self.enable_attention_sink,
            flush=True,
        )
        if self.enable_attention_sink == True:
            self.attn_sink = nn.Parameter(
                torch.empty(self.num_heads), requires_grad=False
            )
        else:
            self.attn_sink = None
        self.attn_sink_cast: Optional[torch.Tensor] = None
        self.qkvg_weight: Optional[torch.Tensor] = None
        self.qkvgm_weight: Optional[torch.Tensor] = None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=not is_dp_attention_enabled(),
            prefix=add_prefix("o_proj", prefix),
        )
        if rope_scaling is None:
            rope_scaling = {"type": "linear", "factor": 1 / self.compress}
        else:
            assert self.compress == 1.0, "Compress must be 1.0 for custom rope scaling."
            if rope_scaling["type"] == "yarn":
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                apply_softmax_scale = rope_scaling.get("apply_softmax_scale", False)
                scaling_factor = rope_scaling["factor"]
                if apply_softmax_scale and mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                    self.scaling = self.scaling * mscale * mscale

        self.rotary_emb = get_rope(
            # self.qk_rope_head_dim,
            self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.rotary_emb_orig = get_rope(
            self.qk_rope_head_dim,
            # self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=self.sliding_window_size,
        )
        self.gated_self_attention_headwise = True
        if self.gated_self_attention_headwise:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        self.kv_mirror_layer_idx = (
            layer_idx
            if not is_nextn
            else layer_idx + total_layer_num
        )
        # Use the global index: the NextN layer is local layer 0 but mirrors
        # as layer num_hidden_layers.
        self.attn.is_kv_mirror = self.kv_mirror_layer_idx in self.kv_mirror_layers
        if get_global_server_args().speculative_algorithm is not None:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_nextn_predict_layers - 1
            )
        else:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_target_layers - 1
            )
        self.is_nextn = is_nextn

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_o_norm: bool = False,
        kv_mirror_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        dump_this_layer = _welm_should_dump_layer(self.layer_idx)
        dump_prefix = f"model.layers.{self.layer_idx}.self_attn"
        use_qkvg = self.qkvg_weight is not None and not dump_this_layer
        gate = None
        if kv_mirror_states is None:
            kv_mirror_states = _get_welm_kv_mirror_states(forward_batch)
        if self.kv_mirror_layer_idx in self.kv_mirror_imitated_layers:
            # Keep the target layer's ordinary Q/K/V projection for its own
            # attention, then project every physical mirror consumer with its
            # own K/V weights.  This is important when 48, 49 and 50 all
            if use_qkvg and self.qkvgm_weight is not None:
                # One GEMM: [q k v | gate | mirror K/V for every consumer].
                out = _welm_linear(hidden_states, self.qkvgm_weight)
                n_qkvg = self.qkvg_weight.shape[0]
                qkv, gate = out[..., :n_qkvg].split(
                    [n_qkvg - self.num_heads, self.num_heads], dim=-1
                )
                mirror_qkv = out[..., n_qkvg:]
                kv_width = self.kv_size
                for i, mirror_layer_id in enumerate(self.welm_kv_mirror_source_ids):
                    off = i * 2 * kv_width
                    kv_mirror_states[mirror_layer_id] = (
                        mirror_qkv[..., off : off + kv_width],
                        mirror_qkv[..., off + kv_width : off + 2 * kv_width],
                    )
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            else:
                if use_qkvg:
                    qkv = _welm_linear(hidden_states, self.qkvg_weight)
                    qkv, gate = qkv.split(
                        [qkv.shape[-1] - self.num_heads, self.num_heads], dim=-1
                    )
                else:
                    qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                for mirror_layer_id, mirror_k, mirror_v in _welm_mirror_kv_from_projection(
                    self, hidden_states, qkv=qkv
                ):
                    kv_mirror_states[mirror_layer_id] = (mirror_k, mirror_v)
        elif self.kv_mirror_layer_idx in self.kv_mirror_layers:
            kv_activation = kv_mirror_states.get(self.kv_mirror_layer_idx)
            # A draft decode may run without a reconstructed target activation;
            # retain the ordinary QKV path in that case.  Target mirror layers
            # and draft-extend must fail loudly rather than read global state.
            use_standard_qkv = (
                self.is_nextn
                and not forward_batch.forward_mode.is_extend_without_speculative()
                and kv_activation is None
            )
            if use_standard_qkv:
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            else:
                if (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                    and forward_batch.custom_last_index is None
                ):
                    forward_batch.custom_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                    hidden_states = hidden_states[forward_batch.custom_last_index]
                kv_activation = kv_mirror_states.pop(self.kv_mirror_layer_idx, None)
                if kv_activation is None:
                    raise RuntimeError(
                        "Missing mirrored KV activation for mirror layer "
                        f"{self.kv_mirror_layer_idx} (imitated layer "
                        f"{self.kv_mirror_imitated_layers[self.kv_mirror_layers.index(self.kv_mirror_layer_idx)]})"
                    )
                k, v = kv_activation
                if self.need_clear_kv_cache:
                    kv_mirror_states.clear()
                if use_qkvg:
                    q, gate = _welm_linear(hidden_states, self.qkvg_weight).split(
                        [self.q_size, self.num_heads], dim=-1
                    )
                else:
                    q = F.linear(
                        hidden_states, self.qkv_proj_weight, self.qkv_proj_bias
                    )
        elif use_qkvg:
            q, k, v, gate = _welm_linear(hidden_states, self.qkvg_weight).split(
                [self.q_size, self.kv_size, self.kv_size, self.num_heads], dim=-1
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.positions", positions)
            if forward_batch.extend_seq_lens is not None:
                _welm_dump_tensor(
                    f"{dump_prefix}.extend_seq_lens",
                    forward_batch.extend_seq_lens,
                )
            _welm_dump_tensor(f"{dump_prefix}.q_pre_rope", q)
            _welm_dump_tensor(f"{dump_prefix}.k_pre_rope", k)
            _welm_dump_tensor(f"{dump_prefix}.v", v)

        q_shape = q.shape
        k_shape = k.shape
        mode = forward_batch.forward_mode
        use_fast_qk = (
            _WELM_FAST_QK
            and not dump_this_layer
            and self.q_norm is None
            and self.head_dim > self.qk_rope_head_dim > 0
            and q.dtype == torch.bfloat16
            and q.dim() == 2
            and k.dim() == 2
            and q.shape[0] == k.shape[0] == positions.shape[0]
            and q.stride(-1) == 1
            and k.stride(-1) == 1
            and not (
                forward_batch.enable_kv_mirror
                and mode.is_extend_without_speculative()
                and self.kv_mirror_layer_idx in self.kv_mirror_layers
            )
        )
        q_prescaled = False
        attn_kwargs_save = True
        if use_fast_qk:
            # The HCU paged-attention path used for decode / target verify
            # rescales q by softmax_scale / default_scale; fold that into the
            # rope kernel (same bf16 rounding) and tell the wrapper.
            q_prescaled = mode.is_decode() or mode.is_target_verify()
            q_ratio = (
                self.scaling / (self.head_dim**-0.5) if q_prescaled else 1.0
            )
            kv_store = None
            if (
                _WELM_FUSED_KV_STORE
                and mode.is_decode()
                and get_global_server_args().speculative_algorithm is None
                and v.dim() == 2
                and v.stride(-1) == 1
                and forward_batch.out_cache_loc is not None
                and forward_batch.out_cache_loc.shape[0] == k.shape[0]
            ):
                bufs = _welm_fused_kv_store_buffers(
                    forward_batch, self.attn.layer_id, k.dtype
                )
                if bufs is not None and bufs[0].shape[1] == k.shape[1] // self.head_dim:
                    kv_store = (v, bufs[0], bufs[1], forward_batch.out_cache_loc)
            k, q = welm_qk_norm_rope(
                q,
                k,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.head_dim,
                self.qk_rope_head_dim,
                self.k_norm.weight if self.k_norm is not None else None,
                self.k_norm.eps if self.k_norm is not None else 1e-6,
                q_scale=q_ratio,
                # q is a strided view of the fused projection output; a
                # contiguous q here saves the attention backend a copy kernel.
                return_q=True,
                kv_store=kv_store,
            )
            if kv_store is not None:
                attn_kwargs_save = False
            q_prescaled = q_prescaled and q_ratio != 1.0
        else:
            q, k = self._qk_norm_rope_ref(
                positions, q, k, forward_batch, dump_this_layer, dump_prefix
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.q_post_rope", q)
            _welm_dump_tensor(f"{dump_prefix}.k_post_rope", k)

        attn_kwargs = {}
        if self.attn_sink is not None:
            attn_sink = self.attn_sink_cast
            if attn_sink is None or attn_sink.dtype != q.dtype:
                # HCU FA paged attention requires sinks to use the same dtype
                # as the query/output tensor.
                attn_sink = self.attn_sink.to(dtype=q.dtype)
                local_num_heads = q.shape[-1] // self.head_dim
                if attn_sink.numel() != local_num_heads:
                    start = self.attn_tp_rank * local_num_heads
                    attn_sink = attn_sink.narrow(0, start, local_num_heads)
            attn_kwargs["sinks"] = attn_sink
        if not attn_kwargs_save:
            attn_kwargs["save_kv_cache"] = False
        if q_prescaled:
            _fa_iface.WELM_Q_PRESCALED = True
            try:
                attn_output = self.attn(q, k, v, forward_batch, **attn_kwargs)
            finally:
                _fa_iface.WELM_Q_PRESCALED = False
        else:
            attn_output = self.attn(q, k, v, forward_batch, **attn_kwargs)
        return self._attn_post(
            attn_output, hidden_states, dump_this_layer, dump_prefix, skip_o_norm, gate
        )

    def build_qkvg_weight(self):
        """Merge gate_proj rows into the (possibly mirror-merged) qkv weight.

        The original weights become views of the merged buffer, so this costs
        no extra memory and later in-place weight updates stay consistent.
        """
        self.qkvg_weight = None
        self.qkvgm_weight = None
        if (
            not _WELM_MERGE_GATE_PROJ
            or not self.gated_self_attention_headwise
            or getattr(self.qkv_proj, "bias", None) is not None
            or getattr(self, "qkv_proj_bias", None) is not None
            or self.gate_proj.weight.dtype != self.qkv_proj.weight.dtype
            or self.gate_proj.weight.device.type == "meta"
        ):
            return
        gate_w = self.gate_proj.weight.data
        if hasattr(self, "qkv_proj_weight"):
            base = self.qkv_proj_weight
        else:
            base = self.qkv_proj.weight.data
        n = base.shape[0]
        g = gate_w.shape[0]
        # A target layer imitated by KV-mirror layers also projects their K/V
        # (welm_kv_mirror_proj_weight).  Append those rows so the layer runs
        # one GEMM instead of three.  LayerManager.post_init calls this again
        # whenever the mirror sources change (e.g. when the MTP draft loads).
        mirror_w = getattr(self, "welm_kv_mirror_proj_weight", None)
        fuse_mirror = (
            mirror_w is not None
            and getattr(self, "welm_kv_mirror_proj_bias", None) is None
            and mirror_w.dtype == base.dtype
            and mirror_w.device == base.device
        )
        parts = [base, gate_w] + ([mirror_w] if fuse_mirror else [])
        merged = torch.cat(parts, dim=0).contiguous()
        if hasattr(self, "qkv_proj_weight"):
            self.qkv_proj_weight = merged[:n]
        else:
            self.qkv_proj.weight.data = merged[:n]
        self.gate_proj.weight.data = merged[n : n + g]
        self.qkvg_weight = merged[: n + g]
        if fuse_mirror:
            self.welm_kv_mirror_proj_weight = merged[n + g :]
            self.qkvgm_weight = merged

    def post_load_weights(self):
        if self.attn_sink is not None:
            sink = self.attn_sink.data.to(torch.bfloat16)
            local_num_heads = self.num_heads
            if sink.numel() != local_num_heads:
                sink = sink.narrow(0, self.attn_tp_rank * local_num_heads, local_num_heads)
            self.attn_sink_cast = sink.contiguous()

    def _qk_norm_rope_ref(
        self, positions, q, k, forward_batch, dump_this_layer, dump_prefix
    ):
        q_shape = q.shape
        k_shape = k.shape
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        if self.q_norm is not None:
            q_by_head, _ = self.q_norm(q_by_head)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.q_after_norm", q_by_head.view(q.shape))
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        if self.k_norm is not None:
            k_by_head = mmq_style_k_rms_norm(
                k_by_head.contiguous(), self.k_norm.weight, self.k_norm.eps
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.k_after_norm", k_by_head.view(k.shape))
        k = k_by_head.view(k.shape)

        qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if qk_nope_head_dim > 0:
            if (
                forward_batch.enable_kv_mirror
                and forward_batch.forward_mode.is_extend_without_speculative()
                and self.kv_mirror_layer_idx in self.kv_mirror_layers
            ):
                self.rotary_emb.forward_cuda(
                    positions, q, k, last_index=forward_batch.custom_last_index
                )
            else:
                self.rotary_emb.forward_cuda(positions, q, k)
            q = q.view(q_shape)
            k = k.view(k_shape)
        else:
            q, k = self.rotary_emb(positions, q, k)
        return q, k

    def _attn_post(
        self,
        attn_output,
        hidden_states,
        dump_this_layer,
        dump_prefix,
        skip_o_norm,
        gate=None,
    ):
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.attn_output", attn_output)
        if gate is not None and attn_output.dim() == 2:
            sigmoid_mul_strided_(gate, attn_output, self.num_heads)
        elif self.gated_self_attention_headwise:
            attn_shape = attn_output.shape
            gate = self.gate_proj(hidden_states)[0].unsqueeze(
                -1
            )  # (bs * seq_len, num_heads, 1)
            if dump_this_layer:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_idx}.attn.router.0", gate.squeeze(-1)
                )
            attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
            inplace_sigmoid_mul(gate, attn_output)
            attn_output = attn_output.view(attn_shape)
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.gated_attn_output", attn_output)

        output, _ = self.o_proj(attn_output)
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_idx}.attn.mixer.o_proj_out", output
            )
        if self.o_norm is not None and not skip_o_norm:
            output, _ = self.o_norm(output)
            if dump_this_layer:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_idx}.attn.mixer.o_norm_out", output
                )
        if dump_this_layer:
            _welm_dump_tensor(f"model.layers.{self.layer_idx}.attn.mixer.0", output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        scale_seq_times = getattr(config, "scale_seq_times", 0)
        if scale_seq_times > 0:
            max_position_embeddings = max_position_embeddings * (scale_seq_times + 1)
        if getattr(config, "qkv_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_bias")
        elif getattr(config, "qkv_proj_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_proj_bias")
        else:
            qkv_bias = True
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        qk_norm = getattr(config, "qk_norm", False)
        k_norm = getattr(config, "k_norm", False)
        out_bias = getattr(config, "out_proj_bias", False)
        head_dim = getattr(
            config, "head_dim", self.hidden_size // config.num_attention_heads
        )
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", head_dim)

        self.kv_mirror_layers = getattr(config, "kv_mirror_layers", [])
        self.kv_mirror_imitated_layers = getattr(
            config, "kv_mirror_imitated_layers", []
        )
        self.sliding_window_size_layerwise = getattr(
            config, "sliding_window_size_layerwise", []
        )
        self.enable_attn_sink_layerwise = getattr(
            config, "enable_attn_sink_layerwise", []
        )
        self.ppln = getattr(config, "ppln", False)
        o_norm = getattr(config, "o_norm", False)
        self.prenorm_layer_idx = getattr(config, "prenorm_layer_idx", [])
        print(
            "self.ppln:",
            self.ppln,
            "o_norm:",
            o_norm,
            "self.prenorm_layer_idx:",
            self.prenorm_layer_idx,
        )
        total_layer_num = getattr(
            config, "num_target_hidden_layers", config.num_hidden_layers
        )

        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            qk_norm=qk_norm,
            k_norm=k_norm,
            qk_rope_head_dim=qk_rope_head_dim,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            prefix=add_prefix("self_attn", prefix),
            kv_mirror_layers=self.kv_mirror_layers,
            kv_mirror_imitated_layers=self.kv_mirror_imitated_layers,
            sliding_window_size_layerwise=self.sliding_window_size_layerwise,
            enable_attn_sink_layerwise=self.enable_attn_sink_layerwise,
            layer_idx=layer_id,
            o_norm=o_norm and layer_id not in self.prenorm_layer_idx,
            rms_norm_eps=config.rms_norm_eps,
            total_layer_num=total_layer_num,
            is_nextn=is_nextn,
        )
        LayerManager.num_nextn_predict_layers = getattr(
            config, "num_nextn_predict_layers", 0
        )
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.is_final_layer = layer_id == total_layer_num - 1 or is_nextn

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen2MoE all layers are sparse (include nextn layers)
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=True,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = WelmV4CommunicatorRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = WelmV4CommunicatorRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.final_mlp_experts_output: Optional[torch.Tensor] = None
        self.final_mlp_shared_output: Optional[torch.Tensor] = None

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )
        LayerManager.set_decoder_layer(self.self_attn.kv_mirror_layer_idx, self)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        kv_mirror_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        if kv_mirror_states is None:
            kv_mirror_states = _get_welm_kv_mirror_states(forward_batch)
        dump_this_layer = _welm_should_dump_layer(self.layer_id)
        if dump_this_layer:
            _welm_dump_tensor(f"model.layers.{self.layer_id}.__input__.0", hidden_states)
        residual_after_layernorm = (
            self.ppln and self.layer_id not in self.prenorm_layer_idx
        )
        use_dp_layer_communicator = is_dp_attention_enabled()
        if use_dp_layer_communicator:
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states, residual, forward_batch
            )
            if residual_after_layernorm:
                residual = hidden_states.to(torch.float32)
        elif residual_after_layernorm:
            hidden_states, _, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
                clone_fp32_out=True,
                output_dtype=self.input_layernorm.weight.dtype
                if hidden_states.dtype == torch.float32
                else hidden_states.dtype,
            )
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
            )
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.input_layernorm.0", hidden_states
            )
            if residual is not None:
                _welm_dump_tensor(f"model.layers.{self.layer_id}.attn.mixer.1", residual)
        use_mmq_norm_after_attn = residual_after_layernorm and self.self_attn.use_o_norm
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                skip_o_norm=use_mmq_norm_after_attn,
                kv_mirror_states=kv_mirror_states,
            )
        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.layer_id == self.kv_mirror_layers[-1]
        ):
            residual = residual[forward_batch.custom_last_index]
            if is_dp_attention_enabled():
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_rank,
                    set_dp_buffer_len,
                )

                dp_rank = get_attention_dp_rank()
                # Attention-TP collectives need the local row count to be a
                # multiple of attn_tp_size; pad the one-row-per-request mirror
                # output with zero rows (trimmed before the final norm).
                rows = hidden_states.shape[0]
                pad = (-rows) % self.attn_tp_size
                forward_batch.welm_mirror_rows = rows
                if pad:
                    hidden_states = torch.cat(
                        [hidden_states, hidden_states.new_zeros((pad, *hidden_states.shape[1:]))]
                    )
                    residual = torch.cat(
                        [residual, residual.new_zeros((pad, *residual.shape[1:]))]
                    )
                new_local_num_tokens = hidden_states.shape[0]
                scale = max(getattr(forward_batch, "scale_seq_factor", 1), 1)
                if scale > 1:
                    new_global_num_tokens_gpu = (
                        forward_batch.global_num_tokens_gpu // scale
                    )
                    forward_batch.global_num_tokens_gpu.copy_(
                        new_global_num_tokens_gpu
                    )
                    new_global_num_tokens = [
                        int(x) for x in new_global_num_tokens_gpu.tolist()
                    ]
                    if forward_batch.global_num_tokens_cpu is not None:
                        forward_batch.global_num_tokens_cpu = new_global_num_tokens
                else:
                    forward_batch.global_num_tokens_gpu[dp_rank] = (
                        new_local_num_tokens
                    )
                    new_global_num_tokens = None
                forward_batch.dp_local_start_pos = None
                forward_batch.dp_local_num_tokens = None
                if new_global_num_tokens is not None:
                    if forward_batch.dp_padding_mode.is_max_len():
                        global_dp_buffer_len = max(new_global_num_tokens) * len(
                            new_global_num_tokens
                        )
                    else:
                        global_dp_buffer_len = sum(new_global_num_tokens)
                    forward_batch.global_dp_buffer_len = global_dp_buffer_len
                else:
                    global_dp_buffer_len = forward_batch.global_dp_buffer_len
                set_dp_buffer_len(
                    global_dp_buffer_len,
                    new_local_num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                    new_global_num_tokens,
                )

        if use_mmq_norm_after_attn and use_dp_layer_communicator and self.attn_tp_size > 1:
            # o_proj does not reduce under DP attention; reduce inside the
            # attention-TP group first (reduce-scatter when the MLP consumes
            # scattered tokens, e.g. DeepEP), matching LayerCommunicator's
            # tensor_split layout, then apply o_norm + residual + post-attn norm.
            from sglang.srt.layers.dp_attention import (
                attn_tp_all_reduce,
                attn_tp_reduce_scatter_tensor,
            )

            full = hidden_states
            if self.layer_scatter_modes.mlp_mode == ScatterMode.SCATTERED:
                hidden_states = full.tensor_split(self.attn_tp_size)[self.attn_tp_rank]
                if full.shape[0] != 0:
                    hidden_states = hidden_states.contiguous()
                    attn_tp_reduce_scatter_tensor(hidden_states, full)
                if residual is not None and residual.shape[0] == full.shape[0]:
                    residual = residual.tensor_split(self.attn_tp_size)[self.attn_tp_rank].contiguous()
            elif full.shape[0] != 0:
                hidden_states = attn_tp_all_reduce(full)
            if hidden_states.shape[0] != 0:
                hidden_states, residual, hidden_states_fp32 = mmq_style_norm_after_attn(
                    hidden_states,
                    residual,
                    self.self_attn.o_norm.weight,
                    self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.eps,
                )
            else:
                hidden_states_fp32 = hidden_states.to(torch.float32)
            if self.layer_scatter_modes.mlp_mode == ScatterMode.FULL:
                # MoE over the whole TP group (no a2a backend): gather every
                # DP replica's tokens; postprocess_layer scatters them back.
                from sglang.srt.layers.dp_attention import (
                    dp_gather_replicate,
                    get_attention_dp_size,
                    get_global_dp_buffer,
                )

                if get_attention_dp_size() != 1:
                    local_hidden_states = hidden_states
                    hidden_states = get_global_dp_buffer(_welm_get_tp_group())
                    dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
                    hidden_states_fp32 = hidden_states.to(torch.float32)
        elif use_mmq_norm_after_attn:
            hidden_states, residual, hidden_states_fp32 = mmq_style_norm_after_attn(
                hidden_states,
                residual,
                self.self_attn.o_norm.weight,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
            if (
                is_dp_attention_enabled()
                and self.attn_tp_size == 1
                and self.layer_scatter_modes.mlp_mode == ScatterMode.FULL
            ):
                from sglang.srt.layers.dp_attention import (
                    dp_gather_partial,
                    get_attention_dp_size,
                    get_global_dp_buffer,
                )

                if get_attention_dp_size() != 1:
                    local_hidden_states = hidden_states
                    hidden_states = get_global_dp_buffer(_welm_get_tp_group())
                    dp_gather_partial(
                        hidden_states, local_hidden_states, forward_batch
                    )
                    hidden_states_fp32 = hidden_states.to(torch.float32)
        else:
            if use_dp_layer_communicator:
                hidden_states, residual = self.layer_communicator.prepare_mlp(
                    hidden_states, residual, forward_batch
                )
                hidden_states_fp32 = hidden_states.to(torch.float32)
            else:
                (
                    hidden_states,
                    residual,
                    hidden_states_fp32,
                ) = self.post_attention_layernorm(
                    hidden_states, residual, clone_fp32_out=True
                )
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.norm_after_attn.output", hidden_states
            )
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.norm_after_attn.output_fp32",
                hidden_states_fp32,
            )
            if residual is not None:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_id}.norm_after_attn.residual", residual
                )
        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        self.final_mlp_experts_output = None
        self.final_mlp_shared_output = None
        mlp_output = self.mlp(
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            return_components=dump_this_layer or self.is_final_layer,
            skip_component_output=(
                self.is_final_layer
                and residual is not None
                and not dump_this_layer
                and getattr(self.mlp, "tp_size", 1) == 1
                and not is_dp_attention_enabled()
            ),
        )
        experts_output = None
        shared_output = None
        if isinstance(mlp_output, tuple):
            hidden_states, experts_output, shared_output = mlp_output
        else:
            hidden_states = mlp_output

        if use_dp_layer_communicator:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        if self.is_final_layer:
            self.final_mlp_experts_output = experts_output
            self.final_mlp_shared_output = shared_output
        if dump_this_layer:
            output_with_residual = hidden_states
            if (
                residual is not None
                and experts_output is not None
                and experts_output.shape == residual.shape
            ):
                output_with_residual = experts_output.float() + residual.float()
                if shared_output is not None:
                    output_with_residual = output_with_residual + shared_output.float()
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.mlp.output_with_residual",
                output_with_residual,
            )
        return hidden_states, residual, kv_mirror_states


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes
        self.scale_seq_times = getattr(config, "scale_seq_times", 0)

        if len(self.oe_vocab_sizes) > 0:
            from sglang.srt.layers.welm_oe_host import HostOEEmbedding, host_dir

            oe_dir = host_dir()
            self.oe_embed = nn.ModuleList(
                [
                    # WELM_OE_HOST_DIR: tables stay in host memory (no HBM).
                    HostOEEmbedding(oe_dir, i, self.oe_vocab_sizes[i], self.oe_dim)
                    if oe_dir
                    else VocabParallelEmbedding(
                        self.oe_vocab_sizes[i],
                        self.oe_dim,
                        **get_embedding_tp_kwargs(),
                    )
                    for i in range(len(self.oe_vocab_sizes))
                ]
            )
            self.oe_gate_up_proj = ReplicatedLinear(
                self.oe_dim * len(self.oe_vocab_sizes),
                config.hidden_size,
                bias=False,
                quant_config=None,
            )

        # Scale sequence length embeddings: N additional embedding groups
        if self.scale_seq_times > 0:
            self.scale_seq_embed_tokens_list = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        config.vocab_size,
                        config.hidden_size,
                        **get_embedding_tp_kwargs(),
                    )
                    for _ in range(self.scale_seq_times)
                ]
            )
            if len(self.oe_vocab_sizes) > 0:
                self.scale_seq_oe_embed_list = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                VocabParallelEmbedding(
                                    self.oe_vocab_sizes[j],
                                    self.oe_dim,
                                    **get_embedding_tp_kwargs(),
                                )
                                for j in range(len(self.oe_vocab_sizes))
                            ]
                        )
                        for _ in range(self.scale_seq_times)
                    ]
                )
                self.scale_seq_oe_up_proj_list = nn.ModuleList(
                    [
                        ReplicatedLinear(
                            self.oe_dim * len(self.oe_vocab_sizes),
                            config.hidden_size,
                            bias=False,
                            quant_config=None,
                        )
                        for _ in range(self.scale_seq_times)
                    ]
                )

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                **get_embedding_tp_kwargs(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        LayerManager.num_target_layers = getattr(
            config, "num_target_hidden_layers", config.num_hidden_layers
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = WelmV4FusedRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def _compute_oe_embedding(
        self,
        input_ids,
        forward_batch,
        base_hidden_states,
        oe_embed_modules=None,
        oe_up_proj_module=None,
    ):
        """Compute over-encoding embedding and combine with base hidden states.
        If oe_embed_modules/oe_up_proj_module are None, use the main OE modules."""
        if oe_embed_modules is None:
            oe_embed_modules = self.oe_embed
        if oe_up_proj_module is None:
            oe_up_proj_module = self.oe_gate_up_proj

        dump_oe = _welm_dump_enabled()
        if dump_oe:
            _welm_dump_tensor("model.oe.input_ids", input_ids)
            _welm_dump_tensor("model.oe.base_hidden_states", base_hidden_states)

        input_ids_ngram = []
        input_ids_ngram_tmp = input_ids
        for g in range(1, max(self.oe_grams)):
            gram_tensor = forward_batch.n_gram_input_ids.get_gram(g + 1)
            if gram_tensor is not None:
                if dump_oe:
                    _welm_dump_tensor(f"model.oe.gram{g + 1}.ids", gram_tensor)
                input_ids_ngram_tmp = input_ids_ngram_tmp + gram_tensor * (
                    self.vocab_size**g
                )
            input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))

        emb_ngram = []
        for i, vs in enumerate(self.oe_vocab_sizes):
            input_ids_ngram_hashed_tmp = input_ids_ngram[self.oe_grams[i] - 2] % vs
            if dump_oe:
                _welm_dump_tensor(
                    f"model.oe.vocab{i}.hashed_ids", input_ids_ngram_hashed_tmp
                )
            emb_ngram_tmp = oe_embed_modules[i](input_ids_ngram_hashed_tmp)
            if dump_oe:
                _welm_dump_tensor(f"model.oe.vocab{i}.embedding", emb_ngram_tmp)
            emb_ngram.append(emb_ngram_tmp)
        emb_new, _ = oe_up_proj_module(torch.cat(emb_ngram, dim=-1))
        hidden_states = (base_hidden_states + emb_new) / 2.0
        if dump_oe:
            _welm_dump_tensor("model.oe.projected", emb_new)
            _welm_dump_tensor("model.oe.output", hidden_states)
        return hidden_states

    def _expand_scale_seq(self, input_ids, forward_batch, hidden_states):
        """Expand hidden_states from (T, D) to (T * scale, D) by interleaving
        main embedding with scale_seq embeddings.

        Layout per original token i:
          [main_emb_i, scale_seq_1_emb_i, ..., scale_seq_N_emb_i]
        """
        scale = self.scale_seq_times + 1
        T = hidden_states.shape[0]
        D = hidden_states.shape[1]

        # (T, D) -> (T, 1, D)
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states_list = [hidden_states]

        for s in range(self.scale_seq_times):
            hs_s = self.scale_seq_embed_tokens_list[s](input_ids)  # (T, D)
            if len(self.oe_grams) > 0 and forward_batch.n_gram_input_ids is not None:
                hs_s = self._compute_oe_embedding(
                    input_ids,
                    forward_batch,
                    hs_s,
                    oe_embed_modules=self.scale_seq_oe_embed_list[s],
                    oe_up_proj_module=self.scale_seq_oe_up_proj_list[s],
                )
            hs_s = hs_s.unsqueeze(1)  # (T, 1, D)
            hidden_states_list.append(hs_s)

        # (T, scale, D) -> (T * scale, D)
        hidden_states = torch.cat(hidden_states_list, dim=1)
        hidden_states = hidden_states.reshape(T * scale, D).contiguous()
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        skip_oe_fusion: bool = False,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        _welm_start_dump_pass()
        forward_batch.welm_mirror_rows = None
        if self.pp_group.is_first_rank:
            kv_mirror_states = _empty_welm_kv_mirror_states()
            if len(self.oe_grams) > 0:
                forward_batch.n_gram_input_ids = welm_spec_n_gram_input_ids(
                    input_ids, positions, forward_batch, is_draft=False
                )
            fused_embed = (
                input_embeds is None
                and len(self.oe_grams) > 0
                and forward_batch.n_gram_input_ids is not None
                and input_ids.shape[0] > 0
                and _can_fuse_embed_ar(self.embed_tokens, self.oe_embed)
            )
            if fused_embed:
                hidden_states = welm_embed_with_oe(
                    self.embed_tokens, self.oe_embed, self.oe_gate_up_proj,
                    input_ids, forward_batch.n_gram_input_ids, self.vocab_size,
                    self.oe_grams, self.oe_vocab_sizes,
                )
            elif input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if _welm_dump_enabled():
                _welm_dump_tensor("model.embed_tokens.output", hidden_states)

            if (
                not fused_embed
                and len(self.oe_grams) > 0
                and not skip_oe_fusion
                and forward_batch.n_gram_input_ids is not None
            ):
                hidden_states = self._compute_oe_embedding(
                    input_ids, forward_batch, hidden_states
                )

            if self.scale_seq_times > 0:
                hidden_states = self._expand_scale_seq(
                    input_ids, forward_batch, hidden_states
                )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            kv_mirror_states = _unpack_welm_kv_mirror_states(pp_proxy_tensors)

        aux_hidden_states = []
        # TBO's operation plan only carries hidden/residual.  Mirror state is
        # a mutable per-forward control object, so use the ordinary loop when
        # this model has KV-mirror pairs.
        if forward_batch.can_run_tbo and not getattr(
            self.config, "kv_mirror_layers", []
        ):
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    layer = self.layers[i]
                    hidden_states, residual, kv_mirror_states = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        kv_mirror_states,
                    )
        _set_welm_kv_mirror_states(forward_batch, kv_mirror_states)
        if not self.pp_group.is_last_rank:
            proxy_tensors = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            _pack_welm_kv_mirror_states(proxy_tensors, kv_mirror_states)
            return PPProxyTensors(proxy_tensors)
        else:
            pre_norm_hidden_states = None
            mirror_rows = getattr(forward_batch, "welm_mirror_rows", None)
            if mirror_rows is not None and hidden_states.shape[0] > mirror_rows:
                hidden_states = hidden_states[:mirror_rows]
                if residual is not None:
                    residual = residual[:mirror_rows]
            if hidden_states.shape[0] != 0:
                if residual is None:
                    pre_norm_hidden_states = hidden_states
                    hidden_states, _ = self.norm(hidden_states)
                else:
                    last_layer = self.layers[self.end_layer - 1]
                    final_experts_output = getattr(
                        last_layer, "final_mlp_experts_output", None
                    )
                    final_shared_output = getattr(
                        last_layer, "final_mlp_shared_output", None
                    )
                    # In TP>1, component tensors are still pre-all-reduce.
                    can_rebuild_final_mlp = (
                        final_experts_output is not None
                        and getattr(last_layer.mlp, "tp_size", 1) == 1
                        and not is_dp_attention_enabled()
                    )
                    if can_rebuild_final_mlp:
                        hidden_states = final_experts_output.float() + residual.float()
                        if final_shared_output is not None:
                            hidden_states = hidden_states + final_shared_output.float()
                        pre_norm_hidden_states = hidden_states.to(self.norm.weight.dtype)
                        hidden_states = F.rms_norm(
                            pre_norm_hidden_states,
                            self.norm.weight.shape,
                            self.norm.weight,
                            eps=self.norm.eps,
                        )
                    else:
                        hidden_states = hidden_states.float() + residual.float()
                        pre_norm_hidden_states = hidden_states.to(self.norm.weight.dtype)
                        hidden_states = F.rms_norm(
                            pre_norm_hidden_states,
                            self.norm.weight.shape,
                            self.norm.weight,
                            eps=self.norm.eps,
                        )

        if (
            len(aux_hidden_states) == 0
            and forward_batch.capture_hidden_mode.need_capture()
            and pre_norm_hidden_states is not None
        ):
            aux_hidden_states = [pre_norm_hidden_states]

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class WeLMV4MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def get_attention_sliding_window_size(self):
        """Window (exclusive of the query, FA convention) of the sliding layers;
        only reported for the hybrid SWA KV pool (WELM_HYBRID_SWA=1)."""
        if os.environ.get("WELM_HYBRID_SWA", "0") != "1":
            return None
        windows = getattr(self.config, "sliding_window_size_layerwise", None) or []
        full = getattr(self.config, "max_position_embeddings", 1 << 30)
        sw = [w for w in windows[: self.config.num_hidden_layers] if 0 < w < full]
        return (min(sw) - 1) if sw else None

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.model = Qwen2MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False
        self.post_init_after_load_weights(is_nextn=False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        skip_oe_fusion: bool = False,
    ) -> torch.Tensor:
        model_output = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            skip_oe_fusion=skip_oe_fusion,
        )
        aux_hidden_states = None
        if isinstance(model_output, tuple):
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
        if self.pp_group.is_last_rank:
            # Contract expanded hidden_states back to logical size for logits.
            # Transformer layers have already processed all T*scale states and
            # written KV cache.  For logits we only need the last state in each
            # scale group (matches MMQ's [:, -1, :] semantic).
            if self.model.scale_seq_times > 0:
                scale = self.model.scale_seq_times + 1
                # Select every scale-th element (last of each group)
                kv_mirror_contracted = (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                )
                if not kv_mirror_contracted:
                    indices = torch.arange(
                        scale - 1,
                        hidden_states.shape[0],
                        scale,
                        device=hidden_states.device,
                    )
                    hidden_states = hidden_states[indices]
                    if aux_hidden_states is not None:
                        aux_hidden_states = [
                            hidden[indices] for hidden in aux_hidden_states
                        ]

                # Restore forward_batch metadata to logical space so that
                # LogitsProcessor sees the un-expanded lengths.
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = (
                        forward_batch.extend_seq_lens // scale
                    )
                    if forward_batch.extend_seq_lens_cpu is not None:
                        forward_batch.extend_seq_lens_cpu = [
                            x // scale for x in forward_batch.extend_seq_lens_cpu
                        ]
                    forward_batch.extend_num_tokens = (
                        forward_batch.extend_num_tokens // scale
                    )

            logits_output = self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
            _welm_trace_target_logits(
                forward_batch, input_ids, positions, logits_output
            )
            logits_output.model_specific_states = forward_batch.model_specific_states
            return logits_output
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        kv_mirror_states = (
            _empty_welm_kv_mirror_states()
            if start == 0
            else _get_welm_kv_mirror_states(forward_batch)
        )
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

            if len(self.model.oe_grams) > 0:
                forward_batch.n_gram_input_ids = welm_spec_n_gram_input_ids(
                    input_ids, positions, forward_batch, is_draft=False
                )
            if (
                len(self.model.oe_grams) > 0
                and forward_batch.n_gram_input_ids is not None
            ):
                forward_batch.hidden_states = self.model._compute_oe_embedding(
                    input_ids, forward_batch, forward_batch.hidden_states
                )

            if self.model.scale_seq_times > 0:
                forward_batch.hidden_states = self.model._expand_scale_seq(
                    input_ids, forward_batch, forward_batch.hidden_states
                )

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                (
                    forward_batch.hidden_states,
                    forward_batch.residual,
                    kv_mirror_states,
                ) = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                    kv_mirror_states,
                )

        _set_welm_kv_mirror_states(forward_batch, kv_mirror_states)
        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states

            # Contract expanded hidden_states back to logical size
            if self.model.scale_seq_times > 0:
                scale = self.model.scale_seq_times + 1
                kv_mirror_contracted = (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                )
                if not kv_mirror_contracted:
                    indices = torch.arange(
                        scale - 1,
                        hidden_states.shape[0],
                        scale,
                        device=hidden_states.device,
                    )
                    forward_batch.hidden_states = hidden_states[indices]
                else:
                    forward_batch.hidden_states = hidden_states
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = (
                        forward_batch.extend_seq_lens // scale
                    )
                    if forward_batch.extend_seq_lens_cpu is not None:
                        forward_batch.extend_seq_lens_cpu = [
                            x // scale for x in forward_batch.extend_seq_lens_cpu
                        ]
                    forward_batch.extend_num_tokens = (
                        forward_batch.extend_num_tokens // scale
                    )

            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
            result.model_specific_states = forward_batch.model_specific_states
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                num_target_layers = LayerManager.num_target_layers
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        if is_nextn:
            # nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            next_layer_prefixes = [
                f"model.layers.{i+num_target_layers}" for i in range(num_nextn_layers)
            ]
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        from sglang.srt.layers.welm_oe_host import host_dir as _oe_host_dir

        oe_on_host = _oe_host_dir() is not None
        for name, loaded_weight in weights:
            if oe_on_host and ".oe_embed." in name:
                # Served from WELM_OE_HOST_DIR (exported once per node).
                continue
            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                flag = False
                matched_prefix = None
                for next_layer_prefix in next_layer_prefixes:
                    if name.startswith(next_layer_prefix):
                        flag = True
                        matched_prefix = next_layer_prefix
                        break
                if not flag:
                    continue
                # if not name.startswith(nextn_layer_prefix):
                #     continue
                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(matched_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    # Use the matched layer, not the loop's last prefix:
                    # otherwise every MTP layer's block loads into the last
                    # decoder slot and the others stay uninitialised.
                    weight_suffix = int(matched_prefix.split(".")[-1])
                    name = name.replace(
                        matched_prefix,
                        f"model.decoder_layers.{weight_suffix-num_target_layers}",
                    )
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "self_attn.gate_proj" in name:
                    continue
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if weight_name == "up_proj" and "scale_seq_oe_up_proj" in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                if name == "model.oe_gate_up_proj.weight":
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "attn_sink" in name:
                            start = get_attention_tp_rank() * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
        self.post_init_after_load_weights(is_nextn=is_nextn)

    def get_embed_and_head(self):
        return [
            self.model.embed_tokens,
            self.model.oe_embed,
            self.model.oe_gate_up_proj,
        ], self.lm_head

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed[0]
        self.model.oe_embed = embed[1]
        self.model.oe_gate_up_proj = embed[2]
        self.lm_head = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def post_init_after_load_weights(self, is_nextn=False):
        if welm_use_device_ngram():
            # Allocate eagerly so CUDA graph capture never creates it.
            param = next(self.parameters(), None)
            if param is not None and param.device.type != "meta":
                WelmSpecNgramRing.ensure(param.device)
        for module in self.modules():
            if module is not self and isinstance(
                module, (Qwen2MoeSparseMoeBlock, Qwen2MoeAttention)
            ):
                param = next(module.parameters(), None)
                if param is not None and param.device.type != "meta":
                    module.post_load_weights()
        if get_global_server_args().speculative_algorithm is not None:
            # Allocate eagerly so CUDA graph capture never creates it.
            param = next(self.parameters(), None)
            if param is not None and param.device.type != "meta":
                WelmSpecNgramRing.ensure(param.device)
        total_kv_mirror_layers = getattr(self.model.config, "kv_mirror_layers", [])
        total_kv_mirror_imitated_layers = getattr(
            self.model.config, "kv_mirror_imitated_layers", []
        )
        if is_nextn:
            kv_mirror_layer_ids = [
                decoder.self_attn.kv_mirror_layer_idx
                for decoder in self.model.decoder_layers
            ]
            kv_mirror_imitated_layers = total_kv_mirror_imitated_layers[
                : len(kv_mirror_layer_ids)
            ]
        else:
            kv_mirror_layer_ids = [
                decoder_layer.self_attn.kv_mirror_layer_idx
                for decoder_layer in self.model.layers
            ]
            kv_mirror_layer_ids = [
                layer_id
                for layer_id in total_kv_mirror_layers
                if layer_id in kv_mirror_layer_ids
            ]  # keep the order of total_kv_mirror_layers
            kv_mirror_imitated_layers = total_kv_mirror_imitated_layers[
                -len(kv_mirror_layer_ids) :
            ]
        LayerManager.post_init(
            kv_mirror_layer_ids, kv_mirror_imitated_layers, is_nextn=is_nextn
        )
        for module in self.modules():
            if isinstance(module, Qwen2MoeAttention):
                module.build_qkvg_weight()
        if not is_nextn:
            from sglang.srt.layers.welmv4_fast_op import warmup_skinny_gemm

            lm_w = getattr(getattr(self, "lm_head", None), "weight", None)
            if lm_w is not None and lm_w.device.type == "cuda":
                warmup_skinny_gemm(lm_w.data)
            for module in self.modules():
                if isinstance(module, Qwen2MoeAttention) and module.qkvg_weight is not None:
                    warmup_skinny_gemm(module.qkvg_weight)
                    break

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = WeLMV4MoeForCausalLM
