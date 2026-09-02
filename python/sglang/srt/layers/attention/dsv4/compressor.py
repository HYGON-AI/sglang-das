# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, Union

import torch
import torch.nn as nn

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.attention.dsv4 import (
    linear_bf16_fp32,
    triton_create_paged_compress_data,
)
from sglang.kernels.ops.attention.dsv4.compress_old import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
)
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_lightop,
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.distributed.parallel_state import get_attn_cp_group
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsv4.rlc import compute_rlc_metadata
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_all_to_all_single,
)
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_lightop,
)
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_finish,
    cp_all_gather_rerange_launch,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    add_prefix,
    get_bool_env_var,
    is_hcu,
    is_hip,
    is_npu,
    set_weight_attrs,
)

_is_hcu = is_hcu()
_is_hip = is_hip()
_is_npu = is_npu()
_use_dpskv4_lightop_quant_k_cache = get_bool_env_var(
    "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE"
)
if _is_hcu:
    from lightop import op


if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class FusedCompressMetadata(NamedTuple):
    write_loc: torch.Tensor
    extra_data: Optional[torch.Tensor]
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan]

    def copy_(self, other: FusedCompressMetadata) -> None:
        from .metadata import maybe_copy_inplace

        self.write_loc.copy_(other.write_loc)
        maybe_copy_inplace(self.extra_data, src=other.extra_data)
        self.plan.copy_(other.plan)


class CompressorBackendMixin:
    def get_paged_compress_metadata(self, compress_ratio: int) -> FusedCompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, FusedCompressMetadata)
        return metadata

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,
        is_paged: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

        assert compress_ratio in (
            4,
            128,
        ), f"DSV4 supports CSA(4x) and HCA(128x) only, got {compress_ratio=}"
        if is_paged:
            metadata = self.get_paged_compress_metadata(compress_ratio)
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
            else:
                last_dim = 2 * head_dim * coff
                assert kv_score_buffer.shape[-1] == last_dim
                kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        else:
            plan = make_compressor_plan(compress_ratio, forward_batch)
            metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
        indices, extra_data, plan = metadata

        if _is_hip:
            if not is_paged:
                raise NotImplementedError("HIP fused compressor expects paged metadata")

            from sglang.kernels.ops.attention.dsv4.fused_compress_triton import (
                hip_compress_forward,
                hip_compress_fused_norm_rope_hadamard_inplace,
                hip_compress_fused_norm_rope_inplace,
            )

            kv_compressed = hip_compress_forward(
                kv_score_buffer=kv_score_buffer,
                kv_score_input=kv_score_input,
                ape=ape,
                indices=indices,
                plan=plan,
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                extra_data=extra_data,
            )
            norm_eps = (
                norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps
            )
            if rotate:
                hip_compress_fused_norm_rope_hadamard_inplace(
                    kv_compressed,
                    norm.weight,
                    norm_eps,
                    freqs_cis_cache,
                    plan,
                    head_dim,
                )
            else:
                hip_compress_fused_norm_rope_inplace(
                    kv_compressed,
                    norm.weight,
                    norm_eps,
                    freqs_cis_cache,
                    plan,
                )
            return kv_compressed

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            indices=indices,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=extra_data,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            getattr(norm, "eps", norm.variance_epsilon),
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = self.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch, attn_backend=self)
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if out_loc.shape[0] > new_compressed_kv.shape[0]:
            out_loc = out_loc[: new_compressed_kv.shape[0]]
        if token_to_kv_pool.is_bf16_attention_kv_cache or (
            envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get()
        ):
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            if (
                _is_hcu
                and _use_dpskv4_lightop_quant_k_cache
                and hasattr(op, "quantize_nope_fp8_rope_bf16_pack_store")
            ):
                token_to_kv_pool.set_extra_key_buffer_lightop_fused(
                    layer_id=layer_id,
                    loc=out_loc,
                    cache_k=new_compressed_kv.bfloat16(),
                )
                return
            if _is_hcu and _use_dpskv4_lightop_quant_k_cache:
                pack = quant_to_nope_fp8_rope_bf16_pack_lightop(
                    new_compressed_kv.bfloat16(), 1e-8
                )
            else:
                pack = quant_to_nope_fp8_rope_bf16_pack_triton(
                    new_compressed_kv.bfloat16()
                )
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        token_to_kv_pool = self.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch, attn_backend=self)
        out_loc = self.forward_metadata.core_metadata.c4_out_loc
        if out_loc.shape[0] > new_compressed_kv.shape[0]:
            out_loc = out_loc[: new_compressed_kv.shape[0]]
        if token_to_kv_pool.use_int8_index_k_cache:
            token_to_kv_pool.set_index_k_int8_buffer(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        elif self.enable_deepseek_v4_fp4_indexer:
            token_to_kv_pool.set_index_k_fp4(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def make_compressor_plan(
    compress_ratio: Literal[4, 128],
    forward_batch: ForwardBatch,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if forward_batch.forward_mode.is_decode():
        seq_lens_32 = forward_batch.seq_lens.to(torch.int32)
        return CompressorDecodePlan(compress_ratio, seq_lens_32)
    if forward_batch.forward_mode.is_prefill():
        assert not forward_batch.forward_mode.is_target_verify()
        extend_lens_list = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_list is not None and seq_lens_cpu is not None
        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            num_q_tokens=sum(extend_lens_list),
            seq_lens=seq_lens_cpu,
            extend_lens=torch.tensor(extend_lens_list),
            device=forward_batch.seq_lens.device,
        )
    elif forward_batch.forward_mode.is_target_verify():
        raise NotImplementedError("target verify mode to be implemented")
    else:
        raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> FusedCompressMetadata:
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # assert ring_size % compress_ratio == 0

    def clip_down(positions: torch.Tensor) -> torch.Tensor:
        return positions // compress_ratio * compress_ratio

    def get_raw_loc(positions: torch.Tensor) -> torch.Tensor:
        positions = positions.masked_fill(positions < 0, 0)
        if compress_ratio == 128:
            state_loc = req_pool_indices * ring_size + positions % ring_size
        else:
            loc = req_to_token[req_pool_indices, positions]
            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(loc)
            swa_pages = swa_loc // swa_page_size
            state_loc = swa_pages * ring_size + swa_loc % ring_size
        return (state_loc // compress_ratio).to(torch.int32)

    is_overlap = is_overlap_compress(compress_ratio)

    if is_prefill:
        assert extend_lens is not None
        write_loc, extra_data = triton_create_paged_compress_data(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
        )

        plan_kwargs: dict
        if seq_lens_cpu is None:
            assert num_q_tokens is not None
            plan_kwargs = dict(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
            )
        else:
            assert extend_lens_cpu is not None
            plan_kwargs = dict(
                num_q_tokens=sum(extend_lens_cpu),
                seq_lens=torch.tensor(seq_lens_cpu),
                extend_lens=torch.tensor(extend_lens_cpu),
            )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            device=seq_lens.device,
            use_cuda_graph=use_prefill_cuda_graph,
            **plan_kwargs,
        )
    else:
        write_positions = clip_down(seq_lens - 1)
        write_loc = get_raw_loc(write_positions)
        if is_overlap:
            write_overlap_loc = get_raw_loc(write_positions - compress_ratio)
            extra_data = write_overlap_loc.view(-1, 1)
        elif _is_hip:
            extra_data = get_raw_loc(write_positions - compress_ratio)
        else:
            extra_data = None
        plan = CompressorDecodePlan(compress_ratio, seq_lens.to(torch.int32))

    return FusedCompressMetadata(write_loc=write_loc, extra_data=extra_data, plan=plan)


# RLC (Repartition-Local Compression) run-time cache: data-independent metadata + derived GPU tensors +
# local compress plan depend only on (ratio, extend_lens, prefix_lens, cp_rank, device), not on the
# layer. One forward's c4 layers reuse the same bundle; rebuilt only when the lengths change.
_rlc_bundle_cache: dict = {}


def _rlc_prefix_aligned(forward_batch: ForwardBatch, ratio: int) -> bool:
    """RLC's prefix-overlap patch assumes each sequence's prefix is a multiple of `ratio`, so the
    extend-first block reads exactly the prefix's last block via the load_normal page. This holds for
    page-aligned chunked prefill (chunks are page_size-aligned, a multiple of ratio); guard the rare
    non-aligned-prefix case by falling back to the base path (which handles any prefix)."""
    seq = forward_batch.seq_lens_cpu
    ext = forward_batch.extend_seq_lens_cpu
    if seq is None or ext is None:
        return False
    return all((int(s) - int(e)) % ratio == 0 for s, e in zip(seq, ext))


def _get_rlc_bundle(extend_lens, prefix_lens, ratio, cp_size, cp_rank, device, write_plan):
    """Build (and cache) the data-independent RLC metadata for one prefill chunk: routing tensors, the
    local compress plan (v1) with PER-SEGMENT PREFIX, expansion rows, the per-segment global sequence
    indices (used to gather load pages from the global paged metadata), and the state-pool WRITE
    indices (compact/remap gather -- pure geometry from `write_plan`). Cached by
    (ratio, cp_size, cp_rank, device, extend_lens, prefix_lens) -> reused across all c4 layers of a
    forward."""
    key = (ratio, cp_size, cp_rank, str(device),
           tuple(int(x) for x in extend_lens), tuple(int(x) for x in prefix_lens))
    bundle = _rlc_bundle_cache.get(key)
    if bundle is not None:
        return bundle

    R = ratio
    meta = compute_rlc_metadata(extend_lens, ratio, cp_size, cp_rank)
    send_idx = torch.tensor(meta.send_idx, dtype=torch.long, device=device)
    recv_perm = torch.tensor(meta.recv_perm, dtype=torch.long, device=device)
    ev = torch.tensor(meta.ev, dtype=torch.long, device=device)

    n_recv = int(sum(meta.out_splits))
    local_plan_c = None
    local_ev = torch.empty(0, dtype=torch.long, device=device)
    if n_recv > 0 and meta.local_extend_lens:
        segs = meta.local_extend_lens
        prefix_local = [int(prefix_lens[meta.local_seg_seqs[i]]) if meta.local_seg_boundary[i] else 0
                        for i in range(len(segs))]
        seq_local = [prefix_local[i] + segs[i] for i in range(len(segs))]
        seq_t = torch.tensor(seq_local, dtype=torch.int64, device=device)
        ext_t = torch.tensor(segs, dtype=torch.int64, device=device)
        local_plan = CompressorPrefillPlan.generate(
            compress_ratio=R, num_q_tokens=n_recv, seq_lens=seq_t, extend_lens=ext_t, device=device)
        lev = local_plan.compress_plan.detach().cpu().contiguous().view(torch.int32)[:, 0]
        local_ev = lev[(lev >= 0) & (lev < n_recv)].sort().values.long().to(device)
        empty16 = torch.zeros((0, 16), dtype=torch.uint8, device=device)
        local_plan_c = local_plan._replace(write_plan=empty16)

    seg_seqs = torch.tensor(meta.local_seg_seqs, dtype=torch.long, device=device)

    empty16 = torch.zeros((0, 16), dtype=torch.uint8, device=device)
    write_W = 0
    write_mine_idx = torch.empty(0, dtype=torch.long, device=device)
    write_src = torch.empty(0, dtype=torch.long, device=device)
    write_remap = empty16
    if write_plan is not None and write_plan.numel() > 0:
        num_q = int(sum(int(x) for x in extend_lens))
        wpi = write_plan.contiguous().view(torch.int32)
        g_all = wpi[:, 0].long()
        wpi = wpi[(g_all >= 0) & (g_all < num_q)]
        write_W = int(wpi.shape[0])
        if write_W > 0:
            wg = wpi[:, 0].long()
            write_mine_idx = (wg % cp_size == cp_rank).nonzero().flatten()
            write_src = (wg[write_mine_idx] // cp_size).contiguous()
            remap = wpi.clone()
            remap[:, 0] = torch.arange(write_W, dtype=torch.int32, device=device)
            write_remap = remap.view(torch.uint8)

    local_ev_kept = local_ev[meta.n_halo:]

    compact_rows = []
    for r in range(cp_size):
        base = r * meta.max_c
        compact_rows.extend(range(base, base + meta.counts[r]))
    compact_idx = torch.tensor(compact_rows, dtype=torch.long, device=device)

    bundle = dict(
        meta=meta, send_idx=send_idx, recv_perm=recv_perm, ev=ev,
        local_plan_c=local_plan_c, local_ev_kept=local_ev_kept, seg_seqs=seg_seqs,
        compact_idx=compact_idx,
        write_W=write_W, write_mine_idx=write_mine_idx, write_src=write_src,
        write_remap=write_remap, empty16=empty16,
    )
    if len(_rlc_bundle_cache) > 8:
        _rlc_bundle_cache.clear()
    _rlc_bundle_cache[key] = bundle
    return bundle


def _rlc_write_state(state_pool, kv_score_local, ape, paged, bundle, ratio, head_dim):
    """Persist this chunk's trailing block(s) into the (replicated) state pool so the NEXT chunk reads
    the correct overlap, using compress_forward's WRITE kernel. All index math is precomputed once per
    forward in `_get_rlc_bundle` (pure geometry). Here we only: fill this rank's owned write rows into
    a compact [W, 4H] buffer, all_reduce(SUM) so every rank holds the full write-set, then run the
    write kernel with the cached remapped plan -- reusing the global write_loc/extra_data so it writes
    the SAME (page, slot) as base. An empty compress plan makes only the write kernel run."""
    W = bundle["write_W"]
    if W == 0:
        return
    write_buf = kv_score_local.new_zeros((W, kv_score_local.shape[1]))
    write_src = bundle["write_src"]
    if write_src.numel():
        write_buf[bundle["write_mine_idx"]] = kv_score_local[write_src]
    write_buf = get_attn_cp_group().all_reduce(write_buf)
    write_plan_c = paged.plan._replace(
        compress_plan=bundle["empty16"], write_plan=bundle["write_remap"]
    )
    compress_forward(
        kv_score_buffer=state_pool, kv_score_input=write_buf, ape=ape,
        indices=paged.write_loc, plan=write_plan_c, compress_ratio=ratio,
        head_dim=head_dim, extra_data=paged.extra_data,
    )


class Compressor(BaseFusedOp):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        freqs_cis: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        set_weight_attrs(self.ape, {"weight_loader": self.load_ape_weight})
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    def _apply_ape_hotfix(self):
        self.ape_converted = True

        if _is_npu:
            return

        if self.overlap:
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            ape = torch.cat([ape[0], ape[1]], dim=0)
            self.ape.data.copy_(ape.view(self.ratio, -1))

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self._apply_ape_hotfix()

    def load_ape_weight(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param is self.ape
        assert loaded_weight.shape == param.shape
        param.data.copy_(loaded_weight)
        self._apply_ape_hotfix()

    def get_state_pool(self, attn_backend: AttentionBackend) -> CompressStatePool:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    def store_rlc_output(
        self,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        layer_id: int,
        out_loc: torch.Tensor,
        new_compressed_kv: torch.Tensor,
    ) -> None:
        """Write the (already normed/roped) RLC output into the extra-key cache.

        Mirrors the store block of ``forward_core_compressor`` so the RLC path
        lands in the exact same cache layout as the base path.
        """
        if out_loc.shape[0] > new_compressed_kv.shape[0]:
            out_loc = out_loc[: new_compressed_kv.shape[0]]
        if token_to_kv_pool.is_bf16_attention_kv_cache or (
            envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get()
        ):
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
            return
        if (
            _is_hcu
            and _use_dpskv4_lightop_quant_k_cache
            and hasattr(op, "quantize_nope_fp8_rope_bf16_pack_store")
        ):
            token_to_kv_pool.set_extra_key_buffer_lightop_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv.bfloat16(),
            )
            return
        if _is_hcu and _use_dpskv4_lightop_quant_k_cache:
            pack = quant_to_nope_fp8_rope_bf16_pack_lightop(
                new_compressed_kv.bfloat16(), 1e-8
            )
        else:
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(
                new_compressed_kv.bfloat16()
            )
        token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)
    def _pending_key(self):
        return ("kv_score", self.layer_id, self.is_in_indexer)

    def prelaunch_kv_score(self, x: torch.Tensor, forward_batch: ForwardBatch):
        """Compute kv_score and start its CP all-gather, without waiting.

        kv_score only needs `x`, which the attention already has at entry, so the
        gather can be issued before the q/kv projections and collected later in
        compute_kv_score -- that projection work is what hides it. Caller must
        guarantee a matching compute_kv_score in the same op (see
        DeepseekV4Attention._forward_prepare).
        """
        if not _is_hip:
            return
        comm_stream = getattr(forward_batch, "_cp_prefetch_comm_stream", None)
        if comm_stream is None or not dsa_use_prefill_cp(forward_batch):
            return
        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
        # Keyed by forward_batch: each TBO ubatch carries its own, so the two
        # ubatches cannot collect each other's gather.
        pending = forward_batch.__dict__.setdefault("_cp_pending_gathers", {})
        pending[self._pending_key()] = cp_all_gather_rerange_launch(
            kv_score, get_parallel().attn_cp_size, comm_stream, self._pending_key()
        )

    def compute_kv_score(self, x: torch.Tensor, forward_batch: ForwardBatch):
        if _is_hip:
            pending = getattr(forward_batch, "_cp_pending_gathers", None)
            handle = pending.pop(self._pending_key(), None) if pending else None
            if handle is not None:
                return cp_all_gather_rerange_finish(handle)

        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)

        # CUDA path: delegate to backend
        if dsa_use_prefill_cp(forward_batch):
            kv_score = cp_materialize_global_token_order(
                kv_score,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return kv_score

    def forward_native(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: Optional[AttentionBackend] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        kv_score = self.compute_kv_score(x, forward_batch)

        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4AttnBackend)
        kv_score_buffer = self.get_state_pool(attn_backend).kv_score_buffer.kv_score
        return attn_backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )

    def _forward_rlc(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
        paged: FusedCompressMetadata,
    ) -> torch.Tensor:
        """Repartition-Local Compression path (attention c4, prefill-CP, round-robin).

        Round-robin-scattered kv_score is all-to-all'd into per-rank contiguous block-runs (+halo),
        each rank compresses its blocks locally with the real v1 kernel, and an all-gather recovers
        the compact output in global block order. For chunked prefill (prefix>0) each sequence's
        extend-first block reads its overlap from the (replicated) state pool -- patched locally on
        the owning rank -- and the chunk's trailing block is persisted back for the next chunk.
        Produces the same per-token output + state pool as the base path.

        ``paged`` is the v1 ``(write_loc, extra_data, plan)`` metadata for the current batch,
        built and passed by the caller (kept explicit so this method does not depend on the
        backend's paged-metadata storage).
        """
        from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

        cp_group = get_attn_cp_group()
        cp_size = cp_group.world_size
        cp_rank = cp_group.rank_in_group
        R, H = self.ratio, self.head_dim
        last_dim = 2 * (1 + self.overlap) * H
        device = x.device

        extend_lens = list(forward_batch.extend_seq_lens_cpu)
        seq_lens = list(forward_batch.seq_lens_cpu)
        prefix_lens = [int(s) - int(e) for s, e in zip(seq_lens, extend_lens)]

        bundle = _get_rlc_bundle(
            extend_lens, prefix_lens, R, cp_size, cp_rank, device, paged.plan.write_plan
        )
        meta = bundle["meta"]

        state_pool = self.get_state_pool(attn_backend).kv_score_buffer.kv_score.view(-1, R, last_dim)
        core_meta = attn_backend.forward_metadata.core_metadata
        out_loc = core_meta.c4_out_loc if R == 4 else core_meta.c128_out_loc
        num_q_out = out_loc.shape[0]

        kv_score_local = linear_bf16_fp32(x, self.wkv_gate.weight)

        assert kv_score_local.shape[0] * cp_size == num_q_out, (
            f"RLC expects an equal-length round-robin shard: n_local({kv_score_local.shape[0]}) "
            f"* cp({cp_size}) != out_loc({num_q_out})"
        )

        send = (kv_score_local[bundle["send_idx"]].contiguous()
                if bundle["send_idx"].numel() else kv_score_local.new_empty((0, last_dim)))
        recv = kv_score_local.new_empty((sum(meta.out_splits), last_dim))
        attn_cp_all_to_all_single(recv, send, meta.out_splits, meta.in_splits)
        local_kv = recv[bundle["recv_perm"]].contiguous() if bundle["recv_perm"].numel() else recv

        ape = self.ape.view(-1, H)
        if local_kv.shape[0] > 0 and bundle["local_plan_c"] is not None:
            seg_seqs = bundle["seg_seqs"]
            local_sparse = compress_forward(
                kv_score_buffer=state_pool, kv_score_input=local_kv, ape=ape,
                indices=paged.write_loc[seg_seqs], plan=bundle["local_plan_c"],
                compress_ratio=R, head_dim=H, extra_data=paged.extra_data[seg_seqs])
            local_out = local_sparse[bundle["local_ev_kept"]].clone()
        else:
            local_out = kv_score_local.new_empty((0, H))

        padded = local_out.new_zeros((meta.max_c, H))
        if local_out.shape[0] > 0:
            padded[:local_out.shape[0]] = local_out
        gathered = local_out.new_empty((cp_size * meta.max_c, H))
        attn_cp_all_gather_into_tensor(gathered, padded)
        compact = gathered[bundle["compact_idx"]]

        full = kv_score_local.new_zeros((num_q_out, H))
        full[bundle["ev"]] = compact.to(full.dtype)

        compress_fused_norm_rope_inplace(
            full,
            self.norm.weight,
            getattr(self.norm, "eps", self.norm.variance_epsilon),
            self.freqs_cis,
            paged.plan,
        )

        _rlc_write_state(state_pool, kv_score_local, ape, paged, bundle, R, H)

        return rotate_activation(full) if self.rotate else full

    def forward_npu(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: Optional[AttentionBackend] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        if dsa_use_prefill_cp(forward_batch):
            x = cp_materialize_global_token_order(
                x,
                forward_batch,
                torch.cuda.current_stream(),
            )

        return get_attn_backend().forward_compress(self, x, forward_batch)
