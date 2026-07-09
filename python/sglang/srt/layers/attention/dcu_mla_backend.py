
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sgl_kernel.flash_mla import dcu_create_flashmla_kv_indices
from sglang.srt.utils import get_bool_env_var, direct_register_custom_op

_use_fused_mla_cat = get_bool_env_var("SGLANG_USE_FUSED_MLA_CAT")
_use_fused_rmsnorm_rope = get_bool_env_var("SGLANG_USE_FUSED_RMSNORM_ROPE")
import inspect
import logging
logger = logging.getLogger(__name__)
from sglang.srt.utils import is_dcu
_is_dcu = is_dcu()
is_fp8 = False
try:
    
    if _is_dcu:
        try:
            from flash_mla import (
                flash_mla_with_kvcache,
                flash_mla_with_kvcache_quantization,
                # get_mla_metadata,
                get_mla_decoding_metadata_dense_fp8 as get_mla_metadata,
                flash_mla_with_kvcache_fp8, # only support fp8_e4m3
            )
            is_fp8 = True
        except Exception:
            from flash_mla import (
                flash_mla_with_kvcache,
                flash_mla_with_kvcache_quantization,
                get_mla_metadata,
            )
    else:
        from flash_mla import (
            flash_mla_with_kvcache,
            flash_mla_with_kvcache_quantization,
            get_mla_metadata,
        )
    if _use_fused_mla_cat:
        from flash_mla import (
            flash_mla_with_kvcache_q_nope_pe,
            flash_mla_with_kvcache_quantization_q_nope_pe,
        )
    _has_flash_mla = True
except Exception:  # TODO: need remove
    try:
        from vllm.attention.ops.flashmla import (
            flash_mla_with_kvcache,
            flash_mla_with_kvcache_quantization,
            get_mla_metadata
        )
        _has_flash_mla = False
    except Exception:
        raise ImportError(
            "Can not import FlashMLA。Please perform the following operations to use flashmla:\n"
            "  pip install flash-mla\n"
            "  or\n"
            "  pip install vllm"
        )

PAGE_SIZE = 64 # 强制64

def is_bmz_fp8(kv_cache: torch.Tensor) -> bool:
    if not (kv_cache.is_cuda and kv_cache.dtype == torch.float8_e5m2):
        return False
    try:
        props = torch.cuda.get_device_properties(kv_cache.device.index)
        gcn_arch = getattr(props, "gcnArchName", "")
        if "gfx936" in gcn_arch:
            return True   
    except Exception:
        pass
    return False


if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

@dataclass
class VllmMLADecodeMetadata:
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices

class DCUMLABackend(AttentionBackend):

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if model_runner.server_args.page_size != PAGE_SIZE:
            raise ValueError(
                f"dcu_mla backend requires page_size={PAGE_SIZE}, "
                f"but got the {model_runner.server_args.page_size}"
            )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype

        self.device = model_runner.device
        # self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.k_scale = torch.ones((1), dtype=torch.float32, device=self.device)
        self.q_scale = torch.ones((1), dtype=torch.float32, device=self.device)
        self.max_context_len = model_runner.model_config.context_len
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.forward_metadata: Union[VllmMLADecodeMetadata] = None
        self.metadata_interface_arguments = len(list(inspect.signature(get_mla_metadata).parameters.keys())) #获取get_mla_metadata参数个数
        self.skip_prefill = skip_prefill
        if not skip_prefill:
            from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
            self.flashattn_backend = FlashAttentionBackend(
                model_runner,
                skip_prefill=False,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        use_sglang_create_flashmla_kv_indices_triton = get_bool_env_var("SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON", default="true")
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            # Match forward_decode cache_seqlens (seq_lens + draft when spec enabled).
            if self.num_draft_tokens:
                seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
                seq_lens_for_kv = forward_batch.seq_lens + self.num_draft_tokens
            else:
                seq_lens_cpu = forward_batch.seq_lens_cpu
                seq_lens_for_kv = forward_batch.seq_lens

            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device
            )
            if use_sglang_create_flashmla_kv_indices_triton:
                dcu_create_flashmla_kv_indices(
                    req_to_token_ptr = self.req_to_token.to(torch.int32),
                    req_pool_indices_ptr = forward_batch.req_pool_indices.to(torch.int32),
                    page_kernel_lens_ptr = seq_lens_for_kv.to(torch.int32),
                    kv_start_idx = None,
                    kv_indices_ptr = block_kv_indices.to(torch.int32),
                    req_to_token_ptr_stride = self.req_to_token.stride(0),
                    kv_indices_ptr_stride = max_seqlen_pad,
                )

            else:
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    seq_lens_for_kv,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_seqlen_pad,
                )
            if self.num_draft_tokens:
                if self.metadata_interface_arguments == 4:
                    mla_metadata, num_splits = get_mla_metadata(
                        seq_lens_for_kv.to(torch.int32),
                        self.num_draft_tokens * self.num_q_heads,
                        1,
                        self.num_q_heads,
                    )
                else:
                    mla_metadata, num_splits = get_mla_metadata(
                        seq_lens_for_kv.to(torch.int32),
                        self.num_draft_tokens * self.num_q_heads,
                        1,
                    )
            elif self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    forward_batch.seq_lens.to(torch.int32),
                    self.num_q_heads,
                    1,
                    self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    forward_batch.seq_lens.to(torch.int32),
                    self.num_q_heads,
                    1,
                )
            self.forward_metadata = VllmMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices
            )
        elif forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend_v2()  or forward_batch.forward_mode.is_draft_extend() :
            seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
            seq_lens = forward_batch.seq_lens + self.num_draft_tokens

            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            if use_sglang_create_flashmla_kv_indices_triton:
                dcu_create_flashmla_kv_indices(
                    req_to_token_ptr = self.req_to_token.to(torch.int32),
                    req_pool_indices_ptr = forward_batch.req_pool_indices.to(torch.int32),
                    page_kernel_lens_ptr = seq_lens.to(torch.int32),
                    kv_start_idx = None,
                    kv_indices_ptr = block_kv_indices.to(torch.int32),
                    req_to_token_ptr_stride = self.req_to_token.stride(0),
                    kv_indices_ptr_stride = max_seqlen_pad,
                )

            else:
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    seq_lens,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_seqlen_pad,
                )
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32),
                    self.num_draft_tokens * self.num_q_heads,
                    1,
                    self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32),
                    self.num_draft_tokens * self.num_q_heads,
                    1,
                )
            self.forward_metadata = VllmMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices
            )
        else:
            if not self.skip_prefill:
                # ===  DRAFT_EXTEND_V2  MLA metadata === nhb
                if forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
                    bs = forward_batch.batch_size
                    seq_lens_cpu = forward_batch.seq_lens_cpu
                    seq_lens = forward_batch.seq_lens

                    max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
                    block_kv_indices = torch.full(
                        (bs, max_seqlen_pad),
                        -1,
                        dtype=torch.int32,
                        device=seq_lens.device,
                    )

                    # 调用 Triton kernel 生成 block_kv_indices
                    if use_sglang_create_flashmla_kv_indices_triton:
                        dcu_create_flashmla_kv_indices(
                            req_to_token_ptr = self.req_to_token.to(torch.int32),
                            req_pool_indices_ptr = forward_batch.req_pool_indices.to(torch.int32),
                            page_kernel_lens_ptr = forward_batch.seq_lens.to(torch.int32),
                            kv_start_idx = None,
                            kv_indices_ptr = block_kv_indices.to(torch.int32),
                            req_to_token_ptr_stride = self.req_to_token.stride(0),
                            kv_indices_ptr_stride = max_seqlen_pad,
                        )

                    else:
                        create_flashmla_kv_indices_triton[(bs,)](
                            self.req_to_token,
                            forward_batch.req_pool_indices,
                            forward_batch.seq_lens,
                            None,
                            block_kv_indices,
                            self.req_to_token.stride(0),
                            max_seqlen_pad,
                        )

                    #  MLA
                    if self.metadata_interface_arguments == 4:
                        mla_metadata, num_splits = get_mla_metadata(
                            seq_lens.to(torch.int32),
                            self.num_q_heads,
                            1,
                            self.num_q_heads,
                        )
                    else:
                        mla_metadata, num_splits = get_mla_metadata(
                            seq_lens.to(torch.int32),
                            self.num_q_heads,
                            1,
                        )

                    # save forward_metadata
                    self.forward_metadata = VllmMLADecodeMetadata(
                        mla_metadata,
                        num_splits,
                        block_kv_indices,
                    )

                self.flashattn_backend.init_forward_metadata(forward_batch)


    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        if self.num_draft_tokens:
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    torch.ones(max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device),
                    self.num_draft_tokens * self.num_q_heads,
                    1,
                    self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    torch.ones(max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device),
                    self.num_draft_tokens * self.num_q_heads,
                    1,
                )
        else:
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    torch.ones(max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device),
                    self.num_q_heads,
                    1,
                    self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    torch.ones(max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device),
                    self.num_q_heads,
                    1,
                )

        self.cuda_graph_mla_metadata = mla_metadata
        self.cuda_graph_num_splits = num_splits
        self.cuda_graph_kv_indices = cuda_graph_kv_indices

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["SpecInput"],
    ):
        use_sglang_create_flashmla_kv_indices_triton = get_bool_env_var("SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON", default="true")
        if forward_mode.is_decode_or_idle():
            seq_lens_for_kv = (
                seq_lens + self.num_draft_tokens if self.num_draft_tokens else seq_lens
            )
            max_seqlen_pad = triton.cdiv(seq_lens_for_kv.max().item(), PAGE_SIZE)
            if use_sglang_create_flashmla_kv_indices_triton:
                        dcu_create_flashmla_kv_indices(
                            req_to_token_ptr = self.req_to_token.to(torch.int32),
                            req_pool_indices_ptr = req_pool_indices.to(torch.int32),
                            page_kernel_lens_ptr = seq_lens_for_kv.to(torch.int32),
                            kv_start_idx = None,
                            kv_indices_ptr = self.cuda_graph_kv_indices.to(torch.int32),
                            req_to_token_ptr_stride =  self.req_to_token.stride(0),
                            kv_indices_ptr_stride = self.cuda_graph_kv_indices.stride(0),
                        )

            else:
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens_for_kv,
                    None,
                    self.cuda_graph_kv_indices,
                    self.req_to_token.stride(0),
                    self.cuda_graph_kv_indices.stride(0),
                )
            num_q_heads = self.num_q_heads * (self.num_draft_tokens or 1)
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens_for_kv.to(torch.int32), num_q_heads, 1, self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens_for_kv.to(torch.int32), num_q_heads, 1,
                )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata = VllmMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()  or forward_mode.is_draft_extend():
            seq_lens = seq_lens + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            if use_sglang_create_flashmla_kv_indices_triton:
                dcu_create_flashmla_kv_indices(
                    req_to_token_ptr = self.req_to_token.to(torch.int32),
                    req_pool_indices_ptr = req_pool_indices.to(torch.int32),
                    page_kernel_lens_ptr = seq_lens.to(torch.int32),
                    kv_start_idx = None,
                    kv_indices_ptr = self.cuda_graph_kv_indices.to(torch.int32),
                    req_to_token_ptr_stride =  self.req_to_token.stride(0),
                    kv_indices_ptr_stride = self.cuda_graph_kv_indices.stride(0),
                )

            else:
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    None,
                    self.cuda_graph_kv_indices,
                    self.req_to_token.stride(0),
                    self.cuda_graph_kv_indices.stride(0),
            )
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32), self.num_draft_tokens * self.num_q_heads, 1, self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32), self.num_draft_tokens * self.num_q_heads, 1,
                )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata = VllmMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
        else:
            if not self.skip_prefill:
                self.flashattn_backend.init_forward_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        use_sglang_create_flashmla_kv_indices_triton = get_bool_env_var("SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON", default="true")
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs]
            if self.num_draft_tokens:
                seq_lens = seq_lens + self.num_draft_tokens
                seq_lens_cpu = seq_lens_cpu + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            if use_sglang_create_flashmla_kv_indices_triton:
                        dcu_create_flashmla_kv_indices(
                            req_to_token_ptr = self.req_to_token.to(torch.int32),
                            req_pool_indices_ptr = req_pool_indices[:bs].to(torch.int32),
                            page_kernel_lens_ptr = seq_lens.to(torch.int32),
                            kv_start_idx = None,
                            kv_indices_ptr = self.cuda_graph_kv_indices.to(torch.int32),
                            req_to_token_ptr_stride =  self.req_to_token.stride(0),
                            kv_indices_ptr_stride = self.cuda_graph_kv_indices.stride(0),
                        )

            else:
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens,
                    None,
                    self.cuda_graph_kv_indices,
                    self.req_to_token.stride(0),
                    self.cuda_graph_kv_indices.stride(0),
            )

            num_q_heads = self.num_q_heads * (self.num_draft_tokens or 1)
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32), num_q_heads, 1, self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32), num_q_heads, 1,
                )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            seq_lens_cpu = seq_lens_cpu[:bs] + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            if use_sglang_create_flashmla_kv_indices_triton:
                        dcu_create_flashmla_kv_indices(
                            req_to_token_ptr = self.req_to_token.to(torch.int32),
                            req_pool_indices_ptr = req_pool_indices[:bs].to(torch.int32),
                            page_kernel_lens_ptr = seq_lens.to(torch.int32),
                            kv_start_idx = None,
                            kv_indices_ptr = self.cuda_graph_kv_indices.to(torch.int32),
                            req_to_token_ptr_stride =  self.req_to_token.stride(0),
                            kv_indices_ptr_stride = self.cuda_graph_kv_indices.stride(0),
                        )

            else:
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens,
                    None,
                    self.cuda_graph_kv_indices,
                    self.req_to_token.stride(0),
                    self.cuda_graph_kv_indices.stride(0),
            )
            if self.metadata_interface_arguments == 4:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32), self.num_draft_tokens * self.num_q_heads, 1, self.num_q_heads,
                )
            else:
                mla_metadata, num_splits = get_mla_metadata(
                    seq_lens.to(torch.int32), self.num_draft_tokens * self.num_q_heads, 1,
                )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
        else:
            if not self.skip_prefill:
                self.flashattn_backend.init_forward_metadata_replay_cuda_graph(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    seq_lens_sum,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                    seq_lens_cpu,
                )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


    @torch._dynamo.disable()  # TODO: register custom op
    def _call_decode(self, q: torch.Tensor,
                     k_cache: torch.Tensor,
                     cache_seqlens: torch.Tensor,
                     bs: int,
                     layer: "RadixAttention",
                     q_rope: Optional[torch.Tensor] = None):
        k_cache_reshaped = k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim)
        block_table = self.forward_metadata.block_kv_indices[:bs]
        scaling = layer.scaling
        if q_rope is not None:  # mla
             # fix_with_mtp
            reshaped_q_nope = q.view(bs, -1, layer.tp_q_head_num, self.kv_lora_rank)
            reshaped_q_rope = q_rope.view(bs, -1, layer.tp_q_head_num, self.qk_rope_head_dim)
            o, _ = flash_mla_with_kvcache_q_nope_pe(
                q_nope=reshaped_q_nope,
                q_pe=reshaped_q_rope,
                k_cache=k_cache_reshaped,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=scaling,
                causal=True,
            )
        else:
            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            o, _ = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache_reshaped,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=scaling,
                causal=True,
            )
        return o

    @torch._dynamo.disable()  # TODO: register custom op
    def _call_fp8_decode(self, q: torch.Tensor,
                           k_cache: torch.Tensor,
                           cache_seqlens: torch.Tensor,
                           bs: int,
                           layer: "RadixAttention",
                           q_rope: Optional[torch.Tensor] = None,
                           k_scale=None, kv_cache_dtype=None):
        assert _has_flash_mla, "FP8 KV cache 需要flash_mla包"
        # reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)  # head_dim = 512 + rope_dim
        k_cache_reshaped = k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim)
        block_table = self.forward_metadata.block_kv_indices[:bs]
        scaling = layer.scaling
        if q_rope is not None:
            # fix_with_mtp
            reshaped_q_nope = q.view(bs, -1, layer.tp_q_head_num, self.kv_lora_rank)
            reshaped_q_rope = q_rope.view(bs, -1, layer.tp_q_head_num, self.qk_rope_head_dim)
            o, _ = flash_mla_with_kvcache_quantization_q_nope_pe(
                q_nope=reshaped_q_nope,
                q_pe=reshaped_q_rope,
                k_cache=k_cache_reshaped,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=scaling,
                causal=True,
                k_scale=k_scale,
                kv_cache_dtype=kv_cache_dtype,
            )
        else:
            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            if is_fp8 and not is_bmz_fp8(k_cache):
                reshape_q = reshape_q.to(k_cache_reshaped.dtype)
                o, _ = flash_mla_with_kvcache_fp8(
                    q=reshape_q,
                    k_cache=k_cache_reshaped,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=scaling,
                    causal=True,
                    descale_k=self.k_scale,
                    descale_q=self.q_scale,# 不能传torch.tensor(1.0)，只能传torch.ones((1))，而且不能和descale_k使用同一个张量
                )
            else:
                o, _ = flash_mla_with_kvcache_quantization(
                    q=reshape_q,
                    k_cache=k_cache_reshaped,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=scaling,
                    causal=True,
                    k_scale=k_scale,
                    kv_cache_dtype=kv_cache_dtype,
                )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            if k_rope is not None:  # cat in save kv cache; when enable fused rmsnorm_rope, skip this cat
                if save_kv_cache and not _use_fused_rmsnorm_rope:
                    get_token_to_kv_pool().set_kv_buffer_opt(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
            else:
                assert v is not None
                if save_kv_cache and not _use_fused_rmsnorm_rope:
                    get_token_to_kv_pool().set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        bs = forward_batch.batch_size
        k_cache = get_token_to_kv_pool().get_key_buffer(layer.layer_id)
        num_draft_tokens = self.num_draft_tokens if self.num_draft_tokens is not None else 0
        if num_draft_tokens == 0:
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)
        else:
            cache_seqlens = (forward_batch.seq_lens + num_draft_tokens).to(torch.int32)

        if self.data_type in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                              torch.float8_e5m2, torch.float8_e5m2fnuz):
            if self.data_type in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
                kv_cache_dtype="fp8_e4m3"
            else:
                kv_cache_dtype="fp8_e5m2"
            k_scale = layer.k_scale if layer.k_scale is not None else self.k_scale
            o = self._call_fp8_decode(
                q,
                k_cache,
                cache_seqlens,
                bs=bs,
                layer=layer,
                k_scale=k_scale,
                q_rope=q_rope,
                kv_cache_dtype=kv_cache_dtype,
            )
        else:
            o = self._call_decode(
                q,
                k_cache,
                cache_seqlens,
                bs=bs,
                layer=layer,
                q_rope=q_rope,
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if ((
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            )
        ):
            if not self.skip_prefill:
                return self.flashattn_backend.forward_extend(
                            q, k, v, layer, forward_batch, save_kv_cache,
                        )
            else:
                raise RuntimeError("skip prefill but use forward_extend")

        cache_loc = forward_batch.out_cache_loc
        if k is not None:
            if k_rope is not None:  # mla maybe better
                if save_kv_cache and not _use_fused_rmsnorm_rope:  # TODO: handwrite kernel, maybe triton is enough
                    get_token_to_kv_pool().set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
            else:
                assert v is not None
                if save_kv_cache and not _use_fused_rmsnorm_rope:
                    get_token_to_kv_pool().set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        bs = forward_batch.batch_size
        k_cache = get_token_to_kv_pool().get_key_buffer(layer.layer_id)
        # 入图+去除非mtp的冗余操作
        num_draft_tokens = self.num_draft_tokens if self.num_draft_tokens is not None else 0
        if num_draft_tokens == 0:
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)
        else:
            cache_seqlens = (forward_batch.seq_lens + num_draft_tokens).to(torch.int32)
        if self.data_type in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                              torch.float8_e5m2, torch.float8_e5m2fnuz):
            if self.data_type in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
                kv_cache_dtype="fp8_e4m3"
            else:
                kv_cache_dtype="fp8_e5m2"
            k_scale = layer.k_scale if layer.k_scale is not None else self.k_scale
            o = self._call_fp8_decode(
                q,
                k_cache,
                cache_seqlens,
                bs=bs,
                layer=layer,
                k_scale=k_scale,
                kv_cache_dtype=kv_cache_dtype,
                q_rope=q_rope,
            )
        else:
            o = self._call_decode(
                q,
                k_cache,
                cache_seqlens,
                bs=bs,
                layer=layer,
                q_rope=q_rope,
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

class DCUMLAMultiStepDraftBackend:
    """
    Wrap multiple flashmla attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        if topk > 1:
            raise ValueError(
                "Currently FlashMLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                DCUMLABackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=None,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)
