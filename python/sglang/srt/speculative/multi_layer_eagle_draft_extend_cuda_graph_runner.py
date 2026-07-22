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

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    LogitsProcessorOutput,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import (
    get_available_gpu_memory,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
        MultiLayerEagleDraftWorker,
    )


logger = logging.getLogger(__name__)


@dataclass
class MultiLayerEagleDraftExtendInputBuffers(ForwardInputBuffers):
    """Persistent buffers shared by all multi-layer EAGLE draft steps."""

    input_ids: torch.Tensor
    out_cache_loc: torch.Tensor
    swa_out_cache_loc: torch.Tensor
    positions: torch.Tensor
    # Shared from parent
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    num_correct_drafts: torch.Tensor
    num_accept_tokens: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_start_loc: torch.Tensor
    select_index: torch.Tensor
    mrope_positions: torch.Tensor
    hidden_states: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class MultiLayerEagleDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker, step: int):
        # Parse args
        self.step = step
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.mtp_model_runner(self.step)
        self.forward_mode = ForwardMode.DRAFT_EXTEND_V2

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.enable_pdmux = model_runner.server_args.enable_pdmux
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.padded_static_len = -1
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        # All MTP steps use the same fixed token window.  Sharing this window
        # lets the worker advance the chain by rotating input ids in place and
        # avoids the old req_to_hidden_states_pool reversion path.
        self.num_tokens_per_bs = self.speculative_num_draft_tokens
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].get_cuda_graph_seq_len_fill_value()

    def init_buffers_and_capture(self, buffers: MultiLayerEagleDraftExtendInputBuffers):
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        self.buffers = buffers

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, forward_batch: ForwardBatch):
        self.graphs[self.bs].replay()

    def capture(self):
        CudaGraphRunner.capture(self)

    def get_forward_batch(self, bs: int) -> ForwardBatch:
        buffers = self.buffers
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        extend_start_loc = buffers.extend_start_loc[:bs]
        num_correct_drafts = buffers.num_correct_drafts[:bs]
        num_accept_tokens = buffers.num_accept_tokens[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_tokens]
        next_token_logits_buffer = buffers.next_token_logits_buffer[
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens
        ]

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        spec_info = EagleDraftExtendInput(
            hidden_states=hidden_states,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
        )
        spec_info.positions = None

        capture_mode = (
            CaptureHiddenMode.NULL
            if self.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_mode,
            attn_backend=self.eagle_worker.draft_extend_attn_backend_list[self.step],
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            padded_static_len=self.padded_static_len,
            # added args
            extend_start_loc=extend_start_loc,
            extend_num_tokens=self.num_tokens_per_bs * bs,
            num_token_non_padded_cpu=self.num_tokens_per_bs * bs,
            return_hidden_states_before_norm=True,
        )
        return forward_batch

    def capture_one_batch_size(self, bs: int, forward: Callable, stream_idx: int = 0):
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream

        self.deepep_adapter.capture(is_extend_in_batch=True)

        num_tokens = bs * self.num_tokens_per_bs
        forward_batch = self.get_forward_batch(bs)

        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                forward_batch.global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            # Backup two fields, which will be modified in-place in `draft_forward`.
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

            # Chain-style MTP writes the step output back to the shared buffer,
            # so the next captured step consumes it directly.
            if (
                self.eagle_worker.chain_mtp_hidden_states
                and ret.hidden_states is not None
            ):
                buffers.hidden_states[:num_tokens].copy_(ret.hidden_states[:num_tokens])

            probs = torch.softmax(
                ret.next_token_logits[buffers.select_index[:bs]], dim=-1
            )
            ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        self._capture_init(run_once)

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def init_replay_state(
        self, forward_batch: ForwardBatch, bs: int, raw_bs: int, num_tokens: int
    ):
        buffers = self.buffers
        # Common inputs
        buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        buffers.positions[:num_tokens].copy_(forward_batch.positions)
        if (
            forward_batch.spec_info.hidden_states.shape[1]
            == buffers.hidden_states.shape[1]
        ):
            buffers.hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.spec_info.num_correct_drafts is not None:
            buffers.num_correct_drafts[:raw_bs].copy_(
                forward_batch.spec_info.num_correct_drafts
            )
            buffers.num_accept_tokens[:raw_bs].copy_(
                forward_batch.spec_info.num_accept_tokens
            )
        arange = torch.arange(bs, device=self.model_runner.device, dtype=torch.int64)
        buffers.select_index[:bs].copy_(
            arange * self.num_tokens_per_bs
            + buffers.num_correct_drafts[:bs].to(torch.int64)
        )
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

    def replay(self, forward_batch: ForwardBatch, init_state: bool = True):
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()
        buffers = self.buffers

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.batch_size
        num_tokens = raw_bs * self.num_tokens_per_bs
        # num_tokens = forward_batch.input_ids.shape[0]
        if self.require_mlp_tp_gather:
            max_batch_size = max(forward_batch.original_global_num_tokens_cpu)
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]

        if init_state:
            self.init_replay_state(forward_batch, bs, raw_bs, num_tokens)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)

        padded_num_tokens = bs * self.num_tokens_per_bs
        forward_batch.spec_info.hidden_states = buffers.hidden_states[
            :padded_num_tokens
        ]
        forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[:bs]
        forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[:bs]
        forward_batch.spec_info.num_tokens_per_req = self.num_tokens_per_bs
        forward_batch.spec_info.num_tokens_for_logprob_per_req = 1
        forward_batch.spec_info.positions = buffers.positions[:padded_num_tokens]
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]

        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum
            + (bs - raw_bs) * self.seq_len_fill_value,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )

        # Replay
        self.raw_bs = raw_bs
        self.bs = bs
        self._replay(forward_batch)
        out = self.output_buffers[bs]

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            # DRAFT_EXTEND_V2: all tokens calculations whether accepted or not.
            unpadding_bs = num_tokens
        elif bs != raw_bs:
            forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[
                :raw_bs
            ]
            forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[
                :raw_bs
            ]
            unpadding_bs = raw_bs
        else:
            unpadding_bs = None

        if unpadding_bs is not None:
            out_copy = out
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:unpadding_bs],
                hidden_states=out.hidden_states[:unpadding_bs],
            )
            out.topk_p = out_copy.topk_p[:raw_bs]
            out.topk_index = out_copy.topk_index[:raw_bs]
        return out


class MultiLayerEagleMultiStepDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        self.eagle_worker = eagle_worker
        self.device = eagle_worker.device
        self.gpu_id = eagle_worker.gpu_id
        self.speculative_num_steps = eagle_worker.speculative_num_steps
        self.draft_extend_attn_backend_list = (
            eagle_worker.draft_extend_attn_backend_list
        )

        self.runners = []
        self.seq_len_fill_value = 1
        self.max_bs = 1
        self.num_tokens_per_bs = 1

        self._init_and_capture()

    def _init_and_capture(self):
        if self.eagle_worker.server_args.disable_cuda_graph:
            self.runners = [None] * self.speculative_num_steps
            return

        self.runners: List[Optional[MultiLayerEagleDraftExtendCudaGraphRunner]] = []

        # 1. Capture loop
        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step]:
                runner = MultiLayerEagleDraftExtendCudaGraphRunner(
                    self.eagle_worker, step
                )
                self.runners.append(runner)
            else:
                self.runners.append(None)

        active_runners = [runner for runner in self.runners if runner is not None]
        if not active_runners:
            raise RuntimeError("No draft-extend attention backend is available")

        runner = active_runners[0]
        shared_signature = (
            tuple(runner.capture_bs),
            runner.max_bs,
            runner.num_tokens_per_bs,
            runner.seq_len_fill_value,
            runner.require_gathered_buffer,
            runner.require_mlp_tp_gather,
            runner.require_attn_tp_gather,
            runner.dp_size,
        )
        for step, candidate in enumerate(self.runners):
            if candidate is None:
                continue
            candidate_signature = (
                tuple(candidate.capture_bs),
                candidate.max_bs,
                candidate.num_tokens_per_bs,
                candidate.seq_len_fill_value,
                candidate.require_gathered_buffer,
                candidate.require_mlp_tp_gather,
                candidate.require_attn_tp_gather,
                candidate.dp_size,
            )
            if candidate_signature != shared_signature:
                raise ValueError(
                    "Multi-layer EAGLE draft steps must use compatible CUDA "
                    f"graph buffers; step {step} differs from step 0"
                )

        self.seq_len_fill_value = runner.seq_len_fill_value
        self.max_bs = runner.max_bs
        self.num_tokens_per_bs = runner.num_tokens_per_bs

        # All steps capture against one fixed-window buffer set.  The worker
        # rotates this buffer between replays instead of copying hidden states
        # through req_to_hidden_states_pool.
        max_num_token = self.max_bs * self.num_tokens_per_bs
        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        with torch.device(self.device):
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            out_cache_loc = torch.ones((max_num_token,), dtype=torch.int64)
            swa_out_cache_loc = torch.ones((max_num_token,), dtype=torch.int64)
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            seq_lens = torch.full(
                (self.max_bs,),
                self.seq_len_fill_value,
                dtype=torch.int32,
            )
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            num_correct_drafts = torch.full((self.max_bs,), 1, dtype=torch.int32)
            num_accept_tokens = torch.full((self.max_bs,), 1, dtype=torch.int32)
            extend_seq_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            extend_start_loc = torch.arange(
                0,
                max_num_token,
                step=self.num_tokens_per_bs,
                dtype=torch.int32,
            )
            select_index = torch.arange(
                self.max_bs, dtype=torch.int64
            ) * self.num_tokens_per_bs + num_correct_drafts.to(torch.int64)
            mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
            hidden_states = torch.zeros(
                (
                    max_num_token,
                    EagleDraftExtendInput.hidden_size_for(self.eagle_worker),
                ),
                dtype=EagleDraftExtendInput.dtype_for(self.eagle_worker),
            )

            if hasattr(runner.model_runner.model_config.hf_config, "draft_vocab_size"):
                vocab_size = runner.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(runner.model_runner.model_config.hf_config, "hot_vocab_size"):
                vocab_size = runner.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = runner.model_runner.model_config.vocab_size
            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size), dtype=torch.float
            )

            if runner.require_gathered_buffer:
                if runner.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (runner.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (runner.dp_size,), dtype=torch.int32
                    )
                else:
                    assert runner.require_attn_tp_gather
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        self.buffers = MultiLayerEagleDraftExtendInputBuffers(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            swa_out_cache_loc=swa_out_cache_loc,
            positions=positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            select_index=select_index,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )

        for step in range(self.speculative_num_steps - 1, -1, -1):
            if self.runners[step] is not None:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph begin (step {step}). This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )

                self.runners[step].init_buffers_and_capture(self.buffers)

                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )

    def reset_buffers(self, forward_batch, batch_result):
        self.buffers.input_ids.zero_()
        self.buffers.seq_lens.fill_(self.seq_len_fill_value)
        self.buffers.out_cache_loc.zero_()
        self.buffers.swa_out_cache_loc.zero_()
        self.buffers.positions.zero_()
        self.buffers.req_pool_indices.zero_()
        self.buffers.num_correct_drafts.fill_(1)
        self.buffers.num_accept_tokens.fill_(1)
        # `batch_result.accept_lens` is drafts + bonus.
        bs = forward_batch.batch_size
        self.buffers.num_correct_drafts[:bs].copy_(batch_result.accept_lens - 1)
        self.buffers.num_accept_tokens[:bs].copy_(batch_result.accept_lens)
        arange = torch.arange(self.max_bs, device=self.device, dtype=torch.int64)
        self.buffers.select_index.copy_(
            arange * self.num_tokens_per_bs
            + self.buffers.num_correct_drafts.to(torch.int64)
        )

    def get_runner(self, step):
        return self.runners[step]

    def get_last_runner(self):
        return self.runners[-1] if self.runners else None

    def can_run(self, forward_batch):
        return self.runners[0].can_run(forward_batch)
