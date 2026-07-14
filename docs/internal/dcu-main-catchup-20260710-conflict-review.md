# Official Main Catch-up 20260710 — Code Conflict Review

> Scope: only the 22 files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `68d965671265f5d3859ba767cc3bd4e94cc03dce`
- Common official base: `bd7e54d7379e437cf5f027382d6ca214e046626b`
- Official endpoint (`theirs`): `e1d51be91f6be39e585756568a8f66b99ac2c512`
- Resolved merge: `18d1216680858500bd12d12a739059a24037f026`
- Reconstructed textual conflicts: 22 files, 37 hunks

Each section compares the saved three-way auto-conflict text with the committed merge resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>python/pyproject.toml</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept official FlashInfer 0.6.14 and repair the adjacent smg/soundfile dependency split.

~~~~diff
--- AUTO-CONFLICT/python/pyproject.toml
+++ RESOLVED/python/pyproject.toml
@@ -26,23 +26,18 @@
   "cuda-python>=13.0",
   "datasets",
   "decord2 ; sys_platform == 'linux' and (platform_machine == 'aarch64' or platform_machine == 'arm64' or platform_machine == 'armv7l')",
   "distro",
   "easydict",  # Required by remote model code (e.g. DeepSeek-OCR) loaded via trust_remote_code; validated by transformers 5.4+ check_imports
   "einops",
   "fastapi",
   "flash-attn-4==4.0.0b15",
-<<<<<<< HEAD
-  "flashinfer_cubin==0.6.12",
-  "flashinfer_python[cu13]==0.6.12", # keep it aligned with jit-cache version in Dockerfile  "gguf",
-=======
   "flashinfer_python[cu13]==0.6.14", # keep it aligned with jit-cache version in Dockerfile
   "gguf",
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
   "interegular",
   "IPython",
   "kernels>=0.14.1,<0.15",
   "llguidance>=0.7.11,<0.8.0",
   "mistral_common>=1.11.5",
   "modelscope",
   "msgspec",
   "ninja",
@@ -66,17 +61,18 @@
   "pyzmq>=25.1.2",
   "quack-kernels>=0.4.1",
   "requests",
   "scipy",
   "sentencepiece",
   "setproctitle",
   "sgl-deep-gemm==0.1.4.post1",
   "sglang-kernel==0.4.4",
-  "smg-grpc-servicer>=0.5.0",  "soundfile==0.13.1",
+  "smg-grpc-servicer>=0.5.0",
+  "soundfile==0.13.1",
   "tiktoken",
   "tilelang==0.1.11",
   "timm==1.0.16",
   "tokenspeed_mla==0.1.7",
   "torch==2.11.0",
   "torch_memory_saver>=0.0.9.post1",
   "torchao==0.17.0",
   "torchaudio==2.11.0",
~~~~

</details>

<details>
<summary><code>python/sglang/srt/batch_overlap/two_batch_overlap.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt runtime ServerArgs access while retaining DCU pinned host buffers.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/batch_overlap/two_batch_overlap.py
+++ RESOLVED/python/sglang/srt/batch_overlap/two_batch_overlap.py
@@ -828,23 +828,18 @@

     @classmethod
     def compute_tbo_children_num_token_non_padded_raw(
         cls, tbo_split_token_index: int, num_token_non_padded: int
     ):
         # TODO we may make padding on both sub-batches to make it slightly more balanced
         value_a = min(tbo_split_token_index, num_token_non_padded)
         value_b = max(0, num_token_non_padded - tbo_split_token_index)
-<<<<<<< HEAD
         return torch.tensor([value_a, value_b], dtype=torch.int32).pin_memory().to(
-            device=get_global_server_args().device, non_blocking=True
-=======
-        return torch.tensor([value_a, value_b], dtype=torch.int32).to(
             device=get_server_args().device, non_blocking=True
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
         )

     @classmethod
     def _compute_split_token_index(cls, batch: ForwardBatch):
         token_num_per_seq = get_token_num_per_seq(
             forward_mode=batch.forward_mode, spec_info=batch.spec_info
         )
         return compute_split_token_index(
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/attention/dsv4/indexer.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Absorb official FP8-FNUZ handling while retaining the DCU LightOp and gfx paths.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsv4/indexer.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsv4/indexer.py
@@ -45,30 +45,20 @@
     from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
     from sglang.srt.layers.attention.dsv4.compressor import (
         CompressorBackendMixin,
     )
     from sglang.srt.layers.quantization import QuantizationConfig
     from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
     from sglang.srt.model_executor.forward_batch_info import ForwardBatch

-<<<<<<< HEAD
 _is_dcu = is_dcu()
 _is_aiter_fp8_paged_mqa_logits_supported = is_gfx942_supported() or is_gfx95_supported()
-if is_hip():
-    FP8_DTYPE = torch.float8_e4m3fnuz
-    FP8_MAX = torch.finfo(FP8_DTYPE).max
-else:
-    FP8_DTYPE = torch.float8_e4m3fn
-    FP8_MAX = torch.finfo(FP8_DTYPE).max
-=======
-
 FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
-
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
+FP8_MAX = torch.finfo(FP8_DTYPE).max

 IndexerQuery: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


 _arange_cache = {}


 def fp8_paged_mqa_logits_torch(
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/attention/triton_backend.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Move imports to sglang.kernels while retaining the optional DCU AITER extend path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/triton_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/triton_backend.py
@@ -119,33 +119,29 @@
         # Lazy import to avoid the initialization of cuda context
         from sglang.kernels.ops.attention.decode_attention import (
             decode_attention_fwd,
         )
         from sglang.kernels.ops.attention.extend_attention import (
             build_unified_kv_indices,
             extend_attention_fwd_unified,
         )
-<<<<<<< HEAD
         self.use_aiter_triton_extend_fwd = (
             os.getenv("SGLANG_USE_TRITON_EXTEND_FROM_AITER", "0") == "1"
         )
         if self.use_aiter_triton_extend_fwd:
             try:
                 from aiter.ops.triton.extend_attention import extend_attention_fwd
             except ImportError:
                 self.use_aiter_triton_extend_fwd = False
         if not self.use_aiter_triton_extend_fwd:
-            from sglang.srt.layers.attention.triton_ops.extend_attention import (
+            from sglang.kernels.ops.attention.extend_attention import (
                 extend_attention_fwd,
             )
-        from sglang.srt.layers.attention.triton_ops.verify_splitkv import (
-=======
         from sglang.kernels.ops.attention.verify_splitkv import (
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
             verify_splitkv_fwd,
         )

         super().__init__()

         self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
         self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
         self.extend_attention_fwd_unified = torch.compiler.disable(
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/linear.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt official runtime flags and parallel groups while preserving DCU fused SiLU and FP8 tuple paths.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/linear.py
+++ RESOLVED/python/sglang/srt/layers/linear.py
@@ -11,17 +11,16 @@
 import torch
 from torch import nn
 from torch.nn.parameter import Parameter, UninitializedParameter

 from sglang.kernel_api_logging import wrap_method_with_debug_kernel_once
 from sglang.srt.distributed import (
     divide,
     get_tp_group,
-    parallel_state,
     split_tensor_along_last_dim,
     tensor_model_parallel_all_gather,
     tensor_model_parallel_all_reduce,
     tensor_model_parallel_quant_all_reduce,
 )
 from sglang.srt.distributed.device_communicators.pynccl_allocator import (
     use_symmetric_memory,
 )
@@ -1620,62 +1619,73 @@
                     loaded_weight,
                     tp_rank=self.tp_rank,
                     use_presharded_weights=self.use_presharded_weights,
                 )
             except TypeError:
                 # Fallback for parameters that don't accept additional args
                 param.load_row_parallel_weight(loaded_weight)

-    def forward(self, input_, skip_all_reduce=False, forward_batch=None, use_fused_silu_mul_quant: Optional[bool] = False, use_fused_silu_mul_fp8_quant: Optional[bool] = False):
+    def forward(
+        self,
+        input_,
+        skip_all_reduce=False,
+        forward_batch=None,
+        use_fused_silu_mul_quant: Optional[bool] = False,
+        use_fused_silu_mul_fp8_quant: Optional[bool] = False,
+    ):
         if self.input_is_parallel:
             input_parallel = input_
         else:
             splitted_input = split_tensor_along_last_dim(
                 input_, num_partitions=self.tp_size
             )
             input_parallel = splitted_input[self.tp_rank].contiguous()

         # Matrix multiply.
         assert self.quant_method is not None
         # Only fuse bias add into GEMM for rank 0 (this ensures that
         # bias will not get added more than once in TP>1 case)
         bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
-<<<<<<< HEAD
+        if self.use_dp_attention_reduce:
+            symm_ctx = use_symmetric_memory(get_parallel().attn_tp_group)
+        else:
+            symm_ctx = use_symmetric_memory(
+                get_tp_group(), disabled=not is_allocation_symmetric()
+            )
+
         if use_fused_silu_mul_quant:
             xq, xs = lm_fuse_silu_mul_quant(input_parallel)
             silu_quant_args = [xq, xs]
-            with use_symmetric_memory(get_tp_group()) as sm:
-                output_parallel = self.quant_method.apply(self, input_parallel,
-                                                          bias=bias_,
-                                                          silu_quant_args=silu_quant_args
+            with symm_ctx as sm:
+                output_parallel = self.quant_method.apply(
+                    self,
+                    input_parallel,
+                    bias=bias_,
+                    silu_quant_args=silu_quant_args,
                 )
                 if sm is not None:
                     sm.tag(output_parallel)
         elif use_fused_silu_mul_fp8_quant:
             output_shape = [*input_.shape[:-1], self.weight.shape[1]]
             input_x, x_scale = fuse_silu_mul_fp8_quant(input_parallel, fp8type=0)

-            with use_symmetric_memory(get_tp_group()) as sm:
-                output = torch.empty(output_shape, device=input_.device, dtype=input_.dtype)
-                deepgemm.fp8_gemm((input_x, x_scale),(self.weight, self.weight_scale),output)
+            with symm_ctx as sm:
+                output = torch.empty(
+                    output_shape, device=input_.device, dtype=input_.dtype
+                )
+                deepgemm.fp8_gemm(
+                    (input_x, x_scale),
+                    (self.weight, self.weight_scale),
+                    output,
+                )
                 output_parallel = output.view(*output_shape)
                 if sm is not None:
                     sm.tag(output_parallel)
-=======
-        if self.use_dp_attention_reduce:
-            symm_ctx = use_symmetric_memory(get_parallel().attn_tp_group)
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
         else:
-            if self.use_dp_attention_reduce:
-                symm_ctx = use_symmetric_memory(get_attention_tp_group())
-            else:
-                symm_ctx = use_symmetric_memory(
-                    get_tp_group(), disabled=not is_allocation_symmetric()
-                )
             with symm_ctx:
                 output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)

         # skip_all_reduce: explicit call-site override. Also honor
         # ForwardFlags (fuse_mlp_allreduce / mlp_reduce_scatter) published by
         # the decoder — callers should not thread those flags into modules.
         if (
             self.reduce_results
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/moe/ep_moe/kernels.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept the official expert-quant block-size API.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/ep_moe/kernels.py
+++ RESOLVED/python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -960,18 +960,16 @@
 def _fwd_kernel_ep_scatter_1_use_groupgemm(
     num_recv_tokens_per_expert,
     expert_start_loc,
     m_indices,
     num_experts: tl.constexpr,
     BLOCK_E: tl.constexpr,
     BLOCK_EXPERT_NUM: tl.constexpr,
 ):
-    cur_expert = tl.program_id(0)
-
     offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
     tokens_per_expert = tl.load(
         num_recv_tokens_per_expert + offset_cumsum,
         mask=offset_cumsum < num_experts,
         other=0,
     )
     cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
     tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
@@ -1212,32 +1210,27 @@
     expert_start_loc: torch.Tensor,
     output_tensor: torch.Tensor,
     output_tensor_scale: torch.Tensor,
     m_indices: torch.Tensor,
     output_index: torch.Tensor,
     scale_ue8m0: bool = False,
     quant_block_size: int = 128,
 ):
-<<<<<<< HEAD
-=======
-    BLOCK_E = 128  # token num of per expert is aligned to 128
-    BLOCK_D = quant_block_size  # block size of quantization
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
     num_warps = 8
     num_experts = num_recv_tokens_per_expert.shape[0]
     hidden_size = recv_x.shape[1]
     # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
     grid = num_experts
     if use_groupgemm:
         BLOCK_E = 256
         scale_hidden_size = recv_x_scale.shape[-1]
     else:
         BLOCK_E = 128  # token num of per expert is aligned to 128
-        BLOCK_D = 128  # block size of quantization
+        BLOCK_D = quant_block_size  # block size of quantization
         scale_hidden_size = hidden_size // BLOCK_D
     if scale_ue8m0:
         # ue8m0 scales are packed here (4 scales per int32),
         # hence the effective size of this dimension is divided by 4.
         scale_hidden_size = ceil_div(scale_hidden_size, 4)

     assert m_indices.shape[0] % BLOCK_E == 0
     is_fp8 = recv_x_scale is not None and recv_x.dtype != torch.bfloat16
@@ -1740,17 +1733,17 @@
         (num_local_experts, m_max, hidden_states.size(1)),
         device=hidden_states.device,
         dtype=output_dtype,
     )

     if block_shape is None:
         block_shape = [128, 128]
     assert len(block_shape) == 2
-    block_n, block_k = block_shape[0], block_shape[1]
+    block_k = block_shape[1]
     is_fp8 = output_dtype == torch.float8_e4m3fn
     if is_fp8 and use_mxfp8:
         from sglang.jit_kernel.minimax_quant_ue8m0 import (
             per_token_quant_fp8_ue8m0_scatter,
         )

         num_groups = hidden_states.size(1) // block_k
         gateup_input_scale = torch.empty(
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/moe/fused_moe_triton/layer.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Combine official environment, TBO, and NPU plumbing with retained DCU LightOp flags.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
+++ RESOLVED/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -62,43 +62,33 @@
 from sglang.srt.layers.quantization.fp8_utils import quantize_block_fp8_weight_to_mxfp4
 from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
 from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
 from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
     get_tc_piecewise_forward_context,
     is_in_tc_piecewise_cuda_graph,
 )
 from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
-<<<<<<< HEAD
-from sglang.srt.runtime_context import get_parallel
-from sglang.srt.server_args import get_global_server_args
-from sglang.srt.environ import envs
-from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
-=======
 from sglang.srt.runtime_context import get_parallel, get_server_args
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
 from sglang.srt.utils import (
     cpu_has_amx_support,
     get_bool_env_var,
     is_cpu,
     is_hip,
     is_npu,
     print_info_once,
     round_up,
 )
 from sglang.srt.utils.custom_op import register_custom_op

 _is_hip = is_hip()
 _is_cpu_amx_available = cpu_has_amx_support()
 _is_cpu = is_cpu()
-<<<<<<< HEAD
 _use_lightop_moe_sum_mul_add = get_bool_env_var("SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD")
-=======
 _is_npu = is_npu()
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
 _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


 def _get_deepep_comm_group(a2a_backend):
     group = get_tp_group().device_group

     if a2a_backend.is_mori():
         group = get_tp_group()
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Follow the official kernel namespace move without changing DCU dispatch semantics.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py
+++ RESOLVED/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py
@@ -12,24 +12,20 @@

 _is_cuda = is_cuda()
 _is_hip = is_hip()
 _is_dcu = is_dcu()
 _is_xpu = is_xpu()
 _is_musa = is_musa()

 if _is_cuda or _is_hip or _is_xpu or _is_musa:
-<<<<<<< HEAD
-    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
+    from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
+
 if _is_dcu:
-    from lightop import op as op
-=======
-    from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
-
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
+    from lightop import op

 def moe_align_block_size(
     topk_ids: torch.Tensor, block_size: int, num_experts: int
 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     """
     Aligns the token distribution across experts to be compatible with block
     size for matrix multiplication.

~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/moe/topk.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Follow the official kernel namespace move and retain DCU LightOp grouped-top-k priority.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/topk.py
+++ RESOLVED/python/sglang/srt/layers/moe/topk.py
@@ -943,22 +943,19 @@
     gating_output: torch.Tensor,
     topk: int,
     renormalize: bool,
     num_expert_group: Optional[int] = None,
     topk_group: Optional[int] = None,
     num_fused_shared_experts: int = 0,
     routed_scaling_factor: Optional[float] = None,
     apply_routed_scaling_factor_on_output: Optional[bool] = False,
-<<<<<<< HEAD
     num_token_non_padded: Optional[torch.Tensor] = None,
     expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
-=======
     scoring_func: str = "softmax",
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
 ):
     assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

     if scoring_func == "softmax":
         scores = torch.softmax(gating_output, dim=-1)
     elif scoring_func == "sigmoid":
         scores = gating_output.sigmoid()
     else:
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/quantization/fp8_utils.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept official runtime-context and MXFP8 updates while retaining DCU DeepGEMM prequantized tuples.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/fp8_utils.py
+++ RESOLVED/python/sglang/srt/layers/quantization/fp8_utils.py
@@ -28,17 +28,16 @@
     pack_mxfp8_scales_triton,
     per_token_group_quant_fp8,
     scaled_fp8_quant,
     sglang_per_token_quant_fp8,
     static_quant_fp8,
     triton_scaled_mm,
     w8a8_block_fp8_matmul_deepgemm,
     w8a8_block_fp8_matmul_triton,
-    vllm_triton_scaled_mm_fp8,
 )
 from sglang.srt.runtime_context import get_server_args
 from sglang.srt.utils import (
     ceil_align,
     ceil_div,
     get_bool_env_var,
     get_cuda_version,
     get_device_capability,
@@ -1824,87 +1823,40 @@
     # for matrices with batch dimension > 16.
     # This could change in the future.
     # We also don't pad when using torch.compile,
     # as it breaks with dynamic shapes.
     if pad_output is None:
         pad_output = not cutlass_fp8_supported and not _sglang_enable_torch_compile
     output_padding = 17 if pad_output else None

-    if not type(input) == tuple:
+    if not isinstance(input, tuple):
         # View input as 2D matrix for fp8 methods
         input_2d = input.view(-1, input.shape[-1])
         output_shape = [*input.shape[:-1], weight.shape[1]]

-<<<<<<< HEAD
         if compressed_tensor_quant:
             # Maybe apply padding to output, see comment in __init__
             num_token_padding = output_padding
             if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
                 num_token_padding = None
             # Let inductor fuse static per-tensor activation quantization with
             # surrounding ops. Eager and decode keep using the custom kernel.
             if (
                 input_scale is not None
                 and input_scale.numel() == 1
-                and get_global_server_args().cuda_graph_config.prefill.tc_compiler
+                and get_server_args().cuda_graph_config.prefill.tc_compiler
                 == "inductor"
             ):
                 qinput = (
                     (input_2d * input_scale.reciprocal())
                     .clamp(min=fp8_min, max=fp8_max)
                     .to(fp8_dtype)
                 )
                 x_scale = input_scale
-=======
-    if compressed_tensor_quant:
-        # Maybe apply padding to output, see comment in __init__
-        num_token_padding = output_padding
-        if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
-            num_token_padding = None
-        # For static per-tensor activation scales when using inductor compiler,
-        # use pure PyTorch ops instead of the opaque sgl_kernel quant kernel.
-        # Inductor fuses these with surrounding ops (RMSNorm, residual add),
-        # eliminating a separate kernel launch per linear layer.
-        # weight_scale shape does not matter here -- it is only used in the
-        # GEMM epilogue, not in the activation quant fusion. Only activates when
-        # cuda_graph_config[prefill].tc_compiler=inductor; eager PCG and
-        # decode both use the faster custom kernel.
-
-        if (
-            input_scale is not None
-            and input_scale.numel() == 1
-            and get_server_args().cuda_graph_config.prefill.tc_compiler == "inductor"
-        ):
-            qinput = (
-                (input_2d * input_scale.reciprocal())
-                .clamp(min=fp8_min, max=fp8_max)
-                .to(fp8_dtype)
-            )
-            x_scale = input_scale
-        else:
-            qinput, x_scale = scaled_fp8_quant(
-                input_2d,
-                input_scale,
-                num_token_padding=num_token_padding,
-                use_per_token_if_dynamic=use_per_token_if_dynamic,
-            )
-    else:
-        # cutlass w8a8 fp8 sgl-kernel only supports per-token scale
-        if input_scale is not None:
-            assert input_scale.numel() == 1
-            # broadcast per-tensor scale to per-token scale when supporting cutlass
-            qinput, x_scale = static_quant_fp8(
-                input_2d, input_scale, repeat_scale=cutlass_fp8_supported
-            )
-        else:
-            # default use per-token quantization if dynamic
-            if _is_cuda:
-                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
             else:
                 qinput, x_scale = scaled_fp8_quant(
                     input_2d,
                     input_scale,
                     num_token_padding=num_token_padding,
                     use_per_token_if_dynamic=use_per_token_if_dynamic,
                 )
         else:
@@ -1935,17 +1887,17 @@
                         )

         if _is_dcu:
             output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
             deepgemm.fp8_gemm((qinput,x_scale),(weight,weight_scale),output)

             return output.view(*output_shape)

-    if _is_dcu and type(input) == tuple:
+    if _is_dcu and isinstance(input, tuple):
         output_shape = [*input[0].shape[:-1], weight.shape[1]]
         output = torch.empty(output_shape, device=input[0].device, dtype=torch.bfloat16)
         deepgemm.fp8_gemm((input[0],input[1]),(weight,weight_scale),output)

         return output.view(*output_shape)

     if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
         cutlass_compatible_b = weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
~~~~

</details>

<details>
<summary><code>python/sglang/srt/managers/overlap_utils.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept official relay/runtime-context types while retaining the DCU-compatible pinned result path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/managers/overlap_utils.py
+++ RESOLVED/python/sglang/srt/managers/overlap_utils.py
@@ -2,22 +2,17 @@

 from dataclasses import dataclass
 from typing import TYPE_CHECKING, Optional, Sequence

 import torch

 from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
 from sglang.srt.environ import envs
-<<<<<<< HEAD
-from sglang.srt.speculative.triton_ops.gather_spec_extras import gather_spec_extras
 from sglang.srt.utils import is_cuda, is_dcu, is_hip, is_npu
-=======
-from sglang.srt.utils import is_cuda, is_hip, is_npu
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512

 if TYPE_CHECKING:
     from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
     from sglang.srt.managers.schedule_batch import ScheduleBatch
     from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
     from sglang.srt.server_args import ServerArgs
     from sglang.srt.speculative.eagle_info import EagleDraftInput
     from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
~~~~

</details>

<details>
<summary><code>python/sglang/srt/mem_cache/common.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Move common Triton helpers to sglang.kernels and retain DCU cache-location behavior.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/mem_cache/common.py
+++ RESOLVED/python/sglang/srt/mem_cache/common.py
@@ -19,23 +19,18 @@
     maybe_evict_dsv4_state_on_swa,
     maybe_write_dsv4_decode,
     maybe_write_dsv4_extend,
 )
 from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
 from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
 from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
 from sglang.srt.runtime_context import get_server_args
-<<<<<<< HEAD
-from sglang.srt.server_args import ServerArgs, get_global_server_args
+from sglang.srt.server_args import ServerArgs
 from sglang.srt.utils import get_bool_env_var, is_cuda, is_dcu, is_hip, is_npu, support_triton
-=======
-from sglang.srt.server_args import ServerArgs
-from sglang.srt.utils import is_cuda, is_hip, is_npu, support_triton
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
 from sglang.srt.utils.common import ceil_align, is_pin_memory_available

 _is_dcu = is_dcu()
 _is_npu = is_npu()
 if _is_dcu:
     from sgl_kernel.kvcacheio import dcu_get_last_loc

 _is_hip = is_hip()
@@ -197,18 +192,18 @@
         return get_last_loc_triton_safe(
             req_to_token, req_pool_indices_tensor, prefix_lens_tensor
         )

     if uses_triton_dispatch:
         impl = get_last_loc_triton
     else:
         if (
-            get_global_server_args().attention_backend != "ascend"
-            and get_global_server_args().attention_backend != "torch_native"
+            get_server_args().attention_backend != "ascend"
+            and get_server_args().attention_backend != "torch_native"
         ):
             impl = get_last_loc_triton
         else:
             impl = get_last_loc_torch
     use_sglang_get_last_loc = get_bool_env_var("SGLANG_GET_LAST_LOC", default="true")
     if use_sglang_get_last_loc:
         impl = dcu_get_last_loc
     return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)
~~~~

</details>

<details>
<summary><code>python/sglang/srt/mem_cache/memory_pool.py</code> — 3 conflict hunks</summary>

**Resolution intent:** Adopt official buffer descriptors and allocation while retaining DCU LightOp and DSA index-cache layouts.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/mem_cache/memory_pool.py
+++ RESOLVED/python/sglang/srt/mem_cache/memory_pool.py
@@ -2937,74 +2937,54 @@


     def _write_mla_kv_buffer(
         self,
         dst_buffer: torch.Tensor,
         loc: torch.Tensor,
         cache_k_nope: torch.Tensor,
         cache_k_rope: torch.Tensor,
-<<<<<<< HEAD
-    ):
-        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
-        layer_id = layer.layer_id
-
+    ) -> None:
         if (
             _is_hip
             and not _is_dcu
             and self.use_dsa
             and self.dtype == fp8_dtype
         ):
-=======
-    ) -> None:
-        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
             # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
             # Fuse BF16/FP16 -> FP8 cast with paged KV write.
             set_mla_kv_buffer_triton_fp8_quant(
                 dst_buffer,
                 loc,
                 cache_k_nope,
                 cache_k_rope,
                 fp8_dtype,
             )
         elif self.dsa_kv_cache_store_fp8:
             if _is_dcu:
                 from lightop import op

-<<<<<<< HEAD
                 op.fused_quantize_and_store_mla_kv_cache(
                     cache_k_nope,
                     cache_k_rope,
-                    self.kv_buffer[layer_id - self.start_layer],
+                    dst_buffer,
                     loc,
                     "fp8_e4m3",
                     1e-6,
                 )
             else:
                 cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                     cache_k_nope, cache_k_rope
                 )
                 set_mla_kv_buffer_triton(
-                    self.kv_buffer[layer_id - self.start_layer],
+                    dst_buffer,
                     loc,
                     cache_k_nope_fp8,
                     cache_k_rope_fp8,
                 )
-=======
-            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
-            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
-            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
-            set_mla_kv_buffer_triton(
-                dst_buffer,
-                loc,
-                cache_k_nope_fp8,
-                cache_k_rope_fp8,
-            )
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
         else:
             if cache_k_nope.dtype != self.dtype:
                 cache_k_nope = cache_k_nope.to(self.dtype)
                 cache_k_rope = cache_k_rope.to(self.dtype)
             if self.store_dtype != self.dtype:
                 cache_k_nope = cache_k_nope.view(self.store_dtype)
                 cache_k_rope = cache_k_rope.view(self.store_dtype)

@@ -3317,88 +3297,71 @@

     def _create_index_buffers(self):
         num_pages = (self.index_buf_size + self.page_size + 1) // self.page_size
         with (
             torch.cuda.use_mem_pool(self.custom_mem_pool)
             if self.custom_mem_pool
             else nullcontext()
         ):
-<<<<<<< HEAD
             self.index_k_with_scale_buffer = None
             self.index_k_buffer = None
             if self.use_fp8_index_k_cache:
                 self.index_k_with_scale_buffer = [
                     torch.zeros(
                         # Layout:
                         #     ref: test_attention.py :: kv_cache_cast_to_fp8
                         #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                         #     data: for page i,
                         #         * buf[i, :page_size * head_dim] for fp8 data
                         #         * buf[i, page_size * head_dim:].view(float32) for scale
-                        (
-                            (index_buf_size + page_size + 1) // self.page_size,
-                            self.page_size
-                            * (
-                                index_head_dim
-                                + index_head_dim // self.quant_block_size * 4
-                            ),
-                        ),
+                        self._index_buffer_shape(num_pages),
                         dtype=self.index_k_with_scale_buffer_dtype,
-                        device=device,
+                        device=self.device,
                     )
-                    for _ in range(layer_num)
+                    for _ in range(self.layer_num)
                 ]
             else:
                 self.index_k_buffer = [
                     torch.zeros(
                         (
-                            (index_buf_size + page_size + 1) // self.page_size,
+                            num_pages,
                             self.page_size,
                             1,
                             self.index_head_dim,
                         ),
                         dtype=self.index_k_buffer_dtype,
-                        device=device,
+                        device=self.device,
                     )
-                    for _ in range(layer_num)
+                    for _ in range(self.layer_num)
                 ]
-        self._finalize_allocation_log(size)
-=======
-            self.index_k_with_scale_buffer = [
-                torch.zeros(
-                    # Layout:
-                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
-                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
-                    #     data: for page i,
-                    #         * buf[i, :page_size * head_dim] for fp8 data
-                    #         * buf[i, page_size * head_dim:].view(float32) for scale
-                    self._index_buffer_shape(num_pages),
-                    dtype=self.index_k_with_scale_buffer_dtype,
-                    device=self.device,
-                )
-                for _ in range(self.layer_num)
-            ]
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512

     def _clear_buffers(self):
         super()._clear_buffers()
-        del self.index_k_with_scale_buffer
+        if hasattr(self, "index_k_with_scale_buffer"):
+            del self.index_k_with_scale_buffer
+        if hasattr(self, "index_k_buffer"):
+            del self.index_k_buffer

     def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
         """Move latent KV and the DSA indexer cache (key + scale) in lockstep."""
         super().move_kv_cache(tgt_loc, src_loc)

         if tgt_loc.numel() == 0:
             return

         tgt_loc_flat = tgt_loc.view(-1).long()
         src_loc_flat = src_loc.view(-1).long()
-        for index_k in self.index_k_with_scale_buffer:
-            index_k[tgt_loc_flat] = index_k[src_loc_flat]
+        if self.use_fp8_index_k_cache:
+            for index_k in self.index_k_with_scale_buffer:
+                index_k[tgt_loc_flat] = index_k[src_loc_flat]
+        else:
+            for index_k in self.index_k_buffer:
+                flat_index_k = index_k.view(-1, 1, self.index_head_dim)
+                flat_index_k[tgt_loc_flat] = flat_index_k[src_loc_flat]

     def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
         assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
         if self.layer_transfer_counter is not None:
             self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
         return self.index_k_with_scale_buffer[layer_id - self.start_layer]

     def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
@@ -3430,16 +3393,17 @@
     def get_index_k_scale_continuous(
         self,
         layer_id: int,
         seq_len: int,
         page_indices: torch.Tensor,
     ):
         if self.layer_transfer_counter is not None:
             self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
+        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
         buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
         return index_buf_accessor.GetS.execute(
             self, buf, seq_len=seq_len, page_indices=page_indices
         )

     def get_index_k_scale_buffer(
         self,
         layer_id: int,
@@ -3454,16 +3418,17 @@

         :param layer_id: Layer index
         :param seq_len: Sequence length
         :param page_indices: Page indices tensor
         :return: tuple of (k_fp8, k_scale) where
                  k_fp8: (seq_len, index_head_dim), uint8
                  k_scale: (seq_len, 4), uint8
         """
+        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
         if self.layer_transfer_counter is not None:
             self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
         buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
         return index_buf_accessor.GetKAndS.execute(
             self,
             buf,
             page_indices=page_indices,
             seq_len_tensor=seq_len_tensor,
~~~~

</details>

<details>
<summary><code>python/sglang/srt/mem_cache/memory_pool_host.py</code> — 3 conflict hunks</summary>

**Resolution intent:** Port DCU host-transfer behavior to the official descriptor and helper interfaces.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/mem_cache/memory_pool_host.py
+++ RESOLVED/python/sglang/srt/mem_cache/memory_pool_host.py
@@ -2174,23 +2174,18 @@
         self.page_size = anchor_host.page_size
         self.layout = layout
         self.pin_memory = pin_memory
         self.device = device
         self.allocator = get_allocator_from_storage(allocator_type)
         self.dtype = device_pool.store_dtype
         self.start_layer = device_pool.start_layer
         self.end_layer = device_pool.end_layer
-<<<<<<< HEAD
-        self.layer_num = device_pool.layer_num
         self.use_fp8 = device_pool.use_fp8_index_k_cache
-=======
         self.layer_num = self._effective_host_layer_num()
-
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
         self.index_head_dim = device_pool.index_head_dim
         self.indexer_quant_block_size = device_pool.quant_block_size
         self.indexer_dtype = DSATokenToKVPool.index_k_with_scale_buffer_dtype
         if self.use_fp8:
             self.indexer_size_per_token = (
                 self.index_head_dim
                 + self.index_head_dim // self.indexer_quant_block_size * 4
             )
@@ -2340,23 +2335,18 @@
         host_page_indices, device_page_indices = self._get_indexer_page_indices(
             host_indices, device_indices
         )
         use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
         device_index_k_cache = self._get_device_index_k_cache_for_transfer(device_pool)
         if use_kernel:
             if self.layout == "layer_first":
                 transfer_kv_per_layer_mla(
-<<<<<<< HEAD
-                    src=self.index_k_with_scale_buffer[layer_id],
+                    src=self.index_k_with_scale_buffer[host_layer],
                     dst=device_index_k_cache[layer_id],
-=======
-                    src=self.index_k_with_scale_buffer[host_layer],
-                    dst=device_pool.index_k_with_scale_buffer[layer_id],
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
                     src_indices=host_page_indices,
                     dst_indices=device_page_indices,
                     item_size=self.indexer_page_stride_size,
                 )
             elif self.layout == "page_first":
                 transfer_kv_per_layer_mla_pf_lf(
                     src=self.index_k_with_scale_buffer,
                     dst=device_index_k_cache[layer_id],
@@ -2366,23 +2356,18 @@
                     item_size=self.indexer_page_stride_size,
                     src_layout_dim=self.indexer_layout_dim,
                 )
             else:
                 raise ValueError(f"Unsupported layout: {self.layout}")
         elif io_backend == "direct":
             if self.layout == "layer_first":
                 transfer_kv_direct(
-<<<<<<< HEAD
-                    src_layers=[self.index_k_with_scale_buffer[layer_id]],
+                    src_layers=[self.index_k_with_scale_buffer[host_layer]],
                     dst_layers=[device_index_k_cache[layer_id]],
-=======
-                    src_layers=[self.index_k_with_scale_buffer[host_layer]],
-                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
                     src_indices=host_page_indices,
                     dst_indices=device_page_indices,
                     page_size=1,
                 )
             elif self.layout == "page_first_direct":
                 transfer_kv_per_layer_direct_pf_lf(
                     src_ptrs=[self.index_k_with_scale_buffer],
                     dst_ptrs=[device_index_k_cache[layer_id]],
@@ -2398,37 +2383,38 @@

     def _backup_from_device_per_layer(
         self, device_pool, host_indices, device_indices, layer_id, io_backend
     ):
         host_layer = self._host_layer_index(layer_id)
         host_page_indices, device_page_indices = self._get_indexer_page_indices(
             host_indices, device_indices
         )
+        device_index_k_cache = self._get_device_index_k_cache_for_transfer(device_pool)
         use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
         if use_kernel:
             if self.layout == "layer_first":
                 transfer_kv_per_layer_mla(
-                    src=device_pool.index_k_with_scale_buffer[layer_id],
+                    src=device_index_k_cache[layer_id],
                     dst=self.index_k_with_scale_buffer[host_layer],
                     src_indices=device_page_indices,
                     dst_indices=host_page_indices,
                     item_size=self.indexer_page_stride_size,
                 )
             elif self.layout == "page_first":
                 raise ValueError(
                     "Layer-sharded DSA indexer HiCache backup with page_first "
                     "layout is not supported without a per-layer LF->PF kernel."
                 )
             else:
                 raise ValueError(f"Unsupported layout: {self.layout}")
         elif io_backend == "direct":
             if self.layout == "layer_first":
                 transfer_kv_direct(
-                    src_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
+                    src_layers=[device_index_k_cache[layer_id]],
                     dst_layers=[self.index_k_with_scale_buffer[host_layer]],
                     src_indices=device_page_indices,
                     dst_indices=host_page_indices,
                     page_size=1,
                 )
             else:
                 raise ValueError(
                     "Layer-sharded direct DSA indexer backup only supports "
~~~~

</details>

<details>
<summary><code>python/sglang/srt/model_executor/model_runner.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Accept official runtime-context and pool lifecycle changes while preserving DCU graph configuration.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/model_executor/model_runner.py
+++ RESOLVED/python/sglang/srt/model_executor/model_runner.py
@@ -126,22 +126,18 @@
     attn_backend_wrapper,
 )
 from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
 from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
 from sglang.srt.layers.cp.utils import (
     get_cp_strategy,
 )
 from sglang.srt.layers.dp_attention import (
-<<<<<<< HEAD
     DpPaddingMode,
-    get_attention_tp_group,
     get_attention_tp_size,
-=======
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
     initialize_dp_attention,
     set_dp_buffer_len,
     set_is_extend_in_batch,
 )
 from sglang.srt.layers.logits_processor import LogitsProcessorOutput
 from sglang.srt.layers.moe.hash_topk import HashTopK
 from sglang.srt.layers.moe.topk import TopK
 from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
@@ -1030,22 +1026,18 @@
             "1",
             "true",
             "yes",
             "on",
         }

         set_global_experts_capturer(
             RoutedExpertsCapturer.create(
-<<<<<<< HEAD
-                enable=get_global_server_args().enable_return_routed_experts
+                enable=get_server_args().enable_return_routed_experts
                 or debug_moe_trace,
-=======
-                enable=get_server_args().enable_return_routed_experts,
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
                 model_config=self.model_config,
                 num_fused_shared_experts=num_fused_shared_experts,
                 num_tokens=self.max_total_num_tokens + self.page_size,
                 max_running_requests=self.max_running_requests,
                 device=self.device,
             )
         )

~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/bailing_moe.py</code> — 5 conflict hunks</summary>

**Resolution intent:** Adopt official scoped forward state and retain DCU fused RMS inputs and SBO flags.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/bailing_moe.py
+++ RESOLVED/python/sglang/srt/models/bailing_moe.py
@@ -76,29 +76,30 @@
 from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
 from sglang.srt.model_executor.runner import get_is_capture_mode
 from sglang.srt.model_loader.weight_utils import default_weight_loader
 from sglang.srt.models.utils import (
     apply_qk_norm,
     create_fused_set_kv_buffer_arg,
     enable_fused_set_kv_buffer,
 )
-<<<<<<< HEAD
-from sglang.srt.runtime_context import get_parallel, get_server_args, get_stream
-from sglang.srt.server_args import get_global_server_args
-from sglang.srt.utils import get_bool_env_var, add_prefix, is_cuda, is_dcu, is_non_idle_and_non_empty, make_layers
-=======
 from sglang.srt.runtime_context import (
     get_forward,
     get_parallel,
     get_server_args,
     get_stream,
 )
-from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
+from sglang.srt.utils import (
+    add_prefix,
+    get_bool_env_var,
+    is_cuda,
+    is_dcu,
+    is_non_idle_and_non_empty,
+    make_layers,
+)

 LoraConfig = None
 logger = logging.getLogger(__name__)
 _is_cuda = is_cuda()
 _is_dcu = is_dcu()

 _use_fused_bailing_silu_mul_fp8_quant = get_bool_env_var("SGLANG_USE_FUSED_BAILING_SILU_MUL_FP8_QUANT")
 _use_fused_bailing_rms_rotary = get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY")
@@ -154,31 +155,28 @@
     ) -> torch.Tensor:
         hidden_states_tensor = (
             hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
         )
         if (self.tp_size == 1) and hidden_states_tensor.shape[0] == 0:
             return hidden_states

         gate_up, _ = self.gate_up_proj(hidden_states)
-<<<<<<< HEAD
         if _use_fused_bailing_silu_mul_fp8_quant:
             hidden_states, _ = self.down_proj(
-                gate_up, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter, use_fused_silu_mul_fp8_quant = True
+                gate_up,
+                forward_batch=forward_batch,
+                use_fused_silu_mul_fp8_quant=True,
             )
         else:
             hidden_states = self.act_fn(gate_up)
             hidden_states, _ = self.down_proj(
-                hidden_states, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
-            )
-
-=======
-        hidden_states = self.act_fn(gate_up)
-        hidden_states, _ = self.down_proj(hidden_states)
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
+                hidden_states,
+                forward_batch=forward_batch,
+            )
         return hidden_states


 class BailingMoEGate(nn.Module):
     def __init__(
         self,
         config,
         params_dtype: Optional[torch.dtype] = None,
@@ -235,17 +233,17 @@
         self.top_k = config.num_experts_per_tok
         self.norm_topk_prob = config.norm_topk_prob
         self.hidden_size = config.hidden_size
         self.num_shared_experts = config.num_shared_experts
         self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
         self.score_function = getattr(config, "score_function", None)
         self.num_fused_shared_experts = (
             0
-            if get_global_server_args().disable_shared_experts_fusion or get_moe_a2a_backend().is_deepep()
+            if get_server_args().disable_shared_experts_fusion or get_moe_a2a_backend().is_deepep()
             else config.num_shared_experts
         )

         if config.hidden_act != "silu":
             raise ValueError(
                 f"Unsupported activation: {config.hidden_act}. "
                 "Only silu is supported for now."
             )
@@ -314,17 +312,17 @@
             quant_config=quant_config,
             routed_scaling_factor=self.routed_scaling_factor,
             fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
         )

         self.experts = get_moe_impl_class(quant_config)(
             num_experts=config.num_experts
             + self.num_fused_shared_experts
-            + get_global_server_args().ep_num_redundant_experts,
+            + get_server_args().ep_num_redundant_experts,
             num_fused_shared_experts=self.num_fused_shared_experts,
             top_k=self.top_k + self.num_fused_shared_experts,
             layer_id=self.layer_id,
             hidden_size=config.hidden_size,
             intermediate_size=config.moe_intermediate_size,
             quant_config=quant_config,
             routed_scaling_factor=self.routed_scaling_factor,
             prefix=add_prefix("experts", prefix),
@@ -367,35 +365,25 @@
                 return_recv_hook=True,
             )
         self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

     def forward(
         self,
         hidden_states: torch.Tensor,
         forward_batch: Optional[ForwardBatch] = None,
-<<<<<<< HEAD
-        should_allreduce_fusion: bool = False,
-        use_reduce_scatter: bool = False,
         moe_i_q: Optional[torch.Tensor] = None,
         moe_i_s: Optional[torch.Tensor] = None,
     ) -> torch.Tensor:
         if not get_moe_a2a_backend().is_deepep():
             return self.forward_normal(
                 hidden_states,
-                should_allreduce_fusion,
-                use_reduce_scatter,
                 moe_i_q=moe_i_q,
                 moe_i_s=moe_i_s,
             )
-=======
-    ) -> torch.Tensor:
-        if not get_moe_a2a_backend().is_deepep():
-            return self.forward_normal(hidden_states)
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
         else:
             return self.forward_deepep(
                 hidden_states, forward_batch, moe_i_q=moe_i_q, moe_i_s=moe_i_s
             )

     def get_moe_weights(self):
         return [
             x.data
@@ -463,23 +451,18 @@
             )
         current_stream.wait_stream(self.alt_stream)

         return router_output, shared_output

     def forward_normal(
         self,
         hidden_states: torch.Tensor,
-<<<<<<< HEAD
-        should_allreduce_fusion: bool = False,
-        use_reduce_scatter: bool = False,
         moe_i_q: Optional[torch.Tensor] = None,
         moe_i_s: Optional[torch.Tensor] = None,
-=======
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
     ) -> torch.Tensor:
         num_tokens, hidden_size = hidden_states.shape
         hidden_states = hidden_states.view(-1, hidden_size)

         if (
             self.alt_stream is not None
             and hidden_states.shape[0] > 0
             and self.num_fused_shared_experts == 0
@@ -802,21 +785,21 @@
             num_kv_heads=self.num_kv_heads,
             layer_id=layer_id,
             prefix=add_prefix("attn", prefix),
         )

         self.alt_stream = alt_stream
         self.page_size = 64
         self.layer_id = layer_id
-        if get_global_server_args().kv_cache_dtype == "fp8_e4m3":
+        if get_server_args().kv_cache_dtype == "fp8_e4m3":
             self.kv_cache_dtype = torch.float8_e4m3fn
-        elif get_global_server_args().kv_cache_dtype == "fp8_e5m2":
+        elif get_server_args().kv_cache_dtype == "fp8_e5m2":
             self.kv_cache_dtype = torch.float8_e5m2
-        elif get_global_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
+        elif get_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
             self.kv_cache_dtype = torch.bfloat16

     def forward(
         self,
         positions: torch.Tensor,
         hidden_states: torch.Tensor,
         forward_batch: ForwardBatch,
     ) -> torch.Tensor:
@@ -1018,23 +1001,23 @@
                     forward_batch=forward_batch,
                 )

         # Keep this fusion path minimal: only first 4 dense layers.
         dense_rms_quant_fusion = (
             _is_dcu
             and _use_fused_bailing_rms_quant
             and (not self.is_layer_sparse)
-            and (not is_dp_attention_enabled() or get_global_server_args().ep_size > 1)
+            and (not is_dp_attention_enabled() or get_server_args().ep_size > 1)
         )
         sparse_rms_quant_fusion = (
             _is_dcu
             and _use_fused_bailing_rms_quant
             and self.is_layer_sparse
-            and (not is_dp_attention_enabled() or get_global_server_args().ep_size > 1)
+            and (not is_dp_attention_enabled() or get_server_args().ep_size > 1)
         )
         rms_quant_fusion = dense_rms_quant_fusion or sparse_rms_quant_fusion
         prev_rms_quant_flag = forward_batch.rms_quant_flag
         prev_sparse_rms_quant_fusion = getattr(
             forward_batch, "bailing_sparse_rms_quant_fusion", False
         )
         prev_sparse_norm_hidden_states = getattr(
             forward_batch, "bailing_sparse_norm_hidden_states", None
@@ -1069,37 +1052,29 @@
             )
         )

         # For DP with padding, reduce scatter can be used instead of all-reduce.
         mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
             forward_batch
         )

-<<<<<<< HEAD
-        if self.is_layer_sparse:
-            hidden_states = self.mlp(
-                hidden_states,
-                forward_batch,
-                should_allreduce_fusion,
-                use_reduce_scatter,
-                moe_i_q=sparse_moe_i_q,
-                moe_i_s=sparse_moe_i_s,
-            )
-        else:
-            hidden_states = self.mlp(
-                hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
-            )
-=======
         with get_forward().scoped(
             fuse_mlp_allreduce=fuse_mlp_allreduce,
             mlp_reduce_scatter=mlp_reduce_scatter,
         ):
-            hidden_states = self.mlp(hidden_states, forward_batch)
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
+            if self.is_layer_sparse:
+                hidden_states = self.mlp(
+                    hidden_states,
+                    forward_batch,
+                    moe_i_q=sparse_moe_i_q,
+                    moe_i_s=sparse_moe_i_s,
+                )
+            else:
+                hidden_states = self.mlp(hidden_states, forward_batch)

         if fuse_mlp_allreduce:
             hidden_states._sglang_needs_allreduce_fusion = True
         else:
             hidden_states, residual = self.layer_communicator.postprocess_layer(
                 hidden_states, residual, forward_batch
             )

@@ -1241,17 +1216,17 @@
                 config.hidden_size,
                 quant_config=quant_config,
                 prefix=add_prefix("lm_head", prefix),
                 use_attn_tp_group=get_server_args().enable_dp_lm_head,
             )
         self.logits_processor = LogitsProcessor(config)
         self.num_fused_shared_experts = (
             0
-            if get_global_server_args().disable_shared_experts_fusion or get_moe_a2a_backend().is_deepep()
+            if get_server_args().disable_shared_experts_fusion or get_moe_a2a_backend().is_deepep()
             else config.num_shared_experts
         )

         self.capture_aux_hidden_states = False

     @property
     def start_layer(self):
         return self.model.start_layer
~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/deepseek_v2.py</code> — 5 conflict hunks</summary>

**Resolution intent:** Adopt official dual-stream and forward-state contracts while retaining DCU fused RMS and tuple-output behavior.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v2.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v2.py
@@ -325,17 +325,17 @@
             out = torch.empty(
                 (A.shape[0], A.shape[1], B.shape[2]),
                 device=A.device,
                 dtype=dtype,
             )
         _bmm_fp8_op(A, B, out, A_scale, B_scale)
         return out
 elif _is_hip:
-    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
+    from sglang.kernels.ops.attention.rocm_mla_decode_rope import (
         decode_attention_fwd_grouped_rope,
     )
     from sgl_kernel import merge_state_v2
 elif _is_npu:
     from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
         forward_dsa_core_npu,
         forward_dsa_prepare_npu,
         forward_mha_core_npu,
@@ -1088,23 +1088,18 @@
             )

     def forward_normal_dual_stream(
         self,
         hidden_states: torch.Tensor,
         gemm_output_zero_allocator: BumpAllocator = None,
         input_ids: Optional[torch.Tensor] = None,
         input_ids_global: Optional[torch.Tensor] = None,
-<<<<<<< HEAD
         rms_weight: Optional[torch.Tensor] = None,
         residual: Optional[torch.Tensor] = None,
-        *,
-        use_flashinfer_trtllm_bypass: bool = False,
-=======
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
     ) -> torch.Tensor:
         # Note(kpham-sgl): issue order satisfies 3 constraints:
         # - no stream explosion: main (routed) issued before alt block -> capture reuses 1 alt stream;
         # - PDL overlap: routed is the last main-stream kernel (fuses w/ residual add);
         # - dispose_tensor: disabled during capture (CaptureFlags.disable_dispose_tensor) so the routed
         #   deep_gemm does not free hidden_states, which the shared expert reads on the alt stream.
         use_flashinfer_trtllm_bypass = get_forward().flashinfer_trtllm_bypass
         current_stream = torch.cuda.current_stream()
@@ -1113,31 +1108,16 @@
             hidden_states.shape[0] > 0 and self.num_fused_shared_experts == 0
         )
         server_args = get_server_args()
         dispatch_info = (
             ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
             if server_args.enable_eplb and not self.is_nextn
             else None
         )
-<<<<<<< HEAD
-        with torch.cuda.stream(self.alt_stream):
-            if _use_fused_rms_quant and rms_weight is not None and residual is not None:
-                shared_output, _, _, _ = self._forward_shared_experts(
-                    hidden_states,
-                    gemm_output_zero_allocator,
-                    rms_weight=rms_weight,
-                    residual=residual,
-                )
-            else:
-                shared_output = self._forward_shared_experts(
-                    hidden_states, gemm_output_zero_allocator
-                )
-=======
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
         # router_logits: (num_tokens, n_experts)
         router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
         if use_flashinfer_trtllm_bypass:
             topk_output = BypassedTopKOutput(
                 hidden_states=hidden_states,
                 router_logits=router_logits,
                 topk_config=self.topk.topk_config,
             )
@@ -1172,19 +1152,29 @@
             and not _is_musa
             and not _use_aiter
             or isinstance(self.experts.quant_method, KTEPWrapperMethod)
         ):
             final_hidden_states *= self.routed_scaling_factor

         # Shared expert on alt stream, issued AFTER the main (routed) branch. See note above.
         with torch.cuda.stream(self.alt_stream):
-            shared_output = self._forward_shared_experts(
-                hidden_states, gemm_output_zero_allocator
-            )
+            if _use_fused_rms_quant and rms_weight is not None and residual is not None:
+                shared_output = self._forward_shared_experts(
+                    hidden_states,
+                    gemm_output_zero_allocator,
+                    rms_weight=rms_weight,
+                    residual=residual,
+                )
+                if isinstance(shared_output, tuple):
+                    shared_output = shared_output[0]
+            else:
+                shared_output = self._forward_shared_experts(
+                    hidden_states, gemm_output_zero_allocator
+                )

         current_stream.wait_stream(self.alt_stream)

         if deferred_finalize:
             from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                 finalize_flashinfer_trtllm_deferred_output,
             )

@@ -3471,35 +3461,29 @@
             layer_scatter_modes=self.layer_scatter_modes,
             prev_topk_indices=prev_topk_indices,
         )
         if isinstance(hidden_states, tuple):
             hidden_states, topk_indices = hidden_states
         else:
             topk_indices = None

-<<<<<<< HEAD
-        # residual = forward_batch.residual_rms_per_quant_int8  # residual在attn中没有变化，不用这句
-        # 判断sbo+moe，如果是则不跳过norm
-        forward_batch.rms_quant_flag = not (self.is_layer_sparse and _is_sbo_enabled)  # NOTE: if _is_sbo_enabled reliable?
+        maybe_prefetch_next_full_attention_kv(
+            forward_batch, next_full_attention_layer_id
+        )
+
+        # DCU fused RMS/quant must keep the norm for SBO sparse-MoE layers.
+        forward_batch.rms_quant_flag = not (
+            self.is_layer_sparse and _is_sbo_enabled
+        )
         hidden_states, residual = self.layer_communicator.prepare_mlp(
             hidden_states, residual, forward_batch
         )
-        should_allreduce_fusion = (
-=======
-        maybe_prefetch_next_full_attention_kv(
-            forward_batch, next_full_attention_layer_id
-        )
-
-        hidden_states, residual = self.layer_communicator.prepare_mlp(
-            hidden_states, residual, forward_batch
-        )

         fuse_mlp_allreduce = (
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
             self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                 forward_batch
             )
         )

         # For DP with padding, reduce scatter can be used instead of all-reduce.
         mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
             forward_batch
@@ -3514,54 +3498,40 @@
             and not torch.compiler.is_compiling()
         ):
             from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

             _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
         else:
             _mlp_ctx = nullcontext()

-<<<<<<< HEAD
-        # Dense layers return four values when the DCU fused RMS/quant path is active.
+        mlp_kwargs = {}
         if (
             _use_fused_rms_quant
             and residual is not None
             and self.post_attention_layernorm.weight.data is not None
-            and isinstance(self.mlp, DeepseekV2MLP)
+            and isinstance(self.mlp, (DeepseekV2MLP, DeepseekV2MoE))
         ):
-            hidden_states, _, _, _ = self.mlp(
-                hidden_states,
-                forward_batch,
-                should_allreduce_fusion,
-                use_reduce_scatter,
-                gemm_output_zero_allocator,
-                rms_weight=self.post_attention_layernorm.weight.data,
-                residual=residual,
-            )
-        else:
-=======
+            mlp_kwargs = {
+                "rms_weight": self.post_attention_layernorm.weight.data,
+                "residual": residual,
+            }
+
         with get_forward().scoped(
             fuse_mlp_allreduce=fuse_mlp_allreduce,
             mlp_reduce_scatter=mlp_reduce_scatter,
         ):
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
             with _mlp_ctx:
-                hidden_states = self.mlp(
+                mlp_output = self.mlp(
                     hidden_states,
                     forward_batch,
-<<<<<<< HEAD
-                    should_allreduce_fusion,
-                    use_reduce_scatter,
                     gemm_output_zero_allocator,
-                    rms_weight=self.post_attention_layernorm.weight.data,
-                    residual=residual,
-=======
-                    gemm_output_zero_allocator,
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
-                )
+                    **mlp_kwargs,
+                )
+        hidden_states = mlp_output[0] if isinstance(mlp_output, tuple) else mlp_output

         if (
             not (self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp)
             and fuse_mlp_allreduce
         ):
             hidden_states._sglang_needs_allreduce_fusion = True

         if not fuse_mlp_allreduce:
~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/deepseek_v4.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Port DCU q, WO-A, rotary, and MHC behavior onto the official MQALayer and HC refactor.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v4.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v4.py
@@ -20,17 +20,16 @@

 import sglang.srt.models.deepseek_v2 as deepseek_v2
 from sglang.jit_kernel.dsv4 import (
     fused_norm_rope_inplace,
     fused_q_norm_rope,
     fused_rope_inplace,
     sglang_per_token_group_quant_fp8_dsv4_wo_a,
 )
-fused_rope = fused_rope_inplace
 from sglang.srt.compilation.compilation_config import register_split_op
 from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
 from sglang.srt.distributed import (
     get_pp_group,
     get_tp_group,
 )
 from sglang.srt.environ import envs
 from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
@@ -412,63 +411,19 @@
             envs.SGLANG_OPT_FUSE_WQA_WKV.get() if fuse_wqa_wkv is None else fuse_wqa_wkv
         )
         fp8: bool = _FP8_WO_A_GEMM if wo_a_fp8 is None else wo_a_fp8
         reduce_results: bool = (
             (self.attn_tp_size == get_parallel().tp_size and self.attn_tp_size > 1)
             if wo_b_reduce_results is None
             else wo_b_reduce_results
         )
-<<<<<<< HEAD
-        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
-        self.freqs_cis: torch.Tensor
-
-        self.register_buffer("cos_sin_cache_fused", None, persistent=False)
-        self.cos_sin_cache_fused: Optional[torch.Tensor]
-        if _is_dcu and _use_fused_qnorm_rope_kv_rope_quant:
-            freqs_real = torch.view_as_real(self.freqs_cis)  # [max_pos, 32, 2]
-            cos_sin_cache = torch.cat(
-                [freqs_real[..., 0], freqs_real[..., 1]], dim=-1
-            ).contiguous()  # [max_pos, 64], first 32 cos, last 32 sin
-            self.cos_sin_cache_fused = cos_sin_cache
-        elif _is_hip and not _is_dcu:
-            cos_cache = freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
-            sin_cache = freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
-            self.register_buffer("cos_cache", cos_cache, persistent=False)
-            self.register_buffer("sin_cache", sin_cache, persistent=False)
-
-        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
-            self.alt_streams = alt_streams[:3]
-            self.alt_streams_indexer = alt_streams[-2:]
-        else:
-            self.alt_streams = None
-            self.alt_streams_indexer = None
-
-        from sglang.srt.utils import is_blackwell_supported
-
-        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64
-
-        self.compressor = None
-        self.indexer = None
-        if self.compress_ratio in (4, 128):
-            self.compressor = Compressor(
-                config,
-                layer_id=self.layer_id,
-                is_in_indexer=False,
-                freqs_cis=freqs_cis,
-                compress_ratio=self.compress_ratio,
-                head_dim=self.head_dim,
-                rotate=False,
-                prefix=add_prefix("compressor", prefix),
-                rotary_emb=getattr(self, "rotary_emb", None),
-=======
         if wo_a_keeps_quant_config is None:
             wo_a_quant_config: Optional[QuantizationConfig] = (
                 quant_config if fp8 else None
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
             )
         elif wo_a_keeps_quant_config:
             wo_a_quant_config = quant_config
         else:
             wo_a_quant_config = None

         self.fuse_wqa_wkv = fuse

@@ -505,33 +460,34 @@
             self.n_heads * self.head_dim,
             bias=False,
             quant_config=quant_config,
             prefix=add_prefix("wq_b", prefix),
             tp_rank=self.attn_tp_rank,
             tp_size=self.attn_tp_size,
         )
         self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
-        if _FP8_WO_A_GEMM and quant_config is not None:
-            quant_config.ignore = [i for i in quant_config.ignore if 'wo_a' not in i]
+        if fp8 and quant_config is not None:
+            quant_config.ignore = [i for i in quant_config.ignore if "wo_a" not in i]
         self.wo_a = ColumnParallelLinear(
             self.n_heads * self.head_dim // self.n_groups,
             self.n_groups * self.o_lora_rank,
             bias=False,
             quant_config=wo_a_quant_config,
             prefix=add_prefix("wo_a", prefix),
             tp_rank=self.attn_tp_rank,
             tp_size=self.attn_tp_size,
             **({} if fp8 else {"params_dtype": torch.bfloat16}),
         )
         if fp8:
-            assert hasattr(
-                self.wo_a, "weight_scale"
-            ), "FP8 quant_config must create weight_scale"
-            self.wo_a.weight_scale_inv = self.wo_a.weight_scale
+            if not hasattr(self.wo_a, "weight_scale_inv"):
+                assert hasattr(
+                    self.wo_a, "weight_scale"
+                ), "FP8 quant_config must create weight_scale or weight_scale_inv"
+                self.wo_a.weight_scale_inv = self.wo_a.weight_scale
             self.wo_a.weight_scale_inv.format_ue8m0 = True
         self.wo_b = RowParallelLinear(
             self.n_groups * self.o_lora_rank,
             self.hidden_size,
             bias=False,
             quant_config=quant_config,
             reduce_results=reduce_results,
             prefix=add_prefix("wo_b", prefix),
@@ -592,17 +548,24 @@
             rotary_dim=self.rope_head_dim,
             max_position=config.max_position_embeddings,
             base=self.rope_base,
             rope_scaling=self.rope_scaling,
             is_neox_style=False,
             device=get_server_args().device,
         )

-        if _is_hip:
+        self.register_buffer("cos_sin_cache_fused", None, persistent=False)
+        self.cos_sin_cache_fused: Optional[torch.Tensor]
+        if _is_dcu and _use_fused_qnorm_rope_kv_rope_quant:
+            freqs_real = torch.view_as_real(self.freqs_cis)
+            self.cos_sin_cache_fused = torch.cat(
+                [freqs_real[..., 0], freqs_real[..., 1]], dim=-1
+            ).contiguous()
+        elif _is_hip and not _is_dcu:
             cos_cache = (
                 self.freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
             )
             sin_cache = (
                 self.freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
             )
             self.register_buffer("cos_cache", cos_cache, persistent=False)
             self.register_buffer("sin_cache", sin_cache, persistent=False)
@@ -684,22 +647,22 @@
         q = q.view(-1, self.n_local_heads, self.head_dim)
         if not _is_dcu:
             if q_out is None:
                 q_out = torch.empty_like(q)
             # Official fused warp-per-(token, head) RMSNorm + RoPE path.
             fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
             return q_out

-        if _is_dcu and _use_dpskv4_lightop_rmsnorm:
+        if _use_dpskv4_lightop_rmsnorm:
             op.rms_norm_no_weight(None, q, None, self.eps)
         else:
-            q = rms_normalize_triton(q, self.eps)
+            q = F.rms_norm(q, (self.head_dim,), eps=self.eps)
         if positions is not None:
-            fused_rope(
+            fused_rope_inplace(
                 q[..., -self.qk_rope_head_dim :],
                 None,
                 self.freqs_cis,
                 positions=positions,
             )
         else:
             apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
         if q_out is not None:
@@ -1267,17 +1230,20 @@
                 self.freqs_cis,
                 positions=positions,
                 inverse=True,
             )

         o = o.view(o.shape[0], self.n_local_groups, -1)

         if _FP8_WO_A_GEMM:
-            import deepgemm as deep_gemm
+            if _is_dcu:
+                import deepgemm as deep_gemm
+            else:
+                import deep_gemm

             T, G, D = o.shape
             R = self.o_lora_rank
             o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_wo_a(o)
             output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)

             deep_gemm.fp8_einsum(
                 "bhr,hdr->bhd",
@@ -2139,31 +2105,22 @@
         if self.pp_group.is_last_rank:
             self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
         else:
             self.norm = PPMissingLayer()
         self.gemm_output_zero_allocator_size = 0
         self.hc_eps = config.hc_eps
         self.hc_mult = hc_mult = config.hc_mult
         self.norm_eps = config.rms_norm_eps
-        hc_dim = hc_mult * config.hidden_size
         if self.pp_group.is_last_rank:
-<<<<<<< HEAD
-            self.hc_head_fn = nn.Parameter(
-                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
-            )
-            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
-            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
-=======
             (
                 self.hc_head_fn,
                 self.hc_head_base,
                 self.hc_head_scale,
             ) = make_hc_head_params(hc_mult, config.hidden_size)
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512

         self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
         self.use_fused_mhc_post_pre = _is_fused_mhc_post_pre_enabled()
         if self.dsa_enable_prefill_cp:
             self.cp_size = get_parallel().attn_cp_size

     def hc_head(
         self,
~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/qwen2.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt runtime ServerArgs access while retaining the DCU fused attention and RMS path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/qwen2.py
+++ RESOLVED/python/sglang/srt/models/qwen2.py
@@ -46,29 +46,27 @@
 )
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
 from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
 from sglang.srt.model_loader.weight_utils import (
     default_weight_loader,
     kv_cache_scales_loader,
 )
 from sglang.srt.platforms import current_platform
-<<<<<<< HEAD
-from sglang.srt.runtime_context import get_parallel
-from sglang.srt.server_args import get_global_server_args
-from sglang.srt.utils import add_prefix, make_layers,is_dcu,get_bool_env_var
-=======
 from sglang.srt.runtime_context import get_parallel, get_server_args
-from sglang.srt.utils import add_prefix, make_layers
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
+from sglang.srt.utils import add_prefix, get_bool_env_var, is_dcu, make_layers
 from sglang.srt.utils.hf_transformers_utils import get_rope_config
+
 _is_dcu = is_dcu()
 if _is_dcu:
     from lightop import split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
-_use_fused_rms_rotary=get_bool_env_var("SGLANG_USE_FUSED_SPLIT_QKV_RMS_ROTARY_EMBEDDING")
+
+_use_fused_rms_rotary = get_bool_env_var(
+    "SGLANG_USE_FUSED_SPLIT_QKV_RMS_ROTARY_EMBEDDING"
+)

 Qwen2Config = None


 logger = logging.getLogger(__name__)


 class Qwen2MLP(nn.Module):
@@ -135,21 +133,21 @@
         self.hidden_size = hidden_size
         tp_size = get_parallel().tp_size
         self.total_num_heads = num_heads
         assert self.total_num_heads % tp_size == 0
         self.num_heads = self.total_num_heads // tp_size
         self.total_num_kv_heads = num_kv_heads
         self.layer_id = layer_id
         self.page_size=64
-        if get_global_server_args().kv_cache_dtype == "fp8_e4m3":
+        if get_server_args().kv_cache_dtype == "fp8_e4m3":
             self.kv_cache_dtype = torch.float8_e4m3fn
-        elif get_global_server_args().kv_cache_dtype == "fp8_e5m2":
+        elif get_server_args().kv_cache_dtype == "fp8_e5m2":
             self.kv_cache_dtype = torch.float8_e5m2
-        elif get_global_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
+        elif get_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
             self.kv_cache_dtype = torch.bfloat16
         else:
             self.kv_cache_dtype = torch.bfloat16
         if self.total_num_kv_heads >= tp_size:
             # Number of KV heads is greater than TP size, so we partition
             # the KV heads across multiple tensor parallel GPUs.
             assert self.total_num_kv_heads % tp_size == 0
         else:
~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/qwen3.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt runtime ServerArgs access while retaining the DCU fused attention and RMS path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/qwen3.py
+++ RESOLVED/python/sglang/srt/models/qwen3.py
@@ -29,23 +29,18 @@
 from sglang.srt.model_loader.weight_utils import (
     default_weight_loader,
     maybe_remap_kv_scale_name,
 )
 from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
 from sglang.srt.models.qwen2 import Qwen2Model
 from sglang.srt.models.utils import apply_qk_norm
 from sglang.srt.runtime_context import get_parallel, get_server_args, get_stream
-<<<<<<< HEAD
-from sglang.srt.server_args import get_global_server_args
 from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_dcu, is_hip, is_npu
 _use_fused_qwen_bailing_rotary = get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY")
-=======
-from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512

 Qwen3Config = None

 logger = logging.getLogger(__name__)
 _is_cuda = is_cuda()
 _is_hip = is_hip()
 _is_dcu = is_dcu()
 _is_npu = is_npu()
@@ -161,21 +156,21 @@
             self.scaling,
             num_kv_heads=self.num_kv_heads,
             layer_id=layer_id,
             prefix=add_prefix("attn", prefix),
         )
         self.alt_stream = alt_stream
         self.page_size = 64
         self.layer_id = layer_id
-        if get_global_server_args().kv_cache_dtype == "fp8_e4m3":
+        if get_server_args().kv_cache_dtype == "fp8_e4m3":
             self.kv_cache_dtype = torch.float8_e4m3fn
-        elif get_global_server_args().kv_cache_dtype == "fp8_e5m2":
+        elif get_server_args().kv_cache_dtype == "fp8_e5m2":
             self.kv_cache_dtype = torch.float8_e5m2
-        elif get_global_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
+        elif get_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
             self.kv_cache_dtype = torch.bfloat16

         self.use_fused_qk_norm_mrope = (
             _has_fused_qk_norm_mrope
             and isinstance(self.rotary_emb, MRotaryEmbedding)
             and getattr(self.rotary_emb, "mrope_section", None) is not None
         )
         if self.use_fused_qk_norm_mrope:
~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/utils.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Combine official runtime flags with retained DCU model utility hooks.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/utils.py
+++ RESOLVED/python/sglang/srt/models/utils.py
@@ -28,23 +28,18 @@
 from sglang.srt.environ import envs
 from sglang.srt.layers.radix_attention import RadixAttention
 from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
 from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
 from sglang.srt.model_executor.runner import get_is_capture_mode
 from sglang.srt.model_loader.weight_utils import default_weight_loader
-<<<<<<< HEAD
-from sglang.srt.server_args import get_global_server_args
+from sglang.srt.runtime_context import get_server_args
 from sglang.srt.utils import get_current_device_stream_fast, is_cuda, is_dcu, is_hip
-=======
-from sglang.srt.runtime_context import get_server_args
-from sglang.srt.utils import get_current_device_stream_fast, is_cuda, is_hip
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512
 from sglang.srt.utils.custom_op import register_custom_op

 if TYPE_CHECKING:
     from sglang.srt.layers.layernorm import RMSNorm

 _is_cuda = is_cuda()
 _is_hip = is_hip()
 _is_dcu = is_dcu()
~~~~

</details>

<details>
<summary><code>test/registered/unit/mem_cache/test_radix_cache_unit.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept official radix-cache test updates and retain DCU registration.

~~~~diff
--- AUTO-CONFLICT/test/registered/unit/mem_cache/test_radix_cache_unit.py
+++ RESOLVED/test/registered/unit/mem_cache/test_radix_cache_unit.py
@@ -12,24 +12,18 @@
 - Boundary conditions with parameterized testing

 Usage:
     python test_radix_cache_unit.py
     python -m pytest test_radix_cache_unit.py -v
     python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic
 """

-<<<<<<< HEAD
-from sglang.srt.mem_cache.common import available_and_evictable_str
 from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
 register_dcu_ci(est_time=5, suite="stage-b-test-1-gpu-small-dcu")
-
-=======
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
->>>>>>> e1d51be91f6be39e585756568a8f66b99ac2c512

 # CPU-based unit test, runs quickly on any GPU runner
 register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
 register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

 import random
 import unittest
 import unittest.mock
~~~~

</details>
