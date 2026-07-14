# Official Main Catch-up 20260703 — Code Conflict Review

> Scope: only the 20 files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `ec49eb80ae6bb9044bc6c31b5fd3d3621516b877`
- Common official base: `f920a37da46e1cbb6ba27b76365a622eba593811`
- Official endpoint (`theirs`): `88db9e033a11b2d366a8f9d037f027a46ccb9940`
- Resolved merge: `b6e37b4a2498f41b20b0bd12a538f2c6e667d7d5`
- Reconstructed textual conflicts: 20 files, 35 hunks

Each section reconstructs Git’s three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>python/pyproject.toml</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Accept the official TileLang 0.1.11 dependency alignment together with the endpoint's TVM FFI and sgl-deep-gemm upgrades.

~~~~diff
--- AUTO-CONFLICT/python/pyproject.toml
+++ RESOLVED/python/pyproject.toml
@@ -63,23 +63,17 @@
   "requests",
   "scipy",
   "sentencepiece",
   "setproctitle",
   "sgl-deep-gemm==0.1.4",
   "sglang-kernel==0.4.4",
   "smg-grpc-servicer>=0.5.0",  "soundfile==0.13.1",
   "tiktoken",
-<<<<<<< DCU main@ec49eb80ae6b
-  "tilelang==0.1.9",
-||||||| official previous@f920a37da46e
-  "tilelang==0.1.8",
-=======
   "tilelang==0.1.11",
->>>>>>> official target@88db9e033a11
   "timm==1.0.16",
   "tokenspeed_mla==0.1.7",
   "torch==2.11.0",
   "torch_memory_saver>=0.0.9.post1",
   "torchao==0.17.0",
   "torchaudio==2.11.0",
   "torchcodec==0.11.1 ; sys_platform != 'linux' or (sys_platform == 'linux' and platform_machine != 'aarch64' and platform_machine != 'arm64' and platform_machine != 'armv7l')", # torchcodec 0.11.1 for torch 2.11.x (0.10 is ABI-incompatible: references the pre-2.11 c10::MessageLogger ctor signature). Not available on Linux ARM.
   "torchvision",
~~~~

</details>


<details>
<summary><code>python/sglang/jit_kernel/csrc/add_constant.cuh</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Adopt the official kDLGPU TensorMatcher dispatch, which maps to kDLROCM under HIP and kDLCUDA otherwise.

~~~~diff
--- AUTO-CONFLICT/python/sglang/jit_kernel/csrc/add_constant.cuh
+++ RESOLVED/python/sglang/jit_kernel/csrc/add_constant.cuh
@@ -57,40 +57,21 @@
 // You can also use struct with static method as an alternative
 template <int32_t kConstant>
 void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
   using namespace host;

   // 1. Validate input tensors
   SymbolicSize N = {"num_elements"};
   SymbolicDevice device_;
-<<<<<<< DCU main@ec49eb80ae6b
-#ifdef USE_ROCM
-  device_.set_options<kDLROCM>();
-#else
-  device_.set_options<kDLCUDA>();
-#endif
-  TensorMatcher({N})          // 1D tensor, must be contiguous
-      .with_dtype<int32_t>()  // must be int32
-      .with_device(device_)
-      .verify(dst)   // check tensor dst
-      .verify(src);  // check tensor src
-||||||| official previous@f920a37da46e
-  TensorMatcher({N})                  // 1D tensor, must be contiguous
-      .with_dtype<int32_t>()          // must be int32
-      .with_device<kDLCUDA>(device_)  // must be on CUDA device
-      .verify(dst)                    // check tensor dst
-      .verify(src);                   // check tensor src
-=======
   TensorMatcher({N})                 // 1D tensor, must be contiguous
       .with_dtype<int32_t>()         // must be int32
       .with_device<kDLGPU>(device_)  // must be on GPU device (CUDA or ROCm)
       .verify(dst)                   // check tensor dst
       .verify(src);                  // check tensor src
->>>>>>> official target@88db9e033a11

   // 2. Extract required parameters, prepare for kernel launch
   const size_t num_elements = N.unwrap();
   const DLDevice device = device_.unwrap();
   [[maybe_unused]]  // optional, can be omitted
   const size_t dynamic_smem = 0;
   [[maybe_unused]]  // optional, LaunchKernel can auto determine stream from device
   const auto stream = LaunchKernel::resolve_device(device);
~~~~

</details>


<details>
<summary><code>python/sglang/jit_kernel/dsv4/elementwise.py</code> — 2 conflict hunk(s)</summary>

**Resolution intent:** Combine official XPU dispatch with the retained DCU-specific JIT module selection and cache behavior.

~~~~diff
--- AUTO-CONFLICT/python/sglang/jit_kernel/dsv4/elementwise.py
+++ RESOLVED/python/sglang/jit_kernel/dsv4/elementwise.py
@@ -3,33 +3,23 @@
 import torch

 from sglang.jit_kernel.utils import (
     cache_once,
     is_arch_support_pdl,
     load_jit,
     make_cpp_args,
 )
-<<<<<<< DCU main@ec49eb80ae6b
-from sglang.srt.utils import is_dcu, is_hip
-||||||| official previous@f920a37da46e
-from sglang.srt.utils import is_hip
-=======
-from sglang.srt.utils import is_hip, is_xpu
->>>>>>> official target@88db9e033a11
+from sglang.srt.utils import is_dcu, is_hip, is_xpu

 from .utils import make_name

 _is_hip = is_hip()
-<<<<<<< DCU main@ec49eb80ae6b
 _is_dcu = is_dcu()
-||||||| official previous@f920a37da46e
-=======
 _is_xpu = is_xpu()
->>>>>>> official target@88db9e033a11


 @cache_once
 def _jit_fused_rope_module():
     args = make_cpp_args(is_arch_support_pdl())
     return load_jit(
         make_name("fused_rope"),
         *args,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/environ.py</code> — 2 conflict hunk(s)</summary>

**Resolution intent:** Preserve DCU FlashMLA, FP4/FP8, and SWA controls while accepting the official sparse-prefill default and its HIP/DCU model-default override.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/environ.py
+++ RESOLVED/python/sglang/srt/environ.py
@@ -895,39 +895,28 @@
     SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
     SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
     SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
     SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
     # Deprecated: DSV4 compressor V2 is always used.
     SGLANG_OPT_USE_COMPRESSOR_V2 = EnvBool(True)
     SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
     SGLANG_TOPK_TRANSFORM_512_TORCH = EnvBool(False)
-<<<<<<< DCU main@ec49eb80ae6b
     SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("kernel")
     SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA = EnvBool(False)
     SGLANG_HACK_SKIP_FP4_FP8_GEMM = EnvBool(False)
     SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(False)
-||||||| official previous@f920a37da46e
-    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(False)
-=======
     SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)
->>>>>>> official target@88db9e033a11

     # SWA radix cache
     # TODO(DSV4): @ispobock this has bug on main branch when retract
     SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
     SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
     SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)
-<<<<<<< DCU main@ec49eb80ae6b
-    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(False)
     SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = EnvBool(False)
-||||||| official previous@f920a37da46e
-    SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = EnvBool(False)
-=======
->>>>>>> official target@88db9e033a11

     # Unified radix cache
     SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(False)

     # DeepGemm Mega MoE
     SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE = EnvBool(False)
     SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/deepseek_v4_backend.py</code> — 3 conflict hunk(s)</summary>

**Resolution intent:** Combine official XPU support with DCU LightOp controls and the validated DCU prefill/decode split; do not let the generic FlashMLA tail become a DCU fallback.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/deepseek_v4_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -51,23 +51,17 @@
 from sglang.srt.layers.attention.dsv4.quant_k_cache import (
     quant_to_nope_fp8_rope_bf16_pack_lightop,
     quant_to_nope_fp8_rope_bf16_pack_triton,
 )
 from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
 from sglang.srt.runtime_context import get_parallel
 from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
-<<<<<<< DCU main@ec49eb80ae6b
-from sglang.srt.utils import ceil_align, get_bool_env_var, is_dcu
-||||||| official previous@f920a37da46e
-from sglang.srt.utils import ceil_align
-=======
-from sglang.srt.utils import ceil_align, is_xpu
->>>>>>> official target@88db9e033a11
+from sglang.srt.utils import ceil_align, get_bool_env_var, is_dcu, is_xpu
 from sglang.srt.utils.common import is_sm120_supported

 _is_dcu = is_dcu()
 _use_dpskv4_lightop_quant_k_cache = get_bool_env_var(
     "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE"
 )

 if TYPE_CHECKING:
@@ -1806,57 +1800,34 @@
                 return self._forward_flash_mla_decode(
                     q=q,
                     swa_k_cache=swa_k_cache,
                     swa_page_indices=swa_page_indices,
                     swa_topk_lengths=swa_topk_lengths,
                     flashmla_metadata=flashmla_metadata,
                     attn_sink=attn_sink,
                     extra_k_cache=extra_k_cache,
-<<<<<<< DCU main@ec49eb80ae6b
                     extra_indices=extra_indices,
                     extra_topk_lengths=extra_topk_lengths,
                     compress_ratio=compress_ratio,
                     layer_id=layer_id,
                 )
-||||||| official previous@f920a37da46e
-                    extra_indices_in_kvcache=extra_indices,
-                    extra_topk_length=extra_topk_lengths,
-                )[0]
-            else:
-                import sgl_kernel.flash_mla as flash_mla
-=======
-                    extra_indices_in_kvcache=extra_indices,
-                    extra_topk_length=extra_topk_lengths,
-                )[0]
-            else:
-                if _is_xpu:
-                    from sgl_kernel import flash_mla_with_kvcache
-                else:
-                    from sgl_kernel.flash_mla import flash_mla_with_kvcache
->>>>>>> official target@88db9e033a11

-<<<<<<< DCU main@ec49eb80ae6b
             if forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
                 if _should_use_sparse_prefill(q, forward_batch):
                     return self._forward_prefill_sparse(
                         q=q,
                         layer_id=layer_id,
                         compress_ratio=compress_ratio,
                         forward_batch=forward_batch,
                         token_to_kv_pool=token_to_kv_pool,
                         core_attn_metadata=core_attn_metadata,
                         attn_sink=attn_sink,
                     )
                 return self._forward_flash_mla_prefill(
-||||||| official previous@f920a37da46e
-                o = flash_mla.flash_mla_with_kvcache(
-=======
-                o = flash_mla_with_kvcache(
->>>>>>> official target@88db9e033a11
                     q=q,
                     swa_k_cache=swa_k_cache,
                     swa_page_indices=swa_page_indices,
                     swa_topk_lengths=swa_topk_lengths,
                     flashmla_metadata=flashmla_metadata,
                     attn_sink=attn_sink,
                     extra_k_cache=extra_k_cache,
                     extra_indices=extra_indices,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/dsa_backend.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Adopt the official TRT-LLM FP8 KV all-gather helper while retaining the existing generic-HIP exclusion for DCU AITER imports.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa_backend.py
@@ -75,21 +75,16 @@
 if is_cuda() and not is_dcu():
     import deep_gemm

 if TYPE_CHECKING:
     from sglang.srt.layers.radix_attention import RadixAttention
     from sglang.srt.model_executor.model_runner import ModelRunner
     from sglang.srt.speculative.spec_info import SpecInput

-<<<<<<< DCU main@ec49eb80ae6b
-||||||| official previous@f920a37da46e
-
-=======
-
 def _all_gather_dsa_trtllm_fp8_kv(
     forward_batch: ForwardBatch,
     k: torch.Tensor,
     k_rope: torch.Tensor,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     kv_lora_rank = k.shape[-1]
     qk_rope_head_dim = k_rope.shape[-1]
     kv_dtype = k.dtype
@@ -98,17 +93,16 @@
         kv,
         get_parallel().attn_cp_size,
         forward_batch,
         torch.cuda.current_stream(),
     ).view(kv_dtype)
     return kv.split((kv_lora_rank, qk_rope_head_dim), dim=-1)


->>>>>>> official target@88db9e033a11
 _is_hip = is_hip()
 _is_dcu = is_dcu()

 if _is_hip:
     from sglang.srt.layers.attention.dsa.triton_kernel import get_valid_kv_indices
     from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/dsv4/compressor.py</code> — 3 conflict hunk(s)</summary>

**Resolution intent:** Retain DCU LightOp quant/store selection, accept the official always-V2 compressor contract, and remove obsolete legacy aliases and tuned-GEMM imports.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsv4/compressor.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsv4/compressor.py
@@ -25,50 +25,30 @@
 from sglang.srt.layers.linear import ReplicatedLinear
 from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
 from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
 from sglang.srt.mem_cache.deepseek_v4_compress_state import (
     CompressStatePool,
 )
 from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
 from sglang.srt.model_executor.forward_context import get_attn_backend
-from sglang.srt.models.deepseek_v2 import _is_hip
 from sglang.srt.runtime_context import get_parallel
-<<<<<<< DCU main@ec49eb80ae6b
-from sglang.srt.utils import add_prefix, get_bool_env_var, is_dcu, is_npu, set_weight_attrs
-||||||| official previous@f920a37da46e
-from sglang.srt.utils import add_prefix, get_bool_env_var, is_npu, set_weight_attrs
-=======
-from sglang.srt.utils import add_prefix, is_npu, set_weight_attrs
->>>>>>> official target@88db9e033a11
+from sglang.srt.utils import (
+    add_prefix,
+    get_bool_env_var,
+    is_dcu,
+    is_npu,
+    set_weight_attrs,
+)

 _is_dcu = is_dcu()
 _is_npu = is_npu()
-<<<<<<< DCU main@ec49eb80ae6b
 _use_dpskv4_lightop_quant_k_cache = get_bool_env_var(
     "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE"
 )
-if _is_dcu:
-    from lightop import op
-
-_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip and not _is_dcu
-_tgemm = None
-if _use_aiter:
-    from aiter.tuned_gemm import tgemm
-
-    _tgemm = tgemm
-||||||| official previous@f920a37da46e
-_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
-_tgemm = None
-if _use_aiter:
-    from aiter.tuned_gemm import tgemm
-
-    _tgemm = tgemm
-=======
->>>>>>> official target@88db9e033a11

 if TYPE_CHECKING:
     from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
     from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
     from sglang.srt.layers.rotary_embedding import RotaryEmbedding
     from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@@ -522,25 +502,8 @@
             x = cp_all_gather_rerange_output(
                 x,
                 get_attention_cp_size(),
                 forward_batch,
                 torch.cuda.current_stream(),
             )

         return get_attn_backend().forward_compress(self, x, forward_batch)
-<<<<<<< DCU main@ec49eb80ae6b
-
-
-# TODO: compatibility impl for dsv4 backend on HIP
-if _is_hip and not _is_dcu and not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
-    from sglang.srt.layers.attention.dsv4.compress_hip import (  # noqa: F811
-        CompressorHip as Compressor,
-    )
-||||||| official previous@f920a37da46e
-
-
-if _is_hip and not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
-    from sglang.srt.layers.attention.dsv4.compress_hip import (  # noqa: F811
-        CompressorHip as Compressor,
-    )
-=======
->>>>>>> official target@88db9e033a11
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/dsv4/indexer.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Add official CUDA-only non-paged indexer planning while preserving DCU detection and the gfx942/gfx95 gate that excludes gfx938 from unsupported AITER paged-MQA logits.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsv4/indexer.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsv4/indexer.py
@@ -26,26 +26,24 @@
 from sglang.srt.model_executor.forward_batch_info import ForwardMode
 from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
     is_in_breakable_cuda_graph,
 )
 from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
     is_in_tc_piecewise_cuda_graph,
 )
 from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
-<<<<<<< DCU main@ec49eb80ae6b
-from sglang.srt.utils import add_prefix, is_dcu, is_gfx95_supported, is_hip
+from sglang.srt.utils import (
+    add_prefix,
+    is_cuda,
+    is_dcu,
+    is_gfx95_supported,
+    is_hip,
+)
 from sglang.srt.utils.common import is_gfx942_supported, is_sm120_supported
-||||||| official previous@f920a37da46e
-from sglang.srt.utils import add_prefix, is_hip
-from sglang.srt.utils.common import is_sm120_supported
-=======
-from sglang.srt.utils import add_prefix, is_cuda, is_hip
-from sglang.srt.utils.common import is_sm120_supported
->>>>>>> official target@88db9e033a11

 if TYPE_CHECKING:
     from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
     from sglang.srt.layers.attention.dsv4.compressor import (
         CompressorBackendMixin,
     )
     from sglang.srt.layers.quantization import QuantizationConfig
     from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/fla/layernorm_gated.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Combine the official XPU Dynamo-safe device context with the DCU LightOp layer_norm_fwd_1pass_opt path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/fla/layernorm_gated.py
+++ RESOLVED/python/sglang/srt/layers/attention/fla/layernorm_gated.py
@@ -252,18 +252,28 @@
         raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
     # heuristics for number of warps
     num_warps = min(max(BLOCK_N // 256, 1), 8)
     # Calculate rows per block based on SM count
     rows_per_block = calc_rows_per_block(M, x.device)
     # Update grid to use rows_per_block
     grid = (cdiv(M, rows_per_block), ngroups)
     pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
-<<<<<<< DCU main@ec49eb80ae6b
-    with device_context(x.device):
+    # Workaround for PyTorch <= 2.12: torch.xpu.device is not Dynamo-compatible
+    # in that release — it creates a DynamoConfigPatchProxy that
+    # SourcelessBuilder cannot wrap, causing a hard error under
+    # torch.compile(fullgraph=True). The device context is a functional no-op
+    # for Triton kernel launches (device is determined by the tensor, not the
+    # surrounding context), so skip it when Dynamo is tracing.
+    device_ctx = (
+        nullcontext()
+        if x.device.type == "xpu" and torch.compiler.is_compiling()
+        else device_context(x.device)
+    )
+    with device_ctx:
         if not _use_prefill_layer_norm_fwd:
             _layer_norm_fwd_1pass_kernel[grid](
                 x,
                 out,
                 weight,
                 bias,
                 z,
                 mean,
@@ -303,83 +313,16 @@
                 BLOCK_N=BLOCK_N,
                 ROWS_PER_BLOCK=rows_per_block,
                 HAS_BIAS=bias is not None,
                 HAS_Z=z is not None,
                 NORM_BEFORE_GATE=norm_before_gate,
                 IS_RMS_NORM=is_rms_norm,
                 ACTIVATION=activation,
             )
-||||||| official previous@f920a37da46e
-    with device_context(x.device):
-        _layer_norm_fwd_1pass_kernel[grid](
-            x,
-            out,
-            weight,
-            bias,
-            z,
-            mean,
-            rstd,
-            x.stride(0),
-            out.stride(0),
-            z.stride(0) if z is not None else 0,
-            M,
-            group_size,
-            eps,
-            BLOCK_N=BLOCK_N,
-            ROWS_PER_BLOCK=rows_per_block,
-            HAS_BIAS=bias is not None,
-            HAS_Z=z is not None,
-            NORM_BEFORE_GATE=norm_before_gate,
-            IS_RMS_NORM=is_rms_norm,
-            num_warps=num_warps,
-            ACTIVATION=activation,
-            **pdl_kwargs,
-        )
-=======
-    # Workaround for PyTorch <= 2.12: torch.xpu.device is not Dynamo-compatible
-    # in that release — it creates a DynamoConfigPatchProxy that
-    # SourcelessBuilder cannot wrap, causing a hard error under
-    # torch.compile(fullgraph=True).  The device context is a functional no-op
-    # for Triton kernel launches (device is determined by the tensor, not the
-    # surrounding context), so we simply skip it when Dynamo is tracing.
-    # PyTorch main already has the proper fix (XPUDeviceVariable registered in
-    # torch/_dynamo/variables/ctx_manager.py analogous to CUDADeviceVariable).
-    # TODO: remove this branch once we upgrade from PyTorch 2.12.
-    device_ctx = (
-        nullcontext()
-        if x.device.type == "xpu" and torch.compiler.is_compiling()
-        else device_context(x.device)
-    )
-    with device_ctx:
-        _layer_norm_fwd_1pass_kernel[grid](
-            x,
-            out,
-            weight,
-            bias,
-            z,
-            mean,
-            rstd,
-            x.stride(0),
-            out.stride(0),
-            z.stride(0) if z is not None else 0,
-            M,
-            group_size,
-            eps,
-            BLOCK_N=BLOCK_N,
-            ROWS_PER_BLOCK=rows_per_block,
-            HAS_BIAS=bias is not None,
-            HAS_Z=z is not None,
-            NORM_BEFORE_GATE=norm_before_gate,
-            IS_RMS_NORM=is_rms_norm,
-            num_warps=num_warps,
-            ACTIVATION=activation,
-            **pdl_kwargs,
-        )
->>>>>>> official target@88db9e033a11
     return out, mean, rstd


 if _is_npu:
     from sgl_kernel_npu.fla.layernorm_gated import layer_norm_fwd_npu as _layer_norm_fwd


 def rms_norm_gated(
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/triton_backend.py</code> — 4 conflict hunk(s)</summary>

**Resolution intent:** Adopt official unified-pool deferred full/SWA location translation and remove superseded CPU-length plumbing while preserving the surrounding backend contract.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/triton_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/triton_backend.py
@@ -403,17 +403,16 @@
         )
         return kv_indptr

     def _update_decode_kv_buffers(
         self,
         bs: int,
         seq_lens: torch.Tensor,
         req_pool_indices: torch.Tensor,
-        seq_lens_cpu: Optional[torch.Tensor] = None,
     ):
         """Fill KV (and SWA) cuda-graph buffers for decode/idle mode.

         Returns ``(kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens)``
         where ``window_kv_lens`` is ``None`` when sliding-window is disabled and
         ``num_kv_splits_lens`` is the per-request length used to size kv splits
         (per-DCP-rank length clamped to >=1 when DCP is enabled, full seq_lens
         otherwise).
@@ -449,34 +448,26 @@
                 self.window_kv_indptr,
                 self.req_to_token,
                 self.sliding_window_size,
                 seq_lens,
                 req_pool_indices,
                 bs,
                 token_to_kv_pool=self.token_to_kv_pool,
                 window_kv_indices=self.cuda_graph_window_kv_indices,
-<<<<<<< DCU main@ec49eb80ae6b
-                kv_last_index_cpu=_window_kv_last_index_cpu(
-                    seq_lens_cpu, bs, self.sliding_window_size
-                ),
-||||||| official previous@f920a37da46e
-=======
                 skip_full_to_swa_translation=(self._translate_kv_loc is not None),
->>>>>>> official target@88db9e033a11
             )
         return kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens

     def _update_target_verify_buffers(
         self,
         bs: int,
         seq_lens: torch.Tensor,
         req_pool_indices: torch.Tensor,
         spec_info,
-        seq_lens_cpu: Optional[torch.Tensor] = None,
     ):
         """Fill all cuda-graph buffers for target_verify mode."""
         qo_indptr = self.qo_indptr[: bs + 1]
         qo_indptr[: bs + 1] = torch.arange(
             0,
             (1 + bs) * self.num_draft_tokens,
             step=self.num_draft_tokens,
             dtype=torch.int32,
@@ -498,19 +489,16 @@
                     self.window_kv_indptr,
                     self.req_to_token,
                     self.sliding_window_size,
                     seq_lens[:bs],
                     req_pool_indices,
                     bs,
                     token_to_kv_pool=self.token_to_kv_pool,
                     window_kv_indices=window_kv_indices,
-                    kv_last_index_cpu=_window_kv_last_index_cpu(
-                        seq_lens_cpu, bs, self.sliding_window_size
-                    ),
                 )
             )
         custom_mask = self.cuda_graph_custom_mask
         if (
             spec_info is not None
             and getattr(spec_info, "custom_mask", None) is not None
         ):
             custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
@@ -532,146 +520,60 @@

     def _update_draft_extend_buffers(
         self,
         bs: int,
         seq_lens: torch.Tensor,
         req_pool_indices: torch.Tensor,
         forward_mode: ForwardMode,
         spec_info: Optional[SpecInput],
-        seq_lens_cpu: Optional[torch.Tensor] = None,
     ):
         """Fill QO + KV cuda-graph buffers for draft_extend mode."""
         seq_lens = seq_lens[:bs]
         # V2 draft-extend fills num_draft_tokens per req; num_steps+1 only equals
         # that when topk == 1.
         num_tokens_per_bs = (
             self.num_draft_tokens
             if forward_mode.is_draft_extend_v2()
             else self.speculative_num_steps + 1
         )
         qo_indptr = self.qo_indptr[: bs + 1]
-<<<<<<< DCU main@ec49eb80ae6b
-        if forward_mode.is_draft_extend_v2():
-            # DRAFT_EXTEND_V2: seq_lens = prefix + extend (bumped by the draft-extend path).
-            # Triton extend kernel receives extend K/V as separate tensors, so
-            # kv_indptr/kv_indices must cover only the prefix portion.
-            # extend_seq_lens_tensor is only attached to spec_info at real
-            # replay (eagle_draft_extend_cuda_graph_runner.replay); during the
-            # capture-time warmup it's absent, so fall back to the default
-            # tokens-per-batch count. Clamp at 0 because
-            # padded rows (raw_bs..bs) leave seq_lens at the fill value (1)
-            # while extend_seq_lens stays at num_tokens_per_bs, which would
-            # otherwise produce negative kv_lens; padded rows reference
-            # reserved req-pool slot 0 and their output is discarded.
-            if (
-                spec_info is not None
-                and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
-            ):
-                extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
-            elif (
-                spec_info is not None
-                and getattr(spec_info, "extend_seq_lens_cpu", None) is not None
-            ):
-                extend_seq_lens = torch.as_tensor(
-                    spec_info.extend_seq_lens_cpu[:bs],
-                    dtype=torch.int32,
-                    device=seq_lens.device,
-                )
-            else:
-                extend_seq_lens = torch.full(
-                    (bs,),
-                    num_tokens_per_bs,
-                    dtype=torch.int32,
-                    device=seq_lens.device,
-                )
-            qo_indptr[0] = 0
-            qo_indptr[1 : bs + 1] = torch.cumsum(extend_seq_lens, dim=0)
-            kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)
-||||||| official previous@f920a37da46e
-        qo_indptr[: bs + 1] = torch.arange(
-            0,
-            bs * num_tokens_per_bs + 1,
-            step=num_tokens_per_bs,
-            dtype=torch.int32,
-            device=self.device,
-        )
-        # DRAFT_EXTEND_V2: seq_lens = prefix + extend (bumped on the draft-extend path).
-        # Triton extend kernel receives extend K/V as separate tensors, so
-        # kv_indptr/kv_indices must cover only the prefix portion.
-        # extend_seq_lens_tensor is only attached to spec_info at real
-        # replay (eagle_draft_extend_cuda_graph_runner.replay); during the
-        # capture-time warmup it's absent, so fall back to zeros (matches
-        # the pre-unification capture path in #26651). Clamp at 0 because
-        # padded rows (raw_bs..bs) leave seq_lens at the fill value (1)
-        # while extend_seq_lens stays at num_tokens_per_bs, which would
-        # otherwise produce negative kv_lens; padded rows reference
-        # reserved req-pool slot 0 and their output is discarded.
-        if (
-            spec_info is not None
-            and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
-        ):
-            extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
-=======
         qo_indptr[: bs + 1] = torch.arange(
             0,
             bs * num_tokens_per_bs + 1,
             step=num_tokens_per_bs,
             dtype=torch.int32,
             device=self.device,
         )
         # DRAFT_EXTEND_V2: kv_indptr/kv_indices cover only the prefix (extend K/V go
         # separately). Capture warmup lacks extend_seq_lens_tensor -> fall back to
         # zeros; clamp at 0 so padded rows (seq_lens==fill 1) don't go negative.
         if (
             spec_info is not None
             and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
         ):
             extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
->>>>>>> official target@88db9e033a11
         else:
-            # DRAFT_EXTEND_V1: seq_lens = prefix only.
-            qo_indptr[: bs + 1] = torch.arange(
-                0,
-                bs * num_tokens_per_bs + 1,
-                step=num_tokens_per_bs,
-                dtype=torch.int32,
-                device=self.device,
+            extend_seq_lens = torch.zeros(
+                bs, dtype=torch.int32, device=seq_lens.device
             )
-            kv_lens = seq_lens
+        kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)
         kv_indptr = self._fill_kv_indptr_and_indices(
             bs, kv_lens, req_pool_indices, self.cuda_graph_kv_indices
         )
-        if self.sliding_window_size is not None and self.sliding_window_size > 0:
-            _, _, _, self.cuda_graph_window_kv_offsets[:bs] = (
-                update_sliding_window_buffer(
-                    self.window_kv_indptr,
-                    self.req_to_token,
-                    self.sliding_window_size,
-                    seq_lens,
-                    req_pool_indices,
-                    bs,
-                    token_to_kv_pool=self.token_to_kv_pool,
-                    window_kv_indices=self.cuda_graph_window_kv_indices,
-                    kv_last_index_cpu=_window_kv_last_index_cpu(
-                        seq_lens_cpu, bs, self.sliding_window_size
-                    ),
-                )
-            )
         return qo_indptr, kv_indptr, num_tokens_per_bs

     def init_forward_metadata_out_graph(
         self,
         forward_batch: ForwardBatch,
         in_capture: bool = False,
     ):
         bs = forward_batch.batch_size
         req_pool_indices = forward_batch.req_pool_indices
         seq_lens = forward_batch.seq_lens
-        seq_lens_cpu = forward_batch.seq_lens_cpu
         forward_mode = forward_batch.forward_mode
         spec_info = forward_batch.spec_info

         if in_capture:
             assert forward_batch.encoder_lens is None, "Not supported"
             # Multi-step spec decode: kv buffers come from spec_info, not the
             # cuda-graph pool, so replay is not involved.
             if forward_mode.is_decode_or_idle() and spec_info is not None:
@@ -692,17 +594,16 @@
                     swa_attn_logits=self.cuda_graph_swa_attn_logits,
                 )
                 return

             self._apply_cuda_graph_metadata(
                 bs=bs,
                 req_pool_indices=req_pool_indices,
                 seq_lens=seq_lens,
-                seq_lens_cpu=seq_lens_cpu,
                 forward_mode=forward_mode,
                 spec_info=spec_info,
             )
             out_cache_loc_full_physical = self._translate_cuda_graph_shared_pool_locs(
                 forward_batch, bs
             )
             swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(forward_batch)
             self.forward_metadata = self._build_cuda_graph_forward_metadata(
@@ -712,17 +613,16 @@
                 swa_out_cache_loc,
                 out_cache_loc_full_physical,
             )
         else:
             self._apply_cuda_graph_metadata(
                 bs=bs,
                 req_pool_indices=req_pool_indices,
                 seq_lens=seq_lens,
-                seq_lens_cpu=seq_lens_cpu,
                 forward_mode=forward_mode,
                 spec_info=spec_info,
             )
             # Metadata view is reused from capture; just refill the buffers.
             self._translate_cuda_graph_shared_pool_locs(forward_batch, bs)
             self._fill_cuda_graph_swa_out_cache_loc(forward_batch)

     def _fill_cuda_graph_swa_out_cache_loc(
@@ -1230,50 +1130,48 @@
         else:
             raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")

     def _apply_cuda_graph_metadata(
         self,
         bs: int,
         req_pool_indices: torch.Tensor,
         seq_lens: torch.Tensor,
-        seq_lens_cpu: Optional[torch.Tensor],
         forward_mode: ForwardMode,
         spec_info: Optional[SpecInput],
     ):
         """Shared capture+replay body for the cuda-graph init path.

         Public entry: :py:meth:`init_forward_metadata_out_graph`.
         """
         # NOTE: encoder_lens expected to be zeros or None
         if forward_mode.is_decode_or_idle():
             assert spec_info is None, "Multi-step cuda graph init is not done here."
             _, _, window_kv_lens, num_kv_splits_lens = self._update_decode_kv_buffers(
-                bs, seq_lens, req_pool_indices, seq_lens_cpu
+                bs, seq_lens, req_pool_indices
             )
             self.get_num_kv_splits(
                 self.cuda_graph_num_kv_splits[:bs], num_kv_splits_lens[:bs]
             )
             if window_kv_lens is not None:
                 self.get_num_kv_splits(
                     self.cuda_graph_window_num_kv_splits[:bs], window_kv_lens[:bs]
                 )
         elif forward_mode.is_target_verify():
             bs = len(req_pool_indices)
             self._update_target_verify_buffers(
-                bs, seq_lens, req_pool_indices, spec_info, seq_lens_cpu
+                bs, seq_lens, req_pool_indices, spec_info
             )
         elif forward_mode.is_draft_extend_v2():
             self._update_draft_extend_buffers(
                 bs,
                 seq_lens,
                 req_pool_indices,
                 forward_mode,
                 spec_info,
-                seq_lens_cpu,
             )

         else:
             raise ValueError(
                 f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
             )

     def get_cuda_graph_seq_len_fill_value(self):
@@ -2093,41 +1991,27 @@
                 forward_batch.seq_lens[:bs],
             )

     def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
         for attn_backend in self.attn_backends:
             attn_backend.init_forward_metadata_in_graph(forward_batch)


-def _window_kv_last_index_cpu(seq_lens_cpu, bs, sliding_window_size):
-    if seq_lens_cpu is None:
-        return None
-    seq_lens_cpu = seq_lens_cpu[:bs]
-    if isinstance(seq_lens_cpu, torch.Tensor):
-        return int(seq_lens_cpu.clamp(max=sliding_window_size).sum())
-    return sum(min(int(seq_len), sliding_window_size) for seq_len in seq_lens_cpu)
-
-
 def update_sliding_window_buffer(
     window_kv_indptr,
     req_to_token,
     sliding_window_size,
     seq_lens,
     req_pool_indices,
     bs,
     device=None,
     token_to_kv_pool=None,
     window_kv_indices=None,
-<<<<<<< DCU main@ec49eb80ae6b
-    kv_last_index_cpu=None,
-||||||| official previous@f920a37da46e
-=======
     skip_full_to_swa_translation=False,
->>>>>>> official target@88db9e033a11
 ):
     """Fill window KV buffers for sliding-window attention.

     Pass ``window_kv_indices`` to write into a pre-allocated buffer (CUDA-graph
     path); omit it (or pass ``None``) to allocate a fresh tensor (eager path,
     requires ``device``).

     ``skip_full_to_swa_translation=True`` leaves ``window_kv_indices`` as VIRTUAL
@@ -2153,30 +2037,18 @@
         req_to_token,
         req_pool_indices,
         window_kv_lens,
         window_kv_indptr,
         window_kv_start_idx,
         window_kv_indices,
         req_to_token.stride(0),
     )
-<<<<<<< DCU main@ec49eb80ae6b
-    if hasattr(token_to_kv_pool, "translate_loc_from_full_to_swa"):
-        kv_last_index = (
-            kv_last_index_cpu
-            if kv_last_index_cpu is not None
-            else window_kv_indptr[-1]
-        )
-||||||| official previous@f920a37da46e
-    if hasattr(token_to_kv_pool, "translate_loc_from_full_to_swa"):
-        kv_last_index = window_kv_indptr[-1]
-=======
     if not skip_full_to_swa_translation and hasattr(
         token_to_kv_pool, "translate_loc_from_full_to_swa"
     ):
         kv_last_index = window_kv_indptr[-1]
->>>>>>> official target@88db9e033a11
         window_kv_indices[:kv_last_index] = (
             token_to_kv_pool.translate_loc_from_full_to_swa(
                 window_kv_indices[:kv_last_index]
             )
         )
     return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/mhc.py</code> — 2 conflict hunk(s)</summary>

**Resolution intent:** Accept the official cleanup of stale commented TileLang GEMM arguments; no DCU runtime branch changes.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/mhc.py
+++ RESOLVED/python/sglang/srt/layers/mhc.py
@@ -498,22 +498,16 @@
                     sqrsum_part[i, j] += x_frag[i, jj * 8 + j] * x_frag[i, jj * 8 + j]

             T.gemm(
                 x_frag,
                 fn_smem,
                 out_frag,
                 transpose_A=False,
                 transpose_B=True,
-<<<<<<< DCU main@ec49eb80ae6b
-                # wg_wait=0,
-||||||| official previous@f920a37da46e
-                wg_wait=0,
-=======
->>>>>>> official target@88db9e033a11
                 clear_accum=False,
             )
         sqrsum_l = T.alloc_fragment(token_block, T.float32)
         T.reduce_sum(sqrsum_part, sqrsum_l)
         for i in T.Parallel(token_block):
             sqrsum[px * token_block + i] = sqrsum_l[i]
         for i, j in T.Parallel(token_block, 32):
             if j < hc_mult3:
@@ -584,22 +578,16 @@
                         sq_part4[i, j] += v * v

                 T.gemm(
                     x_f,
                     fn_smem,
                     out_frag,
                     transpose_A=False,
                     transpose_B=True,
-<<<<<<< DCU main@ec49eb80ae6b
-                    # wg_wait=0,
-||||||| official previous@f920a37da46e
-                    wg_wait=0,
-=======
->>>>>>> official target@88db9e033a11
                     clear_accum=False,
                 )

             sq_l = T.alloc_fragment((token_block,), T.float32)
             T.reduce_sum(sq_part4, sq_l)

             for i in T.Parallel(token_block):
                 t = px * token_block + i
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/fused_moe_triton/layer.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Retain the DCU LightOp sum/mul/add selector while accepting official per-rank shared-slot and FP8-to-FP4 shared-expert loading changes.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
+++ RESOLVED/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -78,23 +78,17 @@
     print_info_once,
     round_up,
 )
 from sglang.srt.utils.custom_op import register_custom_op

 _is_hip = is_hip()
 _is_cpu_amx_available = cpu_has_amx_support()
 _is_cpu = is_cpu()
-<<<<<<< DCU main@ec49eb80ae6b
 _use_lightop_moe_sum_mul_add = get_bool_env_var("SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD")
-_is_npu = is_npu()
-||||||| official previous@f920a37da46e
-_is_npu = is_npu()
-=======
->>>>>>> official target@88db9e033a11
 _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


 def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
     a2a_backend = get_moe_a2a_backend()
     if (
         a2a_backend.is_none()
         or a2a_backend.is_megamoe()
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/topk.py</code> — 3 conflict hunk(s)</summary>

**Resolution intent:** Retain DCU LightOp grouped-top-k and EPLB/padded-token postprocessing, add the official return annotation, and drop the obsolete eager Kimi import.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/topk.py
+++ RESOLVED/python/sglang/srt/layers/moe/topk.py
@@ -186,30 +186,16 @@
                 topk_weights,
                 topk_ids,
                 renormalize,
             )

     except ImportError:
         fused_topk_deepseek = None

-<<<<<<< DCU main@ec49eb80ae6b
-    try:
-        from sgl_kernel import kimi_k2_moe_fused_gate
-    except ImportError:
-        pass
-
-||||||| official previous@f920a37da46e
-    try:
-        from sgl_kernel import kimi_k2_moe_fused_gate
-    except ImportError as e:
-        pass
-
-=======
->>>>>>> official target@88db9e033a11
 if _is_cuda or _is_hip or _is_xpu:
     from sgl_kernel import topk_softmax

     try:
         from sgl_kernel import topk_sigmoid
     except ImportError:
         pass
 if _use_aiter:
@@ -1537,27 +1523,19 @@
     correction_bias: torch.Tensor,
     topk: int,
     renormalize: bool,
     num_expert_group: Optional[int] = None,
     topk_group: Optional[int] = None,
     num_fused_shared_experts: int = 0,
     routed_scaling_factor: Optional[float] = None,
     apply_routed_scaling_factor_on_output: Optional[bool] = False,
-<<<<<<< DCU main@ec49eb80ae6b
     num_token_non_padded: Optional[torch.Tensor] = None,
     expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
-):
-
-||||||| official previous@f920a37da46e
-):
-
-=======
 ) -> Tuple[torch.Tensor, torch.Tensor]:
->>>>>>> official target@88db9e033a11
     num_tokens = gating_output.shape[0]
     num_experts = gating_output.shape[1]
     experts_per_group = (
         num_experts // num_expert_group if num_expert_group else num_experts
     )

     # topk for routed experts only (shared experts are appended separately below)
     topk_routed = topk - num_fused_shared_experts
@@ -1688,39 +1666,34 @@
     ):
         topk_weights, topk_ids = moe_fused_gate(
             gating_output.to(dtype=torch.float32),
             num_fused_shared_experts,
             routed_scaling_factor if routed_scaling_factor is not None else 1.0,
             True,
             apply_routed_scaling_factor_on_output,
         )
-<<<<<<< DCU main@ec49eb80ae6b
     elif _use_lightop:
         assert not apply_routed_scaling_factor_on_output, "Not implemented"
         topk_weights, topk_ids = torch.ops.sglang.moe_fused_gate_dcu(
             gating_output,
             correction_bias,
             num_expert_group,
             topk_group,
             topk,
-            num_fused_shared_experts,
+            num_fused_shared_experts,
             routed_scaling_factor,
         )
         if (expert_location_dispatch_info is not None) or (
             num_token_non_padded is not None
         ):
             topk_ids = _biased_grouped_topk_postprocess(
                 topk_ids, expert_location_dispatch_info, num_token_non_padded
             )
         return topk_weights, topk_ids
-||||||| official previous@f920a37da46e
-=======
-        return topk_weights, topk_ids
->>>>>>> official target@88db9e033a11
     else:
         num_experts = gating_output.shape[1]
         if _is_cuda and num_experts == 384 and num_expert_group == 1:
             # ===== TO BE REFACTORED ====
             _use_jit_bf16_gate = False
             if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                 from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/unquant.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Keep DCU W16A16 Marlin expert packing and fallback behavior ahead of the official NPU postprocess.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/unquant.py
+++ RESOLVED/python/sglang/srt/layers/quantization/unquant.py
@@ -346,17 +346,16 @@
                 )

             layer.w13_weight.data = layer.w13_weight.data.reshape(
                 layer.num_local_experts, *new_shape_w13
             )
             layer.w2_weight.data = layer.w2_weight.data.reshape(
                 layer.num_local_experts, *new_shape_w2
             )
-<<<<<<< DCU main@ec49eb80ae6b
         if (_is_dcu and _use_marlin_w16a16_moe and not _use_aiter_w16a16_moe
             and not self.use_deepep
             and not getattr(layer, "use_nn_moe", False)
             and not getattr(layer, "_marlin_w16a16_moe_packed", False)):
             w1 = layer.w13_weight
             w2 = layer.w2_weight
             N = w1.shape[1]
             if (w1.is_cuda and w2.is_cuda
@@ -425,20 +424,16 @@
                         layer.w13_weight = new_w1
                         layer.w2_weight = new_w2
                         layer._marlin_w16a16_moe_packed = True
                         return
                 except Exception:
                     # If packing dependencies are unavailable, fall back to the
                     # standard (non-Marlin) layouts.
                     pass
-||||||| official previous@f920a37da46e
-
-=======
->>>>>>> official target@88db9e033a11
         if _is_npu:
             for weight_name in ["w13_weight", "w2_weight"]:
                 weight = getattr(layer, weight_name)
                 weight.data = npu_format_cast(weight)

         return

     def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
~~~~

</details>


<details>
<summary><code>python/sglang/srt/model_executor/forward_batch_info.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Retain DCU residual-RMS INT8 fields while accepting the official decode-context-parallel type and lifecycle documentation.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/model_executor/forward_batch_info.py
+++ RESOLVED/python/sglang/srt/model_executor/forward_batch_info.py
@@ -509,30 +509,24 @@

     # For two-batch overlap
     tbo_parent_token_range: Optional[Tuple[int, int]] = None
     tbo_padded_len: Optional[int] = None
     tbo_children: Optional[List[ForwardBatch]] = None

     attn_cp_metadata: Optional[ContextParallelMetadata] = None

-<<<<<<< DCU main@ec49eb80ae6b
     # dcu only
     residual_rms_per_quant_int8: Optional[torch.Tensor] = None
     rms_quant_flag: bool = False

-    # For decode context parallel
-||||||| official previous@f920a37da46e
-    # For decode context parallel
-=======
     # For decode context parallel.
     # NOTE: DecodeContextParallelMetadata is imported under TYPE_CHECKING only (see the
     # import block above) — available for annotations but NOT bound at runtime in this
     # module. Import it from sglang.srt.layers.dcp.metadata if a runtime use is added.
->>>>>>> official target@88db9e033a11
     attn_dcp_metadata: Optional[DecodeContextParallelMetadata] = None

     # Decode context parallel KV write mask.
     dcp_kv_mask: Optional[torch.Tensor] = None

     # For ngram embedding
     ngram_embedding_info: Optional[NgramEmbeddingInfo] = None
~~~~

</details>


<details>
<summary><code>python/sglang/srt/speculative/draft_utils.py</code> — 3 conflict hunk(s)</summary>

**Resolution intent:** Accept official Ascend DeepSeek-V4 draft backends, retain the DCU exclusion from generic HIP draft routing, and remove the stale global ServerArgs import.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/speculative/draft_utils.py
+++ RESOLVED/python/sglang/srt/speculative/draft_utils.py
@@ -1,20 +1,12 @@
 import logging

-<<<<<<< DCU main@ec49eb80ae6b
-from sglang.srt.server_args import ServerArgs, get_global_server_args
-from sglang.srt.utils.common import is_blackwell, is_dcu, is_hip, is_musa, is_npu
-||||||| official previous@f920a37da46e
-from sglang.srt.server_args import ServerArgs, get_global_server_args
-from sglang.srt.utils.common import is_blackwell, is_hip, is_musa, is_npu
-=======
 from sglang.srt.server_args import ServerArgs
-from sglang.srt.utils.common import is_blackwell, is_hip, is_musa, is_npu
->>>>>>> official target@88db9e033a11
+from sglang.srt.utils.common import is_blackwell, is_dcu, is_hip, is_musa, is_npu

 logger = logging.getLogger(__name__)


 class DraftBackendFactory:
     def __init__(
         self,
         server_args: ServerArgs,
@@ -252,32 +244,24 @@

         return AscendAttnMultiStepDraftBackend(
             self.draft_model_runner, self.topk, self.speculative_num_steps
         )

     def _create_dsv4_decode_backend(self):
         # Decode here is the EAGLE multi-step draft decode path.
         if is_npu():
-<<<<<<< DCU main@ec49eb80ae6b
-            return self._create_ascend_decode_backend()
-        elif is_hip() and not is_dcu():
-||||||| official previous@f920a37da46e
-            return self._create_ascend_decode_backend()
-        elif is_hip():
-=======
             from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
                 DeepseekV4AscendMultiStepDraftBackend,
             )

             return DeepseekV4AscendMultiStepDraftBackend(
                 self.draft_model_runner, self.topk, self.speculative_num_steps
             )
-        elif is_hip():
->>>>>>> official target@88db9e033a11
+        elif is_hip() and not is_dcu():
             from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                 DeepseekV4MultiStepBackend,
             )
         else:
             from sglang.srt.layers.attention.deepseek_v4_backend import (
                 DeepseekV4MultiStepBackend,
             )

@@ -364,41 +348,32 @@

     def _create_flashmla_prefill_backend(self):
         from sglang.srt.layers.attention.flashattention_backend import (
             FlashAttentionBackend,
         )

         return FlashAttentionBackend(self.draft_model_runner, skip_prefill=False)

-
     def _create_dcumla_prefill_backend(self):
         logger.warning(
             "flashmla prefill backend is not yet supported for draft extend."
         )
         return None

     def _create_dsv4_prefill_backend(self):
         # On NPU the "dsv4" backend resolves to the Ascend V4 subclass; its
         # draft-extend path uses the registered DSV4 prefill backend.
         if is_npu():
-<<<<<<< DCU main@ec49eb80ae6b
-            return self._create_ascend_prefill_backend()
-        elif is_hip() and not is_dcu():
-||||||| official previous@f920a37da46e
-            return self._create_ascend_prefill_backend()
-        elif is_hip():
-=======
             from sglang.srt.layers.attention.attention_registry import (
                 ATTENTION_BACKENDS,
             )

             return ATTENTION_BACKENDS["dsv4"](self.draft_model_runner)
-        elif is_hip():
->>>>>>> official target@88db9e033a11
+        elif is_hip() and not is_dcu():
             from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                 DeepseekV4HipRadixBackend,
             )

             return DeepseekV4HipRadixBackend(
                 self.draft_model_runner, skip_prefill=False
             )
         from sglang.srt.layers.attention.deepseek_v4_backend import (
~~~~

</details>


<details>
<summary><code>python/sglang/srt/speculative/eagle_utils.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Accept the official Triton tree-build and greedy-verify helpers alongside the existing sgl-kernel route.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/speculative/eagle_utils.py
+++ RESOLVED/python/sglang/srt/speculative/eagle_utils.py
@@ -262,21 +262,16 @@
         tree_mask,
         positions,
         retrieve_index,
         retrieve_next_token,
         retrieve_next_sibling,
         draft_tokens,
     )

-<<<<<<< DCU main@ec49eb80ae6b
-||||||| official previous@f920a37da46e
-
-=======
-
 def sgl_build_tree_kernel_triton(
     parent_list: torch.Tensor,
     selected_index: torch.Tensor,
     verified_seq_len: torch.Tensor,
     tree_mask: torch.Tensor,
     positions: torch.Tensor,
     retrieve_index: torch.Tensor,
     retrieve_next_token: torch.Tensor,
@@ -349,17 +344,16 @@
         retrieve_next_sibling,
         target_predict,
         batch_size=batch_size,
         num_speculative_tokens=num_speculative_tokens,
         num_draft_tokens=num_draft_tokens,
     )


->>>>>>> official target@88db9e033a11
 def verify_tree_greedy_func(
     predicts: torch.Tensor,
     accept_index: torch.Tensor,
     accept_token_num: torch.Tensor,
     candidates: torch.Tensor,
     retrieve_index: torch.Tensor,
     retrieve_next_token: torch.Tensor,
     retrieve_next_sibling: torch.Tensor,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/speculative/triton_ops/cache_locs.py</code> — 2 conflict hunk(s)</summary>

**Resolution intent:** Keep the DCU kvcacheio assign-extend fast path ahead of generic HIP and add official XPU support to the generic Triton path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/speculative/triton_ops/cache_locs.py
+++ RESOLVED/python/sglang/srt/speculative/triton_ops/cache_locs.py
@@ -1,29 +1,24 @@
 from __future__ import annotations

 import torch
 import triton
 import triton.language as tl

-<<<<<<< DCU main@ec49eb80ae6b
 from sglang.srt.utils import (
     get_bool_env_var,
     is_cuda,
     is_dcu,
     is_hip,
     is_musa,
     is_npu,
+    is_xpu,
     next_power_of_2,
 )
-||||||| official previous@f920a37da46e
-from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2
-=======
-from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, next_power_of_2
->>>>>>> official target@88db9e033a11

 _is_cuda = is_cuda()
 _is_hip = is_hip()
 _is_dcu = is_dcu()
 _is_npu = is_npu()
 _is_musa = is_musa()
 _is_xpu = is_xpu()

@@ -374,17 +369,16 @@
     req_pool_indices: torch.Tensor,
     req_to_token: torch.Tensor,
     start_offset: torch.Tensor,
     end_offset: torch.Tensor,
     batch_size: int,
     draft_token_num: int,
     device,
 ) -> torch.Tensor:
-<<<<<<< DCU main@ec49eb80ae6b
     if _is_dcu:
         out_cache_loc = torch.empty(
             (batch_size * draft_token_num,),
             dtype=torch.int64,
             device=device,
         )
         if get_bool_env_var("SGLANG_ASSIGN_EXTEND_CACHE_LOCS", default="true"):
             dcu_assign_extend_cache_locs(
@@ -403,22 +397,17 @@
                 start_offset,
                 end_offset,
                 out_cache_loc,
                 req_to_token.shape[1],
                 next_power_of_2(batch_size),
             )
         return out_cache_loc

-    if _is_cuda or _is_hip or _is_musa:
-||||||| official previous@f920a37da46e
-    if _is_cuda or _is_hip or _is_musa:
-=======
     if _is_cuda or _is_hip or _is_musa or _is_xpu:
->>>>>>> official target@88db9e033a11
         out_cache_loc = torch.empty(
             (batch_size * draft_token_num,),
             dtype=torch.int64,
             device=device,
         )
         assign_extend_cache_locs[(batch_size,)](
             req_pool_indices,
             req_to_token,
~~~~

</details>


<details>
<summary><code>test/registered/quant/test_int8_kernel.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Retain the disabled DCU placeholder and add the official AMD nightly registration.

~~~~diff
--- AUTO-CONFLICT/test/registered/quant/test_int8_kernel.py
+++ RESOLVED/test/registered/quant/test_int8_kernel.py
@@ -3,32 +3,30 @@

 import torch

 from sglang.srt.layers.activation import SiluAndMul
 from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
 from sglang.srt.layers.moe.topk import TopKConfig, select_experts
 from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
 from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
-<<<<<<< DCU main@ec49eb80ae6b
-from sglang.test.ci.ci_register import register_cuda_ci, register_dcu_ci
+from sglang.test.ci.ci_register import (
+    register_amd_ci,
+    register_cuda_ci,
+    register_dcu_ci,
+)

 # DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
 register_dcu_ci(
     est_time=30,
     suite="stage-b-test-1-gpu-small-dcu",
     nightly=False,
     disabled="DCU CSV CI placeholder: INT8 kernel path needs BW1100 numeric validation before enabling.",
 )

-||||||| official previous@f920a37da46e
-from sglang.test.ci.ci_register import register_cuda_ci
-=======
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
->>>>>>> official target@88db9e033a11
 from sglang.test.test_utils import CustomTestCase

 register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
 register_amd_ci(est_time=15, suite="nightly-amd-kernel-1-gpu", nightly=True)


 def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
     """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
~~~~

</details>


<details>
<summary><code>test/run_suite.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Retain all DCU nightly suites and add the official XPU nightly suite list.

~~~~diff
--- AUTO-CONFLICT/test/run_suite.py
+++ RESOLVED/test/run_suite.py
@@ -227,36 +227,30 @@
         "nightly-8-npu-a3",
         "nightly-16-npu-a3",
         "full-1-npu-a3",
         "full-2-npu-a3",
         "full-4-npu-a3",
         "full-8-npu-a3",
         "full-16-npu-a3",
     ],
-<<<<<<< DCU main@ec49eb80ae6b
     HWBackend.DCU: [
         "nightly-dcu",
         "nightly-dcu-1-gpu",
         "nightly-dcu-4-gpu",
         "nightly-dcu-8-gpu",
         "nightly-dcu-accuracy",
         "nightly-dcu-perf",
         "nightly-dcu-vlm",
     ],
-    HWBackend.XPU: [],
-||||||| official previous@f920a37da46e
-    HWBackend.XPU: [],
-=======
     HWBackend.XPU: [
         "nightly-xpu-1-gpu",
         "nightly-xpu-2-gpu",
         "nightly-xpu-4-gpu",
     ],
->>>>>>> official target@88db9e033a11
 }


 OTHER_SUITES = {
     HWBackend.CPU: [
         "default",
     ],
     HWBackend.CUDA: [
~~~~

</details>
