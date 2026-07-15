# Official Main Catch-up 20260714 — Code Conflict Review

> Scope: only the 9 files that produced textual merge conflicts. Automatically merged files and later semantic-only fixes are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `52bf6e27831a1547b1f8eb58be5bf6c1508dc296`
- Common official base: `f49cbbd67dea602f8616892d2a9882c8c30ae942`
- Official endpoint (`theirs`): `7e229e2a817de7d59e919db7ab3809ab4a22e754`
- Resolved merge: `310560cc3595f0739c3fb047c9b99425075e1685`
- Reconstructed textual conflicts: 9 files, 11 hunks

Each section reconstructs Git's three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files


<details>
<summary><code>python/sglang/kernels/ops/moe/ep_moe_kernels.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Port the DCU EP-MoE scatter/gather and quant kernels to the canonical kernels namespace while accepting the official permutation helper.

~~~~diff
--- AUTO-CONFLICT/python/sglang/kernels/ops/moe/ep_moe_kernels.py
+++ RESOLVED/python/sglang/kernels/ops/moe/ep_moe_kernels.py
@@ -2151,27 +2153,24 @@
         x,
         x_scale,
         *x_scale.stride(),
         masked_m,
         output_scale,
         output,
         x.size(1),
         x.size(2),
         K_SCALE_BLOCK_SIZE=K_SCALE_BLOCK_SIZE,
         K_BLOCK_SIZE=K_BLOCK_SIZE,
         num_warps=8,
     )
-<<<<<<< DCU parent@52bf6e27831a
-from triton.language.extra import libdevice
-from typing import Optional
 @triton.jit
 def _per_token_quant_int8_one_kernel_opt(
     x_ptr,
     xq_ptr,
     scale_ptr,
     stride_x,
     stride_xq,
     N,
     T_dim,
     tokens_per_expert_ptr,
     BLOCK: tl.constexpr
 ):
@@ -2282,26 +2281,24 @@
             x_q,
             scales,
             stride_x=x.stride(-2),
             stride_xq=x_q.stride(-2),
             N=N,
             T_dim=T,
             tokens_per_expert_ptr=tokens_per_expert,
             BLOCK=BLOCK,
             num_warps=num_warps,
             num_stages=1,
         )
     return x_q, scales
-||||||| official previous@f49cbbd67dea
-=======


 def moe_permute(
     inputs: torch.Tensor,
     topk_ids: torch.Tensor,
     num_experts: int,
     use_int64_offset: bool = False,
     is_ep: bool = False,
     outputs: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     from sglang.jit_kernel.moe_permute_prepare import moe_permute_prepare

@@ -2355,13 +2352,12 @@
         outputs,
         src2dst,
         topk_ids,
         topk_weights,
         topk_ids.size(1),
         inputs.size(1),
         1.0 if routed_scaling_factor is None else routed_scaling_factor,
         BLOCK_SIZE=512,
     )

     assert outputs is not None
     return outputs
->>>>>>> official target@7e229e2a817d
~~~~

</details>


<details>
<summary><code>python/sglang/srt/configs/model_config.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Keep the DCU SlimQuant method names and add the official Humming quantization method.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/configs/model_config.py
+++ RESOLVED/python/sglang/srt/configs/model_config.py
@@ -1343,31 +1343,27 @@
             "fbgemm_fp8",
             "compressed_tensors",
             "compressed-tensors",
             "experts_int8",
             "w8a8_int8",
             "w8a8_fp8",
             "moe_wna16",
             "qoq",
             "w4afp8",
             "petit_nvfp4",
             "quark",
             "modelslim",
-<<<<<<< DCU parent@52bf6e27831a
             "slimquant_w4a8_marlin",
             "slimquant_marlin",
-||||||| official previous@f49cbbd67dea
-=======
             "humming",
->>>>>>> official target@7e229e2a817d
             "quark_mxfp4",
         ]
         compatible_quantization_methods = {
             "modelopt_fp8": ["modelopt"],
             "modelopt_fp4": ["modelopt"],
             "modelopt_mixed": ["modelopt"],
             "nvfp4_online": ["fp8"],
             "petit_nvfp4": ["modelopt"],
             "w8a8_int8": ["compressed-tensors", "compressed_tensors"],
             "w8a8_fp8": ["compressed-tensors", "compressed_tensors"],
             "auto-round-int8": ["compressed-tensors", "compressed_tensors"],
         }
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/dsa/dsa_indexer.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Add the official XPU Hadamard dispatch without allowing generic HIP or CUDA helpers to replace the DCU LightOp path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -344,33 +343,27 @@
         Return: Anything, since it will be passed to the attention backend
                 for further processing on sparse attention computation.
                 Don't assume it is the topk indices of the input logits.
         """


 def rotate_activation(x: torch.Tensor, apply_scale: bool = True) -> torch.Tensor:
     # DSV4 compressor kernels may return a non-bf16 staging dtype on DCU.
     # The older working dpskv4 branch intentionally allowed this path.
     # from sgl_kernel import hadamard_transform
     if not _is_dcu and _is_hip:
         from fast_hadamard_transform import hadamard_transform
-<<<<<<< DCU parent@52bf6e27831a
-    elif not _is_dcu:
-||||||| official previous@f49cbbd67dea
-    else:
-=======
     elif _is_xpu:
         from sgl_kernel import hadamard_transform
-    else:
->>>>>>> official target@7e229e2a817d
+    elif not _is_dcu:
         from sglang.jit_kernel.hadamard import hadamard_transform

     hidden_size = x.size(-1)
     assert (
         hidden_size & (hidden_size - 1)
     ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

     scale = hidden_size**-0.5 if apply_scale else 1.0

     if _is_dcu and _use_fast_hadamard_transform:
         return hadamard_transform(x, scale=scale)
     elif _is_dcu:
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/linear.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Retain the local RMS epsilon and accept the official with-bias state required by the refactored linear layer.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/linear.py
+++ RESOLVED/python/sglang/srt/layers/linear.py
@@ -588,30 +588,26 @@
         input_size: int,
         output_sizes: List[int],
         bias: bool = True,
         gather_output: bool = False,
         skip_bias_add: bool = False,
         params_dtype: Optional[torch.dtype] = None,
         quant_config: Optional[QuantizationConfig] = None,
         prefix: str = "",
         tp_rank: Optional[int] = None,
         tp_size: Optional[int] = None,
         use_presharded_weights: bool = False,
     ):
-<<<<<<< DCU parent@52bf6e27831a
-        self.eps = 1e-6 #TODO:use config.rms_norm_eps lizhigong
-||||||| official previous@f49cbbd67dea
-=======
+        self.eps = 1e-6  # TODO: use config.rms_norm_eps
         self.with_bias = bias
->>>>>>> official target@7e229e2a817d
         self.output_sizes = output_sizes
         if tp_rank is None:
             tp_rank = get_parallel().tp_rank
         if tp_size is None:
             tp_size = get_parallel().tp_size
         self.tp_rank, self.tp_size = tp_rank, tp_size
         assert all(output_size % tp_size == 0 for output_size in output_sizes)
         self.use_presharded_weights = use_presharded_weights
         super().__init__(
             input_size=input_size,
             output_size=sum(output_sizes),
             bias=bias,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/ep_moe/layer.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Use the new kernel namespaces and Humming selection while retaining DCU AITER as the non-deprecated local EP-MoE path.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/ep_moe/layer.py
+++ RESOLVED/python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -447,46 +448,38 @@
             top_k=top_k,
             hidden_size=hidden_size,
             intermediate_size=intermediate_size,
             layer_id=layer_id,
             num_fused_shared_experts=num_fused_shared_experts,
             params_dtype=params_dtype,
             quant_config=quant_config,
             prefix=prefix,
             activation=activation,
             routed_scaling_factor=routed_scaling_factor,
             **kwargs,
         )
-<<<<<<< DCU parent@52bf6e27831a
-        if _use_aiter or _is_npu:
-            self.deprecate_flag = False
-||||||| official previous@f49cbbd67dea
-        if _use_aiter:
-            self.deprecate_flag = True
-        elif _is_npu:
-            self.deprecate_flag = True
-=======
         is_humming = (
             get_moe_runner_backend().is_humming()
             or get_moe_runner_backend().is_auto()
             and quant_config is not None
             and quant_config.get_name() == "humming"
         )
         if is_humming:
             self.deprecate_flag = True
+        elif _is_dcu and _use_aiter:
+            self.deprecate_flag = False
         elif _use_aiter:
             self.deprecate_flag = True
         elif _is_npu:
             self.deprecate_flag = True
->>>>>>> official target@7e229e2a817d
         elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
             quant_config, Fp8Config
         ):
             self.deprecate_flag = True
         elif (
             deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
             and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
         ):
             self.deprecate_flag = True
         elif (
             get_moe_runner_backend().is_flashinfer_cutedsl()
             and quant_config is not None
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/__init__.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Register official Humming alongside both existing DCU SlimQuant configurations.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/__init__.py
+++ RESOLVED/python/sglang/srt/layers/quantization/__init__.py
@@ -91,31 +78,27 @@
     "gptq": GPTQConfig,
     "gptq_marlin": GPTQMarlinConfig,
     "moe_wna16": MoeWNA16Config,
     "compressed-tensors": CompressedTensorsConfig,
     "qoq": QoQConfig,
     "w4afp8": W4AFp8Config,
     "petit_nvfp4": PetitNvFp4Config,
     "fbgemm_fp8": FBGEMMFp8Config,
     "auto-round": AutoRoundConfig,
     "auto-round-int8": W8A8Int8Config,
     "modelslim": ModelSlimConfig,
     "quark_int4fp8_moe": QuarkInt4Fp8Config,
-<<<<<<< DCU parent@52bf6e27831a
     "slimquant_w4a8_marlin": SlimQuantW4A8Int8MarlinConfig,
     "slimquant_marlin": SlimQuantCompressedTensorsMarlinConfig,
-||||||| official previous@f49cbbd67dea
-=======
     "humming": HummingConfig,
->>>>>>> official target@7e229e2a817d
     "mxfp_w4a8": Mxfp4W4A8Config,
 }
 if QuarkConfig is not None:
     BASE_QUANTIZATION_METHODS["quark"] = QuarkConfig
     BASE_QUANTIZATION_METHODS["quark_mxfp4"] = QuarkConfig

 if is_cpu() or is_cuda() or (_is_mxfp_supported and is_hip()):
     BASE_QUANTIZATION_METHODS.update(
         {
             "mxfp4": Mxfp4Config,
         }
     )
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/fp8_utils.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Keep the DCU-only DeepGEMM import and drop the duplicate fake-op definition now owned by the canonical FP8 kernel module.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/fp8_utils.py
+++ RESOLVED/python/sglang/srt/layers/quantization/fp8_utils.py
@@ -167,44 +167,26 @@
     from sgl_kernel import fp8_scaled_mm

     from sglang.jit_kernel.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
     from sglang.srt.utils.patch_torch import register_fake_if_exists

     @register_fake_if_exists("sgl_kernel::fp8_scaled_mm")
     def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
         # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].
         M = mat_a.shape[-2]
         N = mat_b.shape[-1]
         return mat_a.new_empty((M, N), dtype=out_dtype)

-<<<<<<< DCU parent@52bf6e27831a
-    @register_fake_if_exists("sgl_kernel::fp8_blockwise_scaled_mm")
-    def _fp8_blockwise_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype):
-        # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].
-        M = mat_a.shape[-2]
-        N = mat_b.shape[-1]
-        return mat_a.new_empty((M, N), dtype=out_dtype)
-
 if _is_dcu:
     import deepgemm
-||||||| official previous@f49cbbd67dea
-    @register_fake_if_exists("sgl_kernel::fp8_blockwise_scaled_mm")
-    def _fp8_blockwise_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype):
-        # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].
-        M = mat_a.shape[-2]
-        N = mat_b.shape[-1]
-        return mat_a.new_empty((M, N), dtype=out_dtype)
-
-=======
->>>>>>> official target@7e229e2a817d

 use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")

 # Input scaling factors are no longer optional in _scaled_mm starting
 # from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
 TORCH_DEVICE_IDENTITY = None


 def use_rowwise_torch_scaled_mm():
     if _is_hip:
         # The condition to determine if it is on a platform that supports
         # torch._scaled_mm rowwise feature.
~~~~

</details>


<details>
<summary><code>python/sglang/srt/models/deepseek_v4.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Move generic RoPE and MHC imports to sglang.kernels while retaining DCU AITER TileLang MHC pre/post dispatch.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v4.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v4.py
@@ -1412,47 +1412,41 @@
                 forward_batch=forward_batch,
             )

         if x.shape[0] == 0:
             y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
             post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
             comb = torch.empty(
                 (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
             )
             return y, post, comb, False

         if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
-<<<<<<< DCU parent@52bf6e27831a
             if _is_dcu and _use_aiter_tilelang_mhc:
                 post, comb, y = mhc_pre_big_fuse(
                     residual=x,
                     fn=hc_fn,
                     mhc_scale=hc_scale,
                     mhc_base=hc_base,
                     rms_eps=self.rms_norm_eps,
                     mhc_pre_eps=self.hc_eps,
                     mhc_sinkhorn_eps=self.hc_eps,
                     mhc_post_mult_value=2.0,
                     sinkhorn_repeat=self.hc_sinkhorn_iters,
                     n_splits=16,
                 )
                 # AITER MHC pre does not fuse the decoder-layer RMSNorm.
                 norm_fused = False
             else:
-                from sglang.srt.layers.mhc import mhc_pre
-||||||| official previous@f49cbbd67dea
-            from sglang.srt.layers.mhc import mhc_pre
-=======
-            from sglang.kernels.ops.layernorm.mhc import mhc_pre
->>>>>>> official target@7e229e2a817d
+                from sglang.kernels.ops.layernorm.mhc import mhc_pre

                 norm_kwargs = {}
                 if norm is not None:
                     norm_kwargs["norm_weight"] = norm.weight.data
                     norm_kwargs["norm_eps"] = norm.variance_epsilon

                 post, comb, y = mhc_pre(
                     residual=x,
                     fn=hc_fn,
                     hc_scale=hc_scale,
                     hc_base=hc_base,
                     rms_eps=self.rms_norm_eps,
@@ -1527,46 +1521,30 @@
         comb: torch.Tensor,
     ):

         if x.shape[0] == 0:
             return torch.empty(
                 (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
             )

         if _is_npu:
             return torch.ops.custom.npu_hc_post(x, residual, post, comb)

         if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
-<<<<<<< DCU parent@52bf6e27831a
             if _is_dcu and _use_aiter_tilelang_mhc:
-                out = mhc_post_fwd(
-                    x,
-                    residual,
-                    post,
-                    comb,
-                )
-                return out
+                return mhc_post_fwd(x, residual, post, comb)
             else:
-                from sglang.srt.layers.mhc import mhc_post
+                from sglang.kernels.ops.layernorm.mhc import mhc_post
+
                 return mhc_post(x, residual, post, comb)
-                # return mhc_post_torch(x, residual, post, comb)
-||||||| official previous@f49cbbd67dea
-            from sglang.srt.layers.mhc import mhc_post
-
-            return mhc_post(x, residual, post, comb)
-=======
-            from sglang.kernels.ops.layernorm.mhc import mhc_post
-
-            return mhc_post(x, residual, post, comb)
->>>>>>> official target@7e229e2a817d

         elif _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_POST.get():
             from aiter.ops.mhc import mhc_post

             result = torch.empty_like(residual)
             mhc_post(result, x, residual, post, comb)
             return result

         assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
         assert post.shape == (x.shape[0], self.hc_mult)
         assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)
~~~~

</details>


<details>
<summary><code>python/sglang/srt/server_args.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Keep SlimQuant and LightOp CLI choices and add official Humming choices.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/server_args.py
+++ RESOLVED/python/sglang/srt/server_args.py
@@ -178,31 +177,27 @@
     "auto-round-int8",
     "compressed-tensors",  # for Ktransformers
     "modelslim",  # for NPU
     "mxfp_w4a8",  # for NPU W4A8 (MXFP4 weights + MXFP8 activations)
     "quark",  # AMD Quark quantizer (FP8 / MXFP4 / Int4FP8 etc.)
     "quark_int4fp8_moe",
     "quark_mxfp4",  # Online MOE + linear quantization.
     # Apple Silicon MLX backend — on-the-fly quantization of fp16 weights at load
     # time via mlx.nn.quantize. Only takes effect when SGLANG_USE_MLX=1.
     "mlx_q4",  # 4 bits, group_size=64 (mlx-community default)
     "mlx_q8",  # 8 bits, group_size=64
     "unquant",
-<<<<<<< DCU parent@52bf6e27831a
     "slimquant_w4a8_marlin",
     "slimquant_marlin",
-||||||| official previous@f49cbbd67dea
-=======
     "humming",
->>>>>>> official target@7e229e2a817d
 ]


 SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES = QUANTIZATION_CHOICES

 ATTENTION_BACKEND_CHOICES = [
     # Common
     "triton",
     "torch_native",
     "flex_attention",
     "dsa",
     "nsa",  # Deprecated alias for "dsa"
@@ -263,30 +258,26 @@
     "deep_gemm",
     "triton",
     "triton_kernel",
     "flashinfer_trtllm",
     "experimental_sgl_trtllm",
     "flashinfer_trtllm_routed",
     "flashinfer_cutlass",
     "flashinfer_mxfp4",
     "flashinfer_cutedsl",
     "cutlass",
     "aiter",
     "marlin",
-<<<<<<< DCU parent@52bf6e27831a
     "lightop",
-||||||| official previous@f49cbbd67dea
-=======
     "humming",
->>>>>>> official target@7e229e2a817d
 ]

 MOE_A2A_BACKEND_CHOICES = [
     "none",
     "deepep",
     "mooncake",
     "nixl",
     "mori",
     "ascend_fuseep",
     "flashinfer",
     "megamoe",
 ]
~~~~

</details>
