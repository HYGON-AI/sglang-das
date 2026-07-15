# Official Main Catch-up 20260713 — Code Conflict Review

> Scope: only the 8 files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `71c4c42af24f7dda258df84b79995afa50db3af2`
- Common official base: `82e7cdcff9aa5f49156c3ace73a826f30854ae91`
- Official endpoint (`theirs`): `f49cbbd67dea602f8616892d2a9882c8c30ae942`
- Resolved merge: `b111d8bc66a6ecd8c386fe9110fcf411f9e67650`
- Reconstructed textual conflicts: 8 files, 8 hunks

Each section reconstructs Git’s three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>python/sglang/srt/layers/attention/dsa/index_buf_accessor.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Move the FP8 helper to the canonical kernels namespace while retaining DCU detection and AITER preshuffle dispatch.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa/index_buf_accessor.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa/index_buf_accessor.py
@@ -1,25 +1,17 @@
 from typing import TYPE_CHECKING
 
 import torch
 import triton
 import triton.language as tl
 
 from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
 from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
-<<<<<<< DCU main@71c4c42af24f
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
-from sglang.srt.utils import get_bool_env_var, is_hip, is_dcu
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
-from sglang.srt.utils import get_bool_env_var, is_hip
-=======
-from sglang.srt.utils import get_bool_env_var, is_hip
->>>>>>> official target@f49cbbd67dea
+from sglang.srt.utils import get_bool_env_var, is_dcu, is_hip
 
 _is_hip = is_hip()
 _is_dcu = is_dcu()
 _is_fp8_fnuz = is_fp8_fnuz()
 _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
 # aiter cp_gather kernel with preshuffle=True is only valid when the indexer
 # uses the page_size=64 preshuffle layout (i.e. when the matching MQA gluon path
 # is also enabled).
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/dsa/tilelang_kernel.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Follow the official FP8 helper move while retaining DCU TileLang dispatch and its adapter workaround.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py
@@ -1,26 +1,18 @@
 import functools
 from functools import lru_cache
 from typing import Any, Optional, Tuple
 
 import tilelang
 import tilelang.language as T
 import torch
 
-<<<<<<< DCU main@71c4c42af24f
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
-from sglang.srt.utils import is_gfx95_supported, is_hip, is_dcu
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
-from sglang.srt.utils import is_gfx95_supported, is_hip
-=======
 from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
-from sglang.srt.utils import is_gfx95_supported, is_hip
->>>>>>> official target@f49cbbd67dea
+from sglang.srt.utils import is_dcu, is_gfx95_supported, is_hip
 
 tilelang.set_log_level("WARNING")
 
 # Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
 # `out_idx` list in place when normalising negative indices to positive ones.
 # That breaks any @tilelang.jit factory that compiles two prim_funcs with
 # different param counts (e.g. our unified single/dual partial kernel) — the
 # second compile sees indices already-converted for the first's len(params)
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/ep_moe/layer.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt the canonical FP8 helper import while retaining DCU quant-method dispatch for LightOp, AITER, DeepGEMM, compressed tensors, and Quark.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/ep_moe/layer.py
+++ RESOLVED/python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -39,32 +39,24 @@
     DeepEPNormalCombineInput,
 )
 from sglang.srt.layers.moe.token_dispatcher.moriep import (
     MoriEPLLCombineInput,
     MoriEPNormalCombineInput,
 )
 from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
 from sglang.srt.layers.quantization.base_config import QuantizationConfig
-<<<<<<< DCU main@71c4c42af24f
 from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
     CompressedTensorsFusedMoEMethod,
 )
 from sglang.srt.layers.quantization.compressed_tensors.schemes import (
     NPUCompressedTensorsW4A16Int4DynamicMoE,
 )
 from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
 from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFp4MoE
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.quantization.fp8 import Fp8Config
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
-=======
-from sglang.srt.layers.quantization.fp8 import Fp8Config
->>>>>>> official target@f49cbbd67dea
 from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
 from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
 from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
     is_in_tc_piecewise_cuda_graph,
 )
 from sglang.srt.utils import (
     ceil_div,
     direct_register_custom_op,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt official batch-invariant and kernel imports while retaining the lmslim INT8 quantizer only on DCU.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
+++ RESOLVED/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
@@ -10,47 +10,47 @@
 
 from sglang.kernels.ops.quantization.fp8_kernel import (
     per_token_group_quant_fp8,
     scaled_fp8_quant,
     sglang_per_token_group_quant_fp8,
 )
 from sglang.kernels.ops.quantization.int8_kernel import (
     per_token_group_quant_int8,
-    # per_token_quant_int8,
+    per_token_quant_int8,
     sglang_per_token_group_quant_int8,
 )
-<<<<<<< DCU main@71c4c42af24f
-from lmslim.layers.gemm.int8_utils import per_token_quant_int8
-||||||| official previous@82e7cdcff9aa
-=======
 from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
 from sglang.srt.layers.moe.utils import get_moe_padding_size
->>>>>>> official target@f49cbbd67dea
 from sglang.srt.utils import (
     cpu_has_amx_support,
     get_bool_env_var,
     is_cpu,
     is_cuda,
+    is_dcu,
     is_hip,
     is_sm90_supported,
 )
 
 try:
     from triton.tools.tensor_descriptor import TensorDescriptor
 
     _support_tensor_descriptor = True
 except:
     _support_tensor_descriptor = False
 
 _is_hip = is_hip()
 _is_cuda = is_cuda()
+_is_dcu = is_dcu()
 _is_cpu_amx_available = cpu_has_amx_support()
 _is_cpu = is_cpu()
 _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+
+if _is_dcu:
+    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
 
 if _is_cuda:
     pass
 elif _is_cpu and _is_cpu_amx_available:
     pass
 elif _is_hip:
     pass
 
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Use the official INT8 kernel generically and retain the lmslim quantizer only on DCU.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
+++ RESOLVED/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
@@ -16,33 +16,30 @@
 from sglang.srt.layers.parameter import (
     ChannelQuantScaleParameter,
     ModelWeightParameter,
     PerTensorScaleParameter,
 )
 from sglang.srt.layers.quantization.compressed_tensors.schemes import (
     CompressedTensorsLinearScheme,
 )
-<<<<<<< DCU main@71c4c42af24f
-# from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
-from lmslim.layers.gemm.int8_utils import per_token_quant_int8
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
-=======
->>>>>>> official target@f49cbbd67dea
 from sglang.srt.layers.quantization.utils import requantize_with_max_scale
-from sglang.srt.utils import is_cuda, get_bool_env_var
+from sglang.srt.utils import get_bool_env_var, is_cuda, is_dcu
 from sglang.srt.layers.quantization.compressed_tensors import quant_ops as ops
 _use_fused_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")
 _use_fused_silu_mul_quant = get_bool_env_var("SGLANG_USE_FUSED_SILU_MUL_QUANT")
 
 __all__ = ["CompressedTensorsW8A8Int8", "NPUCompressedTensorsW8A8Int8"]
 
 from lmslim import quant_ops 
 _is_cuda = is_cuda()
+_is_dcu = is_dcu()
+if _is_dcu:
+    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
+
 if _is_cuda:
     from sgl_kernel import int8_scaled_mm
 # TODO: remove vllm deps
 from sglang.srt.utils import W8a8GetCacheJSON
 W8A8_TRITONJSON=W8a8GetCacheJSON()
 
 class CompressedTensorsW8A8Int8(CompressedTensorsLinearScheme):
 
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/w8a8_fp8.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt canonical FP8 kernel imports while retaining DCU MoE runner selection.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/w8a8_fp8.py
+++ RESOLVED/python/sglang/srt/layers/quantization/w8a8_fp8.py
@@ -1,32 +1,26 @@
 from __future__ import annotations
 
 from typing import TYPE_CHECKING, Any, Dict, List, Optional
 
 import torch
 from torch.nn.parameter import Parameter
 
-<<<<<<< DCU main@71c4c42af24f
+from sglang.kernels.ops.quantization.fp8_kernel import (
+    fp8_dtype,
+    is_fp8_fnuz,
+    per_token_group_quant_fp8,
+)
 from sglang.srt.layers.moe import (
     MoeRunner,
     MoeRunnerBackend,
     MoeRunnerConfig,
     get_moe_runner_backend,
 )
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
-=======
-from sglang.kernels.ops.quantization.fp8_kernel import (
-    fp8_dtype,
-    is_fp8_fnuz,
-    per_token_group_quant_fp8,
-)
-from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
->>>>>>> official target@f49cbbd67dea
 from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
 from sglang.srt.layers.moe.utils import get_moe_a2a_backend
 from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
 from sglang.srt.layers.quantization.base_config import (
     FusedMoEMethodBase,
     LinearMethodBase,
     QuantizationConfig,
     QuantizeMethodBase,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/w8a8_int8.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt the official INT8 kernel and retain the lmslim quantizer only on DCU.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/w8a8_int8.py
+++ RESOLVED/python/sglang/srt/layers/quantization/w8a8_int8.py
@@ -18,23 +18,16 @@
 from sglang.srt.layers.quantization.base_config import (
     FusedMoEMethodBase,
     LinearMethodBase,
     QuantizationConfig,
     QuantizeMethodBase,
 )
 from sglang.srt.layers.moe.utils import get_moe_runner_backend
 from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
-<<<<<<< DCU main@71c4c42af24f
-# from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
-from lmslim.layers.gemm.int8_utils import per_token_quant_int8
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
-=======
->>>>>>> official target@f49cbbd67dea
 from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
 from sglang.srt.runtime_context import get_parallel
 from sglang.srt.utils import (
     cpu_has_amx_support,
     is_cpu,
     is_cuda,
     is_dcu,
     is_host_cpu_arm64,
@@ -47,16 +40,19 @@
     from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
 from lmslim import quant_ops
 
 _is_cuda = is_cuda()
 _is_dcu = is_dcu()
 _is_cpu_amx_available = cpu_has_amx_support()
 _is_cpu = is_cpu()
 _is_cpu_arm64 = is_host_cpu_arm64()
+
+if _is_dcu:
+    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
 
 if _is_cuda:
     from sgl_kernel import int8_scaled_mm
 
     @register_fake_if_exists("sgl_kernel::int8_scaled_mm")
     def _int8_scaled_mm_abstract(
         mat_a,
         mat_b,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/models/deepseek_v2.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Move all retained DCU MLA FP8 helpers to the canonical kernels namespace while preserving the endpoint routing-bias fix.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v2.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v2.py
@@ -28,16 +28,22 @@
 import torch
 import torch.nn.functional as F
 from torch import nn
 from transformers import PretrainedConfig
 
 from sglang.jit_kernel.dsv4 import (
     silu_and_mul_clamp,
     silu_and_mul_contig_post_quant,
+)
+from sglang.kernels.ops.quantization.fp8_kernel import (
+    create_per_token_group_quant_fp8_output_scale,
+    fp8_dtype,
+    per_tensor_quant_mla_fp8,
+    per_token_group_quant_mla_deep_gemm_masked_fp8,
 )
 from sglang.kernels.ops.quantization.fp8_kernel import (
     create_per_token_group_quant_fp8_output_scale,
 )
 from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
 from sglang.srt.batch_overlap.two_batch_overlap import (
     MaybeTboDeepEPDispatcher,
     model_forward_maybe_tbo,
@@ -110,29 +116,16 @@
     filter_moe_weight_param_global_expert,
     has_per_rank_fused_shared_slots,
     is_deepep_class_backend,
     is_sbo_enabled,
     is_tbo_enabled,
 )
 from sglang.srt.layers.quantization.base_config import QuantizationConfig
 from sglang.srt.layers.quantization.fp8 import Fp8Config
-<<<<<<< DCU main@71c4c42af24f
-from sglang.srt.layers.quantization.fp8_kernel import (
-    create_per_token_group_quant_fp8_output_scale,
-    fp8_dtype,
-    per_tensor_quant_mla_fp8,
-    per_token_group_quant_mla_deep_gemm_masked_fp8,
-)
-||||||| official previous@82e7cdcff9aa
-from sglang.srt.layers.quantization.fp8_kernel import (
-    create_per_token_group_quant_fp8_output_scale,
-)
-=======
->>>>>>> official target@f49cbbd67dea
 from sglang.srt.layers.quantization.fp8_utils import (
     materialize_bpreshuffle_fp8_scale,
 )
 from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
     maybe_fuse_routed_scale_and_shared_add,
 )
 from sglang.srt.layers.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
 from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton
~~~~

</details>

