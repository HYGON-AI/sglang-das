# Official Main Catch-up 20260712 — Code Conflict Review

> Scope: only the 2 files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `ef85596515098410395f504fb2928e3b28f3520b`
- Common official base: `e1d51be91f6be39e585756568a8f66b99ac2c512`
- Official endpoint (`theirs`): `82e7cdcff9aa5f49156c3ace73a826f30854ae91`
- Resolved merge: `dde320d3772f023256aeb50b51470fefea5cdcf5`
- Reconstructed textual conflicts: 2 files, 2 hunks

Each section reconstructs Git’s three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>python/sglang/srt/layers/attention/deepseek_v4_backend.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt official DSpark and ragged-verify graph metadata while retaining the DCU LightOp quant-cache environment dispatch and DCU-first platform selection.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/deepseek_v4_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -61,31 +61,25 @@
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
 from sglang.srt.runtime_context import get_parallel
 from sglang.srt.speculative.dspark_components.kernels.dspark_attn_metadata import (
     BuildBlockSeqLensCausal,
     BuildDsparkSwaPageIndices,
     ComputeDsparkWindowGather,
 )
 from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
-<<<<<<< DCU main@ef8559651509
-from sglang.srt.utils import ceil_align, get_bool_env_var, is_dcu, is_xpu
-||||||| official previous@e1d51be91f6b
-from sglang.srt.utils import ceil_align, is_xpu
-=======
 from sglang.srt.speculative.ragged_verify import (
     RaggedVerifyMode,
     compute_ragged_extend_lengths,
     compute_target_verify_graph_key,
     compute_uniform_extend_lengths,
     read_ragged_verify_mode,
     resolve_ragged_verify_layout,
 )
-from sglang.srt.utils import ceil_align, is_xpu
->>>>>>> official target@82e7cdcff9aa
+from sglang.srt.utils import ceil_align, get_bool_env_var, is_dcu, is_xpu
 from sglang.srt.utils.common import is_sm120_supported
 
 _is_dcu = is_dcu()
 _use_dpskv4_lightop_quant_k_cache = get_bool_env_var(
     "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE"
 )
 
 if TYPE_CHECKING:
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/mhc.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt official lazy TileLang loading and apply the retained ROCm bool-allocation patch on first real TileLang load while preserving DCU AITER MHC dispatch.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/mhc.py
+++ RESOLVED/python/sglang/srt/layers/mhc.py
@@ -10,106 +10,16 @@
 from sglang.jit_kernel.utils import is_arch_support_pdl
 from sglang.srt.environ import envs
 from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_round_robin_split
 from sglang.srt.layers.utils.common import strict_contiguous
 from sglang.srt.utils import get_bool_env_var, is_dcu
 
 logger = logging.getLogger(__name__)
 
-<<<<<<< DCU main@ef8559651509
-# Tilelang is optional on some platform images. Keep module importable, while
-# preserving DCU's ROCm TileLang patch when the real package is present.
-try:
-    import tilelang
-    import tilelang.language as T
-
-    tilelang.set_log_level("WARNING")
-
-    pass_configs = {
-        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
-        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
-    }
-except ImportError:
-
-    class _TilelangMissing:
-        """Stub so module-level @tilelang.jit and PassConfigKey accesses parse."""
-
-        def __getattr__(self, name):
-            if name == "jit":
-
-                def _jit(*_args, **_kwargs):
-                    def _wrap(fn):
-                        def _raise(*a, **k):
-                            raise RuntimeError(
-                                "tilelang is not installed; this kernel cannot run "
-                                "on the current platform"
-                            )
-
-                        return _raise
-
-                    return _wrap
-
-                return _jit
-            return _TilelangMissing()
-
-        def __call__(self, *_args, **_kwargs):
-            return _TilelangMissing()
-
-    tilelang = _TilelangMissing()
-    T = _TilelangMissing()
-    pass_configs = None
-||||||| official previous@e1d51be91f6b
-# Tilelang isn't packaged on every platform (notably Ascend NPU images) but
-# this module is imported transitively from deepseek_v4.py — module-load
-# must succeed even when tilelang is missing. The kernels themselves still
-# require tilelang at runtime; we replace the package with a stub that lets
-# `@tilelang.jit` decorations and `tilelang.PassConfigKey.*` references parse
-# without ImportError, and any actual call into the kernels raises a clear
-# message at execution time instead of crashing on import.
-try:
-    import tilelang
-    import tilelang.language as T
-
-    tilelang.set_log_level("WARNING")
-
-    pass_configs = {
-        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
-        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
-    }
-except ImportError:
-
-    class _TilelangMissing:
-        """Stub so module-level @tilelang.jit and PassConfigKey accesses parse."""
-
-        def __getattr__(self, name):
-            if name == "jit":
-
-                def _jit(*_args, **_kwargs):
-                    def _wrap(fn):
-                        def _raise(*a, **k):
-                            raise RuntimeError(
-                                "tilelang is not installed; this kernel cannot run "
-                                "on the current platform"
-                            )
-
-                        return _raise
-
-                    return _wrap
-
-                return _jit
-            return _TilelangMissing()
-
-        def __call__(self, *_args, **_kwargs):
-            return _TilelangMissing()
-
-    tilelang = _TilelangMissing()
-    T = _TilelangMissing()
-    pass_configs = None
-=======
 # This module is imported during model-registry discovery. Do not import the real
 # TileLang package here: it loads native CUDA stubs. The proxy below lets
 # module-level @tilelang.jit declarations parse, then imports and applies real
 # TileLang only when a TileLang MHC kernel is actually called.
 _real_tilelang = None
 _real_T = None
 _tilelang_load_lock = threading.Lock()
 
@@ -140,29 +50,64 @@
     # can use lazy TileLang enum values without changing the proxy.
     if isinstance(value, list):
         return [_resolve_lazy_tilelang_value(v) for v in value]
     if isinstance(value, tuple):
         return tuple(_resolve_lazy_tilelang_value(v) for v in value)
     return value
 
 
+def _patch_tilelang_decouple_type_cast_for_rocm() -> None:
+    if torch.version.hip is None:
+        return
+    try:
+        from tilelang.transform import decouple_type_cast as _dtc
+    except Exception:
+        return
+    if getattr(_dtc, "_sglang_rocm_bool_alloc_patch", False):
+        return
+
+    original_allocate = _dtc.Allocate
+
+    def _is_bool_expr(expr) -> bool:
+        try:
+            dtype = expr.dtype
+            if callable(dtype):
+                dtype = dtype()
+            return str(dtype) == "bool8"
+        except Exception:
+            return False
+
+    def _allocate(data, dtype, extents, condition, body, annotations=None, span=None):
+        if not _is_bool_expr(condition):
+            condition = _dtc.tir.const(1) == _dtc.tir.const(1)
+        if annotations is None:
+            return original_allocate(data, dtype, extents, condition, body)
+        if span is None:
+            return original_allocate(data, dtype, extents, condition, body, annotations)
+        return original_allocate(data, dtype, extents, condition, body, annotations, span)
+
+    _dtc.Allocate = _allocate
+    _dtc._sglang_rocm_bool_alloc_patch = True
+
+
 def _load_tilelang():
     global _real_tilelang, _real_T, tilelang, T
     if _real_tilelang is None:
         with _tilelang_load_lock:
             if _real_tilelang is None:
                 try:
                     new_tilelang = importlib.import_module("tilelang")
                     new_T = importlib.import_module("tilelang.language")
                 except ImportError as exc:
                     raise RuntimeError(
                         "tilelang is not installed; this kernel cannot run on the current platform"
                     ) from exc
                 new_tilelang.set_log_level("WARNING")
+                _patch_tilelang_decouple_type_cast_for_rocm()
                 tilelang = new_tilelang
                 T = new_T
                 _real_T = new_T
                 _real_tilelang = new_tilelang
     return _real_tilelang
 
 
 class _LazyTilelang:
@@ -196,62 +141,24 @@
 
 
 tilelang = _LazyTilelang()
 T = _LazyTilelangAttr()
 pass_configs = {
     tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
     tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
 }
->>>>>>> official target@82e7cdcff9aa
 
 # Set once mhc_pre() has compiled every n_splits bucket at startup.
 _mhc_pre_warmed = False
 
 _is_dcu = is_dcu()
 _use_aiter_tilelang_mhc = get_bool_env_var("SGLANG_ROCM_USE_AITER_TILELANG_MHC")
 if _is_dcu and _use_aiter_tilelang_mhc:
     from aiter.ops.tilelang import pre_big_fuse_tilelang
-
-
-def _patch_tilelang_decouple_type_cast_for_rocm() -> None:
-    if torch.version.hip is None:
-        return
-    try:
-        from tilelang.transform import decouple_type_cast as _dtc
-    except Exception:
-        return
-    if getattr(_dtc, "_sglang_rocm_bool_alloc_patch", False):
-        return
-
-    original_allocate = _dtc.Allocate
-
-    def _is_bool_expr(expr) -> bool:
-        try:
-            dtype = expr.dtype
-            if callable(dtype):
-                dtype = dtype()
-            return str(dtype) == "bool8"
-        except Exception:
-            return False
-
-    def _allocate(data, dtype, extents, condition, body, annotations=None, span=None):
-        if not _is_bool_expr(condition):
-            condition = _dtc.tir.const(1) == _dtc.tir.const(1)
-        if annotations is None:
-            return original_allocate(data, dtype, extents, condition, body)
-        if span is None:
-            return original_allocate(data, dtype, extents, condition, body, annotations)
-        return original_allocate(data, dtype, extents, condition, body, annotations, span)
-
-    _dtc.Allocate = _allocate
-    _dtc._sglang_rocm_bool_alloc_patch = True
-
-
-_patch_tilelang_decouple_type_cast_for_rocm()
 
 FP8 = "float8_e4m3"
 BF16 = "bfloat16"
 FP32 = "float32"
 INT32 = "int32"
 
 
 @tilelang.jit(pass_configs=pass_configs)
~~~~

</details>

