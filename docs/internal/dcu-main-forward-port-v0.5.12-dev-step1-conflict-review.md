# v0.5.12_dev Forward-Port Step 1 Conflict Review

Date: 2026-07-15 CST

## Scope

- Current-main parent: `65f3bd9426e51df40987516acd075b646b858cf6`
- Old common base: `d4c6831a107ac03bae80e353d170af15557e4443`
- Old endpoint: `8736a794acee8253019704cf00a901fd7ffcefbe`
- Branch: `forward-port/v0.5.12-dev-20260715`
- Range: 41 full-graph commits, 30 non-merge commits, 54 old-tree files
- Result before documentation: 48 staged files, 6,800 insertions and 350 deletions
- Textual conflicts: 20 files (19 content conflicts and one modify/delete)

## Conflict decisions

| Conflict file | Resolution |
|---|---|
| `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh` | Ported the DCU radix TopK early-exit and launch-bound behavior into the current JIT source. |
| `python/sglang/jit_kernel/deepseek_v4.py` | Kept deleted. The official tree replaced this obsolete wrapper; behavior is carried by current DSV4 modules. |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | Kept the current override architecture and ported the BF16 KV alias through `arg_groups/overrides.py`. |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py` | Preserved the current communicator contract and retained explicit DCU/ROCm diagnostic separation. |
| `python/sglang/srt/layers/attention/aiter_backend.py` | Kept current upstream signatures and retained explicit DCU logging/dispatch behavior. |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | Adapted FP8/BF16 KV detection to the current dtype-driven backend at both refactored call sites. |
| `python/sglang/srt/layers/attention/dsv4/compressor.py` | Ported BF16 KV-cache stores to the current compressor API and retained DCU symbols/imports. |
| `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` | Ported BF16 stores to the v2 compressor without replacing official structure. |
| `python/sglang/srt/layers/attention/dsv4/indexer.py` | Kept current DSA/DSV4 indexer structure; the old VMFault guard is already represented in the current TopK path. |
| `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | Kept the current compatibility shim; active DCU behavior was moved into current DSA paths. |
| `python/sglang/srt/layers/attention/nsa/triton_kernel.py` | Kept the compatibility shim and ported the dense DCU Hadamard fallback to `dsa/triton_kernel.py`. |
| `python/sglang/srt/layers/attention/nsa_backend.py` | Kept the current shim; no obsolete NSA backend implementation was revived. |
| `python/sglang/srt/layers/moe/topk.py` | Retained LightOp TopK and skipped the generic second EPLB remap for that DCU path, preserving MiMo routing semantics. |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | Ported BF16 pool layout, properties, store path, and dtype-aware allocation into the current pool API. |
| `python/sglang/srt/mem_cache/swa_memory_pool.py` | Kept the removed legacy path and moved the stale-page mapping fix to `mem_cache/allocator/swa.py`. |
| `python/sglang/srt/model_executor/pool_configurator.py` | Ported KV dtype and byte-sizing behavior to the official pool configurator. |
| `python/sglang/srt/models/deepseek_v4.py` | Kept the current model structure; related MiMo/DSV4 return and attention behavior is preserved in canonical modules. |
| `python/sglang/srt/server_args.py` | Retained current arguments/overrides and explicit DCU versus generic HIP messages; formatting was normalized after resolution. |
| `python/sglang/srt/speculative/eagle_info.py` | Kept current speculative metadata/API; no removed old contract was restored. |
| `python/sglang/srt/speculative/ngram_info.py` | Kept current speculative metadata/API; no removed old contract was restored. |

## Refactor and `_is_dcu` audit

- The old NSA implementation has become DSA compatibility shims. DCU imports,
  cached dense Hadamard fallback, and FlashMLA hooks were ported to
  `layers/attention/dsa*` rather than added back to deleted files.
- The old SWA pool file has moved into `mem_cache/allocator/swa.py`; the stale
  physical-to-logical page mapping fix is present at the new location.
- DSV4 BF16 KV behavior follows the current canonical
  `DeepSeekV4SingleKVPool`, token-pool store, compressor v1/v2, backend dtype,
  and pool-configurator interfaces.
- `_is_dcu` remains ahead of generic `_is_hip` for LightOp TopK, FlashMLA/DSA,
  AITER diagnostics, and DCU cache behavior. No generic HIP branch was allowed
  to silently replace a dedicated DCU implementation.
- `SGLANG_USE_AITER_AG=0` remains set by the runtime script.

## Validation evidence

- `git ls-files -u`: no output.
- Precise conflict-marker scan: no output.
- `git diff --cached --check`: passed.
- `python3 -m py_compile`: passed for all 35 changed Python files.
- Targeted Ruff `E9,F401,F811,F821` on conflict and semantic-port files:
  passed. Seven broad `F841` findings in `aiter_backend.py` and
  `deepseek_v4_memory_pool.py` pre-exist on the current-main parent and were not
  mixed into this step.
- `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 221 DCU
  registered files and the existing `test/registered/cpu/utils.py` warning.
- `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
  passed.
- `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`:
  passed.
- Immediately before runtime, all eight devices on both `zz-nmz22` and
  `zz-nmz26` reported `VRAM 0% / HCU 0.0%`; `zz-nmz22` was selected.
- The exact pure-TP command loaded all 46 shards, initialized DSV4 pools,
  captured every decode graph bucket from batch size 128 through 1, and reached
  service readiness. `GET /health` returned 200. A short `POST /generate`
  returned 200 with `finish_reason=length` and eight completion tokens without
  a worker exit.
- The response text remained empty and all eight output IDs were zero. This is
  the previously deferred accuracy/NaN issue, not an accuracy pass, and is
  non-blocking under the current function-first forward-port policy.
- The service was stopped after the request.
