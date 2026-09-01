# Official Main Daily 20260721 — Conflict Review

> Scope: official `main` from the previous synced endpoint through the exact
> 2026-07-21 endpoint. This review records all 48 textual conflict files plus
> semantic-only HCU ports caused by upstream file moves.

## Comparison

- HCU parent (`ours`): `ccb9a976e9c3b7556fd6abbca0e4e251c187b678`
- Common official base: `7e229e2a817de7d59e919db7ab3809ab4a22e754`
- Official endpoint (`theirs`): `d6ef68881e263812d4901f632786015005c4d050`
- Official delta: 314 commits, 1,350 changed paths
- Resolved staged tree: 1,348 paths, 131,551 insertions, 25,672 deletions
- Integration branch: `sync/official-main-daily-20260721`

The official target owns the new file layout and API contracts. Existing HCU
behavior is retained through explicit `_is_hcu` dispatch, with generic HIP
kept separate where the implementations differ.

## Textual conflict decisions

| Conflict file | Final resolution |
| --- | --- |
| `.codespellrc` | Unioned the official dictionary update with internal HCU/Hygon terms. |
| `docs_new/index.mdx` | Accepted the official docs landing-page structure while retaining the internal project-specific content that is still reachable. |
| `python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_pull.cuh` | Kept deleted; its HCU pull behavior is ported to the new unified custom-allreduce JIT header. |
| `python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh` | Kept deleted; its HCU push behavior is ported to the new unified custom-allreduce JIT header. |
| `python/sglang/jit_kernel/dsv4/compress.py` | Accepted official DSV4 interfaces and retained HCU dtype/storage and kernel dispatch. |
| `python/sglang/jit_kernel/dsv4/moe.py` | Accepted the current runner signatures and preserved the HCU fused-MoE path. |
| `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` | Combined official helper changes with HCU HIP compatibility definitions. |
| `python/sglang/jit_kernel/utils/compile.py` | Kept official default device flags and restored HCU `hipcc`, gfx938, and FNUZ compile flags. |
| `python/sglang/kernels/ops/attention/fla/fused_recurrent.py` | Accepted the official FLA API while preserving platform-specific HIP/HCU dispatch. |
| `python/sglang/kernels/ops/attention/metadata.py` | Accepted official metadata unions and repaired the conflict-spliced type declaration. |
| `python/sglang/kernels/ops/speculative/cache_locs.py` | Accepted the canonical kernels namespace and retained the HCU request-to-token assignment kernel. |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | Kept current Mooncake transfer contracts and HCU cache-layout/heterogeneous-attention handling. |
| `python/sglang/srt/distributed/device_communicators/pynccl.py` | Accepted current collective contracts while retaining HCU/HIP communicator behavior. |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | Kept official workspace-based sparse prefill and one implementation only; HCU imports Hygon `flash_mla`, while other platforms use `sgl_kernel.flash_mla`. Preserved split prefill/decode and HCU FlashMLA paths. |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | Accepted official indexer refactors and preserved HCU LightOp, fused qnorm/RoPE, cache, and conservative multistream behavior. |
| `python/sglang/srt/layers/attention/dsv4/compressor.py` | Accepted official compressor contracts and retained HCU fused compressor output behavior. |
| `python/sglang/srt/layers/attention/dsv4/indexer.py` | Accepted official indexer structure; restored the HCU Triton scale-kernel imports and retained HCU indexer dispatch. |
| `python/sglang/srt/layers/attention/dsv4/metadata.py` | Accepted official sequence/page metadata and retained the HCU cache-layout fields and builders. |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | Restored official CP-v2 materialization, token-count resolution, and independent KV-write/FA-descale values; retained HCU fused cache-write and custom FlashAttention branches. |
| `python/sglang/srt/layers/attention/flashmla_backend.py` | Accepted official signatures and kept HCU FlashMLA dispatch isolated from generic HIP/CUDA. |
| `python/sglang/srt/layers/attention/triton_backend.py` | Accepted official attention contracts while preserving HCU-specific extend/decode behavior. |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | Accepted current MoE runner contracts and retained HCU LightOp/AITER/DeepEP paths ahead of generic HIP. |
| `python/sglang/srt/layers/quantization/__init__.py` | Merged official quantization registrations with internal HCU methods. |
| `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` | Accepted official compressed-tensors schemes and retained HCU scheme selection. |
| `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py` | Ported the HCU INT8-MoE implementation to the new `MoeRunner` API and restored the official NPU runner path. |
| `python/sglang/srt/layers/quantization/fp8.py` | Accepted official FP8 structure while preserving HCU FNUZ/LightOp behavior. |
| `python/sglang/srt/layers/quantization/unquant.py` | Combined HCU unquant backends with the official NPU method and current runner API. |
| `python/sglang/srt/managers/schedule_batch.py` | Accepted official scheduling contracts and moved HCU allocation calls to the new allocation modules. |
| `python/sglang/srt/mem_cache/common.py` | Accepted the upstream split of allocation helpers; HCU behavior is no longer anchored in this obsolete location. |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | Accepted current DSV4 pool APIs while preserving HCU page layout, FP8 storage, and auxiliary cache mappings. |
| `python/sglang/srt/mem_cache/memory_pool.py` | Accepted official pool refactors and retained HCU pool selection/export compatibility. |
| `python/sglang/srt/model_executor/model_runner.py` | Accepted official runner structure and retained HCU attention, quantization, stream, and memory-pool selection. |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | Kept deleted; model KV sizing/configuration behavior is ported to `pool_configurator.py`. |
| `python/sglang/srt/model_executor/pool_configurator.py` | Ported HCU KV-size and cache-layout calculations onto the new official configurator. |
| `python/sglang/srt/models/deepseek_v2.py` | Repaired a conflict splice between `prepare_qkv_latent` and `q_b_proj_forward`; retained HCU fused RMS quant and accepted the official q-b projection and CP-v1 contracts. |
| `python/sglang/srt/models/deepseek_v4.py` | Accepted official weight-layout/loading changes, retained the HCU scale-name fallback, and skipped incompatible post-load FP8 scale setup on HCU. |
| `python/sglang/srt/models/qwen3_vl.py` | Accepted the official model refactor while retaining current HCU-compatible multimodal paths. |
| `python/sglang/srt/multimodal/processors/base_processor.py` | Accepted official feature-pool sizing and retained current HCU processor behavior. |
| `python/sglang/srt/multimodal/processors/moss_vl.py` | Accepted current processor contracts while preserving internal model support. |
| `python/sglang/srt/server_args.py` | Restored official MXFP8, elastic-EP, and experimental SGL-Marlin validation blocks and retained all HCU validation/default methods. |
| `python/sglang/srt/speculative/draft_utils.py` | Kept HCU draft backends and added the missing module logger used by the HCU prefill warning. |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | Accepted official speculative-v2 state changes while retaining HCU backend selection. |
| `python/sglang/srt/utils/common.py` | Accepted official platform helpers and retained HCU detection/capability behavior. |
| `python/sglang/srt/utils/cuda_ipc_transport_utils.py` | Kept the HCU per-device memory-pool group and reconciled official consumer acknowledgements: normal TP acknowledges the selected pool; encoder-DP/cache-hit acknowledges each source pool once. |
| `sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu` | Kept deleted with the official qserve removal; no live reference remains. |
| `sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu` | Kept deleted with the official qserve removal; no live reference remains. |
| `test/registered/debug_utils/test_tensor_dump_forward_hook.py` | Accepted current test contracts and preserved the HCU-relevant hook coverage. |
| `test/registered/unit/managers/test_mm_process_config.py` | Accepted official multimodal process-config coverage and current fixture APIs. |

## Semantic-only HCU refactor audit

- Allocation helpers moved from `mem_cache/common.py` to
  `mem_cache/allocation.py`; the HCU allocation path was ported to the new
  module and all schedule-batch callers use the canonical location.
- Model KV sizing moved out of the deleted
  `model_runner_kv_cache_mixin.py` into `pool_configurator.py`; HCU sizing and
  cache-layout rules were ported there.
- Pull/push custom-allreduce headers were replaced by the unified JIT
  communicator/custom-allreduce sources. The HCU device behavior was folded
  into the new header rather than reviving deleted files.
- DSV4 sparse prefill was deduplicated after the textual resolution. The
  official `SparsePrefillWorkspace`, live `max_seq_len`, and extra-key page-size
  APIs are used on every platform; only the backend import is selected by
  `_is_hcu`.
- FlashAttention's new CP-v2 path and separate cache-write scales versus
  kernel descales were audited explicitly so a HCU branch cannot bypass the
  official refactor or consume an undefined scale.
- Added the missing `get_global_server_args` import for the retained Qwen3.5
  HCU KV-cache dtype selection. This was a semantic-only issue inherited from
  the pre-merge parent and exposed by the daily static audit.
- Searched the official rename/delete delta and the current/official/old HCU
  trees for `_is_hcu`, `is_hcu`, `hcu_`, `LightOp`, `flash_mla`, `AITER`,
  `DeepEP`, `DeepSeekV4`, and `SGLANG_USE_AITER_AG`. Generic `_is_hip` paths do
  not replace dedicated HCU paths in the high-risk areas reviewed here.
- The `SGLANG_USE_AITER_AG=0` workaround remains unchanged.

## Validation evidence

- `git ls-files -u`: zero entries.
- Precise conflict-marker scan: no `<<<<<<<`, exact `=======`, or `>>>>>>>`
  marker remains in changed code; long documentation separator lines were
  excluded as non-markers.
- `git diff --cached --check`: passed after normalizing the upstream CRLF-only
  `docs_new/docs/sglang-diffusion/models_with_pe.mdx` file.
- Python syntax: every changed Python file passed `python3 -m py_compile`.
- Targeted static names: Ruff is unavailable in the container, so an isolated
  `/tmp` pyflakes install was used without changing the project environment.
  Merge-introduced undefined names and the duplicate sparse-prefill definition
  were fixed; remaining reports are official conditional imports or the
  ServerArgs CLI annotation DSL.
- `python3 scripts/ci/hcu/verify_hcu_registration.py`: passed, 277 registered
  HCU files.
- `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
  passed.
- `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`:
  passed with package `sglang-kernel`, zero unsupported CUDA calls, and 56
  replaced launches.
- High-risk grouped imports reached GPU extension/plugin loading and then the
  child process terminated without a Python traceback; this is recorded as an
  environment-level import limitation, not as a passed module-import gate.
- Pure-TP runtime: blocked before launch. The `zz-nmz26` preflight showed all
  eight devices at VRAM 95% and HCU 25.0%--98.6%; no model process was started.
  The stopped `zz-nmz22` container was not started implicitly.
