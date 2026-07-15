# v0.5.12_dev Forward-Port Step 2 Conflict Review

Date: 2026-07-15 CST

## Scope

- Current forward-port parent: `e7e06b77881d243291fdc29fc815c2da6b28e75e`
- Previous old endpoint: `8736a794acee8253019704cf00a901fd7ffcefbe`
- Step 2 old endpoint: `fde56844fca442108bf3d2c71cbdeacb4ddb8f08`
- Branch: `forward-port/v0.5.12-dev-20260715`
- Range: 68 full-graph commits, 56 non-merge commits, 130 old-range files
- Result before documentation: 127 staged files, 6,914 insertions and 544 deletions
- Textual conflicts: 32 files

## Conflict decisions

| Conflict file | Resolution |
|---|---|
| `.codespellrc` | Kept the current dictionary and added the old DCU/runtime terms. |
| `docs_new/index.mdx` | Kept the current official documentation removal; no obsolete landing page was revived. |
| `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` | Ported C16 masked BF16 KV writes into the current unified compressor and HIP plan representation. |
| `python/sglang/srt/layers/attention/dsv4/indexer.py` | Retained current DSA structure and selected LightOp TopK only for `_is_dcu` with its environment gate. |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | Retained the current backend contract; the old standalone branch was already represented by current platform dispatch. |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | Combined current quant-method dispatch with old DCU EP W4A16/DeepGEMM behavior. |
| `python/sglang/srt/layers/moe/mega_moe.py` | Integrated the old standalone DCU MegaMoE runtime into the current canonical helper implementation. |
| `python/sglang/srt/layers/moe/topk.py` | Preserved current routing and added the DCU LightOp sqrt-softplus custom operation. |
| `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` | Retained current scheme APIs and added DCU MegaMoE plus AITER ASM shuffle weight handling. |
| `python/sglang/srt/layers/quantization/fp8.py` | Retained current runner selection and added the DCU MegaMoE weight builder. |
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | Forwarded `ExpertDistributionReq` before local world-collective handling. |
| `python/sglang/srt/mem_cache/allocator/paged.py` | Kept the current allocator; no old duplicate allocation path was restored. |
| `python/sglang/srt/mem_cache/common.py` | Kept native DCU last-location protection and excluded DCU from generic HIP fallback. |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | Added optional `valid_mask` to the current DSV4 pool store API. |
| `python/sglang/srt/mem_cache/memory_pool.py` | Kept the current refactored pool structure; required DSV4 behavior lives in the specialized pool. |
| `python/sglang/srt/mem_cache/utils.py` | Re-exported the moved masked MLA store helper from the canonical kernel namespace. |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | Kept deleted; MegaMoE graph-token behavior was ported to `runner/decode_cuda_graph_runner.py`. |
| `python/sglang/srt/models/deepseek_v4.py` | Kept current DSV4 model structure and disabled multi-stream for DCU BF16 attention KV cache. |
| `python/sglang/srt/models/hunyuan_v3.py` | Combined current model/loader/stream APIs with old PD, EP, fused RMS/RoPE and DCU behavior. |
| `python/sglang/srt/models/hunyuan_v3_nextn.py` | Preserved the current NextN contract and matching old HY3 runtime behavior. |
| `python/sglang/srt/multimodal/processors/base_processor.py` | Retained current processor caching and replaced the old single-device IPC pool with a TP-device pool group. |
| `python/sglang/srt/server_args.py` | Preserved current argument groups; validated MegaMoE runtime and disabled graphs only for DCU `deep_gemm`, not standalone `megamoe`. |
| `python/sglang/srt/utils/cuda_ipc_transport_utils.py` | Extended current cached IPC transport to per-device pools, with target-device synchronization after copies. |
| `sgl-kernel/python/sgl_kernel/__init__.py` | Unioned current exports with DCU L2Norm/Kimi operator exports. |
| `sgl-kernel/setup_hip.py` | Added L2Norm sources while preserving the current HIP source manifest. |
| `sgl-kernel/setup_rocm.py` | Added L2Norm sources while preserving the current ROCm source manifest. |
| `test/registered/debug_utils/test_crash_dump.py` | Kept current assertions and DCU platform skips/registration. |
| `test/registered/debug_utils/test_soft_watchdog.py` | Kept current watchdog coverage and DCU registration/guards. |
| `test/registered/lora/test_lora_eviction_policy.py` | Combined current test behavior with DCU registration and model mapping. |
| `test/registered/rl/test_update_weights_from_distributed.py` | Retained current distributed-update contract and restored DCU engine kwargs through a local helper. |
| `test/registered/unit/mem_cache/test_radix_cache_slru_accuracy.py` | Kept current SLRU assertions and DCU registration/skip policy. |
| `test/registered/utils/test_type_based_dispatcher.py` | Kept current dispatcher coverage and DCU registration. |

## Refactor and `_is_dcu` audit

- The old MLA masked store in `mem_cache/utils.py` moved to
  `kernels/ops/kvcache/mla_buffer.py`. The canonical kernel now exposes the
  masked Triton launcher; `mem_cache/utils.py` only re-exports it.
- C16/BF16 prefill writes derive `valid_mask`, static output locations, and
  positions from `CompressorPrefillPlan.plan_c`. BF16 attention KV always uses
  its dedicated paged scatter path, even when the generic fused-store
  optimization is disabled, so it cannot fall through to FP8 packing.
- The removed `model_executor/cuda_graph_runner.py` remains absent. MegaMoE
  graph-token state was ported to
  `model_executor/runner/decode_cuda_graph_runner.py` and uses the current
  capture-mode and backend APIs.
- DCU MegaMoE retains standalone/deep-gemm runtime selection, CUDA-graph token
  buffers, PD behavior, and W8A8 builders. Current generic helper
  implementations are canonical instead of removed private `deep_gemm.mega`
  imports.
- HY3 was semantically ported into the current model and stream APIs. Current
  `get_stream`, final-layernorm mapping, and model behavior remain present; old
  PD/EP/fused RMS/RoPE behavior is retained without reviving removed APIs. The
  removed `get_attention_tp_rank/size` helpers were migrated to current
  `get_attn_tensor_model_parallel_rank/world_size`, and both HY3 modules pass a
  direct import smoke.
- Multi-modal IPC retains the current single-pool handle cache and adds one
  source pool per TP device. Each peer copy is synchronized on its target
  device before the proxy state is handed off.
- `_is_dcu` remains ahead of generic `_is_hip` for LightOp, DSV4 cache, AITER,
  MegaMoE, EP W4A16, quantization, and mem-cache behavior. The DeepSeek-V4
  multi-stream exclusion is limited to DCU BF16 attention KV cache; it does not
  reintroduce the rejected fused-qnorm multi-stream bypass.
- The optional `w4a16_marlin_weight` DeepGEMM symbol is now imported lazily only
  when the DCU-only `SGLANG_USE_MARLIN_W4A16_MOE_OPT` path is requested. A
  missing optional symbol no longer prevents every quantization module import.
- `SGLANG_USE_AITER_AG=0` remains set by the pure-TP runtime script.

## Validation evidence

- `git ls-files -u`: no output.
- Precise conflict-marker scan: no output.
- `git diff --cached --check`: passed.
- `python3 -m py_compile`: passed for all 113 changed Python files.
- Targeted Ruff `E9,F401,F811,F821,F841` on conflict and high-risk semantic
  files: passed.
- `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 276 DCU
  registered test files and the existing `test/registered/cpu/utils.py`
  warning.
- `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
  passed, 19 tests.
- `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`:
  passed with package `sglang-kernel`, zero unsupported CUDA calls, and 55
  replaced kernel launches.
- Runtime preflight found `zz-nmz26` occupied at VRAM 93% on all eight devices;
  it was not used. All eight `zz-nmz22` devices were VRAM 0% / HCU 0%, so
  `zz-nmz22` was selected.
- The first pure-TP attempt stopped before model loading because the staged
  W4A16 file imported unavailable optional DeepGEMM symbol
  `w4a16_marlin_weight` unconditionally. The single focused fix made that
  DCU-only optimization lazy and explicit; module import then passed with
  `_is_dcu=True` and the option disabled.
- The confirmation run loaded 46 shards, initialized DSV4 pools, captured all
  decode graph buckets from batch size 128 through 1, and reached readiness.
  `GET /health` returned HTTP 200 and one short `POST /generate` returned HTTP
  200 without a worker exit.
- The response remained empty with eight zero output IDs. This is the existing
  deferred NaN/accuracy issue, is not an accuracy pass, and remains non-blocking
  under the agreed startup/request gate.
- The service was stopped, port 10015 closed, and all eight `zz-nmz22` devices
  returned to VRAM 0% / HCU 0%.

## Integration decision

Step 2 passes the scoped static and pure-TP startup/request gates after one
focused runtime fix. Commit the exact old endpoint as the second no-ff merge
checkpoint and continue to Step 3.
