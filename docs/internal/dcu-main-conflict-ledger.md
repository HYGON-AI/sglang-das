# DCU Main Conflict Ledger

This ledger tracks conflicts and merge decisions while internal `main` catches
up with official SGLang `main`.

## Status Legend


| Status      | Meaning                                                         |
| ----------- | --------------------------------------------------------------- |
| `open`      | Conflict exists or owner decision is pending                    |
| `merged`    | Conflict resolution has been committed to the checkpoint branch |
| `validated` | Required validation passed                                      |
| `waived`    | Validation or issue is explicitly waived with reason            |

## Strategy Legend


| Strategy          | Meaning                                                |
| ----------------- | ------------------------------------------------------ |
| `ours`            | Keep DCU-side implementation                           |
| `theirs`          | Take official implementation                           |
| `manual merge`    | Combine both sides manually                            |
| `drop DCU patch`  | Remove DCU patch because official code supersedes it   |
| `port to new API` | Re-implement DCU behavior on top of official interface |

## Validation Recording Rule

The `Validation` column records completed mechanical or CI checks. Developer
owners should choose the concrete runtime commands for their area and fill the
manual result in the checkpoint notes.

Recommended format for manual validation results:

```text
Manual validation result:
- <area>: <passed / failed / waived>, command or CI job: <...>, evidence: <log / run id / issue>
```

## Recommended Validation Matrix

These are validation recommendations, not fixed commands. Use the closest
available internal CI, local model path, or targeted reproducer, then record the
actual result in the checkpoint note.


| Area                                 | Recommended validation content                                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `ci` / `test`                        | CI registration scan, workflow dry-run if available, PR label/rerun flow, suite partition generation, DCU runner/image selection          |
| `dependency`                         | DCU install flow, wheel build/import check, CUDA-only dependency leakage check,`pip` resolver sanity in the target container              |
| `env` / `server_args`                | CLI parse smoke, env var default/value override check, affected model auto-backend selection                                              |
| `mem_cache`                          | Radix/cache unit or smoke, CPU offload if touched, DCU FA KV layout path, SWA/hybrid cache path if touched                                |
| `scheduler` / `model_executor`       | Stage-a/stage-b scheduler smoke, split prefill, pipeline-parallel smoke, forward-batch init, request abort/retract if touched             |
| `attention`                          | Qwen dense smoke, FlashMLA smoke, DSV4 attention smoke, NSA/topk/index-cache smoke, sparse prefill/chunking if touched                    |
| `jit-kernel` / `sgl-kernel`          | targeted kernel compile/import, kernel unit or smoke whitelist, DSV4 JIT smoke, deleted-file reference scan                               |
| `quantization`                       | FP8/W8A8/W4A8/MXFP4 targeted smoke, AITER shuffle path if touched, quantized MoE smoke, accuracy spot check if weights change             |
| `moe`                                | Qwen3 MoE smoke, EP/TP combination smoke, DeepEP normal and low-latency paths, AITER path, groupgemm/marlin path if touched               |
| `deepep`                             | DeepEP small and large smoke, normal dispatch and low-latency dispatch, topk/dispatch/combine compatibility, BF16/FP8 dispatch mode check |
| `aiter`                              | AITER import/init, W8A8/W16A16 MoE path, custom allreduce if related, eager and cuda-graph path if graph behavior is touched              |
| `model` / `deepseek` / `deepseek-v4` | DeepSeek V2/V3/V4 startup, short request, MTP/NextN if touched, FP8/FP4 checkpoint path, DSV4 JIT and attention path                      |
| `speculative`                        | EAGLE/EAGLE3/MTP/frozen-KV smoke, draft/target worker path, idle batch path, cuda graph draft path if touched                             |

## Active Conflict Board


| Checkpoint          | Merge branch                      | Conflict file                                                                      | Area           | Owner | Strategy        | Reason                                                                                                                               | Risk   | Validation                                                            | Follow-up                                                                                          | Status    |
| ------------------- | --------------------------------- | ---------------------------------------------------------------------------------- | -------------- | ----- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------- |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `.github/workflows/_pr-test-stage.yml`                                             | ci             | Codex | theirs          | Official renamed`check-stage-health` to `check-pr-test-health`; this is a pure workflow action rename                                | low    | precise marker scan passed; DCU registration passed                   | None                                                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `.github/workflows/pr-states.yml`                                                  | ci             | Codex | manual merge    | Keep official`workflow_run` PR lookup and preserve DCU `/rerun-failed-ci` stale-run wording                                          | low    | precise marker scan passed; DCU registration passed                   | Verify real GitHub workflow behavior in CI dry-run                                                 | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/environ.py`                                                     | env            | Codex | manual merge    | Keep both DCU`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` and official `SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN` env switches                  | low    | syntax compile passed; DCU registration passed                        | None                                                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/mem_cache/memory_pool.py`                                       | mem_cache      | Codex | manual merge    | Preserve DCU FA KV layout copy/load while taking official`current_platform.synchronize()`                                            | medium | syntax compile passed; DCU registration passed                        | Add CPU-offload smoke on DCU FA layout when CI command is available                                | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/managers/scheduler_pp_mixin.py`                                 | scheduler      | Codex | manual merge    | Combine official`hc_hidden_size` fallback with DCU proxy hidden-state shape helper                                                   | medium | syntax compile passed; DCU registration passed                        | Pipeline-parallel smoke if available                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `test/registered/tokenizer/test_multi_detokenizer.py`                              | test           | Codex | theirs          | Official C01 renames CUDA suite from`stage-b-*` to `base-b-*`; AMD registration remains from DCU tree                                | low    | syntax compile passed; DCU registration passed                        | None                                                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/moe/ep_moe/layer.py`                                     | moe            | Codex | ours            | High-risk official MoE runner restructuring overlaps DCU DeepEP/AITER paths; preserve known DCU runtime for C01                      | high   | syntax compile passed; DCU registration passed                        | Owner should port official runner-core changes onto DCU path in later checkpoint                   | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/moe/moe_runner/aiter.py`                                 | aiter          | Codex | ours            | Official AITER runner refactor conflicts with DCU W8A8/W16A16 handling; avoid semantic rewrite in first checkpoint                   | high   | syntax compile passed; DCU registration passed                        | Create dedicated AITER merge task before enabling official runner-core behavior                    | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/quantization/unquant.py`                                 | quantization   | Codex | ours            | Conflict is inside MoE execution fallback; keep DCU behavior until AITER/MoE owner ports official path                               | high   | syntax compile passed; DCU registration passed                        | Revisit together with`moe_runner/aiter.py`                                                         | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/models/deepseek_v2.py`                                          | model          | Codex | ours            | DeepSeek forward path overlaps DCU fused RMS/quant behavior; preserve current DCU model path in C01                                  | high   | syntax compile passed; DCU registration passed                        | Run DeepSeek V2/V3 smoke after C01; port official output-buffer context if needed                  | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/models/deepseek_v4.py`                                          | model          | Codex | ours            | DeepSeek V4 has many DCU-specific imports/kernels and official changes are not safe to fold blindly                                  | high   | syntax compile passed; DCU registration passed                        | DSV4 owner should review skipped official C01 hunks before C02/C03                                 | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | ours            | Preserve DCU speculative-algorithm alias helper; official side did not supersede it in C01                                           | medium | syntax compile passed; DCU registration passed                        | Confirm Gemma4 assistant draft CLI path still works if used internally                             | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `.github/workflows/pr-states.yml`                                                  | ci             | Codex | manual merge    | Keep official run status icon/link behavior and preserve DCU`/rerun-failed-ci` stale-run wording                                     | low    | marker scan passed; targeted compile passed; DCU registration passed  | Verify real GitHub workflow behavior in CI dry-run                                                 | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/pyproject.toml`                                                            | dependency     | Codex | ours            | Avoid adding unconditional CUDA`flashinfer_python[cu13]` and `flashinfer_cubin` dependencies to DCU install path                     | medium | marker scan passed; targeted compile passed; DCU registration passed  | Revisit with CUDA/Docker owner if internal main must match official CUDA dependency set exactly    | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh`                            | jit-kernel     | Codex | theirs          | File was deleted upstream and no current tree references remained                                                                    | medium | marker scan passed; targeted compile passed; DCU registration passed  | None                                                                                               | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh` | jit-kernel     | Codex | theirs          | File was deleted upstream and no current tree references remained                                                                    | medium | marker scan passed; targeted compile passed; DCU registration passed  | None                                                                                               | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/jit_kernel/deepseek_v4.py`                                          | jit-kernel     | Codex | manual merge    | Keep DCU BLASLt env path while adding official HIP/AITER imports and guard                                                           | medium | marker scan passed; targeted compile passed; DCU registration passed  | DSV4 JIT smoke on DCU                                                                              | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/dsv4/indexer.py`                               | attention      | Codex | manual merge    | Use official dynamic TOPK from output shape instead of fixed 512                                                                     | low    | marker scan passed; targeted compile passed; DCU registration passed  | DSV4 sparse prefill/topk smoke                                                                     | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/flashmla_backend.py`                           | attention      | Codex | theirs          | Official tuple-style forward mode check is equivalent and cleaner                                                                    | low    | marker scan passed; targeted compile passed; DCU registration passed  | FlashMLA/DSV4 smoke                                                                                | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`                     | attention      | Codex | manual merge    | Preserve DCU page-size 64 assertion while accepting official HIP preshuffle page-size check for non-DCU HIP                          | medium | marker scan passed; targeted compile passed; DCU registration passed  | NSA/DCU index cache smoke                                                                          | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`                            | attention      | Codex | manual merge    | Preserve DCU BF16 index-cache path and adopt official`device_index` budget API                                                       | high   | marker scan passed; targeted compile passed; DCU registration passed  | NSA topk/chunking smoke                                                                            | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/moe/ep_moe/layer.py`                                     | moe            | Codex | ours            | Official low-latency MoE runner updates overlap DCU AITER/DeepEP/groupgemm paths; keep known DCU implementation for C02              | high   | marker scan passed; targeted compile passed; DCU registration passed  | Dedicated MoE owner should port official C02 hunks separately                                      | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`                          | deepep         | Codex | ours            | Official DeepEP dispatcher API changes conflict with DCU quantized dispatch and low-latency dispatch parameters                      | high   | marker scan passed; targeted compile passed; DCU registration passed  | Confirm topk, BF16 dispatch, and low-latency dispatch compatibility before later checkpoints       | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/quantization/fp8.py`                                     | quantization   | Codex | ours            | Preserve DCU AITER/ASM FP8 MoE shuffle behavior; official shuffle path is CUDA/HIP-generic and needs DCU review                      | high   | marker scan passed; targeted compile passed; DCU registration passed  | Review with AITER/MoE owner                                                                        | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/managers/tp_worker.py`                                          | scheduler      | Codex | theirs          | Official fix avoids undefined`model_worker_batch` in split prefill sampling                                                          | low    | marker scan passed; targeted compile passed; DCU registration passed  | Split-prefill smoke if available                                                                   | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/model_executor/forward_batch_info.py`                           | model_executor | Codex | ours            | Preserve current DCU pinned-memory construction for extend lengths                                                                   | medium | marker scan passed; targeted compile passed; DCU registration passed  | Forward batch init smoke                                                                           | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/models/deepseek_v4.py`                                          | model          | Codex | manual merge    | Keep DCU FP8 WO-A GEMM shape compatibility and add official`ceil_to_ue8m0` scale conversion / rotary import                          | high   | marker scan passed; targeted compile passed; DCU registration passed  | DeepSeek V4 startup and short request smoke                                                        | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | theirs          | Official Gemma4 backend selection supports causal and conditional arch plus split backend validation                                 | low    | marker scan passed; targeted compile passed; DCU registration passed  | Server args unit smoke                                                                             | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/speculative/eagle_worker_v2.py`                                 | speculative    | Codex | theirs          | Official side fixes stale`model_worker_batch` references to use `batch`                                                              | medium | marker scan passed; targeted compile passed; DCU registration passed  | EAGLE/MTP smoke if available                                                                       | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/test/ci/ci_register.py`                                             | ci             | Codex | manual merge    | Keep DCU backend/marker and add official XPU backend/marker                                                                          | low    | marker scan passed; targeted compile passed; DCU registration passed  | DCU registration script                                                                            | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh`                          | jit-kernel     | Codex | theirs          | Official unified 512/1024 top-k into`topk_v1.cuh`; the standalone header is no longer referenced                                     | medium | reference scan passed; targeted compile passed                        | Run DSV4 top-k 512/1024 JIT smoke on DCU                                                           | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh`                            | jit-kernel     | Codex | manual merge    | Adopt official dynamic top-k kernel and preserve DCU`SGL_GRID_CONSTANT` plus HIP shared-memory attribute setup                       | high   | marker scan passed; targeted compile passed                           | Run DSV4 top-k 512/1024 compile and numeric smoke on DCU                                           | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/jit_kernel/deepseek_v4.py`                                          | jit-kernel     | Codex | port to new API | Remove the monolithic module and use official`sglang.jit_kernel.dsv4` split modules; model imports were ported                       | high   | deleted-file reference scan passed; targeted compile passed           | Validate DSV4 JIT, BF16-FP32 GEMM, compressor, and fused RoPE paths                                | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/disaggregation/mooncake/conn.py`                                | disaggregation | Codex | manual merge    | Keep DCU requests/ZMQ helpers and add official failed-session probe metrics counter                                                  | medium | marker scan passed; targeted compile passed                           | Mooncake reconnect and failed-session probe smoke                                                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/environ.py`                                                     | env            | Codex | manual merge    | Keep DCU FA KV-layout env and add official Mooncake failed-session probe env settings                                                | low    | DSA env/CLI alias test passed; targeted compile passed                | Verify DCU FA KV-layout env override in server startup                                             | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/dsv4/metadata.py`                              | attention      | Codex | port to new API | Import top-k planning from split`dsv4` module while retaining the non-HIP top-k-v2 enable condition                                  | high   | targeted compile passed; DSA alias test passed                        | DSV4 metadata planning and sparse top-k smoke                                                      | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/flashattention_backend.py`                     | attention      | Codex | port to new API | Preserve DCU fused cache-write guards and move pool access to official ForwardContext-owned backend state                            | high   | targeted compile passed; old ForwardBatch pool reference scan passed  | Qwen dense, MLA, and fused Qwen cache-store smoke                                                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`                     | attention      | Codex | theirs          | Official converted the old NSA file into a compatibility shim; DCU implementation was ported to`dsa/index_buf_accessor.py`           | high   | targeted compile passed; DSA alias test passed                        | DCU DSA index-cache gather/store smoke                                                             | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`                            | attention      | Codex | port to new API | Keep official compatibility shim and three-way port DCU BF16/FP8 indexer behavior into`dsa/dsa_indexer.py`                           | high   | targeted compile passed; DSA alias test passed                        | DSA decode, ragged prefill, chunking, BF16 index-cache, and FP8 index-cache smoke                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`                        | attention      | Codex | port to new API | Keep official compatibility shim and port DCU TileLang/gfx behavior into`dsa/tilelang_kernel.py`                                     | high   | targeted compile passed                                               | TileLang DSA kernel compile and numeric smoke                                                      | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/triton_kernel.py`                          | attention      | Codex | port to new API | Keep official compatibility shim and port DCU Triton helper kernels into`dsa/triton_kernel.py`                                       | high   | targeted compile passed                                               | DCU DSA Triton quant/gate helper smoke                                                             | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa_backend.py`                                | attention      | Codex | port to new API | Keep official compatibility shim and port DCU backend deltas into canonical`dsa_backend.py`                                          | high   | targeted compile passed; DSA registry test passed                     | DSA prefill/decode backend selection and MTP smoke                                                 | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/moe/ep_moe/layer.py`                                     | moe            | Codex | ours            | Preserve current DCU DeepEP/AITER/Marlin/group-GEMM implementation; official C03 mainly removes the legacy NPU path                  | high   | targeted compile passed; DCU registration passed                      | Dedicated owner must port relevant official runner changes; run DeepEP normal/LL and quantized MoE | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`                           | moe            | Codex | manual merge    | Preserve DCU extended forward arguments and add official Ascend FuseEP dispatch                                                      | high   | targeted compile passed                                               | DCU fused MoE, shared-output, and AITER/group-GEMM smoke                                           | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/managers/overlap_utils.py`                                      | scheduler      | Codex | manual merge    | Keep DCU empty-slice helper and add official speculative`seq_lens_cpu` resolution                                                    | medium | targeted compile passed                                               | Overlap scheduler plus speculative decode smoke                                                    | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/mem_cache/memory_pool.py`                                       | mem_cache      | Codex | port to new API | Rename NSA state to DSA, retain DCU BF16 index-cache/lightop store paths, and adopt official non-DCU HIP preshuffle rules            | high   | targeted compile passed; old NSA symbol scan passed                   | DSA cache write/read, retract/offload, BF16/FP8 index cache, and SWA cache smoke                   | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/mem_cache/memory_pool_host.py`                                  | mem_cache      | Codex | manual merge    | Use canonical DSA pool type while preserving DCU BF16 index-cache host sizing                                                        | high   | targeted compile passed; GPU unit collection blocked by no HIP GPU    | Run`test_dsa_pool_host_unit.py` on a DCU runner                                                    | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/deepseek_v2.py`                                          | model          | Codex | port to new API | Adopt DSA naming and ForwardContext access while retaining DCU fused MLA/cache paths                                                 | high   | targeted compile passed; old ForwardBatch context scan passed         | DeepSeek V2/V3.2 startup, DSA, CP, and short-request smoke                                         | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/deepseek_v4.py`                                          | model          | Codex | port to new API | Adopt official split JIT, DSA naming, ForwardContext, and fused cache-write structure while retaining DCU Q/RoPE helpers             | high   | targeted compile passed; old ForwardBatch context scan passed         | DeepSeek V4 startup, CP+EP/DP+EP, MTP, compressor, and FP8 WO-A smoke                              | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/qwen3_5.py`                                              | model          | Codex | manual merge    | Preserve DCU fused RMSNorm/RoPE/KV-store path and add official native/NPU prepare split using ForwardContext pool access             | high   | targeted compile passed                                               | Qwen3.5 dense/MoE short request with fused path on and off                                         | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/utils.py`                                                | model          | Codex | port to new API | Move fused KV-buffer eligibility to ForwardContext pool while preserving DCU exclusion from generic HIP fallback                     | medium | targeted compile passed                                               | Dense fused cache-store path and context-parallel exclusion smoke                                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | manual merge    | Preserve DCU page-size 64 behavior, adopt DSA naming and deprecated aliases, and restore official alias action import                | high   | 24 DSA CLI/registry/env tests passed                                  | DCU DSA backend auto-selection and page-size startup smoke                                         | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `test/registered/core/test_srt_engine.py`                                          | test           | Codex | manual merge    | Preserve DCU stage-b registration while taking official consolidated core test structure                                             | low    | DCU registration passed                                               | Execute the registered test on the normal DCU stage-b runner when enabled                          | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `test/registered/language/test_srt_backend.py`                                     | test           | Codex | theirs          | Official replaced the legacy backend suite with consolidated basic sanity kits                                                       | low    | deleted-file reference scan passed; DCU registration passed           | Decide whether`test_basic_sanity.py` should receive a DCU stage-a registration                     | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`                             | jit-kernel     | Codex | manual merge    | Adopt official`kDLGPU` device checks while retaining the C03 planner extensions                                                      | high   | static/registration passed; DCU runtime pending                       | DSV4 JIT planner compile and runtime smoke                                                         | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/dsv4/elementwise.py`                                     | jit-kernel     | Codex | manual merge    | Keep DCU uint8 JIT storage, use official sgl-kernel on non-DCU HIP, and retain official CUDA JIT                                     | high   | static/registration passed; DCU runtime pending                       | DSV4 FP8 elementwise numeric smoke on DCU                                                          | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh`            | jit-kernel     | Codex | ours            | Preserve the validated DCU HIP FP8 pack; official AMD sgl-kernel keeps its separate software conversion path                         | high   | static/registration passed; DCU runtime pending                       | DCU FP8 pack compile and numeric smoke                                                             | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh`                          | jit-kernel     | Codex | theirs          | Adopt official`kDLGPU` aliases and HIP runtime fallback required by the new JIT interface                                            | medium | static/registration passed; DCU runtime pending                       | Compile a DSV4 JIT module in the target DCU container                                              | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh`                            | jit-kernel     | Codex | manual merge    | Combine official device/memcpy additions with DCU launch, shuffle, sync, and shared-memory helpers                                   | high   | static/registration passed; DCU runtime pending                       | Compile and launch a representative DCU JIT kernel                                                 | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/warp.cuh`                             | jit-kernel     | Codex | manual merge    | Adopt official wave64 mask behavior while retaining DCU shuffle and synchronization helpers                                          | high   | static/registration passed; DCU runtime pending                       | Wave64 shuffle/sync kernel smoke                                                                   | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/disaggregation/mooncake/conn.py`                                | disaggregation | Codex | manual merge    | Use official common`TransferKVChunk` while preserving the DCU FA KV-layout environment and transfer layout                           | medium | static/registration passed; DCU runtime pending                       | Mooncake transfer smoke with`SGLANG_KV_LAYOUT_DCU_FA`                                              | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`                            | attention      | Codex | port to new API | Port official piecewise CUDA-graph structure around the existing DCU BF16/FP8 cache, LightOp, and page-size-64 paths                 | high   | static/registration passed; DCU runtime pending                       | DSA BF16/FP8 cache, sparse prefill, ragged decode, and graph smoke                                 | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/attention/flashattention_backend.py`                     | attention      | Codex | port to new API | Add official MLA context parallelism while retaining DCU fused cache-write ownership guards                                          | high   | static/registration passed; DCU runtime pending                       | Dense, MLA CP, DSV4, and fused cache-write smoke                                                   | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/attention/triton_backend.py`                             | attention      | Codex | manual merge    | Adopt official pool API and cache invalidation while retaining CPU last-index handling for graph capture                             | high   | static/registration passed; DCU runtime pending                       | Decode/extend and CUDA graph cache-invalidation smoke                                              | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/deepseek_v4_rope.py`                                     | dependency     | Codex | theirs          | Adopt the official ImportError-guarded TileLang initialization                                                                       | medium | static/registration passed; DCU runtime pending                       | Import with and without TileLang, then DSV4 RoPE smoke                                             | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/managers/overlap_utils.py`                                      | scheduler      | Codex | manual merge    | Adopt official FutureIndices/spec-extras APIs while keeping the native token resolver on DCU                                         | medium | static/registration passed; DCU runtime pending                       | Overlap scheduler plus speculative decode smoke                                                    | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/managers/schedule_batch.py`                                     | scheduler      | Codex | theirs          | Adopt official tensor flatten and speculative batch interface updates                                                                | medium | static/registration passed; DCU runtime pending                       | Batch flatten, overlap scheduling, and speculative decode smoke                                    | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/models/deepseek_v2.py`                                          | model          | Codex | manual merge    | Keep DCU fused RMS/quant returns while adding official DSA/MLA CP parameters and MoE output-buffer context                           | high   | static/registration passed; DCU runtime pending                       | DeepSeek V2/V3 fused RMS/quant, CP, MoE, and short-request smoke                                   | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/models/deepseek_v4.py`                                          | deepseek-v4    | Codex | manual merge    | Keep DCU fused cos/sin and LightOp/JIT paths; restrict official fused QK/sgl-kernel behavior to non-DCU HIP                          | high   | static/registration passed; DCU runtime pending                       | DSV4 TP, CP+EP, DP+EP, MTP, graph capture, and FP8 WO-A smoke                                      | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `sgl-kernel/csrc/common_extension_rocm.cc`                                         | sgl-kernel     | Codex | manual merge    | Register both DCU decode metadata operators and official DSV4 top-k/norm/RoPE operators                                              | high   | static/registration passed; DCU runtime pending                       | `gfx938` metadata check plus DCU sgl-kernel smoke whitelist                                        | validated |
| C05 /`8805f4cf1666` | `sync/official-main-C05-20260525` | `python/sglang/multimodal_gen/envs.py`                                             | diffusion      | Codex | manual merge    | Keep the DCU ring-attention setting while adopting official CFG gating and model-aware VAE channels-last policy                      | low    | compile and isolated env-default check passed                         | Diffusion runtime smoke on its target platform                                                     | validated |
| C05 /`8805f4cf1666` | `sync/official-main-C05-20260525` | `python/sglang/srt/layers/attention/dsa_backend.py`                                | attention      | Codex | port to new API | Adopt official`DSATopKBackend`; move the DCU LightOp fused transform into the new backend module                                     | high   | compile, CLI/env, and DCU LightOp route mock passed                   | DSA fused/unfused top-k, paged/ragged transform, graph, and sparse prefill smoke                   | validated |
| C05 /`8805f4cf1666` | `sync/official-main-C05-20260525` | `python/sglang/srt/layers/moe/hash_topk.py`                                        | moe            | Codex | manual merge    | Preserve DCU fused hash-top-k and LightOp postprocess, then invoke the official EPLB expert-distribution recorder                    | high   | compile and DCU semantic audit passed; runtime pending                | DSV4 hash top-k with EPLB off/on, padding, and logical-to-physical dispatch                        | validated |
| C05 /`8805f4cf1666` | `sync/official-main-C05-20260525` | `python/sglang/srt/models/deepseek_v4.py`                                          | deepseek-v4    | Codex | manual merge    | Keep DCU tuning-path imports and kernel branches while adding official EPLB per-layer recording context                              | high   | compile and DCU path audit passed; runtime pending                    | DSV4 pure TP, EPLB, MTP, CUDA graph, and short-request inference                                   | validated |
| C05 /`8805f4cf1666` | `sync/official-main-C05-20260525` | `test/registered/distributed/test_dp_attention_large.py`                           | test           | Codex | theirs          | Follow the official registered-test directory split and retain the existing DCU nightly registration at the new path                 | low    | DCU registration passed with 211 files; move scan passed              | Execute the moved nightly test on its configured DCU runner                                        | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/activation.py`                                           | aiter          | Codex | manual merge    | Accept the official AITER activation gate on generic HIP while keeping the existing DCU activation implementation                    | medium | compile and DCU path audit passed; runtime pending                    | AITER import plus activation eager and graph smoke on DCU                                          | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`                        | attention      | Codex | manual merge    | Adopt official FlashMLA metadata API but keep the external`flash_mla` package on DCU and sgl-kernel on other platforms               | high   | compile and DCU path audit passed; runtime pending                    | DSV4 pure TP, context parallel, FlashMLA metadata, and graph capture                               | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/attention/dsv4/compressor.py`                            | attention      | Codex | manual merge    | Restrict official HIP compressor fallback to non-DCU HIP so the validated DCU JIT/compressor-v2 selection remains intact             | high   | compile and DCU path audit passed; runtime pending                    | DSV4 compressor with JIT norm and compressor-v2 toggles                                            | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/attention/dsv4/indexer.py`                               | attention      | Codex | manual merge    | Accept the official cached vectorized fallback while preserving LightOp top-k as the sole DCU-priority transform path                | high   | compile and DCU path audit passed; runtime pending                    | LightOp top-k, index cache, sparse prefill, and graph replay                                       | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/attention/dsv4/metadata.py`                              | attention      | Codex | manual merge    | Use official compact HIP metadata copy only outside DCU and keep full DCU metadata plus DeepGEMM disable guards                      | high   | compile and DCU path audit passed; runtime pending                    | DSV4 extend/decode metadata under pure TP, CP, MTP, and CUDA graph                                 | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/attention/flashattention_backend.py`                     | attention      | Codex | theirs          | Adopt official per-request varlen encoder metadata and retain the previously merged DCU cache-write guards                           | medium | compile and DCU path audit passed; runtime pending                    | Dense/VLM encoder attention plus DCU fused cache-write smoke                                       | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/moe/hash_topk.py`                                        | moe            | Codex | manual merge    | Preserve DCU hash top-k and EPLB recording, then apply the official optional DeepEP waterfill transformation                         | high   | compile and DCU path audit passed; runtime pending                    | Hash top-k with EPLB and DeepEP waterfill independently enabled and combined                       | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/moe/moe_runner/aiter.py`                                 | aiter          | Codex | manual merge    | Add official activation/quant runner inputs for generic HIP while preserving DCU W8A8 and native AITER runner paths                  | high   | compile and DCU path audit passed; runtime pending                    | DCU AITER W8A8/W16A16 MoE, DeepEP, eager, and CUDA graph                                           | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/layers/quantization/fp8.py`                                     | quantization   | Codex | manual merge    | Accept official gfx95 MXFP4/AITER transforms only outside DCU and retain DCU FP8 shuffle and loading behavior                        | high   | compile and DCU path audit passed; runtime pending                    | DSV4 FP8 channel scale plus W8A8/MXFP4 load and accuracy spot check                                | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/managers/overlap_utils.py`                                      | scheduler      | Codex | manual merge    | Adopt official overlap helpers while disabling their torch-compile path on both NPU and DCU                                          | medium | compile and DCU path audit passed; runtime pending                    | Overlap scheduling, idle batch, request retract, and speculative decode                            | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/models/deepseek_v4.py`                                          | deepseek-v4    | Codex | manual merge    | Keep five DCU alternate streams and DCU preparation while accepting the official two-stream generic HIP implementation               | high   | compile and DCU path audit passed; runtime pending                    | DSV4 pure TP, CP+EP, DP+EP, MTP, DeepEP, and CUDA graph                                            | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/srt/speculative/eagle_info.py`                                      | speculative    | Codex | manual merge    | Combine official async NaN/OOB probes with the existing DCU sgl-kernel KV-cache I/O functions                                        | high   | compile and DCU path audit passed; runtime pending                    | EAGLE/MTP draft-target KV transfer, async probes, and graph smoke                                  | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `python/sglang/test/ci/ci_register.py`                                             | ci             | Codex | manual merge    | Export both the internal DCU registration decorator and the newly added official XPU decorator                                       | low    | DCU registration passed with 211 files                                | DCU and XPU suite import/registration scan                                                         | validated |
| C06 /`0abe6a85a51f` | `sync/official-main-C06-20260527` | `test/run_suite.py`                                                                | test           | Codex | manual merge    | Keep DCU suite mappings and schedules while adding the official XPU suite mappings and schedules                                     | low    | compile and DCU registration passed                                   | Generate/list DCU and XPU per-commit and nightly suites                                            | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `.gitignore`                                                                       | build          | Codex | manual merge    | Keep both DCU HIP generated-file ignores and official`.humanize/`                                                                    | low    | static passed; DCU runtime not required                               | None                                                                                               | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/jit_kernel/utils.py`                                                | jit-kernel     | Codex | manual merge    | Accept official MUSA/PDL detection while preserving DCU hipcc, FP8 target flags, and the DCU JIT cache key                           | medium | static/registration passed; DCU runtime pending                       | Compile representative DSV4 JIT kernels on gfx938                                                  | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`                   | attention      | Codex | manual merge    | Keep DCU ring-attention overlap selection and add official varlen FlashAttention fast path                                           | medium | static/registration passed; DCU runtime pending                       | Diffusion ring-attention and varlen attention smoke                                                | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`          | aiter          | Codex | port to new API | Adopt`(group, device)` dispatch and CUDA V2 checks while preserving DCU AITER selection, deterministic AR, and graph registration    | high   | static/registration passed; DCU runtime pending                       | AITER custom-allreduce eager plus CUDA graph replay                                                | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/layers/attention/triton_backend.py`                             | attention      | Codex | port to new API | Use official unified graph metadata helpers; retain DCU CPU seq-lens, SWA location translation/cache invalidation, and draft offsets | high   | static/registration passed; DCU runtime pending                       | Triton SWA/hybrid cache, target verify, EAGLE/MTP draft graph, and replay smoke                    | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/model_executor/forward_batch_info.py`                           | model_executor | Codex | manual merge    | Adopt official grouped fields and GPU-only extend lengths while retaining DCU quant fields and pinned H2D for list inputs            | high   | static/registration passed; DCU runtime pending                       | ForwardBatch list/GPU-only init, overlap scheduling, and speculative decode                        | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/models/deepseek_v4.py`                                          | deepseek-v4    | Codex | port to new API | Follow official`eae03ce3b` no-prewarm lifecycle while retaining DCU `deepgemm`, LightOp paths, and safe HIP multistream              | high   | static/registration and DCU runtime passed                            | DSV4 TP, CP+EP, DP+EP+MTP, graph capture, and GSM8K accuracy                                       | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/debug_utils/test_crash_dump.py`                                   | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Run on the registered DCU suite when available                                                     | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/debug_utils/test_soft_watchdog.py`                                | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Run on the registered DCU suite when available                                                     | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/lora/test_lora_eviction_policy.py`                                | test           | Codex | manual merge    | Preserve DCU registration and disabled reason while accepting official CPU registration                                              | low    | static/registration passed; DCU runtime pending                       | LoRA eviction policy smoke on DCU                                                                  | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/lora/test_lora_openai_api.py`                                     | test           | Codex | manual merge    | Preserve DCU registration and disabled reason while accepting official CPU registration                                              | low    | static/registration passed; DCU runtime pending                       | LoRA OpenAI API smoke on DCU                                                                       | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/model_loading/test_external_models.py`                            | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | External-model loading smoke on DCU                                                                | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/scheduler/test_routing_key_scheduling.py`                         | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Routing-key scheduler smoke and registered CPU unit test                                           | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/unit/managers/test_io_struct.py`                                  | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Run focused CPU unit test and registered DCU suite                                                 | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/unit/managers/test_prefill_adder.py`                              | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Run focused CPU unit test and prefill scheduler smoke                                              | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/unit/managers/test_profile_merger_http_api.py`                    | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Run focused CPU unit test and registered DCU suite                                                 | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/unit/utils/test_profile_merger.py`                                | test           | Codex | manual merge    | Preserve DCU registration while accepting official CPU registration                                                                  | low    | static/registration passed; DCU runtime pending                       | Run focused CPU unit test and registered DCU suite                                                 | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `test/registered/vlm/test_evs.py`                                                  | test           | Codex | manual merge    | Preserve DCU VLM registration and disabled reason while accepting official CPU registration                                          | low    | static/registration passed; DCU runtime pending                       | EVS/VLM startup and short-request smoke on DCU                                                     | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`                          | deepep         | Codex | manual merge    | Accept official dispatcher dtype refresh and NPU INT8 override while retaining DCU group-GEMM, FP8/W16A16, and custom quant paths    | high   | compile/ruff/registration passed; DCU runtime pending                 | DeepEP normal/low-latency BF16/FP8 dispatch, group-GEMM, MTP, and graph smoke                      | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/managers/overlap_utils.py`                                      | scheduler      | Codex | port to new API | Adopt official FutureMap-based deferred prefill H2D and input resolution; keep DCU out of compiled debug-assert helpers              | high   | compile/ruff/registration passed; DCU runtime pending                 | Overlap prefill/decode/mixed batch, eager/graph replay, and speculative decoding                   | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/model_executor/forward_batch_info.py`                           | model_executor | Codex | manual merge    | Accept KV-canary IDs and pin-memory API; retain DCU chunked-prefix operator and native clamp-position implementation                 | high   | compile/ruff/registration passed; DCU runtime pending                 | ForwardBatch prefill/decode/mixed/spec inputs, chunked prefix, token oracle, and CUDA graph        | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/models/deepseek_nextn.py`                                       | speculative    | Codex | manual merge    | Add official NPU MTP unquant context while retaining DCU SBO stream and MTP top-k reuse                                              | high   | compile/ruff/registration passed; DCU runtime pending                 | DeepSeek NextN/MTP eager and graph, reused top-k, CP, and DeepEP dispatch                          | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/models/deepseek_v4.py`                                          | deepseek-v4    | Codex | manual merge    | Keep official fused MHC structure for non-DCU; DCU remains on AITER pre/post and does not restore model prewarm                      | high   | compile/ruff/registration passed; DCU runtime pending                 | DSV4 pure TP, CP+EP, DP+EP+MTP, graph to bs=128, repeated accuracy, and MHC path audit             | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | port to new API | Import moved argparse actions, remove duplicate local definitions, and accept official KV-canary/parallel CLI                        | medium | compile/ruff; 48 unit tests passed and 8 blocked by no-device LightOp | CLI parse for DCU launch arguments, KV-canary default-off, and moe-dense-tp validation             | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/speculative/eagle_info_v2.py`                                   | speculative    | Codex | manual merge    | Use official cache-location helper API with a single`_is_dcu` branch that preserves the DCU kvcache operator                         | high   | compile/ruff/registration passed; DCU runtime pending                 | EAGLE/MTP target verify, page-size-64 cache locations, graph replay, and fallback Triton path      | validated |
| C08 /`373cadc92ea4` | `sync/official-main-C08-20260531` | `python/sglang/srt/speculative/eagle_worker_v2.py`                                 | speculative    | Codex | theirs          | Accept official KV-canary contexts around draft eager/graph execution while retaining existing DCU HIP top-k behavior                | high   | compile/ruff/registration passed; DCU runtime pending                 | EAGLE draft/decode graph, multi-step draft, MTP112, and KV-canary disabled/enabled smoke           | validated |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/pyproject.toml`                                                            | dependency     | Codex | manual merge    | Accept official FlashAttention 4 dependency while retaining the internally validated `sgl-deep-gemm==0.1.0` pin                      | medium | static install-flow audit passed; runtime pending                                       | DCU install/resolver smoke; assess upgrading the internal DeepGEMM package to 0.1.2                 | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`                        | attention      | Codex | port to new API | Adopt the three-method metadata lifecycle and SM120 path; retain DCU external FlashMLA, LightOp quant cache, and sparse prefill       | high   | compile/ruff/registration passed; DCU runtime pending                             | DSV4 TP/CP/PD, split prefill/decode, graph to bs=128, sparse prefill, and FlashMLA                  | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/sglang/srt/layers/attention/dsv4/indexer.py`                               | attention      | Codex | manual merge    | Accept official FP4/SM120 indexer support while preserving DCU LightOp top-k as the sole DCU-priority transform                       | high   | compile/ruff/registration passed; DCU runtime pending                             | DCU FP8 index cache, LightOp top-k, graph replay; verify FP4 flag remains rejected off SM100        | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/sglang/srt/layers/attention/flashattention_backend.py`                     | attention      | Codex | port to new API | Move metadata kernels to official `triton_ops/metadata.py`, keep DCU FA layout/SWA translation, and restore required `cu_seqlens_q` | high   | compile/ruff/registration passed; DCU runtime pending                             | Dense/VLM, DSV4, SWA, cross-attention, speculative graph, and DCU FA-layout smoke                  | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/sglang/srt/model_executor/model_runner.py`                                 | model_executor | Codex | port to new API | Use official `hc_hidden_size` graph-buffer contract, which replaces the removed DSV4 PP proxy-shape argument                         | high   | compile/ruff/registration passed; DCU runtime pending                             | DeepSeek-V4 PP/PD startup, graph capture, flattened MHC hidden-state transfer                      | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/sglang/srt/speculative/eagle_info_v2.py`                                   | speculative    | Codex | port to new API | Consume official centralized cache-location/EAGLE helpers; DCU operator dispatch moved with the helper rather than duplicated        | high   | compile/ruff/registration passed; DCU runtime pending                             | EAGLE/MTP draft and target verify, page-size 1/64, eager/graph, operator-disabled Triton fallback  | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `python/sglang/srt/speculative/spec_utils.py`                                      | speculative    | Codex | port to new API | Remove duplicated Triton cache kernels and import the official centralized helpers, including the migrated DCU req-pool operator     | high   | compile/ruff/registration passed; DCU runtime pending                             | Spec V1/V2 cache assignment, retract/abort, graph replay, and `SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=0`  | merged    |
| C09 /`c55548ba115c` | `sync/official-main-bootstrap`   | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`                           | mem_cache      | Codex | backport upstream fix | Backport official `c9ca56da8c`: compute full-to-SWA write locations once in per-forward graph metadata instead of retaining pool-level locations across replay; preserve DCU LightOp stores | high | compile/ruff/registration passed; DCU runtime pending | Graph on/off parity; standalone CP+EP; PD prefill/decode; EAGLE/MTP; GSM8K 10; HiCache mapping reload | merged |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `test/registered/spec/eagle/test_eagle_infer_a.py`                                 | test           | Codex | port to new API | Accept official test deletion and migrate the DCU placeholder registration to the new core/top-k EAGLE suites                        | medium | registration passed; DCU runtime pending                                            | Enable after BW1100 draft/target model mapping is available                                       | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `test/registered/spec/eagle/test_eagle_infer_b.py`                                 | test           | Codex | port to new API | Accept official test deletion and migrate page-size/tree coverage plus the DCU disabled reason to the split test files                | medium | registration passed; DCU runtime pending                                            | Run new core/page/top-k suites on a DCU runner                                                     | merged    |
| C09 /`c55548ba115c` | `sync/official-main-C09-20260602` | `test/registered/spec/eagle/test_eagle_infer_beta.py`                              | test           | Codex | port to new API | Accept official beta-test deletion and preserve its DCU placeholder intent in the new EAGLE core/page suites                         | medium | registration passed; DCU runtime pending                                            | Validate EAGLE3 overlap, logprob parity, page-size 64, and graph replay on BW1100                  | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `.github/workflows/pr-test-dcu.yml`                                                | ci             | TBD   | manual merge    | Keep official workflow structure and DCU runner/wheel overlays                                                                       | medium | CI dry-run and DCU registration check                                 | Fill exact runner/image validation command                                                         | open      |
| C13 /`125ef888921b` | `sync/official-main-C13-20260610` | `sgl-kernel/**`                                                                    | sgl-kernel     | TBD   | manual merge    | sgl-kernel interfaces and DCU/HIP glue both changed                                                                                  | high   | sgl-kernel DCU smoke whitelist                                        | Assign kernel owner                                                                                | open      |

## Per-Checkpoint Notes

### C01 / `c67b2870569a`

- Expected focus: test registry and CI overlap.
- Owner: Codex for initial mechanical merge; runtime owners still required for high-risk MoE/AITER/DeepSeek follow-ups.
- Required validation:
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`
  - DCU CI dry-run command, to be filled.
- Notes:
  - Branch: `sync/official-main-C01-20260517`.
  - Low-risk CI/test/env conflicts were resolved by taking official suite/action
    naming and preserving DCU-specific workflow wording/env flags.
  - High-risk runtime files were resolved DCU-first for C01:
    `ep_moe/layer.py`, `moe_runner/aiter.py`, `quantization/unquant.py`,
    `deepseek_v2.py`, `deepseek_v4.py`, and `server_args.py`.
  - Reason for DCU-first runtime resolution: these files contain existing DCU
    AITER, DeepEP, fused quant/RMS, and DeepSeek V4 paths; blindly porting the
    official C01 runner/model refactors would be higher risk than preserving
    known behavior for the first checkpoint.
  - Follow-up: assign MoE/AITER/DSV owner to review skipped official hunks before
    C02/C03 if those hunks become required by later checkpoints.
  - Validation completed:
    - precise conflict marker scan: passed.
    - syntax compile for all conflicted Python files: passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed, collected
      212 DCU registered test files.
  - Recommended manual validation:
    - `ci` / `test`: PR workflow dry-run and `/rerun-failed-ci` flow.
    - `mem_cache`: CPU offload smoke covering DCU FA KV layout.
    - `scheduler`: pipeline-parallel proxy tensor smoke.
    - `moe` / `deepep` / `aiter`: Qwen3 MoE or DeepEP smoke covering current
      DCU AITER path.
    - `model` / `deepseek-v4`: DeepSeek V2/V4 startup and short request smoke.
    - `server_args`: speculative algorithm alias CLI parse smoke if used.
  - Manual validation result:
    - TBD

### C02 / `425dffbde339`

- Expected focus: DeepSeek V4 MTP, attention, DeepEP.
- Owner: Codex for initial mechanical merge; runtime owners required for DeepEP/MoE/AITER follow-up.
- Required validation:
  - DCU MoE smoke, exact command to be filled.
  - DSV4 smoke, exact command to be filled.
- Notes:
  - Branch: `sync/official-main-C02-20260519`.
  - Low/medium-risk conflicts were merged hunk-by-hunk.
  - High-risk MoE/DeepEP/FP8 files were resolved DCU-first:
    `ep_moe/layer.py`, `token_dispatcher/deepep.py`, and
    `quantization/fp8.py`.
  - Official C02 DeepEP dispatcher and EP MoE runner changes should be reviewed
    as a dedicated task before relying on later upstream MoE behavior.
  - Validation completed:
    - precise conflict marker scan: passed.
    - syntax compile for all conflicted Python files: passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed, collected
      212 DCU registered test files.
  - Recommended manual validation:
    - `dependency`: DCU install flow to confirm CUDA-only flashinfer cu13 deps
      are not required by internal workflow.
    - `jit-kernel`: DSV4 JIT smoke and deleted-header reference check in the
      target build container.
    - `attention`: DSV4 sparse prefill/topk, FlashMLA, NSA BF16 index-cache, and
      NSA chunking smoke.
    - `moe` / `deepep`: DeepEP normal dispatch and low-latency dispatch smoke,
      including topk/dispatch/combine compatibility.
    - `quantization` / `aiter`: FP8 MoE path, AITER/ASM shuffle path, and
      quantized MoE smoke.
    - `scheduler` / `model_executor`: split prefill and forward-batch init smoke.
    - `model` / `deepseek-v4`: DeepSeek V4 startup, short request, and FP8 WO-A
      GEMM path if the checkpoint is available.
    - `speculative`: EAGLE/MTP smoke covering idle and prefill paths.
    - `ci`: DCU registration, XPU marker coexistence, and PR workflow dry-run.
  - Manual validation result:
    - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112 validation - ✅
    - CI test - ✅

### C03 / `7cf193fe1faf`

- Expected focus: cache, model, attention.
- Owner: Codex for the initial merge; DCU attention, DeepSeek V4, cache, and
  MoE owners are required for runtime validation.
- Required validation:

  - Dense smoke.
  - Cache-related unit or smoke test, exact command to be filled.
- Recommended manual validation:

  - `attention`: Qwen dense smoke and attention backend selection smoke.
  - `mem_cache`: radix cache, SWA/hybrid cache, and retract/decode cache smoke.
  - `model`: affected model startup and short request smoke.
- Manual validation result:

  - TBD
- Notes:

  - Branch: `sync/official-main-C03-20260521`.
  - Resolved 24 Git conflicts.
  - Adopted the official NSA-to-DSA rename as the canonical structure. Old
    `nsa/*` modules remain compatibility shims, while DCU indexer, TileLang,
    Triton, and backend differences were three-way ported into `dsa/*`.
  - Adopted the official DSV4 JIT split under `sglang.jit_kernel.dsv4`.
    The old monolithic `deepseek_v4.py` and standalone `topk_1024.cuh` were
    removed. DCU HIP compile guards were retained in `topk_v1.cuh`.
  - Ported DCU code away from C03-removed `ForwardBatch.token_to_kv_pool` and
    `ForwardBatch.attn_backend` fields to ForwardContext accessors.
  - Kept `ep_moe/layer.py` DCU-first because it contains active DeepEP,
    AITER, Marlin, and group-GEMM implementations. This remains a high-risk
    follow-up rather than a completed upstream runner migration.
  - Validation completed:

    - no unmerged Git entries: passed.
    - precise conflict marker scan: passed.
    - targeted Python compile for conflict and ForwardContext fallout files:
      passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed, collected
      211 DCU registered test files.
    - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
      passed, 24 tests.
    - staged `git diff --check`: passed.
    - `test/registered/unit/mem_cache/test_dsa_pool_host_unit.py`: not run to
      assertions because the current host has no HIP GPU and `lightop`
      initialization cannot determine `LIGHTOP_GPU_CUS`.
  - Additional recommended manual validation:

    - `attention` / `deepseek-v4`: DSA BF16/FP8 index-cache, sparse prefill,
      chunking, DSV4 top-k 512/1024, compressor, and FlashMLA smoke.
    - `mem_cache`: DSA host backup/restore, retract/resume, SWA translation,
      and DCU page-size 64 smoke.
    - `model`: Qwen dense, Qwen3.5 fused path, DeepSeek V2/V3.2, and DeepSeek V4
      startup plus short request.
    - `moe` / `deepep`: DeepEP normal and low-latency dispatch, AITER,
      Marlin/group-GEMM, and quantized MoE smoke.
    - `scheduler`: overlap scheduler plus EAGLE/MTP seq-len publication smoke.
  - Manual validation result:

    - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112 validation - ✅
    - CI test - ✅

### C04 / `af8f66940e9b`

- Expected focus: AMD DSV4 runtime and jit-kernel.
- Owner: Codex for the mechanical merge; DCU DSV4, attention, model, and kernel
  owners are required for runtime validation.
- Required validation:

  - jit-kernel or sgl-kernel DCU smoke.
- Manual validation result:

  - TBD
- Notes:

  - Branch: `sync/official-main-C04-20260523`.
  - Resolved 16 Git conflicts as one checkpoint; no checkpoint split was
    required.
  - Adopted official `kDLGPU`, HIP runtime fallback, context-parallel APIs,
    common disaggregation types, scheduler flattening, and DSV4 sgl-kernel
    registrations.
  - Preserved DCU-first behavior for uint8-backed FP8 JIT output, HIP FP8 pack,
    fused cache writes, DSA page-size-64/BF16/FP8 cache, LightOp top-k, fused
    RMS/quant, and fused cos/sin caches.
  - Non-conflict HIP semantic audit:

    - `dsv4/attn.py` keeps the JIT fused-store path on DCU; Triton store is
      limited to non-DCU HIP.
    - `dsv4/moe.py` keeps the JIT hash top-k path on DCU; Triton hash top-k is
      limited to non-DCU HIP.
    - `dsv4/gemm.py` preserves DCU deep-gemm priority and FP32 AITER output;
      official non-DCU HIP behavior remains unchanged.
    - `dsa_indexer.py` preserves the DCU page-table-64 path for ragged prefill.
    - `deepseek_v2.py` limits the official fused-clamp AITER path to non-DCU
      HIP so DCU keeps the existing JIT elementwise path.
    - `deepseek_v4.py` limits official fused QK norm/RoPE and gfx95 FP8 input
      quantization to non-DCU HIP, while retaining DCU alt streams.
    - `overlap_utils.py` adopts the official speculative data model but keeps
      the native future-token resolver on DCU.
    - `attention/dsv4/indexer.py` still routes DCU top-k to
      `lightop_topk_transform_512`; the removed JIT shared-memory workaround was
      not restored.
  - Automated validation completed:

    - no unmerged Git entries: passed.
    - precise conflict marker scan: passed.
    - staged `git diff --check`: passed.
    - compile for every changed Python file: passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed, collected
      211 DCU registered test files.
    - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
      passed, 24 tests.
    - `AMDGPU_TARGET=gfx938 python3 setup_rocm.py --name`, run from
      `sgl-kernel/`: passed; the new DSV4 top-k and norm/RoPE sources hipified
      successfully.
    - removed JIT cast-file reference scan: passed.
    - official C04 commit object was repacked locally and remains readable
      without an alternate object directory.
  - Recommended manual validation:

    - `jit-kernel` / `sgl-kernel`: DSV4 JIT build/import, FP8 elementwise,
      wave64 helpers, LightOp top-k, compressor, and kernel smoke whitelist.
    - `attention`: DSA BF16/FP8 index cache, sparse prefill, FlashMLA, MLA CP,
      fused cache write, and graph cache invalidation.
    - `model` / `deepseek-v4`: DeepSeek V2/V3 fused RMS/quant and DeepSeek V4
      pure TP, CP+EP, DP+EP, MTP, FP8 WO-A, and CUDA graph capture.
    - `disaggregation`: Mooncake transfer with the DCU FA KV layout.
    - `scheduler` / `speculative`: overlap scheduler and speculative decode.
    - `dependency`: DCU container install/build sanity and guarded TileLang import.
  - Manual validation result:

    - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112 validation - ✅
    - CI test - ✅

### C05 / `8805f4cf1666`

- Expected focus: PD/scheduler fail-fast, DSA top-k backend, DSV4 EPLB,
  hybrid-cache dispatch, and registered-test directory split.
- Owner: Codex for merge and static validation; DCU DSA/DSV4 owners for runtime.
- Required validation:

  - conflict marker scan, compile, and DCU registration.
  - scheduler fail-fast and PD subprocess-exit behavior.
  - DSA top-k backend CLI/env plus DCU LightOp route.
- Notes:

  - Five conflicts were resolved as one checkpoint; no split was required.
  - The new official `DSATopKBackend.SGL_KERNEL` keeps `fast_topk_v2` from
    sgl-kernel, while DCU fused paged/ragged transforms route to LightOp.
  - `ScheduleBatch.loc_tensor` and all existing DeepSeek-V4 `_is_dcu` kernel
    paths remain present after the automatic merge.
  - The official hybrid-cache strategy refactor was accepted unchanged; it
    already contains dedicated DeepSeek-V4 FULL+SWA dispatch.
  - Automated validation completed:

    - no unmerged entries, marker scan, and `git diff --check`: passed.
    - full `python/sglang` and registered-test compile: passed.
    - DCU registration: passed with 211 registered files.
    - existing DSA alias/CLI/env suite: 24 tests passed.
    - C05 DSA top-k CLI/env defaults and DCU LightOp fused-route mock: passed.
    - official metrics collector dependency-injection suite: 14 tests passed.
    - diffusion merged env defaults: passed with isolated module loading.
    - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
      unsupported CUDA calls; C04 DSV4 sources remain in the build manifest.
  - Runtime validation pending:

    - hybrid-cache dispatch/radix unit collection is blocked locally by LightOp
      and lmslim device initialization because no HIP device is available.
    - scheduler/PD fail-fast, DSA kernels, DSV4 EPLB/hash top-k, pure-TP server,
      and CUDA graph validation require a DCU runner.
  - Recommended manual validation:

    - `scheduler` / `PD`: scheduler exception, subprocess exit, overlap idle batch,
      and disaggregation prefill/decode failure propagation.
    - `attention`: DSA `sgl-kernel` backend with fused/unfused paged and ragged
      top-k; confirm DCU uses LightOp only for fused transforms.
    - `deepseek-v4` / `moe`: pure TP, MTP, graph capture, hash top-k, and EPLB
      recording with EPLB both disabled and enabled.
    - `mem_cache`: hybrid SWA/HiCache strategy selection, cache hit/retract, and
      DeepSeek-V4 FULL+SWA pool construction.
    - `test`: DCU registry after the official directory split.
  - Manual validation result:

    - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112 validation - ✅
    - CI test - ✅
    - DCU EPLB recording enabled validation - ✅
    - HiCache - ❓
    - PD disaggregation validation - ✅

### C06 / `0abe6a85a51f`

- Expected focus: DSV4 compressor/FlashMLA/MTP, AITER and FP8 MoE,
  DeepEP waterfill, overlap scheduling, unified radix cache, and HiCache.
- Owner: Codex for merge and static validation; DCU attention, MoE, cache,
  and speculative owners for runtime validation.
- Required validation:
  - conflict marker scan, Python compile, DCU registration, and HIP build metadata.
  - DSV4 DCU-priority path audit for compressor, LightOp top-k, FlashMLA,
    metadata copy, AITER/FP8, alternate streams, and draft attention backend.
  - unified radix cache and HiCache targeted tests where the local environment
    can import the DCU runtime.
- Notes:
  - Fourteen conflicts were resolved as one checkpoint; no split was required.
  - Official C06 was merged at exact SHA `0abe6a85a51f`; later official commits
    are not included.
  - Official generic HIP behavior is preserved for AMD ROCm, while `_is_dcu`
    retains the existing external FlashMLA package, LightOp top-k, full DSV4
    metadata copy, five alternate streams, and DCU AITER/FP8 paths.
  - The automatically merged speculative draft backend was corrected so the
    new HIP radix backend is selected only on non-DCU HIP.
  - `ScheduleBatch.loc_tensor` remains device-resident and the C04 DSV4
    sgl-kernel AOT sources remain in `setup_hip.py`.
  - CUDA Graph runtime follow-up (2026-07-02):
    - with graph disabled, startup and inference accuracy pass; with graph
      enabled, a single request repeats an incorrect short phrase and a
      multi-request GSM8K run faults after prefill enters graph decode.
    - the VMFault queue identifies `per_token_quant_fp8_kernel`, downstream of
      DSV4 attention output; the kernel implementation itself predates C06.
    - root cause: official `3f5e2c7688` added an HIP-specific conservative
      multi-stream path that overlaps only the core/indexer compressors. The C06
      merge resolution changed its dispatch from `_is_hip` to
      `_is_hip and not _is_dcu`, so DCU combined newly enabled fused WQA/WKV
      with the old three-stream Q/KV/indexer path during graph capture.
    - resolution: DCU now follows `_forward_prepare_multi_stream_hip()`.
      Fused WQA/WKV, Q projection, KV cache write, and complete indexer work
      remain on the main stream; only core and indexer compressor work runs on
      auxiliary streams. Existing DCU LightOp, FP8, RoPE, and fused cache-write
      branches remain unchanged.
    - the temporary DCU guard in generic `can_cp_split()` was removed. The
      reported DeepSeek-V4 launch uses DSA round-robin CP through
      `can_dsa_cp_split()`, so that guard was not on the failing call path and
      only disabled multi-request CP for other generic CP users.
    - the old DCU three-stream topology is not restored. Revisit it only with
      dedicated cross-stream lifetime and graph-replay validation.
    - follow-up static validation passed: targeted Python compile,
      `git diff --check`, DCU registration with 211 registered files, HIP helper
      dispatch audit, and DCU LightOp/compressor priority-path audit.
    - follow-up runtime validation remains pending on the target DCU node.
  - Automated validation completed:
    - no unmerged entries, precise marker scan, and `git diff --check`: passed.
    - full `python/sglang` and registered-test syntax compile: passed.
    - DCU registration: passed with 211 registered files.
    - existing DSA alias/CLI/env suite: 24 tests passed.
    - field-validator suite: 18 tests and 13 subtests passed.
    - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
      unsupported CUDA calls; DSV4 top-k and norm/RoPE sources remain present.
    - targeted undefined-name audit found and fixed the merged `is_hip` import
      in DSV4 metadata. Full-file ruff still reports inherited C05 lint debt.
  - Runtime validation pending:
    - unified-radix registry collection is blocked locally because LightOp
      cannot initialize without a visible HIP device.
    - model, kernel, HiCache, DeepEP, MTP, and graph tests require a DCU runner
      and model weights.
    - rerun the reported completion request with CUDA Graph enabled and verify
      a coherent 128-token answer rather than a repeated phrase.
    - run `bench_sglang.py --num-questions 10` and verify no VMFault; the log
      should show multi-request prefill with `cuda graph: False` followed by
      stable graph decode.
  - Recommended manual validation:
    - `deepseek-v4` / `attention`: pure TP, CP batch greater than one, CP+EP,
      DP+EP+MTP, FlashMLA, compressor, sparse prefill, and CUDA graph capture.
    - `moe` / `deepep` / `aiter`: DeepEP normal mode, waterfill disabled/enabled,
      EPLB disabled/enabled, hash top-k, W8A8/W16A16, and graph replay.
    - `quantization`: FP8 channel scale plus FP8/MXFP4 load and accuracy spot check.
    - `mem_cache`: unified radix cache, HiCache, cache hit/retract, and Mooncake
      or CPU-offload path when available.
    - `scheduler` / `speculative`: overlap idle batch and MTP/EAGLE draft-target
      KV-cache transfer with async NaN/OOB probes.
    - `ci` / `test`: DCU and XPU registration plus suite partition generation.
  - Manual validation result:
    - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112 validation - ✅
    - CI test - ✅
    - HiCache - ❓

### C07 / `a5e6a8887a94`

- Expected focus: attention graph unification, ForwardBatch grouping, custom
  allreduce V2 dispatch, DeepSeek-V4 MHC/DeepGEMM, and test registration moves.
- Owner: Codex for merge and static validation; DCU attention, DSV4, AITER,
  speculative, and CI owners for runtime validation.
- Required validation:
  - conflict marker scan, Python compile, DCU registration, and HIP build metadata.
  - custom-allreduce signature/call-site audit and DSV4 DCU-priority path audit.
  - CUDA graph replay through `bs=128`, single-request accuracy, and GSM8K 10.
- Notes:
  - Branch: `sync/official-main-C07-20260529`.
  - Git reported 18 textual conflicts. The planned modify/delete conflict for
    GPT-OSS was automatically recognized as a rename to
    `test_gpt_oss_4gpu_mxfp4.py`; it is recorded as a semantic audit rather
    than an artificial nineteenth conflict.
  - Official C07 unified Triton capture/replay helpers are the canonical
    structure. DCU retains CPU seq-lens publication for the Triton backend,
    SWA full-to-window location translation, cache invalidation, variable MTP
    draft lengths, and sliding-window offsets.
  - The DSV4 backend continues to inherit `needs_cpu_seq_lens=True`; it does
    not enter Triton's non-DCU GPU-only seq-lens path.
  - DeepSeek-V4 keeps the C06 conservative HIP multistream topology: only core
    and indexer compressor work uses auxiliary streams. DCU retains the
    `deepgemm` package path; non-DCU uses the official DeepGEMM wrapper.
  - Runtime follow-up aligned DCU with official `eae03ce3b`: model-specific MHC
    prewarm and its `ModelRunner` hook were removed. This also fixes decode
    startup with a NextN draft model, whose wrapper does not expose the removed
    prewarm API. Target-node runtime validation passed.
  - GPT-OSS DCU nightly placeholder registration now follows both official
    split files: BF16 and MXFP4. Both remain disabled pending BW1100 validation.
  - Non-conflict semantic audit:
    - official DSV4 graph metadata copy/replay structure and page-size 64 are
      retained; DSV4 remains CPU-seq-lens aware.
    - official HIP `Event.synchronize()` TPOT workaround is accepted while the
      DCU native future-token resolver remains present.
    - MXFP4 AITER shuffle imports stay restricted to non-DCU HIP; gfx938 DCU
      does not enter the gfx95 shuffle path.
    - all DCU per-commit and nightly suite mappings remain in `test/run_suite.py`.
  - Automated validation completed:
    - no unmerged entries and precise conflict marker scan: passed.
    - staged `git diff --check`: passed after normalizing the official
      `pr-test-npu.yml` CRLF line endings; workflow content is unchanged.
    - syntax compile for every C07 changed Python file: passed.
    - targeted undefined-name/unused-import ruff gate for conflict and semantic
      audit files: passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed, collected
      212 DCU registered test files. Both GPT-OSS BF16 and MXFP4 registrations
      appear in `nightly-dcu-4-gpu`.
    - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
      passed, 24 tests.
    - custom-allreduce AST signature check: passed with exactly
      `(group, device)`; the sole framework call site passes both arguments.
    - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`, run from
      `sgl-kernel/`: passed with zero unsupported CUDA calls.
    - source audit confirmed the DCU safe HIP multistream dispatch, `deepgemm`,
      LightOp top-k, DCU-only model warmup, and Triton CPU seq-lens contract.
    - conflict-related CPU unit collections were attempted but blocked before
      assertions: this host has no HIP device, so installed `lightop` obtains
      no CU count from `rocminfo` and raises while assigning
      `LIGHTOP_GPU_CUS=None`.
    - DCU model, graph, custom-allreduce, and kernel runtime validation remains
      pending on the target runner.
  - Recommended manual validation:
    - `deepseek-v4` / `attention`: graph capture through `bs=128`, three
      repeated short requests, GSM8K 10, pure TP, CP+EP, DP+EP+MTP, FlashMLA,
      compressor, and confirmation that no model-specific MHC prewarm runs.
    - `speculative` / `mem_cache`: Triton target verify, EAGLE/MTP draft graph,
      SWA/hybrid-cache location translation, and cache invalidation.
    - `aiter`: custom-allreduce eager and graph replay, deterministic mode,
      and the graph registration workaround.
    - `jit-kernel`: representative gfx938 DSV4 JIT compile and FP8 target flags.
    - `model`: Qwen2.5 dense, VLM, embedding, and reranker smoke.
    - `ci` / `test`: stage-a plus available stage-b DCU CI and split GPT-OSS
      registration generation.
  - Manual validation result:
    - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112 validation - ✅
    - CI test - ✅
    - HiCache - ❓

### C08 / `373cadc92ea4`

- Expected focus: KV-canary, allocator package split, deferred prefill H2D,
  EAGLE/NextN, DeepSeek-V4 fused MHC, Mooncake, and CI utilities.
- Owner: Codex for merge and static validation; DCU scheduler, speculative,
  DeepEP, DSV4, and Mooncake owners for runtime validation.
- Required validation:
  - conflict marker scan, changed-Python compile, targeted ruff, DCU
    registration, and `gfx938` HIP build metadata.
  - FutureMap/deferred-H2D prefill, decode, mixed batch, overlap, and graph.
  - DeepSeek-V4 and NextN/MTP with DCU MHC and cache-location paths.
  - Mooncake intra-node custom pool and HiCache double-tag regression.
- Notes:
  - Branch: `sync/official-main-C08-20260531`.
  - C08 contains 57 commits but touches 275 files because it introduces the
    KV-canary subsystem. Git reported eight textual conflicts; checkpoint
    splitting was not required.
  - The official FutureMap/deferred-H2D scheduler structure is retained. DCU
    keeps native clamp-position and disables compiled debug assertions.
  - The official fused MHC post-pre implementation is retained for non-DCU.
    It is gated off on DCU so existing AITER MHC pre/post remains authoritative;
    unused prewarm methods reintroduced by the upstream commit were removed.
  - EAGLE has one cache-location helper. Its `_is_dcu` branch uses
    `dcu_assign_extend_cache_locs`; CUDA, generic HIP, MUSA, and NPU retain the
    official implementations.
  - The allocator module-to-package split keeps existing imports compatible
    through `mem_cache/allocator/__init__.py`.
  - KV-canary is accepted with its default `none` mode. Enabling it on DCU is
    experimental and requires dedicated kernel/graph validation.
  - Automated validation completed:
    - no unmerged entries, precise marker scan, and `git diff --check`: passed.
    - syntax compile for every C08 changed Python file: passed.
    - targeted `E9/F401/F811/F821` ruff gate for conflict files: passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU
      registered files.
    - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
      unsupported CUDA calls.
    - PhaseChecker tests: 29 passed; 32 require a HIP device and failed because
      this host reports no HIP GPU.
    - ServerArgs tests: 48 passed; 8 were blocked when installed `lightop`
      received no CU count from the device-less host.
    - unified-radix-cache test collection was blocked by the same no-device
      `lightop` initialization.
- Recommended manual validation:
  - `scheduler`: overlap on/off, chunked prefill, mixed prefill/decode, PD
    prefill/decode, CUDA graph to bs=128, and repeated short requests.
  - `deepseek-v4` / `speculative`: pure TP, CP+EP, DP+EP+MTP112, NextN top-k
    reuse, EAGLE draft/verify graph, GSM8K 10, and no model prewarm log.
  - `deepep`: normal and low-latency BF16/FP8 dispatch, group-GEMM, eager, and graph.
  - `mem_cache` / `kv-canary`: allocator compatibility, radix/HiCache, default-off
    startup, then opt-in token-oracle and real-KV checks on a dedicated runner.
  - `mooncake`: intra-node custom memory pool plus HiCache existence checks for
    KV-only and hybrid page components.
  - `jit-kernel`: large-tensor add-constant and KV-canary JIT compile on gfx938.
- Manual validation result:
  - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112/PD disaggregation of intranode validation - ✅
  - CI test - ✅

### C09 / `c55548ba115c`

- Expected focus: three-method attention metadata lifecycle, DSV4 FP4/SM120,
  FlashAttention metadata kernel centralization, EAGLE test/helper refactor,
  PD HiCache transfer, embedding replication, and mem-cache changes.
- Owner: Codex for merge and static validation; DCU DSV4, speculative, PD,
  HiCache, PP, and dependency owners for runtime validation.
- Required validation:
  - precise conflict-marker scan, changed-Python compile, targeted ruff, DCU
    registration, DSA alias/CLI/env tests, and `gfx938` HIP build metadata.
  - DSV4 pure TP, CP+EP, DP+EP+MTP, PD prefill/decode, sparse prefill, LightOp
    top-k, FlashMLA, and CUDA graph capture/replay through `bs=128`.
  - EAGLE/MTP cache-location operators and their Triton fallbacks with page
    sizes 1 and 64, overlap on/off, retract/abort, and graph replay.
  - HiCache optimistic prefetch/incremental transfer, dense/VLM attention,
    embedding/reranker, and PP hidden-state transfer.
- Notes:
  - Branch: `sync/official-main-C09-20260602`.
  - C09 contains 103 official commits and reported seven textual conflicts
    plus three modify/delete test conflicts; checkpoint splitting was not
    required.
  - Official attention metadata kernels now live in
    `layers/attention/triton_ops/metadata.py`. The DCU FA layout and SWA
    location translation remain in the backend; `cu_seqlens_q` is retained
    because the DCU varlen/decode paths still consume it.
  - Runtime audit restored the DCU LightOp implementation of
    `normal_decode_set_metadata` in the centralized metadata module. This was a
    real C09 migration omission, but the failing DeepSeek-V4 runs use the DSV4
    backend, so it does not explain their decode corruption.
  - DeepSeek-V4 uses official `init_forward_metadata_out_graph()` plus
    `init_forward_metadata_in_graph()`. Raw metadata is upgraded in-graph;
    the removed `_maybe_upgrade_forward_metadata()` hook is not restored.
  - DCU keeps the external FlashMLA adapter, split prefill/decode behavior,
    sparse-prefill workspace, LightOp KV quantization, and LightOp top-k.
    SM120 and other non-DCU platforms use the official C09 implementations.
  - Official FP4 indexer support is accepted but remains guarded by the
    existing SM100 server-argument check, so gfx938 DCU stays on FP8.
  - EAGLE cache-location kernels are centralized in
    `speculative/triton_ops/cache_locs.py`; both DCU kvcache operators and
    their `SGLANG_ASSIGN_*` fallback switches moved into that module.
  - The old EAGLE A/B/Beta files are removed upstream. Their DCU disabled
    registrations are preserved in the new core, page-size, and top-k suites.
  - `flash-attn-4` is accepted. The internal `sgl-deep-gemm==0.1.0` pin is
    retained pending a DCU package upgrade/compatibility decision for 0.1.2.
  - Runtime accuracy follow-up: standalone CP+EP and PD decode both produce a
    correct first token followed by corrupted CUDA Graph decode output. This
    excludes PD transfer and EAGLE as necessary causes and points to DSV4 KV
    write locations during graph replay.
  - Official C10 commit `c9ca56da8c` fixes this exact lifetime problem by
    recording full-to-SWA translation in `init_forward_metadata_in_graph()` and
    removing the pool-level cross-forward cache. Its DSV4 changes were
    backported while retaining DCU external FlashMLA, LightOp quant/store,
    fused norm/RoPE/cache-write, and conservative HIP multi-stream paths.
  - Direct C10 merge was rejected for this runtime fix: C10 contains 115
    commits and 364 changed files, while merge-tree reports roughly 60
    dual-modified conflicts. C09 accuracy must pass before C10 integration.
  - Automated validation completed:
    - no unmerged entries, precise marker scan, and `git diff --check`: passed.
    - full `python/sglang` and registered-test syntax compile: passed.
    - targeted `E9/F401/F811/F821/F841` ruff gate: passed.
    - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU
      registered files; the new EAGLE core/page/top-k placeholders are present.
    - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
      passed, 24 tests.
    - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
      unsupported CUDA calls and 50 converted kernel launches.
    - AST dispatch check confirmed both centralized DCU cache-location calls
      and their Triton fallback branches.
    - the EAGLE top-k=1 fast-path unit test was attempted but blocked during
      collection because this host has no HIP device; installed `lightop`
      receives `LIGHTOP_GPU_CUS=None`. No test assertion was reached.
    - DCU model, graph, PD, HiCache, EAGLE, and kernel runtime validation remains
      pending on the target runner.
    - SWA-loc backport validation: changed-file `py_compile`, targeted ruff,
      `git diff --check`, and DCU registration with 212 files passed. Runtime
      graph replay remains pending on a DCU node.
- Recommended manual validation:
  - `deepseek-v4` / `attention`: pure TP, CP+EP, DP+EP+MTP112, PD prefill and
    decode, split FlashMLA prefill/decode, sparse prefill, CUDA graph to
    `bs=128`, three repeated requests, and GSM8K 10.
  - `speculative`: EAGLE3/MTP eager and graph, page-size 1/64, cache-location
    operator enabled/disabled, overlap on/off, abort/retract, and logprob parity.
  - `mem_cache` / `disaggregation`: HiCache optimistic prefetch, incremental
    transfer, radix reuse/eviction, and Mooncake/NIXL paths used internally.
  - `model_executor`: DeepSeek-V4 PP hidden-state shape and PD startup.
  - `model`: Qwen2.5 dense/VLM, embedding, reranker, and replicated embedding.
  - `dependency`: DCU install/resolver and import checks with FlashAttention 4
    present or intentionally omitted by the internal image.
- Manual validation result:
  - TBD

### C10 / `47377525cb32`

- Expected focus: CI, mem_cache, and attention.
- Owner: TBD
- Required validation:
  - Stage-b small model smoke.
  - Qwen2.5 dense, VLM, embedding, and reranker smoke.
- Manual validation result:
  - TBD

### C11-C17

- Expected focus: MLA EAGLE, spec naming, MoE, DeepEP, DeepSeek V4, AITER,
  DeepGEMM, and model changes.
- Owner: TBD
- Required validation:
  - Qwen3 MoE smoke.
  - DeepEP small and large.
  - DeepSeek V4 startup and short request.
  - Nightly-dcu.
- Recommended manual validation:
  - `moe`: Qwen3 MoE, EP/TP, groupgemm/marlin, and AITER MoE paths.
  - `deepep`: normal + low-latency dispatch/combine, BF16/FP8 dispatch modes.
  - `deepseek-v4`: startup, short request, MTP/NextN, FP8/FP4 checkpoint path.
  - `aiter`: AITER import/init plus eager and cuda-graph paths if graph code changes.
  - `quantization`: FP8/W8A8/W4A8/MXFP4 smoke and accuracy spot check.
  - `speculative`: EAGLE/MTP/frozen-KV smoke where touched.
  - `nightly`: full nightly-dcu or equivalent internal gate before closing phase.
- Manual validation result:
  - TBD
- Notes:
  - TBD

### C18-C19

- Expected focus: MTP rejection sampling and XPU import guard.
- Owner: TBD
- Required validation:
  - Daily sync smoke gate.
- Recommended manual validation:
  - `daily-sync`: DCU smoke gate, conflict marker scan, DCU registration, and
    official-lag dashboard update.
  - `speculative`: MTP rejection sampling smoke if touched.
  - `platform`: XPU/import guard sanity should not regress DCU imports.
- Manual validation result:
  - TBD
- Notes:
  - TBD
