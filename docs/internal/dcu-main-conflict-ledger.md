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
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `.github/workflows/pr-test-npu.yml`                                                | ci             | Codex | theirs          | Accept the official NPU workflow rewrite; it does not replace or modify the internal DCU workflow                                     | low    | YAML/diff check; runtime pending                                       | NPU workflow syntax check; confirm DCU workflow remains registered                                 | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/pyproject.toml`                                                            | dependency     | Codex | manual merge    | Accept C10 dependencies while retaining the DCU omission of CUDA-only FlashInfer wheels and the internal DeepGEMM pin                 | medium | dependency diff audit; runtime pending                                | DCU editable/wheel install and import smoke                                                       | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/jit_kernel/utils.py`                                                | jit-kernel     | Codex | manual merge    | Add the official local-source hash and MUSA detection while retaining DCU hipcc flags, device key, FP8 target flag, and cache split   | high   | syntax and JIT path audit; runtime pending                            | DCU JIT cache hit/miss, gfx938 FP8 compile, and source-change rebuild                              | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py`     | diffusion      | Codex | manual merge    | Use supplied metadata when available and fall back to forward context; derive query/key lengths independently                         | medium | syntax passed; runtime pending                                        | Diffusion FlashAttention fixed/variable-length smoke                                               | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/configs/model_config.py`                                        | model          | Codex | manual merge    | Preserve internal slimquant/W8A8/modelslim formats and add official Quark MXFP4                                                       | medium | syntax/config audit; runtime pending                                  | Existing DCU quantized model load plus Quark config parse                                          | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/environ.py`                                                     | environment    | Codex | manual merge    | Keep all DCU FlashMLA/split-MLA/FP4/FP8/WO-A knobs and accept the official C10 environment additions                                 | high   | env symbol audit; runtime pending                                     | Parse launch-script envs and test graph/sparse-prefill combinations                                | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`                        | deepseek-v4    | Codex | port to new API | Adopt official per-forward SWA metadata, sparse prefill and direct sgl-kernel paths; retain DCU external FlashMLA and LightOp paths   | high   | syntax/ruff/path audit; runtime pending                               | DSV4 TP/CP/PD, eager/graph, sparse prefill, split MLA, MTP and GSM8K accuracy                       | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`              | attention      | Codex | manual merge    | Accept official lifecycle documentation while preserving the existing HIP radix implementation                                      | medium | syntax passed; runtime pending                                        | HIP radix decode/prefill and graph replay                                                          | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/dsv4/compressor.py`                            | deepseek-v4    | Codex | manual merge    | Add official HIP fused compressor/AITER tgemm only outside DCU; retain DCU LightOp quant-store path                                  | high   | syntax/ruff/DCU guard audit; runtime pending                         | C4/C128 compressor, fused norm/RoPE/Hadamard, LightOp cache store and graph replay                  | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/dsv4/dequant_k_cache.py`                       | deepseek-v4    | Codex | theirs          | Take the official superset with corrected FP32 scale math and reference validation; no historical DCU branch exists                  | high   | syntax passed; runtime pending                                        | FP8 K-cache dequant numerical comparison on gfx938                                                  | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/dsv4/indexer.py`                               | deepseek-v4    | Codex | manual merge    | Accept official local-batch validation while retaining robust option lookup and DCU LightOp TopK priority                            | high   | syntax/path audit; runtime pending                                    | LightOp TopK, FP8 index cache, local batch, eager/graph parity                                      | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/dsv4/metadata.py`                              | deepseek-v4    | Codex | manual merge    | Accept official DeepGEMM/JIT thresholds and top-k-v2, but keep DeepGEMM metadata disabled on DCU                                     | high   | syntax/DCU guard audit; runtime pending                               | Metadata copy/replay, top-k selection, batch-size boundaries and graph to bs=128                   | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`                  | deepseek-v4    | Codex | theirs          | Take the official superset containing request-local rebasing, mask fixes and reusable chunk cache; no DCU branch exists              | high   | syntax passed; runtime pending                                        | Sparse prefill C4/C128 boundary sizes, chunk reuse, CP off/on and accuracy                          | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/attention/triton_backend.py`                             | attention      | Codex | manual merge    | Use official SWA translation lifecycle while retaining DCU CPU seq-lens, AITER extend and GPU-scalar synchronization avoidance        | high   | syntax/path audit; runtime pending                                    | Dense/VLM Triton attention, SWA, target verify, speculative graph and replay                        | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/moe/moe_runner/aiter.py`                                 | aiter          | Codex | manual merge    | Add C10 optional kwargs/no-combine only to generic HIP; DCU remains on validated W8A8/native AITER runners                            | high   | syntax/ruff/DCU dispatch audit; runtime pending                      | DCU W8A8 INT8/FP8, native AITER MoE, eager/graph, DeepEP and no-combine isolation                  | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/quantization/__init__.py`                               | quantization   | Codex | manual merge    | Register both optional Quark loaders without losing internal quantization registrations                                               | medium | import/config audit; runtime pending                                  | Quant-method registry import with and without Quark installed                                      | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`          | quantization   | Codex | manual merge    | Accept online MXFP4 support for capable ROCm GPUs while explicitly excluding DCU from unsupported AITER quant imports                 | medium | syntax/import guard audit; runtime pending                            | DCU import smoke; gfx95 online MXFP4 remains a non-DCU validation                                  | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`                           | mem_cache      | Codex | port to new API | Complete the C10 per-forward SWA translation lifecycle and return native mapping dtype; call sites cast to kernel-required int32      | high   | syntax/call-site audit; runtime pending                               | DSV4 graph replay, allocator reuse, PD transfer and HiCache load-back                              | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/mem_cache/swa_memory_pool.py`                                   | mem_cache      | Codex | port to new API | Accept official allocator relocation and remove obsolete pool-level translation cache; port DCU duplicate-free protection to allocator | high | syntax/import audit; runtime pending                                | SWA allocation/free/reuse, radix eviction, HiCache and graph replay                                | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/srt/model_executor/input_buffers.py`                                | model_executor | Codex | theirs          | Accept the official buffer API cleanup used by CudaGraphBufferRegistry                                                               | high   | syntax/API audit; runtime pending                                     | Decode/prefill graph buffer capture and replay across batch sizes                                  | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`   | test           | Codex | manual merge    | Update the test to use pool-owned SWA location translation                                                                           | medium | syntax passed; runtime pending                                        | Run DSV4 attention unit/accuracy test on DCU                                                      | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `test/registered/lora/test_lora_eviction_policy.py`                                | test           | Codex | manual merge    | Preserve DCU registration and add official XPU registration                                                                          | low    | registration check pending                                            | DCU LoRA eviction smoke                                                                           | merged    |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `test/registered/spec/eagle/test_spec_eagle_topk.py`                               | test           | Codex | manual merge    | Preserve disabled DCU coverage and accept official CUDA registration/estimate update                                                 | medium | registration check pending                                            | EAGLE top-k/page-size graph validation when DCU model assets are available                         | merged    |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `.github/workflows/release-docker-intel-xpu-nightly.yml` | ci | Codex | theirs | Accept official XPU nightly workflow; DCU workflow is independent | low | workflow/static check pending | Confirm DCU workflows remain registered | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `docs_new/index.mdx` | docs | Codex | theirs | Accept official documentation layout | low | docs syntax not a runtime gate | None | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` | jit-kernel | Codex | theirs | Follow official deletion after HiSparse restructuring | medium | deleted-reference scan pending | DSV4 startup/short request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | model_executor | Codex | port to new API | Follow deletion and use TC piecewise runner APIs | high | import/compile pending | Graph capture is optional; model startup remains required | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/speculative/eagle_worker.py` | speculative | Codex | port to new API | Follow official EAGLE worker-v2 migration | high | import/compile pending | EAGLE/MTP startup smoke if enabled | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/jit_kernel/utils.py` | jit-kernel | Codex | manual merge | Combine deterministic prebuilt cache reuse with DCU backend key, source hash, hipcc and gfx flags | high | compile pending | gfx938 JIT compile/import smoke | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/disaggregation/mooncake/conn.py` | disaggregation | Codex | port to new API | Use common KVTransferError while retaining DCU FA KV layout switch | high | compile pending | PD prefill/decode startup and one request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | attention | Codex | port to new API | Adopt TC piecewise/fused-store interfaces while retaining DCU BF16/FP8 index cache and LightOp paged MQA | high | compile pending | DSA model startup and one short request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/attention/dsa_backend.py` | attention | Codex | manual merge | Keep DeepGEMM schedule metadata off DCU and accept official non-DCU target-verify metadata | high | compile pending | DSA metadata init and request smoke | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/attention/fla/layernorm_gated.py` | attention | Codex | manual merge | Keep LightOp layernorm and add official PDL launch kwargs to Triton | medium | compile pending | FLA model startup if available | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/attention/flashattention_backend.py` | attention | Codex | port to new API | Keep DCU FA interface, LightOp metadata and fused-store guards on official KVWriteLoc/SWA lifecycle | high | compile pending | Dense/VLM and DCU FA startup plus short request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/attention/flashmla_backend.py` | attention | Codex | port to new API | DCU uses external FlashMLA and DCU index builder; other platforms use official sgl-kernel FlashMLA and 2D grid | high | compile pending | DeepSeek-V4 startup and short request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/attention/triton_backend.py` | attention | Codex | manual merge | Accept official draft-v2 buffer sizing/SWA metadata and retain DCU token-pool path | high | compile pending | Triton dense startup and one request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/moe/ep_moe/layer.py` | moe | Codex | port to new API | Adopt TC piecewise graph query while retaining DCU DeepEP/AITER/offloader helpers | high | compile pending | Qwen/DeepSeek MoE startup and one request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | moe | Codex | manual merge | Keep DCU LightOp fused sum and accept official NPU dispatch state | medium | compile pending | MoE startup smoke | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/moe/hash_topk.py` | moe | Codex | theirs | Accept official padded-token weight masking and waterfill ordering | medium | compile pending | EPLB/hash-topk model startup | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py` | moe | Codex | manual merge | Add official experimental LoRA env while retaining DCU align helper | medium | compile pending | Triton MoE startup | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/moe/topk.py` | moe | Codex | manual merge | Preserve DCU LightOp top-k priority before official fused top-k pack | high | compile pending | MoE startup and token generation | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/quantization/__init__.py` | quantization | Codex | manual merge | Keep SlimQuant registrations and accept official platform registry | medium | import check pending | Quantized model config/load smoke | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/quantization/fp8_utils.py` | quantization | Codex | manual merge | Preserve tuple input path and add official inductor static-scale path | high | compile pending | FP8 model startup and one request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` | quantization | Codex | manual merge | Keep DCU exclusion from gfx95 AITER path and accept official custom-op registration | medium | import check pending | Clean DCU import; MXFP4 is non-blocking | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/managers/overlap_utils.py` | scheduler | Codex | port to new API | Use official fused gather generally and retain DCU non-Triton compiled gather fallback | high | compile pending | Parallel request startup/inference smoke | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/mem_cache/memory_pool.py` | mem_cache | Codex | manual merge | Add official vectorized 5D layout/OOB checks while retaining DCU FA page layout, graph copy and DSA offload | high | compile pending | KV allocation, model startup and one request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/model_executor/forward_batch_info.py` | model_executor | Codex | manual merge | Keep DCU logging/metadata and accept official deprecation warnings | medium | compile pending | ForwardBatch init through server startup | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | model_executor | Codex | theirs | Drop rename-conflicted legacy runner body and use official phase-aware runner | high | compile pending | Graph is non-blocking; eager startup/request required | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/models/bailing_moe.py` | model | Codex | manual merge | Keep token-pool accessor and add official capture-state API | medium | compile pending | Bailing MoE startup if available | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/models/deepseek_common/utils.py` | model | Codex | manual merge | Keep DCU platform state and accept official gfx95 bpreshuffle capability | high | compile pending | DeepSeek model import/startup | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/models/deepseek_v2.py` | model | Codex | port to new API | Keep DCU fused RMS/quant input layernorm and migrate capture checks to new runner APIs | high | compile pending | DeepSeek V2/V3/V4 startup and request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/models/deepseek_v4.py` | deepseek-v4 | Codex | manual merge | Keep DCU fused RoPE alias and accept official compilation split-op registration | high | compile pending | Primary DeepSeek-V4 startup and short request | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/models/qwen3_5.py` | model | Codex | manual merge | Keep token-pool accessor and add official capture-state API | medium | compile pending | Qwen3.5 startup if available | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` | speculative | Codex | theirs | Use official shape-key/backend capture interface; AMD top-k guard remains upstream | high | compile pending | EAGLE/MTP startup if enabled | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` | speculative | Codex | port to new API | Remove obsolete skip-attention-init argument from target worker call | high | compile pending | Multi-layer EAGLE startup if enabled | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `python/sglang/srt/utils/hf_transformers/config.py` | config | Codex | theirs | Formatting-only conflict around DSA raw config recovery | low | compile pending | DeepSeek config parse | merged |
| C11-C13 /`125ef888921b` | `sync/official-main-C11-C13-20260610` | `test/registered/spec/eagle/test_spec_eagle_page.py` | test | Codex | manual merge | Accept official estimate and preserve disabled DCU registration | low | registration check pending | Enable only after DCU EAGLE page validation | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `.github/workflows/pr-test-npu.yml` | ci | Codex | theirs | Take the official NPU workflow rewrite; internal DCU workflows are independent | low | YAML/static check pending | Confirm DCU workflow and suite registration remain intact | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `.github/workflows/release-docker-intel-xpu-nightly.yml` | ci | Codex | theirs | Take the official isolated checkout/build-context fix for Intel XPU nightly | low | YAML/static check pending | None for DCU runtime | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/bench_serving.py` | benchmark | Codex | theirs | Accept official p90 TTFT/TPOT metrics additions | low | Python compile passed; full static gate pending | Benchmark output schema smoke is non-blocking | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | attention | Codex | manual merge | Add official LoRA-aware gate projection and TC-piecewise guard while retaining DCU BF16/FP8 index cache, LightOp fused Q/K quant-store, and DCU gate-input handling | high | conflict-file compile passed; runtime pending | DeepSeek-V4 startup/request; DSA BF16/FP8 cache and LoRA follow-up | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/dsa_backend.py` | attention | Codex | manual merge | Retain nullable/nested external-FlashMLA metadata copy semantics on DCU; use normal tensor copies elsewhere | high | conflict-file compile passed; runtime pending | DSA metadata capture/replay and startup request | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/dsv4/compressor.py` | deepseek-v4 | Codex | manual merge | Keep explicit DCU detection and LightOp quant-cache path while accepting official weight-attribute plumbing | high | conflict-file compile passed; runtime pending | DSV4 C4/C128/online-MTP compressor startup and request | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/dsv4/metadata.py` | deepseek-v4 | Codex | manual merge | Keep DCU deep-gemm metadata tensor copying separate from generic HIP assignment semantics | high | conflict-file compile passed; runtime pending | Metadata copy under eager/graph and DSV4 request | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/flashattention_backend.py` | attention | Codex | ours | Preserve the validated DCU FA KV layout, varlen/vLLM adapters and page-size behavior; official non-DCU FA skip-KV execution refactor remains outside the DCU functional gate | high | conflict-file compile passed; runtime pending | DCU dense/VLM smoke; review non-DCU `fa_skip_kv_cache` parity separately | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/linear/gdn_backend.py` | attention | Codex | manual merge | Combine official generic HIP fused GDN imports with DCU LightOp/env dispatch | medium | conflict-file compile passed; runtime pending | GDN model smoke when assets are available | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/attention/triton_backend.py` | attention | Codex | manual merge | Retain validated Spec-V2 draft-extend length/QO planning and accept official removal of Spec-V1 graph metadata | high | conflict-file compile passed; runtime pending | Spec-V2 draft extend and Triton startup/request | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/mhc.py` | deepseek-v4 | Codex | manual merge | Preserve DCU AITER TileLang MHC and ROCm bool-allocation patch while accepting official startup prewarm state | high | conflict-file compile passed; runtime pending | DSV4 startup confirms MHC prewarm and no removed model hook | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | moe | Codex | manual merge | Keep DCU bias/i_q/i_s-aware runner call and add official deferred FlashInfer finalize entry point | high | conflict-file compile passed; runtime pending | MoE startup/request and deferred-finalize non-DCU follow-up | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/moe/moe_runner/aiter.py` | aiter | Codex | port to new API | Adopt official Mori/DeepEP quant-aware pre/post-permute contracts and registrations while retaining DCU runner fields, backend key and explicit DCU detection | high | conflict-file compile passed; runtime pending | DCU AITER MoE/DeepEP startup and one request; AG remains disabled | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/quantization/fp8.py` | quantization | Codex | manual merge | Keep DCU on validated native/ASM AITER dispatch; accept generic ROCm FP4 quant-info, padding and shuffled-weight handling outside DCU | high | conflict-file compile passed; runtime pending | FP8 DSV4 startup/request; generic ROCm FP4 follow-up | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/layers/quantization/fp8_kernel.py` | quantization | Codex | manual merge | Keep LightOp import DCU-only and accept official CUDA/MUSA per-token group-quant JIT registrations | high | conflict-file compile passed; runtime pending | DCU FP8 quant startup/request and import smoke | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/managers/overlap_utils.py` | scheduler | Codex | manual merge | Adopt official verified-id/topk/bonus capability flags while routing the fused gather helper through the existing DCU implementation | high | conflict-file compile passed; runtime pending | Overlap startup and short request; Spec-V2 follow-up | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/managers/scheduler.py` | scheduler | Codex | port to new API | Use official `carries_draft_hidden_states()` lifecycle and retain robust fallback for configs without `spec_hidden_size` | high | conflict-file compile passed; runtime pending | PD/standalone metadata buffer startup and request | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/mem_cache/allocator/swa.py` | mem_cache | Codex | manual merge | Adopt official full-page mapping expansion and preserve DCU duplicate/free-page filtering before allocator release | high | conflict-file compile passed; runtime pending | SWA allocate/free/reuse and HiCache load-back | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/mem_cache/common.py` | mem_cache | Codex | manual merge | Accept official pinned-memory helper and keep `dcu_get_last_loc` behind an explicit DCU-only import | high | conflict-file compile passed; runtime pending | Page-size allocation/get-last-loc through model startup | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/model_executor/forward_batch_info.py` | model_executor | Codex | theirs | Take official vectorized multimodal mRoPE delta construction; the prior side was stale list-mutation code | high | conflict-file compile passed; runtime pending | ForwardBatch initialization during service startup | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/server_args.py` | server_args | Codex | port to new API | Drop the obsolete in-class speculative handler; C14-C16 routes validation and defaults through `arg_groups.speculative_hook` | high | conflict-file compile passed; CLI/import gate pending | ServerArgs parse plus service startup | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/speculative/eagle_info.py` | speculative | Codex | port to new API | Accept official compact Spec-V2-only Eagle data model and remove retired Spec-V1 bodies | high | conflict-file compile passed; runtime pending | EAGLE/MTP is non-blocking; startup import must pass | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `python/sglang/srt/speculative/eagle_info_v2.py` | speculative | Codex | port to new API | Accept official allocation/commit-watermark lifecycle; centralized `cache_locs.py` retains explicit DCU `dcu_assign_*` dispatch | high | conflict-file compile passed; runtime pending | Spec-V2 page-size/topk follow-up; service startup import | merged |
| C14-C16 /`2ad00faae1f4` | `sync/official-main-C14-C16-20260616` | `test/registered/unit/server_args/test_server_args.py` | test | Codex | manual merge | Preserve validated DCU suite registration and add official config-merger coverage | low | conflict-file compile passed; registration pending | Run registration and ServerArgs tests | merged |

| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/pyproject.toml` | dependency | Codex | manual merge | Accept official C17-C19 dependency bumps including FlashAttention/FlashInfer/sglang-kernel while retaining current DCU `tilelang==0.1.9` pin | medium | static gates passed; runtime passed | DCU install/resolver smoke if package image changes | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py` | multimodal | Codex | manual merge | Keep official multimodal attention contract; no DCU-specific branch was replaced | low | compile/ruff passed | Diffusion/VLM smoke is non-blocking | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/environ.py` | env | Codex | manual merge | Preserve DCU `SGLANG_USE_LIGHTOP=False` default and existing DSV4 FlashMLA backend env while accepting official C19 env additions | medium | DSA env test passed; import smoke passed | Runtime confirms env defaults during service startup | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/attention/attention_registry.py` | attention | Codex | port to new API | Dispatch NPU first, keep generic HIP only for `_is_hip and not _is_dcu`, and retain DCU DeepSeekV4 backend ownership | high | import smoke and targeted ruff passed | DeepSeek-V4 startup/request passed | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | deepseek-v4 | Codex | manual merge | Keep DCU DSV4 backend behavior on the C19 attention contract; remove stale CP imports after official parallel APIs moved | high | compile/ruff/import passed | DSV4 graph and sparse prefill follow-up | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/attention/dsv4/compressor.py` | deepseek-v4 | Codex | manual merge | Combine official `get_parallel`/NPU support with DCU LightOp compressor path; generic AITER remains guarded by `_is_hip and not _is_dcu` | high | compile/ruff/import passed | DSV4 C4/C128/online-MTP startup | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/attention/flashattention_backend.py` | attention | Codex | manual merge | Adopt official CP-v2 KV materialization hook while retaining validated DCU FlashAttention execution body and FA KV layout | high | compile/ruff/import passed | Dense/VLM attention smoke; non-DCU skip-KV parity follow-up | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/attention/flashmla_backend.py` | attention | Codex | manual merge | Combine official parallel-context API with DCU external FlashMLA import and DCU KV-index creation | high | compile/ruff/import passed | DSV4 FlashMLA startup/request passed | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/attention/triton_backend.py` | attention | Codex | manual merge | Keep DCU optional AITER extend path before official split-KV EAGLE target-verify fast path, then fall back to standard extend | high | compile/ruff/import passed | Spec/EAGLE target-verify follow-up | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/mhc.py` | deepseek-v4 | Codex | manual merge | Preserve DCU AITER TileLang MHC and bool-allocation patch while accepting official TileLang-missing/prewarm handling | high | compile/ruff/import passed | DSV4 MHC startup and graph follow-up | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` | quantization | Codex | manual merge | Combine official parallel/server-args imports with DCU detection and bias/i_q/i_s-aware combine signature | high | compile/ruff/import passed | FP8 compressed MoE smoke | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py` | quantization | Codex | manual merge | Retain official NPU `apply_without_routing_weights` and DCU/GPU `CompressedTensorsW8A8Int8MoE` class | high | compile/ruff/import passed | INT8 compressed MoE smoke | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` | quantization | Codex | manual merge | Keep DCU default zero-point initialization while accepting official zero-point original-shape tracking | medium | compile/ruff/import passed | WNa16 MoE load smoke | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` | quantization | Codex | manual merge | Combine official parallel API with explicit DCU detection; do not let generic HIP override DCU quant path | high | compile/ruff/import passed | Quark INT4/FP8 MoE smoke | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/layers/quantization/unquant.py` | quantization | Codex | manual merge | Preserve DCU W16A16/Marlin paths before generic AITER fallback while adopting official `_aiter_runner`/`AiterMoeQuantInfo` for non-DCU | high | compile/ruff/import passed | W16A16/AITER MoE startup | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/mem_cache/common.py` | mem_cache | Codex | manual merge | Keep `dcu_get_last_loc` DCU-only and accept official NPU platform detection | medium | compile/ruff/import passed | Page-size/cache location smoke | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/model_executor/forward_batch_info.py` | model_executor | Codex | theirs | Use official `get_parallel().attn_tp_size` contract instead of stale local TP helper | medium | compile/ruff/import passed | ForwardBatch initialization during startup | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/model_executor/model_runner.py` | model_executor | Codex | manual merge | Retain current DCU warmup/autotune/dummy-run path and port its imports to C19 runner/base-runner APIs | high | compile/ruff/import passed | Service startup and graph warmup follow-up | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/models/bailing_moe.py` | model | Codex | manual merge | Adopt official `get_parallel()` TP/EP sizing while keeping DCU token-pool/model behavior | medium | compile/ruff/import passed | Bailing MoE startup if model available | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/models/deepseek_v2.py` | model | Codex | port to new API | Keep DCU fused RMS/quant, LightOp and DSA token-pool paths while using official common-utils platform state and C19 MHA/MLA helper imports | high | compile/ruff/import passed | DeepSeek V2/V3/V4 startup/request passed | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/models/deepseek_v4.py` | deepseek-v4 | Codex | manual merge | Combine official NPU/UE8M0 quant changes with DCU LightOp/AITER TileLang MHC and robust WO-A weight layout handling | high | compile/ruff/import passed | Primary DeepSeek-V4 startup/request passed | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/server_args.py` | server_args | Codex | manual merge | Use official strategy-based Mamba radix-cache handling and include DCU in extra-buffer validation | medium | DSA CLI/env test and import smoke passed | Service CLI parse during startup | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `python/sglang/srt/speculative/draft_utils.py` | speculative | Codex | manual merge | Keep DCU ahead of generic HIP for DSV4 draft/prefill backend selection while accepting official NPU dispatch | high | compile/ruff/import passed | MTP/EAGLE smoke is non-blocking | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `test/registered/dp_attn/test_dp_attention.py` | test | Codex | manual merge | Preserve DCU disabled placeholder and official AMD registration imports | low | DCU registration passed | Enable only after BW1100 DP attention validation | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `test/registered/lora/test_lora_update.py` | test | Codex | manual merge | Preserve DCU disabled placeholder and official AMD registration imports | low | DCU registration passed | Enable only after local LoRA adapter/model validation | validated |
| C17-C19 /`62b3c8e17781` | `sync/official-main-C17-C19-20260622` | `test/registered/rl/test_patch_torch.py` | test | Codex | manual merge | Preserve DCU disabled placeholder and official AMD registration imports | low | DCU registration passed | Enable only after multiprocessing HIP tensor validation | validated |

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
  - DCU DeepSeek-V4 CP+EP/DP+EP+MTP112/PD disaggregation of intranode validation - ✅
  - SGLANG_USE_AITER_AG=0 to ensure accuracy ✅ （temporarily disable it）
  - CI test - ✅

### C10 / `47377525cb32`

- Branch: `sync/official-main-C10-20260604`.
- Base: `sync/official-main-bootstrap@f02dd9d1d`.
- Exact official checkpoint: `47377525cb32c21c72949e52804b49ea17ba0c66`.
- Scope: 115 official commits; 23 merge conflicts, kept as one checkpoint.
- Owner: Codex for merge/static validation; DCU runtime owners remain required.
- Resolution summary:
  - Official CudaGraphBufferRegistry, per-forward SWA translation lifecycle,
    DSV4 sparse-prefill/dequant fixes, and moved SWA allocator are the new
    structure.
  - DCU continues to prefer LightOp TopK/cache stores, external FlashMLA,
    gfx938 JIT flags, CPU sequence-length metadata, and the validated AITER
    W8A8/native MoE runners.
  - C10's AITER optional kwargs/no-combine support is enabled only after the
    DCU dispatch returns. The earlier official DeepEP/Mori AITER runner-core
    migration remains a separate tracked task and is not silently enabled here.
  - The SWA allocator relocation retains the internal duplicate/free-page
    protection in `mem_cache/allocator/swa.py`.
  - Automatic HIP semantic audit found and fixed an import hazard in
    `quark_w4a4_mxfp4_moe.py`: online gfx95 MXFP4 quantization is now guarded by
    `_is_hip and not _is_dcu`, matching the dense Quark scheme.
  - Backported `6a4ffcc34a` (`Make Cohere2MoeConfig a dataclass`) from an
    official maintenance branch. Without it, the C10 config blocks
    `server_args` import with current `huggingface_hub` strict dataclasses.
  - Targeted lint found and fixed two additional merge defects: diffusion
    FlashAttention had a duplicated function import and missing forward-context
    import; Triton graph metadata referenced `seq_lens_cpu` without carrying it
    through the helper signature. The latter retains the DCU no-GPU-scalar-sync
    path for sliding-window graph replay.
  - `SGLANG_USE_AITER_AG=0` remains the current DCU accuracy workaround. C10's
    test-only upstream AITER AG workaround does not prove the runtime operator
    correct; re-enable only after the dedicated graph reproducer passes.
- Required validation:
  - `graph-registry`: capture/replay across `bs=1..128`, padding transitions,
    prefill graph, DP attention, PP proxy tensors, MHC hidden states, MTP/EAGLE,
    and graph-on/off output parity.
  - `deepseek-v4`: pure TP, CP+EP, DP+EP+MTP, PD prefill/decode, sparse prefill,
    split FlashMLA, C4/C128 compressor, LightOp TopK, and GSM8K accuracy.
  - `mem-cache`: SWA allocation/free/reuse, radix eviction, duplicate frees,
    HiCache load-back, Mooncake/NIXL PD transfer, and page-size 64 mapping.
  - `aiter`: W8A8 INT8/FP8 and native MoE eager/graph; AITER MLA page-size > 1;
    custom AG must stay disabled for service validation and be tested separately
    with `test/manual/test_aiter_custom_ag_graph.py`.
  - `attention`: Qwen2.5 dense/VLM, Triton SWA, FlashAttention metadata,
    embedding/reranker, target verify, and speculative replay.
  - `jit-kernel`: gfx938 source-hash cache invalidation, FP8 compiler flags,
    DSV4 compressor/dequant numerical comparison, and sgl-kernel smoke.
  - `quantization`: DCU import/load smoke for existing formats; confirm online
    MXFP4 remains rejected cleanly on gfx938 rather than failing at import.
  - `ci`: DCU registration, stage-a, available stage-b suites, and install flow.
- Automated validation result:
  - no unmerged entries, precise conflict-marker scan, and `git diff --check`:
    passed.
  - `python3 -m compileall -q python/sglang test/registered`: passed.
  - targeted `E9/F401/F811/F821/F841` ruff gate for all C10 conflict and DCU
    semantic-fix files: passed. A whole-tree run is not used as a gate because
    the checkpoint contains unrelated existing lint debt.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU
    registered test files.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
    passed, 24 tests, after the Cohere config compatibility backport.
  - CudaGraphBufferRegistry and AITER runner focused CPU tests: 38 passed.
  - SWA allocator focused test: blocked during collection because this host has
    no HIP device and installed LightOp receives `LIGHTOP_GPU_CUS=None`; no SWA
    assertion executed.
  - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
    unsupported CUDA calls and 50 converted kernel launches.
  - DCU service, graph, accuracy, PD, HiCache, AITER and kernel runtime tests
    remain pending on the target runner.
- Manual validation result:
  - DCU DeepSeek-V4 DP+EP+MTP112/PD disaggregation of intranode validation - ✅
  - DCU DeepSeek-V4 TP/CP+EP accuracy validation - ❌(--parallel 1 ✅)
  - CI test - ✅

### C11-C13 / `125ef888921b`

- Branch: `sync/official-main-C11-C13-20260610`.
- Base: `sync/official-main-bootstrap@4bb5b6558`.
- Exact checkpoints: C11 `5160f7914ebf`, C12 `3fe6bc390bdc`, and C13
  `125ef888921bdd657022c4b2f6a264ac86714b38`.
- Scope: 262 official commits and 34 merge conflicts. The checkpoints were
  intentionally grouped under the updated larger-step workflow.
- Resolution summary:
  - Official phase-aware graph runners, TC piecewise APIs, speculative worker
    v2, SWA write-location metadata, vectorized KV layout and test moves form
    the new structure.
  - DCU retains external FlashMLA, DCU FA layout, LightOp metadata/top-k/cache
    store, BF16/FP8 DSA index cache, SlimQuant, DeepEP/AITER helpers and gfx938
    JIT flags through explicit `_is_dcu` paths.
  - Official CUDA DeepGEMM native target-verify layout and PDL optimizations are
    accepted where they do not replace DCU paths. They are not DCU functional
    gates in this checkpoint.
  - Accuracy, throughput, graph replay and large-batch stability are recorded
    as follow-up observations, not merge blockers for the larger-step phase.
- Required functional validation:
  - static import/compile, conflict-marker and DCU registration checks.
  - DeepSeek-V4 service starts successfully on a DCU runner.
  - one short completion request returns normally; content accuracy is not a
    checkpoint gate.
  - when available, one dense or MoE model startup/request smoke confirms the
    generic attention/MoE path is not import-broken.
- Recommended non-blocking validation:
  - graph capture/replay, MTP/EAGLE, PD prefill/decode, DeepEP normal/low
    latency, FlashMLA, DSA, quantized formats and sgl-kernel smoke.
  - accuracy and performance measurements should be attached as observations
    and tracked separately when they fail.
- Automated validation result:
  - no unmerged entries, precise conflict-marker scan and `git diff --check`:
    passed.
  - `python3 -m compileall -q python/sglang test/registered`: passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU
    registered test files.
  - DSA alias/CLI/env manual test: 19 tests passed.
  - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
    unsupported CUDA calls and 50 converted kernel launches.
  - targeted undefined-name audit found one new merge issue in
    `overlap_utils.py` (`Optional` import), which was fixed. Remaining findings
    reproduce on the C10 base and are tracked as pre-existing DCU lint debt.
  - `ServerArgs` import smoke passed. This shell reports no HIP GPU and has no
    Docker CLI, so a real service process cannot be launched here.
- Manual validation result:
  - DCU DeepSeek-V4 DP+EP+MTP112/PD disaggregation of intranode validation - ✅
  - DCU DeepSeek-V4 TP/CP+EP accuracy validation - ❌(--parallel 1 ✅)
  - CI test - ✅

### C14-C16 / `2ad00faae1f4`

- Branch: `sync/official-main-C14-C16-20260616`.
- Base: `sync/official-main-bootstrap@e75c585867`.
- Exact checkpoints:
  - C14 `fda7955890978801727a388d05c14c301f4f7286` (109 commits after C13).
  - C15 `000fc975c7b312ae03f046199e117abbfb7f5b40` (70 commits after C14).
  - C16 `2ad00faae1f4f413330fbc1241dc2c79f44ac4d4` (92 commits after C15).
- Scope: 271 official commits, 896 changed files, and 24 textual conflicts; the
  group stays below the 50-conflict split threshold.
- Resolution summary:
  - Official Spec-V2-only Eagle/DFLASH structures, Mori/DeepEP AITER contracts,
    page-aware SWA release, MHC prewarm state, and new JIT/HiCache interfaces
    form the canonical structure.
  - DCU retains explicit LightOp DSA/cache/top-k, external FA/FlashMLA layout,
    AITER TileLang MHC, native/ASM FP8 MoE, duplicate-safe SWA release, fused
    speculative gather, and gfx938 behavior.
  - `SGLANG_USE_AITER_AG=0` remains the service workaround; AITER allreduce is
    not disabled.
  - The conflict in `flashattention_backend.py` kept the validated DCU FA
    execution body. Official non-DCU `fa_skip_kv_cache` execution parity is a
    named follow-up and is not claimed as validated by the DCU gate.
- Automated validation result:
  - no unmerged entries, conflict-file Python compile, and `git diff --check`:
    passed.
  - `python3 -m compileall -q python/sglang test/registered`: passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU
    registered test files.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
    passed, 19 tests.
  - `AMDGPU_TARGET=gfx938 python3 setup_hip.py --name`: passed with zero
    unsupported CUDA calls and 50 converted kernel launches.
  - current-workspace path check resolved `sglang.__file__` to this repository;
    imports for all 19 conflict runtime modules passed.
  - focused ServerArgs test first found the stale post-init call to the removed
    `_handle_speculative_decoding()` method. It was ported to the official
    `arg_groups.speculative_hook.handle_speculative_decoding(self)` API and the
    focused test then passed.
  - targeted Ruff is blocked: Ruff/Pyflakes/Flake8 are absent in the container,
    and `pip install ruff` could not reach the package index. This is not
    recorded as passed; compile, imports and focused tests are the available
    evidence.
- Required functional validation:
  - verify the imported `sglang` path resolves to this workspace;
  - start the target DeepSeek-V4 service on DCU;
  - complete one short inference request successfully.
- Non-blocking observations:
  - DFlash/Spec-V2 graph, MTP/EAGLE, DeepEP fabric mode, HiCache write-back,
    output accuracy, throughput, graph performance, and topology expansion.
- Manual validation result:
  - DCU DeepSeek-V4 DP+EP+MTP112/PD disaggregation of intranode validation - ✅
  - DCU DeepSeek-V4 TP/CP+EP accuracy validation - ❌(--parallel 1 ✅)
  - CI test - ✅

### C17-C19 / `62b3c8e17781`

- Branch: `sync/official-main-C17-C19-20260622`.
- Base: `sync/official-main-bootstrap@3fe07af4554c`.
- Exact checkpoints:
  - C17 `62ab09a47886c05a664113c7f080a913b20a2924` (107 commits after C16).
  - C18 `f42ec350b431d0305d34d6c70ea45fdfcd29dcad` (64 commits after C17).
  - C19 `62b3c8e17781f9f64653f9bc5b0cb12689ba3ecb` (38 commits after C18).
- Scope: 209 official commits, 694 changed files, and 26 textual conflicts; the
  group stays below the 50-conflict split threshold. The prior placeholder was
  named C18-C19, but merging the C19 endpoint from the live C16 bootstrap also
  includes C17, so this ledger records the actual C17-C19 integration unit.
- Resolution summary:
  - Official C19 remains canonical for dependency manifests, NPU/XPU import
    guards, Spec/MTP rejection-sampling structure, CP-v2 attention hooks,
    staged cache/H2D changes, and the new runner/base-runner helper locations.
  - DCU behavior is preserved through explicit `_is_dcu` or DCU-only imports for
    FlashMLA, FlashAttention KV layout, LightOp compressor/top-k/quant/cache,
    AITER TileLang MHC, W16A16/Marlin/FP8 MoE paths, DSA token-pool access, and
    `dcu_get_last_loc` cache-location helpers.
  - Generic HIP paths that are known to conflict with DCU are guarded as
    `_is_hip and not _is_dcu`, notably DSV4 backend selection, compressor AITER
    usage, and draft/prefill backend dispatch.
  - `model_runner.py` retains the current DCU warmup/autotune/dummy-run logic
    and ports its imports to the C19 `runner.base_runner`/forward-batch APIs.
  - `deepseek_v2.py` uses the official `deepseek_common.utils` platform state as
    authoritative, while adding the helper imports required by retained DCU
    inline MHA/MLA paths.
  - Test registry conflicts preserve DCU disabled placeholders and absorb
    official AMD registration imports.
- Automated validation result:
  - `git ls-files -u | wc -l`: `0`.
  - precise conflict-marker scan with `grep -RInE "^(<<<<<<< |=======$|>>>>>>> )" python test docs/internal`: no output.
  - `git diff --check`: passed.
  - `python3 -m compileall -q python/sglang test/registered`: passed.
  - targeted Ruff `E9/F401/F811/F821/F841` for all conflict files: passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU
    registered test files; it retains the existing warning that
    `test/registered/cpu/utils.py` has no CI registry.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`:
    passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`:
    passed with zero unsupported CUDA calls, 50 replaced kernel launches, and
    package name `sglang-kernel`.
  - `PYTHONPATH=python` import smoke for 20 conflict runtime modules: passed.
- Required functional validation:
  - verify the imported `sglang` path resolves to this workspace;
  - start the target DeepSeek-V4 service on DCU;
  - complete one short inference request successfully.
- Non-blocking observations:
  - MTP rejection sampling, EAGLE/FR-Spec graph replay, draft-weight memory
    accounting, HiCache writing-check/SWA sizing, DP MoE reduce-scatter,
    compressed MoE accuracy, graph performance, throughput, and full topology
    expansion.
- Manual validation result:
  - DCU DeepSeek-V4 DP+EP+MTP112/PD disaggregation of intranode validation - ✅
  - DCU DeepSeek-V4 TP/CP+EP accuracy validation - ❌(--parallel 1 ✅)
  - CI test - ✅


### Main bootstrap landing / `dcu-main-bootstrap-C01-C19-main-20260709`

- Branch: `main`.
- Source branch: `sync/official-main-bootstrap@a5088379e6c886b42cbc95ab9609afabf52830a0`.
- Target base: `origin/main@3ed3bcc06c8e13c80c2cb0f78686bc226cf9c3ee`.
- Scope: land the completed C01-C19 bootstrap into internal `main`; code merged
  without textual conflicts.
- Textual conflicts:
  - `docs/internal/dcu-main-conflict-ledger.md` (`add/add`): resolved with the
    bootstrap-side ledger because it contains the complete C01-C19 conflict and
    validation history; the `origin/main` side was only the initial skeleton.
  - `docs/internal/dcu-main-migration-plan.md` (`add/add`): resolved with the
    bootstrap-side plan, then updated for the post-bootstrap workflow requested
    on 2026-07-09: continue official catch-up on `main`, enter daily sync only
    after reaching current official `main`, then forward-port `v0.5.12_dev` in
    small batches until only `main` remains maintained.
- Validation result:
  - `git ls-files -u | wc -l`: `0` after staging the two resolved documentation
    conflicts.
  - Precise conflict-marker scan on the two resolved docs and whitespace-touched
    files: no output.
  - `git diff --check` and `git diff --cached --check`: passed after normalizing
    upstream CRLF/trailing whitespace in `.github/workflows/pr-test-npu.yml` and
    one space-before-tab instance in `docs_new/docs/advanced_features/pd_disaggregation.mdx`.

### Official main catch-up 20260623 / `6842335fcfb0`

- Branch: `sync/official-main-catchup-20260623`.
- Base: DCU `main@f3b7a57b3d4e1740dc6dce4c8f83a3cc74ccf904`, official previous checkpoint `62b3c8e17781f9f64653f9bc5b0cb12689ba3ecb`.
- Endpoint: official `main@6842335fcfb0cf8fbca537ca79d90576d026cd8f`.
- Scope: 96 official commits, 2026-06-22 to 2026-06-23. This step includes official `v0.5.14` release branch point `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`.
- Release marker note:
  - `v0.5.14` annotated tag object remains `4289f36ef960fad8268a6b94935686e792a81432`, peeled commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb`.
  - Current audit still treats `v0.5.14` as an off-main release tag on `origin/release/v0.5.14`, not as part of official `main`; this step does not merge release-branch cherry-picks.
  - After this step lands on DCU `main` and functional validation passes, create marker tag `dcu-main-official-v0.5.14-branchpoint-20260622` on the DCU main merge commit. The tag message must state the official branch point SHA, DCU main SHA, and that this is a DCU `v0.5.14` baseline point that excludes the release branch's 8 cherry-pick commits.
- Textual conflicts and decisions:
  - `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` (`jit-kernel`, `manual merge`): started from the official HIP/ROCm software FP8 conversion path, then fixed gfx938/DTK compilation by returning the packed FP8x2 value through a `uint16_t`/`fp8x2_e4m3_t` union bit reinterpret instead of relying on an unavailable integer-to-FP8x2 conversion; no separate old-DCU branch existed in the conflicted hunk.
  - `python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py` (`diffusion/attention`, `theirs`): accepted official backend resolver using `AttentionBackendEnum` and global server args; no `_is_dcu` path existed.
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`attention/dsa`, `manual merge`): inserted official `_uses_dsa_attention_backend()` helper while preserving DCU fast-Hadamard, LightOp/topk, device property, and indexer paths; cleaned stale unused DCU temporaries after Ruff.
  - `python/sglang/srt/layers/attention/dsa_backend.py` (`attention/dsa`, `manual merge`): combined official `SGLANG_DSA_TRITON_PREFILL` / gfx95 detection with existing `is_dcu` import and preserved `if _is_hip and not _is_dcu` AITER imports.
  - `python/sglang/srt/layers/attention/triton_backend.py` (`attention`, `manual merge`): kept DCU `seq_lens_cpu` argument while adopting official four-return decode KV buffer update and `num_kv_splits_lens` use.
  - `python/sglang/srt/mem_cache/allocator/paged.py` (`mem_cache`, `manual merge`): retained DCU `dcu_alloc_decode_kernel` / `dcu_alloc_extend_kernel` and `SGLANG_KVALLOC_KERNEL`, while adding official HIP `torch.unique` warm-up.
  - `python/sglang/srt/mem_cache/memory_pool.py` (`mem_cache`, `manual merge`): kept DCU FA KV layout copy path first and returning before generic writes; added official `dcp_kv_mask` masked write for non-DCU-layout cases.
  - `python/sglang/srt/model_executor/forward_batch_info.py` (`model_executor`, `manual merge`): preserved DCU `residual_rms_per_quant_int8` / `rms_quant_flag` and added official `dcp_kv_mask`.
  - `python/sglang/srt/models/deepseek_v2.py` (`deepseek/moe`, `manual merge`): combined DCU residual RMS/LightOp MoE parameters with official `skip_shared_experts`; retained `_is_dcu` exclusion in routed-scaling logic and guarded fused shared-expert hooks with `not skip_shared_experts`.
  - `python/sglang/srt/models/deepseek_v4.py` (`deepseek-v4`, `manual merge`): kept DCU exclusion for generic AITER (`SGLANG_USE_AITER` remains `_is_hip and not _is_dcu`) and added official `_SHARED_EXPERT_LOCAL`.
  - `python/sglang/srt/models/utils.py` (`model/cache`, `manual merge`): added official `dcp_kv_mask` guard while preserving DCU behavior that does not enable the generic HIP fused set-kv path.
  - `python/sglang/srt/server_args.py` (`server_args`, `port to new API`): adopted official `A[...]` CLI auto-registration structure, preserved DCU `record_nolora_graph`, and replaced the stale duplicate manual CLI block with official dynamic/deprecated-only `add_cli_args()` after DSA alias tests caught duplicate `--tool-server` registration.
- Non-textual `_is_dcu` refactor audit fixes:
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`: official functionized `fused_experts_impl` into `_fused_moe_kernel_sequence`; removed an unreachable stale DCU chunk implementation left after the new return path, preserving the active DCU/HIP LightOp path in the new function sequence.
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py` and `fused_moe_triton_kernels.py`: added minimal `deepseek_v4_moe_code_path_checker` objects so retained 2604B DCU instrumentation has a defined counter.
  - `python/sglang/srt/layers/moe/topk.py`: ported DCU LightOp biased grouped top-k to official EPLB/padded-token top-k API by accepting `expert_location_dispatch_info` and `num_token_non_padded` in CPU/GPU helper signatures and forwarding them from `select_experts()`.
  - `python/sglang/srt/layers/attention/dsv4/indexer.py`: official introduced an AITER FP8 paged-MQA logits path that asserts only gfx942/gfx950 are supported. On gfx938 DCU, guard that AITER path with `is_gfx942_supported() or is_gfx95_supported()` so gfx938 falls back to the torch implementation instead of aborting during graph capture; also squeeze `[B, 1]` `seq_lens` to `[B]` before the torch fallback assertion.
- `_is_hip` audit:
  - New official HIP gates were reviewed in DCP, MoE mxfp8, unified-kv-triton, qwen3.5, tests, and server args.
  - Existing high-risk DCU exclusions remain explicit for generic AITER in DeepSeek V4 and DSA backend.
  - `is_unified_kv_triton()` remains generic HIP because `server_args` already forces `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` back to `tilelang` on unsupported non-NVIDIA paths; no DCU path is enabled by default.
- Move/delete audit:
  - `git diff --name-status --find-renames 62b3c8e17781f9f64653f9bc5b0cb12689ba3ecb..6842335fcfb0cf8fbca537ca79d90576d026cd8f` showed deleted Ascend documentation files only; no deleted/renamed DCU runtime file required porting.
  - Three-repo keyword scan (`_is_dcu`, `is_dcu`, `dcu_`, `LightOp`, `flash_mla`, `AITER`, `DeepEP`, `DeepSeekV4`, `SGLANG_USE_AITER_AG`) completed across current DCU, old DCU, and official trees; current DCU retains the expected DCU-specific symbol surface.
- Automated validation result:
  - `git ls-files -u`: no output.
  - Precise conflict marker scan with `grep -RInE "^(<<<<<<< |=======$|>>>>>>> )"`: no output.
  - `git diff --check`: passed.
  - `python3 -m py_compile` over 271 changed Python files: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841` on conflict/high-risk DCU files: passed.
  - Full changed-file Ruff `E9,F821`: passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning about `test/registered/cpu/utils.py` having no CI registry.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed; output package name `sglang-kernel`, zero unsupported CUDA calls, 50 replaced kernel launches.
  - `PYTHONPATH=python python3 -c 'import sglang; print(sglang.__file__)'`: resolved to `/home/proj_sglang_open/dcu-sglang/python/sglang/__init__.py`.
- Required functional validation:
  - Attempt 1 with `/home/scripts/sglang/run_dpsk-v4.sh 31000 /parastor/home/public_user/wanglong/DeepSeek-V4-Flash-FP8-Channel` from this workspace failed during decode graph capture because `fp8_utils.cuh` returned a `uint16_t` where DTK expected `__hip_fp8x2_e4m3`; fixed by the union bit reinterpret above.
  - Attempt 2 passed that JIT compile point but failed on gfx938 because official AITER `deepgemm_fp8_paged_mqa_logits` asserted gfx942/gfx950 only; fixed by the DSV4 indexer support guard above.
  - Attempt 3 reached the torch fallback and failed on `seq_lens.shape == (batch_size,)` because graph capture passed `[B, 1]`; fixed by squeezing singleton trailing dimension before the assertion.
  - Attempt 4 with the default `--cuda-graph-max-bs 128` reached decode graph capture but the gfx938 torch fallback attempted an 8 GiB `torch.bmm()` allocation and failed with HIP OOM. This is recorded as a large-batch graph observation, not a correctness regression in the earlier failed code paths.
  - Attempt 5 used a temporary launcher copied from `run_dpsk-v4.sh` with only `--cuda-graph-max-bs 128` replaced by `--cuda-graph-max-bs-decode 16`, but the machine was blocked by an unrelated existing host SGLang service (`python -m sglang.launch_server ... --port 10030`) occupying all eight DCUs at about 87% VRAM. The smoke failed at weight allocation (`GPU 2 ... 470 MiB free`) before reaching service readiness. Do not kill that external service without owner approval; rerun service `/health` and short `/generate` after GPUs are free.
  - Follow-up DCU pure-TP validation after `72c6a0417` confirmed that the service reaches readiness and `/generate` completes without a worker crash. The returned completion was empty and `SGLANG_DCU_TRACE_FIRST_NAN=1` localized non-finite values to the DeepSeek-V4 attention output and final logits. Per the catch-up acceptance policy and the migration-owner instruction on 2026-07-13, service readiness/request completion satisfy the blocking functional gate; output accuracy remains failed and is explicitly deferred rather than reported as passed.
- Non-blocking observations:
  - v0.5.14 release-branch cherry-picks, accuracy/perf/full topology/large-batch graph, and AITER all-gather graph workaround removal remain out of scope for this catch-up step.
  - The default bs=128 decode graph currently remains unvalidated on gfx938 after the torch fallback OOM. A smaller graph bucket or graph-disabled smoke is acceptable for the catch-up functional gate once the external GPU occupancy is cleared, but do not claim bs=128 graph validation without a successful capture/replay run.

### Official main catch-up 20260625 / `eeee3abbbf81`

- Branch: `sync/official-main-catchup-20260625`.
- Base: DCU `main@d3edaf356`, official previous checkpoint `6842335fcfb0cf8fbca537ca79d90576d026cd8f`.
- Endpoint: official `main@eeee3abbbf8196e54c227faecfd5faba7b1dfc4b`.
- Scope: 110 immutable official commits (the planning estimate was 101), 2026-06-24 to 2026-06-25; 477 files changed in the official range.
- Release marker note:
  - Official `v0.5.14` tag still peels to `49e384ce9d304648e9959666ecb8ce8cd98d0deb`, which is not an ancestor of this official-main endpoint.
  - The tag therefore remains release-branch-only. This step does not merge the eight release-branch cherry-picks and must not create `dcu-main-official-v0.5.14-tag-20260625`.
- Textual conflicts and decisions:
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`attention/dsa`, `manual merge`): preserved DCU LightOp Q/K fusion, BF16/FP8 index-cache paths, page-size-64 handling, and DCU logits behavior; absorbed official BCG/PCG split-op graph surfaces and CUDA-only full-graph split-op checks. Generic HIP FP8 MQA follows official `clean_logits=False`, while DCU keeps `op.mqa_logits`.
  - `python/sglang/srt/layers/attention/triton_backend.py` (`attention`, `manual merge`): retained V1/V2 draft-extend semantics, CPU/tensor extend-length fallback, and SWA metadata updates; replaced the stale deleted `eagle_info_v2` wording with the current draft-extend path.
  - `python/sglang/srt/layers/quantization/unquant.py` (`quantization`, `manual merge`): retained official `topk_output` and `moe_runner_config` plumbing and restored `StandardCombineInput`, which is required by retained DCU Marlin/AITER branches.
  - `python/sglang/srt/managers/overlap_utils.py` (`scheduler/overlap`, `manual merge`): absorbed official `RelayPayload` dataclass typing and removed the obsolete `Union` import.
  - `python/sglang/srt/mem_cache/common.py` (`mem_cache`, `manual merge`): combined official CUDA support with retained DCU environment helpers and `dcu_get_last_loc`.
  - `python/sglang/srt/model_executor/forward_batch_info.py` (`model_executor`, `manual merge`): retained DCU residual RMS quantization fields and added official `attn_dcp_metadata`.
  - `python/sglang/srt/models/deepseek_v2.py` (`deepseek`, `manual merge`): retained DCU BMM/backend mapping and residual-RMS arguments; absorbed official CUDA-only PCG dual-stream and FlashInfer TRT-LLM bypass arguments without widening them to DCU.
  - `python/sglang/srt/models/qwen3_5.py` (`qwen/attention`, `manual merge`): retained DCU LightOp fused RMSNorm/RoPE/KV-store `forward_batch` path and added the official CUDA-only fused QK-norm/RoPE/gate path.
- Non-textual `_is_dcu` refactor audit fixes:
  - `python/sglang/srt/distributed/parallel_state.py`: official added an AITER custom reduce-scatter path whose HIP default would include DCU. Added `not _is_dcu` so DCU uses the registered RCCL fallback until the AITER graph path is independently validated; this does not change AITER all-reduce/all-gather policy.
  - `python/sglang/srt/layers/quantization/fp8.py`: restored the installed AITER `asm_shuffle_weight_b8` import for the retained ASM FP8 MoE path and removed stale unused imports/local state found by Ruff.
  - `python/sglang/srt/utils/common.py`: restored local `get_libnuma` / `numa_bind_to_node` imports after the official NUMA refactor, avoiding undefined names without introducing a top-level circular import.
  - DeepSeek-V4 `enable_multi_stream` remains available to the DCU fused-qnorm path; no prohibited `and not (_is_dcu and _use_fused_qnorm_rope_kv_rope_quant)` bypass was introduced.
- `_is_hip` and move/delete audit:
  - Reviewed new HIP/platform conditions in DCP, DSA graph split, DeepSeek MLA, Qwen fused attention, MoE/DeepEP/top-k, FlashMLA, and AITER collectives. CUDA graph split and Qwen fused paths remain CUDA-only; DCP generic HIP paths intentionally include DCU.
  - `git diff --name-status --find-renames 6842335fcfb0cf8fbca537ca79d90576d026cd8f..eeee3abbbf8196e54c227faecfd5faba7b1dfc4b` removed `moe_runner/flashinfer_mxfp4.py`, `speculative/eagle_info_v2.py`, `dflash_prepare_block.py`, and one NPU test, and renamed `dflash_accept_bonus.py` to `dflash.py`. The removed runtime files contained no DCU path; retained centralized MXFP4 registration and renamed DFlash call sites are not dangling.
  - Three-repo keyword scan (`_is_dcu`, `is_dcu`, `dcu_`, `LightOp`, `flash_mla`, `AITER`, `DeepEP`, `DeepSeekV4`, `SGLANG_USE_AITER_AG`) completed across current DCU, old DCU, and official trees; match counts were 1444, 1244, and 754 respectively.
- Automated validation result:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on changed files: no output.
  - `git diff --check`: passed.
  - `python3 -m py_compile` over 363 changed Python files: passed.
  - Full changed-file Ruff `E9,F821` and targeted conflict/high-risk Ruff `E9,F401,F811,F821,F841`: passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`.
  - `PYTHONPATH=python python3 -c 'import sglang; print(sglang.__file__)'`: resolved to `/home/proj_sglang_open/dcu-sglang/python/sglang/__init__.py`.
- Functional validation policy:
  - Per the 2026-07-13 workflow, only the pure-TP command `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel` is in scope after checking `hy-smi` on both shared-home environments. CI and other models/topologies are owner-run.
  - The previously observed empty-output/NaN accuracy issue is explicitly deferred. Do not add further trace instrumentation or accuracy fixes in this catch-up step.
  - `SGLANG_USE_AITER_AG=0` remains the required operational workaround until the DCU AITER all-gather graph reproducer passes.

### Official main catch-up 20260629 / `f920a37da46e`

- Branch: `sync/official-main-catchup-20260629`.
- Base: DCU `main@b97ca20827da8b9eed8db0cbe0b128f33ccc7aee`, official previous checkpoint `eeee3abbbf8196e54c227faecfd5faba7b1dfc4b`.
- Endpoint: official `main@f920a37da46e1cbb6ba27b76365a622eba593811` (`[AMD] Copy decode result on forward_stream instead of copy_stream (#29642)`).
- Scope: 129 immutable official commits (the planning estimate was 131), 2026-06-26 to 2026-06-29; 799 files changed in the merged result.
- Textual conflicts and decisions (6 files, below the 50-file fallback threshold):
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`attention/dsa`, `manual merge`): adopted official CUDA indexer fusion, graph split-op, and fused K/Q preparation; retained a DCU-first branch for fused LayerNorm/RoPE, BF16 index cache, FP8 QK quant/store, page-size-64 top-k, and DCU logits behavior. The shared helper now returns the official `weights_raw` tuple without sending DCU through CUDA-only JIT fusion.
  - `python/sglang/srt/layers/attention/flashattention_backend.py` (`attention`, `manual merge`): adopted official prefill-aware SWA, scheduler metadata, cascade merge, and device-side page-table generation for non-DCU devices; retained DCU NHD/HND FA layouts, VLLM decode kernel, and `normal_decode_set_metadata_lightop` with bounded host page counts.
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` (`linear-attention`, `manual merge`): kept `causal_conv1d_fn_dcu` dispatch and passed official page-major contiguous state copies and identity cache indices, allowing the official post-kernel scatter back to the strided pool.
  - `python/sglang/srt/managers/scheduler.py` (`scheduler/disaggregation`, `theirs`): used official `get_draft_recurrent_hidden_state_spec()` output for both decode and prefill metadata buffers instead of the stale model-config hidden-size approximation.
  - `python/sglang/srt/mem_cache/memory_pool.py` (`mem_cache`, `manual merge`): kept the validated DCU FA K/V physical layout ahead of generic HND/vectorized/NHD allocation, while accepting official HND copy restrictions and page-major pool infrastructure.
  - `python/sglang/srt/models/deepseek_v2.py` (`deepseek/moe`, `manual merge`): adopted official JIT router/fused-A imports and the shared-expert-before-routed launch order; retained DCU BMM, HIP decode helpers, and fused residual-RMS shared-expert arguments inside the alternate stream.
- High-risk semantic audit:
  - Official page-major MHA/Mamba state layout, HiCache/session changes, DSA fusion, DFlash device-side metadata, DeepSeek-V4 PP SWA layer mapping, and dual-stream MoE ordering were reviewed. Official structure remains canonical, with DCU overrides limited to existing `_is_dcu`/LightOp/layout paths.
  - The endpoint's HIP scheduler change intentionally performs the small decode result D2H copy on the forward stream for DCU/ROCm, avoiding the cross-stream synchronization cost described by the official fix.
  - Deleted runtime files in this range contain no DCU implementation requiring a forward port; the large `.claude` skill deletion and multimodal test moves follow official structure.
  - `/home/scripts/sglang/run_dpsk-v4.sh` still exports `SGLANG_USE_AITER_AG=0`; this catch-up does not remove the workaround.
- Automated validation result before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on changed files: no output.
  - `git diff --check`: passed.
  - `python3 -m py_compile` over 587 changed Python files: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841`: blocked because Ruff/Pyflakes/Flake8 are absent; an isolated `/tmp` Ruff install stalled at the configured package source and was stopped without changing project or system dependencies. This gate is not reported as passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - `PYTHONPATH=python python3 -c 'import sglang; print(sglang.__file__)'`: resolved to `/home/proj_sglang_open/dcu-sglang/python/sglang/__init__.py`.
- Functional validation policy:
  - Run only `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`, then require service readiness, `/health` HTTP 200, and one short `/generate` request completion.
  - Accuracy, throughput, alternate models/topologies, and full CI remain owner-run/non-blocking. If the pure-TP smoke exposes a regression, perform at most one focused fix and one confirmation attempt before returning it for review.
- Pure-TP functional validation result:
  - The exact required command completed weight loading and the default decode graph capture for every bucket from batch size 128 through 1 on all four TP ranks, then reported service readiness on port 10015.
  - `GET /health` returned HTTP 200, and one short `POST /generate` returned HTTP 200 with a completed eight-token response; no worker crash or corrective code change was required.
  - The response contained empty text with eight zero token IDs. This remains the already-deferred, non-blocking accuracy observation and is not reported as an accuracy pass. The test service was stopped cleanly after the request.

### Official main catch-up 20260703 / `88db9e033a11`

- Branch: `sync/official-main-catchup-20260703`.
- Base: DCU `main@ec49eb80ae6bb9044bc6c31b5fd3d3621516b877`, official previous checkpoint `f920a37da46e1cbb6ba27b76365a622eba593811`.
- Endpoint: official `main@88db9e033a11b2d366a8f9d037f027a46ccb9940` (`Adjust KL_THRESHOLD for log probability calculations (#30101)`).
- Scope: 148 immutable official commits (the planning estimate was 156), 2026-06-30 to 2026-07-03; 546 files changed. Git reported 20 textual conflict files, below the 50-file `20260701` split threshold.
- Textual conflicts and decisions:
  - `python/pyproject.toml` (`dependencies`, `theirs`): accepted official TileLang `0.1.11`, paired with the endpoint's tvm-ffi/sgl-deep-gemm upgrade.
  - `python/sglang/jit_kernel/csrc/add_constant.cuh` (`JIT/ROCm`, `theirs`): accepted official `kDLGPU` TensorMatcher dispatch; the shared JIT utility maps it to `kDLROCM` under HIP and to `kDLCUDA` otherwise.
  - `python/sglang/jit_kernel/dsv4/elementwise.py` (`DSV4 JIT`, `manual merge`): combined official XPU dispatch with the retained DCU-specific module selection and cache behavior.
  - `python/sglang/srt/environ.py` (`environment`, `manual merge`): kept DCU FlashMLA split/backend, FP4/FP8, and SWA eviction controls while accepting the official sparse-prefill default. The endpoint's DeepSeek-V4 hook explicitly resets sparse prefill to false on HIP/DCU unless the user sets it.
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` (`DSV4 attention`, `manual merge`): combined XPU imports with DCU LightOp controls and retained the validated DCU decode/prefill split. The official generic FlashMLA tail was not allowed to become a DCU fallback after the existing DCU branches.
  - `python/sglang/srt/layers/attention/dsa_backend.py` (`DSA/CP`, `theirs`): added the official TRT-LLM FP8 KV all-gather helper and retained the existing `_is_hip and not _is_dcu` AITER import boundary.
  - `python/sglang/srt/layers/attention/dsv4/compressor.py` (`DSV4 compressor`, `manual merge`): retained DCU LightOp quant/store selection, accepted the always-V2 official compressor contract, and removed the obsolete non-DCU legacy compressor alias and now-dead AITER tuned-GEMM imports.
  - `python/sglang/srt/layers/attention/dsv4/indexer.py` (`DSV4 indexer`, `manual merge`): added official CUDA-only non-paged indexer planning while retaining DCU detection and the gfx942/gfx95 gate that keeps unsupported gfx938 away from AITER paged-MQA logits.
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` (`linear attention`, `manual merge`): combined the official XPU Dynamo-safe device context with the DCU LightOp `layer_norm_fwd_1pass_opt` branch.
  - `python/sglang/srt/layers/attention/triton_backend.py` (`attention/cache`, `manual merge`): adopted official unified-pool deferred full/SWA location translation and removed the superseded per-step CPU-length plumbing, while preserving the existing backend interfaces outside the refactored graph metadata lifecycle.
  - `python/sglang/srt/layers/mhc.py` (`DSV4 MHC`, `theirs`): removed two stale commented TileLang GEMM arguments; no DCU runtime branch changed.
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` (`MoE`, `manual merge`): retained the DCU LightOp sum/mul/add selector while accepting official per-rank shared-slot and FP8-to-FP4 shared-expert loading changes; dropped an obsolete unused NPU local.
  - `python/sglang/srt/layers/moe/topk.py` (`MoE top-k`, `manual merge`): retained the DCU LightOp grouped-top-k and EPLB/padded-token postprocess arguments, added the official return annotation, and dropped the obsolete eager Kimi import after routing moved to the JIT path.
  - `python/sglang/srt/layers/quantization/unquant.py` (`quant/MoE`, `ours`): retained DCU W16A16 Marlin expert packing and fallback behavior ahead of the official NPU postprocess.
  - `python/sglang/srt/model_executor/forward_batch_info.py` (`model executor`, `manual merge`): retained DCU residual-RMS INT8 fields and accepted the official decode-context-parallel type/lifecycle documentation.
  - `python/sglang/srt/speculative/draft_utils.py` (`spec/DSV4`, `manual merge`): accepted the official Ascend DSV4 draft backends while retaining the DCU exclusion from generic HIP draft routing and removing a stale global-ServerArgs import.
  - `python/sglang/srt/speculative/eagle_utils.py` (`spec`, `theirs`): accepted official Triton tree-build and greedy-verify helpers alongside the existing sgl-kernel route.
  - `python/sglang/srt/speculative/triton_ops/cache_locs.py` (`spec/cache location`, `manual merge`): retained the DCU `kvcacheio` assign-extend fast path ahead of the generic HIP route and added official XPU support to the generic Triton path.
  - `test/registered/quant/test_int8_kernel.py` (`test registry`, `manual merge`): retained the disabled DCU placeholder and added the official AMD nightly registration.
  - `test/run_suite.py` (`test registry`, `manual merge`): retained all DCU nightly suites and added the official XPU nightly suite list.
- High-risk semantic audit:
  - Official DSV4 MHC prewarm is a newer one-shot load-time implementation guarded against NextN reloads; it replaces, rather than restores, the obsolete prewarm hook removed at C07. Its DCU runtime behavior remains part of the pure-TP gate.
  - Official sparse FlashMLA prefill now defaults on globally, but `apply_deepseek_v4_defaults()` disables it by default on HIP/DCU due known ROCm incorrect output. Explicit user overrides remain possible.
  - Unified Mamba/SWA memory-pool location translation, DCP helper moves, shared logits buffers, DSV4 C128 request-state cleanup, and shared-expert fusion were reviewed against retained DCU layout and dispatch paths. Official object lifecycles remain canonical.
  - No deleted runtime file in this range contained a DCU/LightOp/AITER workaround requiring a forward port. `deepseek_v4.py` still excludes DCU from generic AITER, and `/home/scripts/sglang/run_dpsk-v4.sh` still sets `SGLANG_USE_AITER_AG=0`.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on changed files: passed.
  - `git diff --cached --check`: passed.
  - `python3 -m py_compile` over 435 changed Python files: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841`: blocked because Ruff/Pyflakes/Flake8 remain unavailable; no dependency installation was attempted in this step.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - `PYTHONPATH=python python3 -c 'import sglang; print(sglang.__file__)'`: resolved to `/home/proj_sglang_open/dcu-sglang/python/sglang/__init__.py`.
  - `python/pyproject.toml` parsed successfully using setuptools' vendored TOML parser.
- Functional validation policy:
  - Run only `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`, then require readiness, `/health` HTTP 200, and one short `/generate` completion.
  - Accuracy, throughput, alternate models/topologies, and broad CI remain owner-run/non-blocking. If this smoke fails, perform at most one focused fix and one confirmation attempt before returning exact evidence for review.
- Pure-TP functional validation result:
  - Attempt 1 completed distributed initialization and all 46 weight shards, then failed in the new load-time MHC prewarm because `deepseek_v4.py` called `time.perf_counter()` without importing `time`. This was an automatic-merge semantic omission: the official endpoint added both `time` and `fused_q_norm_rope`, while the retained DCU import hunk obscured both additions.
  - Focused fix `ea6273872` restored the official `time` and `fused_q_norm_rope` imports and the official fused-Q path for non-DCU devices while keeping the DCU LightOp RMSNorm/RoPE path unchanged. The file compiled and `git diff --check` passed.
  - The single confirmation attempt again completed distributed initialization and weight loading, then all four TP ranks entered `DeepSeek V4 MHC prenorm prewarm: 16 n_splits buckets`. TP1 failed in the installed DCU AITER TileLang `pre_big_fuse_tilelang` path with `RuntimeError: kernel mhc_pre_big_fuse input gemm_out_mul shape[0] expected 1, but got 32`.
  - The hard stop was reached after one focused fix and one confirmation. Service readiness, `/health`, and `/generate` were therefore not reached; the functional gate is failed and this branch must not advance `main` yet. The test process tree exited and all eight DCUs returned to 2 MiB idle usage.
  - Next review oracle: decide whether the new official load-time MHC prewarm should explicitly skip DCU (preserving the pre-20260703 behavior) or whether the DCU AITER prewarm/cache shape contract should be adapted. A migration-owner-reviewed `_is_dcu` prewarm gate is the smallest candidate; no second fix or retry was performed in this step.
- User-reviewed DCU follow-up and validation (2026-07-14):
  - The owner selected the smallest oracle: preserve the pre-20260703 DCU behavior instead of adapting the installed AITER TileLang prewarm/cache shape contract in this checkpoint.
  - `python/sglang/srt/models/deepseek_v4.py` now returns from `_prewarm_mhc_pre_kernels()` immediately for `_is_dcu`. Non-DCU load-time prewarm remains unchanged; DCU AITER MHC continues to specialize lazily from real request shapes.
  - Focused static checks passed: `git diff --check`, precise marker scan, `python3 -m py_compile python/sglang/srt/models/deepseek_v4.py`, and a workspace import resolving both `sglang` and `deepseek_v4.py` under `/home/proj_sglang_open/dcu-sglang/python` with `_is_dcu=True`.
  - The single confirmation used the exact command `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`. All four TP ranks completed weight loading without entering the incompatible generic MHC bucket prewarm, then captured decode graphs for every configured batch from 128 through 1 and reported service readiness.
  - `GET /health` returned HTTP 200. One short `POST /generate` returned HTTP 200 and completed eight tokens without a worker crash or MHC shape-contract exception. The test service was stopped cleanly afterward and port 10015 was released.
  - The response still contained empty text with eight zero token IDs. This remains a deferred, non-blocking accuracy observation and is not reported as an accuracy pass. The scoped startup/request blocker is resolved; no broader CI or model matrix was run.

### Official main catch-up 20260706 / `9a6f8e599204`

- Branch: `sync/official-main-catchup-20260706`.
- Base: DCU `main@726ca92425c0cac65419686308b9ee2a9c915f80`, official previous checkpoint `88db9e033a11b2d366a8f9d037f027a46ccb9940`.
- Endpoint: official `main@9a6f8e599204aa37481f5f37a1b20938aee98d5c` (`[AMD] Fix DeepSeek V4 MTP accuracy issue (#30333)`).
- Scope: 94 immutable official commits (the planning estimate was 81), 2026-07-04 to 2026-07-06 by official commit date; 414 files changed. Git reported 10 textual conflict files, below the 50-file split threshold.
- Textual conflicts and decisions (10 files, below the 50-file split threshold):
  - `.github/workflows/pr-test-npu.yml` (`CI/NPU`, `theirs + line-ending normalization`): accepted the official single-node NPU job and finish dependency; normalized the official CRLF blob to the LF convention already used by DCU `main`, leaving only the 28-line semantic addition.
  - `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh` (`DSV4 JIT top-k`, `theirs`): accepted the official runtime-topk rewrite and deleted legacy compile-time `SGL_TOPK` implementation. DCU/HIP continues to disable JIT top-k v2 and use the registered backend; the shared JIT header provides the ROCm `__grid_constant__` compatibility macro.
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`DSA/indexer`, `manual merge`): adopted official instance-scoped fusion selection and paged-MQA backend wrappers for CUDA/ROCm, while retaining DCU LightOp fused LayerNorm/RoPE, BF16/FP8 index-cache layouts, page-size-64 handling, and LightOp paged/ragged logits. DCU is selected before generic HIP/AITER dispatch.
  - `python/sglang/srt/layers/moe/topk.py` (`MoE top-k`, `manual merge`): retained DCU LightOp grouped-top-k and postprocess paths, while accepting retirement of the removed AOT fused-gate fake registrations and the unused hard-coded expert-count debug writer.
  - `python/sglang/srt/layers/quantization/__init__.py` (`quantization registry`, `manual merge`): retained both SlimQuant DCU registrations and added the official NPU `mxfp_w4a8` registration.
  - `python/sglang/srt/layers/quantization/unquant.py` (`quantization/MoE`, `manual merge`): added the official type-only `ServerArgs` import and CuTeDSL BF16 GEMM branch, then retained the DCU dtype-alignment fallback before `F.linear`.
  - `python/sglang/srt/models/deepseek_v2.py` (`DeepSeek/MoE`, `manual merge`): retained DCU DSA cache-dequant/attention helpers, added the official unquantized BF16 backend lookup, and accepted Hash-MoE `input_ids` forwarding for TBO sub-batches.
  - `python/sglang/srt/models/deepseek_v4.py` (`DeepSeek-V4/MHC`, `theirs in conflict`): accepted the official `get_flags().enable_dp_lm_head` LM-head structure. The previously validated `_is_dcu` early return from generic load-time MHC prewarm remains intact, so DCU still specializes MHC lazily from real request shapes.
  - `python/sglang/srt/server_args.py` (`ServerArgs/config resolution`, `port to new API`): used the official reordered field layout and declarative override pipeline, replayed compatible DCU fields, and ported DCU DSA page-size, `dcu_mla`, Mamba extra-buffer, LightOp/AITER W8A8, and `record_nolora_graph` semantics to the new view/override APIs.
  - `test/registered/debug_utils/test_dumper.py` (`test registry`, `manual merge`): adopted the official `temp_set_env` import move and preserved the disabled DCU nightly registration alongside AMD/CUDA registration.
- High-risk semantic audit:
  - The endpoint's DeepSeek-V4 MTP fix is present exactly in `deepseek_v4_compress_state.py`: HIP C128 cold state now clears every request row instead of only the sentinel row; this intentionally includes DCU and fixes uninitialized cold-request state.
  - Official ROCm batch-invariant RMSNorm fallback is retained for DCU. DSV4 sparse prefill remains forced off on HIP by the official hook; the pure-TP launcher already requests the same behavior.
  - DSA paged-MQA backend selection is canonical for CUDA/generic ROCm, but DCU LightOp remains an explicit earlier branch and does not execute the new AITER wrapper. DSV4 top-k v2 remains disabled on HIP/DCU by the existing model defaults.
  - `SGLANG_USE_AITER_AG=0` remains unchanged; this checkpoint does not remove or retest the custom all-gather graph workaround.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on staged changed files: no output.
  - `git diff --cached --check`: passed after normalizing the official NPU workflow CRLF blob to the existing LF convention.
  - `python3 -m py_compile` over 304 changed Python files under `python/sglang` and `test/registered`: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841`: blocked because Ruff remains unavailable (`No module named ruff`); no dependency installation was attempted.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - Workspace import resolved `sglang` and `deepseek_v4.py` under `/home/proj_sglang_open/dcu-sglang/python`; `_is_dcu=True`.
- Functional validation status:
  - The only in-scope runtime command remains `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`, followed by `/health` and one short `/generate`.
  - The command was not started during this validation pass because all eight devices were already at 77-79% VRAM usage (about 116-117 GiB of 147 GiB each) from an external/other-container workload not visible in this process namespace. The launcher requests `--mem-fraction-static 0.8`, so starting it with only about 30 GiB free per device would be a guaranteed allocation failure rather than a meaningful checkpoint test.
  - No external process was killed and no runtime fix attempt was consumed. The branch may retain its static merge commit, but must not advance `main` or receive a milestone tag until the pure-TP service gate is run on free devices.
- Pure-TP functional validation follow-up (2026-07-14; supersedes the initial occupied-device status):
  - After the external workload released all eight devices, attempt 1 used the exact required command. All four TP ranks completed distributed initialization, loaded all 46 weight shards, skipped the incompatible generic MHC prewarm through the retained `_is_dcu` gate, and initialized the DSV4 memory pools.
  - The first batch-size-128 decode graph capture then failed in the DCU Hash-MoE fused top-k JIT. The upstream router-GEMM refactor selected `F.linear` on non-CUDA platforms and produced BF16 router logits, while `hash_topk.cuh` requires FP32 (`Tensor<128,256>` matcher rejected `bfloat16`; allowed dtype was `float32`).
  - The single focused fix in `python/sglang/srt/models/deepseek_v2.py` restores the previous DeepSeek-V4 FP32 router contract only for `_is_dcu`: it calls `linear_bf16_fp32` before the new generic `not _is_cuda` branch. Other platforms retain the official endpoint dispatch order. Focused `py_compile`, precise marker scan, and `git diff --check` passed.
  - The one permitted confirmation reran the exact command. All four TP ranks loaded the model, captured every decode graph bucket from batch size 128 through 1, and reported service readiness on port 10015; the Hash-MoE dtype failure did not recur.
  - `GET /health` returned HTTP 200. One short `POST /generate` returned HTTP 200 and completed eight tokens without a worker crash. The response again contained empty text with eight zero token IDs, which remains a non-blocking accuracy observation rather than an accuracy pass.
  - The service was stopped cleanly after the request; port 10015 was released and all eight devices returned to 2 MiB idle usage. No broader CI, model, topology, accuracy, or throughput test was run.
- Code conflict review artifact:
  - `docs/internal/dcu-main-catchup-20260706-conflict-review.md` records the 10 actual textual conflict files and 25 reconstructed conflict hunks against resolved merge `51f025b2d5464a1c35eef12656546d7cc9c56bb1`.
  - The artifact intentionally compares the auto-conflict state with the merge resolution only; the later runtime-only router-dtype fix `0d2e50ec1` remains documented in this ledger rather than being presented as a merge-conflict resolution.

### Official main catch-up 20260709 / `bd7e54d7379e`

- Branch: `sync/official-main-catchup-20260709`.
- Base: DCU `main@b654e63e9815446a27eaf883abf7bf9b9e5e24d8`, official previous checkpoint `9a6f8e599204aa37481f5f37a1b20938aee98d5c`.
- Endpoint: official `main@bd7e54d7379e437cf5f027382d6ca214e046626b` (`[AMD] Fix AITER custom all-gather CUDA-graph capture crash under torch_memory_saver (#30557)`).
- Scope: 90 immutable official commits (the planning estimate was 105), 2026-07-07 to 2026-07-09 by official commit date; 446 files changed. Git reported 11 textual conflict files and 20 conflict hunks, below the 50-file split threshold.
- Textual conflicts and decisions:
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`DSA/indexer`, `manual merge`): accepted official DeepGEMM head padding in generic paged/ragged FP8 logits paths, while preserving the DCU BF16 index cache, LightOp MQA, and the validated DCU CP-ragged FP8 contract. DCU remains selected before generic HIP/CUDA behavior.
  - `python/sglang/srt/layers/dp_attention.py` (`DP attention`, `manual merge`): added official runtime flags while retaining the `is_dcu` import used by the DCU communicator path.
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` (`MoE runner`, `theirs`): accepted official CuTeDSL ModelOpt and unquantized BF16 DeepGEMM/DeepEP low-latency deprecation routing; its explicit `not _is_hip` guard leaves the existing DCU runner selection unchanged.
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` (`DeepEP`, `manual merge`): retained the DCU group-GEMM `quant_type` dispatch ABI, but moved the normal path to official `use_fp8`/`use_nvfp4` low-latency dispatch arguments and scale options.
  - `python/sglang/srt/mem_cache/memory_pool.py` (`KV cache/VMM`, `manual merge`): adopted official `KvBufferDesc` and post-capture VMM backing as canonical, retained the DCU FA physical K/V layout ahead of generic layouts, and ported the DCU per-token copy stride to the new `_create_buffers()` entry point. PD buffer lengths now come from the official descriptors.
  - `python/sglang/srt/model_executor/model_runner.py` (`model executor`, `theirs`): removed the stale local chunked-prefix backend list and used the official `server_args` definition, which already contains `dcu_mla`.
  - `python/sglang/srt/models/bailing_moe.py` (`MoE streams`, `manual merge`): adopted official named stream-pool leasing while preserving stream creation when the DCU SBO path is enabled.
  - `python/sglang/srt/models/deepseek_v4.py` (`DeepSeek-V4`, `manual merge`): retained the DCU rotary helper import and accepted official DSV4 WO-A group-major FP8 quantization/JIT operands. The validated `_is_dcu` MHC prewarm skip remains intact.
  - `python/sglang/srt/speculative/draft_utils.py` (`speculative decoding`, `manual merge`): combined official CPU/AMX helpers with the retained `is_dcu` backend selection.
  - `python/sglang/srt/speculative/triton_ops/cache_locs.py` (`speculative cache`, `manual merge`): kept DCU cache-location wrappers and environment switch before Triton fallback, while adding official CPU wrappers and dispatch.
  - `sgl-kernel/python/sgl_kernel/kvcacheio.py` (`sgl-kernel cache I/O`, `manual merge`): retained all DCU cache-location wrappers and added official CPU all-layer KV-copy binding.
- High-risk semantic audit:
  - The endpoint fix is present exactly in `python/sglang/srt/distributed/parallel_state.py`: AITER custom all-gather uses `all_gather_unreg` during CUDA-graph capture when torch memory saver is active, while the registered path remains for normal capture.
  - `/home/scripts/sglang/run_dpsk-v4.sh` still exports `SGLANG_USE_AITER_AG=0`. This catch-up absorbs and records the official fix but does not remove the operational workaround without a dedicated DCU graph all-gather reproducer.
  - Official template parser files moved from `srt/managers` to `srt/parser`. The retained DCU-only registered OpenAI test import was forward-ported to the canonical parser path; a repository scan found no other stale imports of the removed modules.
  - `deepseek_v2.py` retains the 20260706 `_is_dcu and self.is_deepseek_v4` FP32 router-logit branch. The official FP32 fallback addition applies to non-DSV4 prefill-CP and does not displace that DCU Hash-MoE contract.
  - `attention_registry`, DSV4 JIT/model code, ServerArgs/arg groups, memory cache, and AITER collectives were reviewed. Existing DCU exclusions remain explicit where generic HIP/AITER behavior is not validated.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on staged changed files: no output.
  - `git diff --cached --check`: passed.
  - `python3 -m py_compile` over 352 changed Python files under `python/sglang`, `test/registered`, and `sgl-kernel/python`: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841`: blocked because Ruff remains unavailable (`No module named ruff`); no dependency installation was attempted.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - Workspace import resolved `sglang` and `deepseek_v4.py` under `/home/proj_sglang_open/dcu-sglang/python`; `_is_dcu=True`.
- Functional validation status:
  - Pending the only in-scope runtime command: `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`, followed by `/health` and one short `/generate`.
  - Accuracy, throughput, other models/topologies, broad CI, and AITER all-gather workaround removal remain owner-run/non-blocking.
- Code conflict review artifact:
  - `docs/internal/dcu-main-catchup-20260709-conflict-review.md` records the 11 actual textual conflict files and 20 reconstructed conflict hunks against resolved merge `f4d00bcaae4dd4288fcc206fb70ac27a7211ed3d`.
  - As required by the updated workflow, the artifact contains only conflict-producing files and is intended for VS Code Markdown preview review.
- User-reported DeepEP follow-up (2026-07-14):
  - `/home/scripts/sglang/running_dpsk-v4_nmz22.log` reached EP decode graph capture, then failed in `_DeepEPDispatcherImplLowLatency._dispatch_core()` with `AttributeError: '_DeepEPDispatcherImplLowLatency' object has no attribute 'use_fp8'`.
  - The old DCU implementation computed `use_fp8` as a local variable on every low-latency dispatch. Official refactor `a080358cac7d590bc6582cccf7ec001e62ffd5af` replaced that duplicate logic with canonical dispatcher state and added `self.set_deepep_dispatcher_dtype()` to `_DeepEPDispatcherImplBase.__init__()`.
  - Official checkpoints `9a6f8e599204` and `bd7e54d7379e` both contain that initializer. The DCU parent `b654e63e9815` had silently lost it, and because official did not touch the same initializer hunk in this checkpoint, the three-way merge preserved the stale deletion while accepting later `self.use_fp8` call sites.
  - Focused fix restores the official base-class initializer. It creates `use_fp8`/`use_nvfp4` from the shared dtype resolver before the first graph-capture dispatch; later quantization-specific `set_quant_config()` calls still override the state. DCU group-GEMM retains its existing `quant_type` ABI and does not reintroduce the old duplicated local resolver.
  - Focused `py_compile` and `git diff --check` passed. A constructor-level check forced the FP8 resolver result and confirmed `use_fp8=True` and `use_nvfp4=False` immediately after base initialization.
  - Per the scoped workflow, the EP topology was not rerun by Codex; owner-run EP confirmation should verify that graph capture advances past the former line 788 failure. If a new failure appears, return it for review instead of repeatedly debugging this checkpoint.
- Pure-TP gate note:
  - The first 20260709 pure-TP invocation completed distributed initialization and loaded all 46 shards, but the active tool session was interrupted when the user supplied the EP follow-up request before graph capture/service readiness. It produced no code failure and did not reach `/health` or `/generate`; it is not counted as a passed or failed functional gate.

### Official main catch-up 20260710 / `e1d51be91f6b`

- Branch: `sync/official-main-catchup-20260710`.
- Base: DCU `main@68d965671265f5d3859ba767cc3bd4e94cc03dce`, official previous checkpoint `bd7e54d7379e437cf5f027382d6ca214e046626b`.
- Endpoint: official `main@e1d51be91f6be39e585756568a8f66b99ac2c512` (`[Tiny] Fix a typo in cookbook (#30837)`).
- Scope: 93 immutable official commits; 751 files changed, 34,706 insertions, 11,075 deletions, 65 detected renames, and 9 deletions. Git reported 22 textual conflict files and 37 conflict hunks, below the 50-file split threshold.
- Textual conflicts and decisions:
  - `python/pyproject.toml` (`dependency`, `manual merge`): accepted official FlashInfer `0.6.14` and repaired the adjacent `smg`/`soundfile` dependency split without restoring CUDA-only DCU dependencies.
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` (`batch overlap`, `manual merge`): adopted runtime-context `get_server_args()` while retaining the DCU pinned host buffers.
  - `python/sglang/srt/layers/attention/dsv4/indexer.py` (`DSV4 indexer`, `manual merge`): retained DCU LightOp/gfx support and absorbed official FP8-FNUZ handling.
  - `python/sglang/srt/layers/attention/triton_backend.py` (`attention`, `manual merge`): moved Triton imports to `sglang.kernels`, retained the optional DCU AITER extend path, and kept official metadata contracts.
  - `python/sglang/srt/layers/linear.py` (`linear/FP8`, `manual merge`): adopted official runtime parallel/forward flags while preserving DCU fused SiLU and pre-quantized FP8 tuple paths.
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` (`EP MoE`, `theirs`): accepted the official expert-quant block-size API and removed stale locals exposed by the new contract.
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` (`MoE`, `manual merge`): combined official environment/TBO/NPU plumbing with retained DCU LightOp flags.
  - `python/sglang/srt/layers/moe/moe_align_block_size.py` (`MoE kernel`, `port to new API`): followed the official kernel namespace move without changing DCU dispatch semantics.
  - `python/sglang/srt/layers/moe/topk.py` (`MoE top-k`, `port to new API`): followed the official kernel namespace move and retained DCU LightOp grouped-top-k/postprocess priority.
  - `python/sglang/srt/layers/quantization/fp8_utils.py` (`FP8`, `manual merge`): retained DCU DeepGEMM tuple/prequantized handling while accepting official runtime-context and MXFP8 updates.
  - `python/sglang/srt/managers/overlap_utils.py` (`scheduler overlap`, `manual merge`): accepted official relay/runtime-context types and kept the DCU-compatible pinned result path.
  - `python/sglang/srt/mem_cache/common.py` (`mem cache`, `port to new API`): moved common Triton helpers to `sglang.kernels.ops.memory` and retained DCU cache allocation/location behavior.
  - `python/sglang/srt/mem_cache/memory_pool.py` (`KV cache`, `manual merge`): adopted official descriptor/index-buffer allocation and `_write_mla_kv_buffer(dst_buffer, ...)` contracts while retaining DCU LightOp stores and BF16/FP8 DSA index caches.
  - `python/sglang/srt/mem_cache/memory_pool_host.py` (`host cache`, `manual merge`): ported DCU host transfer behavior to the official descriptor/helper interface.
  - `python/sglang/srt/model_executor/model_runner.py` (`model executor`, `manual merge`): accepted official runtime-context and pool lifecycle changes while preserving the active DCU model/graph configuration.
  - `python/sglang/srt/models/bailing_moe.py` (`MoE/model`, `manual merge`): adopted official scoped forward state and removed retired threaded-allreduce arguments, retaining DCU fused RMS inputs and SBO flags.
  - `python/sglang/srt/models/deepseek_v2.py` (`DeepSeek/MoE`, `manual merge`): adopted official main-first dual-stream/prefetch/forward-flags flow and retained DCU fused RMS inputs plus tuple-output normalization.
  - `python/sglang/srt/models/deepseek_v4.py` (`DeepSeek-V4`, `port to new API`): adopted the official `MqaAttentionBase` to `MQALayer` split and HC helpers; moved the DCU cos/sin buffer and q/WO-A DeepGEMM paths into the new structure, kept generic HIP branches behind DCU-specific behavior, and retained the validated DCU MHC prewarm skip.
  - `python/sglang/srt/models/qwen2.py` (`Qwen`, `manual merge`): adopted runtime `get_server_args()` while retaining the DCU fused attention/RMS path.
  - `python/sglang/srt/models/qwen3.py` (`Qwen`, `manual merge`): adopted runtime `get_server_args()` while retaining the DCU fused attention/RMS path.
  - `python/sglang/srt/models/utils.py` (`model utilities`, `manual merge`): combined official runtime flags with retained DCU model utility hooks.
  - `test/registered/unit/mem_cache/test_radix_cache_unit.py` (`test registry`, `manual merge`): accepted official test updates and retained the DCU registration.
- High-risk semantic and `_is_dcu` move audit:
  - The endpoint introduces the canonical `python/sglang/kernels` namespace. DCU `normal_decode_set_metadata_lightop` moved from the old attention Triton module to `kernels/ops/attention/metadata.py`, and both DCU speculative cache-location operators moved to `kernels/ops/speculative/cache_locs.py`; their DCU call guards and LightOp imports remain present.
  - A source-only stale-import scan found one automatically merged DCU/HIP import in `deepseek_v2.py` still pointing to the removed `srt.layers.attention.triton_ops.rocm_mla_decode_rope`. It was ported to `sglang.kernels.ops.attention.rocm_mla_decode_rope`; the follow-up scan found only intentional migration comments and no stale source imports.
  - DSV4's removed `models/triton_ops/deepseek_v4.py` contained only RMS-normalization helpers. The active DSV4 MHC/normalization implementation remains in `deepseek_v4.py`, `deepseek_common/amd/deepseek_v4_fused_mhc.py`, and the shared/JIT layernorm modules; no deleted symbol remains referenced.
  - Reviewed new HIP conditions in DSA metadata generation, layernorm, MXFP8, DCP cache allocation, MiniMax-M3, ServerArgs overrides, radix attention, and the unified kernel registry. DSA fused metadata remains disabled on HIP; gfx95-only MXFP8 paths do not select gfx938 DCU; DCP/cache and PCG-tail behavior intentionally include DCU. No new generic HIP/AITER branch displaced an existing dedicated DCU path.
  - Three-repository review used official endpoint structure, current DCU behavior, and `/home/proj_dpsk-v4/dcu-sglang` intent for `_is_dcu`, `dcu_`, LightOp, FlashMLA, AITER, DeepEP, DeepSeek-V4, and `SGLANG_USE_AITER_AG` paths. `SGLANG_USE_AITER_AG=0` remains unchanged and is not waived by this merge.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on staged changed files: no output.
  - `git diff --cached --check`: passed after normalizing the official `test/manual/ep/test_eplb.py` CRLF blob to LF.
  - `python3 -m py_compile` over 698 changed Python files: passed.
  - Ruff `E9,F401,F811,F821,F841` over changed Python files: passed.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files; retained the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - Workspace import resolved `sglang`, `deepseek_v2.py`, and `deepseek_v4.py` under `/home/proj_sglang_open/dcu-sglang/python`; `_is_dcu=True`.
- Functional validation status:
  - Pending the only in-scope runtime command: `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`, followed by `/health` and one short `/generate`.
  - Before starting, both `zz-nmz22` and `zz-nmz26` must be checked with `hy-smi`; if neither is fully idle, stop and leave runtime validation to the owner. The two containers share `/home`, so no rsync is permitted.
  - Accuracy, throughput, other models/topologies, broad CI, and the deferred empty-output/NaN issue remain owner-run/non-blocking. A startup/request failure permits one focused fix and one confirmation only.
- Code conflict review artifact:
  - `docs/internal/dcu-main-catchup-20260710-conflict-review.md` records the 22 actual textual conflict files and 37 reconstructed conflict hunks against resolved merge `18d1216680858500bd12d12a739059a24037f026`.
  - The artifact compares the saved auto-conflict state with the merge resolution only; later runtime-only fixes, if any, must stay in this ledger and outside the conflict document.
- Pure-TP functional validation follow-up (2026-07-14):
  - `hy-smi` showed all eight devices on both `zz-nmz22` and `zz-nmz26` at `VRAM 0% / HCU 0.0%`; the test used `zz-nmz22 / rye_sglang_open` and the exact required command.
  - Attempt 1 reached TP worker initialization, then failed before weight loading because conflict-resolved `model_runner.py` still imported the removed `dp_attention.get_attention_tp_size` helper. Official endpoint code now owns attention-TP state in `get_parallel().attn_tp_size`.
  - Focused fix `b37ba82b1631dca586d90772cd2afc6c1a11cf7b` removes the stale import and reads the runtime parallel context for the retained warmup-padding block. File-level `py_compile`, Ruff `E9,F401,F811,F821,F841`, and `git diff --check` passed.
  - Before the one permitted confirmation, `hy-smi` again showed all eight `zz-nmz22` devices idle. The confirmation completed distributed initialization, loaded all 46 shards, initialized the DSV4 pools, and captured every decode graph bucket from batch size 128 through 1.
  - The service reported readiness on port 10015. `GET /health` returned HTTP 200; one short `POST /generate` returned HTTP 200 and completed eight tokens without a worker crash.
  - The response remained empty with eight zero token IDs. This is the already deferred non-blocking NaN/accuracy observation and is not reported as an accuracy pass.
  - The service was stopped cleanly after the request and port 10015 was released. No broader CI, model, topology, accuracy, or throughput test was run.
- Integration decision:
  - Static and functional gates passed under the scoped policy. Fast-forward this validated branch to `main` and create annotated tag `dcu-main-sync-official-20260710`; do not push as part of this step.

### Official main catch-up 20260712 / `82e7cdcff9aa`

- Branch: `sync/official-main-catchup-20260712`.
- Base: DCU `main@ef85596515098410395f504fb2928e3b28f3520b`, official previous checkpoint `e1d51be91f6be39e585756568a8f66b99ac2c512`.
- Endpoint: official `main@82e7cdcff9aa5f49156c3ace73a826f30854ae91` (`[Misc] Remove a few dead code paths in DSA (#30973)`).
- Scope: 34 immutable official commits; 203 files changed, 21,094 insertions, 1,611 deletions, one test-file rename, and no deletions. Git reported 2 textual conflict files, below the 50-file split threshold.
- Textual conflicts and decisions:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` (`DSV4 attention/spec graph`, `manual merge`): accepted official DSpark and compact ragged-verify metadata imports and lifecycle while retaining the DCU `is_dcu` and LightOp quant-cache environment dispatch. The existing DCU branch remains ahead of generic paths.
  - `python/sglang/srt/layers/mhc.py` (`MHC/TileLang`, `port to new API`): adopted official lazy TileLang loading so model-registry discovery does not load native stubs, retained the DCU AITER TileLang MHC route and warmup state, and moved the ROCm `decouple_type_cast` bool-allocation patch into the first real TileLang load before JIT compilation.
- High-risk semantic and move audit:
  - Official DSV4 draft-extend, DSpark, and compact ragged-verify graph metadata are canonical. The retained DCU LightOp quant-cache route and HIP radix backend remain explicit and were not displaced by the new imports or graph-key logic.
  - `MLATokenToKVPoolHost` moved from `memory_pool_host.py` to `mem_cache/pool_host/mla.py`. The moved class retains HIP JIT enablement, staged page-first write-back, layer-sharded transfer behavior, and all supported layouts; all active imports were updated to the new module.
  - DSA fused metadata generation no longer uses the old environment toggle, but every fused path still requires `not _is_hip`, so DCU remains on the existing non-fused metadata route.
  - The only rename is `test_gqa_preill_cp.py` to `test_gqa_prefill_cp.py`; there are no deleted runtime files to forward-port. Source scans found no stale references to the old test name or the previously removed attention/model Triton namespaces.
  - Existing `get_attention_tp_size` references in the legacy DCU MLA backend and KV-cache mixin predate this range and are not touched by Step 1; the planned ModelRunner/ParallelState refactor checkpoint remains the canonical place to reconcile them rather than mixing an unrelated cleanup into this merge.
  - `/home/scripts/sglang/run_dpsk-v4.sh` still sets `SGLANG_USE_AITER_AG=0`, `SGLANG_ROCM_USE_AITER_TILELANG_MHC=1`, and `SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1`; this checkpoint does not remove any workaround.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on staged changed files: no output.
  - `git diff --cached --check`: passed.
  - `python3 -m py_compile` over 192 changed Python files: passed.
  - Ruff `E9,F401,F811,F821,F841`: blocked because neither a Ruff module nor binary is installed in the current host environment; no dependency installation was attempted.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files and the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - Workspace import resolved both `sglang` and `mhc.py` under `/home/proj_sglang_open/sglang-das/python`; `_is_dcu=True`. With the DCU AITER MHC environment switch disabled for the import-only check, real TileLang remained unloaded and absent from `sys.modules`.
- Functional validation status:
  - Before startup, `hy-smi` showed all eight `nmz26` devices at `VRAM 0% / HCU 0.0%` with 2 MiB used per device. Port 10015 was closed, and `sglang.__file__` resolved to `/home/proj_sglang_open/sglang-das/python/sglang/__init__.py`.
  - The exact in-scope command `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel` completed distributed initialization, loaded all 46 shards, initialized the DSV4 pools, and captured all configured decode graph buckets from batch size 128 through 1.
  - The multimem all-gather probe reported `HIP error: invalid argument` and disabled that optional path on every rank; the documented fallback continued through graph capture and request handling without a worker exit.
  - The service reported readiness on port 10015. `GET /health` returned HTTP 200; one short `POST /generate` returned HTTP 200 with `finish_reason=length` and completed eight tokens without a worker crash.
  - The response remained empty with eight zero token IDs. This is the already deferred non-blocking accuracy/NaN observation and is not reported as an accuracy pass; no checkpoint code fix was made for it.
  - The service was stopped cleanly after the request, port 10015 was released, and all eight devices returned to `VRAM 0% / HCU 0.0%`.
  - Accuracy, throughput, alternate models/topologies, DSpark/ragged-verify runtime, broad CI, and the deferred empty-output/NaN observation remain owner-run/non-blocking. A startup/request failure permits one focused fix and one confirmation only.
- Code conflict review artifact:
  - `docs/internal/dcu-main-catchup-20260712-conflict-review.md` was generated from resolved merge `dde320d3772f023256aeb50b51470fefea5cdcf5`.
  - It reconstructs exactly the 2 actual textual conflict files and 2 conflict hunks; automatically merged files are intentionally excluded.
- Integration decision:
  - Static and functional gates passed under the scoped policy. Fast-forward this validated branch to local `main` and create annotated tag `dcu-main-sync-official-20260712`; do not push as part of this step.

### Official main catch-up 20260713 / `f49cbbd67dea`

- Branch: `sync/official-main-catchup-20260713`.
- Base: DCU `main@71c4c42af24f7dda258df84b79995afa50db3af2`, official previous checkpoint `82e7cdcff9aa5f49156c3ace73a826f30854ae91`.
- Endpoint: official `main@f49cbbd67dea602f8616892d2a9882c8c30ae942` (`Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing) (#31001)`).
- Scope: 19 immutable official commits. The official range changes 370 files with 2,668 insertions and 1,573 deletions; the resolved merge changes 373 files with 2,693 insertions and 1,586 deletions because three DCU-only moved-symbol/documentation follow-ups were required. Git reported 8 textual conflict files and 8 conflict hunks, below the 50-file split threshold.
- Textual conflicts and decisions:
  - `python/sglang/srt/layers/attention/dsa/index_buf_accessor.py` (`DSA index cache`, `port to new API`): moved `is_fp8_fnuz` to the canonical `sglang.kernels` namespace while retaining DCU detection and the AITER preshuffle gate.
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` (`DSA TileLang`, `port to new API`): followed the official FP8 helper move and retained `_is_dcu` plus the existing DCU TileLang workaround and dispatch.
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` (`EP MoE`, `manual merge`): accepted the canonical FP8 helper import while retaining DCU quant-method type dispatch for LightOp, AITER, DeepGEMM, compressed tensors, Quark, and NPU paths.
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` (`Triton MoE`, `manual merge`): accepted official batch-invariant/padding imports and the new quantization namespace; the lmslim INT8 quantizer now overrides the canonical helper only under `_is_dcu`.
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` (`W8A8 INT8`, `manual merge`): uses the official INT8 kernel on generic platforms and retains the lmslim quantizer only for DCU.
  - `python/sglang/srt/layers/quantization/w8a8_fp8.py` (`W8A8 FP8`, `manual merge`): accepted canonical FP8 kernel imports while retaining DCU runner selection through `get_moe_runner_backend()`.
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` (`W8A8 INT8`, `manual merge`): accepted the canonical INT8 kernel and retained the lmslim quantizer under `_is_dcu` together with the existing DCU LightOp/AITER runner branches.
  - `python/sglang/srt/models/deepseek_v2.py` (`DeepSeek/DSV4`, `port to new API`): moved all retained DCU MLA FP8 helpers to `sglang.kernels.ops.quantization.fp8_kernel`; the endpoint's BF16 routing-bias fix for modelopt FP4 + FlashInfer TRT-LLM remains present.
- High-risk semantic and move audit:
  - Official moved the quantization kernels/configs into `python/sglang/kernels/ops/quantization` with 166 detected renames. Source scans found two automatically merged DCU references to the removed FP8 module in `jit_kernel/dsv4/elementwise.py` and `debug_flash_mla_adapter.py`; both were ported to the canonical namespace, and the benchmark README output path was updated. A follow-up source scan found no stale imports of the removed FP8/INT8/AWQ modules.
  - The endpoint's NaN-routing correction is present in the merged `MoEGate`: modelopt FP4 with the FlashInfer TRT-LLM runner stores `e_score_correction_bias` as BF16. Existing DCU AITER/compressed-tensor/Quark BF16 routing-bias behavior remains intact.
  - The official range adds no new `is_hip`/`is_dcu` dispatch changes. DCU LightOp, AITER, DeepGEMM, DeepEP, FP8 storage, and DSV4 graph paths therefore remain ahead of generic HIP behavior.
  - The one deletion is an obsolete Ascend GLM example replaced by the new NPU reference/tutorial structure. The Waterfill test/module rename and quantization namespace moves have active imports updated.
  - `/home/scripts/sglang/run_dpsk-v4.sh` still keeps `SGLANG_USE_AITER_AG=0`; this checkpoint does not remove or waive that workaround.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on staged changed files: no unresolved markers.
  - `git diff --cached --check`: passed.
  - `python3 -m py_compile` over 163 changed Python files: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841`: blocked because neither the Ruff module nor binary is installed; no dependency installation was attempted.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files and the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - Import smoke covered all eight conflict modules plus the two moved DCU-only modules and resolved every file from `/home/proj_sglang_open/sglang-das/python`.
- Functional validation status:
  - Pending the only in-scope runtime command: `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel`, followed by `/health` and one short `/generate`.
  - Accuracy, throughput, alternate models/topologies, broad CI, and the deferred empty-output/NaN observation remain owner-run/non-blocking. A startup/request failure permits one focused fix and one confirmation only.
- Code conflict review artifact:
  - `docs/internal/dcu-main-catchup-20260713-conflict-review.md` was generated from resolved merge `b111d8bc66a6ecd8c386fe9110fcf411f9e67650`.
  - It reconstructs exactly the 8 actual textual conflict files and 8 conflict hunks; automatically merged files and the moved-symbol semantic fixes are intentionally excluded.

### Official main catch-up 20260714 / `7e229e2a817d`

- Branch: `sync/official-main-catchup-20260714`.
- Base: DCU catch-up head `52bf6e27831a1547b1f8eb58be5bf6c1508dc296`, official previous checkpoint `f49cbbd67dea602f8616892d2a9882c8c30ae942`.
- Endpoint: official `main@7e229e2a817de7d59e919db7ab3809ab4a22e754` (`support GLM-5.2 MTP index sharing with prefill CP (#30992)`).
- Scope: 26 immutable official commits; the official range changes 534 files with 9,446 insertions and 56,907 deletions, while the resolved merge changes 535 files with 9,515 insertions and 56,969 deletions because the ledger and semantic-audit fixes are included. The large deletion count is dominated by the legacy Sphinx `docs/` removal after the Mintlify cutover. Git reported 9 textual conflict files and 11 conflict hunks, below the 50-file split threshold.
- `v0.5.15.post1` release marker:
  - Official annotated tag object `658e0a942ec771aeeef1b1adf4180764cacd79b2` peels to release-branch commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`; that commit is not an ancestor of official `main` and remains on `release/v0.5.15`.
  - The seven `v0.5.15..v0.5.15.post1` release commits map to main originals `7966f6be` (#30454), `ecb7fb398` (#30627), `24d59d8d` (#30858), `f49cbbd67` (#31001), `78dc58151` (#30839), and `7e229e2a8` (#30992), plus release-only revert `344bd82`; the revert only removes release-branch state and needs no main counterpart.
  - `7e229e2a817d` is therefore the earliest official-main commit containing the complete functional content of `v0.5.15.post1`. This integration is a main-equivalent marker, not a merge or copy of the off-main release tag.
- Textual conflicts and decisions:
  - `python/sglang/kernels/ops/moe/ep_moe_kernels.py` (`MoE/kernel namespace`, `port to new API`): completed the 96%-similar move from `srt/layers/moe/ep_moe/kernels.py`, retaining DCU INT8/FP8 scatter/gather and quant kernels while adding official `moe_permute`; normalized `Optional` and Triton `libdevice` imports at module scope.
  - `python/sglang/srt/configs/model_config.py` (`model config/quantization`, `manual merge`): retained both DCU SlimQuant method names and official Humming registration.
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`DSA/indexer`, `manual merge`): added the official XPU Hadamard implementation without routing DCU away from its LightOp/optimized Hadamard branches. The semantic audit also removed a duplicate runtime-context import, used canonical `get_server_args()` for the DCU PP check, and corrected two non-DCU ragged fallbacks to use the function's `q` tensor instead of an undefined stale name.
  - `python/sglang/srt/layers/linear.py` (`linear`, `manual merge`): retained the local RMS epsilon and accepted official `with_bias` state.
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` (`EP MoE`, `port to new API`): moved kernel and ROCm utility imports to `sglang.kernels`, accepted official Humming selection, and retained DCU AITER as the non-deprecated local path. The audit restored explicit imports for DCU DeepGEMM no-copy-engine transfer, activation, and FP8 group quant helpers after Ruff exposed stale undefined names.
  - `python/sglang/srt/layers/quantization/__init__.py` (`quantization registry`, `manual merge`): retained both SlimQuant registrations and official Humming registration; removed the obsolete placeholder assignment that was immediately overwritten by the real compressed-tensors config.
  - `python/sglang/srt/layers/quantization/fp8_utils.py` (`FP8`, `manual merge`): kept the DCU-only `deepgemm` import and removed a duplicate fake-op definition already supplied by the canonical kernel namespace.
  - `python/sglang/srt/models/deepseek_v4.py` (`DeepSeek-V4/MHC`, `port to new API`): moved RoPE and generic MHC imports to `sglang.kernels` while retaining DCU AITER TileLang `mhc_pre_big_fuse` and `mhc_post_fwd` dispatch ahead of generic HIP.
  - `python/sglang/srt/server_args.py` (`server args`, `manual merge`): retained SlimQuant/LightOp choices and added Humming choices. Removed dead GPT-OSS MXFP4 locals after the endpoint moved those overrides into `arg_groups/overrides.py`.
- High-risk semantic and move audit:
  - Official moved DeepSeek-V4 RoPE, fused Q/K operations, MHC/layernorm, EP-MoE, ROCm MoE utilities, router, sampling hash, and related kernels into `python/sglang/kernels`. Rename detection shows the DCU-bearing MHC file as `R100`, EP-MoE kernels as `R096`, fused MoE Triton kernels as `R098`, and ROCm MoE utilities as `R100`; their DCU/LightOp/AITER symbol counts are retained at the new paths.
  - Source scans found no remaining imports of the removed `srt.layers.mhc`, `deepseek_v4_rope`, `moe.ep_moe.kernels`, or `moe.rocm_moe_utils` module paths. Auto-merge duplicates in `deepseek_v2.py`, `rotary_embedding/mrope.py`, and `schedule_batch.py` were removed.
  - The official range introduces no new `_is_hip`/`is_hip()` or `_is_dcu`/`is_dcu()` dispatch changes. The staged current tree retains 1,372 DCU/high-risk keyword matches versus 822 in the official endpoint.
  - The old semantic-reference path `/home/proj_dpsk-v4/dcu-sglang` is absent in the current shared `/home`, so a fresh three-repository scan is blocked. Decisions use the continuous DCU history already present in `sglang-das`, the prior conflict ledger, and the official endpoint; this limitation is not represented as a completed old-repository audit.
  - `/home/scripts/sglang/run_dpsk-v4.sh` still sets `SGLANG_USE_AITER_AG=0`; this catch-up does not remove or waive that workaround.
- Automated validation before the merge commit:
  - `git ls-files -u`: no output.
  - Precise conflict-marker scan on staged changed files: no output.
  - `git diff --cached --check`: passed.
  - `python3 -m py_compile` over 228 changed Python files: passed.
  - Targeted Ruff `E9,F401,F811,F821,F841` on all 9 conflict files plus 6 high-risk moved/automatically merged files: passed after the semantic fixes above. A broad all-changed-file Ruff run also reported unrelated official/pre-existing warnings outside this targeted gate; no bulk upstream style cleanup was mixed into the merge.
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`: passed with 212 DCU registered test files and the existing warning for `test/registered/cpu/utils.py`.
  - `PYTHONPATH=python python3 test/manual/test_dsa_alias_cli_registry_env.py`: passed, 19 tests.
  - `(cd sgl-kernel && AMDGPU_TARGET=gfx938 python3 setup_hip.py --name)`: passed with package name `sglang-kernel`, zero unsupported CUDA calls, and 50 converted kernel launches.
  - Workspace import resolved to `/home/proj_sglang_open/sglang-das/python/sglang/__init__.py`; `is_dcu()` returned `True`.
- Functional validation status:
  - Preflight `hy-smi` found all eight `zz-nmz22` devices at `VRAM 0% / HCU 0.0%`; all eight `zz-nmz26` devices were occupied at `VRAM 95%`, so only `zz-nmz22` was selected.
  - The exact in-scope command `bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel` failed before model loading because `/usr/local/bin/sglang` could not import the package (`ModuleNotFoundError: No module named 'sglang'`).
  - `pip show sglang` identified the stale editable project location `/home/proj_sglang_open/dcu-sglang/python`, which no longer exists after the workspace move.
  - The one permitted focused environment fix, `python3 -m pip install --no-deps -e python`, failed while building the editable wheel: container Cargo `1.75.0` cannot parse `rust/sglang-grpc/Cargo.toml` because the package requires edition 2024. The installed editable location therefore remains unchanged.
  - Per the migration-owner workflow, no Cargo upgrade, `PYTHONPATH` bypass, or second startup attempt was made. Service readiness, `/health`, and `/generate` remain unvalidated for this step.
  - Owner follow-up on 2026-07-15: after repairing the environment, the migration owner confirmed that the merged branch starts the target service normally and explicitly approved landing it on `main`.
  - This owner result supersedes the earlier environment-only blocker for the catch-up integration gate; request accuracy and the previously deferred empty-output/NaN observation remain outside this startup confirmation.
  - Accuracy, throughput, alternate models/topologies, broad CI, and the deferred empty-output/NaN observation remain owner-run/non-blocking. A startup/request failure permits one focused fix and one confirmation only.
- Code conflict review artifact:
  - `docs/internal/dcu-main-catchup-20260714-conflict-review.md` was generated from resolved merge `310560cc3595f0739c3fb047c9b99425075e1685`.
  - It reconstructs exactly the 9 actual textual conflict files and 11 conflict hunks; automatically merged files and semantic-only fixes are intentionally excluded.
- Integration decision:
  - Code merge, static validation, and owner startup confirmation are complete on `sync/official-main-catchup-20260714`; fast-forward local `main` to this branch.
  - Create annotated tag `v0.5.15.post1` on the resulting DCU `main` commit, explicitly recording official main-equivalent endpoint `7e229e2a817d`, then push `main` and the tag without rewriting remote history.

### v0.5.12_dev forward-port Step 1 / `8736a794acee`

- Branch: `forward-port/v0.5.12-dev-20260715`.
- Base: DCU `main@65f3bd9426e51df40987516acd075b646b858cf6`.
- Old source: `/home/proj_dpsk-v4/sglang-das`, common base
  `d4c6831a107ac03bae80e353d170af15557e4443`, endpoint
  `8736a794acee8253019704cf00a901fd7ffcefbe`.
- Scope: 41 full-graph commits, 30 non-merge commits, 54 old-range files. The
  resolved staged tree changes 48 files with 6,800 insertions and 350 deletions.
- Git reported 20 conflicts: 19 content conflicts plus modify/delete for the
  obsolete `python/sglang/jit_kernel/deepseek_v4.py` wrapper. The wrapper,
  legacy NSA implementation files, and legacy SWA pool file were not revived;
  their required DCU behavior was ported to the current DSV4/DSA and
  `mem_cache/allocator/swa.py` structures.
- Preserved DCU semantics:
  - DSV4 radix TopK early-exit/launch bounds and the existing VMFault guard.
  - BF16 KV-cache layout, pool sizing, canonical store, compressor v1/v2, and
    dtype-aware DSV4 backend behavior.
  - DCU FlashMLA/DSA imports and cached dense Hadamard fallback.
  - LightOp TopK without a duplicate generic EPLB remap.
  - SWA stale-page mapping at its new allocator location.
  - Explicit DCU versus generic ROCm diagnostics; current upstream APIs remain
    canonical.
- Static validation:
  - no unmerged entries or precise conflict markers;
  - staged `git diff --check` passed;
  - all 35 changed Python files compiled;
  - targeted Ruff `E9,F401,F811,F821` passed; seven broad `F841` observations
    are unchanged from the current-main parent and are not Step 1 regressions;
  - DCU registration passed with 221 registered files and the existing CPU
    utils warning;
  - DSA alias/CLI/registry and gfx938 HIP setup gates passed.
- Functional validation:
  - immediate preflight found all devices on both test nodes idle; `zz-nmz22`
    was selected.
  - the exact pure-TP script loaded 46 shards, captured decode graphs from
    `bs=128` through `bs=1`, and reached service readiness;
  - `/health` returned HTTP 200 and one short `/generate` returned HTTP 200
    without worker failure;
  - the response remained empty with eight zero output IDs. This is the known
    deferred NaN/accuracy observation, is not an accuracy pass, and remains
    non-blocking for the current startup/request gate.
- Detailed decisions and validation evidence are recorded in
  `docs/internal/dcu-main-forward-port-v0.5.12-dev-step1-conflict-review.md`.

### v0.5.12_dev forward-port Step 2 / `fde56844fca4`

- Branch: `forward-port/v0.5.12-dev-20260715`.
- Parent: Step 1 merge `e7e06b77881d243291fdc29fc815c2da6b28e75e`.
- Old range: `8736a794acee8253019704cf00a901fd7ffcefbe..fde56844fca442108bf3d2c71cbdeacb4ddb8f08`.
- Scope: 68 full-graph commits, 56 non-merge commits, 130 old-range files;
  resolved code before documentation changes 127 files with 6,914 insertions
  and 544 deletions.
- Git reported exactly 32 textual conflict files. The complete file-by-file
  resolution table is recorded in
  `docs/internal/dcu-main-forward-port-v0.5.12-dev-step2-conflict-review.md`.
- Refactor/move decisions:
  - kept deleted `docs_new/index.mdx` and
    `python/sglang/srt/model_executor/cuda_graph_runner.py` absent;
  - ported graph-token behavior to
    `model_executor/runner/decode_cuda_graph_runner.py`;
  - moved the C16 masked MLA store from old `mem_cache/utils.py` into canonical
    `kernels/ops/kvcache/mla_buffer.py`, with a compatibility re-export;
  - ported old HY3 PD/EP/fused normalization and rotary behavior into current
    model, loader, stream, and capture APIs;
  - migrated HY3 attention rank/size lookup from removed helpers to the current
    attention tensor-parallel API; both HY3 modules pass direct import smoke;
  - extended current multi-modal IPC caching with one source pool per TP
    device and target-device synchronization;
  - integrated DCU standalone/deep-gemm MegaMoE, graph tokens, PD behavior,
    W8A8 builders, AITER ASM shuffle, LightOp TopK, and EP W4A16 into current
    MoE/quant APIs.
- `_is_dcu` audit:
  - dedicated DCU LightOp, DSV4 BF16 cache, MegaMoE, AITER, W4A16, and
    mem-cache paths stay ahead of generic `_is_hip` behavior;
  - DeepSeek-V4 multi-stream is excluded for DCU BF16 attention KV cache, not
    for the previously rejected fused-qnorm condition;
  - BF16 attention KV uses its dedicated scatter regardless of the generic
    fused-store setting and therefore cannot fall through to FP8 packing;
  - the optional DeepGEMM `w4a16_marlin_weight` symbol is now a lazy import
    behind `_is_dcu` and `SGLANG_USE_MARLIN_W4A16_MOE_OPT`;
  - `SGLANG_USE_AITER_AG=0` remains present in the runtime script.
- Static validation:
  - no unmerged entries or precise markers; staged `git diff --check` passed;
  - all 113 changed Python files compiled;
  - targeted Ruff `E9,F401,F811,F821,F841` passed;
  - DCU registration passed with 276 files and the existing CPU-utils warning;
  - DSA alias/CLI/registry passed 19 tests;
  - gfx938 HIP setup passed with package `sglang-kernel`, zero unsupported CUDA
    calls, and 55 replaced kernel launches.
- Functional validation:
  - `zz-nmz26` was not used because all eight devices were at VRAM 93%; all
    eight `zz-nmz22` devices were VRAM 0% / HCU 0% and were selected;
  - the first attempt exposed the unconditional optional DeepGEMM import and
    stopped before weight loading; one focused fix resolved it;
  - confirmation loaded 46 shards, captured decode graphs `bs=128..1`, reached
    readiness, returned HTTP 200 from `/health`, and returned HTTP 200 from one
    short `/generate` without worker failure;
  - response text remained empty with eight zero output IDs, the already
    deferred non-blocking NaN/accuracy observation;
  - service stopped, port 10015 closed, and `zz-nmz22` returned to VRAM 0% /
    HCU 0%.
- Integration decision: scoped static and functional gates passed after one
  focused fix; commit this exact endpoint as the Step 2 no-ff checkpoint.

### v0.5.12_dev forward-port Step 3 / `80571de9491c`

- Branch: `forward-port/v0.5.12-dev-20260715`.
- Parent: Step 2 merge `c7ffa6497a9e783e37a18556639ca7eb6138d292`.
- Old range: `fde56844fca442108bf3d2c71cbdeacb4ddb8f08..80571de9491c8fd80e6822c9fa4efeb02ff67cce`.
- Scope: 57 full-graph commits, 43 non-merge commits, 450 old-range files;
  resolved staged result before documentation changes is 393 files with 8,366
  insertions and 2,117 deletions.
- Git reported exactly 66 conflict files:
  - 12 current CI workflow definitions kept from current `main`;
  - 11 old Sphinx/obsolete documentation paths kept deleted;
  - 34 runtime/model conflicts resolved against current APIs, including FSDP
    streaming, canonical DSA/speculative paths, HY3 precision/PD/EPLB, and
    current disaggregation/communicator contracts;
  - one FlashMLA kernel-Python conflict and eight test conflicts resolved to
    current canonical coverage, with obsolete NSA/EAGLE/PP tests not revived.
- Range-order/revert audit:
  - temporary HY3 PP support and its conflict-resolution commit are followed
    by explicit reverts before the endpoint; the final forward-port does not
    restore that PP implementation;
  - FSDP streaming is retained in the current `WeightLoadPlan`, preprocess,
    and bitsandbytes full-load structure;
  - HY3 retains the range's PD/EPLB, redundant-expert, and precision behavior
    using current graph, stream, parallel, and expert-location APIs.
- `_is_dcu` audit:
  - five runtime Python files initially introduced a non-existent `is_hcu()`
    predicate. They now use `is_dcu/_is_dcu`; HCU remains only the intended
    user-visible compliance label;
  - C++ causal-conv and MoE test helpers follow the same internal-DCU,
    visible-HCU rule;
  - canonical `dsa_*` ServerArgs fields are retained; DCU FP8 DSA resolves to
    `flashmla_auto/flashmla_kv`, DCU non-FP8 to `sparse/sparse`, and generic HIP
    remains `tilelang`;
  - stale `get_attention_tp_size()` use was ported to
    `get_parallel().attn_tp_size`;
  - `SGLANG_USE_AITER_AG=0` remains unchanged.
- Static validation passed: zero unmerged entries, no precise markers,
  `git diff --check`, compilation of all 352 changed Python files, broad
  changed-file Ruff `E9,F821`, targeted high-risk Ruff
  `E9,F401,F811,F821,F841`, ten-module import smoke, 276-file DCU registration,
  19 DSA alias tests, and gfx938 HIP setup-name generation.
- Pure-TP validation:
  - immediate preflight rejected occupied `zz-nmz26` (VRAM 57% on all eight
    devices) and selected idle `zz-nmz22` (VRAM/HCU 0%);
  - the exact required script loaded 46 shards, captured graphs `bs=128..1`,
    reached readiness, and returned HTTP 200 for `/health` and `/generate`;
  - response remained empty with eight zero output IDs, recorded as the known
    non-blocking NaN/accuracy observation rather than an accuracy pass;
  - service stopped, port 10015 closed, and `zz-nmz22` returned to VRAM/HCU 0%.
- Detailed file-by-file groups and evidence:
  `docs/internal/dcu-main-forward-port-v0.5.12-dev-step3-conflict-review.md`.

### v0.5.12_dev forward-port Step 4 / `cf5983854be1`

- Branch: `forward-port/v0.5.12-dev-20260715`.
- Parent: Step 3 merge `d648b38c7f3dd314ca1a5e098144e554949b3a84`.
- Old range: `80571de9491c8fd80e6822c9fa4efeb02ff67cce..cf5983854be1f19237ba28416b438f7b8965cfe6`.
- Scope: 26 full-graph commits, 18 non-merge commits, 30 old-range files;
  resolved code before documentation changes is 32 files with 3,308
  insertions and 295 deletions.
- Git reported exactly 9 textual conflict files:
  - decode KV offload, FlashAttention, DeepEP MoE, HiRadix, the old host-pool
    facade, Mooncake, the removed old CUDA-graph runner, MiniMax M2, and
    ServerArgs;
  - all were resolved against current APIs while retaining the old DCU feature
    intent.
- Refactor/move decisions:
  - moved `MHATokenToKVPoolHostDCU` from the old monolithic host-pool file to
    current `mem_cache/pool_host/mha.py`, including page-major K/V buffers,
    kvcacheio direct/kernel transfers, and explicit unsupported-layout guards;
  - passed `hicache_mem_layout` through decode offload, HiRadix, hybrid pool,
    and draft KV-cache callers so the DCU implementation remains reachable;
  - kept `model_executor/cuda_graph_runner.py` deleted and ported its MiniMax
    gathered-buffer multiple to `runner/base_cuda_graph_runner.py`;
  - retained the current Mooncake hybrid logical-anchor guard and generalized
    recursive registration for tuple/list K/V buffers;
  - adapted old MiniMax/MiMo accessor calls to `get_server_args()` and
    `get_parallel()`.
- `_is_dcu` audit:
  - packed paged-KV can use `SGLANG_KV_LAYOUT_DCU_FA` only when `is_dcu()`;
  - current `_is_hip` DeepEP and quantization paths were reviewed without
    replacing dedicated DCU LightOp/cache behavior;
  - no runtime `is_hcu()` predicate or removed API was introduced;
  - `SGLANG_USE_AITER_AG=0` remains unchanged.
- Static validation passed:
  - zero unmerged entries and no precise markers;
  - `git diff --check` passed after seven upstream trailing-space artifacts in
    `transfer.cu` were removed;
  - all 28 changed Python files compiled;
  - broad Ruff `E9,F821` and targeted high-risk Ruff
    `E9,F401,F811,F821,F841` passed;
  - seven high-risk modules passed direct import smoke;
  - DCU registration passed with 277 files and the existing CPU-utils warning;
  - DSA alias/CLI/registry passed 19 tests;
  - gfx938 HIP setup passed with package `sglang-kernel`, zero unsupported CUDA
    calls, and 56 replaced launches.
- Pure-TP validation:
  - immediate preflight rejected `zz-nmz26`, where all devices were VRAM 93%,
    and selected `zz-nmz22`, where all devices were VRAM/HCU 0%;
  - the exact required script loaded 46 shards, captured graphs `bs=128..1`,
    reached readiness, and returned HTTP 200 for `/health` and `/generate`;
  - response text remained empty with eight zero output IDs, the known
    non-blocking NaN/accuracy observation;
  - no runtime fix or retry was needed; port 10015 closed and all selected
    devices returned to VRAM 0%.
- Detailed conflict decisions and evidence:
  `docs/internal/dcu-main-forward-port-v0.5.12-dev-step4-conflict-review.md`.

### v0.5.12_dev forward-port Step 5 / `5ec8531b096f`

- Branch: `forward-port/v0.5.12-dev-20260715`.
- Parent: Step 4 merge `f76fbea9601d31f7a45cd4b4c063de95c18455d3`.
- Old range: `cf5983854be1f19237ba28416b438f7b8965cfe6..5ec8531b096fa3297ab034dedc873aad215f2c35`.
- Scope: 22 full-graph commits, 13 non-merge commits, 17 old-range files;
  resolved code before documentation changes is 16 files with 2,156
  insertions and 615 deletions.
- Git reported exactly 9 textual conflict files:
  - Mooncake connection, FlashAttention, DeepEP MoE, MiniMax INT8 Marlin,
    fused MoE, W8A8 INT8, SWA pool facade, DeepSeek V2/V3.2, and MiMo V2;
  - all were resolved against current APIs while retaining the endpoint's DCU
    transfer, LightOp/AITER, fused quantization, and model behavior.
- Refactor and semantic decisions:
  - Mooncake keeps current strict C128/SWA-ring validation, canonical transfer
    chunks, MiniMax flat transfer, and recursive registration. Heterogeneous
    attention TP transfers a dedicated SWA/DSA slice and skips the generic
    sender to avoid duplicate writes;
  - `swa_memory_pool.py` remains the pool facade. The old simplified
    `free_swa` edit is superseded by current `mem_cache/allocator/swa.py`,
    which already expands full pages, deduplicates, rejects already-free
    pages, merges released pages, and clears stale mappings;
  - DeepSeek V3.2 fused gate/up RMS quantization and routed/shared expert
    activation scales were ported into current scoped communication and
    deferred-finalization contracts; removed down-projection and all-reduce
    parameters were not restored;
  - DeepEP INT8 uses LightOp only under `_is_dcu`; generic platforms retain
    the current DeepGEMM alias. The endpoint's AITER W4A16 MoE_C support is
    retained on the current fused-MoE runner API;
  - MiMo retains TP1 fused-QKV and MTP-as-SWA loading, KME/RoPE, EPLB, and
    resume behavior using current runtime-context accessors;
  - import smoke found that the installed LightOp lacks
    `mimo_v2_split_rope_vscale_kv_store`. One focused capability guard keeps
    the fused DCU path when available and falls back to the unfused path when
    unavailable instead of failing module import;
  - no runtime `is_hcu()` predicate was introduced, and the generic-HIP fused
    clamp remains explicitly excluded from DCU's dedicated quantized path;
  - `/home/scripts/sglang/run_dpsk-v4.sh` still exports
    `SGLANG_USE_AITER_AG=0`.
- Static validation:
  - no unmerged entries or precise conflict markers;
  - staged `git diff --check` passed;
  - all changed Python files compiled;
  - broad Ruff `E9,F821` and targeted high-risk Ruff
    `E9,F401,F811,F821,F841` passed;
  - eleven high-risk modules imported after the one capability fix;
  - DCU registration passed with 277 files and the existing CPU-utils warning;
  - DSA alias/CLI/registry passed 19 tests;
  - gfx938 HIP setup passed with package `sglang-kernel`, zero unsupported CUDA
    calls, and 56 replaced kernel launches.
- Pure-TP validation:
  - immediate preflight rejected `zz-nmz26`, where all devices were VRAM 93%,
    and selected `zz-nmz22`, where all devices were VRAM/HCU 0%;
  - workspace import resolved to the current
    `/home/proj_sglang_open/sglang-das/python` tree;
  - the exact required script loaded 46 shards, captured graphs `bs=128..1`,
    reached readiness, and returned HTTP 200 for `/health` and `/generate`;
  - response text remained empty with eight zero output IDs, recorded as the
    known non-blocking NaN/accuracy observation rather than an accuracy pass;
  - no runtime retry was needed; port 10015 closed and all selected devices
    returned to VRAM/HCU 0%.
- Status: committed as merge `8a075ddc63af713025cc585fa8d37a84cc99e217`; validated.
- Detailed conflict decisions and evidence:
  `docs/internal/dcu-main-forward-port-v0.5.12-dev-step5-conflict-review.md`.

### Official main daily 20260721 / `d6ef68881e26`

- Branch: `sync/official-main-daily-20260721`.
- DCU parent: `ccb9a976e9c3b7556fd6abbca0e4e251c187b678`.
- Common official base: `7e229e2a817de7d59e919db7ab3809ab4a22e754`.
- Official endpoint: `d6ef68881e263812d4901f632786015005c4d050`
  (`[NPU] Adapt MiMo-V2.5-W8A8 (#29131)`, 2026-07-21).
- Scope: 314 official commits and 1,350 official changed paths; resolved staged
  tree has 1,348 paths with 131,551 insertions and 25,672 deletions.
- Git reported exactly 48 textual conflict files. They cover DSV4/DSA and
  FlashAttention, quantization/MoE, memory-pool and model-runner refactors,
  ServerArgs/speculative code, disaggregation/IPC, two removed qserve sources,
  documentation/configuration, and tests.
- High-risk resolution decisions:
  - allocation DCU behavior moved from `mem_cache/common.py` to
    `mem_cache/allocation.py`;
  - model KV sizing moved from the deleted `model_runner_kv_cache_mixin.py` to
    `pool_configurator.py`;
  - DCU pull/push custom-allreduce behavior moved into the new unified JIT
    custom-allreduce sources, with obsolete headers kept deleted;
  - DSV4 sparse prefill uses the official workspace/live-sequence structure
    and has one implementation; `_is_dcu` selects Hygon `flash_mla` while the
    generic path uses `sgl_kernel.flash_mla`;
  - FlashAttention retains official CP-v2 materialization and separates KV
    cache write scales from FA kernel descales, while preserving DCU cache-write
    and custom-kernel behavior;
  - DeepSeek V2's conflict-spliced qkv functions were repaired; DeepSeek V4
    retains DCU weight-scale fallback and avoids incompatible post-load FP8
    scale setup;
  - CUDA IPC retains per-device DCU pools and applies official consumer-count
    acknowledgement exactly once to every consumed source pool;
  - official MXFP8, elastic-EP, and experimental SGL-Marlin ServerArgs handlers
    are present together with existing DCU handlers;
  - qserve sources and the old model-runner KV mixin remain deleted, with no
    stale references.
- `_is_dcu` refactor audit searched current, official, and old DCU trees for
  platform predicates and DCU backend symbols. Dedicated DCU LightOp, AITER,
  DeepEP, FlashMLA, FP8/cache-layout, and graph paths remain ahead of generic
  HIP where behavior differs. `SGLANG_USE_AITER_AG=0` remains unchanged.
- Static validation:
  - zero unmerged entries and no precise conflict markers;
  - staged `git diff --check` passed after normalizing one upstream CRLF file;
  - all changed Python files compiled;
  - isolated pyflakes found and the merge fixed missing Triton/logger/Qwen
    imports plus a duplicate DSV4 sparse-prefill definition; remaining reports
    were triaged as official conditional-import or CLI-annotation patterns;
  - DCU registration passed with 277 files;
  - DSA alias/CLI/registry validation passed;
  - gfx938 HIP setup-name validation passed with package `sglang-kernel`, zero
    unsupported CUDA calls, and 56 replaced launches.
- Grouped high-risk import smoke terminated during GPU extension/plugin loading
  without a Python traceback; it is not counted as a passed import gate.
- Pure-TP validation: blocked before launch. `zz-nmz26` had VRAM 95% on all
  eight devices and HCU 25.0%--98.6%; no model command was run. The stopped
  `zz-nmz22` container was not started implicitly.
- Detailed file-by-file decisions and evidence:
  `docs/internal/dcu-main-daily-20260721-conflict-review.md`.

### Official main daily 20260721 — post-merge fixed patch review

- Base merge: `d328f8371812c26320ee1409c8ef2f98ee0773cd`.
- Review scope: six uncommitted runtime/import fixes applied after the daily
  merge. No new upstream commit was merged and no v0.5.15.post1_dev work was
  mixed into this review.
- Retained DCU-compatible fixes:
  - renamed DCU `USE_ROCM` JIT dtype specializations from `_dtype_trait` to the
    official `DLDataTypeTrait` interface;
  - skipped the legacy HIP tvm-ffi monkey-patch only when
    `tvm_ffi.cpp.load_inline` itself is absent, which matches the installed
    package's `tvm_ffi.cpp.extension` layout;
  - moved the two DSV4 LightOp imports to the canonical kernels namespace;
  - restored the official `init_cublas` helper required by current attention
    backend setup;
  - repaired a submitted no-op NPU comment into the historical two-value
    `FusedMoEMode` compatibility enum required by the retained local
    `ep_moe/layer.py` post-load path.
- Evidence: `git diff --check`, changed-Python compilation, direct imports of
  `FusedMoEMode`, DSV4 LightOp, `init_cublas`, `DeepEPMoE`, and the installed
  new tvm-ffi entrypoint all passed.
- Runtime status: not run by this review. A fresh `hy-smi` idle preflight is
  required before the single pure-TP service test. The patch is deliberately
  uncommitted, not merged to `main`, and not pushed pending user review.
- Detailed review:
  `docs/internal/dcu-main-daily-20260721-fixed-patch-review.md`.


### Official main daily 20260728 — in progress

- Working branch: `sync/official-main-daily-20260728`; base `main` is
  `49764eb373a`.  Official endpoint is
  `edc0e5489f2ff8b42eb0d48fbcf7137d1931d4d2`.
- The no-commit upstream merge has 32 resolved textual conflict files.  The
  resolution keeps the official `jit_kernel` to `kernels` migration and ports
  HCU behavior to the resulting owners, including DSV4 attention, MoE/quant,
  cache allocation/configuration, server arguments, custom all-reduce, and
  FlashAttention paths.  New platform predicates consistently use `_is_hcu`;
  no runtime `_is_dcu` predicate remains in the HCU paths.
- Static evidence before service validation: no unmerged entries, no precise
  conflict markers, staged `git diff --check`, compilation of all 1,010 changed
  Python files, DSA alias/CLI registry test (19 passed), HCU registration
  validation (274 registered files), and `setup_hip.py --name` (zero unsupported
  CUDA calls, 56 replaced launches).  Ruff was unavailable in the container.
- Pure-TP attempt on the only idle node, `zz-sglang2` / `rye_sglang_0720`,
  reached DeepSeek-V4 construction but stopped at the shared-expert FP8 setup:
  `CompressedTensorsConfig.weight_block_size` is absent.  The generic
  block-size assertion was therefore bypassed only for compressed-tensors.
- The subsequent retry exposed that `_DeepseekV4ConfigAlias` intentionally has
  no `quantization_config`.  `deepseek_v2.py` now queries that optional field
  with `getattr` and, for the alias case, identifies compressed-tensors from
  the instantiated layer quant-method module.  This preserves the ordinary
  FP8 block-size equality assertion and is awaiting one user-run pure-TP retry;
  it has compiled and passed staged whitespace validation.
- Separate non-blocking migration audit observation: optional `mimo_v2_nextn`
  import warns that its old `load_mimo_v2_qkv_proj_weight_v2` helper is absent
  after the official model refactor.  It does not participate in this
  DeepSeek-V4 startup path, but must be audited before finalizing this daily
  sync.
- Follow-up pure-TP startup on `zz-nmz26` reached checkpoint loading and found
  a second removed compatibility path: compressed-tensors supplies
  `wq_a/wkv.*weight_scale_inv`, whereas the fused `wqkv_a` layer owns
  `.*weight_scale`.  The official refactor had removed the old scale-name
  resolver, so the fused loader raised a `KeyError` for the `_inv` target.
  `deepseek_v4.py` restores a parameter-existence based resolver only for these
  scale aliases, applies it after WQA/WKV fusion, accepts both scale spellings,
  and retains the pre-remap cache key.  It also guards `Exception.add_note` for
  the Python 3.10 containers; this prevents masking a future loader error.
  The changed file compiled and staged whitespace validation passed.  No second
  model launch was issued automatically; pure-TP retry is pending user result.
- A 2026-07-28 12:38 user retry then completed all 46 weight shards and
  memory-pool initialization, confirming the compressed-tensors fixes.  CUDA
  Graph setup exposed a separate namespace-conflict omission: official and the
  pre-merge branch retain `ServerArgs.minimax_opt`, and 14 consumers still use
  it, but the merged namespaced `ServerArgs` definition had dropped the field.
  The field is restored under `NS("parallel")` with its original `False`
  default; direct `ServerArgs` construction passed.
- After a clean `hy-smi` preflight (all eight devices at VRAM 0%, HCU 0%), one
  agent-run pure-TP attempt at 12:51 passed weight loading and memory-pool setup
  and entered decode graph capture.  Capture then failed at
  `dsv4/metadata.py:124` because the merge retained an `_is_hcu` condition but
  dropped its `is_hcu` import and module-level initialization.  Those two lines
  are restored from the pre-merge HCU implementation on the official metadata
  structure.  The adjacent `copy_()` audit also restored the validated
  `is_hip() and not _is_hcu` guard so HCU does not enter the generic HIP
  metadata assignment path.  The file compiles and direct import reports
  `_is_hcu=True`.
- A later user retry reached compressed-tensors FP8 GEMM during graph capture
  but failed in `torch._scaled_mm` because hipBLASLt found no valid solution.
  `fp8_utils.py` now restores the validated HCU DeepGEMM path after the
  official quantization flow, avoiding `torch._scaled_mm` for these DSV4
  channel-wise shapes.
- The bounded automatic pure-TP debug sequence on `zz-nmz26` /
  `rye_sglang_0716` used an idle `hy-smi` preflight before every launch and
  stopped after five failed launches as requested:
  1. Graph capture passed the hipBLASLt site, then failed importing
     `aiter.ops.triton.fusions.fused_clamp_act_mul`.  The generic HIP fused
     clamp path is now excluded on `_is_hcu`.  The same audit restored the
     DSV4 HCU FP32 Hash-MoE router and the HCU exclusion from generic routed
     scaling.
  2. Graph capture reached Hash-MoE and failed because the new
     `kernels/ops/attention/dsv4/moe.py` used `_is_hcu` without defining it.
     Its `is_hcu` import and module-level predicate are restored.
  3. Graph capture reached the DSV4 indexer, where AITER's paged-MQA kernel
     asserted that only gfx942/gfx950 are supported.  The pre-merge
     architecture capability predicate is restored; gfx938 no longer enters
     that AITER implementation, and the HCU LightOp fallback remains available.
  4. The resulting Torch fallback received `seq_lens` shaped `[batch, 1]`
     and failed its one-dimensional assertion.  The validated compatibility
     squeeze from the pre-merge HCU implementation is restored.
  5. Graph capture then progressed through that assertion but the Torch paged-
     MQA fallback attempted an additional 8 GiB allocation and failed with HIP
     OOM (136.09 GiB already allocated, 5.52 GiB free).
- Runtime status: not passed.  The five-attempt limit has been reached, so no
  further patch or launch was made.  The next review should decide whether
  `_is_hcu` must select LightOp before
  `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`; do not treat the Torch OOM as a request
  to reduce the established test topology.  After shutdown, all eight devices
  report VRAM 0% and HCU 0%.
- Follow-up graph-memory audit proved that the 80%+ steady-state allocation is
  the configured `--mem-fraction-static 0.8` KV-pool target, not new model
  weight growth: the current and 2026-07-22 successful logs both report
  `bytes_per_full_token=7705.45` and about 46.5--46.8 GiB of model weights.
  The older run captured decode graphs through bs=128 with only 0.67 GiB of
  extra memory.
- The direct graph OOM regression was a merge-resolution loss in
  `server_args.py`: the pre-merge HCU guard from `cba9e72cc0` (renamed by
  `b781224d4d`) was overwritten, enabling the generic Torch paged-MQA fallback
  on HCU.  At bs=128 it materialized an 8 GiB FP32 scores tensor.  The
  `not is_hcu()` guard and its missing import are restored.  As a defensive
  second layer, `dsv4/indexer.py` now selects HCU LightOp before the Torch
  fallback.  Both files compile and staged whitespace validation passes.
- Open-graph retest status: not launched.  The mandatory `hy-smi` preflight
  found HCU 0 at VRAM 3% (all other cards VRAM 0%; all HCU utilization 0%),
  so validation stopped without terminating the unknown owner.  A fresh
  all-zero preflight is required before retrying the pure-TP command.
- 2026-07-28 MHC graph/runtime follow-up on `zz-nmz22` / `rye_sglang_0716`:
  - `mhc_pre_big_fuse` first rejected the split dimension (`expected 1, got
    32`) and then the padded GEMM last dimension (`expected 24, got 32`).
    The HCU path now passes `big_fuse_n_splits` and compacts the valid
    `[..., :hc_mult3]` prefix before calling AITER, preserving the HCU AITER
    implementation while respecting its fixed physical layout contract.
  - Static evidence after the fix: `mhc.py` compiled and staged
    `git diff --check` passed. Decode CUDA graph capture completed all 20
    shapes (bs=128 through bs=1), with about 0.64 GB additional graph memory
    and about 24.6 GB available afterward.
  - Functional smoke: `/health` returned HTTP 200 and `/generate` returned
    HTTP 200. The request produced an empty text/all-zero token IDs; this is
    retained as a non-blocking precision observation per the current user
    workflow and does not block the daily merge.
  - No temporary trace probes were added. The staged branch remains suitable
    for commit; precision investigation is explicitly deferred.
