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


| Checkpoint          | Merge branch                      | Conflict file                                                                      | Area           | Owner | Strategy        | Reason                                                                                                                     | Risk   | Validation                                                           | Follow-up                                                                                          | Status    |
| ------------------- | --------------------------------- | ---------------------------------------------------------------------------------- | -------------- | ----- | --------------- | -------------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------- |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `.github/workflows/_pr-test-stage.yml`                                             | ci             | Codex | theirs          | Official renamed`check-stage-health` to `check-pr-test-health`; this is a pure workflow action rename                      | low    | precise marker scan passed; DCU registration passed                  | None                                                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `.github/workflows/pr-states.yml`                                                  | ci             | Codex | manual merge    | Keep official`workflow_run` PR lookup and preserve DCU `/rerun-failed-ci` stale-run wording                                | low    | precise marker scan passed; DCU registration passed                  | Verify real GitHub workflow behavior in CI dry-run                                                 | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/environ.py`                                                     | env            | Codex | manual merge    | Keep both DCU`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` and official `SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN` env switches        | low    | syntax compile passed; DCU registration passed                       | None                                                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/mem_cache/memory_pool.py`                                       | mem_cache      | Codex | manual merge    | Preserve DCU FA KV layout copy/load while taking official`current_platform.synchronize()`                                  | medium | syntax compile passed; DCU registration passed                       | Add CPU-offload smoke on DCU FA layout when CI command is available                                | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/managers/scheduler_pp_mixin.py`                                 | scheduler      | Codex | manual merge    | Combine official`hc_hidden_size` fallback with DCU proxy hidden-state shape helper                                         | medium | syntax compile passed; DCU registration passed                       | Pipeline-parallel smoke if available                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `test/registered/tokenizer/test_multi_detokenizer.py`                              | test           | Codex | theirs          | Official C01 renames CUDA suite from`stage-b-*` to `base-b-*`; AMD registration remains from DCU tree                      | low    | syntax compile passed; DCU registration passed                       | None                                                                                               | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/moe/ep_moe/layer.py`                                     | moe            | Codex | ours            | High-risk official MoE runner restructuring overlaps DCU DeepEP/AITER paths; preserve known DCU runtime for C01            | high   | syntax compile passed; DCU registration passed                       | Owner should port official runner-core changes onto DCU path in later checkpoint                   | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/moe/moe_runner/aiter.py`                                 | aiter          | Codex | ours            | Official AITER runner refactor conflicts with DCU W8A8/W16A16 handling; avoid semantic rewrite in first checkpoint         | high   | syntax compile passed; DCU registration passed                       | Create dedicated AITER merge task before enabling official runner-core behavior                    | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/quantization/unquant.py`                                 | quantization   | Codex | ours            | Conflict is inside MoE execution fallback; keep DCU behavior until AITER/MoE owner ports official path                     | high   | syntax compile passed; DCU registration passed                       | Revisit together with`moe_runner/aiter.py`                                                         | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/models/deepseek_v2.py`                                          | model          | Codex | ours            | DeepSeek forward path overlaps DCU fused RMS/quant behavior; preserve current DCU model path in C01                        | high   | syntax compile passed; DCU registration passed                       | Run DeepSeek V2/V3 smoke after C01; port official output-buffer context if needed                  | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/models/deepseek_v4.py`                                          | model          | Codex | ours            | DeepSeek V4 has many DCU-specific imports/kernels and official changes are not safe to fold blindly                        | high   | syntax compile passed; DCU registration passed                       | DSV4 owner should review skipped official C01 hunks before C02/C03                                 | validated |
| C01 /`c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | ours            | Preserve DCU speculative-algorithm alias helper; official side did not supersede it in C01                                 | medium | syntax compile passed; DCU registration passed                       | Confirm Gemma4 assistant draft CLI path still works if used internally                             | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `.github/workflows/pr-states.yml`                                                  | ci             | Codex | manual merge    | Keep official run status icon/link behavior and preserve DCU`/rerun-failed-ci` stale-run wording                           | low    | marker scan passed; targeted compile passed; DCU registration passed | Verify real GitHub workflow behavior in CI dry-run                                                 | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/pyproject.toml`                                                            | dependency     | Codex | ours            | Avoid adding unconditional CUDA`flashinfer_python[cu13]` and `flashinfer_cubin` dependencies to DCU install path           | medium | marker scan passed; targeted compile passed; DCU registration passed | Revisit with CUDA/Docker owner if internal main must match official CUDA dependency set exactly    | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh`                            | jit-kernel     | Codex | theirs          | File was deleted upstream and no current tree references remained                                                          | medium | marker scan passed; targeted compile passed; DCU registration passed | None                                                                                               | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh` | jit-kernel     | Codex | theirs          | File was deleted upstream and no current tree references remained                                                          | medium | marker scan passed; targeted compile passed; DCU registration passed | None                                                                                               | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/jit_kernel/deepseek_v4.py`                                          | jit-kernel     | Codex | manual merge    | Keep DCU BLASLt env path while adding official HIP/AITER imports and guard                                                 | medium | marker scan passed; targeted compile passed; DCU registration passed | DSV4 JIT smoke on DCU                                                                              | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/dsv4/indexer.py`                               | attention      | Codex | manual merge    | Use official dynamic TOPK from output shape instead of fixed 512                                                           | low    | marker scan passed; targeted compile passed; DCU registration passed | DSV4 sparse prefill/topk smoke                                                                     | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/flashmla_backend.py`                           | attention      | Codex | theirs          | Official tuple-style forward mode check is equivalent and cleaner                                                          | low    | marker scan passed; targeted compile passed; DCU registration passed | FlashMLA/DSV4 smoke                                                                                | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`                     | attention      | Codex | manual merge    | Preserve DCU page-size 64 assertion while accepting official HIP preshuffle page-size check for non-DCU HIP                | medium | marker scan passed; targeted compile passed; DCU registration passed | NSA/DCU index cache smoke                                                                          | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`                            | attention      | Codex | manual merge    | Preserve DCU BF16 index-cache path and adopt official`device_index` budget API                                             | high   | marker scan passed; targeted compile passed; DCU registration passed | NSA topk/chunking smoke                                                                            | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/moe/ep_moe/layer.py`                                     | moe            | Codex | ours            | Official low-latency MoE runner updates overlap DCU AITER/DeepEP/groupgemm paths; keep known DCU implementation for C02    | high   | marker scan passed; targeted compile passed; DCU registration passed | Dedicated MoE owner should port official C02 hunks separately                                      | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`                          | deepep         | Codex | ours            | Official DeepEP dispatcher API changes conflict with DCU quantized dispatch and low-latency dispatch parameters            | high   | marker scan passed; targeted compile passed; DCU registration passed | Confirm topk, BF16 dispatch, and low-latency dispatch compatibility before later checkpoints       | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/quantization/fp8.py`                                     | quantization   | Codex | ours            | Preserve DCU AITER/ASM FP8 MoE shuffle behavior; official shuffle path is CUDA/HIP-generic and needs DCU review            | high   | marker scan passed; targeted compile passed; DCU registration passed | Review with AITER/MoE owner                                                                        | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/managers/tp_worker.py`                                          | scheduler      | Codex | theirs          | Official fix avoids undefined`model_worker_batch` in split prefill sampling                                                | low    | marker scan passed; targeted compile passed; DCU registration passed | Split-prefill smoke if available                                                                   | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/model_executor/forward_batch_info.py`                           | model_executor | Codex | ours            | Preserve current DCU pinned-memory construction for extend lengths                                                         | medium | marker scan passed; targeted compile passed; DCU registration passed | Forward batch init smoke                                                                           | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/models/deepseek_v4.py`                                          | model          | Codex | manual merge    | Keep DCU FP8 WO-A GEMM shape compatibility and add official`ceil_to_ue8m0` scale conversion / rotary import                | high   | marker scan passed; targeted compile passed; DCU registration passed | DeepSeek V4 startup and short request smoke                                                        | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | theirs          | Official Gemma4 backend selection supports causal and conditional arch plus split backend validation                       | low    | marker scan passed; targeted compile passed; DCU registration passed | Server args unit smoke                                                                             | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/speculative/eagle_worker_v2.py`                                 | speculative    | Codex | theirs          | Official side fixes stale`model_worker_batch` references to use `batch`                                                    | medium | marker scan passed; targeted compile passed; DCU registration passed | EAGLE/MTP smoke if available                                                                       | validated |
| C02 /`425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/test/ci/ci_register.py`                                             | ci             | Codex | manual merge    | Keep DCU backend/marker and add official XPU backend/marker                                                                | low    | marker scan passed; targeted compile passed; DCU registration passed | DCU registration script                                                                            | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh`                          | jit-kernel     | Codex | theirs          | Official unified 512/1024 top-k into`topk_v1.cuh`; the standalone header is no longer referenced                           | medium | reference scan passed; targeted compile passed                       | Run DSV4 top-k 512/1024 JIT smoke on DCU                                                           | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh`                            | jit-kernel     | Codex | manual merge    | Adopt official dynamic top-k kernel and preserve DCU`SGL_GRID_CONSTANT` plus HIP shared-memory attribute setup             | high   | marker scan passed; targeted compile passed                          | Run DSV4 top-k 512/1024 compile and numeric smoke on DCU                                           | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/jit_kernel/deepseek_v4.py`                                          | jit-kernel     | Codex | port to new API | Remove the monolithic module and use official`sglang.jit_kernel.dsv4` split modules; model imports were ported             | high   | deleted-file reference scan passed; targeted compile passed          | Validate DSV4 JIT, BF16-FP32 GEMM, compressor, and fused RoPE paths                                | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/disaggregation/mooncake/conn.py`                                | disaggregation | Codex | manual merge    | Keep DCU requests/ZMQ helpers and add official failed-session probe metrics counter                                        | medium | marker scan passed; targeted compile passed                          | Mooncake reconnect and failed-session probe smoke                                                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/environ.py`                                                     | env            | Codex | manual merge    | Keep DCU FA KV-layout env and add official Mooncake failed-session probe env settings                                      | low    | DSA env/CLI alias test passed; targeted compile passed               | Verify DCU FA KV-layout env override in server startup                                             | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/dsv4/metadata.py`                              | attention      | Codex | port to new API | Import top-k planning from split`dsv4` module while retaining the non-HIP top-k-v2 enable condition                        | high   | targeted compile passed; DSA alias test passed                       | DSV4 metadata planning and sparse top-k smoke                                                      | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/flashattention_backend.py`                     | attention      | Codex | port to new API | Preserve DCU fused cache-write guards and move pool access to official ForwardContext-owned backend state                  | high   | targeted compile passed; old ForwardBatch pool reference scan passed | Qwen dense, MLA, and fused Qwen cache-store smoke                                                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`                     | attention      | Codex | theirs          | Official converted the old NSA file into a compatibility shim; DCU implementation was ported to`dsa/index_buf_accessor.py` | high   | targeted compile passed; DSA alias test passed                       | DCU DSA index-cache gather/store smoke                                                             | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`                            | attention      | Codex | port to new API | Keep official compatibility shim and three-way port DCU BF16/FP8 indexer behavior into`dsa/dsa_indexer.py`                 | high   | targeted compile passed; DSA alias test passed                       | DSA decode, ragged prefill, chunking, BF16 index-cache, and FP8 index-cache smoke                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`                        | attention      | Codex | port to new API | Keep official compatibility shim and port DCU TileLang/gfx behavior into`dsa/tilelang_kernel.py`                           | high   | targeted compile passed                                              | TileLang DSA kernel compile and numeric smoke                                                      | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa/triton_kernel.py`                          | attention      | Codex | port to new API | Keep official compatibility shim and port DCU Triton helper kernels into`dsa/triton_kernel.py`                             | high   | targeted compile passed                                              | DCU DSA Triton quant/gate helper smoke                                                             | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/attention/nsa_backend.py`                                | attention      | Codex | port to new API | Keep official compatibility shim and port DCU backend deltas into canonical`dsa_backend.py`                                | high   | targeted compile passed; DSA registry test passed                    | DSA prefill/decode backend selection and MTP smoke                                                 | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/moe/ep_moe/layer.py`                                     | moe            | Codex | ours            | Preserve current DCU DeepEP/AITER/Marlin/group-GEMM implementation; official C03 mainly removes the legacy NPU path        | high   | targeted compile passed; DCU registration passed                     | Dedicated owner must port relevant official runner changes; run DeepEP normal/LL and quantized MoE | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`                           | moe            | Codex | manual merge    | Preserve DCU extended forward arguments and add official Ascend FuseEP dispatch                                            | high   | targeted compile passed                                              | DCU fused MoE, shared-output, and AITER/group-GEMM smoke                                           | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/managers/overlap_utils.py`                                      | scheduler      | Codex | manual merge    | Keep DCU empty-slice helper and add official speculative`seq_lens_cpu` resolution                                          | medium | targeted compile passed                                              | Overlap scheduler plus speculative decode smoke                                                    | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/mem_cache/memory_pool.py`                                       | mem_cache      | Codex | port to new API | Rename NSA state to DSA, retain DCU BF16 index-cache/lightop store paths, and adopt official non-DCU HIP preshuffle rules  | high   | targeted compile passed; old NSA symbol scan passed                  | DSA cache write/read, retract/offload, BF16/FP8 index cache, and SWA cache smoke                   | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/mem_cache/memory_pool_host.py`                                  | mem_cache      | Codex | manual merge    | Use canonical DSA pool type while preserving DCU BF16 index-cache host sizing                                              | high   | targeted compile passed; GPU unit collection blocked by no HIP GPU   | Run`test_dsa_pool_host_unit.py` on a DCU runner                                                    | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/deepseek_v2.py`                                          | model          | Codex | port to new API | Adopt DSA naming and ForwardContext access while retaining DCU fused MLA/cache paths                                       | high   | targeted compile passed; old ForwardBatch context scan passed        | DeepSeek V2/V3.2 startup, DSA, CP, and short-request smoke                                         | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/deepseek_v4.py`                                          | model          | Codex | port to new API | Adopt official split JIT, DSA naming, ForwardContext, and fused cache-write structure while retaining DCU Q/RoPE helpers   | high   | targeted compile passed; old ForwardBatch context scan passed        | DeepSeek V4 startup, CP+EP/DP+EP, MTP, compressor, and FP8 WO-A smoke                              | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/qwen3_5.py`                                              | model          | Codex | manual merge    | Preserve DCU fused RMSNorm/RoPE/KV-store path and add official native/NPU prepare split using ForwardContext pool access   | high   | targeted compile passed                                              | Qwen3.5 dense/MoE short request with fused path on and off                                         | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/models/utils.py`                                                | model          | Codex | port to new API | Move fused KV-buffer eligibility to ForwardContext pool while preserving DCU exclusion from generic HIP fallback           | medium | targeted compile passed                                              | Dense fused cache-store path and context-parallel exclusion smoke                                  | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `python/sglang/srt/server_args.py`                                                 | server_args    | Codex | manual merge    | Preserve DCU page-size 64 behavior, adopt DSA naming and deprecated aliases, and restore official alias action import      | high   | 24 DSA CLI/registry/env tests passed                                 | DCU DSA backend auto-selection and page-size startup smoke                                         | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `test/registered/core/test_srt_engine.py`                                          | test           | Codex | manual merge    | Preserve DCU stage-b registration while taking official consolidated core test structure                                   | low    | DCU registration passed                                              | Execute the registered test on the normal DCU stage-b runner when enabled                          | validated |
| C03 /`7cf193fe1faf` | `sync/official-main-C03-20260521` | `test/registered/language/test_srt_backend.py`                                     | test           | Codex | theirs          | Official replaced the legacy backend suite with consolidated basic sanity kits                                             | low    | deleted-file reference scan passed; DCU registration passed          | Decide whether`test_basic_sanity.py` should receive a DCU stage-a registration                     | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`                             | jit-kernel     | Codex | manual merge    | Adopt official`kDLGPU` device checks while retaining the C03 planner extensions                                            | high   | static/registration passed; DCU runtime pending                      | DSV4 JIT planner compile and runtime smoke                                                         | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/dsv4/elementwise.py`                                     | jit-kernel     | Codex | manual merge    | Keep DCU uint8 JIT storage, use official sgl-kernel on non-DCU HIP, and retain official CUDA JIT                           | high   | static/registration passed; DCU runtime pending                      | DSV4 FP8 elementwise numeric smoke on DCU                                                          | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh`            | jit-kernel     | Codex | ours            | Preserve the validated DCU HIP FP8 pack; official AMD sgl-kernel keeps its separate software conversion path               | high   | static/registration passed; DCU runtime pending                      | DCU FP8 pack compile and numeric smoke                                                             | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh`                          | jit-kernel     | Codex | theirs          | Adopt official`kDLGPU` aliases and HIP runtime fallback required by the new JIT interface                                  | medium | static/registration passed; DCU runtime pending                      | Compile a DSV4 JIT module in the target DCU container                                              | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh`                            | jit-kernel     | Codex | manual merge    | Combine official device/memcpy additions with DCU launch, shuffle, sync, and shared-memory helpers                         | high   | static/registration passed; DCU runtime pending                      | Compile and launch a representative DCU JIT kernel                                                 | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/jit_kernel/include/sgl_kernel/warp.cuh`                             | jit-kernel     | Codex | manual merge    | Adopt official wave64 mask behavior while retaining DCU shuffle and synchronization helpers                                | high   | static/registration passed; DCU runtime pending                      | Wave64 shuffle/sync kernel smoke                                                                   | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/disaggregation/mooncake/conn.py`                                | disaggregation | Codex | manual merge    | Use official common`TransferKVChunk` while preserving the DCU FA KV-layout environment and transfer layout                 | medium | static/registration passed; DCU runtime pending                      | Mooncake transfer smoke with`SGLANG_KV_LAYOUT_DCU_FA`                                              | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`                            | attention      | Codex | port to new API | Port official piecewise CUDA-graph structure around the existing DCU BF16/FP8 cache, LightOp, and page-size-64 paths       | high   | static/registration passed; DCU runtime pending                      | DSA BF16/FP8 cache, sparse prefill, ragged decode, and graph smoke                                 | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/attention/flashattention_backend.py`                     | attention      | Codex | port to new API | Add official MLA context parallelism while retaining DCU fused cache-write ownership guards                                | high   | static/registration passed; DCU runtime pending                      | Dense, MLA CP, DSV4, and fused cache-write smoke                                                   | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/attention/triton_backend.py`                             | attention      | Codex | manual merge    | Adopt official pool API and cache invalidation while retaining CPU last-index handling for graph capture                   | high   | static/registration passed; DCU runtime pending                      | Decode/extend and CUDA graph cache-invalidation smoke                                              | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/layers/deepseek_v4_rope.py`                                     | dependency     | Codex | theirs          | Adopt the official ImportError-guarded TileLang initialization                                                             | medium | static/registration passed; DCU runtime pending                      | Import with and without TileLang, then DSV4 RoPE smoke                                             | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/managers/overlap_utils.py`                                      | scheduler      | Codex | manual merge    | Adopt official FutureIndices/spec-extras APIs while keeping the native token resolver on DCU                               | medium | static/registration passed; DCU runtime pending                      | Overlap scheduler plus speculative decode smoke                                                    | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/managers/schedule_batch.py`                                     | scheduler      | Codex | theirs          | Adopt official tensor flatten and speculative batch interface updates                                                      | medium | static/registration passed; DCU runtime pending                      | Batch flatten, overlap scheduling, and speculative decode smoke                                    | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/models/deepseek_v2.py`                                          | model          | Codex | manual merge    | Keep DCU fused RMS/quant returns while adding official DSA/MLA CP parameters and MoE output-buffer context                 | high   | static/registration passed; DCU runtime pending                      | DeepSeek V2/V3 fused RMS/quant, CP, MoE, and short-request smoke                                   | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `python/sglang/srt/models/deepseek_v4.py`                                          | deepseek-v4    | Codex | manual merge    | Keep DCU fused cos/sin and LightOp/JIT paths; restrict official fused QK/sgl-kernel behavior to non-DCU HIP                | high   | static/registration passed; DCU runtime pending                      | DSV4 TP, CP+EP, DP+EP, MTP, graph capture, and FP8 WO-A smoke                                      | validated |
| C04 /`af8f66940e9b` | `sync/official-main-C04-20260523` | `sgl-kernel/csrc/common_extension_rocm.cc`                                         | sgl-kernel     | Codex | manual merge    | Register both DCU decode metadata operators and official DSV4 top-k/norm/RoPE operators                                    | high   | static/registration passed; DCU runtime pending                      | `gfx938` metadata check plus DCU sgl-kernel smoke whitelist                                        | validated |
| C07 /`a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/layers/attention/flashmla_backend.py`                           | attention      | TBD   | port to new API | Official attention interfaces changed while DCU FlashMLA paths must remain available                                       | high   | Qwen dense plus DSV4 smoke                                           | Assign attention owner                                                                             | open      |
| C10 /`47377525cb32` | `sync/official-main-C10-20260604` | `.github/workflows/pr-test-dcu.yml`                                                | ci             | TBD   | manual merge    | Keep official workflow structure and DCU runner/wheel overlays                                                             | medium | CI dry-run and DCU registration check                                | Fill exact runner/image validation command                                                         | open      |
| C13 /`125ef888921b` | `sync/official-main-C13-20260610` | `sgl-kernel/**`                                                                    | sgl-kernel     | TBD   | manual merge    | sgl-kernel interfaces and DCU/HIP glue both changed                                                                        | high   | sgl-kernel DCU smoke whitelist                                       | Assign kernel owner                                                                                | open      |

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
  - Post-checkpoint fixes and validation (2026-06-30):
    - Added the missing DSV4 top-k and norm/RoPE sources to the DCU
      `setup_hip.py` manifest; a clean AOT build and operator registration
      check passed.
    - Removed duplicate DCU HIP dtype traits in the DSV4 JIT headers; a
      fresh-cache compressor-plan JIT build passed.
    - Restored the DCU `ScheduleBatch.loc_tensor` initialization required by
      empty-prefix paged allocation.
    - DeepSeek-V4 pure-TP service startup and real inference passed.

### C05-C10

- Expected focus: PD, scheduler, attention, mem_cache, embedding, workflows.
- Owner: TBD
- Required validation:
  - Stage-b small model smoke.
  - Qwen2.5 dense server smoke.
  - Qwen2.5-VL smoke.
- Recommended manual validation:
  - `scheduler`: stage-b small model smoke, split prefill, abort/retract if touched.
  - `attention`: dense + VLM attention backend smoke.
  - `mem_cache`: cache hit/retract/SWA smoke if cache files conflict.
  - `embedding` / `reranker`: embedding and reranker API smoke when touched.
  - `ci`: DCU suite partition generation and runner/image dry-run.
- Manual validation result:
  - TBD
- Notes:
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
