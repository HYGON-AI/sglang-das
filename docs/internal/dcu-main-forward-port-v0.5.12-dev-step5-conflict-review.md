# v0.5.12_dev Forward-Port Step 5 Conflict Review

## Scope

- Branch: `forward-port/v0.5.12-dev-20260715`
- Parent: Step 4 merge `f76fbea9601d31f7a45cd4b4c063de95c18455d3`
- Old range: `cf5983854be1f19237ba28416b438f7b8965cfe6..5ec8531b096fa3297ab034dedc873aad215f2c35`
- Range size: 22 full-graph commits, 13 non-merge commits, 17 old-range files
- Resolved staged code before documentation: 16 files, 2,156 insertions, 615 deletions
- Git-reported textual conflicts: exactly 9 files

The current `main` structure remains canonical. This final step ports
heterogeneous-TP Mooncake transfer, DeepSeek V3.2 fused RMS/quantized MLP,
MiMo KME/RoPE/MTP/EPLB behavior, DCU LightOp INT8 MoE, and AITER W4A16 MoE_C
support without restoring superseded cache or runtime-context APIs.

## Textual conflict decisions

| File | Resolution |
|---|---|
| `srt/disaggregation/mooncake/conn.py` | Retained the current strict C128/SWA-ring checks, canonical `TransferKVChunk`, MiniMax flat transfer, and recursive buffer registration. Added slice-aware SWA/DSA transfer for heterogeneous attention TP, then skips the generic transfer to avoid duplicate sends. |
| `srt/layers/attention/flashattention_backend.py` | Kept the current metadata lifecycle and Step 4 MiniMax sequence-parallel padded-query token accounting. |
| `srt/layers/moe/ep_moe/layer.py` | Preserved current DeepEP dispatch contracts and generic DeepGEMM INT8 symbol; selected the old LightOp contiguous INT8 GEMM only under `_is_dcu`. |
| `srt/layers/moe/fused_moe_int8_marlin_for_minimax_m2.py` | Retained the current module/API shape and ported the old MiniMax INT8 Marlin implementation and modification notice. |
| `srt/layers/moe/moe_runner/triton_utils/fused_moe.py` | Kept current runner contracts and lazy LightOp operations while absorbing the old AITER W4A16 MoE_C scale/config behavior. Adapted the old global ServerArgs access to `get_server_args()`. |
| `srt/layers/quantization/w8a8_int8.py` | Kept current DCU-only LMSlim activation quantization and custom-op registration while porting the old KME ignored-layer and RadixAttention behavior. Removed stale parallel-state access. |
| `srt/mem_cache/swa_memory_pool.py` | Kept the current pool facade. The old simplified `free_swa` change was not restored because allocation/free ownership moved to `mem_cache/allocator/swa.py`, whose current implementation already includes full-page expansion, deduplication, already-free checks, released-page merging, and stale-mapping cleanup. |
| `srt/models/deepseek_v2.py` | Ported the old DeepSeek V3.2 fused gate/up RMS quantization and shared/routed expert `i_q`/`i_s` flow onto current scoped communication and deferred-finalization APIs. Current down-projection and reduce-scatter contracts remain canonical. |
| `srt/models/mimo_v2.py` | Retained the endpoint's TP1 fused-QKV/MTP-as-SWA loading, KME/RoPE, EPLB, and resume behavior; adapted attention TP and ServerArgs access to current runtime context. The new LightOp fused RoPE/KV-store symbol is capability-detected so older installed LightOp builds fall back to the unfused path instead of failing module import. |

## Refactor and `_is_dcu` audit

- `swa_memory_pool.py` still exists, but SWA free-list ownership moved to
  `mem_cache/allocator/swa.py`. The old endpoint's smaller pool-level
  `free_swa` edit is superseded by the stronger current allocator and was not
  copied back into the facade.
- DeepSeek V3.2 MLP changes were placed inside the current
  `get_forward().scoped(...)` communication lifecycle. No deleted
  `skip_all_reduce` parameters or duplicate decoder communication block were
  restored.
- The routed-expert `i_q`/`i_s` values are produced and consumed only in the
  fused RMS quantization path; the current shared-expert stream and deferred
  finalize paths remain intact outside that condition.
- The DeepEP contiguous INT8 GEMM dispatch is explicitly `_is_dcu` first:
  DCU uses LightOp, while generic HIP/CUDA retain the current DeepGEMM alias.
- Mooncake heterogeneous attention-TP transfer uses a dedicated SWA slice and
  does not fall through to the generic chunk sender. Current C128 logical
  anchors and shape validation remain unchanged.
- MiMo attention TP access uses `get_parallel().attn_tp_size/rank`, and current
  `get_server_args()` replaces removed global helpers. Its optional new
  LightOp symbol is guarded by both `_is_dcu` and import capability.
- No runtime `is_hcu()` predicate or broad HIP-to-DCU substitution was
  introduced. The one new generic-HIP fused-clamp condition remains
  `_is_hip and not _is_dcu` because DCU has its own quantized activation path.
- `SGLANG_USE_AITER_AG=0` remains exported by the pure-TP script.

## Static validation evidence

- `git ls-files -u`: zero entries.
- Precise conflict-marker scan: no markers.
- `git diff --cached --check`: passed.
- All changed Python files compiled.
- Broad changed-file Ruff `E9,F821`: passed.
- Targeted conflict-file Ruff `E9,F401,F811,F821,F841`: passed.
- Import smoke passed for Mooncake, FlashAttention, DeepEP MoE, MiniMax INT8
  Marlin, fused MoE, W8A8, SWA pool and allocator, DeepSeek V2/V3.2, MiMo V2,
  and MiMo NextN after one focused optional-LightOp capability fix.
- DCU registration passed with 277 registered files and the existing CPU-utils
  warning.
- DSA alias/CLI/registry passed 19 tests.
- gfx938 `setup_hip.py --name` passed with package `sglang-kernel`, zero
  unsupported CUDA calls, and 56 replaced kernel launches.

## Pure-TP validation evidence

- Immediate preflight found all eight `zz-nmz26` devices occupied at VRAM 93%
  despite HCU 0%, so that node was not used. All eight `zz-nmz22` devices were
  VRAM/HCU 0% and were selected.
- Workspace import resolved to
  `/home/proj_sglang_open/sglang-das/python/sglang/__init__.py`.
- Exact command: `bash /home/scripts/sglang/run_dpsk-v4.sh 10015
  /home/model/DeepSeek-V4-Flash-FP8-Channel`.
- All 46 shards loaded and decode graphs captured for `bs=128..1`.
- `/health` returned HTTP 200 and one short `/generate` returned HTTP 200
  without worker failure.
- The response remained empty with eight zero output IDs. This is the known
  deferred NaN/accuracy issue, is not an accuracy pass, and remains
  non-blocking under the agreed startup/request gate.
- No runtime retry was needed. The only Step 5 fix was the pre-runtime MiMo
  optional-LightOp import compatibility fix found by static import smoke.
- The service was stopped, port 10015 closed, and all eight `zz-nmz22` devices
  returned to VRAM/HCU 0%.

## Integration decision

All scoped static and pure-TP gates passed after one static compatibility fix.
Commit the exact old endpoint as the Step 5 no-ff checkpoint. This completes
the five planned `v0.5.12_dev` forward-port ranges; do not merge the branch to
`main` without migration-owner review.
