# v0.5.12_dev Forward-Port Step 4 Conflict Review

## Scope

- Branch: `forward-port/v0.5.12-dev-20260715`
- Parent: Step 3 merge `d648b38c7f3dd314ca1a5e098144e554949b3a84`
- Old range: `80571de9491c8fd80e6822c9fa4efeb02ff67cce..cf5983854be1f19237ba28416b438f7b8965cfe6`
- Range size: 26 full-graph commits, 18 non-merge commits, 30 old-range files
- Resolved staged tree before documentation: 32 files, 3,308 insertions, 295 deletions
- Git-reported textual conflicts: exactly 9 files

The current `main` layout and runtime-context APIs remain canonical. This step
ports packed paged-KV attention, DCU HiCache/Mooncake transfer, MiniMax M2
sequence-parallel and INT8 Marlin behavior, MiMo TP1 handling, and associated
collective/quantization support without restoring deleted monolithic modules.

## Textual conflict decisions

| File | Resolution |
|---|---|
| `srt/disaggregation/decode_kvcache_offload_manager.py` | Retained the current offload-state lifecycle and passed the configured host-cache layout into the canonical host-pool factory. |
| `srt/layers/attention/flashattention_backend.py` | Combined the current metadata/cascade paths with packed paged-KV handling and MiniMax SP query padding. Query token accounting stays local to the DCU layout branch and only real metadata tokens enter attention. |
| `srt/layers/moe/ep_moe/layer.py` | Preserved current DeepEP tuple and dispatch contracts while porting explicit INT8 activation/scale outputs and the old LightOp behavior. |
| `srt/mem_cache/hiradix_cache.py` | Kept the current hierarchical-cache lifecycle and forwarded `hicache_mem_layout` to host-pool selection. |
| `srt/mem_cache/memory_pool_host.py` | Kept the current compatibility facade; old duplicate MHA host-pool classes were not revived. The DCU implementation moved to `mem_cache/pool_host/mha.py`. |
| `srt/mem_cache/storage/mooncake_store/mooncake_store.py` | Retained current hybrid logical-anchor guards, accepted `layout_dcu`, and generalized buffer registration to recursively register tuple/list K/V buffers. |
| `srt/model_executor/cuda_graph_runner.py` | Kept the obsolete file deleted. Its MiniMax gathered-buffer multiple was ported to canonical `model_executor/runner/base_cuda_graph_runner.py`. |
| `srt/models/minimax_m2.py` | Ported MiniMax optimization, SP, fused RMS quantization, and INT8 Marlin behavior onto current `get_server_args()` and `get_parallel()` APIs. |
| `srt/server_args.py` | Added typed current-dataclass options for MiniMax, packed paged KV, and `layout_dcu`; retained the current resolver and relaxed the MiMo fused-QKV TP check only for effective attention TP1. |

## Refactor and `_is_dcu` audit

- The old `MHATokenToKVPoolHostDCU` implementation was moved from the removed
  monolithic `memory_pool_host.py` structure into current
  `mem_cache/pool_host/mha.py`. It keeps independent K/V page-major buffers,
  DCU kvcacheio transfer operations, page data/meta access, and direct/kernel
  IO selection. Layer-sharded `layout_dcu` is rejected explicitly because the
  old implementation does not support that topology.
- Decode offload, HiRadix, hybrid pool assembly, and draft KV-cache construction
  now pass the layout to the canonical factory, so the DCU pool is selected
  after the refactor rather than only existing as unreachable copied code.
- Mooncake buffer registration is structure-aware instead of globally assuming
  the old DCU tuple layout. The current hybrid logical-anchor safety check is
  retained.
- The removed `cuda_graph_runner.py` remains absent; its one required MiniMax
  behavior now lives in `base_cuda_graph_runner.py`.
- Old `get_global_server_args`, `get_attention_tp_group`, and
  `get_attention_tp_size` references in conflict-resolved model code were
  adapted to current runtime-context accessors.
- The new packed paged-KV helper's `SGLANG_KV_LAYOUT_DCU_FA` assumption is
  gated by `is_dcu()`. It cannot activate the DCU cache layout on generic HIP
  or CUDA merely because the environment default is true.
- Existing generic `_is_hip` paths in DeepEP and quantization were reviewed.
  Dedicated DCU LightOp/layout behavior remains explicit; no `is_hcu()` runtime
  predicate or removed API was introduced.
- `SGLANG_USE_AITER_AG=0` remains exported by the required pure-TP script.

## Static validation evidence

- `git ls-files -u`: zero entries.
- Precise conflict-marker scan: no markers.
- `git diff --check`: passed after removing seven upstream trailing-space
  artifacts in `sgl-kernel/csrc/kvcacheio/transfer.cu`.
- All 28 changed Python files compiled.
- Broad changed-file Ruff `E9,F821`: passed.
- Targeted high-risk Ruff `E9,F401,F811,F821,F841`: passed.
- Import smoke passed for ServerArgs, FlashAttention, MHA host pool, Mooncake,
  MiniMax M2, MiMo V2, and the new MiniMax INT8 Marlin module.
- DCU registration passed with 277 registered files and the existing CPU-utils
  warning.
- DSA alias/CLI/registry passed 19 tests.
- gfx938 `setup_hip.py --name` passed with package `sglang-kernel`, zero
  unsupported CUDA calls, and 56 replaced kernel launches.

## Pure-TP validation evidence

- Immediate preflight found `zz-nmz26` occupied at VRAM 93% on all eight
  devices, despite HCU 0%, so it was not used. All eight `zz-nmz22` devices
  were VRAM/HCU 0% and were selected.
- Exact command: `bash /home/scripts/sglang/run_dpsk-v4.sh 10015
  /home/model/DeepSeek-V4-Flash-FP8-Channel`.
- All 46 shards loaded and decode graphs captured for `bs=128..1`.
- `/health` returned HTTP 200 and one short `/generate` returned HTTP 200
  without worker failure.
- The response remained empty with eight zero output IDs. This is the known
  deferred NaN/accuracy issue, is not an accuracy pass, and remains
  non-blocking under the agreed startup/request gate.
- No runtime code fix or retry was needed. The service was stopped, port 10015
  closed, and all eight `zz-nmz22` devices returned to VRAM 0%.

## Integration decision

All scoped static and pure-TP gates passed without a runtime retry. Commit the
exact old endpoint as the Step 4 no-ff checkpoint, then continue to Step 5.
