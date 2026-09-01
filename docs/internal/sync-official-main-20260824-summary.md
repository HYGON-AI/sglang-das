# Official sync 20260824 — component & model changes

**Range** `92b1d382c7` (20260817 backport endpoint) → `c8e1ddc707` (official main, 2026-08-24)
**Size** 397 commits · 1777 files · +121 101 / −34 135 · 44 conflicted files (63 hunks)
**Merge** `6d77e62f8` on `sync/official-main-daily-20260824`, parents `0d49fbb937` + `c8e1ddc707`

`git merge-base` is exactly `92b1d382c7`, so the topology is clean — no `-s ours` anchoring.

## Release tag marker

`v0.5.18` is a **release branch, not an ancestor of main**:

| | |
|---|---|
| tag object | `ff4c6e641d9f9bb174d34ff651c01c114aea8e40` |
| **tag's last commit** | **`71de97b264b04dcd514cf904003028aefe9775c8`** |
| subject | `[Cherry-pick to release/v0.5.18] [Fix] Support 128-aligned hidden sizes in the W4AFP8 DeepEP low-latency requant kernel (#35593) (#35754)` (2026-08-20) |

All 8 commits reachable only from the tag are cherry-picks whose originals are inside this
range (#35593 `b6dcd393d6`, #35714 `61fa64ae7e`, #35298 `c863760ae1`, #34679 `c7e2c08d14`,
#35221 `0e4a09480c`, #34881 `307a90f6d3`, #35392 `b814a7e812`, #35077 `5f12839591`) — so this
merge contains the whole of v0.5.18. Same shape as v0.5.17 last time.

## Where the 397 commits went

| Area | Count |
|---|---|
| Diffusion / video | 110 |
| CI / build / docs / test | 32 |
| DSpark & speculative decoding | 28 |
| Attention & kernels | 24 |
| DeepSeek / DSV4 / DSA | 22 |
| KV cache & memory | 20 |
| Quantization | 19 |
| PD disaggregation | 18 |
| Model support | 17 |
| MoE / EP / EPLB | 16 |
| Router / gateway / server | 10 |
| Other (config plumbing, XPU/CPU passes) | 51 |

## Model support

| Model | What landed |
|---|---|
| **DeepSeek-V4** | NPU DSpark support + DSV4 cache-management refactor (#33676); AMD prefill-CP two-batch-overlap (#33480); W4A4 MegaMoE server flag (#35918); FP4 checkpoints default to FlashInfer MXFP4 MoE (#35919); shared-experts fusion top-6 (#32340); indexer top-k uses the full 1024-thread block on ROCm (#36004); qk-norm-rope fused on the MTP target-verify path (#34973) |
| **Kimi-K3** | ModelOpt mixed NVFP4/FP8 checkpoints (#35077); Radix-4 MoE top-k router kernel (#34490); FlashInfer MXFP4 auto-MoE on SM107 (#35554); tool calls no longer lost to reasoning/constraint/truncation (#34881); DSpark draft-attn kernel perf (#35499) |
| **Qwen3.5 / Qwen3.8** | MTP keeps online NVFP4 draft quantization on mixed checkpoints (#35545); shared expert overlapped with DeepEP routed experts (#34938); MXFP4 on MI355X with fp8_e4m3 KV cache |
| **Granite** | SWA support via the existing Granite models (#35794) |
| **dots.note.omni** | Native encoders, video preprocessing, MTP decoding (#33829) |
| **Gemma4** | MTP bridge projections now quantized (#32440) |
| **Diffusion** | The single largest area (110 commits): LingBot VAE decode speedup, Hunyuan QKV pack indexing at production video shapes, LoRA-merged weight caching, layerwise-offload fixes |

## Key technology

**Speculative decoding**
- Multi-adapter LoRA with EAGLE / NEXTN / DFLASH / DSPARK (#34337)
- DFlash2: local convolution + candidate selector (#35371); quantized target `lm_head` in the selector (#35496)
- Custom draft worker classes in DSpark (#35397)
- HiCache supports DCP with DSpark (#35221)
- EAGLE sampling borrows CUDA-graph pool storage (#35375)

**MoE / EP**
- Cutlass MoE activation + scales gathered in one launch (#34915)
- W4AFP8 DeepEP low-latency requant kernel handles 128-aligned hidden sizes (#35593)
- DeepEPv2 (ElasticBuffer) A2A backend landed (#29525) then **reverted** (#35568); DeepEP branch refreshed (#34923) with SBO support (#35450)
- Config-driven MoE router scoring ("Laguna", #35362)

**KV cache / memory**
- Unified radix tree is now the default for all cases (#35081); decode-side radix cache for SWA hybrid models under P/D (#27770)
- `index_fill_` for the full→SWA mapping clear, removing a blocking H2D copy (#35773) — **partially adopted**, see below
- Tombstoned SWA locs clamped in `UnifiedSWAKVPool` translation (#35933)
- HiCache: buffer-mode load-back ownership races fixed + prefetch anchor lock (#35769); host budget split across co-located ranks (#35540); retraction host pool may be smaller than the device pool (#35543)
- mxfp8 KV cache: CPU offload (#35888) and PD transfer (#35718)

**PD disaggregation** — a large refactor wave hoisting staging helpers, mooncake failure handling and `_handle_staging_req` into mixins; deferred-release resolution now runs every iteration before the retraction/polling gates.

**Quantization** — `resolve_checkpoint_quant_spec()` centralizes checkpoint quant metadata resolution; bounded post-load device staging shared across methods (#35180); SM107 MXFP8 activation prep fix (#35405).

## Performance highlights

`indexer top-k full 1024-thread block on ROCm (#36004)` · `cutlass MoE single-launch activation+scale gather (#34915)` · `Qwen shared expert overlapped with DeepEP routed experts (#34938)` · `bpreshuffle fp8-scale copies eliminated at MI355X producer sites (#33166)` · `qk-norm-rope fusion on the MTP verify path (#34973)` · `non-blocking full→SWA mapping clear (#35773)` · `Radix-4 MoE top-k router for K3 routing (#34490)`

## Conflict resolution posture

**Adopted official, dropped local:** `resolve_checkpoint_quant_spec` (local kimi_k26
`text_config.compression_config` case folded into official's selector rather than kept
separately) · `_compute_dsa_indexer_cell_size` (HCU bf16 index-K sizing moved *into* the
helper, so all three call sites get it instead of one) · `SRTFp8Config` inheritance in
multimodal_gen (it already carries the `ignored_layers` normalization we duplicated) ·
official's ROCm-7.2 gate in `transfer.cu` (supersedes our HIP-7.0 workaround) ·
`declare_resolution` in `deepseek_v4_hook` · the q8kv8 sparse-prefill dispatch.

**Kept local behind `_is_hcu` / local guards:** the stricter `free_swa` release path ·
the LightOp sqrtsoftplus gate and the extra `biased_grouped_topk_gpu` kwargs · MegaMoE HCU
runtimes and the `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` else arm · the LightOp K-cache store paths ·
the DSpark HCU device gate · the environ HCU section · the `layout_hcu` mamba mapping.

### ⚠️ One deliberate divergence worth a reviewer's eye

`mem_cache/allocator/swa.py` — upstream #35773 frees `swa_indices` directly and clears only
`mapping_indices`. The local path additionally dedupes, skips pages already on the free list,
and clears **every** full index that maps into a freed SWA page. For `page_size > 1` upstream's
form can double-free a page shared by several full tokens and leaves stale full→SWA entries —
the C10 `bs>1` garbled-output suspect. Local guards kept; official's non-blocking
`clear_full_to_swa_mapping()` adopted for the empty-`swa_indices` case only.

## Semantic audit

Ruff `F821/F401/F811` over all 1403 changed files, diffed against **both** parents with line
numbers normalized, found 5 merge artifacts — all in *auto-merged* hunks, none at a conflict:

1. `dsv4/compressor.py` and `deepseek_v4_backend.py` both lost the
   `quant_to_nope_fp8_rope_bf16_pack_triton` import (upstream deleted the block carrying it)
   while still calling it.
2. `memory_pool_host.py` lost `psutil`, `DSATokenToKVPool`, `MLATokenToKVPoolHost` and
   `HICACHE_HOST_MEMORY_RESERVE_BYTES`, all used by the local `DSAIndexerPoolHost`.
3. `mega_moe.py` lost `transform_weights_for_mega_moe` from a function-local import.
4. `compressed_tensors.py` gained a duplicate `BaseKVCacheMethod` import and a **second**
   `CompressedTensorsKVCacheMethod` class — the upstream copy sits later in the file and would
   have won, but it defines `validate_kv_cache_scheme` while local call sites use
   `is_supported_scheme`. Upstream duplicate removed.

After the fixes: **zero findings absent from both parents.**

## Static gates

| Gate | Result |
|---|---|
| `git ls-files -u` / conflict-marker scan / `git diff --cached --check` | clean |
| Changed Python files compile | **1403 / 1403** |
| ruff `F821,F401,F811` vs both parents | **zero new** |
| Key module imports | **30 / 30** |
| `verify_hcu_registration.py` | OK |
| `check_hcu_runtime_text.py` | OK |
| `check_hcu_external_api_compat.py` | OK |
