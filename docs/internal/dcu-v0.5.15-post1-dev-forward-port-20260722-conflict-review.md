# v0.5.15.post1_dev Daily Forward-Port Conflict Review (2026-07-22)

## Scope

- Target branch: `forward-port/v0.5.12-dev-daily-20260722`
- Target parent: `bd8c84eb4f05a88c94e37daf7f21fe5a08376fa9`
- Old branch: `/home/proj_dpsk-v4/sglang-das` `v0.5.12_dev`
- Previous immutable old endpoint:
  `5ec8531b096fa3297ab034dedc873aad215f2c35`
- New exact old endpoint: `023409568bfb83982fbb173ac742baf12dc7dcc3`
- Range size: 29 full-graph commits, 20 non-merge commits
- Resolved code/workflow result before documentation: 23 files, 820
  insertions, 966 deletions
- Git-reported textual conflicts: exactly 11 files

The merge ports the complete endpoint rather than replaying selected patches.
Current `v0.5.15.post1_dev` file locations and APIs remain canonical, while
newer old-branch DCU behavior is preserved or translated to its current
equivalent.

## Commit groups

- HY3 pipeline parallelism and latest static/dynamic EPLB routing fix.
- DCU causal-conv HCU API adaptation.
- AITER custom all-reduce backend and Fabric transport selection.
- DeepSeek-V4 PD prefill full-token-pool admission.
- Qwen2.5-VL FA3 `cu_seqlens` correction.
- DeepSeek fused RMS-quant residual update.
- PD Decode DP-attention anti-hang StepInfo protocol.
- LTX-2 dynamic batching, TP loading, and non-contiguous output support.
- Pinned reusable quality-gate workflow.

## Textual conflict decisions

| File | Resolution |
|---|---|
| `srt/disaggregation/common/conn.py` | Kept current shared ZMQ context, socket cache, disconnect monitor, keepalive, and zero-linger lifecycle. Added the old branch's 5-second send timeout and HWM limit to both manager and receiver PUSH sockets. |
| `srt/disaggregation/decode.py` | Kept current HiCache/offload structure and `DecodeHiCachePreallocMixin`; ported local-progress timing and paused-rank MLPSync participation to `dp_attn_adapter` rather than calling the removed scheduler Mixin method. |
| `srt/disaggregation/prefill.py` | Preserved current PP-aware transfer setup and ported DSV4 full-token-pool admission with a hybrid-SWA fallback cap. |
| `srt/distributed/device_communicators/custom_all_reduce.py` | Added `auto/native/aiter/off` dispatch and AITER Fabric/IPC graph-registration semantics. Retained current CUDA V2 capability checking and corrected the old undefined `_is_hcu` predicate to `_is_dcu`. |
| `srt/distributed/parallel_state.py` | Propagated the selected backend through current `GroupCoordinator`; kept strict explicit-AITER Fabric failure, QuickAllReduce suppression for Fabric/auto transport, and canonical per-rank startup evidence. |
| `srt/environ.py` | Added the live DSV4 full-token-pool switch. Did not retain the old `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS` symbol because current `disaggregation_decode_extra_slots` already owns that behavior and the old symbol would be unused. |
| `srt/layers/moe/topk.py` | Accepted the newest old-branch LightOp EPLB postprocess, which supersedes the earlier no-remap comment and is required by the static/dynamic EPLB accuracy fix. |
| `srt/managers/scheduler_components/dp_attn.py` | Moved the old scheduler-Mixin StepInfo logic into the current Adapter. The current breakable-CUDA-graph column is retained, so the layout is explicitly 7 model fields plus 10 scheduler fields rather than the old 6+10 layout. |
| `srt/models/deepseek_v2.py` | Preserved current fused-path structure and changed the existing-residual path to `update_hd=True`, matching unfused residual advancement. |
| `srt/models/hunyuan_v3.py` | Ported PP ownership/proxy/loading behavior to current `make_layers`, `get_stream("alt")`, and `PPMissingLayer` APIs. EPLB recording contains both routing and expert execution; lazy expert-weight enumeration is restricted to the local PP layer range. |
| `srt/server_args.py` | Added only the new custom-all-reduce backend as an Annotated field and retained environment promotion/disable precedence. The obsolete old hand-written CLI block was not restored. |

## Refactor and `_is_dcu` audit

- The old `scheduler_dp_attn_mixin.py` implementation moved to current
  `scheduler_components/dp_attn.py`; the dedicated Gloo group, epoch check,
  paused-rank participation, and queue diagnostics remain reachable.
- StepInfo uses indexes `0..6` for current model metadata and `7..16` for the
  ten scheduler fields. A direct assertion verified the exact layout.
- HY3 PP uses the current stream registry and current `make_layers` PP
  placeholders. Missing imports introduced by the structural merge were
  restored, and non-local placeholders are not inspected as MoE layers.
- DCU causal convolution imports the installed `causal_conv1d_fn_hcu` symbol;
  the update function remains sourced from the current interface module.
- AITER logging and dispatch use `_is_dcu`; user-visible `HCU` remains only in
  the installed API/model naming. Generic CUDA keeps its V2 availability gate.
- The latest LightOp EPLB remap is retained as the endpoint behavior rather
  than preserving the earlier intermediate MiMo comment.
- `SGLANG_USE_AITER_AG=0` remains exported by the pure-TP launch script.

## Static validation evidence

- `git ls-files -u`: zero entries.
- Precise conflict-marker scan: no markers.
- `git diff --cached --check`: passed.
- All 22 changed Python files compiled.
- Direct imports passed for HY3 and the high-risk disaggregation,
  communicator, DeepSeek, Qwen2.5-VL, TopK, and GDN modules.
- Installed causal-conv exports both `causal_conv1d_fn_hcu` and
  `causal_conv1d_update`.
- ServerArgs accepted `--custom-all-reduce-backend native` through the current
  Annotated CLI builder.
- The DP StepInfo 7+10 layout assertion passed.
- DCU registration passed with 277 registered files and the existing
  CPU-utils warning.
- DSA alias/CLI/registry passed all 19 tests.
- Ruff is not installed in `rye_sglang_0716`; it was recorded as unavailable,
  not reported as passed. Compilation, direct imports, and focused assertions
  were used as the available static substitutes.
- No `sgl-kernel` or build metadata changed, so gfx938 setup-name generation
  was not required.

## Pure-TP validation evidence

- Preflight rejected `zz-nmz22 / rye_sglang_0716` (VRAM 92%),
  `zz-nmz26 / rye_sglang_0716` (VRAM 96%, HCU active), and
  `zz-nmz20 / rye_sglang_0716` (VRAM 91-92%).
- `zz-sglang2 / rye_sglang_0720` was selected with all eight devices at
  VRAM 0% / HCU 0%.
- Exact command:

  ```bash
  cd /home/scripts/sglang
  bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-FP8-Channel
  ```

- All 46 shards loaded and decode graphs captured for `bs=128..1`.
- The new custom-all-reduce path resolved `auto` to AITER IPC on all eight
  ranks with `disabled=False`.
- `/health` returned HTTP 200.
- One short `/generate` returned HTTP 200 with eight non-empty output tokens:
  `World = function() {\n    return "`.
- No runtime fix or retry was needed. The service stopped, port 10015 closed,
  and all eight selected devices returned to VRAM 0% / HCU 0%.

## Integration decision

The exact old endpoint passed the agreed static and pure-TP gates. Commit it
as one no-ff daily forward-port merge on the review branch; do not push or
merge it into `v0.5.15.post1_dev` without owner direction.
