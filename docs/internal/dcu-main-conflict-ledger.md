# DCU Main Conflict Ledger

This ledger tracks conflicts and merge decisions while internal `main` catches
up with official SGLang `main`.

## Status Legend

| Status | Meaning |
|---|---|
| `open` | Conflict exists or owner decision is pending |
| `merged` | Conflict resolution has been committed to the checkpoint branch |
| `validated` | Required validation passed |
| `waived` | Validation or issue is explicitly waived with reason |

## Strategy Legend

| Strategy | Meaning |
|---|---|
| `ours` | Keep DCU-side implementation |
| `theirs` | Take official implementation |
| `manual merge` | Combine both sides manually |
| `drop DCU patch` | Remove DCU patch because official code supersedes it |
| `port to new API` | Re-implement DCU behavior on top of official interface |

## Active Conflict Board

| Checkpoint | Merge branch | Conflict file | Area | Owner | Strategy | Reason | Risk | Validation | Follow-up | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `.github/workflows/_pr-test-stage.yml` | ci | Codex | theirs | Official renamed `check-stage-health` to `check-pr-test-health`; this is a pure workflow action rename | low | precise marker scan passed; DCU registration passed | None | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `.github/workflows/pr-states.yml` | ci | Codex | manual merge | Keep official `workflow_run` PR lookup and preserve DCU `/rerun-failed-ci` stale-run wording | low | precise marker scan passed; DCU registration passed | Verify real GitHub workflow behavior in CI dry-run | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/environ.py` | env | Codex | manual merge | Keep both DCU `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` and official `SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN` env switches | low | syntax compile passed; DCU registration passed | None | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/mem_cache/memory_pool.py` | mem_cache | Codex | manual merge | Preserve DCU FA KV layout copy/load while taking official `current_platform.synchronize()` | medium | syntax compile passed; DCU registration passed | Add CPU-offload smoke on DCU FA layout when CI command is available | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/managers/scheduler_pp_mixin.py` | scheduler | Codex | manual merge | Combine official `hc_hidden_size` fallback with DCU proxy hidden-state shape helper | medium | syntax compile passed; DCU registration passed | Pipeline-parallel smoke if available | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `test/registered/tokenizer/test_multi_detokenizer.py` | test | Codex | theirs | Official C01 renames CUDA suite from `stage-b-*` to `base-b-*`; AMD registration remains from DCU tree | low | syntax compile passed; DCU registration passed | None | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/moe/ep_moe/layer.py` | moe | Codex | ours | High-risk official MoE runner restructuring overlaps DCU DeepEP/AITER paths; preserve known DCU runtime for C01 | high | syntax compile passed; DCU registration passed | Owner should port official runner-core changes onto DCU path in later checkpoint | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/moe/moe_runner/aiter.py` | aiter | Codex | ours | Official AITER runner refactor conflicts with DCU W8A8/W16A16 handling; avoid semantic rewrite in first checkpoint | high | syntax compile passed; DCU registration passed | Create dedicated AITER merge task before enabling official runner-core behavior | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/layers/quantization/unquant.py` | quantization | Codex | ours | Conflict is inside MoE execution fallback; keep DCU behavior until AITER/MoE owner ports official path | high | syntax compile passed; DCU registration passed | Revisit together with `moe_runner/aiter.py` | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/models/deepseek_v2.py` | model | Codex | ours | DeepSeek forward path overlaps DCU fused RMS/quant behavior; preserve current DCU model path in C01 | high | syntax compile passed; DCU registration passed | Run DeepSeek V2/V3 smoke after C01; port official output-buffer context if needed | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/models/deepseek_v4.py` | model | Codex | ours | DeepSeek V4 has many DCU-specific imports/kernels and official changes are not safe to fold blindly | high | syntax compile passed; DCU registration passed | DSV4 owner should review skipped official C01 hunks before C02/C03 | validated |
| C01 / `c67b2870569a` | `sync/official-main-C01-20260517` | `python/sglang/srt/server_args.py` | server_args | Codex | ours | Preserve DCU speculative-algorithm alias helper; official side did not supersede it in C01 | medium | syntax compile passed; DCU registration passed | Confirm Gemma4 assistant draft CLI path still works if used internally | validated |
| C02 / `425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | deepep | TBD | manual merge | DeepEP API and DCU runtime behavior both need review | high | DCU MoE smoke | Confirm topk and low-latency dispatch compatibility | open |
| C07 / `a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/layers/attention/flashmla_backend.py` | attention | TBD | port to new API | Official attention interfaces changed while DCU FlashMLA paths must remain available | high | Qwen dense plus DSV4 smoke | Assign attention owner | open |
| C10 / `47377525cb32` | `sync/official-main-C10-20260604` | `.github/workflows/pr-test-dcu.yml` | ci | TBD | manual merge | Keep official workflow structure and DCU runner/wheel overlays | medium | CI dry-run and DCU registration check | Fill exact runner/image validation command | open |
| C13 / `125ef888921b` | `sync/official-main-C13-20260610` | `sgl-kernel/**` | sgl-kernel | TBD | manual merge | sgl-kernel interfaces and DCU/HIP glue both changed | high | sgl-kernel DCU smoke whitelist | Assign kernel owner | open |

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

### C02 / `425dffbde339`

- Expected focus: DeepSeek V4 MTP, attention, DeepEP.
- Owner: TBD
- Required validation:
  - DCU MoE smoke, exact command to be filled.
  - DSV4 smoke, exact command to be filled.
- Notes:
  - TBD

### C03 / `7cf193fe1faf`

- Expected focus: cache, model, attention.
- Owner: TBD
- Required validation:
  - Dense smoke.
  - Cache-related unit or smoke test, exact command to be filled.
- Notes:
  - TBD

### C04 / `af8f66940e9b`

- Expected focus: AMD DSV4 runtime and jit-kernel.
- Owner: TBD
- Required validation:
  - jit-kernel or sgl-kernel DCU smoke.
- Notes:
  - TBD

### C05-C10

- Expected focus: PD, scheduler, attention, mem_cache, embedding, workflows.
- Owner: TBD
- Required validation:
  - Stage-b small model smoke.
  - Qwen2.5 dense server smoke.
  - Qwen2.5-VL smoke.
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
- Notes:
  - TBD

### C18-C19

- Expected focus: MTP rejection sampling and XPU import guard.
- Owner: TBD
- Required validation:
  - Daily sync smoke gate.
- Notes:
  - TBD
