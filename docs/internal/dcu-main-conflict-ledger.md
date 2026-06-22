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
| C02 / `425dffbde339` | `sync/official-main-C02-20260519` | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | deepep | TBD | manual merge | DeepEP API and DCU runtime behavior both need review | high | DCU MoE smoke | Confirm topk and low-latency dispatch compatibility | open |
| C07 / `a5e6a8887a94` | `sync/official-main-C07-20260529` | `python/sglang/srt/layers/attention/flashmla_backend.py` | attention | TBD | port to new API | Official attention interfaces changed while DCU FlashMLA paths must remain available | high | Qwen dense plus DSV4 smoke | Assign attention owner | open |
| C10 / `47377525cb32` | `sync/official-main-C10-20260604` | `.github/workflows/pr-test-dcu.yml` | ci | TBD | manual merge | Keep official workflow structure and DCU runner/wheel overlays | medium | CI dry-run and DCU registration check | Fill exact runner/image validation command | open |
| C13 / `125ef888921b` | `sync/official-main-C13-20260610` | `sgl-kernel/**` | sgl-kernel | TBD | manual merge | sgl-kernel interfaces and DCU/HIP glue both changed | high | sgl-kernel DCU smoke whitelist | Assign kernel owner | open |

## Per-Checkpoint Notes

### C01 / `c67b2870569a`

- Expected focus: test registry and CI overlap.
- Owner: TBD
- Required validation:
  - `python3 scripts/ci/dcu/verify_dcu_registration.py`
  - DCU CI dry-run command, to be filled.
- Notes:
  - TBD

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

