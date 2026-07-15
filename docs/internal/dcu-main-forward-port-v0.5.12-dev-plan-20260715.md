# v0.5.12_dev to main Forward-Port Plan

Analysis date: 2026-07-15 CST

## 1. Immutable scope

| Role | Repository / ref | Commit |
|---|---|---|
| Current trunk base | `/home/proj_sglang_open/sglang-das`, `main` | `65f3bd9426e51df40987516acd075b646b858cf6` |
| Old delivery branch | `/home/proj_dpsk-v4/sglang-das`, `v0.5.12_dev` | `5ec8531b096fa3297ab034dedc873aad215f2c35` |
| Common base | both histories | `d4c6831a107ac03bae80e353d170af15557e4443` |
| Forward-port branch | current repository | `forward-port/v0.5.12-dev-20260715` |

The old range contains 214 commits in the full graph: 160 non-merge commits,
54 merge commits, and 88 first-parent commits. Patch-id comparison against the
current `main` finds 157 patch-unique non-merge commits and three equivalent
documentation commits (`3d055862`, `fccdfe68`, and `1aac3f456`), which must not
be duplicated. The final tree delta touches 503 files with 28,302 insertions and
4,508 deletions; 47 files are added and 456 are modified.

This is a semantic forward-port, not an attempt to restore the old tree. The
latest official structure and APIs on `main` are canonical. DCU behavior from
`v0.5.12_dev` is carried into those structures explicitly, normally with
`_is_dcu` ahead of generic `_is_hip` behavior.

## 2. Change inventory

The range is concentrated in `python/sglang/srt` (145 files) and DCU registered
tests (56 files). Its major themes are:

- DeepSeek-V4: BF16 KV-cache storage, DSV4/NSA indexer and compressor paths,
  JIT TopK VMFault fixes, C16 KV writes, SWA allocation, cache state sizing,
  and later DSV4/MiMo shared attention changes.
- Models: MiMo startup/TP1/KME/RoPE/PD/MTP/EPLB changes; HY3 disaggregation;
  MiniMax sequence parallel and Marlin; Kimi K2.7; Qwen3.5; DeepSeek V3.2;
  GLM and vision-side compatibility updates.
- MoE and quantization: LightOp TopK and multistream handling, MegaMoE,
  DeepEP/EPLB, W4A16 and DeepGEMM paths, AITER ASM shuffle, group-FP8 and
  fused MoE routing.
- Cache/disaggregation: Mooncake, HiCache offload, PD transfer, SWA stale-page
  handling, L2Norm, and KV write layout changes.
- Kernel/runtime/CI: gfx938 JIT/AITER/LightOp kernels, runtime logging,
  dependency/container scripts, and expanded registered DCU tests.

High-risk review areas are DSV4 JIT/attention/cache, `ServerArgs`, model runner
and parallel state, DeepEP dispatch, fused MoE/quantization, graph capture, and
any official rename that moved a DCU-bearing implementation.

## 3. Forward-port checkpoints

| Step | Old endpoint | Full / non-merge commits | Files | Main themes |
|---|---|---:|---:|---|
| 1 | `8736a794acee8253019704cf00a901fd7ffcefbe` | 41 / 30 | 54 | DSV4 BF16 KV, TopK VMFault, NSA/DCU, MiMo startup, SWA, runtime logging, Kimi Marlin, DCU CI |
| 2 | `fde56844fca442108bf3d2c71cbdeacb4ddb8f08` | 68 / 56 | 130 | LightOp TopK/multistream, HY3 PD, MegaMoE, C16 KV writes, Mooncake/L2Norm, AITER ASM, EPLB, EP W4A16/DeepGEMM |
| 3 | `80571de9491c8fd80e6822c9fa4efeb02ff67cce` | 57 / 43 | 450 | diffusion ROCm, attention API, HY3 PD/disagg/EPLB, MiMo prefill communication, RMSNorm, compliance changes; reverted HY3 PP remains absent |
| 4 | `cf5983854be1f19237ba28416b438f7b8965cfe6` | 26 / 18 | 30 | MiMo-Pro TP1, HiCache/Mooncake offload, MiniMax SP/Marlin, Kimi K2.7, Qwen3.5 |
| 5 | `5ec8531b096fa3297ab034dedc873aad215f2c35` | 22 / 13 | 17 | MiMo KME/RoPE/PD/MTP/EPLB, DeepSeek V3.2, W4A16 fused MoE |

Each step merges the exact old endpoint with `--no-ff`, resolves against current
`main`, the old commit intent, and the previous forward-port result, then records
textual conflicts plus refactor/move audit evidence. An obsolete old file is
deleted rather than revived when its behavior has moved to the official API.

## 4. Validation and execution rules

For every step:

1. Require no unmerged index entries or precise conflict markers.
2. Run `git diff --check`, compile changed Python files, and targeted Ruff
   `E9,F401,F811,F821,F841` on conflicts and high-risk semantic ports.
3. Run DCU registration, DSA alias/CLI/registry, and gfx938 HIP setup gates when
   applicable.
4. Confirm `SGLANG_USE_AITER_AG=0` is retained.
5. Immediately before runtime, inspect both `zz-nmz22` and `zz-nmz26` with
   `hy-smi`. If either selected device set is occupied, do not start there.
6. Run only the pure-TP command:

   ```bash
   bash /home/scripts/sglang/run_dpsk-v4.sh 10015 /home/model/DeepSeek-V4-Flash-FP8-Channel
   ```

7. Gate on service readiness, `/health` HTTP 200, and one short `/generate`
   request without a worker exit. The already deferred empty-output/NaN issue is
   recorded as an accuracy failure but remains non-blocking for forward-port
   integration unless a step newly prevents startup or request handling.
8. Do not run broad CI, other models, or other topologies. Stop after at most
   five focused unsuccessful runtime fixes and return the exact blocker for
   owner review.

GitHub pushes are never attempted from `zz-nmz22 / rye_sglang_open`. Use
`zz-nmz22 / rye_sglang_0601` or `zz-nmz26 / rye_sglang_open` because those
containers have a working GitHub SSH key. The `/home` tree is shared; do not
rsync repositories between the two test machines.

## 5. Execution progress

| Step | Endpoint | Status | Validation |
|---|---|---|---|
| 1 | `8736a794acee8253019704cf00a901fd7ffcefbe` | committed as `e7e06b77881d` | static gates passed; pure-TP startup, health, and request passed; known empty output remains non-blocking |
| 2 | `fde56844fca442108bf3d2c71cbdeacb4ddb8f08` | committed as `c7ffa6497a9e` | 32 textual conflicts; static gates passed; pure-TP passed after one focused optional-DeepGEMM import fix |
| 3 | `80571de9491c8fd80e6822c9fa4efeb02ff67cce` | resolved and validated; merge commit pending | 66 textual conflicts; static gates and pure-TP startup/health/request passed; known empty output remains non-blocking |
| 4 | `cf5983854be1f19237ba28416b438f7b8965cfe6` | pending | not run |
| 5 | `5ec8531b096fa3297ab034dedc873aad215f2c35` | pending | not run |

Step 2 keeps the official-main file/API layout canonical. Its old C16/BF16 KV
store moved from `mem_cache/utils.py` to `kernels/ops/kvcache/mla_buffer.py`;
old CUDA-graph behavior moved from the deleted
`model_executor/cuda_graph_runner.py` to
`model_executor/runner/decode_cuda_graph_runner.py`. Detailed conflict rows,
DCU audits, static evidence, and runtime evidence are in
`docs/internal/dcu-main-forward-port-v0.5.12-dev-step2-conflict-review.md`.

Step 3 retains current DSA/speculative/disaggregation structures, ports FSDP
streaming plus HY3 PD/EPLB intent to current APIs, and preserves the range's
final reverts of temporary HY3 PP support. Internal platform dispatch remains
`is_dcu/_is_dcu`; HCU is retained only in the old range's user-visible
compliance wording. Static gates and the pure-TP startup/health/request gate
passed without a code retry. The complete 66-file conflict inventory and
semantic audit are in
`docs/internal/dcu-main-forward-port-v0.5.12-dev-step3-conflict-review.md`.
