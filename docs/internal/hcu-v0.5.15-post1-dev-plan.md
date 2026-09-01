# v0.5.15.post1_dev Development and Release Plan

Plan date: 2026-07-16 CST

Last workflow update: 2026-07-22 CST

## 1. Baseline and branch roles

The official-main catch-up and the complete `v0.5.12_dev` forward-port are now
landed on internal `main` through merge
`98828d29049179e69d1be31a0163a9546497b9fd`.

| Branch / tag | Role |
|---|---|
| `main` | Internal trunk. Periodically integrates exact official-main checkpoints and receives forward-ports from the release branch after release. |
| `v0.5.15.post1_dev` | Active HCU development/release branch for debugging, optimization, stabilization, and release preparation. |
| `sync/official-main-daily-YYYYMMDD` | Short-lived exact-endpoint official sync branch created from `main`; never created from the release branch. |
| `backport/main-to-v0.5.15-post1-dev-*` | Exceptional selective backport of a required, already-integrated main fix into the active release branch. |
| `forward-port/v0.5.15-post1-dev-*` | Reviewable post-release port of release-branch patches back to `main`. |
| `v0.5.15.post1` | Existing immutable annotated official main-equivalent marker at `65f3bd9426e5`; it is not moved to the later forward-port landing. |

The initial `v0.5.15.post1_dev` base is the `main` commit that contains this
plan and the forward-port landing. The branch must not be cut from the older
`v0.5.15.post1` tag because that tag predates the five forward-port steps.

## 2. Main official-sync lane

Official changes continue to enter `main` independently of release work:

1. Fetch `/home/officials/sglang` and record the exact previous and target
   official SHAs.
2. Create `sync/official-main-daily-YYYYMMDD` from current `main`.
3. Merge only the exact official endpoint with `--no-ff`.
4. Resolve against official structure, current `main`, and retained HCU intent;
   keep `_is_hcu` ahead of generic `_is_hip` where HCU has a dedicated path.
5. Run the current static gates and the agreed pure-TP functional gate.
6. Update the conflict ledger and conflict-review artifact, then merge and push
   `main` without rewriting history.

Official sync commits do not flow wholesale into `v0.5.15.post1_dev`. If the
release needs one of them before release, first integrate and validate it on
`main`, then selectively backport the smallest dependency-complete change via
a dedicated backport branch. Record both the official SHA and the internal
main SHA in the release-branch commit message or PR.

## 3. Release development lane

All near-term HCU debugging, optimization, and release stabilization targets
`v0.5.15.post1_dev`:

- one feature or bug class per commit/PR where practical;
- preserve `_is_hcu` separation for LightOp, AITER, DeepEP, DeepGEMM, DSV4,
  cache layout, graph, FP8, and communicator behavior;
- do not merge official main directly into the release branch;
- do not mix unrelated official sync with a release fix;
- keep the branch linear through normal PR merges; never force-push public
  history;
- record source issue, affected model/topology, validation, and whether the
  patch must later be forward-ported.

Recommended labels:

- `target: v0.5.15.post1_dev`
- `release: v0.5.15.post1`
- `needs-forward-port`
- `main-equivalent-present`
- `hcu-release-only`

## 4. Validation policy

For each release-branch patch, Codex scope remains intentionally bounded:

1. Complete code changes, conflict fixes, and static validation.
2. Compile changed Python files and run targeted Ruff
   `E9,F401,F811,F821,F841`, precise marker scan, and `git diff --check`.
3. Run HCU registration, DSA alias/CLI/registry, and gfx938 HIP metadata gates
   when applicable.
4. Immediately before model validation, check every candidate below with
   `hy-smi`; never launch on a node with VRAM or HCU activity:
   - `zz-nmz22 / rye_sglang_0716`
   - `zz-nmz26 / rye_sglang_0716`
   - `zz-nmz20 / rye_sglang_0716`
   - `zz-sglang2 / rye_sglang_0720`
5. Run only:

   ```bash
   cd /home/scripts/sglang
   bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-FP8-Channel
   ```

6. Gate on service readiness, `/health` HTTP 200, and one short `/generate`
   without worker exit. Broad CI, accuracy suites, other models, and other
   topologies remain owner-run.
7. Stop after at most five focused unsuccessful runtime fixes and return the
   exact blocker for owner review.

Keep `SGLANG_USE_AITER_AG=0` until the registered graph path is explicitly
validated and the workaround is deliberately retired.

## 5. Patch tracking

Every release-branch patch should be classified in the table below as it lands.
This table is the source list for the final forward-port campaign.

| Release commit | Area/model | Main equivalent | Validation | Forward-port status |
|---|---|---|---|---|
| _pending_ | _pending_ | _none / SHA_ | _pending_ | _required / skipped / complete_ |

Rules:

- `main equivalent` means a patch-id or semantic equivalent already exists on
  main; record the SHA and do not duplicate it.
- `release-only` patches require an explicit reason and owner decision before
  they can be skipped.
- high-risk patches touching attention, MoE, DeepEP, quantization, DSV4,
  sgl-kernel, cache, communication, or graph capture require a semantic audit,
  not only a clean cherry-pick.

## 6. Release and post-release forward-port

Before release:

1. Freeze new features on `v0.5.15.post1_dev`.
2. Complete owner-run CI, accuracy, topology, and performance validation.
3. Resolve the internal release-tag name without moving the existing public
   `v0.5.15.post1` marker. A distinct HCU release tag is required unless the
   repository owner explicitly authorizes a public tag migration.
4. Record the release commit, artifacts, known issues, and validation evidence.

After release:

1. Freeze the release source range at the final release commit.
2. Compare every release-branch non-merge patch against current `main` using
   patch-id plus semantic symbol/file-move review.
3. Group dependency-related patches into small
   `forward-port/v0.5.15-post1-dev-*` branches.
4. Port to current main APIs and preserve both newer official behavior and the
   release patch's HCU intent.
5. Validate and merge each group to `main`; update the patch table immediately.
6. Retire `v0.5.15.post1_dev` to release-only maintenance after all required
   rows are `complete` or owner-approved `skipped`.

## 7. Push and workspace rules

- The shared repository is `/home/proj_sglang_open/sglang-das`; do not rsync
  the shared `/home` tree between test hosts.
- Run GitHub pushes only from `zz-nmz22 / rye_sglang_0601` or
  `zz-nmz26 / rye_sglang_open`.
- Do not push from `zz-nmz22 / rye_sglang_open`, whose GitHub SSH key is not
  usable.
- Never force-push `main`, `v0.5.15.post1_dev`, or public tags.
