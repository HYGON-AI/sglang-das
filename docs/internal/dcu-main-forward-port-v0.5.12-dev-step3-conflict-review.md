# v0.5.12_dev Forward-Port Step 3 Conflict Review

## Scope

- Branch: `forward-port/v0.5.12-dev-20260715`
- Parent: Step 2 merge `c7ffa6497a9e783e37a18556639ca7eb6138d292`
- Old range: `fde56844fca442108bf3d2c71cbdeacb4ddb8f08..80571de9491c8fd80e6822c9fa4efeb02ff67cce`
- Range size: 57 full-graph commits, 43 non-merge commits, 450 old-range files
- Resolved staged tree before this review: 393 files, 8,366 insertions, 2,117 deletions
- Git-reported textual conflicts: exactly 66 files

The latest `main` layout and contracts remain canonical. The old range adds
diffusion ROCm support, FSDP streaming, HCU-visible compliance text, HY3
PD/EPLB work, attention and quantization adaptations, and MiMo communication
changes. Its temporary HY3 pipeline-parallel implementation is followed by
explicit reverts in the same range and is intentionally absent from the result.

## Textual conflict decisions

### Current-main workflow definitions retained (12)

The old workflow copies predate the current CI matrix. All twelve conflicts
were resolved to current `main`; the range's non-conflicting HCU manual/nightly
workflow additions remain staged.

- `.github/workflows/amd-aiter-scout.yml`
- `.github/workflows/amd-ci-job-monitor.yml`
- `.github/workflows/ci-coverage-overview.yml`
- `.github/workflows/nightly-test-amd-rocm720.yml`
- `.github/workflows/nightly-test-amd.yml`
- `.github/workflows/nightly-test-intel.yml`
- `.github/workflows/nightly-test-npu.yml`
- `.github/workflows/nightly-test-nvidia.yml`
- `.github/workflows/pr-test-amd-rocm720.yml`
- `.github/workflows/pr-test-amd.yml`
- `.github/workflows/pr-test.yml`
- `.github/workflows/runner-utilization.yml`

### Removed documentation paths retained as removed (11)

These are old Sphinx or obsolete index paths whose maintained content is now
under `docs_new`. They were not revived; automatically merged changes in the
current documentation tree were retained.

- `docs/advanced_features/sgl_model_gateway.md`
- `docs/developer_guide/bench_serving.md`
- `docs/developer_guide/setup_github_runner.md`
- `docs/diffusion/api/openai_api.md`
- `docs/diffusion/disaggregation.md`
- `docs/references/multi_node_deployment/lws_pd/lws-examples/p.yaml`
- `docs/references/multi_node_deployment/lws_pd/lws_pd_deploy.md`
- `docs/references/multi_node_deployment/multi_node.md`
- `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`
- `docs/supported_models/specialized/reward_models.md`
- `docs_new/index.mdx`

### Runtime and model conflicts (34)

| File or group | Resolution |
|---|---|
| `python/sglang/jit_kernel/deepseek_v4.py`, `srt/arg_groups/deepseek_v4_hook.py`, `srt/layers/deepseek_v4_rope.py` | Kept removed compatibility wrappers absent; their DSV4 behavior already lives in current JIT, argument, and rotary modules. |
| `multimodal_gen/runtime/loader/component_loaders/transformer_loader.py`, `multimodal_gen/runtime/loader/fsdp_load.py` | Ported old FSDP streaming and shard iteration into current `WeightLoadPlan`, preprocessing, and bitsandbytes full-load contracts. |
| `srt/disaggregation/decode.py`, `srt/disaggregation/prefill.py` | Kept the current disaggregation API; old HY3 PP-only edits were reverted later in this exact range. |
| `srt/distributed/device_communicators/custom_all_reduce_utils.py`, `srt/distributed/parallel_state.py` | Retained current graph/memory-saver and AITER communicator safeguards. `SGLANG_USE_AITER_AG=0` remains the runtime policy. |
| `srt/environ.py`, `srt/layers/attention/aiter_backend.py`, `srt/layers/attention/dsv4/sparse_prefill_utils.py` | Kept current environment and attention contracts while accepting non-conflicting old functionality. |
| `srt/layers/attention/nsa/index_buf_accessor.py`, `nsa/nsa_indexer.py`, `nsa/tilelang_kernel.py`, `nsa/triton_kernel.py`, `nsa_backend.py` | Did not revive the removed NSA implementation. DSA is canonical; the supported `nsa` CLI/registry alias remains. |
| `srt/layers/layernorm.py` | Kept the current DCU-aware layernorm implementation and its current custom-op API. |
| `srt/layers/quantization/fp8_utils.py` | Combined the old fake-op behavior with the current quantization API and restored its platform guard to `_is_dcu`. |
| `srt/layers/quantization/quark_int4fp8_moe.py` | Kept removed legacy Quark module absent; current compressed-tensors/quantization paths remain canonical. |
| `srt/managers/detokenizer_manager.py`, `srt/models/apertus.py`, `srt/models/deepseek_common/attention_backend_handler.py`, `srt/models/deepseek_v2.py` | Retained current APIs and previously forward-ported DCU behavior. |
| `srt/models/hunyuan_v3.py`, `srt/models/hunyuan_v3_nextn.py` | Ported HY3 precision, PD/EPLB, redundant experts, and current expert-location config APIs; removed obsolete PP imports and preserved the range's final PP reverts. |
| `srt/multimodal/processors/base_processor.py` | Kept current processor API and accepted compatible non-conflicting range changes. |
| `srt/server_args.py` | Applied the old attention-backend defaults and validation to current argument groups. DCU MHA resolves to `fa3`; DCU MLA/DSA uses current `dcu_mla` and canonical `dsa_*` fields. |
| `srt/speculative/eagle_info.py`, `eagle_info_v2.py`, `eagle_utils.py`, `eagle_worker.py`, `ngram_info.py` | Kept current speculative-v2 layout; removed compatibility files were not revived. |
| `srt/utils/common.py` | Kept current exported platform API, including `is_dcu`; no runtime `is_hcu` predicate was introduced. |

### Kernel and test conflicts (9)

- `sgl-kernel/python/sgl_kernel/flash_mla.py`: retained the current FlashMLA
  compatibility surface and current DCU dispatch.
- `test/manual/ep/test_eplb.py`: retained current EPLB coverage and removed one
  unused merge artifact.
- `test/registered/disaggregation/test_disaggregation_pp.py`: retained current
  test API; reverted old HY3 PP coverage was not restored.
- `test/registered/language/test_srt_backend.py`: kept removed obsolete test
  absent.
- `test/registered/spec/eagle/test_eagle_infer_a.py`,
  `test_eagle_infer_b.py`, and `test_eagle_infer_beta.py`: kept removed legacy
  EAGLE suites absent in favor of current registered speculative coverage.
- `test/registered/unit/managers/test_profile_merger_http_api.py` and
  `test/registered/unit/server_args/test_server_args.py`: retained current test
  locations/contracts; equivalent current coverage remains present.

## Refactor and `_is_dcu` audit

- Five runtime Python files from the HCU text-sanitization range initially
  referenced a new `is_hcu()` API that does not exist on current `main`:
  `check_env.py`, `custom_all_reduce.py`, `attention/vision.py`, `mxfp4.py`,
  and `glm4_moe.py`. Their internal dispatch now uses `is_dcu/_is_dcu` while
  retaining the old range's HCU user-visible log and compliance wording.
- The same rule was applied to `sgl-kernel/csrc/mamba/causal_conv1d.cu` and
  `sgl-kernel/tests/test_moe_align.py`: internal helpers remain DCU-named;
  HCU is only the visible platform label.
- `attention/vision.py` and `mxfp4.py` duplicate `get_bool_env_var` imports
  created by automatic merging were removed.
- `model_runner_kv_cache_mixin.py` was adapted from the removed
  `get_attention_tp_size()` helper to `get_parallel().attn_tp_size`.
- `arg_groups/overrides.py` resolves DCU FP8 DSA to
  `flashmla_auto/flashmla_kv`, DCU non-FP8 to `sparse/sparse`, and leaves
  generic HIP on `tilelang`.
- Automatically merged `_is_hip` paths in attention, MoE, quantization,
  DeepEP, DSV4, cache, and graph code were reviewed. Dedicated DCU behavior
  remains ahead of generic HIP behavior; no deleted DSV4/NSA compatibility
  module was revived.
- `SGLANG_USE_AITER_AG=0` remains explicitly exported by the pure-TP script.

## Validation evidence

Static gates passed:

- `git ls-files -u`: zero entries.
- Precise `<<<<<<<`, `=======`, `>>>>>>>` scan: no markers.
- `git diff --check`: passed.
- All 352 changed Python files compiled.
- Broad changed-file Ruff `E9,F821` and targeted high-risk Ruff
  `E9,F401,F811,F821,F841`: passed.
- Import smoke passed for ten high-risk environment, communicator, attention,
  quantization, GLM, ServerArgs, FSDP, and HY3 modules.
- DCU registration passed with 276 registered files and the existing CPU-utils
  warning.
- DSA alias/CLI/registry passed 19 tests.
- gfx938 `setup_hip.py --name` passed.

Pure-TP gate:

- Immediate preflight: `zz-nmz26` was occupied at VRAM 57% on all eight
  devices and was not used; `zz-nmz22` was VRAM/HCU 0% on all eight devices.
- Exact command: `bash /home/scripts/sglang/run_dpsk-v4.sh 10015
  /home/model/DeepSeek-V4-Flash-FP8-Channel`.
- All 46 shards loaded, decode graphs captured for `bs=128..1`, and service
  readiness completed.
- `/health` returned HTTP 200 and one short `/generate` returned HTTP 200
  without worker failure.
- The response remained empty with eight zero output IDs. This is the known
  deferred NaN/accuracy observation, is not an accuracy pass, and remains
  non-blocking under the agreed startup/request gate.
- The service process group was stopped, port 10015 closed, and all eight
  `zz-nmz22` devices returned to VRAM/HCU 0%.

## Integration decision

All scoped static and pure-TP functional gates passed without a code retry.
Commit the exact old endpoint as the Step 3 no-ff checkpoint.
