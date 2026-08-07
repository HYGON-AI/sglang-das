# HCU registered tests

This directory holds HCU (海光 / DTK) backend specific CI tests, mirroring
the convention used by `test/registered/amd/` and `test/registered/ascend/`.

## Layout

- `interface/`       Smoke / API / health-check level tests.
- `accuracy/bw1100/` GSM8K / MMLU / MMMU style accuracy evaluation.
- `srt/bw1100/`      Dense text model server smoke tests.
- `vlm_models/bw1100/` VLM server smoke tests.
- `moe/bw1100/`      MoE model server smoke tests.
- `embedding/bw1100/` Embedding OpenAI API smoke tests.
- `reranker/bw1100/` Reranker OpenAI API smoke tests.
- `kernels/`         HCU-supported `sgl-kernel` whitelist tests.
- `perf/`           Reserved for future BW1100 throughput / TTFT / TPOT / ITL benchmarks.
- `disaggregation/bw1100/` Real two-node PD-disaggregation smoke tests.

Broad HCU registrations for existing `test/registered/**` files live in their
original feature directories.  The `test/registered/hcu/` subtree is only for
HCU-only smoke, accuracy, VLM, MoE, embedding, reranker, and kernel whitelist
tests.

## sglang-tly merge note

The broad HCU registration set was merged from `sglang-tly` into the current
BW1100 suite names. Historical `stage-a-hcu`, `stage-b-hcu`, and `k100/` names
from that repo are not used as current suite or directory names. Disabled and
unverified registrations keep their original reasons so registration coverage is
not mistaken for BW1100 pass status.

## Registering a test

Add a module level call near the top of the file:

```python
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=120, suite="stage-b-test-1-hcu-small")
```

For nightly tests, set `nightly=True` and use a `nightly-hcu-*` suite.

## Accuracy tests

HCU accuracy tests are organized by hardware generation.  BW1100 tests live in
`accuracy/bw1100/` and currently cover:

- `test_gsm8k_eval_hcu.py`: text math reasoning accuracy.
- `test_mmlu_eval_hcu.py`: text general knowledge accuracy.
- `test_mmmu_eval_hcu.py`: VLM multimodal understanding accuracy.

GSM8K and MMLU are registered in `nightly-hcu-accuracy`; MMMU is registered in
`nightly-hcu-vlm` so text nightly runs do not depend on VLM models by default.
Use `SGLANG_HCU_*` environment variables from `accuracy/README.md` to select
models, sample sizes, thresholds, and launch arguments.

Older planning notes may refer to this hardware target as K100. The current
implementation and baseline documents use BW1100 for the accuracy directory and
test class names.

## BW1100 smoke model coverage

The first BW1100 registered smoke layer intentionally checks only startup and
small API requests.  Accuracy thresholds stay in `accuracy/bw1100/`.

- Dense text: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, and
  Qwen2.5-7B-Instruct in `srt/bw1100/`.
- VLM: Qwen2.5-VL-3B-Instruct in `vlm_models/bw1100/`.
- MoE: Qwen3-30B-A3B base and instruct variants in `moe/bw1100/`; BW1100 smoke defaults to TP2 (`--tp-size 2`).
- Embedding: Qwen3-Embedding-0.6B in `embedding/bw1100/`; gte-Qwen2 remains an override candidate after tokenizer compatibility is fixed.
- Reranker: Qwen3-Reranker-0.6B in `reranker/bw1100/`.
- Kernels: only the known BW1100-supported `sgl-kernel` pytest whitelist in
  `kernels/`.

Default HCU server arguments follow the BW1100 cookbook/smoke baseline:

```text
--attention-backend fa3 --page-size 64 --trust-remote-code
--log-level warning --log-level-http warning
```

VLM tests additionally use:

```text
--mm-attention-backend fa3 --enable-multimodal
```

Memory sizing is not hard-coded. Override launch args with these variables when
a specific node or model needs different sizing:

- `SGLANG_HCU_SERVER_ARGS`
- `SGLANG_HCU_VLM_SERVER_ARGS`
- `SGLANG_HCU_MOE_SERVER_ARGS`
- `SGLANG_HCU_EMBEDDING_SERVER_ARGS`
- `SGLANG_HCU_RERANKER_SERVER_ARGS`

Model path overrides:

- `SGLANG_HCU_QWEN25_0P5B_MODEL`
- `SGLANG_HCU_SERVER_SMOKE_MODEL` (backward-compatible alias for the 0.5B smoke)
- `SGLANG_HCU_QWEN25_1P5B_MODEL`
- `SGLANG_HCU_QWEN25_7B_MODEL`
- `SGLANG_HCU_QWEN25_VL_3B_MODEL`
- `SGLANG_HCU_QWEN3_MOE_MODEL`
- `SGLANG_HCU_QWEN3_MOE_INSTRUCT_MODEL`
- `SGLANG_HCU_EMBEDDING_MODEL`
- `SGLANG_HCU_RERANKER_MODEL`

Local default model paths are skipped when absent, while explicitly configured
local paths fail fast. This keeps HCU CI from silently downloading gated or
large models.

Kernel whitelist selection is controlled by `SGLANG_HCU_KERNEL_TEST_SET`:

- `smoke`: default per-commit set.
- `nightly`: smoke plus heavier known-supported kernel files.
- `all-supported`: alias for the current full supported whitelist.

## Running locally

```bash
python3 test/run_suite.py --hw hcu --suite stage-a-test-1-hcu-small
python3 test/run_suite.py --hw hcu --suite stage-b-test-1-hcu-small
python3 test/run_suite.py --hw hcu --suite nightly-hcu-accuracy --nightly
python3 test/run_suite.py --hw hcu --suite nightly-hcu-vlm --nightly
```

## Two-node PD disaggregation

`nightly-hcu-disaggregation-16` is a real two-host suite. It is not a
single-host emulation and must be launched through
`.github/workflows/hcu-pd-1p1d.yml`.

The initial MiniMax-M2.7 configuration uses eight BW1100 HCUs on each host:

```text
Prefill: TP2 / PP4 / DP1
Decode:  TP8 / PP1 / DP1
Router:  runs on the Prefill host
KV:      Mooncake over one explicitly selected RDMA rail
RoCE:    GID index 3 (IPv4 RoCE v2 on the selected rail)
```

The workflow verifies both service identities before sending requests through
the Router. Direct execution of the registered test requires the P, D, and
Router services to be running already.
