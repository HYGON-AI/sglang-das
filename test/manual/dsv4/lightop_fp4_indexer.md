# LightOp FP4 indexer on DCU

Set this variable before starting each SGLang worker:

```bash
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
```

The default is `0`. Set it to `0` (or unset it) and restart the workers to use
the original Triton indexer. An already captured graph keeps the kernels chosen
at capture time. This flag is independent of `SGLANG_USE_LIGHTOP`.

Install a LightOp build containing `op.paged_mqa_logits_fp4`, including the
scale-fold fix in [LightOp PR #54](https://github.com/HYGON-AI/lightop/pull/54).
Check the package from the same environment that starts the workers:

```bash
python -c 'import lightop; from lightop import op; print(lightop.__file__); print(hasattr(op, "paged_mqa_logits_fp4"))'
```

## Scope and fallback

- The switch applies to `fp4_index_logits_decode`, called by the low-ratio
  decode/target-verify path in `DeepseekV4AttnBackend`.
- SGLang passes the BF16 query and the FP4 page table to LightOp without a
  platform, head-count, batch/length or layout allowlist. LightOp owns device
  validation, input validation, BF16 versus lossless FP8 selection, and its
  `49152` crossover. The cache layout remains the packed E2M1 payload followed
  by per-32 E8M0 scales.
- If the package is missing or the wheel does not export the symbol, SGLang
  retains the Triton path and reports the reason once per process. Once the
  operator is present, LightOp validation or launch errors are propagated; they
  are not silently changed into a different result by a second SGLang guard.
- Eager warmup resolves the extension before graph capture. If discovery first
  occurs during capture, that graph uses Triton. Warm up with the switch enabled
  before capturing a graph intended to use LightOp.

LightOp keeps dot accumulation, ReLU, weighting and head reduction in FP32,
matching the DeepGEMM contract. The original Triton fallback has intermediate
BF16 rounding. Their logits and selected top-k indices can therefore differ.
The caller's visibility, candidate masks and top-k postprocessing are unchanged.

## Verification

```bash
python -m pytest -q test/manual/dsv4/test_lightop_fp4_indexer.py
```

The tests cover the environment gate, optional-extension discovery, native FP32
reference parity and graph replay with updated query, cache, weights, slots and
lengths. GPU cases skip when a device or LightOp symbol is unavailable. This is
an operator integration check; model-level accuracy and end-to-end throughput
require a separate service A/B run.
