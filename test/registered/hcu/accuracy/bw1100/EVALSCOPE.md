# HCU EvalScope accuracy

This standalone suite pins `evalscope==1.9.1` and evaluates each model on:

| Dataset | Samples | Max output tokens |
| --- | ---: | ---: |
| GSM8K | 200 | 16384 |
| MATH-500 | 200 | 16384 |
| HumanEval | 164 | 16384 |

Install only for this suite:

```bash
python3 -m pip install -r scripts/ci/hcu/requirements_evalscope.txt
```

HumanEval executes generated Python. Run it only in an isolated,
non-privileged container with no writable host mounts, then opt in:

```bash
export SGLANG_HCU_EVALSCOPE_ALLOW_CODE_EXECUTION=1
python3 test/run_suite.py \
  --hw hcu \
  --nightly \
  --suite nightly-hcu-accuracy-evalscope \
  --timeout-per-file 21600
```

Use `--include-file` to select one model. Small-sample checks can override
`SGLANG_HCU_EVALSCOPE_GSM8K_LIMIT`,
`SGLANG_HCU_EVALSCOPE_MATH_LIMIT`, and
`SGLANG_HCU_EVALSCOPE_HUMANEVAL_LIMIT`; thresholds are report-only for a
partial run unless `SGLANG_HCU_EVALSCOPE_ENFORCE_PARTIAL_THRESHOLDS=1`.

The default thresholds come from the complete July 30 validation. DeepSeek
HumanEval remains report-only because that baseline stopped at 122 of 164
samples. A complete valid run is required before setting its threshold.
