# DSV4.1 HCU decode preparation fusions

These optional paths use LightOp transform/quantization operators on HCU with
ROCm PyTorch. They reuse the current WoA/WoB GEMM implementations.

## Enable the paths

Install the matching LightOp changes before starting a new worker process.

```bash
export SGLANG_OPT_DSV41_SMALL_OP_FUSION=1
# Required by the existing packed INT8 service projection, for INT8 models only:
export SGLANG_DSV41_INT8_SERVICE_OPT=1
```

Both flags default to false. The service flag already existed in the packed
projection implementation; this change registers it in `sglang.srt.environ`.
FP8 models do not require the INT8 service flag.

`SGLANG_HCU_INDEX_Q_ROPE_PACK` remains the independent selector for the existing
index-Q pack path. Small-op fusion also selects that path on gfx936/gfx938 when
its operand/layout guards pass, and allows the native-RoPE-compatible variant.
The two selectors do not represent two newly introduced native kernels.

For an ablation of these fusions, set small-op fusion and the independent
index-Q pack selector to zero, preserving the service flag, GEMM backend,
parallelism, EPLB configuration and model across captures.

## Operations and operands

| Path | Preparation produced for the existing consumer |
| --- | --- |
| INT8 inverse RoPE | Rotate BF16 head tails and quantize full head groups directly into the packed WoA input pitch. |
| INT8 WoA to WoB | Scale INT32 WoA accumulators, round through BF16, and quantize the full token to INT8 without allocating or storing a BF16 carrier. |
| FP8 inverse RoPE | Rotate BF16/FP16 head tails, quantize complete head groups to E4M3, and produce group-major WoA operands in one kernel. |
| FP8 WoA to WoB | Quantize complete token rows directly from group-major BF16/FP16 storage, avoiding a materialized floating transpose. |
| Strided Q RMSNorm | Reuse LightOp RMSNorm arithmetic on a padded Q slice and produce independent normalized, quantized and scale outputs, preserving the original packed Q/KV storage. |
| Index-Q | Extend the existing `_index_q_fp4_pack_weights_hcu_kernel` with native-AOT-compatible RoPE, preserving both FP4 quantization stages and head-weight scaling. |

`Int8QuantizedInput` carries a contiguous GPU `[M,K]` INT8 tensor, FP32 `[M,1]`
scales and the floating output dtype. Only explicitly supported linear methods
consume it. Row-parallel consumers require an already partitioned input and an
inactive sequence-parallel path. The model and DSPARK paths apply this guard
before asking the producer for quantized-only output.

## Numerical contracts

- RoPE rounds the transformed tail to the source dtype before calculating absmax.
  RoPE-to-INT8 uses half-away-from-zero rounding, with a `1e-10` absmax floor.
- INT32 scaling computes `(float(accum) * input_scale) * weight_scale` with
  explicit multiply rounding. WoB quantization observes the original floating
  dtype's rounding and uses round-to-nearest-even INT8. The BF16 buffer is removed;
  its rounding boundary is retained in registers.
- FP8 transform quantization uses E4M3fn and complete-row scales `absmax / 448`.
  Zero rows have zero scales. RMSNorm quantization retains its existing scale
  floor and reduction order.
- Native index-Q RoPE uses separate products for the real component and explicit
  FMA for the imaginary component, with implicit Triton contraction disabled.
  The Torch-compatible fallback keeps its existing rotation path.

## Compatibility and dispatch

Missing LightOp symbols and unsupported source layouts fall back to the existing
paths before rotation/quantization is applied. Invalid explicit prequantized
operands raise an error instead of being silently quantized again. Shape-specific
thread calibration lives in LightOp; unknown shapes and devices use the general
configuration. T40/T48 and the calibrated cases up to T128 do not impose a fixed
batch size on the public operators.

The FlashMLA adapter continues to pass `attn_sink` directly to FlashMLA as it did
on the base branch. No additional external sigmoid scaling is introduced here.

The LightOp BF16-to-FP32 GEMM selector now checks its operand dtype, contiguity
and K tile constraints before dispatch. This is a dispatch guard, not a new GEMM
algorithm. EPLB is configured separately and is not enabled by either fusion flag.

## Measurement scope

The November 15, 2026 investigation includes earlier integrated builds, existing
10-step traces, same-address INT8 graph microbenchmarks and completed GPQA runs.
The PR review adds metadata/layout guards and cleanup without rerunning GPU
experiments. Earlier scores and timings are not validation of this final commit.
The PR description separates kernel-chain timing from cross-run service timing
and lists the reviewed artifact names and limitations.
