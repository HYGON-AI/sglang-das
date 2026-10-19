# GLM-5.3 HCU decode runtime

These changes support the Channel-FP8 GLM-5.3 checkpoint and retain the SGLang
paths used in the gfx938 decode measurements:

- Map `gate_up_proj` to checkpoint `gate_proj` and `up_proj` names so the
  quantization matcher selects the checkpoint's intended weight precision.
- Allow MTP groups of six in the paired sparse MQA and masked TopK route.
  This requires a LightOp build supporting group6; API presence alone does not
  identify the supported group sizes.
- Reuse BF16 MLA absorption weights when their scale is already a Python one.
- Write the HCU BF16 V-BMM result directly in token-major layout, avoiding the
  following transpose/flatten copy. Quantized output-projection paths retain
  their existing handling.
- Use a stream event wait for HCU backends that do not consume CPU sequence
  lengths. CPU consumers retain a host wait before accessing their buffers.
- Keep the four-argument RDMA size hint for the measured HCU DeepEP package;
  other platforms retain their `num_topk` argument.

## Components required to reproduce the measured configuration

The node-local draft LM-head VP and static LP dispatch changes are included in
`feature/glm52-forward-port-v0.5.18`. Installing SGLang alone does not provide
the native operator optimizations or BLAS tuning artifacts.

| Component | Measured dependency |
| --- | --- |
| Platform | gfx938, 64 physical CUs, Python 3.10, Torch 2.10, C++ ABI1, DTK2604 |
| LightOp | `c17922f` integrates native FP8 group6 MQA row reuse and the BS5 dispatch into the normal package |
| FlashMLA | The measured page64-capable build, sparse decode split count 32 |
| DeepGEMM | Base `g2b4d4e` package plus the opt-in M32 masked FP8 library and dispatch adapter |
| hipBLASLt | `hipblaslt.config` for the measured library version |
| rocBLAS | The complete `library_gpu6/` directory, including DAT files and code objects |
| Static EPLB/LP | Matching `static_mapping.pt` and `static_lp_probabilities.pt` |

Select the two BLAS artifacts with:

```bash
export HIPBLASLT_TUNING_OVERRIDE_FILE=/path/to/tuning/hipblaslt.config
export ROCBLAS_TENSILE_LIBPATH=/path/to/tuning/library_gpu6
```

The verified libraries were hipBLASLt `libhipblaslt.so.0.10` and rocBLAS
`librocblas.so.4.3`. Recheck tuning after changing library versions; solution
indices are not portable across versions.

The final native libraries can be identified by SHA256:

| Library | SHA256 |
| --- | --- |
| FlashMLA extension | `969241119b0a9a265a15c845447fceaf548b4949f42b7c85d224c80cdcabf483` |
| LightOp full extension, including MQA row reuse | `d1bbc29cce6675e28c5dd63b7b7e8eb90ab22125d26ac3dc1f52d03f830376ce` |
| DeepGEMM M32 extension | `9b529c15521069b48a881c59513a6a75424e565a7fbaba01df27ec911e94c0be` |

The M32 source is commit `ee428b4efc9f174f1d1958d06cdba85cd7a8fddf` in the
separate DeepGEMM repository. Its source API requires an explicit M32 config;
the default remains the previous kernel. The measured adapter selects M32
for verify gate/up and down projections and retains the previous draft path.

The LightOp source is available on
[`feature/topk-dev-native-fp8-mtp516`](https://github.com/maxiaobetter/lightop/tree/feature/topk-dev-native-fp8-mtp516).
Set `LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3` for the measured row reuse.
The normal package uses 128 persistent CTAs for native FP8 group6 at 30 rows
when the caller supplies no explicit CTA count. The persistent scheduler is
retained, and no separate MQA Python extension is needed.

## Measurement scope

The complete configuration uses two nodes, TP/DP/EP32, five concurrent requests
per DP (160 globally), input 112640/output 1024, MTP5/1/6, VP16, 64 redundant
experts with static EPLB/LP, and 16 tokenizer workers. Its BS5 capacities are
`--max-running-requests 160`, `--cuda-graph-max-bs 5`, and
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32`. Increase these capacities
appropriately before increasing concurrency.

The latest complete experiment used fake-prefill and simulated acceptance4.5.
After one warmup, three unprofiled 160-request runs completed. The full-cycle
estimate from continuous BS5 windows across all32 ranks was 107.79 ms, actual
logged generation throughput averaged 210.63 tokens/s/DP, and throughput
normalized to acceptance4.5 was 208.76 tokens/s/DP. Client mean TPOT was
24.86 ms. These results include all dependencies above and are not the isolated
gain from this SGLang change.

The V-BMM layout change was numerically compared at the actual strides with
1/5/6/30/96 rows, with bitwise matching outputs. Fake-prefill and simulated
acceptance measure performance and execution stability; they do not establish
model accuracy, real MTP acceptance, or P/D transfer correctness.
