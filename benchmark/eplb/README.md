# Static EPLB placement with precomputed LP dispatch

EPLB chooses expert replicas and their physical placement. LP chooses how to
divide each logical expert's tokens among those existing replicas. This
experimental HIP path combines a fixed EPLB placement with an offline LP:
inference samples from cached probabilities without per-layer token counting,
count all-reduce, or LP solving.

## Prepare and use the artifact

Use a frozen placement tensor `physical_to_logical_map` of shape
`[layers, physical_experts]`, plus the corresponding representative EPLB recorder
`logical_count` tensor. Counts may be `[layers, logical_experts]` or
`[steps, layers, logical_experts]`; steps are summed. The benchmark snapshot key
`logical_counts` is also accepted. Both inputs must cover the same model layers
and logical experts. Physical IDs must belong to equal contiguous blocks of
experts per EP rank. The solver keeps the supplied placement unchanged.

The offline command needs CPU PyTorch, NumPy and SciPy. It does not load the model
or use a GPU. Run it on the same statistics window used for the static EPLB
placement:

```bash
python3 benchmark/eplb/prepare_static_lp.py \
  --layout static_mapping.pt \
  --counts expert_distribution.pt \
  --ep-size 32 \
  --output static_lp_probabilities.pt
```

Copy identical placement and probability artifacts to all nodes. Add these
options to the existing HIP DeepEP + DeepGEMM server launch, retaining separate
shared experts:

```bash
export SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS=/path/to/static_lp_probabilities.pt
# Additional sglang serve arguments:
# --init-expert-location /path/to/static_mapping.pt
# --ep-dispatch-algorithm lp
# --ep-num-redundant-experts 64  # Must match the supplied placement.
```

`--enable-eplb` is not needed to load a static EPLB placement. This experiment
does not enable online expert relocation. Initialization checks that the LP
artifact matches the exact placement, validates nonnegative finite masses, and
compacts the padded replica table while retaining every valid replica. Zero-mass
logical experts fall back to uniform sampling over their valid replicas.
Logical expert choices and routing weights are preserved; only the physical
replica changes. Compact static dispatch is currently limited to HIP.

The regular LPLB path without the environment variable computes global per-batch
expert counts and solves a new LP. Its CUDA solver is separate from the offline
SciPy solver. An experimental HIP online solver was measured outside this
change, but is not included or enabled here. With online EPLB relocation, changed
placement requires new LP weights: the fixed artifact is rejected when its map
no longer matches. This change does not implement asynchronous table refresh.

## Measured scope

GLM-5.3 Channel FP8, two 16-GPU gfx938 nodes, TP/DP/EP32, BS5 per DP,
input 112640 / output 1024, fake-prefill, MTP 5/1/6 with simulated acceptance 4.5,
VP16, 16 tokenizer workers, and the same static EPLB placement with 64 redundant
experts. Existing attention and BLAS tuning remained fixed. Each arm used one
warmup and three unprofiled runs of 160 requests, all successful.

| Placement and dispatch | Mean full speculative cycle | Normalized gen tokens/s/DP | Client mean TPOT |
| --- | ---: | ---: | ---: |
| Static EPLB + locality_fair | 120.20 ms | 187.2 | 27.61 ms |
| Same static EPLB + precomputed LP | 114.96 ms | 195.7 | 26.39 ms |

Cycle time is estimated from all 32 ranks' contiguous BS5 scheduler logs in the
same token range, excluding startup and tails. It is not a GPU trace timing.
Normalization uses fixed acceptance 4.5; fake-prefill and simulated acceptance
do not establish model quality or real acceptance rates. These timings apply to
the tested full configuration, not this commit on an otherwise untuned server.

The retained global compact table reduces padded replica width from 320 to 65
for this layout. The HIP dispatch kernel and full replica metadata were checked
with negative padding IDs, zero-probability fallback, deterministic uniforms,
and graph replay. Further per-layer fusion and rank-stratified sampling did not
produce stable end-to-end gains and are excluded.

The offline LP minimizes maximum rank token mass using historical aggregate
counts. It does not minimize network traffic or account for future workload
drift. Low-frequency asynchronous updates and communication-aware objectives
remain possible follow-up work, not measured improvements in this change.
