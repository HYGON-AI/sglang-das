# Packed Paged KV To Varlen Benchmark Notes

## Purpose

This note records the measurements used to choose the default auto policy for
packing paged KV cache into contiguous varlen K/V before calling upstream
`flash_attn_varlen_func`.

The path is useful only when the upstream varlen kernel saves more time than
the extra pack work costs. The current policy is therefore shape-based:

```text
--pack-paged-kv-to-varlen auto
--pack-paged-kv-to-varlen-min-kv-tokens 16384
--pack-paged-kv-to-varlen-min-q-tokens 8192
```

Use `--pack-paged-kv-to-varlen on` to force this path while still keeping
correctness guards.

## Benchmark Setup

- Script: `test/registered/attention/bench_pack_paged_kv_to_varlen.py`
- Kernel-gap command:

```bash
python3 test/registered/attention/bench_pack_paged_kv_to_varlen.py --kernel-gap
```

- Dtype: fp16
- Page size: 64
- Head dim: 128
- Baseline: `vllm_flash_attn_varlen_func` with paged KV cache
- Packed path: pack paged KV into contiguous K/V, then call upstream
  `flash_attn_varlen_func`

The prefix-cache hit rate is modeled as:

```text
query_len = total_input_len * (1 - prefix_cache_hit_rate)
kv_len = total_input_len
```

The important target case is 50K+ input length with at least 70% prefix-cache
hit rate. At 75% hit rate, the query length is still large enough to amortize
packing overhead:

```text
51200 input tokens -> 12800 query tokens
65536 input tokens -> 16384 query tokens
```

## Key Results: Low KV-Head MQA/GQA

These cases are the main support for allowing batch-size > 1 when the per-rank
KV head count is small.

Results for `q=12800`, `total_kv_len=51200`:

| bs | q_head | kv_head | ideal_varlen_ms | paged_ms | pack_only_ms | packed_total_ms | packed_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 9.340 | 12.629 | 0.219 | 9.516 | 1.327 |
| 2 | 6 | 1 | 17.152 | 23.471 | 0.470 | 17.578 | 1.335 |
| 4 | 6 | 1 | 34.105 | 47.324 | 0.905 | 34.983 | 1.353 |
| 8 | 6 | 1 | 69.037 | 94.610 | 1.765 | 70.737 | 1.337 |
| 16 | 6 | 1 | 138.203 | 188.685 | 3.439 | 141.612 | 1.332 |
| 32 | 6 | 1 | 276.044 | 376.695 | 6.928 | 283.095 | 1.331 |
| 64 | 6 | 1 | 551.962 | 752.755 | 13.160 | 565.309 | 1.332 |
| 128 | 6 | 1 | 1105.066 | 1504.873 | 28.677 | 1133.901 | 1.327 |
| 1 | 8 | 2 | 11.711 | 15.834 | 0.436 | 12.138 | 1.305 |
| 2 | 8 | 2 | 22.910 | 31.620 | 1.093 | 23.983 | 1.318 |
| 4 | 8 | 2 | 47.090 | 63.238 | 2.006 | 48.712 | 1.298 |
| 8 | 8 | 2 | 120.281 | 126.103 | 4.027 | 123.445 | 1.022 |
| 16 | 8 | 2 | 245.228 | 252.166 | 7.846 | 258.459 | 0.976 |

Results for `q=16384`, `total_kv_len=65536`, `kv_head=1`:

| bs | q_head | kv_head | ideal_varlen_ms | paged_ms | pack_only_ms | packed_total_ms | packed_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 14.773 | 20.067 | 0.239 | 15.006 | 1.337 |
| 2 | 6 | 1 | 28.764 | 39.595 | 0.592 | 29.361 | 1.349 |
| 4 | 6 | 1 | 56.506 | 78.276 | 1.077 | 57.585 | 1.359 |
| 8 | 6 | 1 | 113.267 | 154.920 | 2.172 | 115.554 | 1.341 |
| 16 | 6 | 1 | 231.212 | 310.053 | 4.287 | 234.624 | 1.321 |
| 32 | 6 | 1 | 460.395 | 618.567 | 8.269 | 467.255 | 1.324 |
| 64 | 6 | 1 | 920.095 | 1234.569 | 16.545 | 936.393 | 1.318 |
| 128 | 6 | 1 | 1883.026 | 2468.655 | 33.105 | 1909.350 | 1.293 |

Additional `kv_head=2` results for `q=16384`, `total_kv_len=65536`:

| bs | q_head | kv_head | ideal_varlen_ms | paged_ms | pack_only_ms | packed_total_ms | packed_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 2 | 202.531 | 207.526 | 4.982 | 207.603 | 1.000 |
| 16 | 8 | 2 | 414.892 | 412.460 | 9.865 | 422.710 | 0.976 |

Conclusion: for `kv_head=1`, batch-size 8 through 128 still benefit in the
50K+/75% hit-rate region. For `kv_head=2`, batch-size 8 is already close to
break-even and batch-size 16 is negative, so the default policy keeps
`kv_head=2` limited to batch-size 4.

## Key Results: Higher KV-Head Counts

These cases justify limiting batch-size > 1 to low KV-head counts in auto mode.

Results for `q=12800`, `total_kv_len=51200`:

| bs | q_head | kv_head | ideal_varlen_ms | paged_ms | pack_only_ms | packed_total_ms | packed_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 4 | 11.849 | 16.119 | 0.953 | 12.759 | 1.263 |
| 2 | 8 | 4 | 23.729 | 31.811 | 2.234 | 25.814 | 1.232 |
| 4 | 8 | 4 | 57.293 | 63.276 | 4.268 | 60.824 | 1.040 |
| 1 | 8 | 8 | 12.334 | 16.223 | 2.315 | 14.682 | 1.105 |
| 2 | 8 | 8 | 32.192 | 31.920 | 5.235 | 36.829 | 0.867 |
| 4 | 8 | 8 | 70.945 | 63.499 | 10.469 | 81.722 | 0.777 |

Conclusion: `kv_head=4` is marginal at larger batch sizes, while `kv_head=8`
is not beneficial for batch-size > 1. The default auto policy therefore uses a
split batch limit internally: batch-size 128 for `kv_head=1`, batch-size 4 for
`kv_head=2`, batch-size 1 for `kv_head=3` and `kv_head=4`, and disables packing
for higher KV-head counts.

## Query-Length Sensitivity

Older synthetic runs with `kv_head=8` are not used as the current batch policy
because they overstate pack cost for MQA/GQA models. They are still useful to
show query-length sensitivity for batch-size 1.

| kv_len | hit_rate | query_len | paged_ms | packed_total_ms | pack_only_ms | packed_speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 51200 | 0.70 | 15360 | 18.876 | 16.593 | 2.356 | 1.138 |
| 51200 | 0.75 | 12800 | 16.302 | 14.731 | 2.341 | 1.107 |
| 51200 | 0.90 | 5120 | 7.579 | 8.164 | 2.344 | 0.928 |
| 51200 | 0.99 | 512 | 2.795 | 4.273 | 2.373 | 0.654 |
| 65536 | 0.70 | 19661 | 31.763 | 26.641 | 2.959 | 1.192 |
| 65536 | 0.75 | 16384 | 26.816 | 23.115 | 2.974 | 1.160 |
| 65536 | 0.90 | 6554 | 13.736 | 13.477 | 2.976 | 1.019 |
| 65536 | 0.99 | 655 | 3.559 | 5.887 | 2.960 | 0.605 |

Conclusion: very high prefix-cache hit rates produce short query lengths, so
the upstream kernel saving no longer pays for packing. This supports the
default `--pack-paged-kv-to-varlen-min-q-tokens 8192` threshold.
