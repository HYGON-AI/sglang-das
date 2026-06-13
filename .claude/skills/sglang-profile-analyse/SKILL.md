---
name: sglang-profile-analyse
description: Analyze SGLang model serving performance from startup logs, benchmark commands, and torch profiler traces or screenshots. Use when diagnosing SGLang prefill/decode/profile bottlenecks, GPU utilization, CUDA kernels, KV cache, attention, sampling, memory, or throughput/latency regressions.
disable-model-invocation: false
---

# SGLang Profile Analyse

## Purpose

Guide performance analysis for models served by the SGLang framework. Always start from the SGLang startup log and the exact benchmark/evaluation command, then use torch profiler artifacts or screenshots to identify decode or profile-stage bottlenecks and recommend targeted next steps.

## Required First Step

Before analyzing profiler data, ask the user to provide:

1. The SGLang startup log file or full pasted startup log.
2. The exact benchmark/evaluation command they ran.
3. The model name/path and served model configuration if not visible in the log.
4. The torch profiler file, trace, exported table, Chrome trace, TensorBoard trace, or screenshot.
5. Whether the target stage is `decode`, `prefill`, `profile`, or unknown.

If the user only provides a torch profiler screenshot or file, do not jump directly to conclusions. First request the missing startup log and benchmark command, because SGLang startup configuration often determines the bottleneck.

Suggested prompt to the user:

```text
请先上传 SGLang 启动日志文件，以及你用于评测的完整命令。随后请上传 torch profiler 文件、trace、导出的表格，或 profiler 截图。请同时说明你希望分析 decode、prefill 还是 profile 阶段；如果不确定，我会先根据日志和 profiler 特征判断。
```

## Analysis Workflow

### 1. Parse SGLang startup log

Extract and summarize:

- SGLang version, commit, Python/PyTorch/CUDA versions when available.
- GPU model, GPU count, tensor parallelism, data parallelism, pipeline parallelism.
- Model path/name, dtype, quantization, load format, trust-remote-code settings.
- Attention backend, decode backend, CUDA graph settings, torch compile settings.
- KV cache configuration, max running requests, max total tokens, chunked prefill settings.
- Context length, memory fraction, available memory, cache block/token capacity.
- Server arguments affecting performance: batch sizes, scheduling policy, speculative decoding, radix cache, overlap scheduling, grammar constraints, sampling options.

Flag anything missing or suspicious before reading profiler details.

### 2. Parse benchmark/evaluation command

Extract:

- Workload type: offline benchmark, HTTP benchmark, eval harness, custom script.
- Prompt length, output length, request rate, concurrency, batch size, dataset, number of prompts.
- Sampling parameters: temperature, top-p, top-k, max tokens, stop conditions.
- Measurement target: TTFT, TPOT, ITL, throughput, QPS, E2E latency.
- Whether the command stresses prefill, decode, or mixed workloads.

Check for command/log mismatches, such as model mismatch, wrong endpoint, unexpected max tokens, low concurrency, or CPU-side benchmark overhead.

### 3. Determine the stage

Use the log and workload shape to classify the profiler data:

- `prefill`: large prompt processing, attention over prompt, high matmul/GEMM volume, TTFT dominated.
- `decode`: one or few tokens per step, repeated small GEMMs, attention/KV cache reads, sampling, scheduler overhead, TPOT/ITL dominated.
- `profile`: SGLang internal profiling or warmup/capacity estimation, often not representative of steady-state serving.
- `mixed`: prefill and decode overlap; separate the two before recommending optimizations.

If the stage cannot be determined, ask for trace time range, profiler labels, or benchmark parameters.

### 4. Analyze torch profiler artifacts

Accept any of these inputs:

- `.json` Chrome trace.
- TensorBoard profiler directory.
- PyTorch profiler text table.
- Exported CSV/table.
- Screenshot of profiler timeline, operator table, CUDA kernel table, memory view, or TensorBoard trace.

When a file is available, inspect it directly when possible. For screenshots, ask the user to include the operator table and CUDA kernel table if the bottleneck is not visible.

Focus on:

- CPU total/self time vs CUDA total/self time.
- CUDA kernel occupancy and gaps between kernels.
- GEMM kernels, attention kernels, paged attention, flash attention, all-reduce, memcpy, sampling kernels.
- Shape patterns: many tiny kernels, long single kernels, excessive synchronization.
- CPU scheduler/tokenizer/sampling overhead.
- H2D/D2H copies, blocking `cudaMemcpy`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`.
- NCCL/all-reduce time for multi-GPU tensor parallel serving.
- Memory allocation/free overhead, fragmentation, KV cache pressure.

### 5. Diagnose common SGLang bottlenecks

#### Decode bottlenecks

Look for:

- Small-batch decode causing low GPU utilization.
- Many short CUDA kernels with large gaps.
- Attention/KV cache kernels dominating TPOT.
- GEMM under-occupancy due to too little concurrency.
- CPU scheduling or sampling dominating between decode steps.
- NCCL all-reduce overhead in tensor parallel mode.
- Frequent request churn preventing stable batching.

Recommended checks/actions:

- Increase concurrency/request rate if workload is under-driving the GPU.
- Compare single-GPU vs tensor-parallel performance for small models or low concurrency.
- Check max running requests, max total tokens, memory fraction, and KV cache capacity.
- Verify CUDA graph usage and whether shapes are captured or repeatedly falling back.
- Reduce expensive sampling settings if sampling dominates.
- Separate network/client overhead from server-side TPOT.

#### Prefill bottlenecks

Look for:

- GEMM/attention kernels dominating TTFT.
- Very long prompts, chunked prefill behavior, or insufficient chunk sizing.
- CPU tokenizer overhead before GPU work begins.
- H2D transfer or input preparation gaps.
- Contention between prefill and decode under mixed workloads.

Recommended checks/actions:

- Test fixed prompt/output lengths to isolate prefill.
- Tune chunked prefill and scheduling settings.
- Compare attention backend choices if applicable.
- Check whether prefix/radix cache should help repeated prompts.

#### Profile-stage bottlenecks

Look for:

- SGLang warmup, memory profiling, CUDA graph capture, or capacity estimation.
- One-time model loading or compilation effects.
- Non-representative synthetic shapes.

Recommended checks/actions:

- Do not treat profile-stage timings as steady-state serving performance.
- Ask for a profiler trace during actual benchmark traffic.
- Separate startup/warmup/profile time from benchmark measurement window.

## Output Format

Respond in Chinese unless the user asks otherwise. Use this structure:

```markdown
## 结论摘要
- 主要瓶颈：...
- 影响阶段：decode / prefill / profile / mixed
- 置信度：高 / 中 / 低

## 依据
### 启动日志关键信息
| 项目 | 值 | 影响 |
|---|---|---|

### 评测命令关键信息
| 项目 | 值 | 影响 |
|---|---|---|

### Profiler 证据
| 现象 | 证据 | 推断 |
|---|---|---|

## 可能原因排序
1. ...
2. ...
3. ...

## 建议的验证实验
1. ...
2. ...
3. ...

## 优化建议
- 短期：...
- 中期：...
- 需要补充信息：...
```

## Guardrails

- Do not claim a bottleneck without tying it to startup-log settings, benchmark shape, or profiler evidence.
- Distinguish one-time startup/profile overhead from steady-state decode throughput.
- Do not over-optimize based on screenshots alone; ask for raw profiler files when screenshots are ambiguous.
- If profiler evidence conflicts with benchmark metrics, explicitly call out the conflict and propose a validation experiment.
- Prefer concrete experiments over generic advice.
- Keep recommendations specific to SGLang serving and the user's observed workload.
