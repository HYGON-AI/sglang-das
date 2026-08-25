# DeepSeek-V4 FP8 KV Gather/Upconvert for BF16 FlashMLA

本文档说明 HCU/DCU 平台上 DeepSeek-V4（DSV4）的 BF16 FlashMLA 优化：KV cache
仍以紧凑的 FP8 格式保存，在调用现有 FlashMLA 之前，只将本次注意力真正访问的 KV
行 gather 到连续 workspace，同时反量化为 BF16。FlashMLA 本身不拆分、不替换，仍通过
原有的 `flash_mla_with_kvcache` 入口执行。

## 1. 目标

该实现希望同时满足以下需求：

- 保留 FP8 KV cache 的显存容量优势。
- 让最终 FlashMLA 使用 BF16 KV 输入进行计算。
- P 端普通 Prefill 与 D 端 Decode 可以使用同一条 BF16 gather 路径。
- 不修改 FlashMLA kernel，不把一次 FlashMLA attention 拆成多次调用。
- 使用环境变量按需启用，关闭后恢复原有行为。

该实现不是将整个 KV cache 永久转换成 BF16，而是只处理 indexer/Top-K 选中的有效
KV 行。因此，额外显存来自临时 BF16 workspace，而不是第二份完整 KV cache。

## 2. 数据流

```text
DSV4 packed FP8 KV cache
        |
        | Top-K token indices + valid lengths
        v
Triton gather/upconvert
  - gather selected rows
  - FP8 NOPE * ue8m0 scale -> BF16
  - copy BF16 ROPE
  - rebuild compact indices
        |
        v
contiguous BF16 KV workspace
        |
        | is_fp8_kvcache=False
        v
existing flash_mla_with_kvcache
        |
        v
attention output
```

`paged_mqa_logits_bf16_fp8_kv` 等 MQA logits kernel 属于 indexer/Top-K 选择阶段，
不是最终的 FlashMLA attention。判断此功能是否生效时，需要同时看到 gather kernel 和
后续 FlashMLA sparse-attention kernel。

## 3. 涉及的源码

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/environ.py` | 增加功能开关 `SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA` |
| `python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py` | 实现 paged FP8 KV 的 Triton gather、反量化和 compact index 构造 |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | 分配 workspace，并在 P/D 的 FlashMLA 入口前接入 gather/upconvert |

## 4. 开关与推荐配置

功能默认关闭。当前在 CP8TP8 P 端和 TP8DP8 DSpark D 端验证过的推荐配置如下：

```bash
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=1
```

然后正常执行 `sglang serve ...`。

三个开关的含义：

- `SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=1`：启用本文档描述的 FP8 gather/upconvert
  和 BF16 FlashMLA 输入。
- `SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0`：使用统一 MLA 路径。普通 Prefill 会进入
  `_forward_flash_mla_decode` 风格的统一入口，因此 P 和 D 都可执行相同的 BF16 gather。
- `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0`：不通过该开关强制进入独立的
  `flash_mla_sparse_fwd` Prefill 路径。

注意：`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0` 只表示“不强制”。在非 CP 场景中，
源码仍可能根据 Query 长度阈值选择独立 sparse-prefill 路径。当前分支的 CP Prefill
还受 `not dsa_use_prefill_cp(forward_batch)` 限制，因此 CP8 不会进入该独立路径。

成功启用后，每个 attention backend/rank 启动时会打印：

```text
Enabled DSV4 Triton FP8 KV gather/upconvert for BF16 FlashMLA prefill/decode forwards
```

## 5. 支持范围与前向模式

| 场景 | 是否执行 BF16 gather | 说明 |
| --- | --- | --- |
| 普通 Prefill / Extend | 是 | 要求走统一 HCU MLA 路径；CP8TP8 P 端已验证 |
| Decode | 是 | D 端正常 Decode 路径已验证 |
| Target Verify / Draft Extend | 是 | 只要 logical forward mode 不是 IDLE |
| IDLE | 否 | 没有真实 Query，不分配和访问 gather workspace |
| KV cache 已经是 BF16 | 否 | gather 属于冗余操作，保留原 BF16 输入并打印提示 |
| 独立 `flash_mla_sparse_fwd` Prefill | 否 | 本次改动没有修改该 kernel 或其索引布局 |
| 非 HCU 平台 | 不支持 | 启用开关时直接报错，避免静默走到未验证路径 |

这里的“Prefill 支持”特指统一路径：

```text
_forward_flash_mla_prefill
    -> SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
    -> _forward_flash_mla_decode
    -> gather/upconvert
    -> flash_mla_with_kvcache
```

## 6. Packed KV 数据布局

DSV4 每个 token 的 K 数据为：

- `448` 个 FP8 NOPE 元素。
- `64` 个 BF16 ROPE 元素，占 `128` 字节。
- `7` 个 ue8m0 scale，每个 scale 对应 `64` 个 FP8 NOPE 元素。
- scale 区按 token 补齐到 `8` 字节。

因此：

```text
NOPE + ROPE data = 448 + 64 * 2 = 576 bytes/token
scale area       = 7 bytes, padded to 8 bytes/token
BF16 output      = (448 + 64) * 2 = 1024 bytes/token
```

页内布局不是简单的每 token 584 字节交错排列，而是：

```text
[page 内所有 token 的 576-byte NOPE+ROPE]
[page 内所有 token 的 8-byte scale]
[可选 page padding]
```

`gather_upconvert_k_cache_paged` 同时接受：

- 原始二维 allocation：`(num_pages, bytes_per_page_padded)`。
- FlashMLA 四维 view：`(num_pages, page_size, 1, 584)`。

四维 view 的逻辑 shape 不包含页尾 padding，通常不是 contiguous。实现必须保留源 tensor
的 `stride(0)` 作为真实 page stride；不能直接 `reshape()`，否则可能触发隐式拷贝并破坏
页地址计算。

## 7. Gather 与反量化算法

Triton grid 为 `(num_queries, topk)`，每个 program 处理一个 Query 的一个候选 KV 行：

1. 读取 `topk_lengths[query_id]`，屏蔽无效的 padding 列。
2. 将物理 token location 换算成 `page_idx` 和 `in_page`。
3. 根据真实 page stride 定位 576-byte 数据区和 8-byte scale 区。
4. 对 7 个 64-element NOPE tile 执行：

   ```text
   bf16_nope = fp8_nope * exp2(scale_u8 - 127)
   ```

5. 直接复制原本已经是 BF16 的 64 维 ROPE。
6. 写入连续 BF16 workspace，并生成新的 compact index：

   ```text
   compact_loc = query_id * output_topk + output_col
   ```

所有参与目标地址计算的 Query ID、Top-K ID、stride 和中间 offset 均提升到 int64。
这是 128K/CP-local 大批量 Prefill 的正确性要求：组合 SWA+C128 workspace 的
`query_id * output_stride` 可能超过 32 位有符号整数范围。使用 int32 会发生地址回绕，
最终表现为 HSA VMFault。

## 8. SWA 与 compressed cache 合并

当前安装的 HCU BF16 sparse-decode FlashMLA kernel 不支持将 `extra_k_cache` 作为独立
参数传入。因此，接入层会将 SWA cache 和 compressed extra cache gather 到同一个
连续 workspace：

```text
query i: [valid SWA rows][valid compressed rows][unused padding]
```

第二段的输出 offset 使用 `swa_topk_lengths[i]`，而不是 SWA tensor 的 padded width。
这样可以保证两个有效前缀首尾相接，不会在注意力集合中插入无效空洞。合并后的有效长度为：

```text
combined_topk_lengths = swa_topk_lengths + extra_topk_lengths
```

完成合并后，独立的 `extra_k_cache`、`extra_indices` 和 `extra_topk_lengths` 会置空，
FlashMLA 只接收一组语义等价的 BF16 KV、compact indices 和 combined lengths。

## 9. Workspace 与显存估算

workspace 按 `(slot, topk)` 缓存，每一组包含：

- BF16 KV：`(capacity, topk, 1, 512)`。
- compact indices：`(capacity, 1, topk)`，dtype 为 int32。

近似显存占用：

```text
BF16 KV bytes = capacity * topk * 512 * 2
index bytes   = capacity * topk * 4
total bytes   = capacity * topk * 1028
```

例如 `capacity=4096, topk=4096` 时，单个 workspace 约为 `16.06 GiB`。因此 128K
场景虽然 KV cache 本体仍是 FP8，临时 gather workspace 仍可能很大。部署时必须为它
预留显存，并结合并发数、每批 Query 数、SWA window、compressed Top-K 和 CUDA Graph
bucket 一起评估。

CUDA Graph 初始化阶段会预分配已知的 combined Top-K 规格。若 graph capture 过程中才
发现 workspace 容量不足，代码会直接报错，不允许在 capture 内动态扩容。

## 10. 为什么不切分 FlashMLA

此前遇到的 128K VMFault 来自 gather kernel 的 32 位目标地址溢出，不是 FlashMLA
单次处理 32768 chunk 的接口限制。当前修复通过 int64 指针运算解决 gather 地址问题，
仍保持每层一次完整的 FlashMLA 调用。

因此不要为了规避旧 VMFault 而把一次 FlashMLA attention 拆成多个调用。注意力的
softmax 归一化跨越全部有效 KV；任意切分都需要正确合并局部 max、sum 和 output，简单
相加会改变结果，并且也会增加 kernel launch 和中间同步开销。

## 11. 启动示例

下面只展示与本功能直接相关的部分，其余 TP/CP/EP、PD 分离和 Mooncake 参数沿用部署
脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=1

sglang serve \
  --model-path /models \
  --tp 8 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --chunked-prefill-size 32768 \
  --kv-cache-dtype fp8_e4m3 \
  ...
```

P、D 是独立进程，环境变量不会从一端自动传到另一端。如果希望 P 和 D 都使用 BF16
FlashMLA 输入，需要分别写入 P、D 启动脚本，并重启两端使其生效。

## 12. 正确性验证

推荐按以下顺序验证，避免直接用长评测掩盖路径或容量问题：

1. 静态语法检查：

   ```bash
   python -m py_compile \
     python/sglang/srt/environ.py \
     python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py \
     python/sglang/srt/layers/attention/deepseek_v4_backend.py
   ```

2. 启动 P、D 和 Router，确认每个目标 rank 都出现启用日志。
3. 发送固定温度、短输入的简单对话，与关闭开关的结果做基本对比。
4. 发送 128K 输入、单并发请求，检查无 OOM、VMFault、NaN 或超时。
5. 使用 EvalScope 回归 `GSM8K`、`HumanEval` 和 `MATH-500`。正式比较时必须保持模型、
   tokenizer、chat template、采样参数和数据集版本一致。
6. 分别抓 P 和 D 的 GPU trace，确认真实执行路径。

本功能改变了 FlashMLA 的 KV 输入精度路径。短对话成功只能证明服务可运行，不能代替
数据集精度回归。

## 13. Profiler 验证

trace 中至少应找到以下两类事件：

```text
_gather_upconvert_k_cache_paged_kernel
gfx93::fwd::sparse_attn_fwd_kernel<...512...>
```

判断标准：

- 只有 gather：说明后续 FlashMLA 没有正常完成，不能算验证通过。
- 只有 FlashMLA：可能开关未生效、KV 已经是 BF16，或走了其他输入路径。
- P 和 D 都需要单独抓 trace；只验证 D 不能证明 P 的 Prefill 已启用。

2026-08-22 的 5-step GPU trace 记录位于：

```text
/public/home/xdb4_10676/flash0731/w4a8/prof/BF16_FlashMLA_5steps_20260822_202941
```

该次 trace 的事件计数为：

| 端 | 每 rank gather | 每 rank FlashMLA | 折算每 step |
| --- | ---: | ---: | ---: |
| P Prefill 128K | 420 | 215 | gather 84，FlashMLA 43 |
| D Decode DP8 | 435 | 230 | gather 87，FlashMLA 46 |

该次 trace 中 FlashMLA kernel 自身的统计为：

| 端 | 单次平均 | P50 | P90 | 每 step 累计/每 rank |
| --- | ---: | ---: | ---: | ---: |
| P Prefill 128K | 4.862 ms | 5.060 ms | 6.120 ms | 209.07 ms |
| D Decode | 37.35 us | 40.16 us | 40.96 us | 1.718 ms |

这些数字只代表当次模型、拓扑、输入和 5-step GPU-only trace，不是通用性能承诺；它们
不包含 gather、indexer、PD 传输、调度、网络和 CPU 开销，也不能直接等同于端到端延迟。

## 14. 常见问题

### 14.1 启动日志中没有 enable 信息

检查：

- 变量是否写成 `SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=1`。
- 变量是否在 `sglang serve` 进程启动前 export。
- P、D 是否分别设置并在修改后重启。
- 当前进程是否确实加载了这份源码，而不是另一套 site-packages。

### 14.2 Profiler 中没有 gather kernel

可能原因：

- 功能开关关闭。
- KV cache 本来就是 BF16，代码无需反量化。
- 当前是 IDLE forward。
- Prefill 进入了独立 `flash_mla_sparse_fwd`，没有经过统一 decode-style 入口。

### 14.3 128K Prefill 出现 HSA VMFault

确认使用的是当前 int64 pointer-arithmetic 版本，不要恢复为 int32 program ID/stride
乘法。还需要检查 page size、真实 `stride(0)`、workspace shape 与 combined Top-K 是否
匹配。不要仅通过把 FlashMLA 拆成多次调用来绕过 gather 地址问题。

### 14.4 出现 OOM

先根据第 9 节公式估算 workspace。优先降低同一时刻的 Prefill Query/请求并发，或减少
最大 running requests，并给临时 workspace 留出显存余量。`max_total_tokens` 主要控制
KV pool，本功能还会额外分配 BF16 workspace，因此只看 KV cache 可容纳 token 数并不足够。

### 14.5 报 `topk_length must have shape (b)`

当前 HCU BF16 sparse-decode ABI 要求 Query batch、indices 和 length 的第一维语义一致。
统一 Prefill 路径会按 token/query 行组织输入并复用 decode-style FlashMLA；若重新修改
shape 或 flatten 逻辑，需要同时检查 `q`、indices、lengths 和 workspace 的第一维。

### 14.6 精度异常

按以下层次定位：

1. 用 `dequantize_k_cache_paged_ref` 对比 Triton 反量化输出。
2. 检查 ue8m0 scale 是否使用 `exp2(scale_u8 - 127)`。
3. 检查 ROPE 是否按 BF16 原样复制，而不是作为 FP8 反量化。
4. 检查 `topk_lengths`、padding mask 和 compact indices。
5. 检查 SWA 与 extra cache 是否按每个 Query 的有效 SWA 长度拼接。
6. 对比开关关闭/打开时的固定输入 logits 或评测集分数。

## 15. 回滚

无需迁移 KV cache，也无需修改模型权重。关闭开关并重启 P/D 即可恢复原路径：

```bash
export SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=0
```

也可以从启动脚本中删除该变量。`SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA` 和
`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` 是否恢复原值，应根据原部署路径决定；它们不是
本功能新增的变量。

## 16. 已知限制

- 仅支持并验证 HCU/DCU 后端。
- Triton gather 当前优先保证数据布局和 128K 地址正确性，尚未针对所有 shape 做极致
  性能优化。
- BF16 workspace 可能显著增加峰值显存，尤其是大 Query 数、大 Top-K 和 CUDA Graph
  多 bucket 场景。
- 当前实现依赖已安装 HCU FlashMLA 的 BF16 sparse-decode 输入 ABI。
- 当前实现不移植或放开 CP-v2 独立 sparse-prefill 的索引重排能力。
- P/D、CP/TP、DSpark、Prefix Cache 和 Mooncake 的其他兼容约束仍由现有 SGLang 逻辑
  决定，本开关不会绕过这些约束。
- 上述 profiler 数据不能代替端到端吞吐、时延和完整精度测试。
