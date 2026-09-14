# 官方同步总结 · 20260914

| 项 | 值 |
|---|---|
| 上次同步点 | `97dcbf9410c1df1b7a4d341c4ed021f06f8d5087` |
| 本次官方 tip | `0d95a9c1ff14773b722009a6c6fd6ad66c7ce395`（2026-09-14） |
| 区间 commit 数 | 332 |
| 变更规模 | 1702 files, +115443 / −43998 |
| 分支 | `sync/official-main-daily-20260914`（基于 main `b5108cc7fe`） |
| 冲突 | 60 文件 / 72 处（55 content、3 file location、2 modify/delete） |

## 0. 同步点确认

20260907 那次同步落 main 依旧是 **squash merge** `ee0f2375e5 (#328)`，单 parent，
`97dcbf9410` 因此不是 main 的祖先，`merge-base(main, officials/main)` 退回到 `92b1d382c7`（8/17），
直接合并要重放 **1502** 个 commit。

`#328` 的 tree 与 squash 前的同步分支 tip `d642fbb8f5` **逐字节相同**，而该分支里的合并
`06316d9b76` 正是以 `97dcbf9410` 为第二 parent —— 内容确实已在 main。于是在新分支上
`git merge -s ours 97dcbf9410` 锚定（commit `c1b59527c9`，tree 不变），重放量降到 **332**。
这是连续第四次 squash 陷阱，范式见 `fp-base-is-second-parent`。

---

## 1. 组件优化点

### 1.1 配置系统：msgspec 化 + 退役全局访问器（影响最大）

- `[Config] msgspec.Struct for the config tier` (#38753)：`arg_groups/fields/*` 的命名空间类从
  `@dataclasses.dataclass` 整体换成 `msgspec.Struct`。
- `[Config] Retire get_global_server_args` (#38375)：`sglang.srt.server_args.get_global_server_args`
  保留名字但**调用即 `RuntimeError`**，理由是它返回的是操作员原始输入，读"解析后决定的值"会静默拿到陈旧值。
  取值改走命名空间 bag：`get_exec().kernel.*`、`get_model().*`、`get_disagg().*`、`get_parallel().*` 等。
- `[Config] One writer for the declaration stash; no exception to the write seal` (#38752)。
- `config: an out-of-tree replacement point for every resolution-pipeline step` (#39134)。

### 1.2 调度 / 内存

- `[Scheduler] Unify per-iteration request intake into ingest_requests()` (#38389)：接收 + 广播 + 派发收拢成一个入口。
- `[mem_cache] Free hybrid SWA pages by one representative per page on page_size > 1` (#38159)：
  `free_swa` 按"每页一个代表 token"释放，full/SWA 两池同步分页并在 debug 模式断言。
- `[AMD][DSV4] Reland unified-KV pool sizing and SWA ring accounting, fully gated` (#38192)：per-request ring，
  JIT plan builder 新增 `use_req_ring` 参数。
- `[unified-memory] PD disaggregation for every unified pool shape` (#37506)。
- `[Unified Tree] Port SWA Branching-Point Caching to the Rust TreeCore` (#37584)。

### 1.3 HiCache / PD / Session

- HiCache：storage 清理时释放 buffer 预取锚锁 (#38483)、storage-prefetch 回填发布 host store 事件 (#38486)、
  file backend 临时文件名限制在 NAME_MAX 内 (#38925)、`@rank_consensus` (#37425)。
- PD：HTTP PD router 支持 `/v1/responses` (#36141)；DCP1→DCP-N 重排时传 DCP 复制的 DSPARK draft KV (#37709)。
- Session：PD 协同与空续写修复 (#39038)、拒绝请求后的 idle 超时修复 (#39035)。

### 1.4 内核 / 量化

- **`Delete cutlass_mla, non-Marlin GPTQ, AWQ AOT kernel, and Dual Chunk Flash Attention` (#32114)**：
  删 AOT 内核与对应 backend；AWQ 在 HIP 上改走 Triton。
- `[MoE][ROCm] Admit the unified Triton router on ROCm, including single-group routing` (#38328)。
- `[AMD] Enable Fast Triton Sparse MLA backend` (#30575)；`[AMD] Fix weight checking for AITER-shuffled block FP8 weights` (#34330)
  （引入 `is_shuffled` 标记与 `unshuffle_fp8_weight`）。
- `[ROCm] Raise HiCache JIT block quota for mapped-host throughput` (#39036)。

### 1.5 测试体系

- `[Test] Consolidate test cleanup and CI taxonomy (net -11.4K lines)` (#37436)：`test/registered/eval/` →
  `accuracy/models/`、`4-gpu-models/` → `e2e/models_large/` 等目录重整。

---

## 2. 模型优化点

### 2.1 DeepSeek-V4 / V4.1（与我们最相关）

- `[Refactor] Clarify DeepSeek V4 metadata names for V4.1` (#38947)：`c4_seq_lens` → `compressed_seq_lens` 等改名。
- `[Refactor] Generalize DeepSeek V4 compressed pool management` (#38954)。
- `[DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103` (#30805)：新增 `trtllm_attn` 路径与 uniform-FP8 KV 布局。
- `[DSV4] Support raw-index output in TopK v2` (#33672)。
- `Fix DeepSeek-V4 routing: sqrtsoftplus underflow and unfloored renorm` (#34459)。
- `[DeepSeek-V4.1] Bump FlashMLA to the fork's rebase head (v4.1 kernels)` (#39171)。
- `[SM120] Use exact query-head widths for DeepSeek-V4 sparse MLA decode` (#36655)。
- AMD：DSpark accept length 修复并减少 host bubble (#39116)、DSV4 dspark 配置 (#39252)、gfx94x block-FP8 加载修复 (#38446)。
- NPU：DSV4 host 内存缓存管理 (#37382)。

### 2.2 其它模型

- **GLM**：GLM-5.3 Flash 恢复并启用 KPool metadata 融合 (#38845)；GLM-5.3 chat 模板自动识别 (#38297)；
  GLM-5.2 MXFP4 在 MI355X 改走 Triton DSA backend (#39106)。
- **Qwen**：Qwen3.8-Next PD 状态传输 (#36651)；Qwen3-VL unique-image serving on H100 优化 (#36411)；
  Qwen3.8 在 DGX Spark 上启用 NVFP4 (#39126)。
- **MiniMax-M3**：Quark MXFP4 `index_qkv_proj` 加载修复 (#37254)。
- **MiMo-V2.5-Pro**：NPU 上 DFlash 投机解码 (#37565)。
- **Gemma-4**：AMD aiter attention backend (#38758)。
- **GraniteMoE**：按 expert 切分的量化 MoE 权重加载 (#37679)。
- **gpt-oss**：RunAI streamer 权重归属修复 (#38908)。

---

## 3. 冲突处理要点

### 3.1 官方修得比我们好，采纳官方

| 位置 | 说明 |
|---|---|
| `mem_cache/allocator/swa.py` | 上两轮都保留了我方"更严格"的 `free_swa`。官方 #38159 改成"两池同步分页 + 按整页展开清映射 + debug 断言"，我方 block 当初防的两个问题（共享页重复释放、遗留陈旧映射）在该不变量下都不再成立；而且新的前导让我方 block 里的 `mapping_indices` 根本没被赋值，取 ours 会 NameError。重复释放由我方 `paged.free()`（unique + `get_all_free_pages()` 过滤）继续兜底。 |
| `layers/attention/dsv4/indexer.py` | 官方把 `c4_sparse_raw_indices` 检查挪到链首；两个 metadata 类都把它声明为 `default=None` 字段，我方末尾的防御式 `getattr` 分支成了死代码。 |
| `kernels/ops/attention/dsv4/compress.py` | 官方给 JIT plan builder 加了 `use_req_ring` 布尔参数，C++ 侧已自动合入，Python 调用必须匹配新签名。 |
| `xpu_backend.py` | 只在 XPU 专用入口里惰性导入，HCU 永不加载；我方当初改 import 属于机械替换，取官方。 |
| `dual_chunk_flashattention_backend.py`、`test_update_weights_from_disk.py` | 官方删除，我方只有版权头 / disabled 的 HCU 注册，接受删除。 |

### 3.2 保留我方（官方版本在 HCU 上不对）

| 位置 | 理由 |
|---|---|
| `kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` | 我方 fork 在 ROCm 上把 `fp8x2_e4m3_t` 定义为 HIP 原生 `__hip_fp8x2_e4m3`（`sgl_kernel/utils.cuh`），官方是 `uint16_t` 别名。官方新的软件 cast 返回 `uint16_t`，在我方类型模型下类型不匹配；硬件转换分支依赖 DCU 没有的指令。保留我方原生类型构造，加注释说明。 |
| `arg_groups/fields/*.py` 的 `from typing import Annotated as A` | 上轮 Quality Gate 修复有意为之：ruff F722 看不穿 `arg_utils.A = Annotated` 的跨模块再导出，会报 352 个语法错。只丢 `import dataclasses`（官方已转 msgspec）。 |
| `layers/attention/deepseek_v4_backend.py` | 官方把 `q.unsqueeze(1)` 升维提前到 assert 之前；HCU 的 attention kernel 要 3D q / 2D indices。升维块加 `if not _is_hcu:`，并删除我方后面那段已重复的升维——非 HCU 顺序与官方一致，HCU 不变。 |
| `models/deepseek_v2.py` shared experts FP8 检查 | DSV4 传进来的 `_DeepseekV4ConfigAlias` 没有 `quantization_config` 属性，官方直接 `config.quantization_config.get(...)` 会 AttributeError（DSV4 模型初始化必经）。保留我方防御式检测，吸收官方 `and not _is_npu`。 |
| `arg_groups/overrides.py` DSA backend 默认值 | 保留 HCU 分支原样（包括"只设了一个时继续落到后面 fp8 默认逻辑"的细节）；官方新的 HIP `triton` 默认值加 `not is_hcu()` 只给 AMD。 |

### 3.3 需要拆开 / 重排的冲突

- **cutlass_mla 与 decode_metadata 捆在同一块**（CMakeLists、`common_extension.cc`、`sgl_kernel_ops.h`、
  `sgl_kernel/__init__.py`、`attention.py`）：cutlass_mla 跟随官方删除；`normal_decode_metadata_general` /
  `decode_metadata.cu` 是我方独有的 HCU 内核（`setup_hip.py` 编译、`common_extension_rocm.cc` 注册），保留。
  **注意：该内核在所有分支都没有 Python 调用方**，是后续清理候选。
- **`layers/quantization/fp8.py` Fp8MoE 权重后处理链**：官方链是 `fnuz(+aiter shuffle 16×16) → aiter 预 shuffle → cpu → deepgemm`；
  我方把 `elif aiter` 改成独立 `if aiter or ASM: asm_shuffle_weight_b8`，让 cpu/deepgemm 改挂到新 `if` 上，
  且没有 `_is_hcu` 保护（AMD 行为也被改了）。重排为：HCU 走 ASM b8 分支，非 HCU fnuz 在 `pass` 处停链，
  非 fnuz 的 aiter 走官方预 shuffle —— 两个平台的 `else: deepgemm` 可达性都与各自原版一致。
- **`token_dispatcher/deepep.py`**：保留 HCU `use_groupgemm` 派发，`else` 分支换成官方（局部 `use_fp8` + A5 MXFP8 `low_latency_quant_kwargs`）。
- **`scheduler_pp_mixin.py`**：官方 #38389 把接收与处理收拢进 `ingest_requests()`，丢了我方 PP 下
  `ExpertDistributionReq` 预转发（dump 是 world-group collective，非末级 stage 必须先转发再本地处理，否则死锁）。
  给 `ingest_requests` 加默认 `None` 的 `before_process` 钩子，PP 循环传入 `_pp_preforward_expert_distribution_req`。
- **`multimodal_gen/runtime/loader/fsdp_load.py`**：保留我方 streaming state-dict 分支结构，`else` 换成官方重做的全量加载
  （外部 `weights_iterator`、`host_spill`、`allow_device_tensor_assignment`），流式分支同时尊重新增的外部 iterator。
- **`mem_cache/deepseek_v4_memory_pool.py`**：两侧函数体相同、触发条件不同，取并集 `is_bf16_attention_kv_cache or uniform_fp8`。

---

## 4. 静态门

| 门 | 结果 |
|---|---|
| `compileall`（python/sglang + test） | 通过，0 错 |
| `environ.py` 全文件符号 diff（按 revision） | 通过：base 600 / ours 659 / theirs 620 / merged 679 = 精确并集，两侧零丢失 |
| `envs.SGLANG_*` 引用对拍 | 通过：merged 88 / ours 108 / theirs 100，无新增悬空引用 |
| ruff F821/F811/F401/**F722**（三方，行号与 "from line N" 归一化） | 通过：merged 226 / ours 255 / theirs 228，合并新引入 0 |
| 跨模块 import（AST，三方） | 通过：合并新引入 0 |
| **预处理器结构门（新增）** | 通过：407 个 C 系文件，合并引入 0 |
| **我方行悬空属性门（新增，启发式）** | 通过：0 |

### 新增的两道门抓的是什么

1. **预处理器结构门**：`fp8_utils.cuh` 的自动合并在**没有任何冲突标记**的区域把两侧的 `#ifdef` 链拼成了
   `#ifdef USE_ROCM … #else … #elif …`（`#elif` 在 `#else` 之后，非法）。JIT 内核只在设备上首次编译时才报错，
   即运行期。新门扫描所有 C 系文件的指令栈并与两个 parent 对比。
2. **悬空属性门**：提取我方独有代码行里 sglang 自有对象（`self.` / `forward_batch.` / `metadata.` 等）上的属性读，
   检查合并树中是否仍有定义——专抓"官方改名/删字段、我方仍按旧名访问"。已用合成用例验证有效。

---

## 5. 静态门看不见的运行期断裂：`get_global_server_args()` 调用即抛

官方 #38375 保留了 `def get_global_server_args() -> NoReturn`（名字仍绑定，import、编译、ruff、import 门全部通过），
但**调用即 `RuntimeError`**。我方有 8 个文件、16 处调用，全部是我方独有代码：

| 文件 | 读的字段 | 迁移到 |
|---|---|---|
| `compressed_tensors_w8a8_fp8_moe.py` | `disaggregation_mode`、`deepep_mode` | `get_disagg()`、`get_exec().moe` —— **DSV4-Flash FP8-Channel 的 MoE 方案** |
| `layers/moe/utils.py` | `disaggregation_mode` | `get_disagg()` |
| `layers/moe/mega_moe.py` | `disaggregation_mode` | `get_disagg()` |
| `token_dispatcher/deepep.py` | `quantization` | `get_model()` |
| `pack_paged_kv_to_varlen.py` | `pack_paged_kv_to_varlen*`、`minimax_opt` | `get_exec().kernel`、`get_parallel()` |
| `flashattention_interface.py` | `quantization`、`kv_cache_dtype` | `get_model()` |
| `models/hunyuan_v3.py`、`hunyuan_v3_nextn.py` | `kv_cache_dtype`、`ep_num_redundant_experts`、`enable_dp_lm_head` | `get_model()`、`get_exec().moe`、`get_parallel()` |

**比崩溃更危险的两处**：`moe/utils.py` 把调用包在 `try/except Exception: pass` 里 —— `RuntimeError` 被吞掉，
**PD 模式下静默返回 `"ifb"`**；`mega_moe.py` 包在 `except ValueError` 里接不住，直接崩。
新 bag 在配置未发布时抛 `ValueError`，正好对上这两处 try/except 原本的防护意图。

**不是合并回归的对照**：同一批调用在我方 parent 上可以正常工作，是本区间 #38375 改变了被调函数的行为。

---

## 6. 既有技术债（main 上本来就有，本次不处理）

用上游 ratchet 单测（`test/registered/unit/*_ratchet.py`，`base-a-test-cpu`）核对，
**在我方 parent `b5108cc7fe` 上失败条目与失败数完全一致**，合并未引入新条目：

- `test_global_config_read_ratchet`：我方 31 处 `get_server_args().field` 直读（官方 0）——
  `communicator.py`、`bailing_moe.py`、`minimax_m2.py`（8 处 `minimax_opt`）、`qwen2.py` / `qwen3.py` / `qwen3_moe.py`
  （`kv_cache_dtype`）、`fused_moe.py`、`loader.py`。运行期不崩，但读的是原始输入而非解析后生效值。
- `test_chain_read_ratchet`：`hcu_mla_backend.py:133/136` 读 `model_runner.server_args.page_size`、
  `:158` 读 `.speculative_num_draft_tokens` —— `page_size` 是解析阶段决定的值（HCU 默认 64），
  **待确认是否已影响 `hcu_mla` backend 的实际行为**。
- `test_supplied_instance_exposure_ratchet`：`scheduler.py` 的 `dp_size` / `enable_dp_attention`、
  `hiradix_cache.py` 的 `hicache_mem_layout`。

建议单独开一个 PR 迁移到命名空间 bag，并补 `hcu_mla` 模型（DeepSeek-V2/V3）的验证。

---

## 7. 未对齐 / 主动不跟进

| 项 | 状态 |
|---|---|
| `pr-test-rust.yml`、`pr-test-amd-rocm720.yml`、`nightly-amd-mi355x-disagg.yml` | 保留我方空触发器（本 fork 跑不起来） |
| `pr-states.yml` 新增的 "AMD ROCm 10" 状态行 | 我方已有意移除 AMD 状态（`3ca61517d0`），官方新行会引用我方已删的变量 |
| `decode_metadata.cu` / `normal_decode_metadata_general` | 保留但无调用方，后续清理候选 |
| `test/registered/e2e/models_large/test_qwen3_next_models.py`（我方旧拷贝） | 删除；`register_hcu_ci` 占位嫁接到官方演进后的 `e2e/models/` 版本 |
| XPU / NPU / SM1xx / gfx95x / trtllm 相关路径、multimodal_gen | 照收，未在 DCU 上验证 |

---

## 8. 验证（zz-nmz26 / rye_sglang_latest，2026-09-14）

sgl-kernel 已按本次合并重编（sglang-kernel 0.4.7，gfx906/926/928/936/938）——官方删了 AWQ/GPTQ AOT 内核、
`dsv4_norm_rope.cu` 有改动，wheel 的 Python 侧也变了。我方 HCU 独有算子 `normal_decode_metadata_general` 编译并导出正常。

`bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel`
（纯 TP8，`mem_fraction_static=0.8`，`max_total_num_tokens=2870016`）：

| 检查项 | 结果 |
|---|---|
| 服务启动 | 正常 |
| Greedy sanity | `The capital of France is **Paris**.` |
| **GSM8K 100 题** | **0.99**（avg 10.9 s / 11.08 tok/s 每请求） |
| 聚合 decode 吞吐峰值 | 802.16 tok/s（上轮 803.26） |
| Traceback / VMFault / 非法访存 / scheduler 异常 | 0 / 0 / 0 / 0 |
| `multimem all-gather disabled` | 8 条，良性 RCCL 回退，与前几轮一致 |

### 0.99 的归因

唯一错题是 GSM8K index 12「柠檬树」：算术完全正确（7×1.5−3=7.5，90/7.5=12），但把"回本"当成"开始赚钱"
（第 12 年末回本，第 13 年起盈利）。这是公认的语义边界题，**不是数值损坏、格式或答案抽取问题**。

与 9/8（1.000）逐题对比：**提示词 100/100 相同**（评测客户端版本差异不是原因），
**完整生成仅 19/100 逐字相同，抽取答案 99/100 相同**——模型数值有微小漂移，贪心解码在某个 token 分叉，
结论基本不变；9/8 在同一题上多推了一步得到 13。

漂移来源已定位：区间内 `Fix DeepSeek-V4 routing: sqrtsoftplus underflow and unfloored renorm` (#34459) 改的正是
HCU 上 DSV4 Hash-MoE 路由用的 `hash_topk.cuh` / `moe_fused_gate.cuh` / `hash_topk.py`——`sqrt(softplus(x))`
改写为数值稳定形式 `max(x,0)+log1p(exp(-|x|))`、renorm 分母加 `1e-20`。数学近似等价，float32 末位舍入不同。

0.99 落在历史纯 TP 区间（0.95~1.00）内，判定精度 OK。

**评测环境备注**：`rye_sglang_latest` 在 9/8 重启后 evalscope 丢失（wheel 目录里只有 1.5.1，与此前 1.11.1 不同代），
为不改动验证容器，评测客户端改由同机 `rye_sglang_open`（evalscope 1.10.0，host 网络）发起，服务仍在 `rye_sglang_latest`。
