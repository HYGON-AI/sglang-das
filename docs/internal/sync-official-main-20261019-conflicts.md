# 2026-10-19 官方 main 同步冲突与验证台账

## 冻结基线

- 内部 main：`f89209aa84a4fd0619d0662f6adf6980740efc78`。
- 上次官方：`438df9a2a4645d39c87e33c4a27792e568ac2701`，由历史总结及旧同步分支确认。
- squash PR #500 (`36b93840b54b8c41fe395e6f3e3f9a5463fe17ba`) 丢失上游 ancestry；本次先做 tree 不变的 ancestry-only merge。Squash 与旧同步末端并非相同 tree，不覆盖其后内部修改。
- 本次官方冻结目标：`a90e9c5dd8973cdb84af4ff6154c7b4958083b31`，722 笔增量。
- 全量试合并出现 89 个冲突，已 abort；拆为约 180 笔一批。

## 第 1 批：6576e50cd5e5817b10e12245104603eb1b0d8a4f

| 冲突文件 | 处理 |
|---|---|
| `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py` | 采用上游 _make_block 工厂；保留 SLA top-k/local blocks/skip-linear 参数解析。 |
| `python/sglang/srt/arg_groups/field_order.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `python/sglang/srt/arg_groups/overrides.py` | 采用上游 CUDA TileLang FP8 KV/平台/DCP 一致性约束。 |
| `python/sglang/srt/configs/glm5_next.py` | 保留注册、KDA 形状/dtype；非 HCU 使用 linear_attn_tp_size，HCU 保留 CP 分片。 |
| `python/sglang/srt/disaggregation/decode.py` | 保留 PD hidden 完成门禁与计时；跟进 host-receive 阈值 1/<1 语义及 HiCache poll 状态。 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `python/sglang/srt/entrypoints/openai/serving_responses.py` | 默认生成长度交由 scheduler 按多模态扩展后的真实长度约束。 |
| `python/sglang/srt/layers/moe/mega_moe.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `python/sglang/srt/layers/quantization/fp8_utils.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `python/sglang/srt/managers/multi_tokenizer_mixin.py` | 接上游批量分组，使用内部 RID 固定 worker 与完成 ACK 的负载管理。 |
| `python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | 非 HCU 采用上游 PPConsensusStore；HCU 保留两轮 bootstrap 资源/失败共识，适配 prefill 队列入口。 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | 同时保留 MLA 维度 override 与新 page-interleave pool kwargs。 |
| `python/sglang/srt/models/kimi_k3.py` | 采用统一 dense MLP group helper，保留旧环境变量别名（NPU 兼容，不误标 HCU 专属）。 |
| `python/sglang/srt/models/minimax_m2.py` | 保留 minimax_opt 条件，移除已退役的 neighbour-layer 构建标志。 |
| `python/sglang/srt/models/qwen4_exp.py` | 保留 PLE INT8/FP8 channelwise；使用新的 config.ple_offload_embedding 属性。 |
| `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | 增加上游 PP load group，同时保留 _is_hcu context-only rank 的 PP placement。 |
| `python/sglang/srt/utils/common.py` | 遵循 gathered_buffer_required 显式参数；HCU minimax/hy3 padding 以 _is_hcu 保护。 |
| `test/registered/amd/test_smallm_fp8_proj_gfx950.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |
| `test/registered/unit/test_runtime_context.py` | 遵循上游新接口/导入/测试结构，保留仍适用的内部字段；详见三方快照及 resolution-ledger.json。 |

### 自动合并语义审查

- `prefill.py`：上游 pop_bootstrapped 移除 PP RID 参数，而内部 PP hidden 资源共识仍依赖它。以 `_is_hcu` 分流，保留 HCU PP 旧资源提交流程，其他平台使用官方流程。
- 路由测试 fixture 改为明确 RID ownership；增加真实 least-load、流式固定和 completion 释放的组合回归。
- 全量 Python compileall 通过；861 个变更 Python 文件 AST 通过；Ruff F821/F822/F823/F811/F722 相对两侧无新增诊断。内部原有 HiRadixCache 类型标注缺失两处，记为基线债务。
- 静态直接调用关键字审计相对两侧无新增问题（不代替运行验证）。
- 上游新增 Hunyuan 文件采用 CRLF；以 `core.whitespace=cr-at-eol` 检查通过，保留原始换行。
- 路由/增量 detokenizer：17 passed；PP deferred-bootstrap/consensus：32 passed、11 subtests passed。DSpark PP context 测试在收集阶段被提交容器旧 LightOp 缺少 fuse_situ_mul_quant_contiguous 阻断，留待验证容器复测。

## 运行验收（合并阶段历史状态，已由最终验收更新）

整理完成后，用户已授权停止 `zz-nmz28 | rye_sglang_dpsk-v41` 的 prefill 进程用于验证。只停止经核对属于该服务的进程。
目标保持 DeepSeek-V4-Flash-0731-FP8-Channel、纯 TP8、端口 10015。需服务就绪、graph capture 与至少 3 次 replay、完整 EvalScope HumanEval；尚无本次精度结论。

证据目录：`/home/proj_sglang_open/sync-evidence/20261019/`。

## 第 2 批：ac187c0cd54d9cdf2772584495afe67c82293a2e

| 冲突文件 | 处理 |
|---|---|
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 合并 state type/elided entry；Mooncake 先 poison unsettled send，再内部跨 rank failure 同步。 |
| `python/sglang/srt/disaggregation/nixl/conn.py` | 合并 state type/elided entry；Mooncake 先 poison unsettled send，再内部跨 rank failure 同步。 |
| `python/sglang/srt/disaggregation/utils.py` | PD state append/SWA layout 并存；DSA effective mode 与 MQA budget helpers 并存。 |
| `python/sglang/srt/distributed/parallel_state.py` | 合入 AITER graph copy-in；保留 HCU 未注册 all-gather/注册能力判断。 |
| `python/sglang/srt/entrypoints/openai/serving_base.py` | 保留内部受控 tokenizer offload 与计时，覆盖上游 off-loop 意图。 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 接新 timestamp、V4.1 多模态编码/default effort、模板缓存、逐 token logprob/tool ordinal；保留 init_glm。 |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | 新版 sparse-prefill eligibility/V4.1 路径进入非 HCU 分支；HCU FlashMLA 分流保留。 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 使用真实 index_page_size；HCU 避开 gfx950 preshuffle。 |
| `python/sglang/srt/layers/attention/dsa/utils.py` | PD state append/SWA layout 并存；DSA effective mode 与 MQA budget helpers 并存。 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 新 HiSparse verify 限制使用 effective mode；修复 topk_transform_method 赋值顺序。 |
| `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py` | 使用上游按请求实际 compressed 长度的扁平 gather，去掉旧矩形 gather。 |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | 无 block-table 才走 HCU causal-conv；否则新官方接口；删除重复 kwargs。 |
| `python/sglang/srt/layers/attention/qsa/mqa.py` | HCU BoltOPs 仅用于 min_scored_len=0；TileLang 透传新阈值，Torch 完整计算。 |
| `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` | 使用 pack_qsa_prefill_kv；HCU sparse-GQA 快路径加 _is_hcu 保护。 |
| `python/sglang/srt/layers/cp/bcg.py` | 官方 enum 能力表；保留 HCU interleave DSV4/MegaMoE token 计数/模型 logits 合约。 |
| `python/sglang/srt/layers/moe/moe_runner/aiter.py` | 保留 HCU input 顺序及 runner；补 EPv2 output view 字段。 |
| `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` | 非 HCU 新 NVFP4 native/weight-only fallback；HCU 保留既有量化分派。 |
| `python/sglang/srt/layers/sampler.py` | 新 watermark/logprob/mask/max-top-k；LightOp 仅 HCU 且 seed/min-p/mask 条件匹配。 |
| `python/sglang/srt/managers/tokenizer_manager.py` | tokenize timing 与批量 request-state 安装并存。 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 迁移 build_mamba_host_pool buffer-info API，保留 HCU 布局映射，去除重复 SWA 分配。 |
| `python/sglang/srt/mem_cache/index_key_cache.py` | 内部 indexer layer mapping 外包新 kernel view，保留 write-buffer API。 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | HiSparse 仅 target；保留 GLM HCU pool/indexer layer，合入 dcp_replicated。 |
| `python/sglang/srt/mem_cache/memory_pool.py` | indexer-layer mapping/HCU page size 保留；新 DCP replication/preshuffle 非 HCU；修复 HCU page descriptor。 |
| `python/sglang/srt/mem_cache/pool_host/base.py` | 新显式几何 API；HCU uneven layer shard host 容量同步；兼容 buffer-only 无 device_pool。 |
| `python/sglang/srt/mem_cache/pool_host/mla.py` | device-buffer view 与 HCU ownership/warp 参数并存，修复 moved-method 局部变量。 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | HCU rms_quant_flag 与 full-prefill capture 字段并存。 |
| `python/sglang/srt/model_executor/pool_configurator.py` | HCU GLM/KDA/layer-split 计费与 HiSparse draft 地址空间比例并存。 |
| `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` | 保留含 capture token sizes 的日志。 |
| `python/sglang/srt/models/deepseek_v4.py` | CP metadata 放入新 language-input 准备入口；保留 PD aux 打包并裁剪真实 tail rows。 |
| `python/sglang/srt/models/deepseek_v4_dspark.py` | 保留 HCU 与 invariant imports。 |
| `python/sglang/srt/models/deepseek_v4_mhc.py` | 新平台 imports 与 HCU AITER/TileLang 分派并存。 |
| `python/sglang/srt/models/dspark.py` | 新 Markov API 结合 d2t reduced-vocab，保留 confidence features。 |
| `python/sglang/srt/server_args.py` | 内部 dataclass 构造结合 watermark 脱敏；repr 使用 dataclasses.fields。 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | NativeSparse 与 QwenSparse draft graph 支持集合合并。 |
| `test/registered/unit/layers/moe/test_aiter_runner.py` | 保留 top-k remap、新 EPv2/dispatch 测试；官方 runner 合约显式模拟非 HCU。 |

### 自动合并与测试

- HCU MHA host pool 构造迁移显式几何 API；host common 的 lru_cache/is_hip 拆分遗漏补回。
- 静态导入 4509 个 merged 模块相对两侧无新增；直接调用关键字相对两侧无新增。
- AITER：9 passed，显式模拟官方非 HCU runner；不据此宣称 HCU kernel 已验收。
- MiniMax host pool 原基线 8 failed/1 passed/1 skipped；修复 descriptor 后剩余 3 个 transfer 失败与基线相同，涉及测试 NHD 期望与 HCU 页式布局。
- Chat/API 初次 171 passed/125 subtests passed、5 failed；logprob/tool ordinal 已接官方修复，相关复测 9 passed/3 subtests passed；其余 3 个失败在原基线同样复现。
- 完整模型运行验收待所有增量整理后执行。

## Step 03：361–540（4afae5568bdc）

26 个冲突文件，保留官方接口重构并迁入内部行为：

- GLM5-Next 采用官方 text/vision 分层配置；保留旧模型类型及架构别名、层覆盖校验、HCU KDA 的 CP 分片、K-head 形状与 state dtype。别名类显式转发构造，避免 Transformers 自动 dataclass 初始化绕过父类配置转换。
- DeepSeek V4 合并 CUDA residual boundary、Mega Attention 与 FP8 einsum；HCU 继续独立的 MHC、PD 辅助隐藏态、CP 和 LightOp 路径，CUDA 专用 sparse/mega dispatch 保持 `_is_hcu` 保护。
- PD 合并新的 DSA page payload、state TP 分片语义和 bootstrap 判断；保留 HCU PP 双轮共识、PD hidden 分块传输及实验性 DSV4 decode radix 限制。非 HCU 使用官方 decode radix 校验。
- MultiDetokenizer 使用官方批量分组/单 tokenizer 空 IPC 支持，保留内部 least-load、RID 粘滞和完成 ACK；控制请求沿原 RID 路由但不增加负载，日志配置广播。
- DSpark 合并 SWA ring 重置，迁移内部 projected-context 注入到官方 ring 接口；修复自动合并遗留的已删除方法引用。
- MoE 保留 HCU DeepEP padding/remap，使用官方 remapping predicate；NVFP4 Marlin 新路径限定非 HCU；MiMo target-verify RoPE 融合限定非 HCU。
- JIT top-k 采用无 PTX 依赖的公共 radix 实现（保留官方溢出修正），公共 dynamic shared-memory helper 增加 HIP 分支；utils.cuh 保留 HIP launcher 并合入 CUDA driver error 支持。
- 自动合并审查另修复 MHC 常量改名和 lru_cache 导入遗漏。

验证：Python 3.10 全树 compileall 通过；相对双方父提交的新增 Ruff F821/F822/F823/F811/F722 为 0。路由 23 passed、2 subtests passed。KDA 策略单测 2 项因提交容器缺少 boltops 阻塞，非精度验证结果。GLM 配置 smoke 通过；直接调用参数审计无新增不匹配，见同批证据目录；完整 TP8/HumanEval 在最后一批完成后执行。

## Step 04：541–722（a90e9c5dd897）

37 个冲突文件。关键决策与自动合并修正：

- MHA/MLA host pool 采用官方 `DevicePoolInfo` / buffer 描述和 controller-layer 映射；保留 `_is_hcu` 下 page-major MHA host pool、GLM KPool/LayerSplit 所有权、indexer 预算及远程缓存失效处理。GLM host 构造显式转换为新版描述输入。
- 保留官方 `kv_layout` 字段，同时保留内部 MHA `kv_cache_layout`（两者语义不同）；PD hidden 分块传输与官方 REQUEST_ROW 状态合并。
- NIXL frame 23 保留官方 state buffer lengths，内部 state types 放在 frame 24；读取兼容旧内部只有 type 的 frame 23。prepared transfer 遇到组件顺序差异时转入按 type/occurrence 映射的通用路径。回归覆盖新组合、纯官方和旧内部三种 packet。
- HTTP 保留内部 warmup owner/result 协调与 spawn bootstrap，合入 post-warmup callback 和自定义 ASGI app path；状态码判断采用官方任意 4xx/5xx abort。
- Wan 合入官方 GELU MLP 融合，同时保留内部 SLA 参数解析；MegaMoE 合入 CUDA EPLB 的 UTCCP scale 参数注册并保留 HCU权重转换。
- 官方删除的旧 compressor、HIP fused compressor 与 c4/c128/common JIT 仍有内部调用，保留；无调用的旧 moe_fused_gate header 按官方删除。修复 split-prefill shared-read helper 移动后的导入。
- 官方 AMD cooperative top-k 保留在 `topk.hip`；已验证 HCU 实现在 `topk_hcu.hip`，构建时按 `_is_hcu` 选择。HCU packed KV 量化/反量化和 int64 地址计算保留。
- 删除已失效的 FormatSignature import；补回 HCU AITER 路径使用的 os 和 SWAKVPool 类型导入。

静态检查：1004 个变更 Python 文件 AST 通过，全树 compileall 通过，相对双方父提交新增 Ruff 错误为 0；直接调用参数审计无新增不匹配。完整导入审计发现 3 项合并问题均已修复（FormatSignature、shared-read helper、HIP compressor）。

NIXL/control/MLA host 单测：67 passed、9 subtests passed。HTTP warmup 单测因提交容器旧 LightOp 缺少 fuse_situ_mul_quant_contiguous 在收集阶段阻塞，需在验证容器复验。最终 TP8/HumanEval 尚未执行，不代表精度通过。

## 最终运行前适配（2026-10-19）

- `setup_hip.py` 与 `setup_rocm.py` 一致，对 gfx928/gfx936/gfx938 的 `_is_hcu` 构建选择保留的 `topk_hcu.hip`，避免 hipify 覆盖官方 AMD `topk.hip`。
- 完整 AOT 编译暴露 `legacy_radix_topk.cuh` 在 HIP 下直接包含 `cuda_fp16.h`；增加 `USE_ROCM` 条件包含 HIP half 头，CUDA 路径不变。重编译成功，产物 `wheels/sglang_kernel-0.4.10-cp39-abi3-linux_x86_64.whl`；日志 `runtime-28/aot-build-retry1.log`。
- nmz28/dpsk-v41 源码导入与 HTTP/server warmup 单测通过：4 passed、2 subtests passed。
- nmz26 卡忙；nmz28/dpsk-v41 实际为 V4.1 decode（10016），未停止。使用此前获准且 8 卡空闲的 nmz22/latest，指定 0731 checkpoint 和纯 TP8；运行结果待记录。
- 验证容器保留 DTK PyTorch 2.11/Python 3.10；editable 安装使用 `--no-deps --ignore-requires-python`，未将上游 CUDA 依赖安装到 HCU 环境。仅代表本次指定路径待验，不宣称全部 Python 3.10 兼容。EvalScope 1.13.0 使用独立 system-site-packages venv。

- nmz22 attempt1 在 wo_a BF16 einsum 的 rocBLAS/Tensile kernel 加载处失败；该表达式同步前已存在，同形状独立 GEMM 通过。attempt2 设置 `TORCH_BLAS_PREFER_HIPBLASLT=1` 后越过该位置，后续环境保留该设置；不能据此归因于本次模型代码变更。
- attempt2 暴露 JIT `CHECK_CUDA` 的 HIP 分支缺失 `runtime::is_cuda_error/get_cuda_error_string` 定义。将共用 runtime error helpers 移到 HIP/CUDA launcher 分支之前，driver `CUresult` overload 仍仅 CUDA 编译。

- Hash TopK GPU probe 进一步发现 padding mask 仅接受 `kDLCUDA`，HIP 张量的 `kDLROCM` 被拒绝；按 `USE_ROCM` 选择设备类型。修复后 n=1/7/32 的索引、sqrtsoftplus 归一化权重及 padding mask 实际 GPU 对照均通过：`runtime-22/jit-hash-topk-probe-fixed.log`。初版 probe 错误将 routed scaling factor 乘到归一化权重，已按现有 API（共享专家权重使用其倒数）修正 probe，未更改生产权重语义。

- 提交容器曾被旧 LightOp 阻塞的 DSpark PP context 测试已在 nmz28/dpsk-v41 补跑。fixture 跟进 `quantizable`、`attn_dp_size` 及 `DecodeInputBuffers.create(num_dp_ranks=...)` 新接口，生产代码未因此修改；最终 15 passed，见 `runtime-28/dspark-pp-context-tests-fixed.log`。

## 最终验收结论

2026-10-19：指定纯 TP8 验收通过。上述各阶段“待执行/尚未执行”均为当时状态；最终结果以本节及 summary 的最终验收章节为准。旧 main 同环境 HumanEval 134/164；同步首次 145/164，最终重复 140/164；三组均无 API 错误、空输出和长度截断。最终 8 rank graph capture 完成，10 条 graph decode 日志，health=200。最终生产代码 diff 与两次评测一致。

验证位置 nmz22/latest；未停止 nmz28 decode。启动需 TORCH_BLAS_PREFER_HIPBLASLT=1，默认 BLAS 环境根因未解决；完整复现命令、依赖限制与证据路径见 [最终验收记录](sync-official-main-20261019-summary.md)。仅验收本次指定纯 TP8，不覆盖全部新模型和分布式组合。
