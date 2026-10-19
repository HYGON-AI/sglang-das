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

## 运行验收（待执行）

整理完成后，用户已授权停止 `zz-nmz28 | rye_sglang_dpsk-v41` 的 prefill 进程用于验证。只停止经核对属于该服务的进程。
目标保持 DeepSeek-V4-Flash-0731-FP8-Channel、纯 TP8、端口 10015。需服务就绪、graph capture 与至少 3 次 replay、完整 EvalScope HumanEval；尚无本次精度结论。

证据目录：`/home/proj_sglang_open/sync-evidence/20261019/`。
