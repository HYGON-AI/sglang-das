# 官方 main 同步 · 20261009 冲突与验证台账

## 固定基线与工作范围

- 工作目录：`/home/proj_sglang_open/sglang-das`；提交环境：`zz-nmz26 | rye_sglang_open`。
- 分支：`sync/official-main-daily-20261009`，内部 main 起点 `fb46d17ad9f633aa7fee521484262b0c26de560d`。
- 上次官方同步：`8b2ca8ecc2410f3ab170adff23fcd2db4cd47d2b`，历史总结与原同步分支祖先关系均确认。
- main 中上次 squash `ab43824b09055fc790bd5a9166a6a6182b1bccc3` 与原同步分支 `aa99d6ac5b51dd350e3447c1d54ac6b3ab94e6df` 的 tree 均为 `e58ad2befa77376afd17e0b0fa54a4f800d0728b`。
- 因 squash 丢失上游祖先，先建立 tree 不变的祖先锚点 `8dd84693716a91af59a57f7361fda4e033213422`，避免重放旧同步。
- 官方目标：`438df9a2a4645d39c87e33c4a27792e568ac2701`，增量 657 commits。整体 merge 有 110 个冲突文件，已中止并按 first-parent 分段。
- 验收：基础纯 TP8，`bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel`；候选环境 nmz26/nmz22/nmz107 的 `rye_sglang_latest`。最终精度通过前不推进 main。

## Step 01：前 100 笔

- 官方终点：`e5cec303ae7e719d27694f46e1c14b9bf34b1894`。
- 初始文本冲突：30 文件；下面逐文件记录，并补充自动合并的语义修复。
- 负责人：本次同步执行者；策略：按官方新 API 迁移当前 HCU 行为，非 HCU 采用官方路径。
- 证据：`/home/proj_sglang_open/sync-evidence/20261009/step-01/`，保留三方文件、merge 输出、静态检查 JSON。
- 静态检查：`python -m compileall -q python/sglang test` 已通过；HCU 注册检查已通过；F821/F811/F722 相对双方新增问题已修复；`git diff --cached --check` 通过。
- 功能状态：pending，最终 TP8 验证统一执行；GLM/MiniMax/Hunyuan/CP/PD 的专有路径本轮不声明运行验证。

| 文件 | 策略、保留内容与原因 | 风险 / 验证 |
|---|---|---|
| `.claude/skills~e5cec303ae7e719d27694f46e1c14b9bf34b1894` | 删除旧 facade 或文件位置冲突副本；保留官方新目录和现有有效实现。 | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/check_env.py` | combine HCU detection and official XPU environment reporting | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py` | keep HCU AOT indices API beside named official FlashMLA format | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/arg_groups/model_hook.py` | retain HYV3 SP validation and adopt official HF-config registry | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/arg_groups/overrides.py` | adopt HF-config based linear-attention extra-buffer registry | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/arg_groups/speculative_hook.py` | official DP speculation coordination diagnostic | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/batch_overlap/two_batch_overlap.py` | preserve official child batch capture contract | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/distributed/parallel_state.py` | HCU-only subgroup override; official custom communicator selection elsewhere | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/entrypoints/openai/protocol.py` | new PD routing request base plus existing constraint API fields | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | adopt official token-preserving template renderer and ordered cache key | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/function_call/function_call_parser.py` | adopt native required-tool grammar for GLM while retaining nonconflicting parsers | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | keep HCU LightOp MQA and Hadamard scaling, add official idle-DP invalid rows | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | retain HCU layouts and padding while adopting official per-request LSE and static decode scheduling | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/layers/communicator/layer.py` | 删除旧 facade 或文件位置冲突副本；保留官方新目录和现有有效实现。 | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/layers/layer_boundary/legacy.py` | 删除旧 facade 或文件位置冲突副本；保留官方新目录和现有有效实现。 | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/layers/moe/mega_moe.py` | merge CUDA fused shared experts with HCU W4A8 stream lifetime, scaling, dual-runtime weights and distinct SM cache key | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/managers/tokenizer_manager.py` | official first/last output timing and abort TTFT policy; keep outbound latency measurement | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/mem_cache/index_key_cache.py` | place HCU layer-split index allocation inside official memory-saver region | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/mem_cache/memory_pool.py` | retain HCU BF16/INT8 index cache methods needed by imported local features | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/bailing_moe.py` | official Bailing stage boundaries and TopK width contract with HCU dense/sparse RMS+FP8 inputs preserved | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | HCU fused QKV/RMS and CP helpers with official shared attention context | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/deepseek_nextn.py` | official residual stream lifecycle with HCU draft KV/page namespace prefetch | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/deepseek_v2.py` | adopt upstream stage boundaries; keep HCU LayerSplit KV lifecycle helpers | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/deepseek_v4.py` | retain HCU CP and aux capture helpers beside official stage context | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/glm5_next.py` | preserve HCU model/draft exports under _is_hcu | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/hunyuan_v4.py` | official attention context plus HCU TileLang IHC hooks | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/kimi_k3.py` | retain conditional HCU packed projection mapping and official loader-seeded mapping on other paths | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/models/minimax_m2.py` | port MiniMax HCU SP into upstream residual stream; fused RMS uses an HCU skip-norm adapter instead of removed facade kwargs | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | combine DP coordinated draft/target plan with HCU PD lifecycle/context-only ranks | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | keep HCU graph runner and LM-head VP, use official PP-aware embedding resolver | 运行时文件需最终 TP8；专有模型路径仅静态审查。 |

### 自动合并语义修复和隔离

- `StageBoundary` 迁移 HCU Bailing 量化 norm 与 MiniMax fused RMS skip-norm；残差在 `ResidualStream` 中流转，避免恢复已删除的公共 facade。
- Hunyuan-V3 仍调用已删除的 communicator：迁移到 `make_stages`，保留旧模型外部 `(hidden,residual)` 交接和 HCU SP 分支。
- 当前 GLM fork 与官方实现差异过大：公共 `glm5_next.py` 保持官方实现，HCU 实现移至 `models/hcu/glm5_next.py`，由 `_is_hcu` 选择；旧 GLM CP/MHC adapter 单独位于 `layer_boundary/hcu_legacy.py`。未覆盖到的官方新 GLM 功能不能由本轮 DSV4 验证推断。
- DSV4 的旧 `communicator_dsa_cp` 导入移至官方 `layer_boundary/adapters/context_parallel`；当前 HCU LayerSplit 的 page-plan 与 prefetch 已随自动重命名迁移。
- 补齐聊天模板 cache 上限和 GLM native grammar 导入，去掉 Kimi 自动合并产生的重复 `layer_id` keyword。
- 原 main 已存在 `all_to_all_single`、`is_dllm_extend` 等重定义静态债务，本段未将其误报为新回归。
- `.claude/skills` 与官方 symlink 冲突：保留内部技能目录，移除 Git 生成的冲突 symlink 副本。

## Step 02：第 101–200 笔

- 官方终点：`77b64e353f10045d8d6d32f56c3c5f49b8d7da55`；初始冲突 40 文件。
- 风险重点：attention DP 字段拆分、HostPoolDecl/HostPoolGroupConfig、ResidualStream 与 CP helpers 的改名。
- 证据：`/home/proj_sglang_open/sync-evidence/20261009/step-02/`。
- 编译与 diff-check 通过，F821/F811/F722 相对双方无新增错误。残差单测 `test_residual_stream.py`：20 passed、2 subtests passed。
- `test_runtime_context.py` 补充检查：初轮 180 passed / 10 failed；其中 7 个在本段合并前快照复现，涉及内部 dataclass config、旧 accessor/ceiling hook、模型 fake 缺少 get_text_config。其余 3 个为 getter census：已迁移真实调用与公共导入，2 项重跑通过；剩余一项只有文档里的旧 getter 字样，已修正，后续复查。HCU custom AR selector 归类为优化策略。
- 不把上述配置测试基线债务报告为通过；最终用户指定纯 TP 精度仍 pending。

| 文件 | 决策与保留内容 | 验证范围 |
|---|---|---|
| `docs/docs/developer_guide/development_guide_using_docker.mdx` | 采用官方开发镜像说明；内部原文件无独有功能差异 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | 官方 attention DP 配置读取与 HCU 分支并存 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/arg_groups/overrides.py` | 采用独立 attention-DP 字段；保留 HCU CP 的模型 hook 规则 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/arg_groups/parallel_hook.py` | 官方 derive_attn_tp_size 与 HCU 参数校验 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/arg_groups/speculative_hook.py` | HCU draft LM-head VP 校验采用新的 attention DP width | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 保留当前 GLM external constraint 扩展，加入 inline-system 模板检测 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/environ.py` | 采用官方 auto backend；HCU 默认在 HIP resolver 单独选择 kernel | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/hip_flash_mla.py` | HCU auto backend 维持外部 FlashMLA kernel；其他设备采用官方 auto | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | CPU seq-lens HCU guard 显式初始化 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 保留 BF16/INT8 索引和 LightOp fused store；官方 gfx950 fused indexer 排除 HCU | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/dsa/utils.py` | 合并 HCU compact Main-KV plan 与下一层 prefetch；CP gather 归属官方 adapter | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 保留 HCU 旧 CP KV materialization fallback 与官方 CUDA gate | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | 保留 HCU FP8 识别和 SP padding；采用新的 parallel config API | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | 将 HCU causal conv 适配迁移到官方 _convolve_prefill | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/linear/kda_backend.py` | HCU causal conv 使用当前 forward 的 prefix state 标记 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/qsa/mqa.py` | HCU num_stages/threads 调优与官方 scoring dtype 同时保留 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/dp_attention.py` | 保留 idle decode 的 MAX_LEN 行；使用官方 gather_width 语义 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/layer_boundary/adapters/context_parallel.py` | 官方纯 attention CP adapter；HCU prefetch 生命周期移动到 DSA utils | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/moe/topk.py` | 保留 simulated balance 限制和 HCU LightOp gate，接入官方 router partials | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 官方按初始请求 state 完成收尾；保留本地 multimodal 原始载荷释放 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | 保留 HCU BF16 KV 与 INT8 index；官方 fused query RoPE store 合同 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 官方声明式 host 构造，HCU GLM/BF16 draft sidecar 走声明 builder | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | 保留 HCU LayerSplit scratch 源和官方 extra mamba budget | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/mem_cache/memory_pool.py` | BF16 HCU index buffer 也声明 host sidecar；保留官方 shared-topk 空层过滤 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/mem_cache/pool_host/dsa.py` | 官方 HostPoolDecl 描述 HCU BF16/INT8/GLM 的真实字节格式，builder 显式 dispatch 与 packed draft offset | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/model_executor/pool_configurator.py` | HCU LayerSplit draft 内存定尺保留；其他分支接入 NPU DCP 扩展预算 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | 保留 HCU GLM 避开 AMD DSA dual graph；采用官方 virtual cache loc | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/bailing_moe.py` | 官方 PP layer construction 配合 HCU fused RMS | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` | 保留 HCU TileLang MHC；增加官方 MHC-post/MoE fusion gate | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/deepseek_v2.py` | LayerSplit page-plan/prefetch 调用迁移到官方 DSA utils | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/deepseek_v4.py` | 官方构造器移除显式 TP 参数后，HCU CP 仍使用 frozen attention rank 并单独置换 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/hunyuan_v3.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/qwen2.py` | 官方 PP layer construction 配合 HCU fused RMS | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/server_args.py` | 内部 parser/Backend 依赖与官方并行配置 helper 合并 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | 保留 HCU context-only/PP 生命周期，迁移新 draft placement API 与自身 vocab 模块支持 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/speculative/spec_utils.py` | 保留 HCU Mamba 逻辑到物理 slot 翻译 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/utils/common.py` | 采用独立 attention DP rank 数；单副本 CP 不伪装为 DP | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/utils/hf_transformers_patches.py` | 保留 torchaudio 导入修复并增加官方 MTP layer-types 校验兼容 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `scripts/ci/slurm/launch_mi355x.sh` | 官方修复未引用 heredoc 中反引号意外执行 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `test/manual/ep/test_eplb.py` | 测试参数采用新 attn-dp-size | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/deepseek_nextn.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/mem_cache/pool_host/glm5_next.py` | HCU GLM host indexer 接受官方 decl builder 输入；保留 LayerSplit 物理所有权与现有 transfer kernels | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/managers/scheduler_components/dp_attn.py` | HCU scheduler skip metadata gather 条件采用新实际 DP replica 数 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `docs/src/snippets/configs/zai-org/glm-5.2.jsx` | 官方参数改名后清理原 CRLF trailing whitespace | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/arg_groups/model_hook.py` | HCU HYV3 SP pure-TP 校验包含独立 attention DP | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/disaggregation/encoder/receiver.py` | 保留内部 encoder 路径并读取新的 attention DP gate | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/managers/scheduler.py` | HCU PD Decode scheduler 控制面采用新 attention DP 配置和拓扑 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/layer_boundary/hcu_norm.py` | HCU fused Bailing norm 仍排除 attention-DP | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/layer_boundary/hcu_legacy.py` | GLM HCU adapter 改读实际 attention-DP gate | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/minimax_m2.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/communicator_glm5_next_mhc.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/sampler.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/speculative/eagle_utils.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/speculative/draft_lm_head_vp.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/models/hcu/glm5_next.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/utils/cp_utils.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/quantization/slimquant_w4a8.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/quantization/slimquant_w4a8_marlin.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/glm5_next/indexer.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/layers/attention/dsv4/compressor.py` | 保留当前 HCU/内部业务行为；已退役的 distributed getter 调用改为 frozen get_parallel 字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `python/sglang/srt/distributed/__init__.py` | 调用者已迁移，删除为旧 HCU 调用恢复的 deprecated 公共 re-export | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |
| `test/registered/unit/test_runtime_context.py` | census 将当前 HCU custom-allreduce backend 选择器归类为优化策略，非拓扑字段 | 静态与 residual 单测；最终 DSV4 纯 TP8。专有模型/PD/CP 未做功能声明。 |

自动合并审查补齐 scheduler 的实际 DP replica 数、GLM/HCU VP 和 HYV3 SP 参数校验、host 索引的物理字节大小及 packed draft 层偏移。HCU FlashMLA default 从全局 EnvStr 迁到 `_is_hcu` resolver；非 HCU 默认跟随官方 auto。DSA utils 的 LayerSplit 引用采用 lazy import，避免循环导入。
