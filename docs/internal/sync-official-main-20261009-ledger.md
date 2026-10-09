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

## Step 03：第 201–330 笔

- 官方终点：`3e9a120adde240ba62abfe78c54ce367e32bfae9`；初始冲突 34 文件。
- 状态：compile、diff-check、相对双方 F821/F811/F722 通过；最终 DSV4 纯 TP8 运行精度 pending。
- 证据：`/home/proj_sglang_open/sync-evidence/20261009/step-03/`。

| 文件 | 决策与保留内容 |
|---|---|
| `docs/src/snippets/configs/zai-org/glm-5.2.jsx` | 采用官方 GLM-5.2 部署示例并统一 LF |
| `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` | 保留 HIP vector 显式字段读取；gfx950 RoPE 舍入顺序仅在其目标架构生效 |
| `python/sglang/srt/arg_groups/parallel_hook.py` | 采用官方新 ROCm CP 支持；仅 MUSA 保留禁用条件 |
| `python/sglang/srt/disaggregation/common/conn.py` | 官方延迟分配/清理通知与内部 PD hidden metadata 释放并存 |
| `python/sglang/srt/disaggregation/decode.py` | 采用官方包含 start_time 的五元组；保留本地 abort RID 记录 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | PD hidden 流传输移植到官方 abort helper；清理只能在所有写入排空后发布，线程在初始化完成后启动一次；PD hidden deferred chunk 保持 outstanding 计数，直至最后一个 packet 写入结束 |
| `python/sglang/srt/disaggregation/prefill.py` | 保留 PD hidden bootstrap 失败处理和重排队；采用官方 checkpoint API 与 prefill_complete 策略 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 采用官方 token-first prompt 接口；显式 external constraint 优先于内置 GLM grammar |
| `python/sglang/srt/environ.py` | 保留内部 HCU/W4A8 开关，增加官方 AITER 排序策略和 Hopper BF16 weight cache |
| `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | 官方 gfx95 能力检测与 HCU CPU seq-lens 路径并存 |
| `python/sglang/srt/layers/attention/dsa/utils.py` | HCU 保留现有 CP round-robin 契约；其他 ROCm 采用官方 CP |
| `python/sglang/srt/layers/attention/dsa_backend.py` | HCU FlashMLA 保留真实行截断/补齐，其他平台使用官方 kv_format ABI |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | 查询准备迁移至官方 helper；HCU 保留 descale、FNUZ 查询转换及 bhsd cache，官方 FA4 capture 上界修复；HCU packed MLA store 使用官方 KVWriteLoc |
| `python/sglang/srt/layers/attention/linear/kda_backend.py` | 官方 ROCm KDA 与 HCU grouped conv 导入并存 |
| `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` | 新增 QSA eager-break context；保留 HCU kernel 配置 |
| `python/sglang/srt/layers/attn_residual.py` | 保留显式 HCU boltops aggregation；官方 TMA capability 使用 is_cuda |
| `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` | 官方 platform context 与内部 custom op 注册并存 |
| `python/sglang/srt/layers/moe/topk.py` | 官方 AITER shared-column 写入契约与 HCU LightOp gate dispatch 合并 |
| `python/sglang/srt/layers/rotary_embedding/base.py` | 保留 rotary stream launcher；官方 HIP cos-sin cache 保持 FP32 |
| `python/sglang/srt/layers/utils/hcu_cp_utils.py` | 隔离 HCU 已验证 CP gather/split/overlap helpers，避免恢复其他平台废弃算法 |
| `python/sglang/srt/layers/utils/cp_utils.py` | 采用官方废弃平台 shim；_is_hcu 显式导出 HCU legacy helper |
| `python/sglang/srt/managers/scheduler.py` | 官方暂停边界/计时与内部 PP collective 控制请求前置 relay 并存 |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | 采用官方 checkpoint_kv_cache API；保留 HCU SWA slot 释放 |
| `python/sglang/srt/mem_cache/dsa_cache_layer_split.py` | 保留 scratch 生命周期；适配官方 KVWriteLoc 解析 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 官方 QSA host sidecar 与 HCU GLM indexer/LayerSplit 定尺分别保留 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 官方 token-major Mamba views 和 stride 修复；HCU bhsd KV slot stride 单独保留 |
| `python/sglang/srt/models/bailing_moe.py` | 官方 StageBoundary 支持 exit fusion；保留 HCU sparse RMS quant payload |
| `python/sglang/srt/models/deepseek_v4.py` | HCU CP rank 置换受 _is_hcu 保护；保留官方未冲突更新；已删除 dsa_use_prefill_cp 改用官方 is_cp_active；HCU legacy CP 使用原 metadata predicate |
| `python/sglang/srt/models/minimax_m2.py` | 保留当前 HCU MiniMax TBO hooks 和优化 prepare；官方未冲突模型流程保留 |
| `python/sglang/srt/models/qwen4_exp.py` | 保留 PLE INT8/FP8-channelwise 识别；采用 QSA eager break 与共享 topk bridge |
| `python/sglang/srt/models/qwen4_exp_mtp.py` | 官方 HC 宽度 graph input slot；内部 fusion loader 保留 prefix 参数 |
| `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | HCU draft VP 与官方 ROCm draft topk1 导入并存 |
| `python/sglang/srt/speculative/eagle_worker_common.py` | 保留 HC pre-norm hidden capture，并加入官方 mrope positions |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | HCU fused top1 仅在不需要 draft distribution 时使用；采用官方全贪心跳过概率构建 |
| `test/registered/unit/models/test_glm5_next_bfg_fusion.py` | 恢复官方 GLM generic fusion 测试；HCU 模型已独立 dispatch |
| `python/sglang/srt/mem_cache/common.py` | checkpoint 不再接受旧 chunked kwargs；跳过 tree 插入的请求仍保留自己已计算的前缀 KV |
| `python/sglang/srt/layers/cp/utils.py` | 官方 generic ROCm CP 生效；仅 HCU 保留 legacy batch layout，避免双重 shard |

功能验收范围为本次用户指定的 DSV4 纯 TP8；其他模型、PD、CP、MTP 只完成代码与静态审查。

## Step 04：第 331–430 笔

- 官方终点：`61ba98443d2a9e67b79b06b01ee227515bf3c9ea`；初始冲突 20 文件。
- 状态：compile、diff-check、相对双方 F821/F811/F722 通过；最终 DSV4 纯 TP8 运行精度 pending。
- 证据：`/home/proj_sglang_open/sync-evidence/20261009/step-04/`。

| 文件 | 决策与保留内容 |
|---|---|
| `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py` | HCU fused temb table 保留；新增官方 fused linear GELU |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py` | 平台支持提示跟随官方 ROCm 支持 |
| `python/sglang/srt/disaggregation/decode.py` | 官方在 match 前限制可复用 full prefix；HCU full-only radix leaf 匹配标志保留 |
| `python/sglang/srt/disaggregation/encoder/receiver.py` | 新的 scoped placement 初始化；保留 attention-DP encoder 集合通信组 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 官方先校验后发布 peer registration；HCU state ABI 校验在发布前仍生效 |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | 官方 RequestWindow sparse-prefill 与 HCU sparse-kernel 排除条件 |
| `python/sglang/srt/layers/attention/dsv4/compressor.py` | 官方 AITER tuned BF16 compressor GEMM；local HIP/HCU capability 保留 |
| `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py` | HCU 真实 query 长度与官方 RequestWindow workspace 同时支持 |
| `python/sglang/srt/layers/moe/topk.py` | HCU DeepEP int64/-1 padding 合并 remap 保留；其他平台采用官方 append+权重 zero 融合 |
| `python/sglang/srt/managers/schedule_policy.py` | max_prefix_len 与 HCU full-only prefix match 控制分别保留 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 采用官方单次 dtype/scale 转换；DSA 物理页拆分并保留活跃 indexer 层映射，HCU 不进入 AITER preshuffle |
| `python/sglang/srt/mem_cache/pool_host/dsa.py` | 官方 pooled Index-K 字节预算，HCU BF16/INT8/GLM 保留真实 storage bytes |
| `python/sglang/srt/model_executor/pool_configurator.py` | 保留 HCU BF16/INT8 workspace 与活跃层计数；generic pooled index 采用压缩后 bytes/token |
| `python/sglang/srt/models/deepseek_nextn.py` | 官方 layer_stack 与 HCU legacy CP gate 并存 |
| `python/sglang/srt/models/dflash.py` | DFlash quantization 声明与内部 vocab-sharing 声明独立保留 |
| `python/sglang/srt/models/hunyuan_v4.py` | 官方 IHCState residual/stage 契约；保留 HCU TileLang iHC operator dispatch |
| `python/sglang/srt/layers/hy4_ihc_tilelang.py` | HCU 专属 BoltOps TileLang iHC 入口增加 _is_hcu 保护 |
| `python/sglang/srt/models/minimax_m2.py` | 保留 HCU MiniMax opt 排除；官方相邻层构造阶段不选择 fused norm |
| `python/sglang/srt/models/qwen4_exp.py` | 保留 PLE INT8/FP8 per-row scale；官方邻层构造禁用 offload |
| `python/sglang/srt/server_args.py` | 保留内部 dataclass resolved-copy 生命周期；新增官方 MLX/MPS runtime 与 deprecated attn-DP readback |
| `python/sglang/srt/speculative/eagle_utils.py` | 官方 block-verification sampling_fn 与 HCU Triton rejection sampling import 合并 |
| `python/sglang/multimodal_gen/runtime/layers/attention/layer.py` | 自动合并审查：清理重复 envs import，保留内部 ring-overlap 选择 |

功能验收范围为本次用户指定的 DSV4 纯 TP8；其他模型、PD、CP、MTP 只完成代码与静态审查。

## Step 05：官方增量 431–530

- 官方终点：`c389cbaf99e1b1ddd77005c9107bb3dae7bb89d7`；初始冲突 18 文件。
- 状态：compile、diff-check、相对双方 F821/F811/F722 通过；最终 DSV4 纯 TP8 运行精度 pending。
- 证据：`/home/proj_sglang_open/sync-evidence/20261009/step-05/`。

| 文件 | 决策与保留内容 |
|---|---|
| `.codespellrc` | 合并拼写白名单 |
| `.github/workflows/release-docker-amd-miles-rocm10-nightly.yml` | 采用官方 nightly 工作流手动构建参数 |
| `docs/src/snippets/configs/zai-org/glm-5.2.jsx` | 采用官方 ROCm GLM 配方并统一 LF |
| `python/sglang/srt/arg_groups/fields/spec.py` | 内部 Annotated 元数据与官方 argparse 兼容 action 并存 |
| `python/sglang/srt/disaggregation/decode.py` | 采用官方 lazy Mamba 初始 slot 计数；保留释放 prefix lock 后重置 root receipt；排队重试释放迁移 TreeLock/unlock，清空 req.lock 防重复释放 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 保留 HCU PD hidden source_event 与 packet metadata，加入官方早发 cached-prefix wait_event |
| `python/sglang/srt/disaggregation/prefill.py` | 采用 release checkpoint=False；保留 HCU hidden pool metadata 释放 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | gfx1250 wave32 官方路径与 HCU BF16/INT8 专属 writer 都跳过不适用 act_quant |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 新增官方 Triton sparse MLA backend；保留 HCU sorted topk 复用 |
| `python/sglang/srt/layers/moe/mega_moe.py` | 保留显式 HCU standalone/deep_gemm 双 runtime 和 buffer key；官方 MMA type 接口由调用者传入 |
| `python/sglang/srt/managers/scheduler_components/dp_attn.py` | 保留 PD decode epoch callback；加入官方 sync-wait carry 指标 |
| `python/sglang/srt/mem_cache/common.py` | 官方 skip-insert prefix resume 与内部 DSV4 prompt-once donation 共用 checkpoint |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | 官方 uniform FP8 fused norm/rope/store；HCU BF16 独立保持现有 two-step 写入 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 保留 HCU packed-draft layer map 与 GLM Mamba page_first host layout；generic kv_split 用 direct；声明工厂统一拥有 packed draft 映射及 indexer entry，移除重复旧拼装，预算读取 root.packed_draft_device_pools |
| `python/sglang/srt/mem_cache/memory_pool.py` | HCU 活跃索引层及物理 page 维持；slots_per_page 随官方改名 index_page_size |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 插入时采用官方 tree-only walk；保留 DSV4 full-only leaf 匹配 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | 完整核对 current-vs-base 差异后迁移官方 AttentionGraphVariants/graph ABI；保留 _is_hcu pre-head capture、MegaMoE token count 与 output info |
| `python/sglang/srt/layers/attention/graph_variants.py` | HCU GLM 排除 AMD DSA dual graph；无 CPU 长度镜像时选择 sparse，避免 replay D2H 同步 |
| `python/sglang/srt/models/deepseek_v4.py` | 官方 mHC state machine/post dispatch；HCU TileLang post、deferred repeat/CP split 与 local capture 层集保留 |
| `python/sglang/srt/models/hunyuan_v3.py` | 补上 Step04 自动合并语义审查：HYV3 stage 构造使用官方 layer_stack 内的 append_stages 与 next_layer_sparse |

功能验收范围为本次用户指定的 DSV4 纯 TP8；其他模型、PD、CP、MTP 只完成代码与静态审查。

## Step 06：官方增量 531–657（最终终点）

- 官方终点：`438df9a2a4645d39c87e33c4a27792e568ac2701`；初始冲突 34 文件。
- 状态：compile、diff-check、相对双方 F821/F811/F722 通过；最终 DSV4 纯 TP8 运行精度 pending。
- 证据：`/home/proj_sglang_open/sync-evidence/20261009/step-06/`。

| 文件 | 决策与保留内容 |
|---|---|
| `.github/workflows/nightly-test-intel.yml` | 采用官方 CI trigger，保留其余 HCU suite 注册 |
| `.github/workflows/pr-test-amd-extra.yml` | 采用官方 CI trigger，保留其余 HCU suite 注册 |
| `.github/workflows/pr-test-extra.yml` | 采用官方 CI trigger，保留其余 HCU suite 注册 |
| `python/sglang/kernels/ops/attention/fla/layernorm_gated.py` | 保留 HCU LightOp 非3D非量化 fused norm；官方 quant_heads 新 ABI 走 Triton |
| `python/sglang/kernels/ops/kvcache/mla_buffer.py` | HCU 静态 masked BF16 scatter 与官方无 Triton naive scatter 并存 |
| `python/sglang/kernels/ops/speculative/dspark/dspark_accept.py` | HCU JIT cache 隔离依赖与官方 partial 并存 |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py` | 保留按模型 contract 提取 video 请求额外字段的内部功能 |
| `python/sglang/srt/arg_groups/model_hook.py` | 保留 HYV3 SP 拓扑校验；官方 noncausal decision checkpoint 禁用缓存/chunk |
| `python/sglang/srt/disaggregation/decode.py` | prefix_len/extend_end 新请求接口；DSV4 prompt donation 状态及 full-only 匹配迁移 canonical match_kv_cache |
| `python/sglang/srt/disaggregation/prefill.py` | 传输边界采用官方 extend_end |
| `python/sglang/srt/layers/attention/flashmla_backend.py` | HCU page builder 保留，读地址翻译采用新 reads_are_translated/KVLocPlan |
| `python/sglang/srt/layers/attention/linear/kda_backend.py` | HCU convolution/source Triton dispatch 保留；官方 FlashInfer prefill graph/chunk 校验并入 |
| `python/sglang/srt/layers/linear.py` | HCU prequantized input迁移；GLM HCU CP head partition显式attn_cp group，仅_is_hcu解析 |
| `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` | 保留 HCU fp8 MoE 路径所需 runtime_context/is_hcu imports |
| `python/sglang/srt/managers/schedule_batch.py` | 使用 canonical match_kv_cache；SWA reprefill 限制现已集中到 common._req_radix_key |
| `python/sglang/srt/managers/schedule_policy.py` | 删除旧 match_prefix_for_req；统一 canonical match_kv_cache |
| `python/sglang/srt/managers/tp_worker.py` | 保留 HCU pre-head capture；新增 official return_kv_loc_plan 输出 contract |
| `python/sglang/srt/mem_cache/allocator/paged.py` | 保留 HCU allocator extend 和 int32 decode ABI；支持官方无 Triton naive fallback，decode先检页预算 |
| `python/sglang/srt/mem_cache/common.py` | 保留官方 prefix_len resume；DSV4 allow-insert-once 迁移新 checkpoint |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | 官方 DSA pool class policy保留；HCU GLM pool与 indexer layer mapping dispatch保留 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 官方 full_kv_pool_class override 与 HCU GLM 专用 pool 分派兼容 |
| `python/sglang/srt/mem_cache/pool_host/dsa.py` | 官方平台 capabilities 与 destroy/unregister 生命周期保留；HCU BF16/INT8 index bytes/transfer保留 |
| `python/sglang/srt/mem_cache/pool_host/mha.py` | 官方平台 capabilities 与 HCU host布局 dispatch 保留 |
| `python/sglang/srt/mem_cache/pool_host/mla.py` | 官方平台 capabilities 与 HCU host布局 dispatch 保留 |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | DSV4 full-only leaf 仍由 canonical _match_tree 返回长度，删除旧 device_indices 读取 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | HCU MainKVPagePlan 与官方 KVLocPlan 类型并存 |
| `python/sglang/srt/models/bailing_moe.py` | ResidualStream hidden tuple zero-row 兼容；并行 policy 采用 is_replicated |
| `python/sglang/srt/models/mimo_v2_nextn.py` | 官方 qkv weight loader 绑定投影模块 |
| `python/sglang/srt/models/minimax_m2.py` | 内部 minimax_opt 复制投影保留；其余投影使用官方 parallel_group contract |
| `python/sglang/srt/speculative/dspark_components/dspark_draft.py` | 保留内部 num_token_non_padded；绑定官方 draft/verify KVLocPlan |
| `python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py` | 保留 HCU DSV4 pool type检查；引入官方 KVLocPlan cols |
| `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | 官方 inject地址从计划取；投影predicate已在该函数上方计算，删除重复 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | 保留 HCU pre-head capture；携带 canonical KVLocPlan/PP topology |
| `test/registered/unit/managers/test_prefill_adder.py` | 保留 Range 测试与 HCU suite注册 |
| `python/sglang/srt/layers/attention/qsa/dsa_indexer.py` | 旧 MultiPlatformOp import迁移官方兼容shim；保留内部 indexer后端 |
| `python/sglang/srt/layers/attention/glm5_next/indexer.py` | 旧 MultiPlatformOp import迁移官方兼容shim；保留内部 indexer后端 |
| `python/sglang/srt/models/glm5_next_vision.py` | 自动合并语义审查：保留原权重分片语义，迁移官方 parallel_group 构造接口 |
| `python/sglang/srt/models/hunyuan_v3.py` | 自动合并语义审查：保留原权重分片语义，迁移官方 parallel_group 构造接口 |
| `python/sglang/srt/models/kimi_k3.py` | 自动合并语义审查：保留原权重分片语义，迁移官方 parallel_group 构造接口 |
| `python/sglang/srt/models/hcu/glm5_next.py` | 自动合并语义审查：保留原权重分片语义，迁移官方 parallel_group 构造接口 |
| `test/registered/amd/test_smallm_fp8_proj_gfx950.py` | 测试构造迁移 replicated group |
| `python/sglang/srt/models/deepseek_v4_mhc.py` | 自动合并语义审查：post空行检查后显式 HCU AITER TileLang dispatch，避免落入通用 TileLang CUDA实现 |

功能验收范围为本次用户指定的 DSV4 纯 TP8；其他模型、PD、CP、MTP 只完成代码与静态审查。
