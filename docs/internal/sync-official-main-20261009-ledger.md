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
