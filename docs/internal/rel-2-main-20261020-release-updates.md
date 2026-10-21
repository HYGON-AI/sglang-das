# Rel-2-Main：release/20260825_v0.5.18 更新盘点（2026-10-20）

## 冻结边界及上次同步

- 本轮 source S：`030f6254f2d9d4abe9e57543e3f38757d12ec4ba`。
- 本轮 main 起点 A：`4975d6490ac2c71c2d88c8a86b96a6f7061050ab`。
- 唯一 Git 分岔点 B：`688659c15694195ee0c6fab18de26b60b313b82f`。
- 上轮任务标识为 **20260929**；实际于 **2026-09-30 20:27:01 +08:00** 通过 squash PR #470（`756ae5b4fe26cf9ad666d46cda6c45b14b3abc60`）进入 main。
- 上轮 source 冻结点：`ea1c297dc23343cbe5c6b619f3094325fb5fe45b`，共 226 笔；本轮新增 51 笔，B..S 合计 277 笔、604 个净变化路径、49 个原始作者身份（name/email，不等同实际人数）。

Squash 未保留原 source 祖先关系，因此 `rev-list S --not A` 返回全部 277 笔，不能当作 277 笔全新功能。上轮最终审计 `de0881e4e4562545f5e04a913b566dad8db3097a` 与 #470 tree 相差 15 个文件，包括 CI、权重加载与文档，不能称两个 tree 完全一致。#470 的提交说明也列出 11 类待核对功能；本轮重新审查，不沿用历史总体 PASS。

本文是冻结范围的更新盘点；代码移植、功能覆盖及真实验证结论另见本轮报告和台账，不能把提交标题当作功能验收。

## 本轮 51 笔增量的模型与 bugfix 要点

| 范围 | release 增量与需保留的行为 |
|---|---|
| DeepSeek-V4 / Flash-0731 | W4A8 API 与 general_fc 精度修复；paged MQA logits 可选行分块；MegaMoE + DeepGEMM；paged split-KV 与 opt-in DSpark V2。保持默认路径及开关语义。 |
| DeepSeek-V3.1/V3.2、MLA | INT8/FP8/BF16 HCU 服务、EAGLE draft、CUDA graph FP8 decode 修复；RMS quantization flag 透传；DeepGEMM 权重打包及 masked FP8 API 对齐。 |
| GLM-5.1/5.2/5.3 | DSA 服务兼容、CP decode FlashMLA；compressed-tensors W4A8 打包元数据、BF16 MTP 加载及 DeepEP 通信；忽略不使用的 expert shape metadata。 |
| Qwen3/3.5/3.6、Qwen4-Exp | MoE gate 量化、channel-FP8 重复输出；DFLASH HCU 加速；BF16 PLE offload OOM、channel FP8/INT8 AITER 精度、PLE/MTP decode graph；QSA e4m3 KV scale 图捕获校准。 |
| Kimi-K3 / KDA | gfx938 decode fusion、packed FP8 Q 保留旧 MLA 路径；可选串行 shared-expert overlap；AITER allreduce shape/接受率异常修复；boltops verify/decode API。 |
| MiniMax-M3/H3 | gfx938 prefill/decode 优化、channel INT8/FP8 推理；MoE 权重副本限制在需要的模型，避免其他模型额外内存开销。 |
| HYV4 | FP8 e4m3 KV cache 与 learnable attention sink 联动；需与现有 CP、scale 和模型入口一起核对。 |
| MoE / 量化 | W4A16 ASM、W4A8 swiglu_limit；DeepEP normal combine、flat dispatch SBO guard；INT8 fallback、AITER W8A8 ABI、FP8 double packing、SlimQuant routing；gfx928 禁用不适用的 smooth ASM；非 AITER compressed-tensors FP8 不做错误 expert padding。 |
| KV / 调度 / 通信 | external KV abort 延后重试与 source drain；Mooncake ReadPlan、PP direct-L3；KV 写入移除不安全 alt_stream overlap；DP health probe 保持生成路径；CP token 非整除；可选预构建 token 异步 H2D。 |
| Diffusion / CI / 测试 | 多视频输出云存储；Wan SLA/FA HCU admission；组织 runner 迁移、多 DTK/Python wheel 矩阵；unified attention KV 索引测试修复。发布分支约束按 main 场景适配，不触发外部发布。 |

## 历史缺口复核范围

上轮 F01–F11 涉及 DSV4 CP Cache LayerSplit、L3 故障恢复/PP 命中、Mooncake direct I/O/ReadPlan、QSA tokenwise/FP8 scale、HYV4 CP、HCU TopK v2、Kimi 缩减专家加载、CP BCG 最大上下文及 MHA/MTP 指针布局。当前 main 已经经历官方重构，需要以真实入口和 producer/consumer 契约判断，不能只匹配符号。纯 TP8 验收不代表上述所有多卡模式均已运行验证。

## 本轮新增提交（上轮冻结点之后，拓扑顺序）

| SHA | 原作者 | 更新 |
|---|---|---|
| `6086af28ddb9` | dingxinlong | ci(hcu): migrate v0.5.18 workflows to organization runners (#465) |
| `10053303694b` | ybhe1110 | [Fix][Unified Cache] Retry external KV abort cleanup after forward and source drain (#455) |
| `1828115e08ab` | jiafei96 | fix w4a16 moe asm error (#462) |
| `4ba29d557c3a` | liuzj512 | [HY4][HCU] Enable fp8_e4m3 KV cache with learnable attention sink (#418) |
| `8effe1350872` | WenQ7 | feat(minimax-m3): optimize prefill and decode on HCU gfx938 (#453) |
| `3854f77546c1` | daishiqiang | opt(kda): update kda verify decode op from boltops (#458) |
| `08ea4dec71b0` | zhengzhi | fix(dsv4): resolve Flash-0731 W4A8 API and general_fc accuracy issues (#450) |
| `76e87a951d7a` | lipenghui | fix(hcu): serve DeepSeek-V3.2 INT8, FP8, and BF16 in sglang0.5.18 (#466) |
| `e885e7223180` | wu ju | fix(hcu): restore ForwardBatch RMS quantization flag (#469) |
| `cfa865d339c0` | chaoyujiang | fix deepgemm masked fp8 api upstream (#472) |
| `f315216140b1` | zhangang1221 | fix(hcu): enable EAGLE draft decoding for DeepSeek MLA (#477) |
| `b7e4eae1243f` | Wenbo Zhu | fix: remove alt_stream overlap copy when writing KV cache (#478) |
| `7fa7729c1d9a` | leichuan | fix(cache): harden Mooncake ReadPlan and PP direct-L3 cache loading (#482) |
| `2cc1bcc28c5c` | maxiaobetter | fix(deepgemm): 统一 INT8/FP8 权重重排与 FP8 masked GEMM 接口 (#484) |
| `723ac998c384` | qieshiniao | 修复 GLM-5.1/5.2 DSA 服务兼容性并优化 CP decode FlashMLA (#475) |
| `2bf2890f8830` | lipenghui | fix(hcu): correct DeepSeek-V3.1 FP8 decode under CUDA graph (#483) |
| `a06c34ef7511` | lizhigong | fix(ci): stage-b include 列表容忍分支差异的测试文件 (#485) |
| `6ca7a1bd6a1e` | maxiaobetter | fix(test): 修复 unified attention 单测 KV 索引构造 (#489) |
| `b3632f0be3bf` | liuzj512 | fix(qwen3_moe): quantise MoE gate for compressed-tensors FP8 checkpoints (#492) |
| `1882acd8c659` | maxiaobetter | fix(moe): restore DeepEP normal combine for HCU groupgemm (#491) |
| `bfbff7f68e02` | WenQ7 | fix(hcu): preserve INT8 MoE fallback and restore AITER W8A8 compatibility (#487) |
| `966f6da1bcf7` | chaoyujiang | fix(hcu): avoid double packing channel FP8 DeepGEMM weights (#497) |
| `e1eaf0829843` | maxiaobetter | fix(moe): guard SBO setup for flat DeepEP dispatch (#501) |
| `f106c53608df` | Yongqiang Duan | perf(hcu): 优化小 batch W8A8 INT8 GEMM 的 Tiling 选择 (#502) |
| `2ac651d8f6b7` | maxiaobetter | fix(w4a8): 兼容 GLM5.3-w4a8 模型的 compressed-tensors 打包权重 (#506) |
| `c9a84fca9624` | daishiqiang | [Perf][HCU] Optimize Kimi K3 decode fusions 20261010 (#507) |
| `49bb75c791c1` | zhangpc3 | fix(diffusion): upload all video outputs to cloud storage |
| `aebf1a9f41ad` | chaoyujiang | fix(hcu): fix SlimQuant INT8 DeepEP routing and DeepGEMM weight packing (#514) |
| `018e5799abb1` | Wenbo Zhu | Fix/hcu aiter moe inplace (#517) |
| `71b2ee4ea67b` | daishiqiang | perf: add opt-in asynchronous prebuilt token H2D (#516) |
| `9dd4e2e80065` | FrankXu0908 | [diffusion] attention: fix HCU Wan SLA and FA family admission (#519) |
| `3b5ca6e33508` | GarrySnail | fix(qwen4_exp): BF16 PLE offload OOM, channel-FP8/INT8 aiter accuracy, and PLE/MTP decode graph fixes (#513) |
| `3604cec316cc` | maxiaobetter | fix(hcu): 修复 GLM W4A8 target 的 BF16 MTP 加载与 DeepEP 通信 (#520) |
| `b290dcf39562` | chenyuege | [feat] 优化 HCU 上 Qwen3.6-35B-A3B 使用 DFLASH 时（--speculative-algorithm DFLASH）的性能 (#522) |
| `5f31aec46b5e` | WenQ7 | fix(aiter): avoid duplicate MoE weights outside MiniMax-M3 (#526) |
| `40c9ec29b31d` | maxiaobetter | fix(scheduler): keep DP health probes on the generation path (#528) |
| `0ac3b2f83eb8` | s1mplezpc | feat(MiniMax-H3): support MiniMax-H3 channel-wise INT8/FP8 inference on HCU (#532) |
| `ea3acdbe95df` | JingLee | feat(dsv4): add opt-in paged MQA logits row chunking (#536) |
| `5b630225295b` | lipenghui | fix(hcu): keep packed FP8 Q on the old MLA path for Kimi K3 (#537) |
| `c2666f4b5c07` | yangyn | [fix] 修复cp 逻辑：解决实际 token 不被整除时不走 cp 的问题 (#538) |
| `3c2b5266717c` | Songge | fix(GLM5.3 w4a8): skip unused packed expert shape metadata (#539) |
| `6fd9d5644212` | chenyuege | [Bugfix] 修复Qwen3.5-122B-A10B-Channel-FP8重复输出的问题 (#545) |
| `6fe59ce1ab68` | Crowder13 | ci: release-pypi-nightly-hcu 多版本编译 (dtk26.04/py3.10+3.12, dtk26.10/py3.12) [rel/20260825_v0.5.18] (#544) |
| `23d1def83c9e` | YuanYuan | [FEAT] support megamoe with deegemm in deepseek-v4 (#525) |
| `ae065131d82c` | daishiqiang | feat(kimi-k3): add opt-in serial shared-expert overlap for DeepEP (#540) |
| `da90edf068c9` | daishiqiang | fix(k3): fix aiter ar shape mismatch and some entries accept rate=1 (#546) |
| `7586e89f5e06` | GarrySnail | fix(qwen4-exp): calibrate QSA e4m3 KV scales during graph capture (#551) |
| `0bb71f0c7fd7` | yangyn | [fix] 解决 w4a8 aiter moe 不支持 swiglu_limit 功能 (#550) |
| `b9e0052e9e35` | GarrySnail | fix(hcu): keep the w8a8 smooth asm fast path off on gfx928 (#555) |
| `a031c979a4a8` | liuzj512 | [fix] 修复非 AITER 路径下 compressed-tensors W8A8 FP8 MoE 权重被错误补齐导致输出乱码 (#564) |
| `030f6254f2d9` | daishiqiang | perf(attention): add paged split-KV support and opt-in DSpark V2 (#553) |

## 原始分岔点至冻结 S 的完整提交索引

以下覆盖全部 277 笔（含 merge、revert 和上轮 226 笔），最终有效需求需结合净变化及后续撤销判断。原作者邮箱、完整消息、父提交、PR 候选和测试明细保存在 inventory，未改写原始对象。

- 001 `e04856c1d4613fcfbc39aa0ed88bc5d3ad5153d7` · Codex · fix(minimax-h3): allow HCU full-loop denoise
- 002 `ce113101ca7d946ccd9ef0a1db8c1d876b7c7c19` · zhangpc3 · fix(minimax-h3): avoid sharing refined prompts with CacheDiT
- 003 `530d6363b30262d10286610938a45e8219306784` · YuanYuan · Merge pull request #231 from s1mplezpc/0518-minimaxh3-dev
- 004 `726db6820363a4f74a42b607b2447f403102e91a` · YuanYuan · Merge all deepseek-v4 updated features including DSpark and more into v0.5.18 release.
- 005 `8b50a269f8057ace50e0fa33a17237855ddb92a8` · zhangyh15 · bugfix router路径不对
- 006 `d5882c133e831c46cab6e37df61f748a9efc4dbf` · zhangyh15 · [feature] 发包流水支持手动触发,手动触发不上传pypi
- 007 `89d8ac7b1d97151c0389e34cc9f4d097fce7ef99` · lizhigong · Merge pull request #233 from Crowder13/release/20260825_v0.5.18
- 008 `32f86c628bcd8baebe8eb2b2a6a0b3924c73f566` · liuzj512 · fix(hy3): align LightOp routed scaling factor handling
- 009 `c063dda9a0f7dfdcab84c15cd7c1baa27cb860e9` · lizhigong · Merge pull request #234 from liuzj512/fix/hy3-lightop-rsf
- 010 `785804bfb994a0afb14e936b163be40803190d28` · shiqiang · fix triton moe path but accuracy fail
- 011 `a0cc114984490e7caee7155d1a4340ef09e3dd2a` · RyeYuan · [BugFix] fix the mxfp4 triton accuracy issues
- 012 `31576cba7da75ca463fc1fee1d4010dc2ea0bd47` · shiqiang · feat: integrate hcu_mla and causal_conv1d modules
- 013 `a7931170f3272df84434f25f9fd757eb6201109f` · shiqiang · sytnx fix
- 014 `a6ce82576e49f7456af4b83d1bae95625e26bb89` · shiqiang · feat: support flash kda, hcu triton kda and hcu triton attn residual
- 015 `12ba645c77fd34eb18379b2bd7dc503f25a3b4e9` · shiqiang · feat: support WFP4A16 hipc moe for kimi-k3
- 016 `6e2576a7498b62845235ea3e57289d673ea3639b` · shiqiang · fix: route hcu_mla prefill through the chunked prefix cache path
- 017 `8084b5cacf215104abd62eb2e5e548a0f019302a` · zhaosong1 · merge: fix kimi-k3 merge conflict
- 018 `9c60fd0fdffdc9fa0ad89e7ecf0bf046482a4d62` · shiqiang · fix: fix situ activation jit error
- 019 `4a004a87bd03e8905631c4dd9ea41f46046f4712` · shiqiang · fix: route KDA packed decode to the HCU operator kernel
- 020 `7d3e54a20ef1c9b979eefeefa05b62f50cb72ebf` · shiqiang · feat: support aiter gdn fwd op
- 021 `9287931edd297ecd3de37dd1ebe9186461093dcd` · shiqiang · feat: support triton moe wfp4 afp8 tc
- 022 `00468ade31979fcdedd2b1586b62219ad7c26dfc` · shiqiang · perf: route K3 attention-residual aggregation to the HCU kernel on large batches
- 023 `39dc297c2bbbade450c65638db9f7217456fb54d` · shiqiang · feat: optimize Triton MXFP4 W4A8 scaling
- 024 `38eea340e42d537d77f6d3a6f651de46e04b5ad5` · shiqiang · fix: correct Kimi-K3 INT4 MoE accuracy via slimquant quant Triton W4A16 path
- 025 `0d55c3dcc1e8c3213313e1350f67a5a7f5ec8f23` · shiqiang · fix: remove debug info
- 026 `c11fc79822819cd025b6a323d3862d1eba430dfc` · shiqiang · feat: add INT4 MoE W4A8 int8 tensor-core path for slimquant HCU
- 027 `288ab2445d9dcf379e188c9b036413ed7f767931` · shiqiang · fix: fix submit_nogds_read failed when use fastsafetensors load
- 028 `1535858cab7a8ea3fbcd843383f118ec572be8bb` · shiqiang · fix: refactor mxfp4 moe for activate 8 or 16 bit
- 029 `68d20891ff89194d21c6197b47a60a2d3e8b2f4d` · shiqiang · fix: refactor kimi k3 int4 moe for activate 8 or 16 bit
- 030 `3190629e74bdc60b933e9194ab1793e8a791fa1e` · shiqiang · fix: correct blockshape is none when use aiter moe for channel w4a8 model
- 031 `38cdca36fe2c8fb27b4ae5475fe99e9a65f12faa` · shiqiang · feat: support w4a8 hipc moe for k3 int4_w4a16 model
- 032 `e3a192d6680163ca7b37ec2c8db1cab2faea5408` · shiqiang · feat: update kda default value and support boltops chunk delta rule
- 033 `ef56737fc97e9776af54f296caad73ebb5779bb6` · shiqiang · fix: update w4a8 moe default backend
- 034 `c4212e94422f5cbe6d1519398ce7003b913a2f25` · WuFan · 适配kimi-k3-w4a8-huawei
- 035 `a6a8e0d5d5f6affea5b5ef4bf6dd06f2d98dbee6` · shiqiang · fix: read MXFP4 group_size from model config instead of hardcoding 32
- 036 `6de39e28a3e6cc03cfb88f71d1c3382dafe34427` · shiqiang · refactor(hcu): replace vendored KDA kernels with boltops
- 037 `238c81b4c22d9b8bd5270153625c68a47d75ad7c` · wufan · 优化因数据拷贝导致的性能问题
- 038 `b921d697a7bff3c5c30584a09bbf67670f1dd3c4` · shiqiang · feat: support contiguous and masked hipc kernel for deepep normal and low_latency mode
- 039 `88f910e94553c50210cd42e4a278c6afbba41b6c` · shiqiang · fix: preserve release fuse_swiglu with source Kimi MoE
- 040 `7164456e710950d6fea3d87a57723f5401f56656` · shiqiang · fix: preserve target lightop int8 quantization
- 041 `038467e8b2792441fffcd7dacc9edb0ef51574dc` · shiqiang · fix: fix lightop moe align vimfault
- 042 `93c930c6fe53ac8631d5c1dc762941dacfe903ea` · shiqiang · fix: fix aiter shuffle func not found
- 043 `5c6fb2e29f5211aac83ad24378771464baec4ac2` · lizhigong · Merge pull request #236 from daishiqiang123/20260825_v0.5.18_kimi_k3
- 044 `6ba4328e9f9878118259979b089bacb9445f0a92` · zhangyh15 · ci(hcu): release 发包流水触发方式改为 push
- 045 `5efbf9c3b043d7875e9e362ae4e967ea04756cbb` · zhangyh15 · debug: gateway 构建加日志排查 wheel 路径，临时只编 router
- 046 `700dd657dce63efa2b6869e337a8522705b5acdd` · lizhigong · Merge pull request #237 from HYGON-AI/zyh/test-release-workflow
- 047 `63610c2fff985eae31cbbc2b53d1851a1c27e941` · dddddddxl · ci(hcu): enable checks for release branches
- 048 `d39c77a236b81eb6206a915e2ba1d2cfd6b4303f` · dddddddxl · ci(hcu): use bash for wheel build jobs
- 049 `5f2c97ef71d6adce5c910a0a78054bb27dcdf3c0` · Lvjiajia · Merge pull request #239 from dddddddxl/hcu/enable-v0518-ci
- 050 `1db0108f3572f0b0cfa0d45ab93dbb058c22c991` · Crowder13 · [CI bugfix]工作目录没有清理导致权限变root (#240)
- 051 `779f1bea28fe63f4e56f1a86e3f1fd8d1b42eb84` · Crowder13 · [bugfix] CI打包版本更新为0.5.18 (#244)
- 052 `f292fa0e3536541ba4c7c8debc0d6b5304c75f9b` · dddddddxl · ci(hcu): trust wheel build workspaces
- 053 `91856a15449b4aeaf8dbc6c6ed73681b86d03f1d` · Lvjiajia · Merge pull request #245 from dddddddxl/hcu/fix-safe-directory-v0518
- 054 `1679428e000d016160df56e75177187730519a9a` · Codex · [FEAT][DSpark] 1. Support the p/d radix cache w/ DSpark by PR:https://github.com/sgl-project/sglang/pull/31097; 2. SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 & SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 & --disaggregation-decode-enable-radix-cache should be enable in D
- 055 `d8a06d0c23d998bfa56a9f49345b08406aee5fc1` · lizhigong · Merge pull request #247 from RyeYuan/feat/rye_20260827_deepseek-v4_dspark_radix-cache
- 056 `6721affc20c2400a021b3fe14bdf737953a653dd` · Crowder13 ·  [feature]适配hcu发包
- 057 `b57ecddf22c012f8f6f3637d27f04dd1f1255cff` · aivictor0901-tech · fix(moe): fix W4A8 DeepEP accuracy with SwiGLU clamp
- 058 `d92a60a7f0500b0d022aaeefaadb02f59c0e903f` · lizhigong · Merge pull request #250 from aivictor0901-tech/release/20260825_v0.5.18
- 059 `561186ca93c8630038574e78654e2a517cb8f8e7` · jiafei96 · kimi-k3 dspark support (#249)
- 060 `2ec9f04ae224b3d5a39bab76c17f7b11772a5f6e` · WuFan · [feat]: use aiter moe for kimi-k3-w4a8-huawei
- 061 `e769cde2e4c551dbe8275468763909d62cd0ce1a` · Crowder13 · hcu ci: docker 镜像 daily 定时构建, router 改用 vendored openssl, 补装 flash_kda (#252)
- 062 `42185e31622b92878b0d922a8a3b8a568c23cfce` · aivictor0901-tech · fix(moe): apply SwiGLU clamp in contiguous W4A8 path
- 063 `3cf5892012c37f06edb9f1e495a2cbb2b872038c` · lizhigong · Merge pull request #253 from aivictor0901-tech/release/20260825_v0.5.18
- 064 `b4b3aaee6e8f6bfe75965f70d1382e0a49e6cf28` · wufan · Merge branch 'HYGON-AI:release/20260825_v0.5.18' into release/20260825_v0.5.18
- 065 `8a65f5ecd8ec0ea9cac3add863bda1f18578d994` · wufan · Merge branch 'HYGON-AI:release/20260825_v0.5.18' into release/20260825_v0.5.18
- 066 `45c5fad8de90135528ff9e02e94a46f4f770774d` · wufan · Merge branch 'release/20260825_v0.5.18' of github.com:wufan-ToBeWater/sglang-das into release/20260825_v0.5.18
- 067 `52cddf2fa5a5f2a3a00b782f52af82b9659f87b4` · wufan · Merge branch 'release/20260825_v0.5.18' of github.com:wufan-ToBeWater/sglang-das into release/20260825_v0.5.18
- 068 `1cac982c2125bee1bda17c1a8472302dd66e9b74` · wufan · Merge branch 'release/20260825_v0.5.18' of github.com:wufan-ToBeWater/sglang-das into release/20260825_v0.5.18
- 069 `c16848ebe82f2466525eb44f4ca369d948ad4301` · lizhigong · Merge pull request #251 from wufan-ToBeWater/release/20260825_v0.5.18
- 070 `72dac21ce7211142fe7194f932a9cafdde7d6be6` · Crowder13 · [bugfix] torchaudio版本装的不对 (#257)
- 071 `83ab58f2eea0c36d0adf36127e313c0eb3389e89` · maxiaobetter · fix(quant): 修复 HCU W8A8 hipBLASLt 权重布局
- 072 `832ed3b143f38f9963fe5c62e723a1b3ee2623a4` · maxiaobetter · feat(moe): 增加 locality-fair static dispatch 策略
- 073 `143df0a3e380fe01f0faab1ef72876df4e55eb41` · maxiaobetter · fix(attention): 恢复 HCU DSA FlashMLA backend 路由
- 074 `ccfc91c5d9f09d62ed8dbdbc630fd40819dcea93` · maxiaobetter · fix(attention): 保留 IFB/MTP 的 DSA effective forward mode
- 075 `75cc77fa24542a42ce626dda6695b9172001dcd0` · maxiaobetter · fix(attention): 修复 IFB/MTP FlashMLA padded rows
- 076 `902de5d3d29246efa7c11e56d24a8cafa33bcd08` · maxiaobetter · fix(attention): 修复 HCU DSA FlashMLA metadata replay
- 077 `18b106418f859fcf1c0670e05ad548543293d303` · maxiaobetter · feat(attention): 恢复 HCU LightOp DSA indexer 路径
- 078 `0dd9ab21133e9afe603f89fcb5c0d382560f9d27` · maxiaobetter · fix(attention): 按 head_dim 归一化 DSA indexer
- 079 `43f1a8b2adfcb90d6d020e205c4e418692433262` · maxiaobetter · fix(attention): 启用 HCU DSA MTP draft-extend CUDA graph
- 080 `45d26d754c953f46146c3dd7f5defea06e693571` · maxiaobetter · perf(kernel): add HCU MLA query concat kernel
- 081 `1653683bdf5c9257135ac1d159b8d43b6e826bd9` · maxiaobetter · perf(hcu): port sparse page MQA with mask TopK to DSA
- 082 `94bacd49a0dd8c64d6cc6aebcb8d118df1bb7d45` · maxiaobetter · perf(hcu): reuse sorted DSA TopK indices
- 083 `f475e65692878e73715d11765fe1011274176099` · maxiaobetter · perf(dsa): compact shared index-K cache
- 084 `a306b802356126288e6adc96bc763e34d45784a4` · maxiaobetter · feat(hcu): port gfx936 INT8 DSA index cache
- 085 `f41bb0f58d63ab025634063c3138dda3ae76ed4e` · maxiaobetter · feat(hcu): support INT8 DSA cache in Mooncake PD
- 086 `3d12a3ce182fdf7738ed4f5027f2a515ca259675` · maxiaobetter · fix(hcu): stabilize DSA MTP metadata replay
- 087 `51e53affe8fbc28bfe50ac1f790e92c7e97f2468` · maxiaobetter · perf(hcu): reuse DSA target verify graph metadata
- 088 `8734656546ebd455e1ee39435de980fec791ad80` · maxiaobetter · fix(dsa): correct compact index cache handling
- 089 `6a51f2a38919c926b14e41f36a9a66a225165857` · maxiaobetter · fix(hcu): register SlimQuant W4A8 Marlin quantization
- 090 `8a11867b1b679a724ee546d40ffedab9c36da4c1` · maxiaobetter · feat(hcu): extend DSA LayerSplit for PP and HiCache
- 091 `0cf9a653eda76ddefec1c3044692433d1d033a36` · maxiaobetter · fix(glm5.2): restore raw DSA config fields
- 092 `afe5219d0291f927c68ad1d4466df5ba57724271` · maxiaobetter · fix(scheduler): use parallel state for PD decode sync
- 093 `0d243f4f13621187f96a9257983ecb1c39731293` · maxiaobetter · fix(hcu): restore async metadata copies and padded dispatch
- 094 `ed017ea38b728c7b7f0c973cb4b861fa34ec4c35` · maxiaobetter · fix(hcu): use LayerSplit scratch for fused index writes
- 095 `650a58304d9a689f3feaab739f049826818ff03e` · maxiaobetter · fix(hcu): skip padded tokens for legacy DeepGEMM MoE
- 096 `b001a8d21336b6824bb5960f031e4f1d1608f9f3` · maxiaobetter · perf(hcu): reduce TopK and DSA replay overhead
- 097 `dcbfd42307a20d006588b5b8762e1d73f94e9de4` · maxiaobetter · perf(hcu): avoid redundant padded weight mask
- 098 `9e3dd48343696446cae70ca6ba18b86aac468050` · maxiaobetter · perf(hcu): fuse DeepEP topk id cast
- 099 `5aedbbcc936a69e5dffff0a4ca5692b046c7e6df` · maxiaobetter · perf(hcu): overlap LayerSplit index cache broadcast
- 100 `5025165399aca0f6c482556c12bf53fbc4b870fd` · maxiaobetter · style: format GLM 5.2 forward-port files
- 101 `d8db8bb3e290dbc8d2a477b953bd773be809ae96` · maxiaobetter · fix(dsa): complete LayerSplit HiCache paths
- 102 `0669c184235c5ee070dbca485b9adf91b167bf4e` · maxiaobetter · fix(quantization): defer SlimQuant MoE backend resolution
- 103 `6e2f11552388438ba08c4da41d641e1f471a2310` · luosyx · fix(hcu): align speculative index cache lengths
- 104 `59fb959732955303211b872cd79e56c8fa50679e` · luosyx · feat(hcu): add INT8 indexer cache for DeepSeek-V4
- 105 `cff97568fd17b74e707f3552bafc924abfa175de` · luosyx · Merge feature/glm52-forward-port-v0.5.18 into release/20260825_v0.5.18
- 106 `dc1b54b42c2aaffe588e033d4e31bc83678f36e1` · luosyx · fix(hcu): tolerate deepgemm without contiguous HIPC kernel
- 107 `b065e598a744246970749810fd16e6619bec8c26` · luosyx · fix(hcu): restore slimquant backend logger
- 108 `962ab75e4ce3b851d7c952aaef73d175d6472ffa` · luosyx · fix(hcu): default to compatible masked W4A8 kernel
- 109 `253e326c3538a2fba5cc8bb05bff1465d7b3a4e1` · luosyx · fix(disagg): avoid duplicate KV transfer in worker
- 110 `11852ed1c2f6b2b26cb168a892c927efe95cbb7a` · lizhigong · Merge pull request #256 from luosyx/feature/release-20260825_v0.5.18-glm52-forward-port
- 111 `31a15956790d995884865015037f72f0c5f9e7b1` · shiqiang · fix: use the model's MoE topk for DeepEP RDMA buffer sizing
- 112 `ea6b50308e972d67bfbce1fb1eed9bbb019c38d5` · lizhigong · Merge pull request #260 from daishiqiang123/0.5.18_kimi_deepep_fix
- 113 `5f314b004bd74aea7a56775fea8ce29e13003931` · aivictor0901-tech · fix(dspark): transfer draft KV from PD prefill
- 114 `1b8d95c85b5a2442939ccc5abc0225b1a7e943f3` · lizhigong · Merge pull request #258 from aivictor0901-tech/release/20260825_v0.5.18
- 115 `845f99085a3841dde0fd75916e80b77d271739e0` · luosyx · fix(spec): handle zero-token verify results
- 116 `c3846f22236b400c9d7d9b17996da259b546e41e` · luosyx · perf(hcu): consume INT8 DSA index cache directly
- 117 `3a160688a388a5ddcb84d8bfe1ca2451df609dda` · lizhigong · Merge pull request #262 from luosyx/feature/release-20260825_v0.5.18-glm52-forward-port
- 118 `bf037e5280c579367cf02b80ea000a121730ea09` · shangxl · fix(hicache): synchronize L3 prefetch across PP ranks
- 119 `2d4eb83cc759a3e318b287cb37efc64bf8f8099c` · lizhigong · Merge pull request #261 from xinleishang/fix/20260829_v0.5.18_hicache_pp
- 120 `bf9245607d920cf11917f31eee665d4aae6f8e86` · dddddddxl · ci-remove-amd-status-from-pr-summaries
- 121 `6378905dc61607a8ad0f04db03b078379da14692` · Lvjiajia · Merge pull request #270 from dddddddxl/hcu/remove-amd-pr-state-v0518
- 122 `75a242b3ab043e57ac78035ee9214eba19329ca7` · xingjinliang · fix(attention): running Qwen3-0.6B dismatch shape of one step off policy
- 123 `84a65ccddc0ba1bc553012a5497bdc8b54069f79` · lizhigong · Merge pull request #271 from xingjinliang/fix/dismatch-shape-of-one-step-off-policy
- 124 `ddbbccde8cc167817faf2f643dfd79c6f4f9e6e6` · JingLee · feat(dsv4): support PP/DSpark and CP-aware sparse prefill on HCU (#274)
- 125 `685051c479c4ae124051c9b2d89a05a26b4f44c2` · YuanYuan · [bugfix] removed the http warning info in the skip-warmup mode (#276)
- 126 `867eeada17adcfe2cbfcd6a0da66cc3b643560f4` · chaoyujiang · fix: gracefully handle unsupported KV cache schemes (#268)
- 127 `c18c8d4bb72a97c17384da51d1a225fb2b573983` · Yongqiang Duan · compressor 通信优化 RLC（all-to-all + 本地压缩 + all-gather） (#275)
- 128 `8ae13700deb972d74166d0eb8a1155b095e865f0` · GarrySnail · fix(hcu): port PD disaggregation accuracy fix to v0.5.18 release (#266)
- 129 `1316e9887593219d484dcc21ff00a70c1f0baff2` · lipenghui · fix(hcu): enable W4A8 DeepEP aiter path and FlashMLA workspace reuse (#265)
- 130 `6614bebc76522ce9092218165b8c62ec35488b9f` · lizhigong · feat(hcu): add BHSD KV-cache support while preserving legacy HCU FA (#282)
- 131 `4fea017b14ad789d2ffecbf29e3776e5f1a6319f` · maxiaobetter · fix(hcu): sample linear MTP target verification (#280)
- 132 `0e2a6e743b8e9338b911f0717e98548ac0aa90ee` · Wenbo Zhu · Feat/bolt recompute w u (#278)
- 133 `556337109e8c9a57b7054e2f5b516e25c313b22b` · wufan · [feat]:用lightop的gemm_w8a8_smooth代替triton实现，提升性能 (#279)
- 134 `0ce4bc71646bca9ba590979a3e98a95470a90bad` · dingxinlong · [CI] Use v0.5.18 HCU image and update model paths (#287)
- 135 `f5a24dc843be70821c51186c22c4211b7326646d` · Crowder13 · [FEATURE]0.5.18镜像新增安装nixl (#283)
- 136 `3b678f29313c946e478e9e2dbf4344cbe7f6b34d` · Crowder13 · [BUGFIX] HCU环境不需要依赖flashinfer  (#289)
- 137 `ab91a15c88647041640a0a21bc93242d3b458673` · YuanYuan · [bugfix] 1.fix the accuracy issue in Decode radix-cache enable due to… (#291)
- 138 `aa89102b1dacca122aec18d43547cb86e979d217` · liuzj512 · fix(hy3): align MoE forward context with current API (#292)
- 139 `2a1b589b3ec390614239fbd6b0d826536db5697a` · lizhigong · fix(hcu): support BSHD multi-batch varlen attention and PD disaggregation (#288)
- 140 `f8bfbc60fbb9400391ab7cbf8f1c725bda5c2b59` · GarrySnail · feat(hcu): Qwen3.8 flash next open support (#281)
- 141 `b0f501598e0633f2bcbc4e069de37f52a7a2ec83` · YuanYuan · [BugFix] fix the bug that certain chat/completion in DSpark goes to C… (#294)
- 142 `f656a2ed25cdedbd8b760e98b2b0ea85e5cdf8ae` · YuanYuan · [BugFix] fix the bug by removing old monkey patch of mhc (#296)
- 143 `2b855b5be60f838f72571caf8b01be238b209748` · dingxinlong · ci(hcu): align llguidance and model asset paths (#300)
- 144 `20c5f927fd804b4c0e73436d1141c68bca7013a6` · YuanYuan · [BugFix] Remove the dependency of prefill radix-cache with decode-pre… (#302)
- 145 `9153913d619d31597b77c205453eadb5d2f355f1` · Wenbo Zhu · fix(speculative): remove unsupported server args (#304)
- 146 `c2544d8ae1a60d2b91ba275ed997bf1da4e1e833` · Jane · fix(hcu): repair two upstream-refactor call-site regressions (#303)
- 147 `71117e8d988a4a0eba516d007cee1f91eadc563b` · GarrySnail · feat(hcu): qwen38 flash next int8 w8a8 support (#310)
- 148 `f35668e41c964583e7b0e0761f311c2893d31b9d` · JingLee · [perf] add opt-in LightOp KV allocation kernel (#307)
- 149 `4c9d666cd36e3aafa5c584f7ff0188877460bb1d` · daishiqiang123 · fix: fix deepep and dspark conflict (#306)
- 150 `44b7bc84619f5a87c239a586ef7fe10cdab0abf4` · YuanYuan · [BugFix] support the MegaMoE func in DSpark model (#309)
- 151 `ff5ffaa1a5b662fb48d2895f414c7a3adcd845b6` · GarrySnail · fix(hcu): keep DeepSeek W4A8 Linear quantized while Qwen3.8 uses expe… (#312)
- 152 `a12f08de78065aa75f5fefa6e6c17c738d022e53` · Jane · Hy3 support sp ep (#311)
- 153 `3bc09034bad8c7679d3eaaa3772669c6126ebcc5` · chaoyujiang · feat(bench): add --gsp-input-ids to skip server tokenization (#315)
- 154 `cdea7a5dbb96b24f80d6330e53c06000ace56e65` · Wenbo Zhu · fix(hcu): use available RMSNorm kernel op (#321)
- 155 `ba2dec637cc60db20a61f27f5ea04daa61db8e51` · lzb16 · fix(fst): reclaim device cache after each weight batch (#322)
- 156 `4c367dbd52b3da177ec12f8bee9f6a5b36eda06f` · JingLee · feat(dsv4,hcu): support BF16/FP8 and INT8/INT8 indexer paths on gfx936 (#323)
- 157 `cd8f78f01f9b6cc9f379e7393a369f378eab35c2` · maxiaobetter · fix(hcu): include MLA absorb q kernel in HIP build (#293)
- 158 `056b21d829f1408d184fc411e1155c7be9d5fa10` · daishiqiang123 · opt: optimize situ_quant op form lightop lib (#324)
- 159 `c338e5fd0e5125d886c53438d42e2fccfd10f3cf` · JingLee · perf(dsv4): optimize EP gather, paged K-cache, and mHC CP split (#327)
- 160 `47b97c0f77e08a5b2287a5f6d0330dec5ccab090` · GarrySnail · fix(hcu): avoid gfx936 QSA LDS overflow (#326)
- 161 `700649f36306ad88b7322362f3262154b7b3adff` · chenyuege · 【Feat】Qwen3.6‑35B‑A3B supported dflash2 (#331)
- 162 `57beb6b4ae453725e0c02b58bb5b3d93d737aa4a` · lizhigong · ci(hcu): align PR wheel version with release 0.5.18 (#295)
- 163 `e2c65d880dcd94e9397040b2b1bcdda38ac73e46` · JingLee · refactor(dsv4): use categorized LightOp mHC API (#332)
- 164 `e6540308c53b7099feb893050d56c1793a3962f1` · lizhigong · fix bug: Refactor is_fp8_fnuz function for clarity
- 165 `b10bb8c6eb52cceb55647620d7e0e9e4e711dc3d` · YuanYuan · [FEAT] support the dspark megamoe-int4 by packing signed-int4 to int8 (#333)
- 166 `6d59319229dbc44db4ed7a13a1af8ffda67d14f2` · lizhigong · test(hcu): derive FP8 quant reference bounds from the output dtype (#337)
- 167 `551a71d4d6966366e76378c06be035079ef96963` · liuzj512 · Feature/hy4 ihc tilelang 5commits (#336)
- 168 `ebe9c77f2189d2e0982deb37a88a1f323dda191c` · Crowder13 · [FEATURE]同步编译python3.10和python3.12, 上传whl包后记录信息至dashub (#338)
- 169 `a494caa1e7974a8ec960c526838f476fd93312c8` · Yongqiang Duan · perf(hcu): use AITER IPC collectives for DP TP-MoE (#340)
- 170 `4cdd9d36e79a61e736bc6712916a23617919fbfa` · wufan · Release/20260825 v0.5.18 (#339)
- 171 `4603df038bccfa86a42db00202c1cd25c8d6529b` · chenyuege · [feat]Qwen3.6-35B-A3B 支持--speculative-algorithm DSPARK (#341)
- 172 `cb50fb561aa174070f79fa98f7bd1638660970d6` · daishiqiang123 · feat: support mimo v2 flash channel fp8 for 5.18 (#343)
- 173 `77ba00139ac13b8a6bdb9ee15554cd9c18537fbc` · daishiqiang123 · fix: fix pd fake prefill mode and attn_tp_size bug (#345)
- 174 `fa95ae0846a943f9a740c56a3dee4d42279d4728` · daishiqiang123 · fix and opt: fix dspark eagle error and optimize attn tp comm from ai… (#342)
- 175 `050129f5b3406b3b003378dccd3009433df76d7a` · chenyuege · [bugfix] Isolate the use of --speculative-algorithm DSPARK for the Qwen3.6-35B-A3B model family via the SGLANG_USE_QWEN_DSPARK environment variable. (#352)
- 176 `44d9bdc2e91331a912ac4d613381c8f044893dd0` · GarrySnail · feat(qwen3_8): support AITER EP and INT8 PLE offload (#350)
- 177 `647229c6e5adaf3f2dda3515671a48c45f58f825` · dingxinlong · ci(hcu): pin v0.5.18 image fallback and shared eval wheels (#348)
- 178 `4af5b4974cc38a4e57896270d069b526e2db00ba` · daishiqiang123 · Mimo v2 pro update and bug fix (#354)
- 179 `4a7532fefd86c6a5f0089a3b6b40eeb1f3ed2119` · chenyuege · fix(hcu): keep DFLASH FP8 sliding attention on native FlashAttention (#356)
- 180 `feff664dfee687c0238ee95f5980467334093f29` · liuzj512 · Hy3 sp ep lightop clean (#355)
- 181 `74ee9054e84ccb9c33007e5726a93d16b248ea37` · dingxinlong · ci(hcu): fix v0.5.18 PR runtime consistency (#363)
- 182 `f2379b09c820f2ad5e9217af1c8a21a20b7b3d73` · Jane · feat:Hy3 support sp ep (#360)
- 183 `5077850b7b646483abe7bd2856ff79d5062ff335` · jiafei96 · support fabric aiter custom allreduce (#361)
- 184 `5817f8359cc0e12df00566e820ed19cccb589d55` · GarrySnail · fix(qwen4): correct INT8 PLE dequant on GPU path and enable AITER EP offload (#365)
- 185 `72df69c8087c51d4ed237c219836a1d0df4d274e` · Crowder13 · [BUGFIX]缺少token声明导致打包信息无法推送至Dashub (#366)
- 186 `86cdce034cbf09a90b7a2a3328174d09634a660e` · dingxinlong · ci(hcu): align v0.5.18 wheel builds with torch 2.11 (#368)
- 187 `45b20a85cb28b0f3cbdb71c3d5ca233c8cb63901` · Wenbo Zhu · Fix ValueError: Add type annotations to fuse_situ_mul_quant_ep (#369)
- 188 `7c7da44a797df5bb370a1789ab7f0aeef55b5530` · dingxinlong · ci(hcu): fix stale v0.5.18 PR test paths (#370)
- 189 `35d915ef222de3b2839bb55c1afa9c99e3a24941` · Fafap131 · feat: support hy4 cp (#358)
- 190 `0f88700ac87668fe19d35f214773ea827a049a8c` · xinleishang · feat(mem-cache): support direct L3 external linker cache without test file (#373)
- 191 `37f23f55a0a348e10240369a3c8d112a62efed35` · Yongqiang Duan · perf(dsv4): auto-dispatch BF16-to-FP32 GEMM backends (#375)
- 192 `b04678a944551b4af5db812e652f36a53eaa89d8` · GarrySnail · fix(qwen4): load INT8 MTP fc projections with ReplicatedLinear (#378)
- 193 `b80d0d1bb6a942853b242ecc38ef3be3db2575d2` · yangyn · [fix] 修复w8a8-fp8-v4pro精度问题 (#380)
- 194 `7df099eb6085eddb781531719bc2af24ff3cf204` · maxiaobetter · feat(moe): 增加 EP 绝对均衡性能测试开关 (#383)
- 195 `2bb5e81e5353c5bc40594d42bd80366b91af15bb` · maxiaobetter · feat(spec): 支持节点内 Draft LM-head 词表并行 (#385)
- 196 `4f224d2b67e3e542d720cc88fc1aa53f6a2c8e49` · JingLee · perf(dsv4): support paged MQA logits for prefill (#389)
- 197 `6bf8197ddaf145e26b22b54a494a0ad7ff2b90aa` · JingLee · Revert "perf(dsv4): support paged MQA logits for prefill (#389)" (#390)
- 198 `f6a092ec7c1693600de5bf530255a91771b00e1c` · GarrySnail · fix(qwen4): flatten MTP fc_hidden input before quantized GEMM (#391)
- 199 `07214443be3c398eab3f874df214cbadf256ae03` · GarrySnail · feat(qsa): use FA sparse GQA for HIP chunk-prefill extend (#392)
- 200 `f4d13c6b88c6a3494b1036d62de54f6849d73eed` · chenyuege · [Feat]Qwen3.6-35B-A3B DFLASH性能优化 (#388)
- 201 `2df70449a7a99196cb2a9497c5f59e57dd556aca` · maxiaobetter · feat(layersplit): 支持单层 MTP Draft KV 分片与缓存复用 (#387)
- 202 `7db5ec1771b90e54071f21394cb3df502af98880` · maxiaobetter · feat(lplb): 支持 HIP 静态 LP 副本概率分发 (#386)
- 203 `98e92353399b9cf708523b02834b4894a3aa3321` · maxiaobetter · fix(glm53): 修正 FP8 权重映射并优化 HCU Decode 路径 (#384)
- 204 `c4ef6d3a48dfc0f79bf13871f3c4b028199a7c95` · maxiaobetter · fix(dsa): 修复 MQA 显存预算与 CP/MTP 索引补齐 (#382)
- 205 `055812e0daa35545c9167f588f9b34c29047d93b` · Yongqiang Duan · [Perf] DSV4：融合 Indexer Q 侧 INT8 量化，减少冗余量化算子 (#396)
- 206 `eb4f0ad89390fc039a5943f241a5a7421bad2a1c` · yangyn · [feat]适配 w4a8-int8-megamoe (#394)
- 207 `fc7ebb3d09b6d29684a1c9a237a1fcfac48075f8` · leichuan · [Unified Cache] Improve Mooncake direct I/O, observability, and reliability (#398)
- 208 `2ed02ec34d6cd61d0c28f55f24f82b30099a706d` · JingLee · feat(dsv4): support paged MQA logits for prefill、support and optimize CP LayerSplit cache with Mooncake direct L3 (#395)
- 209 `16e9b386740e30e4879c696db587db58a2eeeabf` · chenyuege · feat(hcu): support BF16 KV for Qwen DFLASH2 attention (#405)
- 210 `ccc15a6f3ff01d34f7b40441cbb7d29246e39003` · leichuan · feat(cache): integrate Mooncake ReadPlan and optimize page-wise loading for direct linker (#404)
- 211 `c496509fbbf73b0acb7875652cf640791a778294` · lizhigong · fix(mmgen): declare MiniMax H3 request extensions in the OpenAPI document (#399)
- 212 `98ba405a238305be3bd2a483216aa5fb0bb2d47b` · Wenbo Zhu · fix(aiter): w13 权重 shuffle 改回 gemm2 布局，修复 w8a8 变体乱码 (#401)
- 213 `4c219c5f307b655b42b198737229c391f5988900` · daishiqiang123 · opt: opt dense moe ag and nonzero bubble (#406)
- 214 `602fb4a30f93bd40a5eda61e5bc02906fc8d4825` · YuanYuan · [FEAT] Support the BCG with CP-V2 and expect to improve TTFT from 10%… (#407)
- 215 `05c58e82faab83b6bf906cfd70f2845eaee1df6c` · maxiaobetter · fix(dsa): 修复 gfx938 72CU sparse MQA 路由与 HCU 判断 (#409)
- 216 `8eb6acd22bca671c36219947cd25df4be1ef8ef2` · daishiqiang123 · feat(moe): support EPLB for Kimi K3 with DSpark bug fix (#411)
- 217 `2b8d2b04a5a730f2b9031342af00932f583003f1` · GarrySnail · fix(qwen4): support channelwise FP8 and INT8 PLE ngram embeddings (#413)
- 218 `50beda6a90ba4dd7c85d667530c2614b13122fbf` · Lvjiajia · Update quality gate workflow to use stable version
- 219 `31262cd68826a1f3bf8ef186dd692d2b3500b036` · daishiqiang · fix: fix deepep num_rdma_bytes count correct (#437)
- 220 `90f80c6bd5a831f40c29ec8a32200174e749de9f` · Wenbo Zhu · feat(moe): 增加 aiter moe_c 后端选定环境变量 (#415)
- 221 `2df2a10f818a0bf88a2a0b6ac97a3dd919b608f6` · Phy · fix(mmgen): align MiniMax H3 OpenAPI contract (#426)
- 222 `6c0330d8d877d146412e65e5cf6bf717a3004dae` · lipenghui · fix(w4a8): 修复 HIPC MoE scale 重复乘 16 与 HCU RoPE 启动越界 (#439)
- 223 `2f93be6c5e0db8a54fa76d10eab1887e7fcfd145` · daishiqiang · fix(kvcache): handle zero full iterations for asymmetric KV cache (#425)
- 224 `066a95af77c479b644e4477afc42b39246f7f58a` · daishiqiang · fix: fix lightop situ import regression bug and deepgemm w4a8 op interface change (#428)
- 225 `c3bbbd8074795a43345da8ff18e11153bc0424af` · GarrySnail · fix(hcu): serve packed channel INT4 MoE and route QSA MQA through BoltOPs (#434)
- 226 `ea1c297dc23343cbe5c6b619f3094325fb5fe45b` · Yongqiang Duan · perf: 为 HCU gfx936 DeepSeek-V4 增加 INT8 Persistent Paged MQA Logits (#436)
- 227 `6086af28ddb97c1cc4c9cf22fadd2940a1100105` · dingxinlong · ci(hcu): migrate v0.5.18 workflows to organization runners (#465)
- 228 `10053303694ba99167f7e17c585757fe3cf851bb` · ybhe1110 · [Fix][Unified Cache] Retry external KV abort cleanup after forward and source drain (#455)
- 229 `1828115e08aba1659f71c5da67f761343e3553f4` · jiafei96 · fix w4a16 moe asm error (#462)
- 230 `4ba29d557c3a31a852cb7fa4fc0979630c36fff8` · liuzj512 · [HY4][HCU] Enable fp8_e4m3 KV cache with learnable attention sink (#418)
- 231 `8effe13508725f3c871dceb5dfbfbbfc6f959a89` · WenQ7 · feat(minimax-m3): optimize prefill and decode on HCU gfx938 (#453)
- 232 `3854f77546c1a65ebc26d86ab84f2ebd96740cc5` · daishiqiang · opt(kda): update kda verify decode op from boltops (#458)
- 233 `08ea4dec71b0244d41b646c32dd322c9c234fd41` · zhengzhi · fix(dsv4): resolve Flash-0731 W4A8 API and general_fc accuracy issues (#450)
- 234 `76e87a951d7a7423cf9686ef4109f98d25c2a0cf` · lipenghui · fix(hcu): serve DeepSeek-V3.2 INT8, FP8, and BF16 in sglang0.5.18 (#466)
- 235 `e885e7223180a8ddc2c9aa753a4c884cfc9ba72c` · wu ju · fix(hcu): restore ForwardBatch RMS quantization flag (#469)
- 236 `cfa865d339c0d7191dd044e6980006be799a570a` · chaoyujiang · fix deepgemm masked fp8 api upstream (#472)
- 237 `f315216140b16b55ca249cfb07516128fa0949ef` · zhangang1221 · fix(hcu): enable EAGLE draft decoding for DeepSeek MLA (#477)
- 238 `b7e4eae1243fb6de06507484afaddbc1715ee054` · Wenbo Zhu · fix: remove alt_stream overlap copy when writing KV cache (#478)
- 239 `7fa7729c1d9a3007baf471184ca709aece7fdbed` · leichuan · fix(cache): harden Mooncake ReadPlan and PP direct-L3 cache loading (#482)
- 240 `2cc1bcc28c5c4b6e129ae4ce3dadbe3fbc2796cf` · maxiaobetter · fix(deepgemm): 统一 INT8/FP8 权重重排与 FP8 masked GEMM 接口 (#484)
- 241 `723ac998c384e495b66f57119382a3434d2d4152` · qieshiniao · 修复 GLM-5.1/5.2 DSA 服务兼容性并优化 CP decode FlashMLA (#475)
- 242 `2bf2890f8830987c79057b5784515cbe21fdb3e5` · lipenghui · fix(hcu): correct DeepSeek-V3.1 FP8 decode under CUDA graph (#483)
- 243 `a06c34ef7511fbf767d1e8f8314546492d5efb0e` · lizhigong · fix(ci): stage-b include 列表容忍分支差异的测试文件 (#485)
- 244 `6ca7a1bd6a1e7f4719171e4afeb257e4507d3061` · maxiaobetter · fix(test): 修复 unified attention 单测 KV 索引构造 (#489)
- 245 `b3632f0be3bf48b1bf48d0444e4be4f1ee4bc63b` · liuzj512 · fix(qwen3_moe): quantise MoE gate for compressed-tensors FP8 checkpoints (#492)
- 246 `1882acd8c659b9071ad341741e7b5beef1c40ab0` · maxiaobetter · fix(moe): restore DeepEP normal combine for HCU groupgemm (#491)
- 247 `bfbff7f68e024ed71aaec1054c1bdf8ad6df7adf` · WenQ7 · fix(hcu): preserve INT8 MoE fallback and restore AITER W8A8 compatibility (#487)
- 248 `966f6da1bcf76359bbbd3f6919f9da6f83ae905a` · chaoyujiang · fix(hcu): avoid double packing channel FP8 DeepGEMM weights (#497)
- 249 `e1eaf08298435a45133f48366cd75d2ccfd03ecb` · maxiaobetter · fix(moe): guard SBO setup for flat DeepEP dispatch (#501)
- 250 `f106c53608df6269af7296265f9097e9a80abbd2` · Yongqiang Duan · perf(hcu): 优化小 batch W8A8 INT8 GEMM 的 Tiling 选择 (#502)
- 251 `2ac651d8f6b78dcc8214119e733c4ba55be15945` · maxiaobetter · fix(w4a8): 兼容 GLM5.3-w4a8 模型的 compressed-tensors 打包权重 (#506)
- 252 `c9a84fca9624adb32d1848a05298a39ca6f0409f` · daishiqiang · [Perf][HCU] Optimize Kimi K3 decode fusions 20261010 (#507)
- 253 `49bb75c791c10b2114c52c58c8d0433bd03e0856` · zhangpc3 · fix(diffusion): upload all video outputs to cloud storage
- 254 `aebf1a9f41ad5fd1e00ac1570e6b9e6a8128b237` · chaoyujiang · fix(hcu): fix SlimQuant INT8 DeepEP routing and DeepGEMM weight packing (#514)
- 255 `018e5799abb14aa90d05af042db8b273bebc5df9` · Wenbo Zhu · Fix/hcu aiter moe inplace (#517)
- 256 `71b2ee4ea67b6be712a5575ed1dd0950607fc6b5` · daishiqiang · perf: add opt-in asynchronous prebuilt token H2D (#516)
- 257 `9dd4e2e80065ba556c1beb1ffd8b1f2e29bdb085` · FrankXu0908 · [diffusion] attention: fix HCU Wan SLA and FA family admission (#519)
- 258 `3b5ca6e335086811dbc412c69d77226bad8f2722` · GarrySnail · fix(qwen4_exp): BF16 PLE offload OOM, channel-FP8/INT8 aiter accuracy, and PLE/MTP decode graph fixes (#513)
- 259 `3604cec316cca9c1b9fb8da2e5d1a5a7d59e92a2` · maxiaobetter · fix(hcu): 修复 GLM W4A8 target 的 BF16 MTP 加载与 DeepEP 通信 (#520)
- 260 `b290dcf39562e7ae6ed5f1646bbd20d1e1d8c7fc` · chenyuege · [feat] 优化 HCU 上 Qwen3.6-35B-A3B 使用 DFLASH 时（--speculative-algorithm DFLASH）的性能 (#522)
- 261 `5f31aec46b5e557fc00e891e8d91bcdb2e4c6b3b` · WenQ7 · fix(aiter): avoid duplicate MoE weights outside MiniMax-M3 (#526)
- 262 `40c9ec29b31d5a160c8d83d56949c58e76cc6c26` · maxiaobetter · fix(scheduler): keep DP health probes on the generation path (#528)
- 263 `0ac3b2f83eb8c7a57f534810c035dc0feb973cd3` · s1mplezpc · feat(MiniMax-H3): support MiniMax-H3 channel-wise INT8/FP8 inference on HCU (#532)
- 264 `ea3acdbe95df8893fe41b06c1fcf1477cb46d85e` · JingLee · feat(dsv4): add opt-in paged MQA logits row chunking (#536)
- 265 `5b630225295b1bc1e510cfb5fd25e8f9521acef8` · lipenghui · fix(hcu): keep packed FP8 Q on the old MLA path for Kimi K3 (#537)
- 266 `c2666f4b5c07e017ad9195c7435236517cda5f08` · yangyn · [fix] 修复cp 逻辑：解决实际 token 不被整除时不走 cp 的问题 (#538)
- 267 `3c2b5266717c24b831160a3c2f8c09b7f765d313` · Songge · fix(GLM5.3 w4a8): skip unused packed expert shape metadata (#539)
- 268 `6fd9d5644212bf13a4b5394a6b5a2007fa49e68b` · chenyuege · [Bugfix] 修复Qwen3.5-122B-A10B-Channel-FP8重复输出的问题 (#545)
- 269 `6fe59ce1ab68fb4ae69951502b621daa5cde0c95` · Crowder13 · ci: release-pypi-nightly-hcu 多版本编译 (dtk26.04/py3.10+3.12, dtk26.10/py3.12) [rel/20260825_v0.5.18] (#544)
- 270 `23d1def83c9e84999ca0803543f4438a882432bd` · YuanYuan · [FEAT] support megamoe with deegemm in deepseek-v4 (#525)
- 271 `ae065131d82c45bf7135761091fb77b718c5df66` · daishiqiang · feat(kimi-k3): add opt-in serial shared-expert overlap for DeepEP (#540)
- 272 `da90edf068c9a574c3a9e3e7ad69229440edfa01` · daishiqiang · fix(k3): fix aiter ar shape mismatch and some entries accept rate=1 (#546)
- 273 `7586e89f5e06e1ff2a8fb8c0f5a8e8ab3c66bab5` · GarrySnail · fix(qwen4-exp): calibrate QSA e4m3 KV scales during graph capture (#551)
- 274 `0bb71f0c7fd7faeab15402d9a2d5f6759f623120` · yangyn · [fix] 解决 w4a8 aiter moe 不支持 swiglu_limit 功能 (#550)
- 275 `b9e0052e9e35ae3aa58d73ed685ae6191d64fc24` · GarrySnail · fix(hcu): keep the w8a8 smooth asm fast path off on gfx928 (#555)
- 276 `a031c979a4a848bb6905386808d771b5782125e1` · liuzj512 · [fix] 修复非 AITER 路径下 compressed-tensors W8A8 FP8 MoE 权重被错误补齐导致输出乱码 (#564)
- 277 `030f6254f2d9d4abe9e57543e3f38757d12ec4ba` · daishiqiang · perf(attention): add paged split-KV support and opt-in DSpark V2 (#553)
