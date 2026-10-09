# SGLang v0.5.21 release 更新总结

## 版本口径与来源

- 官方 tag：`v0.5.21`，实际提交 `e00930c5489053f26d86b179cee0d087f846acbb`。
- 发布时间：2026-10-02 09:09:04（北京时间；GitHub `published_at=2026-10-02T01:09:04Z`）。tag 指向的提交时间为 2026-09-30，提交时间与发布页面时间不同。
- 本文依据[官方 release 页面](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)、本地官方 tag 对象与源码整理。官方页面统计 779 个 PR、227 位贡献者；此统计不等同于 Git revision range 的提交对象数。
- 这是完整 release 总结，包含 9 月 28 日同步之前已经进入 main 的更新，不能全部算成本次新增。此次 daily sync 的范围另见 `sync-official-main-20261009-summary.md`。
- tag 是独立 release 分支的提交，不是本次 main tip；与 main 的共同祖先为 `bd780950300a0c236509352bab44d8a8a98f36c2`。release 上另有 InternVL/SentencePiece 与 RDMA 检查两笔回补，main 中有对应原始修复。

## 模型新增与优化

| 模型/模型族 | release 更新 | 主要证据 |
|---|---|---|
| DeepSeek-V4.1 Flash | 文本/视觉模型及 runtime 全链路支持；Engram、历史请求、vision tower、chat/tool parsing；候选 indexer、TopK、compression/KV I/O、RoPE/FP4、mHC compensated projection、Hopper FP8 kernel；候选块上使用 DeepGEMM 为 prefill index 层打分 | #38798、#39646–#39677、#40352 |
| DeepSeek-V4 | indexer MQA logits 按显存预算分块；按 compression ratio 统一 metadata、sparse prefill 与 KV pool；C4 state ring、ratio-2 pair-state、FlashMLA 物理页 padding 的内存预算修复；TileLang 缓存纳入 SGLANG_CACHE_DIR | #39095、#39364、#39921、#40337、#41048–#41049、#41090–#41091 |
| GLM-5.2/5.3/5.3-Flash | H200 W4A8 MoE 调优；mHC attention→MLP 融合、KDA projections/prefill metadata 融合、KPool 同步减少及 indexer overlap；ReplaySSM、混合 attention CPU offload/PD mapping、DeepEP v2；PP/MTP、forget-gate shape、GLM-OCR embedding/position 修复 | #38220、#39088、#39200、#39378、#39688、#39695、#39720、#40310、#40466、#40517、#40607 |
| Kimi K3 | expert weight 匹配改为 O(1)；PP prefill/DCP decode/DSpark 集成；支持 DFlash；模型专属优化命名空间收口、packed projection mapping 修复、GB300 collectives 配方 | #38805、#40045、#40794、#40922、#41164、#35330 |
| Qwen3.8 Next / Qwen4-Exp | PLE gate/convolution 的 target-verify 融合；小 CUDA graph 输入 copy 融合；PP serving 与 PD-prefill MTP；Qwen MoE 接 DeepGEMM MegaMoE | #38080、#40041、#40501、#41166 |
| GigaChat 3.5 | 新模型及 reasoning 配方 | #29189、#41118 |
| IQuest-Q1 | 新模型和 MTP draft | #41590 |
| MiMo-V2.6 / V2.6-Pro | day-0 支持，BF16 router/MXFP4 专家量化集成 | #40448 |
| Ling-3.0-flash-VL | 多模态模型支持 | #38526 |
| DiffusionGemma | diffusion language model serving 支持 | #34061 |
| MiniMax-M3 | AMD MXFP8、FP8 index-K、wave64 TopK、shared-expert fusion，以及 HiSparse MHA | #36574、#36549、#36560、#36576、#31446 |
| Gemma4 / Mistral3 / DeepSeek-OCR-2 | tied lm_head 修复；vision tower 不再为读取一层保留所有层；OCR 官方 768px local crop geometry | #35809、#39185、#38996 |
| Nemotron / Step-3.5 / LongCat / Falcon-H1 / Bailing NextN | residual 与 deferred reduction、aux hidden capture、DP attention 下重复求和及 draft 层数修复 | #40800–#40801、#40799、#41082、#41433、#41194 |
| Qwen-Image 2.1 / Anima Base v1.0 / Ming-Image Design、Design-Layer / FLUX 3 Action | diffusion/VLA 新模型；FLUX 3 Action 支持机器人 policy | #39983、#41011、#41067、#41066 |

官方报告 DeepSeek-V4.1 长 prompt 首 token 延迟改善约 22%，Kimi K3 PD prefill 吞吐提升约 20.6%，expert-name 匹配时间从 21.45 秒降至 0.13 秒。这些是官方特定设备与 workload 的结果，不能直接作为 HCU 的实测性能结论。证据见[release 页面](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)与对应 PR。

## 组件更新

| 组件 | release 主要变化 |
|---|---|
| Layer boundary / 并行运行时 | communicator 拆包、独立 attention/FFN stage，残差状态跨边界传递，consumer 选择 reduction fusion，aux capture 在 residual read 完成；修复 PP send、dense TP1 shared expert、DP×CP gather 下 deferred FFN 重复求和；parallel getter 退出 distributed 公共导出，迁向 get_parallel（#40342–#40345、#40868、#41079–#41083、#41193–#41195、#41422–#41436、#41547–#41557） |
| PD / DCP / KV-Shard | 在线 prefill/decode 角色切换；KV-Shard sharded pools/control plane；可选 transfer checksum、自定义 backend decode host receive、DCP radix+HiCache；缓存前缀传输按 pack capacity 分段；失败/abort 时保留 ACK、排空传输后释放 KV，默认 deferred release（#28403、#37615、#38468、#39500、#39731、#40238、#40263、#40376、#40645、#41023、#41402、#41404） |
| Speculative decoding | PP×EAGLE/MTP；XQA verify；内置 draft window attention；DFlash LiLiCorr；DP prefill 期间保留 speculation；GDN target verify 避免物化 QKV；修复 sampling CDF boundary、DSpark graph stream 数量、mixed chunk prefill 与 DP coordination（#30775、#32269、#32673、#33778、#35798、#37462、#40118、#40658、#41179） |
| CUDA graph | 修复 prefill runner/replay input 合同、捕获 KV counter 生命周期、DSA pooled-indexer prefill bridge；DSV4 C4 BCG 降内存；OOT graph backend 与 eager logits hook（#35452、#36534、#39175、#37969、#40222、#40851、#41311） |
| MoE / DeepEP / MegaMoE | DeepEP v2 支持 BF16、batch invariant、MXFP8 和 deferred route weighting；Qwen MXFP4/NVFP4 MegaMoE；clamped SwiGLU、0-token CUTLASS、top-1 非单位 scaling 修复；精度原因默认关闭 FlashInfer fused finalize（#38080、#38160、#38780、#39939、#40030、#40105、#40187） |
| Quant / kernel | SM100 NVFP4 GenMHA/spec KV；32-wide-K ue8m0 block-FP8 接 FlashInfer MXFP8 GEMM；ModelOpt mixed precision/PP shared expert；speculative CUDA/ROCm kernel 迁 JIT；warp copy alignment、TopK cluster fallback、KDA transpose/beta/fence 修复、mHC verify 融合、Mamba2 launch 调优（#36340、#38726、#40039、#40628、#40033、#36176、#40163、#39680、#40685、#39124、#40208、#41223） |
| Radix / unified memory | unified radix 默认 Rust core；移除旧 SWA/Mamba/HiRadix 与实验 C++ tree；统一 shared byte budget、KV row free/protected prefix、component eviction cursor；agentic tail-aware LRU、Inkling physical-slot 修复（#39627、#40313、#40775、#40787、#36729、#38941、#41312、#41276、#34012、#41144） |
| HiCache / storage | host pool 自动 sizing、MHA device-row 宽度、MXFP8 scale backup；PP prefetch tickets；buffer storage pipeline/retry、existence bookkeeping；unified page load-back JIT、sm90+ TMA transfer；SWA/Mamba write-back demotion、批量 backup、host reclaim 与 transfer 顺序解耦；TensorCast 与 LMCache backend（#40135、#40304、#39089、#36700、#39283、#39480、#39726、#40278、#40680、#40712、#40960、#40512、#27265、#38652） |
| Scheduler / loader / RL / observability | shortest-prefill-first；optimistic Mamba prefill；模型懒加载/fork-safe import；weight load 与 postprocess 拆分；权重更新 session、释放 checker snapshot；trace service name、retract queue latency、first/last-token timing；Engine response 避免 timed wait（#40024、#40184、#41061、#40201、#34981、#40777、#37284、#35802、#39312、#39328、#39486） |
| Sampling / logprob | custom logit processor 避免 GPU sync，并明确 full-shape logits 合同；mask 随请求流式下发；selected/support logprob；graph-pool logits 临时内存；OOT sampling capabilities（#39234、#40986、#40932、#40007、#40038） |
| API / grammar / tool parsing | /v1/decisions 与 System One 分类/校准；setwise /v1/score；XGrammar V4.1 DSML；完整 DSML invoke buffer；token bytes、echo logprob、Mistral tokenizer、chat-template cache-key 顺序修复（#41208、#40826、#38965、#41188、#39026、#39632、#38604、#34776、#39773、#41517） |
| Multimodal / LoRA | 图像损坏返回 400、非 CUDA placement；bounded CPU feature hashing、shared-memory/deferred operand；feature offload race 与 padding；ViT FA4；LoRA DP attention 和 shard-aware buffers（#28131、#38750、#39539、#39870、#40357、#40621、#41344、#36389、#39379） |
| Rust frontend / router / renderer | standalone renderer、transport-neutral frontend、chat parsing parity；fleet sampling contract；cache/storage-tier-aware routing、peer snapshot；worker queue/saturation/min-load knobs；503/429 backpressure、failure-class status、disconnect/idle lifetime、SIGTERM drain、IPv6 discovery、h2c；renderer 镜像发布（#36718、#39385、#40477、#39000–#39002、#39108–#39111、#40687–#40689、#39168–#39170、#39463–#39465、#40391、#39015–#39016、#39006、#40639） |
| Diffusion runtime | 多任务 pipeline、/metrics、Cache-DiT 1.5.1、OOT platform、bounded exact conditioning cache；MiniMax-H3 ComfyUI/PDD/Spectrum 与 rounded SwiGLU；Qwen-Image packed QKV、RMSNorm/RoPE/KV fusion；SenseNova/Cosmos/SANA/Wan/LingBot/Klein/Joy 的 lossless fusion；layerwise offload LoRA 与 residency 生命周期修复（#38762、#19084、#40104、#37547、#40470、#35990、#40568、#35684、#40378、#41339、#40374–#40494、#36192、#40590–#40592） |
| Simulator / configuration | KV cache pool interface 兼容；外部 model config 注册（#40418、#39452） |
| Security | 默认拒绝请求自带 chat_template；SafeUnpickler explicit globals（#28135、#39858、#40259） |

## 平台更新与 HCU 影响

- **AMD/ROCm**：GLM-5.3-Flash MI355X FP8/MXFP4/MTP；DSV4 MegaMoEv2、FP8 decode code object/group decode、INT32 offset overflow 修复；V4.1 gfx950 MXFP8、cache/FP4 indexer/compressor/router、sparse decode/top-k、fused mHC/allreduce；DSA HIP TopK、HiCache K-only host pool与 copy rounds。gfx950/AITER/FlyDSL 专用路径需要与 HCU LightOp、FlashMLA、standalone MegaMoE 分离。
- **Ascend/NPU**：CANN 9.1/Python 3.12；Mamba state/async HiCache、GDN chunk kernel、Kimi K3 A5、DSV4 PCP、MXFP4 fused MoE、batch-invariant FIA、sampling 降同步与 offload；A2 镜像停止发布。
- **Intel XPU/CPU**：NGRAM、HiSparse、KV Canary、chunked prefill、Qwen3.8 Flash-Next；W4A16 torch int4pack、block FP8 scaled_mm；CPU FP8-per-tensor、diffusion norm/scale-shift kernel 与依赖更新。
- **NVIDIA**：SM100 NVFP4 KV、SM120 diffusion/PCIe IPC；CUDA 13.4 镜像、Triton 3.8/Rubin 兼容。对应 wheel 和 compute-capability 判断不能直接套用 HCU。

## 依赖和升级兼容性

| 项 | release 变化 |
|---|---|
| xgrammar / cache-dit / sentencepiece / sgl-eval | 0.2.7 / 1.5.1 / 固定 0.2.1 / 0.1.2 |
| CUDA kernel 栈 | sgl-kernel 0.4.7、sgl-deep-gemm 0.2.0、sgl-deep-ep 0.1.2，CUDA 13.4.1 image |
| CPU | torch 2.14.0、torchvision 0.29.0、triton 3.8.0 |
| XPU / NPU / ROCm | sglang-kernel-xpu 0.3.0；sgl-kernel-npu 2026.9.0.post5；AITER pin acf8fdf9 |

升级需要逐项检查：

1. Rust radix unsupported 配置可自动退回 Python；可用 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` 明确选 Python。
2. FlashInfer fused finalize 默认关；PD Mooncake/NIXL deferred KV release 默认开。
3. 请求 chat_template 默认拒绝，只有 `--trust-request-chat-template` 明确允许。
4. 旧 endpoints、NSA env aliases、`SGLANG_ENABLE_SPEC_V2` 等删除；benchmark 模块迁到 `sglang.benchmark.*`；旧 radix 类要改用 unified cache。
5. custom logit processors 必须返回完整 logits shape，不能依赖 `(1,vocab)` 广播。
6. tensor/distributed 权重更新必须在 begin/end session 内，draft selector 字段与 checksum schema 有变化。
7. distributed 公共 parallel getters 退役；模型/通信的导入和生命周期要跟随迁移。
8. HCU 依赖以内部 requirements、DTK torch 和实际构建结果为准，不能按官方 CUDA/CPU pins 替换现有 DTK 栈。

本次按用户要求只运行 DeepSeek-V4-Flash 基础纯 TP 验证，不能据此宣布全部 release 模型或 PD/CP/EP/MTP 特性已经在 HCU 通过。
