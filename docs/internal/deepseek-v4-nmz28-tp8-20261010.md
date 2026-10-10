# DeepSeek-V4 nmz28 纯 TP8 调试与验收 · 2026-10-10

## 验收结果

按用户最新指定的 EvalScope HumanEval 流程完成验收：**第一轮 90.24%（148/164），复跑 89.02%（146/164）**。2026-09-11 同名模型历史为 **89.02%（146/164）**。两轮原始输入、测试 metadata、评测配置与历史逐项相同，均无 API error、空输出或 max_tokens 截断；失败题为生成代码未通过单测，全部 review 正常结束。

服务位于 `zz-nmz28 | rye_sglang_latest`，地址 `http://12.12.12.28:10015`；验收后保留服务运行。工作目录为 `/home/proj_sglang_open/sglang-das`，同步分支 `sync/official-main-daily-20261009`，官方冻结目标仍为 `438df9a2a4645d39c87e33c4a27792e568ac2701`。提交在用户指定 `zz-nmz26 | rye_sglang_open` 完成；最终提交以 `dcu-main-sync-official-20261009` annotated tag 解析，未 push。

## 实际命令、环境与模型

```bash
# /home/proj_sglang_open/scripts_local
bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-FP8-Channel/
# 原 /home/scripts/acc_test/run-all-acc.sh，以同名软链接运行并隔离输出
bash run-all-acc.sh /models/DeepSeek-V4-Flash-FP8-Channel 10015
```

- 服务参数：TP8，DP/CP/EP/PP 均 1，无 MTP / PD；mem_fraction_static=0.8，decode graph max bs32，disable_radix_cache，chunked_prefill_size32768 / max_prefill_tokens16384。
- 原脚本：HumanEval 全量164题、并发64、greedy、stream、max_tokens4096、seed42、review_timeout30、enable_thinking=false；`filters.remove_until=</think>`。没有修改评测脚本或样本。
- 服务：Torch2.11.0 / HIP6.3.26113，八卡 gfx938，SGLang0.5.21 当前目录 editable + HIP AOT，LightOp `0.6.0+dtk2604.torch2110.2609301833.g9d6ed8`；nmz28 EvalScope1.12.0。
- 私有 launcher 增加 `set -o pipefail`，防止 tee 把服务异常退出报告为0；该脚本位于源码 Git 之外。
- 原脚本 SHA256：`bb2ffb8ece0ecbbc9656bf4245380856ac95a56c8409a79514f02af632dcce20`。
- `/module` 和 `/models` 同名 checkpoint 的 config SHA256：`abc44dec7f2f8db883a61af2c5a6dd11a9e594302fcb8213c74cd79ad2e87ca1`；weight index SHA256：`7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a`，两路径完全一致。
- 该模型与原 `DeepSeek-V4-Flash-0731-FP8-Channel` checkpoint 不同；不把本轮结果与原0731的GSM8K分数直接比较。模型实际为43层、hidden4096、64heads、FP8 channel compressed-tensors。

## 修复内容

| 问题 | 根因与修改 | 回归证据 |
|---|---|---|
| startup 构建 Rust TreeCore 失败 | HCU安装不带Rust扩展，官方默认触发JIT，Cargo1.75不能读取edition2024。`tree_core_registry.py` 中 `_is_hcu` 且未显式选择时用Python；显式实例/env配置仍交给官方resolver | TreeCore35passed / 46subtests；HCU默认不probe Rust、显式Rust/Python/custom以及非HCU策略均覆盖 |
| GrammarManager startup AttributeError | Scheduler已删除`enable_dp_attention`。改读发布后的`get_parallel().attn_dp_size > 1`，保留CP同步。旧context bool在发布后清空，不能继续当运行依据 | 新回归1passed / 2subtests：TP8与attnDP2×CP4；真实服务成功 |
| c0 sparse-prefill assertion | ratio0层不存在compressed pool，旧逻辑仍无条件调用`get_extra_key_layout`。仅compressed slice存在时读extra layout；HCU LightOp接受纯SWA | 实际入口c0/c4 × LightOp on/off四种probe全通过；HumanEval全量短/长prefill正常 |
| HumanEval客户端 NFS临时目录清理报错 | multiprocessing manager退出时仍持有NFS文件。复跑将TMPDIR改到容器本地`/tmp/sglang-sync-20261009-humaneval`，数据缓存/输出仍保留共享盘 | 第二轮退出0、无traceback；没有改动generation或review条件 |

官方通用路径、当前main的LightOp/HCU FlashMLA/shared-clamp优化继续保留；没有开启logits sanitization，没有改写模型答案或测试。

## 实际运行与审计

| 运行 | 实际观察 | 状态 |
|---|---|---|
| attempt1 | 八卡graph capture完成后触发Rust Cargo edition2024失败 | 已修复 |
| attempt2 | TreeCore通过，GrammarManager引用删除字段失败 | 已修复 |
| attempt3 | 服务就绪、三次短请求成功，较长prefill暴露c0 extra layout assertion | 已修复 |
| attempt4 | 完整服务；GSM8K100两次0.97/0.93，部分请求批量空输出，失败题串行重放可正确 | 原样记录，不声明GSM8K通过 |
| attempt5 | 仅额外启用`SGLANG_ENABLE_NAN_LOGITS_CHECK=1`；GSM8K期间HSA硬件异常/VMFault，dump显示`_hc_head_kernel`；无明确`NaN detected`信息，未得到首次异常算子定位 | 非默认诊断观察待查，已停止本轮自己的任务 |
| attempt6 | 去掉临时NaN检查开关，原参数重启；10:16服务就绪，八个rank graph capture完整，三次短请求正常 | 服务通过 |
| HumanEval第一次 | 10:16:27–10:17:58，148/164，90.24%，退出0；NFS子进程清理traceback | 精度通过；客户端环境已改 |
| HumanEval本地TMPDIR复跑 | 10:18:51–10:20:28，146/164，89.02%，退出0、无客户端traceback | 最终验收通过 |

- 八个TP rank capture覆盖bs1/2/4/8/12/16/24/32；两轮验证各三次短请求成功，服务日志记录19次`cuda graph: True` decode。
- 当前验收服务日志VMFault、NaN检查报错、worker exit、Python traceback计数均0。这仅描述该服务默认参数下两轮HumanEval，不证明诊断轮问题已修复。
- 当前服务启动于代码基点`02b0aedab168a7e623a5332f6e76af051199657d`加上述源码修复；实际运行快照5583个`python/sglang`文件SHA256与验收结束工作区全部匹配。提交、main快进不修改运行源码。
- 全仓compile、diff-check、精确冲突标记和相对原main/官方的新增F821/F811/F722检查通过，新增错误0；源码以`_is_hcu`隔离HCU专有分派。

## 证据与保留观察

证据根目录：`/home/proj_sglang_open/sync-evidence/20261009/runtime-nmz28/`。

- `humaneval-final-verdict.json`：两轮164题、历史配置/数据一致性、服务日志计数、运行源码hash的最终审计。
- `humaneval-attempt6/`、`humaneval-attempt6-localtmp/`：原始prediction、review、JSON/HTML报告、sanity、validation metadata、prediction-audit与退出码。
- `server.log`、`runtime-metadata.json`、`runtime-code-hashes.json`：验收服务、软件/模型信息、5583文件快照。
- `server-attempt1-rust.log`、`server-attempt2-grammar-context.log`、`server-attempt3-c0-layout.log`、`server-attempt4-accuracy.log`、`server-attempt5-nan-check.log`：先前运行原始证据。
- `tree-core-tests.log`、`grammar-context-tests.log`、`runtime-sparse-prefill-test.log`、`final-static/`：代码回归与静态检查。

用户明确改为HumanEval验收并允许个别题错误。本次仅据完整HumanEval与历史基线判定精度pass。GSM8K批量空输出和非默认NaN检查引发的VMFault仍是待查观察；未将其归为普通算术错题，未声称已定位或修复。不扩展结论至其他模型、EP/DP/CP/PD/MTP或其他长请求负载。
