# <div align="center"><strong>SGLang</strong></div>

## sglang_dcu简介
SGLang是一个用于大型语言模型和多模态模型的高性能服务框架，旨在在从单个GPU到大型分布式集群的各种设置中提供低延迟和高吞吐量的推理，我们基于开源社区做了DCU平台的适配和针对性的优化。
其核心功能包括：快速运行时：通过RadixAttention提供高效的服务，用于前缀缓存、零开销CPU调度器、预填充解码分解、推测解码、连续批处理、分页注意力、张量/流水线/专家/数据并行性、结构化输出、分块预填充、量化（FP4/FP8/INT4/AWQ/GPTQ）和多LoRA批处理。
广泛的模型支持：支持各种语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等）、嵌入模型（e5-Mistral、gte、mcdse）、奖励模型（Skywork）和扩散模型（WAN、Qwen-Image），易于扩展以添加新模型。与大多数Hugging Face模型和OpenAI API兼容。
强化学习和训练后主干：SGLang是一个经过验证的全球推广后端，具有原生强化学习集成，并被AReaL、Miles、slime、Tunix、verl等知名训练后框架采用。

## 使用源码编译方式安装
提供2种环境准备方式:

1. 基于光源pytorch2.5.1基础镜像环境:根据pytorch2.5.1、python、dtk及系统下载对应的镜像版本。

2. 基于现有python环境:安装pytorch2.5.1,pytorch whl包下载目录:[https://cancon.hpccube.com:65024/4/main/pytorch](https://cancon.hpccube.com:65024/4/main/pytorch),根据python、dtk版本,下载对应pytorch2.5.1的whl包。安装命令如下:
```shell
pip install torch* (下载的torch的whl包)
pip install setuptools wheel
```

### 源码编译安装
```shell
git clone  https://developer.sourcefind.cn/codes/OpenDAS/sglang.git #根据需要的分支进行切换
```
安装依赖:
```shell
pip install -r requirements_dcu.txt
```

- 提供2种源码编译方式(进入sglang目录):
```
编译安装sgl_kernel
cd sgl-kernel
python setup_hip.py install

1. 编译whl包并安装
python setup.py bdist_wheel 
cd dist
pip install sglang*

2. 源码编译sglang
pip install -e "python[all_hip]" --no-deps --no-build-isolation --no-index
```
### 运行基础环境准备
1、使用上面基于光源pytorch2.5.1基础镜像环境

2、根据pytorch2.5.1、python、dtk及系统下载对应的依赖包:
- flash_attn: [https://cancon.hpccube.com:65024/4/main/flash_attn](https://cancon.hpccube.com:65024/4/main/flash_attn)
- flash_mla: [https://download.sourcefind.cn:65024/4/main/flash_mla](https://download.sourcefind.cn:65024/4/main/flash_mla)
- lightop: [https://download.sourcefind.cn:65024/4/main/lightop](https://download.sourcefind.cn:65024/4/main/lightop)
- lmslim: [https://cancon.hpccube.com:65024/4/main/lmslim](https://cancon.hpccube.com:65024/4/main/lmslim)
- triton: [https://cancon.hpccube.com:65024/4/main/triton](https://cancon.hpccube.com:65024/4/main/triton)
- vllm: [https://download.sourcefind.cn:65024/4/main/vllm](https://download.sourcefind.cn:65024/4/main/vllm)

### 注意事项
+ 若使用 pip install 下载安装过慢,可添加源:-i  https://mirrors.huaweicloud.com/artifactory/pypi-public/simple

## 验证
- python -c "import sglang; print(sglang.\_\_version__)",

## 手动无人值守大模型测试

仓库提供独立 workflow `Manual DCU Unattended Model Test`（文件：`.github/workflows/dcu-manual-model-test.yml`），用于在 DCU runner 上手动启动长时间模型验证任务。它只支持 `workflow_dispatch`，不会在 `pull_request`、`push` 或 `schedule` 时自动触发，因此不属于 PR gate。

使用方式：进入 GitHub Actions，选择 `Manual DCU Unattended Model Test`，点击 `Run workflow`。GitHub 页面上的分支下拉框决定默认测试 ref；也可以通过 `test_branch` 输入分支、tag、ref 或 SHA 覆盖。默认 suite 为 `nightly-dcu-accuracy`，可通过 `suite` 改成 `nightly-dcu`、`nightly-dcu-vlm`、`nightly-dcu-4-gpu` 等已注册 DCU suite。若只跑单个文件，填写 `include_file`，例如 `test/registered/dcu/accuracy/bw1100/test_gsm8k_eval_dcu.py`。

常用输入包括：`timeout_per_file`（默认 4200 秒，沿用现有 DCU accuracy 长测配置）、`auto_partition_id` 和 `auto_partition_size`（必须成对填写）、`continue_on_error`（默认 true，表示 `run_suite.py` 内部尽量继续跑后续文件，但 workflow 仍会在最终失败时显示失败）、`runner_label`、`container_name` 和 `image`。`model_name` 当前会传给支持全局默认模型变量的测试（`SGLANG_TEST_DEFAULT_MODEL_NAME`）；`run_suite.py` 尚无 `--model-name` 过滤参数，后续如需要按模型精确筛选，应在 runner 或测试注册层补充正式能力。

运行结束后，可在 workflow 日志中查看测试 ref、runner label、容器名、suite、include_file、model_name、timeout、分片配置和最终执行命令。job summary 会展示本次配置和结果；artifact `dcu-manual-model-test-<run_id>` 会上传 `summary.md`、最终命令和 `run.log`。

## PD 分离

### Requirements

```bash
pip list | grep mooncake-transfer-engine
```

### Usage

#### Normal

**加载环境变量：**

```bash
export MC_TOPO_FILE_FORCE=./mc_topo.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
#export MC_IB_GID_INDEX=0 #Roce网络需要设置
```
mc_topo.config

```YAML
0000:9f:00.0 mlx5_2 hip:0
0000:57:00.0 mlx5_3 hip:1
0000:5e:00.0 mlx5_4 hip:2
0000:05:00.0 mlx5_5 hip:3
0000:e5:00.0 mlx5_6 hip:4
0000:c1:00.0 mlx5_7 hip:5
0000:cc:00.0 mlx5_8 hip:6
0000:b1:00.0 mlx5_9 hip:7
```
**DeepSeek-R1-Channel-INT8 模型示例**

##### prefill 
```bash
python -m sglang.launch_server \
  --model-path DeepSeek-R1-Channel-INT8 \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --disaggregation-mode prefill \
  --host ${prefill_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 2 \
  --pp-size 4 \
  --mem-fraction-static 0.9 \
  --attention-backend dcu_mla 
```

##### decode  
```bash
python -m sglang.launch_server \
  --model-path DeepSeek-R1-Channel-INT8  \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --disaggregation-mode decode \
  --host ${decode_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 8 \
  --mem-fraction-static 0.9 \
  --attention-backend dcu_mla \
  --dtype bfloat16 \
  --quantization slimquant_marlin
```
##### 启动路由：
```bash
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://${prefill_ip}:30000 --decode http://${decode_ip}:30000 --policy round_robin --port 30002
```
##### 验证输出结果
在另一个终端中，使用以下命令验证输出结果：
```bash
curl -X POST http://localhost:30002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "介绍一下深度学习的发展",
    "max_tokens": 300,
    "temperature": 0
  }'
```
#### low_latency （使用deepep）
prefill部分同上面normal部分的prefill

##### decode  
```bash
# deep_ep
#export ROCSHMEM_DISABLE_HDP_FLUSH=1 #xdp使用
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_ENABLE_LL_DISPATCH_OPT=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config   // 与上面文件一致

# mooncake
export MC_TOPO_FILE_FORCE=./mc_topo.config      // 与上面文件一致
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
#export MC_IB_GID_INDEX=0 #Roce网络需要设置
```
topo.config

```YAML
0000:9f:00.0 mlx5_2 2
0000:57:00.0 mlx5_3 3
0000:5e:00.0 mlx5_4 4
0000:05:00.0 mlx5_5 5
0000:e5:00.0 mlx5_6 6
0000:c1:00.0 mlx5_7 7
0000:cc:00.0 mlx5_8 8
0000:b1:00.0 mlx5_9 9
```
单机ep8dp2部署示例
```bash
python3 -m sglang.launch_server --model-path DeepSeek-R1-Channel-INT8 \
--disaggregation-mode decode --quantization slimquant_marlin \
--kv-cache-dtype fp8_e4m3 --host ${decode_ip} --port 30000 --trust-remote-code \
--dist-init-addr ${decode_master_ip}:5000 --nnodes 1 --node-rank 0 --dtype bfloat16  \
--tp-size 8 --dp-size 2 --mem-fraction-static 0.85 \
--moe-dense-tp-size 1 --enable-dp-lm-head \
--attention-backend dcu_mla --enable-dp-attention --moe-a2a-backend deepep  \
--ep-size 8 --deepep-mode low_latency \
--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```
多机ep16dp16部署示例
```bash
#node1 作为主节点
python3 -m sglang.launch_server --model-path DeepSeek-R1-Channel-INT8 \
--disaggregation-mode decode --quantization slimquant_marlin \
--kv-cache-dtype fp8_e4m3 --host ${node1_ip} --port 30000 --trust-remote-code \
--dist-init-addr ${node1_ip}:5000 --nnodes 2 --node-rank 0 --dtype bfloat16  \
--tp-size 16 --dp-size 16 --mem-fraction-static 0.85 \
--moe-dense-tp-size 1 --enable-dp-lm-head \
--attention-backend dcu_mla --enable-dp-attention --moe-a2a-backend deepep  \
--ep-size 16 --deepep-mode low_latency \
--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

#node2
python3 -m sglang.launch_server --model-path DeepSeek-R1-Channel-INT8 \
--disaggregation-mode decode --quantization slimquant_marlin \
--kv-cache-dtype fp8_e4m3 --host ${node2_ip} --port 30000 --trust-remote-code \
--dist-init-addr ${node1_ip}:5000 --nnodes 2 --node-rank 1 --dtype bfloat16  \
--tp-size 16 --dp-size 16 --mem-fraction-static 0.85 \
--moe-dense-tp-size 1 --enable-dp-lm-head \
--attention-backend dcu_mla --enable-dp-attention --moe-a2a-backend deepep  \
--ep-size 16 --deepep-mode low_latency \
--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

##### 启动路由：
```bash
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://${prefill_ip}:30000 --decode http://${decode_ip}:30000 --policy round_robin --port 30002
```
##### 验证输出结果
在另一个终端中，使用以下命令验证输出结果：
```bash
curl -X POST http://localhost:30002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "介绍一下深度学习的发展",
    "max_tokens": 300,
    "temperature": 0
  }'
```

## Known Issue
- 无

## 参考资料
- [README_ORIGIN](README_ORIGIN.md)
- [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)

## License

本仓库基于 [SGLang](https://github.com/sgl-project/sglang) `v0.5.12` 版本进行 DCU 平台适配和优化，上游项目采用 Apache License, Version 2.0。

Hygon Information Technology Co., Ltd. 对 DCU 适配、修改和新增贡献部分同样采用 Apache License, Version 2.0。

本仓库保留上游 SGLang 项目的版权声明和许可证条款，详见 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`。
