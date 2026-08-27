# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# 基础镜像由 CI 通过 build-arg 传入 (DTK/HCU 环境, 如 ubuntu22.04 + python3.10 + dtk2604)
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG PYPI_URL
ARG RESOURCE_SERVER_URL

# pip 配置 + das-install
RUN TRUSTED_HOST="${PYPI_URL#*://}" && TRUSTED_HOST="${TRUSTED_HOST%%/*}" && \
    mkdir -p ~/.pip && \
    printf '%s\n' \
    '[global]' \
    "index-url = ${PYPI_URL}/nightly/dtk2604/+simple/" \
    "trusted-host = ${TRUSTED_HOST}" \
    > ~/.pip/pip.conf && \
    pip install --ignore-installed --no-cache-dir blinker && \
    printf '%s\n' \
    '#!/bin/bash' \
    '# usage: das-install <pkg>[==<ver>] <torch_tag>' \
    'pkg=$1 tag=$2 prefix=' \
    'if [[ "$pkg" == *==* ]]; then prefix="${pkg#*==}"; pkg="${pkg%%==*}"; fi' \
    'ver=$(pip index versions "$pkg" 2>/dev/null | grep -oP "${prefix}[^ ()]*${tag}[^ ()]*" | head -1)' \
    '[ -z "$ver" ] && echo "Error: No version for $pkg matching $tag" && exit 1' \
    'echo "Installing $pkg==$ver"' \
    'pip install --no-cache-dir --extra-index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com "$pkg==$ver"' \
    > /usr/local/bin/das-install && chmod +x /usr/local/bin/das-install

# AICC 编译器 (nightly 最新版地址由 CI 通过 build-arg 传入, 在镜像内下载安装)
# AICC_VERSION 由 CI 从同一 URL 解析传入; ENV 指令不支持 bash 参数展开, 无法在 Dockerfile 内由 URL 自动推导
ARG AICC_URL
ARG AICC_VERSION
ENV AICC_VERSION=${AICC_VERSION}

RUN wget -q "${AICC_URL}" -O /tmp/aicc.run && \
    chmod +x /tmp/aicc.run && \
    yes | /tmp/aicc.run --dtk_dir /opt/dtk && \
    rm -f /tmp/aicc.run

RUN pip install --no-cache-dir ninja wheel setuptools \
    && pip install --no-cache-dir ray[data,train,tune,serve] -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir amdsmi==1.0.0+630c16a6.dirty \
    && pip install --no-cache-dir cupy \
    && pip install --no-cache-dir apache-tvm-ffi==0.1.11 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir vllm==0.25.1 \
    && pip install --no-cache-dir vllm-hcu==0.25.1  \
    && pip cache purge

ARG TORCH_VERSION
RUN TORCH_TAG="torch${TORCH_VERSION//./}" \
    && pip install --no-cache-dir torch==${TORCH_VERSION} torchvision \
    && das-install flash_attn ${TORCH_TAG} \
    && das-install lightop ${TORCH_TAG} \
    && das-install lmslim ${TORCH_TAG} \
    && das-install deepgemm ${TORCH_TAG} \
    && das-install aiter ${TORCH_TAG} \
    && pip install --no-cache-dir mooncake_transfer_engine \
    && das-install deep_ep ${TORCH_TAG} \
    && das-install tilelang ${TORCH_TAG} \
    && das-install boltops ${TORCH_TAG} \
    && das-install causal_conv1d==1.5.4 ${TORCH_TAG} \
    && das-install flash_mla ${TORCH_TAG} \
    && das-install flash_kda ${TORCH_TAG} \
    && das-install fastsafetensors ${TORCH_TAG} \
    && das-install triton==3.6.0 ${TORCH_TAG} \
    && pip install --no-cache-dir numpy==1.25.0 \
    && pip cache purge

ARG SGLANG_VERSION
RUN pip uninstall -y starlette fastapi prometheus-fastapi-instrumentator \
    && pip install --no-cache-dir "fastapi==0.115.12" "starlette==0.46.2" "prometheus-fastapi-instrumentator==7.1.0" \
    && pip install --no-cache-dir torchaudio==2.11.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir nvidia-cutlass-dsl==4.4.2 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir sgl-deep-gemm==0.1.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir sglang[diffusion]==${SGLANG_VERSION} sglang-router \
    && pip install --no-cache-dir numpy==1.25.0 \
    && pip install --no-cache-dir setuptools==79.0.1 


# 构建完成后移除内网 pip 源, 避免运行时意外拉取内网依赖
RUN rm -rf ~/.pip/pip.conf
