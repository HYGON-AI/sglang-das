# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# DTK 包由 CI 通过 build-arg 传入, 从 dev 基础镜像内联安装 (分层方式同 ciimage 的框架镜像):
#   第一段下载解压 DTK 并把 lib 移出 (约 20G 的层只存在于本地构建缓存, 不会推到 Harbor),
#   第二段 lib 一层 + 其余一层 COPY 回 /opt/dtk, 避免单个超大层
# 不再依赖预装 DTK 的基础镜像, 便于按次更换 DTK 包
ARG BASE_IMAGE
ARG DTK_PKG_URL

# 第一段: 下载并解压 DTK, 按 ciimage 的方式把 lib 移出单独存放
FROM ${BASE_IMAGE} AS dtk_src
ARG DTK_PKG_URL
RUN wget -q -O - "${DTK_PKG_URL}" | tar -xzf - -C /opt && \
    mv /opt/dtk-* /opt/dtk && \
    mv /opt/dtk/lib /dtk_lib

# 第二段: 正式镜像, 先装入 DTK(lib 一层 + 其余一层), 再装框架
FROM ${BASE_IMAGE}
ARG DTK_VERSION

ENV ROCM_PATH=/opt/dtk
COPY --from=dtk_src /dtk_lib ${ROCM_PATH}/lib
COPY --from=dtk_src /opt/dtk ${ROCM_PATH}

RUN echo "source /opt/dtk/env.sh" >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH' >> ~/.bashrc

ENV DTK_VERSION=${DTK_VERSION}

SHELL ["/bin/bash", "-c"]

ARG PYPI_URL
ARG RESOURCE_SERVER_URL
ARG IMAGE_TAG=unknown
ENV IMAGE_TAG=${IMAGE_TAG}

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

# AICC 编译器 (安装脚本内部解析 nightly 最新版本)
RUN wget -q "${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh" -O /tmp/hcu_llvm_installer.sh && \
    chmod +x /tmp/hcu_llvm_installer.sh && \
    bash /tmp/hcu_llvm_installer.sh --major 1.0.0 && \
    rm -f /tmp/hcu_llvm_installer.sh

# 从 nightly 归档拉取 26.04 最新 blas cut 包, 解压后覆盖到 DTK 安装目录(默认 /opt/dtk)
# 注意: 与 blas cut 一样依赖 /usr/local/lib/python3.10/dist-packages/amdsmi 路径追加, DTK ENV 块在末尾统一固化
RUN wget -q "${RESOURCE_SERVER_URL}/Jenkins/CI_scripts/dtk_blas_cut_installer.sh" -O /tmp/dtk_blas_cut_installer.sh && \
    chmod +x /tmp/dtk_blas_cut_installer.sh && \
    bash /tmp/dtk_blas_cut_installer.sh --version 26.04 --os ubuntu20.04 && \
    rm -f /tmp/dtk_blas_cut_installer.sh

# ---layer--- shca 驱动
RUN wget -q "${RESOURCE_SERVER_URL}/pkg/shca/others/20260706/shca-tools-ofed_3.500.4.B020-Ubuntu22.04_amd64.deb" \
    && dpkg -i --force-all shca-tools-ofed_3.500.4.B020-Ubuntu22.04_amd64.deb \
    && rm -rf shca-tools*.deb

# ---layer---
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
    && pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio \
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
    && das-install nixl ${TORCH_TAG} \
    && pip install --no-cache-dir numpy==1.25.0 \
    && pip cache purge

ARG SGLANG_VERSION
RUN pip uninstall -y starlette fastapi prometheus-fastapi-instrumentator \
    && pip install --no-cache-dir "fastapi==0.115.12" "starlette==0.46.2" "prometheus-fastapi-instrumentator==7.1.0" \
    && pip install --no-cache-dir nvidia-cutlass-dsl==4.4.2 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir sglang[diffusion]==${SGLANG_VERSION} sglang-router \
    && pip install --no-cache-dir numpy==1.25.0 \
    && pip install --no-cache-dir setuptools==79.0.1 \
    && pip install --no-cache-dir ai-dynamo==1.4.1 \
    && pip install --no-cache-dir ai-dynamo-runtime==1.4.1


# 构建完成后移除内网 pip 源, 避免运行时意外拉取内网依赖
RUN rm -rf ~/.pip/pip.conf

# DTK 环境变量: 固化 env.sh 的导出(外加 bashrc 追加的 amdsmi 路径), 使 docker run 直接给命令 / sh /
# k8s 等非交互 bash 场景也能加载; bashrc 保留作兜底(驱动缺失时 env.sh 的 .hyhal 分支只有 source 才命中)。
# 放在文件末尾: 不改动其上方各构建步骤所见的环境。换 DTK 包后对照新 env.sh 同步这里(参考 ciimage「DTK 环境变量」)
ENV HYHAL_PATH=/opt/hyhal
ENV DTKROOT=${ROCM_PATH}
ENV AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"
ENV HIP_PATH=${ROCM_PATH}/hip
ENV MIOPEN_FIND_MODE=3
ENV CMAKE_PREFIX_PATH=/opt/hyhal:$CMAKE_PREFIX_PATH
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/hip/bin/hipify:${HYHAL_PATH}/bin:${ROCM_PATH}/opencl/bin:$PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${HYHAL_PATH}/lib:${HYHAL_PATH}/lib64:${ROCM_PATH}/dushmem/lib:${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/dcc/gcvm/lib:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=${HYHAL_PATH}/lib:${ROCM_PATH}/opencl/lib:${ROCM_PATH}/dushmem/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
ENV C_INCLUDE_PATH=${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include:${ROCM_PATH}/opencl/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
ENV C_INCLUDE_PATH=${ROCM_PATH}/dcc/gcvm/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${ROCM_PATH}/dcc/gcvm/include:${C_INCLUDE_PATH}:${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include:${ROCM_PATH}/dushmem/include:${ROCM_PATH}/opencl/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH

# 镜像完整引用由 CI 构建时通过 --build-arg 传入, 注入 ENV 便于容器运行时读取
ARG IMAGE_TAG
ENV IMAGE_TAG=${IMAGE_TAG}
