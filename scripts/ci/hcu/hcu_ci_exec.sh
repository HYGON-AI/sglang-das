#!/bin/bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Execute a command inside the HCU CI container with the right env vars.
# Usage:
#   hcu_ci_exec.sh [--container-name NAME] [-w WORKDIR] [-e KEY=VAL ...] -- <command>
#
# Defaults:
#   WORKDIR=/sglang-checkout/test
#   SGLANG_IS_IN_CI=1
#   SGLANG_IS_IN_CI_HCU=1

CONTAINER="${HCU_CI_CONTAINER:-${HCU_CI_CONTAINER_NAME:-ci_sglang}}"
WORKDIR="/sglang-checkout/test"

declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI]=1
  [SGLANG_IS_IN_CI_HCU]=1
)

# HCU CI should use local model caches by default. These env vars are read
# by sglang.test.test_utils and keep broad registered tests from falling back
# to gated or remote Hugging Face model names. A caller can still override any
# value with -e KEY=VALUE.
ENV_MAP[SGLANG_TEST_DEFAULT_MODEL_NAME]="${SGLANG_TEST_DEFAULT_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/llama3.2/Llama-3.2-1B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_BASE]="${SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_BASE:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/llama3.2/Llama-3.2-1B}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_SCORE]="${SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_SCORE:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Reranker-0.6B}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_QWEN_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_QWEN_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-1.5B-Instruct}"
ENV_MAP[SGLANG_HCU_QWEN25_0P5B_MODEL]="${SGLANG_HCU_QWEN25_0P5B_MODEL:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_VLM_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_VLM_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_EMBEDDING_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_EMBEDDING_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/Alibaba-NLP/gte-Qwen2-1.5B-instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/BAAI/bge-reranker-base}"
ENV_MAP[SGLANG_TEST_DETERMINISTIC_MODEL_NAME]="${SGLANG_TEST_DETERMINISTIC_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B}"
ENV_MAP[SGLANG_TEST_DEFAULT_REASONING_MODEL_NAME]="${SGLANG_TEST_DEFAULT_REASONING_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/deepseek-r1/DeepSeek-R1-Distill-Qwen-7B}"
ENV_MAP[SGLANG_TEST_DEFAULT_ENABLE_THINKING_MODEL_NAME]="${SGLANG_TEST_DEFAULT_ENABLE_THINKING_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B}"
ENV_MAP[SGLANG_TEST_PRIORITY_METRICS_MODEL]="${SGLANG_TEST_PRIORITY_METRICS_MODEL:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B}"
ENV_MAP[SGLANG_TEST_DEFAULT_DEEPSEEK_V32_MODEL]="${SGLANG_TEST_DEFAULT_DEEPSEEK_V32_MODEL:-/public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2}"
ENV_MAP[SGLANG_HCU_MMLU_DATASET_PATH]="${SGLANG_HCU_MMLU_DATASET_PATH:-/public/opendas/DL_DATA/llm-models/datasets/mmlu}"
ENV_MAP[SGLANG_HCU_MMMU_DATASET_PATH]="${SGLANG_HCU_MMMU_DATASET_PATH:-/public/opendas/DL_DATA/llm-models/multimodal-datasets/MMMU}"
ENV_MAP[SGLANG_TEST_HCU_DEFAULT_SERVER_ARGS]="${SGLANG_TEST_HCU_DEFAULT_SERVER_ARGS:---attention-backend fa3 --page-size 64 --trust-remote-code}"

if [[ -n "${HCU_CI_VISIBLE_DEVICES:-}" ]]; then
  ENV_MAP[HIP_VISIBLE_DEVICES]="${HCU_CI_VISIBLE_DEVICES}"
  ENV_MAP[CUDA_VISIBLE_DEVICES]="${HCU_CI_VISIBLE_DEVICES}"
  if [[ "${HCU_CI_SET_ROCR_VISIBLE_DEVICES:-0}" == "1" ]]; then
    ENV_MAP[ROCR_VISIBLE_DEVICES]="${HCU_CI_VISIBLE_DEVICES}"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      IFS="=" read -r key val <<< "$2"
      ENV_MAP["$key"]="$val"
      shift 2
      ;;
    --container-name)
      CONTAINER="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

# Prefer the current checkout over any preinstalled sglang package in source
# mode.  Wheel mode deliberately avoids this so tests import site-packages.
if [[ "${HCU_CI_USE_INSTALLED_WHEELS:-0}" == "1" || "${HCU_CI_USE_INSTALLED_WHEELS:-0}" == "true" ]]; then
  echo "[hcu-ci] HCU_CI_USE_INSTALLED_WHEELS=${HCU_CI_USE_INSTALLED_WHEELS}; not injecting /sglang-checkout/python into PYTHONPATH"
  ENV_MAP[HCU_CI_USE_INSTALLED_WHEELS]="${HCU_CI_USE_INSTALLED_WHEELS}"
elif [[ -z "${ENV_MAP[PYTHONPATH]+x}" ]]; then
  if [[ "$(basename "${WORKDIR}")" == "test" ]]; then
    SOURCE_ROOT="$(dirname "${WORKDIR}")"
  else
    SOURCE_ROOT="${WORKDIR}"
  fi
  ENV_MAP[PYTHONPATH]="${SOURCE_ROOT}/python"
fi

ENV_ARGS=()
for key in "${!ENV_MAP[@]}"; do
  ENV_ARGS+=("-e" "$key=${ENV_MAP[$key]}")
done

# First attempt: allow HF downloads.
if docker exec \
  -w "${WORKDIR}" \
  "${ENV_ARGS[@]}" \
  "${CONTAINER}" "$@"; then
  exit 0
else
  FIRST_EXIT_CODE=$?
fi

echo "First attempt failed with exit code ${FIRST_EXIT_CODE}"

# Don't retry deterministic failures.
if [[ "${FIRST_EXIT_CODE}" -eq 1 || "${FIRST_EXIT_CODE}" -eq 137 || "${FIRST_EXIT_CODE}" -eq 255 ]]; then
  echo "Exit code ${FIRST_EXIT_CODE} indicates a real failure, not retrying"
  exit "${FIRST_EXIT_CODE}"
fi

echo "Retrying with HF_HUB_OFFLINE=1 (use cached HF models)..."
docker exec \
  -w "${WORKDIR}" \
  "${ENV_ARGS[@]}" \
  -e HF_HUB_OFFLINE=1 \
  "${CONTAINER}" "$@"
