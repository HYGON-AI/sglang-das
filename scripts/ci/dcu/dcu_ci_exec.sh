#!/bin/bash
set -euo pipefail

# Execute a command inside the DCU CI container with the right env vars.
# Usage:
#   dcu_ci_exec.sh [--container-name NAME] [-w WORKDIR] [-e KEY=VAL ...] -- <command>
#
# Defaults:
#   WORKDIR=/sglang-checkout/test
#   SGLANG_IS_IN_CI=1
#   SGLANG_IS_IN_CI_DCU=1

CONTAINER="${DCU_CI_CONTAINER:-${DCU_CI_CONTAINER_NAME:-ci_sglang}}"
WORKDIR="/sglang-checkout/test"

declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI]=1
  [SGLANG_IS_IN_CI_DCU]=1
)

# DCU CI should use local model caches by default. These env vars are read
# by sglang.test.test_utils and keep broad registered tests from falling back
# to gated or remote Hugging Face model names. A caller can still override any
# value with -e KEY=VALUE.
ENV_MAP[SGLANG_TEST_DEFAULT_MODEL_NAME]="${SGLANG_TEST_DEFAULT_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/llama3.2/Llama-3.2-1B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_BASE]="${SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_BASE:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/llama3.2/Llama-3.2-1B}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_SCORE]="${SGLANG_TEST_DEFAULT_SMALL_MODEL_NAME_SCORE:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Reranker-0.6B}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_QWEN_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_QWEN_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-1.5B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_VLM_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_VLM_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_EMBEDDING_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_EMBEDDING_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/Alibaba-NLP/gte-Qwen2-1.5B-instruct}"
ENV_MAP[SGLANG_TEST_DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME]="${SGLANG_TEST_DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/vllm-optest-models/BAAI/bge-reranker-base}"
ENV_MAP[SGLANG_TEST_DETERMINISTIC_MODEL_NAME]="${SGLANG_TEST_DETERMINISTIC_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B}"
ENV_MAP[SGLANG_TEST_DEFAULT_REASONING_MODEL_NAME]="${SGLANG_TEST_DEFAULT_REASONING_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/deepseek-r1/DeepSeek-R1-Distill-Qwen-7B}"
ENV_MAP[SGLANG_TEST_DEFAULT_ENABLE_THINKING_MODEL_NAME]="${SGLANG_TEST_DEFAULT_ENABLE_THINKING_MODEL_NAME:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B}"
ENV_MAP[SGLANG_TEST_DCU_DEFAULT_SERVER_ARGS]="${SGLANG_TEST_DCU_DEFAULT_SERVER_ARGS:---attention-backend fa3 --trust-remote-code}"

if [[ -n "${DCU_CI_VISIBLE_DEVICES:-}" ]]; then
  ENV_MAP[HIP_VISIBLE_DEVICES]="${DCU_CI_VISIBLE_DEVICES}"
  ENV_MAP[ROCR_VISIBLE_DEVICES]="${DCU_CI_VISIBLE_DEVICES}"
  ENV_MAP[CUDA_VISIBLE_DEVICES]="${DCU_CI_VISIBLE_DEVICES}"
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

# Prefer the current checkout over any preinstalled sglang package in the
# container.  This keeps registered tests reproducible when iterating from a
# mounted source tree.
if [[ -z "${ENV_MAP[PYTHONPATH]+x}" ]]; then
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
