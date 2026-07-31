#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCRIPT_DIR}/hcu_evalscope_accuracy.py"

API_URL="${API_URL:-http://127.0.0.1:11000/v1}"
API_KEY="${API_KEY:-sk-123456}"
MODEL_ID="${MODEL_ID:-}"
MODEL_KEY="${MODEL_KEY:-unknown}"
MODEL_NAME="${MODEL_NAME:-${MODEL_KEY}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "${OUTPUT_DIR}" ]]; then
    mkdir -p "${TMPDIR:-/tmp}/hcu-evalscope"
    OUTPUT_DIR="$(
        mktemp -d "${TMPDIR:-/tmp}/hcu-evalscope/${MODEL_KEY}.XXXXXX"
    )"
fi
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/summary.json}"

GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:-/public/opendas/DL_DATA/opencompass_data/gsm8k/train.jsonl}"
GSM8K_TEST_PATH="${GSM8K_TEST_PATH:-/public/opendas/DL_DATA/opencompass_data/gsm8k/test.jsonl}"
MATH500_PATH="${MATH500_PATH:-/public4/home/shenjzh/dataset/math_500/test.jsonl}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/public/opendas/DL_DATA/opencompass_data/humaneval/human-eval-v2-20210705.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RETRIES="${RETRIES:-2}"
RETRY_INTERVAL="${RETRY_INTERVAL:-5}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-null}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-{}}"

GSM8K_LIMIT="${GSM8K_LIMIT:-200}"
MATH_LIMIT="${MATH_LIMIT:-200}"
HUMANEVAL_LIMIT="${HUMANEVAL_LIMIT:-164}"
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_MATH="${RUN_MATH:-1}"
RUN_HUMANEVAL="${RUN_HUMANEVAL:-1}"
HUMANEVAL_REQUEST_TIMEOUT="${HUMANEVAL_REQUEST_TIMEOUT:-6000}"

for command in python3 curl tee; do
    if ! command -v "${command}" >/dev/null; then
        echo "Required command is missing: ${command}" >&2
        exit 2
    fi
done

if ! python3 -c "from importlib.metadata import version; assert version('evalscope') == '1.9.1'" \
    >/dev/null 2>&1; then
    echo "EvalScope 1.9.1 is required. Install scripts/ci/hcu/requirements_evalscope.txt." >&2
    exit 2
fi

export BATCH_SIZE EXTRA_BODY_JSON MAX_TOKENS RETRIES RETRY_INTERVAL TEMPERATURE

python3 "${HELPER}" validate-data \
    --gsm8k-train "${GSM8K_TRAIN_PATH}" \
    --gsm8k-test "${GSM8K_TEST_PATH}" \
    --math-500 "${MATH500_PATH}" \
    --humaneval "${HUMANEVAL_PATH}" || exit 2

if [[ -z "${MODEL_ID}" ]]; then
    MODEL_ID="$(
        curl -fsS \
            -H "Authorization: Bearer ${API_KEY}" \
            "${API_URL%/}/models" |
            python3 -c \
                'import json,sys; data=json.load(sys.stdin)["data"]; assert data; print(data[0]["id"])'
    )" || exit 2
fi

mkdir -p "${OUTPUT_DIR}"
NORMALIZED_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/hcu-evalscope-data.XXXXXX")"
cleanup() {
    rm -rf -- "${NORMALIZED_ROOT}"
}
trap cleanup EXIT

mkdir -p \
    "${NORMALIZED_ROOT}/gsm8k" \
    "${NORMALIZED_ROOT}/math-500" \
    "${NORMALIZED_ROOT}/humaneval"
ln -s "${GSM8K_TRAIN_PATH}" "${NORMALIZED_ROOT}/gsm8k/train.jsonl"
ln -s "${GSM8K_TEST_PATH}" "${NORMALIZED_ROOT}/gsm8k/test.jsonl"
ln -s "${MATH500_PATH}" "${NORMALIZED_ROOT}/math-500/test.jsonl"
ln -s "${HUMANEVAL_PATH}" "${NORMALIZED_ROOT}/humaneval/test.jsonl"

make_dataset_args() {
    local dataset="$1"
    local path="$2"
    DATASET_NAME="${dataset}" DATASET_PATH="${path}" python3 -c '
import json
import os

name = os.environ["DATASET_NAME"]
config = {"dataset_id": os.environ["DATASET_PATH"]}
if name in {"gsm8k", "humaneval"}:
    config["subset_list"] = ["default"]
print(json.dumps({name: config}))
'
}

make_generation_config() {
    local timeout="$1"
    REQUEST_TIMEOUT="${timeout}" python3 -c '
import json
import os

print(json.dumps({
    "timeout": int(os.environ["REQUEST_TIMEOUT"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "retries": int(os.environ["RETRIES"]),
    "retry_interval": int(os.environ["RETRY_INTERVAL"]),
    "extra_body": json.loads(os.environ["EXTRA_BODY_JSON"]),
}))
'
}

run_dataset() {
    local dataset="$1"
    local dataset_path="$2"
    local limit="$3"
    local timeout="$4"
    local work_dir="${OUTPUT_DIR}/${dataset}"
    local dataset_args
    local generation_config

    dataset_args="$(make_dataset_args "${dataset}" "${dataset_path}")" || return 1
    generation_config="$(make_generation_config "${timeout}")" || return 1
    mkdir -p "${work_dir}"

    echo "Running ${MODEL_NAME} on ${dataset}: samples=${limit}, max_tokens=${MAX_TOKENS}"
    python3 "${HELPER}" evalscope eval \
        --model "${MODEL_ID}" \
        --api-url "${API_URL}" \
        --api-key "${API_KEY}" \
        --eval-type openai_api \
        --datasets "${dataset}" \
        --dataset-args "${dataset_args}" \
        --eval-batch-size "${BATCH_SIZE}" \
        --generation-config "${generation_config}" \
        --limit "${limit}" \
        --work-dir "${work_dir}" \
        --no-timestamp \
        2>&1 | tee "${work_dir}.log"
}

evaluation_status=0
summary_args=()

if [[ "${RUN_GSM8K}" == "1" ]]; then
    run_dataset \
        "gsm8k" "${NORMALIZED_ROOT}/gsm8k" "${GSM8K_LIMIT}" "6000" ||
        evaluation_status=1
    summary_args+=(--gsm8k-count "${GSM8K_LIMIT}")
fi

if [[ "${RUN_MATH}" == "1" ]]; then
    run_dataset \
        "math_500" "${NORMALIZED_ROOT}/math-500" "${MATH_LIMIT}" "6000" ||
        evaluation_status=1
    summary_args+=(--math-count "${MATH_LIMIT}")
fi

if [[ "${RUN_HUMANEVAL}" == "1" ]]; then
    if [[ "${SGLANG_HCU_EVALSCOPE_ALLOW_CODE_EXECUTION:-0}" != "1" ]]; then
        echo "HumanEval executes generated Python locally. Set SGLANG_HCU_EVALSCOPE_ALLOW_CODE_EXECUTION=1 only in an isolated, non-privileged container." >&2
        evaluation_status=1
    else
        run_dataset \
            "humaneval" \
            "${NORMALIZED_ROOT}/humaneval" \
            "${HUMANEVAL_LIMIT}" \
            "${HUMANEVAL_REQUEST_TIMEOUT}" ||
            evaluation_status=1
    fi
    summary_args+=(--humaneval-count "${HUMANEVAL_LIMIT}")
fi

metadata_json="$(
    python3 -c '
import json
import os

print(json.dumps({
    "evalscope_version": "1.9.1",
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "retries": int(os.environ["RETRIES"]),
}))
'
)" || exit 2

if python3 "${HELPER}" summarize \
    --output-root "${OUTPUT_DIR}" \
    --summary "${SUMMARY_PATH}" \
    --model-key "${MODEL_KEY}" \
    --model-name "${MODEL_NAME}" \
    --thresholds-json "${THRESHOLDS_JSON}" \
    --metadata-json "${metadata_json}" \
    "${summary_args[@]}"; then
    summary_status=0
else
    summary_status=$?
fi

if [[ "${evaluation_status}" -ne 0 || "${summary_status}" -ne 0 ]]; then
    exit 1
fi
