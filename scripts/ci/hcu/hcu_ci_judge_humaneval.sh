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

MODEL_KEY="${1:?usage: hcu_ci_judge_humaneval.sh MODEL_KEY}"
CHECKOUT="${HCU_CI_CHECKOUT_DIR:-${GITHUB_WORKSPACE:-$PWD}}"
IMAGE="${HCU_CI_IMAGE:?HCU_CI_IMAGE is required}"
ARTIFACT_ROOT="${HCU_REASONING_CODE_ARTIFACT_ROOT:-${CHECKOUT}/test-results/hcu-reasoning-code}"
MODEL_DIR="${ARTIFACT_ROOT}/${MODEL_KEY}"
MANIFEST="${MODEL_DIR}/manifest.json"
SAMPLES="${MODEL_DIR}/humaneval_samples.jsonl"
JUDGE="${CHECKOUT}/test/hcu_humaneval_judge.py"
OUTPUT_DIR="${MODEL_DIR}/humaneval_judge"

for path in "${MANIFEST}" "${SAMPLES}" "${JUDGE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing HumanEval judge input: ${path}" >&2
    exit 1
  fi
done

PROBLEMS="$(jq -er '.humaneval.data_path' "${MANIFEST}")"
EXPECTED="$(jq -er '.humaneval.num_examples' "${MANIFEST}")"
THRESHOLD="$(jq -er '.humaneval.threshold' "${MANIFEST}")"
TIMEOUT="$(jq -er '.humaneval.timeout_seconds' "${MANIFEST}")"
if [[ ! -f "${PROBLEMS}" ]]; then
  echo "Missing HumanEval problem file on the runner: ${PROBLEMS}" >&2
  exit 1
fi

case "${OUTPUT_DIR}" in
  "${ARTIFACT_ROOT}"/*) ;;
  *)
    echo "Refusing to clean output outside ${ARTIFACT_ROOT}: ${OUTPUT_DIR}" >&2
    exit 1
    ;;
esac
rm -rf -- "${OUTPUT_DIR}"
mkdir -m 0777 -p "${OUTPUT_DIR}"

SAFE_MODEL_KEY="${MODEL_KEY//[^A-Za-z0-9_.-]/-}"
CONTAINER="hcu-humaneval-${GITHUB_RUN_ID:-local}-${SAFE_MODEL_KEY}"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
trap 'docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true' EXIT

docker run --rm \
  --name "${CONTAINER}" \
  --network none \
  --read-only \
  --user 65534:65534 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 4g \
  --cpus 4 \
  --ulimit nofile=256:256 \
  --ulimit nproc=128:128 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=256m \
  -v "${JUDGE}:/judge.py:ro" \
  -v "${SAMPLES}:/input/samples.jsonl:ro" \
  -v "${PROBLEMS}:/input/problems.jsonl:ro" \
  -v "${OUTPUT_DIR}:/output:rw" \
  --entrypoint python3 \
  "${IMAGE}" \
  /judge.py \
    --samples /input/samples.jsonl \
    --problems /input/problems.jsonl \
    --expected "${EXPECTED}" \
    --threshold "${THRESHOLD}" \
    --timeout "${TIMEOUT}" \
    --workers 4 \
    --output-dir /output

cat "${OUTPUT_DIR}/humaneval_summary.json"
