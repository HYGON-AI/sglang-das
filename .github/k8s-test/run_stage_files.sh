#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Run every Stage A / Stage B required test file through hcu_ci_exec.sh and write a TSV summary.
set -uo pipefail
out="${1:?output tsv}"
: > "${out}"
fail=0
# list the files inside the CI container (needs PyYAML from the CI image)
bash "${HCU_CI_CHECKOUT_DIR}/scripts/ci/hcu/hcu_ci_exec.sh" -w /sglang-checkout --container-name "${HCU_CI_CONTAINER_NAME}" --   python3 .github/k8s-test/list_stage_files.py > "${out}.files" || { echo "cannot list stage files"; exit 2; }
while IFS= read -r f; do
  [[ -n "${f}" ]] || continue
  start=$(date +%s)
  bash "${HCU_CI_CHECKOUT_DIR}/scripts/ci/hcu/hcu_ci_exec.sh" -w /sglang-checkout/test --container-name "${HCU_CI_CONTAINER_NAME}" \
    -e SGLANG_HCU_RERANKER_MODEL=/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Reranker-0.6B -- \
    timeout 600 python3 "${f}" > "${out}.${start}.log" 2>&1
  rc=$?
  end=$(date +%s)
  printf '%s\t%s\t%s\n' "${f}" "${rc}" "$((end - start))" | tee -a "${out}"
  if [[ ${rc} -ne 0 ]]; then
    fail=$((fail + 1))
    echo "----- tail ${f}"; tail -25 "${out}.${start}.log"
  fi
done < "${out}.files"
echo "files=$(wc -l < "${out}") failed=${fail}"
