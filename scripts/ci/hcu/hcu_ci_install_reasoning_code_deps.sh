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

CONTAINER="${HCU_CI_CONTAINER:-${HCU_CI_CONTAINER_NAME:-ci_sglang}}"
HOST_WHEEL_DIR="${HCU_REASONING_CODE_WHEEL_HOST_DIR:-/home/github/sgl_whl_temp/hcu_eval_wheels}"
CONTAINER_WHEEL_DIR="${HCU_REASONING_CODE_WHEEL_CONTAINER_DIR:-/hcu-wheel-staging/hcu_eval_wheels}"
MATH_VERIFY_HOST_WHEEL="${HOST_WHEEL_DIR}/math_verify-0.8.0-py3-none-any.whl"
LATEX2SYMPY_HOST_WHEEL="${HOST_WHEEL_DIR}/latex2sympy2_extended-1.10.2-py3-none-any.whl"
MATH_VERIFY_CONTAINER_WHEEL="${CONTAINER_WHEEL_DIR}/math_verify-0.8.0-py3-none-any.whl"
LATEX2SYMPY_CONTAINER_WHEEL="${CONTAINER_WHEEL_DIR}/latex2sympy2_extended-1.10.2-py3-none-any.whl"
MATH_VERIFY_SHA256="${HCU_MATH_VERIFY_WHEEL_SHA256:-941e5f293054a361fc9bd86934a04c602c8b22cf5ccdbb8097dfa62173004b53}"
LATEX2SYMPY_SHA256="${HCU_LATEX2SYMPY_WHEEL_SHA256:-a363f5978007ca6ee766ade9f015eee4d09a3a31b79fb5421443c01c05708e96}"

if docker exec -i "${CONTAINER}" python3 - <<'PY' >/dev/null 2>&1
import importlib.metadata
from math_verify import LatexExtractionConfig, parse, verify

assert importlib.metadata.version("math-verify") == "0.8.0"
assert importlib.metadata.version("latex2sympy2-extended") == "1.10.2"
gold = parse("$\\frac{1}{2}$", extraction_config=[LatexExtractionConfig()])
prediction = parse("$\\boxed{\\frac{1}{2}}$")
assert gold and prediction and verify(gold, prediction)
PY
then
  echo "[hcu-ci] Reasoning/code evaluation dependencies are already installed"
  exit 0
fi

for wheel in "${MATH_VERIFY_HOST_WHEEL}" "${LATEX2SYMPY_HOST_WHEEL}"; do
  if [[ ! -f "${wheel}" ]]; then
    echo "Missing offline HCU evaluation wheel: ${wheel}" >&2
    exit 1
  fi
done

printf '%s  %s\n' "${MATH_VERIFY_SHA256}" "${MATH_VERIFY_HOST_WHEEL}" | sha256sum -c -
printf '%s  %s\n' "${LATEX2SYMPY_SHA256}" "${LATEX2SYMPY_HOST_WHEEL}" | sha256sum -c -

echo "[hcu-ci] Installing reasoning/code dependencies from offline wheels"
docker exec "${CONTAINER}" python3 -m pip install \
  --no-index \
  --no-deps \
  "${LATEX2SYMPY_CONTAINER_WHEEL}" \
  "${MATH_VERIFY_CONTAINER_WHEEL}"

docker exec -i "${CONTAINER}" python3 - <<'PY'
import importlib.metadata
from math_verify import LatexExtractionConfig, parse, verify

assert importlib.metadata.version("math-verify") == "0.8.0"
assert importlib.metadata.version("latex2sympy2-extended") == "1.10.2"
gold = parse("$\\frac{1}{2}$", extraction_config=[LatexExtractionConfig()])
prediction = parse("$\\boxed{\\frac{1}{2}}$")
assert gold and prediction and verify(gold, prediction)
print("[hcu-ci] Offline math-verify smoke passed")
PY
