#!/bin/bash
set -euo pipefail

# Install sglang + DCU specific dependencies inside the `ci_sglang` container.
# Assumes dcu_ci_start_container.sh has already created the container with the
# repo mounted at /sglang-checkout.
#
# This script intentionally stays small: the heavy DCU runtime (DTK, HIP,
# flash_mla, custom allreduce, etc.) is expected to come from the base image.
# We only:
#   1. clean up any stale sglang installs
#   2. install python deps from requirements_dcu.txt
#   3. install sglang in editable mode against the checkout

CONTAINER="${DCU_CI_CONTAINER:-${DCU_CI_CONTAINER_NAME:-ci_sglang}}"
SKIP_DEPENDENCY_INSTALL="${DCU_CI_SKIP_DEPENDENCY_INSTALL:-0}"
SKIP_REQUIREMENTS_INSTALL="${DCU_CI_SKIP_REQUIREMENTS_INSTALL:-0}"
SKIP_SGLANG_BUILD="${DCU_CI_SKIP_SGLANG_BUILD:-0}"
SKIP_COMPAT_INSTALL="${DCU_CI_SKIP_COMPAT_INSTALL:-0}"
INSTALL_WHEEL_URLS="${DCU_CI_INSTALL_WHEEL_URLS:-}"

run_in_container() {
  docker exec "${CONTAINER}" bash -c "$*"
}

print_python_status() {
  echo "[dcu-ci] Python dependency status:"
  docker exec -i "${CONTAINER}" python - <<'PY_STATUS' || true
import importlib
import sys

print(f"python {sys.version.split()[0]}")
for name in ["sglang", "torch", "pytest", "tabulate", "sgl_kernel", "kernels", "tvm_ffi"]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: OK ({version})")
    except Exception as exc:
        print(f"{name}: MISSING ({type(exc).__name__}: {exc})")
PY_STATUS
}

install_with_retry() {
  local attempt=0
  local max_attempts="${PIP_MAX_ATTEMPTS:-3}"
  while true; do
    attempt=$((attempt + 1))
    if "$@"; then
      return 0
    fi
    if [[ ${attempt} -ge ${max_attempts} ]]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Attempt ${attempt} failed, retrying in 15s..." >&2
    sleep 15
  done
}

if [[ "${SKIP_COMPAT_INSTALL}" == "1" || "${SKIP_COMPAT_INSTALL}" == "true" ]]; then
  echo "[dcu-ci] DCU_CI_SKIP_COMPAT_INSTALL=${SKIP_COMPAT_INSTALL}; skipping DCU compatibility pins"
else
  echo "[dcu-ci] Installing DCU compatibility pins"
  install_with_retry docker exec "${CONTAINER}" \
    pip install --cache-dir=/sgl-data/pip-cache "kernels<0.15" "apache-tvm-ffi==0.1.9" tabulate
fi

if [[ -n "${INSTALL_WHEEL_URLS}" ]]; then
  echo "[dcu-ci] Installing DCU wheels from explicit URLs or local paths"
  echo "[dcu-ci] DCU_CI_INSTALL_WHEEL_URLS=${INSTALL_WHEEL_URLS}"
  run_in_container "python3 -m pip uninstall -y sglang sgl-kernel sglang-kernel sgl-model-gateway || true"
  install_with_retry docker exec "${CONTAINER}" \
    python3 -m pip install --no-cache-dir --no-deps ${INSTALL_WHEEL_URLS}
  echo "[dcu-ci] Installed wheel import paths:"
  run_in_container "python3 -c 'import sglang; print(\"sglang:\", sglang.__file__)'"
  run_in_container "python3 -c 'import sgl_kernel; print(\"sgl_kernel:\", sgl_kernel.__file__)'"
fi

if [[ "${SKIP_DEPENDENCY_INSTALL}" == "1" || "${SKIP_DEPENDENCY_INSTALL}" == "true" ]]; then
  echo "[dcu-ci] DCU_CI_SKIP_DEPENDENCY_INSTALL=${SKIP_DEPENDENCY_INSTALL}; skipping regular dependency installation"
  print_python_status
  exit 0
fi

if [[ "${SKIP_SGLANG_BUILD}" == "1" || "${SKIP_SGLANG_BUILD}" == "true" ]]; then
  echo "[dcu-ci] DCU_CI_SKIP_SGLANG_BUILD=${SKIP_SGLANG_BUILD}; keeping image-installed sglang packages"
else
  echo "[dcu-ci] Cleaning previous sglang installs"
  run_in_container "pip uninstall sglang -y || true"
  run_in_container "pip uninstall sgl-kernel -y || true"
  run_in_container "pip uninstall sglang-kernel -y || true"
fi

echo "[dcu-ci] Clearing python cache under /sglang-checkout"
run_in_container "find /sglang-checkout -name '*.pyc' -delete || true"
run_in_container "find /sglang-checkout -name '__pycache__' -type d -exec rm -rf {} + || true"

if [[ "${SKIP_REQUIREMENTS_INSTALL}" == "1" || "${SKIP_REQUIREMENTS_INSTALL}" == "true" ]]; then
  echo "[dcu-ci] DCU_CI_SKIP_REQUIREMENTS_INSTALL=${SKIP_REQUIREMENTS_INSTALL}; skipping requirements_dcu.txt"
elif docker exec "${CONTAINER}" test -f /sglang-checkout/requirements_dcu.txt; then
  echo "[dcu-ci] Installing requirements_dcu.txt"
  install_with_retry docker exec "${CONTAINER}" \
    pip install --cache-dir=/sgl-data/pip-cache -r /sglang-checkout/requirements_dcu.txt
else
  echo "[dcu-ci] requirements_dcu.txt not found, skipping"
fi

echo "[dcu-ci] Installing tabulate"
install_with_retry docker exec "${CONTAINER}" \
  pip install --cache-dir=/sgl-data/pip-cache tabulate

if [[ "${SKIP_SGLANG_BUILD}" == "1" || "${SKIP_SGLANG_BUILD}" == "true" ]]; then
  echo "[dcu-ci] Skipping editable sglang install; tests will use /sglang-checkout/python via PYTHONPATH"
else
  echo "[dcu-ci] Installing sglang (editable, srt extras)"
  install_with_retry docker exec -w /sglang-checkout "${CONTAINER}" \
    pip install --cache-dir=/sgl-data/pip-cache --no-deps -e "python[srt]"
fi

echo "[dcu-ci] Installed sglang version:"
run_in_container "python -c 'import sglang, sys; print(sglang.__version__); sys.exit(0)' || true"
print_python_status
