#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Build HCU wheels for sglang / sgl-kernel / sglang-router with a DAS local
# version, e.g. 0.5.18+das.dtk2604.torch2110.2609211530.g023409.
#
# Usage:
#   scripts/build_hcu_wheels.sh [all|sglang|kernel|router]...   (default: all)
#   scripts/build_hcu_wheels.sh clean                           drop build intermediates
#   scripts/build_hcu_wheels.sh clean_all                       clean + output/ + cargo target
#   scripts/build_hcu_wheels.sh clean_all sglang                clean first, then build
#
# Env overrides (all optional):
#   SGLANG_VERSION   base version for the sglang wheel      (default: 0.5.18)
#   TORCH_VERSION    torch version tag, e.g. 2.11.0         (default: installed torch)
#   DTK_VERSION      DTK tag without dots, e.g. 2604        (default: from /opt/dtk symlink)
#   DTK_HOME         DTK install root                       (default: /opt/dtk)
#   GIT_COMMIT       commit used for the g<sha6> suffix     (default: HEAD)
#   BUILD_TIMESTAMP  YYMMDDHHMM in the local version        (default: now)
#   LOCAL_VERSION    full local version, overrides the derived one
#   OUTPUT_DIR       where final wheels land                (default: <repo>/output)
#   USE_CIUPLOAD     1/0, force CIUpload REPAIR on/off      (default: auto-detect)
#   SKIP_AUDITWHEEL  1 to skip auditwheel on the router wheel (default: 0)
#   MATURIN_FEATURES cargo features for the router          (default: vendored-openssl)
#   CLEAN_BEFORE_BUILD  clean|clean_all, run that clean before every build

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="${REPO_ROOT}/python"
KERNEL_DIR="${PYTHON_DIR}/sglang/kernels/aot"
GATEWAY_DIR="${REPO_ROOT}/sgl-model-gateway"
ROUTER_DIR="${GATEWAY_DIR}/bindings/python"

SGLANG_VERSION="${SGLANG_VERSION:-0.5.18}"
DTK_HOME="${DTK_HOME:-/opt/dtk}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output}"
SKIP_AUDITWHEEL="${SKIP_AUDITWHEEL:-0}"
MATURIN_FEATURES="${MATURIN_FEATURES:-vendored-openssl}"

log() { echo "[build_hcu_wheels] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# ----------------------------------------------------------------------------
# Target selection
# ----------------------------------------------------------------------------
BUILD_SGLANG=0
BUILD_KERNEL=0
BUILD_ROUTER=0
CLEAN_MODE=""
if (( $# == 0 )); then
  set -- all
fi
for target in "$@"; do
  case "${target}" in
    all)    BUILD_SGLANG=1; BUILD_KERNEL=1; BUILD_ROUTER=1 ;;
    sglang) BUILD_SGLANG=1 ;;
    kernel|sgl-kernel|sglang-kernel) BUILD_KERNEL=1 ;;
    router|sglang-router|gateway)    BUILD_ROUTER=1 ;;
    clean)     [[ "${CLEAN_MODE}" == "clean_all" ]] || CLEAN_MODE="clean" ;;
    clean_all|clean-all|cleanall) CLEAN_MODE="clean_all" ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown target: ${target} (expected all|sglang|kernel|router|clean|clean_all)" ;;
  esac
done
# A bare `clean` / `clean_all` cleans and exits; combined with a build target it
# runs first. CLEAN_BEFORE_BUILD makes that implicit for every invocation.
CLEAN_ONLY=0
if [[ -n "${CLEAN_MODE}" ]] && (( BUILD_SGLANG + BUILD_KERNEL + BUILD_ROUTER == 0 )); then
  CLEAN_ONLY=1
fi
if [[ -z "${CLEAN_MODE}" && -n "${CLEAN_BEFORE_BUILD:-}" ]]; then
  case "${CLEAN_BEFORE_BUILD}" in
    clean|clean_all) CLEAN_MODE="${CLEAN_BEFORE_BUILD}" ;;
    *) die "CLEAN_BEFORE_BUILD must be clean or clean_all: ${CLEAN_BEFORE_BUILD}" ;;
  esac
fi

# ----------------------------------------------------------------------------
# Clean: drop build history left by earlier runs.
#
# `clean` keeps the finished wheels in OUTPUT_DIR and the cargo target cache (a
# full rebuild of the router costs ~20 min); `clean_all` drops those too.
# ----------------------------------------------------------------------------
remove_path() {
  local path="$1"
  [[ -e "${path}" ]] || return 0
  log "rm -rf ${path#"${REPO_ROOT}/"}"
  rm -rf -- "${path}"
}

# Undo the in-place edits a crashed run may have left behind: the pyproject_hcu
# swap, and the "+das.*" version suffix written when CIUpload is unavailable.
restore_leftover_edits() {
  git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1 || return 0
  if [[ -f "${PYTHON_DIR}/pyproject.toml" ]] \
     && cmp -s "${PYTHON_DIR}/pyproject.toml" "${PYTHON_DIR}/pyproject_hcu.toml"; then
    log "restoring python/pyproject.toml (leftover HCU swap)"
    git -C "${REPO_ROOT}" checkout -- python/pyproject.toml 2>/dev/null || true
  fi
  local pyproject
  for pyproject in "${KERNEL_DIR}/pyproject.toml" "${ROUTER_DIR}/pyproject.toml"; do
    if grep -qm1 '^version = ".*+das\.' "${pyproject}" 2>/dev/null; then
      log "restoring ${pyproject#"${REPO_ROOT}/"} (leftover local version)"
      git -C "${REPO_ROOT}" checkout -- "${pyproject}" 2>/dev/null || true
    fi
  done
}

run_clean() {
  local mode="$1"
  log "=== clean (${mode}) ==="
  restore_leftover_edits

  remove_path "${PYTHON_DIR}/build"
  remove_path "${PYTHON_DIR}/dist"
  remove_path "${PYTHON_DIR}/sglang/_version.py"
  local egg
  for egg in "${PYTHON_DIR}"/*.egg-info "${KERNEL_DIR}"/*.egg-info; do
    remove_path "${egg}"
  done

  remove_path "${KERNEL_DIR}/build"
  remove_path "${KERNEL_DIR}/dist"
  remove_path "${KERNEL_DIR}/_skbuild"
  # hipify rewrites every .cu/.cuh next to the original as .hip/_hip.h; those are
  # gitignored generated files and go stale when the .cu changes.
  if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    log "git clean -Xdf on the kernel csrc/include trees"
    git -C "${REPO_ROOT}" clean -Xdfq -- \
      "python/sglang/kernels/aot/csrc" "python/sglang/kernels/aot/include" || true
  fi

  remove_path "${GATEWAY_DIR}/target/wheels"
  remove_path "${ROUTER_DIR}/target/wheels"

  if [[ "${mode}" == "clean_all" ]]; then
    remove_path "${OUTPUT_DIR}"
    local target_dir
    for target_dir in "${GATEWAY_DIR}/target" "${REPO_ROOT}/rust/target"; do
      remove_path "${target_dir}"
    done
  fi
  log "=== clean done ==="
}

if [[ -n "${CLEAN_MODE}" ]]; then
  run_clean "${CLEAN_MODE}"
  if (( CLEAN_ONLY )); then
    exit 0
  fi
fi

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
if [[ -f "${DTK_HOME}/env.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${DTK_HOME}/env.sh"
  set -u
else
  log "WARNING: ${DTK_HOME}/env.sh not found; assuming DTK already on PATH"
fi
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi

if [[ -z "${TORCH_VERSION:-}" ]]; then
  TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null)" \
    || die "torch not importable; set TORCH_VERSION explicitly"
fi
if [[ -z "${DTK_VERSION:-}" ]]; then
  DTK_VERSION="$(readlink -f "${DTK_HOME}" | grep -oP 'dtk-?\K[0-9]+\.[0-9]+' | head -1 | tr -d '.')"
  [[ -n "${DTK_VERSION}" ]] || die "cannot derive DTK version from ${DTK_HOME}; set DTK_VERSION"
fi

GIT_COMMIT="${GIT_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
GIT_SHORT="${GIT_COMMIT:0:6}"
BUILD_TIMESTAMP="${BUILD_TIMESTAMP:-$(date +%y%m%d%H%M)}"
TORCH_TAG="$(printf '%s' "${TORCH_VERSION}" | tr -d '.')"
LOCAL_VERSION="${LOCAL_VERSION:-das.dtk${DTK_VERSION}.torch${TORCH_TAG}.${BUILD_TIMESTAMP}.g${GIT_SHORT}}"

if [[ -z "${USE_CIUPLOAD:-}" ]]; then
  if command -v CIUpload >/dev/null 2>&1; then USE_CIUPLOAD=1; else USE_CIUPLOAD=0; fi
fi

log "repo:          ${REPO_ROOT}"
log "commit:        ${GIT_COMMIT}"
log "torch:         ${TORCH_VERSION}"
log "dtk:           ${DTK_VERSION}"
log "local version: +${LOCAL_VERSION} (CIUpload=${USE_CIUPLOAD})"
log "targets:       sglang=${BUILD_SGLANG} kernel=${BUILD_KERNEL} router=${BUILD_ROUTER}"
log "output:        ${OUTPUT_DIR}"

# ----------------------------------------------------------------------------
# Restore any files we rewrite (pyproject swaps / version edits) on exit.
# ----------------------------------------------------------------------------
RESTORE_FILES=()
restore_files() {
  for f in "${RESTORE_FILES[@]:-}"; do
    [[ -n "${f}" ]] && git -C "${REPO_ROOT}" checkout -- "${f}" 2>/dev/null || true
  done
}
trap restore_files EXIT INT TERM

# Without CIUpload, bake the local version into each package's own version
# source so the wheel filename and METADATA already carry it.
manual_suffix() {
  if (( USE_CIUPLOAD )); then printf ''; else printf '+%s' "${LOCAL_VERSION}"; fi
}

set_pyproject_version_suffix() {
  local pyproject="$1" suffix="$2"
  [[ -n "${suffix}" ]] || return 0
  RESTORE_FILES+=("${pyproject}")
  sed -i "0,/^version = \"\([^\"+]*\)\"/s//version = \"\1${suffix}\"/" "${pyproject}"
  grep -m1 '^version = ' "${pyproject}" >&2
}

RAW_WHEELS=()

# ----------------------------------------------------------------------------
# sgl-kernel: setup_hip.py reads the version line of aot/pyproject.toml, so
# SETUPTOOLS_SCM_PRETEND_VERSION has no effect here.
# ----------------------------------------------------------------------------
build_kernel() {
  log "=== building sgl-kernel ==="
  set_pyproject_version_suffix "${KERNEL_DIR}/pyproject.toml" "$(manual_suffix)"
  (
    cd "${KERNEL_DIR}"
    rm -rf build dist
    python3 setup_hip.py bdist_wheel
  )
  RAW_WHEELS+=("${KERNEL_DIR}"/dist/*.whl)
}

# ----------------------------------------------------------------------------
# sglang: setuptools-scm; the HCU dependency list lives in pyproject_hcu.toml.
# ----------------------------------------------------------------------------
build_sglang() {
  log "=== building sglang ==="
  RESTORE_FILES+=("${PYTHON_DIR}/pyproject.toml")
  cp "${PYTHON_DIR}/pyproject_hcu.toml" "${PYTHON_DIR}/pyproject.toml"
  (
    cd "${PYTHON_DIR}"
    rm -rf build dist
    SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_VERSION}$(manual_suffix)" \
      python3 -m build --wheel --no-isolation -Cfeatures=all_hip
  )
  RAW_WHEELS+=("${PYTHON_DIR}"/dist/*.whl)
}

# ----------------------------------------------------------------------------
# sglang-router: maturin takes [project].version from bindings/python/pyproject.
# ----------------------------------------------------------------------------
build_router() {
  log "=== building sglang-router ==="
  command -v maturin >/dev/null 2>&1 || die "maturin not found (pip install 'maturin<1.8')"
  command -v cargo >/dev/null 2>&1 || die "cargo not found"
  set_pyproject_version_suffix "${ROUTER_DIR}/pyproject.toml" "$(manual_suffix)"
  local feature_args=()
  [[ -n "${MATURIN_FEATURES}" ]] && feature_args=(--features "${MATURIN_FEATURES}")
  (
    cd "${GATEWAY_DIR}"
    rm -rf target/wheels
    maturin build --manifest-path bindings/python/Cargo.toml --release \
      --skip-auditwheel "${feature_args[@]}"
  )
  RAW_WHEELS+=("${GATEWAY_DIR}"/target/wheels/*.whl)
}

if (( BUILD_KERNEL )); then build_kernel; fi
if (( BUILD_SGLANG )); then build_sglang; fi
if (( BUILD_ROUTER )); then build_router; fi

(( ${#RAW_WHEELS[@]} > 0 )) || die "no wheels were built"

# ----------------------------------------------------------------------------
# Finalize: append local version (CIUpload) or copy, then repair router wheel.
# ----------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
STAGE_DIR="$(mktemp -d)"
if (( USE_CIUPLOAD )); then
  log "=== CIUpload REPAIR ==="
  CIUpload REPAIR \
    --dtk_version "${DTK_VERSION}" \
    --torch_version "${TORCH_VERSION}" \
    --git_commit "${GIT_SHORT}" \
    --outputdir "${STAGE_DIR}" \
    -f "${RAW_WHEELS[@]}"
else
  cp -v "${RAW_WHEELS[@]}" "${STAGE_DIR}/"
fi

for wheel in "${STAGE_DIR}"/*.whl; do
  name="$(basename "${wheel}")"
  if [[ "${name}" == sglang_router-* && "${SKIP_AUDITWHEEL}" != "1" ]] \
     && command -v auditwheel >/dev/null 2>&1; then
    auditwheel repair --plat manylinux_2_28_x86_64 --wheel-dir "${OUTPUT_DIR}" "${wheel}"
  else
    cp -v "${wheel}" "${OUTPUT_DIR}/"
  fi
done
rm -rf "${STAGE_DIR}"

log "=== done ==="
ls -lh "${OUTPUT_DIR}"/*.whl >&2
