#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
umask 0002

MODE="${1:-}"
if [[ "${MODE}" != "build" && "${MODE}" != "wait" ]]; then
  echo "Usage: $0 build|wait" >&2
  exit 2
fi

SHA="${HCU_PD_SHA:?HCU_PD_SHA is required}"
CHECKOUT="${HCU_PD_CHECKOUT:-${GITHUB_WORKSPACE:-$PWD}}"
IMAGE="${HCU_PD_IMAGE:?HCU_PD_IMAGE is required}"
BUILD_IMAGE="${HCU_PD_BUILD_IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/dev:rockylinux8.6-mpi5.0-gcc10.3-cmake3.29-py3.10-mkl2020.4.304}"
WHEEL_ROOT="${HCU_PD_WHEEL_ROOT:-/ci_public/sglang-das/hcu-wheels}"
WAIT_TIMEOUT="${HCU_PD_WHEEL_TIMEOUT:-10800}"
SHARED_GID="${HCU_PD_SHARED_GID:-1002}"

if [[ ! "${SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "HCU_PD_SHA must be a full lowercase Git SHA: ${SHA}" >&2
  exit 2
fi
if [[ ! -d "${CHECKOUT}/python" || ! -d "${CHECKOUT}/python/sglang/kernels/aot" ]]; then
  echo "Invalid HCU PD checkout: ${CHECKOUT}" >&2
  exit 2
fi

FINAL_DIR="${WHEEL_ROOT}/${SHA}"
PIP_INDEX_URL="${HCU_PD_PIP_INDEX_URL:-http://10.16.1.201:9929/nightly/dtk2604/+simple/}"
PIP_TRUSTED_HOST="${HCU_PD_PIP_TRUSTED_HOST:-10.16.1.201}"
TORCH_VERSION="${HCU_PD_TORCH_VERSION:-2.10.0}"
RESOURCE_SERVER="${HCU_PD_RESOURCE_SERVER:-http://10.16.1.201:8000}"
DTK_PKG_URL="${HCU_PD_DTK_PKG_URL:-${RESOURCE_SERVER}/dtk-pkg/dtk26.04/DTK-26.04-rc4-centos8-x86_64.tar.gz}"
PROTOC_URL="${HCU_PD_PROTOC_URL:-${RESOURCE_SERVER}/Jenkins/CompileDep/sglang/protoc-24.4-linux-x86_64.zip}"
RUSTUP_INIT_URL="${HCU_PD_RUSTUP_INIT_URL:-${RESOURCE_SERVER}/Jenkins/CompileDep/sglang/rustup-init.sh}"
RUSTUP_DIST_SERVER="${HCU_PD_RUSTUP_DIST_SERVER:-https://mirrors.tuna.tsinghua.edu.cn/rustup}"
RUSTUP_UPDATE_ROOT="${HCU_PD_RUSTUP_UPDATE_ROOT:-https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup}"

mkdir -p "${WHEEL_ROOT}"
chmod 2775 "${WHEEL_ROOT}" || true
chgrp "${SHARED_GID}" "${WHEEL_ROOT}" || true

validate_bundle() {
  local bundle_dir="$1"
  python3 - "${bundle_dir}" "${SHA}" <<'PY_VALIDATE'
import json
import pathlib
import sys

bundle = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
manifest_path = bundle / "manifest.json"
ready_path = bundle / "READY"
if not ready_path.is_file() or not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
if manifest.get("commit_sha") != expected_sha:
    raise SystemExit(1)
kinds = {item.get("kind") for item in manifest.get("wheels", [])}
if kinds != {"sglang", "sglang-kernel", "sglang-router"}:
    raise SystemExit(1)
for item in manifest["wheels"]:
    path = bundle / item["path"]
    if not path.is_file():
        raise SystemExit(1)
PY_VALIDATE
}

wait_for_bundle() {
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < deadline )); do
    if validate_bundle "${FINAL_DIR}" 2>/dev/null; then
      echo "${FINAL_DIR}"
      return 0
    fi
    sleep 10
  done
  echo "Timed out waiting for shared HCU wheels: ${FINAL_DIR}" >&2
  return 1
}

if validate_bundle "${FINAL_DIR}" 2>/dev/null; then
  echo "Reusing shared HCU wheel bundle: ${FINAL_DIR}" >&2
  echo "${FINAL_DIR}"
  exit 0
fi

if [[ "${MODE}" == "wait" ]]; then
  wait_for_bundle
  exit $?
fi

RUN_KEY="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
LOCK_DIR="${WHEEL_ROOT}/.${SHA}.lock"
TEMP_DIR="${WHEEL_ROOT}/.${SHA}.tmp-${RUN_KEY}-$$"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another process is publishing ${SHA}; waiting for READY." >&2
  wait_for_bundle
  exit $?
fi

cleanup_build_state() {
  if [[ -d "${TEMP_DIR}" ]]; then
    rm -rf -- "${TEMP_DIR}"
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_build_state EXIT INT TERM

mkdir -p "${TEMP_DIR}/wheels"
chmod 2775 "${TEMP_DIR}" "${TEMP_DIR}/wheels" || true
chgrp "${SHARED_GID}" "${TEMP_DIR}" "${TEMP_DIR}/wheels" || true

PACKAGE_VERSION="0.5.15.post1+pd.g${SHA:0:12}"
DTK_VERSION="$(
  printf '%s' "${DTK_PKG_URL}" \
    | grep -oP 'dtk\K[0-9]+\.[0-9]+' \
    | head -1 \
    | tr -d '.'
)"
if [[ -z "${DTK_VERSION}" ]]; then
  echo "Cannot derive the DTK version from ${DTK_PKG_URL}" >&2
  exit 2
fi
echo "Building HCU PD wheels for ${SHA} with ${BUILD_IMAGE}" >&2

docker run --rm \
  --network host \
  --ipc host \
  --privileged \
  --user root \
  -e "HCU_PD_PACKAGE_VERSION=${PACKAGE_VERSION}" \
  -e "HCU_PD_DTK_VERSION=${DTK_VERSION}" \
  -e "HCU_PD_GIT_COMMIT=${SHA:0:6}" \
  -e "PIP_INDEX_URL=${PIP_INDEX_URL}" \
  -e "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}" \
  -e "CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse" \
  -e "CARGO_NET_GIT_FETCH_WITH_CLI=true" \
  -e "GIT_TERMINAL_PROMPT=0" \
  -e "HCU_PD_TORCH_VERSION=${TORCH_VERSION}" \
  -e "HCU_PD_DTK_PKG_URL=${DTK_PKG_URL}" \
  -e "HCU_PD_PROTOC_URL=${PROTOC_URL}" \
  -e "HCU_PD_RUSTUP_INIT_URL=${RUSTUP_INIT_URL}" \
  -e "RUSTUP_DIST_SERVER=${RUSTUP_DIST_SERVER}" \
  -e "RUSTUP_UPDATE_ROOT=${RUSTUP_UPDATE_ROOT}" \
  -v "${CHECKOUT}:/source:ro" \
  -v "${TEMP_DIR}:/publish" \
  "${BUILD_IMAGE}" \
  bash -lc '
    set -euo pipefail
    mkdir -p /root/.cargo
    cat > /root/.cargo/config << "EOF_CARGO"
[source.crates-io]
replace-with = "rsproxy-sparse"

[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"

[net]
retry = 5
git-fetch-with-cli = true

[http]
low-speed-limit = 1
EOF_CARGO
    python3 -m pip install --no-cache-dir \
      "torch==${HCU_PD_TORCH_VERSION}" \
      build \
      scikit-build-core \
      cmake \
      ninja \
      wheel \
      setuptools \
      setuptools-scm \
      setuptools-rust \
      "maturin==1.9.6" \
      ciupload \
      auditwheel \
      patchelf
    wget -q "${HCU_PD_PROTOC_URL}" -O /tmp/protoc.zip
    unzip -q -o /tmp/protoc.zip -d /usr/local
    chmod +x /usr/local/bin/protoc
    ln -sf /usr/local/bin/protoc /usr/bin/protoc
    wget -q "${HCU_PD_RUSTUP_INIT_URL}" -O /tmp/rustup-init.sh
    bash /tmp/rustup-init.sh -y
    source /root/.cargo/env

    wget -q "${HCU_PD_DTK_PKG_URL}" -O /tmp/dtk.tar.gz
    rm -rf /opt/dtk /opt/dtk-*
    tar -xzf /tmp/dtk.tar.gz -C /opt
    mv /opt/dtk-* /opt/dtk
    set +u
    source /opt/dtk/env.sh
    set -u
    rustc --version
    cargo --version
    rm -rf /tmp/sglang-pd-source
    mkdir -p /tmp/sglang-pd-source
    mkdir -p /tmp/raw-wheels
    cp -a /source/. /tmp/sglang-pd-source/

    cd /tmp/sglang-pd-source/python/sglang/kernels/aot
    rm -rf build dist
    SETUPTOOLS_SCM_PRETEND_VERSION="${HCU_PD_PACKAGE_VERSION}" \
      python3 setup_hip.py bdist_wheel
    cp -v dist/*.whl /tmp/raw-wheels/

    cd /tmp/sglang-pd-source/python
    rm -rf build dist
    SETUPTOOLS_SCM_PRETEND_VERSION="${HCU_PD_PACKAGE_VERSION}" \
      python3 -m build --wheel --no-isolation -Cfeatures=all_hip
    cp -v dist/*.whl /tmp/raw-wheels/

    cd /tmp/sglang-pd-source/sgl-model-gateway/bindings/python
    RUSTUP_TOOLCHAIN=stable maturin build --release --skip-auditwheel
    cp -v target/wheels/*.whl /tmp/raw-wheels/

    mkdir -p /tmp/ci-repaired-wheels
    mapfile -t raw_wheels < <(find /tmp/raw-wheels -maxdepth 1 -type f -name "*.whl" -print | sort)
    if (( ${#raw_wheels[@]} == 0 )); then
      echo "No HCU PD wheels were built" >&2
      exit 1
    fi
    CIUpload REPAIR \
      --dtk_version "${HCU_PD_DTK_VERSION}" \
      --torch_version "${HCU_PD_TORCH_VERSION}" \
      --git_commit "${HCU_PD_GIT_COMMIT}" \
      --outputdir /tmp/ci-repaired-wheels \
      -f "${raw_wheels[@]}"
    for wheel in /tmp/ci-repaired-wheels/*.whl; do
      if [[ "$(basename "${wheel}")" == sglang_router-* ]]; then
        auditwheel repair \
          --plat manylinux_2_28_x86_64 \
          --wheel-dir /publish/wheels \
          "${wheel}"
      else
        cp -v "${wheel}" /publish/wheels/
      fi
    done
  '

export SHA IMAGE BUILD_IMAGE PACKAGE_VERSION TEMP_DIR
python3 - <<'PY_MANIFEST'
from email.parser import Parser
import hashlib
import json
import os
import pathlib
import zipfile

temp_dir = pathlib.Path(os.environ["TEMP_DIR"])
wheel_dir = temp_dir / "wheels"


def wheel_local_version(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected one METADATA file in {path.name}")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
    version = metadata.get("Version", "")
    _, separator, local_version = version.partition("+")
    if not separator or not local_version:
        raise RuntimeError(f"Missing local version in {path.name}: {version}")
    return local_version


def wheel_kind(path: pathlib.Path) -> str | None:
    name = path.name.lower().replace("-", "_")
    if name.startswith("sglang_router_"):
        return "sglang-router"
    if name.startswith(("sglang_kernel_", "sgl_kernel_")):
        return "sglang-kernel"
    if name.startswith("sglang_"):
        return "sglang"
    return None


found = {}
wheel_local_versions = {}
for wheel in sorted(wheel_dir.glob("*.whl")):
    kind = wheel_kind(wheel)
    if kind is None:
        continue
    if kind in found:
        raise SystemExit(f"duplicate {kind} wheels: {found[kind]} and {wheel}")
    found[kind] = wheel
    wheel_local_versions[wheel.name] = wheel_local_version(wheel)

expected = {"sglang", "sglang-kernel", "sglang-router"}
if set(found) != expected:
    raise SystemExit(f"wheel bundle mismatch: expected={expected}, found={set(found)}")

unique_local_versions = set(wheel_local_versions.values())
local_version = next(iter(unique_local_versions)) if len(unique_local_versions) == 1 else None

wheels = []
for kind in sorted(found):
    path = found[kind]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    wheels.append(
        {
            "kind": kind,
            "path": str(path.relative_to(temp_dir)),
            "filename": path.name,
            "sha256": digest,
            "size": path.stat().st_size,
            "source": "repository-build",
        }
    )

manifest = {
    "repository": os.environ.get("GITHUB_REPOSITORY", ""),
    "run_id": os.environ.get("GITHUB_RUN_ID", ""),
    "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
    "commit_sha": os.environ["SHA"],
    "image": os.environ["IMAGE"],
    "build_image": os.environ["BUILD_IMAGE"],
    "package_version": os.environ["PACKAGE_VERSION"],
    "local_version": local_version,
    "wheel_local_versions": wheel_local_versions,
    "wheels": wheels,
}
(temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY_MANIFEST

find "${TEMP_DIR}" -type d -exec chmod 2775 {} +
find "${TEMP_DIR}" -type f -exec chmod 0664 {} +
chgrp -R "${SHARED_GID}" "${TEMP_DIR}" || true
printf '%s\n' "${SHA}" > "${TEMP_DIR}/READY"
chmod 0664 "${TEMP_DIR}/READY"
chgrp "${SHARED_GID}" "${TEMP_DIR}/READY" || true

if [[ -e "${FINAL_DIR}" ]]; then
  INVALID_DIR="${WHEEL_ROOT}/.${SHA}.invalid-$(date +%s)"
  mv "${FINAL_DIR}" "${INVALID_DIR}"
fi
mv "${TEMP_DIR}" "${FINAL_DIR}"
rmdir "${LOCK_DIR}" 2>/dev/null || true
trap - EXIT INT TERM

validate_bundle "${FINAL_DIR}"
echo "Published shared HCU wheel bundle: ${FINAL_DIR}" >&2
echo "${FINAL_DIR}"
