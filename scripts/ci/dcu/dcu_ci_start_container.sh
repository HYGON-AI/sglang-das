#!/bin/bash
set -euo pipefail

# Start a DCU CI container.
#
# Required env / inputs:
#   DCU_CI_IMAGE     Docker image to use. Must point to a DTK/DCU enabled image
#                    that has sglang build dependencies preinstalled.
#                    Override with: --custom-image <image> or --image <image>
#   GITHUB_WORKSPACE Mount point for the checkout. Defaults to $PWD.
#   HF_TOKEN         Optional, forwarded into the container.
#
# Optional env:
#   DCU_CI_CONTAINER / DCU_CI_CONTAINER_NAME  Container name. Defaults to ci_sglang.
#   DCU_DEVICE_FLAGS / DCU_CI_DEVICE_FLAGS    Additional `--device ...` flags.
#   DCU_CI_VISIBLE_DEVICES                    Comma-separated DCU devices to expose.
#   DCU_CACHE_HOST / DCU_CI_CACHE_HOST         Host-side cache directory mounted into /sgl-data.

CUSTOM_IMAGE=""
CONTAINER="${DCU_CI_CONTAINER:-${DCU_CI_CONTAINER_NAME:-ci_sglang}}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --custom-image|--image) CUSTOM_IMAGE="$2"; shift 2;;
    --container-name) CONTAINER="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--custom-image IMAGE|--image IMAGE] [--container-name NAME]"
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ -n "${CUSTOM_IMAGE}" ]]; then
  IMAGE="${CUSTOM_IMAGE}"
elif [[ -n "${DCU_CI_IMAGE:-}" ]]; then
  IMAGE="${DCU_CI_IMAGE}"
else
  echo "Error: DCU_CI_IMAGE env var not set and --custom-image not provided." >&2
  echo "Set DCU_CI_IMAGE to a DTK/DCU enabled sglang dev image." >&2
  exit 1
fi

echo "Using DCU image: ${IMAGE}"

# Pull only if not already present locally, unless explicitly skipped.
if [[ -z "${DCU_CI_SKIP_PULL:-}" ]] && ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Pulling Docker image: ${IMAGE}"
  docker pull "${IMAGE}"
fi

# DCU exposes /dev/kfd + /dev/dri the same way ROCm does. Allow override.
DEVICE_FLAGS="${DCU_DEVICE_FLAGS:-${DCU_CI_DEVICE_FLAGS:---device=/dev/kfd --device=/dev/dri}}"
VISIBLE_DEVICES="${DCU_CI_VISIBLE_DEVICES:-}"
DTK_ROOT="${DCU_DTK_ROOT:-/opt/dtk}"
DCU_LD_LIBRARY_PATH="${DCU_LD_LIBRARY_PATH:-${DTK_ROOT}/hip/lib:${DTK_ROOT}/lib:${DTK_ROOT}/lib64:${DTK_ROOT}/hsa/lib:${DTK_ROOT}/llvm/lib:${DTK_ROOT}/dcc/gcvm/lib:${DTK_ROOT}/.hyhal/lib:${DTK_ROOT}/.hyhal/lib64:${DTK_ROOT}/.hyhal/rocm_smi/lib:${DTK_ROOT}/.hyhal/hydm/lib:/opt/hyhal/lib:/opt/hyhal/lib64}"

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
  if [[ "${DCU_CI_SET_ROCR_VISIBLE_DEVICES:-0}" == "1" ]]; then
    VISIBLE_ENV_ARGS+=(-e "ROCR_VISIBLE_DEVICES=${VISIBLE_DEVICES}")
  fi
fi

CACHE_HOST="${DCU_CACHE_HOST:-${DCU_CI_CACHE_HOST:-/home/runner/sgl-data}}"
if [[ -d "${CACHE_HOST}" ]]; then
  CACHE_VOLUME="-v ${CACHE_HOST}:/sgl-data"
else
  CACHE_VOLUME=""
fi

MODEL_HOST_PATH="${DCU_MODEL_HOST_PATH:-/public/opendas/DL_DATA/llm-models}"
if [[ -d "${MODEL_HOST_PATH}" ]]; then
  # This mount also exposes the default DCU accuracy datasets under llm-models.
  MODEL_VOLUME="-v ${MODEL_HOST_PATH}:${MODEL_HOST_PATH}:ro"
else
  MODEL_VOLUME=""
fi

# Remove any leftover container from a previous run.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "Launching container: ${CONTAINER}"
docker run -dt --user root --privileged \
  --network=host \
  --ipc=host \
  ${DEVICE_FLAGS} \
  --ulimit nofile=65536:65536 \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  -v /opt/hyhal:/opt/hyhal:ro \
  ${CACHE_VOLUME} \
  ${MODEL_VOLUME} \
  --group-add video \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/sgl-data/hf-cache \
  -e HF_HUB_ETAG_TIMEOUT=300 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  -e ROCM_PATH="${DTK_ROOT}" \
  -e LD_LIBRARY_PATH="${DCU_LD_LIBRARY_PATH}" \
  "${VISIBLE_ENV_ARGS[@]}" \
  -e SGLANG_IS_IN_CI=1 \
  -e SGLANG_IS_IN_CI_DCU=1 \
  -e SGLANG_USE_AITER=0 \
  -e SGLANG_ROCM_USE_AITER_MOE=0 \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name "${CONTAINER}" \
  "${IMAGE}"

# Git >= 2.35.2 refuses cross-user repos; mark the mount as safe.
docker exec "${CONTAINER}" git config --global --add safe.directory /sglang-checkout
