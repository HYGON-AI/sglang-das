#!/bin/bash
set -euo pipefail

# Normalize ownership of the mounted checkout before removing a HCU CI
# container. HCU jobs run containers as root, so files created under the
# bind-mounted checkout can otherwise poison the self-hosted runner workspace
# and break the next actions/checkout.

CONTAINER="${HCU_CI_CONTAINER:-${HCU_CI_CONTAINER_NAME:-ci_sglang}}"
CHECKOUT_MOUNT="${HCU_CI_CHECKOUT_MOUNT:-/sglang-checkout}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
  host_uid="$(id -u)"
  host_gid="$(id -g)"
  docker exec "${CONTAINER}" bash -lc \
    "chown -R ${host_uid}:${host_gid} '${CHECKOUT_MOUNT}' || true" || true
  docker rm -f "${CONTAINER}" || true
fi
