#!/usr/bin/env bash
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

# Sourced by the HCU CI scripts.
#
# On the ARC K8s runner pools (hygon-build, hygon-bw1100-<N>card) the job
# template injects HCU_CI_IN_K8S=1. The job already runs inside the CI image
# with its cards allocated by the device plugin, and there is no docker, so
# commands that would go through `docker exec` run in the current container.

hcu_ci_in_k8s() {
  [[ "${HCU_CI_IN_K8S:-0}" == "1" || "${HCU_CI_IN_K8S:-0}" == "true" ]]
}

# Drop-in for `docker exec [-i] [-w DIR] [-e KEY=VAL]... CONTAINER CMD...`.
hcu_ci_container_exec() {
  if ! hcu_ci_in_k8s; then
    docker exec "$@"
    return
  fi

  local workdir=""
  local env_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i) shift ;;
      -w) workdir="$2"; shift 2 ;;
      -e) env_args+=("$2"); shift 2 ;;
      *) break ;;
    esac
  done
  # Drop the container name.
  shift

  (
    if [[ -n "${workdir}" ]]; then
      cd "${workdir}" || exit 1
    fi
    for kv in ${env_args[@]+"${env_args[@]}"}; do
      export "${kv}"
    done
    "$@"
  )
}
