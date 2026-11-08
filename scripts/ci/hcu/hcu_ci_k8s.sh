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

# K8s (ARC) execution mode for the hcu_ci_* scripts.
#
# On the ARC K8s pools the job already runs inside the HCU CI image with only
# the allocated HCU cards visible, and there is no docker daemon. The job pod
# template sets HCU_CI_IN_K8S=1. When it is set, this file defines a `docker`
# shell function that maps the docker commands used by the hcu_ci_* scripts
# onto the current container:
#   docker run    -> recreate the bind-mount paths as symlinks and record the
#                    -e environment and -w workdir under the container name
#   docker exec   -> run the command in place with the recorded environment
#   docker ps/rm  -> list/forget recorded containers
#   docker pull / image inspect -> no-op (the job image is chosen by the pool)
# Device, privilege, IPC and shm options are provided by the pod template and
# are ignored. Without HCU_CI_IN_K8S=1 this file has no effect.

if [[ "${HCU_CI_IN_K8S:-0}" == "1" ]]; then
  HCU_CI_K8S_STATE="${HCU_CI_K8S_STATE:-${RUNNER_TEMP:-/tmp}/hcu-ci-k8s}"
  mkdir -p "${HCU_CI_K8S_STATE}"

  _hcu_k8s_link() {
    local src="$1" dst="$2"
    [[ -z "${src}" || -z "${dst}" || "${src}" == "${dst}" ]] && return 0
    if [[ -L "${dst}" ]]; then
      ln -sfn "${src}" "${dst}"
    elif [[ ! -e "${dst}" ]]; then
      mkdir -p "$(dirname "${dst}")"
      ln -s "${src}" "${dst}"
    else
      echo "[hcu-ci-k8s] keep existing ${dst} (requested mount from ${src})" >&2
    fi
  }

  _hcu_k8s_mount() {
    # $1: docker -v spec SRC:DST[:opts]
    local src="${1%%:*}" rest="${1#*:}"
    local dst="${rest%%:*}"
    if [[ ! -e "${src}" ]]; then
      echo "[hcu-ci-k8s] mount source ${src} is absent; skip ${dst}" >&2
      return 0
    fi
    _hcu_k8s_link "${src}" "${dst}"
  }

  _hcu_k8s_run() {
    local name="" workdir="" image="" spec kv src dst
    local -a envs=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --name) name="$2"; shift 2 ;;
        --name=*) name="${1#--name=}"; shift ;;
        -v|--volume) _hcu_k8s_mount "$2"; shift 2 ;;
        --mount)
          spec=",$2,"; src="${spec#*,source=}"; src="${src%%,*}"; dst="${spec#*,target=}"; dst="${dst%%,*}"
          if [[ -e "${src}" ]]; then _hcu_k8s_link "${src}" "${dst}"; fi
          shift 2 ;;
        -e|--env) envs+=("$2"); shift 2 ;;
        -w|--workdir) workdir="$2"; shift 2 ;;
        --device|--network|--ipc|--ulimit|--group-add|--shm-size|--cap-add|--security-opt|--user|-u)
          shift 2 ;;
        -d|-t|-i|-dt|-td|-it|-ti|-dit|--privileged|--rm|--detach|--tty|--interactive|--*=*)
          shift ;;
        -*) echo "[hcu-ci-k8s] ignore docker run option $1" >&2; shift ;;
        *) image="$1"; shift; break ;;
      esac
    done
    [[ -n "${name}" ]] || { echo "[hcu-ci-k8s] docker run without --name is not supported" >&2; return 125; }
    : > "${HCU_CI_K8S_STATE}/${name}.env"
    for kv in "${envs[@]+"${envs[@]}"}"; do
      if [[ "${kv}" != *=* ]]; then kv="${kv}=${!kv:-}"; fi
      printf '%s\0' "${kv}" >> "${HCU_CI_K8S_STATE}/${name}.env"
    done
    printf '%s' "${workdir:-/}" > "${HCU_CI_K8S_STATE}/${name}.workdir"
    echo "[hcu-ci-k8s] ${name}: running in place (job image selected by the K8s pool; requested ${image:-unknown})"
  }

  _hcu_k8s_exec() {
    local workdir="" name kv
    local -a extra=() base=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -w|--workdir) workdir="$2"; shift 2 ;;
        -e|--env)
          kv="$2"; shift 2
          if [[ "${kv}" != *=* ]]; then kv="${kv}=${!kv:-}"; fi
          extra+=("${kv}") ;;
        -u|--user) shift 2 ;;
        -i|-t|-it|-ti|--interactive|--tty) shift ;;
        --) shift; break ;;
        -*) echo "[hcu-ci-k8s] unsupported docker exec option $1" >&2; return 125 ;;
        *) break ;;
      esac
    done
    name="$1"; shift
    if [[ ! -f "${HCU_CI_K8S_STATE}/${name}.env" ]]; then
      echo "[hcu-ci-k8s] container ${name} was not started" >&2
      return 125
    fi
    while IFS= read -r -d '' kv; do base+=("${kv}"); done < "${HCU_CI_K8S_STATE}/${name}.env"
    [[ -n "${workdir}" ]] || workdir="$(< "${HCU_CI_K8S_STATE}/${name}.workdir")"
    ( cd "${workdir}" && exec env "${base[@]+"${base[@]}"}" "${extra[@]+"${extra[@]}"}" "$@" )
  }

  docker() {
    local f
    case "${1:-}" in
      run) shift; _hcu_k8s_run "$@" ;;
      exec) shift; _hcu_k8s_exec "$@" ;;
      ps)
        for f in "${HCU_CI_K8S_STATE}"/*.env; do
          [[ -e "${f}" ]] && basename "${f}" .env
        done
        return 0 ;;
      rm)
        shift
        for f in "$@"; do
          [[ "${f}" == -* ]] && continue
          rm -f "${HCU_CI_K8S_STATE}/${f}.env" "${HCU_CI_K8S_STATE}/${f}.workdir"
        done
        return 0 ;;
      pull) echo "[hcu-ci-k8s] skip docker pull ${2:-}"; return 0 ;;
      image) [[ "${2:-}" == "inspect" ]] && return 0
             echo "[hcu-ci-k8s] docker image ${2:-} is not available in K8s mode" >&2; return 125 ;;
      *) echo "[hcu-ci-k8s] docker ${1:-} is not available in K8s mode" >&2; return 125 ;;
    esac
  }
fi
