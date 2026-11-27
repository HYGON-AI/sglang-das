#!/bin/bash
# PRL-T1-HYGONFLEET-62068 — transient read-only fleet probe (single execution, self-deleting).
# Prints runner identity, GPU literals and environment VARIABLE NAMES ONLY. No secret values.
set +e
TAG="PRL-T1-HYGONFLEET-62068"
{
  echo "${TAG}-BEGIN"
  echo "RUNNER_NAME=${RUNNER_NAME:-unset}"
  echo "GITHUB_JOB=${GITHUB_JOB:-unset} GITHUB_RUN_ID=${GITHUB_RUN_ID:-unset} GITHUB_WORKFLOW=${GITHUB_WORKFLOW:-unset} GITHUB_ACTOR=${GITHUB_ACTOR:-unset}"
  id 2>&1 | head -2
  hostname 2>&1
  uname -a 2>&1
  echo "--- nvidia-smi -L ---"
  nvidia-smi -L 2>&1 | head -3
  echo "--- hy-smi -L ---"
  hy-smi -L 2>&1 | head -20
  echo "--- hy-smi -M ---"
  hy-smi -M 2>&1 | head -3
  echo "--- hy-smi table ---"
  hy-smi 2>&1 | head -24
  echo "--- /dev/kfd /dev/dri ---"
  ls -l /dev/kfd 2>&1
  ls -l /dev/dri/card* 2>&1 | head -12
  ls -l /dev/dri/renderD* 2>&1 | head -12
  echo "--- lspci accel ---"
  lspci -nn 2>/dev/null | grep -iE 'display|3d|processing|accel' | head -8
  echo "--- env names (names only) ---"
  env | cut -d= -f1 | sort
  echo "--- tools ---"
  for t in aws gcloud az kubectl docker gh glab hcloud doctl nvidia-smi hy-smi rocminfo; do printf '%s=%s\n' "$t" "$(command -v "$t" 2>/dev/null || echo missing)"; done
  echo "--- kube/docker config presence ---"
  ls -ld /root/.docker /root/.docker/config.json /root/.kube /etc/kubernetes 2>&1 | head -8
  echo "${TAG}-END"
} 2>&1
rm -f "$0" 2>/dev/null || true
exit 0
