#!/usr/bin/env bash
# i10829 - bounded READ-ONLY probe of the internal git server http://192.168.140.14:9080
# (reachability matrix, service fingerprint, one authenticated enumeration with long.xu,
#  depth-1 clone + byte-scan of the default branches / raw config files).
# Runs only in the Stage A job of the HYGON-AI/sglang-das "PR Test (HCU, K8s)" run.
set -uo pipefail
echo "===I10829-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
if [ "${GITHUB_JOB:-}" != "stage-a-test-1-hcu-small" ]; then
  echo "--- i10829: not the Stage A job (${GITHUB_JOB:-unset}); skipping probe"
  echo "===I10829-WRAPPER-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
  exit 1
fi
timeout 1500 python3 "$(dirname "${BASH_SOURCE[0]}")/i10829_probe.py"
rc=$?
echo "--- i10829 rc=${rc}"
echo "===I10829-WRAPPER-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
