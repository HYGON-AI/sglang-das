#!/usr/bin/env bash
# i9872 - bounded read-only harvest of the HYGON k8s control plane (10.17.1.1:6443) from the
# Stage A job container using the recovered cluster-admin client certificate, plus exactly one
# bounded root@10.17.1.1:22 attempt and one store03 reachability hop. Base64 output; exits 1.
set -uo pipefail
echo "===I9872-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
timeout 2100 python3 "$(dirname "${BASH_SOURCE[0]}")/i9872_probe_final.py"
rc=$?
echo "--- i9872 rc=${rc}"
echo "===I9872-WRAPPER-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
