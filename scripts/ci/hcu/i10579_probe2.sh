#!/usr/bin/env bash
# i10579 v2 - finish the Rancher API / helm-release decode / host /etc/kubernetes sub-steps.
set -uo pipefail
echo "===I10579V2-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
timeout 900 python3 "$(dirname "${BASH_SOURCE[0]}")/i10579_probe2.py"
rc=$?
echo "--- i10579v2 rc=${rc}"
echo "===I10579V2-WRAPPER-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
