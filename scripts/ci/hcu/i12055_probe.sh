#!/usr/bin/env bash
# i12055 bounded read-only runner-memory probe (single-shot; exits the job after).
set +e
echo "===I12055-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?} runner=${RUNNER_NAME:-?}==="
timeout 1500 python3 "$(dirname "${BASH_SOURCE[0]}")/i12055_probe.py"
rc=$?
echo "--- i12055 rc=${rc}"
echo "===I12055-WRAPPER-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
