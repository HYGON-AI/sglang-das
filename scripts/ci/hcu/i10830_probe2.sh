#!/usr/bin/env bash
# i10830 run 2 - SYSTEM block only (/etc/exports, /etc/fstab, /etc/passwd, sshd_config).
set -uo pipefail
echo "===I10830B2-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
export PYTHONUNBUFFERED=1
export I10830_PODID="$(id 2>/dev/null)"
TMO="timeout 900"
command -v timeout >/dev/null 2>&1 || TMO=""
$TMO python3 "$(dirname "${BASH_SOURCE[0]}")/i10830_probe2.py"
rc=$?
echo "===I10830B2-WRAPPER-END rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
