#!/usr/bin/env bash
# i9454 - bounded read-only layer scan of the PRIVATE HYGON Harbor projects from the HCU
# Stage A job container, plus FTP package-host credential validation and one internal
# pip-host probe. All probe output is base64 encoded (C211B64 lines). Read-only.
# Exits 1 right after printing evidence so no test suite runs.
set -uo pipefail
umask 022
echo "===C211-I9454-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) host=$(hostname 2>/dev/null || echo unknown) job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
timeout 2900 python3 "$(dirname "${BASH_SOURCE[0]}")/i9454_probe.py"
rc=$?
echo "--- i9454 rc=${rc}"
echo "===C211-I9454-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
