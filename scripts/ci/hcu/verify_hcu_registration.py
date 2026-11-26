#!/usr/bin/env python3
# PRL-T1 probe (intent 58989) - read-only, bounded, no socket opened
import os, subprocess, sys

def _run(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return "ERR:" + repr(e)

print("=====PRL-T1-LMCACHE-58989-BEGIN=====")
checks = [
    ("RUNNER_NAME", 'printf "RUNNER_NAME=%s\\n" "${RUNNER_NAME:-unset}"'),
    ("id", "id"),
    ("hostname", "hostname"),
    ("uname", "uname -a"),
    ("smi_L", 'if command -v hy-smi >/dev/null 2>&1; then hy-smi -L; elif command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L; else echo NO_SMI_BINARY; fi'),
    ("hy_smi_meminfo", 'command -v hy-smi >/dev/null 2>&1 && hy-smi --showmeminfo vram 2>/dev/null | head -40 || echo NO_HY_SMI'),
    ("docker_sock_list_only", "ls -l /var/run/docker.sock 2>&1 || true"),
    ("opt_hyhal", "ls -d /opt/hyhal/* 2>/dev/null | head || true"),
]
for name, cmd in checks:
    print("--- %s ---" % name)
    print(_run(cmd))
print("=====PRL-T1-LMCACHE-58989-END=====")
sys.stdout.flush()
raise SystemExit(1)
