#!/usr/bin/env python3
# PRL-T1-HYGONK8S-64228
# Transient read-only Kubernetes RBAC / kubelet matrix probe.
# Executed once from a HYGON CI runner pod (uid=0). Metadata only:
# the service-account token value is NEVER printed, only its JWT claims.
# Read-only: 2 apiserver GETs + 2 SelfSubjectAccessReview POSTs + 2 anonymous
# kubelet GETs. No kubelet exec, no pod writes, no secret reads, no cloud calls.
import base64
import json
import os
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request

TAG = "PRL-T1-HYGONK8S-64228"
SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
KUBELET_HOST = "10.17.1.4"
TIMEOUT = 8


def p(*args):
    try:
        print(*args, flush=True)
    except Exception:
        pass


def http(url, headers=None, ctx=None, data=None, method=None, timeout=TIMEOUT, maxb=4096):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.status, resp.read(maxb)
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read(1024)
        except Exception:
            return e.code, b""
    except Exception as e:
        return 0, (type(e).__name__ + ": " + str(e)[:200]).encode()


def b64json(seg):
    return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))


def main():
    p(TAG + "-BEGIN")
    p("hostname=" + socket.gethostname())
    p("uid=" + (str(os.getuid()) if hasattr(os, "getuid") else "na"))
    ksh = os.environ.get("KUBERNETES_SERVICE_HOST", "")
    ksp = os.environ.get("KUBERNETES_SERVICE_PORT", "")
    p("KUBERNETES_SERVICE_HOST=" + (ksh or "unset"))
    p("KUBERNETES_SERVICE_PORT=" + (ksp or "unset"))
    p("POD_NAMESPACE_env=" + os.environ.get("POD_NAMESPACE", "unset"))
    p("RUNNER_NAME=" + os.environ.get("RUNNER_NAME", "unset"))
    p("GITHUB_JOB=" + os.environ.get("GITHUB_JOB", "unset"))
    try:
        p("SA_DIR_LIST=" + ",".join(sorted(os.listdir(SA_DIR))))
    except Exception as e:
        p("SA_DIR_LIST_ERR=" + type(e).__name__)

    tok = ""
    present = os.path.isfile(os.path.join(SA_DIR, "token"))
    p("SA_TOKEN_PRESENT=" + ("yes" if present else "no"))
    ns = ""
    try:
        ns = open(os.path.join(SA_DIR, "namespace")).read().strip()
        p("SA_NAMESPACE=" + (ns or "empty"))
    except Exception as e:
        p("SA_NAMESPACE_ERR=" + type(e).__name__)
    if present:
        try:
            tok = open(os.path.join(SA_DIR, "token")).read().strip()
            p("SA_TOKEN_LEN=" + str(len(tok)))
            parts = tok.split(".")
            if len(parts) == 3:
                hdr = b64json(parts[0])
                cl = b64json(parts[1])
                p("SA_JWT_HDR=" + json.dumps({k: hdr.get(k) for k in ("alg", "kid")}, sort_keys=True))
                p("SA_JWT_CLAIMS=" + json.dumps(
                    {k: cl.get(k) for k in ("iss", "sub", "aud", "namespace", "exp", "iat")},
                    sort_keys=True, default=str))
        except Exception as e:
            p("SA_JWT_DECODE_ERR=" + type(e).__name__ + ": " + str(e)[:120])

    if not ksh:
        p("APISERVER_SKIP=no KUBERNETES_SERVICE_HOST")
    else:
        ca = os.path.join(SA_DIR, "ca.crt")
        try:
            if os.path.isfile(ca):
                ctx = ssl.create_default_context(cafile=ca)
                p("APISERVER_CA=present")
            else:
                ctx = ssl._create_unverified_context()
                p("APISERVER_CA=absent")
        except Exception as e:
            ctx = ssl._create_unverified_context()
            p("APISERVER_CA_ERR=" + type(e).__name__)
        base = "https://" + ksh + (":" + ksp if ksp else "")
        hdrs = {"Accept": "application/json"}
        if tok:
            hdrs["Authorization"] = "Bearer " + tok

        # READ 1: nodes (list verdict + GPU node inventory)
        st, body = http(base + "/api/v1/nodes?limit=50", headers=hdrs, ctx=ctx, maxb=8_000_000)
        p("APISERVER_NODES_STATUS=" + str(st))
        if st == 200:
            try:
                items = json.loads(body).get("items", []) or []
                p("APISERVER_NODES_COUNT=" + str(len(items)))
                for n in items[:50]:
                    md = n.get("metadata", {}) or {}
                    labels = md.get("labels", {}) or {}
                    keep = {k: v for k, v in labels.items()
                            if re.search(r"gpu|nvidia|hygon|hcu|accel|dcgm|device|instance-type", k, re.I)}
                    p("NODE " + str(md.get("name", "?")) + " LABELS " + json.dumps(keep, sort_keys=True))
            except Exception as e:
                p("APISERVER_NODES_PARSE_ERR=" + type(e).__name__)
        else:
            p("APISERVER_NODES_BODY=" + body[:200].decode("utf-8", "replace").replace("\n", " "))

        # READ 2: pods in own namespace (count only)
        if ns:
            st, body = http(base + "/api/v1/namespaces/" + ns + "/pods?limit=20", headers=hdrs, ctx=ctx, maxb=8_000_000)
            p("APISERVER_PODS_STATUS=" + str(st))
            if st == 200:
                try:
                    items = json.loads(body).get("items", []) or []
                    p("APISERVER_PODS_COUNT=" + str(len(items)))
                except Exception as e:
                    p("APISERVER_PODS_PARSE_ERR=" + type(e).__name__)
            else:
                p("APISERVER_PODS_BODY=" + body[:200].decode("utf-8", "replace").replace("\n", " "))

        # READ 3+4: SelfSubjectAccessReview get on nodes and pods
        # (list verdicts are implied by the two GETs above)
        if tok:
            for res in ("nodes", "pods"):
                ssar = json.dumps({"apiVersion": "authorization.k8s.io/v1",
                                   "kind": "SelfSubjectAccessReview",
                                   "spec": {"resourceAttributes": {"verb": "get", "resource": res}}}).encode()
                h2 = dict(hdrs)
                h2["Content-Type"] = "application/json"
                st, body = http(base + "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
                                headers=h2, ctx=ctx, data=ssar, method="POST")
                allowed = None
                reason = ""
                try:
                    j = json.loads(body)
                    allowed = j.get("status", {}).get("allowed")
                    reason = j.get("status", {}).get("reason", "") or ""
                except Exception:
                    pass
                p("SSAR get/" + res + " status=" + str(st) + " allowed=" + str(allowed)
                  + ((" reason=" + reason[:80]) if reason else ""))
        else:
            p("SSAR_SKIP=no token")

    # READ 5+6: anonymous kubelet TLS
    kctx = ssl._create_unverified_context()
    for path in ("/version", "/pods?limit=50"):
        st, body = http("https://" + KUBELET_HOST + ":10250" + path,
                        headers={"Accept": "application/json"}, ctx=kctx, timeout=8)
        txt = body[:200].decode("utf-8", "replace").replace("\n", " ")
        p("KUBELET " + path + " status=" + str(st) + " body=" + txt)

    p(TAG + "-END")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        p("PRL_PROBE_FATAL=" + type(e).__name__ + ": " + str(e)[:200])
    sys.exit(0)
