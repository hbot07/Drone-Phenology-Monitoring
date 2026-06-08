#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sudo-password>" >&2
  exit 1
fi

SUDO_PASSWORD="$1"
HTTP_PROXY_URL="http://proxy62.iitd.ac.in:3128"
HTTPS_PROXY_URL="http://proxy62.iitd.ac.in:3128"

run_sudo() {
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' "$@"
}

run_sudo_bash() {
  local script="$1"
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' bash -lc "$script"
}

echo "Configuring Docker daemon proxy..."
run_sudo mkdir -p /etc/systemd/system/docker.service.d
run_sudo_bash "cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<'EOF'
[Service]
Environment=\"HTTP_PROXY=$HTTP_PROXY_URL\"
Environment=\"HTTPS_PROXY=$HTTPS_PROXY_URL\"
Environment=\"NO_PROXY=localhost,127.0.0.1,::1\"
EOF"

echo "Configuring apt proxy..."
run_sudo_bash "cat > /etc/apt/apt.conf.d/95proxy <<'EOF'
Acquire::http::Proxy \"$HTTP_PROXY_URL\";
Acquire::https::Proxy \"$HTTPS_PROXY_URL\";
EOF"

echo "Installing NVIDIA container toolkit prerequisites..."
export http_proxy="$HTTP_PROXY_URL"
export https_proxy="$HTTPS_PROXY_URL"
export HTTP_PROXY="$HTTP_PROXY_URL"
export HTTPS_PROXY="$HTTPS_PROXY_URL"

run_sudo_bash "export http_proxy='$HTTP_PROXY_URL'; export https_proxy='$HTTPS_PROXY_URL'; export HTTP_PROXY='$HTTP_PROXY_URL'; export HTTPS_PROXY='$HTTPS_PROXY_URL'; apt-get update"
run_sudo_bash "export DEBIAN_FRONTEND=noninteractive; export http_proxy='$HTTP_PROXY_URL'; export https_proxy='$HTTPS_PROXY_URL'; export HTTP_PROXY='$HTTP_PROXY_URL'; export HTTPS_PROXY='$HTTPS_PROXY_URL'; apt-get install -y curl gnupg2 ca-certificates"

echo "Adding NVIDIA container toolkit repository..."
run_sudo_bash "export http_proxy='$HTTP_PROXY_URL'; export https_proxy='$HTTPS_PROXY_URL'; export HTTP_PROXY='$HTTP_PROXY_URL'; export HTTPS_PROXY='$HTTPS_PROXY_URL'; curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
run_sudo_bash "export http_proxy='$HTTP_PROXY_URL'; export https_proxy='$HTTPS_PROXY_URL'; export HTTP_PROXY='$HTTP_PROXY_URL'; export HTTPS_PROXY='$HTTPS_PROXY_URL'; curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list"

echo "Installing NVIDIA container toolkit..."
run_sudo_bash "export http_proxy='$HTTP_PROXY_URL'; export https_proxy='$HTTPS_PROXY_URL'; export HTTP_PROXY='$HTTP_PROXY_URL'; export HTTPS_PROXY='$HTTPS_PROXY_URL'; apt-get update"
run_sudo_bash "export DEBIAN_FRONTEND=noninteractive; export http_proxy='$HTTP_PROXY_URL'; export https_proxy='$HTTPS_PROXY_URL'; export HTTP_PROXY='$HTTP_PROXY_URL'; export HTTPS_PROXY='$HTTPS_PROXY_URL'; apt-get install -y nvidia-container-toolkit"

echo "Configuring Docker runtime for NVIDIA GPUs..."
run_sudo nvidia-ctk runtime configure --runtime=docker
run_sudo systemctl daemon-reload
run_sudo systemctl restart docker

echo "GPU setup complete."