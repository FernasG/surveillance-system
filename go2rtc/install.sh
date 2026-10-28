#!/bin/bash
# Installs go2rtc on the Raspberry Pi host as a systemd service.
# Usage: ./go2rtc/install.sh [--dev]
set -euo pipefail

GO2RTC_VERSION="v1.9.14"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

configs="-c $REPO_DIR/go2rtc/go2rtc.yaml"
if [ "${1:-}" = "--dev" ]; then
    configs="$configs -c $REPO_DIR/go2rtc/go2rtc.dev.yaml"
fi

sudo curl -fsSL -o /usr/local/bin/go2rtc \
    "https://github.com/AlexxIT/go2rtc/releases/download/$GO2RTC_VERSION/go2rtc_linux_arm64"
sudo chmod +x /usr/local/bin/go2rtc

sudo tee /etc/systemd/system/go2rtc.service > /dev/null <<EOF
[Unit]
Description=go2rtc camera restreamer
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
EnvironmentFile=$REPO_DIR/.env
ExecStart=/usr/local/bin/go2rtc $configs
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable go2rtc
sudo systemctl restart go2rtc
