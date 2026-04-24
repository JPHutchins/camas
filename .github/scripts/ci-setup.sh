#!/usr/bin/env bash
# Installs asciinema, agg, and gifsicle for the demo-GIF workflow.
# Prebuilt binaries where possible — no `apt-get update`.

set -euo pipefail

ASCIINEMA_VERSION="3.2.0"
AGG_VERSION="1.7.0"

BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN_DIR"

echo "::group::Install asciinema ${ASCIINEMA_VERSION}"
curl -fsSL --retry 3 \
	-o "$BIN_DIR/asciinema" \
	"https://github.com/asciinema/asciinema/releases/download/v${ASCIINEMA_VERSION}/asciinema-x86_64-unknown-linux-gnu"
chmod +x "$BIN_DIR/asciinema"
"$BIN_DIR/asciinema" --version
echo "::endgroup::"

echo "::group::Install agg ${AGG_VERSION}"
curl -fsSL --retry 3 \
	-o "$BIN_DIR/agg" \
	"https://github.com/asciinema/agg/releases/download/v${AGG_VERSION}/agg-x86_64-unknown-linux-gnu"
chmod +x "$BIN_DIR/agg"
"$BIN_DIR/agg" --version
echo "::endgroup::"

echo "::group::Install gifsicle"
sudo apt-get install -y --no-install-recommends gifsicle
gifsicle --version | head -1
echo "::endgroup::"

if [[ -n "${GITHUB_PATH:-}" ]]; then
	echo "$BIN_DIR" >>"$GITHUB_PATH"
fi

echo "Installed to $BIN_DIR"
