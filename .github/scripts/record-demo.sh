#!/usr/bin/env bash
# Records a camas demo into an asciicast and renders it to GIF.
# Outputs .demo/demo.cast and .demo/demo.gif.

set -euo pipefail

OUT_DIR="${OUT_DIR:-.demo}"
mkdir -p "$OUT_DIR"

unset VIRTUAL_ENV

DEMO_CMD="uv run python tests/test_demos.py test-demo-matrix-full-ci-pipeline"

echo "::group::Record asciicast"
asciinema rec \
	--window-size 100x30 \
	--idle-time-limit 2 \
	--overwrite \
	-c "$DEMO_CMD" \
	"$OUT_DIR/demo.cast"
echo "::endgroup::"

echo "::group::Render GIF"
# Custom theme: monokai palette with pure-black background.
# Format: bg,fg,c0..c15 (RGB hex, comma-separated, no # prefix).
THEME="000000,f8f8f2,272822,f92672,a6e22e,f4bf75,66d9ef,ae81ff,a1efe4,f8f8f2,75715e,f92672,a6e22e,f4bf75,66d9ef,ae81ff,a1efe4,f9f8f5"
agg \
	--theme "$THEME" \
	--font-size 16 \
	--line-height 1.0 \
	--speed 1.2 \
	"$OUT_DIR/demo.cast" "$OUT_DIR/demo.gif"
echo "::endgroup::"

if command -v gifsicle >/dev/null 2>&1; then
	echo "::group::Shrink GIF"
	gifsicle --lossy=80 -k 128 -O2 -Okeep-empty \
		"$OUT_DIR/demo.gif" -o "$OUT_DIR/demo.gif"
	echo "::endgroup::"
else
	echo "gifsicle not found — skipping shrink step"
fi
ls -lh "$OUT_DIR/demo.gif"
