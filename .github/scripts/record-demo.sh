#!/usr/bin/env bash
# Records a `camas all` demo on the tauri-app fixture into an asciicast and
# renders it to GIF. Designed to be run after a warm-cache pass (CI does this
# automatically; locally, run `camas all` once from the fixture dir first).
#
# Prereqs on PATH: asciinema, agg, camas. Optional: gifsicle (Linux) for
# shrinking. Outputs to .demo/demo.cast and .demo/demo.gif at the repo root.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.demo}"
FIXTURE="$REPO_ROOT/tests/fixtures/tauri-app"
PAYLOAD="$REPO_ROOT/.github/scripts/demo-payload.sh"

for tool in asciinema agg camas; do
	if ! command -v "$tool" >/dev/null 2>&1; then
		echo "error: '$tool' not found on PATH" >&2
		case "$tool" in
		asciinema | agg) echo "  hint: run .github/scripts/ci-setup.sh to install" >&2 ;;
		camas) echo "  hint: run 'uv tool install .' from the repo root" >&2 ;;
		esac
		exit 1
	fi
done

mkdir -p "$OUT_DIR"

ROWS=$(cd "$FIXTURE" && camas all --dry-run --effects='(Summary(SummaryOptions(Fixed(90))),)' 2>/dev/null | wc -l)
ROWS=$((ROWS + 3))

echo "::group::Record asciicast"
asciinema rec \
	--window-size 90x${ROWS} \
	--idle-time-limit 2 \
	--overwrite \
	-c "cd '$FIXTURE' && '$PAYLOAD'" \
	"$OUT_DIR/demo.cast"
echo "::endgroup::"

echo "::group::Render GIF"
# Monokai palette with pure-black background; format: bg,fg,c0..c15 (hex).
THEME="000000,f8f8f2,272822,f92672,a6e22e,f4bf75,66d9ef,ae81ff,a1efe4,f8f8f2,75715e,f92672,a6e22e,f4bf75,66d9ef,ae81ff,a1efe4,f9f8f5"
agg \
	--theme "$THEME" \
	--font-size 16 \
	--line-height 1.2 \
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
