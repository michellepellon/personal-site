#!/usr/bin/env zsh
# ABOUTME: Renders og-card.html to img/michelle-pellon-card.png with headless Chrome.
# ABOUTME: Fails if the image is not exactly 1200x630, the size social platforms expect.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: scripts/build-og-card.sh

Builds img/michelle-pellon-card.png from og-card.html. Rerun it whenever
the name, positioning line, title or employer on the card changes.

Env: CHROME  path to a Chrome binary (default: /Applications/Google Chrome.app)
USAGE
}
[[ "${1:-}" == (-h|--help) ]] && { usage; exit 0 }

root=${0:A:h:h}
src=$root/og-card.html
out=$root/img/michelle-pellon-card.png
chrome=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}

[[ -x "$chrome" ]] || { print -u2 "Chrome not found at: $chrome (set CHROME)"; exit 1 }

# Virtual time budget lets the web fonts load before the capture.
"$chrome" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
  --window-size=1200,630 --virtual-time-budget=5000 \
  --screenshot="$out" "file://$src" 2>/dev/null

size="$(sips -g pixelWidth -g pixelHeight "$out" | awk '/pixel/ {print $2}' | paste -sd x -)"
if [[ "$size" != 1200x630 ]]; then
  print -u2 "expected 1200x630, got $size"
  exit 1
fi
print "built img/${out:t} ($size, $(( $(stat -f%z "$out") / 1024 )) KB)"
