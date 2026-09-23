#!/usr/bin/env zsh
# ABOUTME: Renders one-pager.html to michelle-pellon-one-pager.pdf with headless Chrome.
# ABOUTME: Fails if a shared figure is missing from index.html or the one-pager, or the PDF is not one page.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: scripts/build-one-pager.sh

Builds michelle-pellon-one-pager.pdf at the repo root from one-pager.html.
Before rendering, checks that every figure in FACTS appears in both
index.html and one-pager.html, so the two pages cannot drift apart.
After rendering, checks the PDF is exactly one page.

Env: CHROME  path to a Chrome binary (default: /Applications/Google Chrome.app)
USAGE
}
[[ "${1:-}" == (-h|--help) ]] && { usage; exit 0 }

root=${0:A:h:h}
src=$root/one-pager.html
out=$root/michelle-pellon-one-pager.pdf
chrome=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}

# Figures stated on both pages. Add one here whenever a number appears on both.
FACTS=('$40K' '$10K' '15+' '200–400' '90 days' '3,000+' '150+' '$8M' 'since 2019'
       'DaVita' 'Waud' 'September 2023' '$4M' '$16M' '45 days' '2008 — 2017'
       'help desk costs 30%' '$360K' 'CentralReach'
       'Walmart Supply Chain Cybersecurity' 'Every role was a mandate')

missing=0
for fact in $FACTS; do
  for page in $root/index.html $src; do
    if ! grep -qF -- "$fact" $page; then
      print -u2 "missing: '$fact' in ${page:t}"
      missing=1
    fi
  done
done
(( missing )) && { print -u2 "Fix the copy so both pages agree, then rerun."; exit 1 }

[[ -x "$chrome" ]] || { print -u2 "Chrome not found at: $chrome (set CHROME)"; exit 1 }
(( $+commands[pdfinfo] )) || { print -u2 "pdfinfo not found: brew install poppler"; exit 1 }

# Virtual time budget lets the web fonts load before printing.
"$chrome" --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=5000 \
  --print-to-pdf="$out" "file://$src" 2>/dev/null

pages=$(pdfinfo "$out" | awk '/^Pages:/ {print $2}')
if [[ "$pages" != 1 ]]; then
  print -u2 "expected 1 page, got $pages: tighten one-pager.html"
  exit 1
fi
print "built ${out:t} (1 page, $(( $(stat -f%z "$out") / 1024 )) KB)"
