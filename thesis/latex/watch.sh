#!/usr/bin/env bash
# Poll-based alternative to `latexmk -pvc`, which hangs in this environment
# (its file-watcher needs libfilesys-notify-simple-perl, not installed here).
# Recompiles tcc.pdf whenever tcc.tex, tcc.bib, or the vendored class/style
# files change. No extra dependencies -- just mtime polling.
cd "$(dirname "$0")"
last=""
echo "Watching for changes (Ctrl-C to stop)..."
while true; do
  current=$(stat -c '%Y' tcc.tex tcc.bib infufrgs.cls abntex2cite.sty 2>/dev/null | md5sum)
  if [ "$current" != "$last" ]; then
    echo "[$(date +%H:%M:%S)] change detected, recompiling..."
    latexmk -pdf -interaction=nonstopmode tcc.tex > /tmp/watch_compile.log 2>&1
    if [ $? -eq 0 ]; then
      echo "[$(date +%H:%M:%S)] OK -- tcc.pdf updated"
    else
      echo "[$(date +%H:%M:%S)] FAILED -- see /tmp/watch_compile.log"
    fi
    last="$current"
  fi
  sleep 2
done
