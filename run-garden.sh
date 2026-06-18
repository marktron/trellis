#!/bin/zsh
# Nightly trellis run: refresh the embedding index, then garden.
# Invoked by com.trellis.garden launchd agent. Ollama must be running.

PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
DIR=/Users/mark/Developer/trellis
LOG="$DIR/garden.log"

echo "===== run $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
"$PY" "$DIR/trellis.py" index  >> "$LOG" 2>&1
"$PY" "$DIR/trellis.py" garden >> "$LOG" 2>&1
echo "===== done $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
