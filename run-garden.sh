#!/bin/zsh
# Nightly trellis run: refresh the embedding index, triage new notes, then garden.
# Invoked by com.trellis.garden launchd agent. Ollama must be running.

# Resolve this script's own directory so there are no hardcoded user paths.
DIR=${0:A:h}
PY="$DIR/.venv/bin/python3"
LOG="$DIR/garden.log"

echo "===== run $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
"$PY" "$DIR/trellis.py" index  >> "$LOG" 2>&1
"$PY" "$DIR/trellis.py" triage >> "$LOG" 2>&1
"$PY" "$DIR/trellis.py" garden >> "$LOG" 2>&1
echo "===== done $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
