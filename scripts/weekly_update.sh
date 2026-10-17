#!/bin/bash
# Refresh the latest season from StatsLook (all competitions and league-only)
# and rebuild it. Run by hand or on a schedule (see scripts/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/update_$(date +%Y-%m-%d).log"
{
  echo "== $(date) =="
  "$PYTHON" -m src.data_sources.statslook_fetch --build
  "$PYTHON" -m src.data_sources.statslook_fetch --league-only --build
  # Other seasons take the latest season's clubs as their current club.
  "$PYTHON" -m src.pipeline
  echo "== done $(date) =="
} >>"$LOG" 2>&1
