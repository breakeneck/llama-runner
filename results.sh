#!/usr/bin/env bash
# ── Results Display Script ───────────────────────────────────────────────
# Reads results.json and displays formatted benchmark tables.
#
# Usage:
#   ./results.sh
# ─────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$SCRIPT_DIR/results.json" ]; then
    echo "❌ No results.json found. Run ./test.sh first."
    exit 1
fi

# Activate venv (silently skip if not available)
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "⚠️  Virtual environment not found. Installing dependencies..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$SCRIPT_DIR"
python show_results.py
