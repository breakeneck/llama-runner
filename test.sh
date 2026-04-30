#!/usr/bin/env bash
# ── Model Benchmark Test Script ──────────────────────────────────────────
# Tests all discovered models with multiple tasks and temperature configs.
# Results are saved to results.json for resumability.
#
# Usage:
#   ./test.sh                              # Run all temperature configs
#   ./test.sh --temperature 0.2,0.6        # Run only specific temperatures
#   ./test.sh --task 3                     # Run only specific task(s)
# ─────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run the benchmark
cd "$SCRIPT_DIR"
python run_test.py "$@"
