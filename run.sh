#!/usr/bin/env bash
set -e

git fetch origin &&
git reset --hard origin/main &&
git clean -fd &&
git log -1 --pretty=format:"%h %s"


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Run the app
cd "$SCRIPT_DIR"
python main.py "$@"
