#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -e .

if [ ! -f .env ]; then
  cp .env.example .env
fi

printf '%s\n' \
  "" \
  "Setup complete." \
  "1. Add your Groq key to .env." \
  "2. Activate with: . .venv/bin/activate" \
  "3. Test with: python -m unittest discover -s tests -v" \
  "4. Run with: robot-task \"Put the apple in the basket\""
