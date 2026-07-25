#!/usr/bin/env bash
# EcoLoop setup (Linux/macOS/Git-Bash).
#  - verifies EnergyPlus is installed and discoverable
#  - adds the E+ dir to PYTHONPATH so `import pyenergyplus` works
#  - pulls the default Ollama model
#  - creates a venv and pip-installs the pinned deps
#
# Usage: bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "== EcoLoop setup =="
echo "repo: $ROOT"

# --- 1. EnergyPlus -----------------------------------------------------------
EPLUS_DIR="${ENERGYPLUS_DIR:-}"
if [[ -z "$EPLUS_DIR" ]]; then
  for c in /usr/local/EnergyPlus-24-2-0 /usr/local/EnergyPlus-24-1-0 \
           /Applications/EnergyPlus-24-2-0 "C:/EnergyPlusV24-2-0" "C:/EnergyPlusV24-1-0"; do
    [[ -d "$c" ]] && EPLUS_DIR="$c" && break
  done
fi
if [[ -z "$EPLUS_DIR" || ! -d "$EPLUS_DIR" ]]; then
  echo "!! EnergyPlus not found. Install v24.x from https://energyplus.net/downloads"
  echo "   then re-run:  ENERGYPLUS_DIR=/path/to/EnergyPlus bash scripts/setup.sh"
else
  echo "-> EnergyPlus: $EPLUS_DIR"
  if [[ -d "$EPLUS_DIR/pyenergyplus" ]]; then
    export PYTHONPATH="$EPLUS_DIR:${PYTHONPATH:-}"
    echo "-> added to PYTHONPATH (pyenergyplus importable this shell)"
    echo "   (persist it: echo 'export PYTHONPATH=\"$EPLUS_DIR:\$PYTHONPATH\"' >> ~/.bashrc)"
  else
    echo "!! $EPLUS_DIR/pyenergyplus missing — is the install complete?"
  fi
  "$EPLUS_DIR/energyplus" --version 2>/dev/null || true
fi

# --- 2. Python venv + deps ---------------------------------------------------
echo "== Python venv =="
python -m venv .venv || python3 -m venv .venv
# shellcheck disable=SC1091
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
echo "-> deps installed into .venv"

# --- 3. Ollama model ---------------------------------------------------------
echo "== Ollama =="
if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen2.5:3b-instruct || echo "!! model pull failed (is 'ollama serve' running?)"
else
  echo "!! ollama not found. Install from https://ollama.com/download, then:"
  echo "   ollama pull qwen2.5:3b-instruct"
fi

echo "== Done. Next: bash scripts/run_all.sh --smoke =="
