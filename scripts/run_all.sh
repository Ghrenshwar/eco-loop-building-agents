#!/usr/bin/env bash
# EcoLoop end-to-end: baseline -> AI -> compare -> dashboard.
# Usage:  bash scripts/run_all.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi

SMOKE="${1:-}"
echo "== 1/4 baseline =="
python -m ecoloop.pipeline.run_baseline $SMOKE
echo "== 2/4 AI closed-loop =="
python -m ecoloop.pipeline.run_ai $SMOKE
echo "== 3/4 compare =="
python -m ecoloop.pipeline.compare
echo "== 4/4 dashboard (Ctrl-C to stop) =="
streamlit run dashboard/app.py
