"""Replay recorded telemetry as a live-looking stream — for the demo video (Shot 3).

Reads a run's telemetry (default: the AI run) and prints each simulated
timestep's sensor readings — zone air temperature, relative humidity,
occupancy, the facility power meter — together with the PMV/PPD comfort values
computed in Python, one timestep at a time with a small delay so it scrolls
like live telemetry streaming out of EnergyPlus.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts\\stream_telemetry.py            # AI run
    .venv\\Scripts\\python.exe scripts\\stream_telemetry.py baseline   # baseline run
    .venv\\Scripts\\python.exe scripts\\stream_telemetry.py ai 0.05    # faster
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

run = sys.argv[1] if len(sys.argv) > 1 else "ai"
delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.18

csv = ROOT / "outputs" / run / "telemetry.csv"
if not csv.exists():
    sys.exit(f"No telemetry at {csv} — run the pipeline first "
             f"(python -m ecoloop.pipeline.run_{run if run=='baseline' else 'ai'} --smoke)")

df = pd.read_csv(csv)

BOLD = "\033[1m"; GREEN = "\033[32m"; CYAN = "\033[36m"; YELLOW = "\033[33m"; RESET = "\033[0m"
try:
    import os
    os.system("")  # enable ANSI colours on Windows terminals
except Exception:
    pass

print(f"\n{BOLD}EcoLoop - live telemetry streaming out of EnergyPlus  (run: {run}){RESET}")
print(f"{'sim_hr':>6} {'zone':<9} {'Tair_C':>7} {'RH_%':>6} {'occ':>4} "
      f"{'kW_facility':>12} {'PMV':>7} {'PPD_%':>6}")
print("-" * 62)

last_j = {}
for _, r in df.iterrows():
    # facility meter is per-timestep energy (J); show it as an instantaneous-ish kW
    kw = r["facility_elec_j"] / 1000.0 / (15 * 60) if r["facility_elec_j"] else 0.0
    pmv = r["pmv"]
    band = GREEN if -0.5 <= pmv <= 0.5 else YELLOW
    print(f"{r['sim_time_s']/3600:>6.2f} {CYAN}{r['zone']:<9}{RESET} "
          f"{r['air_temp_c']:>7.2f} {r['rel_humidity_pct']:>6.1f} {r['occupancy']:>4.0f} "
          f"{r['facility_elec_j']/3.6e6:>12.4f} {band}{pmv:>7.3f}{RESET} {r['ppd']:>6.1f}")
    time.sleep(delay)

print(f"\n{BOLD}{GREEN}Telemetry stream complete - {len(df)} readings across "
      f"{df['zone'].nunique()} zones.{RESET}\n")
