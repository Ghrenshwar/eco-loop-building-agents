"""Pretty-print the supervisor decision log for the demo video (Shot 4).

Shows, for every supervisory cycle: which MCP tools the LLM called, the setpoints
it chose, its one-line rationale, any safety clamps applied, and whether the
cycle was a self-correction. This is the "money shot" — proof the LLM is driving
control through real tools, with every setpoint validated and clamped.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts\\show_decisions.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else "ai"
path = ROOT / "outputs" / run / "decisions.jsonl"

B = "\033[1m"; G = "\033[32m"; C = "\033[36m"; Y = "\033[33m"; M = "\033[35m"; R = "\033[0m"
import os
os.system("")  # enable ANSI colours on Windows

if not path.exists():
    sys.exit(f"No decisions at {path} - run:  python -m ecoloop.pipeline.run_ai --smoke")

rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"\n{B}EcoLoop - Supervisor decision log  ({len(rows)} LLM decisions, run: {run}){R}\n")

for x in rows:
    tag = f"{M}[SELF-CORRECTION]{R}" if x.get("self_correction") else ""
    fb = f"{Y}[FALLBACK]{R}" if x.get("fallback") else ""
    print(f"{B}Decision #{x['version']}  (sim hour {x['hour']:02d}){R}  "
          f"latency {x.get('latency_s','?')}s  {tag}{fb}")
    print(f"  {C}MCP tools called:{R} {', '.join(x['tool_calls']) or '(none)'}")
    sps = x.get("setpoints", {})
    if sps:
        shown = "  ".join(f"{z}: {v['heating_sp']:.0f}/{v['cooling_sp']:.0f}C"
                          for z, v in list(sps.items())[:5])
        print(f"  {C}Setpoints (htg/clg):{R} {shown}")
    ecm = x.get("ecm", {})
    on = [k for k, v in ecm.items() if v]
    if on:
        print(f"  {C}ECM flags on:{R} {', '.join(on)}")
    rationale = (x.get("rationale") or "").strip()
    if rationale and rationale.lower() != "no rationale given":
        print(f"  {G}Rationale:{R} {rationale}")
    else:
        # LLM applied its decision through tool calls without restating a reason;
        # describe the action it actually took (factual, from the log).
        acts = []
        if "set_zone_setpoints" in x["tool_calls"] and sps:
            v = next(iter(sps.values()))
            acts.append(f"set zone setpoints to {v['heating_sp']:.0f}/{v['cooling_sp']:.0f}C")
        if on:
            acts.append("applied " + ", ".join(on))
        acts = acts or ["read building state via MCP tools"]
        print(f"  {G}Action (via MCP tool calls):{R} " + "; ".join(acts))
    if x.get("repairs"):
        print(f"  {Y}Safety clamps applied:{R} {x['repairs'][0]}")
    print()

nfb = sum(r.get("fallback") for r in rows)
nsc = sum(r.get("self_correction") for r in rows)
tools = sorted({t for r in rows for t in r["tool_calls"]})
print(f"{B}{G}Summary:{R} {len(rows)} decisions - {nsc} self-corrections - "
      f"{nfb} safety fallbacks - tools used: {', '.join(tools)}\n")
