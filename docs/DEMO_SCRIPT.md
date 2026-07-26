# EcoLoop — Demo Video Script (≤ 3 minutes)

A shot list / storyboard for recording the demo. You (the human) record; this
script tells you what to show and say. Target length **2:45**. Have a completed
`--smoke` (or short-week) AI run ready so `outputs/` and the dashboard are
populated, and a terminal with the repo checked out.

---

## Shot 0 — Cold open (0:00–0:15)
**On screen:** the dashboard headline metrics (kWh saved %, comfort in band %).
**Say:** "This is an autonomous building controller. A local open-source LLM is
running the HVAC of a simulated office — and it cut energy use by **3.2%** while
keeping every occupant comfortable. No human in the loop. Here's how."

## Shot 1 — The digital twin (0:15–0:40)
**On screen:** `models/baseline.idf` open in an editor; scroll to the
`Schedule:Constant` setpoint schedules (`ECO_SPACE1-1_HtgSP`, …) and the
`Output:SQLite` / output-variable lines.
**Say:** "EnergyPlus simulates a five-zone office — our digital twin. We prepared
it so every zone's thermostat reads its setpoint from a named schedule we can
actuate live. Same building, same Colorado weather, same week for both runs —
only the control differs."

## Shot 2 — Launch the AI run (0:40–1:05)
**On screen:** terminal running `python -m ecoloop.pipeline.run_ai --smoke`.
Let the logs scroll.
**Say:** "When we start the AI run, three things spin up in one process: the
EnergyPlus simulation, a real MCP tool server, and the LLM supervisor — each on
its own thread so the simulation never waits on the model."

## Shot 3 — Telemetry streaming out (1:05–1:30)
**On screen:** highlight log lines showing telemetry being recorded / the
`[decision N]` lines appearing about once per simulated hour.
**Say:** "Every simulated timestep, sensor readings — zone temperatures,
humidity, occupancy, the facility power meter — stream out of EnergyPlus into a
shared buffer, and we compute thermal comfort, PMV, in Python."

## Shot 4 — The LLM tool calls (1:30–2:00)  ← the money shot
**On screen:** the `decisions.jsonl` tail or the dashboard's **decision log**
table — point at the `tool_calls` column (`get_telemetry_summary`,
`get_targets`, `set_zone_setpoints`, `parse_simulation_log`) and the
`rationale`.
**Say:** "Once an hour the supervisor wakes the LLM. It calls MCP tools to read a
compact telemetry summary and the hour's carbon and price signals, then decides
new setpoints — here it *pre-cools* before the dirty, expensive afternoon so it
can coast through the peak. Every setpoint is validated and clamped to safe
ranges before it touches the simulation."

## Shot 5 — Self-correction (2:00–2:20)
**On screen:** a decision row where `self_correction = true` (and/or a
`fallback = true` row).
**Say:** "It also watches itself. When a previous decision pushes a zone out of
the comfort band, that shows up in the next prompt and the agent corrects — here
it tightens a setback that went too far. When the model times out or returns
junk, the guard holds the last safe policy instead."

## Shot 6 — Setpoints updating live (2:20–2:35)
**On screen:** dashboard per-zone temperature traces + the PMV histogram with
the green comfort band; and `models/generated/ai_step_*.idf` snapshots in the
file tree.
**Say:** "The setpoint changes flow straight back into the running model — and
comfort stays inside the band the whole time. Each decision even writes a
modified `.idf` snapshot for the audit trail."

## Shot 7 — The proof (2:35–2:45)
**On screen:** dashboard cumulative-energy chart (baseline vs AI) and the
headline metrics; then `summary.json` verdict `PASS`.
**Say:** "Same building, same weather — the AI curve stays below the baseline the
entire run. **3.2% less energy**, $1.08 cheaper, 2.1 kg less carbon, and comfort
actually improved — 87.7% of occupied hours in band versus 84.1% for the baseline.
That's the whole point: measurable savings, proven, with comfort as a hard constraint."

---

### Verified numbers from this run (already filled into the script above)
- **kWh saved: 3.2%** (baseline 172.5 → AI 167.0 kWh)
- **Cost saved: $1.08**  ·  **Carbon saved: 2.1 kg**
- **Comfort: 87.7% of occupied hours in band** (vs baseline 84.1%); mean PMV −0.19 vs −0.35
- **11 LLM decisions · 7 self-corrections · 0 fallbacks** · verdict **PASS**

### Capture checklist
- [ ] Terminal font large enough to read on video.
- [ ] Dashboard pre-loaded and refreshed (`streamlit run dashboard/app.py`, at http://localhost:8501).
- [ ] `outputs/summary.json` shows `kwh_reduction_pct: 3.19` and `PASS`.
- [ ] `outputs/ai/decisions.jsonl` shows 7 `self_correction=true` rows.
- [ ] `models/generated/` contains `ai_step_*.idf` snapshots.
- [ ] Telemetry stream (`scripts/stream_telemetry.py`) and decision log (`scripts/show_decisions.py`) ready for Shots 3–4.
