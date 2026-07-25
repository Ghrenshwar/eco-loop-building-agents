# EcoLoop Building Agents

An **autonomous closed-loop building-energy control system**. EnergyPlus
simulates a real multi-zone office (the *digital twin*); a locally-hosted
open-source LLM acts as a **supervisory HVAC controller** that ingests live
telemetry streamed out of the running simulation, reasons about
comfort / energy / carbon, and injects updated setpoints back into the *same
running simulation* — over a real **MCP** tool server, with no human in the loop.
It then proves, quantitatively, that the AI loop uses **less energy** than a
fixed rule-based baseline **while keeping occupants comfortable** (PMV in band).

> Digital twin: **EnergyPlus** (developed & verified on **v26.1.0**; adapts to
> the installed API — works on 24.x–26.x), building **`5ZoneAirCooled.idf`**
> (5 conditioned zones), weather **`USA_CO_Golden-NREL.724666_TMY3.epw`**
> (Golden, Colorado — strong heating *and* cooling load). `setup` confirms the
> exact installed E+ version and copies these from the install.
>
> **Verified result (design-day smoke run, qwen2.5:3b-instruct on CPU):**
> AI **−3.2% kWh** vs the fixed baseline (167.0 vs 172.5 kWh) with comfort
> **improved** (+3.6 pp occupied-hours in band, mean PMV −0.19 vs −0.35) — 11
> LLM decisions, 0 fallbacks, 7 self-corrections, all MCP tools exercised.
> Verdict: **PASS**.

---

## Prerequisites

| Dependency | Why | Install |
|---|---|---|
| **Python 3.11+** | everything | python.org |
| **EnergyPlus 24.x** | the simulation + `pyenergyplus` API (not on PyPI — ships in the install) | <https://energyplus.net/downloads> |
| **Ollama** + a tool-calling model | the local LLM controller | <https://ollama.com/download> |

Model: the **CPU default is `qwen2.5:3b-instruct`** (~20 s/call on CPU); on a
GPU switch to `qwen2.5:7b-instruct` for stronger tool-calling (set `llm.model`
in `config/config.yaml`). Pull it with `ollama pull qwen2.5:3b-instruct`.
Because EnergyPlus runs far faster than real time, the AI run paces the sim to
wall-clock (`supervisor.realtime_pace_s_per_step`) so the async supervisor lands
~hourly decisions; the baseline runs at full speed. This never blocks on the LLM.

## One-command setup

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```
**Linux / macOS / Git-Bash:**
```bash
bash scripts/setup.sh
```
Setup verifies EnergyPlus, creates `.venv`, installs pinned deps + the editable
package, and pulls the Ollama model. If E+ is in a non-standard location, set
`ENERGYPLUS_DIR` (or edit `energyplus.install_dir` in `config/config.yaml`).

## Run it end-to-end

Start with a fast **smoke** run (single design day) to iterate, then the full
period (a representative week by default — edit `run_period` in
`config/config.yaml`, e.g. `end_day: 31` for a month):

**Windows:**
```powershell
.\scripts\run_all.ps1 -Smoke      # baseline -> AI -> compare -> dashboard
```
**Linux / macOS:**
```bash
bash scripts/run_all.sh --smoke
```

Or step by step:
```bash
python -m ecoloop.pipeline.run_baseline --smoke   # fixed-schedule control
python -m ecoloop.pipeline.run_ai --smoke         # LLM closed-loop over MCP
python -m ecoloop.pipeline.compare                # writes outputs/summary.json
streamlit run dashboard/app.py                    # baseline-vs-AI dashboard
```

## What to expect

- `outputs/baseline/` and `outputs/ai/` — per-timestep telemetry
  (`telemetry.parquet` / `.csv`), the agent's `decisions.jsonl`, and each run's
  `eplusout.sql`.
- `outputs/summary.json` — headline **% kWh reduction**, cost & carbon savings,
  PMV distribution, **% of occupied hours in band**, unmet hours, and a
  `PASS/REVIEW` verdict. It passes only when the AI **saves energy AND holds
  comfort** (≥ 95 % of occupied zone-hours inside PMV ∈ [−0.5, +0.5]).
- `models/generated/ai_step_*.idf` — AI-modified `.idf` snapshots written at
  runtime.
- The **dashboard** — % saved, cumulative baseline-vs-AI energy curve, per-zone
  temperature traces, PMV band overlay, unmet-hours comparison, cost/carbon, and
  the live **decision log** proving the LLM actually called the MCP tools and
  self-corrected. Export `comparison.csv` from the dashboard.

## Inspect the MCP server directly

The control surface is a real, inspectable MCP server:
```bash
python -m ecoloop.mcp.server          # serves http://127.0.0.1:8765/mcp
```
Point any MCP client at it to `list_tools` / `call_tool`
(`get_telemetry_summary`, `set_zone_setpoints`, `parse_simulation_log`, …).

## Tests

```bash
pytest            # bus, policy-guard (robustness), PMV, MCP tools
```

## Repository layout

```
config/            config.yaml + targets.yaml (comfort band, carbon/tariff curves)
models/            baseline.idf, weather.epw, generated/ (runtime AI snapshots)
src/ecoloop/
  config.py        pydantic-validated settings
  energyplus/      runner (callback loop), handles, actuators, idf_prep (eppy)
  bus/             TelemetryBuffer + ControlPolicy (thread-safe)
  comfort/pmv.py   PMV/PPD via pythermalcomfort (+ ISO 7730 fallback)
  mcp/             FastMCP server + tools (the graded MCP layer)
  agent/           supervisor, llm_client (Ollama), mcp_client, prompts, policy_guard
  logging/         recorder (Parquet/CSV/JSONL)
  pipeline/        run_baseline, run_ai, compare
dashboard/app.py   Streamlit baseline-vs-AI dashboard
docs/              ARCHITECTURE.md, DEMO_SCRIPT.md
scripts/           setup.(sh|ps1), run_all.(sh|ps1)
tests/             pytest suite
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the threading model,
tool-calling architecture, prompt-engineering & latency strategy, long-log
handling, and the self-correction design.

## How it works (one paragraph)

One process, three threads over a thread-safe bus. The **main thread** runs
EnergyPlus; a per-timestep callback reads sensors, computes PMV, buffers
telemetry, and writes the current policy to `Schedule:Constant` actuators — it
never calls the LLM and never lets an exception kill the sim. The **supervisor
thread** wakes ~hourly, summarizes telemetry, and runs an Ollama tool-calling
agent that reads/writes the building through the **MCP server thread**; every
proposed setpoint is validated and clamped (`policy_guard`) before it reaches
E+, and the agent self-corrects when comfort drifts. Baseline and AI runs use an
identical building/weather/period, so `compare.py`'s savings claim is valid.
```
