# EcoLoop Building Agents — System Architecture Document

An autonomous **closed-loop** building-energy control system. EnergyPlus
simulates a real multi-zone office (the *digital twin*); a locally-hosted
open-source LLM acts as a **supervisory controller** that reads live telemetry,
reasons about comfort / energy / carbon, and injects HVAC setpoints back into
the *same running simulation* — with no human in the loop. We then prove,
quantitatively, that the AI loop uses less energy than a fixed rule-based
baseline while keeping occupants comfortable.

---

## 1. Process model — three cooperating threads

A single OS process runs three threads over a thread-safe shared bus. The
decoupling is deliberate: **the simulation must never block on LLM inference.**

```
                       ┌──────────────────────────────────────────────┐
                       │                MAIN THREAD                     │
                       │      EnergyPlus (pyenergyplus) run loop         │
                       │   .idf + .epw simulated timestep-by-timestep   │
                       └───────────────┬───────────────▲────────────────┘
        every zone timestep (fast)     │ read sensors  │ write actuators
                                       ▼               │
                   ┌───────────────────────────────────────────────────┐
                   │      SHARED BUS (thread-safe singletons)           │
                   │  TelemetryBuffer (ring buffer of recent readings)  │
                   │  ControlPolicy   (current setpoints + ECM flags)   │
                   └───────▲───────────────────────────┬────────────────┘
       reads telemetry     │                           │ writes new policy
       (once per sim-hour) │                           ▼
                   ┌───────┴──────────────┐   ┌────────────────────────┐
                   │   SUPERVISOR THREAD   │   │      MCP SERVER THREAD  │
                   │  throttled loop:      │   │  FastMCP over localhost │
                   │  1 summarize telemetry│   │  exposes tools that     │
                   │  2 run LLM agent loop ├──►│  read TelemetryBuffer / │
                   │  3 validate + clamp   │◄──┤  write ControlPolicy /  │
                   │  4 update ControlPolicy│  │  parse E+ logs / snap idf│
                   └──────────┬────────────┘   └────────────────────────┘
                              │ agent loop (tool calls)
                              ▼
                   ┌────────────────────────┐
                   │  Ollama (local OSS LLM) │  ⇄ MCP tools ⇄ shared bus
                   └────────────────────────┘

        Both runs (baseline & AI) are recorded, then compared:
   Recorder → CSV/Parquet + eplusout.sql  →  compare.py  →  Streamlit dashboard
```

### Fast control loop (main thread, every zone timestep)
`energyplus/runner.py` registers a callback at a per-zone-timestep calling
point. Each step it:
1. guards on `api.exchange.api_data_fully_ready(state)`;
2. lazily fetches & caches variable/actuator/meter handles (`handles.py`);
3. reads sensors (zone air temp, RH, occupancy, `Electricity:Facility`),
   computes **PMV** via `pythermalcomfort` (`comfort/pmv.py`), and pushes a
   `TelemetryRecord` to the `TelemetryBuffer`;
4. reads the current `ControlPolicy` and writes setpoints via the built-in
   `Zone Temperature Control` actuator, keyed by zone (`actuators.py`).

> **Actuation note:** the spec suggested actuating named `Schedule:Constant`
> objects. The bundled `5ZoneAirCooled.idf` *shares* its thermostat setpoint
> objects across zones and selects among three control types via a schedule, so
> rewiring per-zone schedules is unreliable (it silently no-ops, leaving both
> runs on identical setpoints). We therefore actuate EnergyPlus's built-in
> `Zone Temperature Control` actuator (control types `Heating Setpoint` /
> `Cooling Setpoint`, key = zone name), which overrides the active setpoint
> directly and robustly. `idf_prep` still creates the named `Schedule:Constant`
> objects so the runtime `.idf` snapshots record each decision's setpoints.

The entire callback body is wrapped in `try/except`; **an exception here can
never kill the sim** — on error it holds the last applied policy and logs once.
It never calls the LLM; between supervisory decisions it simply keeps applying
the current policy.

### Supervisory reasoning loop (supervisor thread, throttled ≈ hourly)
`agent/supervisor.py` watches simulated time on the bus and wakes once per
simulated hour (or early on a comfort breach). Each cycle it summarizes the last
hour of telemetry, runs the LLM agent loop, validates/clamps the result, and
atomically updates `ControlPolicy`. If the LLM times out or returns garbage it
keeps the last known-good policy and logs the fallback.

**Why this shape:** an LLM call takes seconds; a week-long sim has tens of
thousands of timesteps. Calling the model per-timestep is impossible.
Supervisory cadence (≈ hourly) mirrors real Building Management Systems and is
the core of prompt-latency management.

---

## 2. Tool-calling architecture (MCP)

The supervisor does **not** call Python functions directly. It talks to a real
**MCP server** (`mcp/server.py`, FastMCP over localhost) as an MCP client
(`agent/mcp_client.py`). This makes the control surface genuinely inspectable —
point any MCP client at `http://127.0.0.1:8765/mcp` and list/call the tools.

Tools (`mcp/tools.py`), all operating on the shared bus:

| Tool | Purpose |
|------|---------|
| `get_current_telemetry()` | latest per-zone snapshot |
| `get_telemetry_summary(window_minutes)` | compact aggregate (keeps prompts small) |
| `get_targets()` | comfort band + this hour's carbon / tariff / peak threshold |
| `set_zone_setpoints(zone, heating_sp, cooling_sp)` | propose setpoints (through the guard) |
| `set_ecm_flags(night_setback, precool, demand_response)` | toggle conservation measures |
| `parse_simulation_log(level)` | counts + last few unique Warning/Severe/Fatal lines only |
| `snapshot_current_idf()` | write an AI-modified `.idf` to `models/generated/` |

The agent loop (`agent/llm_client.py`): system prompt + few-shot + user message
(telemetry summary + targets) → `ollama.chat(model, messages, tools=<mcp tools>)`
→ if the model emits `tool_calls`, execute each via the MCP client, append the
results as `tool` messages, and loop → terminate on a final JSON decision or a
max-iteration cap. Tool schemas are translated from MCP into the format Ollama
expects, and **the tool docstrings double as the descriptions the LLM reads.**

---

## 3. Prompt-engineering strategy

- **Strict JSON output schema** (per-zone setpoints + ECM flags + one-line
  rationale), enforced downstream by a pydantic model in `policy_guard.py`.
- **Explicit hard safety rules** in the system prompt (keep PMV in band;
  setpoint ranges; widen deadband when unoccupied; pre-cool before high-carbon /
  peak hours; relax during DR only if comfort holds).
- **1–2 few-shot exemplars** showing an occupied pre-cool decision and an
  unoccupied night-setback decision.
- **Telemetry compressed to a compact summary** — the model sees per-zone
  mean/min/max temps, mean PMV, occupancy, and window kWh, never raw rows.
- **Tool descriptions as usage instructions** — the docstrings guide correct
  tool use without bloating the prompt.
- **Low `num_predict` and low temperature** for fast, near-deterministic control.

---

## 4. Latency management

- **Decoupled threads** — the sim never waits on the LLM (its own thread).
- **Supervisory cadence throttling** — one decision per simulated hour, not per
  timestep. A design week (≈ 1000 hourly decisions) stays tractable.
- **Small models** — default `qwen2.5:7b-instruct`; strong tool-calling on
  modest hardware. `llama3.1:8b` / `mistral-nemo` are configurable alternates.
- **Short prompts from summaries**, capped output tokens (`num_predict`), and a
  hard per-call timeout (tenacity). On timeout the previous policy is kept.
- **Optional system-prompt caching** — the system prompt + few-shot are constant
  across cycles, so the model's prompt cache is reused between decisions.

---

## 5. Long-log handling

EnergyPlus writes a multi-thousand-line `eplusout.err`. The
`parse_simulation_log(level)` tool tails and filters it to **counts + the last
few unique** Warning/Severe/Fatal lines. The raw log **never** enters a prompt.
This both controls token cost and gives the agent a clean signal for
self-correction ("did my last decision cause a new Severe line?").

---

## 6. Robustness & self-correction (design)

Robustness is a first-class concern (see `agent/policy_guard.py` and its tests):

- **Callback exceptions never stop the sim** — fall back to the last safe policy.
- **Hard timeout on every LLM call** — on timeout keep the previous policy.
- **Every setpoint is validated and clamped** before it reaches E+: heating
  ∈ [18, 23] °C, cooling ∈ [22, 28] °C, and cooling − heating ≥ 2 °C deadband.
  Impossible values are repaired where possible; unusable proposals trigger a
  `fallback=True` return of the last known-good policy.
- **Clean shutdown** — the supervisor and MCP client join / close when the sim
  ends; the MCP server runs as a daemon thread torn down with the process.

**Self-correction loop:** before each new decision the supervisor compares the
*realized* PMV (and any new Severe/Fatal log lines) against the previous
decision's intent. If comfort was violated or a runtime error appeared, a
`SELF-CORRECTION CONTEXT` note is injected into the next prompt and the agent is
required to correct — e.g. tighten a too-aggressive setback. Every correction is
flagged in `decisions.jsonl` and surfaced on the dashboard.

---

## 7. Measurement — why the savings claim is valid

`pipeline/run_baseline.py` and `pipeline/run_ai.py` run the **identical**
building, weather, and run period; only the control differs (fixed schedule vs
AI). `pipeline/compare.py` loads both runs and computes total site electricity
(kWh) and % reduction, cost (time-of-use tariff × kWh), carbon (hourly
intensity × kWh), the PMV distribution and % of occupied hours in band, and
unmet-load hours — writing `summary.json`. Energy is derived from recorded
telemetry (`facility_elec_j` de-duplicated per timestep) and cross-checked
against `eplusout.sql` for unmet hours. Comfort is a **hard constraint**: the
verdict only passes when the AI both saves energy *and* holds ≥ 95 % of occupied
zone-hours inside the PMV band.
