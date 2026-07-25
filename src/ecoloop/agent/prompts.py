"""System prompt, decision schema, and few-shot exemplars for the supervisor.

Prompts are kept short: the LLM is fed compact telemetry summaries (never raw
logs), a strict JSON output schema, explicit safety rules, and 1-2 exemplars.
Tool descriptions (from the MCP server) double as usage instructions.
"""

from __future__ import annotations

import json
from typing import Optional

SYSTEM_PROMPT = """\
You are the supervisory HVAC controller for a live building-energy simulation.
Your job: keep every occupied zone thermally comfortable while minimizing
energy use, cost, and carbon. You act about once per simulated hour.

HARD RULES (never violate):
- Keep PMV within [-0.5, +0.5] for every OCCUPIED zone. Comfort is a hard
  constraint; never trade it away for energy savings during occupied hours.
- Heating setpoint must stay in 18-23 C, cooling in 22-28 C, and
  cooling_sp - heating_sp >= 2 C. Values outside these are clamped for you, but
  propose valid values.

DEFAULT STRATEGY (this is a summer / cooling-season week — follow unless the
telemetry clearly says otherwise):
- If the zone is UNOCCUPIED: apply maximum setback -> heating 18, cooling 28.
  There is nobody to keep comfortable, so save as much energy as possible.
- If the zone is OCCUPIED: use heating 20, cooling 26. In summer a cooling
  setpoint of 26 is comfortable (PMV near 0 with light clothing) AND uses far
  LESS cooling energy than a low setpoint. Do NOT cool below 24 when occupied —
  that wastes energy and makes people too cold (PMV below -0.5).
- Only lower the cooling setpoint (pre-cool) if a zone is occupied AND its PMV
  is already above +0.5 (too warm). Never pre-cool an empty zone.
- If a previous decision left a zone too cold (PMV < -0.5), RAISE its setpoints
  toward comfort. If too warm (PMV > +0.5), lower the cooling setpoint.
- During a demand-response window, relax cooling (raise it) only if comfort holds.

WORKFLOW each cycle:
1. Call get_telemetry_summary to read recent building state.
2. Call get_targets to read the comfort band and this hour's carbon/tariff, the
   peak threshold, and whether the building is occupied now.
3. If a prior decision may have caused problems, call parse_simulation_log.
4. Decide per-zone setpoints and ECM flags. You MAY call set_zone_setpoints /
   set_ecm_flags to apply them, and snapshot_current_idf to save a snapshot.
5. Emit your FINAL answer as a single JSON object exactly matching the schema
   below. Use EXACTLY the keys "heating_sp" and "cooling_sp" (not "heating_c"
   or "cooling_sp_c"). No prose around the JSON.

DECISION JSON SCHEMA:
{schema}

Keep rationale to one short sentence. Set night_setback=true when unoccupied.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "setpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "heating_sp": {"type": "number"},
                    "cooling_sp": {"type": "number"},
                },
                "required": ["zone", "heating_sp", "cooling_sp"],
            },
        },
        "night_setback": {"type": "boolean"},
        "precool": {"type": "boolean"},
        "demand_response": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["setpoints", "rationale"],
}

FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Summary: hour=14, OCCUPIED. Zones SPACE1-1..SPACE5-1 all "
            "PMV ~-0.4 (slightly cold), temps ~24C. Carbon 390 gCO2/kWh (high), "
            "tariff $0.18, peak threshold 40kW."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "setpoints": [
                    {"zone": "SPACE1-1", "heating_sp": 20.0, "cooling_sp": 26.0},
                    {"zone": "SPACE2-1", "heating_sp": 20.0, "cooling_sp": 26.0},
                    {"zone": "SPACE3-1", "heating_sp": 20.0, "cooling_sp": 26.0},
                    {"zone": "SPACE4-1", "heating_sp": 20.0, "cooling_sp": 26.0},
                    {"zone": "SPACE5-1", "heating_sp": 20.0, "cooling_sp": 26.0},
                ],
                "night_setback": False,
                "precool": False,
                "demand_response": False,
                "rationale": "Occupied summer: raise cooling to 26 to save energy and warm slightly into band.",
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Summary: hour=2, UNOCCUPIED. Temps ~22C, PMV n/a (no occupants). "
            "Carbon low (195). No peak."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "setpoints": [
                    {"zone": "SPACE1-1", "heating_sp": 18.0, "cooling_sp": 28.0},
                    {"zone": "SPACE2-1", "heating_sp": 18.0, "cooling_sp": 28.0},
                    {"zone": "SPACE3-1", "heating_sp": 18.0, "cooling_sp": 28.0},
                    {"zone": "SPACE4-1", "heating_sp": 18.0, "cooling_sp": 28.0},
                    {"zone": "SPACE5-1", "heating_sp": 18.0, "cooling_sp": 28.0},
                ],
                "night_setback": True,
                "precool": False,
                "demand_response": False,
                "rationale": "Unoccupied overnight: wide setback deadband to save energy.",
            }
        ),
    },
]


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(schema=json.dumps(DECISION_SCHEMA, indent=2))


def build_user_message(summary: dict, targets: dict, correction_note: Optional[str] = None) -> str:
    """Compose the per-cycle user message from compact summaries."""
    lines = [
        "Current building telemetry summary (compact):",
        json.dumps(summary, separators=(",", ":")),
        "",
        "Targets for this hour:",
        json.dumps(targets, separators=(",", ":")),
    ]
    if correction_note:
        lines += ["", "SELF-CORRECTION CONTEXT (address this):", correction_note]
    lines += [
        "",
        "Decide setpoints and ECM flags. You may call tools first, then emit the "
        "final decision JSON.",
    ]
    return "\n".join(lines)
