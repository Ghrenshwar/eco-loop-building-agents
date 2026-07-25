"""Tool implementations for the EcoLoop MCP server.

These functions operate on the shared bus (telemetry + control policy) and the
run's config/targets. They are deliberately plain Python so they can be unit
tested directly (see tests/test_mcp_tools.py) *and* registered on a FastMCP
server (see server.py). Each returns a typed, JSON-serializable dict; the
docstrings double as the tool descriptions the LLM reads.

A single module-level :class:`ToolContext` wires the tools to the live bus. The
server sets it via :func:`bind_context` before serving; tests set it directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..bus.control_state import ControlPolicy, Policy
from ..bus.telemetry import TelemetryBuffer
from ..config import Config, SafetyBounds, Targets


@dataclass
class ToolContext:
    config: Config
    targets: Targets
    telemetry: TelemetryBuffer
    control: ControlPolicy
    err_file: Path                       # eplusout.err of the active run
    baseline_idf: Path
    generated_dir: Path
    # Optional hook so set_zone_setpoints/set_ecm_flags route through the guard
    # and get logged as decisions. When None, tools update the policy directly.
    apply_setpoints_hook: Optional[callable] = None
    apply_ecm_hook: Optional[callable] = None
    snapshot_counter: int = 0


_CTX: Optional[ToolContext] = None


def bind_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("ToolContext not bound; call bind_context() first")
    return _CTX


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def get_current_telemetry() -> dict:
    """Return the most recent sensor snapshot for every zone.

    Use this to see the building's current state: per-zone air temperature (C),
    relative humidity (%), occupancy, PMV, PPD, and the currently-applied
    heating/cooling setpoints, plus the facility electricity meter reading.
    """
    ctx = _ctx()
    per_zone = ctx.telemetry.latest_per_zone()
    if not per_zone:
        return {"available": False, "reason": "no telemetry yet"}
    zones = {
        z: {
            "air_temp_c": r.air_temp_c,
            "rel_humidity_pct": r.rel_humidity_pct,
            "occupancy": r.occupancy,
            "pmv": r.pmv,
            "ppd": r.ppd,
            "heating_sp_c": r.heating_sp_c,
            "cooling_sp_c": r.cooling_sp_c,
        }
        for z, r in per_zone.items()
    }
    any_rec = next(iter(per_zone.values()))
    return {
        "available": True,
        "sim_time_s": any_rec.sim_time_s,
        "hour": any_rec.hour,
        "day_of_year": any_rec.day_of_year,
        "facility_elec_j_last": any_rec.facility_elec_j,
        "zones": zones,
    }


def get_telemetry_summary(window_minutes: int = 60) -> dict:
    """Return a compact aggregated summary over the last ``window_minutes``.

    This is the preferred way to read building state before making a decision:
    it keeps prompts small. Includes per-zone temp mean/min/max, mean PMV, mean
    occupancy, energy used (kWh) in the window, and whether any zone drifted out
    of the comfort band.
    """
    ctx = _ctx()
    summary = ctx.telemetry.hourly_summary(float(window_minutes))
    if summary is None:
        return {"available": False, "reason": "no telemetry yet"}
    d = summary.to_dict()
    d["available"] = True
    return d


def get_targets() -> dict:
    """Return the control targets for the current simulated hour.

    Includes the comfort PMV band, the peak-demand threshold (kW), and the
    carbon intensity (gCO2/kWh) and electricity tariff ($/kWh) for the current
    hour — use these to decide when to pre-cool or shed load.
    """
    ctx = _ctx()
    latest = ctx.telemetry.latest()
    hour = latest.hour if latest else 12
    # Occupancy is taken from the live simulation (the People schedule), not a
    # synthetic calendar, so it always matches what the building is actually doing.
    per_zone = ctx.telemetry.latest_per_zone()
    occupied_now = any(r.occupancy > 0 for r in per_zone.values()) if per_zone else False
    return {
        "hour": hour,
        "pmv_band": list(ctx.targets.comfort.pmv_band),
        "ppd_max_pct": ctx.targets.comfort.ppd_max_pct,
        "peak_demand_threshold_kw": ctx.targets.demand.peak_threshold_kw,
        "carbon_intensity_gco2_per_kwh": ctx.targets.carbon_at(hour),
        "tariff_usd_per_kwh": ctx.targets.tariff_at(hour),
        "occupied_now": occupied_now,
        "safe_ranges": {
            "heating_c": [ctx.targets.safety.heating_min_c, ctx.targets.safety.heating_max_c],
            "cooling_c": [ctx.targets.safety.cooling_min_c, ctx.targets.safety.cooling_max_c],
            "min_deadband_c": ctx.targets.safety.min_deadband_c,
        },
    }


def set_zone_setpoints(zone: str, heating_sp: float, cooling_sp: float) -> dict:
    """Propose new heating/cooling setpoints (C) for one zone.

    The proposal is validated and clamped to the safe ranges before it reaches
    the simulation (heating 18-23, cooling 22-28, cooling-heating >= 2). Returns
    the values actually applied after clamping, and any repairs that were made.
    """
    ctx = _ctx()
    if ctx.apply_setpoints_hook is not None:
        return ctx.apply_setpoints_hook(zone, heating_sp, cooling_sp)

    # Direct path (used in tests): clamp locally and update the policy.
    from ..agent.policy_guard import clamp_setpoints

    repairs: List[str] = []
    sp = clamp_setpoints(heating_sp, cooling_sp, ctx.targets.safety, repairs, zone)
    cur = ctx.control.get()
    new_sps = dict(cur.setpoints)
    new_sps[zone] = sp
    new_policy = Policy(
        version=cur.version,
        timestamp_s=cur.timestamp_s,
        setpoints=new_sps,
        ecm=cur.ecm,
        rationale=f"set_zone_setpoints({zone})",
    )
    applied = ctx.control.update(new_policy)
    return {
        "applied": True,
        "zone": zone,
        "heating_sp_c": sp.heating_sp,
        "cooling_sp_c": sp.cooling_sp,
        "repairs": repairs,
        "policy_version": applied.version,
    }


def set_ecm_flags(
    night_setback: bool = False, precool: bool = False, demand_response: bool = False
) -> dict:
    """Toggle energy-conservation measures for the building.

    ``night_setback`` widens the deadband when unoccupied; ``precool`` cools
    ahead of high-carbon/peak hours; ``demand_response`` relaxes cooling during
    a DR window (only when comfort holds). Returns the active flags.
    """
    ctx = _ctx()
    if ctx.apply_ecm_hook is not None:
        return ctx.apply_ecm_hook(night_setback, precool, demand_response)

    from ..bus.control_state import ECMFlags

    cur = ctx.control.get()
    new_policy = Policy(
        version=cur.version,
        timestamp_s=cur.timestamp_s,
        setpoints=dict(cur.setpoints),
        ecm=ECMFlags(
            night_setback=night_setback, precool=precool, demand_response=demand_response
        ),
        rationale="set_ecm_flags",
    )
    applied = ctx.control.update(new_policy)
    return {
        "applied": True,
        "night_setback": night_setback,
        "precool": precool,
        "demand_response": demand_response,
        "policy_version": applied.version,
    }


_SEVERITY_RE = re.compile(r"\*\*\s*(Warning|Severe|Fatal)", re.IGNORECASE)


def parse_simulation_log(level: str = "Warning") -> dict:
    """Parse ``eplusout.err`` and return only counts + the last few unique lines.

    The raw error log can be thousands of lines; this never returns the whole
    file. ``level`` is the minimum severity to include ("Warning", "Severe", or
    "Fatal"). Use it to check whether the last decision caused runtime problems.
    """
    ctx = _ctx()
    order = {"warning": 0, "severe": 1, "fatal": 2}
    min_rank = order.get(level.lower(), 0)
    counts = {"Warning": 0, "Severe": 0, "Fatal": 0}
    unique_lines: List[str] = []
    seen = set()

    if not ctx.err_file.exists():
        return {"available": False, "reason": f"{ctx.err_file} not found", "counts": counts}

    with open(ctx.err_file, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = _SEVERITY_RE.search(raw)
            if not m:
                continue
            sev = m.group(1).capitalize()
            counts[sev] = counts.get(sev, 0) + 1
            if order.get(sev.lower(), 0) >= min_rank:
                line = raw.strip()
                key = line[:160]
                if key not in seen:
                    seen.add(key)
                    unique_lines.append(line)

    return {
        "available": True,
        "counts": counts,
        "min_level": level,
        "n_unique_at_level": len(unique_lines),
        "last_unique_lines": unique_lines[-5:],
    }


def snapshot_current_idf() -> dict:
    """Write an .idf snapshot reflecting the current AI setpoints to disk.

    Saves ``models/generated/ai_step_<simtime>.idf`` for the audit trail /
    "modified .idf generated at runtime" deliverable. Returns the file path.
    """
    ctx = _ctx()
    from ..energyplus.idf_prep import snapshot_idf

    policy = ctx.control.get()
    latest = ctx.telemetry.latest()
    tag = int(latest.sim_time_s) if latest else ctx.snapshot_counter
    setpoints = {z: (sp.heating_sp, sp.cooling_sp) for z, sp in policy.setpoints.items()}
    out = ctx.generated_dir / f"ai_step_{tag}.idf"
    try:
        snapshot_idf(ctx.baseline_idf, out, ctx.config.energyplus.install_dir, setpoints)
        ctx.snapshot_counter += 1
        return {"written": True, "path": str(out), "setpoints": setpoints}
    except Exception as exc:  # noqa: BLE001
        return {"written": False, "reason": str(exc)}


# Registry used by the server + tests to enumerate tools uniformly.
TOOL_FUNCTIONS = [
    get_current_telemetry,
    get_telemetry_summary,
    get_targets,
    set_zone_setpoints,
    set_ecm_flags,
    parse_simulation_log,
    snapshot_current_idf,
]
