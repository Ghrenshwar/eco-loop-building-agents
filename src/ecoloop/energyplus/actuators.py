"""Apply control-policy setpoints to EnergyPlus actuators, with clamping.

We actuate ``Schedule:Constant`` objects (control type ``Schedule Value``),
which is the most reliable setpoint-override mechanism in E+: the zone
thermostat reads its setpoint from the named schedule, and we overwrite that
schedule's value each timestep. This module is a thin, defensive wrapper that
clamps once more at the boundary (belt-and-braces with policy_guard) and never
writes an invalid handle.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..bus.control_state import Policy
from ..config import SafetyBounds
from .handles import HandleCache


def apply_policy(
    api,
    state,
    handles: HandleCache,
    policy: Policy,
    bounds: SafetyBounds,
) -> Dict[str, Tuple[float, float]]:
    """Write *policy*'s setpoints to the E+ actuators.

    Returns the per-zone (heating, cooling) values actually written, so the
    caller can record exactly what the simulation saw.
    """
    applied: Dict[str, Tuple[float, float]] = {}
    ex = api.exchange
    for zone, zh in handles.per_zone.items():
        sp = policy.setpoints.get(zone)
        if sp is None:
            continue
        h = _clamp(sp.heating_sp, bounds.heating_min_c, bounds.heating_max_c)
        c = _clamp(sp.cooling_sp, bounds.cooling_min_c, bounds.cooling_max_c)
        if c - h < bounds.min_deadband_c:
            c = min(h + bounds.min_deadband_c, bounds.cooling_max_c)
        if zh.heating_actuator != -1:
            ex.set_actuator_value(state, zh.heating_actuator, h)
        if zh.cooling_actuator != -1:
            ex.set_actuator_value(state, zh.cooling_actuator, c)
        applied[zone] = (h, c)
    return applied


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
