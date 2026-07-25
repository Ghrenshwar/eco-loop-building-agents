"""Validation, clamping, and fallback for LLM-proposed control policies.

This is the safety boundary between the (fallible) LLM and the live simulation.
Every setpoint the model proposes passes through :func:`guard_policy`, which:

* validates the raw proposal against a pydantic schema,
* clamps each setpoint into the configured safe range,
* enforces the minimum cooling-heating deadband (repairing, not rejecting,
  when possible),
* and, if the proposal is unusable, returns the last known-good policy with
  ``fallback=True`` so the caller can log a self-correction event.

The guard NEVER raises on bad LLM output — bad input yields a safe policy, not
a crash. That property is what the robustness tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from ..bus.control_state import ECMFlags, Policy, ZoneSetpoint
from ..config import SafetyBounds

# Small models often emit setpoint keys under slightly different names (echoing
# telemetry field names, abbreviations, etc.). We normalize these aliases before
# validation so a well-intentioned proposal isn't wasted on a field-name typo.
_HEATING_ALIASES = ("heating_sp", "heating_sp_c", "heating_c", "heating",
                    "htg", "htg_sp", "heat_sp", "heating_setpoint")
_COOLING_ALIASES = ("cooling_sp", "cooling_sp_c", "cooling_c", "cooling",
                    "clg", "clg_sp", "cool_sp", "cooling_setpoint")


class ProposedZone(BaseModel):
    """Loosely-typed zone proposal from the LLM (bounds enforced later)."""

    zone: str
    heating_sp: float
    cooling_sp: float

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "heating_sp" not in d:
            for a in _HEATING_ALIASES:
                if a in d:
                    d["heating_sp"] = d[a]
                    break
        if "cooling_sp" not in d:
            for a in _COOLING_ALIASES:
                if a in d:
                    d["cooling_sp"] = d[a]
                    break
        return d


class ProposedDecision(BaseModel):
    """The strict JSON schema we ask the LLM to emit."""

    setpoints: List[ProposedZone] = Field(default_factory=list)
    night_setback: bool = False
    precool: bool = False
    demand_response: bool = False
    rationale: str = ""


@dataclass
class GuardResult:
    policy: Policy
    fallback: bool
    repairs: List[str]              # human-readable notes on every clamp/repair

    @property
    def ok(self) -> bool:
        return not self.fallback


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def clamp_setpoints(
    heating_sp: float,
    cooling_sp: float,
    bounds: SafetyBounds,
    repairs: Optional[List[str]] = None,
    zone: str = "",
) -> ZoneSetpoint:
    """Clamp a heating/cooling pair into safe ranges and enforce the deadband."""
    repairs = repairs if repairs is not None else []
    h, hc = _clamp(float(heating_sp), bounds.heating_min_c, bounds.heating_max_c)
    c, cc = _clamp(float(cooling_sp), bounds.cooling_min_c, bounds.cooling_max_c)
    if hc:
        repairs.append(f"{zone}: heating {heating_sp}->{h} (range "
                       f"{bounds.heating_min_c}-{bounds.heating_max_c})")
    if cc:
        repairs.append(f"{zone}: cooling {cooling_sp}->{c} (range "
                       f"{bounds.cooling_min_c}-{bounds.cooling_max_c})")

    # Enforce deadband: widen cooling upward first, then heating downward,
    # staying inside the safe ranges.
    if c - h < bounds.min_deadband_c:
        needed = bounds.min_deadband_c
        new_c = min(h + needed, bounds.cooling_max_c)
        if new_c - h < needed:
            new_h = max(new_c - needed, bounds.heating_min_c)
            if new_h != h:
                repairs.append(f"{zone}: heating {h}->{new_h} to hold "
                               f"{needed}C deadband")
                h = new_h
        if new_c != c:
            repairs.append(f"{zone}: cooling {c}->{new_c} to hold "
                           f"{needed}C deadband")
            c = new_c
    return ZoneSetpoint(heating_sp=round(h, 2), cooling_sp=round(c, 2))


def guard_policy(
    raw: Any,
    *,
    zones: List[str],
    bounds: SafetyBounds,
    last_good: Policy,
    sim_time_s: float,
) -> GuardResult:
    """Validate + clamp *raw* LLM output into a safe :class:`Policy`.

    *raw* may be a dict, a :class:`ProposedDecision`, or malformed junk. On any
    validation failure we return *last_good* flagged as a fallback.
    """
    repairs: List[str] = []

    # 1. Parse into the strict schema. Any failure -> fallback.
    try:
        decision = raw if isinstance(raw, ProposedDecision) else ProposedDecision.model_validate(raw)
    except (ValidationError, TypeError, ValueError) as exc:
        return GuardResult(
            policy=_fallback(last_good, sim_time_s, f"schema invalid: {exc}"),
            fallback=True,
            repairs=[f"validation failed: {exc}"],
        )

    # 2. Clamp each proposed zone. Unknown zones are ignored (logged as repair).
    setpoints: Dict[str, ZoneSetpoint] = {}
    for pz in decision.setpoints:
        if pz.zone not in zones:
            repairs.append(f"ignored unknown zone '{pz.zone}'")
            continue
        try:
            setpoints[pz.zone] = clamp_setpoints(
                pz.heating_sp, pz.cooling_sp, bounds, repairs, pz.zone
            )
        except (ValidationError, ValueError) as exc:
            repairs.append(f"{pz.zone}: unrepairable ({exc}); kept last-good")

    # 3. Any zone the LLM omitted keeps its last-good setpoint (never leave a
    #    controlled zone without a setpoint).
    for z in zones:
        if z not in setpoints:
            prev = last_good.setpoints.get(z)
            if prev is not None:
                setpoints[z] = prev
            else:
                setpoints[z] = ZoneSetpoint(heating_sp=21.0, cooling_sp=24.0)
                repairs.append(f"{z}: no proposal and no last-good; used defaults")

    # 4. If nothing valid survived, fall back entirely.
    if not setpoints:
        return GuardResult(
            policy=_fallback(last_good, sim_time_s, "no valid setpoints produced"),
            fallback=True,
            repairs=repairs or ["empty setpoint proposal"],
        )

    policy = Policy(
        version=last_good.version,          # ControlPolicy.update() re-increments
        timestamp_s=sim_time_s,
        setpoints=setpoints,
        ecm=ECMFlags(
            night_setback=decision.night_setback,
            precool=decision.precool,
            demand_response=decision.demand_response,
        ),
        rationale=(decision.rationale or "no rationale given")[:400],
        fallback=False,
    )
    return GuardResult(policy=policy, fallback=False, repairs=repairs)


def _fallback(last_good: Policy, sim_time_s: float, reason: str) -> Policy:
    return Policy(
        version=last_good.version,
        timestamp_s=sim_time_s,
        setpoints=dict(last_good.setpoints),
        ecm=last_good.ecm,
        rationale=f"FALLBACK ({reason}); held last-good policy",
        fallback=True,
    )
