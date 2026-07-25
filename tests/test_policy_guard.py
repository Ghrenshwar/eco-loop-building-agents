"""Hard tests for policy_guard — the safety boundary (robustness component)."""

from __future__ import annotations

import pytest

from ecoloop.bus.control_state import Policy, ZoneSetpoint, default_policy
from ecoloop.config import SafetyBounds
from ecoloop.agent.policy_guard import (
    ProposedDecision,
    clamp_setpoints,
    guard_policy,
)

ZONES = ["Z1", "Z2"]
BOUNDS = SafetyBounds()  # defaults: htg 18-23, clg 22-28, deadband 2


def _last_good(sim_time=0.0):
    return default_policy(ZONES, heating_sp=21.0, cooling_sp=24.0, timestamp_s=sim_time)


# -- clamping --------------------------------------------------------------- #

def test_clamp_below_range():
    sp = clamp_setpoints(heating_sp=5.0, cooling_sp=24.0, bounds=BOUNDS)
    assert sp.heating_sp == 18.0


def test_clamp_above_range():
    sp = clamp_setpoints(heating_sp=21.0, cooling_sp=40.0, bounds=BOUNDS)
    assert sp.cooling_sp == 28.0


def test_clamp_repairs_deadband():
    # heating 23, cooling 22 -> deadband negative; must be widened to >= 2C.
    sp = clamp_setpoints(heating_sp=23.0, cooling_sp=22.0, bounds=BOUNDS)
    assert sp.cooling_sp - sp.heating_sp >= 2.0


def test_clamp_records_repairs():
    repairs = []
    clamp_setpoints(50.0, -10.0, BOUNDS, repairs=repairs, zone="Z1")
    assert any("heating" in r for r in repairs)
    assert any("cooling" in r for r in repairs)


# -- guard_policy: happy path ---------------------------------------------- #

def test_guard_valid_decision_applies():
    raw = {
        "setpoints": [
            {"zone": "Z1", "heating_sp": 20.0, "cooling_sp": 24.0},
            {"zone": "Z2", "heating_sp": 19.0, "cooling_sp": 25.0},
        ],
        "night_setback": True,
        "rationale": "ok",
    }
    res = guard_policy(raw, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=100.0)
    assert res.ok and not res.fallback
    assert res.policy.ecm.night_setback is True
    assert res.policy.heating_for("Z1") == 20.0


def test_guard_clamps_out_of_range_values():
    raw = {"setpoints": [{"zone": "Z1", "heating_sp": 30.0, "cooling_sp": 15.0}],
           "rationale": "extreme"}
    res = guard_policy(raw, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=1.0)
    assert not res.fallback
    sp = res.policy.setpoints["Z1"]
    assert 18.0 <= sp.heating_sp <= 23.0
    assert 22.0 <= sp.cooling_sp <= 28.0
    assert sp.cooling_sp - sp.heating_sp >= 2.0
    assert res.repairs  # repairs were recorded


# -- guard_policy: fallback paths ------------------------------------------ #

def test_guard_none_input_falls_back():
    res = guard_policy(None, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=5.0)
    assert res.fallback is True
    assert res.policy.heating_for("Z1") == 21.0  # last-good retained


def test_guard_garbage_input_falls_back():
    res = guard_policy("not a decision", zones=ZONES, bounds=BOUNDS,
                       last_good=_last_good(), sim_time_s=5.0)
    assert res.fallback is True


def test_guard_missing_required_field_falls_back():
    # setpoints entries missing cooling_sp -> schema invalid.
    raw = {"setpoints": [{"zone": "Z1", "heating_sp": 20.0}], "rationale": "x"}
    res = guard_policy(raw, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=5.0)
    assert res.fallback is True


def test_guard_unknown_zone_ignored_but_others_apply():
    raw = {"setpoints": [
        {"zone": "GHOST", "heating_sp": 20, "cooling_sp": 24},
        {"zone": "Z1", "heating_sp": 20, "cooling_sp": 24},
    ], "rationale": "x"}
    res = guard_policy(raw, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=5.0)
    assert not res.fallback
    assert "Z1" in res.policy.setpoints
    assert any("unknown zone" in r.lower() for r in res.repairs)


def test_guard_omitted_zone_keeps_last_good():
    # Only Z1 proposed; Z2 must retain last-good setpoints.
    raw = {"setpoints": [{"zone": "Z1", "heating_sp": 19, "cooling_sp": 24}], "rationale": "x"}
    res = guard_policy(raw, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=5.0)
    assert res.policy.heating_for("Z2") == 21.0


def test_guard_accepts_proposeddecision_object():
    dec = ProposedDecision(
        setpoints=[{"zone": "Z1", "heating_sp": 20, "cooling_sp": 24}], rationale="obj"
    )
    res = guard_policy(dec, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=5.0)
    assert not res.fallback


def test_guard_never_raises_on_bad_types():
    for bad in [123, [], {"setpoints": "nope"}, {"setpoints": [1, 2, 3]}]:
        res = guard_policy(bad, zones=ZONES, bounds=BOUNDS, last_good=_last_good(), sim_time_s=1.0)
        assert res.fallback is True  # safe fallback, no exception
