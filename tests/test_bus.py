"""Tests for the thread-safe telemetry + control-state bus."""

from __future__ import annotations

import threading

import pytest

from ecoloop.bus.control_state import (
    ControlPolicy,
    ECMFlags,
    Policy,
    ZoneSetpoint,
    default_policy,
)
from ecoloop.bus.telemetry import TelemetryBuffer, TelemetryRecord


def _rec(t, zone, temp, pmv, elec, occ=1.0, hour=12):
    return TelemetryRecord(
        sim_time_s=t, day_of_year=196, hour=hour, zone=zone, air_temp_c=temp,
        mean_radiant_c=temp, rel_humidity_pct=50.0, occupancy=occ, pmv=pmv,
        ppd=10.0, facility_elec_j=elec, heating_sp_c=21.0, cooling_sp_c=24.0,
    )


def test_ring_buffer_is_bounded():
    buf = TelemetryBuffer(maxlen=10)
    for i in range(50):
        buf.push(_rec(i * 600, "Z1", 24.0, 0.1, 1e6))
    assert len(buf) == 10


def test_latest_per_zone():
    buf = TelemetryBuffer(maxlen=100)
    buf.push(_rec(0, "Z1", 23.0, 0.1, 1e6))
    buf.push(_rec(0, "Z2", 25.0, 0.2, 1e6))
    buf.push(_rec(600, "Z1", 24.0, 0.15, 1.1e6))
    per_zone = buf.latest_per_zone()
    assert set(per_zone) == {"Z1", "Z2"}
    assert per_zone["Z1"].air_temp_c == 24.0  # most recent Z1


def test_hourly_summary_dedups_energy():
    buf = TelemetryBuffer(maxlen=1000, pmv_band=(-0.5, 0.5))
    # Two timesteps, two zones each; facility energy repeats per zone.
    buf.push(_rec(0, "Z1", 24.0, 0.1, 1_000_000))
    buf.push(_rec(0, "Z2", 24.0, 0.1, 1_000_000))
    buf.push(_rec(600, "Z1", 24.0, 0.2, 2_000_000))
    buf.push(_rec(600, "Z2", 24.0, 0.2, 2_000_000))
    s = buf.hourly_summary(window_minutes=60)
    assert s is not None
    # kWh should sum unique timesteps: (1e6 + 2e6) J = 3e6 J = 0.8333 kWh
    # (summary rounds to 4 dp, so compare with matching absolute tolerance).
    assert s.kwh_in_window == pytest.approx(3_000_000 / 3.6e6, abs=1e-4)
    assert not s.any_out_of_band


def test_hourly_summary_flags_out_of_band():
    buf = TelemetryBuffer(maxlen=100, pmv_band=(-0.5, 0.5))
    buf.push(_rec(0, "Z1", 28.0, 1.2, 1e6))  # PMV way high
    s = buf.hourly_summary(60)
    assert s.any_out_of_band is True
    assert s.per_zone["Z1"]["out_of_band"] is True


def test_empty_summary_is_none():
    assert TelemetryBuffer().hourly_summary() is None


def test_control_policy_atomic_update_increments_version():
    cp = ControlPolicy(default_policy(["Z1", "Z2"]))
    assert cp.version == 0
    new = Policy(setpoints={"Z1": ZoneSetpoint(heating_sp=20, cooling_sp=25)})
    applied = cp.update(new)
    assert applied.version == 1
    assert cp.get().heating_for("Z1") == 20


def test_zone_setpoint_enforces_deadband():
    with pytest.raises(ValueError):
        ZoneSetpoint(heating_sp=23.0, cooling_sp=24.0)  # only 1C deadband


def test_control_policy_thread_safe_under_contention():
    cp = ControlPolicy(default_policy(["Z1"]))
    errors = []

    def writer():
        try:
            for i in range(200):
                cp.update(Policy(setpoints={"Z1": ZoneSetpoint(heating_sp=20, cooling_sp=24)}))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                _ = cp.get().heating_for("Z1")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert cp.version == 600
