"""Tests for the MCP tool implementations against a bound in-memory bus."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecoloop.bus.control_state import ControlPolicy, default_policy
from ecoloop.bus.telemetry import TelemetryBuffer, TelemetryRecord
from ecoloop.config import load_config, load_targets
from ecoloop.mcp import tools as T


@pytest.fixture
def ctx(tmp_path):
    cfg = load_config()
    tgt = load_targets()
    telemetry = TelemetryBuffer(pmv_band=tuple(tgt.comfort.pmv_band))
    control = ControlPolicy(default_policy(["SPACE1-1", "SPACE2-1"]))
    err = tmp_path / "eplusout.err"
    context = T.ToolContext(
        config=cfg,
        targets=tgt,
        telemetry=telemetry,
        control=control,
        err_file=err,
        baseline_idf=tmp_path / "baseline.idf",
        generated_dir=tmp_path / "generated",
    )
    T.bind_context(context)
    return context


def _push(telemetry, zone, temp, pmv, elec, hour=14, occ=1.0):
    telemetry.push(TelemetryRecord(
        sim_time_s=hour * 3600.0, day_of_year=196, hour=hour, zone=zone,
        air_temp_c=temp, mean_radiant_c=temp, rel_humidity_pct=50.0, occupancy=occ,
        pmv=pmv, ppd=10.0, facility_elec_j=elec, heating_sp_c=21.0, cooling_sp_c=24.0,
    ))


def test_get_current_telemetry_empty(ctx):
    out = T.get_current_telemetry()
    assert out["available"] is False


def test_get_current_telemetry_reports_zones(ctx):
    _push(ctx.telemetry, "SPACE1-1", 24.0, 0.2, 1e6)
    _push(ctx.telemetry, "SPACE2-1", 25.0, 0.4, 1e6)
    out = T.get_current_telemetry()
    assert out["available"] is True
    assert set(out["zones"]) == {"SPACE1-1", "SPACE2-1"}
    assert out["zones"]["SPACE1-1"]["air_temp_c"] == 24.0


def test_get_telemetry_summary(ctx):
    _push(ctx.telemetry, "SPACE1-1", 24.0, 0.2, 1e6)
    out = T.get_telemetry_summary(60)
    assert out["available"] is True
    assert "per_zone" in out


def test_get_targets_uses_current_hour(ctx):
    _push(ctx.telemetry, "SPACE1-1", 24.0, 0.2, 1e6, hour=17)
    out = T.get_targets()
    assert out["hour"] == 17
    assert out["carbon_intensity_gco2_per_kwh"] == ctx.targets.carbon_at(17)
    assert "safe_ranges" in out


def test_set_zone_setpoints_clamps_and_updates(ctx):
    out = T.set_zone_setpoints("SPACE1-1", heating_sp=40.0, cooling_sp=10.0)
    assert out["applied"] is True
    assert 18.0 <= out["heating_sp_c"] <= 23.0
    assert 22.0 <= out["cooling_sp_c"] <= 28.0
    # Reflected in the shared policy.
    assert ctx.control.get().heating_for("SPACE1-1") == out["heating_sp_c"]


def test_set_ecm_flags_updates_policy(ctx):
    out = T.set_ecm_flags(night_setback=True, precool=True)
    assert out["night_setback"] is True
    assert ctx.control.get().ecm.night_setback is True


def test_parse_simulation_log_counts_and_tails(ctx):
    ctx.err_file.write_text(
        "\n".join([
            "Program Version ...",
            "   ** Warning ** Zone air temperature outside range A",
            "   ** Warning ** Zone air temperature outside range A",  # dup
            "   ** Warning ** Different warning B",
            "   ** Severe  ** Something serious happened",
        ]),
        encoding="utf-8",
    )
    out = T.parse_simulation_log("Warning")
    assert out["available"] is True
    assert out["counts"]["Warning"] == 3
    assert out["counts"]["Severe"] == 1
    # Dedup: only 2 unique warning lines + severe is >= warning rank.
    assert out["n_unique_at_level"] <= 4
    assert len(out["last_unique_lines"]) <= 5


def test_parse_simulation_log_missing_file(ctx):
    out = T.parse_simulation_log("Severe")
    assert out["available"] is False


def test_tool_registry_has_all_seven():
    assert len(T.TOOL_FUNCTIONS) == 7
