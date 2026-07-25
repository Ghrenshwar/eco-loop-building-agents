"""Tests for the PMV/PPD comfort model."""

from __future__ import annotations

import pytest

from ecoloop.comfort.pmv import (
    ComfortResult,
    clo_for_season,
    compute_pmv,
    in_band,
    _fanger_iso7730,
)


def test_neutral_conditions_near_zero_pmv():
    # ~24C, 50% RH, light summer clothing, office met -> comfortable (|PMV| small).
    res = compute_pmv(air_temp_c=24.0, rel_humidity_pct=50.0, clo=0.5, met_rate=1.1)
    assert isinstance(res, ComfortResult)
    assert abs(res.pmv) < 0.5
    assert 0.0 <= res.ppd <= 20.0


def test_hot_conditions_positive_pmv():
    res = compute_pmv(air_temp_c=30.0, rel_humidity_pct=60.0, clo=0.5, met_rate=1.1)
    assert res.pmv > 0.5


def test_cold_conditions_negative_pmv():
    res = compute_pmv(air_temp_c=17.0, rel_humidity_pct=40.0, clo=0.5, met_rate=1.1)
    assert res.pmv < -0.5


def test_fanger_reference_iso7730_example():
    # ISO 7730 worked example: tdb=tr=22, vr=0.1, rh=60, met=1.2, clo=0.5
    # -> PMV ~ -0.75, PPD ~ 17 (reference values).
    res = _fanger_iso7730(22.0, 22.0, 0.1, 60.0, 1.2, 0.5)
    assert res.pmv == pytest.approx(-0.75, abs=0.15)
    assert res.ppd == pytest.approx(17.0, abs=5.0)


def test_ppd_minimum_around_5pct_at_neutral():
    res = _fanger_iso7730(23.5, 23.5, 0.1, 50.0, 1.1, 0.5)
    assert res.ppd >= 5.0


def test_clo_seasonal_switch():
    assert clo_for_season(196, 0.5, 1.0) == 0.5   # mid-July -> summer
    assert clo_for_season(15, 0.5, 1.0) == 1.0    # mid-Jan -> winter


def test_in_band_helper():
    assert in_band(0.0)
    assert in_band(-0.5)
    assert not in_band(0.7)


def test_mrt_defaults_to_air_temp():
    a = compute_pmv(air_temp_c=25.0, rel_humidity_pct=50.0)
    b = compute_pmv(air_temp_c=25.0, rel_humidity_pct=50.0, mean_radiant_c=25.0)
    assert a.pmv == pytest.approx(b.pmv, abs=1e-6)
