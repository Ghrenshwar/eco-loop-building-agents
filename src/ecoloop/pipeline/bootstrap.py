"""Shared setup for the baseline / AI pipeline runs.

Locates the example building + weather file, prepares the canonical baseline
.idf via :mod:`..energyplus.idf_prep`, and applies ``--smoke`` (single design
day) overrides so both runs use an identical building/weather/period.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..config import Config, Targets
from ..energyplus.idf_prep import prepare_idf

# Preferred example building + weather bundled with EnergyPlus.
PREFERRED_IDF = "5ZoneAirCooled.idf"
PREFERRED_EPW = "USA_CO_Golden-NREL.724666_TMY3.epw"


@dataclass
class PreparedModel:
    idf: Path
    epw: Path
    wired: Dict[str, dict]


def _find_example_idf(cfg: Config) -> Path:
    ex_dir = cfg.energyplus.example_files_dir
    cand = ex_dir / PREFERRED_IDF
    if cand.exists():
        return cand
    # Fall back to any 5-zone example.
    for p in ex_dir.glob("5Zone*.idf"):
        return p
    raise FileNotFoundError(
        f"Could not find {PREFERRED_IDF} (or any 5Zone*.idf) in {ex_dir}. "
        "Copy an example building into models/ and set paths.idf."
    )


def _find_epw(cfg: Config) -> Path:
    wx_dir = cfg.energyplus.weather_data_dir
    cand = wx_dir / PREFERRED_EPW
    if cand.exists():
        return cand
    for p in wx_dir.glob("USA_CO_*.epw"):
        return p
    for p in wx_dir.glob("*.epw"):
        return p
    raise FileNotFoundError(
        f"No .epw found in {wx_dir}. Copy a weather file into models/ and set paths.epw."
    )


def prepare_baseline(cfg: Config, smoke: bool = False, force: bool = False) -> PreparedModel:
    """Ensure models/baseline.idf + models/weather.epw exist and are prepared."""
    idf_out = cfg.paths.idf
    epw_out = cfg.paths.epw
    idf_out.parent.mkdir(parents=True, exist_ok=True)

    # Weather: copy the bundled file into models/ once.
    if force or not epw_out.exists():
        src_epw = cfg.paths.epw if cfg.paths.epw.exists() else _find_epw(cfg)
        if src_epw.resolve() != epw_out.resolve():
            shutil.copy2(src_epw, epw_out)

    run_period = {
        "begin_month": cfg.run_period.begin_month,
        "begin_day": cfg.run_period.begin_day,
        "end_month": cfg.run_period.end_month,
        "end_day": cfg.run_period.end_day,
    }
    if smoke:
        # One representative summer design day for fast iteration.
        run_period = {"begin_month": 7, "begin_day": 15, "end_month": 7, "end_day": 15}

    # Choose the raw source: an existing prepared baseline is reused unless
    # forced; otherwise start from the bundled example.
    if force or not idf_out.exists():
        raw = _find_example_idf(cfg)
    else:
        raw = idf_out

    wired = prepare_idf(
        raw_idf=raw,
        out_idf=idf_out,
        energyplus_dir=cfg.energyplus.install_dir,
        zones=list(cfg.zones),
        run_period=run_period,
    )
    return PreparedModel(idf=idf_out, epw=epw_out, wired=wired)


def require_energyplus(cfg: Config) -> None:
    if not cfg.energyplus.exists():
        raise SystemExit(
            "EnergyPlus not found. Set energyplus.install_dir in config/config.yaml "
            "or the ENERGYPLUS_DIR env var, then re-run. "
            f"(looked at {cfg.energyplus.install_dir})"
        )
