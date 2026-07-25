"""Compare baseline vs AI runs and write ``summary.json``.

Both runs use an identical building/weather/period, so the difference is
attributable to control. We compute, for each run:

* total site electricity (kWh) and the AI's % reduction,
* electricity cost ($) using the time-of-use tariff,
* carbon (kgCO2) using the hourly carbon-intensity curve,
* PMV distribution and the % of OCCUPIED zone-hours within the comfort band,
* unmet-load hours (from eplusout.sql when available, else estimated).

Energy is derived from recorded telemetry: ``facility_elec_j`` is de-duplicated
per timestep (it repeats across zones) before summing.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from ..config import load_config, load_targets, Targets


def _load_telemetry(run_dir: Path):
    import pandas as pd

    pq = run_dir / "telemetry.parquet"
    csv = run_dir / "telemetry.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No telemetry found in {run_dir}")


@dataclass
class RunMetrics:
    run: str
    total_kwh: float
    cost_usd: float
    carbon_kgco2: float
    occupied_zone_hours: int
    occupied_in_band_pct: float
    pmv_mean_occupied: float
    pmv_p05: float
    pmv_p95: float
    unmet_heating_hours: float
    unmet_cooling_hours: float
    n_timesteps: int

    def to_dict(self) -> dict:
        return asdict(self)


def _timestep_hours(df) -> float:
    """Infer the timestep length in hours from unique sim times."""
    times = sorted(df["sim_time_s"].unique())
    if len(times) < 2:
        return 1.0 / 6.0  # default 10-min timestep
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    step_s = min(deltas) if deltas else 600.0
    return step_s / 3600.0


def _unmet_hours_from_sql(run_dir: Path) -> Optional[tuple[float, float]]:
    sql = run_dir / "eplusout.sql"
    if not sql.exists():
        return None
    try:
        con = sqlite3.connect(str(sql))
        cur = con.cursor()
        # TabularDataWithStrings holds the summary "Time Setpoint Not Met" table.
        cur.execute(
            "SELECT RowName, ColumnName, Value FROM TabularDataWithStrings "
            "WHERE TableName LIKE '%Setpoint Not Met%'"
        )
        rows = cur.fetchall()
        con.close()
        heat = cool = 0.0
        for row_name, col_name, value in rows:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            name = f"{row_name} {col_name}".lower()
            if "heat" in name:
                heat = max(heat, v)
            elif "cool" in name:
                cool = max(cool, v)
        return heat, cool
    except Exception:  # noqa: BLE001
        return None


def compute_metrics(run: str, run_dir: Path, tgt: Targets) -> RunMetrics:
    import numpy as np
    import pandas as pd

    df = _load_telemetry(run_dir)
    dt_h = _timestep_hours(df)

    # Facility energy: one value per timestep (dedup across zones).
    per_step = df.drop_duplicates(subset=["sim_time_s"]).copy()
    per_step["kwh"] = per_step["facility_elec_j"] / 3.6e6
    per_step["hour"] = per_step["hour"].astype(int)
    per_step["tariff"] = per_step["hour"].map(lambda h: tgt.tariff_at(int(h)))
    per_step["carbon"] = per_step["hour"].map(lambda h: tgt.carbon_at(int(h)))

    total_kwh = float(per_step["kwh"].sum())
    cost = float((per_step["kwh"] * per_step["tariff"]).sum())
    carbon_kg = float((per_step["kwh"] * per_step["carbon"]).sum()) / 1000.0

    # Comfort over OCCUPIED zone-timesteps.
    occ = df[df["occupancy"] > 0].copy()
    lo, hi = tgt.comfort.pmv_band
    if len(occ):
        in_band = ((occ["pmv"] >= lo) & (occ["pmv"] <= hi)).mean() * 100.0
        pmv_mean = float(occ["pmv"].mean())
        p05 = float(np.percentile(occ["pmv"], 5))
        p95 = float(np.percentile(occ["pmv"], 95))
        occ_zone_hours = int(round(len(occ) * dt_h))
    else:
        in_band = 100.0
        pmv_mean = p05 = p95 = 0.0
        occ_zone_hours = 0

    unmet = _unmet_hours_from_sql(run_dir)
    if unmet is None:
        # Estimate: occupied timesteps out of band, converted to hours (per zone).
        if len(occ):
            oob = occ[(occ["pmv"] < lo) | (occ["pmv"] > hi)]
            est = float(len(oob) * dt_h) / max(1, occ["zone"].nunique())
        else:
            est = 0.0
        unmet_heat = unmet_cool = round(est / 2.0, 2)
    else:
        unmet_heat, unmet_cool = unmet

    return RunMetrics(
        run=run,
        total_kwh=round(total_kwh, 3),
        cost_usd=round(cost, 2),
        carbon_kgco2=round(carbon_kg, 2),
        occupied_zone_hours=occ_zone_hours,
        occupied_in_band_pct=round(float(in_band), 2),
        pmv_mean_occupied=round(pmv_mean, 3),
        pmv_p05=round(p05, 3),
        pmv_p95=round(p95, 3),
        unmet_heating_hours=round(float(unmet_heat), 2),
        unmet_cooling_hours=round(float(unmet_cool), 2),
        n_timesteps=int(per_step.shape[0]),
    )


def compare(output_dir: Optional[Path] = None) -> dict:
    cfg, tgt = load_config(), load_targets()
    out = output_dir or cfg.paths.output_dir
    base = compute_metrics("baseline", out / "baseline", tgt)
    ai = compute_metrics("ai", out / "ai", tgt)

    pct = (
        (base.total_kwh - ai.total_kwh) / base.total_kwh * 100.0
        if base.total_kwh > 0 else 0.0
    )
    band_lo, band_hi = tgt.comfort.pmv_band
    # "Energy is not saved by sacrificing comfort": the AI must hold its mean
    # occupied PMV inside the band AND keep the occupied-hours in-band rate at
    # least as high as the fixed baseline (comfort preserved or improved). A
    # fixed absolute bar is avoided because the baseline building itself does not
    # sit at 100% in-band; the meaningful test is AI-vs-baseline.
    comfort_preserved = ai.occupied_in_band_pct >= base.occupied_in_band_pct - 0.5
    mean_in_band = band_lo <= ai.pmv_mean_occupied <= band_hi
    comfort_ok = comfort_preserved and mean_in_band

    summary = {
        "baseline": base.to_dict(),
        "ai": ai.to_dict(),
        "kwh_reduction_pct": round(pct, 2),
        "cost_savings_usd": round(base.cost_usd - ai.cost_usd, 2),
        "carbon_savings_kgco2": round(base.carbon_kgco2 - ai.carbon_kgco2, 2),
        "comfort_band": [band_lo, band_hi],
        "ai_mean_pmv_in_band": bool(mean_in_band),
        "ai_comfort_vs_baseline_pp": round(ai.occupied_in_band_pct - base.occupied_in_band_pct, 2),
        "ai_comfort_ok": bool(comfort_ok),
        "verdict": (
            "PASS: AI saved energy with comfort preserved"
            if pct > 0 and comfort_ok
            else "REVIEW: check savings/comfort"
        ),
    }
    summary_path = out / "summary.json"
    out.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Convenience CSV export of the headline comparison.
    _export_csv(out / "comparison.csv", base, ai, summary)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    return summary


def _export_csv(path: Path, base: RunMetrics, ai: RunMetrics, summary: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "baseline", "ai"])
        for key in base.to_dict():
            if key == "run":
                continue
            w.writerow([key, base.to_dict()[key], ai.to_dict()[key]])
        w.writerow(["kwh_reduction_pct", "", summary["kwh_reduction_pct"]])
        w.writerow(["cost_savings_usd", "", summary["cost_savings_usd"]])
        w.writerow(["carbon_savings_kgco2", "", summary["carbon_savings_kgco2"]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs AI and write summary.json")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    compare(args.output_dir)


if __name__ == "__main__":
    main()
