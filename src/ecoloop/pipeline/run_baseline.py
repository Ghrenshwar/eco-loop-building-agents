"""Baseline run: fixed rule-based setpoints, no LLM.

Applies a constant comfortable policy (21/24 C) every timestep — a naive fixed
schedule. Records per-timestep telemetry and copies eplusout.sql. This is the
control against which the AI run's savings are measured; the building, weather,
and run period are identical to the AI run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bus.control_state import ControlPolicy, default_policy
from ..bus.telemetry import TelemetryBuffer
from ..config import load_config, load_targets
from ..energyplus.runner import EnergyPlusRunner
from ..logging.recorder import Recorder
from .bootstrap import prepare_baseline, require_energyplus


def run_baseline(smoke: bool = False, force_prepare: bool = False) -> Path:
    cfg, tgt = load_config(), load_targets()
    require_energyplus(cfg)

    model = prepare_baseline(cfg, smoke=smoke, force=force_prepare)
    telemetry = TelemetryBuffer(pmv_band=tuple(tgt.comfort.pmv_band))
    control = ControlPolicy(default_policy(cfg.zones, heating_sp=21.0, cooling_sp=24.0))
    recorder = Recorder("baseline", cfg.paths.output_dir)

    runner = EnergyPlusRunner(
        config=cfg,
        targets=tgt,
        control_policy=control,
        telemetry=telemetry,
        wired_scheds=model.wired,
        on_timestep=recorder.record_telemetry,
    )
    outdir = cfg.paths.output_dir / "baseline"
    artifacts = runner.run(model.idf, model.epw, outdir)

    written = recorder.flush(sql_source=artifacts.sql_file)
    print(f"[baseline] telemetry rows={recorder.n_telemetry} "
          f"exit={artifacts.exit_code} outputs={list(written.values())}")
    return outdir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fixed-schedule baseline simulation.")
    ap.add_argument("--smoke", action="store_true", help="single design day (fast)")
    ap.add_argument("--force-prepare", action="store_true", help="rebuild baseline.idf from example")
    args = ap.parse_args()
    run_baseline(smoke=args.smoke, force_prepare=args.force_prepare)


if __name__ == "__main__":
    main()
