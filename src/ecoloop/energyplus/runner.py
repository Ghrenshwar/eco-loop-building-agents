"""EnergyPlus Python-API wrapper: the fast control loop.

This owns the E+ ``state`` and registers a per-zone-timestep callback that:

* guards on ``api_data_fully_ready``,
* lazily acquires + caches handles (see :mod:`.handles`),
* reads sensors, computes PMV (:mod:`..comfort.pmv`), pushes a
  :class:`TelemetryRecord` onto the shared bus,
* reads the current :class:`Policy` and writes setpoints to the actuators.

The callback body is wrapped in ``try/except`` — an exception here must never
kill the simulation. On error we hold the last applied policy and log once.

The exact callback registration method name and exchange signatures vary
between EnergyPlus releases, so we resolve them from the installed
``pyenergyplus.api`` at runtime rather than hardcoding.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..bus.control_state import ControlPolicy, Policy
from ..bus.telemetry import TelemetryBuffer, TelemetryRecord
from ..comfort import pmv as pmvmod
from ..config import Config, Targets
from .actuators import apply_policy
from .handles import HandleCache


def ensure_pyenergyplus_importable(energyplus_dir: Path) -> None:
    """Add the EnergyPlus install dir to sys.path so ``pyenergyplus`` imports."""
    d = str(energyplus_dir)
    if d not in sys.path:
        sys.path.insert(0, d)


@dataclass
class RunArtifacts:
    output_dir: Path
    err_file: Path
    sql_file: Path
    exit_code: int = -1


class EnergyPlusRunner:
    """Runs one E+ simulation with the EcoLoop control callback attached."""

    def __init__(
        self,
        config: Config,
        targets: Targets,
        control_policy: ControlPolicy,
        telemetry: TelemetryBuffer,
        wired_scheds: Dict[str, dict],
        on_timestep: Optional[Callable[[TelemetryRecord], None]] = None,
        pace_s: float = 0.0,
    ):
        self.cfg = config
        self.tgt = targets
        self.policy = control_policy
        self.telemetry = telemetry
        self.on_timestep = on_timestep
        self.pace_s = max(0.0, float(pace_s))  # wall-clock pacing per timestep

        # zone -> heating/cooling Schedule:Constant names (from idf_prep).
        heating_of = {z: w["heating_sched"] for z, w in wired_scheds.items()}
        cooling_of = {z: w["cooling_sched"] for z, w in wired_scheds.items()}
        self.handles = HandleCache(
            zones=list(config.zones),
            heating_sched_of=heating_of,
            cooling_sched_of=cooling_of,
        )

        self._api = None
        self._state = None
        self._callback_errors = 0
        self._last_applied: Dict[str, Tuple[float, float]] = {}
        self._log = _make_logger(config)

    # ------------------------------------------------------------------ #
    def _load_api(self):
        ensure_pyenergyplus_importable(self.cfg.energyplus.install_dir)
        from pyenergyplus.api import EnergyPlusAPI  # type: ignore

        self._api = EnergyPlusAPI()
        self._state = self._api.state_manager.new_state()
        return self._api, self._state

    def _resolve_callback_registrar(self, api) -> Callable:
        """Pick the per-timestep calling point available in this E+ version."""
        rt = api.runtime
        for name in (
            "callback_begin_zone_timestep_after_init_heat_balance",
            "callback_begin_system_timestep_before_predictor",
            "callback_begin_zone_timestep_before_init_heat_balance",
        ):
            fn = getattr(rt, name, None)
            if callable(fn):
                self._log.info(f"Using E+ calling point: {name}")
                return fn
        raise RuntimeError(
            "No supported per-timestep callback found on this pyenergyplus.runtime"
        )

    def _request_variables(self, api, state) -> None:
        """Request non-default output variables so their handles resolve."""
        want = [
            ("Zone Mean Air Temperature", self.cfg.zones),
            ("Zone Air Relative Humidity", self.cfg.zones),
            ("Zone People Occupant Count", self.cfg.zones),
            ("Zone Mean Radiant Temperature", self.cfg.zones),
        ]
        for var, keys in want:
            for key in keys:
                try:
                    api.exchange.request_variable(state, var, key)
                except Exception as exc:  # noqa: BLE001
                    self._log.warning(f"request_variable({var},{key}) failed: {exc}")

    # ------------------------------------------------------------------ #
    def _timestep_callback(self, state) -> None:
        """Fast control loop. Must never raise out of this function."""
        api = self._api
        try:
            ex = api.exchange
            # Skip EnergyPlus warmup: during warmup the first day is repeated to
            # converge, so current_sim_time accumulates and meters are transient.
            # Recording it would double-count energy and corrupt the day index.
            if hasattr(ex, "warmup_flag") and ex.warmup_flag(state):
                return
            if not self.handles.acquire(api, state):
                return
            # sim time in seconds and calendar context.
            sim_time_s = float(ex.current_sim_time(state)) * 3600.0 \
                if _current_sim_time_is_hours(ex) else float(ex.current_time(state))
            hour = int(ex.hour(state))
            doy = int(ex.day_of_year(state))

            facility_j = 0.0
            for mh in self.handles.elec_meter_handles:
                facility_j += float(ex.get_meter_value(state, mh))

            policy: Policy = self.policy.get()
            clo = pmvmod.clo_for_season(
                doy, self.tgt.comfort.clo_summer, self.tgt.comfort.clo_winter
            )

            for zone, zh in self.handles.per_zone.items():
                if zh.air_temp == -1:
                    continue
                tair = float(ex.get_variable_value(state, zh.air_temp))
                rh = (
                    float(ex.get_variable_value(state, zh.rel_humidity))
                    if zh.rel_humidity != -1 else 50.0
                )
                mrt = (
                    float(ex.get_variable_value(state, zh.mean_radiant))
                    if zh.mean_radiant != -1 else tair
                )
                occ = (
                    float(ex.get_variable_value(state, zh.occupancy))
                    if zh.occupancy != -1 else 0.0
                )
                comfort = pmvmod.compute_pmv(
                    air_temp_c=tair,
                    rel_humidity_pct=rh,
                    mean_radiant_c=mrt,
                    air_speed_ms=self.tgt.comfort.air_speed_ms,
                    met_rate=self.tgt.comfort.met_rate,
                    clo=clo,
                )
                sp = policy.setpoints.get(zone)
                rec = TelemetryRecord(
                    sim_time_s=sim_time_s,
                    day_of_year=doy,
                    hour=hour,
                    zone=zone,
                    air_temp_c=round(tair, 3),
                    mean_radiant_c=round(mrt, 3),
                    rel_humidity_pct=round(rh, 2),
                    occupancy=round(occ, 2),
                    pmv=comfort.pmv,
                    ppd=comfort.ppd,
                    facility_elec_j=facility_j,
                    heating_sp_c=sp.heating_sp if sp else float("nan"),
                    cooling_sp_c=sp.cooling_sp if sp else float("nan"),
                )
                self.telemetry.push(rec)
                if self.on_timestep is not None:
                    self.on_timestep(rec)

            # Write current policy to actuators (fast loop applies latest policy).
            self._last_applied = apply_policy(
                api, state, self.handles, policy, self.tgt.safety
            )
        except Exception as exc:  # noqa: BLE001 — must not kill the sim
            self._callback_errors += 1
            if self._callback_errors <= 5 or self._callback_errors % 500 == 0:
                self._log.error(
                    f"callback error #{self._callback_errors}: {exc!r} "
                    f"(holding last policy)"
                )
        # Optional wall-clock pacing so the async supervisor can keep up. This
        # sleeps a fixed amount; it never waits on LLM inference.
        if self.pace_s:
            import time as _t
            _t.sleep(self.pace_s)

    # ------------------------------------------------------------------ #
    def run(self, idf: Path, epw: Path, outdir: Path) -> RunArtifacts:
        """Run the simulation to completion. Blocks the calling thread."""
        outdir.mkdir(parents=True, exist_ok=True)
        api, state = self._load_api()

        registrar = self._resolve_callback_registrar(api)
        registrar(state, self._timestep_callback)
        self._request_variables(api, state)

        argv = ["-d", str(outdir), "-w", str(epw), str(idf)]
        self._log.info(f"Starting EnergyPlus: {' '.join(argv)}")
        exit_code = api.runtime.run_energyplus(state, argv)
        self._log.info(f"EnergyPlus finished exit_code={exit_code} "
                       f"callback_errors={self._callback_errors}")

        ok, bad = self.handles.critical_ok()
        if not ok:
            self._log.warning(f"Critical handles were invalid: {bad}")

        api.state_manager.reset_state(state)
        return RunArtifacts(
            output_dir=outdir,
            err_file=outdir / "eplusout.err",
            sql_file=outdir / "eplusout.sql",
            exit_code=exit_code,
        )


def _current_sim_time_is_hours(ex) -> bool:
    """current_sim_time returns hours in modern E+; guard for older builds."""
    return hasattr(ex, "current_sim_time")


def _make_logger(cfg: Config):
    import logging as _logging

    logger = _logging.getLogger("ecoloop.runner")
    if not logger.handlers:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except Exception:  # noqa: BLE001
            handler = _logging.StreamHandler()
            fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(_logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(cfg.logging.level.upper())
    logger.propagate = False
    return logger
