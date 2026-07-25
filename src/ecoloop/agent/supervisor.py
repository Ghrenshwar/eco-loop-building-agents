"""Throttled supervisory loop.

Runs in its own thread. It watches the telemetry bus for simulated-time
progress and triggers a decision about once per simulated hour (or early on a
comfort/threshold breach). Each decision:

1. builds a compact summary + targets,
2. runs the LLM agent loop (which calls MCP tools),
3. validates + clamps the proposal (policy_guard) — falling back to the last
   known-good policy on timeout/garbage,
4. atomically updates ControlPolicy,
5. snapshots the .idf every N decisions,
6. records the decision (with tool calls + latency + repairs).

Self-correction: before each new decision, it checks whether the *previous*
decision's intent was met (comfort held, no new Severe/Fatal log lines). If not,
it passes a correction note into the next prompt and flags the decision as a
self-correction in the log.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..bus.control_state import ControlPolicy, Policy
from ..bus.telemetry import TelemetryBuffer
from ..config import Config, Targets
from ..logging.recorder import DecisionRecord, Recorder
from . import prompts
from .llm_client import AgentLoop
from .mcp_client import MCPClient
from .policy_guard import guard_policy
from . import mcp_client as _mcpmod  # noqa: F401 (kept for symmetry/testing)


class Supervisor:
    def __init__(
        self,
        config: Config,
        targets: Targets,
        telemetry: TelemetryBuffer,
        control: ControlPolicy,
        mcp_client: MCPClient,
        recorder: Recorder,
        parse_log_fn=None,
    ):
        self.cfg = config
        self.tgt = targets
        self.telemetry = telemetry
        self.control = control
        self.recorder = recorder
        self.agent = AgentLoop(config.llm, mcp_client)
        self.parse_log_fn = parse_log_fn  # callable() -> parse_simulation_log dict

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log = _logger(config)

        self._interval_s = config.supervisor.interval_sim_minutes * 60.0
        self._next_decision_sim_s = 0.0
        self._decisions = 0
        self._last_intent: Optional[Policy] = None
        self._last_severe_count = 0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        self._log.info("Supervisor started.")
        while not self._stop.is_set():
            latest = self.telemetry.latest()
            if latest is None:
                self._sleep(0.2)
                continue

            sim_now = latest.sim_time_s
            breach = self._breach_now()
            due = sim_now >= self._next_decision_sim_s
            if due or (breach and self.cfg.supervisor.breach_triggers_early):
                try:
                    self._make_decision(sim_now, triggered_by="breach" if (breach and not due) else "interval")
                except Exception as exc:  # noqa: BLE001
                    self._log.error(f"decision cycle failed: {exc!r}; holding policy")
                self._next_decision_sim_s = sim_now + self._interval_s
            self._sleep(0.2)
        self._log.info(f"Supervisor stopped after {self._decisions} decisions.")

    def _sleep(self, s: float) -> None:
        # Interruptible sleep so stop() is responsive.
        self._stop.wait(timeout=s)

    # ------------------------------------------------------------------ #
    def _breach_now(self) -> bool:
        summary = self.telemetry.hourly_summary(15.0)
        if summary is None:
            return False
        # Breach if any occupied zone is out of band right now.
        return bool(summary.any_out_of_band)

    def _correction_note(self) -> Optional[str]:
        """Compare realized state against last intent; build a note if violated."""
        if self._last_intent is None:
            return None
        notes = []

        summary = self.telemetry.hourly_summary(self.cfg.supervisor.interval_sim_minutes)
        if summary and summary.any_out_of_band:
            oob = [z for z, d in summary.per_zone.items() if d.get("out_of_band")]
            notes.append(
                f"Previous decision left zones {oob} outside the comfort band "
                f"(PMV min {summary.pmv_min}, max {summary.pmv_max}). Correct it — "
                f"if you widened a setback too far, tighten it."
            )

        if self.parse_log_fn is not None:
            try:
                logres = self.parse_log_fn()
                severe = logres.get("counts", {}).get("Severe", 0) + logres.get("counts", {}).get("Fatal", 0)
                if severe > self._last_severe_count:
                    notes.append(
                        f"New Severe/Fatal simulation log lines appeared since the "
                        f"last decision: {logres.get('last_unique_lines', [])}. Adjust."
                    )
                self._last_severe_count = severe
            except Exception:  # noqa: BLE001
                pass

        return " ".join(notes) if notes else None

    # ------------------------------------------------------------------ #
    def _make_decision(self, sim_now: float, triggered_by: str) -> None:
        summary = self.telemetry.hourly_summary(self.cfg.supervisor.interval_sim_minutes)
        if summary is None:
            return
        targets = self._targets_snapshot(summary.hour, summary.occupancy_mean)
        correction = self._correction_note()

        decision_raw, meta = self.agent.decide(summary.to_dict(), targets, correction)

        last_good = self.control.get()
        guard = guard_policy(
            decision_raw if decision_raw is not None else {},
            zones=list(self.cfg.zones),
            bounds=self.tgt.safety,
            last_good=last_good,
            sim_time_s=sim_now,
        )
        applied = self.control.update(guard.policy)
        self._decisions += 1

        is_correction = correction is not None and not guard.fallback
        self._last_intent = applied

        # Snapshot the .idf every N decisions.
        if self._decisions % self.cfg.supervisor.snapshot_every_n_decisions == 0:
            self._maybe_snapshot()

        self._log.info(
            f"[decision {self._decisions}] trigger={triggered_by} "
            f"fallback={guard.fallback} correction={is_correction} "
            f"tools={meta.get('tool_calls')} latency={meta.get('latency_s')}s "
            f"rationale={applied.rationale!r}"
        )
        if guard.repairs:
            self._log.info(f"  repairs: {guard.repairs}")

        self.recorder.record_decision(
            DecisionRecord(
                sim_time_s=sim_now,
                hour=summary.hour,
                day_of_year=summary.day_of_year,
                version=applied.version,
                fallback=guard.fallback,
                self_correction=is_correction,
                rationale=applied.rationale,
                setpoints={
                    z: {"heating_sp": sp.heating_sp, "cooling_sp": sp.cooling_sp}
                    for z, sp in applied.setpoints.items()
                },
                ecm={
                    "night_setback": applied.ecm.night_setback,
                    "precool": applied.ecm.precool,
                    "demand_response": applied.ecm.demand_response,
                },
                repairs=guard.repairs,
                tool_calls=meta.get("tool_calls", []),
                latency_s=meta.get("latency_s", 0.0),
            )
        )

    def _targets_snapshot(self, hour: int, occupancy_mean: float) -> dict:
        return {
            "hour": hour,
            "pmv_band": list(self.tgt.comfort.pmv_band),
            "peak_demand_threshold_kw": self.tgt.demand.peak_threshold_kw,
            "carbon_intensity_gco2_per_kwh": self.tgt.carbon_at(hour),
            "tariff_usd_per_kwh": self.tgt.tariff_at(hour),
            "occupied_now": occupancy_mean > 0.0,
            "safe_ranges": {
                "heating_c": [self.tgt.safety.heating_min_c, self.tgt.safety.heating_max_c],
                "cooling_c": [self.tgt.safety.cooling_min_c, self.tgt.safety.cooling_max_c],
                "min_deadband_c": self.tgt.safety.min_deadband_c,
            },
        }

    def _maybe_snapshot(self) -> None:
        try:
            from ..energyplus.idf_prep import snapshot_idf

            policy = self.control.get()
            setpoints = {z: (sp.heating_sp, sp.cooling_sp) for z, sp in policy.setpoints.items()}
            latest = self.telemetry.latest()
            tag = int(latest.sim_time_s) if latest else self._decisions
            out = self.cfg.paths.generated_dir / f"ai_step_{tag}.idf"
            snapshot_idf(self.cfg.paths.idf, out, self.cfg.energyplus.install_dir, setpoints)
            self._log.info(f"  snapshot written: {out}")
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"snapshot failed: {exc}")


def _logger(cfg: Config):
    import logging

    logger = logging.getLogger("ecoloop.supervisor")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(cfg.logging.level.upper())
    logger.propagate = False
    return logger
