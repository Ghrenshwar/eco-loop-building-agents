"""Thread-safe telemetry bus shared between the E+ main thread (producer) and
the supervisor / MCP threads (consumers).

The E+ callback pushes one :class:`TelemetryRecord` per zone timestep. The
buffer is a bounded ring so memory stays flat over a multi-day run. Consumers
read compact aggregates via :meth:`TelemetryBuffer.hourly_summary`, which keeps
LLM prompts small (no raw log ever enters a prompt).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass(frozen=True)
class TelemetryRecord:
    """One sensor snapshot for one zone at one simulation timestep."""

    sim_time_s: float                 # seconds since sim start (E+ current_sim_time)
    day_of_year: int
    hour: int                         # local hour of day, 0-23
    zone: str
    air_temp_c: float
    mean_radiant_c: float
    rel_humidity_pct: float
    occupancy: float                  # people count
    pmv: float
    ppd: float
    # Facility-wide electricity for this timestep (J). Same value repeated across
    # zones within a timestep; the summary de-duplicates by timestep.
    facility_elec_j: float
    heating_sp_c: float               # setpoint actually applied this step
    cooling_sp_c: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HourlySummary:
    """Compact aggregate over a recent window (fed to the LLM)."""

    sim_time_s: float
    hour: int
    day_of_year: int
    window_minutes: float
    n_records: int
    per_zone: Dict[str, dict]         # zone -> {temp_mean/min/max, pmv_mean, occ_mean, ...}
    pmv_mean: float
    pmv_min: float
    pmv_max: float
    occupancy_mean: float
    kwh_in_window: float              # facility electricity consumed over the window
    any_out_of_band: bool             # any zone PMV outside [-0.5, 0.5]

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryBuffer:
    """Bounded, thread-safe ring buffer of :class:`TelemetryRecord`."""

    def __init__(self, maxlen: int = 20000, pmv_band: tuple[float, float] = (-0.5, 0.5)):
        self._buf: Deque[TelemetryRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._pmv_band = pmv_band

    # -- producer ---------------------------------------------------------- #
    def push(self, rec: TelemetryRecord) -> None:
        with self._lock:
            self._buf.append(rec)

    # -- consumers --------------------------------------------------------- #
    def latest(self) -> Optional[TelemetryRecord]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def latest_per_zone(self) -> Dict[str, TelemetryRecord]:
        """Most recent record for each zone."""
        out: Dict[str, TelemetryRecord] = {}
        with self._lock:
            for rec in reversed(self._buf):
                if rec.zone not in out:
                    out[rec.zone] = rec
        return out

    def snapshot(self) -> List[TelemetryRecord]:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def hourly_summary(self, window_minutes: float = 60.0) -> Optional[HourlySummary]:
        """Aggregate the last *window_minutes* of telemetry.

        Returns ``None`` if no telemetry has been recorded yet. Facility energy
        is summed over unique timesteps (records repeat the meter value per
        zone, so we de-duplicate by ``sim_time_s``).
        """
        with self._lock:
            if not self._buf:
                return None
            latest = self._buf[-1]
            cutoff = latest.sim_time_s - window_minutes * 60.0
            window = [r for r in self._buf if r.sim_time_s >= cutoff]

        if not window:
            window = [latest]

        lo, hi = self._pmv_band
        by_zone: Dict[str, List[TelemetryRecord]] = {}
        for r in window:
            by_zone.setdefault(r.zone, []).append(r)

        per_zone: Dict[str, dict] = {}
        all_pmv: List[float] = []
        all_occ: List[float] = []
        any_oob = False
        for zone, recs in by_zone.items():
            temps = [r.air_temp_c for r in recs]
            pmvs = [r.pmv for r in recs]
            occs = [r.occupancy for r in recs]
            oob = any((p < lo or p > hi) for p in pmvs)
            any_oob = any_oob or oob
            per_zone[zone] = {
                "temp_mean_c": round(_mean(temps), 3),
                "temp_min_c": round(min(temps), 3),
                "temp_max_c": round(max(temps), 3),
                "pmv_mean": round(_mean(pmvs), 3),
                "occ_mean": round(_mean(occs), 3),
                "heating_sp_c": recs[-1].heating_sp_c,
                "cooling_sp_c": recs[-1].cooling_sp_c,
                "out_of_band": oob,
            }
            all_pmv.extend(pmvs)
            all_occ.extend(occs)

        # De-duplicate facility energy by timestep, then convert J -> kWh.
        seen: Dict[float, float] = {}
        for r in window:
            seen[r.sim_time_s] = r.facility_elec_j
        kwh = sum(seen.values()) / 3.6e6

        return HourlySummary(
            sim_time_s=latest.sim_time_s,
            hour=latest.hour,
            day_of_year=latest.day_of_year,
            window_minutes=window_minutes,
            n_records=len(window),
            per_zone=per_zone,
            pmv_mean=round(_mean(all_pmv), 3),
            pmv_min=round(min(all_pmv), 3),
            pmv_max=round(max(all_pmv), 3),
            occupancy_mean=round(_mean(all_occ), 3),
            kwh_in_window=round(kwh, 4),
            any_out_of_band=any_oob,
        )


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
