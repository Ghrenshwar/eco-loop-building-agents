"""Record per-timestep telemetry and every supervisory decision to disk.

Writes are keyed by run name (``baseline`` / ``ai``). Telemetry is buffered in
memory and flushed to Parquet (with a CSV convenience copy); decisions are
appended to a JSONL file as they happen so a live tail shows the agent working.
The recorder is thread-safe: the E+ thread records telemetry while the
supervisor thread records decisions.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bus.telemetry import TelemetryRecord


@dataclass
class DecisionRecord:
    """One supervisory decision, for the audit log."""

    sim_time_s: float
    hour: int
    day_of_year: int
    version: int
    fallback: bool
    self_correction: bool
    rationale: str
    setpoints: Dict[str, Dict[str, float]]
    ecm: Dict[str, bool]
    repairs: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Recorder:
    def __init__(self, run_name: str, output_dir: Path):
        self.run_name = run_name
        self.dir = Path(output_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._telemetry: List[dict] = []
        self._decisions: List[dict] = []
        self._lock = threading.Lock()
        self._decisions_path = self.dir / "decisions.jsonl"
        # Truncate any prior decisions log for a clean run.
        self._decisions_path.write_text("", encoding="utf-8")

    # -- telemetry --------------------------------------------------------- #
    def record_telemetry(self, rec: TelemetryRecord) -> None:
        with self._lock:
            self._telemetry.append(rec.to_dict())

    # -- decisions --------------------------------------------------------- #
    def record_decision(self, dec: DecisionRecord) -> None:
        d = dec.to_dict()
        with self._lock:
            self._decisions.append(d)
            with open(self._decisions_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(d) + "\n")

    # -- finalize ---------------------------------------------------------- #
    def flush(self, sql_source: Optional[Path] = None) -> Dict[str, Path]:
        """Write telemetry/decisions to Parquet+CSV and copy eplusout.sql."""
        import pandas as pd

        out: Dict[str, Path] = {}
        with self._lock:
            tele_df = pd.DataFrame(self._telemetry)
            dec_df = pd.DataFrame(self._decisions)

        tele_parquet = self.dir / "telemetry.parquet"
        tele_csv = self.dir / "telemetry.csv"
        if not tele_df.empty:
            try:
                tele_df.to_parquet(tele_parquet, index=False)
                out["telemetry_parquet"] = tele_parquet
            except Exception:  # pyarrow missing -> CSV only
                pass
            tele_df.to_csv(tele_csv, index=False)
            out["telemetry_csv"] = tele_csv

        if not dec_df.empty:
            dec_csv = self.dir / "decisions.csv"
            dec_df.to_csv(dec_csv, index=False)
            out["decisions_csv"] = dec_csv

        if sql_source and Path(sql_source).exists():
            import shutil

            dest = self.dir / "eplusout.sql"
            # E+ often writes the .sql straight into this run dir already; only
            # copy when the source is a different file.
            if Path(sql_source).resolve() != dest.resolve():
                shutil.copy2(sql_source, dest)
            out["sql"] = dest

        out["decisions_jsonl"] = self._decisions_path
        return out

    @property
    def n_telemetry(self) -> int:
        with self._lock:
            return len(self._telemetry)

    @property
    def n_decisions(self) -> int:
        with self._lock:
            return len(self._decisions)
