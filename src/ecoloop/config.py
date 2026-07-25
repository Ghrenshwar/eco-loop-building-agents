"""Load and validate ``config/config.yaml`` and ``config/targets.yaml``.

All configuration flows through the pydantic models below, so the rest of the
codebase can rely on typed, validated settings. Paths are resolved to absolute
paths relative to the repository root, and the EnergyPlus install directory is
auto-discovered when the configured one does not exist.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Repo-root resolution
# --------------------------------------------------------------------------- #

# src/ecoloop/config.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _resolve(path: str | Path) -> Path:
    """Resolve *path* against the repo root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------- #
# config.yaml models
# --------------------------------------------------------------------------- #

# Common EnergyPlus install locations, newest first. Used when the configured
# install_dir is missing (e.g. a teammate on a different machine/OS).
_EPLUS_CANDIDATES = [
    "C:/EnergyPlusV26-1-0",
    "C:/EnergyPlusV25-1-0",
    "C:/EnergyPlusV24-2-0",
    "C:/EnergyPlusV24-1-0",
    "C:/EnergyPlusV23-2-0",
    "/usr/local/EnergyPlus-26-1-0",
    "/usr/local/EnergyPlus-24-2-0",
    "/Applications/EnergyPlus-26-1-0",
    "/Applications/EnergyPlus-24-2-0",
]


class EnergyPlusCfg(BaseModel):
    install_dir: Path
    version: str = "unknown"

    @model_validator(mode="after")
    def _discover(self) -> "EnergyPlusCfg":
        # Env var wins, then configured path, then well-known locations.
        env = os.environ.get("ENERGYPLUS_DIR")
        candidates = [env] if env else []
        candidates.append(str(self.install_dir))
        candidates.extend(_EPLUS_CANDIDATES)
        for c in candidates:
            if c and Path(c).expanduser().exists():
                self.install_dir = Path(c).expanduser().resolve()
                return self
        # Not found — keep the configured value so callers can emit a helpful
        # error at the point they actually need EnergyPlus (not at import time).
        self.install_dir = _resolve(self.install_dir)
        return self

    @property
    def pyenergyplus_dir(self) -> Path:
        return self.install_dir / "pyenergyplus"

    @property
    def example_files_dir(self) -> Path:
        return self.install_dir / "ExampleFiles"

    @property
    def weather_data_dir(self) -> Path:
        return self.install_dir / "WeatherData"

    def exists(self) -> bool:
        return self.install_dir.exists() and self.pyenergyplus_dir.exists()


class PathsCfg(BaseModel):
    idf: Path
    epw: Path
    output_dir: Path
    generated_dir: Path

    @field_validator("idf", "epw", "output_dir", "generated_dir", mode="before")
    @classmethod
    def _abs(cls, v):
        return _resolve(v)


class RunPeriodCfg(BaseModel):
    begin_month: int = Field(ge=1, le=12)
    begin_day: int = Field(ge=1, le=31)
    end_month: int = Field(ge=1, le=12)
    end_day: int = Field(ge=1, le=31)

    def as_dates(self, year: int = 2017) -> tuple[date, date]:
        return date(year, self.begin_month, self.begin_day), date(
            year, self.end_month, self.end_day
        )


class LLMCfg(BaseModel):
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    timeout_s: float = 45.0
    num_predict: int = 512
    temperature: float = 0.1
    max_tool_iters: int = 6


class MCPCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    transport: str = "streamable-http"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"


class SupervisorCfg(BaseModel):
    interval_sim_minutes: int = 60
    snapshot_every_n_decisions: int = 6
    breach_triggers_early: bool = True
    # Wall-clock pacing (seconds slept per zone timestep) applied ONLY to the AI
    # run, so the sim advances slowly enough for the async supervisor to make
    # ~hourly LLM decisions. This paces to wall-clock, never blocks on the LLM,
    # and does not affect physics/energy (the baseline runs at 0 = full speed).
    realtime_pace_s_per_step: float = 0.6


class LoggingCfg(BaseModel):
    level: str = "INFO"
    console_rich: bool = True


class Config(BaseModel):
    energyplus: EnergyPlusCfg
    paths: PathsCfg
    run_period: RunPeriodCfg
    zones: List[str]
    llm: LLMCfg = LLMCfg()
    mcp: MCPCfg = MCPCfg()
    supervisor: SupervisorCfg = SupervisorCfg()
    logging: LoggingCfg = LoggingCfg()

    @field_validator("zones")
    @classmethod
    def _nonempty_zones(cls, v):
        if not v:
            raise ValueError("config.zones must list at least one thermal zone")
        return v


# --------------------------------------------------------------------------- #
# targets.yaml models
# --------------------------------------------------------------------------- #


class ComfortTargets(BaseModel):
    pmv_band: tuple[float, float]
    ppd_max_pct: float = 10.0
    air_speed_ms: float = 0.1
    met_rate: float = 1.1
    clo_summer: float = 0.5
    clo_winter: float = 1.0

    @field_validator("pmv_band")
    @classmethod
    def _ordered(cls, v):
        lo, hi = v
        if lo >= hi:
            raise ValueError("pmv_band must be [low, high] with low < high")
        return v


class OccupancyTargets(BaseModel):
    occupied_start_hour: int = Field(ge=0, le=23)
    occupied_end_hour: int = Field(ge=1, le=24)
    occupied_weekdays: List[int]

    def is_occupied(self, weekday: int, hour: int) -> bool:
        return (
            weekday in self.occupied_weekdays
            and self.occupied_start_hour <= hour < self.occupied_end_hour
        )


class DemandTargets(BaseModel):
    peak_threshold_kw: float = 40.0


class SafetyBounds(BaseModel):
    heating_min_c: float = 18.0
    heating_max_c: float = 23.0
    cooling_min_c: float = 22.0
    cooling_max_c: float = 28.0
    min_deadband_c: float = 2.0


class Targets(BaseModel):
    comfort: ComfortTargets
    occupancy: OccupancyTargets
    demand: DemandTargets
    carbon_intensity_gco2_per_kwh: List[float]
    tariff_usd_per_kwh: List[float]
    safety: SafetyBounds

    @field_validator("carbon_intensity_gco2_per_kwh", "tariff_usd_per_kwh")
    @classmethod
    def _len24(cls, v):
        if len(v) != 24:
            raise ValueError("hourly curves must have exactly 24 values")
        return v

    def carbon_at(self, hour: int) -> float:
        return self.carbon_intensity_gco2_per_kwh[hour % 24]

    def tariff_at(self, hour: int) -> float:
        return self.tariff_usd_per_kwh[hour % 24]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: Optional[Path | str] = None) -> Config:
    path = Path(path) if path else CONFIG_DIR / "config.yaml"
    return Config.model_validate(_read_yaml(path))


def load_targets(path: Optional[Path | str] = None) -> Targets:
    path = Path(path) if path else CONFIG_DIR / "targets.yaml"
    return Targets.model_validate(_read_yaml(path))


@lru_cache(maxsize=1)
def get_settings() -> tuple[Config, Targets]:
    """Cached (config, targets) pair for convenient global access."""
    return load_config(), load_targets()


if __name__ == "__main__":  # quick manual sanity check
    cfg, tgt = load_config(), load_targets()
    print("Repo root:", REPO_ROOT)
    print("EnergyPlus dir:", cfg.energyplus.install_dir, "exists:", cfg.energyplus.exists())
    print("Zones:", cfg.zones)
    print("PMV band:", tgt.comfort.pmv_band)
    print("Carbon @ 17h:", tgt.carbon_at(17), "Tariff @ 17h:", tgt.tariff_at(17))
