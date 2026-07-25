"""Thread-safe control state shared between the supervisor (writer) and the E+
callback (reader).

:class:`ControlPolicy` is the single source of truth for what setpoints and
energy-conservation measures the fast control loop applies. It is validated
(setpoint ranges + deadband) and versioned so every change is auditable.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ZoneSetpoint(BaseModel):
    """A validated per-zone heating/cooling setpoint pair (deg C)."""

    heating_sp: float = Field(ge=10.0, le=30.0)
    cooling_sp: float = Field(ge=15.0, le=35.0)

    @model_validator(mode="after")
    def _deadband(self) -> "ZoneSetpoint":
        if self.cooling_sp - self.heating_sp < 2.0:
            raise ValueError(
                f"cooling_sp ({self.cooling_sp}) must exceed heating_sp "
                f"({self.heating_sp}) by >= 2 C deadband"
            )
        return self


class ECMFlags(BaseModel):
    """Energy-conservation-measure toggles."""

    night_setback: bool = False
    precool: bool = False
    demand_response: bool = False


class Policy(BaseModel):
    """An immutable snapshot of the full control policy at a version."""

    version: int = 0
    timestamp_s: float = 0.0          # sim time this policy became active
    setpoints: Dict[str, ZoneSetpoint]
    ecm: ECMFlags = ECMFlags()
    rationale: str = "initial policy"
    fallback: bool = False            # True if this was a safety fallback

    def heating_for(self, zone: str) -> Optional[float]:
        sp = self.setpoints.get(zone)
        return sp.heating_sp if sp else None

    def cooling_for(self, zone: str) -> Optional[float]:
        sp = self.setpoints.get(zone)
        return sp.cooling_sp if sp else None


class ControlPolicy:
    """Thread-safe holder for the current :class:`Policy`.

    Reads (the fast E+ loop) and writes (the supervisor) are guarded by a lock.
    Updates are atomic: a whole new ``Policy`` object replaces the old one, so a
    reader never sees a half-updated setpoint map.
    """

    def __init__(self, initial: Policy):
        self._policy = initial
        self._lock = threading.Lock()

    def get(self) -> Policy:
        with self._lock:
            return self._policy

    def update(self, new_policy: Policy) -> Policy:
        """Atomically install *new_policy*, auto-incrementing its version."""
        with self._lock:
            new_policy.version = self._policy.version + 1
            self._policy = new_policy
            return new_policy

    @property
    def version(self) -> int:
        with self._lock:
            return self._policy.version


def default_policy(
    zones: list[str],
    heating_sp: float = 21.0,
    cooling_sp: float = 24.0,
    timestamp_s: float = 0.0,
) -> Policy:
    """A safe, comfortable starting policy applied to every zone."""
    return Policy(
        setpoints={z: ZoneSetpoint(heating_sp=heating_sp, cooling_sp=cooling_sp) for z in zones},
        timestamp_s=timestamp_s,
        rationale="default comfortable setpoints",
    )
