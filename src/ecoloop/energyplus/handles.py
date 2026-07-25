"""Lazy acquisition and caching of EnergyPlus runtime handles.

Handles (variable / actuator / meter) are only valid *after* E+ has finished
building its data structures, so they must be fetched inside a callback once
``api.exchange.api_data_fully_ready(state)`` is true — never at registration
time. This helper fetches each handle exactly once, caches it, and reports any
handle that E+ returns as invalid (-1) so the runner can log a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ZoneHandles:
    air_temp: int = -1
    rel_humidity: int = -1
    mean_radiant: int = -1
    occupancy: int = -1
    heating_actuator: int = -1
    cooling_actuator: int = -1


@dataclass
class HandleCache:
    """Caches all handles for a run. Populated lazily on the first ready call."""

    zones: List[str]
    heating_sched_of: Dict[str, str]        # zone -> heating Schedule:Constant name
    cooling_sched_of: Dict[str, str]        # zone -> cooling Schedule:Constant name

    ready: bool = False
    facility_elec_meter: int = -1              # kept for compatibility (first valid)
    elec_meter_handles: List[int] = field(default_factory=list)  # summed for total
    per_zone: Dict[str, ZoneHandles] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    def acquire(self, api, state) -> bool:
        """Fetch and cache every handle. Returns True once fully acquired.

        Safe to call every timestep; it no-ops after the first success.
        """
        if self.ready:
            return True
        if not api.exchange.api_data_fully_ready(state):
            return False

        ex = api.exchange
        problems: List[str] = []

        # Total facility electricity. The 'Electricity:Facility' rollup meter is
        # not always resolvable via get_meter_handle in every E+ build, so we
        # prefer it but fall back to summing its components (Building + HVAC),
        # which equal the facility total for this model.
        facility = ex.get_meter_handle(state, "Electricity:Facility")
        if facility != -1:
            self.elec_meter_handles = [facility]
        else:
            # Facility = Building + HVAC + Plant (chiller) + exterior. Sum every
            # component that resolves so we capture cooling electricity too.
            for comp in ("Electricity:Building", "Electricity:HVAC",
                         "Electricity:Plant", "ExteriorEquipment:Electricity"):
                h = ex.get_meter_handle(state, comp)
                if h != -1:
                    self.elec_meter_handles.append(h)
            if not self.elec_meter_handles:
                problems.append("no electricity meter handle resolved "
                                "(Facility/Building/HVAC/Plant all -1)")
        self.facility_elec_meter = self.elec_meter_handles[0] if self.elec_meter_handles else -1

        for zone in self.zones:
            zh = ZoneHandles()
            zh.air_temp = ex.get_variable_handle(state, "Zone Mean Air Temperature", zone)
            zh.rel_humidity = ex.get_variable_handle(
                state, "Zone Air Relative Humidity", zone
            )
            # Mean radiant temp is optional; approximate with air temp if absent.
            zh.mean_radiant = ex.get_variable_handle(
                state, "Zone Mean Radiant Temperature", zone
            )
            zh.occupancy = ex.get_variable_handle(
                state, "Zone People Occupant Count", zone
            )
            # Actuate the zone thermostat setpoints directly via the built-in
            # "Zone Temperature Control" actuator (keyed by zone). This reliably
            # overrides the active setpoint regardless of how the thermostat's
            # control objects/schedules are wired in the source .idf — more
            # robust than rewiring shared Schedule:Constant objects. The named
            # ECO schedules from idf_prep still back the runtime .idf snapshots.
            zh.heating_actuator = ex.get_actuator_handle(
                state, "Zone Temperature Control", "Heating Setpoint", zone
            )
            zh.cooling_actuator = ex.get_actuator_handle(
                state, "Zone Temperature Control", "Cooling Setpoint", zone
            )
            for label, h in (
                ("air_temp", zh.air_temp),
                ("rel_humidity", zh.rel_humidity),
                ("occupancy", zh.occupancy),
                ("heating_actuator", zh.heating_actuator),
                ("cooling_actuator", zh.cooling_actuator),
            ):
                if h == -1:
                    problems.append(f"{zone}: handle '{label}' invalid (-1)")
            self.per_zone[zone] = zh

        self.problems = problems
        # We consider ourselves ready even with some optional handles missing;
        # the runner decides how strict to be, but core actuators must exist.
        self.ready = True
        return True

    def critical_ok(self) -> Tuple[bool, List[str]]:
        """Are the handles required for control all valid?"""
        bad: List[str] = []
        if not self.elec_meter_handles:
            bad.append("electricity meter (Facility/Building/HVAC)")
        for zone, zh in self.per_zone.items():
            if zh.air_temp == -1:
                bad.append(f"{zone} air temp")
            if zh.heating_actuator == -1:
                bad.append(f"{zone} heating actuator")
            if zh.cooling_actuator == -1:
                bad.append(f"{zone} cooling actuator")
        return (len(bad) == 0, bad)
