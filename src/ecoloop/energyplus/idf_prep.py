"""Prepare a raw example .idf into EcoLoop's canonical, actuatable baseline.

Responsibilities (all via eppy, before any simulation runs):

1. Give every controlled zone a dedicated **heating** and **cooling**
   ``Schedule:Constant`` and rewire its ``ThermostatSetpoint:DualSetpoint`` to
   point at those schedules. Actuating a per-zone ``Schedule:Constant`` is the
   most reliable setpoint-override path in E+.
2. Ensure the output variables and meter we need are requested: Zone Mean Air
   Temperature, Zone Air Relative Humidity, Zone People Occupant Count, and the
   ``Electricity:Facility`` meter, plus unmet-hours reporting.
3. Enable ``Output:SQLite`` (Simple) so post-run analysis can read
   ``eplusout.sql``.
4. Set the RunPeriod to the configured dates.

The eppy field/object names differ slightly across E+ IDD versions, so we look
objects up defensively and log what we changed. ``snapshot_idf`` writes an
.idf reflecting the current AI policy into ``models/generated/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# eppy is imported lazily inside functions so the module imports even when eppy
# is not yet installed (keeps `import ecoloop...` cheap for unit tests).


def _idd_path(energyplus_dir: Path) -> Path:
    idd = energyplus_dir / "Energy+.idd"
    if not idd.exists():
        raise FileNotFoundError(
            f"Energy+.idd not found at {idd}. Is the EnergyPlus install_dir correct?"
        )
    return idd


def _get_idf(idf_path: Path, energyplus_dir: Path):
    from eppy.modeleditor import IDF  # type: ignore

    if getattr(IDF, "iddname", None) is None:
        IDF.setiddname(str(_idd_path(energyplus_dir)))
    return IDF(str(idf_path))


def sched_names(zone: str) -> Tuple[str, str]:
    """Deterministic (heating, cooling) Schedule:Constant names for a zone."""
    safe = zone.replace(" ", "_")
    return f"ECO_{safe}_HtgSP", f"ECO_{safe}_ClgSP"


def _ensure_schedule_typelimits(idf) -> str:
    """Ensure a temperature ScheduleTypeLimits exists; return its name."""
    name = "ECO_Temperature"
    for stl in idf.idfobjects.get("SCHEDULETYPELIMITS", []):
        if stl.Name.lower() == name.lower():
            return name
    stl = idf.newidfobject("SCHEDULETYPELIMITS", Name=name)
    # Fields vary by version; set what exists.
    for field, value in (
        ("Lower_Limit_Value", -60.0),
        ("Upper_Limit_Value", 200.0),
        ("Numeric_Type", "CONTINUOUS"),
        ("Unit_Type", "Temperature"),
    ):
        if hasattr(stl, field):
            setattr(stl, field, value)
    return name


def _ensure_constant_schedule(idf, name: str, value: float, type_limits: str) -> None:
    for sc in idf.idfobjects.get("SCHEDULE:CONSTANT", []):
        if sc.Name.lower() == name.lower():
            sc.Hourly_Value = value
            return
    kwargs = {"Name": name, "Hourly_Value": value}
    if type_limits:
        kwargs["Schedule_Type_Limits_Name"] = type_limits
    idf.newidfobject("SCHEDULE:CONSTANT", **kwargs)


def _zone_control_thermostats(idf) -> List:
    return list(idf.idfobjects.get("ZONECONTROL:THERMOSTAT", []))


def _dualsetpoints_by_name(idf) -> Dict[str, object]:
    out = {}
    for ds in idf.idfobjects.get("THERMOSTATSETPOINT:DUALSETPOINT", []):
        out[ds.Name.lower()] = ds
    return out


def prepare_idf(
    raw_idf: Path,
    out_idf: Path,
    energyplus_dir: Path,
    zones: List[str],
    run_period: Optional[dict] = None,
    default_heating: float = 21.0,
    default_cooling: float = 24.0,
) -> Dict[str, dict]:
    """Prepare *raw_idf* -> *out_idf*. Returns a mapping of what was wired.

    The returned dict maps zone -> {"heating_sched", "cooling_sched"} so the
    runner knows which schedule name backs each zone's actuator.
    """
    idf = _get_idf(raw_idf, energyplus_dir)
    type_limits = _ensure_schedule_typelimits(idf)

    wired: Dict[str, dict] = {}

    # Map zone -> its thermostat's dual-setpoint object, via ZoneControl:Thermostat.
    ds_by_name = _dualsetpoints_by_name(idf)
    tstats = _zone_control_thermostats(idf)

    def _dualsetpoint_for_zone(zone: str):
        for zc in tstats:
            zname = getattr(zc, "Zone_or_ZoneList_Name", "") or getattr(zc, "Zone_or_ZoneList_or_Space_or_SpaceList_Name", "")
            if zname.lower() != zone.lower():
                continue
            ctrl_type_sched = getattr(zc, "Control_1_Object_Type", "")
            ctrl_name = getattr(zc, "Control_1_Name", "")
            if "dualsetpoint" in ctrl_type_sched.lower():
                return ds_by_name.get(ctrl_name.lower())
        return None

    for zone in zones:
        h_name, c_name = sched_names(zone)
        _ensure_constant_schedule(idf, h_name, default_heating, type_limits)
        _ensure_constant_schedule(idf, c_name, default_cooling, type_limits)

        ds = _dualsetpoint_for_zone(zone)
        if ds is not None:
            ds.Heating_Setpoint_Temperature_Schedule_Name = h_name
            ds.Cooling_Setpoint_Temperature_Schedule_Name = c_name
        # If we couldn't find a dual-setpoint (unusual IDF layout), the schedules
        # still exist and can be actuated; we record the names regardless.
        wired[zone] = {"heating_sched": h_name, "cooling_sched": c_name,
                       "rewired": ds is not None}

    _ensure_outputs(idf, zones)
    _ensure_sqlite(idf)
    if run_period:
        _set_run_period(idf, run_period)

    out_idf.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(out_idf))
    return wired


def _ensure_outputs(idf, zones: List[str]) -> None:
    """Request the output variables and meter EcoLoop reads."""
    wanted_vars = [
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone People Occupant Count",
        "Zone Mean Radiant Temperature",
    ]
    existing = {
        (o.Variable_Name.lower(), str(o.Key_Value).lower())
        for o in idf.idfobjects.get("OUTPUT:VARIABLE", [])
    }
    for var in wanted_vars:
        if (var.lower(), "*") not in existing:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value="*",
                Variable_Name=var,
                Reporting_Frequency="Timestep",
            )
    # Facility electricity meter.
    have_meter = any(
        m.Key_Name.lower() == "electricity:facility"
        for m in idf.idfobjects.get("OUTPUT:METER", [])
    )
    if not have_meter:
        idf.newidfobject(
            "OUTPUT:METER", Key_Name="Electricity:Facility", Reporting_Frequency="Timestep"
        )
    # Unmet hours summary table.
    have_unmet = any(
        "setpointnotmet" in "".join(getattr(o, "Report_1_Name", "") for _ in [0]).lower()
        for o in idf.idfobjects.get("OUTPUT:TABLE:SUMMARYREPORTS", [])
    )
    if not idf.idfobjects.get("OUTPUT:TABLE:SUMMARYREPORTS", []):
        idf.newidfobject(
            "OUTPUT:TABLE:SUMMARYREPORTS", Report_1_Name="SystemSummary"
        )


def _ensure_sqlite(idf) -> None:
    objs = idf.idfobjects.get("OUTPUT:SQLITE", [])
    if objs:
        objs[0].Option_Type = "Simple"
    else:
        idf.newidfobject("OUTPUT:SQLITE", Option_Type="Simple")


def _set_run_period(idf, rp: dict) -> None:
    rps = idf.idfobjects.get("RUNPERIOD", [])
    if not rps:
        idf.newidfobject(
            "RUNPERIOD",
            Name="EcoLoop_Run",
            Begin_Month=rp["begin_month"],
            Begin_Day_of_Month=rp["begin_day"],
            End_Month=rp["end_month"],
            End_Day_of_Month=rp["end_day"],
        )
        return
    r = rps[0]
    r.Begin_Month = rp["begin_month"]
    r.Begin_Day_of_Month = rp["begin_day"]
    r.End_Month = rp["end_month"]
    r.End_Day_of_Month = rp["end_day"]


def snapshot_idf(
    baseline_idf: Path,
    out_path: Path,
    energyplus_dir: Path,
    setpoints: Dict[str, Tuple[float, float]],
) -> Path:
    """Write a snapshot .idf whose Schedule:Constant values reflect *setpoints*.

    *setpoints* maps zone -> (heating_sp, cooling_sp). Satisfies the
    "AI-modified .idf files generated at runtime" deliverable.
    """
    idf = _get_idf(baseline_idf, energyplus_dir)
    for zone, (h, c) in setpoints.items():
        h_name, c_name = sched_names(zone)
        for sc in idf.idfobjects.get("SCHEDULE:CONSTANT", []):
            if sc.Name.lower() == h_name.lower():
                sc.Hourly_Value = round(h, 2)
            elif sc.Name.lower() == c_name.lower():
                sc.Hourly_Value = round(c, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(out_path))
    return out_path
