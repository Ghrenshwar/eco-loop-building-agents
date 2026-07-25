"""PMV / PPD comfort computation via ``pythermalcomfort``.

We compute comfort ourselves from sensor values rather than relying on the .idf
having Fanger People objects configured. ``pythermalcomfort`` changed its
public API across releases (the ``pmv_ppd`` return shape and the ``standard``
argument differ between 2.x lines), so this module probes what is installed and
adapts, and falls back to a self-contained ISO 7730 implementation if the
import or call signature is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ComfortResult:
    pmv: float
    ppd: float


def clo_for_season(day_of_year: int, summer_clo: float = 0.5, winter_clo: float = 1.0) -> float:
    """Rough seasonal clothing insulation (N. hemisphere): summer ~ May-Sep."""
    return summer_clo if 120 <= day_of_year <= 273 else winter_clo


# --------------------------------------------------------------------------- #
# pythermalcomfort adapter (probed once, cached)
# --------------------------------------------------------------------------- #

_PMV_FN = None          # resolved callable: (tdb, tr, vr, rh, met, clo) -> ComfortResult
_PROBED = False


def _probe_pythermalcomfort():
    """Return an adapter callable using whatever pythermalcomfort exposes.

    Handles the API rename across releases: v4.x exposes ``pmv_ppd_iso`` (which
    returns a ``PMVPPD`` object with ``.pmv``/``.ppd``); v2.x exposed
    ``pmv_ppd`` (returning a dict / namedtuple). We probe for whichever exists.
    """
    iso_fn = legacy_fn = None
    try:
        from pythermalcomfort.models import pmv_ppd_iso as iso_fn  # type: ignore  # noqa: F401
    except Exception:
        iso_fn = None
    if iso_fn is None:
        try:
            from pythermalcomfort.models import pmv_ppd as legacy_fn  # type: ignore  # noqa: F401
        except Exception:
            legacy_fn = None
    if iso_fn is None and legacy_fn is None:
        return None

    def _coerce(pmv, ppd) -> ComfortResult:
        # Values may be numpy scalars/0-d arrays; float() handles both.
        return ComfortResult(pmv=float(pmv), ppd=float(ppd))

    def _call(tdb, tr, vr, rh, met, clo):
        if iso_fn is not None:
            # v4.x: ISO 7730 model, scalars in -> PMVPPD object out.
            res = iso_fn(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, limit_inputs=False)
            return _coerce(res.pmv, res.ppd)
        # v2.x fallback.
        try:
            res = legacy_fn(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ISO")
        except TypeError:
            res = legacy_fn(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo)
        if isinstance(res, dict):
            return _coerce(res["pmv"], res["ppd"])
        pmv = getattr(res, "pmv", None)
        ppd = getattr(res, "ppd", None)
        if pmv is None:
            pmv, ppd = res[0], res[1]
        return _coerce(pmv, ppd)

    return _call


def _resolve_fn():
    global _PMV_FN, _PROBED
    if not _PROBED:
        _PMV_FN = _probe_pythermalcomfort()
        _PROBED = True
    return _PMV_FN


# --------------------------------------------------------------------------- #
# Self-contained ISO 7730 fallback (used only if the library is unavailable)
# --------------------------------------------------------------------------- #


def _fanger_iso7730(tdb, tr, vr, rh, met, clo) -> ComfortResult:
    """Reference Fanger PMV/PPD (ISO 7730 Annex D iterative surface-temp solve)."""
    # Water-vapour partial pressure (Pa) via the ISO 7730 saturation formula
    # FNPS(T)=exp(16.6536 - 4030.183/(T+235)) [kPa]; *10*RH% converts to Pa.
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (tdb + 235.0))
    m = met * 58.15
    w = 0.0
    mw = m - w
    icl = 0.155 * clo
    fcl = 1.05 + 0.645 * icl if icl > 0.078 else 1.0 + 1.29 * icl
    hcf = 12.1 * math.sqrt(max(vr, 0.0))
    taa = tdb + 273.0
    tra = tr + 273.0
    tcla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = xn
    eps = 1e-5
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= eps:
            break
    tcl = 100.0 * xn - 273.0

    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)
    hl5 = 3.96 * fcl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - tdb)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-(0.03353 * pmv**4 + 0.2179 * pmv**2))
    return ComfortResult(pmv=round(pmv, 3), ppd=round(ppd, 2))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def compute_pmv(
    air_temp_c: float,
    rel_humidity_pct: float,
    mean_radiant_c: Optional[float] = None,
    air_speed_ms: float = 0.1,
    met_rate: float = 1.1,
    clo: float = 0.5,
) -> ComfortResult:
    """Compute PMV/PPD from sensor values.

    ``mean_radiant_c`` defaults to ``air_temp_c`` when a radiant sensor is
    unavailable (a standard approximation for lightly-loaded interior zones).
    """
    tr = air_temp_c if mean_radiant_c is None else mean_radiant_c
    rh = min(max(rel_humidity_pct, 0.0), 100.0)
    fn = _resolve_fn()
    if fn is not None:
        try:
            res = fn(air_temp_c, tr, air_speed_ms, rh, met_rate, clo)
            # Guard against NaN from out-of-range inputs.
            if not (math.isnan(res.pmv) or math.isnan(res.ppd)):
                return res
        except Exception:
            pass
    return _fanger_iso7730(air_temp_c, tr, air_speed_ms, rh, met_rate, clo)


def in_band(pmv: float, band: tuple[float, float] = (-0.5, 0.5)) -> bool:
    return band[0] <= pmv <= band[1]
