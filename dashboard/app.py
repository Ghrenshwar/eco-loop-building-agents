"""Streamlit dashboard: baseline vs AI closed-loop comparison.

Reads the recorded run outputs (telemetry Parquet/CSV, decisions JSONL,
summary.json) and shows: headline % kWh saved, cumulative energy curves,
per-zone temperature traces with the comfort band overlaid, PMV distribution,
unmet-hours comparison, and cost & carbon savings. The underlying comparison is
downloadable as CSV.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the src package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecoloop.config import load_config, load_targets  # noqa: E402

st.set_page_config(page_title="EcoLoop — Baseline vs AI", layout="wide")

cfg = load_config()
tgt = load_targets()
OUT = cfg.paths.output_dir


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_run(run: str) -> pd.DataFrame:
    d = OUT / run
    pq, csv = d / "telemetry.parquet", d / "telemetry.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    p = OUT / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_decisions(run: str) -> pd.DataFrame:
    p = OUT / run / "decisions.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def cumulative_kwh(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    per = df.drop_duplicates(subset=["sim_time_s"]).sort_values("sim_time_s").copy()
    per["kwh"] = per["facility_elec_j"] / 3.6e6
    per["cum_kwh"] = per["kwh"].cumsum()
    per["hours"] = per["sim_time_s"] / 3600.0
    return per


# --------------------------------------------------------------------------- #
# Header + headline metric
# --------------------------------------------------------------------------- #
st.title("EcoLoop Building Agents — Baseline vs AI")
st.caption(
    "An EnergyPlus digital twin supervised by a local LLM over MCP. "
    "Identical building, weather, and run period; only the control differs."
)

summary = load_summary()
base_df = load_run("baseline")
ai_df = load_run("ai")

if not summary and base_df.empty and ai_df.empty:
    st.warning(
        "No run outputs found yet. Run the pipeline first:\n\n"
        "```\npython -m ecoloop.pipeline.run_baseline --smoke\n"
        "python -m ecoloop.pipeline.run_ai --smoke\n"
        "python -m ecoloop.pipeline.compare\n```"
    )
    st.stop()

if summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("kWh saved", f"{summary.get('kwh_reduction_pct', 0):.1f}%")
    c2.metric("Cost saved", f"${summary.get('cost_savings_usd', 0):,.2f}")
    c3.metric("Carbon saved", f"{summary.get('carbon_savings_kgco2', 0):,.1f} kg")
    ai_ok = summary.get("ai", {}).get("occupied_in_band_pct", 0)
    c4.metric("AI comfort in band", f"{ai_ok:.1f}%")
    verdict = summary.get("verdict", "")
    (st.success if verdict.startswith("PASS") else st.info)(verdict)


# --------------------------------------------------------------------------- #
# Cumulative energy
# --------------------------------------------------------------------------- #
st.subheader("Cumulative facility electricity")
base_cum = cumulative_kwh(base_df)
ai_cum = cumulative_kwh(ai_df)
fig = go.Figure()
if not base_cum.empty:
    fig.add_trace(go.Scatter(x=base_cum["hours"], y=base_cum["cum_kwh"],
                             name="Baseline", line=dict(color="#888")))
if not ai_cum.empty:
    fig.add_trace(go.Scatter(x=ai_cum["hours"], y=ai_cum["cum_kwh"],
                             name="AI", line=dict(color="#2e7d32")))
fig.update_layout(xaxis_title="Simulation hour", yaxis_title="Cumulative kWh",
                  height=380, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Per-zone temperature traces with comfort context
# --------------------------------------------------------------------------- #
st.subheader("Per-zone air temperature (AI run)")
if not ai_df.empty:
    zones = sorted(ai_df["zone"].unique())
    sel = st.multiselect("Zones", zones, default=zones[: min(3, len(zones))])
    tfig = go.Figure()
    for z in sel:
        zd = ai_df[ai_df["zone"] == z].sort_values("sim_time_s")
        tfig.add_trace(go.Scatter(x=zd["sim_time_s"] / 3600.0, y=zd["air_temp_c"], name=z))
    tfig.update_layout(xaxis_title="Simulation hour", yaxis_title="Air temp (C)", height=360)
    st.plotly_chart(tfig, use_container_width=True)


# --------------------------------------------------------------------------- #
# PMV band overlay (comfort held?)
# --------------------------------------------------------------------------- #
st.subheader("Comfort: PMV distribution over occupied hours")
lo, hi = tgt.comfort.pmv_band
pcol1, pcol2 = st.columns(2)
for col, df, label, color in ((pcol1, base_df, "Baseline", "#888"),
                              (pcol2, ai_df, "AI", "#2e7d32")):
    with col:
        if df.empty:
            st.info(f"No {label} data")
            continue
        occ = df[df["occupancy"] > 0]
        pfig = go.Figure()
        pfig.add_trace(go.Histogram(x=occ["pmv"], nbinsx=40, marker_color=color, name=label))
        pfig.add_vrect(x0=lo, x1=hi, fillcolor="green", opacity=0.12, line_width=0,
                       annotation_text="comfort band")
        in_band = ((occ["pmv"] >= lo) & (occ["pmv"] <= hi)).mean() * 100 if len(occ) else 0
        pfig.update_layout(title=f"{label} — {in_band:.1f}% in band",
                           xaxis_title="PMV", yaxis_title="occupied timesteps", height=320)
        st.plotly_chart(pfig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Unmet hours + cost/carbon table
# --------------------------------------------------------------------------- #
if summary:
    st.subheader("Unmet-load hours & economics")
    b, a = summary.get("baseline", {}), summary.get("ai", {})
    ucol1, ucol2 = st.columns([2, 3])
    with ucol1:
        ufig = go.Figure()
        cats = ["Heating unmet h", "Cooling unmet h"]
        ufig.add_trace(go.Bar(name="Baseline", x=cats,
                              y=[b.get("unmet_heating_hours", 0), b.get("unmet_cooling_hours", 0)],
                              marker_color="#888"))
        ufig.add_trace(go.Bar(name="AI", x=cats,
                              y=[a.get("unmet_heating_hours", 0), a.get("unmet_cooling_hours", 0)],
                              marker_color="#2e7d32"))
        ufig.update_layout(barmode="group", height=320)
        st.plotly_chart(ufig, use_container_width=True)
    with ucol2:
        table = pd.DataFrame(
            {
                "metric": ["total_kwh", "cost_usd", "carbon_kgco2",
                           "occupied_in_band_pct", "pmv_mean_occupied"],
                "baseline": [b.get(k) for k in
                             ["total_kwh", "cost_usd", "carbon_kgco2",
                              "occupied_in_band_pct", "pmv_mean_occupied"]],
                "ai": [a.get(k) for k in
                       ["total_kwh", "cost_usd", "carbon_kgco2",
                        "occupied_in_band_pct", "pmv_mean_occupied"]],
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Agent decision log (proof the LLM actually drove control)
# --------------------------------------------------------------------------- #
st.subheader("Supervisor decision log (AI)")
dec = load_decisions("ai")
if dec.empty:
    st.info("No decisions recorded yet.")
else:
    n_fallback = int(dec.get("fallback", pd.Series(dtype=bool)).sum())
    n_correction = int(dec.get("self_correction", pd.Series(dtype=bool)).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Decisions", len(dec))
    m2.metric("Self-corrections", n_correction)
    m3.metric("Safety fallbacks", n_fallback)
    show_cols = [c for c in ["hour", "version", "fallback", "self_correction",
                             "rationale", "tool_calls", "latency_s"] if c in dec.columns]
    st.dataframe(dec[show_cols], use_container_width=True, hide_index=True, height=300)


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #
comp_csv = OUT / "comparison.csv"
if comp_csv.exists():
    st.download_button("Download comparison.csv", comp_csv.read_bytes(),
                       file_name="comparison.csv", mime="text/csv")
