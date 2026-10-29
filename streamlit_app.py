# -*- coding: utf-8 -*-
"""
Streamlit front-end for the Kia-style P2 hybrid with a 6-speed DCT.

    streamlit run streamlit_app.py

Physics, controllers and analysis: hev_p2dct_sim.py (uses component models from
hev_ips_sim.py).
"""
import io
import math
import os
import tempfile
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
import streamlit as st  # noqa: E402

import hev_ips_sim as common  # noqa: E402
import hev_p2dct_sim as sim  # noqa: E402

st.set_page_config(page_title="Kia P2 Hybrid · 6DCT Simulator", page_icon="🚙",
                   layout="wide", initial_sidebar_state="expanded")

RPM = common.RPM
DEFAULTS = sim.P2Params()
GEAR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# -----------------------------------------------------------------------------
# Parameter metadata
# -----------------------------------------------------------------------------
GROUPS = {
    "Vehicle body": [
        ("mass", "Mass (kerb + driver)", "kg"), ("Cd", "Drag coefficient", "-"),
        ("A", "Frontal area", "m²"), ("Crr", "Rolling resistance coeff.", "-"),
        ("r_wheel", "Tire rolling radius", "m"), ("J_wheels", "Wheel inertia (all four)", "kg·m²")],
    "6-speed DCT": [
        ("g1", "1st gear", "-"), ("g2", "2nd gear", "-"), ("g3", "3rd gear", "-"),
        ("g4", "4th gear", "-"), ("g5", "5th gear", "-"), ("g6", "6th gear", "-"),
        ("fd_odd_low", "Final drive, gears 1-4", "-"), ("fd_high", "Final drive, gears 5-6", "-"),
        ("gb_eta", "Load efficiency", "-"), ("gb_spin_Nm", "Spin drag at input", "N·m"),
        ("gb_visc", "Viscous drag at input", "N·m/(rad/s)"), ("input_J", "Input-shaft inertia", "kg·m²"),
        ("dct_Tin_max", "Max input torque", "N·m")],
    "Shift schedule (TCU)": [
        ("up12_lo", "1→2 light pedal", "km/h"), ("up12_hi", "1→2 full pedal", "km/h"),
        ("up23_lo", "2→3 light pedal", "km/h"), ("up23_hi", "2→3 full pedal", "km/h"),
        ("up34_lo", "3→4 light pedal", "km/h"), ("up34_hi", "3→4 full pedal", "km/h"),
        ("up45_lo", "4→5 light pedal", "km/h"), ("up45_hi", "4→5 full pedal", "km/h"),
        ("up56_lo", "5→6 light pedal", "km/h"), ("up56_hi", "5→6 full pedal", "km/h"),
        ("dn_hyst_kmh", "Downshift hysteresis", "km/h"), ("shift_min_s", "Min. time between shifts", "s"),
        ("shift_tau", "Speed-sync time constant", "s")],
    "Engine": [
        ("eng_disp_L", "Displacement", "L"), ("eng_Pmax_kW", "Max power", "kW"),
        ("eng_rpm_idle", "Idle speed", "rpm"), ("eng_rpm_max", "Max speed", "rpm"),
        ("eng_couple_rpm", "Min. coupling speed (K0)", "rpm"), ("eng_J", "Inertia", "kg·m²"),
        ("eng_eta_ind", "Peak indicated efficiency", "-"), ("eng_tau", "Torque response", "s")],
    "HSG (starter-generator)": [
        ("hsg_Pmax_kW", "Max power", "kW"), ("hsg_Tmax", "Max torque", "N·m"),
        ("hsg_ratio", "Belt ratio", "-"), ("hsg_rpm_max", "Max speed", "rpm"), ("hsg_J", "Rotor inertia", "kg·m²")],
    "P2 traction motor": [
        ("mot_Pmax_kW", "Max power", "kW"), ("mot_Tmax", "Max torque", "N·m"),
        ("mot_rpm_max", "Max speed", "rpm"), ("mot_J", "Rotor inertia", "kg·m²")],
    "Battery": [
        ("batt_cells", "Cells in series", "-"), ("batt_Ah", "Cell capacity", "Ah"),
        ("batt_Rcell_mOhm", "Cell resistance @25 °C", "mΩ"), ("batt_Pdis_kW", "Max discharge power", "kW"),
        ("batt_Pchg_kW", "Max charge power", "kW"), ("soc_init", "Initial SOC", "-"),
        ("dcdc_eff", "DC-DC efficiency (1 = none)", "-"), ("aux_kW", "Auxiliary load", "kW")],
    "Energy management (ECMS)": [
        ("soc_target", "SOC target", "-"), ("soc_ke_drop", "SOC target drop at 130 km/h", "-"),
        ("soc_low", "Low SOC (force engine)", "-"), ("ecms_s0", "Equivalence factor s₀", "-"),
        ("ecms_Kp", "SOC feedback Kp", "1/SOC"), ("ecms_Ki", "SOC feedback Ki", "1/(SOC·s)"),
        ("ecms_hyst_gps", "On/off hysteresis", "g/s"), ("ecms_Te_min_frac", "Min. engine load fraction", "-"),
        ("ev_vmax_kmh", "Max EV speed", "km/h"), ("eng_min_on_s", "Min. engine on-time", "s"),
        ("eng_min_off_s", "Min. engine off-time", "s"), ("eng_P_rate", "Engine torque ramp", "N·m/s"),
        ("idle_chg_kW", "Idle charging power", "kW"), ("idle_chg_rpm", "Idle charging speed", "rpm"),
        ("regen_fade_kmh", "Regen fade-out speed", "km/h")],
    "Driver": [
        ("drv_Kp", "Speed PI Kp (× mass)", "1/s"), ("drv_Ki", "Speed PI Ki (× mass)", "1/s²"),
        ("drv_preview", "Preview time", "s")],
    "Environment": [
        ("grade_pct", "Road grade", "%"), ("wind_kmh", "Head wind", "km/h"),
        ("T_amb_C", "Ambient temperature", "°C"), ("p_amb_kPa", "Ambient pressure", "kPa")],
    "Fuel": [
        ("fuel_LHV_MJkg", "Lower heating value", "MJ/kg"), ("fuel_rho", "Density", "kg/L"),
        ("co2_per_kg", "CO₂ per kg fuel", "kg/kg")],
    "Scenario & solver": [
        ("const_kmh", "Constant-speed target", "km/h"), ("const_hold_s", "Constant-speed hold", "s"),
        ("dt", "Time step", "s"), ("log_dt", "Logging interval", "s")],
}
_listed = {f for g in GROUPS.values() for f, _, _ in g}
_missing = [f.name for f in fields(sim.P2Params) if f.name not in _listed]
if _missing:
    GROUPS["Other"] = [(m, m, "") for m in _missing]

INT_FIELDS = {"batt_cells"}
LIMITS = {"dt": (0.005, 0.03), "log_dt": (0.02, 1.0), "soc_init": (0.2, 0.95),
          "soc_target": (0.3, 0.9), "gb_eta": (0.8, 1.0), "dcdc_eff": (0.8, 1.0)}


def _step_fmt(name, v):
    if name in INT_FIELDS:
        return 1.0, "%.0f"
    av = abs(v)
    if av >= 1000:
        return 100.0, "%.0f"
    if av >= 100:
        return 5.0, "%.0f"
    if av >= 10:
        return 1.0, "%.1f"
    if av >= 1:
        return 0.01, "%.3f"
    if av >= 0.1:
        return 0.005, "%.3f"
    if av == 0:
        return 1.0, "%.1f"
    return 0.0005, "%.4f"


def _key(name):
    return f"k_{name}"


def reset_defaults():
    for f in fields(sim.P2Params):
        st.session_state[_key(f.name)] = float(getattr(DEFAULTS, f.name))


def param_input(name, label, unit):
    d = float(getattr(DEFAULTS, name))
    step, fmt = _step_fmt(name, d)
    lo, hi = LIMITS.get(name, (None, None))
    if _key(name) not in st.session_state:
        st.session_state[_key(name)] = d
    return st.number_input(f"{label} [{unit}]" if unit and unit != "-" else label,
                           min_value=None if lo is None else float(lo),
                           max_value=None if hi is None else float(hi),
                           step=step, format=fmt, key=_key(name),
                           help=sim.PARAM_HELP.get(name) or common.PARAM_HELP.get(name))


# -----------------------------------------------------------------------------
# Computation & exports
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=24)
def run_cached(params: dict, scenario: str, csv_bytes: bytes):
    p = sim.P2Params(**params)
    scen = common.make_scenario(scenario, p, csv_bytes if csv_bytes else "")
    return sim.simulate(p, scen)


_EXPORTS = {}


def export_files(res, run_key):
    if run_key not in _EXPORTS:
        with tempfile.TemporaryDirectory() as d:
            out = {}
            for f in sim.export_results(res, d):
                with open(f, "rb") as fh:
                    out[os.path.basename(f)] = fh.read()
        while len(_EXPORTS) >= 6:
            _EXPORTS.pop(next(iter(_EXPORTS)))
        _EXPORTS[run_key] = out
    return _EXPORTS[run_key]


def export_names(res):
    base = res.scen.name.split()[0].replace("/", "-")
    return [f"kia_p2dct_{base}_timeseries.csv", f"kia_p2dct_{base}_energy_summary.csv",
            f"kia_p2dct_{base}_results.xlsx"]


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
PLOT_FONT = dict(family="Inter, DejaVu Sans, sans-serif", size=12, color=common.C_TXT2)


def _finish(fig, n_rows, height):
    fig.update_layout(height=height, hovermode="x unified", font=PLOT_FONT,
                      plot_bgcolor=common.C_SURF, paper_bgcolor="white",
                      margin=dict(l=60, r=20, t=40, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0, font=dict(size=11)))
    fig.update_xaxes(showgrid=True, gridcolor=common.C_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=common.C_GRID, zeroline=False)
    fig.update_xaxes(title_text="Time [s]", row=n_rows, col=1)
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=common.C_TXT); a.x = 0; a.xanchor = "left"
    return fig


MAX_PTS = 6000   # per trace; SVG traces this size draw in well under a second


def _thinned(res):
    """Log decimated for plotting (every k-th sample, k chosen from the record length)."""
    n = len(res.log["t"])
    k = max(1, int(math.ceil(n / MAX_PTS)))
    return {key: val[::k] for key, val in res.log.items()}


def results_figure(res):
    lg = _thinned(res)
    t = lg["t"]
    rows = [("Speed [km/h]", [("Reference", lg["v_ref"] * 3.6, common.C_REF, 2.4, None),
                              ("Vehicle", lg["v"] * 3.6, common.C_MG2, 1.5, None)]),
            ("Gear", [("Gear", lg["gear"], common.C_TXT2, 1.6, "hv")]),
            ("SOC [%]", [("SOC", lg["soc"] * 100, common.C_BAT, 1.8, None)]),
            ("Speed [rpm]", [("Engine", lg["w_e"] / RPM, common.C_ENG, 1.5, None),
                             ("Motor / DCT input", lg["w_i"] / RPM, common.C_MG2, 1.2, None)]),
            ("Torque [N·m]", [("Engine", lg["Te"], common.C_ENG, 1.5, None),
                              ("Motor", lg["Tm"], common.C_MG2, 1.2, None),
                              ("HSG", lg["T_hsg"], sim.C_HSG, 1.0, None)]),
            ("Power [kW]", [("Engine", lg["P_eng"] / 1e3, common.C_ENG, 1.5, None),
                            ("Motor (mech)", lg["P_mot"] / 1e3, common.C_MG2, 1.2, None),
                            ("Battery", lg["P_batt"] / 1e3, common.C_BAT, 1.2, None)]),
            ("Fuel [L]", [("Fuel used", lg["fuel_L"], common.C_ENG, 1.8, None)])]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=[r[0] for r in rows])
    shown = set()
    for i, (title, traces) in enumerate(rows, start=1):
        for name, y, col, w, shape in traces:
            line = dict(color=col, width=w)
            if shape:
                line["shape"] = shape
            fig.add_trace(go.Scatter(x=t, y=y, name=name, line=line, legendgroup=name,
                                       showlegend=name not in shown, hovertemplate="%{y:.2f}"),
                          row=i, col=1)
            shown.add(name)
    mode = lg["mode"].astype(int)
    shapes, start = [], 0
    for j in range(1, len(t) + 1):
        if j == len(t) or mode[j] != mode[start]:
            shapes.append(dict(type="rect", xref="x", yref="y domain", x0=float(t[start]),
                               x1=float(t[min(j, len(t) - 1)]), y0=0, y1=1, layer="below",
                               fillcolor=sim.MODE_COLORS[mode[start]], opacity=0.2, line_width=0))
            start = j
    fig.update_layout(shapes=shapes)
    for m, c in zip(sim.MODE_NAMES, sim.MODE_COLORS):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Mode: {m}",
                                 marker=dict(size=10, color=c, symbol="square", opacity=0.55)),
                      row=1, col=1)
    fig.update_yaxes(range=[0.5, 6.5], dtick=1, row=2, col=1)
    fig.add_hline(y=res.p.soc_target * 100, line=dict(color=common.C_REF, dash="dash", width=1), row=3, col=1)
    return _finish(fig, len(rows), 1150)


SIGNAL_INFO = {
    "v_ref": ("Reference speed", "km/h", 3.6), "v": ("Vehicle speed", "km/h", 3.6),
    "accel": ("Accelerator pedal", "-", 1), "brake": ("Brake pedal", "-", 1),
    "mode": ("Operating mode (0 stat., 1 EV, 2 HEV chg, 3 HEV assist, 4 idle chg)", "-", 1),
    "gear": ("Gear", "-", 1), "shifting": ("Shift in progress", "-", 1),
    "k0": ("Engine clutch K0 closed", "-", 1),
    "eng_state": ("Engine state (0 off, 1 start, 2 run, 3 charge, 4 stop)", "-", 1),
    "w_e": ("Engine speed", "rpm", 1 / RPM), "w_i": ("Motor / DCT input speed", "rpm", 1 / RPM),
    "Te": ("Engine torque", "N·m", 1), "Tm": ("Motor torque", "N·m", 1),
    "T_hsg": ("HSG torque", "N·m", 1), "T_fb": ("Friction brake torque", "N·m", 1),
    "Tc": ("DCT clutch torque while slipping", "N·m", 1),
    "P_eng": ("Engine power", "kW", 1e-3), "P_mot": ("Motor mech. power", "kW", 1e-3),
    "P_hsg": ("HSG mech. power", "kW", 1e-3), "P_mot_e": ("Motor elec. power", "kW", 1e-3),
    "P_hsg_e": ("HSG elec. power", "kW", 1e-3), "P_batt": ("Battery terminal power", "kW", 1e-3),
    "P_wheel": ("Wheel power", "kW", 1e-3), "soc": ("State of charge", "%", 100),
    "I_batt": ("Battery current", "A", 1), "V_batt": ("Battery voltage", "V", 1),
    "mdot": ("Fuel flow", "g/s", 1e3), "fuel_L": ("Fuel used", "L", 1), "dist": ("Distance", "km", 1e-3),
    "P_dis_lim": ("BMS discharge limit", "kW", 1e-3), "P_chg_lim": ("BMS charge limit", "kW", 1e-3),
    "ecms_s": ("ECMS equivalence factor", "-", 1), "eta_eng": ("Engine efficiency", "%", 100),
    "grade": ("Road grade", "%", 1),
}


def explorer_figure(res, names):
    lg = _thinned(res)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[f"{SIGNAL_INFO[n][0]} [{SIGNAL_INFO[n][1]}]" for n in names])
    for i, n in enumerate(names, start=1):
        fig.add_trace(go.Scatter(x=lg["t"], y=lg[n] * SIGNAL_INFO[n][2], name=SIGNAL_INFO[n][0],
                                   line=dict(color=common.C_MG2, width=1.4), showlegend=False,
                                   hovertemplate="%{y:.3g}"), row=i, col=1)
    return _finish(fig, len(names), max(260, 190 * len(names)))


def energy_figure(res):
    rows = [(it, v) for g, it, v in sim.energy_table(res) if g != "Source" and abs(v) > 1e-5]
    rows.sort(key=lambda r: r[1])
    cmap = {"Engine losses": common.C_ENG, "Motor losses": common.C_MG2, "HSG losses": sim.C_HSG,
            "Battery resistive losses": common.C_BAT, "Friction brakes": common.C_BRK}
    fig = go.Figure(go.Bar(
        x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
        marker=dict(color=[cmap.get(r[0], common.C_REF) for r in rows]),
        text=[f"{r[1]:.3f}" for r in rows], textposition="outside",
        hovertemplate="%{y}: %{x:.4f} kWh<extra></extra>"))
    lo = min([r[1] for r in rows] + [0]); hi = max([r[1] for r in rows] + [0])
    pad = 0.15 * (hi - lo or 1)
    fig.update_layout(height=520, font=PLOT_FONT, plot_bgcolor=common.C_SURF, paper_bgcolor="white",
                      margin=dict(l=10, r=40, t=10, b=40), xaxis_title="Energy [kWh]")
    fig.update_xaxes(showgrid=True, gridcolor=common.C_GRID, zeroline=True, zerolinecolor="#b9b8b2",
                     range=[lo - (pad if lo < 0 else 0), hi + pad])
    return fig


def draw_shift_map(ax, p, lg=None, i=None):
    ped = np.linspace(0, 1, 41)
    ups = np.array([sim.shift_speeds(p, a) for a in ped])
    for n in range(5):
        ax.plot(ups[:, n], ped, color=GEAR_COLORS[n + 1], lw=1.8)
        ax.plot(ups[:, n] - p.dn_hyst_kmh, ped, color=GEAR_COLORS[n + 1], lw=1.0, ls="--")
        ax.text(ups[-1, n], 1.02, f"{n + 1}→{n + 2}", fontsize=7.5, color=common.C_TXT2, ha="center")
    if lg is not None:
        m = lg["accel"] > 0.005
        for g in range(1, 7):
            s = m & (lg["gear"] == g)
            ax.scatter(lg["v"][s] * 3.6, lg["accel"][s], s=6, color=GEAR_COLORS[g - 1], alpha=0.3,
                       edgecolors="none", label=f"gear {g}")
        if i is not None:
            ax.scatter([lg["v"][i] * 3.6], [lg["accel"][i]], s=110, color="white",
                       edgecolors=common.C_TXT, linewidths=2, zorder=5)
    ax.set_xlim(0, max(150, float(np.max(ups)) + 10)); ax.set_ylim(0, 1.08)
    common._style_ax(ax, "Accelerator pedal [-]", "Vehicle speed [km/h]")
    ax.set_title("Shift map: upshift (solid) / downshift (dashed)", fontsize=9.5, color=common.C_TXT, loc="left")


def maps_figure(res):
    eng, mot, lg, p = res.eng, res.mot, res.log, res.p
    fig = plt.figure(figsize=(14, 9.6), facecolor="white")
    # engine BSFC
    ax = fig.add_axes([0.06, 0.52, 0.47, 0.43])
    rpm = np.linspace(p.eng_rpm_idle * 0.8, p.eng_rpm_max, 120)
    wv = rpm * RPM
    Tg = np.linspace(2, max(eng.tq_T) * 1.02, 100)
    W, T = np.meshgrid(wv, Tg)
    B = np.where(T <= eng.tmax_v(W), eng.bsfc_v(W, T), np.nan)
    lev = [215, 220, 225, 230, 240, 250, 260, 275, 300, 350, 400, 500, 700]
    cs = ax.contourf(rpm, Tg, B, levels=lev, cmap="Blues_r", extend="both", alpha=0.85)
    cl = ax.contour(rpm, Tg, B, levels=lev, colors="white", linewidths=0.5)
    ax.clabel(cl, fontsize=6.5, fmt="%d")
    ax.plot(np.array(eng.tq_w) / RPM, eng.tq_T, color=common.C_TXT, lw=1.6, label="Full load")
    run = lg["k0"] > 0.5
    ax.scatter(lg["w_e"][run] / RPM, lg["Te"][run], s=8, color=common.C_ENG, alpha=0.35,
               edgecolors="none", label="Coupled (parallel)")
    ch = lg["mode"] == 4
    if np.any(ch):
        ax.scatter(lg["w_e"][ch] / RPM, lg["Te"][ch], s=10, color=sim.C_HSG, alpha=0.5,
                   edgecolors="none", label="Idle charging")
    ax.axvline(p.eng_couple_rpm, color=common.C_REF, lw=0.8, ls=":")
    ax.set_xlim(rpm[0], rpm[-1]); ax.set_ylim(0, max(eng.tq_T) * 1.05)
    common._style_ax(ax, "Engine torque [N·m]", "Engine speed [rpm]")
    ax.set_title("Engine BSFC [g/kWh] with operating points", fontsize=9.5, color=common.C_TXT, loc="left")
    ax.legend(fontsize=7, loc="lower right", facecolor="white", edgecolor="none", labelcolor=common.C_TXT2)
    cb = fig.colorbar(cs, ax=ax, pad=0.01, fraction=0.04); cb.ax.tick_params(labelsize=7)
    # shift map
    ax = fig.add_axes([0.62, 0.52, 0.36, 0.43])
    draw_shift_map(ax, p, lg)
    # motor map
    ax = fig.add_axes([0.06, 0.06, 0.40, 0.36])
    wv = np.linspace(0, mot.wmax, 120)
    Tv = np.linspace(-mot.Tmax, mot.Tmax, 100)
    W, T = np.meshgrid(wv, Tv)
    Eff = np.where(np.abs(T) <= mot.tlim_v(W), mot.eff_map(W, T) * 100, np.nan)
    cs = ax.contourf(wv / RPM, Tv, Eff, levels=[70, 80, 85, 88, 90, 92, 94, 96, 98], cmap="Greens",
                     extend="min", alpha=0.85)
    ax.plot(wv / RPM, mot.tlim_v(wv), color=common.C_TXT, lw=1.2)
    ax.plot(wv / RPM, -mot.tlim_v(wv), color=common.C_TXT, lw=1.2)
    ax.scatter(lg["w_i"] / RPM, lg["Tm"], s=5, color=common.C_MG2, alpha=0.3, edgecolors="none")
    common._style_ax(ax, "Motor torque [N·m]", "Motor speed [rpm]")
    ax.set_title("P2 motor efficiency [%] with operating points", fontsize=9.5, color=common.C_TXT, loc="left")
    cb = fig.colorbar(cs, ax=ax, pad=0.01, fraction=0.05); cb.ax.tick_params(labelsize=7)
    # gear usage
    ax = fig.add_axes([0.55, 0.06, 0.18, 0.36])
    tt = np.array(res.gear_time)
    ax.bar(range(1, 7), tt, color=GEAR_COLORS, width=0.7)
    for g, v in enumerate(tt, 1):
        ax.text(g, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=common.C_TXT2)
    common._style_ax(ax, "Time [s]", "Gear")
    ax.set_title("Time in gear", fontsize=9.5, color=common.C_TXT, loc="left")
    # engine efficiency histogram
    ax = fig.add_axes([0.80, 0.06, 0.18, 0.36])
    ef = res.eng_eff_samples * 100
    if len(ef):
        bins = np.arange(0, 44, 2)
        cnt, _ = np.histogram(ef, bins=bins)
        ax.bar(bins[:-1] + 1, cnt * p.dt, width=1.7, color=common.C_ENG)
    common._style_ax(ax, "Time [s]", "Engine efficiency [%]")
    ax.set_title("Time at engine efficiency", fontsize=9.5, color=common.C_TXT, loc="left")
    return fig


@st.cache_data(show_spinner=False, max_entries=12)
def maps_png(_res, run_key: str):
    fig = maps_figure(_res)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def draw_power_flow_p2(ax, d):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax.cla(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = {"Engine": (0.09, 0.60, common.C_ENG), "HSG": (0.09, 0.92, sim.C_HSG),
             f"K0 {'closed' if d['k0'] else 'open'}": (0.29, 0.60, common.C_REF),
             "Motor": (0.48, 0.60, common.C_MG2), f"DCT · {d['gear']}": (0.69, 0.60, common.C_TXT2),
             "Wheels": (0.91, 0.60, common.C_TXT2), "Battery": (0.48, 0.12, common.C_BAT)}
    for name, (x, y, c) in boxes.items():
        ax.add_patch(FancyBboxPatch((x - 0.075, y - 0.055), 0.15, 0.11,
                                    boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=c, lw=1.6))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.5, color=common.C_TXT)

    def arrow(p0, p1, P, col, lp, label=None):
        if abs(P) < 0.2:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#dedcd6", lw=1.2, ls="--")
        else:
            a, b = (p0, p1) if P > 0 else (p1, p0)
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=col,
                                         lw=1.2 + 4.0 * min(abs(P) / 60.0, 1.0), shrinkA=0, shrinkB=0))
        ax.text(*lp, label if label else f"{abs(P):.1f} kW", fontsize=7, color=common.C_TXT2,
                ha="center", va="center")

    P_e_out = d["P_eng"] if d["k0"] else 0.0
    arrow((0.09, 0.865), (0.09, 0.655), -d["P_hsg"], sim.C_HSG, (0.02, 0.76))           # HSG <-> engine
    arrow((0.165, 0.60), (0.215, 0.60), P_e_out, common.C_ENG, (0.19, 0.69))
    arrow((0.365, 0.60), (0.405, 0.60), P_e_out, common.C_ENG, (0.385, 0.69))
    arrow((0.555, 0.60), (0.615, 0.60), P_e_out + d["P_mot"], common.C_TXT2, (0.585, 0.69))
    arrow((0.765, 0.60), (0.835, 0.60), d["P_wheel"], common.C_TXT2, (0.80, 0.69))
    arrow((0.48, 0.175), (0.48, 0.545), d["P_mot_e"], common.C_MG2, (0.56, 0.36),
          f"motor el\n{abs(d['P_mot_e']):.1f} kW")
    ch = sim.C_HSG if abs(d["P_hsg_e"]) > 0.2 else "#dedcd6"
    ax.plot([0.165, 0.25, 0.25], [0.92, 0.92, 0.12], color=ch, lw=1.0)
    arrow((0.25, 0.12), (0.405, 0.12), -d["P_hsg_e"], sim.C_HSG, (0.32, 0.30),
          f"HSG el\n{abs(d['P_hsg_e']):.1f} kW")
    ax.text(0.48, 0.0, f"Battery {d['P_batt']:+.1f} kW (+ = discharge)   SOC {d['soc'] * 100:.1f} %",
            ha="center", va="bottom", fontsize=7.5, color=common.C_TXT)
    ax.text(0.09, 0.47, f"fuel {d['mdot'] * 1000:.2f} g/s", ha="center", fontsize=7, color=common.C_TXT2)
    ax.text(0.0, 1.03, "Power flow", fontsize=9.5, color=common.C_TXT, transform=ax.transAxes)


def dashboard_figure(res, i):
    lg, p = res.log, res.p
    fig = plt.figure(figsize=(14, 7.6), facecolor="white")
    gw = 0.245
    axs = [fig.add_axes([0.01 + k * gw, 0.57, gw - 0.01, 0.42]) for k in range(4)]
    gs = [common.make_gauge(axs[0], 0, 180, "Vehicle speed", "km/h", 20, common.C_MG2),
          common.make_gauge(axs[1], 0, 7000, "Engine", "rpm", 1000, common.C_ENG),
          common.make_gauge(axs[2], 0, 7000, "Motor / DCT input", "rpm", 1000, common.C_MG2)]
    for g, val in zip(gs, (lg["v"][i] * 3.6, lg["w_e"][i] / RPM, lg["w_i"][i] / RPM)):
        common.set_gauge(g, val)
    ax = axs[3]; ax.axis("off")
    gear = int(lg["gear"][i])
    ax.text(0.5, 0.62, f"{gear}", ha="center", va="center", fontsize=64, weight="bold",
            color=GEAR_COLORS[gear - 1], transform=ax.transAxes)
    ax.text(0.5, 0.30, "SHIFTING" if lg["shifting"][i] else "gear", ha="center", fontsize=10,
            color=common.C_TXT2, transform=ax.transAxes)
    es = sim.ENG_STATES[int(lg["eng_state"][i])]
    ax.text(0.5, 0.15, f"engine {es} · K0 {'closed' if lg['k0'][i] else 'open'}", ha="center",
            fontsize=9, color=common.C_TXT, transform=ax.transAxes)
    ax.text(0.5, 0.03, f"ECMS s = {lg['ecms_s'][i]:.2f}", ha="center", fontsize=9,
            color=common.C_TXT2, transform=ax.transAxes)
    ax_sm = fig.add_axes([0.06, 0.07, 0.24, 0.40])
    draw_shift_map(ax_sm, p, None)
    ax_sm.scatter([lg["v"][i] * 3.6], [lg["accel"][i]], s=110, color=GEAR_COLORS[gear - 1],
                  edgecolors=common.C_TXT, linewidths=1.5, zorder=5)
    ax_pf = fig.add_axes([0.33, 0.04, 0.42, 0.45])
    draw_power_flow_p2(ax_pf, dict(
        k0=bool(lg["k0"][i]), gear=gear, P_eng=lg["P_eng"][i] / 1e3, P_mot=lg["P_mot"][i] / 1e3,
        P_hsg=lg["P_hsg"][i] / 1e3, P_mot_e=lg["P_mot_e"][i] / 1e3, P_hsg_e=lg["P_hsg_e"][i] / 1e3,
        P_wheel=lg["P_wheel"][i] / 1e3, P_batt=lg["P_batt"][i] / 1e3, soc=lg["soc"][i], mdot=lg["mdot"][i]))
    ax_tr = fig.add_axes([0.79, 0.30, 0.20, 0.17])
    m = lg["mode"].astype(int); s0 = 0
    for j in range(1, len(m) + 1):
        if j == len(m) or m[j] != m[s0]:
            ax_tr.axvspan(lg["t"][s0], lg["t"][min(j, len(m) - 1)], color=sim.MODE_COLORS[m[s0]],
                          alpha=0.25, lw=0)
            s0 = j
    ax_tr.plot(lg["t"], lg["v_ref"] * 3.6, color=common.C_REF, lw=1.2)
    ax_tr.plot(lg["t"], lg["v"] * 3.6, color=common.C_MG2, lw=0.9)
    ax_tr.axvline(lg["t"][i], color=common.C_TXT, lw=1)
    common._style_ax(ax_tr, "km/h"); ax_tr.set_xlim(lg["t"][0], lg["t"][-1])
    ax_tr.set_title("Speed trace & mode", fontsize=9, color=common.C_TXT, loc="left")
    ax_soc = fig.add_axes([0.79, 0.07, 0.20, 0.15])
    ax_soc.plot(lg["t"], lg["soc"] * 100, color=common.C_BAT, lw=1.2)
    ax_soc.axvline(lg["t"][i], color=common.C_TXT, lw=1)
    common._style_ax(ax_soc, "SOC %", "Time [s]"); ax_soc.set_xlim(lg["t"][0], lg["t"][-1])
    fig.text(0.79, 0.50, f"{sim.MODE_NAMES[int(lg['mode'][i])]}   t = {lg['t'][i]:.1f} s",
             fontsize=10.5, weight="bold", color=common.C_TXT)
    return fig


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox("Drive cycle / test", common.CYCLES, index=0, key="scenario")
    csv_bytes = b""
    if scenario == "Custom CSV":
        up = st.file_uploader("Cycle CSV: time [s], speed [km/h], optional grade [%]", type=["csv", "txt"])
        if up is not None:
            csv_bytes = up.getvalue()
        st.caption("Comma- or semicolon-separated; a header row is allowed.")
    elif scenario == "Constant speed":
        st.caption("Target speed and hold time are under *Scenario & solver*.")
    st.button("Reset all parameters to defaults", on_click=reset_defaults, width="stretch")
    with st.form("params"):
        st.header("Parameters")
        st.caption("Defaults: 2nd-generation Kia Niro HEV (Kia published specs).")
        with st.expander("Environment", expanded=True):
            for name, label, unit in GROUPS["Environment"]:
                param_input(name, label, unit)
        for gname, items in GROUPS.items():
            if gname == "Environment":
                continue
            with st.expander(gname, expanded=False):
                for name, label, unit in items:
                    param_input(name, label, unit)
        submitted = st.form_submit_button("▶ Run simulation", type="primary", width="stretch")

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
params = {f.name: float(st.session_state.get(_key(f.name), getattr(DEFAULTS, f.name)))
          for f in fields(sim.P2Params)}
run_key = repr((sorted(params.items()), scenario, hash(csv_bytes)))

st.title("Kia P2 Hybrid · 6-speed DCT Simulator")
st.caption("Parallel hybrid in Kia's layout: engine → engine clutch K0 → motor → 6-speed dual-clutch "
           "transmission → wheels, with a belt starter-generator (HSG). Structure follows MathWorks' "
           "HEV P2 Reference Application (ECMS energy management, speed/load shift schedule).")

need_run = submitted or "res" not in st.session_state or st.session_state.get("scenario_ran") != scenario
if need_run:
    if scenario == "Custom CSV" and not csv_bytes:
        st.info("Upload a cycle CSV in the sidebar to run the custom scenario.")
        st.stop()
    try:
        with st.spinner(f"Simulating {scenario} …"):
            res = run_cached(params, scenario, csv_bytes)
    except Exception as e:  # noqa
        st.error(f"Simulation failed: {e}")
        st.stop()
    st.session_state.res = res
    st.session_state.run_key = run_key
    st.session_state.scenario_ran = scenario
    hist = st.session_state.setdefault("history", [])
    if not hist or hist[-1]["_key"] != run_key:
        changed = {k: v for k, v in params.items() if v != float(getattr(DEFAULTS, k))}
        s = res.summary
        hist.append({"_key": run_key, "Run": len(hist) + 1, "Scenario": scenario,
                     "Changed parameters": ", ".join(f"{k}={v:g}" for k, v in changed.items()) or "defaults",
                     "L/100 km": round(s["fuel_L100km"], 3),
                     "L/100 km (SOC-corr.)": round(s["fuel_L100km_soc_corrected"], 3),
                     "CO₂ g/km": round(s["co2_g_km"], 1), "SOC end": round(s["soc_end"], 3),
                     "Engine on %": round(s["engine_on_pct"], 1), "Starts": s["engine_starts"],
                     "Shifts": s["shifts"], "EV %": round(s["ev_pct"], 1),
                     "Regen kWh": round(s["regen_recovered_kWh"], 3),
                     "0-100 s": None if s.get("t_0_100_s") is None else round(s["t_0_100_s"], 2)})

res = st.session_state.res
s = res.summary
if st.session_state.get("run_key") != run_key and not submitted:
    st.warning("Parameters changed since the last run — press **Run simulation** to update.")

c = st.columns(6)
c[0].metric("Fuel [L/100 km]", f"{s['fuel_L100km']:.2f}",
            help=f"{s['fuel_L']:.3f} L over {s['distance_km']:.2f} km ({s['mpg_US']:.1f} mpg US)")
c[1].metric("SOC-corr. [L/100 km]", f"{s['fuel_L100km_soc_corrected']:.2f}",
            help="Net battery energy valued at the average engine × generator efficiency")
c[2].metric("CO₂ [g/km]", f"{s['co2_g_km']:.0f}")
c[3].metric("SOC", f"{s['soc_end'] * 100:.1f} %", f"{(s['soc_end'] - s['soc_start']) * 100:+.1f} pts")
c[4].metric("Engine on [% time]", f"{s['engine_on_pct']:.0f} %",
            help=f"{s['engine_starts']} starts, average efficiency {s['engine_avg_eff_pct']:.1f} %")
if s.get("t_0_100_s") is not None:
    c[5].metric("0–100 km/h [s]", f"{s['t_0_100_s']:.1f}",
                help=f"0–130 km/h: {res.t130:.1f} s" if res.t130 else None)
else:
    c[5].metric("Shifts", f"{s['shifts']}", help=f"EV driving {s['ev_pct']:.0f} % of time")
st.caption(f"{s['scenario']} · {s['distance_km']:.2f} km in {s['duration_s']:.0f} s · "
           f"EV {s['ev_pct']:.0f} % of time · speed error RMS {s['speed_err_rms_kmh']:.2f} km/h "
           f"(max {s['speed_err_max_kmh']:.2f}) · regen {s['regen_recovered_kWh']:.3f} kWh · "
           f"energy-balance error {s['energy_balance_err_pct']:+.3f} %")

tabs = st.tabs(["Results", "Maps & shifting", "Energy", "Dashboard", "Compare runs", "Model"])

with tabs[0]:
    st.plotly_chart(results_figure(res), width="stretch")
    with st.expander("Signal explorer"):
        pick = st.multiselect("Signals", list(SIGNAL_INFO), default=["accel", "ecms_s", "Tc"],
                              format_func=lambda n: SIGNAL_INFO[n][0])
        if pick:
            st.plotly_chart(explorer_figure(res, pick), width="stretch")
    st.markdown("**Downloads**")
    mimes = {".csv": "text/csv",
             ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    names = export_names(res)
    cols = st.columns(len(names))
    rk = st.session_state.run_key
    for col, fn in zip(cols, names):
        col.download_button(f"⬇ {fn}", (lambda fn=fn, r=res, k=rk: export_files(r, k)[fn]),
                            file_name=fn, mime=mimes[os.path.splitext(fn)[1]], width="stretch")

with tabs[1]:
    st.caption("Engine BSFC map with coupled and idle-charging operating points; TCU shift map with the "
               "points driven in each gear; motor efficiency map; time in gear; engine efficiency histogram.")
    st.image(maps_png(res, st.session_state.run_key), width="stretch")

with tabs[2]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Where the energy went")
        st.plotly_chart(energy_figure(res), width="stretch")
    with right:
        st.subheader("Sources")
        src = [(it, v) for g, it, v in sim.energy_table(res) if g == "Source"]
        st.dataframe(pd.DataFrame(src, columns=["Item", "kWh"]).style.format({"kWh": "{:.3f}"}),
                     hide_index=True, width="stretch")
        st.subheader("Component efficiencies")
        st.dataframe(pd.DataFrame(sim.efficiency_table(res), columns=["Efficiency", "%"])
                     .style.format({"%": "{:.1f}"}, na_rep="–"), hide_index=True, width="stretch")
        st.subheader("Time in mode")
        tot = sum(res.mode_time) or 1
        st.dataframe(pd.DataFrame({"Mode": sim.MODE_NAMES, "%": [100 * x / tot for x in res.mode_time]})
                     .style.format({"%": "{:.1f}"}), hide_index=True, width="stretch")
    st.caption(f"Energy-balance closure error: {s['energy_balance_err_pct']:+.3f} % of throughput "
               "(fuel + battery = losses + road loads + Δ kinetic energy).")

with tabs[3]:
    tmax = float(res.log["t"][-1])
    st.session_state.setdefault("dash_t", min(60.0, tmax))
    st.session_state.setdefault("playing", False)
    if st.session_state.dash_t > tmax:
        st.session_state.dash_t = 0.0

    def _toggle_play():
        st.session_state.playing = not st.session_state.playing

    cc = st.columns([1, 1, 4])
    cc[0].button("⏸ Pause" if st.session_state.playing else "▶ Play", on_click=_toggle_play, width="stretch")
    cc[1].selectbox("Speed", [1.0, 2.0, 5.0, 10.0], index=1, key="play_step",
                    format_func=lambda x: f"{x:g} s / frame", label_visibility="collapsed")

    @st.fragment(run_every=0.6 if st.session_state.playing else None)
    def dashboard():
        lg = st.session_state.res.log
        tm = float(lg["t"][-1])
        if st.session_state.playing:
            nt = st.session_state.dash_t + float(st.session_state.play_step)
            st.session_state.dash_t = nt if nt <= tm else 0.0
        st.slider("Time [s]", 0.0, tm, step=0.1, key="dash_t")
        i = int(min(np.searchsorted(lg["t"], st.session_state.dash_t), len(lg["t"]) - 1))
        f = dashboard_figure(st.session_state.res, i)
        st.pyplot(f, clear_figure=True)
        plt.close(f)

    dashboard()

with tabs[4]:
    hist = st.session_state.get("history", [])
    st.caption("Every distinct run in this session, with the parameters that differ from the defaults.")
    if hist:
        df = pd.DataFrame(hist).drop(columns="_key")
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇ runs.csv", df.to_csv(index=False).encode(), "kia_p2dct_runs.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

with tabs[5]:
    p = res.p
    rat = sim.gear_ratios(p)
    st.markdown("""
**Structure** (as in the Simulink P2 reference application): *Scenarios* → *Driver* → *Controllers*
(VCU with ECMS, ECU, TCU) → *Vehicle* (engine + HSG, engine clutch K0, P2 motor, 6-speed DCT, battery,
longitudinal body) → *Visualization / energy analysis*.

**Overall ratios** (gear × final drive):
""")
    st.dataframe(pd.DataFrame({"Gear": range(1, 7), "Gear ratio": [p.g1, p.g2, p.g3, p.g4, p.g5, p.g6],
                               "Final drive": [p.fd_odd_low] * 4 + [p.fd_high] * 2,
                               "Overall": [round(r, 3) for r in rat],
                               "km/h per 1000 rpm": [round(1000 * RPM / r * p.r_wheel * 3.6, 1) for r in rat]}),
                 hide_index=True)
    st.markdown("**Plant**: three rotating bodies — engine (+HSG through the belt), DCT input (motor), "
                "vehicle — joined by K0 and the active DCT clutch, each locked or slipping:")
    st.latex(r"J_E\dot\omega_e = T_e + G_h T_{hsg} - T_{K0},\qquad "
             r"J_I\dot\omega_i = T_m - T_{gb} + T_{K0} - T_c,\qquad "
             r"J_V\dot\omega_w = R\,T_c - (F_{aero}+F_{roll}+F_{grade})\,r_w - T_{brk}")
    st.markdown("With a clutch locked its two bodies merge (e.g. K0 and DCT both locked: "
                "$\\left[(J_E+J_I)R^2+J_V\\right]\\dot\\omega_w = R(T_e+T_m-T_{gb}) + T_V$).")
    st.markdown("**ECMS** (every 0.1 s) chooses EV or the engine torque that minimises the equivalent fuel rate")
    st.latex(r"J(T_e)=\dot m_f(\omega_i,T_e)+s\,\frac{P_{batt,chem}(T_m)}{H_{LHV}},\qquad T_m=T_{in,req}-T_e,"
             r"\qquad s = s_0 + K_p\,(SOC^*(v)-SOC) + K_i\!\int(SOC^*-SOC)\,dt")
    st.markdown("""
with $SOC^*(v) = SOC_{target} - \\Delta SOC_{KE}\\,(v/130\\,\\text{km/h})^2$ leaving room for regeneration,
an on/off hysteresis and minimum on/off times, and the motor/battery/engine limits as constraints.

| Mode | When | What happens |
|---|---|---|
| Stationary | at rest | engine off; HSG + engine charge if SOC is low |
| EV | ECMS prefers electric, all braking, below engine coupling speed | K0 open, motor drives through the DCT; series regenerative braking, friction brakes cover the rest |
| HEV · motor charging | K0 closed, engine above the load request | load-point shifting up: the motor generates |
| HEV · motor assist | K0 closed, engine below the request | motor adds torque (up to the DCT input limit) |
| Idle charging | vehicle too slow to couple the engine and SOC low | engine runs decoupled, HSG charges the battery |

**Engine start**: HSG cranks, the ECU synchronises engine speed to the motor speed, K0 closes.
**Shifts**: clutch-to-clutch with an inertia phase — the clutch whose slip sign matches the demand carries
the torque (capped to what the input side can react), the motor synchronises the input speed and the
on-coming clutch locks; slip energy is booked as a loss.

Component maps are analytic approximations (Willans-type engine, loss-coefficient motor maps), so absolute
fuel figures are indicative.
""")
