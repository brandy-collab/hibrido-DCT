#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEV Input Power-Split Simulator
===============================

A Python re-creation of the structure of MathWorks' "HEV Input Power-Split
Reference Application" (Powertrain Blockset), for a Toyota Prius-like
full hybrid (engine + MG1 on a planetary power-split device, MG2 on the ring).

Model structure (mirrors the Simulink reference application):

  Scenarios        drive cycle source (WLTC class 3b, UDDS/FTP-72, HWFET, US06,
                   NEDC), wide-open-throttle test, constant speed, custom CSV
  Environment      road grade, head wind, ambient temperature and pressure
  Driver Commands  longitudinal driver (feed-forward + PI with preview)
                   -> accelerator and brake pedal
  Controllers      VCU: pedal -> wheel torque request, regenerative braking
                        blended with friction brakes, virtual BMS (SOC-dependent
                        charge/discharge power limits), rule-based mode logic
                        (Stationary / EV / HEV charge-sustaining /
                        HEV charge-depleting), engine off/start/run logic,
                        optimal-BSFC engine operating line, MG1 PI engine-speed
                        regulation, MG2 power management
                   ECU: engine torque command (first-order torque response)
  Vehicle          2-DOF power-split dynamics (carrier + ring), mapped SI engine
                   (Willans-type fuel model), MG1/MG2 with loss maps,
                   Li-ion pack (OCV(SOC), temperature-dependent resistance),
                   DC-DC converter, drivetrain spin losses, longitudinal body
  Visualization    speed tracking, SOC, machine speeds/torques/powers, fuel,
                   engine BSFC map with operating points, motor efficiency maps,
                   energy analysis (component- and system-level, closure check),
                   animated dashboard with gauges, lever diagram and power flow

Default parameters are representative of a 3rd-generation Prius (1.8 L Atkinson
engine 73 kW, MG1 42 kW, MG2 60 kW, sun 30 / ring 78 teeth, MG2 reduction 2.636,
final drive 3.267). The component maps are analytic approximations, not
measured data -- treat absolute fuel figures as indicative.

Usage
-----
  python hev_ips_sim.py                       # GUI
  python hev_ips_sim.py --cycle UDDS --no-gui # headless run, prints summary
  python hev_ips_sim.py --cycle WLTC --no-gui --export out/
  python hev_ips_sim.py --set mass=1600 --set soc_init=0.5
  python hev_ips_sim.py --list-params

Requires numpy and matplotlib (openpyxl optional, for .xlsx export).
"""

import argparse
import base64
import bisect
import csv
import math
import os
import sys
import zlib
from dataclasses import dataclass, fields, asdict

import numpy as np

G = 9.81
RPM = math.pi / 30.0          # rpm -> rad/s

# =============================================================================
# Palette (categorical slots in fixed order; text never wears series colour)
# =============================================================================
C_ENG = "#eb6834"     # engine          (orange)
C_MG2 = "#2a78d6"     # MG2 / traction  (blue)
C_MG1 = "#6250d6"     # MG1 / generator (violet)
C_BAT = "#1baf7a"     # battery         (aqua)
C_BRK = "#e34948"     # friction brake  (red)
C_REF = "#8a8984"     # reference / neutral
C_TXT = "#0b0b0b"
C_TXT2 = "#52514e"
C_GRID = "#e4e3df"
C_SURF = "#fcfcfb"
MODE_NAMES = ["Stationary", "EV", "HEV-CS", "HEV-CD"]
MODE_COLORS = ["#c9c8c2", "#1baf7a", "#2a78d6", "#eb6834"]

# =============================================================================
# Drive cycle data (1 Hz, speed in 0.1 km/h, zlib+base64)
#   WLTC class 3b, UDDS (FTP-72), HWFET, US06 -- from NREL FASTSim resources
# =============================================================================
_CYCLE_DATA = {
    "WLTC": (
        "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"
    ),
    "UDDS": (
        "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"
    ),
    "HWFET": (
        "eNqVVluS4zAIvJA+JAQC7n+xbR5ONJmZ3dpKdcWWEYKmwZ5j4rdpqI+151hGg/YepDb2krFljW1n"
        "MK554x/3fHSw7cHOQ6YOWTaEfMgGGBCsHQJkiAbwXA/urdZF2ma+bWwBG8C64zp8u/wCHWdOwMah"
        "NQ7CP8zjwO85ZxzkcnwOnQEbujZwhsJWCdfEAO437rcMZdhx/MNW8Bx+VPBccI8Y9az6j/tclwtt"
        "99jeNuEv/fLlX9oW54APVVAPbhT5q+FZxO3U/4B54bWOfR52eqH3gTu18Ad75fKf50T8cR/rq89c"
        "F+jCKntwcoJn7D+o+wFXh6zhtcbBN+wQ/0G9dAbPs/lGTCu49+Zci2toJLn4xvP5iGVVvBGLUHEY"
        "e9MHVe3Iq5ahA5yfdUfOCbVLAzh7UuXjVLZutZ6xWse4Ku6p+fyA1wL3vo+9oa3QIfojzwz9hQ4T"
        "/ubGtPfJx7l+6VDf+kjNPXVr7T146evR4cVj2vAPNX3ApbUEl1ZCM4/WbqTutGxTR9zndy2iZyLm"
        "zV0LrVpETtFrazU3XhyEVqJXJ9csQB7C6HXoSMC9QDcMfhnPGLkz5gPjPI7ZMIEdM2Tn/tQfhzbh"
        "L9aQR54TvTDl4pZbd6s1o93rdPW6dR9q5myI2RCLIVbDHuM9DPwa9GmwMbNhqKGj9o48fSngw4kA"
        "XGOGOuanQ6eOs5zjH3Y4z2UW4jrXdtlsLfvc134QuyOG9J1naJ93rnP/hbCDnwkfM/7Xhd057LZ7"
        "cuDr7MdH2J+PPWEvV74/geo5eSFt6YOf4AX+oV/Hu8GhswKuxWsddSqbhztp7lb7CL/2ncdv6HMT"
        "ekH+kcNPe+3C7WMWL6+6PVyv1I1hfqeGLLQEXRn/AvkF+sNe6NSgWcw/Q78YeruAdcyCFzBP/4q0"
        "o9Y5XXG07lP7OAd9mnlgnpmf7of5oa//wXyhOJI645OnV36dV/QkNGLo4exRaMQwF7JnoQlDbQy1"
        "+Yp7DXt29DquoX2D1rP3EVO+Z43fs/LLDO3Z+HovX+/p1ztYP95j1nP52veasVRzKOcqNbjfBX7N"
        "JXCAPsrcOHJGnmdftZPSQXCWNaLiLuoZdugbi3dW5MT2nuPP90/Mw9M5W31jhH3yEvzkPJzlJ3mc"
        "tR6zMkHFZdgFv4+tdLxZG2p0PLludZ8+qWuBe8ycqEXkrvHe7O+d5GJKx8Yd2+x6do1zbq/SxuEP"
        "XUvqtzS3e7ZZzcnXPNSeh179HfM6Z1DPopzfq/qeenZmz7/nbfaIdm0iBjmt0Y6TTuaZecX3X3wD"
        "RQ3inYq5caC/+O7K72fMFEYOjPuNb4mN8zfOItSbkCshlwX/K3nD6zK+6eEjvu7nH9lxX30="
    ),
    "US06": (
        "eNptVlmS6CAIvBAfIipw/4u9boxmMm8qFeKCLA1ImrT7mKiKhnSXdOl9iFmToUNG6zKmy+xNJr/h"
        "ssC5RpflKd6mOA6vxFosrKcs083Dl2PI48uzlDMG5M+EtoDSKQn1+cOa/ShWB0waTTo4DMqHhUwo"
        "nBAwE4qw5i3EB4xYSxzGBxTFNAmfEgnh2M9meLEOIwNG1J7397tUAg6FYc+wZuP5Yg7ZAWMjhiQA"
        "ybGA0RRtEKrNSBanCQRVHYRraoskSJS7GDWHVw22a+slAJKiS05I7QkLYYVD42zlBb1xWFfewYu1"
        "Higbw5AyV8gIIEN0AGAH9OqUrjJV4MMfqI4HVXhkMHs4w9q23IQu2OFAr/QSTaUt8B5RiOEbWSB9"
        "X6fd8EPxArEEX0J2Lvi08HV+FXux95FW2eEzAEikTQKw+rasCFW0an6ixjF0R38i9tgSeHPt88C7"
        "ZM+19RLTwHnsF8iHJImeqc6zxtxvSuCsMzZxRkYWxnAHbZJv9UvynhifYxXcfdbuRo0Yf2t3rUZ+"
        "RpqH5d0tKYxZG++aHetL0Tvq/fhRa3t0hdboRUP78XxDMk5GbwIY8x7YbN/Nu/5uqr4nyZAsvxqf"
        "N66mX4pLlX+we71ehTv5eGG04DSovywkUZ5VurQrEBWluuB1x22BEcuQMdtk8sQin3PKqtTJcmUB"
        "KUP9EFY0zXjIeoRuebgPIJ4E15x2nugU3x18rC/t3bgLRZ2Bq/rsc5KFG84p/dgk+yG1sQn5aOk+"
        "y5tEcx0LjO5HnFR0OzBtwOaZBsMf8wvYOpdSRTIvX4mqY9M/mb5HfqNQx+wRgMCRlORxovCS/r0W"
        "C9hyZt+SHDF5ldFXZr8yZVXnXavQxpkSzh+kwmN5JccNXmnzu7tObtC3Jy3GJe3kxru7EyTurl4d"
        "pXLe6WtGOx5VOlZSb5zzjioedkCsCMbN7EJ349xvGPNTBvNnCdVVB2nfbzzV976b+Qyecvtz5Ofy"
        "OBsldsbufTrq5g3X6gzOrhFP/0XCL+DG3lw/C8CBHaaaPXjY8DsDyI7ggsvyd3tC54Edy9Cg0MGA"
        "70A6Dagd8HlAnCUbF34J2IKjsoI1hsbHBhiQCSFojsN4Jvc5RMLG879BADobkjAlIKxuE4wosxXf"
        "QM5unVpyqsHC7I5GrGzOaKNex2Eg65sFXP8nSFN25EEOZKshIzrDVsGGwP8drutOeI8Yqsd4GBZP"
        "FNvCGv+mnD804AnbvwWhe+5ICucfFtBa8GAi7WmxIYusZNZtC8O+Kv8BzZbv+Q=="
    ),
}


def _decode_cycle(name):
    raw = zlib.decompress(base64.b64decode("".join(_CYCLE_DATA[name]))).decode()
    v = np.array([int(x) for x in raw.split(",")], dtype=float) / 36.0   # m/s
    return np.arange(len(v), dtype=float), v


def _nedc():
    """NEDC = 4 x ECE-15 + EUDC, built from its segment definition."""
    def seg(pts):
        t, v = [0.0], [0.0]
        for dur, vend in pts:
            t.append(t[-1] + dur)
            v.append(vend)
        return np.array(t), np.array(v)
    ece = [(11, 0), (4, 15), (8, 15), (5, 0), (21, 0), (12, 32), (24, 32), (11, 0),
           (21, 0), (26, 50), (12, 50), (8, 35), (13, 35), (12, 0), (7, 0)]
    eudc = [(20, 0), (41, 70), (50, 70), (8, 50), (69, 50), (13, 70), (50, 70),
            (35, 100), (30, 100), (20, 120), (10, 120), (34, 0), (20, 0)]
    pts = ece * 4 + eudc
    tk, vk = seg(pts)
    t = np.arange(0, tk[-1] + 1)
    return t, np.interp(t, tk, vk) / 3.6


CYCLES = ["WLTC class 3b", "UDDS (FTP-72)", "HWFET", "US06", "NEDC",
          "WOT 0-130 km/h", "Constant speed", "Custom CSV"]
_CYCLE_KEY = {"WLTC class 3b": "WLTC", "UDDS (FTP-72)": "UDDS",
              "HWFET": "HWFET", "US06": "US06"}


@dataclass
class Scenario:
    name: str
    t: np.ndarray            # s
    v: np.ndarray            # m/s reference speed
    grade_pct: np.ndarray    # %
    kind: str = "cycle"      # "cycle" | "wot"

    @property
    def duration(self):
        return float(self.t[-1])


def make_scenario(name, p, csv_path=""):
    lname = name.lower()
    kind = "cycle"
    grade = None
    if name in _CYCLE_KEY:
        t, v = _decode_cycle(_CYCLE_KEY[name])
    elif lname.startswith("nedc"):
        t, v = _nedc()
    elif lname.startswith("wot"):
        t = np.arange(0, 41.0)
        v = np.zeros_like(t)
        kind = "wot"
    elif lname.startswith("constant"):
        vt = p.const_kmh / 3.6
        tr = vt / 1.5
        t = np.arange(0, tr + p.const_hold_s + 1.0)
        v = np.minimum(t * 1.5, vt)
    elif lname.startswith("custom"):
        t, v, grade = load_csv_cycle(csv_path)
    else:
        raise ValueError(f"unknown scenario '{name}'")
    if grade is None:
        grade = np.full_like(t, p.grade_pct, dtype=float)
    return Scenario(name, t.astype(float), v.astype(float), grade.astype(float), kind)


def load_csv_cycle(src):
    """CSV with columns: time [s], speed [km/h], optional grade [%]. Header allowed.

    `src` is a file path, raw bytes / str contents, or a file-like object
    (e.g. a Streamlit UploadedFile)."""
    import io
    if isinstance(src, (bytes, bytearray)):
        text = bytes(src).decode("utf-8-sig", errors="replace")
    elif hasattr(src, "read"):
        data = src.read()
        text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    elif isinstance(src, str) and "\n" in src:
        text = src
    else:
        if not src or not os.path.isfile(src):
            raise FileNotFoundError(f"custom cycle file not found: '{src}'")
        with open(src, encoding="utf-8-sig") as f:
            text = f.read()
    # European spreadsheets: ';' separator with decimal commas
    semi = text.count(";") > 0 and text.count(";") >= text.count("\n") // 2
    t, v, g = [], [], []
    for row in csv.reader(io.StringIO(text), delimiter=";" if semi else ","):
        try:
            vals = [float(x.replace(",", ".") if semi else x) for x in row if x.strip() != ""]
        except ValueError:
            continue          # header / comment
        if len(vals) < 2:
            continue
        t.append(vals[0]); v.append(vals[1] / 3.6)
        g.append(vals[2] if len(vals) > 2 else 0.0)
    if len(t) < 2:
        raise ValueError("custom cycle needs at least two rows of time, speed")
    return np.array(t), np.array(v), np.array(g)


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class Params:
    # --- Vehicle body ---
    mass: float = 1500.0          # kg, test mass
    Cd: float = 0.25
    A: float = 2.17               # m2
    Crr: float = 0.0075
    r_wheel: float = 0.31         # m, rolling radius
    J_wheels: float = 3.2         # kg m2, all four wheels
    # --- Power-split driveline ---
    sun_teeth: float = 30
    ring_teeth: float = 78
    mg2_ratio: float = 2.636      # MG2 -> ring reduction
    final_drive: float = 3.267    # ring -> wheel (counter + differential)
    dl_spin_Nm: float = 1.2       # drivetrain drag at ring, Nm
    dl_visc: float = 0.004        # Nm/(rad/s) at ring
    # --- Engine (mapped SI) ---
    eng_disp_L: float = 1.8
    eng_Pmax_kW: float = 73.0
    eng_rpm_idle: float = 1000.0
    eng_rpm_max: float = 5200.0
    eng_J: float = 0.18
    eng_eta_ind: float = 0.44     # peak indicated efficiency (Willans)
    eng_tau: float = 0.15         # s, torque response
    # --- MG1 (generator) ---
    mg1_Pmax_kW: float = 42.0
    mg1_Tmax: float = 80.0
    mg1_rpm_max: float = 10000.0
    mg1_J: float = 0.025
    # --- MG2 (traction motor) ---
    mg2_Pmax_kW: float = 60.0
    mg2_Tmax: float = 207.0
    mg2_rpm_max: float = 13500.0
    mg2_J: float = 0.035
    # --- Battery (Li-ion) + DC-DC ---
    batt_cells: float = 60
    batt_Ah: float = 6.0
    batt_Rcell_mOhm: float = 3.0
    batt_Pdis_kW: float = 27.0
    batt_Pchg_kW: float = 25.0
    soc_init: float = 0.60
    dcdc_eff: float = 0.98
    aux_kW: float = 0.40
    # --- Supervisory controller (VCU) ---
    soc_target: float = 0.60
    soc_low: float = 0.45
    soc_stat_chg: float = 0.45
    ev_vmax_kmh: float = 55.0
    ev_Pmax_kW: float = 10.0
    eng_min_on_s: float = 4.0
    eng_on_delay_s: float = 0.5   # request must persist before a start
    eng_off_delay_s: float = 2.0  # no-request must persist before a stop
    eng_P_min_kW: float = 4.0
    cs_gain_kW: float = 80.0      # charge-sustain power per unit SOC error
    cs_Pmax_kW: float = 10.0
    eng_P_rate_up: float = 20.0   # kW/s  (torque blending EV -> HEV)
    eng_P_rate_dn: float = 40.0   # kW/s
    stat_chg_kW: float = 5.0
    regen_fade_kmh: float = 8.0
    gen_Kp: float = 1.2           # MG1 speed PI, Nm/(rad/s)
    gen_Ki: float = 4.0           # Nm/rad
    # --- Driver ---
    drv_Kp: float = 1.0           # x mass, N/(m/s)
    drv_Ki: float = 0.15          # x mass
    drv_preview: float = 0.4      # s
    # --- Environment ---
    grade_pct: float = 0.0
    wind_kmh: float = 0.0         # head wind (+)
    T_amb_C: float = 25.0
    p_amb_kPa: float = 101.325
    # --- Fuel ---
    fuel_LHV_MJkg: float = 43.0
    fuel_rho: float = 0.745       # kg/L
    co2_per_kg: float = 3.09      # kg CO2 / kg fuel
    # --- Scenario helpers ---
    const_kmh: float = 90.0
    const_hold_s: float = 300.0
    # --- Solver ---
    dt: float = 0.02
    log_dt: float = 0.1


PARAM_HELP = {
    "mass": "vehicle test mass [kg]", "Cd": "drag coefficient [-]",
    "A": "frontal area [m2]", "Crr": "rolling resistance coefficient [-]",
    "r_wheel": "tire rolling radius [m]", "sun_teeth": "planetary sun teeth",
    "ring_teeth": "planetary ring teeth", "mg2_ratio": "MG2 reduction to ring",
    "final_drive": "ring-to-wheel ratio", "eng_Pmax_kW": "engine max power [kW]",
    "mg1_Pmax_kW": "MG1 max power [kW]", "mg2_Pmax_kW": "MG2 max power [kW]",
    "batt_Ah": "cell capacity [Ah]", "soc_init": "initial SOC [-]",
    "soc_target": "charge-sustaining SOC target [-]",
    "ev_vmax_kmh": "max EV-mode speed [km/h]", "ev_Pmax_kW": "max EV wheel power [kW]",
    "grade_pct": "road grade [%]", "wind_kmh": "head wind [km/h]",
    "T_amb_C": "ambient temperature [C]", "aux_kW": "auxiliary electrical load [kW]",
}


def interp1(x, xp, fp):
    """Fast scalar linear interpolation with end clamping (xp ascending)."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    i = bisect.bisect_right(xp, x)
    x0, x1 = xp[i - 1], xp[i]
    return fp[i - 1] + (fp[i] - fp[i - 1]) * (x - x0) / (x1 - x0)


# =============================================================================
# Engine: mapped SI engine with Willans-type fuel model
# =============================================================================
class Engine:
    # Normalised full-load torque shape of a 1.8 L Atkinson engine (73 kW @ 5200)
    _RPM = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5200]
    _TQ = [102, 118, 128, 134, 138, 141, 142, 141, 137, 134]
    _P_REF = 73.0          # kW of the reference curve (subclasses may redefine)
    _RPM_REF = 5200.0      # max speed of the reference curve

    def __init__(self, p):
        self.p = p
        self.Vd = p.eng_disp_L * 1e-3
        self.LHV = p.fuel_LHV_MJkg * 1e6
        scale = p.eng_Pmax_kW / self._P_REF
        r = np.array(self._RPM, float) * p.eng_rpm_max / self._RPM_REF
        self.tq_w = list(r * RPM)
        self.tq_T = list(np.array(self._TQ, float) * scale * self._RPM_REF / p.eng_rpm_max)
        self.w_idle = p.eng_rpm_idle * RPM
        self.w_max = p.eng_rpm_max * RPM
        self._build_ool()

    # full-load torque
    def tmax(self, w):
        if w > self.w_max * 1.02:
            return 0.0
        return interp1(w, self.tq_w, self.tq_T)

    def tmax_v(self, w):
        return np.interp(w, self.tq_w, self.tq_T)

    # friction (motoring) torque from FMEP correlation
    def tfric(self, w):
        if w < 0.5:
            return 0.0
        n = w / RPM / 1000.0
        fmep = (0.6 + 0.1 * n + 0.03 * n * n) * 1e5
        return fmep * self.Vd / (4 * math.pi)

    def tfric_v(self, w):
        n = np.asarray(w) / RPM / 1000.0
        return (0.6 + 0.1 * n + 0.03 * n * n) * 1e5 * self.Vd / (4 * math.pi)

    def eta_ind_v(self, w, T):
        w = np.maximum(np.asarray(w, float), 1.0)
        rpm = w / RPM
        x = np.asarray(T, float) / np.maximum(self.tmax_v(w), 1e-6)
        f_speed = 1 - 0.05 * (1500.0 / np.maximum(rpm, 600)) ** 2
        f_load = 1 - 2.0 * np.maximum(0.0, x - 0.8) ** 2
        return self.p.eng_eta_ind * f_speed * f_load

    def fuel_rate_v(self, w, T):
        """kg/s for brake torque T at speed w (fuelled)."""
        Ti = np.asarray(T, float) + self.tfric_v(w)
        return np.maximum(Ti, 0) * np.asarray(w) / (self.eta_ind_v(w, T) * self.LHV)

    def fuel_rate(self, w, T):
        if w < 1.0:
            return 0.0
        rpm = w / RPM
        tm = max(self.tmax(w), 1e-6)
        x = T / tm
        fs = 1 - 0.05 * (1500.0 / max(rpm, 600)) ** 2
        fl = 1 - 2.0 * max(0.0, x - 0.8) ** 2
        Ti = T + self.tfric(w)
        if Ti <= 0:
            return 0.0
        return Ti * w / (self.p.eng_eta_ind * fs * fl * self.LHV)

    def bsfc_v(self, w, T):
        """g/kWh (nan where T<=0)."""
        P = np.asarray(T) * np.asarray(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = self.fuel_rate_v(w, T) * 1000 * 3.6e6 / P
        return np.where(P > 0, b, np.nan)

    def _build_ool(self):
        """Optimal operating line: min fuel speed/torque for each power level."""
        w = np.arange(self.w_idle, self.w_max + 1e-9, 5 * RPM)
        Tm = self.tmax_v(w)
        self.P_max = float(np.max(Tm * w))
        P_lv = np.linspace(0, self.P_max, 241)
        ws, Ts = [self.w_idle], [0.0]
        for P in P_lv[1:]:
            T = P / w
            ok = T <= Tm
            if not np.any(ok):
                ws.append(ws[-1]); Ts.append(P / ws[-1]); continue
            fr = self.fuel_rate_v(w[ok], T[ok])
            i = int(np.argmin(fr))
            ws.append(float(w[ok][i])); Ts.append(float(T[ok][i]))
        ws = np.maximum.accumulate(np.array(ws))       # monotone speed schedule
        ker = np.ones(15) / 15.0                         # remove grid steps
        wp = np.pad(ws, 7, mode="edge")
        ws = np.maximum.accumulate(np.convolve(wp, ker, mode="valid"))
        ws[0] = self.w_idle
        Ts = np.where(ws > 0, P_lv / ws, 0.0)
        Ts = np.minimum(Ts, self.tmax_v(ws))
        self.ool_P, self.ool_w, self.ool_T = list(P_lv), list(ws), list(Ts)

    def ool(self, P):
        return interp1(P, self.ool_P, self.ool_w), interp1(P, self.ool_P, self.ool_T)


# =============================================================================
# Electric machines: analytic loss map (copper + iron + windage + inverter)
# =============================================================================
class Machine:
    def __init__(self, Pmax_kW, Tmax, rpm_max, J):
        self.Pmax = Pmax_kW * 1e3
        self.Tmax = Tmax
        self.wmax = rpm_max * RPM
        self.J = J
        self.kc = 0.08 * self.Pmax / Tmax ** 2           # copper W/Nm^2
        self.ki = 0.02 * self.Pmax / self.wmax           # iron   W/(rad/s)
        self.kw = 0.01 * self.Pmax / self.wmax ** 3      # windage
        self.c0 = 0.002 * self.Pmax                      # inverter/const

    def tlim(self, w):
        aw = abs(w)
        if aw > self.wmax * 1.05:
            return 0.0
        if aw < 1e-3:
            return self.Tmax
        return min(self.Tmax, self.Pmax / aw)

    def tlim_v(self, w):
        aw = np.abs(np.asarray(w, float))
        t = np.minimum(self.Tmax, self.Pmax / np.maximum(aw, 1e-3))
        return np.where(aw > self.wmax * 1.05, 0.0, t)

    def loss(self, T, w):
        aw = abs(w)
        if abs(T) < 0.05 and aw < 1.0:
            return 0.0
        return self.c0 + self.kc * T * T + self.ki * aw + self.kw * aw ** 3

    def pelec(self, T, w):
        return T * w + self.loss(T, w)

    def torque_for_pelec(self, P, w, Tlo, Thi):
        """Torque whose electrical power equals P (clipped to [Tlo, Thi])."""
        aw = abs(w)
        c = self.c0 + self.ki * aw + self.kw * aw ** 3 - P
        disc = w * w - 4 * self.kc * c
        if disc < 0:
            T = -w / (2 * self.kc)            # minimum-power torque
        else:
            T = (-w + math.sqrt(disc)) / (2 * self.kc)
        return min(max(T, Tlo), Thi)

    def eff_map(self, w, T):
        w = np.asarray(w, float); T = np.asarray(T, float)
        L = self.c0 + self.kc * T ** 2 + self.ki * np.abs(w) + self.kw * np.abs(w) ** 3
        Pm = T * w
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(Pm >= 0, Pm / (Pm + L), (-Pm - L) / (-Pm))
        return np.clip(eta, 0, 1)


# =============================================================================
# Battery: Li-ion datasheet-style pack
# =============================================================================
class Battery:
    _SOC = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    _OCV = [3.00, 3.30, 3.45, 3.55, 3.60, 3.65, 3.70, 3.78, 3.87, 3.96, 4.06, 4.12, 4.18]

    def __init__(self, p: Params):
        self.p = p
        self.Ns = p.batt_cells
        self.Q = p.batt_Ah * 3600.0
        T = p.T_amb_C + 273.15
        self.Rcell = p.batt_Rcell_mOhm * 1e-3 * math.exp(3500 * (1 / T - 1 / 298.15))
        self.R = self.Ns * self.Rcell
        self.E_nom = self.Ns * 3.7 * p.batt_Ah / 1000.0      # kWh

    def ocv(self, soc):
        return self.Ns * interp1(soc, self._SOC, self._OCV)

    def r(self, soc):
        return self.R * (1 + 1.5 * max(0.0, 0.2 - soc) / 0.2)

    def limits(self, soc):
        """Virtual BMS: dynamic discharge / charge power limits [W]."""
        p = self.p
        fd = min(max((soc - 0.25) / 0.10, 0.0), 1.0)
        fc = min(max((0.95 - soc) / 0.10, 0.0), 1.0)
        V, R = self.ocv(soc), self.r(soc)
        # voltage-window limits (2.8 V / 4.2 V per cell)
        Vmin, Vmax = 2.8 * self.Ns, 4.2 * self.Ns
        Pv_d = Vmin * (V - Vmin) / R
        Pv_c = Vmax * (Vmax - V) / R
        return min(p.batt_Pdis_kW * 1e3 * fd, Pv_d), min(p.batt_Pchg_kW * 1e3 * fc, Pv_c)

    def current(self, P, soc):
        """Current [A] for terminal power P [W] (+ discharge)."""
        V, R = self.ocv(soc), self.r(soc)
        disc = V * V - 4 * R * P
        if disc < 0:
            return V / (2 * R)
        return (V - math.sqrt(disc)) / (2 * R)


# =============================================================================
# Simulation results container
# =============================================================================
class Results:
    pass


SIGNALS = ["t", "v_ref", "v", "accel", "brake", "mode", "eng_on",
           "w_e", "w_g", "w_m", "Te", "Tg", "Tm", "T_fb",
           "P_eng", "P_g", "P_m", "P_ge", "P_me", "P_batt", "P_wheel",
           "soc", "I_batt", "V_batt", "mdot", "fuel_L", "dist",
           "P_dis_lim", "P_chg_lim", "eta_eng", "grade"]


# =============================================================================
# The simulator
# =============================================================================
def simulate(p: Params, scen: Scenario, progress=None):
    eng = Engine(p)
    mg1 = Machine(p.mg1_Pmax_kW, p.mg1_Tmax, p.mg1_rpm_max, p.mg1_J)
    mg2 = Machine(p.mg2_Pmax_kW, p.mg2_Tmax, p.mg2_rpm_max, p.mg2_J)
    bat = Battery(p)

    # --- driveline constants
    S, R = p.sun_teeth, p.ring_teeth
    k = R / S
    Gm, Gf, rw = p.mg2_ratio, p.final_drive, p.r_wheel
    M11 = p.eng_J + mg1.J * (1 + k) ** 2
    M12 = -mg1.J * (1 + k) * k
    M22 = mg1.J * k * k + mg2.J * Gm * Gm + (p.mass * rw * rw + p.J_wheels) / Gf ** 2
    det = M11 * M22 - M12 * M12
    M_eq = M22 * Gf * Gf / (rw * rw)                       # equivalent mass, kg
    wsg_max = mg1.wmax * 0.95

    # --- environment
    rho = p.p_amb_kPa * 1e3 / (287.05 * (p.T_amb_C + 273.15))
    v_wind = p.wind_kmh / 3.6
    cdA = 0.5 * rho * p.Cd * p.A
    eta_dc = p.dcdc_eff
    P_aux = p.aux_kW * 1e3

    dt = p.dt
    n_steps = int(scen.duration / dt) + 1
    log_every = max(1, int(round(p.log_dt / dt)))
    t_cyc, v_cyc, g_cyc = list(scen.t), list(scen.v), list(scen.grade_pct)
    T_brake_max = 1.0 * p.mass * G * rw
    Te_ool_max = eng.ool(eng.P_max)[1]
    T_ring_eng_max = Te_ool_max * k / (1 + k)

    # --- states
    wc = 0.0          # carrier = engine speed
    wr = 0.0          # ring speed
    Te = 0.0          # engine net torque (fuelled state)
    soc = p.soc_init
    drv_I = 0.0
    gen_I = 0.0
    mode = 0
    eng_state = "OFF"         # OFF | START | RUN | STOP
    eng_on_timer = 0.0
    want_t = 0.0
    nowant_t = 0.0
    P_cmd = 0.0              # rate-limited engine power command
    w_ref = 0.0               # rate-limited engine speed reference
    dist = 0.0
    fuel_kg = 0.0
    wot_done_t = None
    t100 = None

    # --- energy integrators [J]
    E = dict(fuel=0.0, eng_mech=0.0, eng_loss=0.0, g_mech=0.0, g_elec=0.0, g_loss=0.0,
             m_mech=0.0, m_elec=0.0, m_loss=0.0, dcdc_loss=0.0, batt_chem=0.0,
             batt_loss=0.0, batt_term=0.0, aux=0.0, aero=0.0, roll=0.0, grade=0.0,
             brake=0.0, driveline=0.0, standstill=0.0, regen_m=0.0, wheel_pos=0.0,
             batt_out=0.0, batt_in=0.0, m_motor_mech=0.0, m_motor_elec=0.0,
             m_gen_mech=0.0, m_gen_elec=0.0, g_gen_mech=0.0, g_gen_elec=0.0,
             g_mot_mech=0.0, g_mot_elec=0.0)
    KE0 = 0.0
    mode_time = [0.0, 0.0, 0.0, 0.0]
    n_starts = 0
    eng_eff_samples = []      # (eta, dt) while running & loaded

    log = {s: [] for s in SIGNALS}

    for step in range(n_steps):
        t = step * dt
        # ------------------------------------------------------------ speeds
        ws = (1 + k) * wc - k * wr            # sun = MG1
        wm = Gm * wr                          # MG2
        ww = wr / Gf                          # wheel
        v = ww * rw
        grade = interp1(t, t_cyc, g_cyc)
        th = math.atan(grade / 100.0)
        # ------------------------------------------------------------ BMS
        P_dis_lim, P_chg_lim = bat.limits(soc)

        # ============================================================ DRIVER
        Tw_max = (T_ring_eng_max + mg2.tlim(wm) * Gm) * Gf
        if scen.kind == "wot":
            if wot_done_t is None and v >= 130 / 3.6:
                wot_done_t = t
            if wot_done_t is None and t < 30:
                accel, brake = 1.0, 0.0
            else:
                accel, brake = 0.0, 0.25
            v_r = v
        else:
            v_r = interp1(t, t_cyc, v_cyc)
            v_tg = interp1(t + p.drv_preview, t_cyc, v_cyc)
            a_r = (interp1(t + p.drv_preview + 0.5, t_cyc, v_cyc)
                   - interp1(t + p.drv_preview - 0.5, t_cyc, v_cyc))
            F_ff = (M_eq * a_r + cdA * (v_tg + v_wind) * abs(v_tg + v_wind)
                    + (p.Crr * p.mass * G * math.cos(th) if v_tg > 0.05 else 0.0)
                    + p.mass * G * math.sin(th))
            err = v_tg - v
            F_cmd = F_ff + p.drv_Kp * p.mass * err + drv_I
            Tw_cmd = F_cmd * rw
            if v_tg < 0.05 and v < 0.3:
                accel, brake = 0.0, 0.3            # hold at standstill
                drv_I = 0.0
            elif Tw_cmd >= 0:
                accel, brake = min(Tw_cmd / Tw_max, 1.0), 0.0
            else:
                accel, brake = 0.0, min(-Tw_cmd / T_brake_max, 1.0)
            sat = (accel >= 1.0 and err > 0) or (brake >= 1.0 and err < 0)
            if not sat and not (v_tg < 0.05 and v < 0.3):
                drv_I += p.drv_Ki * p.mass * err * dt
        if t100 is None and v >= 100 / 3.6:
            t100 = t

        # ============================================================ VCU
        Tw_req = accel * Tw_max - brake * T_brake_max
        Pw_req = Tw_req * ww
        soc_tgt = p.soc_target - 0.08 * (v / (130 / 3.6)) ** 2   # room for regen at speed
        P_cs = min(max(p.cs_gain_kW * 1e3 * (soc_tgt - soc), -p.cs_Pmax_kW * 1e3),
                   p.cs_Pmax_kW * 1e3)
        eng_running = eng_state in ("START", "RUN")
        P_on = p.ev_Pmax_kW * 1e3 * min(max(1 + 4 * (soc - p.soc_target), 0.3), 1.5)
        mg1_overspeed = k * wr > 0.9 * mg1.wmax

        # ---- operating mode
        if v < 0.3 and accel < 0.02:
            new_mode = 0
        elif Tw_req <= 0:
            new_mode = 1
        else:
            need = (Pw_req > P_on or v > p.ev_vmax_kmh / 3.6 or soc < p.soc_low
                    or accel > 0.7 or mg1_overspeed or Pw_req > 0.8 * P_dis_lim)
            ok_off = (Pw_req < 0.7 * P_on and v < p.ev_vmax_kmh / 3.6 - 1.5
                      and soc > p.soc_low + 0.02 and accel < 0.6 and not mg1_overspeed)
            if eng_running:
                hev = not (ok_off and eng_on_timer > p.eng_min_on_s)
            else:
                hev = need
            if hev:
                need_P = max(Pw_req, 0) / 0.88 + P_cs
                new_mode = 3 if need_P > 0.98 * eng.P_max else 2
            else:
                new_mode = 1
        mode = new_mode

        # ---- engine on/off request & power request
        if mode in (2, 3):
            eng_want = True
            P_req = max(Pw_req, 0) / 0.88 + P_cs
            P_req = min(P_req, max(Pw_req, 0) / 0.88 + 0.8 * P_chg_lim)
            P_req = min(max(P_req, p.eng_P_min_kW * 1e3), eng.P_max)
            if mode == 3:
                P_req = eng.P_max
        elif mode == 0:
            eng_want = soc < p.soc_stat_chg or (eng_running and soc < p.soc_stat_chg + 0.03)
            P_req = min(p.stat_chg_kW * 1e3, 0.8 * P_chg_lim)
        else:
            eng_want = False
            P_req = 0.0

        # ---- engine state machine (off / start / run / stop) with debouncing:
        #      start needs a sustained request (unless urgent), stop needs a
        #      sustained no-request AND the minimum on-time.
        urgent = accel > 0.7 or mg1_overspeed or soc < p.soc_low - 0.03
        if eng_want:
            want_t += dt; nowant_t = 0.0
        else:
            nowant_t += dt; want_t = 0.0
        if eng_state in ("START", "RUN") and not eng_want:
            delay = 0.5 if brake > 0.1 else p.eng_off_delay_s
            if nowant_t < delay or eng_on_timer < p.eng_min_on_s:
                eng_want = True                    # keep running (low power)
                if mode == 1 or mode == 0:
                    P_req = p.eng_P_min_kW * 1e3 if mode == 1 and Tw_req > 0 else 0.0
                    if mode == 1 and Tw_req > 0:
                        mode = 2
        elif eng_state in ("OFF", "STOP") and eng_want and not urgent:
            if want_t < p.eng_on_delay_s:
                eng_want = False
        if eng_want and eng_state in ("OFF", "STOP"):
            eng_state = "START"; gen_I = 0.0; P_cmd = 0.0; n_starts += 1
        elif not eng_want and eng_state in ("START", "RUN"):
            eng_state = "STOP"; gen_I = 0.0
        if eng_state == "START" and wc >= 0.9 * eng.w_idle:
            eng_state = "RUN"; w_ref = wc
        if eng_state == "STOP" and wc < 3.0:
            eng_state = "OFF"
        eng_on_timer = eng_on_timer + dt if eng_state in ("START", "RUN") else 0.0

        # ---- engine power / speed / torque commands (optimal BSFC line)
        if eng_state == "RUN":
            dP = P_req - P_cmd
            P_cmd += min(max(dP, -p.eng_P_rate_dn * 1e3 * dt), p.eng_P_rate_up * 1e3 * dt)
            w_tgt, Te_cmd = eng.ool(P_cmd)
            lo = max(eng.w_idle, (k * wr - wsg_max) / (1 + k))
            hi = min(eng.w_max, (k * wr + wsg_max) / (1 + k))
            w_tgt = min(max(w_tgt, lo), hi)
            Te_cmd = min(P_cmd / max(w_tgt, 1.0), eng.tmax(max(wc, eng.w_idle)))
            w_ref += min(max(w_tgt - w_ref, -250 * dt), 250 * dt)
            fuel_on = True
        elif eng_state == "START":
            w_ref = eng.w_idle; Te_cmd = 0.0; fuel_on = False
        else:
            w_ref = 0.0; Te_cmd = 0.0; fuel_on = False; P_cmd = 0.0

        # ---- MG1: engine-speed regulation (PI + reaction feed-forward)
        Tf_now = eng.tfric(wc)
        Te_net_est = Te if fuel_on else -Tf_now
        Tg_lim = mg1.tlim(ws)
        if eng_state == "OFF" and wc <= 1e-6:
            Tg = 0.0; gen_I = 0.0
        else:
            e = w_ref - wc
            Tg_ff = -Te_net_est / (1 + k)
            Tg_raw = Tg_ff + p.gen_Kp * e + gen_I
            if eng_state == "STOP":
                Tg_lim = min(Tg_lim, 25.0)
            Tg = min(max(Tg_raw, -Tg_lim), Tg_lim)
            if Tg == Tg_raw or (Tg_raw > Tg_lim and e < 0) or (Tg_raw < -Tg_lim and e > 0):
                gen_I += p.gen_Ki * e * dt
                gen_I = min(max(gen_I, -Tg_lim), Tg_lim)

        # ---- MG2: traction / regenerative braking with power management
        T_dl = (p.dl_spin_Nm + p.dl_visc * wr) if wr > 0.1 else 0.0
        Tm_lim = mg2.tlim(wm)
        T_fb = 0.0
        if Tw_req >= 0:
            T_ring_req = Tw_req / Gf + (T_dl if Tw_req > 0 else 0.0)
            Tm = (T_ring_req + k * Tg) / Gm
        else:
            fade = min(max((v * 3.6 - 2.0) / max(p.regen_fade_kmh - 2.0, 0.1), 0.0), 1.0)
            Tm = (Tw_req / Gf * fade + k * Tg) / Gm
        Tm = min(max(Tm, -Tm_lim), Tm_lim)
        # battery power limits (virtual BMS)
        P_ge = mg1.pelec(Tg, ws)
        P_me = mg2.pelec(Tm, wm)
        P_bus = P_ge + P_me + P_aux
        P_bt = P_bus / eta_dc if P_bus > 0 else P_bus * eta_dc
        if P_bt > P_dis_lim:
            P_allow = P_dis_lim * eta_dc - P_ge - P_aux
            Tm = mg2.torque_for_pelec(P_allow, wm, -Tm_lim, Tm)
        elif P_bt < -P_chg_lim:
            P_allow = -P_chg_lim / eta_dc - P_ge - P_aux
            Tm = mg2.torque_for_pelec(P_allow, wm, Tm, Tm_lim)
        P_me = mg2.pelec(Tm, wm)
        P_bus = P_ge + P_me + P_aux
        P_bt = P_bus / eta_dc if P_bus > 0 else P_bus * eta_dc
        # friction brakes take what regen cannot
        if Tw_req < 0:
            T_ring_el = Gm * Tm - k * Tg
            regen_w = max(0.0, -T_ring_el * Gf)
            T_fb = max(0.0, -Tw_req - regen_w)
        elif brake > 0:
            T_fb = brake * T_brake_max

        # ============================================================ PLANT
        # engine (ECU torque tracking, first order)
        if fuel_on:
            Te += (Te_cmd - Te) * min(dt / p.eng_tau, 1.0)
            Te = min(Te, eng.tmax(wc))
            Te_net = Te
            mdot = eng.fuel_rate(wc, Te)
        else:
            Te = 0.0
            Te_net = -Tf_now
            mdot = 0.0

        # battery
        I = bat.current(P_bt, soc)
        Voc = bat.ocv(soc)
        Rb = bat.r(soc)
        P_chem = Voc * I
        P_bl = I * I * Rb

        # road loads
        vrel = v + v_wind
        F_aero = cdA * vrel * abs(vrel)
        F_roll = p.Crr * p.mass * G * math.cos(th) if v > 0.01 else 0.0
        F_grade = p.mass * G * math.sin(th)
        T_fb_eff = T_fb if v > 0.01 else 0.0

        # generalized forces & 2-DOF dynamics
        Qc = Te_net + (1 + k) * Tg
        Qr = -k * Tg + Gm * Tm - T_dl - ((F_aero + F_roll + F_grade) * rw + T_fb_eff) / Gf
        ac = (M22 * Qc - M12 * Qr) / det
        ar = (M11 * Qr - M12 * Qc) / det

        # ---- energy bookkeeping (start-of-step speeds)
        P_eng = Te_net * wc
        P_g = Tg * ws
        P_m = Tm * wm
        E["fuel"] += mdot * eng.LHV * dt
        E["eng_mech"] += P_eng * dt
        E["eng_loss"] += (mdot * eng.LHV - P_eng) * dt
        E["g_mech"] += P_g * dt; E["g_elec"] += P_ge * dt; E["g_loss"] += (P_ge - P_g) * dt
        E["m_mech"] += P_m * dt; E["m_elec"] += P_me * dt; E["m_loss"] += (P_me - P_m) * dt
        if P_m >= 0:
            E["m_motor_mech"] += P_m * dt; E["m_motor_elec"] += P_me * dt
        else:
            E["m_gen_mech"] += -P_m * dt; E["m_gen_elec"] += -P_me * dt
        if P_g >= 0:
            E["g_mot_mech"] += P_g * dt; E["g_mot_elec"] += P_ge * dt
        else:
            E["g_gen_mech"] += -P_g * dt; E["g_gen_elec"] += -P_ge * dt
        E["dcdc_loss"] += (P_bt - P_bus) * dt
        E["batt_chem"] += P_chem * dt
        E["batt_loss"] += P_bl * dt
        E["batt_term"] += (P_chem - P_bl) * dt
        if P_chem >= 0:
            E["batt_out"] += P_chem * dt
        else:
            E["batt_in"] += -P_chem * dt
        E["aux"] += P_aux * dt
        E["aero"] += F_aero * v * dt
        E["roll"] += F_roll * v * dt
        E["grade"] += F_grade * v * dt
        E["brake"] += T_fb_eff * ww * dt
        E["driveline"] += T_dl * wr * dt
        mode_time[mode] += dt
        if fuel_on and P_eng > 1000:
            eta_e = P_eng / (mdot * eng.LHV) if mdot > 0 else 0.0
            eng_eff_samples.append(eta_e)
        else:
            eta_e = float("nan")

        # ---- logging
        if step % log_every == 0:
            fuel_L = fuel_kg / p.fuel_rho
            vals = (t, v_r, v, accel, brake, mode, 1.0 if eng_state in ("START", "RUN") else 0.0,
                    wc, ws, wm, Te_net, Tg, Tm, T_fb_eff, P_eng, P_g, P_m, P_ge, P_me, P_bt,
                    (Gm * Tm - k * Tg - T_dl) * wr, soc, I, Voc - I * Rb, mdot, fuel_L, dist,
                    P_dis_lim, P_chg_lim, eta_e, grade)
            for name, val in zip(SIGNALS, vals):
                log[name].append(val)

        # ---- integrate (semi-implicit Euler) with one-way constraints
        KE_before = 0.5 * (M11 * wc * wc + 2 * M12 * wc * wr + M22 * wr * wr)
        wc_n = wc + ac * dt
        wr_n = wr + ar * dt
        if wc_n < 0.0:                       # engine cannot run backwards
            wc_n = 0.0
            wr_n = wr + (Qr / M22) * dt      # carrier locked
        if wr_n < 0.0:                       # brake / no reverse
            wr_n = 0.0
        KE_pred = KE_before + (Qc * wc + Qr * wr) * dt
        wc, wr = wc_n, wr_n
        KE_after = 0.5 * (M11 * wc * wc + 2 * M12 * wc * wr + M22 * wr * wr)
        E["standstill"] += max(0.0, KE_pred - KE_after) if (wc == 0.0 or wr == 0.0) else 0.0

        soc -= I * dt / bat.Q
        soc = min(max(soc, 0.0), 1.0)
        fuel_kg += mdot * dt
        dist += v * dt

        if progress is not None and step % 2000 == 0:
            progress(step / n_steps)

        if scen.kind == "wot" and wot_done_t is not None and v < 0.1 and t > wot_done_t:
            break

    # ------------------------------------------------------------ summary
    res = Results()
    res.p = p
    res.scen = scen
    res.eng, res.mg1, res.mg2, res.bat = eng, mg1, mg2, bat
    res.log = {kk: np.array(vv) for kk, vv in log.items()}
    KE_end = 0.5 * (M11 * wc * wc + 2 * M12 * wc * wr + M22 * wr * wr)
    E["dKE"] = KE_end - KE0
    res.E = E
    res.mode_time = mode_time
    res.n_starts = n_starts
    res.eng_eff_samples = np.array(eng_eff_samples)
    res.t100 = t100
    res.t130 = wot_done_t
    res.summary = summarize(res, fuel_kg, dist, soc)
    return res


def summarize(res, fuel_kg, dist, soc_end):
    p, E, lg = res.p, res.E, res.log
    km = dist / 1000.0
    fuel_L = fuel_kg / p.fuel_rho
    L100 = fuel_L / km * 100 if km > 0.05 else float("nan")
    # SOC correction: net battery energy valued at average engine x generator path
    eta_e = E["eng_mech"] / E["fuel"] if E["fuel"] > 0 else 0.35
    eta_path = max(eta_e, 0.2) * 0.85
    dE_batt = E["batt_chem"]                       # +: net taken from battery
    fuel_corr_kg = fuel_kg + dE_batt / (eta_path * p.fuel_LHV_MJkg * 1e6)
    L100c = fuel_corr_kg / p.fuel_rho / km * 100 if km > 0.05 else float("nan")
    sources = E["fuel"] + E["batt_chem"]
    sinks = (E["eng_loss"] + E["g_loss"] + E["m_loss"] + E["dcdc_loss"] + E["batt_loss"]
             + E["aux"] + E["aero"] + E["roll"] + E["grade"] + E["brake"] + E["driveline"]
             + E["standstill"] + E["dKE"])
    bal_err = (sources - sinks) / max(abs(E["fuel"]) + abs(E["batt_out"]) + abs(E["batt_in"]), 1.0)
    err = lg["v"] - lg["v_ref"]
    moving = lg["v_ref"] > 0.5
    tot_t = sum(res.mode_time)
    s = dict(
        scenario=res.scen.name,
        duration_s=float(lg["t"][-1]) if len(lg["t"]) else 0.0,
        distance_km=km,
        fuel_L=fuel_L,
        fuel_L100km=L100,
        fuel_L100km_soc_corrected=L100c,
        mpg_US=235.215 / L100 if L100 and L100 > 0 else float("nan"),
        co2_g_km=fuel_kg * p.co2_per_kg * 1000 / km if km > 0.05 else float("nan"),
        soc_start=p.soc_init, soc_end=soc_end,
        engine_starts=res.n_starts,
        engine_on_pct=100 * (res.mode_time[2] + res.mode_time[3]) / tot_t if tot_t else 0,
        ev_pct=100 * res.mode_time[1] / tot_t if tot_t else 0,
        engine_avg_eff_pct=100 * eta_e,
        speed_err_rms_kmh=float(np.sqrt(np.mean(err[moving] ** 2)) * 3.6) if np.any(moving) else 0.0,
        speed_err_max_kmh=float(np.max(np.abs(err)) * 3.6) if len(err) else 0.0,
        energy_balance_err_pct=100 * bal_err,
        regen_recovered_kWh=E["m_gen_elec"] / 3.6e6,
        friction_brake_kWh=E["brake"] / 3.6e6,
        t_0_100_s=res.t100 if res.scen.kind == "wot" else None,
    )
    return s


def energy_table(res):
    """Rows of (group, item, kWh) for the system-level energy summary."""
    E = res.E
    kwh = lambda j: j / 3.6e6
    rows = [
        ("Source", "Fuel chemical energy", kwh(E["fuel"])),
        ("Source", "Battery net discharge (chemical)", kwh(E["batt_chem"])),
        ("Loss", "Engine losses", kwh(E["eng_loss"])),
        ("Loss", "MG1 losses", kwh(E["g_loss"])),
        ("Loss", "MG2 losses", kwh(E["m_loss"])),
        ("Loss", "Battery resistive losses", kwh(E["batt_loss"])),
        ("Loss", "DC-DC converter losses", kwh(E["dcdc_loss"])),
        ("Loss", "Drivetrain spin losses", kwh(E["driveline"])),
        ("Loss", "Friction brakes", kwh(E["brake"])),
        ("Load", "Auxiliary load", kwh(E["aux"])),
        ("Load", "Aerodynamic drag", kwh(E["aero"])),
        ("Load", "Rolling resistance", kwh(E["roll"])),
        ("Load", "Grade (potential energy)", kwh(E["grade"])),
        ("Storage", "Kinetic energy change", kwh(E["dKE"])),
        ("Loss", "Standstill clamps (numerical)", kwh(E["standstill"])),
    ]
    return rows


def efficiency_table(res):
    E = res.E
    def r(a, b):
        return 100 * a / b if b > 1e-6 else float("nan")
    return [
        ("Engine plant (mech out / fuel)", r(E["eng_mech"], E["fuel"])),
        ("MG2 motoring (mech / elec)", r(E["m_motor_mech"], E["m_motor_elec"])),
        ("MG2 generating (elec / mech)", r(E["m_gen_elec"], E["m_gen_mech"])),
        ("MG1 generating (elec / mech)", r(E["g_gen_elec"], E["g_gen_mech"])),
        ("MG1 motoring (mech / elec)", r(E["g_mot_mech"], E["g_mot_elec"])),
        ("Battery (1 - loss / throughput)",
         100 - r(E["batt_loss"], E["batt_out"] + E["batt_in"])),
        ("Road loads / fuel (tank-to-wheel)",
         r(E["aero"] + E["roll"] + max(E["grade"], 0), E["fuel"] + max(E["batt_chem"], 0))),
    ]


# =============================================================================
# Export
# =============================================================================
def export_results(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    base = res.scen.name.split()[0].replace("/", "-")
    ts = os.path.join(outdir, f"hev_ips_{base}_timeseries.csv")
    with open(ts, "w", newline="") as f:
        w = csv.writer(f)
        units = {"t": "s", "v_ref": "m/s", "v": "m/s", "w_e": "rad/s", "w_g": "rad/s",
                 "w_m": "rad/s", "Te": "Nm", "Tg": "Nm", "Tm": "Nm", "T_fb": "Nm",
                 "soc": "-", "I_batt": "A", "V_batt": "V", "mdot": "kg/s", "fuel_L": "L",
                 "dist": "m", "grade": "%"}
        w.writerow([f"{s} [{units.get(s, 'W' if s.startswith('P') else '-')}]" for s in SIGNALS])
        for row in zip(*[res.log[s] for s in SIGNALS]):
            w.writerow([f"{x:.6g}" for x in row])
    sm = os.path.join(outdir, f"hev_ips_{base}_energy_summary.csv")
    with open(sm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Item", "Energy [kWh]"])
        for g, it, val in energy_table(res):
            w.writerow([g, it, f"{val:.5f}"])
        w.writerow([])
        w.writerow(["Efficiency", "Value [%]"])
        for it, val in efficiency_table(res):
            w.writerow([it, f"{val:.2f}"])
        w.writerow([])
        w.writerow(["Result", "Value"])
        for kk, vv in res.summary.items():
            w.writerow([kk, vv])
    out = [ts, sm]
    try:
        import openpyxl
        from openpyxl.styles import Font
        wb = openpyxl.Workbook()
        ws1 = wb.active; ws1.title = "Energy summary"
        ws1.append(["Group", "Item", "Energy [kWh]"])
        for g, it, val in energy_table(res):
            ws1.append([g, it, round(val, 5)])
        ws1.append([])
        ws1.append(["Efficiency", "", "Value [%]"])
        for it, val in efficiency_table(res):
            ws1.append(["", it, None if math.isnan(val) else round(val, 2)])
        ws1.append([])
        ws1.append(["Result", "", "Value"])
        for kk, vv in res.summary.items():
            ws1.append(["", kk, vv if not isinstance(vv, float) else round(vv, 4)])
        for row in (1, ):
            for c in ws1[row]:
                c.font = Font(bold=True)
        ws1.column_dimensions["A"].width = 12
        ws1.column_dimensions["B"].width = 38
        ws1.column_dimensions["C"].width = 14
        ws2 = wb.create_sheet("Parameters")
        ws2.append(["Parameter", "Value"])
        for kk, vv in asdict(res.p).items():
            ws2.append([kk, vv])
        ws2.column_dimensions["A"].width = 18
        ws3 = wb.create_sheet("Time series")
        ws3.append(SIGNALS)
        for row in zip(*[res.log[s] for s in SIGNALS]):
            ws3.append([None if (isinstance(x, float) and math.isnan(x)) else float(x) for x in row])
        xl = os.path.join(outdir, f"hev_ips_{base}_results.xlsx")
        wb.save(xl)
        out.append(xl)
    except ImportError:
        pass
    return out


def print_summary(res):
    s = res.summary
    print(f"\n=== {s['scenario']} ===")
    print(f"  distance            {s['distance_km']:.2f} km in {s['duration_s']:.0f} s")
    print(f"  fuel                {s['fuel_L']:.3f} L  ->  {s['fuel_L100km']:.2f} L/100 km"
          f"  ({s['mpg_US']:.1f} mpg US)")
    print(f"  SOC-corrected       {s['fuel_L100km_soc_corrected']:.2f} L/100 km"
          f"   CO2 {s['co2_g_km']:.0f} g/km")
    print(f"  SOC                 {s['soc_start']:.3f} -> {s['soc_end']:.3f}")
    print(f"  engine on           {s['engine_on_pct']:.1f} % of time, {s['engine_starts']} starts,"
          f" avg eff {s['engine_avg_eff_pct']:.1f} %")
    print(f"  EV                  {s['ev_pct']:.1f} % of time")
    print(f"  speed tracking      RMS {s['speed_err_rms_kmh']:.2f} km/h, max {s['speed_err_max_kmh']:.2f} km/h")
    print(f"  regen recovered     {s['regen_recovered_kWh']:.3f} kWh, friction brakes {s['friction_brake_kWh']:.3f} kWh")
    print(f"  energy balance err  {s['energy_balance_err_pct']:.3f} %")
    if s.get("t_0_100_s") is not None:
        print(f"  0-100 km/h          {s['t_0_100_s']:.1f} s")
        if res.t130:
            print(f"  0-130 km/h          {res.t130:.1f} s")


# =============================================================================
# Plotting helpers
# =============================================================================
def _style_ax(ax, ylabel=None, xlabel=None):
    ax.set_facecolor(C_SURF)
    ax.grid(True, color=C_GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b2")
    ax.tick_params(colors=C_TXT2, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=C_TXT2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=C_TXT2)


def _mode_spans(ax, t, mode, alpha=0.18):
    if len(t) == 0:
        return
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or mode[i] != mode[start]:
            ax.axvspan(t[start], t[min(i, len(t) - 1)], color=MODE_COLORS[int(mode[start])],
                       alpha=alpha, linewidth=0)
            start = i


def draw_results(fig, rect, res):
    """Time-history view; returns list of axes."""
    lg = res.log
    t = lg["t"]
    x0, y0, w, h = rect
    n = 6
    gap = 0.012
    hh = (h - gap * (n - 1)) / n
    axes = []
    for i in range(n):
        ax = fig.add_axes([x0, y0 + h - (i + 1) * hh - i * gap, w, hh],
                          sharex=axes[0] if axes else None)
        axes.append(ax)
    a = axes[0]
    _mode_spans(a, t, lg["mode"])
    a.plot(t, lg["v_ref"] * 3.6, color=C_REF, lw=2.0, label="Reference")
    a.plot(t, lg["v"] * 3.6, color=C_MG2, lw=1.2, label="Vehicle")
    _style_ax(a, "Speed\n[km/h]")
    from matplotlib.patches import Patch
    hd = [a.lines[0], a.lines[1]] + [Patch(color=c, alpha=0.35, label=m) for m, c in zip(MODE_NAMES, MODE_COLORS)]
    a.legend(handles=hd, fontsize=7, ncol=6, loc="upper left", frameon=False,
             labelcolor=C_TXT2, bbox_to_anchor=(0, 1.32))
    a = axes[1]
    a.plot(t, lg["soc"] * 100, color=C_BAT, lw=1.6)
    a.axhline(res.p.soc_target * 100, color=C_REF, lw=0.8, ls="--")
    _style_ax(a, "SOC\n[%]")
    a = axes[2]
    a.plot(t, lg["w_e"] / RPM, color=C_ENG, lw=1.2, label="Engine")
    a.plot(t, lg["w_g"] / RPM, color=C_MG1, lw=1.0, label="MG1")
    a.plot(t, lg["w_m"] / RPM, color=C_MG2, lw=1.0, label="MG2")
    _style_ax(a, "Speed\n[rpm]")
    a.legend(fontsize=7, ncol=3, loc="upper left", frameon=False, labelcolor=C_TXT2)
    a = axes[3]
    a.plot(t, lg["Te"], color=C_ENG, lw=1.2, label="Engine")
    a.plot(t, lg["Tg"], color=C_MG1, lw=1.0, label="MG1")
    a.plot(t, lg["Tm"], color=C_MG2, lw=1.0, label="MG2")
    _style_ax(a, "Torque\n[Nm]")
    a.legend(fontsize=7, ncol=3, loc="upper left", frameon=False, labelcolor=C_TXT2)
    a = axes[4]
    a.plot(t, lg["P_eng"] / 1e3, color=C_ENG, lw=1.2, label="Engine")
    a.plot(t, lg["P_g"] / 1e3, color=C_MG1, lw=1.0, label="MG1 (mech)")
    a.plot(t, lg["P_m"] / 1e3, color=C_MG2, lw=1.0, label="MG2 (mech)")
    a.plot(t, lg["P_batt"] / 1e3, color=C_BAT, lw=1.0, label="Battery")
    a.axhline(0, color="#b9b8b2", lw=0.6)
    _style_ax(a, "Power\n[kW]")
    a.legend(fontsize=7, ncol=4, loc="upper left", frameon=False, labelcolor=C_TXT2)
    a = axes[5]
    a.plot(t, lg["fuel_L"], color=C_ENG, lw=1.6)
    _style_ax(a, "Fuel\n[L]", "Time [s]")
    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    axes[0].set_xlim(t[0], t[-1])
    return axes


def draw_maps(fig, rect, res):
    eng, mg1, mg2 = res.eng, res.mg1, res.mg2
    lg = res.log
    x0, y0, w, h = rect
    axes = []
    # --- engine BSFC map
    ax = fig.add_axes([x0, y0 + h * 0.42, w * 0.55, h * 0.58])
    axes.append(ax)
    rpm = np.linspace(eng.p.eng_rpm_idle * 0.8, eng.p.eng_rpm_max, 120)
    wv = rpm * RPM
    Tg = np.linspace(2, max(eng.tq_T) * 1.02, 100)
    W, T = np.meshgrid(wv, Tg)
    B = eng.bsfc_v(W, T)
    B = np.where(T <= eng.tmax_v(W), B, np.nan)
    lev = [225, 230, 235, 240, 250, 260, 275, 300, 350, 400, 500, 700]
    cs = ax.contourf(rpm, Tg, B, levels=lev, cmap="Blues_r", extend="both", alpha=0.85)
    cl = ax.contour(rpm, Tg, B, levels=lev, colors="white", linewidths=0.5)
    ax.clabel(cl, fontsize=6.5, fmt="%d")
    ax.plot(np.array(eng.tq_w) / RPM, eng.tq_T, color=C_TXT, lw=1.6, label="Full load")
    ax.plot(np.array(eng.ool_w) / RPM, eng.ool_T, color=C_BRK, lw=1.8, label="Optimal BSFC line")
    run = lg["eng_on"] > 0.5
    ax.scatter(lg["w_e"][run] / RPM, lg["Te"][run], s=8, color=C_ENG, alpha=0.35,
               edgecolors="none", label="Operating points")
    for P in (10, 20, 30, 40, 50, 60, 70):
        pr = np.linspace(800, eng.p.eng_rpm_max, 50)
        tp = P * 1e3 / (pr * RPM)
        m = tp < max(eng.tq_T) * 1.02
        ax.plot(pr[m], tp[m], color=C_REF, lw=0.5, ls=":")
        if np.any(m):
            ax.text(pr[m][-1], tp[m][-1], f" {P} kW", fontsize=6, color=C_TXT2, va="center")
    ax.set_xlim(rpm[0], rpm[-1]); ax.set_ylim(0, max(eng.tq_T) * 1.05)
    _style_ax(ax, "Engine torque [Nm]", "Engine speed [rpm]")
    ax.set_title("Engine BSFC map [g/kWh] with operating points", fontsize=9.5, color=C_TXT, loc="left")
    ax.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white", edgecolor="none",
              labelcolor=C_TXT2)
    cb = fig.colorbar(cs, ax=ax, pad=0.01, fraction=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("BSFC [g/kWh]", fontsize=7.5, color=C_TXT2)
    axes.append(cb.ax)
    # --- efficiency histogram
    ax = fig.add_axes([x0 + w * 0.67, y0 + h * 0.42, w * 0.33, h * 0.58])
    axes.append(ax)
    ef = res.eng_eff_samples * 100
    if len(ef):
        bins = np.arange(0, 42, 2)
        cnt, _ = np.histogram(ef, bins=bins)
        tt = cnt * res.p.dt
        ax.bar(bins[:-1] + 1, tt, width=1.7, color=C_ENG, edgecolor="none")
    _style_ax(ax, "Time [s]", "Engine brake efficiency [%]")
    ax.set_title("Time at engine efficiency (loaded)", fontsize=9.5, color=C_TXT, loc="left")
    # --- motor maps
    for j, (mc, name, wcol, Tcol, col) in enumerate(
            [(mg1, "MG1", "w_g", "Tg", C_MG1), (mg2, "MG2", "w_m", "Tm", C_MG2)]):
        ax = fig.add_axes([x0 + j * w * 0.52, y0, w * 0.46, h * 0.33])
        axes.append(ax)
        wv = np.linspace(-mc.wmax if j == 0 else 0, mc.wmax, 120)
        Tv = np.linspace(-mc.Tmax, mc.Tmax, 100)
        W, T = np.meshgrid(wv, Tv)
        Eff = mc.eff_map(W, T) * 100
        Eff = np.where(np.abs(T) <= mc.tlim_v(W), Eff, np.nan)
        cs = ax.contourf(wv / RPM, Tv, Eff, levels=[70, 80, 85, 88, 90, 92, 94, 96, 98],
                         cmap="Greens", extend="min", alpha=0.85)
        ax.plot(wv / RPM, mc.tlim_v(wv), color=C_TXT, lw=1.2)
        ax.plot(wv / RPM, -mc.tlim_v(wv), color=C_TXT, lw=1.2)
        ax.scatter(lg[wcol] / RPM, lg[Tcol], s=5, color=col, alpha=0.3, edgecolors="none")
        _style_ax(ax, f"{name} torque [Nm]", f"{name} speed [rpm]")
        ax.set_title(f"{name} efficiency [%] with operating points", fontsize=9.5, color=C_TXT, loc="left")
        cb = fig.colorbar(cs, ax=ax, pad=0.01, fraction=0.05)
        cb.ax.tick_params(labelsize=7)
        axes.append(cb.ax)
    return axes


def draw_energy(fig, rect, res):
    x0, y0, w, h = rect
    rows = energy_table(res)
    axes = []
    ax = fig.add_axes([x0 + w * 0.24, y0 + h * 0.08, w * 0.40, h * 0.86])
    axes.append(ax)
    sink = [(it, v) for g, it, v in rows if g != "Source"]
    sink = [s for s in sink if abs(s[1]) > 1e-5]
    sink.sort(key=lambda r: r[1])
    labels = [s[0] for s in sink]
    vals = [s[1] for s in sink]
    cmap = {"Engine losses": C_ENG, "MG1 losses": C_MG1, "MG2 losses": C_MG2,
            "Battery resistive losses": C_BAT, "Friction brakes": C_BRK}
    cols = [cmap.get(l, C_REF) for l in labels]
    ax.barh(range(len(vals)), vals, color=cols, height=0.7, edgecolor="none")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8, color=C_TXT)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=C_TXT2)
    ax.axvline(0, color="#b9b8b2", lw=0.8)
    _style_ax(ax, None, "Energy [kWh]")
    ax.set_title("Where the energy went (system level)", fontsize=10, color=C_TXT, loc="left")
    xm = max(abs(min(vals + [0])), max(vals + [0]))
    ax.set_xlim(min(vals + [0]) * 1.25 - 0.02 * xm, max(vals + [0]) * 1.25)
    # text panel
    ax2 = fig.add_axes([x0 + w * 0.70, y0 + h * 0.08, w * 0.30, h * 0.86])
    axes.append(ax2)
    ax2.axis("off")
    src = [(it, v) for g, it, v in rows if g == "Source"]
    lines = ["SOURCES (kWh)"]
    for it, v in src:
        lines.append(f"  {it:<34s}{v:8.3f}")
    lines.append("")
    lines.append("COMPONENT EFFICIENCIES (%)")
    for it, v in efficiency_table(res):
        lines.append(f"  {it:<34s}{v:8.1f}" if not math.isnan(v) else f"  {it:<34s}{'-':>8s}")
    lines.append("")
    lines.append("TIME IN MODE (%)")
    tot = sum(res.mode_time) or 1
    for m, tt in zip(MODE_NAMES, res.mode_time):
        lines.append(f"  {m:<34s}{100 * tt / tot:8.1f}")
    s = res.summary
    lines += ["", "ENERGY BALANCE",
              f"  closure error {s['energy_balance_err_pct']:+.3f} % of throughput",
              f"  regen recovered {s['regen_recovered_kWh']:.3f} kWh (elec.)"]
    ax2.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.6,
             color=C_TXT, transform=ax2.transAxes)
    return axes


# ---- dashboard primitives ---------------------------------------------------
def make_gauge(ax, vmin, vmax, label, unit, major, color):
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.85, 1.15); ax.set_aspect("equal"); ax.axis("off")
    a0, a1 = math.radians(225), math.radians(-45)
    th = np.linspace(a0, a1, 100)
    ax.plot(np.cos(th), np.sin(th), color="#b9b8b2", lw=3, solid_capstyle="round")
    val = vmin
    while val <= vmax + 1e-9:
        f = (val - vmin) / (vmax - vmin)
        a = a0 + (a1 - a0) * f
        ax.plot([0.86 * math.cos(a), math.cos(a)], [0.86 * math.sin(a), math.sin(a)], color=C_TXT2, lw=1)
        lab = f"{val / 1000:g}" if abs(vmax) >= 2000 else f"{val:g}"
        ax.text(0.70 * math.cos(a), 0.70 * math.sin(a), lab, ha="center", va="center",
                fontsize=6.5, color=C_TXT2)
        val += major
    arc, = ax.plot([], [], color=color, lw=4, solid_capstyle="round", alpha=0.85)
    needle, = ax.plot([0, 0], [0, 0.8], color=C_TXT, lw=2)
    ax.add_patch(__import__("matplotlib").patches.Circle((0, 0), 0.06, color=C_TXT))
    txt = ax.text(0, -0.45, "", ha="center", va="center", fontsize=11, color=C_TXT, weight="bold")
    scale_note = " x1000" if abs(vmax) >= 2000 else ""
    ax.text(0, -0.72, f"{label} [{unit}]{scale_note}", ha="center", va="center", fontsize=7.5, color=C_TXT2)
    g = dict(vmin=vmin, vmax=vmax, a0=a0, a1=a1, needle=needle, txt=txt, arc=arc, fmt="{:.0f}")
    return g


def set_gauge(g, val):
    f = min(max((val - g["vmin"]) / (g["vmax"] - g["vmin"]), 0), 1)
    a = g["a0"] + (g["a1"] - g["a0"]) * f
    g["needle"].set_data([0, 0.8 * math.cos(a)], [0, 0.8 * math.sin(a)])
    f0 = min(max((0 - g["vmin"]) / (g["vmax"] - g["vmin"]), 0), 1)
    th = np.linspace(g["a0"] + (g["a1"] - g["a0"]) * f0, a, 40)
    g["arc"].set_data(np.cos(th), np.sin(th))
    g["txt"].set_text(g["fmt"].format(val))


def draw_power_flow(ax, d):
    """d: dict of instantaneous powers [kW] and states."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax.cla(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = {"Engine": (0.10, 0.62, C_ENG), "Power split": (0.42, 0.62, C_REF),
             "MG1": (0.42, 0.93, C_MG1), "Ring / MG2": (0.72, 0.62, C_MG2),
             "Wheels": (0.93, 0.62, C_TXT2), "Battery": (0.42, 0.12, C_BAT)}
    for name, (x, y, c) in boxes.items():
        ax.add_patch(FancyBboxPatch((x - 0.085, y - 0.055), 0.17, 0.11,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="white", ec=c, lw=1.6))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.5, color=C_TXT)

    def arrow(p0, p1, P, col, label_pos, label=None):
        thr = 0.2
        if abs(P) < thr:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#dedcd6", lw=1.2, ls="--")
        else:
            a, b = (p0, p1) if P > 0 else (p1, p0)
            lw = 1.2 + 4.0 * min(abs(P) / 60.0, 1.0)
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                         color=col, lw=lw, shrinkA=0, shrinkB=0))
        ax.text(*label_pos, f"{P:+.1f} kW" if label is None else label, fontsize=7,
                color=C_TXT2, ha="center", va="center")

    arrow((0.185, 0.62), (0.335, 0.62), d["P_eng"], C_ENG, (0.26, 0.68))
    arrow((0.42, 0.675), (0.42, 0.875), -d["P_g"], C_MG1, (0.50, 0.78))     # + = to MG1
    arrow((0.505, 0.62), (0.635, 0.62), d["P_psd_out"], C_REF, (0.57, 0.68))
    arrow((0.805, 0.62), (0.845, 0.62), d["P_wheel"], C_TXT2, (0.84, 0.50))
    # electrical paths
    # electrical paths: arrows point in the direction energy flows
    c1 = C_MG1 if abs(d["P_ge"]) > 0.2 else "#dedcd6"
    ax.plot([0.335, 0.25, 0.25], [0.93, 0.93, 0.12], color=c1, lw=1.0)
    arrow((0.25, 0.12), (0.335, 0.12), -d["P_ge"], C_MG1, (0.175, 0.30),
          f"MG1 el\n{abs(d['P_ge']):.1f} kW")
    c2 = C_MG2 if abs(d["P_me"]) > 0.2 else "#dedcd6"
    ax.plot([0.72, 0.72], [0.565, 0.12], color=c2, lw=1.0)
    arrow((0.505, 0.12), (0.72, 0.12), d["P_me"], C_MG2, (0.80, 0.30),
          f"MG2 el\n{abs(d['P_me']):.1f} kW")
    ax.text(0.42, 0.0, f"Battery {d['P_batt']:+.1f} kW (+ = discharge)   SOC {d['soc'] * 100:.1f} %",
            ha="center", va="bottom", fontsize=7.5, color=C_TXT)
    ax.text(0.10, 0.48, f"fuel {d['mdot'] * 1000:.2f} g/s", ha="center", fontsize=7, color=C_TXT2)
    ax.text(0.0, 1.03, "Power flow", fontsize=9.5, color=C_TXT, transform=ax.transAxes)


def draw_lever(ax, res, i):
    lg = res.log
    S, R = res.p.sun_teeth, res.p.ring_teeth
    xs, xc, xr = 0.0, R / (S + R), 1.0
    ws, wc, wr = lg["w_g"][i] / RPM, lg["w_e"][i] / RPM, lg["w_m"][i] / RPM / res.p.mg2_ratio
    lim = max(res.p.mg1_rpm_max, res.p.eng_rpm_max) * 1.05
    ax.cla()
    ax.set_xlim(-0.25, 1.25); ax.set_ylim(-lim, lim)
    for x, c, nm in ((xs, C_MG1, "Sun\nMG1"), (xc, C_ENG, "Carrier\nEngine"), (xr, C_MG2, "Ring\nMG2/out")):
        ax.plot([x, x], [-lim * 0.95, lim * 0.95], color="#cfcdc6", lw=6, solid_capstyle="round", zorder=1)
        ax.text(x, -lim * 1.0, nm, ha="center", va="top", fontsize=7, color=C_TXT2)
    ax.axhline(0, color="#b9b8b2", lw=0.8)
    ax.plot([xs, xc, xr], [ws, wc, wr], color=C_TXT, lw=1.6, zorder=2)
    for x, y, c in ((xs, ws, C_MG1), (xc, wc, C_ENG), (xr, wr, C_MG2)):
        ax.scatter([x], [y], s=60, color=c, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(x + 0.06, y, f"{y:.0f}", fontsize=7, color=C_TXT, va="center")
    _style_ax(ax, "Speed [rpm]")
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title("Lever diagram (planetary speeds)", fontsize=9.5, color=C_TXT, loc="left")


# =============================================================================
# GUI
# =============================================================================
GUI_PARAMS = [
    ("mass", "Mass", "kg"), ("Cd", "Cd", "-"), ("A", "Frontal area", "m²"),
    ("Crr", "Rolling coeff.", "-"), ("r_wheel", "Tire radius", "m"),
    ("sun_teeth", "Sun teeth", "-"), ("ring_teeth", "Ring teeth", "-"),
    ("mg2_ratio", "MG2 reduction", "-"), ("final_drive", "Final drive", "-"),
    ("eng_Pmax_kW", "Engine Pmax", "kW"), ("mg1_Pmax_kW", "MG1 Pmax", "kW"),
    ("mg2_Pmax_kW", "MG2 Pmax", "kW"), ("batt_Ah", "Battery cap.", "Ah"),
    ("soc_init", "Initial SOC", "-"), ("soc_target", "Target SOC", "-"),
    ("ev_vmax_kmh", "EV max speed", "km/h"), ("ev_Pmax_kW", "EV max power", "kW"),
    ("grade_pct", "Road grade", "%"), ("wind_kmh", "Head wind", "km/h"),
    ("T_amb_C", "Ambient temp.", "°C"), ("aux_kW", "Aux load", "kW"),
    ("const_kmh", "Const. speed", "km/h"),
]


class App:
    def __init__(self, p: Params, cycle: str, csv_path=""):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, TextBox, Slider
        self.plt = plt
        self.p = p
        self.csv_path = csv_path
        self.res = None
        self.view = "Results"
        self.view_axes = []
        self.playing = False
        self.dash = None

        plt.rcParams["font.family"] = "DejaVu Sans"
        self.fig = fig = plt.figure(figsize=(16, 9.4), facecolor="white")
        try:
            fig.canvas.manager.set_window_title("HEV Input Power-Split Simulator")
        except Exception:
            pass
        fig.text(0.008, 0.975, "HEV Input Power-Split", fontsize=13, weight="bold", color=C_TXT)
        fig.text(0.008, 0.958, "Prius-type planetary hybrid", fontsize=8.5, color=C_TXT2)

        # --- scenario radio
        fig.text(0.008, 0.938, "SCENARIO", fontsize=7.5, color=C_TXT2, weight="bold")
        axr = fig.add_axes([0.004, 0.785, 0.115, 0.15], frameon=False)
        idx = CYCLES.index(cycle) if cycle in CYCLES else 0
        self.radio = RadioButtons(axr, CYCLES, active=idx)
        for lab in self.radio.labels:
            lab.set_fontsize(7.5)
        axcsv = fig.add_axes([0.034, 0.762, 0.085, 0.02])
        self.tb_csv = TextBox(axcsv, "CSV ", initial=csv_path)
        self.tb_csv.label.set_fontsize(7)
        self.tb_csv.text_disp.set_fontsize(7)

        # --- parameter boxes (two columns)
        fig.text(0.008, 0.742, "PARAMETERS", fontsize=7.5, color=C_TXT2, weight="bold")
        self.tbs = {}
        rows = (len(GUI_PARAMS) + 1) // 2
        for i, (key, lab, unit) in enumerate(GUI_PARAMS):
            col, r = divmod(i, rows)
            x = 0.062 + col * 0.108
            y = 0.718 - r * 0.0265
            ax = fig.add_axes([x, y, 0.040, 0.021])
            tb = TextBox(ax, f"{lab} ", initial=f"{getattr(p, key):g}")
            tb.label.set_fontsize(7)
            tb.text_disp.set_fontsize(7.5)
            fig.text(x + 0.042, y + 0.006, unit, fontsize=6.5, color=C_TXT2)
            self.tbs[key] = tb

        # --- buttons
        yb = 0.718 - rows * 0.0265 - 0.012
        self.btn_run = Button(fig.add_axes([0.008, yb - 0.04, 0.105, 0.040]), "▶  Run simulation",
                              color="#2a78d6", hovercolor="#3987e5")
        self.btn_run.label.set_color("white"); self.btn_run.label.set_fontsize(9)
        self.btn_run.on_clicked(self.on_run)
        self.btn_exp = Button(fig.add_axes([0.118, yb - 0.04, 0.095, 0.040]), "Export CSV/XLSX",
                              color="#f1f0ec", hovercolor="#e4e3df")
        self.btn_exp.label.set_fontsize(8)
        self.btn_exp.on_clicked(self.on_export)
        self.btn_def = Button(fig.add_axes([0.118, yb - 0.085, 0.095, 0.035]), "Reset defaults",
                              color="#f1f0ec", hovercolor="#e4e3df")
        self.btn_def.label.set_fontsize(8)
        self.btn_def.on_clicked(self.on_defaults)

        fig.text(0.008, yb - 0.105, "VIEW", fontsize=7.5, color=C_TXT2, weight="bold")
        self.view_btns = {}
        for j, name in enumerate(["Results", "Maps", "Energy", "Dashboard"]):
            b = Button(fig.add_axes([0.008 + (j % 2) * 0.104, yb - 0.150 - (j // 2) * 0.042,
                                     0.100, 0.036]), name, color="#f1f0ec", hovercolor="#e4e3df")
            b.label.set_fontsize(8.5)
            b.on_clicked(lambda ev, n=name: self.set_view(n))
            self.view_btns[name] = b

        # --- summary + status
        self.status = fig.text(0.008, 0.012, "Press Run.", fontsize=7.5, color=C_TXT2)
        self.summary_txt = fig.text(0.25, 0.965, "", fontsize=9, color=C_TXT, va="center",
                                    family="monospace")
        self.main_rect = [0.27, 0.10, 0.71, 0.80]

        # --- dashboard playback controls (hidden unless Dashboard)
        self.ax_slider = fig.add_axes([0.32, 0.035, 0.50, 0.02])
        self.slider = Slider(self.ax_slider, "Time [s] ", 0, 1, valinit=0, color=C_MG2)
        self.slider.label.set_fontsize(8)
        self.slider.on_changed(self.on_slider)
        self.ax_play = fig.add_axes([0.89, 0.028, 0.06, 0.034])
        self.btn_play = Button(self.ax_play, "Play ▶", color="#f1f0ec", hovercolor="#e4e3df")
        self.btn_play.label.set_fontsize(8)
        self.btn_play.on_clicked(self.on_play)
        self._show_playback(False)
        self.timer = fig.canvas.new_timer(interval=80)
        self.timer.add_callback(self.on_tick)

        self.on_run(None)

    # ----------------------------------------------------------------- helpers
    def _show_playback(self, show):
        self.ax_slider.set_visible(show)
        self.ax_play.set_visible(show)

    def read_params(self):
        p = Params(**asdict(self.p))
        for key, tb in self.tbs.items():
            try:
                setattr(p, key, float(tb.text))
            except ValueError:
                raise ValueError(f"'{tb.text}' is not a number for {key}")
        return p

    def on_defaults(self, _):
        d = Params()
        for key, tb in self.tbs.items():
            tb.set_val(f"{getattr(d, key):g}")
        self.status.set_text("Defaults restored - press Run.")
        self.fig.canvas.draw_idle()

    def on_run(self, _):
        try:
            p = self.read_params()
            name = self.radio.value_selected
            scen = make_scenario(name, p, self.tb_csv.text.strip())
        except Exception as e:  # noqa
            self.status.set_text(f"Error: {e}")
            self.fig.canvas.draw_idle()
            return
        self.status.set_text(f"Running {name} ...")
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        self.p = p
        self.res = simulate(p, scen)
        s = self.res.summary
        line1 = (f"{s['scenario']}:  {s['distance_km']:.2f} km   "
                 f"fuel {s['fuel_L100km']:.2f} L/100 km  (SOC-corr. {s['fuel_L100km_soc_corrected']:.2f})   "
                 f"CO₂ {s['co2_g_km']:.0f} g/km   SOC {s['soc_start'] * 100:.0f}→{s['soc_end'] * 100:.1f} %")
        line2 = (f"engine on {s['engine_on_pct']:.0f} % ({s['engine_starts']} starts, avg eff {s['engine_avg_eff_pct']:.1f} %)   "
                 f"EV {s['ev_pct']:.0f} %   regen {s['regen_recovered_kWh']:.3f} kWh   "
                 f"speed err RMS {s['speed_err_rms_kmh']:.2f} km/h   balance {s['energy_balance_err_pct']:+.2f} %")
        if s.get("t_0_100_s") is not None:
            line2 = f"0-100 km/h {s['t_0_100_s']:.1f} s   " + (
                f"0-130 km/h {self.res.t130:.1f} s   " if self.res.t130 else "") + line2
        self.summary_txt.set_text(line1 + "\n" + line2)
        self.status.set_text(f"Done: {name}, {len(self.res.log['t'])} samples.")
        self.slider.valmax = float(self.res.log["t"][-1])
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.set_val(0)
        self.set_view(self.view)

    def on_export(self, _):
        if self.res is None:
            return
        files = export_results(self.res, os.path.join(os.getcwd(), "hev_ips_results"))
        self.status.set_text("Exported: " + ", ".join(os.path.basename(f) for f in files))
        self.fig.canvas.draw_idle()

    def clear_view(self):
        for ax in self.view_axes:
            try:
                ax.remove()
            except Exception:
                pass
        self.view_axes = []
        self.dash = None

    def set_view(self, name):
        if self.res is None:
            return
        self.view = name
        self.playing = False
        self.timer.stop()
        self.btn_play.label.set_text("Play ▶")
        for n, b in self.view_btns.items():
            b.color = "#d6e4f7" if n == name else "#f1f0ec"
            b.ax.set_facecolor(b.color)
        self.clear_view()
        r = self.main_rect
        if name == "Results":
            self.view_axes = draw_results(self.fig, r, self.res)
        elif name == "Maps":
            self.view_axes = draw_maps(self.fig, r, self.res)
        elif name == "Energy":
            self.view_axes = draw_energy(self.fig, r, self.res)
        elif name == "Dashboard":
            self.build_dashboard()
        self._show_playback(name == "Dashboard")
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------- dashboard
    def build_dashboard(self):
        fig, res, p = self.fig, self.res, self.p
        x0, y0, w, h = self.main_rect
        gw = w / 4
        axs = [fig.add_axes([x0 + i * gw, y0 + h * 0.60, gw * 0.95, h * 0.40]) for i in range(4)]
        g_speed = make_gauge(axs[0], 0, 180, "Vehicle speed", "km/h", 20, C_MG2)
        g_eng = make_gauge(axs[1], 0, 6000, "Engine", "rpm", 1000, C_ENG)
        g_mg1 = make_gauge(axs[2], -10000, 10000, "MG1", "rpm", 2500, C_MG1)
        g_mg2 = make_gauge(axs[3], 0, 14000, "MG2", "rpm", 2000, C_MG2)
        ax_lev = fig.add_axes([x0, y0 + h * 0.02, w * 0.28, h * 0.50])
        ax_pf = fig.add_axes([x0 + w * 0.33, y0 + h * 0.02, w * 0.38, h * 0.50])
        ax_tr = fig.add_axes([x0 + w * 0.77, y0 + h * 0.30, w * 0.23, h * 0.22])
        ax_soc = fig.add_axes([x0 + w * 0.77, y0 + h * 0.04, w * 0.23, h * 0.16])
        lg = res.log
        _mode_spans(ax_tr, lg["t"], lg["mode"], alpha=0.25)
        ax_tr.plot(lg["t"], lg["v_ref"] * 3.6, color=C_REF, lw=1.2)
        ax_tr.plot(lg["t"], lg["v"] * 3.6, color=C_MG2, lw=0.9)
        cur = ax_tr.axvline(0, color=C_TXT, lw=1)
        _style_ax(ax_tr, "km/h"); ax_tr.set_xlim(lg["t"][0], lg["t"][-1])
        ax_tr.set_title("Speed trace & mode", fontsize=9, color=C_TXT, loc="left")
        ax_soc.plot(lg["t"], lg["soc"] * 100, color=C_BAT, lw=1.2)
        cur2 = ax_soc.axvline(0, color=C_TXT, lw=1)
        _style_ax(ax_soc, "SOC %", "Time [s]"); ax_soc.set_xlim(lg["t"][0], lg["t"][-1])
        mode_txt = fig.text(x0 + w * 0.77, y0 + h * 0.56, "", fontsize=10, weight="bold", color=C_TXT)
        self.dash = dict(g=[g_speed, g_eng, g_mg1, g_mg2], lev=ax_lev, pf=ax_pf, cur=[cur, cur2],
                         mode_txt=mode_txt)
        self.view_axes = axs + [ax_lev, ax_pf, ax_tr, ax_soc]
        self._dash_text = mode_txt
        self.update_dashboard(self.slider.val)

    def update_dashboard(self, tsel):
        if not self.dash:
            return
        res, lg = self.res, self.res.log
        i = int(min(max(np.searchsorted(lg["t"], tsel), 0), len(lg["t"]) - 1))
        g = self.dash["g"]
        set_gauge(g[0], lg["v"][i] * 3.6)
        set_gauge(g[1], lg["w_e"][i] / RPM)
        set_gauge(g[2], lg["w_g"][i] / RPM)
        set_gauge(g[3], lg["w_m"][i] / RPM)
        draw_lever(self.dash["lev"], res, i)
        P_eng, P_g = lg["P_eng"][i] / 1e3, lg["P_g"][i] / 1e3
        d = dict(P_eng=P_eng, P_g=P_g, P_psd_out=P_eng + P_g, P_wheel=lg["P_wheel"][i] / 1e3,
                 P_ge=lg["P_ge"][i] / 1e3, P_me=lg["P_me"][i] / 1e3, P_batt=lg["P_batt"][i] / 1e3,
                 P_batt_bus=(lg["P_ge"][i] + lg["P_me"][i]) / 1e3, soc=lg["soc"][i], mdot=lg["mdot"][i])
        draw_power_flow(self.dash["pf"], d)
        for c in self.dash["cur"]:
            c.set_xdata([lg["t"][i], lg["t"][i]])
        m = int(lg["mode"][i])
        self.dash["mode_txt"].set_text(f"{MODE_NAMES[m]}   t = {lg['t'][i]:.0f} s")
        self.dash["mode_txt"].set_color(C_TXT)

    def on_slider(self, val):
        if self.view == "Dashboard":
            self.update_dashboard(val)
            self.fig.canvas.draw_idle()

    def on_play(self, _):
        if self.view != "Dashboard":
            return
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause ❚❚" if self.playing else "Play ▶")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def on_tick(self):
        if not self.playing:
            return
        nv = self.slider.val + 1.0          # 1 s simulated per frame (~12x real time)
        if nv > self.slider.valmax:
            nv = 0.0
        self.slider.set_val(nv)

    def show(self):
        self.plt.show()


# =============================================================================
# Snapshots (render every view to PNG without a display)
# =============================================================================
def save_snapshots(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    out = []
    rect = [0.07, 0.07, 0.90, 0.84]
    for name, fn in (("results", draw_results), ("maps", draw_maps), ("energy", draw_energy)):
        fig = plt.figure(figsize=(14, 9), facecolor="white")
        fn(fig, rect, res)
        fig.suptitle(f"{res.scen.name} - {name}", x=0.07, ha="left", fontsize=12, color=C_TXT)
        f = os.path.join(outdir, f"{name}.png")
        fig.savefig(f, dpi=90); plt.close(fig); out.append(f)
    return out


# =============================================================================
# CLI
# =============================================================================
def parse_set(p, items):
    names = {f.name for f in fields(Params)}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown parameter '{k}' (see --list-params)")
        setattr(p, k, float(v))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="HEV input power-split simulator")
    ap.add_argument("--cycle", default="WLTC",
                    help="WLTC | UDDS | HWFET | US06 | NEDC | WOT | CONST | CSV")
    ap.add_argument("--csv", default="", help="custom cycle CSV: time[s], speed[km/h], grade[%%]")
    ap.add_argument("--set", action="append", help="override parameter, e.g. --set mass=1600")
    ap.add_argument("--no-gui", action="store_true", help="run headless and print summary")
    ap.add_argument("--export", default="", help="directory for CSV/XLSX export (headless)")
    ap.add_argument("--snapshot", default="", help="directory for PNG views (headless)")
    ap.add_argument("--list-params", action="store_true")
    a = ap.parse_args(argv)

    p = parse_set(Params(), a.set)
    if a.list_params:
        for f in fields(Params):
            print(f"  {f.name:<16s} = {getattr(p, f.name):<10g} {PARAM_HELP.get(f.name, '')}")
        return
    alias = {"WLTC": "WLTC class 3b", "UDDS": "UDDS (FTP-72)", "FTP": "UDDS (FTP-72)",
             "HWFET": "HWFET", "US06": "US06", "NEDC": "NEDC", "WOT": "WOT 0-130 km/h",
             "CONST": "Constant speed", "CSV": "Custom CSV"}
    cyc = alias.get(a.cycle.upper(), a.cycle)
    if a.csv and cyc not in CYCLES:
        cyc = "Custom CSV"

    if a.no_gui or a.export or a.snapshot:
        scen = make_scenario(cyc, p, a.csv)
        res = simulate(p, scen)
        print_summary(res)
        if a.export:
            for f in export_results(res, a.export):
                print("  wrote", f)
        if a.snapshot:
            for f in save_snapshots(res, a.snapshot):
                print("  wrote", f)
        return res
    App(p, cyc, a.csv).show()


if __name__ == "__main__":
    main()
