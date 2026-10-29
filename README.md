# Kia P2 Hybrid · 6-speed DCT Simulator (Streamlit)

Parallel (P2) full hybrid in Kia/Hyundai's "TMED" layout — engine → engine clutch K0 →
traction motor → 6-speed dry dual-clutch transmission → wheels, plus a belt
starter-generator (HSG) — structured like MathWorks' *HEV P2 Reference Application*
(ECMS energy management, speed/load shift schedule).

Default parameters: 2nd-generation Kia Niro HEV from Kia's published specifications
(1.6 GDi 77.2 kW / 144 Nm, motor 32 kW / 170 Nm, HSG 8 kW, 240 V 1.32 kWh Li-ion
polymer, 6DCT 3.600 / 2.217 / 1.371 / 0.953 / 0.978 / 0.767 with final drives
4.438 (gears 1-4) and 3.227 (gears 5-6), Cd 0.29, 205/60 R16).

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web interface (entry point) |
| `hev_p2dct_sim.py` | P2 + DCT model, controllers, energy analysis; also a command-line tool |
| `hev_ips_sim.py` | Shared component library (engine map, motor loss maps, Li-ion pack, drive cycles, gauges) — the same file as the power-split simulator |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme and upload limit |
| `example_cycle.csv` | Sample custom cycle (time s, speed km/h, grade %) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Put this folder in its own GitHub repository (include the hidden `.streamlit/` folder).
2. At https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**,
   select the repository, set **Main file path** to `streamlit_app.py`, Python 3.11+.
3. **Deploy**. Every push redeploys.

## What the model does

* **TCU** – upshift speeds interpolated between light- and full-pedal tables
  (downshift = upshift − hysteresis), engine-protect downshifts when K0 is closed,
  minimum time between shifts.
* **Shifts** – clutch-to-clutch with an inertia phase: the clutch whose slip agrees
  with the demand carries the torque (limited to what the input side can react),
  the motor synchronises the input shaft, the on-coming clutch locks. Slip losses are
  booked in the energy analysis.
* **VCU (ECMS)** – every 0.1 s compares EV against engine-coupled torque splits on
  fuel rate + s × battery chemical power / LHV; s adapts to the SOC error, and the
  SOC target drops with speed to leave room for regeneration. Hysteresis and
  minimum on/off times prevent chattering.
* **Engine start** – HSG cranks, ECU synchronises engine to motor speed, K0 closes.
  Below the coupling speed the engine can run decoupled and charge via the HSG.
* **Regenerative braking** – series: the motor first, friction brakes for the rest.

## Command line

```bash
python hev_p2dct_sim.py --cycle WLTC
python hev_p2dct_sim.py --cycle UDDS --set mass=1550 --set ecms_s0=2.4 --export out/
python hev_p2dct_sim.py --list-params
```

## Notes

Engine and motor maps are analytic approximations, so absolute fuel figures are
indicative (the model is warm-started and has no cold-start or emissions model);
comparisons between settings and controllers are more reliable.
