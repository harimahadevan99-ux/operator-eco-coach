# Operator Eco-Driving Coach

![tests](https://github.com/harimahadevan99-ux
/operator-eco-coach/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A behavioral-analytics pipeline that scores excavator operators on fuel
efficiency, cycle by cycle, from raw telemetry — the same kind of coaching
tool commercial systems like Cat SmartFuel or Komatsu KOMTRAX build on top
of. Built and tested entirely on synthetic telemetry, so it needs no
proprietary data or live machine access.

## Why this matters

Two operators on the *identical* machine can differ 20-30% in fuel burn per
ton moved, purely from throttle habits and idle discipline. Fuel modeling
(Example 1 of this series) tells you a machine is burning more than
expected; this tells you *why*, at the operator level, in terms a
supervisor or the operator themselves can act on.

## Architecture

```
generate_operator_data.py --> per-second telemetry --> detect_cycles() --> per-cycle features
                                                                                  |
                                                            +---------------------+----------------------+
                                                            v                                              v
                                                    eco_score.py                                     train.py
                                            (transparent formula +                          (supervised classifier,
                                             unsupervised clustering                          held-out operators)
                                             sanity check)
                                                            |
                                                            v
                                                 dashboard/app.py (replay + ranking)
```

Each cycle = dig → swing_loaded → dump → swing_empty, optionally preceded by
an idle gap. Cycle boundaries are detected with a simple, auditable
rule (not a black box) — see `src/features/build_features.py`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic operator telemetry (8 operators, 2 behavior styles)
python -m src.simulation.generate_operator_data --out data/operator_telemetry.csv

# 2. See the eco-score ranking + unsupervised clustering sanity check
python -m src.models.eco_score --data data/operator_telemetry.csv

# 3. Train a classifier that predicts style from a new operator's cycles
python -m src.models.train --data data/operator_telemetry.csv --out models_artifacts/

# 4. Interactive replay dashboard
streamlit run src/dashboard/app.py

pytest -v
```

## Results

On 1,200 detected cycles across 8 synthetic operators (4 "efficient", 4
"aggressive"):

| Metric | Efficient operators | Aggressive operators |
|---|---|---|
| Avg EcoScore (0-100) | ~88 | ~61-67 |
| Avg idle % per cycle | ~1.8% | ~11-17% |
| Avg fuel per cycle | ~0.037 gal | ~0.057 gal |

- **Supervised classifier** (Gradient Boosting), validated on 2 operators
  held out entirely from training: **99.3% accuracy** predicting style from
  cycle features alone.
- **Unsupervised clustering** (KMeans, k=2, no labels used): **~66%
  purity** against the true styles — real signal, clearly beating a 50%
  coin flip, but far from perfect. The reason is itself a useful finding:
  roughly a third of "aggressive" cycles have no idle gap at all and look
  similar to efficient cycles on a single-cycle basis — it's the
  *aggregate pattern* across many cycles (which the continuous EcoScore
  captures) that reliably separates styles, not any single cycle in
  isolation. This is exactly why a per-operator rolling EcoScore is a
  better coaching tool than a one-shot cycle classification.

## Moving from synthetic to real data

Keep the same per-second schema
(`t, cycle_id, mode, load_factor, rpm, hyd_pressure_bar, fuel_rate_gal_hr,
operator_id`) and feed in a real per-second CAN/J1939 export with a mode
signal (or derive "mode" from hydraulic pressure + boom/bucket angle
thresholds if no explicit state signal exists). No other code should need
to change.

**Do not** feed this pipeline real operator data from an employer without
explicit authorization — operator performance data is sensitive.

## Next steps

- Real-time streaming version (MQTT) feeding the dashboard live instead of replaying a CSV
- Add a per-operator trend view (is this operator improving over weeks?)
- Feed each operator's average EcoScore into the Example 3 dispatch
  simulation as a per-truck/excavator efficiency multiplier
