"""
Synthetic per-second operator telemetry generator.

Every excavator cycle is modeled as 4 phases: dig -> swing_loaded -> dump ->
swing_empty -> (back to dig). Two operator archetypes are simulated:

  "efficient":  smooth throttle, short/rare idle gaps between cycles,
                consistent phase timing.
  "aggressive": higher RPM volatility (throttle jabbing), frequent and
                longer idle gaps between cycles, less consistent timing.

This mirrors the real-world finding that operator behavior alone can swing
fuel-per-ton 20-30% between two operators on an identical machine.

Run:
    python -m src.simulation.generate_operator_data --out data/operator_telemetry.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

RATED_HP = 220
BSFC = 0.37
UNIT_FACTOR = 6.0

# (mean_seconds, std_seconds, load_factor_mean, load_factor_std)
PHASE_PARAMS = {
    "efficient": {
        "dig":          (8, 1.0, 0.55, 0.03),
        "swing_loaded": (6, 1.0, 0.45, 0.03),
        "dump":         (4, 0.8, 0.25, 0.03),
        "swing_empty":  (5, 1.0, 0.35, 0.03),
    },
    "aggressive": {
        "dig":          (10, 2.5, 0.60, 0.08),
        "swing_loaded": (7, 2.0, 0.55, 0.09),
        "dump":         (5, 1.8, 0.30, 0.08),
        "swing_empty":  (6, 2.0, 0.45, 0.09),
    },
}
IDLE_GAP_PROB = {"efficient": 0.08, "aggressive": 0.35}
IDLE_GAP_RANGE = {"efficient": (3, 10), "aggressive": (10, 45)}
PHASE_ORDER = ["dig", "swing_loaded", "dump", "swing_empty"]


def _simulate_second_level(archetype: str, n_cycles: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    t = 0
    params = PHASE_PARAMS[archetype]
    for cycle_id in range(n_cycles):
        # occasional idle gap before the cycle starts (operator repositioning,
        # waiting on a truck, distraction, etc.)
        if rng.random() < IDLE_GAP_PROB[archetype]:
            lo, hi = IDLE_GAP_RANGE[archetype]
            idle_secs = int(rng.integers(lo, hi))
            for _ in range(idle_secs):
                rows.append(_make_row(t, cycle_id, "idle", 0.12, 0.02, rng))
                t += 1

        for phase in PHASE_ORDER:
            mean_s, std_s, lf_mean, lf_std = params[phase]
            duration = max(1, int(round(rng.normal(mean_s, std_s))))
            for _ in range(duration):
                rows.append(_make_row(t, cycle_id, phase, lf_mean, lf_std, rng))
                t += 1

    df = pd.DataFrame(rows)
    df["archetype"] = archetype
    return df


def _make_row(t, cycle_id, mode, lf_mean, lf_std, rng):
    load_factor = float(np.clip(rng.normal(lf_mean, lf_std), 0.05, 0.95))
    rpm = 800 + load_factor * 1400 + rng.normal(0, 25)
    hyd_pressure = 50 + load_factor * 250 + rng.normal(0, 12)
    fuel_rate = RATED_HP * load_factor * BSFC / UNIT_FACTOR + rng.normal(0, 0.08)
    return {
        "t": t, "cycle_id": cycle_id, "mode": mode,
        "load_factor": load_factor, "rpm": rpm,
        "hyd_pressure_bar": max(hyd_pressure, 0),
        "fuel_rate_gal_hr": max(fuel_rate, 0.1),
    }


def generate_operators(n_cycles_per_operator: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    operators = [
        ("OP-01", "efficient"), ("OP-02", "efficient"),
        ("OP-03", "efficient"), ("OP-04", "efficient"),
        ("OP-05", "aggressive"), ("OP-06", "aggressive"),
        ("OP-07", "aggressive"), ("OP-08", "aggressive"),
    ]
    frames = []
    for operator_id, archetype in operators:
        df = _simulate_second_level(archetype, n_cycles_per_operator, rng)
        df["operator_id"] = operator_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/operator_telemetry.csv")
    parser.add_argument("--cycles", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = generate_operators(n_cycles_per_operator=args.cycles, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} second-level rows for {df.operator_id.nunique()} operators to {args.out}")


if __name__ == "__main__":
    main()
