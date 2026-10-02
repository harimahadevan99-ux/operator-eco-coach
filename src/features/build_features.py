"""
Turns raw per-second telemetry into per-cycle behavioral features.

Cycle detection is intentionally simple and rule-based (not a black box):
a new cycle starts the moment the machine leaves "swing_empty" and enters
either an idle gap or the next "dig" phase. This is the same kind of logic
you'd implement against a real machine's mode/state signal -- more robust
general-purpose segmentation (e.g. change-point detection on raw signals)
is a natural upgrade once this rule-based version is working end to end.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CYCLE_FEATURE_COLUMNS = [
    "cycle_time_s", "idle_pct", "rpm_std", "hyd_pressure_peak",
    "mean_load_factor", "fuel_used_gal",
]


def detect_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `detected_cycle_id` column, computed independently per operator_id."""
    df = df.sort_values(["operator_id", "t"]).copy()
    out_frames = []
    for operator_id, g in df.groupby("operator_id", sort=False):
        g = g.reset_index(drop=True)
        cycle_ids = np.zeros(len(g), dtype=int)
        current = 0
        prev_mode = None
        for i, mode in enumerate(g["mode"].values):
            if prev_mode == "swing_empty" and mode in ("idle", "dig"):
                current += 1
            cycle_ids[i] = current
            prev_mode = mode
        g["detected_cycle_id"] = cycle_ids
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def extract_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses second-level rows into one row per detected cycle."""
    df = detect_cycles(df)

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        idle_pct = 100 * (g["mode"] == "idle").mean()
        fuel_used_gal = g["fuel_rate_gal_hr"].sum() / 3600.0  # gal/hr * seconds -> gal
        return pd.Series({
            "cycle_time_s": n,
            "idle_pct": idle_pct,
            "rpm_std": g["rpm"].std() if n > 1 else 0.0,
            "hyd_pressure_peak": g["hyd_pressure_bar"].max(),
            "mean_load_factor": g["load_factor"].mean(),
            "fuel_used_gal": fuel_used_gal,
            "archetype": g["archetype"].iloc[0],
        })

    cycles = (
        df.groupby(["operator_id", "detected_cycle_id"], sort=False)
        .apply(_agg)
        .reset_index()
    )
    # every detected cycle is a complete dig->swing->dump->swing sequence by
    # construction of the generator, so no partial-cycle trimming is needed here.
    return cycles
