"""
Segment raw per-second telemetry into work cycles and extract per-cycle
behavioral features.

A "cycle" is one dig -> swing_loaded -> dump -> swing_empty sequence,
already tagged by the simulator's cycle_id. In real telemetry you would not
have this label handed to you -- you'd derive it from contiguous runs of
"non-idle" mode, which is exactly what `detect_cycle_boundaries` does, used
here to double check our segmentation logic is sound before trusting it on
data where we don't already know the answer.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

CYCLE_FEATURE_COLUMNS = [
    "cycle_time_s", "idle_before_s", "rpm_mean", "rpm_std",
    "pressure_mean", "pressure_std", "fuel_used_gal",
]


def detect_cycle_boundaries(df: pd.DataFrame) -> pd.Series:
    """Derive cycle groups from contiguous non-idle runs (no ground-truth label used)."""
    is_working = (df["mode"] != "idle").astype(int)
    # a new cycle starts every time we transition from idle -> working
    cycle_start = (is_working.diff() == 1).astype(int)
    derived_cycle_id = cycle_start.cumsum()
    derived_cycle_id[is_working == 0] = -1  # idle rows belong to no cycle
    return derived_cycle_id


def extract_cycle_features(df: pd.DataFrame, use_derived_segmentation: bool = False) -> pd.DataFrame:
    df = df.sort_values(["operator_id", "t_s"]).copy()

    if use_derived_segmentation:
        df["cycle_group"] = df.groupby("operator_id", group_keys=False).apply(
            lambda g: detect_cycle_boundaries(g)
        )
    else:
        df["cycle_group"] = df["cycle_id"]

    working = df[df["cycle_group"] >= 0]
    idle = df[df["mode"] == "idle"]

    cycle_stats = (
        working.groupby(["operator_id", "operator_style", "cycle_group"])
        .agg(
            cycle_time_s=("t_s", lambda s: s.max() - s.min() + 1),
            rpm_mean=("rpm", "mean"),
            rpm_std=("rpm", "std"),
            pressure_mean=("hyd_pressure_bar", "mean"),
            pressure_std=("hyd_pressure_bar", "std"),
            fuel_used_gal=("fuel_rate_gal_hr", lambda s: s.sum() / 3600),
        )
        .reset_index()
    )

    idle_stats = (
        idle.groupby(["operator_id"])
        .size()
        .rename("idle_seconds_total")
        .reset_index()
    )
    # approximate idle-before-cycle as total idle time / number of cycles (simple, legible)
    cycles_per_op = cycle_stats.groupby("operator_id")["cycle_group"].count().rename("n_cycles")
    idle_stats = idle_stats.merge(cycles_per_op, on="operator_id")
    idle_stats["idle_before_s"] = idle_stats["idle_seconds_total"] / idle_stats["n_cycles"]

    cycle_stats = cycle_stats.merge(idle_stats[["operator_id", "idle_before_s"]], on="operator_id")
    cycle_stats["rpm_std"] = cycle_stats["rpm_std"].fillna(0)
    cycle_stats["pressure_std"] = cycle_stats["pressure_std"].fillna(0)

    return cycle_stats
