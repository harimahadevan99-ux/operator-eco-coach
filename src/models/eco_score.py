"""
Two independent views of the same per-cycle features:

1. EcoScore: a simple, fully transparent weighted formula (0-100, higher is
   better) -- nothing fancy, and that's the point: it's auditable by anyone,
   including a non-technical operator or supervisor.

2. KMeans clustering (unsupervised, no labels used): sanity-checks that the
   features we engineered actually separate real behavioral styles on their
   own, *before* we trust any supervised model built on top of them.

Run:
    python -m src.models.eco_score --data data/operator_telemetry.csv
"""
from __future__ import annotations
import argparse

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features.build_features import extract_cycle_features, CYCLE_FEATURE_COLUMNS

WEIGHTS = {"idle_pct": 0.40, "rpm_std": 0.30, "fuel_used_gal": 0.30}


def compute_eco_scores(cycles: pd.DataFrame) -> pd.DataFrame:
    cycles = cycles.copy()
    for col in WEIGHTS:
        lo, hi = cycles[col].min(), cycles[col].max()
        cycles[f"{col}_norm"] = 0.0 if hi == lo else (cycles[col] - lo) / (hi - lo)

    penalty = sum(WEIGHTS[c] * cycles[f"{c}_norm"] for c in WEIGHTS)
    cycles["eco_score"] = (100 * (1 - penalty)).round(1)
    return cycles


def cluster_purity(cycles: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    X = StandardScaler().fit_transform(cycles[CYCLE_FEATURE_COLUMNS])
    cycles = cycles.copy()
    cycles["cluster"] = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)

    crosstab = pd.crosstab(cycles["cluster"], cycles["archetype"])
    purity = crosstab.max(axis=1).sum() / len(cycles)
    return cycles, purity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/operator_telemetry.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    cycles = extract_cycle_features(raw)
    cycles = compute_eco_scores(cycles)
    cycles, purity = cluster_purity(cycles)

    print(f"Detected {len(cycles)} cycles across {cycles.operator_id.nunique()} operators.\n")

    summary = (
        cycles.groupby(["operator_id", "archetype"])
        .agg(avg_eco_score=("eco_score", "mean"),
             avg_idle_pct=("idle_pct", "mean"),
             avg_rpm_std=("rpm_std", "mean"),
             avg_fuel_per_cycle=("fuel_used_gal", "mean"),
             n_cycles=("eco_score", "count"))
        .round(2)
        .sort_values("avg_eco_score", ascending=False)
    )
    print("Operator ranking (highest eco-score first):\n")
    print(summary.to_string())

    print(f"\nUnsupervised clustering purity vs. true archetype: {purity:.2%}")
    print("(How well 2 clusters found with NO labels line up with the real")
    print(" efficient/aggressive styles -- validates the features are meaningful.)")


if __name__ == "__main__":
    main()
