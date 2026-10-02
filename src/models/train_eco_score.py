"""
Train a classifier to predict operator style from cycle features, and
compute a human-facing 0-100 EcoScore per cycle.

Group-split by operator_id (not random rows) -- same principle as the fuel
model: we need to know this generalizes to an operator the model has never
scored before, not just memorized these four people's habits.

Run:
    python -m src.models.train_eco_score --data data/operator_telemetry.csv --out models_artifacts/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from src.features.cycle_features import extract_cycle_features, CYCLE_FEATURE_COLUMNS


def compute_eco_score(cycles: pd.DataFrame) -> pd.Series:
    """Composite 0-100 score: higher = more efficient. Weights are the one
    place domain judgement enters the formula -- tune these against real
    fuel-per-ton outcomes once you have real data."""
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    idle_term = norm(cycles["idle_before_s"])
    jerk_term = norm(cycles["rpm_std"])
    fuel_term = norm(cycles["fuel_used_gal"])

    penalty = 0.4 * idle_term + 0.3 * jerk_term + 0.3 * fuel_term
    return (100 * (1 - penalty)).round(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/operator_telemetry.csv")
    parser.add_argument("--out", default="models_artifacts/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    cycles = extract_cycle_features(df)
    cycles["eco_score"] = compute_eco_score(cycles)

    X = cycles[CYCLE_FEATURE_COLUMNS]
    y = (cycles["operator_style"] == "aggressive").astype(int)  # 1 = aggressive
    groups = cycles["operator_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    held_out = sorted(groups.iloc[test_idx].unique())

    clf = GradientBoostingClassifier(random_state=42).fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = clf.predict(X.iloc[test_idx])
    acc = accuracy_score(y.iloc[test_idx], preds)

    joblib.dump(clf, out_dir / "eco_classifier.joblib")

    score_by_operator = cycles.groupby(["operator_id", "operator_style"])["eco_score"].mean().round(1)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "held_out_operators": held_out,
            "held_out_accuracy": round(float(acc), 3),
            "avg_eco_score_by_operator": {
                f"{idx[0]} ({idx[1]})": val for idx, val in score_by_operator.items()
            },
        }, f, indent=2)

    print(f"Held-out operators (never seen in training): {held_out}")
    print(f"Held-out accuracy predicting efficient vs aggressive: {acc:.1%}\n")
    print("Average EcoScore by operator (100 = best):")
    print(score_by_operator.to_string())
    print(f"\nSaved classifier -> {out_dir/'eco_classifier.joblib'}")


if __name__ == "__main__":
    main()
