"""
Train a classifier that predicts operator style (efficient/aggressive) from
per-cycle behavioral features, validated on operators the model never saw
during training (group-aware split -- same principle as Example 1).

This is the supervised layer that sits on top of the EcoScore formula and
the unsupervised clustering check: once you trust the features (clustering)
and have a transparent baseline (EcoScore formula), a classifier lets you
generalize to a brand-new operator from just their first handful of cycles.

Run:
    python -m src.models.train --data data/operator_telemetry.csv --out models_artifacts/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.features.build_features import extract_cycle_features, CYCLE_FEATURE_COLUMNS


def stratified_group_holdout(cycles: pd.DataFrame, frac: float = 0.25, seed: int = 42):
    """Hold out whole operators, but guarantee at least one of EACH archetype
    ends up in the test set. A plain GroupShuffleSplit can (and did, when
    this was first built) select held-out groups that are all the same
    class, making the accuracy number meaningless -- this is the fix."""
    rng = np.random.default_rng(seed)
    held_out = []
    for archetype, ops in cycles.groupby("archetype")["operator_id"].unique().items():
        ops = list(ops)
        n_test = max(1, round(len(ops) * frac))
        held_out.extend(rng.choice(ops, size=n_test, replace=False))
    return held_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/operator_telemetry.csv")
    parser.add_argument("--out", default="models_artifacts/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    cycles = extract_cycle_features(raw)

    X = cycles[CYCLE_FEATURE_COLUMNS]
    y = (cycles["archetype"] == "aggressive").astype(int)  # 1 = aggressive

    held_out_operators = stratified_group_holdout(cycles)
    test_mask = cycles["operator_id"].isin(held_out_operators)
    train_idx, test_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
    held_out_operators = sorted(str(op) for op in held_out_operators)

    model = GradientBoostingClassifier(random_state=42).fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])

    acc = accuracy_score(y.iloc[test_idx], preds)
    report = classification_report(y.iloc[test_idx], preds, target_names=["efficient", "aggressive"])

    joblib.dump(model, out_dir / "operator_style_model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"held_out_operators": held_out_operators, "accuracy": round(acc, 4)}, f, indent=2)

    print("Held-out operators (never seen in training):", held_out_operators)
    print(f"\nAccuracy on held-out operators: {acc:.2%}\n")
    print(report)
    print(f"Saved model -> {out_dir/'operator_style_model.joblib'}")


if __name__ == "__main__":
    main()
