"""
Before trusting any supervised label, check that the engineered cycle
features actually separate operator behavior on their own, with NO labels
given to the algorithm. If KMeans with k=2 can't find the two styles
unsupervised, the features aren't capturing what we think they are, and a
supervised model built on top of them would just be memorizing noise.

Run:
    python -m src.models.validate_clustering --data data/operator_telemetry.csv
"""
from __future__ import annotations
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features.cycle_features import extract_cycle_features, CYCLE_FEATURE_COLUMNS


def validate_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    cycles = extract_cycle_features(df)
    X = StandardScaler().fit_transform(cycles[CYCLE_FEATURE_COLUMNS])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    cycles["cluster"] = kmeans.labels_

    # purity: for each true style, what fraction fell into its majority cluster
    crosstab = pd.crosstab(cycles["operator_style"], cycles["cluster"])
    purity = crosstab.max(axis=1).sum() / crosstab.values.sum()

    return cycles, purity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/operator_telemetry.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cycles, purity = validate_clustering(df)

    print(pd.crosstab(cycles["operator_style"], cycles["cluster"]))
    print(f"\nCluster purity vs. true style: {purity:.1%}")
    print(
        "(>85% means the unlabeled cycle features alone separate efficient "
        "from aggressive operators -- safe to build a supervised scorer on top)"
    )


if __name__ == "__main__":
    main()
