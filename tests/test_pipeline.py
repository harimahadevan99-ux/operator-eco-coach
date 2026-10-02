import pandas as pd

from src.simulation.generate_operator_data import generate_operators
from src.features.build_features import extract_cycle_features, CYCLE_FEATURE_COLUMNS
from src.models.eco_score import compute_eco_scores, cluster_purity


def _sample_cycles(n_cycles=60, seed=1):
    raw = generate_operators(n_cycles_per_operator=n_cycles, seed=seed)
    return raw, extract_cycle_features(raw)


def test_cycle_detection_roughly_matches_requested_count():
    raw, cycles = _sample_cycles(n_cycles=60)
    for operator_id, g in cycles.groupby("operator_id"):
        # detection is rule-based on mode transitions; should land within a
        # couple of cycles of the ground-truth count for every operator
        assert abs(len(g) - 60) <= 2


def test_no_nans_in_features():
    _, cycles = _sample_cycles()
    assert not cycles[CYCLE_FEATURE_COLUMNS].isna().any().any()


def test_efficient_operators_score_higher_than_aggressive():
    _, cycles = _sample_cycles(n_cycles=100)
    scored = compute_eco_scores(cycles)
    avg_by_style = scored.groupby("archetype")["eco_score"].mean()
    assert avg_by_style["efficient"] > avg_by_style["aggressive"]


def test_unsupervised_clustering_finds_real_structure():
    _, cycles = _sample_cycles(n_cycles=100)
    scored = compute_eco_scores(cycles)
    _, purity = cluster_purity(scored)
    # clustering with NO labels should beat random (50%) by a wide margin,
    # but won't be anywhere near 100%: a third-or-so of aggressive cycles
    # happen to have no idle gap at all and look similar to efficient cycles
    # on a single-cycle basis. See README "Results" for why this matters.
    assert purity > 0.60
