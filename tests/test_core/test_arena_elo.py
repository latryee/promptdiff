"""Unit tests for Bayesian Bradley-Terry and ELO Prompt Arena Rating System."""

from __future__ import annotations

import promptdiff
from promptdiff.core.arena_elo import (
    BradleyTerryRatingEngine,
    ELORatingEngine,
    PairwiseMatch,
    compute_bradley_terry_ratings,
    compute_elo_ratings,
)


def test_elo_rating_engine_basic() -> None:
    """Winner should gain ELO points and loser should drop points."""
    engine = ELORatingEngine(initial_rating=1500.0, k_factor=32.0)
    matches = [
        PairwiseMatch(prompt_a="prompt_v1", prompt_b="prompt_v2", winner="A"),
        PairwiseMatch(prompt_a="prompt_v1", prompt_b="prompt_v2", winner="A"),
        PairwiseMatch(prompt_a="prompt_v1", prompt_b="prompt_v3", winner="A"),
        PairwiseMatch(prompt_a="prompt_v2", prompt_b="prompt_v3", winner="B"),
    ]

    res = engine.compute_ratings(matches)
    assert res.total_matches == 4
    assert len(res.ratings) == 3

    # prompt_v1 won all matches, so it must lead the leaderboard
    leader = res.ratings[0]
    assert leader.prompt_name == "prompt_v1"
    assert leader.rating > 1500.0
    assert leader.wins == 3
    assert leader.losses == 0

    # prompt_v2 lost both matches, so its rating must drop
    last = res.ratings[-1]
    assert last.prompt_name == "prompt_v2"
    assert last.rating < 1500.0


def test_bradley_terry_convergence() -> None:
    """Bradley-Terry Minorization-Maximization algorithm should converge on tournament matrix."""
    engine = BradleyTerryRatingEngine(max_iter=50)

    # 3 prompts: A beats B, B beats C, A beats C
    matches = [
        PairwiseMatch(prompt_a="A", prompt_b="B", winner="A"),
        PairwiseMatch(prompt_a="A", prompt_b="B", winner="A"),
        PairwiseMatch(prompt_a="B", prompt_b="C", winner="A"),  # B beats C
        PairwiseMatch(prompt_a="A", prompt_b="C", winner="A"),  # A beats C
        PairwiseMatch(prompt_a="B", prompt_b="C", winner="tie"),
    ]

    res = engine.compute_ratings(matches)
    assert res.total_matches == 5
    assert len(res.ratings) == 3
    assert res.convergence_iterations > 0

    # Rank order must strictly be A > B > C
    ranks = [r.prompt_name for r in res.ratings]
    assert ranks == ["A", "B", "C"]
    assert res.ratings[0].rating > res.ratings[1].rating > res.ratings[2].rating


def test_elo_and_bt_tied_matches() -> None:
    """Tied matches should keep ratings close to baseline 1500."""
    matches = [
        PairwiseMatch(prompt_a="p1", prompt_b="p2", winner="tie"),
        PairwiseMatch(prompt_a="p1", prompt_b="p2", winner="tie"),
    ]

    res_elo = compute_elo_ratings(matches)
    assert abs(res_elo.ratings[0].rating - 1500.0) < 1.0
    assert abs(res_elo.ratings[1].rating - 1500.0) < 1.0

    res_bt = compute_bradley_terry_ratings(matches)
    assert abs(res_bt.ratings[0].rating - 1500.0) < 1.0


def test_sdk_rating_exports() -> None:
    """Verify top-level promptdiff SDK exports."""
    matches = [
        {"prompt_a": "v1", "prompt_b": "v2", "winner": "A"},
        {"prompt_a": "v1", "prompt_b": "v2", "winner": "B"},
    ]
    res_elo = promptdiff.compute_elo_ratings(matches)
    assert len(res_elo.ratings) == 2

    res_bt = promptdiff.compute_bradley_terry_ratings(matches)
    assert len(res_bt.ratings) == 2
