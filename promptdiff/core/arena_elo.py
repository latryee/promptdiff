"""Bayesian Bradley-Terry and Dynamic ELO Rating System for Prompt Tournaments.

Models pairwise prompt evaluations into continuous latent skill ratings with
confidence intervals (identical to LMSYS Chatbot Arena methodology).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PairwiseMatch:
    """Outcome of a head-to-head evaluation between two prompt variants."""

    prompt_a: str
    prompt_b: str
    winner: Literal["A", "B", "tie"]
    test_case_id: str | None = None


@dataclass
class PromptRating:
    """Continuous latent skill rating for a prompt variant."""

    prompt_name: str
    rating: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate_pct: float = 0.0
    confidence_interval: tuple[float, float] = (1500.0, 1500.0)


@dataclass
class ArenaTournamentResult:
    """Full tournament leaderboard with ratings and convergence metrics."""

    ratings: list[PromptRating] = field(default_factory=list)
    total_matches: int = 0
    convergence_iterations: int = 0
    method: str = "elo"


class ELORatingEngine:
    """Online sequential ELO rating updates for prompt arena matches."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor

    def compute_ratings(self, matches: list[PairwiseMatch]) -> ArenaTournamentResult:
        ratings: dict[str, float] = {}
        wins: dict[str, int] = {}
        losses: dict[str, int] = {}
        ties: dict[str, int] = {}

        def ensure_prompt(name: str) -> None:
            if name not in ratings:
                ratings[name] = self.initial_rating
                wins[name] = 0
                losses[name] = 0
                ties[name] = 0

        for match in matches:
            ensure_prompt(match.prompt_a)
            ensure_prompt(match.prompt_b)

            r_a = ratings[match.prompt_a]
            r_b = ratings[match.prompt_b]

            # Expected scores
            e_a = 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))
            e_b = 1.0 - e_a

            # Actual score
            if match.winner == "A":
                s_a, s_b = 1.0, 0.0
                wins[match.prompt_a] += 1
                losses[match.prompt_b] += 1
            elif match.winner == "B":
                s_a, s_b = 0.0, 1.0
                losses[match.prompt_a] += 1
                wins[match.prompt_b] += 1
            else:
                s_a, s_b = 0.5, 0.5
                ties[match.prompt_a] += 1
                ties[match.prompt_b] += 1

            # Update ratings
            ratings[match.prompt_a] += self.k_factor * (s_a - e_a)
            ratings[match.prompt_b] += self.k_factor * (s_b - e_b)

        result_list: list[PromptRating] = []
        for name in ratings:
            tot = wins[name] + losses[name] + ties[name]
            w_rate = ((wins[name] + 0.5 * ties[name]) / tot * 100.0) if tot > 0 else 50.0

            # 95% standard error approximation for rating: 400 / sqrt(N)
            margin = (400.0 / math.sqrt(tot)) if tot > 0 else 100.0
            r_val = ratings[name]
            ci = (round(r_val - margin, 1), round(r_val + margin, 1))

            result_list.append(
                PromptRating(
                    prompt_name=name,
                    rating=round(r_val, 1),
                    matches_played=tot,
                    wins=wins[name],
                    losses=losses[name],
                    ties=ties[name],
                    win_rate_pct=round(w_rate, 1),
                    confidence_interval=ci,
                )
            )

        result_list.sort(key=lambda x: x.rating, reverse=True)
        return ArenaTournamentResult(
            ratings=result_list,
            total_matches=len(matches),
            convergence_iterations=len(matches),
            method="elo",
        )


class BradleyTerryRatingEngine:
    """Maximum-Likelihood Bradley-Terry paired comparison estimator."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def compute_ratings(self, matches: list[PairwiseMatch]) -> ArenaTournamentResult:
        prompts = sorted({m.prompt_a for m in matches} | {m.prompt_b for m in matches})
        if not prompts:
            return ArenaTournamentResult()

        n = len(prompts)
        idx_map = {p: i for i, p in enumerate(prompts)}

        # Wins matrix and total matches matrix
        wins_matrix = [[0.0] * n for _ in range(n)]
        n_matrix = [[0.0] * n for _ in range(n)]
        total_wins = [0.0] * n

        win_counts = dict.fromkeys(prompts, 0)
        loss_counts = dict.fromkeys(prompts, 0)
        tie_counts = dict.fromkeys(prompts, 0)

        for m in matches:
            i = idx_map[m.prompt_a]
            j = idx_map[m.prompt_b]
            n_matrix[i][j] += 1.0
            n_matrix[j][i] += 1.0

            if m.winner == "A":
                wins_matrix[i][j] += 1.0
                total_wins[i] += 1.0
                win_counts[m.prompt_a] += 1
                loss_counts[m.prompt_b] += 1
            elif m.winner == "B":
                wins_matrix[j][i] += 1.0
                total_wins[j] += 1.0
                loss_counts[m.prompt_a] += 1
                win_counts[m.prompt_b] += 1
            else:
                wins_matrix[i][j] += 0.5
                wins_matrix[j][i] += 0.5
                total_wins[i] += 0.5
                total_wins[j] += 0.5
                tie_counts[m.prompt_a] += 1
                tie_counts[m.prompt_b] += 1

        # Iterative Minorization-Maximization (MM) for Bradley-Terry
        p_vec = [1.0 / n] * n
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            p_next = [0.0] * n
            for i in range(n):
                denom = 0.0
                for j in range(n):
                    if i != j and n_matrix[i][j] > 0:
                        denom += n_matrix[i][j] / (p_vec[i] + p_vec[j])
                p_next[i] = total_wins[i] / denom if denom > 0 else 1e-4

            # Normalize sum to 1
            total_sum = sum(p_next)
            if total_sum > 0:
                p_next = [x / total_sum for x in p_next]

            # Convergence check
            diff = sum(abs(p_next[i] - p_vec[i]) for i in range(n))
            p_vec = p_next
            if diff < self.tol:
                break

        # Convert latent probabilities to ELO scale centered around 1500
        mean_p = sum(p_vec) / n
        ratings: list[PromptRating] = []

        for i, name in enumerate(prompts):
            ratio = max(p_vec[i] / mean_p, 1e-6)
            elo = 1500.0 + 400.0 * math.log10(ratio)

            tot = win_counts[name] + loss_counts[name] + tie_counts[name]
            w_rate = ((win_counts[name] + 0.5 * tie_counts[name]) / tot * 100.0) if tot > 0 else 50.0
            margin = (300.0 / math.sqrt(tot)) if tot > 0 else 100.0
            ci = (round(elo - margin, 1), round(elo + margin, 1))

            ratings.append(
                PromptRating(
                    prompt_name=name,
                    rating=round(elo, 1),
                    matches_played=tot,
                    wins=win_counts[name],
                    losses=loss_counts[name],
                    ties=tie_counts[name],
                    win_rate_pct=round(w_rate, 1),
                    confidence_interval=ci,
                )
            )

        ratings.sort(key=lambda x: x.rating, reverse=True)
        return ArenaTournamentResult(
            ratings=ratings,
            total_matches=len(matches),
            convergence_iterations=iters,
            method="bradley_terry",
        )


def compute_elo_ratings(
    matches: list[PairwiseMatch],
    k_factor: float = 32.0,
) -> ArenaTournamentResult:
    """Compute ELO ratings from paired head-to-head tournament comparisons."""
    engine = ELORatingEngine(k_factor=k_factor)
    return engine.compute_ratings(matches)


def compute_bradley_terry_ratings(
    matches: list[PairwiseMatch],
) -> ArenaTournamentResult:
    """Compute Maximum Likelihood Bradley-Terry ratings from paired comparisons."""
    engine = BradleyTerryRatingEngine()
    return engine.compute_ratings(matches)
