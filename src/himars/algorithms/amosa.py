from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import HiMARSConfig
from ..pareto import dominates, first_front, select_by_crowding, update_archive
from ..problem import RecommendationProblem, Solution
from .common import rng_from_seed


def _sigmoid(x: float) -> float:
    # numerically stable enough for this scale
    x = float(np.clip(x, -700, 700))
    return float(1.0 / (1.0 + np.exp(-x)))


def domination_amount(a: Solution, b: Solution, reference: Sequence[Solution]) -> float:
    """Normalized domination amount used in AMOSA-style acceptance."""

    values = np.asarray([s.objectives for s in [*reference, a, b]], dtype=float)
    ranges = values.max(axis=0) - values.min(axis=0)
    ranges = np.where(np.abs(ranges) <= 1e-12, 1.0, ranges)
    delta = np.abs(np.asarray(a.objectives) - np.asarray(b.objectives)) / ranges
    delta = delta[delta > 1e-12]
    if delta.size == 0:
        return 0.0
    return float(np.prod(delta))


def _dominated_by_archive(candidate: Solution, archive: Sequence[Solution]) -> list[Solution]:
    return [sol for sol in archive if dominates(sol, candidate)]


def nlists(
    current: Solution,
    reference: Sequence[Solution],
    problem: RecommendationProblem,
    tau: float,
    config: HiMARSConfig,
    rng: np.random.Generator,
) -> tuple[list[Solution], Solution]:
    """AMOSA neighborhood update around ``current``.

    This is the cleaned implementation of Algorithm 3 / ``New_lists`` from the
    notebooks. It keeps the same replacement-neighborhood idea but guards all
    divisions and archive updates.
    """

    archive = list(reference)
    available = [item for item in problem.candidates if item not in set(current.items)]
    if not available:
        return select_by_crowding(first_front(archive), config.hard_limit), current

    positions = list(range(problem.top_s))
    if len(available) < problem.top_s:
        positions = rng.choice(np.asarray(positions), size=len(available), replace=False).tolist()

    for pos in positions:
        current_items = list(current.items)
        available = [item for item in problem.candidates if item not in set(current_items)]
        if not available:
            break
        current_items[pos] = rng.choice(np.array(available, dtype=object))
        candidate = problem.make_solution(current_items)

        if dominates(candidate, current):
            current = candidate
            archive = update_archive(archive, candidate, hard_limit=None)
        elif dominates(current, candidate):
            amount = domination_amount(candidate, current, archive)
            if rng.random() < _sigmoid(-amount * tau):
                current = candidate
        else:
            dominators = _dominated_by_archive(candidate, archive)
            if dominators:
                avg_amount = float(np.mean([domination_amount(candidate, d, archive) for d in dominators]))
                if rng.random() < _sigmoid(-avg_amount * tau):
                    current = candidate
            else:
                current = candidate
                archive = update_archive(archive, candidate, hard_limit=None)

        if len(archive) > config.soft_limit:
            archive = select_by_crowding(first_front(archive), config.hard_limit)

    archive = first_front(archive)
    if len(archive) > config.hard_limit:
        archive = select_by_crowding(archive, config.hard_limit)
    return archive, current


def run_amosa(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_archive: Sequence[Solution] | None = None,
) -> list[Solution]:
    """Run AMOSA for the top-s recommendation-list problem."""

    rng = rng or rng_from_seed(seed)
    if initial_archive:
        archive = list(initial_archive)
    else:
        archive = problem.random_population(config.hard_limit, rng)
    archive = select_by_crowding(first_front(archive), config.hard_limit)
    if not archive:
        archive = problem.random_population(config.hard_limit, rng)
    current = archive[int(rng.integers(0, len(archive)))]
    tau = config.tau
    for _ in range(config.max_iter):
        archive, current = nlists(current, archive, problem, tau, config, rng)
        tau *= config.cooling_rate
    return select_by_crowding(first_front(archive), config.hard_limit)
