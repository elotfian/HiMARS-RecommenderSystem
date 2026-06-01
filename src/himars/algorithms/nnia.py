from __future__ import annotations

import numpy as np

from ..config import HiMARSConfig
from ..pareto import assign_crowding_distance, first_front, select_by_crowding, unique_solutions
from ..problem import RecommendationProblem, Solution
from .common import clone_population, crossover_between, mutate_population, rng_from_seed


def _nnia_step(
    population: list[Solution],
    problem: RecommendationProblem,
    config: HiMARSConfig,
    rng: np.random.Generator,
) -> tuple[list[Solution], list[Solution], list[Solution], list[Solution]]:
    non_dominated = first_front(population)
    assign_crowding_distance(non_dominated)
    dominant = select_by_crowding(non_dominated, config.nd)
    active = select_by_crowding(dominant, min(config.na, len(dominant)))
    clones = clone_population(active, config.nc, rng)
    crossover = crossover_between(clones, active, problem, rng)
    mutated = mutate_population(crossover, config.nnia_mutation_probability, problem, rng)
    next_population = unique_solutions([*mutated, *dominant])
    return next_population, dominant, active, mutated


def run_nnia(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    max_iter: int | None = None,
) -> list[Solution]:
    """Run the NNIA baseline."""

    rng = rng or rng_from_seed(seed)
    n_iter = config.max_iter if max_iter is None else int(max_iter)
    population = problem.random_population(config.nd, rng)
    for _ in range(n_iter):
        population, _, _, _ = _nnia_step(population, problem, config, rng)
        if not population:
            population = problem.random_population(config.nd, rng)
    return select_by_crowding(first_front(population), config.nd)
