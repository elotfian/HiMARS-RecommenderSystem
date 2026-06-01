from __future__ import annotations

import numpy as np

from ..config import HiMARSConfig
from ..pareto import first_front, truncate_by_rank_and_crowding
from ..problem import RecommendationProblem, Solution
from .common import make_crossover_offspring, make_mutation_offspring, rng_from_seed


def run_nsga2(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Solution]:
    """Run NSGA-II for the top-s recommendation-list problem."""

    rng = rng or rng_from_seed(seed)
    pop = problem.random_population(config.pop_size, rng)
    pop = truncate_by_rank_and_crowding(pop, config.pop_size)

    n_cross = 2 * round(config.pop_size * config.crossover_probability / 2)
    n_mut = round(config.pop_size * config.nsga_mutation_probability)

    for _ in range(config.max_iter):
        crossover = make_crossover_offspring(pop, n_cross, problem, rng)
        mutation = make_mutation_offspring(
            crossover or pop,
            n_mut,
            config.nsga_mutation_probability,
            problem,
            rng,
        )
        pop = truncate_by_rank_and_crowding([*pop, *crossover, *mutation], config.pop_size)
    return first_front(pop)
