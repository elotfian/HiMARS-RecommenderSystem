from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import HiMARSConfig
from ..pareto import first_front, truncate_by_rank_and_crowding
from ..problem import RecommendationProblem, Solution
from .amosa import nlists
from .common import make_crossover_offspring, make_mutation_offspring, rng_from_seed


def run_han_v1(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Solution]:
    """Run HANv1: dynamic AMOSA archive generated from the NSGA-II population."""

    rng = rng or rng_from_seed(seed)
    pop = problem.random_population(config.pop_size, rng)
    pop = truncate_by_rank_and_crowding(pop, config.pop_size)
    tau = config.tau
    n_cross = 2 * round(config.pop_size * config.crossover_probability / 2)
    n_mut = round(config.pop_size * config.nsga_mutation_probability)

    for _ in range(config.max_iter):
        crossover = make_crossover_offspring(pop, n_cross, problem, rng)
        mutation = make_mutation_offspring(
            crossover or pop, n_mut, config.nsga_mutation_probability, problem, rng
        )
        pop = truncate_by_rank_and_crowding([*pop, *crossover, *mutation], config.pop_size)
        front = first_front(pop)
        if front:
            current = front[int(rng.integers(0, len(front)))]
            archive, _ = nlists(current, pop, problem, tau, config, rng)
            pop = truncate_by_rank_and_crowding([*pop, *archive], config.pop_size)
        tau *= config.cooling_rate
    return first_front(pop)


def run_han_v2(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_archive: Sequence[Solution] | None = None,
) -> list[Solution]:
    """Run HANv2: fixed AMOSA archive plus NSGA-II crossover/mutation."""

    rng = rng or rng_from_seed(seed)
    pop = problem.random_population(config.pop_size, rng)
    pop = truncate_by_rank_and_crowding(pop, config.pop_size)
    archive = list(initial_archive) if initial_archive else problem.random_population(config.hard_limit, rng)
    archive = truncate_by_rank_and_crowding(first_front(archive), config.hard_limit)
    current = archive[int(rng.integers(0, len(archive)))]
    tau = config.tau
    n_cross = 2 * round(config.pop_size * config.crossover_probability / 2)
    n_mut = round(config.pop_size * config.nsga_mutation_probability)

    for _ in range(config.max_iter):
        archive, current = nlists(current, archive, problem, tau, config, rng)
        tau *= config.cooling_rate
        crossover = make_crossover_offspring(pop, n_cross, problem, rng)
        mutation = make_mutation_offspring(
            crossover or pop, n_mut, config.nsga_mutation_probability, problem, rng
        )
        pop = truncate_by_rank_and_crowding([*pop, *crossover, *mutation, *archive], config.pop_size)
    return first_front(pop)
