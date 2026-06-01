from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import HiMARSConfig
from ..pareto import first_front, select_by_crowding, truncate_by_rank_and_crowding, unique_solutions
from ..problem import RecommendationProblem, Solution
from .amosa import nlists
from .common import clone_population, crossover_between, mutate_population, rng_from_seed


def _active_clone_mutate(
    population: list[Solution],
    problem: RecommendationProblem,
    config: HiMARSConfig,
    rng: np.random.Generator,
) -> tuple[list[Solution], list[Solution], list[Solution]]:
    dominant = select_by_crowding(first_front(population), config.nd)
    active = select_by_crowding(dominant, min(config.na, len(dominant)))
    clones = clone_population(active, config.nc, rng)
    crossover = crossover_between(clones, active, problem, rng)
    mutated = mutate_population(crossover, config.nnia_mutation_probability, problem, rng)
    return dominant, active, mutated


def run_hani_v1(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Solution]:
    """Run HANIv1: NNIA exploration plus dynamic AMOSA neighborhood search."""

    rng = rng or rng_from_seed(seed)
    population = problem.random_population(config.nd, rng)
    tau = config.tau
    for _ in range(config.max_iter):
        dominant, active, mutated = _active_clone_mutate(population, problem, config, rng)
        if active:
            current = active[int(rng.integers(0, len(active)))]
            archive, _ = nlists(current, population, problem, tau, config, rng)
        else:
            archive = []
        tau *= config.cooling_rate
        population = unique_solutions([*mutated, *dominant, *archive])
        if not population:
            population = problem.random_population(config.nd, rng)
        population = truncate_by_rank_and_crowding(population, max(config.nd, config.hard_limit))
    return select_by_crowding(first_front(population), config.nd)


def run_hani_v2(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_archive: Sequence[Solution] | None = None,
) -> list[Solution]:
    """Run HANIv2: fixed AMOSA archive plus NNIA cloning/crossover/mutation.

    Important correction: the uploaded notebook's ``HANv4`` built ``CT`` but then
    updated the population using ``Ct`` without first assigning ``Ct = Mutate(CT)``.
    This implementation performs the mutation step explicitly, matching Algorithm 6.
    """

    rng = rng or rng_from_seed(seed)
    population = problem.random_population(config.nd, rng)
    archive = list(initial_archive) if initial_archive else problem.random_population(config.hard_limit, rng)
    archive = select_by_crowding(first_front(archive), config.hard_limit)
    current = archive[int(rng.integers(0, len(archive)))]
    tau = config.tau

    for _ in range(config.max_iter):
        archive, current = nlists(current, archive, problem, tau, config, rng)
        tau *= config.cooling_rate
        dominant, _active, mutated = _active_clone_mutate(population, problem, config, rng)
        # Corrected population update: mutated offspring are included.
        population = unique_solutions([*mutated, *dominant, *archive])
        if not population:
            population = problem.random_population(config.nd, rng)
        population = truncate_by_rank_and_crowding(population, max(config.nd, config.hard_limit))
    return select_by_crowding(first_front(population), config.nd)
