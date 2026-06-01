from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..operators import mutate_items, one_point_crossover
from ..pareto import select_by_crowding, unique_solutions
from ..problem import RecommendationProblem, Solution


def rng_from_seed(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_crossover_offspring(
    parents: Sequence[Solution],
    n_offspring: int,
    problem: RecommendationProblem,
    rng: np.random.Generator,
) -> list[Solution]:
    if not parents or n_offspring <= 0:
        return []
    n_offspring = n_offspring if n_offspring % 2 == 0 else n_offspring + 1
    children: list[Solution] = []
    for _ in range(n_offspring // 2):
        i, j = rng.integers(0, len(parents), size=2)
        child1, child2 = one_point_crossover(
            parents[int(i)].items, parents[int(j)].items, problem.candidates, problem.top_s, rng
        )
        children.append(problem.make_solution(child1))
        children.append(problem.make_solution(child2))
    return children[:n_offspring]


def make_mutation_offspring(
    parents: Sequence[Solution],
    n_offspring: int,
    mutation_probability: float,
    problem: RecommendationProblem,
    rng: np.random.Generator,
) -> list[Solution]:
    if not parents or n_offspring <= 0:
        return []
    offspring: list[Solution] = []
    for _ in range(n_offspring):
        parent = parents[int(rng.integers(0, len(parents)))]
        items = mutate_items(parent.items, problem.candidates, rng, mutation_probability)
        offspring.append(problem.make_solution(items))
    return offspring


def mutate_population(
    population: Sequence[Solution],
    mutation_probability: float,
    problem: RecommendationProblem,
    rng: np.random.Generator,
) -> list[Solution]:
    return [
        problem.make_solution(mutate_items(sol.items, problem.candidates, rng, mutation_probability))
        for sol in population
    ]


def crossover_between(
    clones: Sequence[Solution],
    active: Sequence[Solution],
    problem: RecommendationProblem,
    rng: np.random.Generator,
) -> list[Solution]:
    if not clones or not active:
        return []
    offspring: list[Solution] = []
    for clone in clones:
        mate = active[int(rng.integers(0, len(active)))]
        child1, child2 = one_point_crossover(mate.items, clone.items, problem.candidates, problem.top_s, rng)
        offspring.append(problem.make_solution(child1))
        offspring.append(problem.make_solution(child2))
    return offspring


def clone_population(active: Sequence[Solution], n_clones: int, rng: np.random.Generator) -> list[Solution]:
    """Crowding-proportional cloning used by NNIA/HANI."""

    if not active or n_clones <= 0:
        return []
    crowd = np.asarray([s.crowding_distance for s in active], dtype=float)
    finite = crowd[np.isfinite(crowd)]
    max_finite = float(finite.max()) if finite.size else 1.0
    crowd = np.where(np.isfinite(crowd), crowd, 2.0 * max_finite)
    crowd = np.maximum(crowd, 0.0)
    if crowd.sum() <= 1e-12:
        probs = np.full(len(active), 1 / len(active))
    else:
        probs = crowd / crowd.sum()
    idx = rng.choice(np.arange(len(active)), size=n_clones, replace=True, p=probs)
    return [active[int(i)] for i in idx]


def dedupe_and_limit(solutions: Iterable[Solution], limit: int | None = None) -> list[Solution]:
    out = unique_solutions(solutions)
    return select_by_crowding(out, limit) if limit is not None and len(out) > limit else out
