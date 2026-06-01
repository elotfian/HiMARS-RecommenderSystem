from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .problem import Solution


def dominates(a: Solution, b: Solution, atol: float = 1e-12) -> bool:
    """Return True if solution ``a`` Pareto-dominates ``b`` for maximization."""

    av = np.asarray(a.objectives, dtype=float)
    bv = np.asarray(b.objectives, dtype=float)
    return bool(np.all(av >= bv - atol) and np.any(av > bv + atol))


def objective_equal(a: Solution, b: Solution, atol: float = 1e-12) -> bool:
    return bool(np.allclose(a.objectives, b.objectives, atol=atol, rtol=0.0))


def unique_solutions(solutions: Iterable[Solution]) -> list[Solution]:
    """Remove duplicate item lists while preserving order."""

    seen = set()
    out: list[Solution] = []
    for sol in solutions:
        key = sol.items
        if key not in seen:
            out.append(sol)
            seen.add(key)
    return out


def non_dominated_sort(solutions: Sequence[Solution]) -> list[list[Solution]]:
    """Fast-enough non-dominated sorting for the small populations in HiMARS."""

    n = len(solutions)
    if n == 0:
        return []
    domination_sets: list[list[int]] = [[] for _ in range(n)]
    dominated_counts = [0] * n
    fronts_idx: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(solutions[i], solutions[j]):
                domination_sets[i].append(j)
                dominated_counts[j] += 1
            elif dominates(solutions[j], solutions[i]):
                domination_sets[j].append(i)
                dominated_counts[i] += 1
        if dominated_counts[i] == 0:
            solutions[i].rank = 0
            fronts_idx[0].append(i)

    k = 0
    while k < len(fronts_idx) and fronts_idx[k]:
        next_front: list[int] = []
        for i in fronts_idx[k]:
            for j in domination_sets[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    solutions[j].rank = k + 1
                    next_front.append(j)
        if next_front:
            fronts_idx.append(next_front)
        k += 1

    fronts = [[solutions[i] for i in front] for front in fronts_idx if front]
    for front in fronts:
        assign_crowding_distance(front)
    return fronts


def assign_crowding_distance(front: Sequence[Solution]) -> None:
    """Assign NSGA-II crowding distance in-place."""

    n = len(front)
    if n == 0:
        return
    for sol in front:
        sol.crowding_distance = 0.0
    if n <= 2:
        for sol in front:
            sol.crowding_distance = float("inf")
        return

    values = np.asarray([s.objectives for s in front], dtype=float)
    n_obj = values.shape[1]
    for j in range(n_obj):
        order = np.argsort(values[:, j])
        front[order[0]].crowding_distance = float("inf")
        front[order[-1]].crowding_distance = float("inf")
        denom = values[order[-1], j] - values[order[0], j]
        if abs(denom) <= 1e-12:
            continue
        for pos in range(1, n - 1):
            idx = order[pos]
            if np.isinf(front[idx].crowding_distance):
                continue
            front[idx].crowding_distance += float(
                (values[order[pos + 1], j] - values[order[pos - 1], j]) / denom
            )


def first_front(solutions: Sequence[Solution]) -> list[Solution]:
    fronts = non_dominated_sort(list(solutions))
    return fronts[0] if fronts else []


def truncate_by_rank_and_crowding(solutions: Iterable[Solution], size: int) -> list[Solution]:
    """Keep at most ``size`` solutions using NSGA-II rank/crowding order."""

    solutions = unique_solutions(solutions)
    if len(solutions) <= size:
        non_dominated_sort(solutions)
        return list(solutions)
    selected: list[Solution] = []
    for front in non_dominated_sort(solutions):
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            front = sorted(front, key=lambda s: s.crowding_distance, reverse=True)
            selected.extend(front[: size - len(selected)])
            break
    return selected


def select_by_crowding(solutions: Iterable[Solution], size: int) -> list[Solution]:
    """Select diverse solutions from a non-dominated set using crowding distance."""

    solutions = unique_solutions(solutions)
    if len(solutions) <= size:
        assign_crowding_distance(solutions)
        return list(solutions)
    assign_crowding_distance(solutions)
    return sorted(solutions, key=lambda s: s.crowding_distance, reverse=True)[:size]


def update_archive(archive: Iterable[Solution], candidate: Solution, hard_limit: int | None = None) -> list[Solution]:
    """Add a solution to an archive if it is not dominated, removing dominated entries."""

    archive = unique_solutions(archive)
    if any(sol.items == candidate.items for sol in archive):
        return archive
    if any(dominates(sol, candidate) or objective_equal(sol, candidate) for sol in archive):
        return archive
    archive = [sol for sol in archive if not dominates(candidate, sol)]
    archive.append(candidate)
    archive = first_front(archive)
    if hard_limit is not None and len(archive) > hard_limit:
        archive = select_by_crowding(archive, hard_limit)
    return archive
