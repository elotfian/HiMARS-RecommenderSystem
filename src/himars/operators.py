from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def one_point_crossover(
    parent1: Sequence[Hashable],
    parent2: Sequence[Hashable],
    candidates: Sequence[Hashable],
    top_s: int,
    rng: np.random.Generator,
) -> tuple[tuple[Hashable, ...], tuple[Hashable, ...]]:
    """One-point crossover with duplicate repair.

    This mirrors the notebook/paper crossover idea but guarantees valid unique
    top-s lists even when parents share many items.
    """

    if top_s < 2:
        return tuple(parent1[:top_s]), tuple(parent2[:top_s])
    point = int(rng.integers(1, top_s))
    child1 = list(parent1[:point]) + list(parent2[point:top_s])
    child2 = list(parent2[:point]) + list(parent1[point:top_s])
    return (
        repair_items(child1, candidates, top_s, rng),
        repair_items(child2, candidates, top_s, rng),
    )


def mutate_items(
    items: Sequence[Hashable],
    candidates: Sequence[Hashable],
    rng: np.random.Generator,
    mutation_probability: float = 1.0,
) -> tuple[Hashable, ...]:
    """Replace one random item by an item not currently in the list."""

    items = list(items)
    if rng.random() > mutation_probability:
        return tuple(items)
    available = [item for item in candidates if item not in set(items)]
    if not available or not items:
        return tuple(items)
    pos = int(rng.integers(0, len(items)))
    items[pos] = rng.choice(np.array(available, dtype=object))
    return tuple(items)


def repair_items(
    items: Sequence[Hashable],
    candidates: Sequence[Hashable],
    top_s: int,
    rng: np.random.Generator,
) -> tuple[Hashable, ...]:
    seen: set[Hashable] = set()
    candidate_set = set(candidates)
    out: list[Hashable] = []
    for item in items:
        if item in candidate_set and item not in seen:
            out.append(item)
            seen.add(item)
        if len(out) == top_s:
            break
    if len(out) < top_s:
        available = [item for item in candidates if item not in seen]
        fill_n = top_s - len(out)
        if fill_n > len(available):
            raise ValueError("Candidate pool too small to repair child.")
        out.extend(rng.choice(np.array(available, dtype=object), size=fill_n, replace=False).tolist())
    return tuple(out)
