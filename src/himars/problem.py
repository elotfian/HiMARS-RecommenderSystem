from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass
class Solution:
    """A top-s recommendation list and its objective values.

    ``objectives`` are stored as maximization values: ``(accuracy_objective,
    diversity_objective)``. This avoids the negative-cost convention used in the
    notebooks and makes dominance checks less error-prone.
    """

    items: tuple[Hashable, ...]
    objectives: tuple[float, float]
    rank: int = 0
    crowding_distance: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def f1(self) -> float:
        return float(self.objectives[0])

    @property
    def f2(self) -> float:
        return float(self.objectives[1])

    @property
    def costs(self) -> tuple[float, float]:
        """Minimization convention compatible with the old notebooks."""

        return (-self.f1, -self.f2)

    def to_dict(self) -> dict:
        return {
            "items": list(self.items),
            "f1": self.f1,
            "f2": self.f2,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            **self.metadata,
        }


class RecommendationProblem:
    """Bi-objective top-s recommendation problem over a candidate top-k pool."""

    def __init__(self, candidates: Sequence[Hashable], top_s: int, evaluator):
        self.candidates = tuple(dict.fromkeys(candidates))
        self.top_s = int(top_s)
        self.evaluator = evaluator
        if self.top_s <= 0:
            raise ValueError("top_s must be positive.")
        if len(self.candidates) < self.top_s:
            raise ValueError(
                f"Need at least top_s={self.top_s} unique candidates, got {len(self.candidates)}."
            )

    def make_solution(self, items: Iterable[Hashable]) -> Solution:
        repaired = self.repair(items)
        return Solution(items=repaired, objectives=self.evaluator.objectives(repaired))

    def repair(self, items: Iterable[Hashable], rng: np.random.Generator | None = None) -> tuple[Hashable, ...]:
        """Repair a list so it is unique, feasible, and exactly length top_s."""

        rng = rng or np.random.default_rng()
        seen: set[Hashable] = set()
        repaired: list[Hashable] = []
        candidate_set = set(self.candidates)
        for item in items:
            if item in candidate_set and item not in seen:
                repaired.append(item)
                seen.add(item)
            if len(repaired) == self.top_s:
                break
        if len(repaired) < self.top_s:
            missing = [item for item in self.candidates if item not in seen]
            fill_n = self.top_s - len(repaired)
            if fill_n > len(missing):
                raise ValueError("Cannot repair solution because candidate pool is too small.")
            fill = rng.choice(np.array(missing, dtype=object), size=fill_n, replace=False).tolist()
            repaired.extend(fill)
        return tuple(repaired)

    def random_solution(self, rng: np.random.Generator) -> Solution:
        items = rng.choice(np.array(self.candidates, dtype=object), size=self.top_s, replace=False)
        return self.make_solution(items.tolist())

    def random_population(self, size: int, rng: np.random.Generator) -> list[Solution]:
        return [self.random_solution(rng) for _ in range(size)]
