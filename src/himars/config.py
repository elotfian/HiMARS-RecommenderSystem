from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HiMARSConfig:
    """Central configuration for paper-style HiMARS experiments."""

    # general experiment settings
    test_size: float = 0.20
    random_state: int = 42
    n_simulations: int = 20
    max_iter: int = 200
    top_k: int = 100
    top_s: int = 10
    n_neighbors: int = 20
    rating_threshold: float = 3.0

    # NSGA-II
    pop_size: int = 100
    crossover_probability: float = 0.70
    nsga_mutation_probability: float = 0.20

    # NNIA
    nd: int = 100
    na: int = 10
    nc: int = 40
    nnia_mutation_probability: float = 0.10

    # AMOSA
    soft_limit: int = 140
    hard_limit: int = 100
    tau: float = 1.0
    cooling_rate: float = 0.90

    # archive initialization used by AMOSA/HANv2/HANIv2
    archive_init_iter: int = 50

    # reproducibility/performance controls
    show_progress: bool = False
    positive_neighbors_only: bool = True
    similarity: str = "adjusted_cosine"  # "adjusted_cosine" or "cosine"

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.top_s <= 0 or self.top_k <= 0:
            raise ValueError("top_s and top_k must be positive.")
        if self.top_s > self.top_k:
            raise ValueError("top_s must be <= top_k.")
        if self.max_iter < 0 or self.archive_init_iter < 0:
            raise ValueError("iteration counts must be non-negative.")
        for name in ["pop_size", "nd", "na", "nc", "hard_limit", "soft_limit"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.na > self.nd:
            raise ValueError("na must be <= nd.")
        if self.hard_limit > self.soft_limit:
            raise ValueError("hard_limit must be <= soft_limit.")
