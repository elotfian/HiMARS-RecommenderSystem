"""HiMARS: Hybrid multi-objective algorithms for recommender systems."""

from .config import HiMARSConfig
from .recommender import ItemBasedCF
from .problem import RecommendationProblem, Solution
from .objectives import ObjectiveEvaluator
from .selection import select_by_ideal_point, topsis_rank
from .metrics import recommendation_metrics, pareto_front_metrics

__all__ = [
    "HiMARSConfig",
    "ItemBasedCF",
    "RecommendationProblem",
    "Solution",
    "ObjectiveEvaluator",
    "select_by_ideal_point",
    "topsis_rank",
    "recommendation_metrics",
    "pareto_front_metrics",
]
