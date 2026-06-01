from __future__ import annotations

import numpy as np
import pandas as pd

from himars.algorithms import run_amosa, run_han_v1, run_han_v2, run_hani_v1, run_hani_v2, run_nnia, run_nsga2
from himars.config import HiMARSConfig
from himars.data import make_rating_matrix, split_ratings
from himars.experiment import build_problem_for_user
from himars.metrics import pareto_front_metrics
from himars.operators import mutate_items, one_point_crossover
from himars.pareto import dominates, first_front
from himars.problem import Solution
from himars.recommender import ItemBasedCF
from himars.selection import select_by_ideal_point, topsis_rank


def synthetic_ratings() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(123)
    for user in range(1, 9):
        for item in range(1, 15):
            if rng.random() < 0.72:
                rows.append({"user_id": user, "item_id": item, "rating": int(rng.integers(1, 6))})
    return pd.DataFrame(rows)


def test_adjusted_cosine_recommender_predicts_candidates():
    ratings = synthetic_ratings()
    train, _ = split_ratings(ratings, random_state=1)
    matrix = make_rating_matrix(train)
    user = matrix.index[0]
    rec = ItemBasedCF(n_neighbors=3).fit(matrix)
    top = rec.recommend_top_k(user, k=5)
    assert list(top.columns) == ["item_id", "predicted_rating"]
    assert len(top) <= 5
    assert rec.similarity_df.shape[0] == rec.similarity_df.shape[1]


def test_dominance_and_selection():
    a = Solution((1, 2), (2.0, 2.0))
    b = Solution((2, 3), (1.0, 2.0))
    c = Solution((3, 4), (2.0, 1.0))
    assert dominates(a, b)
    assert len(first_front([a, b, c])) == 1
    assert select_by_ideal_point([a, b, c]).items == a.items


def test_operators_return_unique_feasible_children():
    rng = np.random.default_rng(1)
    candidates = list(range(10))
    c1, c2 = one_point_crossover((1, 2, 3), (3, 4, 5), candidates, 3, rng)
    assert len(c1) == len(set(c1)) == 3
    assert len(c2) == len(set(c2)) == 3
    m = mutate_items(c1, candidates, rng, mutation_probability=1.0)
    assert len(m) == len(set(m)) == 3


def test_small_algorithms_smoke():
    ratings = synthetic_ratings()
    train, test = split_ratings(ratings, random_state=2)
    matrix = make_rating_matrix(train)
    user = matrix.index[0]
    cfg = HiMARSConfig(
        top_k=8,
        top_s=3,
        n_neighbors=3,
        max_iter=2,
        archive_init_iter=1,
        pop_size=8,
        nd=8,
        na=3,
        nc=4,
        hard_limit=8,
        soft_limit=10,
        random_state=7,
    )
    rec = ItemBasedCF(n_neighbors=cfg.n_neighbors).fit(matrix)
    problem, _, _, _ = build_problem_for_user(user, matrix, test, ratings, cfg, recommender=rec)
    initial = run_nnia(problem, cfg, seed=10, max_iter=1)
    runners = [
        lambda: run_nnia(problem, cfg, seed=1),
        lambda: run_nsga2(problem, cfg, seed=2),
        lambda: run_amosa(problem, cfg, seed=3, initial_archive=initial),
        lambda: run_han_v1(problem, cfg, seed=4),
        lambda: run_han_v2(problem, cfg, seed=5, initial_archive=initial),
        lambda: run_hani_v1(problem, cfg, seed=6),
        lambda: run_hani_v2(problem, cfg, seed=7, initial_archive=initial),
    ]
    for run in runners:
        sols = run()
        assert len(sols) > 0
        assert all(len(s.items) == cfg.top_s for s in sols)
        assert all(len(set(s.items)) == cfg.top_s for s in sols)
        metrics = pareto_front_metrics(sols)
        assert set(["SM", "MID", "DM", "SNS", "n_solutions"]).issubset(metrics)


def test_topsis_rank_order():
    table = pd.DataFrame(
        {
            "SM": [0.2, 0.1],
            "MID": [0.4, 0.2],
            "DM": [1.0, 2.0],
            "SNS": [1.0, 2.0],
        },
        index=["A", "B"],
    )
    ranked = topsis_rank(table)
    assert ranked.index[0] == "B"
