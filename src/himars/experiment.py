from __future__ import annotations

from dataclasses import asdict
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd

from .algorithms import (
    run_amosa,
    run_han_v1,
    run_han_v2,
    run_hani_v1,
    run_hani_v2,
    run_nnia,
    run_nsga2,
)
from .config import HiMARSConfig
from .data import make_rating_matrix, split_ratings
from .metrics import pareto_front_metrics
from .objectives import ObjectiveEvaluator
from .problem import RecommendationProblem, Solution
from .recommender import ItemBasedCF
from .selection import select_by_ideal_point, topsis_rank

ALGORITHM_NAMES = ("NNIA", "NSGAII", "AMOSA", "HANv1", "HANv2", "HANIv1", "HANIv2")


def build_problem_for_user(
    user_id: Hashable,
    train_matrix: pd.DataFrame,
    test_ratings: pd.DataFrame,
    all_ratings: pd.DataFrame,
    config: HiMARSConfig,
    recommender: ItemBasedCF | None = None,
) -> tuple[RecommendationProblem, pd.DataFrame, ObjectiveEvaluator, ItemBasedCF]:
    """Fit/use ICF, generate top-k candidates, and build a HiMARS problem."""

    if recommender is None:
        recommender = ItemBasedCF(
            n_neighbors=config.n_neighbors,
            similarity=config.similarity,
            positive_neighbors_only=config.positive_neighbors_only,
        ).fit(train_matrix)
    top_k_df = recommender.recommend_top_k(user_id, config.top_k)
    evaluator = ObjectiveEvaluator(
        user_id=user_id,
        train_matrix=train_matrix,
        similarity_df=recommender.similarity_df,
        test_ratings=test_ratings,
        all_ratings=all_ratings,
        rating_threshold=config.rating_threshold,
    )
    problem = RecommendationProblem(top_k_df["item_id"].tolist(), config.top_s, evaluator)
    return problem, top_k_df, evaluator, recommender


def run_algorithms_for_user(
    problem: RecommendationProblem,
    config: HiMARSConfig,
    seed: int,
    algorithms: Sequence[str] = ALGORITHM_NAMES,
) -> dict[str, list[Solution]]:
    """Run selected algorithms for one target user."""

    rng_master = np.random.default_rng(seed)

    def next_seed() -> int:
        return int(rng_master.integers(0, 2**32 - 1))

    results: dict[str, list[Solution]] = {}
    # Archive initialization is generated once per simulation/user for methods that need it.
    archive_cfg = HiMARSConfig(**asdict(config))
    archive_cfg.max_iter = config.archive_init_iter
    initial_archive = run_nnia(problem, archive_cfg, seed=next_seed(), max_iter=config.archive_init_iter)

    for name in algorithms:
        key = name.upper()
        if key == "NNIA":
            results["NNIA"] = run_nnia(problem, config, seed=next_seed())
        elif key in {"NSGAII", "NSGA-II"}:
            results["NSGAII"] = run_nsga2(problem, config, seed=next_seed())
        elif key == "AMOSA":
            results["AMOSA"] = run_amosa(problem, config, seed=next_seed(), initial_archive=initial_archive)
        elif key == "HANV1":
            results["HANv1"] = run_han_v1(problem, config, seed=next_seed())
        elif key == "HANV2":
            results["HANv2"] = run_han_v2(problem, config, seed=next_seed(), initial_archive=initial_archive)
        elif key == "HANIV1":
            results["HANIv1"] = run_hani_v1(problem, config, seed=next_seed())
        elif key == "HANIV2":
            results["HANIv2"] = run_hani_v2(problem, config, seed=next_seed(), initial_archive=initial_archive)
        else:
            raise ValueError(f"Unknown algorithm: {name}")
    return results


def summarize_user_results(
    results: dict[str, list[Solution]],
    evaluator: ObjectiveEvaluator,
    icf_solution: Solution | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return solution-level, selected-list, and Pareto metric tables."""

    solution_rows = []
    selected_rows = []
    metric_rows = []

    if icf_solution is not None:
        q = evaluator.recommendation_quality(icf_solution.items)
        selected_rows.append({"algorithm": "Item_CF", **icf_solution.to_dict(), **q})

    for algorithm, sols in results.items():
        metrics = pareto_front_metrics(sols)
        metric_rows.append({"algorithm": algorithm, **metrics})
        if sols:
            selected = select_by_ideal_point(sols)
            q = evaluator.recommendation_quality(selected.items)
            selected_rows.append({"algorithm": algorithm, **selected.to_dict(), **q})
        for sol in sols:
            solution_rows.append({"algorithm": algorithm, **sol.to_dict()})

    solutions_df = pd.DataFrame(solution_rows)
    selected_df = pd.DataFrame(selected_rows)
    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        ranked = topsis_rank(metrics_df.set_index("algorithm"))
        metrics_df = ranked.reset_index().rename(columns={"index": "algorithm"})
    return solutions_df, selected_df, metrics_df


def run_single_user_experiment(
    ratings: pd.DataFrame,
    user_id: Hashable,
    config: HiMARSConfig,
    simulation: int = 0,
    algorithms: Sequence[str] = ALGORITHM_NAMES,
) -> dict[str, pd.DataFrame]:
    """Run the complete HiMARS workflow for one user and one simulation."""

    config.validate()
    train, test = split_ratings(ratings, test_size=config.test_size, random_state=config.random_state)
    train_matrix = make_rating_matrix(train)
    problem, top_k_df, evaluator, _ = build_problem_for_user(user_id, train_matrix, test, ratings, config)
    seed = config.random_state + 1009 * simulation + hash(user_id) % 100000
    results = run_algorithms_for_user(problem, config, seed=seed, algorithms=algorithms)
    icf_items = top_k_df["item_id"].head(config.top_s).tolist()
    icf_solution = problem.make_solution(icf_items)
    solutions_df, selected_df, metrics_df = summarize_user_results(results, evaluator, icf_solution)
    for df in [solutions_df, selected_df, metrics_df]:
        if not df.empty:
            df.insert(0, "simulation", simulation + 1)
            df.insert(0, "user_id", user_id)
    return {
        "top_k": top_k_df,
        "solutions": solutions_df,
        "selected": selected_df,
        "pareto_metrics": metrics_df,
    }


def run_many_users(
    ratings: pd.DataFrame,
    users: Iterable[Hashable],
    config: HiMARSConfig,
    algorithms: Sequence[str] = ALGORITHM_NAMES,
) -> dict[str, pd.DataFrame]:
    """Run the full paper-style experiment for multiple users/simulations."""

    all_solutions = []
    all_selected = []
    all_metrics = []
    for sim in range(config.n_simulations):
        for user_id in users:
            out = run_single_user_experiment(ratings, user_id, config, sim, algorithms)
            all_solutions.append(out["solutions"])
            all_selected.append(out["selected"])
            all_metrics.append(out["pareto_metrics"])
    return {
        "solutions": pd.concat(all_solutions, ignore_index=True) if all_solutions else pd.DataFrame(),
        "selected": pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame(),
        "pareto_metrics": pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame(),
    }
