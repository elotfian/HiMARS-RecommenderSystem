from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate HiMARS selected-list and Pareto metric outputs.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    selected_path = results_dir / "selected.csv"
    metrics_path = results_dir / "pareto_metrics.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        agg = selected.groupby(["algorithm"])[["accuracy", "diversity", "novelty", "f1", "f2"]].agg(["min", "max", "mean"])
        agg.to_csv(results_dir / "selected_summary.csv")
        print(f"Saved {results_dir / 'selected_summary.csv'}")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        cols = [c for c in ["SM", "MID", "DM", "SNS", "CLO"] if c in metrics.columns]
        agg = metrics.groupby(["algorithm"])[cols].agg(["min", "max", "mean"])
        agg.to_csv(results_dir / "pareto_metric_summary.csv")
        print(f"Saved {results_dir / 'pareto_metric_summary.csv'}")


if __name__ == "__main__":
    main()
