from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import pandas as pd
import yaml

from himars.config import HiMARSConfig
from himars.data import load_ratings
from himars.experiment import ALGORITHM_NAMES, run_many_users


def load_config(path: str | Path) -> tuple[dict, HiMARSConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg_keys = {f.name for f in fields(HiMARSConfig)}
    cfg = HiMARSConfig(**{k: v for k, v in raw.items() if k in cfg_keys})
    cfg.validate()
    return raw, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HiMARS experiments.")
    parser.add_argument("--config", required=True, help="YAML config path, e.g. configs/movielens.yaml")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV outputs")
    parser.add_argument("--algorithms", nargs="*", default=list(ALGORITHM_NAMES))
    parser.add_argument("--n-simulations", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    args = parser.parse_args()

    raw, config = load_config(args.config)
    if args.n_simulations is not None:
        config.n_simulations = args.n_simulations
    if args.max_iter is not None:
        config.max_iter = args.max_iter

    ratings_path = raw.get("ratings_path")
    dataset = raw.get("dataset")
    users = raw.get("users")
    if not ratings_path or not users:
        raise ValueError("Config must contain ratings_path and users.")

    ratings = load_ratings(ratings_path, dataset=dataset)
    result = run_many_users(ratings, users, config, algorithms=args.algorithms)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in result.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    print(f"Saved outputs in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
