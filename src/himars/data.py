from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

_REQUIRED_COLUMNS = ("user_id", "item_id", "rating")


def _normalise_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical user_id, item_id and rating columns.

    The original notebooks used MovieLens ``ratings.dat`` and a ModCloth CSV. This
    helper accepts common column variants so that a user does not have to edit the
    package for small dataset-format differences.
    """

    aliases = {
        "user_id": ["user_id", "userid", "user", "userID", "userId"],
        "item_id": ["item_id", "itemid", "item", "movie_id", "movieId", "movieID", "product_id"],
        "rating": ["rating", "score", "overall", "quality", "fit_rating"],
    }
    lower_to_original = {str(c).lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            key = name.lower()
            if key in lower_to_original:
                rename[lower_to_original[key]] = canonical
                break
    out = df.rename(columns=rename).copy()
    missing = [c for c in _REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            "Missing required rating columns "
            f"{missing}. Available columns are {list(df.columns)!r}."
        )
    out = out.loc[:, [c for c in ["user_id", "item_id", "rating", "timestamp"] if c in out.columns]]
    out = out.dropna(subset=["user_id", "item_id", "rating"])
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna(subset=["rating"])
    return out


def load_movielens_1m(path: str | Path) -> pd.DataFrame:
    """Load the MovieLens 1M ratings.dat file.

    Expected format: ``user_id::item_id::rating::timestamp``.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MovieLens ratings file not found: {path}")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        header=None,
    )
    return _normalise_rating_columns(df)


def load_modcloth(path: str | Path) -> pd.DataFrame:
    """Load a ModCloth-style CSV and normalize its columns."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ModCloth ratings file not found: {path}")
    return _normalise_rating_columns(pd.read_csv(path))


def load_ratings(path: str | Path, dataset: str | None = None) -> pd.DataFrame:
    """Load ratings for a supported dataset.

    Parameters
    ----------
    path:
        Dataset file path.
    dataset:
        ``"movielens"``, ``"modcloth"`` or ``None``. If ``None``, a CSV is
        assumed unless the filename is ``ratings.dat``.
    """

    path = Path(path)
    name = (dataset or "").lower()
    if name in {"movielens", "ml-1m"} or path.name == "ratings.dat":
        return load_movielens_1m(path)
    if name == "modcloth" or path.suffix.lower() == ".csv":
        return load_modcloth(path)
    raise ValueError("dataset must be 'movielens', 'modcloth', or inferable from path.")


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify_by_user: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets.

    The original notebooks used a global random split. ``stratify_by_user`` is
    optional because sparse datasets may contain users with too few ratings for
    stratified splitting.
    """

    ratings = _normalise_rating_columns(ratings)
    stratify = ratings["user_id"] if stratify_by_user else None
    try:
        train, test = train_test_split(
            ratings, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_rating_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create a user-item rating matrix with users as rows and items as columns."""

    ratings = _normalise_rating_columns(ratings)
    matrix = ratings.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
    return matrix.sort_index(axis=0).sort_index(axis=1)


def sparsity(ratings: pd.DataFrame) -> float:
    """Return sparsity percentage of a ratings dataframe."""

    ratings = _normalise_rating_columns(ratings)
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    if n_users == 0 or n_items == 0:
        return 100.0
    observed = ratings.drop_duplicates(["user_id", "item_id"]).shape[0]
    return 100.0 * (1.0 - observed / (n_users * n_items))


def ensure_users_exist(users: Iterable, rating_matrix: pd.DataFrame) -> list:
    """Filter user IDs to those available in a fitted rating matrix."""

    available = set(rating_matrix.index)
    return [u for u in users if u in available]
