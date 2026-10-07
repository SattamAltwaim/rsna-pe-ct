"""Leak-free, stratified study-level splits, decided once in NB00.

* ``pool``: ``test`` (frozen, 20 %) or ``dev`` (everything else).
* ``in_test_sample``: the subset of the test pool evaluated once at the end.
* ``in_dev``: the working dev subset (train + val, 5-fold CV via ``fold``).
* ``in_eda``: a class-balanced subset of ``in_dev`` whose PE half covers every sub-type.

Stratification key: ``y`` x ``central_pe`` x ``rv_lv_ratio_gte_1`` so that rare,
clinically important PE kinds appear in the same proportion everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from pe_ct.config import SUBLABEL_COLS, Config

STRAT_COLS = ["y", "central_pe", "rv_lv_ratio_gte_1"]


def strat_key(df: pd.DataFrame) -> pd.Series:
    return df[STRAT_COLS].astype(int).astype(str).agg("-".join, axis=1)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of ``df`` keeping the strat-key proportions (falls back if too few per key)."""
    if n >= len(df):
        return df
    key = strat_key(df)
    counts = key.value_counts()
    rare = key.isin(counts[counts < 2].index)
    try:
        picked, _ = train_test_split(df[~rare], train_size=n, stratify=key[~rare], random_state=seed)
    except ValueError:  # n too small for the number of strata
        picked = df[~rare].sample(n=n, random_state=seed)
    return picked


def make_splits(studies: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build the split table from the *usable* study table (indeterminate already dropped)."""
    df = studies[["study_uid", "y"] + SUBLABEL_COLS].copy().reset_index(drop=True)
    key = strat_key(df)

    dev_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=cfg.test_frac, stratify=key, random_state=cfg.seed
    )
    df["pool"] = "dev"
    df.loc[test_idx, "pool"] = "test"

    test_pool = df[df["pool"] == "test"]
    test_sample = _stratified_sample(test_pool, cfg.n_test_sample, cfg.seed)
    df["in_test_sample"] = df["study_uid"].isin(test_sample["study_uid"])

    dev_pool = df[df["pool"] == "dev"]
    dev = _stratified_sample(dev_pool, cfg.n_dev, cfg.seed + 1)
    df["in_dev"] = df["study_uid"].isin(dev["study_uid"])

    # EDA subset: half PE (stratified over sub-types), half negative, drawn from the dev subset.
    half = cfg.n_eda // 2
    eda_pe = _stratified_sample(dev[dev["y"] == 1], half, cfg.seed + 2)
    eda_neg = dev[dev["y"] == 0].sample(n=min(half, (dev["y"] == 0).sum()), random_state=cfg.seed + 2)
    df["in_eda"] = df["study_uid"].isin(pd.concat([eda_pe, eda_neg])["study_uid"])

    # 5-fold stratified CV inside the dev subset.
    df["fold"] = -1
    dev_rows = df.index[df["in_dev"]]
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_pos) in enumerate(skf.split(dev_rows, key[dev_rows])):
        df.loc[dev_rows[val_pos], "fold"] = fold

    df["fold"] = df["fold"].astype("int8")
    for col in ["in_test_sample", "in_dev", "in_eda"]:
        df[col] = df[col].astype(bool)
    return df


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Rows = named subsets; columns = n, n_pe, pe fraction, and sub-type fractions."""
    subsets = {
        "all usable": splits,
        "dev pool": splits[splits["pool"] == "dev"],
        "test pool": splits[splits["pool"] == "test"],
        "dev subset": splits[splits["in_dev"]],
        "eda subset": splits[splits["in_eda"]],
        "test sample": splits[splits["in_test_sample"]],
    }
    for f in range(int(splits["fold"].max()) + 1):
        subsets[f"dev fold {f}"] = splits[splits["fold"] == f]
    rows = []
    for name, sub in subsets.items():
        pe = sub[sub["y"] == 1]
        rows.append(
            {
                "subset": name,
                "n": len(sub),
                "n_pe": int(sub["y"].sum()),
                "pe_frac": round(sub["y"].mean(), 4) if len(sub) else np.nan,
                "central_pe_frac_of_pe": round(pe["central_pe"].mean(), 4) if len(pe) else np.nan,
                "rv_lv_gte_1_frac_of_pe": round(pe["rv_lv_ratio_gte_1"].mean(), 4) if len(pe) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("subset")


def check_no_leakage(splits: pd.DataFrame) -> None:
    dev = set(splits.loc[splits["in_dev"], "study_uid"])
    test = set(splits.loc[splits["pool"] == "test", "study_uid"])
    if dev & test:
        raise AssertionError(f"{len(dev & test)} studies in both dev and test")
    eda = set(splits.loc[splits["in_eda"], "study_uid"])
    if not eda <= dev:
        raise AssertionError("EDA subset must be inside the dev subset")
    if splits["study_uid"].duplicated().any():
        raise AssertionError("duplicate study_uid in splits")
