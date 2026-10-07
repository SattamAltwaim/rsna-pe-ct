"""From the slice-level ``train.csv`` to one row per study with a binary target.

Study groups (mutually exclusive):

* ``negative``      ``negative_exam_for_pe == 1``
* ``pe``            ``negative_exam_for_pe == 0 and indeterminate == 0``
* ``indeterminate`` ``indeterminate == 1`` (poor-quality scans; also has
  ``negative_exam_for_pe == 0``, so it must not be counted as positive)

Binary target ``y``: 1 for ``pe``, 0 for ``negative``; indeterminate studies are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pe_ct.config import GROUP_COLS, QA_COLS, SUBLABEL_COLS

ID_COLS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
SLICE_LABEL = "pe_present_on_image"
STUDY_LEVEL_COLS = GROUP_COLS + SUBLABEL_COLS + QA_COLS


def load_train_csv(path) -> pd.DataFrame:
    """``train.csv`` with compact dtypes (1.79M rows)."""
    dtypes = {c: "string" for c in ID_COLS}
    dtypes.update({c: "int8" for c in [SLICE_LABEL] + STUDY_LEVEL_COLS})
    return pd.read_csv(path, dtype=dtypes)


def assign_group(df: pd.DataFrame) -> pd.Series:
    """One of ``negative`` / ``pe`` / ``indeterminate`` per row."""
    group = np.where(
        df["indeterminate"] == 1,
        "indeterminate",
        np.where(df["negative_exam_for_pe"] == 1, "negative", "pe"),
    )
    return pd.Series(group, index=df.index, dtype="string")


def study_table(train: pd.DataFrame) -> pd.DataFrame:
    """Collapse slice rows to one row per study (asserts the dataset's invariants)."""
    g = train.groupby("StudyInstanceUID", sort=True)
    n_series = g["SeriesInstanceUID"].nunique()
    if (n_series != 1).any():
        raise ValueError(f"{(n_series != 1).sum()} studies have != 1 series")
    varying = g[STUDY_LEVEL_COLS].nunique().max()
    if (varying > 1).any():
        raise ValueError(f"study-level columns vary within a study: {varying[varying > 1].to_dict()}")

    out = g.agg(
        series_uid=("SeriesInstanceUID", "first"),
        n_slices=("SOPInstanceUID", "size"),
        n_pos_slices=(SLICE_LABEL, "sum"),
        **{c: (c, "first") for c in STUDY_LEVEL_COLS},
    ).reset_index().rename(columns={"StudyInstanceUID": "study_uid"})
    out["group"] = assign_group(out)
    out["y"] = (out["group"] == "pe").astype("int8")
    out["n_pos_slices"] = out["n_pos_slices"].astype("int32")
    out["frac_pos_slices"] = out["n_pos_slices"] / out["n_slices"]
    cols = ["study_uid", "series_uid", "n_slices", "n_pos_slices", "frac_pos_slices", "group", "y"]
    return out[cols + STUDY_LEVEL_COLS]


def usable_studies(studies: pd.DataFrame) -> pd.DataFrame:
    """Drop the indeterminate (poor-quality) studies."""
    return studies[studies["group"] != "indeterminate"].reset_index(drop=True)


def group_counts(studies: pd.DataFrame) -> pd.DataFrame:
    counts = studies["group"].value_counts().reindex(["negative", "pe", "indeterminate"]).fillna(0)
    out = counts.rename("n_studies").to_frame()
    out["fraction"] = (out["n_studies"] / out["n_studies"].sum()).round(4)
    return out.astype({"n_studies": int})


def sublabel_prevalence(studies: pd.DataFrame) -> pd.Series:
    """Fraction of PE-positive studies carrying each sub-label, descending."""
    pe = studies[studies["group"] == "pe"]
    return pe[SUBLABEL_COLS].mean().sort_values(ascending=False)


def sublabel_cooccurrence(studies: pd.DataFrame) -> pd.DataFrame:
    """P(col | row) among PE-positive studies: how often col is set when row is set."""
    pe = studies[studies["group"] == "pe"][SUBLABEL_COLS].astype(float)
    joint = pe.T @ pe
    marg = np.diag(joint.values)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = joint.values / marg[:, None]
    return pd.DataFrame(np.nan_to_num(cond), index=SUBLABEL_COLS, columns=SUBLABEL_COLS)


class SliceLabelIndex:
    """Fast ``study_uid -> (sop_uids, labels)`` lookup for aligning labels to volumes."""

    def __init__(self, train: pd.DataFrame):
        self._groups = train.groupby("StudyInstanceUID", sort=False)

    def __contains__(self, study_uid: str) -> bool:
        return study_uid in self._groups.groups

    def for_study(self, study_uid: str) -> pd.Series:
        """``pe_present_on_image`` indexed by ``SOPInstanceUID`` (csv row order kept)."""
        g = self._groups.get_group(study_uid)
        return pd.Series(g[SLICE_LABEL].to_numpy(dtype=np.int8), index=g["SOPInstanceUID"].to_numpy())


def align_slice_labels(sop_uids_in_z_order, slice_labels: pd.Series) -> np.ndarray:
    """``labels[z]`` for the volume's z-th slice. Raises if a slice has no label."""
    sop_uids_in_z_order = list(sop_uids_in_z_order)
    missing = [u for u in sop_uids_in_z_order if u not in slice_labels.index]
    if missing:
        raise KeyError(f"{len(missing)} slices without a label, e.g. {missing[0]}")
    extra = len(slice_labels) - len(sop_uids_in_z_order)
    if extra != 0:
        raise ValueError(f"label count {len(slice_labels)} != slice count {len(sop_uids_in_z_order)}")
    return slice_labels.loc[sop_uids_in_z_order].to_numpy(dtype=np.int8)


def positive_runs(labels: np.ndarray) -> list:
    """``[(start, end_exclusive), ...]`` of consecutive positive slices."""
    labels = np.asarray(labels).astype(bool)
    if labels.size == 0:
        return []
    padded = np.concatenate([[False], labels, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def positive_band(labels: np.ndarray):
    """``(first_positive, last_positive_exclusive)`` or ``None`` if no positives."""
    pos = np.where(np.asarray(labels) == 1)[0]
    if pos.size == 0:
        return None
    return int(pos.min()), int(pos.max()) + 1


def representative_positive_slice(labels: np.ndarray):
    """Middle of the longest positive run (the slice most likely to show the clot)."""
    runs = positive_runs(labels)
    if not runs:
        return None
    start, end = max(runs, key=lambda r: r[1] - r[0])
    return (start + end - 1) // 2
