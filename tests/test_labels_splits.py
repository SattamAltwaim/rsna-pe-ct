import numpy as np
import pandas as pd
import pytest

from pe_ct import labels, splits
from pe_ct.config import Config


def test_study_table_groups_and_target(tiny_train_csv):
    st = labels.study_table(tiny_train_csv)
    assert len(st) == 30
    counts = labels.group_counts(st)
    assert counts.loc["negative", "n_studies"] == 18
    assert counts.loc["pe", "n_studies"] == 9
    assert counts.loc["indeterminate", "n_studies"] == 3
    assert set(st.loc[st.group == "pe", "y"]) == {1}
    assert set(st.loc[st.group != "pe", "y"]) == {0}
    assert (st.loc[st.group == "negative", "n_pos_slices"] == 0).all()
    assert (st.loc[st.group == "pe", "n_pos_slices"] > 0).all()
    assert len(labels.usable_studies(st)) == 27


def test_study_table_rejects_varying_study_columns(tiny_train_csv):
    bad = tiny_train_csv.copy()
    bad.loc[0, "central_pe"] = 1 - bad.loc[0, "central_pe"]
    with pytest.raises(ValueError):
        labels.study_table(bad)


def test_slice_label_alignment(tiny_train_csv):
    idx = labels.SliceLabelIndex(tiny_train_csv)
    s = idx.for_study("study020")
    sops = list(s.index)[::-1]  # pretend the volume is in reverse order
    aligned = labels.align_slice_labels(sops, s)
    np.testing.assert_array_equal(aligned, s.to_numpy()[::-1])
    with pytest.raises(KeyError):
        labels.align_slice_labels(sops + ["missing"], s)
    with pytest.raises(ValueError):
        labels.align_slice_labels(sops[:-1], s)


def test_positive_runs_and_band():
    lab = np.array([0, 1, 1, 0, 0, 1, 0])
    assert labels.positive_runs(lab) == [(1, 3), (5, 6)]
    assert labels.positive_band(lab) == (1, 6)
    assert labels.representative_positive_slice(lab) == 1
    assert labels.positive_band(np.zeros(3)) is None
    assert labels.representative_positive_slice(np.zeros(3)) is None


def test_sublabel_tables(tiny_train_csv):
    st = labels.study_table(tiny_train_csv)
    prev = labels.sublabel_prevalence(st)
    assert prev["leftsided_pe"] == 1.0
    co = labels.sublabel_cooccurrence(st)
    assert co.loc["leftsided_pe", "leftsided_pe"] == 1.0
    assert co.loc["chronic_pe", "leftsided_pe"] == 0.0  # no chronic studies -> 0, not NaN


def _bigger_studies(n=400, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({"study_uid": [f"s{i}" for i in range(n)], "y": y, "group": np.where(y == 1, "pe", "negative")})
    for c in ["rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1", "central_pe", "leftsided_pe", "rightsided_pe", "chronic_pe", "acute_and_chronic_pe"]:
        df[c] = ((rng.random(n) < 0.4) & (y == 1)).astype(int)
    return df


def test_make_splits_is_stratified_leakfree_and_deterministic():
    studies = _bigger_studies()
    cfg = Config(test_frac=0.2, n_eda=40, n_dev=200, n_test_sample=50, n_folds=5, seed=3)
    sp = splits.make_splits(studies, cfg)
    sp2 = splits.make_splits(studies, cfg)
    pd.testing.assert_frame_equal(sp, sp2)
    splits.check_no_leakage(sp)
    assert (sp.pool == "test").sum() == 80
    assert sp.in_test_sample.sum() == 50 and sp.in_dev.sum() == 200 and sp.in_eda.sum() == 40
    assert sp.loc[sp.in_eda, "y"].mean() == 0.5
    assert set(sp.loc[sp.in_dev, "fold"]) == {0, 1, 2, 3, 4}
    assert (sp.loc[~sp.in_dev, "fold"] == -1).all()
    summary = splits.split_summary(sp)
    overall = sp.y.mean()
    for name in ["dev pool", "test pool", "dev subset", "test sample"]:
        assert abs(summary.loc[name, "pe_frac"] - overall) < 0.03
