import numpy as np
import pandas as pd
import pytest
import torch

from pe_ct import probe


def _data(n=600, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += 2.0 * y  # one informative direction
    X[:, 1] *= 10  # un-standardized feature
    folds = rng.integers(0, 5, n)
    return X, y, folds


def test_train_probe_learns_separable_signal():
    X, y, folds = _data()
    tr, va = folds != 0, folds == 0
    model, hist = probe.train_probe(X[tr], y[tr], X[va], y[va], epochs=300, lr=0.05, patience=50)
    p = probe.predict_proba(model, X[va])
    m = probe.compute_metrics(y[va], p, probe.best_threshold(y[va], p))
    assert m["auroc"] > 0.85 and m["macro_f1"] > 0.7
    assert {"epoch", "train_loss", "val_loss", "train_auprc", "val_auprc"} <= set(hist.columns)
    assert hist.attrs["best_epoch"] <= len(hist) - 1
    sk = probe.sklearn_baseline(X[tr], y[tr], X[va])
    assert abs(probe.compute_metrics(y[va], sk)["auroc"] - m["auroc"]) < 0.05
    fn = probe.predict_logit_fn(model)
    logit = fn(torch.as_tensor(X[va][:3]))
    np.testing.assert_allclose(torch.sigmoid(logit).numpy(), p[:3], atol=1e-5)


def test_metrics_and_thresholds():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.4, 0.6, 0.3, 0.8, 0.9])
    m = probe.compute_metrics(y, prob, 0.5)
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 2
    assert m["recall_pe"] == pytest.approx(2 / 3) and m["accuracy"] == pytest.approx(4 / 6)
    assert m["macro_f1"] == pytest.approx(2 / 3)
    sweep = probe.threshold_sweep(y, prob)
    assert sweep.f1.max() >= m["f1_pe"]
    assert probe.confusion(y, prob, 0.5).tolist() == [[2, 1], [1, 2]]
    rep = probe.report_table(y, prob, y, prob, 0.5)
    assert ("train", "recall") in rep.columns and "PE" in rep.index
    ci = probe.bootstrap_metrics(y, prob, 0.5, n_boot=50)
    assert (ci["ci_lo"] <= ci["value"] + 1e-9).all() and (ci["ci_hi"] >= ci["value"] - 1e-9).all()


def test_cross_validate_learning_curve_and_subgroups():
    X, y, folds = _data(n=400)
    per_fold, oof = probe.cross_validate(X, y, folds, n_folds=5, epochs=60, lr=0.05, patience=20)
    assert len(per_fold) == 5 and np.isfinite(oof).all()
    summary = probe.cv_summary(per_fold)
    assert "auroc" in summary.index and summary.loc["auroc", "mean"] > 0.8
    lc = probe.learning_curve(X, y, folds, sizes=[50, 150], val_fold=0, epochs=60, lr=0.05, patience=20)
    assert lc.n_train.tolist() == [50, 150]
    df = pd.DataFrame({"y": y, "pred": (oof >= 0.5).astype(int), "n_pos_slices": np.where(y == 1, 20, 0),
                       "central_pe": 0, "rv_lv_ratio_gte_1": 0, "rv_lv_ratio_lt_1": 1, "chronic_pe": 0, "acute_and_chronic_pe": 0})
    rb = probe.recall_by_subgroup(df)
    assert rb.loc["11-50 slices", "n"] == int(y.sum()) and 0 <= rb.loc["11-50 slices", "recall"] <= 1
    assert np.isnan(rb.loc["central", "recall"])
