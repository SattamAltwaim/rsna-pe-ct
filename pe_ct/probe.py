"""Linear probe on frozen scan embeddings, plus every metric the notebooks report.

Headline metric for classification is **macro F1** (reported with accuracy);
AUROC / AUPRC are threshold-free companions, and the operating threshold is
chosen on validation data only.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn


class LinearProbe(nn.Module):
    """Standardize (train-fold statistics) -> one ``nn.Linear(d, 1)`` -> logit."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.linear = nn.Linear(len(mean), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear((x - self.mean) / self.std).squeeze(-1)


def fit_standardizer(X_train: np.ndarray) -> tuple:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def _tensor(X, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)


def train_probe(X_tr, y_tr, X_val, y_val, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-2,
                patience: int = 20, seed: int = 0, device: str = "cpu", verbose: bool = False):
    """Full-batch AdamW on BCE with ``pos_weight = n_neg / n_pos``; early stop on val AUPRC.

    Returns ``(probe, history)`` where history has one row per epoch.
    """
    torch.manual_seed(seed)
    mean, std = fit_standardizer(np.asarray(X_tr))
    probe = LinearProbe(mean, std).to(device)
    Xt, Xv = _tensor(X_tr, device), _tensor(X_val, device)
    yt = torch.as_tensor(np.asarray(y_tr, dtype=np.float32), device=device)
    yv = torch.as_tensor(np.asarray(y_val, dtype=np.float32), device=device)
    pos_weight = torch.tensor([(yt == 0).sum() / max(1, (yt == 1).sum())], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    rows, best, best_state, since_best = [], -np.inf, None, 0
    for epoch in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        probe.eval()
        with torch.no_grad():
            val_logits = probe(Xv)
            val_loss = loss_fn(val_logits, yv).item()
            p_tr = torch.sigmoid(logits).cpu().numpy()
            p_val = torch.sigmoid(val_logits).cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss,
            "train_auprc": average_precision_score(y_tr, p_tr),
            "val_auprc": average_precision_score(y_val, p_val),
            "train_auroc": roc_auc_score(y_tr, p_tr),
            "val_auroc": roc_auc_score(y_val, p_val),
        }
        rows.append(row)
        if row["val_auprc"] > best + 1e-6:
            best, best_state, since_best = row["val_auprc"], copy.deepcopy(probe.state_dict()), 0
        else:
            since_best += 1
        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:4d}  train loss {row['train_loss']:.4f}  val loss {val_loss:.4f}  val AUPRC {row['val_auprc']:.4f}")
        if since_best >= patience:
            break
    probe.load_state_dict(best_state)
    probe.eval()
    history = pd.DataFrame(rows)
    history.attrs["best_epoch"] = int(history["val_auprc"].idxmax())
    return probe, history


@torch.no_grad()
def predict_proba(probe: LinearProbe, X) -> np.ndarray:
    device = probe.mean.device
    return torch.sigmoid(probe(_tensor(X, device))).cpu().numpy()


def predict_logit_fn(probe: LinearProbe):
    """Torch-in / torch-out logit function (used by the occlusion code in NB05)."""
    probe.eval()

    @torch.no_grad()
    def fn(cls: torch.Tensor) -> torch.Tensor:
        return probe(cls.to(device=probe.mean.device, dtype=torch.float32))

    return fn


def sklearn_baseline(X_tr, y_tr, X_val, C: float = 1.0, seed: int = 0) -> np.ndarray:
    """Balanced logistic regression on standardized features; sanity check for the torch probe."""
    mean, std = fit_standardizer(np.asarray(X_tr))
    clf = LogisticRegression(class_weight="balanced", C=C, max_iter=5000, random_state=seed)
    clf.fit((X_tr - mean) / std, y_tr)
    return clf.predict_proba((X_val - mean) / std)[:, 1]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def compute_metrics(y, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "threshold": float(threshold),
        "auroc": roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan,
        "auprc": average_precision_score(y, prob) if y.sum() > 0 else np.nan,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "f1_pe": f1_score(y, pred, pos_label=1, zero_division=0),
        "precision_pe": tp / max(1, tp + fp),
        "recall_pe": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "accuracy": (tp + tn) / max(1, len(y)),
        "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    return out


def threshold_sweep(y, prob, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0.02, 0.98, 97) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = compute_metrics(y, prob, t)
        rows.append({"threshold": t, "precision": m["precision_pe"], "recall": m["recall_pe"],
                     "f1": m["f1_pe"], "macro_f1": m["macro_f1"]})
    return pd.DataFrame(rows)


def best_threshold(y, prob, metric: str = "f1") -> float:
    sweep = threshold_sweep(y, prob)
    return float(sweep.loc[sweep[metric].idxmax(), "threshold"])


def curves(y, prob) -> dict:
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


def confusion(y, prob, threshold: float) -> np.ndarray:
    return confusion_matrix(np.asarray(y).astype(int), (np.asarray(prob) >= threshold).astype(int), labels=[0, 1])


def report_table(y_tr, p_tr, y_val, p_val, threshold: float) -> pd.DataFrame:
    """Per-class precision / recall / F1 for train and val side by side."""
    frames = []
    for name, y, p in [("train", y_tr, p_tr), ("val", y_val, p_val)]:
        rep = classification_report(np.asarray(y).astype(int), (np.asarray(p) >= threshold).astype(int),
                                    target_names=["no PE", "PE"], output_dict=True, zero_division=0)
        df = pd.DataFrame(rep).T[["precision", "recall", "f1-score", "support"]]
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1).round(3)


def bootstrap_metrics(y, prob, threshold: float, n_boot: int = 1000, seed: int = 0,
                      keys=("auroc", "auprc", "macro_f1", "f1_pe", "recall_pe", "precision_pe", "accuracy")) -> pd.DataFrame:
    """Point estimate + 95 % percentile CI per metric by resampling studies."""
    rng = np.random.default_rng(seed)
    y, prob = np.asarray(y), np.asarray(prob)
    point = compute_metrics(y, prob, threshold)
    samples = {k: [] for k in keys}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], prob[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    rows = [{"metric": k, "value": point[k], "ci_lo": np.percentile(samples[k], 2.5), "ci_hi": np.percentile(samples[k], 97.5)}
            for k in keys]
    return pd.DataFrame(rows).set_index("metric").round(4)


def cross_validate(X, y, folds, n_folds: int = 5, threshold_metric: str = "f1", **train_kwargs) -> tuple:
    """Leave-one-fold-out over the dev set. Returns ``(per-fold metrics, out-of-fold probabilities)``."""
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    oof = np.full(len(y), np.nan)
    rows = []
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        probe, hist = train_probe(X[tr], y[tr], X[va], y[va], **train_kwargs)
        p = predict_proba(probe, X[va])
        oof[va] = p
        thr = best_threshold(y[va], p, threshold_metric)
        m = compute_metrics(y[va], p, thr)
        m.update({"fold": f, "best_epoch": hist.attrs["best_epoch"], "n_train": int(tr.sum()), "n_val": int(va.sum())})
        rows.append(m)
    return pd.DataFrame(rows).set_index("fold"), oof


def cv_summary(per_fold: pd.DataFrame, keys=("auroc", "auprc", "macro_f1", "f1_pe", "recall_pe", "accuracy")) -> pd.DataFrame:
    return pd.DataFrame({"mean": per_fold[list(keys)].mean(), "std": per_fold[list(keys)].std()}).round(4)


def learning_curve(X, y, folds, sizes, val_fold: int = 0, seed: int = 0, **train_kwargs) -> pd.DataFrame:
    """Train on growing stratified subsets of the training folds; evaluate on one fixed val fold."""
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    rng = np.random.default_rng(seed)
    tr_idx = np.where(folds != val_fold)[0]
    va = folds == val_fold
    rows = []
    for n in sizes:
        n = min(int(n), len(tr_idx))
        pos, neg = tr_idx[y[tr_idx] == 1], tr_idx[y[tr_idx] == 0]
        n_pos = int(round(n * len(pos) / len(tr_idx)))
        pick = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n - n_pos, replace=False)])
        probe, _ = train_probe(X[pick], y[pick], X[va], y[va], seed=seed, **train_kwargs)
        p = predict_proba(probe, X[va])
        m = compute_metrics(y[va], p, best_threshold(y[va], p))
        rows.append({"n_train": n, "auroc": m["auroc"], "auprc": m["auprc"], "macro_f1": m["macro_f1"], "recall_pe": m["recall_pe"]})
    return pd.DataFrame(rows)


def recall_by_subgroup(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Recall among PE-positive studies, split by sub-type and by positive-slice count bins.

    ``df`` needs ``y``, ``pred`` and the sub-label / ``n_pos_slices`` columns.
    """
    pe = df[df["y"] == 1].copy()
    pe["slice_bin"] = pd.cut(pe["n_pos_slices"], bins=[0, 10, 50, 10_000], labels=["1-10 slices", "11-50 slices", "51+ slices"])
    groups = {
        "central": pe["central_pe"] == 1, "not central": pe["central_pe"] == 0,
        "RV/LV >= 1": pe["rv_lv_ratio_gte_1"] == 1, "RV/LV < 1": pe["rv_lv_ratio_lt_1"] == 1,
        "chronic": (pe["chronic_pe"] == 1) | (pe["acute_and_chronic_pe"] == 1),
        "acute only": (pe["chronic_pe"] == 0) & (pe["acute_and_chronic_pe"] == 0),
    }
    for b in ["1-10 slices", "11-50 slices", "51+ slices"]:
        groups[b] = pe["slice_bin"] == b
    rows = [{"subgroup": name, "n": int(mask.sum()), "recall": float(pe.loc[mask, pred_col].mean()) if mask.sum() else np.nan}
            for name, mask in groups.items()]
    return pd.DataFrame(rows).set_index("subgroup")
