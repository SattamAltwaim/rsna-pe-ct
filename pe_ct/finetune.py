"""Fine-tune SPECTRE's feature combiner (+ an MLP head) on cached crop descriptors.

Why this part of the model: the ViT-L *backbone* (339M params) turns every
128 x 128 x 64 crop into a 2160-d descriptor; the *feature combiner* (58M, four
transformer blocks over the ~50 crop descriptors of a scan) turns those into the
scan embedding. Running the backbone once per study and caching the descriptors
makes a training epoch over thousands of scans take seconds, so cross-validation
and early stopping are cheap. Two ways to adapt the combiner:

* ``lora``        low-rank adapters on every attention projection (q, kv, proj) of
                  all blocks, about 0.2M trainable params. The default for a small subset.
* ``last_blocks`` unfreeze the last ``n`` blocks (about 14M params each).
* ``head_only``   nothing in the combiner trains (a non-linear probe).

The MLP head always trains. Training: BCE with a positive-class weight, AdamW with
separate learning rates for head and adapters, cosine schedule with linear warm-up,
mixed precision on GPU, early stopping on validation macro F1.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from pe_ct.probe import compute_metrics

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(dropout(x)))`` with a frozen ``base`` and ``B`` initialised to zero."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


def inject_lora(combiner: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                targets=("q", "kv", "proj")) -> int:
    """Wrap the attention projections of every block; returns the number of LoRA parameters."""
    n = 0
    for blk in combiner.blocks:
        for name in targets:
            base = getattr(blk.attn, name)
            if isinstance(base, LoRALinear):
                continue
            lora = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout).to(base.weight.device)
            setattr(blk.attn, name, lora)
            n += sum(p.numel() for p in [lora.lora_a.weight, lora.lora_b.weight])
    return n


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


class CombinerClassifier(nn.Module):
    """``crop descriptors (B, N, 2F) -> combiner -> CLS -> MLP -> logit (B,)``."""

    def __init__(self, combiner: nn.Module, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.combiner = combiner
        d = combiner.embed_dim
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, desc: torch.Tensor, grid) -> torch.Tensor:
        tokens = self.combiner(desc, tuple(int(g) for g in grid))
        return self.head(tokens[:, 0]).squeeze(-1)

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if self._is_trainable(k)}

    def _is_trainable(self, key: str) -> bool:
        return key in self._trainable_keys

    def mark_trainable(self) -> None:
        self._trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}


def set_trainable(combiner: nn.Module, mode: str, n_last_blocks: int = 2, lora_rank: int = 8,
                  lora_alpha: float = 16.0, lora_dropout: float = 0.05) -> dict:
    """Freeze the combiner, then open up what ``mode`` says. Returns parameter counts."""
    for p in combiner.parameters():
        p.requires_grad_(False)
    n_lora = 0
    if mode == "lora":
        n_lora = inject_lora(combiner, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        for n, p in combiner.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
    elif mode == "last_blocks":
        for blk in list(combiner.blocks)[-n_last_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
    elif mode != "head_only":
        raise ValueError(f"unknown mode {mode!r}; use 'lora', 'last_blocks' or 'head_only'")
    for p in combiner.norm.parameters():  # the final LayerNorm is cheap to adapt in every mode
        p.requires_grad_(mode != "head_only")
    total = sum(p.numel() for p in combiner.parameters())
    trainable = sum(p.numel() for p in combiner.parameters() if p.requires_grad)
    return {"mode": mode, "combiner_params_M": round(total / 1e6, 2), "combiner_trainable_M": round(trainable / 1e6, 3), "lora_params": n_lora}


def build_classifier(model, mode: str = "lora", hidden: int = 256, dropout: float = 0.2, device=None, **mode_kwargs) -> tuple:
    """A fresh classifier around a *copy* of the model's pretrained combiner (fp32). Returns ``(clf, info)``."""
    combiner = copy.deepcopy(model.feature_combiner).float()
    info = set_trainable(combiner, mode, **mode_kwargs)
    clf = CombinerClassifier(combiner, hidden=hidden, dropout=dropout)
    clf.mark_trainable()
    info["head_params_M"] = round(sum(p.numel() for p in clf.head.parameters()) / 1e6, 3)
    info["total_trainable_M"] = round(sum(p.numel() for p in clf.parameters() if p.requires_grad) / 1e6, 3)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return clf.to(device), info


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class DescriptorSet:
    """Cached per-study crop descriptors (float16 in RAM) with grid, label and uid."""

    def __init__(self, records: dict, uids: list):
        self.uids = list(uids)
        self.desc = [np.asarray(records[u]["crop_desc"], dtype=np.float16) for u in self.uids]
        self.grid = [tuple(int(g) for g in records[u]["grid"]) for u in self.uids]
        self.y = np.array([int(records[u]["y"]) for u in self.uids], dtype=np.int64)
        self.cls = np.stack([np.asarray(records[u]["cls"], dtype=np.float32) for u in self.uids])

    def __len__(self) -> int:
        return len(self.uids)

    def grid_counts(self) -> pd.Series:
        return pd.Series([f"{g[0]}x{g[1]}x{g[2]}" for g in self.grid]).value_counts()


def grid_batches(dataset: DescriptorSet, indices, batch_size: int, rng: np.random.Generator | None = None) -> list:
    """Batches of indices that share one crop grid (the combiner's RoPE needs that); shuffled if ``rng``."""
    buckets: dict = {}
    for i in indices:
        buckets.setdefault(dataset.grid[i], []).append(int(i))
    batches = []
    for grid, idx in buckets.items():
        idx = list(idx)
        if rng is not None:
            rng.shuffle(idx)
        batches += [idx[k: k + batch_size] for k in range(0, len(idx), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(dataset: DescriptorSet, idx: list, device, air: np.ndarray | None = None,
            crop_dropout: float = 0.0, rng: np.random.Generator | None = None) -> tuple:
    desc = np.stack([dataset.desc[i] for i in idx]).astype(np.float32)
    if crop_dropout > 0 and air is not None and rng is not None:
        mask = rng.random(desc.shape[:2]) < crop_dropout
        desc[mask] = air.astype(np.float32)
    y = torch.as_tensor(dataset.y[idx], dtype=torch.float32, device=device)
    return torch.as_tensor(desc, device=device), dataset.grid[idx[0]], y


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def predict(clf: CombinerClassifier, dataset: DescriptorSet, indices, batch_size: int = 64) -> np.ndarray:
    """Probabilities in the order of ``indices``."""
    clf.eval()
    device = next(clf.parameters()).device
    out = np.zeros(len(indices), dtype=np.float32)
    pos = {int(i): k for k, i in enumerate(indices)}
    use_amp = device.type == "cuda"
    for idx in grid_batches(dataset, indices, batch_size):
        desc, grid, _ = collate(dataset, idx, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = clf(desc, grid)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        for i, p in zip(idx, probs):
            out[pos[i]] = p
    return out


def train_model(clf: CombinerClassifier, dataset: DescriptorSet, train_idx, val_idx, epochs: int = 40,
                batch_size: int = 32, lr_head: float = 1e-3, lr_adapt: float = 2e-4, weight_decay: float = 1e-2,
                warmup_epochs: float = 2.0, patience: int = 6, crop_dropout: float = 0.1, air: np.ndarray | None = None,
                grad_clip: float = 1.0, seed: int = 0, monitor: str = "val_macro_f1", progress=None, log=None) -> dict:
    """Train until ``monitor`` stops improving; restores the best weights.

    Returns ``{"history": DataFrame, "best_epoch", "val_prob" (best epoch, in val_idx order)}``.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(clf.parameters()).device
    use_amp = device.type == "cuda"
    train_idx, val_idx = [int(i) for i in train_idx], [int(i) for i in val_idx]

    y_tr = dataset.y[train_idx]
    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    head_params = [p for p in clf.head.parameters() if p.requires_grad]
    adapt_params = [p for n, p in clf.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": head_params, "lr": lr_head}]
    if adapt_params:
        groups.append({"params": adapt_params, "lr": lr_adapt})
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(grid_batches(dataset, train_idx, batch_size)))
    sched = cosine_with_warmup(opt, int(warmup_epochs * steps_per_epoch), epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rows, best, best_state, best_prob, since = [], -np.inf, None, None, 0
    epoch_iter = range(epochs) if progress is None else progress(range(epochs), desc="epochs")
    for epoch in epoch_iter:
        clf.train()
        t0, losses = time.time(), []
        for idx in grid_batches(dataset, train_idx, batch_size, rng):
            desc, grid, y = collate(dataset, idx, device, air, crop_dropout, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = clf(desc, grid)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val_prob = predict(clf, dataset, val_idx, batch_size * 2)
        y_val = dataset.y[val_idx]
        with torch.no_grad():
            val_logit = torch.logit(torch.as_tensor(val_prob, dtype=torch.float32).clamp(1e-6, 1 - 1e-6))
            val_loss = loss_fn(val_logit.to(device), torch.as_tensor(y_val, dtype=torch.float32, device=device)).item()
        m = compute_metrics(y_val, val_prob, 0.5)
        row = {"epoch": epoch, "lr_head": opt.param_groups[0]["lr"], "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_auroc": m["auroc"], "val_auprc": m["auprc"], "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        if log is not None:
            log(f"epoch {epoch:3d}  train {row['train_loss']:.4f}  val {val_loss:.4f}  acc {row['val_acc']:.3f}  macroF1 {row['val_macro_f1']:.3f}  AUROC {row['val_auroc']:.3f}")
        if row[monitor] > best + 1e-6:
            best, best_state, best_prob, since = row[monitor], clf.trainable_state_dict(), val_prob, 0
        else:
            since += 1
            if since >= patience:
                break
    clf.load_state_dict(best_state, strict=False)
    clf.eval()
    history = pd.DataFrame(rows)
    return {"history": history, "best_epoch": int(history[monitor].idxmax()), "val_prob": best_prob, "state": best_state}


# ---------------------------------------------------------------------------
# cross-validation and ensembling
# ---------------------------------------------------------------------------


def group_stratified_folds(y, groups, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per sample: stratified on ``y``, with every group (study) in exactly one fold."""
    folds = np.full(len(y), -1, dtype=np.int64)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, val) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        folds[val] = f
    return folds


def cross_validate(model, dataset: DescriptorSet, dev_idx, folds, build_kwargs: dict, train_kwargs: dict,
                   progress=None, log=print) -> dict:
    """Train one classifier per fold. Returns per-fold table, histories, out-of-fold probabilities, fold states."""
    dev_idx = np.asarray([int(i) for i in dev_idx])
    oof = np.full(len(dataset), np.nan, dtype=np.float32)
    rows, histories, states = [], [], []
    for f in sorted(set(int(v) for v in folds)):
        tr, va = dev_idx[folds != f], dev_idx[folds == f]
        clf, _ = build_classifier(model, **build_kwargs)
        t0 = time.time()
        res = train_model(clf, dataset, tr, va, progress=progress, **train_kwargs)
        oof[va] = res["val_prob"]
        m = compute_metrics(dataset.y[va], res["val_prob"], 0.5)
        rows.append({"fold": f, "best_epoch": res["best_epoch"], "epochs_run": len(res["history"]), "n_train": len(tr), "n_val": len(va),
                     "val_loss": float(res["history"].loc[res["best_epoch"], "val_loss"]), "acc": m["accuracy"], "macro_f1": m["macro_f1"],
                     "f1_pe": m["f1_pe"], "recall_pe": m["recall_pe"], "auroc": m["auroc"], "auprc": m["auprc"], "minutes": round((time.time() - t0) / 60, 1)})
        log(f"fold {f}: best epoch {res['best_epoch']}  acc {m['accuracy']:.3f}  macro F1 {m['macro_f1']:.3f}  AUROC {m['auroc']:.3f}")
        histories.append(res["history"].assign(fold=f))
        states.append(res["state"])
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"per_fold": pd.DataFrame(rows).set_index("fold"), "histories": pd.concat(histories, ignore_index=True), "oof": oof, "states": states}


def ensemble_predict(model, dataset: DescriptorSet, indices, states: list, build_kwargs: dict, batch_size: int = 64) -> np.ndarray:
    """Average the probabilities of the fold models on ``indices``."""
    probs = []
    for state in states:
        clf, _ = build_classifier(model, **build_kwargs)
        clf.load_state_dict(state, strict=False)
        probs.append(predict(clf, dataset, indices, batch_size))
        del clf
    return np.mean(probs, axis=0)


def cv_summary(per_fold: pd.DataFrame, keys=("acc", "macro_f1", "f1_pe", "recall_pe", "auroc", "auprc", "val_loss")) -> pd.DataFrame:
    return pd.DataFrame({"mean": per_fold[list(keys)].mean(), "std": per_fold[list(keys)].std()}).round(4)


def plot_histories(histories: pd.DataFrame, axes=None):
    """Loss (train vs val), val accuracy and val macro F1 per epoch, one line per fold."""
    import matplotlib.pyplot as plt

    from pe_ct.viz import PALETTE

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    for f, h in histories.groupby("fold"):
        c = PALETTE[int(f) % len(PALETTE)]
        axes[0].plot(h.epoch, h.train_loss, color=c, ls="--", lw=1, label=f"fold {f} train")
        axes[0].plot(h.epoch, h.val_loss, color=c, lw=1.5, label=f"fold {f} val")
        axes[1].plot(h.epoch, h.val_acc, color=c, lw=1.5, label=f"fold {f}")
        axes[2].plot(h.epoch, h.val_macro_f1, color=c, lw=1.5, label=f"fold {f}")
    axes[0].set_title("loss per epoch (dashed = train, solid = val)")
    axes[1].set_title("validation accuracy")
    axes[2].set_title("validation macro F1")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=7)
    return axes
