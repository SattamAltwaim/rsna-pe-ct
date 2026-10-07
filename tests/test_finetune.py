import numpy as np
import pytest
import torch

from pe_ct import embed, finetune

spectre = pytest.importorskip("spectre")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return embed.load_model("spectre-small", device="cpu", dtype=torch.float32, pretrained=False)


def _records(model, n=120, seed=0):
    """Synthetic descriptors whose label is linearly readable from the first dimension."""
    rng = np.random.default_rng(seed)
    d2 = 2 * model.backbone.embed_dim
    recs = {}
    grids = [(2, 1, 1), (1, 2, 1), (2, 2, 1)]
    for i in range(n):
        y = int(rng.random() < 0.35)
        grid = grids[i % len(grids)]
        n_crops = int(np.prod(grid))
        desc = rng.normal(size=(n_crops, d2)).astype(np.float32)
        desc[:, :8] += 4.0 * y
        recs[f"s{i:03d}"] = {"crop_desc": desc.astype(np.float16), "grid": np.array(grid), "y": y,
                             "cls": rng.normal(size=model.feature_combiner.embed_dim).astype(np.float32)}
    return recs


def test_lora_is_identity_at_init_and_trainable_counts(tiny_model):
    clf, info = finetune.build_classifier(tiny_model, mode="lora", lora_rank=4, device="cpu")
    ref = copy_combiner_out(tiny_model, clf)
    assert info["lora_params"] > 0 and info["combiner_trainable_M"] > 0
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert all(("lora_" in n) or n.startswith("head.") or n.startswith("combiner.norm.") for n in trainable)
    torch.testing.assert_close(ref["lora"], ref["orig"], atol=1e-5, rtol=1e-5)  # B = 0 -> same output as pretrained
    clf2, info2 = finetune.build_classifier(tiny_model, mode="last_blocks", n_last_blocks=1, device="cpu")
    assert info2["lora_params"] == 0 and info2["combiner_trainable_M"] > info["combiner_trainable_M"]
    clf3, info3 = finetune.build_classifier(tiny_model, mode="head_only", device="cpu")
    assert info3["combiner_trainable_M"] == 0
    with pytest.raises(ValueError):
        finetune.build_classifier(tiny_model, mode="everything", device="cpu")
    # the original model is untouched
    assert not any(isinstance(getattr(b.attn, "q"), finetune.LoRALinear) for b in tiny_model.feature_combiner.blocks)


def copy_combiner_out(model, clf):
    torch.manual_seed(1)
    desc = torch.randn(2, 4, 2 * model.backbone.embed_dim)
    with torch.no_grad():
        return {"lora": clf.combiner(desc, (2, 2, 1))[:, 0], "orig": model.feature_combiner(desc, (2, 2, 1))[:, 0]}


def test_grid_batches_keep_one_grid_per_batch(tiny_model):
    ds = finetune.DescriptorSet(_records(tiny_model, 30), [f"s{i:03d}" for i in range(30)])
    batches = finetune.grid_batches(ds, list(range(30)), 4, np.random.default_rng(0))
    assert sorted(i for b in batches for i in b) == list(range(30))
    for b in batches:
        assert len({ds.grid[i] for i in b}) == 1 and len(b) <= 4
    assert set(ds.grid_counts().index) == {"2x1x1", "1x2x1", "2x2x1"}


def test_train_cv_and_ensemble_learn_synthetic_signal(tiny_model):
    recs = _records(tiny_model, 150)
    uids = sorted(recs)
    ds = finetune.DescriptorSet(recs, uids)
    folds = finetune.group_stratified_folds(ds.y, np.array(uids), n_splits=3, seed=0)
    assert set(folds) == {0, 1, 2}
    for f in range(3):
        assert 0 < ds.y[folds == f].mean() < 1
    air = np.zeros(2 * tiny_model.backbone.embed_dim, dtype=np.float32)
    build_kwargs = dict(mode="lora", lora_rank=4, device="cpu")
    train_kwargs = dict(epochs=12, batch_size=16, lr_head=5e-3, lr_adapt=1e-3, warmup_epochs=1, patience=5, crop_dropout=0.1, air=air, seed=0)
    out = finetune.cross_validate(tiny_model, ds, np.arange(len(ds)), folds, build_kwargs, train_kwargs, log=lambda *a: None)
    assert len(out["per_fold"]) == 3 and np.isfinite(out["oof"]).all()
    assert {"train_loss", "val_loss", "val_acc", "val_macro_f1", "fold"} <= set(out["histories"].columns)
    from pe_ct.probe import compute_metrics

    assert compute_metrics(ds.y, out["oof"], 0.5)["auroc"] > 0.75
    summary = finetune.cv_summary(out["per_fold"])
    assert "macro_f1" in summary.index
    ens = finetune.ensemble_predict(tiny_model, ds, np.arange(20), out["states"], build_kwargs)
    assert ens.shape == (20,) and (0 <= ens).all() and (ens <= 1).all()
    axes = finetune.plot_histories(out["histories"])
    assert len(axes) == 3
