"""The self-contained fine-tuning notebook's own cells (metrics, LoRA, training, CV) run on a tiny model."""

import numpy as np
import pytest
import torch

spectre = pytest.importorskip("spectre")
nbformat = pytest.importorskip("nbformat")

from tests.conftest import REPO  # noqa: E402


def _cells():
    nb = nbformat.read(REPO / "notebooks" / "02_finetune.ipynb", as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _find(cells, marker):
    hits = [c for c in cells if marker in c]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def test_notebook_training_cells_run_end_to_end(tmp_path):
    from pe_ct import embed

    cells = _cells()
    ns = {}
    exec(_find(cells, "import io, json, math"), ns)
    config = _find(cells, "INPUT = Path(").replace('Path("/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection")', f'Path("{tmp_path}/input")').replace('Path("/kaggle/working/rsna-pe")', f'Path("{tmp_path}/out")')
    exec(config, ns)
    assert ns["DEVICE"] == "cpu" and ns["DTYPE"] == torch.float32
    ns["EPOCHS"], ns["PATIENCE"], ns["BATCH_SIZE"], ns["N_FOLDS"], ns["LORA_RANK"] = 4, 3, 16, 2, 4

    torch.manual_seed(0)
    model = embed.load_model("spectre-small", device="cpu", dtype=torch.float32, pretrained=False)
    import copy

    ns["combiner_pretrained"] = copy.deepcopy(model.feature_combiner).float()
    d2 = 2 * model.backbone.embed_dim
    rng = np.random.default_rng(0)
    n = 90
    ns["uids"] = [f"s{i:03d}" for i in range(n)]
    ns["Y"] = (rng.random(n) < 0.35).astype(np.int64)
    ns["GRID"] = [[(2, 1, 1), (1, 2, 1)][i % 2] for i in range(n)]
    ns["DESC"] = [(rng.normal(size=(int(np.prod(g)), d2)) + np.r_[np.full(8, 4.0 * y), np.zeros(d2 - 8)]).astype(np.float16) for g, y in zip(ns["GRID"], ns["Y"])]
    ns["AIR"] = np.zeros(d2, np.float32)
    ns["dev_idx"], ns["test_idx"] = np.arange(70), np.arange(70, 90)

    exec(_find(cells, "def metrics(y, prob"), ns) if "frozen = metrics" not in _find(cells, "def metrics(y, prob") else None
    metrics_cell = _find(cells, "def metrics(y, prob")
    ns["va"], ns["p_lin"] = ns["dev_idx"][:20], rng.random(20)
    exec(metrics_cell, ns)
    assert set(ns["frozen"]) >= {"accuracy", "macro_f1", "auroc"}

    exec(_find(cells, "class LoRALinear"), ns)
    exec(_find(cells, "def grid_batches"), ns)
    exec(_find(cells, "def train_model"), ns)
    clf = ns["build_classifier"]()
    assert sum(p.numel() for p in clf.parameters() if p.requires_grad) > 0
    res = ns["train_model"](clf, ns["dev_idx"][:50], ns["dev_idx"][50:70])
    assert {"train_loss", "val_loss", "val_acc", "val_macro_f1"} <= set(res["history"].columns)
    assert res["val_prob"].shape == (20,) and np.isfinite(res["val_prob"]).all()

    exec(_find(cells, "folds = np.full(len(dev_idx)"), ns)
    assert set(ns["folds"]) == {0, 1}
    exec(_find(cells, "oof = np.full(len(uids)"), ns)
    assert len(ns["per_fold"]) == 2 and np.isfinite(ns["oof"][ns["dev_idx"]]).all()
    exec(_find(cells, "y_dev, p_dev = Y[dev_idx]"), ns)
    assert 0.05 <= ns["THR"] <= 0.95
    exec(_find(cells, "p_test = np.zeros(len(test_idx)"), ns)
    assert np.isfinite(ns["p_test"]).all() and ns["p_test"].shape == (20,)
    assert ns["bootstrap"](ns["y_test"], ns["p_test"], ns["THR"], n=20).shape[1] == 3
