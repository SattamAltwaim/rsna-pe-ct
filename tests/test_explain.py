import numpy as np
import pytest
import torch

from pe_ct import embed, explain, probe

spectre = pytest.importorskip("spectre")


def test_importance_volume_profile_and_hits():
    boxes = np.array([[[0, 10], [0, 8], [0, 8]], [[10, 20], [0, 8], [0, 8]]])
    imp = np.array([0.2, 1.5])
    heat = explain.importance_volume(imp, boxes, (24, 8, 8))
    assert heat[5, 0, 0] == pytest.approx(0.2) and heat[15, 0, 0] == pytest.approx(1.5) and np.isnan(heat[22, 0, 0])
    prof = explain.z_profile(imp, boxes, 24)
    assert prof[0] == pytest.approx(0.2) and prof[19] == pytest.approx(1.5) and np.isnan(prof[23])
    assert explain.covered_z_range(boxes) == (0, 20)
    labels = np.zeros(24, dtype=int)
    labels[12:15] = 1
    assert explain.z_hit(imp, boxes, labels) is True
    assert explain.z_hit(imp[::-1], boxes, labels) is False
    assert explain.z_hit(imp, boxes, np.zeros(24)) is False
    rate = explain.random_z_hit_rate(boxes, labels, n_draws=400)
    assert 0.35 < rate < 0.65
    assert explain.band_importance_fraction(imp, boxes, labels) == pytest.approx(1.5 / 1.7)


def test_occlusion_with_tiny_model():
    torch.manual_seed(0)
    model = embed.load_model("spectre-small", device="cpu", dtype=torch.float32, pretrained=False)
    tensor = torch.rand(1, 128, 256, 128) * 1500 - 1000
    rec = embed.extract(model, tensor)
    d = model.feature_combiner.embed_dim
    lp = probe.LinearProbe(np.zeros(d, np.float32), np.ones(d, np.float32))
    fn = probe.predict_logit_fn(lp)
    air = embed.air_descriptor(model)
    out = explain.occlusion_importance(model, rec["crop_desc"], rec["grid"], fn, air, batch_size=3)
    assert out["importance"].shape == (4,) and np.isfinite(out["importance"]).all()
    # replacing a crop by its own descriptor changes nothing
    same = explain.occlusion_importance(model, rec["crop_desc"], rec["grid"], fn, rec["crop_desc"][0].astype(np.float32))
    assert abs(same["importance"][0]) < 1e-3
    assert abs(out["logit_full"] - float(fn(torch.as_tensor(rec["cls"]).unsqueeze(0))[0])) < 2e-2
