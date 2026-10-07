import numpy as np
import pytest
import torch

from pe_ct import embed, volume

spectre = pytest.importorskip("spectre")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return embed.load_model("spectre-small", device="cpu", dtype=torch.float32, pretrained=False)


@pytest.mark.parametrize("shape", [(512, 512, 234), (300, 260, 64), (100, 512, 70), (128, 128, 64), (129, 255, 130)])
def test_crop_layout_matches_window_scan(shape):
    from spectre import window_scan

    layout = embed.crop_layout(shape)
    crops, grid = window_scan(torch.full((1, *shape), -500.0))
    assert layout["grid"] == tuple(grid)
    assert crops.shape[0] == int(np.prod(grid))
    boxes = embed.crop_boxes_ras(layout)
    assert boxes.shape == (crops.shape[0], 3, 2)
    for axis in range(3):
        inside = boxes[:, axis, :].clip(0, shape[axis])
        assert (inside[:, 1] > inside[:, 0]).all()


def test_crop_boxes_ras_match_window_scan_voxels():
    """Voxel values encode position; each RAS box must contain exactly its crop's voxels."""
    from spectre import window_scan

    shape = (256, 129, 130)
    coords = np.indices(shape).astype(np.float32)
    code = coords[0] * 1e6 + coords[1] * 1e3 + coords[2]  # unique per voxel
    tensor = torch.from_numpy(code).unsqueeze(0)
    crops, grid = window_scan(tensor, scale_intensity=False)
    layout = embed.crop_layout(shape)
    boxes = embed.crop_boxes_ras(layout)
    for n in range(crops.shape[0]):
        (r0, r1), (a0, a1), (s0, s1) = boxes[n]
        np.testing.assert_array_equal(crops[n, 0].numpy(), code[r0:r1, a0:a1, s0:s1])
        assert embed.grid_index(n, grid) == tuple(int(v) for v in np.unravel_index(n, grid))


def test_crop_boxes_lpi_cover_the_same_voxels(tiny_volume):
    vol, meta = tiny_volume
    big = np.zeros((70, 140, 300), dtype=np.int16)  # (z, y, x) -> RAS (300, 140, 70)
    rng = np.random.default_rng(1)
    big[:] = rng.integers(-1000, 1000, big.shape, dtype=np.int16)
    m = dict(meta, shape_zyx=list(big.shape))
    img = volume.array_to_image(big, m)
    ras = volume.to_ras_tensor_array(img)
    layout = embed.crop_layout(ras.shape)
    ras_boxes = embed.crop_boxes_ras(layout)
    lpi_boxes = embed.crop_boxes_lpi(layout)
    assert layout["grid"] == (2, 1, 1) and layout["padded"] == (False, False, False)
    for n in range(len(ras_boxes)):
        (r0, r1), (a0, a1), (s0, s1) = ras_boxes[n]
        (z0, z1), (y0, y1), (x0, x1) = lpi_boxes[n]
        from_ras = ras[r0:r1, a0:a1, s0:s1]
        from_lpi = big[z0:z1, y0:y1, x0:x1]
        np.testing.assert_array_equal(from_lpi, from_ras.transpose(2, 1, 0)[::-1, ::-1, ::-1])
    assert lpi_boxes[:, 0, 0].min() == 70 - 64 - layout["start"][2] and lpi_boxes.min() >= 0


def test_crop_boxes_lpi_clipped_when_padded():
    layout = embed.crop_layout((100, 512, 40))  # R and S shorter than a crop -> padded
    assert layout["padded"] == (True, False, True) and layout["grid"] == (1, 4, 1)
    boxes = embed.crop_boxes_lpi(layout)
    assert boxes[:, 0, :].min() == 0 and boxes[:, 0, :].max() == 40
    assert boxes[:, 2, :].min() == 0 and boxes[:, 2, :].max() == 100


def test_extract_end_to_end_with_tiny_model(tiny_model):
    torch.manual_seed(0)
    tensor = torch.rand(1, 128, 256, 64) * 1500 - 1000
    rec = embed.extract(tiny_model, tensor, max_crops_per_forward=1)
    assert tuple(rec["grid"]) == (1, 2, 1) and rec["n_crops"] == 2
    d = tiny_model.feature_combiner.embed_dim
    assert rec["cls"].shape == (d,) and rec["crop_tokens"].shape == (2, d) and rec["crop_desc"].shape == (2, 2 * tiny_model.backbone.embed_dim)
    # our chunked backbone+combiner path reproduces the library's own forward
    with torch.no_grad():
        ref = tiny_model(tensor)
    np.testing.assert_allclose(rec["cls"], ref[0].numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(rec["crop_tokens"].astype(np.float32), ref[1:].numpy(), atol=2e-2, rtol=2e-2)


def test_pool_matches_library(tiny_model):
    feats = torch.randn(3, 5, tiny_model.backbone.embed_dim)
    ours = embed.pool_crop_tokens(feats)
    if hasattr(tiny_model, "_pool_crop_tokens"):
        torch.testing.assert_close(ours, tiny_model._pool_crop_tokens(feats))
    assert ours.shape == (3, 2 * tiny_model.backbone.embed_dim)


def test_embed_image_and_air_descriptor(tiny_model, tiny_volume):
    vol, meta = tiny_volume
    big = np.full((64, 128, 128), -1000, dtype=np.int16)
    big[20:40, 40:90, 40:90] = 40
    img = volume.array_to_image(big, dict(meta, shape_zyx=list(big.shape)))
    rec = embed.embed_image(tiny_model, img)
    assert tuple(rec["grid"]) == (1, 1, 1) and rec["boxes_lpi"].tolist() == [[[0, 64], [0, 128], [0, 128]]]
    air = embed.air_descriptor(tiny_model)
    assert air.shape == (2 * tiny_model.backbone.embed_dim,) and np.isfinite(air).all()
    info = embed.model_info(tiny_model)
    assert info["crop_size"] == (128, 128, 64) and info["embed_dim"] == tiny_model.feature_combiner.embed_dim


def test_resample_image_changes_size(tiny_volume):
    vol, meta = tiny_volume
    img = volume.array_to_image(vol, meta)  # spacing (0.8, 0.8, 2.0), size (32, 32, 20)
    out = embed.resample_image(img, (1.6, 1.6, 1.0))
    assert out.GetSize() == (16, 16, 40)
    assert out.GetSpacing() == (1.6, 1.6, 1.0)
