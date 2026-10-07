"""Frozen SPECTRE backbone: preprocessing, extraction, and the crop-grid geometry.

SPECTRE (``spectre-fm``) tiles a RAS-oriented HU volume into non-overlapping
128 x 128 x 64 crops, runs a ViT-L *backbone* on each crop, pools each crop to a
2160-d descriptor (CLS ++ mean patch token), and runs a small *feature
combiner* transformer over the crop descriptors to produce one 1080-d scan
embedding (its CLS) plus one 1080-d token per crop.

We store, per study:

* ``cls``          (1080,)  float32   scan embedding -> linear probe (NB04)
* ``crop_desc``    (N, 2160) float16  combiner *inputs* -> occlusion heatmaps (NB05)
* ``crop_tokens``  (N, 1080) float16  combiner outputs (for a later attention head)
* ``grid``         (3,)  (n_h, n_w, n_d) in RAS order (R, A, S)
* ``boxes_lpi``    (N, 3, 2) each crop's [lo, hi) along (z, y, x) of the stored LPI volume

Crop ``n`` sits at grid position ``n = (h * n_w + w) * n_d + d`` (depth fastest),
exactly as ``spectre.windowing.grid_patch`` orders them.
"""

from __future__ import annotations

import time

import numpy as np
import SimpleITK as sitk
import torch

CROP_SIZE = (128, 128, 64)  # (H, W, D) = (R, A, S) voxels


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def load_model(name: str = "spectre-large", device=None, dtype=None, pretrained: bool = True):
    """Load SPECTRE in eval mode. bf16 on GPU by default, fp32 on CPU."""
    from spectre import SpectreImageFeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if str(device).startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    return SpectreImageFeatureExtractor.from_pretrained(name, pretrained=pretrained, device=device, dtype=dtype)


def model_info(model) -> dict:
    p = next(model.parameters())
    return {
        "n_params_backbone_M": round(sum(x.numel() for x in model.backbone.parameters()) / 1e6, 1),
        "n_params_combiner_M": round(sum(x.numel() for x in model.feature_combiner.parameters()) / 1e6, 1),
        "crop_size": tuple(model.crop_size),
        "embed_dim": int(model.feature_combiner.embed_dim),
        "device": str(p.device),
        "dtype": str(p.dtype),
    }


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


def resample_image(image: sitk.Image, spacing, interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resample to a fixed voxel spacing (mm); geometry preserved."""
    old_spacing = np.array(image.GetSpacing())
    old_size = np.array(image.GetSize())
    new_spacing = np.array(spacing, dtype=float)
    new_size = np.maximum(1, np.round(old_size * old_spacing / new_spacing)).astype(int).tolist()
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator, image.GetOrigin(),
                         new_spacing.tolist(), image.GetDirection(), -1000.0, image.GetPixelID())


def prepare_input(image: sitk.Image, resample_spacing=None) -> tuple:
    """SimpleITK image (any orientation) -> ``(torch (1, R, A, S) float32 HU, info)``."""
    from pe_ct.volume import to_ras_tensor_array

    if resample_spacing is not None:
        image = resample_image(image, resample_spacing)
    arr = to_ras_tensor_array(image)
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    info = {
        "ras_shape": tuple(int(s) for s in arr.shape),
        "spacing_xyz_mm": [float(s) for s in image.GetSpacing()],
        "resampled": resample_spacing is not None,
    }
    return tensor, info


# ---------------------------------------------------------------------------
# crop geometry (mirrors spectre.windowing.window_scan exactly)
# ---------------------------------------------------------------------------


def crop_layout(ras_shape, crop_size=CROP_SIZE) -> dict:
    """Where the crops land in the RAS volume: grid, per-axis start offset, padded axes.

    Follows ``window_scan``: axes shorter than one crop are padded symmetrically
    (start is then negative), longer axes are centre-cropped to the largest
    whole multiple with ``start = size // 2 - roi // 2``.
    """
    grid, start, padded = [], [], []
    for size, crop in zip(ras_shape, crop_size):
        size, crop = int(size), int(crop)
        if size < crop:
            grid.append(1)
            start.append(-((crop - size) // 2))
            padded.append(True)
        else:
            roi = (size // crop) * crop
            grid.append(roi // crop)
            start.append(max(size // 2 - roi // 2, 0))
            padded.append(False)
    return {"grid": tuple(grid), "start": tuple(start), "padded": tuple(padded),
            "crop_size": tuple(int(c) for c in crop_size), "ras_shape": tuple(int(s) for s in ras_shape)}


def crop_boxes_ras(layout: dict) -> np.ndarray:
    """``(N, 3, 2)`` [lo, hi) per crop along (R, A, S), in ``grid_patch`` order."""
    n_h, n_w, n_d = layout["grid"]
    boxes = []
    for h in range(n_h):
        for w in range(n_w):
            for d in range(n_d):
                box = []
                for axis, i in enumerate((h, w, d)):
                    lo = layout["start"][axis] + i * layout["crop_size"][axis]
                    box.append((lo, lo + layout["crop_size"][axis]))
                boxes.append(box)
    return np.asarray(boxes, dtype=np.int64)


def crop_boxes_lpi(layout: dict) -> np.ndarray:
    """Crop boxes in the stored volume's ``(z, y, x)`` index space, clipped to the volume.

    LPI index = (size - 1) - RAS index on every axis, so a RAS range [a, b)
    becomes [size - b, size - a).
    """
    ras = crop_boxes_ras(layout)
    sizes = np.array(layout["ras_shape"])  # (R, A, S) sizes == (x, y, z) sizes
    lpi = np.empty_like(ras)
    for axis in range(3):
        lpi[:, axis, 0] = sizes[axis] - ras[:, axis, 1]
        lpi[:, axis, 1] = sizes[axis] - ras[:, axis, 0]
    lpi = lpi[:, ::-1, :]  # (x, y, z) -> (z, y, x)
    zyx_sizes = sizes[::-1]
    for axis in range(3):
        lpi[:, axis, :] = np.clip(lpi[:, axis, :], 0, zyx_sizes[axis])
    return np.ascontiguousarray(lpi)


def grid_index(n: int, grid) -> tuple:
    """Crop number -> (h, w, d) grid position."""
    n_h, n_w, n_d = grid
    d = n % n_d
    w = (n // n_d) % n_w
    h = n // (n_d * n_w)
    return h, w, d


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def pool_crop_tokens(backbone_tokens: torch.Tensor) -> torch.Tensor:
    """``(N, 513, F)`` backbone tokens -> ``(N, 2F)`` = CLS ++ mean(patch tokens)."""
    return torch.cat([backbone_tokens[:, 0, :], backbone_tokens[:, 1:, :].mean(dim=1)], dim=-1)


@torch.inference_mode()
def backbone_descriptors(model, crops: torch.Tensor, max_crops_per_forward: int = 16) -> torch.Tensor:
    """Run the backbone chunk-wise and pool: ``(N, 1, 128, 128, 64)`` -> ``(N, 2160)``."""
    p = next(model.parameters())
    out = []
    for chunk in torch.split(crops, max_crops_per_forward, dim=0):
        feats = model.backbone(chunk.to(device=p.device, dtype=p.dtype))
        out.append(pool_crop_tokens(feats))
    return torch.cat(out, dim=0)


@torch.inference_mode()
def combine(model, desc: torch.Tensor, grid) -> torch.Tensor:
    """Feature combiner on ``(N, 2160)`` or ``(B, N, 2160)`` descriptors -> ``(B, 1+N, 1080)``."""
    p = next(model.parameters())
    if desc.ndim == 2:
        desc = desc.unsqueeze(0)
    return model.feature_combiner(desc.to(device=p.device, dtype=p.dtype), tuple(int(g) for g in grid))


@torch.inference_mode()
def extract(model, tensor: torch.Tensor, max_crops_per_forward: int = 16) -> dict:
    """Whole pipeline on one ``(1, R, A, S)`` HU tensor; returns numpy arrays + grid."""
    from spectre import window_scan

    t0 = time.time()
    crops, grid = window_scan(tensor)
    desc = backbone_descriptors(model, crops, max_crops_per_forward)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_backbone = time.time() - t0
    out = combine(model, desc, grid)[0]
    return {
        "cls": out[0].float().cpu().numpy(),
        "crop_tokens": out[1:].float().cpu().numpy().astype(np.float16),
        "crop_desc": desc.float().cpu().numpy().astype(np.float16),
        "grid": np.asarray(grid, dtype=np.int64),
        "n_crops": int(crops.shape[0]),
        "t_gpu_s": round(time.time() - t0, 3),
        "t_backbone_s": round(t_backbone, 3),
    }


def embed_image(model, image: sitk.Image, resample_spacing=None, max_crops_per_forward: int = 16) -> dict:
    """SimpleITK image -> record with embeddings and crop boxes in stored (z, y, x) space."""
    tensor, info = prepare_input(image, resample_spacing)
    rec = extract(model, tensor, max_crops_per_forward)
    layout = crop_layout(info["ras_shape"], tuple(model.crop_size))
    rec["boxes_lpi"] = crop_boxes_lpi(layout)
    rec["ras_shape"] = np.asarray(info["ras_shape"], dtype=np.int64)
    rec["crop_start_ras"] = np.asarray(layout["start"], dtype=np.int64)
    rec["padded_axes"] = np.asarray(layout["padded"])
    rec["resampled"] = bool(info["resampled"])
    return rec


@torch.inference_mode()
def air_descriptor(model) -> np.ndarray:
    """Descriptor of a crop of pure air (-1000 HU): the occlusion baseline for NB05."""
    from spectre import window_scan

    crop_size = tuple(model.crop_size)
    crops, _ = window_scan(torch.full((1, *crop_size), -1000.0))
    return backbone_descriptors(model, crops)[0].float().cpu().numpy()
