"""Where does the model look? Crop-level occlusion on the saved descriptors.

For one study we have the ``(N, 2160)`` crop descriptors that feed SPECTRE's
feature combiner. Occluding crop *i* means replacing its descriptor with the
descriptor of a crop of pure air (so the combiner still sees a full grid and
RoPE positions are unchanged), re-running only the combiner + the linear
probe, and recording the drop in PE logit:

    importance_i = logit(all crops) - logit(crop i replaced by air)

Large positive = the model relied on that region for its PE call. Negative
values mean the region argued *against* PE. The heatmap resolution is one
crop (128 x 128 x 64 voxels), so it says *which region*, not which pixel.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def occlusion_importance(model, crop_desc, grid, logit_fn, baseline_desc, batch_size: int = 32) -> dict:
    """Returns ``{"logit_full", "logit_occluded" (N,), "importance" (N,)}``."""
    from pe_ct.embed import combine

    p = next(model.parameters())
    desc = torch.as_tensor(np.asarray(crop_desc, dtype=np.float32), device=p.device)
    base = torch.as_tensor(np.asarray(baseline_desc, dtype=np.float32), device=p.device)
    n = desc.shape[0]

    full_cls = combine(model, desc, grid)[0, 0].float()
    logit_full = float(logit_fn(full_cls.unsqueeze(0))[0])

    occluded = []
    for start in range(0, n, batch_size):
        idx = list(range(start, min(n, start + batch_size)))
        batch = desc.unsqueeze(0).repeat(len(idx), 1, 1)
        for row, i in enumerate(idx):
            batch[row, i] = base
        cls = combine(model, batch, grid)[:, 0].float()
        occluded.append(logit_fn(cls).float().cpu())
    logit_occ = torch.cat(occluded).numpy()
    return {"logit_full": logit_full, "logit_occluded": logit_occ, "importance": logit_full - logit_occ}


def importance_volume(importance, boxes_lpi, shape_zyx, fill=np.nan) -> np.ndarray:
    """Paint each crop's importance into its box of a ``(z, y, x)`` volume (nearest upsampling)."""
    heat = np.full(tuple(int(s) for s in shape_zyx), fill, dtype=np.float32)
    for value, box in zip(importance, boxes_lpi):
        (z0, z1), (y0, y1), (x0, x1) = box
        heat[z0:z1, y0:y1, x0:x1] = value
    return heat


def z_profile(importance, boxes_lpi, n_slices: int, reduce: str = "max") -> np.ndarray:
    """Importance along the head-to-feet axis: per slice, max (or sum) over crops covering it."""
    prof = np.full(int(n_slices), np.nan, dtype=np.float32)
    for value, box in zip(importance, boxes_lpi):
        z0, z1 = box[0]
        seg = prof[z0:z1]
        if reduce == "max":
            prof[z0:z1] = np.where(np.isnan(seg), value, np.maximum(seg, value))
        else:
            prof[z0:z1] = np.where(np.isnan(seg), value, seg + value)
    return prof


def covered_z_range(boxes_lpi) -> tuple:
    """Slices the crop grid actually covers (edges beyond the grid are never looked at)."""
    return int(boxes_lpi[:, 0, 0].min()), int(boxes_lpi[:, 0, 1].max())


def z_hit(importance, boxes_lpi, slice_labels, top_k: int = 1) -> bool:
    """Does one of the ``top_k`` most important crops overlap the positive slice band?"""
    labels = np.asarray(slice_labels)
    pos = np.where(labels == 1)[0]
    if pos.size == 0:
        return False
    lo, hi = int(pos.min()), int(pos.max()) + 1
    order = np.argsort(-np.asarray(importance))[:top_k]
    for i in order:
        z0, z1 = boxes_lpi[i, 0]
        if z0 < hi and z1 > lo:
            return True
    return False


def random_z_hit_rate(boxes_lpi, slice_labels, top_k: int = 1, n_draws: int = 2000, seed: int = 0) -> float:
    """Chance level for :func:`z_hit`: pick ``top_k`` random crops instead."""
    rng = np.random.default_rng(seed)
    n = len(boxes_lpi)
    hits = 0
    for _ in range(n_draws):
        fake = rng.random(n)
        hits += z_hit(fake, boxes_lpi, slice_labels, top_k)
    return hits / n_draws


def band_importance_fraction(importance, boxes_lpi, slice_labels) -> float:
    """Share of positive importance that falls on crops overlapping the ground-truth band."""
    imp = np.clip(np.asarray(importance, dtype=float), 0, None)
    if imp.sum() <= 0:
        return np.nan
    labels = np.asarray(slice_labels)
    pos = np.where(labels == 1)[0]
    if pos.size == 0:
        return np.nan
    lo, hi = int(pos.min()), int(pos.max()) + 1
    overlap = (boxes_lpi[:, 0, 0] < hi) & (boxes_lpi[:, 0, 1] > lo)
    return float(imp[overlap].sum() / imp.sum())
