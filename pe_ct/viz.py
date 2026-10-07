"""Plotting helpers: windowing presets, slice / orthogonal views, overlays, galleries.

Every function draws onto a matplotlib ``Axes`` (or returns a ``Figure``) and
never saves; notebooks call :func:`save_fig` explicitly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pe_ct.config import HU_REFERENCE

# (level, width) in HU. "full" = the raw dynamic range of the slice.
WINDOWS = {
    "lung": (-600, 1500),
    "soft_tissue": (40, 400),
    "vessel": (100, 700),
    "full": None,
}
PALETTE = plt.get_cmap("tab10").colors
COLOR_PE = PALETTE[3]       # red
COLOR_NEG = PALETTE[0]      # blue
COLOR_IND = PALETTE[7]      # gray
COLOR_GT = PALETTE[2]       # green: ground-truth bands/outlines


def apply_window(hu, window="vessel") -> np.ndarray:
    """HU -> float in [0, 1] using a named preset or an explicit ``(level, width)``."""
    hu = np.asarray(hu, dtype=np.float32)
    if isinstance(window, str):
        window = WINDOWS[window]
    if window is None:
        lo, hi = float(hu.min()), float(hu.max())
    else:
        level, width = window
        lo, hi = level - width / 2, level + width / 2
    return np.clip((hu - lo) / max(hi - lo, 1e-6), 0, 1)


def show_slice(ax, hu2d, window="vessel", title=None, aspect=1.0, cmap="gray"):
    ax.imshow(apply_window(hu2d, window), cmap=cmap, vmin=0, vmax=1, aspect=aspect, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def mark_z(ax, z, color=COLOR_GT, **kwargs):
    """Horizontal line at slice ``z`` on a coronal/sagittal view."""
    ax.axhline(z, color=color, lw=1, **kwargs)


def shade_z_band(ax, band, color=COLOR_GT, alpha=0.25, label=None):
    """Shade the slice range ``band=(lo, hi_exclusive)`` on a coronal/sagittal view."""
    if band is None:
        return
    lo, hi = band
    ax.axhspan(lo - 0.5, hi - 0.5, color=color, alpha=alpha, lw=0, label=label)


def slice_label_bar(ax, labels, color=COLOR_PE):
    """Thin vertical strip: red where a slice is PE-positive, white elsewhere (z downwards)."""
    labels = np.asarray(labels)
    strip = np.zeros((len(labels), 1, 3)) + 1.0
    strip[labels == 1] = color
    ax.imshow(strip, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def orthoviews(vol, meta, window="vessel", z=None, y=None, x=None, labels=None, title=None):
    """Axial / coronal / sagittal views through one voxel, with crosshair lines."""
    from pe_ct.volume import aspect_coronal, aspect_sagittal

    nz, ny, nx = vol.shape
    z = nz // 2 if z is None else z
    y = ny // 2 if y is None else y
    x = nx // 2 if x is None else x
    fig, axes = plt.subplots(1, 4 if labels is not None else 3, figsize=(15, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.08] if labels is not None else [1, 1, 1]})
    show_slice(axes[0], vol[z], window, f"axial  z={z}")
    axes[0].axhline(y, color=PALETTE[1], lw=0.6)
    axes[0].axvline(x, color=PALETTE[1], lw=0.6)
    show_slice(axes[1], vol[:, y, :], window, f"coronal  y={y}", aspect=aspect_coronal(meta))
    axes[1].axhline(z, color=PALETTE[1], lw=0.6)
    axes[1].axvline(x, color=PALETTE[1], lw=0.6)
    show_slice(axes[2], vol[:, :, x], window, f"sagittal  x={x}", aspect=aspect_sagittal(meta))
    axes[2].axhline(z, color=PALETTE[1], lw=0.6)
    axes[2].axvline(y, color=PALETTE[1], lw=0.6)
    if labels is not None:
        slice_label_bar(axes[3], labels)
        axes[3].set_title("PE slices", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def window_comparison(hu2d, windows=("full", "lung", "soft_tissue", "vessel"), title=None):
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4.3))
    for ax, name in zip(axes, windows):
        w = WINDOWS[name]
        label = f"{name}  (L {w[0]} / W {w[1]})" if w else "full range"
        show_slice(ax, hu2d, name, label)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def hu_histogram(ax, volumes, labels=None, bins=200, hu_range=(-1100, 1500)):
    """Overlaid HU histograms (log y) with reference lines for known tissues."""
    for i, vol in enumerate(volumes):
        flat = np.asarray(vol).reshape(-1)
        sample = flat[:: max(1, flat.size // 1_000_000)]
        ax.hist(sample, bins=bins, range=hu_range, histtype="step", lw=1.2,
                color=PALETTE[i % len(PALETTE)], label=labels[i] if labels else None)
    for name, hu in HU_REFERENCE.items():
        ax.axvline(hu, color="k", lw=0.6, ls="--", alpha=0.6)
        ax.text(hu, ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Hounsfield units")
    ax.set_ylabel("voxels (log)")
    if labels:
        ax.legend(fontsize=8)


def crop_center(hu2d, frac=0.5, center=None):
    """Central square crop (default: the middle half) of an axial slice."""
    h, w = hu2d.shape
    cy, cx = (h // 2, w // 2) if center is None else center
    hh, hw = int(h * frac / 2), int(w * frac / 2)
    return hu2d[max(cy - hh, 0): cy + hh, max(cx - hw, 0): cx + hw]


def image_grid(images, titles=None, ncols=4, window="vessel", figsize_per=3.2, aspects=None, suptitle=None):
    """Grid of 2D HU images with captions."""
    n = len(images)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows + 0.3), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        show_slice(ax, images[i], window, titles[i] if titles else None,
                   aspect=aspects[i] if aspects else 1.0)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def overlay_heatmap(ax, hu2d, heat2d, window="vessel", alpha=0.45, aspect=1.0, cmap="inferno", vmax=None):
    """Grayscale slice with a semi-transparent importance map on top."""
    show_slice(ax, hu2d, window, aspect=aspect)
    heat = np.asarray(heat2d, dtype=float)
    vmax = float(np.nanmax(heat)) if vmax is None else vmax
    masked = np.ma.masked_where(~np.isfinite(heat), heat)
    im = ax.imshow(masked, cmap=cmap, alpha=alpha, vmin=0, vmax=max(vmax, 1e-9), aspect=aspect, interpolation="nearest")
    return im


def draw_grid(ax, starts, steps, color=PALETTE[1], lw=0.8):
    """Draw crop boundaries: ``starts=(x0, y0)``, ``steps=(dx, dy, nx, ny)``."""
    x0, y0 = starts
    dx, dy, nx, ny = steps
    for i in range(nx + 1):
        ax.axvline(x0 + i * dx - 0.5, color=color, lw=lw)
    for j in range(ny + 1):
        ax.axhline(y0 + j * dy - 0.5, color=color, lw=lw)


def bar_counts(ax, counts, colors=None, title=None, xlabel=None, horizontal=False, annotate=True):
    """Bar chart of a pandas Series of counts, value labels on the bars."""
    names = [str(k) for k in counts.index]
    values = np.asarray(counts.values, dtype=float)
    colors = colors or [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    if horizontal:
        bars = ax.barh(names, values, color=colors)
        ax.invert_yaxis()
    else:
        bars = ax.bar(names, values, color=colors)
    if annotate:
        for b, v in zip(bars, values):
            txt = f"{v:,.0f}" if v >= 1 or v == 0 else f"{v:.3f}"
            if horizontal:
                ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + txt, va="center", fontsize=8)
            else:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt, ha="center", va="bottom", fontsize=8)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    return bars


def save_fig(fig, figures_dir, notebook: str, name: str, dpi: int = 130) -> Path:
    """``figures/<notebook>/<name>.png``; returns the path."""
    out = Path(figures_dir) / notebook / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def use_notebook_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "font.size": 9,
    })


def slice_scroller(load_volume, uids, labels_lookup=None, window="vessel"):
    """ipywidgets slice browser: dropdown over studies, slider over z, red bar marks PE slices."""
    import ipywidgets as widgets
    from IPython.display import display

    cache: dict = {}

    def get(uid):
        if uid not in cache:
            cache.clear()
            cache[uid] = load_volume(uid)
        return cache[uid]

    study = widgets.Dropdown(options=list(uids), description="study")
    z = widgets.IntSlider(min=0, max=1, value=0, description="slice z", continuous_update=False)
    win = widgets.Dropdown(options=list(WINDOWS), value=window, description="window")
    out = widgets.Output()

    def redraw(*_):
        vol, meta = get(study.value)
        z.max = vol.shape[0] - 1
        labels = labels_lookup(study.value, meta) if labels_lookup else None
        with out:
            out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 6), gridspec_kw={"width_ratios": [1, 0.05]})
            tag = ""
            if labels is not None:
                tag = "  PE on this slice" if labels[z.value] == 1 else ""
            show_slice(axes[0], vol[z.value], win.value, f"{study.value}  z={z.value}/{vol.shape[0]-1}{tag}")
            if labels is not None:
                slice_label_bar(axes[1], labels)
                axes[1].axhline(z.value, color=PALETTE[1], lw=1.5)
            else:
                axes[1].axis("off")
            plt.show()

    study.observe(redraw, "value")
    z.observe(redraw, "value")
    win.observe(redraw, "value")
    display(widgets.VBox([widgets.HBox([study, win]), z, out]))
    redraw()
