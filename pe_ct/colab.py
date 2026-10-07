"""Colab session helpers. Everything degrades gracefully on a workstation."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def in_kaggle() -> bool:
    return os.path.exists("/kaggle/input")


def mount_drive(mount_point: str = "/content/drive") -> Path | None:
    """Mount Google Drive on Colab; returns the MyDrive path (None locally and on Kaggle)."""
    if not in_colab() or in_kaggle():
        return None
    mount = Path(mount_point)
    if not (mount / "MyDrive").exists():
        from google.colab import drive  # type: ignore

        drive.mount(str(mount))
    return mount / "MyDrive"


def link_kaggle_inputs(cfg, pattern: str = "/kaggle/input/*/rsna-pe") -> dict:
    """Make earlier notebooks' outputs visible under ``cfg.root`` on Kaggle.

    On Kaggle a notebook's ``/kaggle/working`` is persisted as its output and can be
    attached to another notebook as an input, where it appears read-only under
    ``/kaggle/input/<notebook-slug>/``. Every file found under ``<input>/rsna-pe/`` is
    symlinked to the same relative path under ``cfg.root`` (existing files are kept),
    so shards, markers, labels and results from previous runs are picked up as if
    they had been written here. Returns ``{source_dir: n_linked}``.
    """
    out: dict = {}
    root = Path(cfg.root)
    for src_root in sorted(Path(p) for p in __import__("glob").glob(pattern)):
        if src_root.resolve() == root.resolve():
            continue
        n = 0
        for src in src_root.rglob("*"):
            if not src.is_file():
                continue
            dst = root / src.relative_to(src_root)
            if dst.exists() or dst.is_symlink():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, dst)
            n += 1
        out[str(src_root)] = n
    return out


def free_space_gb(path) -> float:
    """Free disk space (GiB) of the filesystem holding ``path``."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return round(shutil.disk_usage(path).free / 2**30, 1)


def session_report(cfg=None) -> dict:
    """Print and return the facts that determine what this session can do."""
    import platform

    report = {
        "python": platform.python_version(),
        "in_colab": in_colab() and not in_kaggle(),
        "in_kaggle": in_kaggle(),
        "cpu_count": os.cpu_count(),
        "local_disk_free_gb": free_space_gb("/content" if in_colab() else Path.cwd()),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = props.name
            report["vram_gb"] = round(props.total_memory / 2**30, 1)
    except Exception as exc:  # torch missing or broken
        report["torch"] = f"unavailable ({exc})"
    if cfg is not None:
        report["source"] = cfg.source + (f" ({cfg.kaggle_input})" if cfg.source == "kaggle" else "")
        report["output_root"] = str(cfg.root)
        report["output_free_gb"] = free_space_gb(cfg.root)
    for key, value in report.items():
        print(f"{key:>20}: {value}")
    return report
