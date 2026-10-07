"""Colab session helpers. Everything degrades gracefully on a workstation."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def mount_drive(mount_point: str = "/content/drive") -> Path | None:
    """Mount Google Drive on Colab; returns the MyDrive path (None locally)."""
    if not in_colab():
        return None
    mount = Path(mount_point)
    if not (mount / "MyDrive").exists():
        from google.colab import drive  # type: ignore

        drive.mount(str(mount))
    return mount / "MyDrive"


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
        "in_colab": in_colab(),
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
        report["drive_root"] = str(cfg.root)
        report["drive_free_gb"] = free_space_gb(cfg.root)
    for key, value in report.items():
        print(f"{key:>20}: {value}")
    return report
