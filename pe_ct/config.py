"""Paths, seeds, subset sizes and physical constants.

Everything a notebook might want to tune lives in :class:`Config`; constants
that must never change (HU reference values, the bucket URL) are module-level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Public AWS Open Data bucket (no credentials needed, supports HTTP range requests).
ZIP_URL = "https://pulmonary-embolism-detection.s3.us-west-2.amazonaws.com/rsna-ped-dataset.zip"

# Hounsfield reference values (calibrated density scale, same on every scanner).
HU_REFERENCE = {
    "air": -1000,
    "fat": -100,
    "water": 0,
    "soft tissue": 40,
    "contrast blood": 300,
    "bone": 1000,
}

# The dataset's study-level label columns (constant within a study).
SUBLABEL_COLS = [
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "central_pe",
    "leftsided_pe",
    "rightsided_pe",
    "chronic_pe",
    "acute_and_chronic_pe",
]
QA_COLS = ["qa_motion", "qa_contrast", "flow_artifact", "true_filling_defect_not_pe"]
GROUP_COLS = ["negative_exam_for_pe", "indeterminate"]


@dataclass
class Config:
    """One object holds every tunable; notebooks construct it in their CONFIG cell."""

    drive_root: str = "/content/drive/MyDrive/rsna-pe"
    work_dir: str = "/content/work"
    zip_url: str = ZIP_URL
    seed: int = 0

    # Splits (NB00)
    test_frac: float = 0.20
    n_eda: int = 300
    n_dev: int = 2000
    n_test_sample: int = 600
    n_folds: int = 5

    # Storage / download
    shard_size: int = 50
    download_workers: int = 24
    download_retries: int = 4

    # Backbone (NB03)
    spectre_model: str = "cclaess/SPECTRE-Large"
    crop_size: tuple = (128, 128, 64)  # (x, y, z) voxels per SPECTRE crop
    resample_spacing: tuple | None = None  # None = native spacing (recorded in meta)

    # Probe (NB04)
    probe_epochs: int = 200
    probe_lr: float = 1e-3
    probe_weight_decay: float = 1e-2
    probe_patience: int = 20

    extra: dict = field(default_factory=dict)

    # ---- derived paths -------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.drive_root)

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def volumes_eda_dir(self) -> Path:
        return self.root / "volumes_eda"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings" / "spectre"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def work(self) -> Path:
        return Path(self.work_dir)

    def all_dirs(self) -> list:
        return [
            self.labels_dir,
            self.splits_dir,
            self.volumes_eda_dir,
            self.embeddings_dir,
            self.results_dir,
            self.figures_dir,
            self.work,
        ]

    def ensure_dirs(self) -> None:
        for d in self.all_dirs():
            Path(d).mkdir(parents=True, exist_ok=True)

    # file paths used by more than one notebook
    @property
    def train_csv(self) -> Path:
        return self.labels_dir / "train.csv"

    @property
    def study_labels_path(self) -> Path:
        return self.labels_dir / "study_labels.parquet"

    @property
    def zip_index_path(self) -> Path:
        return self.labels_dir / "zip_index.parquet"

    @property
    def splits_path(self) -> Path:
        return self.splits_dir / "splits.parquet"

    def failures_csv(self, notebook: str) -> Path:
        return self.results_dir / f"failures_{notebook}.csv"


def default_config() -> Config:
    """Colab paths on Colab; a local ``./data`` tree on a workstation."""
    if os.path.exists("/content"):
        return Config()
    local = Path(os.environ.get("PE_CT_DATA", Path.cwd() / "data")).resolve()
    return Config(drive_root=str(local), work_dir=str(local / "work"))
