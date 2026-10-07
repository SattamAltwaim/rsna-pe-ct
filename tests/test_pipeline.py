"""Offline end-to-end: the real 8-slice fixture flows through fetch -> volume -> shard / embed."""

import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from pe_ct import io as pio
from pe_ct import labels, pipeline, storage
from pe_ct.config import QA_COLS, SUBLABEL_COLS, Config
from tests.conftest import FIXTURES

UID = "bc855cd8bdc9"


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """A Config on tmp disk, a locator, a label index, and a fake downloader that unpacks the fixture."""
    cfg = Config(drive_root=str(tmp_path / "drive"), work_dir=str(tmp_path / "work"), shard_size=1)
    cfg.ensure_dirs()
    index = pio.add_uid_columns(pd.read_csv(FIXTURES / f"{UID}_zip_index.csv"))
    locator = pio.StudyLocator(index)
    lab = pd.read_csv(FIXTURES / f"{UID}_labels.csv", dtype=str)
    lab["pe_present_on_image"] = lab["pe_present_on_image"].astype(np.int8)
    slice_index = labels.SliceLabelIndex(lab)
    row = {"study_uid": UID, "series_uid": "bab8e5153e7b", "n_slices": 8, "n_pos_slices": 3, "y": 1, "group": "pe"}
    row.update({c: 0 for c in SUBLABEL_COLS + QA_COLS})
    row["central_pe"] = 1
    studies = pd.DataFrame([row, dict(row, study_uid="missing_study", y=0, group="negative", n_pos_slices=0)])

    def fake_download(url, entries, dest_dir, workers=1, retries=0, progress=None):
        if entries.study_uid.iloc[0] != UID:
            raise IOError("simulated network failure")
        with tarfile.open(FIXTURES / f"{UID}_8slices.tar.gz") as tar:
            tar.extractall(dest_dir.parent, filter="data")
        return sorted(dest_dir.glob("*.dcm"))

    monkeypatch.setattr(pio, "download_study", fake_download)
    return cfg, locator, slice_index, studies


def test_process_study_aligns_labels_and_cleans_up(world):
    cfg, locator, slice_index, studies = world
    by_uid = studies.set_index("study_uid")
    image, meta, timings = pipeline.process_study(cfg, locator, slice_index, UID, by_uid.loc[UID], cfg.work / "dicom")
    assert image.GetSize() == (512, 512, 8)
    assert meta["slice_labels"] == [0, 1, 1, 1, 0, 0, 0, 0] and meta["y"] == 1 and meta["central_pe"] == 1
    assert meta["hu_sanity"]["frac_air"] > 0.2
    assert set(timings) == {"t_download_s", "t_decode_s"}
    assert not (cfg.work / "dicom" / UID).exists()  # DICOMs deleted


def test_run_volume_download_is_resumable_and_logs_failures(world):
    cfg, locator, slice_index, studies = world
    log = storage.FailureLog(cfg.failures_csv("nb01"))
    stats = pipeline.run_volume_download(cfg, studies, [UID, "missing_study"], locator, slice_index, log, log=lambda *a: None)
    assert stats["done"] == 1 and stats["failed"] == 1
    assert storage.done_uids(cfg.volumes_eda_dir, "tar") == {UID}
    assert log.uids() == {"missing_study"} and "not in zip index" in log.df.reason.iloc[0]
    assert list((cfg.work / "shards").glob("*")) == []
    # second run: nothing left to do, and the failed study is not retried
    stats2 = pipeline.run_volume_download(cfg, studies, [UID, "missing_study"], locator, slice_index, log, log=lambda *a: None)
    assert stats2["done"] == 0 and stats2["failed"] == 0
    storage.prepare_local(cfg.volumes_eda_dir, cfg.work / "volumes_eda")
    vol, meta = storage.VolumeStore(cfg.work / "volumes_eda" / "cache").load_volume(UID)
    assert vol.shape == (8, 512, 512) and meta["slice_labels"][1:4] == [1, 1, 1]


def test_run_embedding_extraction_with_tiny_model(world):
    spectre = pytest.importorskip("spectre")
    from pe_ct import embed

    cfg, locator, slice_index, studies = world
    torch.manual_seed(0)
    model = embed.load_model("spectre-small", device="cpu", dtype=torch.float32, pretrained=False)
    log = storage.FailureLog(cfg.failures_csv("nb03"))
    stats = pipeline.run_embedding_extraction(cfg, model, studies, [UID, "missing_study"], locator, slice_index, log, prefetch=1, log=lambda *a: None)
    assert stats["done"] == 1 and stats["failed"] == 1
    recs = storage.load_all_records(cfg.embeddings_dir)
    rec = recs[UID]
    d = model.feature_combiner.embed_dim
    assert rec["cls"].shape == (d,) and rec["crop_desc"].shape[1] == 2 * model.backbone.embed_dim
    # 8 slices < one crop depth: the S axis is padded, grid 4 x 4 x 1
    assert tuple(rec["grid"]) == (4, 4, 1) and rec["boxes_lpi"].shape == (16, 3, 2)
    assert rec["boxes_lpi"][:, 0, :].tolist() == [[0, 8]] * 16
    assert rec["slice_labels"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0] and rec["y"] == 1
    assert rec["manufacturer"] == "unknown" and rec["scanner"] == "kernel B30f"
    vols = pipeline.fetch_volumes_for_uids(cfg, studies.set_index("study_uid"), [UID], locator, slice_index)
    assert vols[UID][0].shape == (8, 512, 512)
