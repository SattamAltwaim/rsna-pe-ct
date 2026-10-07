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


@pytest.fixture()
def kaggle_world(tmp_path):
    """A fake /kaggle layout: mounted dataset with the fixture study, plus an attached earlier output."""
    inp = tmp_path / "input" / "rsna-str-pulmonary-embolism-detection"
    series_dir = inp / "train" / UID / "bab8e5153e7b"
    series_dir.mkdir(parents=True)
    with tarfile.open(FIXTURES / f"{UID}_8slices.tar.gz") as tar:
        tar.extractall(tmp_path / "unpack", filter="data")
    for f in (tmp_path / "unpack" / UID).glob("*.dcm"):
        f.rename(series_dir / f.name)
    lab = pd.read_csv(FIXTURES / f"{UID}_labels.csv", dtype=str)
    # a tiny train.csv: the fixture study plus a few synthetic negatives/positives so splits work
    rows = [lab]
    rng = np.random.default_rng(0)
    for s in range(60):
        n = 5
        y = s % 3 == 0
        df = pd.DataFrame({c: [lab[c].iloc[0]] * n for c in lab.columns})
        df["StudyInstanceUID"], df["SeriesInstanceUID"] = f"syn{s:03d}", f"ser{s:03d}"
        df["SOPInstanceUID"] = [f"sop{s:03d}_{i}" for i in range(n)]
        df["pe_present_on_image"] = [1, 1, 0, 0, 0] if y else [0] * n
        df["negative_exam_for_pe"] = 0 if y else 1
        for c in ["central_pe", "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1", "leftsided_pe", "rightsided_pe", "chronic_pe", "acute_and_chronic_pe", "indeterminate", "qa_motion", "qa_contrast", "flow_artifact", "true_filling_defect_not_pe"]:
            df[c] = 0
        if y:
            df["leftsided_pe"] = 1
            df["rv_lv_ratio_lt_1"] = 1
            df["central_pe"] = int(rng.random() < 0.3)
        rows.append(df)
    pd.concat(rows, ignore_index=True).to_csv(inp / "train.csv", index=False)
    cfg = Config(source="kaggle", kaggle_input=str(inp), drive_root=str(tmp_path / "working" / "rsna-pe"),
                 work_dir=str(tmp_path / "tmpwork"), n_eda=4, n_dev=30, n_test_sample=8, shard_size=5)
    cfg.ensure_dirs()
    return cfg, inp, tmp_path


def test_kaggle_fetch_reads_in_place_and_never_deletes(kaggle_world):
    cfg, inp, _ = kaggle_world
    d, owned = pipeline.fetch_study(cfg, None, UID, cfg.work, series_uid="bab8e5153e7b")
    assert d == inp / "train" / UID / "bab8e5153e7b" and owned is False
    d2, _ = pipeline.fetch_study(cfg, None, UID, cfg.work)  # series found by glob
    assert d2 == d
    with pytest.raises(FileNotFoundError):
        pipeline.fetch_study(cfg, None, "nope", cfg.work)
    studies, _ = pipeline.ensure_labels_and_splits(cfg, log=lambda *a: None)
    row = studies.set_index("study_uid").loc[UID]
    slice_index = labels.SliceLabelIndex(pipeline.load_train(cfg))
    image, meta, _ = pipeline.process_study(cfg, None, slice_index, UID, row, cfg.work)
    assert image.GetSize() == (512, 512, 8) and meta["slice_labels"][1:4] == [1, 1, 1]
    assert len(list(d.glob("*.dcm"))) == 8  # input untouched


def test_kaggle_labels_splits_are_deterministic_and_locator_is_none(kaggle_world):
    cfg, _, _ = kaggle_world
    assert pipeline.make_locator(cfg) is None
    studies, splits1 = pipeline.ensure_labels_and_splits(cfg, log=lambda *a: None)
    assert cfg.train_csv.exists() and cfg.splits_path.exists()
    cfg.splits_path.unlink()
    _, splits2 = pipeline.ensure_labels_and_splits(cfg, log=lambda *a: None)
    pd.testing.assert_frame_equal(splits1, splits2)
    assert UID in set(studies.study_uid)


def test_link_kaggle_inputs_symlinks_earlier_outputs(kaggle_world):
    from pe_ct import colab

    cfg, _, tmp_path = kaggle_world
    earlier = tmp_path / "input" / "nb03-output" / "rsna-pe" / "embeddings" / "spectre"
    earlier.mkdir(parents=True)
    (earlier / "shard_000.npz").write_bytes(b"x")
    (earlier / "shard_000.done.json").write_text("{}")
    (cfg.embeddings_dir / "shard_001.npz").write_bytes(b"mine")
    linked = colab.link_kaggle_inputs(cfg, pattern=str(tmp_path / "input" / "*" / "rsna-pe"))
    assert linked == {str(tmp_path / "input" / "nb03-output" / "rsna-pe"): 2}
    assert (cfg.embeddings_dir / "shard_000.npz").is_symlink()
    assert (cfg.embeddings_dir / "shard_000.npz").read_bytes() == b"x"
    assert (cfg.embeddings_dir / "shard_001.npz").read_bytes() == b"mine"
    # idempotent
    assert colab.link_kaggle_inputs(cfg, pattern=str(tmp_path / "input" / "*" / "rsna-pe")) == {str(tmp_path / "input" / "nb03-output" / "rsna-pe"): 0}
    # a rewritten marker/csv replaces the symlink, not the read-only source
    storage.atomic_write_text(cfg.embeddings_dir / "shard_000.done.json", '{"new": 1}')
    assert not (cfg.embeddings_dir / "shard_000.done.json").is_symlink()
    assert (earlier / "shard_000.done.json").read_text() == "{}"


def test_ensure_eda_volumes_builds_then_noops(kaggle_world):
    cfg, _, _ = kaggle_world
    studies, split_table = pipeline.ensure_labels_and_splits(cfg, log=lambda *a: None)
    split_table["in_eda"] = split_table.study_uid == UID   # only the real study has files
    slice_index = labels.SliceLabelIndex(pipeline.load_train(cfg))
    stats = pipeline.ensure_eda_volumes(cfg, studies, split_table, None, slice_index, log=lambda *a: None)
    assert stats["done"] == 1
    assert pipeline.ensure_eda_volumes(cfg, studies, split_table, None, slice_index, log=lambda *a: None) is None
