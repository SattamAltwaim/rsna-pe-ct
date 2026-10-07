"""Per-study processing used by NB01 (EDA volumes) and NB03 (streaming embeddings).

    fetch DICOMs (parallel range requests) -> SimpleITK series -> LPI HU volume
    -> align slice labels -> [save to shard | embed] -> delete DICOMs

Both loops are resumable: studies already present in a marked shard (or in the
failure log) are skipped, and a shard that was being built when the session
died simply never got its marker and is rebuilt.
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pe_ct import io as pio
from pe_ct import labels as plabels
from pe_ct import storage, volume
from pe_ct.config import QA_COLS, SUBLABEL_COLS, Config


def _by_uid(studies) -> "pd.DataFrame":
    """Study table indexed by ``study_uid`` whether or not it already is."""
    return studies if studies.index.name == "study_uid" else studies.set_index("study_uid")


def study_meta_from_row(study_row) -> dict:
    """The study-level labels we copy into every volume / embedding record."""
    out = {"y": int(study_row["y"]), "group": str(study_row["group"]),
           "n_slices_csv": int(study_row["n_slices"]), "n_pos_slices": int(study_row["n_pos_slices"])}
    for c in SUBLABEL_COLS + QA_COLS:
        out[c] = int(study_row[c])
    return out


def fetch_study(cfg: Config, locator, study_uid: str, work_dir, series_uid: str | None = None) -> tuple:
    """Where one study's DICOM files are: ``(directory, owned)``.

    * ``source == "kaggle"``: the read-only series folder of the mounted dataset (not owned,
      never deleted).
    * ``source == "s3"``: downloaded into ``work_dir/<uid>/`` with range requests (owned,
      deleted by the caller after use).
    """
    if cfg.source == "kaggle":
        d = cfg.kaggle_study_dir(study_uid, series_uid)
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not found in the mounted dataset")
        return d, False
    if locator is None:
        raise ValueError("an s3 source needs a StudyLocator (zip index)")
    if study_uid not in locator:
        raise KeyError(f"{study_uid} not in zip index")
    entries = locator.entries(study_uid)
    dest = Path(work_dir) / study_uid
    if dest.exists():
        shutil.rmtree(dest)
    pio.download_study(cfg.zip_url, entries, dest, workers=cfg.download_workers, retries=cfg.download_retries)
    return dest, True


def build_volume(dcm_dir, study_uid: str, study_row, slice_labels) -> tuple:
    """DICOM dir -> ``(sitk.Image LPI, meta)`` with aligned ``meta["slice_labels"]``."""
    image, sop_uids, header = volume.read_series(dcm_dir)
    meta = volume.build_meta(image, sop_uids, header, study_uid, str(study_row["series_uid"]))
    meta["slice_labels"] = plabels.align_slice_labels(sop_uids, slice_labels).tolist()
    meta.update(study_meta_from_row(study_row))
    meta["hu_sanity"] = volume.hu_sanity(volume.image_to_hu_array(image))
    return image, meta


def process_study(cfg: Config, locator, slice_index: plabels.SliceLabelIndex, study_uid: str, study_row, work_dir) -> tuple:
    """fetch -> volume -> labels. Returns ``(image, meta, timings)``; DICOMs are deleted.

    ``study_row`` is the study's row of the study table (indexed by ``study_uid`` or not).
    """
    uid = str(study_uid)
    t0 = time.time()
    dcm_dir, owned = fetch_study(cfg, locator, uid, work_dir, series_uid=str(study_row["series_uid"]))
    t1 = time.time()
    try:
        image, meta = build_volume(dcm_dir, uid, study_row, slice_index.for_study(uid))
    finally:
        if owned:
            shutil.rmtree(dcm_dir, ignore_errors=True)
    timings = {"t_download_s": round(t1 - t0, 2), "t_decode_s": round(time.time() - t1, 2)}
    return image, meta, timings


# ---------------------------------------------------------------------------
# labels, splits and the study locator (built on demand, deterministic)
# ---------------------------------------------------------------------------


def load_train(cfg: Config) -> pd.DataFrame:
    """``train.csv`` from the output root, fetched from the source the first time."""
    if not cfg.train_csv.exists():
        if cfg.source == "kaggle":
            src = Path(cfg.kaggle_input) / "train.csv"
            storage.atomic_write_bytes(cfg.train_csv, src.read_bytes())
        else:
            storage.atomic_write_bytes(cfg.train_csv, pio.read_member_bytes(cfg.zip_url, "train.csv"))
    return plabels.load_train_csv(cfg.train_csv)


def ensure_labels_and_splits(cfg: Config, log=print) -> tuple:
    """``(studies, split_table)``: read if present, otherwise rebuilt (seeded, so identical)."""
    from pe_ct import splits as psplits

    if cfg.study_labels_path.exists() and cfg.splits_path.exists():
        return pd.read_parquet(cfg.study_labels_path), pd.read_parquet(cfg.splits_path)
    log("labels/splits not found: building them from train.csv")
    studies = plabels.study_table(load_train(cfg))
    storage.atomic_write_parquet(studies, cfg.study_labels_path)
    split_table = psplits.make_splits(plabels.usable_studies(studies), cfg)
    psplits.check_no_leakage(split_table)
    storage.atomic_write_parquet(split_table, cfg.splits_path)
    return studies, split_table


def make_locator(cfg: Config, log=print):
    """``StudyLocator`` over the zip index for the s3 source (built once); ``None`` on Kaggle."""
    if cfg.source == "kaggle":
        return None
    if cfg.zip_index_path.exists():
        index = pd.read_parquet(cfg.zip_index_path)
    else:
        log("zip index not found: reading the zip central directory (about a minute)")
        index = pio.build_zip_index(cfg.zip_url)
        storage.atomic_write_parquet(index, cfg.zip_index_path)
    return pio.StudyLocator(index)


# ---------------------------------------------------------------------------
# NB01: EDA volumes -> tar shards on Drive
# ---------------------------------------------------------------------------


def pending_uids(all_uids, shard_dir, ext: str, failure_log: storage.FailureLog) -> list:
    done = storage.done_uids(shard_dir, ext)
    failed = failure_log.uids()
    return [u for u in all_uids if u not in done and u not in failed]


def run_volume_download(cfg: Config, studies, uids, locator, slice_index, failure_log, progress=None, log=print) -> dict:
    """Resumable loop writing ``volumes_eda/shard_XXX.tar`` (+ markers). Returns a summary."""
    by_uid = _by_uid(studies)
    todo = pending_uids(uids, cfg.volumes_eda_dir, "tar", failure_log)
    log(f"{len(uids)} requested, {len(uids) - len(todo)} already done/failed, {len(todo)} to do")
    work = cfg.work / "dicom"
    build = cfg.work / "shards"
    stats = {"done": 0, "failed": 0, "seconds": [], "bytes": 0}
    writer = None
    iterator = progress(todo) if progress is not None else todo
    for uid in iterator:
        if writer is None:
            writer = storage.TarShardWriter(cfg.volumes_eda_dir, storage.next_shard_index(cfg.volumes_eda_dir, "tar"), build_dir=build)
        t0 = time.time()
        try:
            image, meta, timings = process_study(cfg, locator, slice_index, uid, by_uid.loc[uid], work)
            meta.update(timings)
            nii = build / f"{uid}.nii.gz"
            volume.save_nifti(image, nii)
            writer.add_file(uid, f"{uid}.nii.gz", nii)
            writer.add_bytes(uid, f"{uid}.meta.json", storage_json(meta))
            stats["bytes"] += nii.stat().st_size
            nii.unlink()
            stats["done"] += 1
        except Exception as exc:  # one bad study must not stop the loop
            failure_log.add(uid, "volume", repr(exc))
            stats["failed"] += 1
        stats["seconds"].append(time.time() - t0)
        if len(writer) >= cfg.shard_size:
            path = writer.close()
            writer = None
            log(f"shard written: {path.name}")
    if writer is not None:
        path = writer.close()
        if path is not None:
            log(f"shard written: {path.name}")
    return stats


def ensure_eda_volumes(cfg: Config, studies, split_table, locator, slice_index, progress=None, log=print):
    """Build any EDA volume shard that is missing (no-op when NB01 already ran). Returns stats or None."""
    uids = split_table.loc[split_table["in_eda"], "study_uid"].tolist()
    failure_log = storage.FailureLog(cfg.failures_csv("01_download_subset"))
    if not pending_uids(uids, cfg.volumes_eda_dir, "tar", failure_log):
        return None
    log("EDA volumes incomplete: building them now")
    return run_volume_download(cfg, studies, uids, locator, slice_index, failure_log, progress=progress, log=log)


def storage_json(meta: dict) -> bytes:
    import json

    return json.dumps(meta, default=storage._json_default).encode("utf-8")


# ---------------------------------------------------------------------------
# NB03: streaming embeddings -> npz shards on Drive
# ---------------------------------------------------------------------------


def embedding_record(uid: str, meta: dict, rec: dict) -> dict:
    """Flatten what NB04/NB05 need into one npz-friendly record."""
    out = {
        "uid": uid,
        "y": int(meta["y"]),
        "cls": rec["cls"].astype(np.float32),
        "crop_tokens": rec["crop_tokens"],
        "crop_desc": rec["crop_desc"],
        "grid": rec["grid"],
        "boxes_lpi": rec["boxes_lpi"],
        "ras_shape": rec["ras_shape"],
        "shape_zyx": np.asarray(meta["shape_zyx"], dtype=np.int64),
        "spacing_xyz_mm": np.asarray(meta["spacing_xyz_mm"], dtype=np.float32),
        "slice_labels": np.asarray(meta["slice_labels"], dtype=np.int8),
        "n_pos_slices": int(meta["n_pos_slices"]),
        "manufacturer": str(meta.get("manufacturer", "unknown")),
        "scanner": str(meta.get("scanner", meta.get("manufacturer", "unknown"))),
        "slice_thickness_mm": float(meta.get("slice_thickness_mm", np.nan)),
        "n_missing_slices_est": int(meta.get("n_missing_slices_est", 0)),
        "resampled": bool(rec.get("resampled", False)),
        "t_gpu_s": float(rec.get("t_gpu_s", np.nan)),
        "t_download_s": float(meta.get("t_download_s", np.nan)),
        "t_decode_s": float(meta.get("t_decode_s", np.nan)),
    }
    for c in SUBLABEL_COLS + QA_COLS:
        out[c] = int(meta[c])
    return out


def run_embedding_extraction(cfg: Config, model, studies, uids, locator, slice_index, failure_log,
                             prefetch: int = 2, progress=None, log=print) -> dict:
    """Resumable streaming loop: CPU prefetch of the next studies while the GPU embeds the current one."""
    from pe_ct import embed

    by_uid = _by_uid(studies)
    todo = pending_uids(uids, cfg.embeddings_dir, "npz", failure_log)
    log(f"{len(uids)} requested, {len(uids) - len(todo)} already done/failed, {len(todo)} to do")
    work = cfg.work / "dicom"
    build = cfg.work / "shards"
    stats = {"done": 0, "failed": 0, "seconds": [], "t_gpu": [], "t_cpu": []}
    writer = None

    def prepare(uid):
        return process_study(cfg, locator, slice_index, uid, by_uid.loc[uid], work / "prefetch")

    iterator = progress(todo) if progress is not None else todo
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        futures = {}
        queue = list(todo)
        for uid in queue[:prefetch]:
            futures[uid] = pool.submit(prepare, uid)
        for i, uid in enumerate(iterator):
            if writer is None:
                writer = storage.NpzShardWriter(cfg.embeddings_dir, storage.next_shard_index(cfg.embeddings_dir, "npz"), build_dir=build)
            nxt = i + prefetch
            if nxt < len(queue):
                futures[queue[nxt]] = pool.submit(prepare, queue[nxt])
            t0 = time.time()
            try:
                image, meta, timings = futures.pop(uid).result()
                meta.update(timings)
                rec = embed.embed_image(model, image, cfg.resample_spacing)
                writer.add(embedding_record(uid, meta, rec))
                stats["done"] += 1
                stats["t_gpu"].append(rec["t_gpu_s"])
                stats["t_cpu"].append(timings["t_download_s"] + timings["t_decode_s"])
            except Exception as exc:
                failure_log.add(uid, "embed", repr(exc))
                stats["failed"] += 1
            stats["seconds"].append(time.time() - t0)
            if len(writer) >= cfg.shard_size:
                path = writer.close()
                writer = None
                log(f"shard written: {path.name}")
    if writer is not None:
        path = writer.close()
        if path is not None:
            log(f"shard written: {path.name}")
    return stats


def fetch_volumes_for_uids(cfg: Config, studies, uids, locator, slice_index, progress=None) -> dict:
    """Small helper for NB05 galleries: ``{uid: (volume int16, meta)}`` for a handful of studies."""
    by_uid = _by_uid(studies)
    out = {}
    iterator = progress(uids) if progress is not None else uids
    for uid in iterator:
        image, meta, _ = process_study(cfg, locator, slice_index, uid, by_uid.loc[uid], cfg.work / "dicom")
        out[uid] = (volume.image_to_hu_array(image), meta)
    return out
