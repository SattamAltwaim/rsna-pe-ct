"""Drive-safe storage: marker-gated shards and atomic single-file writes.

Google Drive is very slow with many small files and a half-written file that
looks complete is the worst failure mode, so:

* Per-study outputs are grouped into shards (~50 studies). A shard is built on
  fast local disk and only *counts* once its ``.done.json`` marker exists; the
  marker is written last and lists the shard's study UIDs and a checksum.
  An interrupted shard leaves no marker and is rebuilt from scratch.
* Every single-file output (parquet, json, npz, csv) is written to a temporary
  name and moved into place with ``os.replace``.

Two shard flavours share the same marker logic:

* **tar shards** (NB01): members ``{uid}.nii.gz`` + ``{uid}.meta.json`` per study.
* **npz shards** (NB03): one compressed npz holding arrays for ~50 studies.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# atomic single-file writes
# ---------------------------------------------------------------------------


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_write_bytes(path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=1, default=_json_default))


def atomic_write_npy(path, array: np.ndarray) -> Path:
    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return atomic_write_bytes(path, buf.getvalue())


def atomic_write_parquet(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def atomic_write_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def atomic_copy(src, dst) -> Path:
    """Copy a finished file to Drive without ever exposing a partial copy."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def sha256_file(path, chunk_mb: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_mb * 2**20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------


def shard_name(index: int, ext: str, prefix: str = "shard") -> str:
    return f"{prefix}_{index:03d}.{ext}"


def marker_path(shard_path) -> Path:
    """``shard_000.tar`` -> ``shard_000.done.json`` (also for ``.npz``)."""
    shard_path = Path(shard_path)
    stem = shard_path.name.split(".", 1)[0]
    return shard_path.with_name(stem + ".done.json")


def write_marker(shard_path, uids: list, extra: dict | None = None) -> Path:
    shard_path = Path(shard_path)
    payload = {
        "shard": shard_path.name,
        "n_samples": len(uids),
        "uids": list(map(str, uids)),
        "sha256": sha256_file(shard_path),
        "bytes": shard_path.stat().st_size,
    }
    if extra:
        payload.update(extra)
    return atomic_write_json(marker_path(shard_path), payload)


def list_done_shards(shard_dir, ext: str, prefix: str = "shard") -> list:
    """Shards whose marker exists, sorted by name (unmarked shards are invisible)."""
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        return []
    return sorted(
        p for p in shard_dir.glob(f"{prefix}_*.{ext}") if marker_path(p).exists()
    )


def read_markers(shard_dir, ext: str, prefix: str = "shard") -> list:
    return [json.loads(marker_path(p).read_text()) for p in list_done_shards(shard_dir, ext, prefix)]


def done_uids(shard_dir, ext: str, prefix: str = "shard") -> set:
    """All study UIDs already stored in marked shards (reads only the markers)."""
    out: set = set()
    for marker in read_markers(shard_dir, ext, prefix):
        out.update(marker["uids"])
    return out


def next_shard_index(shard_dir, ext: str, prefix: str = "shard") -> int:
    done = list_done_shards(shard_dir, ext, prefix)
    if not done:
        return 0
    return max(int(p.name.split(".", 1)[0].rsplit("_", 1)[1]) for p in done) + 1


# ---------------------------------------------------------------------------
# tar shards (volumes)
# ---------------------------------------------------------------------------


class TarShardWriter:
    """Build one tar shard locally; ``close()`` moves it into place and writes the marker.

    ``out_dir`` is where the finished shard must end up (usually Drive);
    ``build_dir`` is fast local disk where the tar is assembled.
    """

    def __init__(self, out_dir, shard_index: int, build_dir=None, prefix: str = "shard"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.final_path = self.out_dir / shard_name(shard_index, "tar", prefix)
        build_dir = Path(build_dir) if build_dir is not None else self.out_dir
        build_dir.mkdir(parents=True, exist_ok=True)
        self.build_path = build_dir / (self.final_path.name + ".building")
        self._tar = tarfile.open(self.build_path, "w")
        self.uids: list = []

    def add_file(self, uid: str, arcname: str, path) -> None:
        self._tar.add(str(path), arcname=arcname)
        if uid not in self.uids:
            self.uids.append(uid)

    def add_bytes(self, uid: str, arcname: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))
        if uid not in self.uids:
            self.uids.append(uid)

    def __len__(self) -> int:
        return len(self.uids)

    def close(self) -> Path | None:
        self._tar.close()
        if not self.uids:
            self.build_path.unlink(missing_ok=True)
            return None
        atomic_copy(self.build_path, self.final_path)
        self.build_path.unlink(missing_ok=True)
        write_marker(self.final_path, self.uids)
        return self.final_path

    def abort(self) -> None:
        self._tar.close()
        self.build_path.unlink(missing_ok=True)


def iter_tar_members(tar_path):
    """Yield ``(uid, {member_name: bytes})`` groups from one tar shard."""
    with tarfile.open(tar_path, "r") as tar:
        groups: dict = {}
        order: list = []
        for member in tar.getmembers():
            uid = member.name.split(".", 1)[0]
            if uid not in groups:
                groups[uid] = {}
                order.append(uid)
            groups[uid][member.name] = tar.extractfile(member).read()
        for uid in order:
            yield uid, groups[uid]


def prepare_local(remote_dir, local_dir, progress=None, prefix: str = "shard") -> list:
    """Copy finished tar shards to local disk and extract them once (sentinel-gated).

    Returns the names of shards extracted by this call; re-running is a no-op.
    """
    local_dir = Path(local_dir)
    cache = local_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fresh = []
    shard_paths = list_done_shards(remote_dir, "tar", prefix)
    iterator = progress(shard_paths) if progress is not None else shard_paths
    for remote_tar in iterator:
        sentinel = cache / f".extracted_{remote_tar.name.split('.', 1)[0]}"
        if sentinel.exists():
            continue
        local_tar = local_dir / remote_tar.name
        shutil.copyfile(remote_tar, local_tar)
        with tarfile.open(local_tar, "r") as tar:
            tar.extractall(cache, filter="data")
        local_tar.unlink()
        sentinel.touch()
        fresh.append(remote_tar.name)
    return fresh


class VolumeStore:
    """Read volumes + meta from an extracted shard cache (see :func:`prepare_local`)."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def uids(self) -> list:
        return sorted(p.name[: -len(".meta.json")] for p in self.cache_dir.glob("*.meta.json"))

    def has(self, uid: str) -> bool:
        return (self.cache_dir / f"{uid}.meta.json").exists()

    def load_meta(self, uid: str) -> dict:
        return json.loads((self.cache_dir / f"{uid}.meta.json").read_text())

    def load_volume(self, uid: str):
        """Returns ``(volume_hu int16 (z, y, x), meta dict)``."""
        from pe_ct.volume import load_nifti

        vol = load_nifti(self.cache_dir / f"{uid}.nii.gz")
        return vol, self.load_meta(uid)


class LiveVolumeStore:
    """Same interface as :class:`VolumeStore`, but reads straight from DICOM folders on disk.

    Used on Kaggle, where the dataset is mounted: metadata comes from the headers
    (fast, cached) and a volume is decoded only when asked for.
    """

    def __init__(self, cfg, studies, slice_index, uids):
        self.cfg = cfg
        self.slice_index = slice_index
        self._uids = list(uids)
        self._by_uid = studies if studies.index.name == "study_uid" else studies.set_index("study_uid")
        self._meta_cache: dict = {}

    def uids(self) -> list:
        return sorted(self._uids)

    def has(self, uid: str) -> bool:
        return uid in self._uids

    def load_meta(self, uid: str) -> dict:
        from pe_ct import pipeline

        if uid not in self._meta_cache:
            self._meta_cache[uid] = pipeline.study_meta_only(self.cfg, self.slice_index, uid, self._by_uid.loc[uid])
        return self._meta_cache[uid]

    def load_volume(self, uid: str):
        from pe_ct import pipeline, volume

        image, meta, _ = pipeline.process_study(self.cfg, None, self.slice_index, uid, self._by_uid.loc[uid], self.cfg.work / "dicom")
        self._meta_cache[uid] = meta
        return volume.image_to_hu_array(image), meta


# ---------------------------------------------------------------------------
# npz shards (embeddings)
# ---------------------------------------------------------------------------


class NpzShardWriter:
    """Accumulate per-study records in memory; ``close()`` writes one npz + marker.

    A record is a dict of ``{name: np.ndarray | scalar}``; every record must
    carry a ``uid`` string. Arrays may differ in shape between records.
    """

    def __init__(self, out_dir, shard_index: int, build_dir=None, prefix: str = "shard"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.final_path = self.out_dir / shard_name(shard_index, "npz", prefix)
        build_dir = Path(build_dir) if build_dir is not None else self.out_dir
        build_dir.mkdir(parents=True, exist_ok=True)
        self.build_path = build_dir / (self.final_path.name + ".building")
        self.records: list = []

    def add(self, record: dict) -> None:
        if "uid" not in record:
            raise ValueError("record needs a 'uid'")
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def uids(self) -> list:
        return [str(r["uid"]) for r in self.records]

    def close(self) -> Path | None:
        if not self.records:
            return None
        arrays = {"__uids__": np.array(self.uids)}
        for i, rec in enumerate(self.records):
            for key, value in rec.items():
                if key == "uid":
                    continue
                arrays[f"{i:04d}__{key}"] = np.asarray(value)
        with open(self.build_path, "wb") as f:
            np.savez_compressed(f, **arrays)
        atomic_copy(self.build_path, self.final_path)
        self.build_path.unlink(missing_ok=True)
        write_marker(self.final_path, self.uids)
        return self.final_path


def read_npz_shard(path) -> list:
    """Inverse of :class:`NpzShardWriter`: list of record dicts (with ``uid``)."""
    with np.load(path, allow_pickle=False) as data:
        uids = [str(u) for u in data["__uids__"]]
        records = [{"uid": uid} for uid in uids]
        for key in data.files:
            if key == "__uids__":
                continue
            idx, name = key.split("__", 1)
            value = data[key]
            records[int(idx)][name] = value.item() if value.ndim == 0 else value
    return records


def load_all_records(shard_dir, prefix: str = "shard", progress=None) -> dict:
    """``{uid: record}`` across every marked npz shard in ``shard_dir``."""
    out: dict = {}
    paths = list_done_shards(shard_dir, "npz", prefix)
    iterator = progress(paths) if progress is not None else paths
    for path in iterator:
        for rec in read_npz_shard(path):
            out[rec["uid"]] = rec
    return out


# ---------------------------------------------------------------------------
# failure log
# ---------------------------------------------------------------------------


class FailureLog:
    """``results/failures_<notebook>.csv``: uid, stage, reason, timestamp.

    Kept in memory and rewritten atomically on every ``add`` (it is tiny).
    """

    columns = ["uid", "stage", "reason", "timestamp"]

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.df = pd.read_csv(self.path, dtype=str).fillna("")
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def add(self, uid: str, stage: str, reason: str) -> None:
        row = pd.DataFrame(
            [[str(uid), stage, str(reason)[:500], pd.Timestamp.utcnow().isoformat()]],
            columns=self.columns,
        )
        self.df = pd.concat([self.df, row], ignore_index=True)
        atomic_write_csv(self.df, self.path)

    def uids(self) -> set:
        return set(self.df["uid"].astype(str))

    def __len__(self) -> int:
        return len(self.df)
