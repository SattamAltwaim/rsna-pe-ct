"""Remote zip access: central-directory index and parallel range-request downloads.

The whole dataset is one ~534 GB zip in a public S3 bucket. Zip files store
each member contiguously, so once we know a member's byte offset we can fetch
just that member with an HTTP ``Range`` request. The central directory
(~250 MB, 1.95M entries) is read once with ``remotezip`` and saved as a parquet
*zip index*; after that every download is plain ``requests`` with offsets.
"""

from __future__ import annotations

import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

INDEX_COLUMNS = ["filename", "header_offset", "compress_size", "file_size", "compress_type"]
LOCAL_HEADER_SIZE = 30
LOCAL_HEADER_SIG = b"PK\x03\x04"
# Local headers carry a filename + an "extra" field whose length is unknown until
# the header is read; over-fetching this many bytes avoids a second request.
EXTRA_SLACK = 128


# ---------------------------------------------------------------------------
# zip index
# ---------------------------------------------------------------------------


def build_zip_index(url: str, initial_buffer_mb: int = 64) -> pd.DataFrame:
    """Read the zip central directory once; one row per entry (dirs included)."""
    from remotezip import RemoteZip

    with RemoteZip(url, initial_buffer_size=initial_buffer_mb * 2**20) as zf:
        rows = [
            (i.filename, i.header_offset, i.compress_size, i.file_size, i.compress_type)
            for i in zf.infolist()
        ]
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return add_uid_columns(df)


def add_uid_columns(index: pd.DataFrame) -> pd.DataFrame:
    """Parse ``split/study_uid/series_uid/sop_uid.dcm`` into columns (empty for non-DICOM rows)."""
    df = index.copy()
    is_dcm = df["filename"].str.endswith(".dcm")
    for col in ["split", "study_uid", "series_uid", "sop_uid"]:
        df[col] = ""
    if is_dcm.any():
        parts = df.loc[is_dcm, "filename"].str.split("/", expand=True)
        if parts.shape[1] != 4:
            raise ValueError("expected DICOM paths of the form split/study/series/sop.dcm")
        df.loc[is_dcm, "split"] = parts[0].values
        df.loc[is_dcm, "study_uid"] = parts[1].values
        df.loc[is_dcm, "series_uid"] = parts[2].values
        df.loc[is_dcm, "sop_uid"] = parts[3].str.replace(".dcm", "", regex=False).values
    df["is_dcm"] = is_dcm
    return df


def read_member_bytes(url: str, filename: str) -> bytes:
    """Fetch one member by name through remotezip (used for the CSVs in NB00)."""
    from remotezip import RemoteZip

    with RemoteZip(url) as zf, zf.open(filename) as f:
        return f.read()


class StudyLocator:
    """Fast ``study_uid -> index rows`` lookup over the 1.9M-row zip index."""

    def __init__(self, index: pd.DataFrame, split: str = "train"):
        dcm = index[(index["is_dcm"]) & (index["split"] == split)]
        self._groups = dcm.groupby("study_uid", sort=False)
        self.study_uids = list(self._groups.groups.keys())

    def __contains__(self, study_uid: str) -> bool:
        return study_uid in self._groups.groups

    def entries(self, study_uid: str) -> pd.DataFrame:
        return self._groups.get_group(study_uid).reset_index(drop=True)

    def n_slices(self, study_uid: str) -> int:
        return len(self._groups.groups[study_uid])


# ---------------------------------------------------------------------------
# range requests
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _session() -> requests.Session:
    """One ``requests.Session`` per thread (connection reuse, thread-safe)."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        sess.mount("https://", adapter)
        _thread_local.session = sess
    return sess


def fetch_range(url: str, start: int, length: int, timeout: float = 60.0) -> bytes:
    """Exactly ``length`` bytes starting at ``start`` (raises on short reads)."""
    if length <= 0:
        return b""
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    resp = _session().get(url, headers=headers, timeout=timeout)
    if resp.status_code not in (200, 206):
        raise IOError(f"HTTP {resp.status_code} for range {headers['Range']}")
    data = resp.content
    if len(data) != length:
        raise IOError(f"short read: wanted {length} bytes, got {len(data)}")
    return data


def parse_local_header(buf: bytes) -> int:
    """Return the number of header bytes preceding the member data."""
    if buf[:4] != LOCAL_HEADER_SIG:
        raise IOError("bad local file header signature (wrong offset?)")
    name_len, extra_len = struct.unpack("<HH", buf[26:30])
    return LOCAL_HEADER_SIZE + name_len + extra_len


def fetch_entry(url: str, header_offset: int, compress_size: int, file_size: int,
                compress_type: int, filename_len: int = 0) -> bytes:
    """Fetch and decode one zip member given its central-directory fields.

    One range request covers header + data in the common case; a second one is
    issued only if the local header's extra field is longer than our slack.
    """
    want = LOCAL_HEADER_SIZE + filename_len + EXTRA_SLACK + compress_size
    buf = fetch_range(url, header_offset, want)
    data_start = parse_local_header(buf)
    data = buf[data_start : data_start + compress_size]
    if len(data) < compress_size:
        missing = compress_size - len(data)
        data += fetch_range(url, header_offset + len(buf), missing)
    if compress_type == 8:
        out = zlib.decompress(data, -15)
    elif compress_type == 0:
        out = data
    else:
        raise IOError(f"unsupported compression type {compress_type}")
    if len(out) != file_size:
        raise IOError(f"decoded {len(out)} bytes, expected {file_size}")
    return out


def fetch_entry_with_retry(url: str, row, retries: int = 4, backoff: float = 1.5) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_entry(
                url,
                int(row.header_offset),
                int(row.compress_size),
                int(row.file_size),
                int(row.compress_type),
                len(str(row.filename).encode("utf-8")),
            )
        except Exception as exc:  # network hiccups, throttling
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff**attempt * (0.5 + np.random.rand()))
    raise IOError(f"failed after {retries + 1} attempts: {last_exc}")


def download_study(url: str, entries: pd.DataFrame, dest_dir, workers: int = 24,
                   retries: int = 4, progress=None) -> list:
    """Fetch every DICOM of one study into ``dest_dir`` in parallel.

    Returns the written file paths. Any failed slice raises after the pool has
    drained, so a study is either complete on disk or reported as failed.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = list(entries.itertuples(index=False))
    paths: list = []
    errors: list = []

    def job(row):
        data = fetch_entry_with_retry(url, row, retries=retries)
        path = dest_dir / Path(row.filename).name
        path.write_bytes(data)
        return path

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(rows)))) as pool:
        futures = {pool.submit(job, row): row for row in rows}
        iterator = as_completed(futures)
        if progress is not None:
            iterator = progress(iterator, total=len(futures))
        for fut in iterator:
            try:
                paths.append(fut.result())
            except Exception as exc:
                errors.append((futures[fut].filename, str(exc)))
    if errors:
        raise IOError(f"{len(errors)}/{len(rows)} slices failed, first: {errors[0]}")
    return sorted(paths)
