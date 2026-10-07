import io
import zipfile
import zlib

import numpy as np
import pandas as pd
import pytest

from pe_ct import io as pio
from tests.conftest import FIXTURES


def _make_zip(members, compression):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _index_from_bytes(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        rows = [(i.filename, i.header_offset, i.compress_size, i.file_size, i.compress_type) for i in zf.infolist()]
    return pio.add_uid_columns(pd.DataFrame(rows, columns=pio.INDEX_COLUMNS))


@pytest.mark.parametrize("compression", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_fetch_entry_matches_zipfile(monkeypatch, compression):
    rng = np.random.default_rng(0)
    members = {
        "train/aaa/bbb/ccc.dcm": rng.integers(0, 256, 5000, dtype=np.uint8).tobytes(),
        "train/aaa/bbb/ddd.dcm": b"hello" * 1000,
        "train.csv": b"a,b\n1,2\n",
    }
    blob = _make_zip(members, compression)
    monkeypatch.setattr(pio, "fetch_range", lambda url, start, length, timeout=60.0: blob[start: start + length])
    index = _index_from_bytes(blob)
    for row in index.itertuples(index=False):
        out = pio.fetch_entry_with_retry("http://x", row, retries=0)
        assert out == members[row.filename]


def test_fetch_entry_second_request_when_extra_is_long(monkeypatch):
    blob = _make_zip({"f.bin": b"x" * 3000}, zipfile.ZIP_STORED)
    monkeypatch.setattr(pio, "EXTRA_SLACK", 0)
    calls = []

    def fake(url, start, length, timeout=60.0):
        calls.append((start, length))
        return blob[start: start + length]

    monkeypatch.setattr(pio, "fetch_range", fake)
    row = _index_from_bytes(blob).iloc[0]
    assert pio.fetch_entry_with_retry("u", row, retries=0) == b"x" * 3000


def test_bad_offset_raises(monkeypatch):
    monkeypatch.setattr(pio, "fetch_range", lambda *a, **k: b"\x00" * 200)
    with pytest.raises(IOError):
        pio.fetch_entry("u", 0, 10, 10, 0, 1)


def test_add_uid_columns_and_locator():
    df = pd.DataFrame({
        "filename": ["train/", "train/s1/se1/a.dcm", "train/s1/se1/b.dcm", "test/s2/se2/c.dcm", "train.csv"],
        "header_offset": [0, 10, 20, 30, 40], "compress_size": [0, 5, 5, 5, 5], "file_size": [0, 5, 5, 5, 5], "compress_type": [0, 8, 8, 8, 8],
    })
    idx = pio.add_uid_columns(df)
    assert idx.is_dcm.tolist() == [False, True, True, True, False]
    assert idx.loc[1, "study_uid"] == "s1" and idx.loc[1, "sop_uid"] == "a"
    loc = pio.StudyLocator(idx, split="train")
    assert loc.study_uids == ["s1"] and "s2" not in loc and loc.n_slices("s1") == 2
    assert loc.entries("s1").sop_uid.tolist() == ["a", "b"]


def test_download_study_writes_files(monkeypatch, tmp_path):
    blob = _make_zip({"train/s/se/a.dcm": b"A" * 100, "train/s/se/b.dcm": b"B" * 100}, zipfile.ZIP_DEFLATED)
    monkeypatch.setattr(pio, "fetch_range", lambda url, start, length, timeout=60.0: blob[start: start + length])
    index = _index_from_bytes(blob)
    paths = pio.download_study("u", index, tmp_path / "s", workers=2)
    assert sorted(p.name for p in paths) == ["a.dcm", "b.dcm"]
    assert (tmp_path / "s" / "a.dcm").read_bytes() == b"A" * 100


def _online() -> bool:
    try:
        import requests

        return requests.head(pio.ZIP_URL if hasattr(pio, "ZIP_URL") else
                             "https://pulmonary-embolism-detection.s3.us-west-2.amazonaws.com/rsna-ped-dataset.zip",
                             timeout=5).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _online(), reason="needs network access to the public bucket")
def test_real_range_request_matches_fixture(tmp_path):
    """Fetch one real slice from S3 and compare it byte-for-byte with the committed fixture."""
    import tarfile

    from pe_ct.config import ZIP_URL

    index = pio.add_uid_columns(pd.read_csv(FIXTURES / "bc855cd8bdc9_zip_index.csv"))
    row = index.iloc[0]
    data = pio.fetch_entry_with_retry(ZIP_URL, row, retries=2)
    with tarfile.open(FIXTURES / "bc855cd8bdc9_8slices.tar.gz") as tar:
        member = tar.extractfile(f"bc855cd8bdc9/{row.sop_uid}.dcm").read()
    assert data == member
