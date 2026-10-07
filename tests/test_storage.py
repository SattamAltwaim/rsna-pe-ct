import json

import numpy as np
import pandas as pd

from pe_ct import storage


def test_atomic_writes_leave_no_tmp(tmp_path):
    storage.atomic_write_json(tmp_path / "a" / "x.json", {"k": np.int64(3), "arr": np.arange(2)})
    storage.atomic_write_parquet(pd.DataFrame({"a": [1, 2]}), tmp_path / "x.parquet")
    storage.atomic_write_csv(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")
    assert json.loads((tmp_path / "a" / "x.json").read_text()) == {"k": 3, "arr": [0, 1]}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_tar_shard_roundtrip_and_marker(tmp_path):
    out, build = tmp_path / "drive", tmp_path / "local"
    w = storage.TarShardWriter(out, 0, build_dir=build)
    w.add_bytes("u1", "u1.meta.json", b'{"a": 1}')
    w.add_bytes("u1", "u1.nii.gz", b"xyz")
    w.add_bytes("u2", "u2.meta.json", b'{"a": 2}')
    path = w.close()
    assert path.name == "shard_000.tar"
    assert storage.marker_path(path).exists()
    marker = json.loads(storage.marker_path(path).read_text())
    assert marker["uids"] == ["u1", "u2"] and marker["sha256"] == storage.sha256_file(path)
    groups = dict(storage.iter_tar_members(path))
    assert set(groups) == {"u1", "u2"} and groups["u1"]["u1.nii.gz"] == b"xyz"
    assert list(build.glob("*")) == []  # build dir cleaned up


def test_unmarked_shards_are_invisible(tmp_path):
    w = storage.TarShardWriter(tmp_path, 0)
    w.add_bytes("u1", "u1.meta.json", b"{}")
    w.close()
    (tmp_path / "shard_001.tar").write_bytes(b"partial")
    assert [p.name for p in storage.list_done_shards(tmp_path, "tar")] == ["shard_000.tar"]
    assert storage.done_uids(tmp_path, "tar") == {"u1"}
    assert storage.next_shard_index(tmp_path, "tar") == 1
    assert storage.list_done_shards(tmp_path / "missing", "tar") == []
    assert storage.next_shard_index(tmp_path / "missing", "tar") == 0


def test_empty_shard_writes_nothing(tmp_path):
    assert storage.TarShardWriter(tmp_path, 0).close() is None
    assert storage.NpzShardWriter(tmp_path, 0).close() is None
    assert list(tmp_path.glob("shard_*")) == []


def test_npz_shard_roundtrip(tmp_path):
    w = storage.NpzShardWriter(tmp_path, 3)
    w.add({"uid": "a", "cls": np.ones(4, np.float32), "tokens": np.zeros((3, 2), np.float16), "y": 1})
    w.add({"uid": "b", "cls": np.zeros(4, np.float32), "tokens": np.ones((5, 2), np.float16), "y": 0})
    path = w.close()
    assert path.name == "shard_003.npz"
    recs = storage.read_npz_shard(path)
    assert [r["uid"] for r in recs] == ["a", "b"]
    assert recs[1]["tokens"].shape == (5, 2) and recs[0]["y"] == 1
    allrecs = storage.load_all_records(tmp_path)
    assert set(allrecs) == {"a", "b"}


def test_prepare_local_and_volume_store(tmp_path, tiny_volume):
    from pe_ct import volume

    vol, meta = tiny_volume
    remote, local = tmp_path / "remote", tmp_path / "local"
    img = volume.array_to_image(vol, meta)
    nii = tmp_path / "s.nii.gz"
    volume.save_nifti(img, nii)
    w = storage.TarShardWriter(remote, 0)
    w.add_file("synthetic", "synthetic.nii.gz", nii)
    w.add_bytes("synthetic", "synthetic.meta.json", json.dumps(meta).encode())
    w.close()
    assert storage.prepare_local(remote, local) == ["shard_000.tar"]
    assert storage.prepare_local(remote, local) == []
    store = storage.VolumeStore(local / "cache")
    assert store.uids() == ["synthetic"]
    v2, m2 = store.load_volume("synthetic")
    np.testing.assert_array_equal(v2, vol)
    assert m2["slice_labels"] == meta["slice_labels"]


def test_failure_log(tmp_path):
    log = storage.FailureLog(tmp_path / "f.csv")
    log.add("u1", "download", "timeout")
    log2 = storage.FailureLog(tmp_path / "f.csv")
    assert log2.uids() == {"u1"} and len(log2) == 1


def test_atomic_write_npy(tmp_path):
    arr = np.arange(6, dtype=np.float32)
    storage.atomic_write_npy(tmp_path / "a.npy", arr)
    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), arr)
    assert list(tmp_path.glob("*.tmp")) == []
