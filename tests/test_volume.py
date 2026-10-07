import tarfile

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from pe_ct import labels, volume
from tests.conftest import FIXTURES

UID = "bc855cd8bdc9"


@pytest.fixture(scope="module")
def real_series_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("real")
    with tarfile.open(FIXTURES / f"{UID}_8slices.tar.gz") as tar:
        tar.extractall(out, filter="data")
    return out / UID


def _write_synthetic_series(dirpath, n=6, z_step=2.0, ascending=True, value_fn=None):
    """Axial series with identity orientation; slice k at z = k*z_step (or descending)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    series_uid, study_uid = generate_uid(), generate_uid()
    sops = []
    for k in range(n):
        z = k * z_step if ascending else -k * z_step
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(dirpath / f"{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.Modality = "CT"
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [0.5, 0.5]
        ds.SliceThickness = z_step
        ds.Rows = ds.Columns = 16
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept, ds.RescaleSlope = -1024.0, 1.0
        ds.Manufacturer = "TestCo"
        ds.KVP = 120
        px = np.full((16, 16), 24, dtype=np.int16)  # 24 + (-1024) = -1000 HU (air)
        if value_fn is not None:
            px = value_fn(k, px)
        ds.PixelData = px.tobytes()
        ds.save_as(dirpath / f"{k}.dcm", enforce_file_format=True)
        sops.append((ds.SOPInstanceUID, z))
    return sops


def test_real_fixture_reads_as_head_first_hu_volume(real_series_dir):
    img, sops, header = volume.read_series(real_series_dir)
    vol = volume.image_to_hu_array(img)
    assert vol.shape == (8, 512, 512)
    assert len(sops) == 8 and len(set(sops)) == 8
    meta = volume.build_meta(img, sops, header, UID, "bab8e5153e7b")
    assert meta["orientation"] == "LPI" and meta["direction"][8] == -1.0
    z = meta["z_positions_mm"]
    assert all(z[i] > z[i + 1] for i in range(7)), "index 0 must be the head end"
    assert meta["slice_spacing_median_mm"] == pytest.approx(5.0, abs=0.01)
    assert meta["n_missing_slices_est"] == 0
    assert header["kvp"] == 120.0 and header["manufacturer"] == "unknown"
    assert header["scanner"] == "kernel B30f" and header["transfer_syntax"] == "1.2.840.10008.1.2.1"
    sanity = volume.hu_sanity(vol)
    assert -1100 < sanity["min"] <= -1000 and sanity["frac_air"] > 0.2 and sanity["max"] > 300
    # sop uid in the file name matches the tag SimpleITK reported
    files = {pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID: p.stem for p in real_series_dir.glob("*.dcm")}
    assert all(files[s] == s for s in sops)


def test_real_fixture_label_alignment(real_series_dir):
    img, sops, _ = volume.read_series(real_series_dir)
    lab = pd.read_csv(FIXTURES / f"{UID}_labels.csv", dtype={"SOPInstanceUID": str})
    series = pd.Series(lab.pe_present_on_image.to_numpy(np.int8), index=lab.SOPInstanceUID)
    aligned = labels.align_slice_labels(sops, series)
    assert aligned.sum() == 3
    assert labels.positive_runs(aligned) == [(1, 4)]  # instances 22-24 of 21-28 -> indices 1,2,3


@pytest.mark.parametrize("ascending", [True, False])
def test_synthetic_series_orientation_is_independent_of_file_order(tmp_path, ascending):
    def value_fn(k, px):
        px = px.copy()
        px[k, :] = 1024 + 500  # bright row index k -> HU 500 (encodes slice identity)
        return px

    sops = _write_synthetic_series(tmp_path / "s", n=6, ascending=ascending, value_fn=value_fn)
    img, sop_order, header = volume.read_series(tmp_path / "s")
    vol = volume.image_to_hu_array(img)
    meta = volume.build_meta(img, sop_order, header, "s", "se")
    z = meta["z_positions_mm"]
    assert all(z[i] > z[i + 1] for i in range(5))
    by_sop = {s: zz for s, zz in sops}
    assert [by_sop[s] for s in sop_order] == sorted(by_sop.values(), reverse=True)
    # slice z of the volume must be the file whose bright row encodes it
    for zi, s in enumerate(sop_order):
        k = [i for i, (ss, _) in enumerate(sops) if ss == s][0]
        assert vol[zi, k, :].max() == 500 and vol[zi].max() == 500
    assert header["manufacturer"] == "TestCo" and header["scanner"] == "TestCo"
    assert meta["slice_spacing_median_mm"] == pytest.approx(2.0)


def test_missing_slice_is_detected(tmp_path):
    sops = _write_synthetic_series(tmp_path / "s", n=7)
    (tmp_path / "s" / "3.dcm").unlink()
    _, _, header = volume.read_series(tmp_path / "s")
    assert header["n_missing_slices_est"] == 1
    assert header["slice_gap_max_mm"] == pytest.approx(4.0)


def test_ras_conversion_and_roundtrip(tiny_volume, tmp_path):
    vol, meta = tiny_volume
    img = volume.array_to_image(vol, meta)
    ras = volume.to_ras_tensor_array(img)
    nz, ny, nx = vol.shape
    assert ras.shape == (nx, ny, nz)
    rng = np.random.default_rng(0)
    for _ in range(20):
        z, y, x = rng.integers(0, nz), rng.integers(0, ny), rng.integers(0, nx)
        assert ras[nx - 1 - x, ny - 1 - y, nz - 1 - z] == vol[z, y, x]
    assert volume.ras_to_lpi_z(np.array([0, nz - 1]), nz).tolist() == [nz - 1, 0]
    path = volume.save_nifti(img, tmp_path / "v.nii.gz")
    np.testing.assert_array_equal(volume.load_nifti(path), vol)
    assert np.allclose(volume.load_nifti_image(path).GetDirection(), meta["direction"])


def test_scanner_proxy():
    assert volume.scanner_proxy("SIEMENS", "B30f") == "SIEMENS"
    assert volume.scanner_proxy("unknown", "STANDARD") == "kernel STANDARD"
    assert volume.scanner_proxy("", "") == "unknown"


def test_view_helpers(tiny_volume):
    vol, meta = tiny_volume
    assert volume.coronal_slab(vol).shape == (20, 32)
    assert volume.sagittal_slab(vol).shape == (20, 32)
    assert volume.coronal_mip(vol).max() == 300
    assert volume.aspect_coronal(meta) == pytest.approx(2.5)
    assert volume.normalized_z([0, 19], 20).tolist() == [0.0, 1.0]


def test_read_series_headers_matches_full_read(real_series_dir):
    img, sops, h = volume.read_series(real_series_dir)
    img2, sops2, h2 = volume.read_series_headers(real_series_dir)
    assert sops == sops2
    m, m2 = volume.build_meta(img, sops, h, UID, "s"), volume.build_meta(img2, sops2, h2, UID, "s")
    assert m == m2
    assert img2.GetSize() == img.GetSize() and img2.GetSpacing() == img.GetSpacing()
