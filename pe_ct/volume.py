"""DICOM series -> oriented HU volume with slice labels aligned to the z-axis.

Conventions used everywhere in this project:

* A stored volume is an ``int16`` array ``(z, y, x)`` in Hounsfield Units.
* Axes are oriented **LPI**: x grows toward the patient's Left, y toward
  Posterior, z toward Inferior. So ``vol[0]`` is the top (head end) of the
  scan, and ``plt.imshow(vol[z])`` is the standard radiological axial view
  (anterior at the top, patient's left on the right of the screen).
* ``meta["sop_uids"][z]`` is the DICOM file that produced slice ``z``, and
  ``meta["slice_labels"][z]`` its ``pe_present_on_image`` label.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk

STORED_ORIENTATION = "LPI"

# The dataset's anonymised UIDs are short hex strings, which pydicom flags on every read.
warnings.filterwarnings("ignore", message="Invalid value for VR UI")

DICOM_TAGS = {
    "manufacturer": "0008|0070",
    "model_name": "0008|1090",
    "slice_thickness_mm": "0018|0050",
    "kvp": "0018|0060",
    "convolution_kernel": "0018|1210",
    "patient_position": "0018|5100",
    "study_description": "0008|1030",
}


def _tag(reader: sitk.ImageSeriesReader, slice_index: int, tag: str, default=""):
    try:
        if reader.HasMetaDataKey(slice_index, tag):
            return reader.GetMetaData(slice_index, tag).strip()
    except RuntimeError:
        pass
    return default


def _to_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_series(dcm_dir) -> tuple:
    """Read one DICOM series directory into ``(sitk.Image in LPI, sop_uids_in_z_order, header)``.

    SimpleITK (GDCM) sorts slices by position, applies RescaleSlope/Intercept
    and builds the 3D geometry. We then reorient to LPI and recover which file
    ended up at which z index by matching physical slice positions.

    SimpleITK assumes uniform slice spacing; when slices are missing it keeps
    the stack as-is and averages the spacing. We therefore also return the true
    per-slice positions read from each file (``header["z_positions_true_mm"]``)
    and gap statistics so such studies can be flagged.
    """
    dcm_dir = str(dcm_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
    if len(series_ids) == 0:
        raise IOError(f"no DICOM series found in {dcm_dir}")
    if len(series_ids) > 1:
        raise IOError(f"{len(series_ids)} series in one directory; expected 1")
    file_names = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    reader.SetGlobalWarningDisplay(False)  # the non-uniform-spacing warning is handled below
    raw = reader.Execute()

    n = raw.GetSize()[2]
    sop_raw = [_tag(reader, i, "0008|0018") for i in range(n)]
    if any(not s for s in sop_raw):
        raise IOError("a slice is missing its SOPInstanceUID")
    z_raw = np.array([raw.TransformIndexToPhysicalPoint((0, 0, i))[2] for i in range(n)])
    z_true_raw = np.array([_position_z(_tag(reader, i, "0020|0032"), z_raw[i]) for i in range(n)])

    oriented = sitk.DICOMOrient(raw, STORED_ORIENTATION)
    if oriented.GetSize()[2] != n:
        raise IOError("series is not axial: slice axis changed under reorientation")
    z_oriented = np.array([oriented.TransformIndexToPhysicalPoint((0, 0, j))[2] for j in range(n)])
    order = np.abs(z_oriented[:, None] - z_raw[None, :]).argmin(axis=1)
    if len(set(order.tolist())) != n or np.abs(z_oriented - z_raw[order]).max() > 0.01:
        raise IOError("could not match slices to files after reorientation")
    sop_uids = [sop_raw[i] for i in order]
    z_true = z_true_raw[order]

    header = {name: _tag(reader, 0, tag) for name, tag in DICOM_TAGS.items()}
    header["slice_thickness_mm"] = _to_float(header["slice_thickness_mm"])
    header["kvp"] = _to_float(header["kvp"])
    header["manufacturer"] = header["manufacturer"] or "unknown"
    header["n_files"] = len(file_names)
    header["z_positions_true_mm"] = [float(z) for z in z_true]
    header.update(spacing_stats(z_true))
    return oriented, sop_uids, header


def _position_z(image_position: str, fallback: float) -> float:
    """``ImagePositionPatient`` is stored as ``"x\\y\\z"``; return z."""
    try:
        return float(image_position.split("\\")[2])
    except (IndexError, ValueError, AttributeError):
        return float(fallback)


def spacing_stats(z_true) -> dict:
    """Median slice step, largest step, and an estimate of how many slices are missing."""
    z_true = np.asarray(z_true, dtype=float)
    if len(z_true) < 2:
        return {"slice_spacing_median_mm": np.nan, "slice_gap_max_mm": np.nan, "n_missing_slices_est": 0}
    steps = np.abs(np.diff(z_true))
    median = float(np.median(steps))
    missing = int(round(float(steps.sum() / median) - len(steps))) if median > 0 else 0
    return {
        "slice_spacing_median_mm": round(median, 4),
        "slice_gap_max_mm": round(float(steps.max()), 4),
        "n_missing_slices_est": max(0, missing),
    }


def image_to_hu_array(image: sitk.Image) -> np.ndarray:
    """``int16 (z, y, x)`` HU array (SimpleITK already applied slope/intercept)."""
    arr = sitk.GetArrayFromImage(image)
    return np.clip(np.rint(arr), -32768, 32767).astype(np.int16)


def array_to_image(vol: np.ndarray, meta: dict) -> sitk.Image:
    """Rebuild a geometrically correct SimpleITK image from a stored array + meta."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol))
    img.SetSpacing(tuple(float(s) for s in meta["spacing_xyz_mm"]))
    img.SetOrigin(tuple(float(o) for o in meta["origin_xyz_mm"]))
    img.SetDirection(tuple(float(d) for d in meta["direction"]))
    return img


def build_meta(image: sitk.Image, sop_uids: list, header: dict, study_uid: str, series_uid: str) -> dict:
    """Everything a notebook needs to interpret the stored volume."""
    size = image.GetSize()  # (x, y, z)
    spacing = image.GetSpacing()
    meta = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "orientation": STORED_ORIENTATION,
        "shape_zyx": [int(size[2]), int(size[1]), int(size[0])],
        "spacing_xyz_mm": [float(s) for s in spacing],
        "origin_xyz_mm": [float(o) for o in image.GetOrigin()],
        "direction": [float(d) for d in image.GetDirection()],
        "sop_uids": list(sop_uids),
    }
    meta.update(header)
    # true per-slice positions (from the files) take precedence over the uniform assumption
    meta["z_positions_mm"] = header.get(
        "z_positions_true_mm",
        [float(image.TransformIndexToPhysicalPoint((0, 0, j))[2]) for j in range(size[2])],
    )
    meta.pop("z_positions_true_mm", None)
    return meta


def hu_sanity(vol: np.ndarray) -> dict:
    """Quick calibration check: lots of air at about -1000 HU, nothing absurd."""
    flat = vol.reshape(-1)
    sample = flat[:: max(1, flat.size // 2_000_000)]
    return {
        "min": int(sample.min()),
        "p01": int(np.percentile(sample, 1)),
        "median": int(np.median(sample)),
        "p99": int(np.percentile(sample, 99)),
        "max": int(sample.max()),
        "frac_air": float(((sample > -1100) & (sample < -900)).mean()),
        "frac_soft_tissue": float(((sample > -100) & (sample < 100)).mean()),
        "frac_contrast_or_bone": float((sample > 150).mean()),
    }


def save_nifti(image: sitk.Image, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(image, sitk.sitkInt16), str(path), useCompression=True)
    return path


def load_nifti(path) -> np.ndarray:
    return image_to_hu_array(sitk.ReadImage(str(path)))


def load_nifti_image(path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write_meta(meta: dict, path) -> Path:
    from pe_ct.storage import atomic_write_json

    return atomic_write_json(path, meta)


def read_meta(path) -> dict:
    return json.loads(Path(path).read_text())


def coronal_slab(vol: np.ndarray, y: int | None = None) -> np.ndarray:
    """2D coronal view ``(z, x)`` through row ``y`` (default: middle). Head at the top."""
    y = vol.shape[1] // 2 if y is None else y
    return vol[:, y, :]


def sagittal_slab(vol: np.ndarray, x: int | None = None) -> np.ndarray:
    """2D sagittal view ``(z, y)`` through column ``x`` (default: middle). Head at the top."""
    x = vol.shape[2] // 2 if x is None else x
    return vol[:, :, x]


def coronal_mip(vol: np.ndarray, y_lo: int | None = None, y_hi: int | None = None) -> np.ndarray:
    """Maximum-intensity projection over a slab of rows: vessels pop out."""
    y_lo = 0 if y_lo is None else y_lo
    y_hi = vol.shape[1] if y_hi is None else y_hi
    return vol[:, y_lo:y_hi, :].max(axis=1)


def aspect_coronal(meta: dict) -> float:
    """``imshow(aspect=...)`` so that mm are square in a (z, x) view."""
    sx, _, sz = meta["spacing_xyz_mm"]
    return sz / sx


def aspect_sagittal(meta: dict) -> float:
    _, sy, sz = meta["spacing_xyz_mm"]
    return sz / sy


def normalized_z(z_index, n_slices: int) -> np.ndarray:
    """Slice index -> position in [0, 1], 0 = top of scan (head end), 1 = bottom."""
    return np.asarray(z_index, dtype=float) / max(1, n_slices - 1)


def to_ras_tensor_array(image: sitk.Image) -> np.ndarray:
    """Reorient a SimpleITK image to RAS and return a ``(R, A, S)``-ordered HU array.

    This is the layout SPECTRE expects: axis 0 grows toward Right, axis 1 toward
    Anterior, axis 2 toward Superior (nibabel's canonical orientation).
    """
    ras = sitk.DICOMOrient(image, "RAS")
    arr = sitk.GetArrayFromImage(ras)  # (S, A, R)
    return np.ascontiguousarray(arr.transpose(2, 1, 0))


def ras_to_lpi_z(ras_k, n_slices: int):
    """RAS superior-axis index -> stored (head-first) slice index. Works on ranges too."""
    return (n_slices - 1) - np.asarray(ras_k)
