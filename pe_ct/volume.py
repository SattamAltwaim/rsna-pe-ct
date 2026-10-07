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
    "manufacturer": "Manufacturer",
    "model_name": "ManufacturerModelName",
    "slice_thickness_mm": "SliceThickness",
    "kvp": "KVP",
    "convolution_kernel": "ConvolutionKernel",
    "patient_position": "PatientPosition",
    "study_description": "StudyDescription",
}


def _to_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _header_value(ds, keyword: str, default=""):
    value = ds.get(keyword, default)
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return " ".join(str(v) for v in value)
    return str(value).strip()


def _read_sorted_datasets(dcm_dir, stop_before_pixels: bool) -> tuple:
    """All slices of one series sorted by position along the slice normal, plus geometry.

    Returns ``(datasets, positions, along, geometry)`` where ``geometry`` holds the
    in-plane spacing, the slice step and the direction matrix (columns = image axes in LPS).
    """
    import pydicom

    files = sorted(Path(dcm_dir).glob("*.dcm"))
    if not files:
        raise IOError(f"no DICOM files in {dcm_dir}")
    datasets = [pydicom.dcmread(str(f), stop_before_pixels=stop_before_pixels) for f in files]

    series = {str(ds.get("SeriesInstanceUID", "")) for ds in datasets}
    if len(series) != 1:
        raise IOError(f"{len(series)} series in one directory; expected 1")
    for ds in datasets:
        if "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
            raise IOError("a slice lacks ImagePositionPatient/ImageOrientationPatient")

    iop = np.array(datasets[0].ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    row_dir /= np.linalg.norm(row_dir)
    normal = np.cross(row_dir, col_dir)
    normal /= np.linalg.norm(normal)
    col_dir = np.cross(normal, row_dir)
    for ds in datasets[1:]:
        if np.abs(np.array(ds.ImageOrientationPatient, dtype=float) - iop).max() > 1e-3:
            raise IOError("slices have different orientations")

    positions = np.array([ds.ImagePositionPatient for ds in datasets], dtype=float)
    along = positions @ normal
    order = np.argsort(along, kind="stable")
    datasets = [datasets[i] for i in order]
    positions, along = positions[order], along[order]
    shapes = {(int(ds.Rows), int(ds.Columns)) for ds in datasets}
    if len(shapes) != 1:
        raise IOError(f"slices have different shapes: {shapes}")
    sop_uids = [str(ds.SOPInstanceUID) for ds in datasets]
    if len(set(sop_uids)) != len(sop_uids):
        raise IOError("duplicate SOPInstanceUIDs in series")

    row_spacing, col_spacing = (float(v) for v in datasets[0].PixelSpacing)  # (y, x)
    steps = np.diff(along)
    dz = float(np.median(np.abs(steps))) if len(steps) else _to_float(datasets[0].get("SliceThickness", 1.0), 1.0)
    if not np.isfinite(dz) or dz <= 0:
        raise IOError("could not determine slice spacing")
    geometry = {
        "rows": int(datasets[0].Rows),
        "cols": int(datasets[0].Columns),
        "spacing": (col_spacing, row_spacing, dz),
        "origin": tuple(float(v) for v in positions[0]),
        "direction": tuple(float(v) for v in np.stack([row_dir, col_dir, normal], axis=1).reshape(-1)),
    }
    return datasets, positions, along, geometry


def _orient_and_match(raw: sitk.Image, datasets: list, positions: np.ndarray, along: np.ndarray) -> tuple:
    """Reorient ``raw`` to LPI and work out which file landed at which z index."""
    n = len(datasets)
    sop_raw = [str(ds.SOPInstanceUID) for ds in datasets]
    z_raw = np.array([raw.TransformIndexToPhysicalPoint((0, 0, i))[2] for i in range(n)])

    oriented = sitk.DICOMOrient(raw, STORED_ORIENTATION)
    if oriented.GetSize()[2] != n:
        raise IOError("series is not axial: slice axis changed under reorientation")
    z_oriented = np.array([oriented.TransformIndexToPhysicalPoint((0, 0, j))[2] for j in range(n)])
    match = np.abs(z_oriented[:, None] - z_raw[None, :]).argmin(axis=1)
    if len(set(match.tolist())) != n or np.abs(z_oriented - z_raw[match]).max() > 0.01:
        raise IOError("could not match slices to files after reorientation")
    sop_uids = [sop_raw[i] for i in match]
    z_true = positions[match, 2]

    first = datasets[0]
    header = {name: _header_value(first, keyword) for name, keyword in DICOM_TAGS.items()}
    header["slice_thickness_mm"] = _to_float(header["slice_thickness_mm"])
    header["kvp"] = _to_float(header["kvp"])
    header["manufacturer"] = header["manufacturer"] or "unknown"
    header["scanner"] = scanner_proxy(header["manufacturer"], header["convolution_kernel"])
    header["transfer_syntax"] = str(getattr(first.file_meta, "TransferSyntaxUID", ""))
    header["n_files"] = n
    header["z_positions_true_mm"] = [float(z) for z in z_true]
    header.update(spacing_stats(along))
    return oriented, sop_uids, header


def _apply_geometry(image: sitk.Image, geometry: dict) -> sitk.Image:
    image.SetSpacing(geometry["spacing"])
    image.SetOrigin(geometry["origin"])
    image.SetDirection(geometry["direction"])
    return image


def read_series(dcm_dir) -> tuple:
    """Read one DICOM series directory into ``(sitk.Image in LPI, sop_uids_in_z_order, header)``.

    Every slice is read with pydicom (GDCM's series scanner rejects this dataset's
    anonymised, non-standard UIDs on some scanners). Slices are sorted by their
    position along the slice normal, intensities are mapped to HU with each file's
    RescaleSlope/Intercept, and the 3D geometry (spacing, origin, direction) is
    built from the headers. The volume is then reoriented to LPI with SimpleITK
    and we record which file ended up at which z index.

    The true per-slice positions and gap statistics are returned in the header so
    studies with missing slices can be flagged (the array is simply the stack of
    the files that exist; nothing is interpolated).
    """
    datasets, positions, along, geometry = _read_sorted_datasets(dcm_dir, stop_before_pixels=False)
    slices = []
    for ds in datasets:
        slope = _to_float(ds.get("RescaleSlope", 1.0), 1.0)
        intercept = _to_float(ds.get("RescaleIntercept", 0.0), 0.0)
        slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)
    raw = _apply_geometry(sitk.GetImageFromArray(np.stack(slices).astype(np.float32)), geometry)
    return _orient_and_match(raw, datasets, positions, along)


def read_series_headers(dcm_dir) -> tuple:
    """Like :func:`read_series` but without decoding pixels: ``(geometry-only image, sop_uids, header)``.

    Roughly ten times faster; the returned image has the right size, spacing, origin and
    direction (its voxels are zeros), so :func:`build_meta` works on it unchanged.
    """
    datasets, positions, along, geometry = _read_sorted_datasets(dcm_dir, stop_before_pixels=True)
    raw = _apply_geometry(sitk.Image([geometry["cols"], geometry["rows"], len(datasets)], sitk.sitkUInt8), geometry)
    return _orient_and_match(raw, datasets, positions, along)


def scanner_proxy(manufacturer: str, convolution_kernel: str) -> str:
    """Manufacturer if present, else the reconstruction kernel name (kernel names are
    vendor-specific, e.g. ``B30f`` is Siemens, ``STANDARD`` is GE), else ``unknown``.
    The Manufacturer tag is missing from almost every study in this dataset."""
    if manufacturer and manufacturer != "unknown":
        return manufacturer
    if convolution_kernel:
        return f"kernel {convolution_kernel}"
    return "unknown"


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
