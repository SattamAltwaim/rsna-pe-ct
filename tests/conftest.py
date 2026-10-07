import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIXTURES = REPO / "tests" / "fixtures"


@pytest.fixture(scope="session")
def tiny_train_csv() -> pd.DataFrame:
    """A synthetic train.csv with 30 studies: 18 negative, 9 PE, 3 indeterminate."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(30):
        n = int(rng.integers(5, 12))
        if s < 18:
            group = "neg"
        elif s < 27:
            group = "pe"
        else:
            group = "ind"
        pos = np.zeros(n, dtype=int)
        if group == "pe":
            start = int(rng.integers(0, n - 2))
            pos[start: start + int(rng.integers(1, 3))] = 1
        central = int(group == "pe" and s % 2 == 0)
        rv = int(group == "pe" and s % 3 == 0)
        for i in range(n):
            rows.append(
                {
                    "StudyInstanceUID": f"study{s:03d}",
                    "SeriesInstanceUID": f"series{s:03d}",
                    "SOPInstanceUID": f"sop{s:03d}_{i:03d}",
                    "pe_present_on_image": int(pos[i]),
                    "negative_exam_for_pe": int(group == "neg"),
                    "qa_motion": int(group == "ind" and s % 2 == 0),
                    "qa_contrast": int(group == "ind" and s % 2 == 1),
                    "flow_artifact": int(s % 7 == 0),
                    "rv_lv_ratio_gte_1": rv,
                    "rv_lv_ratio_lt_1": int(group == "pe" and not rv),
                    "leftsided_pe": int(group == "pe"),
                    "chronic_pe": 0,
                    "true_filling_defect_not_pe": int(s % 11 == 0),
                    "rightsided_pe": int(group == "pe" and s % 2 == 1),
                    "acute_and_chronic_pe": 0,
                    "central_pe": central,
                    "indeterminate": int(group == "ind"),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_volume():
    """Synthetic (z, y, x) HU volume: air background, a soft-tissue box, a bright 'vessel'."""
    vol = np.full((20, 32, 32), -1000, dtype=np.int16)
    vol[2:18, 6:26, 6:26] = 40
    vol[:, 14:18, 14:18] = 300
    meta = {
        "study_uid": "synthetic",
        "series_uid": "synthetic",
        "orientation": "LPI",
        "shape_zyx": [20, 32, 32],
        "spacing_xyz_mm": [0.8, 0.8, 2.0],
        "origin_xyz_mm": [0.0, 0.0, 0.0],
        "direction": [1, 0, 0, 0, 1, 0, 0, 0, -1],
        "z_positions_mm": [float(-2.0 * z) for z in range(20)],
        "sop_uids": [f"sop{z:03d}" for z in range(20)],
        "manufacturer": "unknown",
        "slice_labels": [0] * 8 + [1] * 4 + [0] * 8,
    }
    return vol, meta
