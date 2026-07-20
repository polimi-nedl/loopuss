"""Test fixtures shared across loopuss tests."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(test_name: str) -> dict:
    """Load a ``data.mat`` reference fixture, scalarizing 1x1 arrays."""
    mat = loadmat(str(FIXTURE_DIR / test_name / "data.mat"))
    out = {}
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.shape == (1, 1):
            out[k] = float(v.flat[0]) if v.dtype.kind in "fui" else v.flat[0]
        else:
            out[k] = v
    return out
