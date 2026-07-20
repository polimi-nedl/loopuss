"""Pole-zero analysis test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, LoopUS, OAModel


@pytest.fixture
def pz_data():
    return load_fixture("pole_zero_analysis")


def _sorted_real(p: np.ndarray) -> np.ndarray:
    """Sort complex poles by real part descending (matches LoopUS pz convention)."""
    return p[np.argsort(p.real)[::-1]]


def test_pz_symmetric_pd(pz_data):
    """Symmetric PD matrix -> all real, LHP poles."""
    D = pz_data
    A = D["A"]
    G0 = D["G0"]

    circ = LoopUS(None, CrossBar(A * G0), oa_model=OAModel("1P"))
    poles, *_ = circ.pz()

    # All poles should be in the LHP (real part < 0)
    assert np.all(poles.real < 0), f"Found non-LHP pole: {poles}"

    # Compare to reference (sort both by real-part desc)
    ref = D["poles1"].ravel()
    py_sorted = _sorted_real(poles)
    ref_sorted = _sorted_real(ref.astype(complex))
    np.testing.assert_allclose(
        py_sorted.real, ref_sorted.real, rtol=1e-3, err_msg="Symmetric PD poles mismatch"
    )


def test_pz_nonsymmetric(pz_data):
    """Nonsymmetric matrix -> may have complex (split) poles."""
    D = pz_data
    B = D["B"]
    G0 = D["G0"]

    circ = LoopUS(None, CrossBar(B * G0), oa_model=OAModel("1P"))
    poles, *_ = circ.pz()

    ref = D["poles2"].ravel()
    py_sorted = _sorted_real(poles)
    ref_sorted = _sorted_real(ref.astype(complex))
    np.testing.assert_allclose(
        py_sorted.real, ref_sorted.real, rtol=1e-3, err_msg="Nonsymmetric pole real-parts mismatch"
    )


def test_pz_symmetric_not_pd(pz_data):
    """Symmetric non-PD matrix -> real poles, possibly RHP."""
    D = pz_data
    C = D["C"]
    G0 = D["G0"]

    circ = LoopUS(None, CrossBar(C * G0), oa_model=OAModel("1P"))
    poles, *_ = circ.pz()

    ref = D["poles3"].ravel()
    py_sorted = _sorted_real(poles)
    ref_sorted = _sorted_real(ref.astype(complex))
    np.testing.assert_allclose(
        py_sorted.real,
        ref_sorted.real,
        rtol=1e-3,
        err_msg="Symmetric non-PD pole real-parts mismatch",
    )
