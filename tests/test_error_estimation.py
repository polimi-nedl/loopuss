"""Worst-case error bound (errbound) test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, CurrentSignal, LoopUS, relerr


@pytest.fixture
def err_data():
    return load_fixture("error_estimation")


def _build(D):
    A = D["A"]
    G0 = float(D["G0"])
    N = int(D["N"])
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0))
    return circ, A, G0, float(D["V0"])


def test_errbound_matches_reference(err_data):
    D = err_data
    circ, _, _, _ = _build(D)
    maxerr, y, minerr, z = circ.errbound()

    np.testing.assert_allclose(maxerr, float(D["maxerr"]), rtol=1e-6)

    # The maximising input is defined up to sign; align before comparing.
    yf = D["y"].ravel()
    s = np.sign(np.dot(y, yf)) or 1.0
    np.testing.assert_allclose(s * y, yf, rtol=1e-5, atol=1e-8)

    # Best-case error is bounded below by zero.
    assert 0.0 <= minerr <= maxerr
    assert np.isclose(np.linalg.norm(z), 1.0)


def test_errbound_input_realises_bound(err_data):
    """Applying the worst-case input achieves the predicted relative error."""
    D = err_data
    circ, A, G0, V0 = _build(D)
    maxerr, y, _, _ = circ.errbound()

    yf = D["y"].ravel()
    s = np.sign(np.dot(y, yf)) or 1.0
    y = s * y
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0 * V0))

    time = np.concatenate([[0.0], np.logspace(np.log10(np.finfo(float).eps), -4, 50)])
    Vmodel, _ = circ.tran("model", time)
    np.testing.assert_allclose(Vmodel, D["Vmodel"], rtol=1e-5, atol=1e-6)

    ref = -np.linalg.solve(A, y) * V0
    np.testing.assert_allclose(relerr(Vmodel[:, -1], ref), maxerr, rtol=1e-3)


def test_errbound_rejects_unknown_error_types():
    c = LoopUS(None, CrossBar(np.eye(3) * 1e-4))
    with pytest.raises(ValueError, match="error_type"):
        c.errbound("statik")
    c.errbound("Static")
    c.errbound("DYNAMIC")
