"""Time-to-solution (tts) test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, CurrentSignal, LoopUS


@pytest.fixture
def tts_data():
    return load_fixture("time_to_solution")


def _build(D):
    A = D["A"]
    G0 = float(D["G0"])
    N = int(D["N"])
    V0 = float(D["V0"])
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0))
    circ.set_inputs(CurrentSignal.ytoi(D["y"].ravel(), I0=G0 * V0))
    return circ


def test_tts_bounds_ordering(tts_data):
    """tts sits between its lazy (min) and greedy (max) bounds."""
    tts, tts_min, tts_max = _build(tts_data).tts()
    assert tts_min <= tts <= tts_max
    assert np.isclose(tts, 0.5 * (tts_min + tts_max))


def test_tts_transient_matches_reference(tts_data):
    """The transient on a tts-derived grid reproduces the reference."""
    D = tts_data
    circ = _build(D)
    tts, *_ = circ.tts()
    time = np.logspace(-15, np.log10(5 * tts), 1000)
    Vmodel, tmodel = circ.tran("model", time)

    np.testing.assert_allclose(tmodel, D["tmodel"].ravel(), rtol=1e-6, atol=1e-20)
    np.testing.assert_allclose(Vmodel, D["Vmodel"], rtol=1e-6, atol=1e-9)
