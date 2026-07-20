"""Model-vs-SPICE tests against the model_vs_SPICE reference fixture.

The analytical model is checked against the saved reference model output; the
SPICE co-simulation (when LTSpice is available) is checked against the saved
LTSpice output. SPICE comparisons use the final settled value since LTSpice's
adaptive time grid differs from the reference grid.
"""

import numpy as np
import pytest
from conftest import load_fixture

import loopuss as lp
from loopuss import spice


@pytest.fixture
def mvs_data():
    return load_fixture("model_vs_SPICE")


def _build(D):
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    Gf = lp.CrossBar(A * G0)
    Gin = lp.CrossBar(np.empty((N, 0)))
    circ = lp.LoopUS(Gin, Gf, oa_model=lp.OAModel("1P"))
    circ.set_inputs(lp.CurrentSignal.ytoi(D["y"].ravel(), I0))
    return circ


def test_model_final_matches_reference(mvs_data):
    D = mvs_data
    circ = _build(D)
    V, t = circ.tran("model", np.logspace(-10, np.log10(10e-6), 400))
    np.testing.assert_allclose(V[:, -1], D["Vmod"][:, -1], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not spice.spice_available(), reason="LTSpice not installed")
def test_spice_fast_matches_reference(mvs_data):
    D = mvs_data
    circ = _build(D)
    V, t = circ.tran("SPICE/fast", 10e-6)
    np.testing.assert_allclose(V[:, -1], D["Vfas"][:, -1], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not spice.spice_available(), reason="LTSpice not installed")
def test_spice_accurate_matches_reference(mvs_data):
    D = mvs_data
    circ = _build(D)
    V, t = circ.tran("SPICE/accurate", 10e-6)
    np.testing.assert_allclose(V[:, -1], D["Vacc"][:, -1], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not spice.spice_available(), reason="LTSpice not installed")
def test_spice_matches_model(mvs_data):
    """SPICE and model final outputs should agree to ~6 digits."""
    D = mvs_data
    circ = _build(D)
    Vm, _ = circ.tran("model", np.logspace(-10, np.log10(10e-6), 400))
    Vs, _ = circ.tran("SPICE/fast", 10e-6)
    np.testing.assert_allclose(Vs[:, -1], Vm[:, -1], rtol=1e-4, atol=1e-5)
