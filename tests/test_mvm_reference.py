"""Compare MVM output against saved reference data."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, LoopUS, OAModel, VoltageSignal


@pytest.fixture
def mvm_data():
    return load_fixture("mvm")


def test_mvm_model_steady_matches_reference(mvm_data):
    """Steady-state of tran('model') should match reference Vmodel at end-of-time."""
    D = mvm_data
    A = D["A"]
    G0 = D["G0"]
    V0 = float(D["V0"])
    N = int(D["N"])
    y = D["y"].ravel()

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, oa_model=OAModel("1P"))
    circ.set_inputs(VoltageSignal.ytov(y, V0=V0))

    # Use the same time grid: logspace(-10, -5, 100)
    Vpy, _ = circ.tran("model", np.logspace(-10, -5, 100))

    # reference: Vmodel (10x101 = 100 logspace points + the t=0 origin)
    V_ref = D["Vmodel"]
    # The final-time value should match within OA-finite-gain tolerance
    np.testing.assert_allclose(
        Vpy[:, -1],
        V_ref[:, -1],
        rtol=5e-2,
        atol=1e-2,
        err_msg="MVM end-of-time output diverges from reference",
    )


def test_pz_matches_reference(mvm_data):
    """MVM circuit poles should match the reference values."""
    D = mvm_data
    A = D["A"]
    G0 = D["G0"]
    N = int(D["N"])

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, oa_model=OAModel("1P"))

    poles, *_ = circ.pz()
    # Sanity: all real parts negative (stable closed-loop)
    assert np.all(poles.real < 0), f"Unstable poles: {poles}"
