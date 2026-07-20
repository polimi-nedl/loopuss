"""Matrix-vector multiplication test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, LoopUS, OAModel, VoltageSignal


@pytest.fixture
def mvm_data():
    return load_fixture("mvm")


def test_mvm_steady_state(mvm_data):
    """At t->∞, MVM circuit converges to -A @ y (with finite-gain error)."""
    D = mvm_data
    A = D["A"]
    G0 = D["G0"]
    V0 = float(D["V0"])
    N = int(D["N"])
    y = D["y"].ravel()

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, oa_model=OAModel("1P"))

    sigs = VoltageSignal.ytov(y, V0=V0)
    circ.set_inputs(sigs)

    Vmodel, tmodel = circ.tran("model", np.logspace(-10, -5, 100))

    expected = -A @ y * V0  # ideal output
    # With finite OA gain (A0=1e4), expect small but bounded error
    final = Vmodel[:, -1]
    np.testing.assert_allclose(
        final, expected, rtol=1e-2, atol=5e-3, err_msg="MVM steady state mismatch"
    )


def test_mvm_initial_zero(mvm_data):
    """At t=0, all outputs should be ≈ 0 (initial condition)."""
    D = mvm_data
    A = D["A"]
    G0 = D["G0"]
    V0 = float(D["V0"])
    N = int(D["N"])
    y = D["y"].ravel()

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, oa_model=OAModel("1P"))

    sigs = VoltageSignal.ytov(y, V0=V0)
    circ.set_inputs(sigs)

    Vmodel, _ = circ.tran("model", np.logspace(-10, -5, 100))
    np.testing.assert_allclose(Vmodel[:, 0], 0.0, atol=1e-9)
