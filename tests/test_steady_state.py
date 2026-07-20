"""Steady-state (OP) analysis test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, LoopUS, OAModel, VoltageSignal


@pytest.fixture
def ss_data():
    return load_fixture("steady_state_analysis")


def test_op_matches_minus_A_y(ss_data):
    """For an MVM circuit, OP analysis gives Vout ≈ -A @ y."""
    D = ss_data
    A = D["A"]
    G0 = D["G0"]
    I0 = D["I0"]
    N = int(D["N"])
    y = D["y"].ravel()
    V0 = I0 / G0

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, oa_model=OAModel("1P"))

    sigs = VoltageSignal.ytov(y, V0=V0, stype="DC")
    circ.set_inputs(sigs)

    Vmodel, _ = circ.op("model")

    expected = -A @ y * V0
    np.testing.assert_allclose(Vmodel, expected, rtol=1e-2, atol=5e-3)
