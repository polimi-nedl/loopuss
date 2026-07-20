"""Piecewise-linear (PWL) input transient test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, lassoreg


@pytest.fixture
def pwl_data():
    return load_fixture("pwl_input")


def test_pwl_transient_matches_reference(pwl_data):
    D = pwl_data
    A = D["A"]
    G0 = float(D["G0"])
    M, N = int(D["M"]), int(D["N"])
    V0 = float(D["V0"])
    VLim = 0.1
    T = D["T"].astype(float)
    R = D["R"].astype(float)
    dGBWP = D["dGBWP"].ravel()

    L, S = lassoreg(T, R, A)
    oa = [OAModel("1P", GBWP=1e6) for _ in range(M + N)]
    oa += [
        OAModel("1P", A0=1e4, GBWP=1e9 + 1e9 * dGBWP[i], Vsupply=[-VLim, VLim]) for i in range(N)
    ]

    circ = LoopUS(
        CrossBar(np.empty((M + 2 * N, 0))),
        CrossBar(L * G0),
        sign_vector=S,
        oa_model=oa,
    )
    circ.set_inputs(CurrentSignal.ytoi(D["y"], I0=G0 * V0, stype="PWL", tv=D["t"].ravel()))

    time = np.concatenate([[0.0], np.logspace(-12, np.log10(5e-5), 5000)])
    Vout, tout = circ.tran("model", time)

    assert Vout.shape == D["Vout"].shape
    np.testing.assert_allclose(tout, D["time"].ravel(), rtol=0, atol=1e-9)
    # Steady state is essentially exact; the transient tracks within a small
    # tolerance set by the adaptive saturation subdivision.
    np.testing.assert_allclose(Vout[:, -1], D["Vout"][:, -1], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(Vout, D["Vout"], rtol=0, atol=1e-4)
