"""LASSO regression circuit transient test."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, lassoreg


@pytest.fixture
def lasso_data():
    return load_fixture("lasso_regression")


def test_lasso_transient_matches_reference(lasso_data):
    D = lasso_data
    A = D["A"]
    G0 = float(D["G0"])
    M, N = int(D["M"]), int(D["N"])
    V0 = float(D["V0"])
    VLim = float(D["VLim"])
    t, lam = float(D["t"]), float(D["l"])

    L, S = lassoreg(t, lam, A)
    # Ideal transconductors feed the supply-limited LASSO branch.
    oa = [OAModel("GM") for _ in range(M + N)]
    oa += [OAModel("1P", Vsupply=[-VLim, VLim]) for _ in range(N)]

    circ = LoopUS(
        CrossBar(np.empty((M + 2 * N, 0))),
        CrossBar(L * G0),
        sign_vector=S,
        oa_model=oa,
    )
    circ.set_inputs(CurrentSignal.ytoi(D["y"].ravel(), I0=G0 * V0))

    time = np.concatenate([[0.0], np.logspace(-10, -3, 1000)])
    Vmodel, tmodel = circ.tran("model", time)

    np.testing.assert_allclose(tmodel, D["tmodel"].ravel(), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(Vmodel, D["Vmodel"], rtol=1e-6, atol=1e-9)
