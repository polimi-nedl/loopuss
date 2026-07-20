"""Pole-zero analysis of a non-linear LASSO circuit."""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import CrossBar, LoopUS, OAModel, lassoreg
from loopuss.analysis import natsortpoles


@pytest.fixture
def lasso_pz_data():
    return load_fixture("lasso_pz")


def test_lasso_pz_matches_reference(lasso_pz_data):
    D = lasso_pz_data
    A = D["A"]
    G0 = float(D["G0"])
    M, N = int(D["M"]), int(D["N"])
    VLim = float(D["VLim"])
    t, lam = float(D["t"]), float(D["l"])

    L, S = lassoreg(t, lam, A)
    oa = [OAModel("1P") for _ in range(M + N)]
    oa += [OAModel("1P", Vsupply=[-VLim, VLim]) for _ in range(N)]

    size = M + 2 * N
    circ = LoopUS(CrossBar(np.empty((size, 0))), CrossBar(L * G0), sign_vector=S, oa_model=oa)

    poles = np.zeros((size, 2**N), dtype=complex)
    for col in range(2**N):
        bits = np.array([not ((col >> (N - 1 - k)) & 1) for k in range(N)])
        l_id = np.ones(size, dtype=bool)
        l_id[M + N : M + 2 * N] = bits
        p, *_ = circ.pz("hz", l_id)
        poles[:, col] = natsortpoles(p)

    ref = D["p"]
    assert np.all(np.isinf(poles) == np.isinf(ref))
    finite = np.isfinite(ref)
    np.testing.assert_allclose(poles[finite], ref[finite], rtol=1e-6)
