"""Tests for the GM (pure-transconductor) op-amp model in its ideal limit
A0 = Inf, FP = 0."""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, VoltageSignal


def _op(A, y, oa):
    G0 = 1.0 / 10e3
    circ = LoopUS(CrossBar(np.empty((A.shape[0], 0))), CrossBar(A * G0), oa_model=oa)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0))
    Vop, _ = circ.op("model")
    Vt, _ = circ.tran("model", np.concatenate([[0.0], np.logspace(-9, -4, 400)]))
    return Vop, Vt[:, -1]


def test_gm_partial_saturation_op_matches_transient():
    """A supply-limited ideal transconductor can itself partially saturate. Its
    operating point (static solver) must agree with the transient steady state,
    which resolves saturation by entirely independent machinery."""
    N = 4
    Vlim = 0.1
    A = np.array(
        [[2.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0], [0.0, 0.0, 1.0, 2.0]]
    )
    y = np.array([0.8, 0.1, -0.9, 0.2])

    gm = [OAModel("GM", GBWP=20e6, Vsupply=[-Vlim, Vlim]) for _ in range(N)]
    gm_op, gm_tr = _op(A, y, gm)

    # Genuinely partial saturation: at least one rail-clamped and one interior OA.
    clamped = np.abs(gm_op) > Vlim - 1e-6
    assert clamped.any() and (~clamped).any()

    # Exact infinite-gain handling: op and transient steady state coincide.
    np.testing.assert_allclose(gm_op, gm_tr, rtol=0, atol=1e-9)


def test_gm_ideal_limits():
    gm = OAModel("GM", GBWP=50e6)
    assert np.isinf(gm.A0)
    np.testing.assert_array_equal(gm.FP, [0.0])
    assert gm.GBWP == 50e6
    assert gm.P0 == pytest.approx(50e6)  # P0 falls back to GBWP for GM


def test_gm_spice_uses_gbwp():
    gm = OAModel("GM", GBWP=50e6)
    assert gm.model_str().startswith("NonLinearOA_GM")
    assert gm.spice_params() == "GBWP={5e+07}"


def test_gm_pole_zero_is_finite():
    N = 4
    G0 = 1e-4
    A = np.array(
        [[2.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0], [0.0, 0.0, 1.0, 2.0]]
    )
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0), oa_model=OAModel("GM", GBWP=20e6))
    poles, *_ = circ.pz()
    assert np.all(np.isfinite(poles))


def test_gm_infinite_gain_has_no_finite_gain_error():
    """An ideal GM integrator MVM gives exactly -A*y (no finite-gain offset)."""
    N = 3
    G0 = 1e-4
    A = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    y = np.array([0.1, 0.2, 0.3])
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0), oa_model=OAModel("GM", GBWP=20e6))
    circ.set_inputs(VoltageSignal.ytov(y, V0=1.0, stype="DC"))
    V, _ = circ.op("model")
    np.testing.assert_allclose(V, -A @ y, rtol=1e-9, atol=1e-9)
