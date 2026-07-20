"""Integration tests against the reference fixtures.

Each test rebuilds the circuit for the corresponding reference case, runs the
model (and, where LTSpice is available, the SPICE backend), and checks the
result against:
  * the saved reference model output ``Vmodel`` (same algorithm -> machine
    precision), and
  * the mathematical reference (e.g. -A\\y) within engineering tolerance.
"""

import numpy as np
import pytest
from conftest import load_fixture

from loopuss import (
    CrossBar,
    CurrentSignal,
    LoopUS,
    OAModel,
    VoltageSignal,
    linreg,
    posneg,
    ridgereg,
)
from loopuss.spice import spice_available

needs_spice = pytest.mark.skipif(not spice_available(), reason="LTSpice not installed")

# generous-but-meaningful tolerance vs the reference model output
MODEL = dict(rtol=1e-4, atol=1e-4)
# looser tolerance vs the ideal mathematical reference (finite-gain error)
MATH = dict(rtol=2e-2, atol=1e-2)
LOG = np.concatenate([[0.0], np.logspace(-10, -4, 1000)])


# --- matrix-vector multiplication ---
def test_mvm_model_and_spice():
    D = load_fixture("mvm")
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    y = D["y"].ravel()
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0), oa_model=OAModel("1P"))
    circ.set_inputs(VoltageSignal.ytov(y, V0=I0 / G0, stype="DC"))
    V, _ = circ.tran("model", D["tmodel"].ravel())
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    np.testing.assert_allclose(V[:, -1], -A @ y * (I0 / G0), **MATH)


@needs_spice
def test_mvm_spice_matches_fixture():
    D = load_fixture("mvm")
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0), oa_model=OAModel("1P"))
    circ.set_inputs(VoltageSignal.ytov(D["y"].ravel(), V0=I0 / G0, stype="DC"))
    V, _ = circ.tran("SPICE/fast", float(D["tspice"].ravel()[-1]))
    np.testing.assert_allclose(V[:, -1], D["Vspice"][:, -1], rtol=1e-3, atol=1e-4)


# --- steady-state (operating point, MVM construction) ---
def test_steady_state_op_model():
    D = load_fixture("steady_state_analysis")
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0))
    circ.set_inputs(VoltageSignal.ytov(D["y"].ravel(), V0=I0 / G0, stype="DC"))
    V, _ = circ.op("model")
    np.testing.assert_allclose(V, D["Vmodel"].ravel(), **MODEL)


@needs_spice
def test_steady_state_op_spice():
    D = load_fixture("steady_state_analysis")
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0))
    circ.set_inputs(VoltageSignal.ytov(D["y"].ravel(), V0=I0 / G0, stype="DC"))
    V, _ = circ.op("SPICE/fast")
    np.testing.assert_allclose(V, D["Vspice"].ravel(), rtol=1e-3, atol=1e-4)


# --- linear regression ---
def test_linear_regression_positive():
    D = load_fixture("linear_regression_positive")
    A, G0, V0 = D["A"], float(D["G0"]), float(D["V0"])
    y, M, Nc = D["y"].ravel(), A.shape[0], A.shape[1]
    L, S = linreg(1.0, A)
    circ = LoopUS(
        CrossBar(np.empty((L.shape[0], 0))),
        CrossBar(L * G0),
        sign_vector=S,
        output_mask=M + np.arange(Nc),
    )
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0 * V0))
    V, _ = circ.tran("model", LOG)
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    np.testing.assert_allclose(V[:, -1], -np.linalg.pinv(A) @ y * V0, **MATH)


def test_linear_regression_posneg():
    D = load_fixture("linear_regression_posneg")
    A, G0, V0 = D["A"], float(D["G0"]), float(D["V0"])
    y, M, Nc = D["y"].ravel(), A.shape[0], A.shape[1]
    L, S = linreg(1.0, A)
    ncomp = L.shape[0]
    L, S, *_ = posneg(L, 1.0, sign_vector=S)
    nbuf = L.shape[0] - ncomp
    oas = [OAModel("1P", GBWP=1e6) for _ in range(ncomp)] + [
        OAModel("1P", GBWP=100e6) for _ in range(nbuf)
    ]
    circ = LoopUS(
        CrossBar(np.empty((L.shape[0], 0))),
        CrossBar(L * G0),
        oa_model=oas,
        sign_vector=S,
        output_mask=M + np.arange(Nc),
    )
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0 * V0))
    V, _ = circ.tran("model", np.logspace(-10, -4, 1000))
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    np.testing.assert_allclose(V[:, -1], -np.linalg.pinv(A) @ y * V0, **MATH)


def test_ridge_regression():
    D = load_fixture("ridge_regression")
    A, G0, V0, lam = D["A"], float(D["G0"]), float(D["V0"]), float(D["l"])
    y, M, Nc = D["y"].ravel(), A.shape[0], A.shape[1]
    L, S = ridgereg(np.sqrt(lam), np.sqrt(lam), A)
    L, S, *_ = posneg(L, 1.0, sign_vector=S)
    circ = LoopUS(
        CrossBar(np.empty((L.shape[0], 0))),
        CrossBar(L * G0),
        sign_vector=S,
        output_mask=M + np.arange(Nc),
    )
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0 * V0))
    V, _ = circ.tran("model", LOG)
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    ref = -np.linalg.solve(A.T @ A + lam * np.eye(Nc), A.T @ y) * V0
    np.testing.assert_allclose(V[:, -1], ref, **MATH)


# --- PD linear systems ---
def test_linear_system_pd_positive():
    D = load_fixture("linear_system_PD_positive")
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    y = D["y"].ravel()
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0))
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))
    V, _ = circ.tran("model", LOG)
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    np.testing.assert_allclose(V[:, -1], -np.linalg.solve(A, y) * float(D["V0"]), **MATH)


def test_linear_system_pd_posneg():
    D = load_fixture("linear_system_PD_posneg")
    A, G0, V0, N = D["A"], float(D["G0"]), float(D["V0"]), int(D["N"])
    y = D["y"].ravel()
    L, S, *_ = posneg(A, 1.0)
    circ = LoopUS(
        CrossBar(np.empty((L.shape[0], 0))),
        CrossBar(L * G0),
        sign_vector=S,
        output_mask=np.arange(N),
    )
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0 * V0))
    V, _ = circ.tran("model", np.logspace(-10, -4, 1000))
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    np.testing.assert_allclose(V[:, -1], -np.linalg.solve(A, y) * V0, **MATH)


# --- initial condition ---
def test_initial_condition_relaxation():
    D = load_fixture("initial_condition")
    A, G0, I0, N = D["A"], float(D["G0"]), float(D["I0"]), int(D["N"])
    y = D["y"].ravel()
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0), initial_condition=y)
    circ.set_inputs(CurrentSignal.ytoi(np.zeros(N), I0))
    V, _ = circ.tran("model", np.concatenate([[0.0], np.logspace(-12, -7, 1000)]))
    # starts at the imposed IC ...
    np.testing.assert_allclose(V[:, 0], y, rtol=1e-6, atol=1e-9)
    # ... and matches the reference trajectory endpoint
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)


# --- mixed current/voltage inputs ---
def test_iv_input():
    D = load_fixture("iv_input")
    A, B, G0, V0 = D["A"], D["B"], float(D["G0"]), float(D["V0"])
    yI, yV = D["yI"].ravel(), D["yV"].ravel()
    circ = LoopUS(CrossBar(A * G0), CrossBar(B * G0))
    circ.set_inputs(VoltageSignal.ytov(yV, V0), CurrentSignal.ytoi(yI, G0 * V0))
    V, _ = circ.tran("model", np.logspace(-10, -4, 100))
    np.testing.assert_allclose(V[:, -1], D["Vmodel"][:, -1], **MODEL)
    ref = -np.linalg.solve(B, yI + A @ yV)
    np.testing.assert_allclose(V[:, -1], ref, **MATH)


# --- pole-zero analysis ---
@pytest.mark.parametrize("mat_key,pole_key", [("A", "poles1"), ("B", "poles2"), ("C", "poles3")])
def test_pole_zero(mat_key, pole_key):
    D = load_fixture("pole_zero_analysis")
    M, G0, N = D[mat_key], float(D["G0"]), int(D["N"])
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(M * G0))
    poles, *_ = circ.pz()
    got = np.sort_complex(poles)
    exp = np.sort_complex(D[pole_key].ravel())
    np.testing.assert_allclose(got, exp, rtol=1e-4, atol=1e-2)


# --- power iteration (op loop + DSP normalisation) ---
def test_power_iteration_converges_to_dominant_eigenvector():
    D = load_fixture("power_iteration")
    A, G0, N = D["A"], float(D["G0"]), int(D["N"])
    V0 = 1.0
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0))
    circ.set_inputs(VoltageSignal.ytov(D["y0"].ravel(), V0=V0, stype="DC"))
    y = None
    for _ in range(100):
        y, _ = circ.op("model")
        y = y / np.max(np.abs(y))
        circ.set_inputs(VoltageSignal.ytov(y, V0=V0, stype="DC"))
    eigvals, eigvecs = np.linalg.eig(A)
    v_ref = eigvecs[:, int(np.argmax(eigvals.real))].real
    v_ref = v_ref / np.max(np.abs(v_ref))
    if np.dot(y, v_ref) < 0:
        v_ref = -v_ref
    # cosine similarity close to 1 (converged up to sign)
    cos = np.dot(y, v_ref) / (np.linalg.norm(y) * np.linalg.norm(v_ref))
    assert cos > 0.99
