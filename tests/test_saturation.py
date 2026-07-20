"""Exercise the nonlinear OA-saturation paths of the model solver, covering
partial saturation, full (all-OAs) saturation, and saturation with voltage
inputs. These regimes previously tripped bugs in the solver (now fixed)."""

import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, VoltageSignal


def _partial_sat_circuit(vsupply=0.3):
    """N independent integrators (Gf = I): output_i ≈ -y_i.

    y is chosen so that one OA saturates and the other stays linear.
    """
    N = 2
    G0 = 1e-4
    circ = LoopUS(
        CrossBar(np.empty((N, 0))),
        CrossBar(np.eye(N) * G0),
        oa_model=OAModel("1P", Vsupply=vsupply),
    )
    y = np.array([5.0, 0.1])  # -> outputs ≈ [-5, -0.1]
    circ.set_inputs(CurrentSignal.ytoi(y, I0=G0 * 1.0))
    return circ, y, vsupply


def test_nltran_partial_saturation_clips_to_rails():
    circ, y, vsup = _partial_sat_circuit()
    V, t = circ.tran("model", np.concatenate([[0.0], np.logspace(-10, -5, 300)]))
    assert np.all(np.isfinite(V))
    assert abs(V[0, -1]) > 0.99 * vsup  # channel 0 saturates
    assert abs(V[1, -1]) < vsup  # channel 1 stays linear
    assert np.all(np.abs(V) <= vsup + 1e-6)


def test_nlop_partial_saturation_clips_to_rails():
    circ, y, vsup = _partial_sat_circuit()
    V, _ = circ.op("model")
    assert np.all(np.isfinite(V))
    assert abs(V[0]) > 0.99 * vsup
    assert abs(V[1]) < vsup


def test_unsaturated_matches_linear_solution():
    """With wide rails the saturating solver reduces to the linear answer."""
    circ, y, _ = _partial_sat_circuit(vsupply=100.0)
    V, _ = circ.op("model")
    np.testing.assert_allclose(V, -y, rtol=2e-2, atol=1e-2)


def test_all_oas_saturated():
    """Every OA driven past its rail (n_lin == 0): all outputs clamp."""
    N, G0, vsup = 4, 1e-4, 0.3
    circ = LoopUS(
        CrossBar(np.empty((N, 0))), CrossBar(np.eye(N) * G0), oa_model=OAModel("1P", Vsupply=vsup)
    )
    circ.set_inputs(CurrentSignal.ytoi(np.array([5.0, 6.0, 7.0, 8.0]), I0=G0))
    Vop, _ = circ.op("model")
    np.testing.assert_allclose(Vop, -vsup, atol=1e-9)  # all at -rail
    Vt, _ = circ.tran("model", np.concatenate([[0.0], np.logspace(-10, -5, 200)]))
    assert np.all(np.isfinite(Vt))
    assert np.all(np.abs(Vt) <= vsup + 1e-6)


def test_saturation_with_voltage_inputs():
    """Saturation while voltage inputs are present (Vin is time-resolved)."""
    N, G0, vsup = 4, 1e-4, 0.5
    rng = np.random.default_rng(0)
    M = rng.standard_normal((N, N))
    A = M @ M.T + N * np.eye(N)
    A = A / A.max()
    circ = LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0), oa_model=OAModel("1P", Vsupply=vsup))
    circ.set_inputs(VoltageSignal.ytov(np.array([2.0, 3.0, 4.0, 5.0]), 1.0, stype="DC"))
    Vt, _ = circ.tran("model", np.logspace(-10, -6, 200))
    assert np.all(np.isfinite(Vt))  # previously raised ValueError
    assert np.all(np.abs(Vt) <= vsup + 1e-6)
    assert np.any(np.abs(Vt[:, -1]) > 0.99 * vsup)  # at least one rail-clamped
