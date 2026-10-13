"""The analytical transient applies inputs that change after t = 0 at the right time."""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentDAC, CurrentSignal, LoopUS
from loopuss.spice import spice_available

N = 3
G0 = 1e-4
TD = 1e-9
TIME = np.concatenate([[0.0], np.logspace(-11, -7, 400)])


def _run(signals, **kwargs):
    circ = LoopUS(None, CrossBar(np.eye(N) * G0), **kwargs)
    circ.set_inputs(signals)
    return circ.tran("model", TIME)


def test_delayed_step_starts_at_td():
    V, t = _run([CurrentSignal("STEP", i1=0.0, i2=1e-5, td=TD) for _ in range(N)])
    assert np.abs(V[:, t < TD]).max() == 0.0
    np.testing.assert_allclose(V[:, -1], -1e-5 / G0, rtol=1e-3)


def test_dac_latency_delays_the_input():
    V, t = _run(CurrentSignal.ytoi(np.full(N, 1e-5)), current_dac=CurrentDAC(latency=TD))
    assert np.abs(V[:, t < TD]).max() == 0.0
    np.testing.assert_allclose(V[:, -1], -1e-5 / G0, rtol=1e-3)


def test_pwl_change_is_applied_at_its_knot():
    pwl = [CurrentSignal("PWL", ival=[0.0, 1e-5], time=[0.0, TD]) for _ in range(N)]
    V, t = _run(pwl)
    assert np.abs(V[:, t < TD]).max() == 0.0
    np.testing.assert_allclose(V[:, -1], -1e-5 / G0, rtol=1e-3)


def test_dac_latency_delays_a_pwl_input():
    pwl = [CurrentSignal("PWL", ival=[0.0, 1e-5], time=[0.0, TD]) for _ in range(N)]
    V, t = _run(pwl, current_dac=CurrentDAC(latency=TD))
    # The knot at TD moves to 2 TD.
    assert np.abs(V[:, t < 2 * TD]).max() == 0.0
    np.testing.assert_allclose(V[:, -1], -1e-5 / G0, rtol=1e-3)


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
def test_pwl_dac_latency_matches_spice():
    # Knots 1 fs apart: a step, read the same by the held model and by LTspice.
    pwl = [CurrentSignal("PWL", ival=[0.0, 0.0, 1e-5], time=[0.0, TD, TD + 1e-15])]
    c = LoopUS(None, CrossBar(np.eye(1) * G0), current_dac=CurrentDAC(latency=TD))
    c.set_inputs(pwl)
    vm, tm = c.tran("model", TIME)
    vs, ts = c.tran("SPICE/fast", TIME)
    late = TIME > 3 * TD
    np.testing.assert_allclose(
        np.interp(TIME[late], tm, vm[0]), np.interp(TIME[late], ts, vs[0]), atol=1e-3
    )
    assert np.abs(vs[0, ts < 1.9 * TD]).max() < 1e-6  # LTspice holds 0 until 2 TD


def test_input_that_switches_twice_follows_both_segments():
    # 0 until 1 ns, 10 uA until 20 ns, then 0 again: the output must rise and decay back.
    pwl = [CurrentSignal("PWL", ival=[0.0, 1e-5, 0.0], time=[0.0, TD, 20 * TD]) for _ in range(N)]
    V, t = _run(pwl)
    assert np.abs(V[:, t < TD]).max() == 0.0
    mid = np.searchsorted(t, 19 * TD)
    assert V[0, mid] == pytest.approx(-1e-5 / G0, rel=1e-2)
    assert abs(V[0, -1]) < 1e-3 * 1e-5 / G0


def test_step_at_time_zero_is_unchanged():
    V, t = _run(CurrentSignal.ytoi(np.full(N, 1e-5)))
    assert np.abs(V[:, 1]).max() > 0.0
    np.testing.assert_allclose(V[:, -1], -1e-5 / G0, rtol=1e-3)


def test_breakpoints_after_the_end_are_ignored():
    # A knot at 50 us must not stretch a 2 us run.
    far = [CurrentSignal("PWL", ival=[0.0, 1e-5, 0.0], time=[0.0, TD, 50e-6]) for _ in range(N)]
    near = [CurrentSignal("PWL", ival=[0.0, 1e-5], time=[0.0, TD]) for _ in range(N)]
    t_req = np.linspace(0, 2e-6, 5)
    c = LoopUS(None, CrossBar(np.eye(N) * G0), bench_enable=True)
    c.set_inputs(far)
    V, t = c.tran("model", t_req)
    assert t[-1] == pytest.approx(2e-6)
    assert c.latency["Computing"] == pytest.approx(2e-6)
    c_ref = LoopUS(None, CrossBar(np.eye(N) * G0))
    c_ref.set_inputs(near)
    V_ref, _ = c_ref.tran("model", t_req)
    np.testing.assert_allclose(V[:, -1], V_ref[:, -1])
