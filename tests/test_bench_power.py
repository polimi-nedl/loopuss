"""Benchmark power and run duration."""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS, MemWriteModel, OAModel

N, G0 = 10, 1e-4
RNG = np.random.default_rng(0)
A = RNG.random((N, N))
A = A.T @ A + N * np.eye(N)
A /= A.max()
Y = RNG.random(N)


def _circ():
    c = LoopUS(None, CrossBar(A * G0), oa_model=OAModel("1P", Vsupply=[-1, 1]), bench_enable=True)
    c.set_inputs(CurrentSignal.ytoi(Y, I0=0.1 * G0, stype="PULSE", ton=1e-6))
    return c


T_LOG = np.concatenate([[0.0], np.logspace(-10, -4, 1000)])
T_LIN = np.linspace(0, 1e-4, 1001)


def test_mean_power_is_energy_over_time_on_any_grid():
    means = []
    for t in (T_LOG, T_LIN):
        c = _circ()
        c.tran("model", t)
        r = c.benchmark()
        assert r.power[0] == pytest.approx(r.energy["Total"] / r.latency["Total"])
        means.append(r.power[0])
    # A sample average over the log grid was 54x the linear-grid value; what
    # is left is the trapezoid integration of the energy on each grid.
    assert means[0] == pytest.approx(means[1], rel=5e-2)


def test_peak_power_is_at_least_the_mean():
    c = _circ()
    c.tran("model", T_LIN)
    mean, peak = c.benchmark().power
    assert peak >= mean > 0


def test_run_is_charged_from_t0_like_its_leakage():
    # The inputs are applied at t = 0, so a grid starting at 1 us still costs 2 us.
    t = np.linspace(1e-6, 2e-6, 101)
    c = LoopUS(None, CrossBar(A * G0), oa_model=OAModel("Gupta16"), bench_enable=True)
    c.set_inputs(CurrentSignal.ytoi(Y, I0=0.1 * G0))
    c.tran("model", t)
    assert c.latency["Computing"] == pytest.approx(2e-6)
    oa_leak = sum(m.leakage_power for m in c.oa_models)
    assert c.energy["OA"]["Computing"]["Static"] == pytest.approx(oa_leak * 2e-6)


def test_no_runs_no_power():
    c = LoopUS(None, CrossBar(A * G0), bench_enable=True)
    np.testing.assert_array_equal(c.power, [0.0, 0.0])


def test_nowrite_mean_power_leaves_out_programming():
    c = LoopUS(
        None,
        CrossBar(A * G0),
        oa_model=OAModel("1P", Vsupply=[-1, 1]),
        mem_write_model=MemWriteModel("IGVVA10"),
        bench_enable=True,
    )
    c.reprogram()
    c.set_inputs(CurrentSignal.ytoi(Y, I0=0.1 * G0))
    c.tran("model", T_LIN)
    full, nw = c.benchmark(), c.benchmark("-nowrite")
    assert nw.power[0] == pytest.approx(nw.energy["Total"] / nw.latency["Total"])
    # Programming draws more than computing here, so leaving it in raises the mean.
    assert full.power[0] > nw.power[0]
    assert nw.power[1] == full.power[1]
