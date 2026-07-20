"""Benchmark tracking tests.

The benchmark reference fixture stores BenchReport objects that are awkward to
load, so these tests validate the structure, positivity and the documented flag
behaviour of the benchmarking rather than an exact object match.
"""

import numpy as np
import pytest

import loopuss as lp


def _bench_circuit():
    N = 6
    rng = np.random.default_rng(0)
    M = rng.standard_normal((N, N))
    A = M @ M.T + N * np.eye(N)  # SPD
    G0 = 1e-4
    # Wide supply keeps the OAs in their linear regime (the nonlinear
    # saturation path with voltage inputs is exercised separately by SPICE).
    oa = lp.OAModel("1P", GBWP=20e6, Vsupply=10.0, leakage_power=1e-6)
    circ = lp.LoopUS(
        lp.CrossBar(A * G0), lp.CrossBar(np.eye(N) * G0), oa_model=oa, bench_enable=True
    )
    circ.set_inputs(lp.VoltageSignal.ytov(np.linspace(0.01, 0.05, N), 1.0, stype="DC"))
    return circ, N


def test_benchmark_structure_and_positivity():
    circ, N = _bench_circuit()
    circ.tran("model", np.concatenate([[0], np.logspace(-10, -7, 500)]))
    B = circ.benchmark()

    # Latency / Energy / Area structures
    assert set(B.latency) >= {"Programming", "Computing", "PostProcessing", "Total"}
    assert B.energy["Breakdown"].shape == (7, 6)
    assert B.energy["Labels"][0] == "MemWrite"
    assert set(B.area) >= {"DAC", "ADC", "OA", "MemWrite", "InputArray", "FeedbackArray", "Total"}

    # one tran => OPs == complexity == N^3
    assert B.ops == pytest.approx(float(N**3))
    assert B.latency["Computing"] > 0
    assert B.energy["Total"] >= 0
    assert np.all(B.power >= 0)
    assert np.isfinite(B.throughput)


def test_benchmark_nowrite_zeros_programming():
    circ, _ = _bench_circuit()
    circ.tran("model", np.concatenate([[0], np.logspace(-10, -7, 500)]))
    full = circ.benchmark()
    nw = circ.benchmark("-nowrite")

    assert nw.latency["Programming"] == 0.0
    assert nw.area["MemWrite"] == 0.0
    # programming (MemWrite) row of the breakdown is zeroed
    np.testing.assert_array_equal(nw.energy["Breakdown"][0, :], 0.0)
    # totals shrink (or stay equal if there was no programming cost)
    assert nw.latency["Total"] <= full.latency["Total"] + 1e-30
    assert nw.area["Total"] <= full.area["Total"] + 1e-30


def test_benchmark_report_aggregation():
    circ, _ = _bench_circuit()
    circ.tran("model", np.concatenate([[0], np.logspace(-10, -7, 500)]))
    B = circ.benchmark("-nowrite")
    # serial sum of two identical reports doubles OPs/energy
    C = B + B
    assert C.ops == pytest.approx(2 * B.ops)
    assert C.energy["Total"] == pytest.approx(2 * B.energy["Total"])
