"""SPICE operating point with time-varying sources on a dense array.

LTspice's direct Newton iteration fails on dense feedback arrays, and with
``gminsteps=0``/``itl6=0`` (SPICE/fast, SPICE/accurate) it falls back to
pseudo-transient, which evaluates PULSE sources at t = 0 and ignores a
``DC x PULSE(...)`` override. A STEP input then read as 0 and the operating
point came out all zero. ``.op`` netlists now write every input as a DC source
at its t = 0 value, which is how ``op("model")`` samples them.
"""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentDAC, CurrentSignal, LoopUS, VoltageSignal
from loopuss.spice import SpiceAnalysis, spice_available

N, G0 = 10, 1e-4
I0 = G0
RNG = np.random.default_rng(0)
A = RNG.random((N, N))
A = A.T @ A + 2 * np.eye(N)  # dense and positive definite: a stable loop
A = A / A.max()
B = RNG.random((N, 2))
Y = np.linspace(-0.5, 0.5, N)

SPICE_SOLVERS = ["SPICE/fast", "SPICE/accurate", "SPICE/extreme"]


def _current(sig):
    return LoopUS(None, CrossBar(A * G0)), [sig]


# Each case builds a circuit and the inputs to load into it. All of them are
# non-zero at t = 0, so a source read as 0 shows up as a wrong answer.
CASES = {
    "current STEP": lambda: _current(CurrentSignal.ytoi(Y, I0=I0)),
    "current DC": lambda: _current(CurrentSignal.ytoi(Y, I0=I0, stype="DC")),
    "current PULSE": lambda: _current(CurrentSignal.ytoi(Y, I0=I0, stype="PULSE", ton=1e-6)),
    "current SINE with phase": lambda: _current(
        [CurrentSignal("SINE", ioff=0.1 * I0, iamp=0.4 * I0, freq=1e5, phi=60.0)] * N
    ),
    "voltage STEP": lambda: (
        LoopUS(CrossBar(B * G0), CrossBar(A * G0)),
        [VoltageSignal.ytov([0.3, -0.2])],
    ),
    # The DAC latency delays the STEP, so t = 0 reads i1, not i2.
    "STEP delayed by DAC latency": lambda: (
        LoopUS(None, CrossBar(A * G0), current_dac=CurrentDAC(latency=1e-6)),
        [[CurrentSignal("STEP", i1=0.2 * I0, i2=0.5 * I0, td=0.0)] * N],
    ),
}


def _build(case):
    circ, inputs = CASES[case]()
    circ.set_inputs(*inputs)
    return circ


@pytest.mark.parametrize(
    "sig, expected",
    [
        (VoltageSignal("STEP", v1=0.0, v2=0.3, td=0.0), 0.3),
        (VoltageSignal("STEP", v1=0.1, v2=0.3, td=1e-6), 0.1),
        (VoltageSignal("PULSE", v1=0.0, v2=0.3, td=0.0, ton=1e-6), 0.3),
        (VoltageSignal("SINE", voff=0.1, vamp=0.2, freq=1e3, phi=90.0), 0.3),
        (VoltageSignal("PWL", time=[0.0, 1e-6], vval=[0.25, 1.0]), 0.25),
        (VoltageSignal("DC", dcvalue=-0.4), -0.4),
    ],
)
def test_dc_only_writes_the_value_at_t0(sig, expected):
    line = sig.to_spice("1", "p", "n", dc_only=True)
    assert line.startswith("V1 p n DC(")
    assert float(line[len("V1 p n DC(") : -1]) == pytest.approx(expected)
    assert float(line[len("V1 p n DC(") : -1]) == pytest.approx(sig.to_array([0.0])[0])


def test_dc_only_keeps_empty_signals_out():
    assert VoltageSignal("EMPTY").to_spice("1", "p", "n", dc_only=True) == ""


@pytest.mark.parametrize("case", list(CASES))
def test_op_netlist_has_only_dc_inputs_and_tran_keeps_waveforms(case, tmp_path):
    circ = _build(case)
    circ.file_path = str(tmp_path)
    (op_net,), _ = circ.netlist("SPICE/fast", SpiceAnalysis("op"))
    sources = [ln for ln in open(op_net).read().splitlines() if ln.startswith(("VIN", "IIN"))]
    assert sources and all(" DC(" in ln for ln in sources)
    assert not any(kw in ln for ln in sources for kw in ("PULSE", "SINE", "PWL"))

    (tran_net,), _ = circ.netlist("SPICE/fast", SpiceAnalysis("tran", Tstop=1e-6))
    tran_text = open(tran_net).read()
    if "DC" not in case:
        assert any(kw in tran_text for kw in ("PULSE(", "SINE("))


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
@pytest.mark.parametrize("solver", SPICE_SOLVERS)
@pytest.mark.parametrize("case", list(CASES))
def test_dense_op_spice_matches_model(case, solver):
    circ = _build(case)
    v_model = circ.op("model")[0]
    v_spice = circ.op(solver)[0]
    assert np.max(np.abs(v_model)) > 1e-2  # guard: the case drives a real output
    np.testing.assert_allclose(v_spice, v_model, rtol=1e-3, atol=1e-4 * np.max(np.abs(v_model)))


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
def test_dense_tran_spice_still_starts_from_zero_with_step_inputs():
    circ = _build("current STEP")
    v, t = circ.tran("SPICE/fast", 1e-6)
    np.testing.assert_allclose(v[:, 0], 0.0, atol=1e-9)
    v_op = circ.op("model")[0]
    np.testing.assert_allclose(v[:, -1], v_op, rtol=1e-3, atol=1e-4 * np.max(np.abs(v_op)))
