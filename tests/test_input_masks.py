"""Partial input masks: only some rows get currents, only some columns voltages.

The DAC waveforms are scattered into full ``(n, T)`` / ``(p, T)`` matrices, so
the unmasked rows carry no current and the unmasked columns sit at 0 V.
"""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS, VoltageSignal
from loopuss.spice import SpiceAnalysis, spice_available

N, G0 = 3, 1e-4
A = np.array([[3.0, 1.0, 0.5], [1.0, 2.5, 0.2], [0.5, 0.2, 2.0]])
B = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.6], [0.4, 0.1, 1.0]])
TOL = dict(rtol=2e-3, atol=1e-6)


def _current_masked():
    c = LoopUS(None, CrossBar(A * G0), input_mask_i=[0, 2])
    c.set_inputs(CurrentSignal.ytoi([1e-5, 2e-5]))
    i_full = np.array([1e-5, 0.0, 2e-5])
    return c, -np.linalg.solve(A * G0, i_full)


def _voltage_masked():
    c = LoopUS(CrossBar(B * G0), CrossBar(A * G0), input_mask_v=[1])
    c.set_inputs(VoltageSignal.ytov([0.5]))
    v_full = np.array([0.0, 0.5, 0.0])
    return c, -np.linalg.solve(A * G0, B * G0 @ v_full)


CASES = {"current mask": _current_masked, "voltage mask": _voltage_masked}


@pytest.mark.parametrize("case", list(CASES))
def test_masked_op_and_tran_match_closed_form(case):
    c, ref = CASES[case]()
    np.testing.assert_allclose(c.op("model")[0], ref, **TOL)
    v, _ = c.tran("model", 1e-5)
    np.testing.assert_allclose(v[:, -1], ref, **TOL)


@pytest.mark.parametrize("case", list(CASES))
def test_masked_reports_have_full_shapes(case):
    c, _ = CASES[case]()
    _, _, R = c.tran("model", 1e-6, return_report=True)
    T = R.time.size
    assert R.Iin.shape == (c.n, T)
    assert R.Vin.shape == ((c.p, T) if c.p else (0, T))
    assert R.I_Vin.shape == R.Vin.shape
    _, _, Rop = c.op("model", return_report=True)
    assert Rop.Iin.shape == (c.n, 1)


def test_unmasked_rows_and_columns_are_zero():
    c, _ = _current_masked()
    _, _, R = c.op("model", return_report=True)
    assert R.Iin[1, 0] == 0.0 and R.Iin[0, 0] == pytest.approx(1e-5)
    c, _ = _voltage_masked()
    _, _, R = c.op("model", return_report=True)
    np.testing.assert_allclose(R.Vin[:, 0], [0.0, 0.5, 0.0])


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
@pytest.mark.parametrize("case", list(CASES))
def test_masked_op_spice_matches_model(case):
    c, ref = CASES[case]()
    np.testing.assert_allclose(c.op("SPICE/fast")[0], ref, **TOL)


def test_dacs_outside_the_masks_are_not_netlisted(tmp_path):
    c = LoopUS(CrossBar(B * G0), CrossBar(A * G0))
    c.set_inputs(CurrentSignal.ytoi([1e-5] * N), VoltageSignal.ytov([0.1, 0.2, 0.3]))
    # Narrow the masks after loading: the old waveforms stay on the DACs.
    c.input_mask_i = np.array([0])
    c.input_mask_v = np.array([2])
    c.file_path = str(tmp_path)
    (net,), _ = c.netlist("SPICE/fast", SpiceAnalysis("op"))
    sources = {ln.split()[0] for ln in open(net).read().splitlines() if ln[:3] in ("IIN", "VIN")}
    assert sources == {"IIN1", "VIN3"}


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
def test_dacs_outside_the_masks_spice_matches_model():
    c = LoopUS(CrossBar(B * G0), CrossBar(A * G0))
    c.set_inputs(CurrentSignal.ytoi([1e-5] * N), VoltageSignal.ytov([0.1, 0.2, 0.3]))
    c.input_mask_i = np.array([0])
    c.input_mask_v = np.array([2])
    ref = -np.linalg.solve(A * G0, np.array([1e-5, 0, 0]) + B * G0 @ np.array([0, 0, 0.3]))
    np.testing.assert_allclose(c.op("model")[0], ref, **TOL)
    np.testing.assert_allclose(c.op("SPICE/fast")[0], ref, **TOL)
