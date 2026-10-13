"""Input-array columns whose DAC drives nothing.

Such a column sits at its input reference (DAC output 0 V) in both backends.
"""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS, VoltageSignal
from loopuss.spice import SpiceAnalysis, spice_available

N, G0, VREF = 3, 1e-4, 0.2
REF = VoltageSignal("DC", dcvalue=VREF)
A = np.array([[3.0, 1.0, 0.5], [1.0, 2.5, 0.2], [0.5, 0.2, 2.0]])
B = np.array([[1.0, 0.5], [0.3, 1.0], [0.4, 0.1]])
TOL = dict(rtol=2e-3, atol=1e-6)


def _closed_form(i_in, v_col):
    return -np.linalg.solve(A * G0, i_in + B * G0 @ v_col)


def _reference_only():
    """Current inputs only: both columns sit at the reference."""
    c = LoopUS(CrossBar(B * G0), CrossBar(A * G0), input_reference=REF)
    c.set_inputs(CurrentSignal.ytoi([1e-5] * N))
    return c, _closed_form(np.full(N, 1e-5), np.full(2, VREF))


def test_unloaded_columns_use_the_reference_node(tmp_path):
    c, _ = _reference_only()
    c.file_path = str(tmp_path)
    (net,), _ = c.netlist("SPICE/fast", SpiceAnalysis("op"))
    cells = [ln for ln in open(net).read().splitlines() if ln.startswith("RXRIN")]
    assert cells and all("Vrefin_" in ln for ln in cells)


def test_unloaded_columns_model_matches_closed_form():
    c, ref = _reference_only()
    np.testing.assert_allclose(c.op("model")[0], ref, **TOL)


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
def test_unloaded_columns_spice_matches_model():
    c, ref = _reference_only()
    np.testing.assert_allclose(c.op("SPICE/fast")[0], ref, **TOL)


def _short_voltage_list():
    """One voltage signal for two columns: the second is padded with EMPTY."""
    c = LoopUS(CrossBar(B * G0), CrossBar(A * G0), input_reference=REF)
    c.set_inputs(VoltageSignal.ytov([0.5, np.nan]))
    return c, _closed_form(np.zeros(N), np.array([0.5 + VREF, VREF]))


def test_empty_voltage_holds_column_at_reference():
    c, ref = _short_voltage_list()
    np.testing.assert_allclose(c.op("model")[0], ref, **TOL)
    v, _ = c.tran("model", 1e-5)
    np.testing.assert_allclose(v[:, -1], ref, **TOL)


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
def test_empty_voltage_spice_matches_model():
    c, ref = _short_voltage_list()
    np.testing.assert_allclose(c.op("SPICE/fast")[0], ref, **TOL)
