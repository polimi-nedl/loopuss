"""Extra unit tests for loopuss.signal beyond the basic test_signal.py."""

import numpy as np
import pytest

from loopuss import CurrentSignal, VoltageSignal


# --- waveform evaluation ---
def test_dc_waveform():
    s = VoltageSignal("DC", dcvalue=0.7)
    np.testing.assert_array_equal(s.to_array(np.linspace(0, 1, 5)), 0.7)


def test_step_waveform_switches_at_td():
    s = VoltageSignal("STEP", v1=0.0, v2=1.0, td=0.5)
    out = s.to_array(np.array([0.0, 0.4, 0.5, 0.6]))
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0])


def test_pulse_waveform_window():
    s = VoltageSignal("PULSE", v1=0.0, v2=2.0, td=1.0, ton=2.0)
    out = s.to_array(np.array([0.5, 1.0, 2.0, 3.0, 3.5]))
    np.testing.assert_array_equal(out, [0.0, 2.0, 2.0, 2.0, 0.0])


def test_pwl_previous_interpolation():
    s = VoltageSignal("PWL", vval=[0, 1, 3], time=[0, 1, 2])
    out = s.to_array(np.array([-1, 0.0, 0.5, 1.0, 1.5, 5.0]))
    np.testing.assert_array_equal(out, [0, 0, 0, 1, 1, 3])


def test_sine_waveform_values():
    s = VoltageSignal("SINE", vamp=1.0, voff=0.0, freq=1.0, phi=0.0, td=0.0)
    t = np.array([0.0, 0.25, 0.5])
    np.testing.assert_allclose(s.to_array(t), [0.0, 1.0, 0.0], atol=1e-12)


def test_sine_with_offset_before_delay():
    s = VoltageSignal("SINE", vamp=1.0, voff=0.3, freq=1.0, td=1.0)
    assert s.to_array(np.array([0.0]))[0] == 0.3  # offset before delay


def test_empty_voltage_is_nan_current_is_zero():
    assert np.isnan(VoltageSignal("EMPTY").to_array(np.array([0.0]))[0])
    assert CurrentSignal("EMPTY").to_array(np.array([0.0]))[0] == 0.0


# --- breakpoints ---
def test_breakpoints():
    assert VoltageSignal("DC", dcvalue=1).breakpoints().size == 0
    assert VoltageSignal("EMPTY").breakpoints().size == 0
    bp = VoltageSignal("PULSE", v1=0, v2=1, td=1.0, ton=2.0).breakpoints()
    assert set(bp.tolist()) == {1.0, 3.0}
    bp = VoltageSignal("PWL", vval=[0, 1], time=[0.2, 0.8]).breakpoints()
    assert set(bp.tolist()) == {0.2, 0.8}
    assert VoltageSignal("SINE", vamp=1, freq=1).breakpoints().size == 0


# --- min / max ---
def test_min_max():
    assert VoltageSignal("DC", dcvalue=0.5).max_value() == 0.5
    s = VoltageSignal("STEP", v1=-1, v2=2)
    assert s.max_value() == 2 and s.min_value() == -1
    sine = VoltageSignal("SINE", vamp=2, voff=1, freq=1)
    assert sine.max_value() == 3 and sine.min_value() == -1
    pwl = VoltageSignal("PWL", vval=[-3, 4, 1], time=[0, 1, 2])
    assert pwl.max_value() == 4 and pwl.min_value() == -3


# --- add_delay ---
def test_add_delay_is_pure():
    s = VoltageSignal("STEP", v1=0, v2=1, td=0.0)
    s2 = s.add_delay(0.3)
    assert s2.td == 0.3 and s.td == 0.0  # original unchanged


# --- conversions ---
def test_vtoy_itoy_roundtrip():
    assert VoltageSignal("DC", dcvalue=2.0).vtoy(V0=2.0) == pytest.approx(1.0)
    assert CurrentSignal("DC", dcvalue=4.0).itoy(I0=2.0) == pytest.approx(2.0)


def test_ytov_step_dc_pulse_pwl_empty():
    out = VoltageSignal.ytov(np.array([0.1, np.nan]), V0=2.0, stype="DC")
    assert out[0].is_dc() and out[0].dcvalue == pytest.approx(0.2)
    assert out[1].is_empty()

    out = VoltageSignal.ytov(np.array([0.5]), V0=1.0, stype="STEP")
    assert out[0].is_step() and out[0].val2 == 0.5

    out = VoltageSignal.ytov(np.array([0.5]), V0=1.0, stype="PULSE", ton=1e-6)
    assert out[0].is_pulse()

    out = VoltageSignal.ytov(np.array([0.1, 0.2]), V0=1.0, stype="PWL", tv=np.array([0.0, 1.0]))
    assert out[0].is_pwl()


def test_ytov_pulse_requires_ton():
    with pytest.raises(ValueError):
        VoltageSignal.ytov(np.array([1.0]), stype="PULSE")


def test_ytoi_variants():
    out = CurrentSignal.ytoi(np.array([3.0]), I0=2.0, stype="DC")
    assert out[0].is_dc() and out[0].dcvalue == pytest.approx(6.0)
