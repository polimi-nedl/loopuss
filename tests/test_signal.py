"""Signal waveform tests."""

import numpy as np
import pytest

from loopuss import CurrentSignal, VoltageSignal


def test_dc_signal():
    s = VoltageSignal("DC", dcvalue=1.5)
    t = np.linspace(0, 1, 5)
    np.testing.assert_array_equal(s.to_array(t), [1.5, 1.5, 1.5, 1.5, 1.5])


def test_step_signal():
    s = VoltageSignal("STEP", v1=0.0, v2=2.0, td=0.5)
    t = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    out = s.to_array(t)
    np.testing.assert_array_equal(out, [0.0, 0.0, 2.0, 2.0, 2.0])


def test_pulse_signal():
    s = VoltageSignal("PULSE", v1=0.0, v2=1.0, td=0.2, ton=0.4)
    t = np.array([0.0, 0.1, 0.3, 0.6, 0.8])
    out = s.to_array(t)
    # ton interval: [0.2, 0.6] inclusive
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0, 0.0])


def test_pwl_signal():
    s = VoltageSignal("PWL", time=[0.0, 1.0, 2.0], vval=[0.0, 1.0, -1.0])
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    out = s.to_array(t)
    # 'previous' interpolation
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0, -1.0, -1.0])


def test_sine_signal():
    s = VoltageSignal("SINE", vamp=1.0, freq=1.0, voff=0.0, td=0.0)
    t = np.array([0.0, 0.25, 0.5, 0.75])
    out = s.to_array(t)
    # sin(2π * t) at these points
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0, -1.0], atol=1e-10)


def test_empty_signal():
    s = VoltageSignal("EMPTY")
    t = np.linspace(0, 1, 3)
    out = s.to_array(t)
    assert np.all(np.isnan(out))


def test_ytov_step():
    sigs = VoltageSignal.ytov([0.5, -1.0], V0=2.0)
    assert len(sigs) == 2
    assert sigs[0].v2 == 1.0
    assert sigs[1].v2 == -2.0


def test_current_signal_dc():
    s = CurrentSignal("DC", dcvalue=1e-3)
    t = np.array([0.0, 1.0])
    np.testing.assert_array_equal(s.to_array(t), [1e-3, 1e-3])


def test_add_delay():
    s = VoltageSignal("STEP", v1=0.0, v2=1.0, td=0.5)
    s2 = s.add_delay(0.2)
    assert s2.td == pytest.approx(0.7)
    # Original unchanged
    assert s.td == pytest.approx(0.5)


def test_breakpoints_pulse():
    s = VoltageSignal("PULSE", v1=0.0, v2=1.0, td=0.2, ton=0.4)
    bp = s.breakpoints()
    np.testing.assert_allclose(sorted(bp), [0.2, 0.6])


def test_breakpoints_dc():
    s = VoltageSignal("DC", dcvalue=1.0)
    bp = s.breakpoints()
    assert bp.size == 0
