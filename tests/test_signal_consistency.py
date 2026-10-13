"""Signals sample in the model as LTspice evaluates the sources it is given."""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS, VoltageSignal
from loopuss.spice import spice_available

G0 = 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [dict(voff=0.1, freq=1e6), dict(voff=0.1, vamp=0.2), dict(voff=0.1)],
)
def test_sine_needs_amplitude_and_frequency(kwargs):
    with pytest.raises(ValueError, match="amplitude and a frequency"):
        VoltageSignal("SINE", **kwargs)


def test_delayed_sine_holds_its_starting_value():
    s = VoltageSignal("SINE", voff=0.1, vamp=0.2, freq=1e3, phi=90.0, td=1e-3)
    np.testing.assert_allclose(s.to_array([0.0, 0.5e-3, 1e-3]), 0.3)


@pytest.mark.skipif(not spice_available(), reason="LTspice not available")
def test_delayed_sine_before_td_matches_spice():
    c = LoopUS(None, CrossBar(np.eye(1) * G0))
    c.set_inputs([CurrentSignal("SINE", ioff=0.0, iamp=1e-5, freq=1e5, phi=90.0, td=2e-6)])
    t = np.linspace(0, 1.9e-6, 20)
    vm, tm = c.tran("model", t)
    vs, ts = c.tran("SPICE/fast", t)
    np.testing.assert_allclose(np.interp(t, tm, vm[0])[1:], -0.1, rtol=1e-3)
    np.testing.assert_allclose(np.interp(t, ts, vs[0])[1:], -0.1, rtol=1e-3)


def test_sine_stops_after_ncycles():
    s = VoltageSignal("SINE", voff=0.1, vamp=1, freq=1e6, td=1e-6, theta=2e5, phi=30, ncycles=2)
    end = 0.1 + np.exp(-2e5 * 2e-6) * np.sin(np.deg2rad(30))
    np.testing.assert_allclose(s.to_array([3e-6, 3.5e-6, 5e-6]), end)
    np.testing.assert_allclose(s.breakpoints(), [3e-6])
    endless = VoltageSignal("SINE", voff=0.1, vamp=1, freq=1e6, td=1e-6, phi=30)
    assert endless.to_array([3.25e-6])[0] == pytest.approx(0.1 + np.sin(np.deg2rad(120)))
