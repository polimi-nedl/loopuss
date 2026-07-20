"""Edge-case coverage for loopuss.signal."""

import numpy as np
import pytest

from loopuss import CurrentSignal, VoltageSignal


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        VoltageSignal("BOGUS")


def test_empty_min_max_zero():
    s = VoltageSignal("EMPTY")
    assert s.max_value() == 0.0 and s.min_value() == 0.0


def test_current_signal_aliases_roundtrip():
    s = CurrentSignal("PULSE", i1=0.0, i2=2e-3, td=0.0, ton=1e-6)
    assert s.i2 == 2e-3
    s.iamp = 1e-3
    assert s.ampsine == 1e-3
    s.ioff = 0.5e-3
    assert s.offsine == 0.5e-3
    s.ival = [1.0, 2.0]
    np.testing.assert_array_equal(s.valpwl, [1.0, 2.0])


def test_voltage_signal_aliases():
    s = VoltageSignal("PULSE", v1=0.0, v2=1.0, td=0.0, ton=1e-6)
    s.vamp = 2.0
    s.voff = 0.5
    s.vval = [3.0]
    assert s.ampsine == 2.0 and s.offsine == 0.5
    np.testing.assert_array_equal(s.vval, [3.0])


def test_ytoi_pulse_and_pwl_and_nan():
    out = CurrentSignal.ytoi(np.array([1.0]), I0=1.0, stype="PULSE", ton=1e-6)
    assert out[0].is_pulse()
    out = CurrentSignal.ytoi(np.array([0.1, 0.2]), I0=1.0, stype="PWL", tv=np.array([0.0, 1.0]))
    assert out[0].is_pwl()
    out = CurrentSignal.ytoi(np.array([np.nan]), stype="STEP")
    assert out[0].is_empty()


def test_ytoi_pwl_requires_tv():
    with pytest.raises(ValueError):
        CurrentSignal.ytoi(np.array([1.0]), stype="PWL")


def test_stype_property_and_repr():
    s = VoltageSignal("DC", dcvalue=1.0)
    assert s.stype == "DC"
    assert "VoltageSignal" in repr(s)


def test_get_char_times_alias():
    s = VoltageSignal("PULSE", v1=0, v2=1, td=0.2, ton=0.3)
    np.testing.assert_array_equal(s.get_char_times(), s.breakpoints())
