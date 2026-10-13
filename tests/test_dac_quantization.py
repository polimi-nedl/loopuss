"""DAC resolution is applied to the loaded waveforms."""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentDAC, CurrentSignal, LoopUS, VoltageDAC, VoltageSignal


def _expected(dac, value, ref):
    return dac.quantclip_array(np.array([[value]]), np.asarray(ref).reshape(1, 2))[0, 0]


def test_step_levels_are_quantized():
    dac = CurrentDAC("Ishraqul17")  # 8 bits over +-19.5 uA
    dac.load_waveform(CurrentSignal("STEP", i1=0.0, i2=1.2345e-6, td=0.0))
    level = dac.todouble(np.array([1.0]))[0, 0]
    assert level == pytest.approx(_expected(dac, 1.2345e-6, dac.Iref))
    assert level != pytest.approx(1.2345e-6)


def test_dc_and_pwl_values_are_quantized():
    dac = VoltageDAC(Nbit=2, Vref=[0.0, 1.0])  # levels 0, 0.25, 0.5, 0.75
    dac.load_waveform(VoltageSignal("DC", dcvalue=0.3))
    assert dac.todouble(np.array([0.0]))[0, 0] == pytest.approx(0.25)
    dac.load_waveform(VoltageSignal("PWL", vval=[0.1, 0.6, 0.9], time=[0.0, 1.0, 2.0]))
    np.testing.assert_allclose(dac.todouble(np.array([0.5, 1.5, 2.5]))[0], [0.0, 0.5, 0.75])


def test_values_are_clipped_to_the_range():
    dac = CurrentDAC("Ishraqul17")
    dac.load_waveform(CurrentSignal("DC", dcvalue=1.0))  # far above +19.5 uA
    assert dac.todouble(np.array([0.0]))[0, 0] <= dac.Iref[1]


def test_ideal_dac_and_sine_are_unchanged():
    ideal = VoltageDAC()
    ideal.load_waveform(VoltageSignal("DC", dcvalue=0.123456789))
    assert ideal.todouble(np.array([0.0]))[0, 0] == 0.123456789

    coarse = VoltageDAC(Nbit=2, Vref=[-1.0, 1.0])
    sine = VoltageSignal("SINE", vamp=0.3, voff=0.1, freq=1.0)
    coarse.load_waveform(sine)
    t = np.array([0.0, 0.1, 0.2])
    np.testing.assert_allclose(coarse.todouble(t)[0], sine.to_array(t))


def test_loading_does_not_modify_the_given_signal():
    sig = VoltageSignal("DC", dcvalue=0.3)
    VoltageDAC(Nbit=2, Vref=[0.0, 1.0]).load_waveform(sig)
    assert sig.dcvalue == 0.3


def test_netlist_uses_quantized_levels():
    dac = VoltageDAC(Nbit=2, Vref=[0.0, 1.0])
    dac.load_waveform(VoltageSignal("DC", dcvalue=0.3))
    (line,) = dac.to_spice("IN1", ("a", "0"))
    assert "DC(0.25)" in line


def test_quantclip_accepts_signals():
    dac = CurrentDAC(Nbit=2, Iref=[0.0, 1.0])
    step, dc = dac.quantclip(
        [CurrentSignal("STEP", i1=0.0, i2=0.3, td=0.0), CurrentSignal("DC", dcvalue=0.6)]
    )
    assert (step.val1, step.val2, dc.dcvalue) == (0.0, 0.25, 0.5)


def test_set_inputs_quantizes_through_the_circuit_dacs():
    N = 3
    circ = LoopUS(None, CrossBar(np.eye(N) * 1e-4), current_dac=CurrentDAC("Ishraqul17"))
    y = np.array([0.1, 0.2, 0.3]) * 1e-5
    circ.set_inputs(CurrentSignal.ytoi(y))
    dac = circ.current_dac[1]
    level = dac.todouble(np.array([1.0]))[0, 0]
    assert level == pytest.approx(_expected(dac, y[1], dac.Iref))
