"""Unit tests for the peripheral models: PeriphCirc base, converters,
OAModel, MemoryModel, MemWriteModel, DSPModel, ADCModel, DACs."""

import numpy as np
import pytest

from loopuss import (
    ADCModel,
    CrossBar,
    CurrentDAC,
    DSPModel,
    MemoryModel,
    MemWriteModel,
    OAModel,
    TechModel,
    VoltageDAC,
)


# --- PeriphCirc base (via ADCModel) ---
def test_energy_tracking_and_reset():
    adc = ADCModel("Fan19")
    adc.add_energy("Dynamic", 3.0)
    adc.add_energy("Static", 1.0)
    assert adc.energy["Total"] == 4.0
    adc.reset()
    assert adc.energy["Total"] == 0.0


def test_scale_tech_changes_area():
    adc = ADCModel("Fan19")  # 28 nm
    a0 = adc.area
    adc._scale_tech(TechModel(node=14e-9))
    assert adc.area != a0


# --- ConverterModel.quantclip_array ---
def test_quantclip_infinite_nbit_infinite_fsr_passthrough():
    adc = ADCModel()  # CustomADC: Nbit inf, Vref inf
    x = np.array([[0.1, -0.2, 0.3]])
    np.testing.assert_array_equal(adc.quantclip(x), x)


def test_quantclip_finite_nbit_levels():
    adc = ADCModel(Nbit=2, Vref=[0.0, 1.0])  # 4 levels over [0,1]
    x = np.array([[0.0, 0.24, 0.26, 0.99]])
    out = adc.quantclip(x)
    # values snap to multiples of LSB = 1/4
    for v in out.ravel():
        assert np.isclose((v / 0.25) % 1, 0, atol=1e-9)


def test_quantclip_clips_to_fsr():
    adc = ADCModel(Vref=[-1.0, 1.0])  # inf Nbit -> pure clip
    x = np.array([[-5.0, 0.0, 5.0]])
    np.testing.assert_array_equal(adc.quantclip(x), [[-1.0, 0.0, 1.0]])


# --- OAModel ---
def test_oa_default_is_1p():
    oa = OAModel("1P")
    assert oa.type == "1P"
    assert oa.P0 == pytest.approx(oa.A0 * oa.FP[0])


def test_oa_gbwp_consistency_update():
    # Supplying A0 and FP that disagree with GBWP updates GBWP.
    oa = OAModel("1P", A0=1e4, FP=5e3)
    assert oa.GBWP == pytest.approx(1e4 * 5e3)


def test_oa_preset_values():
    oa = OAModel("Sancandi21")
    assert oa.Vos != 0
    assert np.any(np.isfinite(oa.Iout))


def test_oa_iout_scalar_becomes_symmetric():
    oa = OAModel("1P", Iout=3e-3)
    np.testing.assert_array_equal(oa.Iout, [-3e-3, 3e-3])


# --- MemoryModel ---
def test_memory_default_ideal():
    m = MemoryModel()
    assert m.type == "Ideal"
    assert m.LRS == 1e-3


def test_memory_window_from_lrs_hrs():
    m = MemoryModel(LRS=100e-6, HRS=1e-6)
    assert m.window == pytest.approx(100.0)


def test_memory_preset_and_area():
    m = MemoryModel("SRAM")
    assert m.type == "SRAM"
    assert m.area > 0


def test_memory_area_scales_with_tech():
    base = MemoryModel("RRAM")
    scaled = MemoryModel("RRAM", technology=TechModel(node=45e-9))
    assert scaled.area != base.area


# --- MemWriteModel ---
def test_memwrite_program_no_noise_preserves_conductance():
    mw = MemWriteModel("SRAM", SigmaG=0.0)
    G = np.array([[1e-4, 2e-4], [3e-4, 4e-4]])
    Y, T, E, Ed, Es = mw.program(CrossBar(G))
    np.testing.assert_allclose(Y.G, G, rtol=1e-9)
    assert T >= 0 and E == pytest.approx(Ed + Es)


def test_memwrite_program_adds_energy():
    mw = MemWriteModel("IGVVA10", SigmaG=0.0)
    before = mw.energy["Total"]
    mw.program(CrossBar(np.array([[1e-4, 1e-4]])))
    assert mw.energy["Total"] >= before


def test_memwrite_absolute_noise_changes_values():
    mw = MemWriteModel("SRAM", SigmaG=1e-5)
    np.random.seed(0)
    G = np.full((4, 4), 1e-3)
    Y, *_ = mw.program(CrossBar(G))
    assert not np.allclose(Y.G, G)


def test_memwrite_empty_crossbar():
    mw = MemWriteModel("SRAM")
    Y, T, E, Ed, Es = mw.program(CrossBar(np.empty((2, 0))))
    assert T == 0 and E == 0


# --- DSPModel ---
def test_dsp_eval_identity():
    dsp = DSPModel()
    x = np.arange(5.0)
    y, T, Ed = dsp.eval(x)
    np.testing.assert_array_equal(y, x)
    assert T == 0 and Ed == 0


def test_dsp_eval_custom_function_and_energy():
    dsp = DSPModel(DSPFunction=lambda v: v * 2, complexity=10.0)
    dsp.dynamic_energy = 1e-12
    dsp.latency = 1e-9
    y, T, Ed = dsp.eval(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y, [2.0, 4.0])
    assert T == pytest.approx(1e-9 * 10) and Ed == pytest.approx(1e-12 * 10)
    assert dsp.ops == 10.0


# --- ADCModel ---
def test_adc_convert_returns_latency_energy():
    adc = ADCModel("Fan19")
    y, lat, ed = adc.convert(np.array([[0.1, 0.2]]))
    assert lat == adc.latency and ed == adc.dynamic_energy
    assert adc.energy["Dynamic"] == pytest.approx(ed)


def test_adc_preset_fields():
    adc = ADCModel("Tang21")
    assert adc.Nbit == 10 and adc.Nchannel == 16


# --- DACs ---
def test_voltage_dac_load_and_evaluate():
    from loopuss import VoltageSignal

    dac = VoltageDAC()
    assert not dac.is_loaded()
    dac.load_waveform([VoltageSignal("DC", dcvalue=0.5)])
    assert dac.is_loaded()
    out = dac.todouble(np.array([0.0, 1.0]))
    np.testing.assert_array_equal(out, [[0.5, 0.5]])


def test_voltage_dac_extra_times_within_range():
    from loopuss import VoltageSignal

    dac = VoltageDAC()
    dac.load_waveform([VoltageSignal("PULSE", v1=0, v2=1, td=0.3, ton=0.4)])
    et = dac.extra_times(np.array([0.0, 1.0]))
    assert set(np.round(et, 6)) >= {0.3, 0.7}


def test_current_dac_preset_and_quantclip():
    dac = CurrentDAC("Tseng14")
    assert dac.Nbit == 10
    out = dac.quantclip(np.array([[1e3, -1e3]]))  # huge -> clipped to Iref
    assert np.all(np.abs(out) <= np.max(np.abs(dac.Iref)) + 1e-12)
