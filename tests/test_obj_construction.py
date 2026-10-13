"""Basic object construction tests."""

import math

import numpy as np

from loopuss import (
    ADCModel,
    CrossBar,
    CurrentDAC,
    DSPModel,
    LoopUS,
    MemoryModel,
    MemWriteModel,
    OAModel,
    TechModel,
    VoltageDAC,
)


# --- ADCModel ---
def test_adc_default():
    adc = ADCModel()
    assert math.isinf(adc.Nbit) or adc.Nbit > 0
    assert isinstance(adc.technology, TechModel)


def test_adc_with_technology():
    tech = TechModel(node=14e-9)
    adc = ADCModel(technology=tech)
    assert adc.technology.node == 14e-9


def test_adc_preset_sancandi21():
    adc = ADCModel("Sancandi21")
    assert adc.type == "Sancandi21"


# --- OAModel ---
def test_oa_default():
    oa = OAModel("1P")
    assert oa.type == "1P"
    assert float(np.atleast_1d(oa.A0)[0]) > 0
    assert float(np.atleast_1d(oa.GBWP)[0]) > 0


def test_oa_with_gbwp():
    oa = OAModel("1P", GBWP=1e6)
    assert float(np.atleast_1d(oa.GBWP)[0]) == 1e6


# --- DACs ---
def test_voltage_dac():
    dac = VoltageDAC()
    assert hasattr(dac, "Vref")
    assert hasattr(dac, "Nbit")


def test_voltage_dac_preset():
    dac = VoltageDAC("Sancandi21")
    assert dac.type == "Sancandi21"
    assert dac.Nbit == 7


def test_current_dac_preset():
    dac = CurrentDAC("Tseng14")
    assert dac.type == "Tseng14"
    assert dac.Nbit == 10


# --- MemoryModel / MemWriteModel ---
def test_memory_model_default():
    mm = MemoryModel(LRS=1e-4, HRS=1e-6, ZRS=0.0, window=100, technology=TechModel(), area=1e-12)
    assert mm.LRS == 1e-4
    assert mm.HRS == 1e-6


def test_mem_write_default():
    mw = MemWriteModel()
    assert mw.SigmaG == 0.0


def test_mem_write_preset():
    mw = MemWriteModel("SRAM")
    assert mw.type == "SRAM"


# --- DSPModel ---
def test_dsp_default():
    dsp = DSPModel()
    out, lat, e = dsp.eval(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(out, [1.0, 2.0])


# --- CrossBar ---
def test_crossbar_empty():
    cb = CrossBar(np.empty((2, 0)))
    assert cb.isempty()
    assert cb.M == 2


def test_crossbar_basic():
    G = np.eye(3) * 1e-4
    cb = CrossBar(G)
    assert cb.M == 3 and cb.N == 3
    np.testing.assert_allclose(cb.G, G)


def test_crossbar_transfer_zero_lr():
    """For zero line resistance, V->I col->row transfer == G itself."""
    G = np.array([[1, 2], [3, 4]], dtype=float) * 1e-4
    cb = CrossBar(
        G,
        line_resistance=0.0,
        memory_model=MemoryModel(
            LRS=1.0,
            HRS=0.0,
            ZRS=0.0,
            window=np.inf,
            technology=TechModel(),
            area=0.0,
        ),
        skip_quant=True,
    )
    T = cb.transfer(output_variable="current", transfer_direction=("col", "row"))
    np.testing.assert_allclose(T, G)


# --- LoopUS construction ---
def test_loopus_construct():
    G0 = 1e-4
    Gf = CrossBar(np.eye(2) * G0)
    Gin = CrossBar(np.ones((2, 1)) * G0)
    circ = LoopUS(Gin, Gf, oa_model=OAModel("1P"))
    assert circ.n == 2
    assert circ.p == 1


def test_loopus_construct_feedback_only():
    G0 = 1e-4
    Gf = CrossBar(np.eye(3) * G0)
    circ = LoopUS(None, Gf)
    assert circ.n == 3
    assert circ.p == 0
