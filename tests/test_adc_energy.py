"""ADC conversion energy reaches the benchmark."""

import numpy as np
import pytest

from loopuss import ADCModel, CrossBar, CurrentSignal, LoopUS

N = 4
ADC_ROW = 5  # "ADC" row of the energy breakdown
COMPUTING_DYNAMIC = 3


def _circuit(adc, output_mask=None):
    A = np.eye(N) + 0.1
    circ = LoopUS(
        None, CrossBar(A * 1e-4), adc_model=adc, output_mask=output_mask, bench_enable=True
    )
    circ.set_inputs(CurrentSignal.ytoi(np.linspace(0.1, 0.4, N), I0=1e-4))
    return circ


def _adc_dynamic(circ):
    return circ.energy["Breakdown"][ADC_ROW, COMPUTING_DYNAMIC]


def test_each_masked_output_costs_one_conversion():
    adc = ADCModel("Fan19")
    circ = _circuit(adc, output_mask=[0, 2, 3])
    circ.tran("model", 1e-6)
    assert _adc_dynamic(circ) == pytest.approx(3 * adc.dynamic_energy)
    circ.tran("model", 1e-6)
    assert _adc_dynamic(circ) == pytest.approx(6 * adc.dynamic_energy)
    assert circ.benchmark().energy["Breakdown"][ADC_ROW, COMPUTING_DYNAMIC] > 0


def test_multichannel_adc_charges_every_output():
    # Wang18 serves 2 channels per ADC: 4 outputs need 2 ADCs, 4 conversions.
    adc = ADCModel("Wang18")
    circ = _circuit(adc)
    circ.tran("model", 1e-6)
    assert len(circ._adcs) == 2
    assert _adc_dynamic(circ) == pytest.approx(N * adc.dynamic_energy)


def test_conversion_values_unchanged():
    # Quantization still uses the whole output matrix.
    circ = _circuit(ADCModel("Fan19"))
    vout, _, report = circ.tran("model", 1e-6, return_report=True)
    expected = ADCModel("Fan19").quantclip(report.Vout[circ.output_mask, :])
    np.testing.assert_allclose(vout, expected)


def test_set_output_mask_keeps_accumulated_energy():
    circ = _circuit(ADCModel("Fan19"), output_mask=[0, 1, 2])
    circ.tran("model", 1e-6)
    before = circ.energy["Breakdown"][ADC_ROW].copy()
    assert before[COMPUTING_DYNAMIC] > 0
    circ.set_output_mask([0, 1])
    np.testing.assert_allclose(circ.energy["Breakdown"][ADC_ROW], before)
    assert len(circ._adcs) == 2
