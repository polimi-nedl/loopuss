"""Supply current and leakage power of peripherals.

With finite rails the leakage is always (Vdd - Vss) * Isupply; with unbounded
rails an explicit leakage_power is kept as given.
"""

import numpy as np
import pytest

from loopuss import ADCModel, DSPModel, OAModel, TechModel, VoltageDAC


def test_preset_leakage_matches_publication_values():
    # Presets that store a power figure keep it exactly.
    assert VoltageDAC("Sancandi21").leakage_power == pytest.approx(547.2e-6)
    assert DSPModel("ADSP2148-450MHz").leakage_power == pytest.approx(182e-3)
    assert ADCModel("Kadetotad15").Isupply == pytest.approx(0.6216e-6 / 0.9)


def test_vsupply_override_keeps_current_and_rescales_leakage():
    oa = OAModel("Feinberg21", Vsupply=[-1.0, 1.0])
    assert oa.Isupply == pytest.approx(15e-6)
    assert oa.leakage_power == pytest.approx(2.0 * 15e-6)


def test_isupply_override_changes_leakage():
    adc = ADCModel("Kadetotad15", Isupply=1e-6)
    assert adc.Isupply == pytest.approx(1e-6)
    assert adc.leakage_power == pytest.approx(0.9e-6)


def test_leakage_power_is_converted_to_isupply_with_rails():
    oa = OAModel("1P", Vsupply=1.0, leakage_power=2e-3)
    assert oa.Isupply == pytest.approx(1e-3)
    assert oa.leakage_power == pytest.approx(2e-3)


def test_leakage_power_is_kept_with_unbounded_rails():
    # An op-amp that never saturates can still account for leakage.
    oa = OAModel("1P", leakage_power=1e-6)
    assert np.all(np.isinf(oa.Vsupply))
    assert oa.Isupply == 0.0
    assert oa.leakage_power == pytest.approx(1e-6)


def test_isupply_and_leakage_power_together_raise():
    with pytest.raises(ValueError):
        OAModel("1P", Vsupply=1.0, Isupply=1e-3, leakage_power=2e-3)


def test_isupply_without_rails_raises():
    with pytest.raises(ValueError):
        OAModel("1P", Isupply=1e-3)


def test_leakage_power_is_read_only():
    oa = OAModel("1P", Vsupply=1.0)
    with pytest.raises(AttributeError):
        oa.leakage_power = 1e-3


def test_tech_scaling_scales_leakage_once():
    oa = OAModel("Sancandi21", technology=TechModel(node=45e-9))
    assert oa.leakage_power == pytest.approx(1.2 * 456e-6 * 0.5)
