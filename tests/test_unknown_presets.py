"""Unknown preset names raise instead of silently loading defaults."""

import pytest

from loopuss import (
    ADCModel,
    CurrentDAC,
    DSPModel,
    GPUModel,
    MemoryModel,
    MemWriteModel,
    OAModel,
    TechModel,
    VoltageDAC,
)

MODELS = [
    OAModel,
    ADCModel,
    VoltageDAC,
    CurrentDAC,
    MemWriteModel,
    DSPModel,
    TechModel,
    GPUModel,
    MemoryModel,
]


@pytest.mark.parametrize("cls", MODELS)
def test_unknown_preset_raises(cls):
    with pytest.raises(ValueError, match="NotAPreset"):
        cls("NotAPreset")


@pytest.mark.parametrize("cls", MODELS)
def test_default_construction_still_works(cls):
    cls()


@pytest.mark.parametrize(
    ("cls", "name"),
    [(ADCModel, "ideal"), (VoltageDAC, "ideal"), (CurrentDAC, "ideal"), (MemWriteModel, "ideal")],
)
def test_ideal_is_accepted_explicitly(cls, name):
    assert cls(name).type == name


def test_dsp_bypass_is_accepted_explicitly():
    assert DSPModel("bypass").type == "bypass"


def test_renamed_presets_raise():
    with pytest.raises(ValueError):
        GPUModel("Artix7")
    with pytest.raises(ValueError):
        TechModel("PTM-MG-LTSP-14nm")


def test_memory_model_custom_device_needs_no_name():
    with pytest.raises(ValueError):
        MemoryModel("GenericMemory")
    assert MemoryModel(LRS=1e-4, HRS=1e-6).LRS == 1e-4
