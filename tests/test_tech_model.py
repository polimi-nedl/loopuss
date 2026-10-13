"""Unit tests for loopuss.TechModel."""

import pytest

from loopuss import TechModel


def test_default_custom():
    t = TechModel()
    assert t.type == "Custom"
    assert t.node == 14e-9
    assert t.Vdd == 0.8


def test_preset_loading():
    t = TechModel("PTM-MG-HP-7nm")
    assert t.type == "PTM-MG-HP-7nm"
    assert t.node == 7e-9
    assert t.Vdd == 0.70
    assert t.Rswn == 50


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="MyFab"):
        TechModel("MyFab")


def test_kwargs_override():
    t = TechModel("PTM-HP-45nm", Vdd=1.5)
    assert t.Vdd == 1.5


def test_effective_node_multipatterning():
    assert TechModel(node=14e-9).effective_node == 42e-9
    assert TechModel(node=16e-9).effective_node == 64e-9
    assert TechModel(node=45e-9).effective_node == 45e-9  # no remap


def test_resistor_size_value_roundtrip():
    t = TechModel("PTM-HP-45nm")
    R = 10e3
    W, L, A = t.resistor_size(R, "n")
    assert A == pytest.approx(W * L)
    assert t.resistor_value(W, L, "n") == pytest.approx(R)


def test_resistor_p_vs_n_type():
    t = TechModel("PTM-HP-22nm")  # Rswn != Rswp
    _, Ln, _ = t.resistor_size(1e3, "n")
    _, Lp, _ = t.resistor_size(1e3, "p")
    assert Ln != Lp


def test_repr():
    assert "7nm" in repr(TechModel("PTM-MG-HP-7nm"))
