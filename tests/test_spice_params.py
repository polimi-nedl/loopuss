"""Validation of to_spice netlist generation for the in-scope OA models (1P, GM)
and the resistive crossbar, plus the .raw round-trip via SpiceAnalysis.
"""

import numpy as np
import pytest

import loopuss as lp
from loopuss import spice


def test_1p_model_string_and_params():
    oa = lp.OAModel("1P", GBWP=20e6)
    assert oa.model_str() == "NonLinearOA_1P_Y_Y_Y_Y"
    s = oa.spice_params()
    assert s.startswith("A0=") and "FP1=" in s


def test_1p_with_supply_and_iout():
    oa = lp.OAModel("1P", Vsupply=1.2, Iout=3.6e-3, Rout=920.0, Vos=-96e-6)
    m = oa.model_str()
    assert m == "NonLinearOA_1P_O_Y_V_IR"
    s = oa.spice_params()
    assert "VOS=" in s and "ROUT=" in s and "ISS=" in s and "IDD=" in s


def test_gm_model_uses_gbwp():
    """GM (pure integrator) must emit the GBWP form, not A0/FP."""
    oa = lp.OAModel("GM", GBWP=50e6)
    assert oa.model_str().startswith("NonLinearOA_GM")
    assert oa.spice_params().startswith("GBWP=")


def test_noisy_model_string():
    oa = lp.OAModel("1P", noise_voltage=2.9e-9, noise_current=[14e-12, 3.2e-12])
    assert oa.model_str() == "NonLinearOA_1P_Y_Y_Y_Y_noisy_vinip"
    s = oa.spice_params()
    assert "SVP=" in s and "SIN=" in s and "SIP=" in s


def test_crossbar_reduced_netlist():
    G0 = 1e-4
    cb = lp.CrossBar(np.array([[1.0, 0.0], [2.0, 3.0]]) * G0)
    lines = cb.to_spice("RIN", ["a0", "a1"], ["b0", "b1"])
    # zero-conductance cell (0,1) is skipped; three resistors remain
    assert len(lines) == 3
    assert all(ln.startswith("R") for ln in lines)
    assert "10000" in lines[0]  # 1/(1*G0)


def test_signal_to_spice_types():
    assert lp.VoltageSignal("DC", dcvalue=0.5).to_spice("1", "p", "n") == "V1 p n DC(0.5)"
    assert lp.VoltageSignal("EMPTY").to_spice("1", "p", "n") == ""
    step_sig = lp.VoltageSignal("STEP", v1=0.0, v2=0.3, td=0.0)
    assert step_sig.to_spice("1", "p", "n").startswith("V1 p n PULSE(")
    # .op netlists hold every source at its t = 0 value, the post-step value here
    assert step_sig.to_spice("1", "p", "n", dc_only=True) == "V1 p n DC(0.3)"


def test_spice_analysis_to_double_noise():
    sp = spice.SpiceAnalysis("noise", Fmin=1, Fmax=1e3)
    f = sp.to_double()
    assert f[0] == pytest.approx(1.0) and f[-1] == pytest.approx(1e3)
