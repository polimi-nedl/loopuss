"""Analysis reports with a non-zero input reference.

The input DACs sit on top of the input reference, so each input column is driven
at DAC + reference.
"""

import numpy as np
import pytest

from loopuss import CrossBar, LoopUS, VoltageSignal
from loopuss.spice import spice_available

N, G0, V0, VREF = 3, 1e-4, 0.1, 0.05
A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
Y = np.array([1.0, 2.0, 3.0])

# Closed form (virtual ground at the junctions): each column draws the sum of its
# conductances times the full column voltage.
VIN = Y * V0 + VREF
I_VIN = (A * G0).sum(axis=0) * VIN


@pytest.fixture
def circ():
    c = LoopUS(
        CrossBar(A * G0),
        CrossBar(np.eye(N) * G0),
        input_reference=VoltageSignal("DC", dcvalue=VREF),
    )
    c.set_inputs(VoltageSignal.ytov(Y, V0=V0, stype="DC"))
    return c


def test_op_report_includes_input_reference(circ):
    _, _, R = circ.op("model", return_report=True)
    np.testing.assert_allclose(R.Vin[:, 0], VIN, atol=1e-12)
    np.testing.assert_allclose(R.I_Vin[:, 0], I_VIN, rtol=1e-3)


def test_tran_report_includes_input_reference(circ):
    _, _, Rop = circ.op("model", return_report=True)
    _, _, R = circ.tran("model", np.logspace(-9, -4, 50), return_report=True)
    np.testing.assert_allclose(R.Vin[:, -1], VIN, atol=1e-12)
    # Settled transient and operating point agree on the input-array quantities.
    np.testing.assert_allclose(R.I_Vin[:, -1], Rop.I_Vin[:, 0], rtol=1e-6)
    np.testing.assert_allclose(R.power["InputArray"][-1], Rop.power["InputArray"][0], rtol=1e-6)


@pytest.mark.skipif(not spice_available(), reason="LTSpice not installed")
def test_op_report_model_vs_spice(circ):
    _, _, Rm = circ.op("model", return_report=True)
    _, _, Rs = circ.op("SPICE/fast", return_report=True)
    np.testing.assert_allclose(Rm.Vin, Rs.Vin, atol=1e-9)
    np.testing.assert_allclose(Rm.I_Vin, Rs.I_Vin, rtol=1e-6)
    np.testing.assert_allclose(Rm.power["InputArray"], Rs.power["InputArray"], rtol=1e-6)


# One reference per column, with a zero one so the netlist must ground only that column.
VREF_COL = np.array([0.05, 0.0, -0.02])
VIN_COL = Y * V0 + VREF_COL


@pytest.fixture
def circ_col():
    c = LoopUS(
        CrossBar(A * G0),
        CrossBar(np.eye(N) * G0),
        input_reference=[VoltageSignal("DC", dcvalue=v) for v in VREF_COL],
    )
    c.set_inputs(VoltageSignal.ytov(Y, V0=V0, stype="DC"))
    return c


def test_single_reference_applies_to_every_column(circ):
    assert len(circ.input_reference) == N
    assert all(r.dcvalue == VREF for r in circ.input_reference)


def test_input_reference_length_mismatch_raises():
    with pytest.raises(ValueError, match="1 or 3"):
        LoopUS(
            CrossBar(A * G0),
            CrossBar(np.eye(N) * G0),
            input_reference=[VoltageSignal("DC", dcvalue=0.1)] * 2,
        )


def test_input_reference_rejects_non_signals(circ):
    with pytest.raises(TypeError):
        circ.input_reference = [0.1, 0.2, 0.3]


def test_per_column_reference_model(circ_col):
    Vout, _, R = circ_col.op("model", return_report=True)
    np.testing.assert_allclose(R.Vin[:, 0], VIN_COL, atol=1e-12)
    # MVM with unit feedback: Vout = -A @ Vin, up to the finite op-amp gain.
    np.testing.assert_allclose(Vout, -A @ VIN_COL, rtol=1e-3)
    _, _, Rt = circ_col.tran("model", np.logspace(-9, -4, 50), return_report=True)
    np.testing.assert_allclose(Rt.Vout[:, -1], R.Vout[:, 0], rtol=1e-6)


@pytest.mark.skipif(not spice_available(), reason="LTSpice not installed")
def test_per_column_reference_model_vs_spice(circ_col):
    Vm, _, Rm = circ_col.op("model", return_report=True)
    Vs, _, Rs = circ_col.op("SPICE/fast", return_report=True)
    np.testing.assert_allclose(Vm, Vs, rtol=1e-6)
    np.testing.assert_allclose(Rm.I_Vin, Rs.I_Vin, rtol=1e-6)


# References are DC only: the model solvers treat them as constant.
NON_DC = [
    VoltageSignal("SINE", voff=0.0, vamp=0.01, freq=1e6),
    VoltageSignal("STEP", v1=0.0, v2=0.05, td=1e-6),
    VoltageSignal("PWL", vval=[0.0, 0.05], time=[0.0, 1e-6]),
]


@pytest.mark.parametrize("ref", NON_DC, ids=lambda r: r.s_type)
@pytest.mark.parametrize("name", ["input_reference", "oa_reference"])
def test_non_dc_reference_raises(name, ref):
    with pytest.raises(ValueError, match="must be a DC signal"):
        LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0), **{name: ref})


@pytest.mark.parametrize("ref", NON_DC, ids=lambda r: r.s_type)
@pytest.mark.parametrize("name", ["input_reference", "oa_reference"])
def test_non_dc_reference_setter_raises(circ, name, ref):
    with pytest.raises(ValueError, match="must be a DC signal"):
        setattr(circ, name, ref)


def test_non_dc_reference_in_per_column_list_raises():
    refs = [VoltageSignal("DC", dcvalue=0.05), NON_DC[0], VoltageSignal("DC", dcvalue=0.0)]
    with pytest.raises(ValueError, match="must be a DC signal"):
        LoopUS(CrossBar(A * G0), CrossBar(np.eye(N) * G0), input_reference=refs)


def test_dc_reference_without_value_raises(circ):
    with pytest.raises(ValueError, match="without a dcvalue"):
        circ.oa_reference = VoltageSignal("DC")


def test_empty_reference_reads_as_zero(circ):
    circ.input_reference = VoltageSignal()
    circ.oa_reference = VoltageSignal()
    assert all(r.is_dc() and r.dcvalue == 0.0 for r in circ.input_reference)
    assert circ.oa_reference.is_dc() and circ.oa_reference.dcvalue == 0.0
    _, _, R = circ.op("model", return_report=True)
    np.testing.assert_allclose(R.Vin[:, 0], Y * V0, atol=1e-12)
