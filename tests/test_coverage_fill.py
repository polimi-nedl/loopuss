"""Targeted tests to exercise less-trodden branches: line-resistance crossbar
transfers, vpart variants, multi-pole OA models, DSP preset, binary .raw
parsing, and assorted edge cases."""

import struct

import numpy as np
import pytest

import loopuss as lp
from loopuss import CrossBar, DSPModel, OAModel
from loopuss.spice import raw_parse


# --- DSP preset ---
def test_dsp_adsp_preset():
    dsp = DSPModel("ADSP2148-450MHz")
    assert dsp.type == "ADSP"
    assert dsp.latency == pytest.approx(1.11e-9)
    assert dsp.leakage_power > 0
    assert dsp.area > 0


# --- multi-pole / RC OA models ---
def test_oa_two_pole_model():
    oa = OAModel("2P", A0=1e4, FP=[1e3, 1e6])
    assert oa.model_str().startswith("NonLinearOA_2P")
    s = oa.spice_params()
    assert "FP1=" in s and "FP2=" in s


def test_oa_two_pole_one_zero():
    oa = OAModel("2P1Z", A0=1e4, FP=[1e3, 1e6], FZ=[1e5])
    assert "2P1Z" in oa.model_str()
    assert "FZ1=" in oa.spice_params()


def test_oa_input_rc_model_string():
    oa = OAModel("1P", input_resistance=1e6, input_capacitance=1e-12)
    m = oa.model_str()
    assert "_RC" in m
    s = oa.spice_params()
    assert "RIN=" in s and "CIN=" in s


def test_oa_rout_only():
    oa = OAModel("1P", Rout=500.0)
    assert oa.model_str() == "NonLinearOA_1P_Y_Y_Y_R"
    assert "ROUT=" in oa.spice_params()


# --- crossbar with line resistance (extended/analytical path) ---
def test_transfer_with_line_resistance_runs():
    G = np.array([[1.0, 2.0], [3.0, 4.0]]) * 1e-4
    cb = CrossBar(G, line_resistance=10.0)
    T = cb.transfer(
        output_variable="current",
        input_variable="voltage",
        transfer_direction=("col", "row"),
        xbar_orient=("bottom", "left"),
    )
    assert T.shape == (2, 2)
    assert np.all(np.isfinite(T))
    # current->voltage transfer also resolves
    Tv = cb.transfer(
        output_variable="voltage",
        input_variable="current",
        transfer_direction=("col", "row"),
        xbar_orient=("bottom", "left"),
    )
    assert np.all(np.isfinite(Tv))


def test_extended_netlist_has_array_cells():
    G = np.array([[1.0, 2.0], [3.0, 4.0]]) * 1e-4
    cb = CrossBar(G, line_resistance=5.0)
    assert cb.netlist_type == "extended"
    lines = cb.to_spice("RF", ["l0", "l1"], ["t0", "t1"], ["r0", "r1"], ["u0", "u1"])
    assert any("ArrayCell_rprc" in ln for ln in lines)


def test_vpart_first_last_and_current():
    gA, gB = 2e-4, 6e-4
    A = CrossBar(np.array([[gA]]))
    B = CrossBar(np.array([[gB]]))
    assert np.all(np.isfinite(np.atleast_1d(CrossBar.vpart(A, B, output_node="first"))))
    assert np.all(np.isfinite(np.atleast_1d(CrossBar.vpart(A, B, output_node="last"))))
    assert np.all(np.isfinite(np.atleast_1d(CrossBar.vpart(A, B, output_variable="current"))))


def test_vpart_full_returns_tuple():
    A = CrossBar(np.array([[2e-4]]))
    B = CrossBar(np.array([[6e-4]]))
    out = CrossBar.vpart_full(A, B)
    assert len(out) == 6


# --- binary .raw parsing ---
def _binary_raw(plotname, flags, names, values, axis=None, double_time=True):
    """Build a minimal LTSpice-IV-style (ASCII header) binary .raw."""
    nvars = len(names)
    npts = values.shape[1]
    hdr = (
        f"Title: t\r\nDate: d\r\nPlotname: {plotname}\r\n"
        f"Flags: {flags}\r\nNo. Variables: {nvars}\r\nNo. Points: {npts}\r\n"
        "Variables:\r\n"
    )
    for i, nm in enumerate(names):
        hdr += f"\t{i}\t{nm}\t{'time' if i == 0 and axis is not None else 'voltage'}\r\n"
    hdr += "Binary:\n"
    buf = bytearray(hdr.encode("latin-1"))
    if axis is not None:  # tran: [f8 axis][f4 var...] per point
        for k in range(npts):
            buf += struct.pack("<d", axis[k])
            for r in range(nvars - 1):
                buf += struct.pack("<f", values[r, k])
    else:  # op: first var f8, rest f4 (single point)
        buf += struct.pack("<d", values[0, 0])
        for r in range(1, nvars):
            buf += struct.pack("<f", values[r, 0])
    return bytes(buf)


def test_raw_parse_binary_tran(tmp_path):
    names = ["time", "V(a)", "V(b)"]
    vals = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
    axis = np.array([0.0, 1e-6, 2e-6])
    f = tmp_path / "tran.raw"
    f.write_bytes(_binary_raw("Transient Analysis", "real forward", names, vals, axis))
    res = raw_parse(str(f))
    assert res["variable_name_list"] == ["V(a)", "V(b)"]
    np.testing.assert_allclose(res["time_vect"], axis)
    np.testing.assert_allclose(res["variable_mat"], vals)


def test_raw_parse_binary_op(tmp_path):
    names = ["V(a)", "V(b)"]
    vals = np.array([[-0.4], [-0.8]])
    f = tmp_path / "op.raw"
    f.write_bytes(_binary_raw("Operating Point", "real", names, vals, axis=None))
    res = raw_parse(str(f))
    assert res["variable_name_list"] == ["V(a)", "V(b)"]
    np.testing.assert_allclose(res["variable_mat"].ravel(), [-0.4, -0.8], rtol=1e-6)


def test_raw_parse_rejects_complex(tmp_path):
    f = tmp_path / "ac.raw"
    f.write_bytes(
        _binary_raw(
            "AC Analysis",
            "complex forward",
            ["freq", "V(a)"],
            np.array([[1.0, 2.0]]),
            axis=np.array([1.0, 2.0]),
        )
    )
    with pytest.raises(ValueError):
        raw_parse(str(f))


# --- LoopUS transfer + noise input schemes ---
def test_transfer_noninverting_matrix():
    N = 3
    circ = lp.LoopUS(CrossBar(np.empty((N, 0))), CrossBar(np.eye(N) * 1e-4), oa_model=OAModel("1P"))
    T = circ.transfer("noninverting")
    assert T.shape == (N, N)
    with pytest.raises(ValueError):
        circ.transfer("bogus")


def test_noise_model_with_explicit_freq_vector():
    N = 3
    circ = lp.LoopUS(
        CrossBar(np.empty((N, 0))),
        CrossBar(np.eye(N) * 1e-4),
        oa_model=OAModel("1P", GBWP=20e6, noise_voltage=2e-9),
    )
    freqs = np.logspace(0, 6, 50)
    S, f, R = circ.noise("model", freqs)
    assert S.shape == (N, 50)
    np.testing.assert_array_equal(f, freqs)


def test_noise_ltspice_style_scalar_args():
    N = 2
    circ = lp.LoopUS(
        CrossBar(np.empty((N, 0))),
        CrossBar(np.eye(N) * 1e-4),
        oa_model=OAModel("1P", noise_voltage=2e-9),
    )
    S, f, R = circ.noise("model", 1e6)  # Fmax as a scalar
    assert f[-1] == pytest.approx(1e6)
