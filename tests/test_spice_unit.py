"""Unit tests for loopuss.spice (SpiceAnalysis, .raw parser, helpers) and the
LoopUS netlist helper functions, without invoking LTSpice."""

import numpy as np
import pytest

from loopuss import spice
from loopuss.netlist import _vto_supply_net
from loopuss.spice import SpiceAnalysis, _natkey, _spice_node_matrix, raw_parse


# --- SpiceAnalysis ---
def test_to_spice_op_tran_noise():
    assert SpiceAnalysis("op").to_spice() == ".op"
    assert SpiceAnalysis("tran", Tstop=1e-5).to_spice().startswith(".tran")
    s = SpiceAnalysis("noise", Fmax=1e6).to_spice(("V(o)", "VNOISE"))
    assert s.startswith(".noise V(o) VNOISE dec")


def test_tstep_default():
    sp = SpiceAnalysis("tran", Tstart=0, Tstop=1.0)
    assert sp.Tstep == pytest.approx(0.01)


def test_to_double_tran_and_op():
    assert SpiceAnalysis("op").to_double().tolist() == [0.0]
    v = SpiceAnalysis("tran", Tstart=0, Tstop=1e-6, Tstep=1e-7).to_double()
    assert v[0] == 0.0 and v[-1] == pytest.approx(1e-6)


def test_to_double_noise_dec_oct_lin():
    dec = SpiceAnalysis("noise", Fmin=1, Fmax=1e3, Fstep=100, Fsweep="dec").to_double()
    assert dec[0] == pytest.approx(1.0) and dec[-1] == pytest.approx(1e3)
    octv = SpiceAnalysis("noise", Fmin=1, Fmax=1024, Fstep=10, Fsweep="oct").to_double()
    assert octv[0] == pytest.approx(1.0)
    lin = SpiceAnalysis("noise", Fmin=1, Fmax=10, Fstep=1, Fsweep="lin").to_double()
    assert lin[0] == pytest.approx(1.0) and lin[-1] == pytest.approx(10.0)


def test_validate_fixes_fmax_and_rejects_bad_type():
    sp = SpiceAnalysis("noise", Fmin=100, Fmax=10)
    assert sp.Fmax == 1000  # 10*Fmin
    with pytest.raises(ValueError):
        SpiceAnalysis("dc")


def test_is_helpers():
    assert SpiceAnalysis("op").is_op()
    assert SpiceAnalysis("tran").is_tran()
    assert SpiceAnalysis("noise").is_noise()


# --- helpers ---
def test_natkey_natural_order():
    names = ["v(vout_10)", "v(vout_2)", "v(vout_1)"]
    assert sorted(names, key=_natkey) == ["v(vout_1)", "v(vout_2)", "v(vout_10)"]


@pytest.mark.parametrize(
    "V,expected",
    [
        (1.2, "+1V200"),
        (-0.8, "-0V800"),
        (0.0, "+0V0"),
        (-1.5, "-1V500"),
    ],
)
def test_vto_supply_net(V, expected):
    assert _vto_supply_net(V) == expected


def test_vto_supply_net_inf_is_none():
    assert _vto_supply_net(np.inf) is None


def test_spice_node_matrix_lookup_case_insensitive():
    res = {
        "variable_name_list": ["V(a)", "V(b)", "I(x)"],
        "variable_mat": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    }
    M = _spice_node_matrix(res, ["v(b)", "V(A)"])
    np.testing.assert_array_equal(M, [[3.0, 4.0], [1.0, 2.0]])
    with pytest.raises(KeyError):
        _spice_node_matrix(res, ["V(missing)"])


# --- .raw parser round-trip (ASCII) ---
_ASCII_RAW = (
    "Title: * test\r\n"
    "Date: now\r\n"
    "Plotname: Transient Analysis\r\n"
    "Flags: real forward\r\n"
    "No. Variables: 3\r\n"
    "No. Points: 2\r\n"
    "Variables:\r\n"
    "\t0\ttime\ttime\r\n"
    "\t1\tV(a)\tvoltage\r\n"
    "\t2\tV(b)\tvoltage\r\n"
    "Values:\r\n"
    "0\t0.0\t1.0\t2.0\r\n"
    "1\t1e-06\t1.5\t2.5\r\n"
)


def test_raw_parse_ascii_tran(tmp_path):
    f = tmp_path / "sim.raw"
    f.write_bytes(_ASCII_RAW.encode("latin-1"))
    res = raw_parse(str(f))
    assert res["plotname"].lower().startswith("transient")
    assert res["variable_name_list"] == ["V(a)", "V(b)"]
    np.testing.assert_allclose(res["time_vect"], [0.0, 1e-6])
    np.testing.assert_allclose(res["variable_mat"], [[1.0, 1.5], [2.0, 2.5]])


def test_raw_parse_missing_marker_raises(tmp_path):
    f = tmp_path / "bad.raw"
    f.write_bytes(b"Title: nope\r\nNo data here\r\n")
    with pytest.raises(ValueError):
        raw_parse(str(f))


def test_spice_available_is_bool():
    assert isinstance(spice.spice_available(), bool)
