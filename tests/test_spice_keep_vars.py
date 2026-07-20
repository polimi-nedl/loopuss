"""``raw_parse(keep_vars=...)`` must be a pure column filter.

The selective path skips building the full nvars x npts matrix, so it reads the
payload differently from the reference path. These tests pin the two together on
the same bytes: a .noise raw carries one column per device (N^2 of them for an
NxN array), and only V(onoise) is ever read back, so the fast path is what runs
in anger - it must not drift from the parser it replaces.
"""

import struct

import numpy as np
import pytest
from test_coverage_fill import _binary_raw

from loopuss.spice import raw_parse


def _noise_raw(tmp_path, names, values, freq, double=False):
    """Write a binary .noise raw; ``double`` selects the all-f8 payload variant."""
    if not double:
        f = tmp_path / "noise.raw"
        f.write_bytes(_binary_raw("Noise Spectral Density", "real forward", names, values, freq))
        return f
    nvars, npts = len(names), values.shape[1]
    hdr = (
        f"Title: t\r\nDate: d\r\nPlotname: Noise Spectral Density\r\n"
        f"Flags: real forward double\r\nNo. Variables: {nvars}\r\n"
        f"No. Points: {npts}\r\nVariables:\r\n"
    )
    for i, nm in enumerate(names):
        hdr += f"\t{i}\t{nm}\t{'frequency' if i == 0 else 'voltage'}\r\n"
    hdr += "Binary:\n"
    buf = bytearray(hdr.encode("latin-1"))
    for k in range(npts):
        buf += struct.pack("<d", freq[k])
        for r in range(nvars - 1):
            buf += struct.pack("<d", values[r, k])
    f = tmp_path / "noise_double.raw"
    f.write_bytes(bytes(buf))
    return f


# Stand-ins for the real thing: V(onoise) is the wanted column, the r* columns
# are the per-device contributions that dominate a real .noise raw.
NAMES = ["frequency", "V(onoise)", "V(inoise)", "V(r1)", "V(r2)"]
VALS = np.array([[1e-9, 2e-9, 3e-9], [4e-9, 5e-9, 6e-9], [7e-9, 8e-9, 9e-9], [1e-8, 2e-8, 3e-8]])
FREQ = np.array([1.0, 10.0, 100.0])


@pytest.mark.parametrize("double", [False, True])
def test_keep_vars_matches_full_parse(tmp_path, double):
    f = _noise_raw(tmp_path, NAMES, VALS, FREQ, double=double)
    full = raw_parse(str(f))
    one = raw_parse(str(f), keep_vars=["V(onoise)"])

    assert one["variable_name_list"] == ["V(onoise)"]
    assert one["num_data_pnts"] == full["num_data_pnts"]
    # Identical bytes in, so identical values out - not merely close.
    np.testing.assert_array_equal(one["freq_vect"], full["freq_vect"])
    i = full["variable_name_list"].index("V(onoise)")
    np.testing.assert_array_equal(one["variable_mat"][0], full["variable_mat"][i])


def test_keep_vars_preserves_requested_order(tmp_path):
    f = _noise_raw(tmp_path, NAMES, VALS, FREQ)
    full = raw_parse(str(f))
    sub = raw_parse(str(f), keep_vars=["V(r2)", "V(onoise)"])

    assert sub["variable_name_list"] == ["V(r2)", "V(onoise)"]
    for r, nm in enumerate(sub["variable_name_list"]):
        i = full["variable_name_list"].index(nm)
        np.testing.assert_array_equal(sub["variable_mat"][r], full["variable_mat"][i])


def test_keep_vars_is_case_insensitive(tmp_path):
    f = _noise_raw(tmp_path, NAMES, VALS, FREQ)
    got = raw_parse(str(f), keep_vars=["v(ONOISE)"])
    ref = raw_parse(str(f), keep_vars=["V(onoise)"])
    np.testing.assert_array_equal(got["variable_mat"], ref["variable_mat"])


def test_keep_vars_drops_absent_names(tmp_path):
    # Silently dropped rather than raised: the caller's own lookup names the
    # variable it wanted, which is a better error than one from in here.
    f = _noise_raw(tmp_path, NAMES, VALS, FREQ)
    res = raw_parse(str(f), keep_vars=["V(onoise)", "V(nope)"])
    assert res["variable_name_list"] == ["V(onoise)"]
    assert res["num_data_pnts"] == 3  # still truthy -> no spurious retry


def test_keep_vars_on_binary_tran(tmp_path):
    names = ["time", "V(a)", "V(b)"]
    vals = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
    axis = np.array([0.0, 1e-6, 2e-6])
    f = tmp_path / "tran.raw"
    f.write_bytes(_binary_raw("Transient Analysis", "real forward", names, vals, axis))
    res = raw_parse(str(f), keep_vars=["V(b)"])
    assert res["variable_name_list"] == ["V(b)"]
    np.testing.assert_allclose(res["time_vect"], axis)
    np.testing.assert_allclose(res["variable_mat"][0], vals[1], rtol=1e-6)


def test_keep_vars_on_ascii_op(tmp_path):
    raw = (
        "Title: t\r\nDate: d\r\nPlotname: Operating Point\r\n"
        "Flags: real\r\nNo. Variables: 2\r\nNo. Points: 1\r\n"
        "Variables:\r\n\t0\tV(a)\tvoltage\r\n\t1\tV(b)\tvoltage\r\n"
        "Values:\r\n0\t-0.4\t-0.8\r\n"
    )
    f = tmp_path / "op.raw"
    f.write_bytes(raw.encode("latin-1"))
    res = raw_parse(str(f), keep_vars=["V(b)"])
    assert res["variable_name_list"] == ["V(b)"]
    np.testing.assert_allclose(res["variable_mat"].ravel(), [-0.8])


def test_keep_vars_on_ascii_tran(tmp_path):
    raw = (
        "Title: t\r\nDate: d\r\nPlotname: Transient Analysis\r\n"
        "Flags: real forward\r\nNo. Variables: 3\r\nNo. Points: 2\r\n"
        "Variables:\r\n\t0\ttime\ttime\r\n\t1\tV(a)\tvoltage\r\n"
        "\t2\tV(b)\tvoltage\r\n"
        "Values:\r\n0\t0.0\t1.0\t2.0\r\n1\t1e-6\t1.5\t2.5\r\n"
    )
    f = tmp_path / "tran.raw"
    f.write_bytes(raw.encode("latin-1"))
    res = raw_parse(str(f), keep_vars=["V(b)"])
    assert res["variable_name_list"] == ["V(b)"]
    np.testing.assert_allclose(res["variable_mat"][0], [2.0, 2.5])
