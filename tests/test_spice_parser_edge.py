"""Edge-format coverage for the .raw parser: ASCII .op and double-flag binary."""

import struct

import numpy as np

from loopuss.spice import raw_parse


def test_raw_parse_ascii_op(tmp_path):
    raw = (
        "Title: t\r\nDate: d\r\nPlotname: Operating Point\r\n"
        "Flags: real\r\nNo. Variables: 2\r\nNo. Points: 1\r\n"
        "Variables:\r\n\t0\tV(a)\tvoltage\r\n\t1\tV(b)\tvoltage\r\n"
        "Values:\r\n0\t-0.4\t-0.8\r\n"
    )
    f = tmp_path / "op.raw"
    f.write_bytes(raw.encode("latin-1"))
    res = raw_parse(str(f))
    assert res["variable_name_list"] == ["V(a)", "V(b)"]
    np.testing.assert_allclose(res["variable_mat"].ravel(), [-0.4, -0.8])


def test_raw_parse_binary_double_flag_tran(tmp_path):
    axis = np.array([0.0, 1e-6])
    vals = np.array([[1.0, 1.5], [2.0, 2.5]])
    hdr = (
        "Title: t\r\nDate: d\r\nPlotname: Transient Analysis\r\n"
        "Flags: real forward double\r\nNo. Variables: 3\r\nNo. Points: 2\r\n"
        "Variables:\r\n\t0\ttime\ttime\r\n\t1\tV(a)\tvoltage\r\n"
        "\t2\tV(b)\tvoltage\r\nBinary:\n"
    )
    buf = bytearray(hdr.encode("latin-1"))
    for k in range(2):  # all values float64 (double flag)
        buf += struct.pack("<d", axis[k])
        buf += struct.pack("<d", vals[0, k])
        buf += struct.pack("<d", vals[1, k])
    f = tmp_path / "trand.raw"
    f.write_bytes(bytes(buf))
    res = raw_parse(str(f))
    np.testing.assert_allclose(res["time_vect"], axis)
    np.testing.assert_allclose(res["variable_mat"], vals)


def test_raw_parse_appends_raw_suffix(tmp_path):
    raw = (
        "Title: t\r\nDate: d\r\nPlotname: Operating Point\r\n"
        "Flags: real\r\nNo. Variables: 1\r\nNo. Points: 1\r\n"
        "Variables:\r\n\t0\tV(a)\tvoltage\r\n"
        "Values:\r\n0\t1.25\r\n"
    )
    (tmp_path / "noext.raw").write_bytes(raw.encode("latin-1"))
    # pass the path WITHOUT the .raw suffix -> parser appends it
    res = raw_parse(str(tmp_path / "noext"))
    np.testing.assert_allclose(res["variable_mat"].ravel(), [1.25])
