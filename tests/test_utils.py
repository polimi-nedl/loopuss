"""Unit tests for loopuss.utils."""

import pytest

from loopuss.utils import _ci_get, si_format, vparse


def test_ci_get_found_and_default():
    found, v = _ci_get({"Foo": 3}, "foo")
    assert found and v == 3
    found, v = _ci_get({"Foo": 3}, "bar", default=9)
    assert not found and v == 9


def test_vparse_case_insensitive():
    assert vparse({"GBWP": 7}, "gbwp") == 7
    assert vparse({"a": 1}, "missing", default="d") == "d"


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        (0, "V", "0 V"),
        (0, "", "0"),
        (1e9, "Hz", "1 GHz"),
        (1e6, "Hz", "1 MHz"),
        (1e3, "Ohm", "1 kOhm"),
        (1.0, "F", "1 F"),
        (1e-3, "A", "1 mA"),
        (1e-6, "s", "1 us"),
        (1e-9, "s", "1 ns"),
        (1e-12, "F", "1 pF"),
        (1e-15, "F", "1 fF"),
    ],
)
def test_si_format(value, unit, expected):
    assert si_format(value, unit) == expected


def test_si_format_negative_and_fractional():
    assert si_format(-2.5e-6, "A") == "-2.5 uA"
    # Below the smallest prefix still uses femto.
    assert si_format(1e-18, "s").endswith("fs")
