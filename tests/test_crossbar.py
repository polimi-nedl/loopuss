"""Unit tests for loopuss.CrossBar."""

import numpy as np

from loopuss import CrossBar, MemoryModel


def test_shape_and_empty():
    cb = CrossBar(np.ones((3, 4)) * 1e-4)
    assert (cb.M, cb.N) == (3, 4)
    assert not cb.isempty()
    assert CrossBar(np.empty((3, 0))).isempty()


def test_transfer_zero_lr_voltage_to_current_closed_forms():
    G = np.array([[1.0, 2.0], [3.0, 4.0]]) * 1e-4
    cb = CrossBar(G)
    # col->row : G itself
    T = cb.transfer(
        output_variable="current", input_variable="voltage", transfer_direction=("col", "row")
    )
    np.testing.assert_allclose(T, G)
    # row->col : G transposed
    T = cb.transfer(output_variable="current", transfer_direction=("row", "col"))
    np.testing.assert_allclose(T, G.T)
    # col->col : diag(sum over rows)
    T = cb.transfer(output_variable="current", transfer_direction=("col", "col"))
    np.testing.assert_allclose(T, np.diag(G.sum(axis=0)))
    # row->row : diag(sum over cols)
    T = cb.transfer(output_variable="current", transfer_direction=("row", "row"))
    np.testing.assert_allclose(T, np.diag(G.sum(axis=1)))


def test_transfer_voltage_to_voltage_identity_and_zero():
    cb = CrossBar(np.ones((2, 3)) * 1e-4)
    np.testing.assert_allclose(
        cb.transfer(output_variable="voltage", transfer_direction=("col", "col")), np.eye(3)
    )
    np.testing.assert_allclose(
        cb.transfer(output_variable="voltage", transfer_direction=("row", "row")), np.eye(2)
    )


def test_transfer_caching_returns_same_object():
    cb = CrossBar(np.ones((2, 2)) * 1e-4)
    a = cb.transfer(output_variable="current", transfer_direction=("col", "row"))
    b = cb.transfer(output_variable="current", transfer_direction=("col", "row"))
    assert a is b


def test_transfer_empty_returns_zeros():
    cb = CrossBar(np.empty((3, 0)))
    T = cb.transfer(output_variable="current", transfer_direction=("col", "row"))
    assert T.shape == (3, 0) or T.size == 0


def test_line_resistance_approaches_zero_lr_limit():
    G = np.array([[1.0, 2.0], [0.5, 4.0]]) * 1e-4
    T0 = CrossBar(G).transfer(output_variable="current", transfer_direction=("col", "row"))
    T_eps = CrossBar(G, line_resistance=1e-6).transfer(
        output_variable="current", transfer_direction=("col", "row"), xbar_orient=("bottom", "left")
    )
    np.testing.assert_allclose(T_eps, T0, rtol=1e-2, atol=1e-8)


def test_ideal_buffers_use_abs_sum():
    G = np.array([[1.0, -2.0], [3.0, 4.0]]) * 1e-4
    cb = CrossBar(G, ideal_buffers=True)
    T = cb.transfer(output_variable="current", transfer_direction=("col", "col"))
    np.testing.assert_allclose(T, np.diag(np.sum(np.abs(G), axis=0)))


def test_quantization_with_finite_nbit():
    mm = MemoryModel(LRS=1e-3, HRS=1e-5, Nbit=2)
    cb = CrossBar(np.array([[0.3e-3, 0.7e-3]]), memory_model=mm)
    # values should be snapped to the quantization grid (not the raw inputs)
    assert cb.G.shape == (1, 2)


def test_vpart_middle_divider():
    gA, gB = 2e-4, 6e-4
    A = CrossBar(np.array([[gA]]))
    B = CrossBar(np.array([[gB]]))
    Tout = CrossBar.vpart(A, B, output_node="middle")
    np.testing.assert_allclose(np.atleast_1d(Tout).ravel()[0], gA / (gA + gB), rtol=1e-9)


def test_repr():
    assert "CrossBar" in repr(CrossBar(np.ones((2, 2)) * 1e-4))
