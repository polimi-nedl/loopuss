"""Static circuit mappers: linreg, ridgereg, lassoreg, eigdecomp, posneg."""

import numpy as np

from loopuss import eigdecomp, lassoreg, linreg, posneg, ridgereg


def test_linreg_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # m=3, n=2
    M, S = linreg(1.0, X)
    # Block form: [[T, X], [Y=X.T, -D=0]]
    assert M.shape == (5, 5)
    assert S.shape == (5,)
    np.testing.assert_array_equal(M[:3, :3], np.eye(3))
    np.testing.assert_array_equal(M[:3, 3:], X)
    np.testing.assert_array_equal(M[3:, :3], X.T)
    np.testing.assert_array_equal(M[3:, 3:], np.zeros((2, 2)))
    np.testing.assert_array_equal(S, np.array([-1, -1, -1, 1, 1]))


def test_ridgereg_basic():
    X = np.eye(3)
    M, S = ridgereg(1.0, 0.1, X)
    assert M.shape == (6, 6)
    np.testing.assert_array_equal(M[:3, :3], np.eye(3))
    np.testing.assert_array_equal(M[3:, 3:], -0.1 * np.eye(3))


def test_lassoreg_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # m=n=2
    M, S = lassoreg(1.0, 0.1, X)
    # [[T X 0], [Y 0 D], [0 I 0]]
    assert M.shape == (6, 6)
    assert S.shape == (6,)


def test_eigdecomp_basic():
    X = np.eye(3)
    M, S = eigdecomp(X, 1.0, 0.5, 0.01)
    assert M.shape == (12, 12)
    assert len(S) == 12


def test_posneg_basic():
    A = np.array([[1.0, -2.0, 3.0], [-1.0, 0.0, 2.0]])
    # auto_reduce=False keeps B and C the same shape as A
    M, S, B, C, P = posneg(A, auto_reduce=False)
    assert np.all(B >= 0)
    assert np.all(C >= 0)
    np.testing.assert_array_equal(B - C, A)


def test_posneg_autoreduce_prunes_zero_cols():
    """A column where C is all zero is pruned out (column with no negative entries)."""
    A = np.array([[1.0, -2.0, 3.0], [-1.0, 0.0, 2.0]])
    M, S, B, C, P = posneg(A)
    # Last column of A has no negative entries -> C drops a column
    assert C.shape[1] == 2
    # B keeps all columns
    assert B.shape == A.shape


def test_posneg_no_negative():
    """Pure-positive matrix -> C is all zero, prune reduces it to zero columns."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    M, S, B, C, P = posneg(A)
    np.testing.assert_array_equal(B, A)
    # All-zero negative-side columns get pruned
    assert C.shape[1] == 0
