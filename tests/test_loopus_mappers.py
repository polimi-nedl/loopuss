"""Unit tests for the static circuit mappers on LoopUS:
linreg / ridgereg / lassoreg / eigdecomp / posneg."""

import numpy as np

from loopuss import eigdecomp, lassoreg, linreg, posneg, ridgereg


def test_ridgereg_block_structure():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    M, S = ridgereg(1.0, 0.5, X)
    m, n = X.shape
    assert M.shape == (m + n, m + n)
    assert S.shape == (m + n,)
    # top-left block is T = 1*I(m); bottom-right is -D = -0.5*I(n)
    np.testing.assert_array_equal(M[:m, :m], np.eye(m))
    np.testing.assert_array_equal(M[m:, m:], -0.5 * np.eye(n))
    # off-diagonal blocks are X and X.T
    np.testing.assert_array_equal(M[:m, m:], X)
    np.testing.assert_array_equal(M[m:, :m], X.T)
    np.testing.assert_array_equal(S, np.concatenate([-np.ones(m), np.ones(n)]))


def test_linreg_is_ridge_with_zero_d():
    X = np.array([[1.0, 0.5], [0.2, 1.0], [0.3, 0.7]])
    M1, S1 = linreg(1.0, X)
    M2, S2 = ridgereg(1.0, 0.0, X)
    np.testing.assert_array_equal(M1, M2)
    np.testing.assert_array_equal(S1, S2)


def test_ridgereg_solves_regression():
    """The mapped system M*v = rhs reproduces the ridge solution."""
    rng = np.random.default_rng(1)
    X = rng.random((6, 3))
    y = rng.random(6)
    lam = 0.7
    M, S = ridgereg(np.sqrt(lam), np.sqrt(lam), X)
    m, n = X.shape
    # solve the linear system the circuit represents:  [T X; Y -D] [u; w] = [-y; 0]
    rhs = np.concatenate([-y, np.zeros(n)])
    sol = np.linalg.solve(M, rhs)
    w = sol[m:]
    ref = -np.linalg.solve(X.T @ X + lam * np.eye(n), X.T @ y)
    np.testing.assert_allclose(w, ref, rtol=1e-8)


def test_lassoreg_shapes():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    M, S = lassoreg(1.0, 0.5, X)
    m, n = X.shape
    assert M.shape == (m + 2 * n, m + 2 * n)
    assert S.shape == (m + 2 * n,)


def test_eigdecomp_shapes_and_signs():
    # eigendecomposition is defined for a square matrix
    X = np.array([[1.0, 2.0, 0.5], [0.3, 5.0, 1.0], [0.1, 0.2, 4.0]])  # 3x3
    M, S = eigdecomp(X, lam=0.1, t=1.0, d=1.0)
    m, n = X.shape
    assert M.shape == (2 * m + 2 * n, 2 * m + 2 * n)
    assert S.shape == (2 * m + 2 * n,)


def test_posneg_splits_sign():
    A = np.array([[1.0, -2.0], [-3.0, 4.0]])
    M, S, B, C, P = posneg(A, 1.0, auto_reduce=False)
    # B is the positive part, C the negative part; B - C == A
    np.testing.assert_allclose(B - C, A)
    assert np.all(B >= 0) and np.all(C >= 0)
    # all-positive conductances in the mapped matrix
    assert np.all(M >= -1e-12)


def test_posneg_auto_reduce_drops_zero_negative_columns():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])  # all positive -> C all zero
    M_full, *_ = posneg(A, 1.0, auto_reduce=False)
    M_red, *_ = posneg(A, 1.0, auto_reduce=True)
    assert M_red.shape[1] < M_full.shape[1]
