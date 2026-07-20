"""Static circuit mappers.

Each maps a problem statement onto the block conductance matrix ``M`` and the
sign vector ``S`` that a :class:`~loopuss.LoopUS` circuit is built from.
"""

from __future__ import annotations

import numpy as np


def linreg(T, X: np.ndarray, Y: np.ndarray | None = None):
    """Linear regression circuit mapper."""
    if Y is None:
        Y = X.T
    return ridgereg(T, 0, X, Y)


def ridgereg(T, D, X: np.ndarray, Y: np.ndarray | None = None):
    """Ridge regression circuit mapper."""
    if Y is None:
        Y = X.T
    if np.isscalar(T):
        T = T * np.eye(X.shape[0])
    if np.isscalar(D):
        D = D * np.eye(X.shape[1])
    assert X.shape == Y.T.shape, "Incompatible sizes of X and Y."
    M = np.block([[T, X], [Y, -D]])
    S = np.concatenate([-np.ones(X.shape[0]), np.ones(Y.shape[0])])
    return M, S


def lassoreg(T, D, X: np.ndarray, Y: np.ndarray | None = None):
    """LASSO regression circuit mapper."""
    if Y is None:
        Y = X.T
    if np.isscalar(T):
        T = T * np.eye(X.shape[0])
    if np.isscalar(D):
        D = D * np.eye(X.shape[1])
    assert X.shape == Y.T.shape, "Incompatible sizes of X and Y."
    m, n = X.shape
    Zn = np.zeros((n, n))
    Znm = np.zeros((n, m))
    Zmn = np.zeros((m, n))
    In = np.eye(n)
    M = np.block([[T, X, Zmn], [Y, Zn, D], [Znm, In, Zn]])
    S = np.concatenate([-np.ones(m), np.ones(n), -np.ones(n)])
    return M, S


def eigdecomp(X: np.ndarray, lam, t: float, d: float):
    """Eigendecomposition circuit mapper."""
    m, n = X.shape
    L = lam * np.eye(m) if np.isscalar(lam) else np.asarray(lam)
    T_mat = t * np.eye(m)
    D_mat = d * np.eye(n)
    Zn = np.zeros((n, n))
    Zm = np.zeros((m, m))
    Zmn = np.zeros((m, n))
    Znm = np.zeros((n, m))
    In = np.eye(n)
    Im = np.eye(m)
    Xp = 0.5 * (np.abs(X) + X)
    Xn = 0.5 * (np.abs(X) - X)
    M = np.block(
        [
            [T_mat, Xp, Zm, Xn + L],
            [Xp.T, D_mat, Xn.T + L.T, Zn],
            [Im, Zmn, Im, Zmn],
            [Znm, In, Znm, In],
        ]
    )
    S = np.concatenate([-np.ones(m), np.ones(n), -np.ones(m), -np.ones(n)])
    return M, S


def posneg(
    A: np.ndarray,
    b: float = 1.0,
    *,
    splitting_threshold: float = 0.0,
    sign_vector: np.ndarray | None = None,
    prog_mask: np.ndarray | None = None,
    auto_reduce: bool = True,
):
    """Mixed-sign matrix circuit mapper."""
    if sign_vector is None:
        sign_vector = -np.ones(A.shape[0])
    if prog_mask is None:
        prog_mask = np.ones(A.shape, dtype=bool)

    A_direct = np.where(np.abs(A) >= splitting_threshold, A, np.nan)
    B_direct = 0.5 * (np.abs(A_direct) + A_direct)
    C_direct = 0.5 * (np.abs(A_direct) - A_direct)

    A_inverse = np.where(np.abs(A) < splitting_threshold, A, np.nan)
    B_inverse = np.ones_like(A) - 0.5 * (np.abs(A_inverse) - A_inverse)
    C_inverse = np.ones_like(A) - 0.5 * (np.abs(A_inverse) + A_inverse)

    B = np.where(~np.isnan(B_direct), B_direct, B_inverse)
    C = np.where(~np.isnan(C_direct), C_direct, C_inverse)
    P = prog_mask.copy()

    I_buf = np.eye(A.shape[1])
    BufIn = I_buf.copy()
    BufF = b * I_buf.copy()
    if auto_reduce:
        p_zero = np.all(C == 0, axis=0)
        BufIn = BufIn[~p_zero, :]
        BufF = BufF[np.ix_(~p_zero, ~p_zero)]
        C = C[:, ~p_zero]
        P = P[:, ~p_zero]

    M = np.block([[B, C], [BufIn, BufF]])
    S = np.concatenate([sign_vector, -np.ones(BufF.shape[0])])
    P_full = np.block([[prog_mask, P], [(BufIn != 0), (BufF != 0)]])
    return M, S, B, C, P_full
