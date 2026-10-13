"""Static circuit mappers.

Each maps a problem statement onto the block conductance matrix ``M`` and the
sign vector ``S`` that a :class:`~loopuss.LoopUS` circuit is built from. With
ideal op-amps the circuit settles on ``M v = -i``, where ``i`` holds the input
currents, so the solution is read from the op-amp outputs ``v``.
"""

from __future__ import annotations

import numpy as np


def linreg(T, X: np.ndarray, Y: np.ndarray | None = None):
    """Least-squares regression mapper.

    Same as ``ridgereg(T, 0, X, Y)``: with ``Y = X.T`` and currents ``y`` on
    the first ``m`` rows, rows ``m:`` settle to ``-inv(X.T X) X.T y``. See
    :func:`ridgereg` for the parameters.
    """
    if Y is None:
        Y = X.T
    return ridgereg(T, 0, X, Y)


def ridgereg(T, D, X: np.ndarray, Y: np.ndarray | None = None):
    """Ridge regression mapper.

    Builds ``M = [[T, X], [Y, -D]]``, with sign ``-1`` on the first ``m`` rows
    and ``+1`` on the last ``n``. Currents ``y`` injected on the first ``m``
    rows give ``v[m:] = -inv(D + Y inv(T) X) Y inv(T) y``. For scalars ``t``
    and ``d`` and ``Y = X.T``, this is ``-inv(X.T X + t d I) X.T y``.

    ``M`` has negative entries, so split it with :func:`posneg` before scaling
    it into a :class:`~loopuss.CrossBar`.

    Parameters
    ----------
    T : float or ndarray, shape (m, m)
        Top-left block; a scalar means ``T * I``.
    D : float or ndarray, shape (n, n)
        Regularization block; a scalar means ``D * I``.
    X : ndarray, shape (m, n)
        Data matrix.
    Y : ndarray, shape (n, m), optional
        Bottom-left block. Default ``X.T``.

    Returns
    -------
    M : ndarray, shape (m + n, m + n)
        Block matrix.
    S : ndarray, shape (m + n,)
        Sign vector.
    """
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
    """LASSO regression mapper.

    Builds ``M = [[T, X, 0], [Y, 0, D], [0, I, 0]]`` with signs ``-1``
    (``m`` rows), ``+1`` (``n`` rows) and ``-1`` (``n`` rows). The last ``n``
    op-amps must be supply-limited: their saturation implements the
    soft-thresholding that makes the solution on rows ``m:m+n`` sparse. See
    the ``lasso_regression`` example.

    Parameters
    ----------
    T : float or ndarray, shape (m, m)
        Top-left block; a scalar means ``T * I``.
    D : float or ndarray, shape (n, n)
        Regularization block; a scalar means ``D * I``.
    X : ndarray, shape (m, n)
        Data matrix.
    Y : ndarray, shape (n, m), optional
        Second-row block. Default ``X.T``.

    Returns
    -------
    M : ndarray, shape (m + 2n, m + 2n)
        Block matrix.
    S : ndarray, shape (m + 2n,)
        Sign vector.
    """
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
    """Eigendecomposition mapper.

    Splits ``X`` into its positive and negative parts ``Xp`` and ``Xn`` and
    builds::

        M = [[t I,  Xp,  0,        Xn + L],
             [Xp.T, d I, Xn.T + L.T, 0   ],
             [I,    0,   I,          0   ],
             [0,    I,   0,          I   ]]

    with signs ``-1`` (``m``), ``+1`` (``n``), ``-1`` (``m``) and ``-1``
    (``n``). The last two block rows are unity inverting buffers that provide
    the negated outputs of the first two blocks.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Matrix to decompose.
    lam : float or ndarray, shape (m, n)
        Eigenvalue shift ``L``; a scalar means ``lam * I`` and needs ``m == n``.
    t, d : float
        Diagonal gains of the first two blocks.

    Returns
    -------
    M : ndarray, shape (2(m + n), 2(m + n))
        Block matrix.
    S : ndarray, shape (2(m + n),)
        Sign vector.
    """
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
    """Split a mixed-sign matrix into non-negative conductances.

    Writes ``A = B - C`` with ``B, C >= 0`` and returns
    ``M = [[B, C], [I, b I]]``. The appended inverting buffer rows produce
    ``-x / b`` for every column, so that ``C`` acts on non-negative
    conductances and the first rows realize ``(B - C / b) x``, i.e. ``A x``
    for ``b = 1``.

    Entries with ``|A| >= splitting_threshold`` are split directly:
    ``B = max(A, 0)`` and ``C = max(-A, 0)``. Smaller entries use
    ``B = 1 - max(-A, 0)`` and ``C = 1 - max(A, 0)``, which keeps both
    conductances near 1 and still gives ``B - C = A``.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Matrix to split.
    b : float, optional
        Buffer feedback gain. Default 1.
    splitting_threshold : float, optional
        Default 0, i.e. direct splitting everywhere.
    sign_vector : ndarray, shape (m,), optional
        Signs of the rows of ``A``; the buffer rows are always ``-1``.
        Default: all ``-1``.
    prog_mask : ndarray of bool, shape (m, n), optional
        Programmable entries of ``A``. Default: all.
    auto_reduce : bool, optional
        Drop the all-zero columns of ``C`` together with their buffers.
        Default ``True``.

    Returns
    -------
    M : ndarray, shape (m + k, n + k)
        Block matrix, where ``k`` is the number of buffers (``n`` without
        reduction).
    S : ndarray, shape (m + k,)
        Sign vector.
    B : ndarray, shape (m, n)
        Positive part.
    C : ndarray, shape (m, k)
        Negative part.
    P : ndarray of bool, shape (m + k, n + k)
        Programming mask of ``M``.
    """
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
