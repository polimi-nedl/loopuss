"""Nonlinear behavioural solvers (operating point + transient)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from .loopus import LoopUS


def _vnonlin(sat_id: np.ndarray, vsupply: np.ndarray) -> np.ndarray:
    """Return saturation voltages for saturated OAs, in OA-index order.
    sat_id: (n,) int array  {-1, 0, +1}
    vsupply: (n, 2) array   [Vss, Vdd] per OA
    Returns Vnlin: rail voltage (Vss if sat_id<0 else Vdd) for each saturated
    OA, ordered by OA index so it lines up with the positional assignment in
    ``_nlstate``.

    Rails must be taken one per saturated OA in index order; a naive NumPy
    ``vs[mask]`` is row-major and instead returns all low rails followed by all
    high rails, mis-assigning rails when saturated OAs have interleaved mixed
    signs.
    """
    sat_id = np.asarray(sat_id)
    rail = np.where(sat_id < 0, vsupply[:, 0], vsupply[:, 1])  # Vss if -, else Vdd
    return rail[sat_id != 0]


def _nlstate(Vlin: np.ndarray, Vnlin: np.ndarray, sat_id: np.ndarray) -> np.ndarray:
    """Assemble full-state vector from linear solution + saturation clamps."""
    n = len(sat_id)
    nt = Vlin.shape[1] if Vlin.ndim > 1 else 1
    if Vlin.ndim == 1:
        Vlin = Vlin.reshape(-1, 1)
    V = np.zeros((n, nt))
    lin_mask = sat_id == 0
    sat_mask = sat_id != 0
    V[lin_mask, :] = Vlin
    if np.any(sat_mask):
        V[sat_mask, :] = Vnlin[:, np.newaxis] * np.ones((1, nt))
    return V if nt > 1 else V.ravel()


def _lintran(
    t: np.ndarray, W: np.ndarray, d: np.ndarray, Wdiff: np.ndarray, Wende: np.ndarray
) -> np.ndarray:
    """Modal transient projection.
    t: (T,) time array (local, starting at 0)
    W: (n, n) eigenvector matrix
    d: (n,) eigenvalues
    Wdiff, Wende: (n, cols) initial-condition and steady-state modal vectors
    Returns V: (n, T) voltage matrix.
    """
    if d.size == 0:
        return np.empty((0, len(t)))
    t = np.asarray(t, dtype=float).ravel()
    # exp(d * t): (n, T)
    V = np.exp(d[:, np.newaxis] * t[np.newaxis, :]) * Wdiff
    # Avoid Inf * 0 - need 1D row-mask to avoid 3D indexing on (n, T) array
    zero_rows = np.all(np.asarray(Wdiff) == 0, axis=1) if np.ndim(Wdiff) > 1 else (Wdiff == 0)
    V[zero_rows, :] = 0.0
    return np.real(W @ (V + Wende))


def _endstat(
    T: list, A: np.ndarray, sat_id: np.ndarray, iin, vin, vsat, vbiasp=None, vbiasn=None
) -> np.ndarray:
    """Compute steady-state OA output vector.
    T: list of 10 transfer matrices (0-indexed, i.e. T[0]..T[9])
    A:   (n,) signed gain array  S * A0
    sat_id: (n,) int array {-1, 0, +1}
    iin: (n, cols) or None
    vin: (p, cols) or None
    vsat: saturation voltages for saturated OAs (length nnz(sat_id!=0))
    vbiasp: (n,) OA+ bias (Vos + OARef)
    vbiasn: (p,) input bias or None
    Returns v: (nlin, cols) steady-state output for linear OAs.
    """
    if vbiasp is None:
        vbiasp = np.zeros(len(sat_id))
    if vbiasn is None:
        vbiasn = None

    lin_id = sat_id == 0  # bool mask
    sat_id_b = sat_id != 0  # bool mask
    n_lin = int(np.sum(lin_id))

    # All OAs saturated -> no linear sub-system to solve; the caller fills the
    # full state from the saturation clamps. Return an empty linear solution.
    if n_lin == 0:
        return np.zeros(0)

    v = np.zeros((n_lin, 1))

    has_input = (
        iin is not None
        or vin is not None
        or (vsat is not None and len(vsat) > 0)
        or vbiasn is not None
    )

    if has_input:
        # Tinv = (T2[l,l] + T4[l,l]) * (T9[l,l] \ (diag(1/A[l]) - T10[l,l])) - T3[l,l]
        T2l = T[1][np.ix_(lin_id, lin_id)]
        T4l = T[3][np.ix_(lin_id, lin_id)]
        T9l = T[8][np.ix_(lin_id, lin_id)]
        T10l = T[9][np.ix_(lin_id, lin_id)]
        T3l = T[2][np.ix_(lin_id, lin_id)]

        inv_A_lin = np.diag(1.0 / A[lin_id])
        inner = scipy.linalg.solve(T9l, inv_A_lin - T10l)
        Tinv = (T2l + T4l) @ inner - T3l

        ieval = np.zeros((n_lin, 1))

        if iin is not None and np.asarray(iin).size > 0:
            iin_arr = np.asarray(iin)
            if iin_arr.ndim == 1 and iin_arr.shape[0] == len(lin_id):
                ieval = ieval + iin_arr[lin_id].reshape(n_lin, -1)
            elif iin_arr.ndim == 2 and iin_arr.shape[0] == len(lin_id):
                ieval = ieval + iin_arr[lin_id, :]

        if T[0] is not None and T[0].size > 0 and vin is not None:
            T1l = T[0][lin_id, :]
            ieval = ieval + T1l @ np.atleast_2d(vin).reshape(T[0].shape[1], -1)

        if T[2].size > 0 and np.any(sat_id_b) and vsat is not None and len(vsat) > 0:
            T3ls = T[2][np.ix_(lin_id, sat_id_b)]
            ieval = ieval + T3ls @ vsat.reshape(-1, 1)

        if vbiasn is not None:
            if T[0] is not None and T[0].size > 0:
                T1l = T[0][lin_id, :]
                ieval = ieval + T1l @ np.atleast_2d(vbiasn).reshape(T[0].shape[1], -1)

        if np.linalg.cond(Tinv) < 1e11:
            v = v + scipy.linalg.solve(Tinv, ieval)
        else:
            v = v + np.linalg.pinv(Tinv) @ ieval

    vbiasp_lin = vbiasp[lin_id].ravel()
    if np.any(vbiasp_lin != 0):
        T10l = T[9][np.ix_(lin_id, lin_id)]
        T9l = T[8][np.ix_(lin_id, lin_id)]
        T2l = T[1][np.ix_(lin_id, lin_id)]
        T4l = T[3][np.ix_(lin_id, lin_id)]
        T3l = T[2][np.ix_(lin_id, lin_id)]
        inv_A_lin = np.diag(1.0 / A[lin_id])

        inner2 = scipy.linalg.solve(T2l + T4l, T3l)
        Tninv = T10l + T9l @ inner2 - inv_A_lin
        v = v + scipy.linalg.solve(Tninv, vbiasp_lin.reshape(-1, 1))

    return v.ravel()


def _ctrlvec(
    T: list,
    sat_id: np.ndarray,
    Vlin: np.ndarray,
    Iin,
    Vin,
    SA0: np.ndarray,
    mask: np.ndarray,
    T1_cache=None,
    T2_cache=None,
    vbiasp=None,
    vbiasn=None,
):
    """Compute control vector seen by OAs in saturation.
    Returns (V_ctrl, T1_new, T2_new) where V_ctrl is (nnz(mask), cols).
    """
    if vbiasp is None:
        vbiasp = np.zeros(len(sat_id))
    if vbiasn is None:
        vbiasn = np.zeros_like(Vin) if Vin is not None else 0.0

    n_mask = int(np.sum(mask))
    nt = Vlin.shape[1] if Vlin.ndim > 1 else 1
    if Vlin.ndim == 1:
        Vlin = Vlin.reshape(-1, 1)

    V_ctrl = np.zeros((n_mask, nt))

    # Linear OAs: their control voltage == their output voltage
    lin_in_mask = (~(sat_id != 0)) & mask
    if np.any(lin_in_mask):
        # index within mask
        lin_pos_in_mask = ~(sat_id[mask] != 0)
        lin_pos = np.where(lin_pos_in_mask)[0]
        V_ctrl[lin_pos, :] = Vlin[lin_in_mask, :]

    # Saturated OAs: compute control from divider
    sat_in_mask = (sat_id != 0) & mask
    if np.any(sat_in_mask):
        nMask_sat = sat_in_mask
        n_sat = int(np.sum(nMask_sat))

        if T1_cache is None or T2_cache is None:
            T9_ss = T[8][np.ix_(nMask_sat, nMask_sat)]
            T2_ss = T[1][np.ix_(nMask_sat, nMask_sat)]
            T4_ss = T[3][np.ix_(nMask_sat, nMask_sat)]
            T3_ss = T[2][nMask_sat, :]
            T10_ss = T[9][nMask_sat, :]

            tmp = scipy.linalg.solve(T2_ss + T4_ss, np.eye(n_sat))
            T1_new = T9_ss @ tmp @ T3_ss + T10_ss  # (n_sat, n)
            # Input-current contribution onto the saturated OAs' junctions.
            iin_sat = (
                np.asarray(Iin)[nMask_sat].reshape(n_sat, -1)
                if Iin is not None
                else np.zeros((n_sat, 1))
            )
            if T[0] is not None and T[0].size > 0:
                p = T[0].shape[1]
                # Reshape Vin to (p, cols) - works for 1-D (p,), column (p,1) and
                # time-resolved (p, T) inputs alike. A naive transpose breaks the
                # (p, 1) and (p, T) cases.
                vin_2d = np.asarray(Vin).reshape(p, -1) if Vin is not None else np.zeros((p, 1))
                vbn = np.asarray(vbiasn, dtype=float)
                if vbn.size == p:
                    vin_2d = vin_2d + vbn.reshape(p, -1)
                elif vbn.size == 1:
                    vin_2d = vin_2d + float(vbn)
                T2_new = T9_ss @ tmp @ (iin_sat + T[0][nMask_sat, :] @ vin_2d)
            else:
                T2_new = T9_ss @ tmp @ iin_sat
            T1_cache = T1_new
            T2_cache = T2_new

        # control = (T2_cache + T1_cache @ Vlin - vbiasp[nMask_sat]) * SA0[nMask_sat]
        vbp_sat = vbiasp[nMask_sat].reshape(-1, 1)
        sa_sat = SA0[nMask_sat].reshape(-1, 1)
        Vcheck = (T2_cache + T1_cache @ Vlin - vbp_sat) * sa_sat

        sat_pos_in_mask = np.where(sat_id[mask] != 0)[0]
        V_ctrl[sat_pos_in_mask, :] = Vcheck

    return V_ctrl, T1_cache, T2_cache


def _chkregn(
    Vcontrol: np.ndarray, sat_id: np.ndarray, Vss: np.ndarray, Vdd: np.ndarray, tol: float = 0.0
) -> np.ndarray:
    """Check that Vcontrol is consistent with sat_id.
    Returns bool array of shape (cols,).
    """
    nt = Vcontrol.shape[1] if Vcontrol.ndim > 1 else 1
    if Vcontrol.ndim == 1:
        Vcontrol = Vcontrol.reshape(-1, 1)

    y = np.ones(nt, dtype=bool)
    lin_id = sat_id == 0
    if np.any(lin_id):
        Vlin = Vcontrol[lin_id, :]
        y = y & np.all(Vlin >= Vss[lin_id, np.newaxis] * (1 + tol), axis=0)
        y = y & np.all(Vlin <= Vdd[lin_id, np.newaxis] * (1 + tol), axis=0)
    if np.any(sat_id > 0):
        y = y & np.all(Vcontrol[sat_id > 0, :] > Vdd[sat_id > 0, np.newaxis] * (1 - tol), axis=0)
    if np.any(sat_id < 0):
        y = y & np.all(Vcontrol[sat_id < 0, :] < Vss[sat_id < 0, np.newaxis] * (1 - tol), axis=0)
    return y


def _getregn(Vcontrol: np.ndarray, Vss: np.ndarray, Vdd: np.ndarray) -> np.ndarray:
    """Determine saturation state from control voltage."""
    s = np.zeros(Vcontrol.shape[0], dtype=int)
    s[Vcontrol.ravel() > Vdd] = 1
    s[Vcontrol.ravel() < Vss] = -1
    return s


def _sigprep(Ie, Ve, time: np.ndarray):
    """Pre-process input signals: deduplicate time-steps where inputs are constant.
    Returns (S, T) where:
      S is a list of (Iin_slice, Vin_slice) for each unique-input segment,
      T is the array of time-boundary points.
    """
    i_empty = Ie is None or Ie.size == 0
    v_empty = Ve is None or Ve.size == 0

    Ti = np.array([], dtype=int)
    Tv = np.array([], dtype=int)
    IC = np.array([], dtype=int)
    VC = np.array([], dtype=int)
    I_uniq = None
    V_uniq = None

    if not i_empty:
        _, IC_full, IC = np.unique(Ie.T, axis=0, return_index=True, return_inverse=True)
        I_uniq = Ie[:, IC_full]
        # Transition indices (1-based transition points)
        diff_IC = np.diff(IC)
        Ti = np.unique(np.concatenate([1 + np.where(diff_IC)[0], [len(time) - 1]])).astype(int)
        IC = IC[Ti - 1]

    if not v_empty:
        _, VC_full, VC = np.unique(Ve.T, axis=0, return_index=True, return_inverse=True)
        V_uniq = Ve[:, VC_full]
        diff_VC = np.diff(VC)
        Tv = np.unique(np.concatenate([1 + np.where(diff_VC)[0], [len(time) - 1]])).astype(int)
        VC = VC[Tv - 1]

    T_idx = np.sort(np.unique(np.concatenate([Ti, Tv]))).astype(int)
    if T_idx.size == 0:
        T_idx = np.array([len(time) - 1], dtype=int)

    S = []
    for t_ii in T_idx:
        I_sl = None
        V_sl = None
        if not i_empty:
            idx_i = np.where(Ti >= t_ii)[0]
            col = IC[idx_i[0]] if idx_i.size else IC[-1]
            I_sl = I_uniq[:, col]
        if not v_empty:
            idx_v = np.where(Tv >= t_ii)[0]
            col = VC[idx_v[0]] if idx_v.size else VC[-1]
            V_sl = V_uniq[:, col]
        S.append((I_sl, V_sl))

    T_times = time[T_idx]
    return S, T_times


def nlop_model(circ: LoopUS, Vin: np.ndarray, Iin: np.ndarray):
    """Nonlinear operating-point solver. Returns (Vout, time=0, info)."""
    S = circ.sign_vector
    A = S * np.array([float(np.atleast_1d(m.A0)[0]) for m in circ.oa_models])
    T = circ._transfer_core()

    Voffset = np.array([float(np.atleast_1d(getattr(m, "Vos", 0.0))[0]) for m in circ.oa_models])
    oa_bias = circ.oa_reference.to_array(np.array([0.0]))[0]
    in_bias = circ.input_reference.to_array(np.array([0.0]))[0]

    vbiasn = in_bias if (in_bias != 0) else None
    vbiasp = S * Voffset + oa_bias

    vsupply = np.array([list(m.Vsupply) for m in circ.oa_models])
    Vss = vsupply[:, 0]
    Vdd = vsupply[:, 1]
    oa_mask = np.any(~np.isinf(vsupply), axis=1)

    n = circ.n

    if not np.any(oa_mask):
        # No supply-limited op-amps => nothing can saturate => exact linear solve (below)
        sat_id = np.zeros(n, dtype=int)
    else:
        # Drive of each OA as a function of Vout: u = M @ Vout + c
        # Full circuit relation:
        #   Vjunc  = (T2 + T4)^-1 (Iin + T1 (Vin + Vbiasn) + T3 Vout)
        #   Voainn = T9 Vjunc + T10 Vout
        #   u      = (S A0) (Voainn - Vbiasp)
        # u = open-loop output the op-amps would drive to without clamping.
        #   Vout = clip(u(Vout), Vss, Vdd).
        inv_junc = scipy.linalg.inv(T[1] + T[3])

        cjun = np.zeros(n)
        if Iin is not None and np.asarray(Iin).size:
            cjun = cjun + np.asarray(Iin, float).reshape(n)
        if T[0] is not None and T[0].size:
            p = T[0].shape[1]
            vin_part = np.zeros(p)
            if Vin is not None and np.asarray(Vin).size:
                vin_part = vin_part + np.asarray(Vin, float).reshape(p)
            if vbiasn is not None:
                vin_part = vin_part + vbiasn
            if np.any(vin_part):
                cjun = cjun + T[0] @ vin_part

        # Gain-free drive of each OA: drive(Vout) = M0 @ Vout + c0. A finite-gain
        # OA settles to Vout = clip(A * drive). An ideal one (A = +-inf) cannot
        # use that relation, but its rail behaviour is either interior with drive = 0,
        # or pinned to a rail with the drive != 0, which is captured by the equation:
        # Vout = clip(Vout + kappa * s * drive) for any kappa > 0
        # s = sign(A) sets the drive polarity.
        # This keeps the ideal case finite and exact rather than a large-gain
        # approximation.
        M0 = T[8] @ inv_junc @ T[2] + T[9]
        c0 = T[8] @ inv_junc @ cjun - vbiasp

        inf_mask = np.isinf(A)
        fin = ~inf_mask
        inf_idx = np.where(inf_mask)[0]
        inv_A = np.zeros(n)
        inv_A[fin] = 1.0 / A[fin]
        s_inf = np.sign(A[inf_mask])
        kappa = 1.0

        # Find the saturation state via a bounded least-squares solve of
        # the fixed point.  The piecewise-linear residual is solved globally
        # (one continuous problem) instead of searching the 3^n discrete
        # states, so it does not get trapped in the cycles that made the old
        # heuristic / brute-force fallback fail. Non-supply-limited OAs get
        # infinite rails (free variables); supply-limited OAs are bounded to
        # [Vss, Vdd].
        lo = np.where(oa_mask, Vss, -np.inf)
        hi = np.where(oa_mask, Vdd, np.inf)

        # Seed from the purely-linear (unsaturated) solution:
        # (diag(1/A) - M0) Vout = c0, clipped to the rails.
        K = np.diag(inv_A) - M0
        try:
            x_lin = scipy.linalg.solve(K, c0)
        except scipy.linalg.LinAlgError:
            x_lin = np.linalg.lstsq(K, c0, rcond=None)[0]
        x_lin = np.nan_to_num(x_lin, nan=0.0, posinf=0.0, neginf=0.0)
        x0 = np.clip(x_lin, lo, hi)

        def _open_loop(x, drive):
            """Unclamped target each OA drives toward (finite: A*drive; ideal:
            Vout + kappa*s*drive)."""
            u = np.empty(n)
            u[fin] = A[fin] * drive[fin]
            u[inf_mask] = x[inf_mask] + kappa * s_inf * drive[inf_mask]
            return u

        def _resid(x):
            drive = M0 @ x + c0
            return x - np.clip(_open_loop(x, drive), lo, hi)

        def _jac(x):
            drive = M0 @ x + c0
            u = _open_loop(x, drive)
            active = (u > lo) & (u < hi)  # d(clip)/du = 1 in the interior
            # d(open_loop)/dx: A*M0 on finite rows, I + kappa*s*M0 on ideal rows.
            dU = np.zeros((n, n))
            dU[fin, :] = A[fin, np.newaxis] * M0[fin, :]
            dU[inf_mask, :] = kappa * s_inf[:, np.newaxis] * M0[inf_mask, :]
            dU[inf_idx, inf_idx] += 1.0
            J = np.eye(n) - active[:, np.newaxis] * dU
            return J

        sol = least_squares(
            _resid,
            x0,
            jac=_jac,
            bounds=(lo, hi),
            method="trf",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        drive = M0 @ sol.x + c0
        u = _open_loop(sol.x, drive)
        sat_id = np.zeros(n, dtype=int)
        sat_id[oa_mask] = _getregn(u[oa_mask], Vss[oa_mask], Vdd[oa_mask])

    # Exact solve for the resolved saturation state, reusing linear sub-solver.
    Vctrl = np.zeros((int(np.sum(oa_mask)), 1))
    for _ in range(n + 2):
        Vnlin = _vnonlin(sat_id, vsupply)
        Vout = _nlstate(_endstat(T, A, sat_id, Iin, Vin, Vnlin, vbiasp, vbiasn), Vnlin, sat_id)
        Vctrl, _, _ = _ctrlvec(
            T,
            sat_id,
            Vout.reshape(-1, 1),
            Iin,
            Vin,
            A,
            oa_mask,
            vbiasp=vbiasp,
            vbiasn=vbiasn if vbiasn is not None else 0.0,
        )
        if np.all(_chkregn(Vctrl, sat_id[oa_mask], Vss[oa_mask], Vdd[oa_mask])):
            break
        new_sat = _getregn(Vctrl.ravel(), Vss[oa_mask], Vdd[oa_mask])
        if np.array_equal(new_sat, sat_id[oa_mask]):
            break
        sat_id[oa_mask] = new_sat

    info = {"Vout": Vout, "sat_id": sat_id, "Vcontrol": Vctrl}
    return Vout, 0.0, info


def nltran_model(circ: LoopUS, time: np.ndarray, Vin, Iin):
    """Nonlinear transient solver. Returns (VGlobal, tGlobal)."""
    T = circ._transfer_core()

    vsupply = np.array([list(m.Vsupply) for m in circ.oa_models])
    Vss = vsupply[:, 0]
    Vdd = vsupply[:, 1]
    P0 = 2 * np.pi * np.array([float(np.atleast_1d(m.GBWP)[0]) for m in circ.oa_models])
    S = circ.sign_vector
    A = S * np.array([float(np.atleast_1d(m.A0)[0]) for m in circ.oa_models])
    oa_mask = np.any(~np.isinf(vsupply), axis=1)

    Voffset = np.array([float(np.atleast_1d(getattr(m, "Vos", 0.0))[0]) for m in circ.oa_models])
    oa_bias = circ.oa_reference.to_array(np.array([0.0]))[0]
    in_bias = circ.input_reference.to_array(np.array([0.0]))[0]

    vbiasn = in_bias if (in_bias != 0) else None
    vbiasp = S * Voffset + oa_bias

    all_linear = not np.any(oa_mask)

    D0 = float(np.max(P0)) if P0.size else 1.0

    # Signal preparation
    Sin, t_input = _sigprep(Iin, Vin, time)
    IEval = Sin[0][0] if Sin[0][0] is not None else None
    VinEval = Sin[0][1] if Sin[0][1] is not None else None

    time_s = np.sort(np.unique(np.concatenate([time, t_input]))) * D0
    t_input_s = np.concatenate([t_input, [np.inf]]) * D0
    dT0 = min(1e2 / (float(np.abs(P0).mean()) + 1e-300), 1e-8) * D0
    dTmin0 = min(1e-6 / (float(np.abs(P0).max()) + 1e-300), 1e-16) * D0
    chunk_size = len(time_s) if all_linear else min(100, len(time_s))

    # Preallocate
    n = circ.n
    sat_id = np.zeros(n, dtype=int)
    sat_id_L = sat_id.copy()
    t_global = np.full(2 * len(time_s), np.nan)
    V_global = np.full((n, len(t_global)), np.nan)
    Vende_cache: dict = {}
    Vbias_cache: dict = {}
    Wende_cache: dict = {}
    VSat_cache: dict = {}
    useExpm_cache: dict = {}
    decW_cache: dict = {}

    t_global[0] = time_s[0]
    e_idx = 1
    t_valid = 0.0
    state_changed = True
    input_changed = True
    inp_id = 0
    c_iter = 0

    # Initial condition
    if circ.initial_condition is None:
        _, _, iC = nlop_model(circ, np.zeros(circ.p) if circ.p > 0 else None, np.zeros(n))
        V_global[:, 0] = iC["Vout"]
        sat_id = iC["sat_id"]
    else:
        V_global[:, 0] = circ.initial_condition

    # --- Main loop ---
    while (t_global[e_idx - 1] - time_s[-1]) < -np.spacing(time_s[-1]):
        c_iter += 1
        s_idx = e_idx
        t_init = t_global[s_idx - 1]
        V_init = V_global[:, s_idx - 1]

        if input_changed:
            inp_id = min(inp_id + 1, len(Sin) - 1)
        if state_changed or input_changed:
            t_ref0 = t_init

        t_local_full = np.concatenate(
            [[t_init], time_s[(time_s > t_init) & (time_s <= t_input_s[inp_id])]]
        )
        cur_chunk = min(chunk_size, len(t_local_full))
        t_local = t_local_full[:cur_chunk]
        input_bndry = t_local[-1] == t_input_s[inp_id]
        t_local = t_local - t_ref0

        # STEP 2: linearise
        if state_changed:
            poles_rad, W, l_eig, Xhat = circ.pz("rad", ~(sat_id != 0), True, True)
            d = poles_rad / D0

            key = tuple(sat_id.tolist())
            if key not in VSat_cache:
                Vnlin = _vnonlin(sat_id, vsupply)
                useExpm = np.linalg.cond(W) > 1e11 if W.size > 0 else False
                VSat_cache[key] = Vnlin
                useExpm_cache[key] = useExpm
                decW_cache[key] = W
            else:
                Vnlin = VSat_cache[key]
                useExpm = useExpm_cache[key]

        if input_changed:
            sl = Sin[min(inp_id, len(Sin) - 1)]
            IEval = sl[0]
            VinEval = sl[1]

        if state_changed or input_changed:
            key2 = (tuple(sat_id.tolist()), inp_id)
            if key2 not in Vende_cache:
                Vende = _endstat(T, A, sat_id, IEval, VinEval, Vnlin)
                Vende_cache[key2] = Vende
            else:
                Vende = Vende_cache[key2]

            key3 = tuple(sat_id.tolist())
            if key3 not in Vbias_cache:
                Vbias_v = _endstat(T, A, sat_id, None, None, None, vbiasp, vbiasn)
                Vbias_cache[key3] = Vbias_v
            else:
                Vbias_v = Vbias_cache[key3]

            # Bias vector expanded to full n
            lin_id = sat_id == 0
            Vbias = np.zeros(n)
            Vbias[lin_id] = Vbias_v

            W_sub = decW_cache[key]
            if key2 not in Wende_cache:
                if useExpm and W_sub.size > 0:
                    Wende = np.linalg.pinv(W_sub) @ Vende.reshape(-1, 1)
                elif W_sub.size > 0:
                    Wende = scipy.linalg.solve(W_sub, Vende.reshape(-1, 1))
                else:
                    Wende = np.zeros((0, 1))
                Wende_cache[key2] = Wende
            else:
                Wende = Wende_cache[key2]

            V_init_lin = V_init[lin_id] - Vbias[lin_id]
            if W_sub.size > 0:
                if useExpm:
                    Winit = np.linalg.pinv(W_sub) @ V_init_lin.reshape(-1, 1)
                else:
                    Winit = scipy.linalg.solve(W_sub, V_init_lin.reshape(-1, 1))
            else:
                Winit = np.zeros((0, 1))
            Wdiff = Winit - Wende

        # STEP 3: linear transient
        if d.size > 0 and len(t_local) > 1:
            Vlin_inner = _lintran(t_local[1:], W_sub, d, Wdiff, Wende)  # (n_lin, T-1)
        else:
            Vlin_inner = np.zeros((0, max(len(t_local) - 1, 0)))

        # Assemble full-state voltage matrix (n, T)
        Vlocal_cols = []
        Vlocal_cols.append(V_init)
        for ti in range(Vlin_inner.shape[1]):
            Vfull = np.zeros(n)
            Vfull[lin_id] = Vlin_inner[:, ti] + Vbias[lin_id]
            Vfull[sat_id != 0] = Vnlin
            Vlocal_cols.append(Vfull)
        V_local = np.column_stack(Vlocal_cols) if len(Vlocal_cols) > 1 else V_init.reshape(-1, 1)

        Iin_eval = IEval.reshape(-1, 1) if IEval is not None else None
        Vin_eval = VinEval.reshape(-1, 1) if VinEval is not None else None
        Vctrl, T1_c, T2_c = _ctrlvec(
            T,
            sat_id,
            V_local,
            Iin_eval,
            Vin_eval,
            A,
            oa_mask,
            vbiasp=vbiasp,
            vbiasn=vbiasn if vbiasn is not None else 0.0,
        )
        b_check = _chkregn(Vctrl, sat_id[oa_mask], Vss[oa_mask], Vdd[oa_mask])

        # First-point boundary relaxation
        if not b_check[0] and not input_changed:
            b_check[0] = _chkregn(
                Vctrl[:, :1], sat_id[oa_mask], Vss[oa_mask], Vdd[oa_mask], tol=1e-2
            )[0]
            if not b_check[0]:
                warnings.warn("Transient convergence failed.", stacklevel=2)
                break

        if not b_check[0] and input_changed:
            new_sat = _getregn(Vctrl[:, 0], Vss[oa_mask], Vdd[oa_mask])
            # Keep previously-saturated OAs on their side
            mask_paradox = (sat_id_L[oa_mask] != 0) & (new_sat != sat_id_L[oa_mask])
            new_sat[mask_paradox] = 0
            sat_id[oa_mask] = new_sat
            state_changed = True
            input_changed = False
            continue

        state_changed = not np.all(b_check)

        if state_changed:
            b_idx = int(np.where(~b_check)[0][0])
            chunk_size = b_idx

            t_valid = t_local[b_idx - 1]
            t_invld = t_local[b_idx]
            dT = min((t_invld - t_valid) / 10.0, dT0)
            dTmin = max(min(dT / 1e6, dTmin0), np.spacing(t_valid))

            # Binary-search refinement
            inner_count = 0
            while True:
                inner_count += 1
                t_grid = np.linspace(t_valid, t_invld, 10)
                dT = t_grid[1] - t_grid[0]

                Vlin_g = _lintran(t_grid, W_sub, d, Wdiff, Wende)
                Vstate_g = np.zeros((n, 10))
                Vstate_g[lin_id, :] = Vlin_g + Vbias[lin_id, np.newaxis]
                if np.any(sat_id != 0):
                    Vstate_g[sat_id != 0, :] = Vnlin[:, np.newaxis]

                Vctrl_g, _, _ = _ctrlvec(
                    T,
                    sat_id,
                    Vstate_g,
                    Iin_eval,
                    Vin_eval,
                    A,
                    oa_mask,
                    T1_c,
                    T2_c,
                    vbiasp,
                    vbiasn if vbiasn is not None else 0.0,
                )
                b_chk_g = _chkregn(Vctrl_g, sat_id[oa_mask], Vss[oa_mask], Vdd[oa_mask])
                b_chk_g[0] = True
                b_chk_g[-1] = False

                b_idx_g = int(np.where(~b_chk_g)[0][0]) - 1
                if dT > dTmin:
                    t_valid = t_grid[b_idx_g]
                    t_invld = t_grid[b_idx_g + 1]
                else:
                    t_valid = t_grid[b_idx_g + 1]
                    Vstate = Vstate_g[:, b_idx_g + 1]
                    sat_id_L = sat_id.copy()
                    sat_id[oa_mask] = _getregn(Vctrl_g[:, b_idx_g + 1], Vss[oa_mask], Vdd[oa_mask])
                    break

            mask_tv = t_local < t_valid
            V_local = np.column_stack([V_local[:, mask_tv], Vstate])
            t_local = np.concatenate([t_local[mask_tv], [t_valid]])
        else:
            sat_id_L = sat_id.copy()
            t_valid = t_local[-1]
            b_idx = cur_chunk
            chunk_size = len(time_s) if all_linear else min(2 * chunk_size, len(time_s))

        input_changed = not state_changed and input_bndry

        # STEP 5: global buffer update
        n_new = V_local.shape[1] - 1
        end_idx = s_idx + n_new - 1
        # Expand if needed
        if end_idx >= len(t_global):
            extra = max(n_new, len(time_s))
            t_global = np.concatenate([t_global, np.full(extra, np.nan)])
            V_global = np.concatenate([V_global, np.full((n, extra), np.nan)], axis=1)

        t_global[s_idx : s_idx + n_new] = t_local[1:] + t_ref0
        V_global[:, s_idx : s_idx + n_new] = V_local[:, 1:]
        e_idx = s_idx + n_new

    # --- Finishing ---
    valid_mask = ~np.isnan(t_global)
    t_global = t_global[valid_mask] / D0
    V_global = V_global[:, valid_mask]

    _, u_idx = np.unique(t_global, return_index=True)
    t_global = t_global[u_idx]
    V_global = V_global[:, u_idx]

    return V_global, t_global
