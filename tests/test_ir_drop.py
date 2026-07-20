"""IR-drop (interconnect / line-resistance) tests.

Two IR-drop scenarios are covered:
  * an analytical ``Geq`` ground truth (matrix-vector multiply with IR drop)
  * a closed-loop solver (linear-system solve with IR drop)

There is no saved ``data.mat`` fixture for IR drop, so the ground truth is the
analytical equivalent-conductance matrix built by hand:

    Geq = kron(Im, jn') * inv(Imn + Gd*(rr*kron(Im,MR) + rc*kron(MC,In)))
                        * Gd * kron(jm, In)

with ``Gd = diag(vec(G'))``, ``MR = minij(N)`` and ``MC = R*minij(M)*R`` for the
anti-diagonal exchange (reflector) ``R``. This matches the package's extended
transfer matrix exactly when the array uses full-segment terminal resistances
(``lineResistance = [rr, rc, rr, rc]``).
"""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS


def _geq_ir(G, rr, rc):
    """Analytical IR-drop equivalent conductance (mvm_ir_drop.m)."""
    M, N = G.shape
    Im, In = np.eye(M), np.eye(N)
    jm, jn = np.ones((M, 1)), np.ones((N, 1))
    Imn = np.eye(M * N)
    Gd = np.diag(G.ravel(order="C"))  # diag(vec(G'))
    idx_n = np.arange(1, N + 1)
    idx_m = np.arange(1, M + 1)
    MR = np.minimum.outer(idx_n, idx_n).astype(float)  # gallery('minij', N)
    R = np.fliplr(np.eye(M))  # reflector
    MC = R @ np.minimum.outer(idx_m, idx_m).astype(float) @ R
    Rrow = rr * np.kron(Im, MR)
    Rcol = rc * np.kron(MC, In)
    return np.kron(Im, jn.T) @ np.linalg.inv(Imn + Gd @ (Rrow + Rcol)) @ Gd @ np.kron(jm, In)


# --- MVM with IR drop: extended transfer matrix vs analytical Geq ---
@pytest.mark.parametrize("shape", [(5, 5), (4, 6), (7, 3)])
def test_extended_transfer_matches_analytical_geq(shape):
    rng = np.random.default_rng(0)
    M, N = shape
    G0 = 1.0 / 1e4
    rr = rc = 100.0
    A = rng.random((M, N))
    A = A / A.max()
    G = A * G0

    cb = CrossBar(G, line_resistance=[rr, rc, rr, rc])
    T = cb.transfer(
        output_variable="current",
        input_variable="voltage",
        transfer_direction=("col", "row"),
        xbar_orient=("bottom", "left"),
    )

    np.testing.assert_allclose(T, _geq_ir(G, rr, rc), rtol=1e-9, atol=1e-12)


def test_ir_drop_perturbs_ideal_mvm():
    """Finite line resistance distorts the ideal G transfer matrix."""
    rng = np.random.default_rng(1)
    G = rng.random((5, 5)) * (1.0 / 1e4)
    ideal = CrossBar(G).transfer(output_variable="current", transfer_direction=("col", "row"))
    ir = CrossBar(G, line_resistance=[100, 100, 100, 100]).transfer(
        output_variable="current", transfer_direction=("col", "row"), xbar_orient=("bottom", "left")
    )
    # IR drop lowers the effective conductances measurably.
    assert np.abs(ir - ideal).max() > 1e-7
    assert np.all(np.abs(ir) <= np.abs(ideal) + 1e-12)


def test_extended_transfer_converges_to_ideal_as_lr_zero():
    rng = np.random.default_rng(2)
    G = rng.random((5, 5)) * (1.0 / 1e4)
    ideal = CrossBar(G).transfer(output_variable="current", transfer_direction=("col", "row"))
    near = CrossBar(G, line_resistance=[1e-6, 1e-6, 1e-6, 1e-6]).transfer(
        output_variable="current", transfer_direction=("col", "row"), xbar_orient=("bottom", "left")
    )
    np.testing.assert_allclose(near, ideal, rtol=1e-3, atol=1e-10)


# --- closed-loop PD solver with IR drop (linear_system_PD_positive_IRdrop) ---
def _pd_solution(line_resistance, seed=3):
    rng = np.random.default_rng(seed)
    N = 10
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0
    A = rng.random((N, N))
    A = A.T @ A + 10 * np.eye(N)
    A = A / A.max()

    Gin = CrossBar(np.empty((N, 0)))
    Gf = (
        CrossBar(A * G0)
        if line_resistance is None
        else CrossBar(A * G0, line_resistance=line_resistance)
    )
    circ = LoopUS(Gin, Gf)
    y = rng.random(N)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))
    time = np.concatenate([[0.0], np.logspace(-10, np.log10(100e-6), 1000)])
    Vout, _ = circ.tran("model", time)
    ref = -np.linalg.solve(A, y) * V0
    return Vout[:, -1], ref


def test_pd_ideal_solver_matches_reference():
    V, ref = _pd_solution(None)
    np.testing.assert_allclose(V, ref, rtol=2e-2, atol=1e-2)


def test_pd_ir_drop_deviates_from_reference():
    """IR drop introduces a solver error well beyond the finite-gain error."""
    V_ideal, ref = _pd_solution(None)
    V_ir, _ = _pd_solution(100)
    assert np.linalg.norm(V_ir - ref) > 5 * np.linalg.norm(V_ideal - ref)


def test_pd_ir_drop_recovers_ideal_as_lr_zero():
    V_ideal, _ = _pd_solution(None)
    V_near, _ = _pd_solution(1e-6)
    np.testing.assert_allclose(V_near, V_ideal, rtol=1e-4, atol=1e-6)


# --- dynamic feedback coupling (pole accuracy) under IR drop ---
# The op-amp input sits at the FAR end of each resistive row rail; the input
# current enters at the NEAR end. Under line resistance those are distinct
# nodes. pz()'s closed-loop coupling Xhat must sense the far end, otherwise the
# poles (hence the transient) are biased by a few percent. Ground truth is a
# direct resistive MNA of the array (matches the extended SPICE netlist).


def _mna_coupling(G, rr, rc, sense):
    """Exact static coupling dV_sense/dV_colBottom for the resistive array.
    ``sense`` is 'far' (VinOA, row right end) or 'near' (Vjunc, row left end).
    Both row ends and column tops are open; column bottoms are the drive."""
    m, n = G.shape
    src = {f"Vout_{j}": j for j in range(n)}
    edges = []

    def res(a, b, R):
        edges.append((a, b, 1.0 / R))

    for i in range(m):
        chain = [f"Vjunc_{i}"]
        for j in range(n):
            chain.append(f"BE_{i}_{j}")
            if j < n - 1:
                chain.append(f"rb_{i}_{j}")
        chain.append(f"VinOA_{i}")
        for a, b in zip(chain[:-1], chain[1:]):
            res(a, b, rr / 2)
    for j in range(n):
        chain = [f"ctop_{j}"]
        for i in range(m):
            chain.append(f"TE_{i}_{j}")
            if i < m - 1:
                chain.append(f"cb_{i}_{j}")
        chain.append(f"Vout_{j}")
        for a, b in zip(chain[:-1], chain[1:]):
            res(a, b, rc / 2)
    for i in range(m):
        for j in range(n):
            if G[i, j] != 0:
                res(f"BE_{i}_{j}", f"TE_{i}_{j}", 1.0 / G[i, j])
    free = sorted({nm for e in edges for nm in e[:2] if nm not in src})
    fidx = {nm: k for k, nm in enumerate(free)}
    Gff = np.zeros((len(free), len(free)))
    Gfs = np.zeros((len(free), n))
    for a, b, g in edges:
        sa, sb = a in src, b in src
        if not sa:
            Gff[fidx[a], fidx[a]] += g
        if not sb:
            Gff[fidx[b], fidx[b]] += g
        if sa and not sb:
            Gfs[fidx[b], src[a]] -= g
        elif sb and not sa:
            Gfs[fidx[a], src[b]] -= g
        elif not sa and not sb:
            Gff[fidx[a], fidx[b]] -= g
            Gff[fidx[b], fidx[a]] -= g
    end = "VinOA" if sense == "far" else "Vjunc"
    rows = [fidx[f"{end}_{i}"] for i in range(m)]
    return -np.linalg.solve(Gff, Gfs)[rows, :]


def test_pz_ir_coupling_senses_far_end():
    """pz()'s Xhat must match the far-end (op-amp input) coupling, not the
    near-end (current-injection) coupling that the lumped-port model gives."""
    rng = np.random.default_rng(0)
    N, G0, rr = 6, 1.0 / 10e3, 100.0
    A = rng.random((N, N))
    A = A.T @ A + 10 * np.eye(N)
    A = A / A.max()
    G = A * G0
    circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(G, line_resistance=rr))
    _, _, _, Xhat = circ.pz("hz")

    M_far = _mna_coupling(G, rr, rr, "far")
    M_near = _mna_coupling(G, rr, rr, "near")
    # the two physical sense nodes are genuinely different under IR drop
    assert np.abs(M_far - M_near).max() > 1e-2
    # pz must track the far end to high accuracy ...
    np.testing.assert_allclose(Xhat, M_far, rtol=1e-6, atol=1e-9)
    # ... and must NOT be the near-end coupling (the pre-fix behaviour)
    assert np.abs(Xhat - M_near).max() > 1e-2
