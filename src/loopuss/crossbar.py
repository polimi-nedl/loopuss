"""CrossBar — resistive crosspoint array."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from .memory_model import MemoryModel
from .tech_model import TechModel


class CrossBar:
    """Crosspoint array of resistive memory devices."""

    def __init__(
        self,
        G: np.ndarray,
        C: np.ndarray | float = 0.0,
        *,
        line_resistance: float | np.ndarray = 0.0,
        prog_mask: np.ndarray | None = None,
        noiseless: bool = True,
        ideal_buffers: bool = False,
        memory_model: MemoryModel | None = None,
        skip_quant: bool = False,
    ):

        assert isinstance(G, np.ndarray), "G must be a numpy array."
        self.G = G.astype(float, copy=True)
        if isinstance(C, np.ndarray):
            self.C = C.astype(float, copy=True)
        else:
            self.C = np.full(self.G.shape, float(C))

        self.noiseless: bool = bool(noiseless)
        self.ideal_buffers: bool = bool(ideal_buffers)
        self.prog_mask: np.ndarray = prog_mask if prog_mask is not None else ~np.isnan(G)

        # Expand lineResistance to [rr, rc, rrt, rct]
        lr = np.asarray(line_resistance, dtype=float).ravel()
        if lr.size == 1:
            self.line_resistance = np.array([lr[0], lr[0], lr[0] / 2, lr[0] / 2])
        elif lr.size == 2:
            self.line_resistance = np.array([lr[0], lr[1], lr[0] / 2, lr[1] / 2])
        else:
            self.line_resistance = lr[:4]

        self._Teq: dict = {}  # (io_key, td_key, xo_key) -> T
        self._dec_req: dict = {}  # xo_key -> (lu, piv)

        self._validate(memory_model, skip_quant)

    # --- helpers ---
    @property
    def M(self) -> int:
        return self.G.shape[0]

    @property
    def N(self) -> int:
        return self.G.shape[1]

    def isempty(self) -> bool:
        return self.G.size == 0

    # --- validation & quantisation ---
    def _validate(self, memory_model: MemoryModel | None, skip_quant: bool) -> None:
        self.G[np.isnan(self.G)] = 0.0

        if not np.isscalar(self.C) and self.C.shape != self.G.shape:
            self.C = np.full(self.G.shape, float(self.C.flat[0]))

        # Build MemoryModel if not provided
        G = self.G
        if memory_model is None:
            g_abs = np.abs(G[G != 0])
            lrs = float(np.max(np.abs(G))) if G.size else 0.0
            hrs = float(np.min(g_abs)) if g_abs.size else 0.0
            win = (lrs / hrs) if hrs > 0 else np.inf
            tech = TechModel()
            area = 4 * tech.node**2 if G.size else 0.0
            memory_model = MemoryModel(
                "GenericMemory", LRS=lrs, HRS=hrs, ZRS=0.0, window=win, technology=tech, area=area
            )
        self.memory_model = memory_model

        if G.size and not skip_quant:
            S = np.sign(G)
            Gabs = np.abs(G)
            mm = self.memory_model
            # Clamp to [HRS, LRS]
            Gabs[(Gabs > 0) & (Gabs < mm.HRS)] = mm.HRS
            Gabs[(Gabs > 0) & (Gabs > mm.LRS)] = mm.LRS
            # Quantise if Nbit is finite
            if (Gabs != 0).any() and len(np.unique(Gabs)) > 1 and np.isfinite(mm.Nbit):
                levels = 2**mm.Nbit - 1
                if mm.HRS == mm.ZRS:
                    Gabs = np.round(Gabs / mm.LRS * levels) / levels * mm.LRS
                else:
                    span = mm.LRS - mm.HRS
                    Gabs = np.round((Gabs - mm.HRS) / span * levels) / levels * span + mm.HRS
            Gabs[S == 0] = mm.ZRS
            Gabs[S != 0] = Gabs[S != 0] * S[S != 0]
            Gabs[~self.prog_mask] = G[~self.prog_mask]
            self.G = Gabs

        self.area = self.memory_model.area * int(np.count_nonzero(self.prog_mask))

    # --- transfer matrix ---
    def transfer(
        self,
        *,
        output_variable: str = "current",
        input_variable: str = "voltage",
        transfer_direction: tuple = ("col", "row"),
        xbar_orient: tuple = ("bottom", "left"),
    ) -> np.ndarray:
        """Compute (and cache) a CrossBar transfer matrix."""
        iv = input_variable.lower()
        ov = output_variable.lower()
        td = (transfer_direction[0].lower(), transfer_direction[1].lower())
        xo = (xbar_orient[0].lower(), xbar_orient[1].lower())
        key = ((iv, ov), td, xo)

        if key in self._Teq:
            return self._Teq[key]

        # Empty CrossBar -> zero matrix of correct shape
        if self.isempty():
            rows = self.M if td[1] == "row" else self.N
            cols = self.N if td[0] == "col" else self.M
            T = np.zeros((rows, cols))
            self._Teq[key] = T
            return T

        # Zero line resistance -> closed-form expressions
        if np.all(self.line_resistance == 0):
            T = self._transfer_zero_lr(iv, ov, td)
            self._Teq[key] = T
            return T

        if self.ideal_buffers and not np.all(self.line_resistance == 0):
            raise ValueError("idealBuffers requires lineResistance == 0.")

        # General analytical model
        Ev, Ac, Os = self._get_help_matrices(iv, ov, td, xo)
        lu_key = xo
        if lu_key not in self._dec_req:
            self._build_dec(td, xo)
        lu, piv = self._dec_req[lu_key]
        X = scipy.linalg.lu_solve((lu, piv), Ev)
        T = Os + Ac @ X
        T = np.real(T)
        self._Teq[key] = T
        return T

    def _transfer_zero_lr(self, iv: str, ov: str, td: tuple) -> np.ndarray:
        """Transfer matrix for zero line resistance."""
        G = self.G
        M, N = G.shape
        if iv == "voltage" and ov == "current":
            if td == ("col", "row"):
                return G.copy()
            if td == ("col", "col"):
                s = np.sum(np.abs(G), axis=0) if self.ideal_buffers else np.sum(G, axis=0)
                return np.diag(s)
            if td == ("row", "row"):
                s = np.sum(np.abs(G), axis=1) if self.ideal_buffers else np.sum(G, axis=1)
                return np.diag(s)
            if td == ("row", "col"):
                return G.T.copy()
        elif iv == "voltage" and ov == "voltage":
            if td == ("col", "row"):
                return np.zeros((M, N))
            if td == ("col", "col"):
                return np.eye(N)
            if td == ("row", "row"):
                return np.eye(M)
            if td == ("row", "col"):
                return np.zeros((N, M))
        elif iv == "current" and ov == "current":
            if td == ("col", "row"):
                return np.zeros((M, N))
            if td == ("col", "col"):
                return np.eye(N)
            if td == ("row", "row"):
                return np.eye(M)
            if td == ("row", "col"):
                return np.zeros((N, M))
        elif iv == "current" and ov == "voltage":
            rows = M if td[1] == "row" else N
            cols = N if td[0] == "col" else M
            return np.zeros((rows, cols))
        raise ValueError(f"Unrecognised transfer ({iv},{ov},{td})")

    def _build_dec(self, td: tuple, xo: tuple) -> None:
        """Build and cache LU factorisation of the hyperresistance matrix."""
        m, n = self.M, self.N
        rr, rc, rrt, rct = self.line_resistance
        Im = np.eye(m)
        In = np.eye(n)
        jn = np.ones(n)
        jm = np.ones(m)
        vR = np.concatenate([[rrt / rr], jn[:-1]])
        vC = np.concatenate([[rct / rc], jm[:-1]])
        TR = np.tril(np.ones((n, n)))
        TC = np.tril(np.ones((m, m)))
        if xo[0] == "bottom":
            TC = np.flipud(Im) @ TC
        if xo[1] == "right":
            TR = np.flipud(In) @ TR

        MR = TR @ (vR[:, None] * TR.T)
        MC = TC @ (vC[:, None] * TC.T)
        Rrow = np.kron(rr * Im, MR)
        Rcol = np.kron(MC, rc * In)
        # Rdev: diagonal of 1/g for each cell (G stored row-major -> ravel C order = row-major)
        g_vec = self.G.ravel(order="C")
        with np.errstate(divide="ignore"):
            rdev_diag = np.where(g_vec != 0, 1.0 / g_vec, np.inf)
        Rdev = np.diag(rdev_diag)
        lu, piv = scipy.linalg.lu_factor(Rdev + Rrow + Rcol)
        self._dec_req[xo] = (lu, piv)

    def _get_help_matrices(self, iv: str, ov: str, td: tuple, xo: tuple) -> tuple:
        """Compute Ev, Ac, Os for the analytical transfer formula."""
        m, n = self.M, self.N
        rr, rc, rrt, rct = self.line_resistance
        Im = np.eye(m)
        In = np.eye(n)
        jm = np.ones((m, 1))
        jn = np.ones((n, 1))
        vR = np.concatenate([[rrt / rr], np.ones(n - 1)])
        vC = np.concatenate([[rct / rc], np.ones(m - 1)])
        TR = np.tril(np.ones((n, n)))
        TC = np.tril(np.ones((m, m)))
        if xo[0] == "bottom":
            TC = np.flipud(Im) @ TC
        if xo[1] == "right":
            TR = np.flipud(In) @ TR

        # Ensure LU is built
        if xo not in self._dec_req:
            self._build_dec(td, xo)

        # Input expansion matrix Ev
        if iv == "voltage":
            if td[0] == "col":
                Ev = np.kron(jm, In)  # (m*n, n)
            else:
                Ev = np.kron(Im, jn)  # (m*n, m)
        else:  # current
            if td[0] == "col":
                Ev = np.kron(rc * TC @ vC[:, None], In)  # (m*n, n)
            else:
                Ev = np.kron(Im, rr * TR @ vR[:, None])  # (m*n, m)

        # Output accumulation matrix Ac
        if ov == "voltage":
            if td[1] == "col":
                Ac = np.kron((vC @ TC.T * rc).reshape(1, -1), In)  # (n, m*n)
            else:
                Ac = np.kron(Im, (vR @ TR.T * rr))  # (m, m*n)
            if td[0] == td[1]:
                Ac = -Ac
        else:  # current
            if td[1] == "col":
                Ac = np.kron(jm.T, In)  # (n, m*n)
            else:
                Ac = np.kron(Im, jn.T)  # (m, m*n)

        # Transfer offset Os
        n_rows = Ac.shape[0]
        n_cols = Ev.shape[1]
        Os = np.zeros((n_rows, n_cols))
        if iv == ov:
            if td[0] == td[1] == "col":
                Os = In.copy()
            if td[0] == td[1] == "row":
                Os = Im.copy()
        elif ov == "voltage" and iv == "current":
            if td[0] == td[1] == "col":
                Os = -(m - 1 + 2 * rct / rc) * rc * In
            if td[0] == td[1] == "row":
                Os = (n - 1 + 2 * rrt / rr) * rr * Im

        return Ev, Ac, Os

    # --- static: voltage divider ---
    @staticmethod
    def vpart(
        A: CrossBar,
        B: CrossBar,
        *,
        output_node: str = "middle",
        output_variable: str = "voltage",
        xbar_orient: tuple | list = (("bottom", "left"), ("bottom", "right")),
    ) -> np.ndarray:
        """Voltage divider between two cascade CrossBars.

        Returns Tout, and optionally (TsA, TsB, TcA, Tpro, Tdir).
        """
        if isinstance(xbar_orient[0], str):
            xbar_orient = [xbar_orient, xbar_orient]
        xoA = tuple(xbar_orient[0])
        xoB = tuple(xbar_orient[1])

        TcA = A.transfer(
            output_variable="current", transfer_direction=("col", "row"), xbar_orient=xoA
        )
        TsA = A.transfer(
            output_variable="current", transfer_direction=("row", "row"), xbar_orient=xoA
        )
        TsB = B.transfer(
            output_variable="current", transfer_direction=("row", "row"), xbar_orient=xoB
        )

        ov = output_variable.lower()
        on = output_node.lower()

        if ov == "voltage":
            if on == "first":
                Tdir = A.transfer(
                    output_variable="voltage", transfer_direction=("col", "row"), xbar_orient=xoA
                )
                Tpro = A.transfer(
                    output_variable="voltage", transfer_direction=("row", "row"), xbar_orient=xoA
                )
            elif on == "last":
                Tdir = 0.0
                Tpro = B.transfer(
                    output_variable="voltage", transfer_direction=("row", "row"), xbar_orient=xoB
                )
            else:  # middle
                Tdir = 0.0
                Tpro = 1.0
        else:  # current
            if on == "first":
                Tdir = A.transfer(
                    output_variable="current", transfer_direction=("col", "col"), xbar_orient=xoA
                )
                Tpro = -A.transfer(
                    output_variable="current", transfer_direction=("row", "col"), xbar_orient=xoA
                )
            elif on == "last":
                Tdir = 0.0
                Tpro = B.transfer(
                    output_variable="current", transfer_direction=("row", "col"), xbar_orient=xoB
                )
            else:  # middle
                Tdir = 0.0
                Tpro = B.transfer(
                    output_variable="current", transfer_direction=("row", "row"), xbar_orient=xoB
                )

        denom = TsA + TsB
        X = scipy.linalg.solve(denom, TcA)
        Tout = Tdir + Tpro @ X if not np.isscalar(Tpro) else Tdir + Tpro * X
        Tout = np.where(np.isnan(Tout), 0.0, Tout)
        return Tout

    @staticmethod
    def vpart_full(
        A: CrossBar,
        B: CrossBar,
        *,
        output_node: str = "middle",
        output_variable: str = "voltage",
        xbar_orient: tuple | list = (("bottom", "left"), ("bottom", "right")),
    ) -> tuple:
        """Like vpart but returns (Tout, TsA, TsB, TcA, Tpro, Tdir)."""
        if isinstance(xbar_orient[0], str):
            xbar_orient = [xbar_orient, xbar_orient]
        xoA = tuple(xbar_orient[0])
        xoB = tuple(xbar_orient[1])

        TcA = A.transfer(
            output_variable="current", transfer_direction=("col", "row"), xbar_orient=xoA
        )
        TsA = A.transfer(
            output_variable="current", transfer_direction=("row", "row"), xbar_orient=xoA
        )
        TsB = B.transfer(
            output_variable="current", transfer_direction=("row", "row"), xbar_orient=xoB
        )

        ov = output_variable.lower()
        on = output_node.lower()

        if ov == "voltage":
            if on == "first":
                Tdir = A.transfer(
                    output_variable="voltage", transfer_direction=("col", "row"), xbar_orient=xoA
                )
                Tpro = A.transfer(
                    output_variable="voltage", transfer_direction=("row", "row"), xbar_orient=xoA
                )
            elif on == "last":
                Tdir = 0.0
                Tpro = B.transfer(
                    output_variable="voltage", transfer_direction=("row", "row"), xbar_orient=xoB
                )
            else:
                Tdir = 0.0
                Tpro = 1.0
        else:
            if on == "first":
                Tdir = A.transfer(
                    output_variable="current", transfer_direction=("col", "col"), xbar_orient=xoA
                )
                Tpro = -A.transfer(
                    output_variable="current", transfer_direction=("row", "col"), xbar_orient=xoA
                )
            elif on == "last":
                Tdir = 0.0
                Tpro = B.transfer(
                    output_variable="current", transfer_direction=("row", "col"), xbar_orient=xoB
                )
            else:
                Tdir = 0.0
                Tpro = B.transfer(
                    output_variable="current", transfer_direction=("row", "row"), xbar_orient=xoB
                )

        denom = TsA + TsB
        X = scipy.linalg.solve(denom, TcA)
        Tout = Tdir + Tpro @ X if not np.isscalar(Tpro) else Tdir + Tpro * X
        Tout = np.where(np.isnan(Tout), 0.0, Tout)
        return Tout, TsA, TsB, TcA, Tpro, Tdir

    # --- SPICE ---
    @property
    def netlist_type(self) -> str:
        """'reduced' for line-resistance-free arrays, else 'extended'."""
        return "reduced" if np.all(self.line_resistance == 0) else "extended"

    def _parse_array_cell(
        self, cell_id: str, G: float, C: float, nodes: list[str], noiseless: bool
    ) -> str:
        """Single array cell SPICE string.

        Resistive-only: capacitive cells (C != 0) are out of scope.
        """
        if C != 0:
            raise NotImplementedError("Capacitive crossbar cells are out of scope.")

        nt = self.netlist_type
        nsl = "noiseless" if noiseless else ""

        if self.ideal_buffers and nt == "extended":
            raise ValueError("idealBuffers is incompatible with extended netlisting.")
        if self.ideal_buffers and G < 0:
            G = abs(G)
            nodes = [nodes[0], f"buf_{nodes[1]}"] + list(nodes[2:])

        if nt == "reduced":
            if G == 0:
                return ""
            return f"R{cell_id} {nodes[0]} {nodes[1]} {1.0 / G:.16g} {nsl}".rstrip()

        # extended netlisting (resistive)
        model = "ArrayCell_oprc" if G == 0 else "ArrayCell_rprc"
        if not noiseless:
            model += "_noisy"
        params = (
            f"{model} params: R={{1/{G:.16g}}} C={{0}} "
            f"RPR={{{self.line_resistance[0]:.16g}}} "
            f"RPC={{{self.line_resistance[1]:.16g}}}"
        )
        return f"{cell_id} {nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]} {params}"

    def to_spice(self, cb_id: str, bel, ted, ber=None, teu=None, ridx=None, cidx=None) -> list[str]:
        """SPICE netlist lines for the array.

        ``bel``/``ber`` are row node lists (left/right); ``ted``/``teu`` are
        column node lists (bottom/top). At least one of bel/ber and one of
        ted/teu must be provided.
        """
        if self.isempty():
            return []

        M, N = self.M, self.N
        nt = self.netlist_type
        ridx = list(range(M)) if ridx is None else [i for i in ridx if 0 <= i < M]
        cidx = list(range(N)) if cidx is None else [j for j in cidx if 0 <= j < N]
        bel = list(bel) if bel else []
        ber = list(ber) if ber else []
        ted = list(ted) if ted else []
        teu = list(teu) if teu else []

        if not bel and not ber:
            raise ValueError("At least one between bel and ber must be specified.")
        if not ted and not teu:
            raise ValueError("At least one between ted and teu must be specified.")

        be_int = f"{cb_id}_R"
        te_int = f"{cb_id}_C"

        if not bel:
            bel = [f"{be_int}{i}_0" for i in range(M)] if nt == "extended" else ber
        if not ber:
            ber = [f"{be_int}{i}_{N}" for i in range(M)] if nt == "extended" else bel
        if not ted:
            ted = [f"{te_int}{j}_{M}" for j in range(N)] if nt == "extended" else teu
        if not teu:
            teu = [f"{te_int}{j}_0" for j in range(N)] if nt == "extended" else ted

        lines: list[str] = []
        for ii in ridx:
            for jj in cidx:
                cell_id = f"X{cb_id}_R{ii}C{jj}"
                if nt == "extended":
                    n1 = bel[ii] if jj == cidx[0] else f"{be_int}{ii}_{jj}"
                    n2 = ber[ii] if jj == cidx[-1] else f"{be_int}{ii}_{jj + 1}"
                    n3 = teu[jj] if ii == ridx[0] else f"{te_int}{jj}_{ii}"
                    n4 = ted[jj] if ii == ridx[-1] else f"{te_int}{jj}_{ii + 1}"
                    nodes = [n1, n2, n3, n4]
                else:
                    nodes = [bel[ii], ted[jj]]
                s = self._parse_array_cell(
                    cell_id,
                    self.G[ii, jj],
                    self.C[ii, jj] if not np.isscalar(self.C) else 0.0,
                    nodes,
                    self.noiseless,
                )
                if s:
                    lines.append(s)

        if self.ideal_buffers:
            buf_mask = np.any(self.G[np.ix_(ridx, cidx)] < 0, axis=0)
            buf_cols = [cidx[k] for k in range(len(cidx)) if buf_mask[k]]
            for n, jj in enumerate(buf_cols, start=1):
                lines.append(f"EbufOA{n} buf_{ted[jj]} 0 0 {ted[jj]} 1")

        return lines

    def __repr__(self) -> str:
        return f"CrossBar(shape={self.G.shape}, lr={self.line_resistance[0]:.3g})"
