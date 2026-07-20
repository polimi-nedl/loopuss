"""LoopUS - main closed-loop in-memory computing circuit class."""

from __future__ import annotations

import atexit
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.linalg

from . import spice as _spice
from .adc_model import ADCModel
from .crossbar import CrossBar
from .dac import CurrentDAC, VoltageDAC
from .dsp_model import DSPModel
from .mem_write_model import MemWriteModel
from .netlist import build_netlist
from .noise import noise_model, noise_spice
from .oa_model import OAModel
from .reports import BenchReport, NoiseReport, TranReport, _safe_div
from .signal import CurrentSignal, VoltageSignal
from .solver import nlop_model, nltran_model
from .utils import GroupedDisplay, si_format

# --- private helpers (module-level) ---
# Default LTSpice working directory, made process-private. Every SPICE run
# writes a fixed ``spiceSimulation.net``/``.raw`` pair, and run_spice() deletes
# the stale ``.raw`` before each invocation; if two processes shared one
# directory they would clobber each other's files and a run could observe "no
# output" mid-race. Anchoring the default to ``simfolder/<pid>/`` lets several
# scripts (or pytest-xdist workers) drive LTSpice in parallel safely. Passing an
# explicit ``file_path`` opts out (and is never auto-removed).
_SIM_ROOT = "simfolder"
_proc_sim_dir: str | None = None


def _default_num_workers() -> int:
    """Physical cores available to this process.

    Hyperthread siblings are deliberately excluded: an LTSpice solve is CPU- and
    memory-bandwidth-bound, so sibling threads add contention and memory
    pressure rather than throughput. Falls back to the logical count wherever
    the topology isn't readable.
    """
    try:
        cpus = os.sched_getaffinity(0)  # respects taskset/cgroup limits
    except AttributeError:
        cpus = set(range(os.cpu_count() or 1))
    cores = set()
    for c in cpus:
        sibs = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        try:
            cores.add(sibs.read_text().strip())
        except OSError:
            return len(cpus)
    return len(cores) or len(cpus)


def _default_sim_dir() -> str:
    """Absolute, per-process SPICE working directory (``<cwd>/simfolder/<pid>``).

    Computed once per interpreter (anchored to the cwd of the first LoopUS) and
    registered for removal at exit, so concurrent processes never share netlist
    or ``.raw`` filenames. See _SIM_ROOT for why isolation matters.
    """
    global _proc_sim_dir
    if _proc_sim_dir is None:
        _proc_sim_dir = str(Path.cwd() / _SIM_ROOT / str(os.getpid()))
        atexit.register(shutil.rmtree, _proc_sim_dir, ignore_errors=True)
    return _proc_sim_dir


# --- LoopUS ---
class LoopUS(GroupedDisplay):
    """Closed-loop in-memory computing circuit."""

    def __init__(
        self,
        input_array: CrossBar | None = None,
        feedback_array: CrossBar | None = None,
        *,
        oa_model: OAModel | list[OAModel] | None = None,
        adc_model: ADCModel | None = None,
        dsp_model: DSPModel | None = None,
        mem_write_model: MemWriteModel | None = None,
        voltage_dac: VoltageDAC | None = None,
        current_dac: CurrentDAC | None = None,
        sign_vector: float | np.ndarray = -1.0,
        output_mask: np.ndarray | None = None,
        input_mask_i: np.ndarray | None = None,
        input_mask_v: np.ndarray | None = None,
        oa_reference: VoltageSignal | None = None,
        input_reference: VoltageSignal | None = None,
        initial_condition: np.ndarray | None = None,
        xbar_orient: list | None = None,
        bench_enable: bool = False,
        complexity: float | None = None,
        verbose: bool = False,
        file_path: str | None = None,
        file_name: str = "spiceSimulation",
        num_workers: int | None = None,
        nl_cache_size: int = 10000,
    ):

        self.verbose = verbose
        self.bench_enable = bool(bench_enable)
        # Parallelism for multi-netlist SPICE runs (.noise emits one per
        # output).
        self.num_workers = _default_num_workers() if num_workers is None else int(num_workers)
        # None -> a process-private SPICE working dir so parallel runs (separate
        # scripts / test workers) don't clobber each other's netlist/.raw files.
        self.file_path = file_path if file_path is not None else _default_sim_dir()
        self.file_name = file_name
        self._nl_cache_size = nl_cache_size

        # --- crossbars ---
        if feedback_array is None:
            raise ValueError("feedback_array is required.")
        assert feedback_array.M == feedback_array.N, "Feedback matrix must be square."
        self.feedback_array = feedback_array
        n = feedback_array.M

        if input_array is None:
            self.input_array = CrossBar(np.empty((n, 0)))
        else:
            assert input_array.M == n, (
                "input_array must have the same number of rows as feedback_array."
            )
            self.input_array = input_array

        # --- orientation: default {'bottom','right'; 'bottom','left'} ---
        if xbar_orient is None:
            self.xbar_orient = [["bottom", "right"], ["bottom", "left"]]
        else:
            self.xbar_orient = xbar_orient

        # --- OA models ---
        if oa_model is None:
            oa_model = OAModel("1P")
        if isinstance(oa_model, OAModel):
            self.oa_models = [deepcopy(oa_model) for _ in range(n)]
        else:
            oa_list = list(oa_model)
            if len(oa_list) == 1:
                self.oa_models = [deepcopy(oa_list[0]) for _ in range(n)]
            else:
                self.oa_models = oa_list[:n]

        # --- sign vector ---
        sv = np.asarray(sign_vector, dtype=float).ravel()
        if sv.size == 1:
            self.sign_vector = np.full(n, sv[0])
        else:
            self.sign_vector = sv[:n]

        # --- ADC ---
        self.adc_model = adc_model if adc_model is not None else ADCModel()

        # --- DSP ---
        self.dsp_model = dsp_model if dsp_model is not None else DSPModel()

        # --- MemWrite ---
        self.mem_write = mem_write_model if mem_write_model is not None else MemWriteModel()

        # --- references ---
        self.oa_reference = (
            oa_reference if oa_reference is not None else VoltageSignal("DC", dcvalue=0.0)
        )
        self.input_reference = (
            input_reference if input_reference is not None else VoltageSignal("DC", dcvalue=0.0)
        )

        # --- initial condition ---
        self.initial_condition = initial_condition

        # --- output mask ---
        m = output_mask if output_mask is not None else np.arange(n)
        self.output_mask = np.asarray(m, dtype=int).ravel()
        n_out = len(self.output_mask)
        nc = max(1, int(np.ceil(n_out / self.adc_model.Nchannel)))
        self._adcs = [deepcopy(self.adc_model) for _ in range(nc)]

        # --- input masks ---
        p = self.input_array.N
        self.input_mask_i = (
            np.arange(n) if input_mask_i is None else np.asarray(input_mask_i, dtype=int).ravel()
        )
        self.input_mask_v = (
            np.arange(p) if input_mask_v is None else np.asarray(input_mask_v, dtype=int).ravel()
        )

        # --- DACs ---
        self.voltage_dac = [
            deepcopy(voltage_dac if voltage_dac is not None else VoltageDAC()) for _ in range(p)
        ]
        self.current_dac = [
            deepcopy(current_dac if current_dac is not None else CurrentDAC()) for _ in range(n)
        ]

        # --- complexity ---
        self.complexity = float(n**3) if complexity is None else float(complexity)

        # --- benchmark state ---
        self._energy_breakdown = np.zeros((7, 6))
        self._latency = {"Programming": 0.0, "Computing": 0.0, "PostProcessing": 0.0}
        self._ops = 0.0
        self._power = np.array([0.0, 0.0])

        # --- caches ---
        self._trans_matrix: list | None = None
        self._eigen_decomp: dict | None = None
        self._nl_cache_index = 0

        # --- reprogram (applies MemWrite noise on construction) ---
        self.reprogram()

    # --- size helpers ---
    @property
    def n(self) -> int:
        """Number of OAs / rows of feedback array."""
        return self.feedback_array.M

    @property
    def p(self) -> int:
        """Number of columns of input array."""
        return self.input_array.N

    # --- benchmark properties ---
    _ENERGY_LABELS = ["MemWrite", "DAC", "InputArray", "FeedbackArray", "OA", "ADC", "DSP"]

    @property
    def area(self) -> dict:
        """Area footprint broken down by component."""
        cdac = self.current_dac[0].area if self.current_dac else 0.0
        vdac = self.voltage_dac[0].area if self.voltage_dac else 0.0
        A = {
            "MemWrite": self.mem_write.area * float(np.sum(self.mem_write.parallel_write)),
            "DAC": cdac * len(self.input_mask_i) + vdac * len(self.input_mask_v),
            "InputArray": getattr(self.input_array, "area", 0.0),
            "FeedbackArray": getattr(self.feedback_array, "area", 0.0),
            "OA": float(sum(m.area for m in self.oa_models)),
            "ADC": float(sum(a.area for a in self._adcs)),
            "DSP": float(self.dsp_model.area),
        }
        A["Total"] = sum(v for k, v in A.items())
        return A

    @property
    def energy(self) -> dict:
        """Energy breakdown structure.

        Breakdown is a 7x6 table: rows = MemWrite/DAC/InputArray/FeedbackArray/
        OA/ADC/DSP; cols = Programming(S,D), Computing(S,D), PostProcessing(S,D).
        Computing-dynamic for the arrays/OA comes from the internal accumulator;
        the remaining cells are read from the peripheral energy counters.
        """
        bd = self._energy_breakdown.copy()
        mw = self.mem_write.energy
        bd[0, 0], bd[0, 1] = mw["Static"], mw["Dynamic"]
        dac_s = sum(self.voltage_dac[i].energy["Static"] for i in self.input_mask_v) + sum(
            self.current_dac[i].energy["Static"] for i in self.input_mask_i
        )
        bd[1, 2] = dac_s
        bd[4, 2] = float(sum(m.energy["Static"] for m in self.oa_models))
        bd[5, 2] = float(sum(a.energy["Static"] for a in self._adcs))
        bd[5, 3] = float(sum(a.energy["Dynamic"] for a in self._adcs))
        bd[6, 4] = self.dsp_model.energy["Static"]
        bd[6, 5] = self.dsp_model.energy["Dynamic"]

        E = {"Breakdown": bd, "Labels": list(self._ENERGY_LABELS), "Total": float(np.nansum(bd))}
        for ii, lab in enumerate(self._ENERGY_LABELS):
            row = bd[ii]
            E[lab] = {
                "Programming": {"Static": row[0], "Dynamic": row[1], "Total": row[0] + row[1]},
                "Computing": {"Static": row[2], "Dynamic": row[3], "Total": row[2] + row[3]},
                "PostProcessing": {"Static": row[4], "Dynamic": row[5], "Total": row[4] + row[5]},
                "Total": float(np.nansum(row)),
            }
        return E

    @property
    def latency(self) -> dict:
        """Latency structure."""
        L = dict(self._latency)
        L["Total"] = L["Programming"] + L["Computing"] + L["PostProcessing"]
        return L

    @property
    def power(self) -> np.ndarray:
        """[mean, max] tracked power."""
        return self._power

    @property
    def ops(self) -> float:
        return self._ops

    @property
    def throughput(self) -> float:
        return _safe_div(self._ops, self.latency["Total"])

    @property
    def energy_efficiency(self) -> float:
        return _safe_div(self._ops, self.energy["Total"])

    @property
    def area_efficiency(self) -> float:
        return _safe_div(self.throughput, self.area["Total"])

    def benchmark(self, *flags) -> BenchReport:
        """Assemble a BenchReport from tracked metrics.

        Flags: ``'-nowrite'`` excludes programming energy/latency/area;
        ``'-noise'`` is accepted for compatibility.
        """
        nowrite = "-nowrite" in flags

        latency = self.latency
        energy = self.energy
        area = self.area
        ops = self.ops

        if nowrite:
            latency = dict(latency)
            latency["Total"] = latency["Total"] - latency["Programming"]
            latency["Programming"] = 0.0
            energy = deepcopy(energy)
            energy["Breakdown"][0, :] = 0.0  # zero the MemWrite (programming) row
            energy["Total"] = float(np.nansum(energy["Breakdown"]))
            area = dict(area)
            area["Total"] = area["Total"] - area["MemWrite"]
            area["MemWrite"] = 0.0

        return BenchReport(
            latency=latency,
            energy=energy,
            power=self.power,
            area=area,
            complexity=self.complexity,
            ops=ops,
            throughput=_safe_div(ops, latency["Total"]),
            energy_efficiency=_safe_div(ops, energy["Total"]),
            area_efficiency=_safe_div(_safe_div(ops, latency["Total"]), area["Total"]),
            circuit=deepcopy(self),
        )

    # --- reprogram / reset ---
    def reprogram(
        self, new_input: CrossBar | None = None, new_feedback: CrossBar | None = None
    ) -> None:
        """Re-program memory arrays with MemWrite noise model."""
        if new_input is None and new_feedback is None:
            new_input = self.input_array
            new_feedback = self.feedback_array

        t_prog = 0.0
        changed = False

        if new_input is not None and new_input.G.size > 0:
            new_in, T, *_ = self.mem_write.program(new_input)
            self.input_array = new_in
            t_prog += T
            changed = True

        if new_feedback is not None and new_feedback.G.size > 0:
            new_fb, T, *_ = self.mem_write.program(new_feedback)
            self.feedback_array = new_fb
            t_prog += T
            changed = True

        if changed:
            self._trans_matrix = None
            self._eigen_decomp = None

        if self.bench_enable:
            self._latency["Programming"] += t_prog

    # --- input setters ---
    def set_inputs(self, *args) -> None:
        """Load CurrentSignal / VoltageSignal into DACs."""
        for sig in args:
            if isinstance(sig, (list, np.ndarray)):
                sig_list = list(sig)
            else:
                sig_list = [sig]

            if sig_list and isinstance(sig_list[0], CurrentSignal):
                n_needed = len(self.input_mask_i)
                while len(sig_list) < n_needed:
                    sig_list.append(CurrentSignal("EMPTY"))
                for jj, idx in enumerate(self.input_mask_i):
                    dac = self.current_dac[idx]
                    s = sig_list[jj].add_delay(dac.latency)
                    dac.load_waveform([s])

            elif sig_list and isinstance(sig_list[0], VoltageSignal):
                n_needed = len(self.input_mask_v)
                while len(sig_list) < n_needed:
                    sig_list.append(VoltageSignal("EMPTY"))
                for jj, idx in enumerate(self.input_mask_v):
                    dac = self.voltage_dac[idx]
                    s = sig_list[jj].add_delay(dac.latency)
                    dac.load_waveform([s])

    def set_output_mask(self, mask: np.ndarray) -> None:
        self.output_mask = np.asarray(mask, dtype=int).ravel()
        n_out = len(self.output_mask)
        nc = max(1, int(np.ceil(n_out / self.adc_model.Nchannel)))
        self._adcs = [deepcopy(self.adc_model) for _ in range(nc)]

    def set_initial_condition(self, ic: np.ndarray) -> None:
        ic = np.asarray(ic, dtype=float).ravel()
        if ic.size == 1:
            ic = np.full(self.n, ic[0])
        self.initial_condition = ic[: self.n]

    # --- transfer core (T1-T10) ---
    def _transfer_core(self) -> list:
        """Compute and cache the 10 characteristic transfer matrices."""
        if self._trans_matrix is not None:
            return self._trans_matrix

        xo_in = tuple(self.xbar_orient[0])
        xo_fb = tuple(self.xbar_orient[1])

        # T1: input array col->row current transfer
        T1 = self.input_array.transfer(
            output_variable="current",
            input_variable="voltage",
            transfer_direction=("col", "row"),
            xbar_orient=xo_in,
        )

        # T2, T3, T4, T9, T10 from vpart(feedbackArray, inputArray, outputNode='first')
        # Order of returns: [_, T4, T2, T3, T9, T10] with xbar_orient rows swapped.
        # Default output_variable='voltage' -> T9, T10 are V->V; T2, T3, T4 are
        # always V->I (conductance).
        xbar_orient_vpart = [self.xbar_orient[1], self.xbar_orient[0]]
        Tout_vp, TsA, TsB, TcA, Tpro, Tdir = CrossBar.vpart_full(
            self.feedback_array,
            self.input_array,
            output_node="first",
            output_variable="voltage",
            xbar_orient=xbar_orient_vpart,
        )
        # mapping: Tout_vp is unused, TsA=T4, TsB=T2, TcA=T3, Tpro=T9, Tdir=T10
        T4 = TsA
        T2 = TsB
        T3 = TcA
        T9 = Tpro  # V->V: row-end to row-end voltage propagation
        T10 = Tdir  # V->V: col to row-end feedforward

        # T5: feedback col->col
        T5 = self.feedback_array.transfer(
            output_variable="current",
            input_variable="voltage",
            transfer_direction=("col", "col"),
            xbar_orient=xo_fb,
        )

        # T6: feedback row->col
        T6 = self.feedback_array.transfer(
            output_variable="current",
            input_variable="voltage",
            transfer_direction=("row", "col"),
            xbar_orient=xo_fb,
        )

        # T7: input col->col
        T7 = self.input_array.transfer(
            output_variable="current",
            input_variable="voltage",
            transfer_direction=("col", "col"),
            xbar_orient=xo_in,
        )

        # T8: input row->col
        T8 = self.input_array.transfer(
            output_variable="current",
            input_variable="voltage",
            transfer_direction=("row", "col"),
            xbar_orient=xo_in,
        )

        # Store as 0-indexed list [T1..T10]
        self._trans_matrix = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]
        return self._trans_matrix

    # --- pz (pole-zero analysis) ---
    def pz(
        self,
        units: str = "hz",
        l_id: np.ndarray | None = None,
        omit_inf: bool = False,
        skip_check: bool = False,
    ):
        """Compute poles of the closed-loop circuit.
        Returns (poles, V, l, Xhat).
        """
        n = self.n
        if l_id is None:
            l_id = np.ones(n, dtype=bool)
        l_id = np.asarray(l_id, dtype=bool).ravel()

        # Try cache
        if self._eigen_decomp is not None and not all(l_id):
            if self._eigen_decomp.get("Xhat") is None:
                self.pz()

        if self._eigen_decomp is not None:
            lID_cache = self._eigen_decomp["lID"]  # (n, cache_size)
            match = np.where(np.all(lID_cache == l_id[:, np.newaxis], axis=0))[0]
            if match.size > 0:
                idx = match[0]
                poles = self._eigen_decomp["poles"][:, idx]
                V = self._eigen_decomp["V"][idx]
                l = self._eigen_decomp["l"][:, idx]
                Xhat = self._eigen_decomp["Xhat"][np.ix_(l_id, l_id)]
                if omit_inf:
                    pmask = ~np.isinf(poles)
                    poles = poles[pmask]
                    l = l[pmask]
                    V = V[np.ix_(np.where(pmask)[0], np.where(pmask)[0])]
                    Xhat = Xhat[np.ix_(np.where(pmask)[0], np.where(pmask)[0])]
                if units.lower() == "rad":
                    poles = 2 * np.pi * poles
                return poles, V, l, Xhat

        # --- compute Xhat ---
        if self._eigen_decomp is None:
            # vpart takes (feedback, input) order
            xbar_orient_vpart = [self.xbar_orient[1], self.xbar_orient[0]]
            Xhat_full = CrossBar.vpart(
                self.feedback_array,
                self.input_array,
                xbar_orient=xbar_orient_vpart,
                output_node="first",
            )
        else:
            Xhat_full = self._eigen_decomp["Xhat"]

        Xhat = Xhat_full[np.ix_(l_id, l_id)]

        # --- check for RC (capacitive) case ---
        par_caps = (
            np.any(self.input_array.C)
            or np.any(self.feedback_array.C)
            or any(getattr(m, "input_capacitance", 0.0) for m in self.oa_models)
        )

        if not par_caps:
            F0 = np.array([float(np.atleast_1d(m.FP)[0]) for m in self.oa_models])[l_id]
            A0 = np.array([float(np.atleast_1d(m.A0)[0]) for m in self.oa_models])[l_id]
            # GM stores A0=Inf, FP=0 -> A0*F0 is an intentional Inf*0=NaN that is
            # replaced by GBWP just below; ignore the resulting warning.
            with np.errstate(invalid="ignore"):
                P0 = A0 * F0

            # GM model: use GBWP directly
            for ii, m in enumerate(m for jj, m in enumerate(self.oa_models) if l_id[jj]):
                if getattr(m, "type", "").upper() == "GM":
                    P0[ii] = m.GBWP

            # No linear OAs (all saturated) -> no modal sub-system to normalise.
            mGBWP = np.exp(np.mean(np.log(np.abs(P0) + 1e-300))) if P0.size else 1.0
            F0_s = F0 / mGBWP
            P0_s = self.sign_vector[l_id] * P0 / mGBWP

            nl = int(np.sum(l_id))
            Ehat = P0_s[:, np.newaxis] * Xhat
            Ehat[np.arange(nl), np.arange(nl)] -= F0_s

            l_eig, V = np.linalg.eig(Ehat)
            poles_raw = l_eig * mGBWP

            sort_idx = np.argsort(poles_raw.real)[::-1]
            poles = poles_raw[sort_idx]
            l = l_eig[sort_idx]
            V = V[:, sort_idx]
        else:
            raise NotImplementedError("RC pole-zero analysis not yet implemented.")

        # Pad with -inf for saturated OAs
        if not np.all(l_id):
            n_sat = n - int(np.sum(l_id))
            V = np.block(
                [
                    [V, np.full((V.shape[0], n_sat), np.nan)],
                    [np.full((n_sat, V.shape[1]), np.nan), np.full((n_sat, n_sat), np.nan)],
                ]
            )
            l = np.concatenate([l, -np.inf * np.ones(n_sat)])
            poles = np.concatenate([poles, -np.inf * np.ones(n_sat)])

        # --- store in cache ---
        cache_size = min(10000, self._nl_cache_size)
        if self._eigen_decomp is None:
            self._eigen_decomp = {
                "lID": np.full((n, cache_size), np.nan),
                "poles": np.full((n, cache_size), np.nan, dtype=complex),
                "V": [None] * cache_size,
                "l": np.full((n, cache_size), np.nan, dtype=complex),
                "Xhat": Xhat_full,
            }
            self._nl_cache_index = 0
        else:
            self._nl_cache_index = (self._nl_cache_index % (cache_size - 1)) + 1

        idx = self._nl_cache_index
        self._eigen_decomp["lID"][:, idx] = l_id.astype(float)
        self._eigen_decomp["poles"][:, idx] = poles
        self._eigen_decomp["V"][idx] = V
        self._eigen_decomp["l"][:, idx] = l

        if omit_inf:
            pmask = ~np.isinf(poles)
            poles = poles[pmask]
            l = l[pmask]
            V = V[np.ix_(np.where(pmask)[0], np.where(pmask)[0])]
            Xhat = Xhat[np.ix_(np.where(pmask)[0], np.where(pmask)[0])]

        if units.lower() == "rad":
            poles = 2 * np.pi * poles

        return poles, V, l, Xhat

    def tts(
        self,
        target_err: float | None = None,
        strict: bool = False,
        l_id: np.ndarray | None = None,
    ):
        """Estimate the time-to-solution of the circuit.

        Returns ``(tts, tts_min, tts_max)``. ``tts_min``/``tts_max`` are the
        lazy and greedy bounds for the dynamic error norm to fall below
        ``target_err``; ``tts`` is their mean. ``target_err`` defaults to
        ``exp(-2*pi)``, the settling threshold of an equivalent single-pole
        stage. With ``strict=True`` the result is ``nan`` unless the circuit is
        stable (all poles in the left half-plane).
        """
        if target_err is None:
            target_err = np.exp(-2 * np.pi)

        poles, V, _, _ = self.pz("hz", l_id, omit_inf=True)
        pmin = -2 * np.pi * np.max(poles.real)

        if strict and np.any(poles.real > 0):
            return np.nan, np.nan, np.nan

        tts_max = -np.log(target_err / np.linalg.cond(V)) / pmin
        tts_min = -np.log(target_err) / pmin
        return 0.5 * (tts_max + tts_min), tts_min, tts_max

    def errbound(self, error_type: str = "static"):
        """Worst- and best-case relative transfer error of the circuit.

        Returns ``(maxerr, y, minerr, z)`` where ``y``/``z`` are the unit input
        vectors realising the maximal and minimal relative error. ``error_type``
        is ``'static'`` (steady-state) or ``'dynamic'`` (the coefficients at
        ``t = 0`` of the free evolution). If the ideal transfer is singular the
        relative error is unbounded above (``maxerr = inf``); if the absolute
        error is singular it is sharply bounded below by zero (``minerr = 0``).
        """
        # Ideal transfer T = -inv(Gf) and the realised composite transfer.
        T_exac = -scipy.linalg.solve(self.feedback_array.G, np.eye(self.feedback_array.M))
        T_comp = scipy.linalg.solve(self.transfer("inverting"), np.eye(self.n))

        ix = np.ix_(self.output_mask, self.input_mask_i)
        T_exac = T_exac[ix]
        T_comp = T_comp[ix]
        T_exac_pinv = np.linalg.pinv(T_exac)

        rel = T_comp @ T_exac_pinv
        if error_type.lower() == "static":
            rel = rel - T_exac @ T_exac_pinv

        # Maximal error: nullspace of the ideal transfer makes it unbounded,
        # otherwise it is the largest singular value of the relative-error map.
        N = scipy.linalg.null_space(T_exac)
        if N.size == 0:
            _, s, Vh = np.linalg.svd(rel)
            maxerr = float(s[0])
            y = T_exac_pinv @ Vh[0].conj()
            y = y / np.linalg.norm(y)
        else:
            maxerr = np.inf
            y = N[:, 0]

        # Minimal error: nullspace of the absolute error bounds it by zero,
        # otherwise it is the smallest singular value of the same map.
        N = scipy.linalg.null_space(T_comp - T_exac)
        if N.size == 0:
            _, s, Vh = np.linalg.svd(rel)
            minerr = float(s[-1])
            z = T_exac_pinv @ Vh[-1].conj()
            z = z / np.linalg.norm(z)
        else:
            minerr = 0.0
            z = N[:, 0]

        return maxerr, y, minerr, z

    # --- SPICE netlist + invocation ---
    def netlist(self, solver_type: str, sp_analysis: _spice.SpiceAnalysis):
        """Generate LTSpice netlist(s) for the circuit.

        Returns ``(net_paths, raw_paths)`` as lists (one element for tran/op,
        one per output for noise).
        """
        return build_netlist(self, solver_type, sp_analysis)

    def _spice_tran(self, solver_type, sp_analysis):
        """Run a SPICE .tran. Returns (V_raw nxnpts, t_raw) on the native grid."""
        net, raw = self.netlist(solver_type, sp_analysis)
        res = _spice.run_spice(net, raw)
        n = self.feedback_array.M
        t_raw = np.asarray(res["time_vect"], dtype=float).ravel()
        V_raw = _spice._spice_node_matrix(res, [f"V(Vout_{i + 1})" for i in range(n)])
        return V_raw, t_raw

    def _spice_op(self, solver_type, sp_analysis):
        """Run a SPICE .op and return the output voltage vector."""
        net, raw = self.netlist(solver_type, sp_analysis)
        res = _spice.run_spice(net, raw)
        n = self.feedback_array.M
        V = _spice._spice_node_matrix(res, [f"V(Vout_{i + 1})" for i in range(n)])
        return V[:, 0]

    def tran(
        self,
        solver_type: str = "model",
        time: np.ndarray | float | _spice.SpiceAnalysis | None = None,
        return_report: bool = False,
        *,
        tstep: float | None = None,
        dtmax: float | None = None,
    ):
        """Run transient analysis. Returns (Vout, time_out) or (Vout, time_out, R)."""
        if solver_type.lower() == "spice":
            solver_type = "SPICE/fast"
        is_spice = solver_type in _spice._SPICE_FOOTERS

        if solver_type.lower() != "model" and not is_spice:
            raise NotImplementedError(f"Solver {solver_type!r} not yet implemented.")

        # SpiceAnalysis or explicit timestep controls only make sense for SPICE.
        sp_obj = time if isinstance(time, _spice.SpiceAnalysis) else None
        if sp_obj is not None:
            time = None
        if not is_spice and (sp_obj is not None or tstep is not None or dtmax is not None):
            raise ValueError("tstep/dtmax/SpiceAnalysis are only valid for SPICE solvers.")

        if sp_obj is None:
            if time is None:
                raise ValueError("time array is required for model solver.")
            if np.isscalar(time):
                time = np.array([0.0, float(time)])
            time = np.asarray(time, dtype=float).ravel()

        if is_spice:
            if sp_obj is not None:
                sp = sp_obj
            else:
                sp_kw = {}
                if tstep is not None:
                    sp_kw["Tstep"] = float(tstep)
                if dtmax is not None:
                    sp_kw["dTmax"] = float(dtmax)
                sp = _spice.SpiceAnalysis(
                    "tran", Tstart=float(time[0]), Tstop=float(time[-1]), **sp_kw
                )
            V_raw, t_raw = self._spice_tran(solver_type, sp)
            # sample DAC inputs on the SPICE native time grid for the report
            Vin = (
                np.vstack([self.voltage_dac[idx].todouble(t_raw) for idx in self.input_mask_v])
                if (self.p > 0 and self.input_mask_v.size)
                else None
            )
            Iin = (
                np.vstack([self.current_dac[idx].todouble(t_raw) for idx in self.input_mask_i])
                if self.input_mask_i.size
                else None
            )
        else:
            # --- gather waveforms ---
            t_extra = []
            for idx in self.input_mask_v:
                t_extra.extend(self.voltage_dac[idx].extra_times(time).tolist())
            for idx in self.input_mask_i:
                t_extra.extend(self.current_dac[idx].extra_times(time).tolist())
            time_full = np.sort(np.unique(np.concatenate([time, t_extra])))

            if self.p > 0:
                Vin = (
                    np.vstack(
                        [self.voltage_dac[idx].todouble(time_full) for idx in self.input_mask_v]
                    )
                    if self.input_mask_v.size
                    else None
                )
            else:
                Vin = None

            Iin = (
                np.vstack([self.current_dac[idx].todouble(time_full) for idx in self.input_mask_i])
                if self.input_mask_i.size
                else None
            )

            V_raw, t_raw = nltran_model(self, time_full, Vin, Iin)

        # ADC + DSP post-processing
        adc_lat = max((m.latency for m in self._adcs), default=0.0)
        if hasattr(self.adc_model, "convert"):
            Vout_adc, _, _ = self.adc_model.convert(V_raw[self.output_mask, :])
        else:
            Vout_adc = V_raw[self.output_mask, :]
        Vout_dsp, dsp_lat, _ = self.dsp_model.eval(Vout_adc)
        time_adc = t_raw + adc_lat

        Vout = Vout_dsp
        t_out = time_adc

        # The report quantities are also needed for continuous benchmarking.
        if return_report or self.bench_enable:
            T = self._transfer_core()
            # Rebase the DAC waveforms onto the solver's output grid when the
            # model solver changed the number of time points (nltran saturation
            # subdivision); otherwise V_raw (on t_raw) and Vin/Iin (on time_full)
            # would not align. The SPICE branch
            # already sampled Vin/Iin on t_raw.
            if not is_spice and len(t_raw) != len(time_full):
                Vin = (
                    np.vstack([self.voltage_dac[idx].todouble(t_raw) for idx in self.input_mask_v])
                    if (self.p > 0 and self.input_mask_v.size)
                    else None
                )
                Iin = (
                    np.vstack([self.current_dac[idx].todouble(t_raw) for idx in self.input_mask_i])
                    if self.input_mask_i.size
                    else None
                )
            Vin_r = Vin if Vin is not None else np.empty((0, V_raw.shape[1]))
            Iin_r = Iin if Iin is not None else np.empty((0, V_raw.shape[1]))
            Ijunc = T[2] @ V_raw
            if Iin_r.size:
                Ijunc = Ijunc + Iin_r
            if Vin_r.size:
                Ijunc = Ijunc + T[0] @ Vin_r
            Vjunc = scipy.linalg.solve(T[1] + T[3], Ijunc)
            Iout = T[4] @ V_raw - T[5] @ Vjunc
            I_Vin = (T[6] @ Vin_r - T[7] @ Vjunc) if Vin_r.size else np.empty((0, V_raw.shape[1]))

            # --- dissipated power and energy ---
            Vsup = np.array([np.diff(np.asarray(m.Vsupply, float))[0] for m in self.oa_models])[
                :, np.newaxis
            ]
            Vsup = np.broadcast_to(Vsup, V_raw.shape).copy()
            inf_mask = np.isinf(Vsup)
            Vsup[inf_mask] = V_raw[inf_mask]
            P_OA = np.sum((np.abs(Vsup) - np.abs(V_raw)) * np.abs(Iout), axis=0)
            P_IA = (
                np.sum(np.abs(I_Vin) * np.abs(Vin_r), axis=0)
                if Vin_r.size
                else np.zeros(V_raw.shape[1])
            )
            P_FA = np.sum(np.abs(Iout) * np.abs(V_raw), axis=0)
            P_tot = P_IA + P_FA + P_OA
            power = {"OA": P_OA, "InputArray": P_IA, "FeedbackArray": P_FA, "Total": P_tot}
            if len(t_raw) > 1:
                energy = {
                    "InputArray": float(np.trapezoid(P_IA, t_raw)),
                    "FeedbackArray": float(np.trapezoid(P_FA, t_raw)),
                    "OA": float(np.trapezoid(P_OA, t_raw)),
                }
                energy["Total"] = energy["InputArray"] + energy["FeedbackArray"] + energy["OA"]
            else:
                energy = {
                    "InputArray": np.nan,
                    "FeedbackArray": np.nan,
                    "OA": np.nan,
                    "Total": np.nan,
                }

        # --- continuous benchmark accumulation ---
        if self.bench_enable:
            self._ops += self.complexity
            t_end = float(time_adc[-1]) if len(time_adc) else 0.0
            # computing dynamic energy (InputArray=row2, FeedbackArray=row3, OA=row4)
            self._energy_breakdown[2, 3] += energy["InputArray"]
            self._energy_breakdown[3, 3] += energy["FeedbackArray"]
            self._energy_breakdown[4, 3] += energy["OA"]
            # static (leakage) energies on peripherals
            for a in self._adcs:
                a.add_energy("Static", a.leakage_power * t_end)
            for idx in self.input_mask_v:
                d = self.voltage_dac[idx]
                d.add_energy("Static", d.leakage_power * t_end)
            for idx in self.input_mask_i:
                d = self.current_dac[idx]
                d.add_energy("Static", d.leakage_power * t_end)
            for m in self.oa_models:
                m.add_energy("Static", m.leakage_power * t_end)
            self.dsp_model.add_energy("Static", self.dsp_model.leakage_power * dsp_lat)
            # power: weighted-mean / max over runtime
            periph_leak = (
                sum(a.leakage_power for a in self._adcs)
                + sum(self.voltage_dac[i].leakage_power for i in self.input_mask_v)
                + sum(self.current_dac[i].leakage_power for i in self.input_mask_i)
                + sum(m.leakage_power for m in self.oa_models)
                + self.dsp_model.leakage_power
            )
            lat_tot = (
                self._latency["Programming"]
                + self._latency["Computing"]
                + self._latency["PostProcessing"]
            )
            denom = lat_tot + t_end
            if denom > 0:
                new_mean = (
                    self._power[0] * lat_tot / denom + np.mean(P_tot + periph_leak) * t_end / denom
                )
            else:
                new_mean = self._power[0]
            self._power = np.array([new_mean, max(self._power[1], np.max(P_tot + periph_leak))])
            self._latency["Computing"] += t_end
            self._latency["PostProcessing"] += dsp_lat

        if return_report:
            R = TranReport(
                solver_type=solver_type,
                time=t_raw,
                Vout=V_raw,
                Voainn=T[8] @ Vjunc + T[9] @ V_raw,
                Voainp=np.zeros((self.n, V_raw.shape[1])),
                Iout=Iout,
                Vin=Vin_r,
                I_Vin=I_Vin,
                Iin=Iin_r,
                power=power,
                energy=energy,
            )
            R.VoutADC = Vout
            R.TimeADC = t_out
            return Vout, t_out, R

        return Vout, t_out

    # --- op (public) ---
    def op(self, solver_type: str = "model", return_report: bool = False):
        """Run operating-point analysis. Returns (Vout, time=0) or (Vout, time, R)."""
        if solver_type.lower() == "spice":
            solver_type = "SPICE/fast"
        is_spice = solver_type in _spice._SPICE_FOOTERS
        if solver_type.lower() != "model" and not is_spice:
            raise NotImplementedError(f"Solver {solver_type!r} not yet implemented.")

        if self.p > 0 and self.input_mask_v.size:
            Vin_raw = np.vstack(
                [self.voltage_dac[idx].todouble(np.array([0.0])) for idx in self.input_mask_v]
            )
            Vin = Vin_raw if Vin_raw.shape[0] > 0 else np.zeros((self.p, 1))
        else:
            Vin = np.zeros((self.p, 1)) if self.p > 0 else None

        if self.input_mask_i.size:
            Iin_raw = np.vstack(
                [self.current_dac[idx].todouble(np.array([0.0])) for idx in self.input_mask_i]
            )
            Iin = Iin_raw if Iin_raw.shape[0] > 0 else np.zeros((self.n, 1))
        else:
            Iin = np.zeros((self.n, 1))

        Vin_1d = Vin[:, 0] if (Vin is not None and Vin.size > 0) else None
        Iin_1d = Iin[:, 0]

        if is_spice:
            Vout_raw = self._spice_op(solver_type, _spice.SpiceAnalysis("op"))
        else:
            Vout_raw, _, info = nlop_model(self, Vin_1d, Iin_1d)
        Vout = Vout_raw[self.output_mask]

        if return_report:
            T = self._transfer_core()
            Vin_r = Vin if Vin is not None else np.empty((0, 1))
            Iin_r = Iin
            Ijunc = T[2] @ Vout_raw.reshape(-1, 1)
            if Iin_r.size:
                Ijunc = Ijunc + Iin_r
            if Vin_r.size:
                Ijunc = Ijunc + T[0] @ Vin_r
            Vjunc = scipy.linalg.solve(T[1] + T[3], Ijunc)
            R = {
                "Vout": Vout_raw,
                "Voainn": (T[8] @ Vjunc + T[9] @ Vout_raw.reshape(-1, 1)).ravel(),
                "Voainp": np.zeros(self.n),
                "Iout": (T[4] @ Vout_raw.reshape(-1, 1) - T[5] @ Vjunc).ravel(),
                "Vin": Vin_r.ravel() if Vin_r.size else np.array([]),
                "I_Vin": (T[6] @ Vin_r - T[7] @ Vjunc).ravel() if Vin_r.size else np.array([]),
                "Iin": Iin_r.ravel(),
                "time": np.array([0.0]),
            }
            return Vout, np.array([0.0]), R

        return Vout, np.array([0.0])

    # --- transfer + noise ---
    def transfer(self, kind: str = "inverting", l_id: np.ndarray | None = None):
        """Composite input/output transfer matrix.

        ``kind`` is 'inverting' (current-on-inverting-input -> Vout) or
        'noninverting' (voltage-on-noninverting-input -> Vout). Returns the
        matrix; only the 'matrix' output format is supported.
        """
        n = self.n
        if l_id is None:
            l_id = np.ones(n, dtype=bool)
        l_id = np.asarray(l_id, dtype=bool).ravel()

        A0 = np.array([float(np.atleast_1d(m.A0)[0]) for m in self.oa_models])
        Ai = np.diag(1.0 / (self.sign_vector * A0))
        T = self._transfer_core()
        ix = np.ix_(l_id, l_id)

        if kind.lower() == "inverting":
            return (T[1][ix] + T[3][ix]) @ scipy.linalg.solve(T[8][ix], Ai[ix] - T[9][ix]) - T[2][
                ix
            ]
        elif kind.lower() == "noninverting":
            return T[9][ix] + T[8][ix] @ scipy.linalg.solve(T[1][ix] + T[3][ix], T[2][ix]) - Ai[ix]
        raise ValueError(f"Unknown transfer kind {kind!r}")

    def noise(self, solver_type: str = "SPICE/fast", *args):
        """Perform a .noise analysis.

        Returns ``(Sout, freq, R)`` where R is a NoiseReport. ``args`` may be:
        empty (default sweep), an LTSpice-style ``Fmax[, Fmin, Fstep, Fsweep]``,
        a SpiceAnalysis, or (model only) an explicit frequency vector.
        """
        if solver_type.lower() == "spice":
            solver_type = "SPICE/fast"
        is_spice = solver_type in _spice._SPICE_FOOTERS

        # --- resolve the analysis description ---
        if len(args) == 0:
            sp = _spice.SpiceAnalysis("noise")
        elif isinstance(args[0], _spice.SpiceAnalysis):
            sp = args[0]
        elif np.isscalar(args[0]):
            kw = dict(Fmax=float(args[0]))
            if len(args) > 1:
                kw["Fmin"] = float(args[1])
            if len(args) > 2:
                kw["Fstep"] = float(args[2])
            if len(args) > 3:
                kw["Fsweep"] = args[3]
            sp = _spice.SpiceAnalysis("noise", **kw)
        else:  # explicit frequency vector (model only)
            sp = np.asarray(args[0], dtype=float).ravel()

        # --- dispatch ---
        if is_spice:
            if not isinstance(sp, _spice.SpiceAnalysis):
                raise ValueError("SPICE noise requires a SpiceAnalysis or Fmax.")
            Sout, freq = noise_spice(self, solver_type, sp)
        else:
            freq = sp.to_double() if isinstance(sp, _spice.SpiceAnalysis) else sp
            Sout, freq = noise_model(self, freq)

        Pout = np.sqrt(np.trapezoid(Sout**2, freq, axis=1))
        R = NoiseReport(solver_type=solver_type, freq=freq, Sout=Sout, Pout=Pout)
        return Sout, freq, R

    # --- display: six property groups. The concise ---
    # n/p/lr summary becomes the header line above the grouped properties.
    def _display_header(self) -> str:
        return f"LoopUS(n={self.n}, p={self.p}, lr={self.feedback_array.line_resistance[0]:.3g})"

    def _property_groups(self):
        has_v = bool(self.voltage_dac)
        has_i = bool(self.current_dac)

        cfg = {"sign_vector": self.sign_vector}
        if has_i and not has_v:
            cfg["input_mask_i"] = self.input_mask_i
        elif has_v and not has_i:
            cfg["input_mask_v"] = self.input_mask_v
        else:
            cfg["input_mask_i"] = self.input_mask_i
            cfg["input_mask_v"] = self.input_mask_v
        cfg["output_mask"] = self.output_mask

        periph = {"oa_models": self.oa_models}
        if has_v:
            periph["voltage_dac"] = self.voltage_dac
        if has_i:
            periph["current_dac"] = self.current_dac
        periph["adc_model"] = self.adc_model
        periph["mem_write"] = self.mem_write
        periph["dsp_model"] = self.dsp_model

        p = np.asarray(self.power, float).ravel()
        return [
            (
                "Memory arrays",
                {
                    "input_array": self.input_array,
                    "feedback_array": self.feedback_array,
                    "xbar_orient": self.xbar_orient,
                },
            ),
            ("Circuit configuration", cfg),
            ("Circuit peripherals", periph),
            (
                "Reference signals",
                {
                    "input_reference": self.input_reference,
                    "oa_reference": self.oa_reference,
                    "initial_condition": self.initial_condition,
                },
            ),
            (
                "Performance",
                {
                    "Latency": self.latency,
                    "Energy": self.energy,
                    "Power": f"{si_format(p[0])}W|{si_format(p[1])}W",
                    "Area": self.area,
                    "Complexity": self.complexity,
                    "OPs": f"{si_format(self.ops)}OPs",
                    "Throughput": f"{si_format(self.throughput)}OPS",
                    "EnergyEfficiency": f"{si_format(self.energy_efficiency)}OPS/W",
                    "AreaEfficiency": f"{si_format(self.area_efficiency / 1e6)}OPS/mm2",
                },
            ),
            (
                "Debug info",
                {
                    "verbose": self.verbose,
                    "bench_enable": self.bench_enable,
                    "file_path": self.file_path,
                    "file_name": self.file_name,
                },
            ),
        ]
