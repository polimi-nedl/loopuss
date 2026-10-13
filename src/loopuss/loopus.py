"""LoopUS - main closed-loop in-memory computing circuit class."""

from __future__ import annotations

import atexit
import os
import shutil
import warnings
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
def _dc_reference(ref, name: str) -> VoltageSignal:
    """Validate a reference signal: DC only, with EMPTY read as DC 0 V.

    The model solvers treat references as constant, so a time-varying one would
    be simulated by SPICE but not by the model.
    """
    if not isinstance(ref, VoltageSignal):
        raise TypeError(f"{name} must be a VoltageSignal or a sequence of them.")
    if ref.is_empty():
        return VoltageSignal("DC", dcvalue=0.0)
    if not ref.is_dc():
        raise ValueError(f"{name} must be a DC signal, got {ref.s_type}.")
    if ref.dcvalue is None:
        raise ValueError(f"{name} is a DC signal without a dcvalue.")
    return ref


class LoopUS(GroupedDisplay):
    """Open- or closed-loop analog in-memory computing circuit.

    The circuit has ``n`` op-amps. Row ``i`` of the ``n × n`` feedback array
    and row ``i`` of the ``n × p`` input array meet at a junction wired to one
    input of op-amp ``i``, whose output drives column ``i`` of the feedback
    array. Inputs are voltages on the input-array columns (one
    :class:`~loopuss.VoltageDAC` per column) and currents injected at the
    junctions (one :class:`~loopuss.CurrentDAC` per row). With ideal op-amps
    and ideal lines the circuit settles on

    .. math:: G_f V_{out} = -(I_{in} + G_{in} V_{in}).

    Every analysis takes a ``solver_type``: ``"model"`` runs the analytical
    solvers, while ``"SPICE/fast"``, ``"SPICE/accurate"`` and
    ``"SPICE/extreme"`` run LTspice with increasingly tight tolerances.
    ``"spice"`` is an alias of ``"SPICE/fast"``.

    Construction programs both arrays once through ``mem_write_model`` (see
    :meth:`reprogram`).

    Parameters
    ----------
    input_array : CrossBar, optional
        Input array, ``n × p``. Default: an empty ``n × 0`` array, i.e. current
        inputs only.
    feedback_array : CrossBar
        Feedback array, ``n × n``. Required.
    oa_model : OAModel or list of OAModel, optional
        Op-amp model. A single model, or a one-element list, is copied for
        every op-amp; otherwise give one per row. Default ``OAModel("1P")``:
        80 dB gain, 100 MHz gain-bandwidth and no supply limits.
    adc_model : ADCModel, optional
        Output converter. The area and leakage count
        ``ceil(len(output_mask) / Nchannel)`` of them, and every transient
        costs one conversion (``dynamic_energy``) per masked output. Default:
        ideal.
    dsp_model : DSPModel, optional
        Post-processing applied after the ADC. Default: bypass.
    mem_write_model : MemWriteModel, optional
        Programming peripheral. Default: ideal.
    voltage_dac : VoltageDAC, optional
        Template copied for every input-array column. Default: ideal.
    current_dac : CurrentDAC, optional
        Template copied for every row. Default: ideal.
    sign_vector : float or array_like, optional
        ``-1`` wires row ``i`` to the inverting input of op-amp ``i``, ``+1`` to
        the non-inverting input; the other input goes to ``oa_reference``. A
        scalar applies to every op-amp. Default ``-1``.
    output_mask : array_like of int, optional
        0-based indices of the op-amps whose outputs the analyses return.
        Default: all.
    input_mask_i : array_like of int, optional
        0-based rows that receive current inputs. Default: all.
    input_mask_v : array_like of int, optional
        0-based input-array columns that receive voltage inputs. Default: all.
    oa_reference : VoltageSignal, optional
        DC reference on the free op-amp input. Default: DC 0 V. EMPTY reads as
        0 V; other signal types raise ``ValueError``.
    input_reference : VoltageSignal or sequence of VoltageSignal, optional
        DC reference of the input-array voltage sources: one signal for every
        column, or one per column (length ``p``). Default: DC 0 V. Stored as a
        list of ``p`` signals. EMPTY reads as 0 V; other signal types raise
        ``ValueError``.
    initial_condition : array_like, shape (n,), optional
        Op-amp outputs at the start of a transient [V]. Default: the operating
        point with every input at zero.
    xbar_orient : list, optional
        Where each array is wired:
        ``[[input_column_end, input_row_end], [feedback_column_end,
        feedback_row_end]]``. Column ends are ``"bottom"``/``"top"`` and row
        ends ``"left"``/``"right"``. This only matters with non-zero line
        resistance. Default ``[["bottom", "right"], ["bottom", "left"]]``.
    bench_enable : bool, optional
        Accumulate latency, energy, power and operation count on every
        :meth:`tran` and :meth:`reprogram`, for :meth:`benchmark`. Default
        ``False``.
    complexity : float, optional
        Operations credited to each transient run [OPs]. Default ``n**3``.
    file_path : str, optional
        Working directory for SPICE files. Default ``simfolder/<pid>`` under
        the current directory, private to the process and deleted at exit. An
        explicit path is never deleted.
    file_name : str, optional
        Base name of the netlist and ``.raw`` files. Default
        ``"spiceSimulation"``.
    num_workers : int, optional
        LTspice processes run in parallel when an analysis needs several
        netlists (:meth:`noise` uses one per output). Default: the physical
        cores available.
    nl_cache_size : int, optional
        Number of saturation states whose eigendecomposition :meth:`pz`
        caches, capped at 10000. Default 10000.
    """

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
        file_path: str | None = None,
        file_name: str = "spiceSimulation",
        num_workers: int | None = None,
        nl_cache_size: int = 10000,
    ):

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
        self._peak_power = 0.0

        # --- caches ---
        self._trans_matrix: list | None = None
        self._trans_key: tuple | None = None
        self._eigen_key: tuple | None = None
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

    @property
    def oa_reference(self) -> VoltageSignal:
        """DC reference on the free op-amp input."""
        return self._oa_reference

    @oa_reference.setter
    def oa_reference(self, ref):
        self._oa_reference = _dc_reference(ref, "oa_reference")

    @property
    def input_reference(self) -> list[VoltageSignal]:
        """DC reference of each input-array voltage source, one per column."""
        return self._input_reference

    @input_reference.setter
    def input_reference(self, ref):
        refs = [ref] if isinstance(ref, VoltageSignal) else list(ref)
        refs = [_dc_reference(r, "input_reference") for r in refs]
        if len(refs) == 1:
            refs = [deepcopy(refs[0]) for _ in range(self.p)]
        elif len(refs) != self.p:
            raise ValueError(
                f"input_reference needs 1 or {self.p} signals (one per column), got {len(refs)}."
            )
        self._input_reference = refs

    def _input_bias(self) -> np.ndarray:
        """Input reference of each column, shape (p,) [V]."""
        return np.array([r.dcvalue for r in self.input_reference], dtype=float)

    # --- benchmark properties ---
    _ENERGY_LABELS = ["MemWrite", "DAC", "InputArray", "FeedbackArray", "OA", "ADC", "DSP"]

    @property
    def area(self) -> dict:
        """Area of each component [m²].

        Keys: ``MemWrite`` (one driver area times ``sum(parallel_write)``),
        ``DAC``, ``InputArray``, ``FeedbackArray``, ``OA``, ``ADC``, ``DSP``
        and ``Total``.
        """
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
        """Accumulated energy [J].

        ``Breakdown`` is a 7x6 table: rows = MemWrite/DAC/InputArray/FeedbackArray/
        OA/ADC/DSP (also in ``Labels``); cols = Programming(S,D), Computing(S,D),
        PostProcessing(S,D). ``Total`` is its sum, and each label maps to a dict
        with the per-phase ``Static``/``Dynamic``/``Total`` values.
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
        """Accumulated latency [s]: ``Programming``, ``Computing``,
        ``PostProcessing`` and ``Total``."""
        L = dict(self._latency)
        L["Total"] = L["Programming"] + L["Computing"] + L["PostProcessing"]
        return L

    @property
    def power(self) -> np.ndarray:
        """``[mean, peak]`` power, including peripheral leakage [W].

        The mean is ``energy["Total"] / latency["Total"]``, every phase
        included; the peak is the largest instantaneous power of any
        transient plus the peripheral leakage.
        """
        lat = self.latency["Total"]
        mean = self.energy["Total"] / lat if lat > 0 else 0.0
        return np.array([mean, self._peak_power])

    @property
    def ops(self) -> float:
        """Accumulated operations [OPs]."""
        return self._ops

    @property
    def throughput(self) -> float:
        """``ops / latency["Total"]`` [OPS]."""
        return _safe_div(self._ops, self.latency["Total"])

    @property
    def energy_efficiency(self) -> float:
        """``ops / energy["Total"]`` [OPS/W]."""
        return _safe_div(self._ops, self.energy["Total"])

    @property
    def area_efficiency(self) -> float:
        """``throughput / area["Total"]`` [OPS/m²]."""
        return _safe_div(self.throughput, self.area["Total"])

    def benchmark(self, *flags) -> BenchReport:
        """Snapshot the accumulated metrics into a report.

        Metrics accumulate only when ``bench_enable`` is set.

        Parameters
        ----------
        *flags : str
            ``"-nowrite"`` removes the programming latency, energy (the
            MemWrite row) and area from the report, and its mean power is
            computed without them. Other flags are ignored.

        Returns
        -------
        BenchReport
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

        # From the report's own totals, so "-nowrite" also applies to it.
        mean_power = energy["Total"] / latency["Total"] if latency["Total"] > 0 else 0.0

        return BenchReport(
            latency=latency,
            energy=energy,
            power=np.array([mean_power, self._peak_power]),
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
        """Program the arrays through ``mem_write_model``.

        Parameters
        ----------
        new_input, new_feedback : CrossBar, optional
            Replacement arrays, programmed and then stored. They must keep the
            circuit's shape. With neither given, the current arrays are
            programmed again, which adds another round of programming error on
            top of the existing one.

        Notes
        -----
        Clears the transfer-matrix and pole caches when an array changes. The
        programming latency is added to the benchmark only with
        ``bench_enable``; the programming energy always accumulates in
        ``mem_write_model``.
        """
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
        """Load input waveforms into the DACs.

        Parameters
        ----------
        *args : Signal or list of Signal
            Each argument is a signal or a list of signals of one kind.
            Current signals go, in order, to the rows in ``input_mask_i``, and
            voltage signals to the columns in ``input_mask_v``. Short lists
            are padded with ``EMPTY`` signals; an ``EMPTY`` voltage holds its
            column at the input reference (DAC output 0 V) in every solver.
            Each signal is delayed by its DAC's ``latency``.
        """
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
        """Set ``output_mask`` (0-based op-amp indices) and resize the ADC bank."""
        self.output_mask = np.asarray(mask, dtype=int).ravel()
        n_out = len(self.output_mask)
        nc = max(1, int(np.ceil(n_out / self.adc_model.Nchannel)))
        spent = [a.energy for a in self._adcs]
        self._adcs = [deepcopy(self.adc_model) for _ in range(nc)]
        # Keep the energy the old ADC copies accumulated in the benchmark.
        for kind in ("Static", "Dynamic"):
            self._adcs[0].add_energy(kind, sum(e[kind] for e in spent))

    def set_initial_condition(self, ic: np.ndarray) -> None:
        """Set the op-amp outputs at the start of a transient [V]; a scalar
        applies to every op-amp."""
        ic = np.asarray(ic, dtype=float).ravel()
        if ic.size == 1:
            ic = np.full(self.n, ic[0])
        self.initial_condition = ic[: self.n]

    # --- cache keys ---
    # The transfer matrices and the poles are cached. Rather than relying on
    # every change going through a setter (``oa_models[0].GBWP = ...`` does
    # not), each cache remembers what it was built from and is rebuilt when
    # that differs. Arrays are compared by identity: assign a new CrossBar,
    # or call reprogram(), to change one.
    def _array_key(self) -> tuple:
        orient = tuple(tuple(str(x).lower() for x in end) for end in self.xbar_orient)
        return (self.input_array, self.feedback_array, orient)

    def _pole_key(self) -> tuple:
        oas = tuple(
            (
                str(m.type).upper(),
                float(np.atleast_1d(m.A0)[0]),
                float(np.atleast_1d(m.FP)[0]),
                float(m.GBWP),
                float(m.input_capacitance),
            )
            for m in self.oa_models
        )
        return self._array_key() + (np.asarray(self.sign_vector, float).tobytes(), oas)

    @staticmethod
    def _same_key(a, b) -> bool:
        if a is None or b is None or len(a) != len(b):
            return False
        # arrays by identity, the rest by value
        return a[0] is b[0] and a[1] is b[1] and a[2:] == b[2:]

    # --- transfer core (T1-T10) ---
    def _transfer_core(self) -> list:
        """Compute and cache the 10 characteristic transfer matrices."""
        key = self._array_key()
        if self._trans_matrix is not None and self._same_key(key, self._trans_key):
            return self._trans_matrix
        self._trans_key = key

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
    ):
        """Poles of the linearized circuit.

        The poles are the eigenvalues of ``diag(s * P0) @ Xhat - diag(FP)``.
        Here ``P0 = A0 * FP`` is each op-amp's gain-bandwidth product
        (``GBWP`` for ``"GM"`` models) and ``s`` is ``sign_vector``. ``Xhat``
        is the divider from the op-amp outputs to their inputs. The problem is
        normalized by the geometric mean of ``P0``, and results are cached for
        each ``l_id``.

        Parameters
        ----------
        units : {"hz", "rad"}, optional
            ``"hz"`` (default) or ``"rad"`` (multiplied by 2π).
        l_id : array_like of bool, shape (n,), optional
            Op-amps in their linear region. Saturated op-amps leave the loop
            and contribute a pole at ``-inf``. Default: all linear.
        omit_inf : bool, optional
            Drop the ``-inf`` poles and the matching rows and columns of the
            other outputs. Default ``False``.

        Returns
        -------
        poles : ndarray
            Poles, sorted by decreasing real part.
        V : ndarray
            Eigenvectors, one per column; ``NaN`` for saturated op-amps.
        l : ndarray
            Normalized eigenvalues, i.e. the poles divided by the mean ``P0``.
        Xhat : ndarray
            Divider matrix restricted to ``l_id``.

        Raises
        ------
        ValueError
            If ``units`` is neither ``"hz"`` nor ``"rad"``, or if an op-amp
            is not ``"1P"`` or ``"GM"``. Since :meth:`tts`, the model transient
            and the model noise analysis call ``pz``, they raise as well.
        NotImplementedError
            If an array or op-amp has non-zero capacitance.
        """
        n = self.n
        if l_id is None:
            l_id = np.ones(n, dtype=bool)
        l_id = np.asarray(l_id, dtype=bool).ravel()

        if units.lower() not in ("hz", "rad"):
            raise ValueError(f"units must be 'hz' or 'rad', not {units!r}.")
        unsupported = sorted({m.type for m in self.oa_models if m.type.upper() not in ("1P", "GM")})
        if unsupported:
            raise ValueError(
                f"The analytical model supports only '1P' and 'GM' op-amps, got {unsupported}. "
                "Use a SPICE solver for multi-pole op-amps."
            )

        # The cache holds poles for the parameters it was built from.
        key = self._pole_key()
        if self._eigen_decomp is not None and not self._same_key(key, self._eigen_key):
            self._eigen_decomp = None
        self._eigen_key = key

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
        """Estimate the time-to-solution.

        ``tts_min`` and ``tts_max`` are the lazy and greedy bounds on the time
        the dynamic error norm takes to fall below ``target_err``. With ``p``
        the slowest pole in rad/s, ``-2π max Re(poles)``, they are
        ``-ln(target_err) / p`` and ``-ln(target_err / cond(V)) / p``, where
        ``V`` holds the eigenvectors returned by :meth:`pz`.

        Parameters
        ----------
        target_err : float, optional
            Error threshold. Default ``exp(-2π)``, the settling threshold of
            an equivalent single-pole stage.
        strict : bool, optional
            Return ``NaN`` if any pole has a positive real part. Default
            ``False``.
        l_id : array_like of bool, shape (n,), optional
            Op-amps in their linear region, see :meth:`pz`.

        Returns
        -------
        tts : float
            Mean of the two bounds [s].
        tts_min, tts_max : float
            Lazy and greedy bounds [s].
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
        """Worst- and best-case relative error of the circuit transfer.

        Compares the realized current-to-output transfer, the inverse of
        :meth:`transfer`, with the ideal ``-inv(G_f)``. Both are restricted to
        ``output_mask`` × ``input_mask_i``.

        Parameters
        ----------
        error_type : {"static", "dynamic"}, optional
            ``"static"`` (default) for the steady-state error, ``"dynamic"``
            for the coefficients at ``t = 0`` of the free evolution.

        Returns
        -------
        maxerr : float
            Largest relative error over unit inputs; ``inf`` if the ideal
            transfer is singular.
        y : ndarray
            Unit input that realizes ``maxerr``.
        minerr : float
            Smallest relative error; 0 if the absolute error is singular.
        z : ndarray
            Unit input that realizes ``minerr``.

        Raises
        ------
        ValueError
            For an ``error_type`` other than ``"static"`` or ``"dynamic"``.
        """
        kind = str(error_type).lower()
        if kind not in ("static", "dynamic"):
            raise ValueError(f"error_type must be 'static' or 'dynamic', not {error_type!r}.")

        # Ideal transfer T = -inv(Gf) and the realised composite transfer.
        T_exac = -scipy.linalg.solve(self.feedback_array.G, np.eye(self.feedback_array.M))
        T_comp = scipy.linalg.solve(self.transfer("inverting"), np.eye(self.n))

        ix = np.ix_(self.output_mask, self.input_mask_i)
        T_exac = T_exac[ix]
        T_comp = T_comp[ix]
        T_exac_pinv = np.linalg.pinv(T_exac)

        rel = T_comp @ T_exac_pinv
        if kind == "static":
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
        """Write the LTspice netlist(s) of the circuit into ``file_path``.

        Parameters
        ----------
        solver_type : str
            A SPICE solver, which sets the tolerance options.
        sp_analysis : SpiceAnalysis
            Analysis to run.

        Returns
        -------
        net_paths, raw_paths : list of str
            One entry for ``.tran`` and ``.op``; one per masked output for
            ``.noise``.
        """
        return build_netlist(self, solver_type, sp_analysis)

    def _spice_tran(self, solver_type, sp_analysis):
        """Run a SPICE .tran.

        Returns (V_raw, Iout, t_raw) on the native grid; V_raw and Iout are n x npts.
        """
        net, raw = self.netlist(solver_type, sp_analysis)
        res = _spice.run_spice(net, raw)
        t_raw = np.asarray(res["time_vect"], dtype=float).ravel()
        V_raw, Iout = self._spice_outputs(res)
        return V_raw, Iout, t_raw

    def _spice_op(self, solver_type, sp_analysis):
        """Run a SPICE .op and return the output voltage and current vectors."""
        net, raw = self.netlist(solver_type, sp_analysis)
        res = _spice.run_spice(net, raw)
        V, Iout = self._spice_outputs(res)
        return V[:, 0], Iout[:, 0]

    def _spice_outputs(self, res):
        """Op-amp output voltages and currents from a parsed .raw."""
        n = self.feedback_array.M
        V = _spice._spice_node_matrix(res, [f"V(Vout_{i + 1})" for i in range(n)])
        # LTspice reports the current flowing into the subcircuit pin. We negate it
        # so that Iout is the current sourced by the op-amp output.
        Iout = -_spice._spice_node_matrix(res, [f"Ix(XOA{i + 1}:OUT)" for i in range(n)])
        return V, Iout

    def tran(
        self,
        solver_type: str = "model",
        time: np.ndarray | float | _spice.SpiceAnalysis | None = None,
        return_report: bool = False,
        *,
        tstep: float | None = None,
        dtmax: float | None = None,
    ):
        """Transient analysis.

        Parameters
        ----------
        solver_type : str, optional
            ``"model"`` (default) or a SPICE solver.
        time : array_like, float or SpiceAnalysis
            Time points [s], or a stop time meaning ``[0, time]``; SPICE
            solvers also accept a ``SpiceAnalysis``. The model solver adds the
            source breakpoints and saturation events to the grid. SPICE runs
            from ``time[0]`` to ``time[-1]`` and returns LTspice's own grid.
        return_report : bool, optional
            Also return a :class:`~loopuss.TranReport`. Default ``False``.
        tstep : float, optional
            SPICE initial timestep [s]. Default ``(Tstop - Tstart) / 100``.
        dtmax : float, optional
            SPICE maximum timestep [s]. Default 1000.

        Returns
        -------
        Vout : ndarray, shape (len(output_mask), T)
            Masked outputs after the ADC and DSP [V].
        time : ndarray, shape (T,)
            Time points shifted by the ADC latency [s].
        report : TranReport
            Only with ``return_report``.

        Raises
        ------
        ValueError
            If ``time`` is missing, or if ``tstep``, ``dtmax`` or a
            ``SpiceAnalysis`` is given to the model solver.
        NotImplementedError
            For an unknown ``solver_type``.

        Notes
        -----
        With ``bench_enable``, each call adds ``complexity`` operations, the
        computing and post-processing latency, the dissipated energy and the
        peripheral leakage to the benchmark. Every call also charges one ADC
        conversion per masked output to the ADC dynamic energy.
        """
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
            V_raw, Iout_spice, t_raw = self._spice_tran(solver_type, sp)
            # sample DAC inputs on the SPICE native time grid for the report
            Vin, Iin = self._input_matrices(t_raw)
        else:
            # --- gather waveforms ---
            t_extra = []
            for idx in self.input_mask_v:
                t_extra.extend(self.voltage_dac[idx].extra_times(time).tolist())
            for idx in self.input_mask_i:
                t_extra.extend(self.current_dac[idx].extra_times(time).tolist())
            # Drop breakpoints past the requested end.
            t_extra = np.asarray(t_extra, dtype=float)
            t_extra = t_extra[t_extra <= time[-1]]
            time_full = np.sort(np.unique(np.concatenate([time, t_extra])))

            Vin, Iin = self._input_matrices(time_full)
            V_raw, t_raw = nltran_model(self, time_full, Vin, Iin)

        # ADC + DSP post-processing
        adc_lat = max((m.latency for m in self._adcs), default=0.0)
        # Quantize outputs
        Vout_adc = self.adc_model.quantclip(V_raw[self.output_mask, :])
        n_out = len(self.output_mask)
        n_ch = self.adc_model.Nchannel
        # Add energy to the per-channel ADC copies (what the benchmark reads)
        for k, adc in enumerate(self._adcs):
            served = max(0, min(n_ch, n_out - k * n_ch))
            adc.add_energy("Dynamic", adc.dynamic_energy * served)
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
                Vin, Iin = self._input_matrices(t_raw)
            Vin_r = self._report_vin(Vin, V_raw.shape[1])
            Iin_r = Iin
            Ijunc = T[2] @ V_raw + Iin_r
            if Vin_r.size:
                Ijunc = Ijunc + T[0] @ Vin_r
            Vjunc = scipy.linalg.solve(T[1] + T[3], Ijunc)
            # SPICE measures the op-amp output currents directly.
            Iout = Iout_spice if is_spice else T[4] @ V_raw - T[5] @ Vjunc
            I_Vin = (T[6] @ Vin_r - T[7] @ Vjunc) if Vin_r.size else np.empty((0, V_raw.shape[1]))

            power, energy = self._dissipation(V_raw, Iout, Vin_r, I_Vin, t_raw)

        # --- continuous benchmark accumulation ---
        if self.bench_enable:
            self._ops += self.complexity
            # The inputs are applied at t = 0 and both solvers simulate from
            # there, so a run lasts until its last time point plus the
            # conversion, whatever time the requested grid starts at.
            t_run = float(time_adc[-1]) if len(time_adc) else 0.0
            # computing dynamic energy (InputArray=row2, FeedbackArray=row3, OA=row4)
            self._energy_breakdown[2, 3] += energy["InputArray"]
            self._energy_breakdown[3, 3] += energy["FeedbackArray"]
            self._energy_breakdown[4, 3] += energy["OA"]
            # static (leakage) energies on peripherals
            for a in self._adcs:
                a.add_energy("Static", a.leakage_power * t_run)
            for idx in self.input_mask_v:
                d = self.voltage_dac[idx]
                d.add_energy("Static", d.leakage_power * t_run)
            for idx in self.input_mask_i:
                d = self.current_dac[idx]
                d.add_energy("Static", d.leakage_power * t_run)
            for m in self.oa_models:
                m.add_energy("Static", m.leakage_power * t_run)
            self.dsp_model.add_energy("Static", self.dsp_model.leakage_power * dsp_lat)
            # Peak power: instantaneous dissipation plus peripheral leakage. The
            # mean power is derived from the accumulated energy (see ``power``).
            periph_leak = (
                sum(a.leakage_power for a in self._adcs)
                + sum(self.voltage_dac[i].leakage_power for i in self.input_mask_v)
                + sum(self.current_dac[i].leakage_power for i in self.input_mask_i)
                + sum(m.leakage_power for m in self.oa_models)
                + self.dsp_model.leakage_power
            )
            self._peak_power = max(self._peak_power, float(np.max(power["Total"] + periph_leak)))
            self._latency["Computing"] += t_run
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

    def _input_matrices(self, time):
        """DAC outputs sampled on ``time``, one row per line.

        Returns ``(Vin, Iin)``: ``Vin`` has shape ``(p, T)`` (``None`` without
        input columns) and ``Iin`` shape ``(n, T)``. Columns and rows outside
        ``input_mask_v``/``input_mask_i``, DACs with nothing loaded and EMPTY
        channels are 0, so such a column sits at its input reference.
        """
        t = np.asarray(time, dtype=float).ravel()
        Vin = None
        if self.p > 0:
            Vin = np.zeros((self.p, t.size))
            for idx in self.input_mask_v:
                v = self.voltage_dac[idx].todouble(t)
                if v.shape[0]:
                    # EMPTY voltages sample as NaN; the netlist holds that
                    # column at the reference, so the model does too.
                    Vin[idx] = np.nan_to_num(v[0], nan=0.0)
        Iin = np.zeros((self.n, t.size))
        for idx in self.input_mask_i:
            i = self.current_dac[idx].todouble(t)
            if i.shape[0]:
                Iin[idx] = i[0]
        return Vin, Iin

    def _report_vin(self, Vin, npts):
        """Input-column voltages for the reports: DAC waveforms plus the input reference."""
        Vin_r = Vin if Vin is not None else np.empty((0, npts))
        # The DACs sit on top of the input reference, so the columns are driven at
        # DAC + reference.
        in_bias = self._input_bias()
        if np.any(in_bias != 0):
            if not Vin_r.size:
                Vin_r = np.zeros((self.p, npts))
            Vin_r = Vin_r + in_bias[:, np.newaxis]
        return Vin_r

    def _dissipation(self, Vout, Iout, Vin, I_Vin, time):
        """Power and energy dissipated by op-amps and arrays.

        Arrays are (lines x len(time)). Returns ``(power, energy)`` dicts keyed by
        ``OA``, ``InputArray``, ``FeedbackArray`` and ``Total``. Energy is NaN
        for a single time point.
        """
        # OAs dissipate (Vdd - Vout) * I+ when sourcing current and
        # (Vout - Vss) * I- when sinking it; ideal (infinite) rails dissipate nothing.
        Vrail = np.array([np.asarray(m.Vsupply, float) for m in self.oa_models])
        Vss, Vdd = Vrail[:, [0]], Vrail[:, [1]]
        finite = ~np.isinf(Vrail).any(axis=1)
        P_OA = np.zeros_like(Vout)
        P_OA[finite] = (Vdd[finite] - Vout[finite]) * np.maximum(Iout[finite], 0) + (
            Vout[finite] - Vss[finite]
        ) * np.maximum(-Iout[finite], 0)
        P_OA = np.sum(P_OA, axis=0)
        P_IA = np.sum(np.abs(I_Vin) * np.abs(Vin), axis=0) if Vin.size else np.zeros(len(time))
        P_FA = np.sum(np.abs(Iout) * np.abs(Vout), axis=0)
        power = {"OA": P_OA, "InputArray": P_IA, "FeedbackArray": P_FA, "Total": P_IA + P_FA + P_OA}
        if len(time) > 1:
            energy = {
                k: float(np.trapezoid(power[k], time))  # Time integral
                for k in ("InputArray", "FeedbackArray", "OA")
            }
            energy["Total"] = energy["InputArray"] + energy["FeedbackArray"] + energy["OA"]
        else:
            energy = dict.fromkeys(("InputArray", "FeedbackArray", "OA", "Total"), np.nan)
        return power, energy

    # --- op (public) ---
    def op(self, solver_type: str = "model", return_report: bool = False):
        """Operating-point analysis.

        Sources are sampled at t = 0, so a STEP delayed by a non-zero DAC
        latency still reads its initial value. The ADC and DSP are not
        applied.

        Parameters
        ----------
        solver_type : str, optional
            ``"model"`` (default) or a SPICE solver.
        return_report : bool, optional
            Also return a :class:`~loopuss.TranReport` for the single time point
            ``t = 0``, with signal arrays of shape ``(lines, 1)``. Its ``power``
            is computed as in ``tran``; its ``energy`` is ``NaN``, since an
            operating point has no duration. Default ``False``.

        Returns
        -------
        Vout : ndarray, shape (len(output_mask),)
            Masked op-amp outputs [V].
        time : ndarray
            ``[0.0]``.
        report : TranReport
            Only with ``return_report``.
        """
        if solver_type.lower() == "spice":
            solver_type = "SPICE/fast"
        is_spice = solver_type in _spice._SPICE_FOOTERS
        if solver_type.lower() != "model" and not is_spice:
            raise NotImplementedError(f"Solver {solver_type!r} not yet implemented.")

        Vin, Iin = self._input_matrices(np.array([0.0]))
        Vin_1d = Vin[:, 0] if Vin is not None else None
        Iin_1d = Iin[:, 0]

        if is_spice:
            Vout_raw, Iout_spice = self._spice_op(solver_type, _spice.SpiceAnalysis("op"))
        else:
            Vout_raw, _, info = nlop_model(self, Vin_1d, Iin_1d)
        Vout = Vout_raw[self.output_mask]

        if return_report:
            T = self._transfer_core()
            Vin_r = self._report_vin(Vin, 1)
            Iin_r = Iin
            Ijunc = T[2] @ Vout_raw.reshape(-1, 1) + Iin_r
            if Vin_r.size:
                Ijunc = Ijunc + T[0] @ Vin_r
            Vjunc = scipy.linalg.solve(T[1] + T[3], Ijunc)
            V = Vout_raw.reshape(-1, 1)
            # SPICE measures the op-amp output currents directly.
            Iout = Iout_spice.reshape(-1, 1) if is_spice else T[4] @ V - T[5] @ Vjunc
            I_Vin = T[6] @ Vin_r - T[7] @ Vjunc if Vin_r.size else np.empty((0, 1))
            time = np.array([0.0])
            power, energy = self._dissipation(V, Iout, Vin_r, I_Vin, time)
            R = TranReport(
                solver_type=solver_type,
                time=time,
                Vout=V,
                Voainn=T[8] @ Vjunc + T[9] @ V,
                Voainp=np.zeros((self.n, 1)),
                Iout=Iout,
                Vin=Vin_r,
                I_Vin=I_Vin,
                Iin=Iin_r,
                power=power,
                energy=energy,
            )
            return Vout, np.array([0.0]), R

        return Vout, np.array([0.0])

    # --- transfer + noise ---
    def transfer(self, kind: str = "inverting", l_id: np.ndarray | None = None):
        """Composite transfer matrix with finite op-amp gain.

        Parameters
        ----------
        kind : {"inverting", "noninverting"}, optional
            ``"inverting"`` (default) gives ``T`` with ``Iin = T @ Vout`` for
            currents injected at the junctions:
            ``(T2 + T4) inv(T9) (diag(1/(s A0)) - T10) - T3``. It tends to
            ``-G_f`` for ideal op-amps, and its inverse is the
            current-to-output transfer. ``"noninverting"`` gives the
            corresponding matrix for voltages on the non-inverting inputs:
            ``T10 + T9 inv(T2 + T4) T3 - diag(1/(s A0))``.
        l_id : array_like of bool, shape (n,), optional
            Restrict to the op-amps in their linear region. Default: all.

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            For an unknown ``kind``.

        Notes
        -----
        ``T1``–``T10`` are the array transfers cached by the circuit.
        ``T1``, ``T2``, ``T7`` and ``T8`` are the input array's column-to-row,
        row self, column self and row-to-column current transfers. ``T3``,
        ``T4``, ``T5`` and ``T6`` are the same four for the feedback array.
        ``T9`` and ``T10`` are the feedback array's row-to-row and
        column-to-row voltage transfers. ``s`` is ``sign_vector``.
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
        """Noise analysis.

        Parameters
        ----------
        solver_type : str, optional
            ``"SPICE/fast"`` (default), another SPICE solver, or ``"model"``.
        *args
            Frequency sweep: nothing for the default
            :class:`~loopuss.spice.SpiceAnalysis` (1 Hz to 10 GHz, 100 points
            per decade), ``Fmax[, Fmin[, Fstep[, Fsweep]]]``, a
            ``SpiceAnalysis``, or, with ``"model"`` only, a frequency vector.

        Returns
        -------
        Sout : ndarray, shape (len(output_mask), F)
            Output noise density [V/√Hz].
        freq : ndarray, shape (F,)
            Frequencies [Hz].
        report : NoiseReport

        Notes
        -----
        The model includes the thermal noise of both arrays and the op-amp
        noise; see :func:`~loopuss.noise.noise_model`. SPICE runs one
        ``.noise`` netlist per masked output, ``num_workers`` at a time. A
        warning is issued when the feedback array uses ``ideal_buffers``,
        whose in-loop inverters are noiseless.
        """
        if solver_type.lower() == "spice":
            solver_type = "SPICE/fast"
        is_spice = solver_type in _spice._SPICE_FOOTERS

        if self.feedback_array.ideal_buffers:
            warnings.warn(
                "Noise analysis with ideal_buffers on the feedback array may be "
                "inaccurate: the in-loop inverting buffers are noiseless.",
                stacklevel=2,
            )

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
                    "bench_enable": self.bench_enable,
                    "file_path": self.file_path,
                    "file_name": self.file_name,
                },
            ),
        ]
