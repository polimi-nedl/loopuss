"""Analysis report containers: BenchReport, TranReport, NoiseReport and the
common grouped-display base.

BenchReport.barplot / .bdplot render with matplotlib.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import GroupedDisplay


# --- BenchReport ---
@dataclass(repr=False)
class BenchReport(GroupedDisplay):
    """Benchmark report.

    Returned by :meth:`LoopUS.benchmark <loopuss.LoopUS.benchmark>` and
    :meth:`GPUModel.benchmark <loopuss.GPUModel.benchmark>`. Reports combine
    arithmetically: ``A + B`` is ``combine(A, B, "serial", noarea=True,
    noprog=True)`` and ``b * A`` is ``scale(A, b, "parallel", noarea=False,
    noprog=True)``.

    Attributes
    ----------
    latency : dict
        ``Programming``, ``Computing``, ``PostProcessing`` and ``Total`` [s].
    energy : dict
        ``Breakdown`` is an array with one row per component, named in
        ``Labels``. Its six columns are the static and dynamic energy of the
        Programming, Computing and PostProcessing phases [J]. ``Total`` is the
        sum [J]. Reports from ``LoopUS.benchmark`` also hold one nested dict
        per component; combined reports do not.
    power : ndarray, shape (2,)
        Mean and peak power [W].
    area : dict
        Area of each component, plus ``Total`` [m²].
    complexity : float
        Operations per run [OPs].
    ops : float
        Total operations [OPs].
    throughput : float
        ``ops / latency["Total"]`` [OPS].
    energy_efficiency : float
        ``ops / energy["Total"]`` [OPS/W].
    area_efficiency : float
        ``throughput / area["Total"]`` [OPS/m²].
    circuit : LoopUS, GPUModel or str
        Copy of the benchmarked object, or a label for derived reports.
    """

    latency: dict | None = None
    energy: dict | None = None
    power: Any = None
    area: dict | None = None
    complexity: float = 0.0
    ops: float = 0.0
    throughput: float = 0.0
    energy_efficiency: float = 0.0
    area_efficiency: float = 0.0
    circuit: Any = None

    # --- display: three property groups ---
    def _property_groups(self):
        from .utils import si_format

        p = (
            np.asarray(self.power, float).ravel()
            if self.power is not None
            else np.array([np.nan, np.nan])
        )
        return [
            (
                "Observables",
                {
                    "Latency": self.latency,
                    "Energy": self.energy,
                    "Power": f"{si_format(p[0])}W|{si_format(p[1])}W",
                    "Area": self.area,
                },
            ),
            (
                "Performance",
                {
                    "Complexity": f"{si_format(self.complexity)}OPs/run",
                    "OPs": f"{si_format(self.ops)}OPs",
                    "Throughput": f"{si_format(self.throughput)}OPS",
                    "EnergyEfficiency": f"{si_format(self.energy_efficiency)}OPS/W",
                    "AreaEfficiency": f"{si_format(self.area_efficiency / 1e6)}OPS/mm2",
                },
            ),
            (
                "Circuit configuration",
                {
                    "Circuit": self.circuit,
                },
            ),
        ]

    # --- operators ---
    def __add__(self, other: BenchReport) -> BenchReport:
        # addition default: mode='serial', noarea=True, noprog=True
        return BenchReport.combine(self, other, "serial", True, True)

    def __mul__(self, b: float) -> BenchReport:
        # scalar-multiply default: mode='parallel', noarea=False, noprog=True
        return BenchReport.scale(self, float(b), "parallel", False, True)

    __rmul__ = __mul__

    @staticmethod
    def combine(
        A: BenchReport,
        B: BenchReport,
        mode: str = "serial",
        noarea: bool = True,
        noprog: bool = True,
    ) -> BenchReport:
        """Combine two reports.

        Parameters
        ----------
        A, B : BenchReport
            Reports to combine.
        mode : {"serial", "parallel"}, optional
            ``"serial"`` (default): ``B`` runs after ``A``. Computing and
            post-processing latencies add, mean powers are averaged, the peak
            power is the larger one and complexities are averaged.
            ``"parallel"``: ``B`` runs alongside ``A``. Latencies take the
            larger value, while powers and complexities add.
        noarea : bool, optional
            Keep ``A``'s area (default). Otherwise areas combine per component,
            taking the larger one in serial mode and the sum in parallel mode.
        noprog : bool, optional
            Count programming once, from ``A`` (default). This drops ``B``'s
            programming latency and the first row of its energy breakdown.

        Returns
        -------
        BenchReport
            Energies and ``ops`` add, and the efficiencies are recomputed.
        """
        if mode not in ("serial", "parallel"):
            raise ValueError("mode must be either serial or parallel.")

        C = BenchReport()

        # Latency
        lat = {}
        if mode == "serial":
            lat["Programming"] = A.latency["Programming"]
            if not noprog:
                lat["Programming"] += B.latency["Programming"]
            lat["Computing"] = A.latency["Computing"] + B.latency["Computing"]
            lat["PostProcessing"] = A.latency["PostProcessing"] + B.latency["PostProcessing"]
            lat["Total"] = lat["Computing"] + lat["Programming"] + lat["PostProcessing"]
        else:  # parallel
            lat["Programming"] = A.latency["Programming"]
            if not noprog:
                lat["Programming"] = max(A.latency["Programming"], B.latency["Programming"])
            lat["Computing"] = max(A.latency["Computing"], B.latency["Computing"])
            lat["PostProcessing"] = max(A.latency["PostProcessing"], B.latency["PostProcessing"])
            lat["Total"] = max(A.latency["Total"], B.latency["Total"])
        C.latency = lat

        # Energy
        bd_A = np.asarray(A.energy["Breakdown"], dtype=float)
        bd_B = np.asarray(B.energy["Breakdown"], dtype=float)
        emask = np.ones_like(bd_A)
        if noprog:
            emask[0, :] = 0.0
        breakdown = bd_A + emask * bd_B
        C.energy = {
            "Breakdown": breakdown,
            "Labels": list(A.energy["Labels"]),
            "Total": float(breakdown.sum()),
        }

        pa = np.asarray(A.power, dtype=float).ravel()
        pb = np.asarray(B.power, dtype=float).ravel()
        if mode == "serial":
            C.power = np.array([np.mean([pa[0], pb[0]]), max(pa[1], pb[1])])
        else:
            C.power = np.array([pa[0] + pb[0], pa[1] + pb[1]])

        # Area
        if mode == "serial":
            if noarea:
                C.area = deepcopy(A.area)
            else:
                C.area = {k: max(A.area[k], B.area[k]) for k in A.area}
        else:
            if noarea:
                C.area = deepcopy(A.area)
            else:
                C.area = {k: A.area[k] + B.area[k] for k in A.area}

        # Performance
        if mode == "serial":
            C.complexity = float(np.mean([A.complexity, B.complexity]))
        else:
            C.complexity = A.complexity + B.complexity
        C.ops = A.ops + B.ops
        C.throughput = _safe_div(C.ops, C.latency["Total"])
        C.energy_efficiency = _safe_div(C.ops, C.energy["Total"])
        C.area_efficiency = _safe_div(C.throughput, C.area["Total"])
        C.circuit = f"{mode}-sum"
        return C

    @staticmethod
    def scale(
        A: BenchReport,
        b: float,
        mode: str = "parallel",
        noarea: bool = False,
        noprog: bool = True,
    ) -> BenchReport:
        """Scale a report to ``b`` repetitions of its workload.

        Parameters
        ----------
        A : BenchReport
            Report to scale.
        b : float
            Number of repetitions.
        mode : {"parallel", "serial"}, optional
            ``"parallel"`` (default): ``b`` copies run side by side, so latency
            is unchanged while power and area (see ``noarea``) scale by ``b``.
            ``"serial"``: ``b`` runs in sequence, so computing and
            post-processing latencies scale by ``b`` while power and area are
            unchanged.
        noarea : bool, optional
            Leave the area unscaled in parallel mode. Default ``False``.
        noprog : bool, optional
            Leave programming unscaled: its latency (serial mode) and the
            first energy-breakdown row. Default ``True``.

        Returns
        -------
        BenchReport
            The energy (except as per ``noprog``), ``complexity`` and ``ops``
            scale by ``b``, and the efficiencies are recomputed.
        """
        if mode not in ("serial", "parallel"):
            raise ValueError("mode must be either serial or parallel.")

        C = BenchReport()

        lat = {}
        lat["Programming"] = A.latency["Programming"]
        if mode == "serial":
            if not noprog:
                lat["Programming"] = A.latency["Programming"] * b
            lat["Computing"] = A.latency["Computing"] * b
            lat["PostProcessing"] = A.latency["PostProcessing"] * b
        else:  # parallel
            lat["Computing"] = A.latency["Computing"]
            lat["PostProcessing"] = A.latency["PostProcessing"]
        lat["Total"] = lat["Computing"] + lat["Programming"] + lat["PostProcessing"]
        C.latency = lat

        bd_A = np.asarray(A.energy["Breakdown"], dtype=float)
        emask = np.ones_like(bd_A)
        if noprog:
            emask[0, :] = 0.0
        breakdown = bd_A + (b - 1.0) * emask * bd_A
        C.energy = {
            "Breakdown": breakdown,
            "Labels": list(A.energy["Labels"]),
            "Total": float(breakdown.sum()),
        }

        pa = np.asarray(A.power, dtype=float).ravel()
        C.power = pa.copy() if mode == "serial" else pa * b

        if mode == "serial":
            C.area = deepcopy(A.area)
        else:
            C.area = deepcopy(A.area) if noarea else {k: A.area[k] * b for k in A.area}

        C.complexity = A.complexity * b
        C.ops = A.ops * b
        C.throughput = _safe_div(C.ops, C.latency["Total"])
        C.energy_efficiency = _safe_div(C.ops, C.energy["Total"])
        C.area_efficiency = _safe_div(C.throughput, C.area["Total"])
        C.circuit = f"{mode}-sum"
        return C

    # --- plotting ---
    #
    # Both accept either a single report (``report.barplot()``) or a list of
    # reports (``BenchReport.barplot([A, B], ...)``).

    def barplot(self, *flags: str, ax=None, labels=None):
        """Grouped bar chart of the efficiency metrics.

        Plots Throughput [OPS], Energy Efficiency [OPS/W] and Area Efficiency
        [OPS/mm^2] on a shared log y-axis, one bar per report within each
        metric group. The metric flags are: ``'-latency'`` ->
        Throughput, ``'-energy'`` -> Energy Efficiency, ``'-area'`` -> Area
        Efficiency (all three if none given). ``'-notext'`` hides the
        per-bar value annotations.

        Call it on one report (``report.barplot()``) or on a list
        (``BenchReport.barplot([A, B], labels=["A", "B"])``).

        Parameters
        ----------
        *flags : str
            Flags described above.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure by default.
        labels : list of str, optional
            One label per report. Default ``BR1``, ``BR2``, ...

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        from .utils import si_format, si_prefix, si_scale

        reports = list(self) if isinstance(self, (list, tuple)) else [self]
        n = len(reports)

        latF, enrF, areF = ("-latency" in flags, "-energy" in flags, "-area" in flags)
        notext = "-notext" in flags
        if not (latF or enrF or areF):
            latF = enrF = areF = True

        # (name, unit, values) in metric order.
        metrics = []
        if latF:
            metrics.append(("Throughput", "OPS", np.array([r.throughput for r in reports], float)))
        if enrF:
            metrics.append(
                (
                    "Energy Efficiency",
                    "OPS/W",
                    np.array([r.energy_efficiency for r in reports], float),
                )
            )
        if areF:
            metrics.append(
                (
                    "Area Efficiency",
                    "OPS/mm$^2$",
                    np.array([r.area_efficiency for r in reports], float) / 1e6,
                )
            )
        nmetr = len(metrics)

        # Common SI exponent for all bars: round(median(SIscale)/3)*3.
        allvals = np.concatenate([m[2] for m in metrics])
        finite = allvals[np.isfinite(allvals) & (allvals > 0)]
        d = int(round(np.median([si_scale(v) for v in finite]) / 3) * 3) if finite.size else 0
        scale = 10.0**d

        if labels is None:
            labels = [f"BR{i + 1}" for i in range(n)]
        labels = list(labels)[:n]

        if ax is None:
            _, ax = plt.subplots()

        x = np.arange(nmetr)
        bar_w = 0.8 / n
        for i, lab in enumerate(labels):
            heights = np.array([metrics[j][2][i] for j in range(nmetr)]) / scale
            offs = (i - (n - 1) / 2.0) * bar_w
            bars = ax.bar(x + offs, heights, bar_w, label=lab, edgecolor="k")
            if not notext:
                for j, rect in enumerate(bars):
                    raw = metrics[j][2][i]
                    ax.annotate(
                        si_format(raw),
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        ax.set_yscale("log")
        ax.set_xticks(x)
        # Every metric is scaled by the same 10^d, so the prefix lives in each
        # group's unit and the bar height is just the value in those units.
        ax.set_xticklabels([f"{m[0]}\n[{si_prefix(d)}{m[1]}]" for m in metrics])
        ax.set_ylabel("value (in group units)")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        if n > 1:
            ax.legend()
        return ax

    def bdplot(self, *flags: str, fig=None, labels=None):
        """Breakdown pie charts of Area / Energy / Latency.

        One pie per (report, metric). Metric flags ``'-area'`` / ``'-energy'`` /
        ``'-latency'`` select the columns (all if none given); ``'-noprog'``
        drops the programming (MemWrite / Programming) contribution;
        ``'-vert'`` / ``'-horiz'`` set the grid orientation (horizontal
        default); ``'-notitle'`` hides the subplot titles.

        Call it on one report or on a list, like :meth:`barplot`.

        Parameters
        ----------
        *flags : str
            Flags described above.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on; a new one by default.
        labels : list of str, optional
            One label per report. Default ``BR1``, ``BR2``, ...

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        reports = list(self) if isinstance(self, (list, tuple)) else [self]
        nrep = len(reports)

        latF, areF, enrF = ("-latency" in flags, "-area" in flags, "-energy" in flags)
        notitle = "-notitle" in flags
        noprog = "-noprog" in flags
        vert, horiz = ("-vert" in flags, "-horiz" in flags)
        if not (latF or areF or enrF):
            latF = areF = enrF = True
        if vert and horiz:
            raise ValueError("Orientation must be either vert or horiz.")
        if not vert and not horiz:
            horiz = True

        # metric order: area, energy, latency.
        metrics = [m for m, on in (("Area", areF), ("Energy", enrF), ("Latency", latF)) if on]
        nmetr = len(metrics)

        if labels is None:
            labels = [f"BR{i + 1}" for i in range(nrep)]
        labels = list(labels)[:nrep]

        def _area(r):
            items = [(k, v) for k, v in r.area.items() if k != "Total"]
            if noprog:
                items = items[1:]  # drop MemWrite
            return [k for k, _ in items], np.array([v for _, v in items], float)

        def _energy(r):
            bd = np.asarray(r.energy["Breakdown"], float)
            vals = np.nansum(bd, axis=1)  # per-component total
            labs = list(r.energy["Labels"])
            if noprog:
                vals, labs = vals[1:], labs[1:]  # drop MemWrite
            return labs, vals

        def _latency(r):
            L = r.latency
            labs = ["Programming", "Computing", "PostProcessing"]
            vals = np.array([L[k] for k in labs], float)
            if noprog:
                labs, vals = labs[1:], vals[1:]  # drop Programming
            return labs, vals

        getter = {"Area": _area, "Energy": _energy, "Latency": _latency}
        nrows, ncols = (nrep, nmetr) if horiz else (nmetr, nrep)

        if fig is None:
            fig = plt.figure(figsize=(3.2 * ncols, 3.2 * nrows))

        for i, r in enumerate(reports):
            for j, met in enumerate(metrics):
                labs, vals = getter[met](r)
                pos = (i * ncols + j + 1) if horiz else (j * ncols + i + 1)
                ax = fig.add_subplot(nrows, ncols, pos)
                mask = vals > 0  # pie cannot show 0/neg wedges
                if mask.any():
                    ax.pie(
                        vals[mask],
                        labels=[l for l, m in zip(labs, mask, strict=True) if m],
                        autopct="%1.0f%%",
                        textprops={"fontsize": 7},
                    )
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                    ax.set_axis_off()
                if not notitle:
                    ax.set_title(f"{labels[i]} - {met}")
        fig.tight_layout()
        return fig


def sum_reports(
    reports, mode: str = "serial", noarea: bool = True, noprog: bool = True
) -> BenchReport:
    """Fold a sequence of reports with :meth:`BenchReport.combine`, using the
    given ``mode``, ``noarea`` and ``noprog``."""
    reports = list(reports)
    C = reports[0]
    for r in reports[1:]:
        C = BenchReport.combine(C, r, mode, noarea, noprog)
    return C


def mean_reports(reports) -> BenchReport:
    """Average a sequence of reports.

    Energy, latency, area, mean power, ``ops`` and ``complexity`` are
    averaged; the peak power is the maximum over the reports.
    """
    reports = list(reports)
    n = len(reports)
    # Re-derive as a plain average from the breakdowns/areas.
    bd = np.zeros_like(np.asarray(reports[0].energy["Breakdown"], dtype=float))
    lat = {"Programming": 0.0, "Computing": 0.0, "PostProcessing": 0.0, "Total": 0.0}
    area = {k: 0.0 for k in reports[0].area}
    pmean, pmax = 0.0, -np.inf
    ops = cmplx = 0.0
    for r in reports:
        bd += np.asarray(r.energy["Breakdown"], dtype=float)
        for k in lat:
            lat[k] += r.latency[k]
        for k in area:
            area[k] += r.area[k]
        p = np.asarray(r.power, dtype=float).ravel()
        pmean += p[0]
        pmax = max(pmax, p[1])
        ops += r.ops
        cmplx += r.complexity
    bd /= n
    for k in lat:
        lat[k] /= n
    for k in area:
        area[k] /= n
    S = BenchReport()
    S.latency = lat
    S.energy = {
        "Breakdown": bd,
        "Labels": list(reports[0].energy["Labels"]),
        "Total": float(bd.sum()),
    }
    S.power = np.array([pmean / n, pmax])
    S.area = area
    S.complexity = cmplx / n
    S.ops = ops / n
    S.throughput = _safe_div(S.ops, S.latency["Total"])
    S.energy_efficiency = _safe_div(S.ops, S.energy["Total"])
    S.area_efficiency = _safe_div(S.throughput, S.area["Total"])
    S.circuit = "averaged"
    return S


# --- TranReport ---
@dataclass(repr=False)
class TranReport(GroupedDisplay):
    """Transient analysis report.

    Returned by :meth:`LoopUS.tran <loopuss.LoopUS.tran>` with
    ``return_report=True``. It covers every op-amp and active input, not only
    the masked outputs, on the solver's time grid before the ADC latency is
    added.

    Attributes
    ----------
    solver_type : str
        Solver that produced the report.
    time : ndarray, shape (T,)
        Time points [s].
    Vout : ndarray, shape (n, T)
        Op-amp output voltages [V].
    Voainn : ndarray, shape (n, T)
        Voltage at the op-amp input wired to the arrays [V].
    Voainp : ndarray, shape (n, T)
        Voltage at the other op-amp input. Always zero, because reference
        signals are not included.
    Iout : ndarray, shape (n, T)
        Op-amp output currents into the feedback array [A].
    Vin : ndarray, shape (p, T)
        Voltages applied to the input-array columns, DAC plus input
        reference [V]. Columns outside ``input_mask_v`` sit at the reference.
    I_Vin : ndarray, shape (p, T)
        Currents drawn from those voltage sources [A].
    Iin : ndarray, shape (n, T)
        Input currents injected at the junctions [A]; 0 on rows outside
        ``input_mask_i``.
    power : dict of ndarray
        Instantaneous dissipation of ``OA``, ``InputArray``,
        ``FeedbackArray`` and ``Total`` [W].
    energy : dict of float
        Time integral of each ``power`` entry [J]; ``NaN`` for single-point
        runs.
    VoutADC, TimeADC : ndarray
        The outputs and time returned by ``LoopUS.tran``, after the ADC and
        DSP.
    """

    solver_type: Any = None
    time: Any = None
    Voainn: Any = None
    Voainp: Any = None
    Vout: Any = None
    Iout: Any = None
    Vin: Any = None
    I_Vin: Any = None
    Iin: Any = None
    power: Any = None
    energy: Any = None

    @property
    def peak_power(self) -> float:
        """Peak total power [W]."""
        return float(np.max(self.power["Total"]))

    @property
    def mean_power(self) -> float:
        """Mean total power over the time samples, not time-weighted [W]."""
        return float(np.mean(self.power["Total"]))

    def _property_groups(self):
        from .utils import si_format

        power = {}
        if self.power is not None:
            power["MeanPower"] = f"{si_format(self.mean_power)}W"
            power["PeakPower"] = f"{si_format(self.peak_power)}W"
        power["Power"] = self.power
        power["Energy"] = self.energy
        return [
            ("Solver", {"SolverType": self.solver_type}),
            (
                "Signals",
                {
                    "Time": self.time,
                    "Vin": self.Vin,
                    "Vout": self.Vout,
                    "Voainn": self.Voainn,
                    "Voainp": self.Voainp,
                    "Iout": self.Iout,
                    "I_Vin": self.I_Vin,
                    "Iin": self.Iin,
                },
            ),
            ("Power & energy", power),
        ]


# --- NoiseReport ---
@dataclass(repr=False)
class NoiseReport(GroupedDisplay):
    """Noise analysis report, returned by :meth:`LoopUS.noise
    <loopuss.LoopUS.noise>`.

    Attributes
    ----------
    solver_type : str
        Solver that produced the report.
    freq : ndarray, shape (F,)
        Frequencies [Hz].
    Sout : ndarray, shape (len(output_mask), F)
        Output noise density [V/√Hz].
    Pout : ndarray, shape (len(output_mask),)
        RMS output noise over the sweep, ``sqrt(∫ Sout² df)`` [V].
    """

    solver_type: Any = None
    freq: Any = None
    Sout: Any = None
    Pout: Any = None

    @property
    def mean_power(self) -> float:
        """Mean of ``Pout`` over the outputs [V]."""
        return float(np.mean(self.Pout))

    @property
    def peak_power(self) -> float:
        """Largest ``Pout`` [V]."""
        return float(np.max(self.Pout))

    def _property_groups(self):
        from .utils import si_format

        noise = {}
        if self.Pout is not None:
            noise["MeanPower"] = si_format(self.mean_power)
            noise["PeakPower"] = si_format(self.peak_power)
        noise["Pout"] = self.Pout
        return [
            ("Solver", {"SolverType": self.solver_type}),
            ("Spectra", {"Freq": self.freq, "Sout": self.Sout}),
            ("Noise power", noise),
        ]


# --- helpers ---
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) else np.inf
