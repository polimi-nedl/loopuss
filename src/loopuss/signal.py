"""Electrical signal waveforms: Signal, VoltageSignal, CurrentSignal."""

from __future__ import annotations

from abc import ABC
from copy import deepcopy

import numpy as np

_VALID_TYPES = {"DC", "STEP", "PULSE", "SINE", "PWL", "EMPTY"}

# Mapping from subclass-specific aliases to internal storage names.
_V_ALIASES = {"v1": "val1", "v2": "val2", "vamp": "ampsine", "voff": "offsine", "vval": "valpwl"}
_I_ALIASES = {"i1": "val1", "i2": "val2", "iamp": "ampsine", "ioff": "offsine", "ival": "valpwl"}

_KNOWN_ATTRS = {
    "dcvalue",
    "tr",
    "tf",
    "ton",
    "tperiod",
    "freq",
    "theta",
    "phi",
    "td",
    "ncycles",
    "time",
    "val1",
    "val2",
    "ampsine",
    "offsine",
    "valpwl",
}


class Signal(ABC):
    """Abstract base for voltage and current waveforms.

    Waveforms use the LTspice source vocabulary. :meth:`to_array` samples one
    for the analytical solvers, and :meth:`to_spice` emits it as a SPICE source
    line. Instantiate :class:`~loopuss.VoltageSignal` or
    :class:`~loopuss.CurrentSignal`.

    Parameters
    ----------
    s_type : {"EMPTY", "DC", "STEP", "PULSE", "SINE", "PWL"}, optional
        Source type, case-insensitive. Default ``"EMPTY"``: no source. It
        samples as 0 for currents and as ``NaN`` (floating) for voltages, and
        is left out of netlists.
    **kwargs
        Waveform parameters, case-insensitive; unknown names are ignored.

        * ``dcvalue``: value of DC sources.
        * ``td``: delay [s] of STEP, PULSE, SINE and PWL sources; a PWL
          source has its knot times shifted by it.
        * ``val1``, ``val2``: initial and final value of STEP and PULSE
          sources, also accepted as ``v1``/``v2`` or ``i1``/``i2``.
        * ``ton``: on-time [s] of PULSE sources.
        * ``tr``, ``tf``, ``tperiod``, ``ncycles``: rise time, fall time,
          period and cycle count of PULSE sources. They are used only in
          netlists; the model samples a single ideal pulse.
        * ``ampsine``, ``offsine`` (also ``vamp``/``voff`` or
          ``iamp``/``ioff``), ``freq`` [Hz], ``theta`` [1/s], ``phi`` [°]:
          amplitude, offset, frequency, damping and phase of SINE sources; the
          amplitude and frequency are required. ``ncycles`` stops a SINE after
          that many periods, holding its last value (unset or 0: endless).
        * ``time``, ``valpwl`` (also ``vval`` or ``ival``): knot times [s] and
          values of PWL sources.
    """

    e_type: str  # 'V' or 'I'

    def __init__(self, s_type: str = "EMPTY", **kwargs):
        s_type = s_type.upper()
        if s_type not in _VALID_TYPES:
            raise ValueError(f"Unknown signal type {s_type!r}")
        self.s_type: str = s_type

        self.dcvalue: float | None = None
        self.tr: float | None = None
        self.tf: float | None = None
        self.ton: float | None = None
        self.tperiod: float | None = None
        self.freq: float | None = None
        self.theta: float | None = None
        self.phi: float | None = None
        self.td: float | None = None
        self.ncycles: float | None = None
        self.time: np.ndarray | None = None
        self.val1: float | None = None
        self.val2: float | None = None
        self.ampsine: float | None = None
        self.offsine: float | None = None
        self.valpwl: np.ndarray | None = None

        aliases = _V_ALIASES if self.e_type == "V" else _I_ALIASES
        for k, v in kwargs.items():
            attr = aliases.get(k.lower(), k.lower())
            if attr not in _KNOWN_ATTRS:
                continue
            if attr in {"time", "valpwl"}:
                setattr(self, attr, np.asarray(v, dtype=float).ravel())
            else:
                setattr(self, attr, float(v))

        # No default would serve both backends: an unset field is written to
        # the netlist as 0, which for these two gives a flat source.
        if self.is_sine() and (self.ampsine is None or self.freq is None):
            raise ValueError("SINE signals need an amplitude and a frequency.")

    # --- queries ---
    def is_voltage(self) -> bool:
        """True for voltage signals."""
        return self.e_type == "V"

    def is_current(self) -> bool:
        """True for current signals."""
        return self.e_type == "I"

    def is_dc(self) -> bool:
        """True for DC sources."""
        return self.s_type == "DC"

    def is_step(self) -> bool:
        """True for STEP sources."""
        return self.s_type == "STEP"

    def is_pulse(self) -> bool:
        """True for PULSE sources."""
        return self.s_type == "PULSE"

    def is_sine(self) -> bool:
        """True for SINE sources."""
        return self.s_type == "SINE"

    def is_pwl(self) -> bool:
        """True for PWL sources."""
        return self.s_type == "PWL"

    def is_empty(self) -> bool:
        """True for EMPTY signals."""
        return self.s_type == "EMPTY"

    # --- waveform ---
    def to_array(self, t: np.ndarray) -> np.ndarray:
        """Sample the waveform.

        Parameters
        ----------
        t : array_like
            Time points [s].

        Returns
        -------
        ndarray, shape (len(t),)
            STEP: ``val1`` before ``td`` and ``val2`` from ``td`` on.
            PULSE: ``val2`` on ``[td, td + ton]`` and ``val1`` elsewhere.
            SINE: ``offsine + ampsine * sin(phi π/180)`` before ``td``, then
            ``offsine + ampsine * exp(-theta (t - td)) * sin(2π freq (t - td) + phi π/180)``.
            PWL: each knot value held until the next knot, the knots shifted
            by ``td``, and the first value before the first knot.
        """
        t = np.asarray(t, dtype=float).ravel()
        td = float(self.td) if self.td is not None else 0.0

        if self.is_empty():
            return np.zeros(len(t)) if self.is_current() else np.full(len(t), np.nan)

        if self.is_dc():
            return np.full(len(t), float(self.dcvalue))

        if self.is_step():
            v1 = float(self.val1) if self.val1 is not None else 0.0
            v2 = float(self.val2) if self.val2 is not None else 0.0
            return np.where(t < td, v1, v2)

        if self.is_pulse():
            v1 = float(self.val1) if self.val1 is not None else 0.0
            v2 = float(self.val2) if self.val2 is not None else 0.0
            ton = float(self.ton) if self.ton is not None else 0.0
            return np.where((t >= td) & (t <= td + ton), v2, v1)

        if self.is_pwl():
            t_pts = np.asarray(self.time, dtype=float).ravel() + td
            v_pts = np.asarray(self.valpwl, dtype=float).ravel()
            # 'previous' interpolation with left-extrapolation using first value.
            idx = np.searchsorted(t_pts, t, side="right") - 1
            idx = np.clip(idx, 0, len(v_pts) - 1)
            return v_pts[idx].astype(float)

        if self.is_sine():
            amp = float(self.ampsine)
            off = float(self.offsine) if self.offsine is not None else 0.0
            freq = float(self.freq)
            # phi is in degrees, as in the SPICE SINE source the netlist emits.
            phi = np.deg2rad(float(self.phi)) if self.phi is not None else 0.0
            damp = float(self.theta) if self.theta is not None else 0.0
            # Before td LTspice holds the value the sine starts from, not the offset.
            out = np.full(len(t), off + amp * np.sin(phi))
            mask = t >= td
            # After ncycles periods LTspice holds the value the last one ends on.
            u = t[mask] - td
            if self.ncycles:
                u = np.minimum(u, float(self.ncycles) / freq)
            out[mask] = off + amp * np.exp(-damp * u) * np.sin(2 * np.pi * freq * u + phi)
            return out

        raise NotImplementedError(f"to_array not implemented for {self.s_type}")

    @property
    def stype(self) -> str:
        """Source type, same as ``s_type``."""
        return self.s_type

    # --- SPICE ---
    def to_spice(self, name: str, pos: str, neg: str, *, dc_only: bool = False) -> str:
        """SPICE source line for this signal.

        Returns '' for EMPTY signals. ``name`` is the source id (the V/I prefix
        is prepended); ``pos``/``neg`` are the terminal node names.

        STEP sources are written as a PULSE with 1 fs edges. With ``dc_only``,
        every type is written as a DC source holding its value at t = 0, which
        is how the model solver samples sources for an operating point.
        """
        if self.is_empty():
            return ""
        head = f"{self.e_type}{name} {pos} {neg}"

        def f(x) -> str:
            return f"{(float(x) if x is not None else 0.0):.16g}"

        # A "DC x PULSE(...)" override would not do for .op: when LTspice's
        # direct Newton iteration fails and it falls back to pseudo-transient
        # (gminsteps=0 and itl6=0 skip the other fallbacks), it evaluates the
        # PULSE at t = 0 and ignores the override.
        if dc_only:
            return f"{head} DC({f(self.to_array(np.array([0.0]))[0])})"
        if self.is_dc():
            return f"{head} DC({f(self.dcvalue)})"
        if self.is_pulse():
            return (
                f"{head} PULSE({f(self.val1)}, {f(self.val2)}, {f(self.td)} "
                f"{f(self.tr)} {f(self.tf)} {f(self.ton)} "
                f"{f(self.tperiod)} {f(self.ncycles)})"
            )
        if self.is_sine():
            return (
                f"{head} SINE({f(self.offsine)}, {f(self.ampsine)}, "
                f"{f(self.freq)} {f(self.td)} {f(self.theta)} "
                f"{f(self.phi)} {f(self.ncycles)})"
            )
        if self.is_step():
            return (
                f"{head} PULSE({f(self.val1)},{f(self.val2)}, {f(self.td)} 1e-15 1e-15 1e3 1e3 1)"
            )
        if self.is_pwl():
            # LTspice's PWL has no delay field, so the knots carry it.
            time = np.asarray(self.time, dtype=float).ravel() + (self.td or 0.0)
            val = np.asarray(self.valpwl, dtype=float).ravel()
            pwl = np.empty(time.size + val.size)
            pwl[0::2] = time
            pwl[1::2] = val
            return f"{head} PWL({''.join(f'{v:.16g} ' for v in pwl)})"
        return ""

    def breakpoints(self) -> np.ndarray:
        """Return critical time points (signal discontinuities)."""
        if self.s_type in ("DC", "EMPTY"):
            return np.array([], dtype=float)
        if self.s_type == "SINE":
            if not self.ncycles:
                return np.array([], dtype=float)
            return np.array([float(self.td or 0.0) + float(self.ncycles) / float(self.freq)])
        pts: list[float] = []
        if self.td is not None:
            pts.append(float(self.td))
        if self.s_type == "PULSE" and self.ton is not None and self.td is not None:
            pts.append(float(self.td) + float(self.ton))
        if self.s_type == "PWL" and self.time is not None:
            pts.extend((np.asarray(self.time, dtype=float).ravel() + (self.td or 0.0)).tolist())
        return np.array(pts, dtype=float)

    def get_char_times(self) -> np.ndarray:
        """Alias for breakpoints()."""
        return self.breakpoints()

    def add_delay(self, dt: float) -> Signal:
        """Return a copy with ``td`` increased by ``dt`` [s]."""
        s = deepcopy(self)
        s.td = (s.td or 0.0) + float(dt)
        return s

    def max_value(self) -> float:
        """Largest value the waveform reaches; 0 for EMPTY signals."""
        if self.is_dc():
            return float(self.dcvalue)
        if self.is_step() or self.is_pulse():
            return float(max(self.val1 or 0.0, self.val2 or 0.0))
        if self.is_pwl():
            return float(np.max(self.valpwl))
        if self.is_sine():
            return float((self.offsine or 0.0) + abs(self.ampsine or 0.0))
        return 0.0

    def min_value(self) -> float:
        """Smallest value the waveform reaches; 0 for EMPTY signals."""
        if self.is_dc():
            return float(self.dcvalue)
        if self.is_step() or self.is_pulse():
            return float(min(self.val1 or 0.0, self.val2 or 0.0))
        if self.is_pwl():
            return float(np.min(self.valpwl))
        if self.is_sine():
            return float((self.offsine or 0.0) - abs(self.ampsine or 0.0))
        return 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.s_type!r})"


class VoltageSignal(Signal):
    """Voltage waveform; see :class:`~loopuss.Signal` for the parameters.

    ``v1``, ``v2``, ``vamp``, ``voff`` and ``vval`` are read/write aliases of
    ``val1``, ``val2``, ``ampsine``, ``offsine`` and ``valpwl``.
    """

    e_type = "V"

    # Convenience aliases that map to the underlying storage fields.
    @property
    def v1(self):
        return self.val1

    @v1.setter
    def v1(self, v):
        self.val1 = float(v)

    @property
    def v2(self):
        return self.val2

    @v2.setter
    def v2(self, v):
        self.val2 = float(v)

    @property
    def vamp(self):
        return self.ampsine

    @vamp.setter
    def vamp(self, v):
        self.ampsine = float(v)

    @property
    def voff(self):
        return self.offsine

    @voff.setter
    def voff(self, v):
        self.offsine = float(v)

    @property
    def vval(self):
        return self.valpwl

    @vval.setter
    def vval(self, v):
        self.valpwl = np.asarray(v, dtype=float).ravel()

    def vtoy(self, V0: float = 1.0) -> float:
        """Value at t = 0 divided by ``V0``."""
        return self.to_array(np.array([0.0]))[0] / V0

    @staticmethod
    def ytov(
        y: np.ndarray,
        V0: float = 1.0,
        stype: str = "STEP",
        ton: float | None = None,
        tv: np.ndarray | None = None,
    ) -> list[VoltageSignal]:
        """Convert a numeric vector into one voltage source per channel.

        Parameters
        ----------
        y : array_like
            One value per channel. For PWL, a 2-D array with one row of
            values per channel, sampled at ``tv``. ``NaN`` channels (all-``NaN``
            rows for PWL) become EMPTY signals.
        V0 : float, optional
            Scale factor [V per unit of ``y``]. Default 1.
        stype : {"STEP", "DC", "PULSE", "PWL"}, optional
            Source type, case-insensitive. Default ``"STEP"``, from 0 to
            ``y * V0`` at t = 0. PULSE is a single pulse from t = 0 lasting
            ``ton``. Any other value gives EMPTY signals.
        ton : float, optional
            Pulse on-time [s]; required for PULSE.
        tv : array_like, optional
            Knot times [s] shared by all channels; required for PWL.

        Returns
        -------
        list of VoltageSignal
        """
        stype = stype.upper()
        if stype == "PWL":
            if tv is None:
                raise ValueError("tv is required for PWL sources")
            tv = np.asarray(tv, dtype=float).ravel()
            rows = np.atleast_2d(np.asarray(y, dtype=float))
            out = []
            for row in rows:
                if np.all(np.isnan(row)):
                    out.append(VoltageSignal("EMPTY"))
                else:
                    out.append(VoltageSignal("PWL", vval=row * V0, time=tv))
            return out
        y = np.asarray(y, dtype=float).ravel()
        out: list[VoltageSignal] = []
        for yi in y:
            if np.isnan(yi):
                out.append(VoltageSignal("EMPTY"))
            elif stype == "DC":
                out.append(VoltageSignal("DC", dcvalue=float(yi * V0)))
            elif stype == "STEP":
                out.append(VoltageSignal("STEP", v1=0.0, v2=float(yi * V0), td=0.0))
            elif stype == "PULSE":
                if ton is None:
                    raise ValueError("ton is required for PULSE sources")
                out.append(
                    VoltageSignal(
                        "PULSE",
                        v1=0.0,
                        v2=float(yi * V0),
                        td=0.0,
                        tr=0.0,
                        tf=0.0,
                        ton=float(ton),
                        tperiod=1.0,
                        ncycles=1.0,
                    )
                )
            else:
                out.append(VoltageSignal("EMPTY"))
        return out


class CurrentSignal(Signal):
    """Current waveform; see :class:`~loopuss.Signal` for the parameters.

    ``i1``, ``i2``, ``iamp``, ``ioff`` and ``ival`` are read/write aliases of
    ``val1``, ``val2``, ``ampsine``, ``offsine`` and ``valpwl``.
    """

    e_type = "I"

    @property
    def i1(self):
        return self.val1

    @i1.setter
    def i1(self, v):
        self.val1 = float(v)

    @property
    def i2(self):
        return self.val2

    @i2.setter
    def i2(self, v):
        self.val2 = float(v)

    @property
    def iamp(self):
        return self.ampsine

    @iamp.setter
    def iamp(self, v):
        self.ampsine = float(v)

    @property
    def ioff(self):
        return self.offsine

    @ioff.setter
    def ioff(self, v):
        self.offsine = float(v)

    @property
    def ival(self):
        return self.valpwl

    @ival.setter
    def ival(self, v):
        self.valpwl = np.asarray(v, dtype=float).ravel()

    def itoy(self, I0: float = 1.0) -> float:
        """Value at t = 0 divided by ``I0``."""
        return self.to_array(np.array([0.0]))[0] / I0

    @staticmethod
    def ytoi(
        y: np.ndarray,
        I0: float = 1.0,
        stype: str = "STEP",
        ton: float | None = None,
        tv: np.ndarray | None = None,
    ) -> list[CurrentSignal]:
        """Convert a numeric vector into one current source per channel.

        Parameters
        ----------
        y : array_like
            One value per channel. For PWL, a 2-D array with one row of
            values per channel, sampled at ``tv``. ``NaN`` channels (all-``NaN``
            rows for PWL) become EMPTY signals.
        I0 : float, optional
            Scale factor [A per unit of ``y``]. Default 1.
        stype : {"STEP", "DC", "PULSE", "PWL"}, optional
            Source type, case-insensitive. Default ``"STEP"``, from 0 to
            ``y * I0`` at t = 0. PULSE is a single pulse from t = 0 lasting
            ``ton``. Any other value gives EMPTY signals.
        ton : float, optional
            Pulse on-time [s]; required for PULSE.
        tv : array_like, optional
            Knot times [s] shared by all channels; required for PWL.

        Returns
        -------
        list of CurrentSignal
        """
        stype = stype.upper()
        if stype == "PWL":
            if tv is None:
                raise ValueError("tv is required for PWL sources")
            tv = np.asarray(tv, dtype=float).ravel()
            rows = np.atleast_2d(np.asarray(y, dtype=float))
            out = []
            for row in rows:
                if np.all(np.isnan(row)):
                    out.append(CurrentSignal("EMPTY"))
                else:
                    out.append(CurrentSignal("PWL", ival=row * I0, time=tv))
            return out
        y = np.asarray(y, dtype=float).ravel()
        out: list[CurrentSignal] = []
        for yi in y:
            if np.isnan(yi):
                out.append(CurrentSignal("EMPTY"))
            elif stype == "DC":
                out.append(CurrentSignal("DC", dcvalue=float(yi * I0)))
            elif stype == "STEP":
                out.append(CurrentSignal("STEP", i1=0.0, i2=float(yi * I0), td=0.0))
            elif stype == "PULSE":
                if ton is None:
                    raise ValueError("ton is required for PULSE sources")
                out.append(
                    CurrentSignal(
                        "PULSE",
                        i1=0.0,
                        i2=float(yi * I0),
                        td=0.0,
                        tr=0.0,
                        tf=0.0,
                        ton=float(ton),
                        tperiod=1.0,
                        ncycles=1.0,
                    )
                )
            else:
                out.append(CurrentSignal("EMPTY"))
        return out
