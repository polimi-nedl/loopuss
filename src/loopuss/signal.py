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
    "acamplitude",
    "acphase",
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
    """Abstract base for voltage and current waveforms."""

    e_type: str  # 'V' or 'I'

    def __init__(self, s_type: str = "EMPTY", **kwargs):
        s_type = s_type.upper()
        if s_type not in _VALID_TYPES:
            raise ValueError(f"Unknown signal type {s_type!r}")
        self.s_type: str = s_type

        self.dcvalue: float | None = None
        self.acamplitude: float | None = None
        self.acphase: float | None = None
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

    # --- queries ---
    def is_voltage(self) -> bool:
        return self.e_type == "V"

    def is_current(self) -> bool:
        return self.e_type == "I"

    def is_dc(self) -> bool:
        return self.s_type == "DC"

    def is_step(self) -> bool:
        return self.s_type == "STEP"

    def is_pulse(self) -> bool:
        return self.s_type == "PULSE"

    def is_sine(self) -> bool:
        return self.s_type == "SINE"

    def is_pwl(self) -> bool:
        return self.s_type == "PWL"

    def is_empty(self) -> bool:
        return self.s_type == "EMPTY"

    # --- waveform ---
    def to_array(self, t: np.ndarray) -> np.ndarray:
        """Evaluate waveform on time vector t. Returns 1-D float64 array."""
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
            t_pts = np.asarray(self.time, dtype=float).ravel()
            v_pts = np.asarray(self.valpwl, dtype=float).ravel()
            # 'previous' interpolation with left-extrapolation using first value.
            idx = np.searchsorted(t_pts, t, side="right") - 1
            idx = np.clip(idx, 0, len(v_pts) - 1)
            return v_pts[idx].astype(float)

        if self.is_sine():
            amp = float(self.ampsine) if self.ampsine is not None else 1.0
            off = float(self.offsine) if self.offsine is not None else 0.0
            freq = float(self.freq) if self.freq is not None else 1.0
            phi = float(self.phi) if self.phi is not None else 0.0
            damp = float(self.theta) if self.theta is not None else 0.0
            out = np.full(len(t), off)
            mask = t >= td
            out[mask] = off + amp * np.exp(-damp * (t[mask] - td)) * np.sin(
                2 * np.pi * freq * (t[mask] - td) + phi
            )
            return out

        raise NotImplementedError(f"to_array not implemented for {self.s_type}")

    @property
    def stype(self) -> str:
        return self.s_type

    # --- SPICE ---
    def to_spice(self, name: str, pos: str, neg: str) -> str:
        """SPICE source line for this signal.

        Returns '' for EMPTY signals. ``name`` is the source id (the V/I prefix
        is prepended); ``pos``/``neg`` are the terminal node names.
        """
        if self.is_empty():
            return ""
        head = f"{self.e_type}{name} {pos} {neg}"

        def f(x) -> str:
            return f"{(float(x) if x is not None else 0.0):.16g}"

        if self.is_dc():
            return f"{head} DC({f(self.dcvalue)})"
        if self.is_pulse():
            # DC operating-point override so .op matches model.
            dc = f"DC {f(self.to_array(np.array([0.0]))[0])} "
            return (
                f"{head} {dc}PULSE({f(self.val1)}, {f(self.val2)}, {f(self.td)} "
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
            # DC operating-point override so .op matches model.
            dc = f"DC {f(self.to_array(np.array([0.0]))[0])} "
            return (
                f"{head} {dc}PULSE({f(self.val1)},{f(self.val2)}, {f(self.td)} "
                f"1e-15 1e-15 1e3 1e3 1)"
            )
        if self.is_pwl():
            time = np.asarray(self.time, dtype=float).ravel()
            val = np.asarray(self.valpwl, dtype=float).ravel()
            pwl = np.empty(time.size + val.size)
            pwl[0::2] = time
            pwl[1::2] = val
            return f"{head} PWL({''.join(f'{v:.16g} ' for v in pwl)})"
        return ""

    def breakpoints(self) -> np.ndarray:
        """Return critical time points (signal discontinuities)."""
        if self.s_type in ("DC", "EMPTY", "SINE"):
            return np.array([], dtype=float)
        pts: list[float] = []
        if self.td is not None:
            pts.append(float(self.td))
        if self.s_type == "PULSE" and self.ton is not None and self.td is not None:
            pts.append(float(self.td) + float(self.ton))
        if self.s_type == "PWL" and self.time is not None:
            pts.extend(np.asarray(self.time).ravel().tolist())
        return np.array(pts, dtype=float)

    def get_char_times(self) -> np.ndarray:
        """Alias for breakpoints()."""
        return self.breakpoints()

    def add_delay(self, dt: float) -> Signal:
        s = deepcopy(self)
        s.td = (s.td or 0.0) + float(dt)
        return s

    def max_value(self) -> float:
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
    """Voltage waveform."""

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
        return self.to_array(np.array([0.0]))[0] / V0

    @staticmethod
    def ytov(
        y: np.ndarray,
        V0: float = 1.0,
        stype: str = "STEP",
        ton: float | None = None,
        tv: np.ndarray | None = None,
    ) -> list[VoltageSignal]:
        """Convert numeric vector y to list of VoltageSignal objects.

        For PWL sources ``y`` is a 2-D array whose rows are the per-channel
        value sequences sampled at the shared time vector ``tv``; every other
        source type takes a 1-D ``y`` with one scalar per channel.
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
    """Current waveform."""

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
        return self.to_array(np.array([0.0]))[0] / I0

    @staticmethod
    def ytoi(
        y: np.ndarray,
        I0: float = 1.0,
        stype: str = "STEP",
        ton: float | None = None,
        tv: np.ndarray | None = None,
    ) -> list[CurrentSignal]:
        """Convert numeric vector y to list of CurrentSignal objects.

        For PWL sources ``y`` is a 2-D array whose rows are the per-channel
        value sequences sampled at the shared time vector ``tv``; every other
        source type takes a 1-D ``y`` with one scalar per channel.
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
