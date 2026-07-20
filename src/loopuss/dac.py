"""VoltageDAC and CurrentDAC - digital-to-analog converter models."""

from __future__ import annotations

import math

import numpy as np

from .converter_model import ConverterModel
from .signal import CurrentSignal, VoltageSignal
from .tech_model import TechModel

# --- VoltageDAC presets ---
_VDAC_PRESETS: dict[str, dict] = {
    "Reaz19": dict(
        Nbit=10,
        Nchannel=1,
        latency=500e-9,
        dynamic_energy=0,
        area=0.18e-6,
        technology=TechModel(node=130e-9),
        Vsupply=[0, 0.9],
        leakage_power=0.10e-3,
        Vref=[0, 0.9],
    ),
    "Sancandi21": dict(
        Nbit=7,
        Nchannel=1,
        latency=50e-9,
        dynamic_energy=0,
        area=3.14e-9,
        technology=TechModel(node=90e-9),
        Vsupply=[-0.6, 0.6],
        leakage_power=547.2e-6,
        Vref=[-0.6, 0.6],
    ),
}

# --- CurrentDAC presets ---
_CDAC_PRESETS: dict[str, dict] = {
    "Tseng14": dict(
        Nbit=10,
        Nchannel=1,
        latency=1.0417e-9,
        dynamic_energy=0,
        area=0.01e-6,
        technology=TechModel(node=20e-9),
        Vsupply=[0, 1.8],
        leakage_power=9e-3,
        Iref=[-256e-6, 256e-6],
    ),
    "Ishraqul17": dict(
        Nbit=8,
        Nchannel=1,
        latency=0.4e-9,
        dynamic_energy=0,
        area=3.07e-12,
        technology=TechModel(node=16e-9),
        Vsupply=[0, 0.9],
        leakage_power=0.160e-3,
        Iref=[-19.5e-6, 19.5e-6],
    ),
    "Mao18": dict(
        Nbit=12,
        Nchannel=1,
        latency=10e-9,
        dynamic_energy=0,
        area=0.21e-12,
        technology=TechModel(node=130e-9),
        Vsupply=[0, 1.2],
        leakage_power=18e-3,
        Iref=[-16e-3, 16e-3],
    ),
    "Bandali18": dict(
        Nbit=8,
        Nchannel=1,
        latency=7.1429e-9,
        dynamic_energy=0,
        area=0.202e-6,
        technology=TechModel(node=180e-9),
        Vsupply=[0, 0.9],
        leakage_power=0.62381e-3,
        Iref=[-637.2e-6, 637.2e-6],
    ),
    "Chang18": dict(
        Nbit=14,
        Nchannel=1,
        latency=0.5e-9,
        dynamic_energy=0,
        area=0.19e-6,
        technology=TechModel(node=65e-9),
        Vsupply=[0, 1.3],
        leakage_power=490e-3,
        Iref=[-20e-3, 20e-3],
    ),
    "Kim19": dict(
        Nbit=12,
        Nchannel=1,
        latency=1e-9,
        dynamic_energy=0,
        area=0.093e-6,
        technology=TechModel(node=14e-9),
        Vsupply=[0, 1.8],
        leakage_power=12.5e-3,
        Iref=[-2e-3, 2e-3],
    ),
    "Kommangunta21": dict(
        Nbit=10,
        Nchannel=1,
        latency=10e-9,
        dynamic_energy=0,
        area=0.135e-6,
        technology=TechModel(node=180e-9),
        Vsupply=[0, 1.2],
        leakage_power=2.1e-3,
        Iref=[-5.115e-3, 5.115e-3],
    ),
    "Kumar22": dict(
        Nbit=12,
        Nchannel=1,
        latency=1.25e-9,
        dynamic_energy=0,
        area=0.0303e-6,
        technology=TechModel(node=180e-9),
        Vsupply=[0, 1.8],
        leakage_power=9.1e-3,
        Iref=[-4.15e-3, 4.15e-3],
    ),
    "Sancandi21_1X": dict(
        Nbit=6,
        Nchannel=1,
        latency=20e-9,
        dynamic_energy=0,
        area=5e-9,
        technology=TechModel(node=90e-9),
        Vsupply=[0, 1.2],
        leakage_power=130.8e-6,
        Iref=[-20e-6, 20e-6],
    ),
}


class VoltageDAC(ConverterModel):
    def __init__(self, preset_type: str = "ideal", *, Vref=None, signalWaveform=None, **kwargs):
        self.Vref: np.ndarray = np.array([-np.inf, np.inf])
        self._waveforms: list[VoltageSignal] = []
        super().__init__(preset_type=preset_type, **kwargs)

        if Vref is not None:
            v = np.asarray(Vref, dtype=float).ravel()
            self.Vref = np.array([-v[0], v[0]]) if v.size == 1 else v

        if signalWaveform is not None:
            self.load_waveform(signalWaveform)

    def _load_preset(self, preset_type: str) -> None:
        if preset_type not in _VDAC_PRESETS:
            return
        p = _VDAC_PRESETS[preset_type]
        self.type = preset_type
        self.Nbit = float(p.get("Nbit", math.inf))
        self.Nchannel = int(p.get("Nchannel", 1))
        self.latency = float(p.get("latency", 0.0))
        self.dynamic_energy = float(p.get("dynamic_energy", 0.0))
        self.area = float(p.get("area", 0.0))
        self.technology = p.get("technology", TechModel())
        self.Vsupply = np.asarray(p.get("Vsupply", [-np.inf, np.inf]), dtype=float)
        self.leakage_power = float(p.get("leakage_power", 0.0))
        vspan = float(np.diff(self.Vsupply)[0])
        self.Isupply = self.leakage_power / vspan if (np.isfinite(vspan) and vspan > 0) else 0.0
        vr = p.get("Vref", [-np.inf, np.inf])
        self.Vref = np.asarray(vr, dtype=float)

    def load_waveform(self, signals) -> None:
        """Load signal waveforms, applying quantization."""
        if isinstance(signals, VoltageSignal):
            signals = [signals]
        self._waveforms = list(signals)
        # Pad to Nchannel if needed
        while len(self._waveforms) < self.Nchannel:
            self._waveforms.append(VoltageSignal("EMPTY"))

    def is_loaded(self) -> bool:
        return bool(self._waveforms) and not all(
            w.stype.upper() == "EMPTY" for w in self._waveforms
        )

    def todouble(
        self, time: np.ndarray, extra_only: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Evaluate waveforms at time vector. Returns (n_ch, len(time)) array.

        If extra_only=True, returns extra breakpoint times instead.
        """
        t = np.asarray(time, dtype=float).ravel()
        if not self._waveforms:
            return np.empty((0, len(t)))
        rows = [w.to_array(t) for w in self._waveforms]
        return np.vstack(rows)

    def extra_times(self, time: np.ndarray) -> np.ndarray:
        """Return the source breakpoints (delays and PWL knots).

        Every breakpoint is returned, including those outside the requested
        time span, so that a source discontinuity at ``t = 0`` is honoured even
        when the requested grid starts above it.
        """
        if not self._waveforms:
            return np.array([])
        return np.concatenate([w.breakpoints() for w in self._waveforms])

    def quantclip(self, x: np.ndarray) -> np.ndarray:
        return self.quantclip_array(np.atleast_2d(x), self.Vref.reshape(1, 2))

    def to_spice(self, names, nodes) -> list[str]:
        """SPICE source lines for the DAC channels."""
        return _dac_to_spice(self._waveforms, names, nodes)


class CurrentDAC(ConverterModel):
    def __init__(self, preset_type: str = "ideal", *, Iref=None, signalWaveform=None, **kwargs):
        self.Iref: np.ndarray = np.array([-np.inf, np.inf])
        self._waveforms: list[CurrentSignal] = []
        super().__init__(preset_type=preset_type, **kwargs)

        if Iref is not None:
            v = np.asarray(Iref, dtype=float).ravel()
            self.Iref = np.array([-v[0], v[0]]) if v.size == 1 else v

        if signalWaveform is not None:
            self.load_waveform(signalWaveform)

    def _load_preset(self, preset_type: str) -> None:
        if preset_type not in _CDAC_PRESETS:
            return
        p = _CDAC_PRESETS[preset_type]
        self.type = preset_type
        self.Nbit = float(p.get("Nbit", math.inf))
        self.Nchannel = int(p.get("Nchannel", 1))
        self.latency = float(p.get("latency", 0.0))
        self.dynamic_energy = float(p.get("dynamic_energy", 0.0))
        self.area = float(p.get("area", 0.0))
        self.technology = p.get("technology", TechModel())
        self.Vsupply = np.asarray(p.get("Vsupply", [-np.inf, np.inf]), dtype=float)
        self.leakage_power = float(p.get("leakage_power", 0.0))
        vspan = float(np.diff(self.Vsupply)[0])
        self.Isupply = self.leakage_power / vspan if (np.isfinite(vspan) and vspan > 0) else 0.0
        ir = p.get("Iref", [-np.inf, np.inf])
        self.Iref = np.asarray(ir, dtype=float)

    def load_waveform(self, signals) -> None:
        if isinstance(signals, CurrentSignal):
            signals = [signals]
        self._waveforms = list(signals)
        while len(self._waveforms) < self.Nchannel:
            self._waveforms.append(CurrentSignal("EMPTY"))

    def is_loaded(self) -> bool:
        return bool(self._waveforms) and not all(
            w.stype.upper() == "EMPTY" for w in self._waveforms
        )

    def todouble(self, time: np.ndarray, extra_only: bool = False) -> np.ndarray:
        t = np.asarray(time, dtype=float).ravel()
        if not self._waveforms:
            return np.empty((0, len(t)))
        rows = [w.to_array(t) for w in self._waveforms]
        return np.vstack(rows)

    def extra_times(self, time: np.ndarray) -> np.ndarray:
        """Return the source breakpoints (delays and PWL knots).

        Every breakpoint is returned, including those outside the requested
        time span, so that a source discontinuity at ``t = 0`` is honoured even
        when the requested grid starts above it.
        """
        if not self._waveforms:
            return np.array([])
        return np.concatenate([w.breakpoints() for w in self._waveforms])

    def quantclip(self, x: np.ndarray) -> np.ndarray:
        return self.quantclip_array(np.atleast_2d(x), self.Iref.reshape(1, 2))

    def to_spice(self, names, nodes) -> list[str]:
        """SPICE source lines for the DAC channels."""
        return _dac_to_spice(self._waveforms, names, nodes)


def _dac_to_spice(waveforms, names, nodes) -> list[str]:
    """Emit SPICE source lines for a list of waveforms.

    ``names`` is a single id or one id per channel; ``nodes`` is a single
    (pos, neg) pair (shared) or one pair per channel.
    """
    lines: list[str] = []
    per_channel_nodes = bool(nodes) and isinstance(nodes[0], (list, tuple))
    for i, w in enumerate(waveforms):
        name = names[i] if isinstance(names, (list, tuple)) else names
        pos, neg = nodes[i] if per_channel_nodes else nodes
        s = w.to_spice(name, pos, neg)
        if s:
            lines.append(s)
    return lines
