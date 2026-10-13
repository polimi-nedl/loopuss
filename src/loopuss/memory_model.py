"""Resistive memory cell model."""

from __future__ import annotations

import math

import numpy as np

from .tech_model import TechModel

_DEFAULT_TECH = TechModel()  # 14 nm

# Preset table of published memory-cell characterisations.
_PRESETS: dict[str, dict] = {
    "RRAM": dict(
        type="RRAM",
        LRS=1 / 10e3,
        HRS=1 / 1e6,
        ZRS=1 / 1e6,
        Nbit=3,
        area_factor=4,
    ),
    "Jo10": dict(
        type="RRAM",
        LRS=1 / 26e6,
        HRS=1 / 325e6,
        ZRS=1 / 325e6,
        Nbit=7,
        area_factor=4,
    ),
    "Park13": dict(
        type="RRAM",
        LRS=1 / 23e6,
        HRS=1 / 157.32e6,
        ZRS=1 / 157.32e6,
        Nbit=6,
        area_factor=4,
    ),
    "Gao15": dict(
        type="RRAM",
        LRS=1 / 5e6,
        HRS=1 / 10e6,
        ZRS=1 / 10e6,
        Nbit=7,
        area_factor=4,
    ),
    "Woo16": dict(
        type="RRAM",
        LRS=1 / 16.9e3,
        HRS=1 / 74.867e3,
        ZRS=1 / 74.867e3,
        Nbit=5,
        area_factor=14,
    ),
    "STT-MRAM": dict(
        type="STT-MRAM",
        LRS=1 / 10e3,
        HRS=1 / 20e3,
        ZRS=1 / 20e3,
        Nbit=2,
        area_factor=13,
    ),
    "PCM": dict(
        type="PCM",
        LRS=1 / 1e3,
        HRS=1 / 1e10,
        ZRS=1 / 1e10,
        Nbit=3,
        area_factor=12,
    ),
    "Kuzum11": dict(
        type="PCM",
        LRS=1 / 4.71e3,
        HRS=1 / 93.258e3,
        ZRS=1 / 93.258e3,
        Nbit=7,
        area_factor=27,
    ),
    "FEFET": dict(
        type="FeFET",
        LRS=1 / 500e3,
        HRS=1 / 1e9,
        ZRS=1 / 1e9,
        Nbit=5,
        area_factor=17,
    ),
    "Jerry17": dict(
        type="FeFET",
        LRS=1 / 559.28e3,
        HRS=1 / 25.1676e6,
        ZRS=1 / 25.1676e6,
        Nbit=5,
        area_factor=5,
    ),
    "Oh17": dict(
        type="FeFET",
        LRS=1 / 500e3,
        HRS=1 / 650e6,
        ZRS=1 / 650e6,
        Nbit=5,
        area_factor=5,
    ),
    "SRAM": dict(
        type="SRAM",
        LRS=1 / 1e3,
        HRS=1 / 1e10,
        ZRS=1 / 1e10,
        Nbit=6,
        area_factor=100,
    ),
    "Sancandi21": dict(
        type="CMOS",
        LRS=31 / 160e3,
        HRS=1 / 160e3,
        ZRS=0.0,
        Nbit=5,
        area_factor=None,
        area_abs=442e-12,
        technology=TechModel(node=90e-9),
    ),
}

# Area-node mapping for sub-28 nm nodes (minimum metal pitch table).
_MMP = {14e-9: 42e-9, 16e-9: 64e-9}


def _eff_node(node: float) -> float:
    return _MMP.get(node, node)


class MemoryModel:
    """Resistive memory cell model.

    Describes the conductance range and precision of the cells of a
    :class:`~loopuss.CrossBar`, which uses it at construction to clamp and
    quantize its conductances. See :doc:`/presets` for the device presets.

    Parameters
    ----------
    preset : str, optional
        Preset name. An unknown name raises ``ValueError``. Default: an ideal
        cell with ``type="Ideal"``. For a custom device, pass the overrides
        below without a preset.
    **kwargs
        Overrides applied after the preset. Names are case-insensitive:

        ``LRS``
            Highest programmable conductance (low-resistance state) [S].
            Default 1 mS.
        ``HRS``
            Lowest non-zero programmable conductance (high-resistance state)
            [S]. Default 0.
        ``ZRS``
            Conductance of cells meant to be zero [S]. Default 0.
        ``window``
            Memory window ``LRS / HRS``. Default ``inf``. If ``HRS`` is
            non-zero and the window differs from ``LRS / HRS`` by more than
            1 %, it is replaced by ``LRS / HRS``. A window alone does not set
            ``HRS``.
        ``Nbit``
            Equivalent bits of precision. Default ``inf``: no quantization.
        ``area``
            Cell area [m²]. Default ``4 F²`` at 14 nm.
        ``technology``
            A :class:`~loopuss.TechModel`. If its node differs from the
            current one, ``area`` (explicit or not) is scaled by the squared
            ratio of the effective nodes.
    """

    def __init__(self, preset: str | None = None, **kwargs):
        self.type: str = "Ideal"
        self.LRS: float = 1 / 1e3
        self.HRS: float = 0.0
        self.ZRS: float = 0.0
        self.window: float = math.inf
        self.technology: TechModel = TechModel()
        self.area: float = 4 * self.technology.node**2
        self.Nbit: float = math.inf

        if preset is not None:
            self._load_preset(preset)

        # Apply keyword overrides.
        def _get(key):
            for k, v in kwargs.items():
                if k.lower() == key.lower():
                    return v
            return None

        v = _get("area")
        if v is not None:
            self.area = float(v)

        v = _get("Nbit")
        if v is not None:
            self.Nbit = float(v)

        v = _get("LRS")
        if v is not None:
            self.LRS = float(v)

        v = _get("HRS")
        if v is not None:
            self.HRS = float(v)

        v = _get("ZRS")
        if v is not None:
            self.ZRS = float(v)

        v = _get("window")
        if v is not None:
            self.window = float(v)

        new_tech = _get("technology")
        if new_tech is not None and new_tech.node != self.technology.node:
            old_eff = _eff_node(self.technology.node)
            new_eff = _eff_node(new_tech.node)
            self.area *= (new_eff / old_eff) ** 2
            self.technology = new_tech

        # Validate Window vs LRS/HRS.
        if self.HRS != 0:
            computed_win = self.LRS / self.HRS
            if not np.isclose(computed_win, self.window, rtol=0.01):
                self.window = computed_win
        elif math.isinf(self.window):
            self.window = math.inf

    def _load_preset(self, preset: str) -> None:
        if preset not in _PRESETS:
            raise ValueError(
                f"Unknown MemoryModel preset {preset!r}. Available: {', '.join(_PRESETS)}."
            )
        p = _PRESETS[preset]
        self.type = p.get("type", preset)
        self.LRS = p["LRS"]
        self.HRS = p["HRS"]
        self.ZRS = p.get("ZRS", p["HRS"])
        self.Nbit = p.get("Nbit", math.inf)

        tech = p.get("technology", TechModel())
        self.technology = tech

        area_abs = p.get("area_abs")
        if area_abs is not None:
            self.area = float(area_abs)
        else:
            factor = p.get("area_factor", 4)
            self.area = factor * tech.node**2

        # Compute window from LRS/HRS.
        if self.HRS != 0:
            self.window = self.LRS / self.HRS
        else:
            self.window = math.inf

    def __repr__(self) -> str:
        return (
            f"MemoryModel(type={self.type!r}, "
            f"LRS={self.LRS:.3g} S, HRS={self.HRS:.3g} S, "
            f"Window={self.window:.3g})"
        )
