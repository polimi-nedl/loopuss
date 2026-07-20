"""DSPModel — digital signal-processing block model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .periph_circ import PeriphCirc
from .tech_model import TechModel


class DSPModel(PeriphCirc):
    def __init__(
        self,
        preset_type: str = "bypass",
        *,
        DSPFunction=None,
        complexity=None,
        latency=None,
        **kwargs,
    ):
        self.dsp_function: Callable = lambda x: x
        self.complexity: float = 0.0
        self.ops: float = 0.0
        self.latency: float = 0.0
        super().__init__(preset_type=preset_type, **kwargs)

        if DSPFunction is not None:
            self.dsp_function = DSPFunction
        if complexity is not None:
            self.complexity = float(complexity)
        if latency is not None:
            self.latency = float(latency)

    def _load_preset(self, preset_type: str) -> None:
        if preset_type == "ADSP2148-450MHz":
            self.type = "ADSP"
            self.dsp_function = lambda x: x
            self.complexity = 0.0
            self.latency = 1.11e-9
            self.leakage_power = 182e-3
            self.Vsupply = np.array([0, 0.975])
            vspan = float(np.diff(self.Vsupply)[0])
            self.Isupply = self.leakage_power / vspan
            self.dynamic_energy = 389.3e-12
            self.area = 0.6 * 1.434e-4
            self.technology = TechModel(node=65e-9)

    def eval(self, *args: Any) -> tuple:
        """Evaluate DSP function; returns (y, latency, dynamic_energy)."""
        y = self.dsp_function(*args)
        T = self.latency * self.complexity
        Ed = self.dynamic_energy * self.complexity
        self.ops += self.complexity
        self._energy["Dynamic"] += Ed
        return y, T, Ed

    def __repr__(self) -> str:
        return f"DSPModel(type={self.type!r})"
