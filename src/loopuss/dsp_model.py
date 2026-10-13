"""DSPModel — digital signal-processing block model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .periph_circ import PeriphCirc
from .tech_model import TechModel


class DSPModel(PeriphCirc):
    """Digital signal-processing block model.

    Applied to the ADC output at the end of every
    :meth:`LoopUS.tran <loopuss.LoopUS.tran>`. Its cost scales with
    ``complexity``: each :meth:`eval` takes ``latency * complexity`` seconds
    and ``dynamic_energy * complexity`` joules, so ``latency`` and
    ``dynamic_energy`` are per-operation figures.

    Parameters
    ----------
    preset_type : str, optional
        ``"bypass"`` (default): identity function at no cost.
        ``"ADSP2148-450MHz"``: 1.11 ns and 389.3 pJ per operation, 182 mW
        leakage, 0.975 V supply and 86 mm² at 65 nm. This preset sets
        ``type`` to ``"ADSP"``.
    dsp_function : callable, optional
        Function applied to the data. Default: identity.
    complexity : float, optional
        Operations per evaluation [OPs]. Default 0.
    latency : float, optional
        Latency per operation [s]. Default 0.
    **kwargs
        Forwarded to :class:`~loopuss.PeriphCirc`; ``dynamic_energy`` is the
        energy per operation [J].

    Attributes
    ----------
    dsp_function : callable
        Function applied by :meth:`eval`.
    ops : float
        Operations accumulated by :meth:`eval` [OPs].
    """

    def __init__(
        self,
        preset_type: str = "bypass",
        *,
        dsp_function=None,
        complexity=None,
        latency=None,
        **kwargs,
    ):
        self.dsp_function: Callable = lambda x: x
        self.complexity: float = 0.0
        self.ops: float = 0.0
        self.latency: float = 0.0
        super().__init__(preset_type=preset_type, **kwargs)

        if dsp_function is not None:
            self.dsp_function = dsp_function
        if complexity is not None:
            self.complexity = float(complexity)
        if latency is not None:
            self.latency = float(latency)

    def _load_preset(self, preset_type: str) -> None:
        if preset_type == "bypass":
            return
        if preset_type != "ADSP2148-450MHz":
            raise self._unknown_preset(preset_type, ["bypass", "ADSP2148-450MHz"])
        if preset_type == "ADSP2148-450MHz":
            self.type = "ADSP"
            self.dsp_function = lambda x: x
            self.complexity = 0.0
            self.latency = 1.11e-9
            self.Vsupply = np.array([0, 0.975])
            self._set_supply_from_power(182e-3)
            self.dynamic_energy = 389.3e-12
            self.area = 0.6 * 1.434e-4
            self.technology = TechModel(node=65e-9)

    def eval(self, *args: Any) -> tuple:
        """Apply the DSP function and account for its cost.

        Parameters
        ----------
        *args
            Passed to ``dsp_function``.

        Returns
        -------
        y
            Result of ``dsp_function(*args)``.
        latency : float
            ``latency * complexity`` [s].
        dynamic_energy : float
            ``dynamic_energy * complexity`` [J], also added to the accumulator.
        """
        y = self.dsp_function(*args)
        T = self.latency * self.complexity
        Ed = self.dynamic_energy * self.complexity
        self.ops += self.complexity
        self._energy["Dynamic"] += Ed
        return y, T, Ed

    def __repr__(self) -> str:
        return f"DSPModel(type={self.type!r})"
