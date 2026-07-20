"""ConverterModel - abstract base for data converters."""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np

from .periph_circ import PeriphCirc


class ConverterModel(PeriphCirc):
    """Abstract base for data converters (ADC/DAC)."""

    def __init__(
        self, preset_type: str, *, Nbit=None, Nchannel=None, latency=None, DNL=None, **kwargs
    ):
        self.latency: float = 0.0
        self.Nbit: float = math.inf
        self.Nchannel: int = 1
        self.DNL: float = 0.0
        super().__init__(preset_type=preset_type, **kwargs)

        if Nbit is not None:
            self.Nbit = float(Nbit)
        if Nchannel is not None:
            self.Nchannel = int(Nchannel)
        if latency is not None:
            self.latency = float(latency)
        if DNL is not None:
            self.DNL = float(DNL)

    def quantclip_array(self, x: np.ndarray, fsr: np.ndarray) -> np.ndarray:
        """Quantize and clip a numeric array.

        x:   (n, m) float64 - n channels x m time samples
        fsr: (n, 2) [lo, hi] per channel, or (1, 2) broadcast
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        fsr = np.atleast_2d(np.asarray(fsr, dtype=float))
        if fsr.shape[0] == 1:
            fsr = np.repeat(fsr, x.shape[0], axis=0)

        if math.isinf(self.Nbit) and np.all(np.isinf(fsr)):
            return x.copy()

        fsr = fsr.copy()
        fsr[np.isinf(fsr[:, 0]), 0] = np.min(x)
        fsr[np.isinf(fsr[:, 1]), 1] = np.max(x)

        out = x.copy()
        for ii in range(x.shape[0]):
            lo, hi = fsr[ii]
            xi = x[ii]
            if not math.isinf(self.Nbit):
                lsb = (hi - lo) / (2.0**self.Nbit)
                fsrd = 2.0**self.Nbit - 1.0
                out[ii] = np.clip(np.round((xi - lo) / lsb), 0.0, fsrd) * lsb + lo
            else:
                out[ii] = np.clip(xi, lo, hi)
        return out

    @abstractmethod
    def _load_preset(self, preset_type: str) -> None: ...
