"""ConverterModel - abstract base for data converters."""

from __future__ import annotations

import math
from abc import abstractmethod
from copy import deepcopy

import numpy as np

from .periph_circ import PeriphCirc
from .signal import Signal


class ConverterModel(PeriphCirc):
    """Abstract base class for data converters.

    Adds resolution, channel count and latency to
    :class:`~loopuss.PeriphCirc`. See :class:`~loopuss.ADCModel`,
    :class:`~loopuss.VoltageDAC` and :class:`~loopuss.CurrentDAC`.

    Parameters
    ----------
    preset_type : str
        Preset name, see :class:`~loopuss.PeriphCirc`.
    Nbit : float, optional
        Resolution [bits]. Default ``inf``: no quantization.
    Nchannel : int, optional
        Channels served by one converter. Default 1.
    latency : float, optional
        Conversion latency [s]. Default 0.
    **kwargs
        Forwarded to :class:`~loopuss.PeriphCirc`.
    """

    def __init__(self, preset_type: str, *, Nbit=None, Nchannel=None, latency=None, **kwargs):
        self.latency: float = 0.0
        self.Nbit: float = math.inf
        self.Nchannel: int = 1
        super().__init__(preset_type=preset_type, **kwargs)

        if Nbit is not None:
            self.Nbit = float(Nbit)
        if Nchannel is not None:
            self.Nchannel = int(Nchannel)
        if latency is not None:
            self.latency = float(latency)

    def quantclip_array(self, x, fsr):
        """Quantize and clip values or signal levels, channel by channel.

        Parameters
        ----------
        x : array_like, shape (n, m), or list of Signal
            ``n`` channels of ``m`` samples, or one signal per channel.
        fsr : array_like, shape (n, 2) or (1, 2)
            Full-scale range ``[lo, hi]`` of each channel. Infinite ends are
            replaced by the minimum or maximum of the whole ``x``: of all the
            samples, or of all the signals' values.

        Returns
        -------
        ndarray, shape (n, m), or list of Signal
            For an array, with finite ``Nbit``,
            ``lo + lsb * clip(round((x - lo) / lsb), 0, 2**Nbit - 1)`` with
            ``lsb = (hi - lo) / 2**Nbit``; with infinite ``Nbit``, ``x`` clipped
            to ``[lo, hi]``. For signals, copies whose STEP and PULSE levels, DC
            values and PWL values are quantized the same way; SINE and EMPTY
            signals are copied unchanged.
        """
        signals = isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], Signal)
        if not signals:
            x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x) if signals else x.shape[0]
        fsr = np.atleast_2d(np.asarray(fsr, dtype=float))
        if fsr.shape[0] == 1:
            fsr = np.repeat(fsr, n, axis=0)

        if math.isinf(self.Nbit) and np.all(np.isinf(fsr)):
            return [deepcopy(w) for w in x] if signals else x.copy()

        fsr = fsr.copy()
        if signals:
            loaded = [w for w in x if not w.is_empty()]
            lo_all = min((w.min_value() for w in loaded), default=0.0)
            hi_all = max((w.max_value() for w in loaded), default=0.0)
        else:
            lo_all, hi_all = np.min(x), np.max(x)
        fsr[np.isinf(fsr[:, 0]), 0] = lo_all
        fsr[np.isinf(fsr[:, 1]), 1] = hi_all

        if not signals:
            out = x.copy()
            for ii in range(n):
                lo, hi = fsr[ii]
                out[ii] = self._quantize(x[ii], lo, hi)
            return out

        out = []
        for ii, w in enumerate(x):
            lo, hi = fsr[ii]
            q = deepcopy(w)
            if q.is_step() or q.is_pulse():
                q.val1 = float(self._quantize(q.val1 if q.val1 is not None else 0.0, lo, hi))
                q.val2 = float(self._quantize(q.val2 if q.val2 is not None else 0.0, lo, hi))
            elif q.is_dc():
                q.dcvalue = float(self._quantize(q.dcvalue, lo, hi))
            elif q.is_pwl():
                q.valpwl = np.asarray(self._quantize(np.asarray(q.valpwl, dtype=float), lo, hi))
            out.append(q)
        return out

    def _quantize(self, x, lo: float, hi: float):
        """Quantize ``x`` to ``Nbit`` levels over ``[lo, hi]`` and clip it."""
        if math.isinf(self.Nbit):
            return np.clip(x, lo, hi)
        lsb = (hi - lo) / (2.0**self.Nbit)
        fsrd = 2.0**self.Nbit - 1.0
        # If lsb is zero, every code maps to the single level, so we just need to clip
        # (to avoid 0/0)
        if lsb == 0.0:
            return np.clip(x, lo, hi)
        return np.clip(np.round((x - lo) / lsb), 0.0, fsrd) * lsb + lo

    @abstractmethod
    def _load_preset(self, preset_type: str) -> None: ...
