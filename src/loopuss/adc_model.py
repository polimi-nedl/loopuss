"""ADCModel — analog-to-digital converter model."""

from __future__ import annotations

import math

import numpy as np

from .converter_model import ConverterModel
from .tech_model import TechModel

_PRESETS: dict[str, dict] = {
    "Sandner05": dict(
        Nbit=6,
        Nchannel=1,
        latency=1.7e-9,
        dynamic_energy=0.5 * 150e-12,
        area=0.12e-6,
        technology=TechModel(node=130e-9),
        Vsupply=[0, 1.5],
        leakage_power=0,
    ),
    "Pang09": dict(
        Nbit=10,
        Nchannel=1,
        latency=2e-6,
        dynamic_energy=0.5 * 84e-12,
        area=0.24e-6,
        technology=TechModel(node=180e-9),
        Vsupply=[0, 1],
        leakage_power=0,
    ),
    "Ahmadi13": dict(
        Nbit=10,
        Nchannel=1,
        latency=4e-6,
        dynamic_energy=0.5 * 2.4e-12,
        area=0.072e-6,
        technology=TechModel(node=65e-9),
        Vsupply=[-0.5, 0.5],
        leakage_power=0,
    ),
    "Brenna14": dict(
        Nbit=10,
        Nchannel=1,
        latency=1.11e-6,
        dynamic_energy=0.5 * 5.833e-12,
        area=0.045e-6,
        technology=TechModel(node=130e-9),
        Vsupply=[0, 0.8],
        leakage_power=0,
    ),
    "Kadetotad15": dict(
        Nbit=3,
        Nchannel=1,
        latency=0.4e-9,
        dynamic_energy=0.5 * 0.68e-12,
        area=6.035e-12,
        technology=TechModel(node=32e-9),
        Vsupply=[0, 0.9],
        leakage_power=0.6216e-6,
    ),
    "Wang18": dict(
        Nbit=12,
        Nchannel=2,
        latency=0.5e-9,
        dynamic_energy=0.5 * 25e-12,
        area=0.01e-6,
        technology=TechModel(node=28e-9),
        Vsupply=[0, 0.9],
        leakage_power=0,
    ),
    "Shi18": dict(
        Nbit=10,
        Nchannel=1,
        latency=20e-9,
        dynamic_energy=0.5 * 0.826e-12,
        area=0.1e-6,
        technology=TechModel(node=14e-9),
        Vsupply=[0, 1.6],
        leakage_power=0,
    ),
    "Fan19": dict(
        Nbit=8,
        Nchannel=1,
        latency=1e-9,
        dynamic_energy=0.5 * 2.1e-12,
        area=0.0073e-6,
        technology=TechModel(node=28e-9),
        Vsupply=[0, 1.1],
        leakage_power=0,
    ),
    "Zhang21": dict(
        Nbit=10,
        Nchannel=1,
        latency=40e-9,
        dynamic_energy=0.5 * 2.7456e-12,
        area=0.1e-6,
        technology=TechModel(node=40e-9),
        Vsupply=[0, 1.1],
        leakage_power=0,
    ),
    "Tang21": dict(
        Nbit=10,
        Nchannel=16,
        latency=1e-6,
        dynamic_energy=0.5 * 0.743e-9,
        area=0.3353e-6,
        technology=TechModel(node=180e-9),
        Vsupply=[0, 1.8],
        leakage_power=0,
    ),
    "Bermak21": dict(
        Nbit=10,
        Nchannel=1,
        latency=10e-9,
        dynamic_energy=0.5 * 11e-12,
        area=0.026e-6,
        technology=TechModel(node=28e-9),
        Vsupply=[0, 0.9],
        leakage_power=0,
    ),
    "Sancandi21": dict(
        Nbit=7,
        Nchannel=1,
        latency=51e-6,
        dynamic_energy=28e-9,
        area=12.81e-9,
        technology=TechModel(node=90e-9),
        Vsupply=[0, 1.2],
        leakage_power=0,
    ),
}


class ADCModel(ConverterModel):
    """Analog-to-digital converter model.

    At the end of every :meth:`LoopUS.tran <loopuss.LoopUS.tran>`, the ADC
    quantizes and clips the op-amp outputs selected by ``output_mask``. Its
    latency, area, leakage and dynamic energy feed the benchmark. See
    :doc:`/presets` for the literature presets.

    Parameters
    ----------
    preset_type : str, optional
        Preset name. Default ``"ideal"``: infinite resolution and no latency,
        area or energy, so conversion leaves the signal unchanged.
    Vref : float or array_like of 2 floats, optional
        Full-scale range ``[lo, hi]`` [V]; a scalar ``v`` means ``[-v, v]``.
        Presets do not set it. Default ``[-inf, inf]``: infinite ends are
        replaced by the minimum and maximum of the converted signal.
    **kwargs
        Forwarded to :class:`~loopuss.ConverterModel` (``Nbit``,
        ``Nchannel``, ``latency``) and :class:`~loopuss.PeriphCirc`
        (``technology``, ``area``, ``dynamic_energy``, ``Vsupply``,
        ``Isupply``, ``leakage_power``).
    """

    def __init__(self, preset_type: str = "ideal", *, Vref=None, **kwargs):
        self.Vref: np.ndarray = np.array([-np.inf, np.inf])
        super().__init__(preset_type=preset_type, **kwargs)
        if Vref is not None:
            v = np.asarray(Vref, dtype=float).ravel()
            self.Vref = np.array([-v[0], v[0]]) if v.size == 1 else v

    def _load_preset(self, preset_type: str) -> None:
        if preset_type == "ideal":
            return
        if preset_type not in _PRESETS:
            raise self._unknown_preset(preset_type, ["ideal", *_PRESETS])
        p = _PRESETS[preset_type]
        self.type = preset_type
        self.Nbit = float(p.get("Nbit", math.inf))
        self.Nchannel = int(p.get("Nchannel", 1))
        self.latency = float(p.get("latency", 0.0))
        self.dynamic_energy = float(p.get("dynamic_energy", 0.0))
        self.area = float(p.get("area", 0.0))
        self.technology = p.get("technology", TechModel())
        self.Vsupply = np.asarray(p.get("Vsupply", [-np.inf, np.inf]), dtype=float)
        self._set_supply_from_power(float(p.get("leakage_power", 0.0)))
        self.Vref = np.array([-np.inf, np.inf])

    def quantclip(self, x: np.ndarray) -> np.ndarray:
        """Quantize and clip ``x`` over ``Vref``, see
        :meth:`~loopuss.ConverterModel.quantclip_array`."""
        return self.quantclip_array(np.atleast_2d(x), self.Vref.reshape(1, 2))

    def convert(self, x: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Convert a signal.

        Parameters
        ----------
        x : array_like, shape (channels, samples)
            Analog values [V].

        Returns
        -------
        y : ndarray
            Quantized and clipped values, same shape as ``x``.
        latency : float
            Conversion latency [s].
        dynamic_energy : float
            Energy of this call [J], added to the accumulator once per call
            regardless of the number of samples.
        """
        y = self.quantclip(x)
        self._energy["Dynamic"] += self.dynamic_energy
        return y, self.latency, self.dynamic_energy

    def __repr__(self) -> str:
        return f"ADCModel(type={self.type!r}, Nbit={self.Nbit}, Latency={self.latency:.3g} s)"
