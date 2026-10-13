"""MemWriteModel — memory programming/write model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .periph_circ import PeriphCirc
from .tech_model import TechModel

# Preset table of published programming-peripheral characterisations. A string
# ``dynamic_energy`` names a per-bit energy law, resolved in _load_preset.
_PRESETS: dict[str, dict] = {
    "SRAM": dict(
        type="SRAM",
        Vsupply=[0, 1.2],
        leakage_power=0.11e-7,
        latency=1e-9,
        dynamic_energy=7.0539e-12,
        area=3.09e-12,
        SigmaG=0,
    ),
    "Sancandi21": dict(
        type="SRAM",
        Vsupply=[0, 1.2],
        leakage_power=5.4e-9,
        latency=2.5e-9,
        dynamic_energy="lambda_n",
        area=6.46e-12,
        SigmaG=0,
        technology=TechModel(node=90e-9),
    ),
    "Zahoor20_SRAM": dict(
        type="SRAM",
        Vsupply=[0, 1.2],
        leakage_power=0,
        latency=1e-9,
        dynamic_energy="lambda_n1",
        area=3e-12,
        SigmaG=0,
    ),
    "Gul22_6T": dict(
        type="SRAM",
        Vsupply=[0, 1.2],
        leakage_power=5.6e-9,
        latency=33.53e-12,
        dynamic_energy="lambda_n0815",
        area=4.22e-15,
        SigmaG=0,
    ),
    "Zahoor20_RRAM": dict(
        type="RRAM",
        Vsupply=[0, 1.2],
        leakage_power=0,
        latency=5e-9,
        dynamic_energy="lambda_n01",
        area=3e-12,
        SigmaG=0,
    ),
    "IGVVA10": dict(
        type="RRAM",
        Vsupply=[0, 1.2],
        leakage_power=0,
        latency=7e-6,
        dynamic_energy=1.5e-9,
        area=1.79e-12,
        SigmaG=5.8e-6,
    ),
}


class MemWriteModel(PeriphCirc):
    """Memory programming (write) peripheral model.

    :meth:`LoopUS.reprogram <loopuss.LoopUS.reprogram>` uses it to program the
    crossbar conductances; this also happens once at construction. It adds
    programming variability and accounts for programming latency and energy.
    See :doc:`/presets` for the literature presets.

    Parameters
    ----------
    preset_type : str, optional
        Preset name. Default ``"ideal"``: exact, instantaneous and free.
    SigmaG : float, optional
        Standard deviation of the programming error. Default 0.
    sigma_type : str, optional
        ``"absolute"`` (default): ``SigmaG`` is a conductance [S] added to
        every programmable cell. Any other value: ``SigmaG`` is relative to
        each cell's target conductance.
    latency : float, optional
        Time to program one chunk of cells [s]. Default 0.
    parallel_write : float or array_like of 2 floats, optional
        Rows and columns programmed simultaneously; a scalar applies to both,
        and ``inf`` means the whole dimension. Default ``[1, 1]``, one cell at
        a time. The circuit area counts ``sum(parallel_write)`` drivers.
    **kwargs
        Forwarded to :class:`~loopuss.PeriphCirc`. ``dynamic_energy`` is the
        energy per programmed cell [J]; some presets set it to a function of
        the cell's ``Nbit`` instead (energy per bit).
    """

    def __init__(
        self,
        preset_type: str = "ideal",
        *,
        SigmaG=None,
        sigma_type=None,
        latency=None,
        parallel_write=None,
        **kwargs,
    ):
        self.latency: float | Callable = 0.0
        self.parallel_write: np.ndarray = np.array([1, 1])
        self.SigmaG: float = 0.0
        self.sigma_type: str = "absolute"
        super().__init__(preset_type=preset_type, **kwargs)

        if SigmaG is not None:
            self.SigmaG = float(SigmaG)
        if sigma_type is not None:
            self.sigma_type = str(sigma_type)
        if latency is not None:
            self.latency = float(latency)
        if parallel_write is not None:
            v = np.asarray(parallel_write, dtype=float).ravel()
            self.parallel_write = np.array([v[0], v[0]]) if v.size == 1 else v[:2]

    def _load_preset(self, preset_type: str) -> None:
        if preset_type == "ideal":
            return
        if preset_type not in _PRESETS:
            raise self._unknown_preset(preset_type, ["ideal", *_PRESETS])
        p = _PRESETS[preset_type]
        self.type = p.get("type", preset_type)
        self.Vsupply = np.asarray(p.get("Vsupply", [-np.inf, np.inf]), dtype=float)
        self._set_supply_from_power(float(p.get("leakage_power", 0.0)))
        self.area = float(p.get("area", 0.0))
        self.SigmaG = float(p.get("SigmaG", 0.0))
        self.technology = p.get("technology", TechModel())
        self.parallel_write = np.array([1, 1])

        lat = p.get("latency", 0.0)
        self.latency = float(lat)

        de = p.get("dynamic_energy", 0.0)
        # Handle lambda variants (energy per bit)
        if de == "lambda_n":
            self.dynamic_energy = lambda n: 3.526e-15 * n
        elif de == "lambda_n1":
            self.dynamic_energy = lambda n: 1e-15 * n
        elif de == "lambda_n01":
            self.dynamic_energy = lambda n: 0.1e-12 * n
        elif de == "lambda_n0815":
            self.dynamic_energy = lambda n: 8.15e-16 * n
        else:
            self.dynamic_energy = float(de)

    def program(self, crossbar) -> tuple:
        """Program a crossbar.

        Every cell in ``crossbar.prog_mask`` receives a Gaussian programming
        error (see ``SigmaG`` and ``sigma_type``), drawn from NumPy's global
        random generator. Cells whose sign would flip are set to zero. The
        result is not re-quantized.

        Programming takes ``latency * Nchunks`` seconds, with
        ``Nchunks = prod(ceil(shape / parallel_write))``. Dynamic energy is
        ``dynamic_energy`` per programmable cell. Static energy is
        ``leakage_power`` times the programming time. Both are added to the
        accumulator.

        Parameters
        ----------
        crossbar : CrossBar
            Target array.

        Returns
        -------
        Y : CrossBar
            Programmed array. It keeps the line resistance, memory model,
            programming mask and the noise and buffer settings of ``crossbar``.
        T : float
            Programming latency [s].
        E : float
            Total programming energy ``Ed + Es`` [J].
        Ed : float
            Dynamic energy [J].
        Es : float
            Static energy [J].
        """
        from .crossbar import CrossBar

        if crossbar.G.size == 0:
            return crossbar, 0.0, 0.0, 0.0, 0.0

        G = crossbar.G.copy()
        if self.sigma_type.lower() == "absolute":
            G_prog = G + self.SigmaG * np.random.randn(*G.shape) * crossbar.prog_mask
        else:
            G_prog = G * (1 + crossbar.prog_mask * self.SigmaG * np.random.randn(*G.shape))
        # Zero out sign-flipped cells
        G_prog[np.sign(G_prog) != np.sign(G)] = 0.0

        Y = CrossBar(
            G_prog,
            skip_quant=True,
            line_resistance=crossbar.line_resistance,
            memory_model=crossbar.memory_model,
            prog_mask=crossbar.prog_mask,
            noiseless=crossbar.noiseless,
            ideal_buffers=crossbar.ideal_buffers,
        )

        pW = self.parallel_write.copy()
        pW[np.isinf(pW)] = max(crossbar.G.shape)
        pW = pW.astype(int)
        Nchunks = int(np.prod(np.ceil(np.array(crossbar.G.shape) / pW)))

        T = float(self.latency) * Nchunks

        n_prog = int(np.count_nonzero(crossbar.prog_mask))
        de = self.dynamic_energy
        if callable(de):
            Ed = float(de(crossbar.memory_model.Nbit)) * n_prog
        else:
            Ed = float(de) * n_prog
        Es = self.leakage_power * T
        E = Ed + Es

        self._energy["Dynamic"] += Ed
        self._energy["Static"] += Es
        return Y, T, E, Ed, Es

    def __repr__(self) -> str:
        return f"MemWriteModel(type={self.type!r})"
