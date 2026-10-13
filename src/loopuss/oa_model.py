"""Operational amplifier model."""

from __future__ import annotations

import numpy as np

from .periph_circ import PeriphCirc
from .tech_model import TechModel

# Preset table of published op-amp characterisations.
_PRESETS: dict[str, dict] = {
    "1P": {},  # default values, no preset overrides
    "GM": {},
    "2P": {},
    "2P1Z": {},
    "Valero12": dict(
        type="1P",
        A0=354.8134,
        GBWP=40e3,
        Vsupply=np.array([0.0, 0.8]),
        Isupply=1.25e-6,
        area=104.46e-12,
        technology=TechModel(node=180e-9),
    ),
    "Gupta16": dict(
        type="1P",
        A0=4000.0,
        GBWP=13.33e6,
        Vsupply=np.array([0.0, 1.8]),
        Isupply=71.974e-6,
        area=656e-12,
        technology=TechModel(node=180e-9),
    ),
    "Kouhalvandi17": dict(
        type="1P",
        A0=7.08e5,
        GBWP=22.85e6,
        Vsupply=np.array([0.0, 1.8]),
        Isupply=666.67e-6,
        area=104.46e-12,
        technology=TechModel(node=180e-9),
    ),
    "Nidhi19": dict(
        type="1P",
        A0=5734.6,
        GBWP=7.73e6,
        Vsupply=np.array([0.0, 1.5]),
        Isupply=93e-12,
        area=1.7365e-9,
        technology=TechModel(node=90e-9),
    ),
    "Parthipan19": dict(
        type="1P",
        A0=2.512e4,
        GBWP=1.45e9,
        Vsupply=np.array([0.0, 1.5]),
        Isupply=149.87e-6,
        area=0.042e-6,
        technology=TechModel(node=90e-9),
    ),
    "Rahaman19": dict(
        type="1P",
        A0=14.96,
        GBWP=2.37e6,
        Vsupply=np.array([-10.0, 10.0]),
        Isupply=2.55e-3,
        area=0.7e-6,
        technology=TechModel(node=20e-9),
    ),
    "AlQaysi20": dict(
        type="1P",
        A0=4518.6,
        GBWP=14.9e6,
        Vsupply=np.array([-1.0, 1.0]),
        Isupply=455e-6,
        area=122.24e-12,
        technology=TechModel(node=90e-9),
    ),
    "Padma20": dict(
        type="1P",
        A0=3.1623e4,
        GBWP=530e6,
        Vsupply=np.array([-0.9, 0.9]),
        Isupply=2.4e-3,
        area=0.0138e-6,
        technology=TechModel(node=180e-9),
    ),
    "Feinberg21": dict(
        type="1P",
        A0=63.0957,
        GBWP=500e6,
        Vsupply=np.array([0.0, 0.8]),
        Isupply=15e-6,
        area=50e-12,
        technology=TechModel(node=14e-9),
    ),
    "Sancandi21": dict(
        type="1P",
        A0=2000.0,
        GBWP=20e6,
        Vsupply=np.array([0.0, 1.2]),
        Isupply=456e-6,
        area=6292e-12,
        Vos=-96e-6,
        Rout=920.0,
        Iout=np.array([-3.6e-3, 3.6e-3]),
        technology=TechModel(node=90e-9),
    ),
}


class OAModel(PeriphCirc):
    """Operational amplifier model.

    A gain stage with DC gain ``A0``, poles ``FP`` and zeros ``FZ`` sets the
    frequency response. An output stage with rails ``Vsupply``, current limits
    ``Iout`` and output resistance ``Rout`` models saturation. The analytical
    solvers use ``A0``, the first pole, ``GBWP``, ``Vos`` and ``Vsupply``, so
    the output-stage limits only take effect in SPICE, and multi-pole
    topologies (``"2P"``, ``"2P1Z"``) are rejected by the analytical pole,
    transient and noise solvers. There, each op-amp becomes the ``NonLinearOA`` subcircuit that
    matches the non-idealities in use. See :doc:`/presets` for the literature
    presets.

    Parameters
    ----------
    preset_type : str, optional
        Topology or preset name. The topologies are ``"1P"`` (single pole,
        the default), ``"GM"`` (ideal integrator: ``A0 = inf`` and a pole at
        0, characterized by ``GBWP``), ``"2P"`` (two poles) and ``"2P1Z"``
        (two poles, one zero). All literature presets are single-pole. An
        unknown name raises ``ValueError``.
    A0 : float, optional
        DC differential gain [V/V]. Default 1e4 (80 dB).
    GBWP : float, optional
        Gain-bandwidth product [Hz]. Default 100 MHz. It is replaced by
        ``A0 * FP[0]`` when the two differ by more than 0.1 %.
    FP : array_like, optional
        Pole frequencies [Hz], sorted ascending. Default ``GBWP / A0``.
    FZ : array_like, optional
        Zero frequencies [Hz], sorted descending. Default: none.
    Vos : float, optional
        Input offset voltage [V]. Default 0.
    Rout : float, optional
        Output resistance [Ω]. Default 0.
    Iout : float or array_like of 2 floats, optional
        Negative and positive output current limits [A]; a scalar ``i`` means
        ``[-i, i]``. Default unlimited.
    input_capacitance : float, optional
        Differential input capacitance [F]. Default 0. A non-zero value makes
        :meth:`LoopUS.pz <loopuss.LoopUS.pz>` raise ``NotImplementedError``.
    input_resistance : float, optional
        Differential input resistance [Ω]. Default ``inf``.
    noise_voltage : float, optional
        Input voltage noise density [V/√Hz]. Default 0.
    noise_current : float or array_like of 2 floats, optional
        Current noise densities of the inverting and non-inverting inputs
        [A/√Hz]; a scalar sets the inverting input only. Default 0.
    **kwargs
        Forwarded to :class:`~loopuss.PeriphCirc` (``Vsupply``,
        ``technology``, ``area``, ...). Finite ``Vsupply`` rails make the
        op-amp saturate.
    """

    def __init__(
        self,
        preset_type: str = "1P",
        *,
        A0=None,
        GBWP=None,
        FZ=None,
        Vos=None,
        Rout=None,
        Iout=None,
        input_capacitance=None,
        input_resistance=None,
        noise_voltage=None,
        noise_current=None,
        FP=None,
        **kwargs,
    ):
        # Initialise OA-specific defaults before super().__init__ calls
        # _load_preset, which will override these with preset values.
        self.A0: float = 1e4
        self.GBWP: float = 100e6
        self.FP: np.ndarray = np.array([10e3])
        self.FZ: np.ndarray = np.array([])
        self.Vos: float = 0.0
        self.Rout: float = 0.0
        self.Iout: np.ndarray = np.array([-np.inf, np.inf])
        self.input_capacitance: float = 0.0
        self.input_resistance: float = np.inf
        self.noise_voltage: float = 0.0
        self.noise_current: np.ndarray = np.array([0.0, 0.0])

        super().__init__(preset_type, **kwargs)

        # Apply OA-specific keyword overrides (after preset loading).
        if A0 is not None:
            self.A0 = float(A0)

        if GBWP is not None:
            self.GBWP = float(GBWP)

        if FZ is not None:
            self.FZ = np.sort(np.atleast_1d(np.asarray(FZ, dtype=float)))[::-1]

        if Vos is not None:
            self.Vos = float(Vos)

        if Rout is not None:
            self.Rout = float(Rout)

        if Iout is not None:
            ia = np.atleast_1d(np.asarray(Iout, dtype=float))
            self.Iout = np.array([-ia[0], ia[0]]) if ia.size == 1 else ia

        if input_capacitance is not None:
            self.input_capacitance = float(input_capacitance)

        if input_resistance is not None:
            self.input_resistance = float(input_resistance)

        if noise_voltage is not None:
            self.noise_voltage = float(noise_voltage)

        if noise_current is not None:
            nc = np.atleast_1d(np.asarray(noise_current, dtype=float))
            self.noise_current = np.array([nc[0], 0.0]) if nc.size == 1 else nc

        # FP: explicit override or derived from GBWP/A0.
        if FP is not None:
            self.FP = np.sort(np.atleast_1d(np.asarray(FP, dtype=float)))
        else:
            self.FP = np.atleast_1d(self.GBWP / self.A0)

        self._validate()

    def _load_preset(self, preset_type: str) -> None:
        if preset_type not in _PRESETS:
            raise self._unknown_preset(preset_type, _PRESETS)
        for k, v in _PRESETS[preset_type].items():
            if k == "type":
                self.type = v
            elif hasattr(self, k):
                setattr(self, k, v)
        # Presets store Isupply. PeriphCirc.leakage_power derives
        # the leakage from it and the rails.

    def _validate(self) -> None:
        """Validate/normalise the model. For GM force the ideal-integrator
        limit (A0 = Inf, FP = 0); otherwise keep GBWP consistent with A0 x FP."""
        if str(self.type).upper() == "GM":
            # Pure transconductor: infinite open-loop gain, pole at the origin.
            # GBWP (the unity-gain frequency) carries the model information.
            self.A0 = np.inf
            self.FP = np.array([0.0])
            return
        computed_gbwp = float(self.A0 * self.FP[0])
        if not np.isclose(computed_gbwp, self.GBWP, rtol=1e-3):
            self.GBWP = computed_gbwp

    @property
    def P0(self) -> float:
        """Gain-bandwidth product (A0 x FP). For GM this is the GBWP, since the
        ideal model stores A0 = Inf and FP = 0."""
        if str(self.type).upper() == "GM":
            return float(self.GBWP)
        return float(self.A0 * self.FP[0])

    # --- SPICE ---
    def model_str(self) -> str:
        """NonLinearOA library subcircuit name."""
        m = self.type

        m += "_Y" if self.Vos == 0 else "_O"

        if not np.isinf(self.input_resistance) and self.input_capacitance != 0:
            m += "_RC"
        elif not np.isinf(self.input_resistance):
            m += "_R"
        elif self.input_capacitance != 0:
            m += "_C"
        else:
            m += "_Y"

        m += "_V" if np.any(~np.isinf(self.Vsupply)) else "_Y"

        if np.any(~np.isinf(self.Iout)) and self.Rout != 0:
            m += "_IR"
        elif np.any(~np.isinf(self.Iout)):
            m += "_I"
        elif self.Rout != 0:
            m += "_R"
        else:
            m += "_Y"

        nc = np.asarray(self.noise_current, dtype=float).ravel()
        if self.noise_voltage != 0 or np.any(nc != 0):
            m += "_noisy_"
            if self.noise_voltage != 0:
                m += "v"
            if nc[0] != 0:
                m += "in"
            if nc[1] != 0:
                m += "ip"

        return f"NonLinearOA_{m}"

    def spice_params(self) -> str:
        """NonLinearOA parameter string."""
        FP = np.atleast_1d(np.asarray(self.FP, dtype=float))
        FZ = np.atleast_1d(np.asarray(self.FZ, dtype=float)) if np.size(self.FZ) else np.array([])
        nc = np.asarray(self.noise_current, dtype=float).ravel()

        # GM (pure integrator) uses GBWP directly. The ideal GM has FP=0/A0=Inf,
        # but finite placeholders may be stored, so key off the type too.
        if (FP.size == 1 and FP[0] == 0) or self.type.upper() == "GM":
            s = f"GBWP={{{self.GBWP:g}}}"
        elif not np.isinf(self.A0):
            s = f"A0={{{self.A0:g}}}"
            for i, fp in enumerate(FP, start=1):
                s += f" FP{i}={{{fp:g}}}"
            for i, fz in enumerate(FZ, start=1):
                s += f" FZ{i}={{{fz:g}}}"
        else:
            s = ""

        if self.Vos != 0:
            s += f" VOS={{{self.Vos:g}}}"
        if not np.isinf(self.input_resistance):
            s += f" RIN={{{self.input_resistance:g}}}"
        if self.input_capacitance != 0:
            s += f" CIN={{{self.input_capacitance:g}}}"
        if self.Rout != 0:
            s += f" ROUT={{{self.Rout:g}}}"
        if np.any(~np.isinf(self.Iout)):
            s += f" ISS={{{-self.Iout[0]:g}}} IDD={{{self.Iout[1]:g}}}"
        if self.noise_voltage != 0:
            s += f" SVP={{{self.noise_voltage:g}}}"
        if nc[0] != 0:
            s += f" SIN={{{nc[0]:g}}}"
        if nc[1] != 0:
            s += f" SIP={{{nc[1]:g}}}"
        return s

    def to_spice(
        self, oa_id, inn: str, inp: str, out: str, vss: str | None = None, vdd: str | None = None
    ) -> str:
        """SPICE XOA instance line."""
        s = f"XOA{oa_id} {inn} {inp} {out}"
        if vss and vdd:
            s += f" {vss} {vdd}"
        s += f" {self.model_str()} params: {self.spice_params()}"
        return s
