"""Technology node model."""

from __future__ import annotations

_PRESETS: dict[str, dict] = {
    "PTM-MG-HP-7nm": dict(node=7e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.70),
    "PTM-MG-HP-10nm": dict(node=10e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.75),
    "PTM-MG-HP-14nm": dict(node=14e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.80),
    "PTM-MG-HP-16nm": dict(node=16e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.85),
    "PTM-MG-HP-20nm": dict(node=20e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.90),
    "PTM-MG-LSTP-7nm": dict(node=7e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.70),
    "PTM-MG-LSTP-10nm": dict(node=10e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.75),
    "PTM-MG-LSTP-14nm": dict(node=14e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.80),
    "PTM-MG-LSTP-16nm": dict(node=16e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.85),
    "PTM-MG-LSTP-20nm": dict(node=20e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.90),
    "PTM-HP-16nm": dict(node=16e-9, Rswn=75, Rswp=70.0, Rsh=5, Wr=1, Vdd=0.70),
    "PTM-HP-22nm": dict(node=22e-9, Rswn=75, Rswp=72.5, Rsh=5, Wr=1, Vdd=0.80),
    "PTM-HP-32nm": dict(node=32e-9, Rswn=75, Rswp=75.0, Rsh=5, Wr=1, Vdd=0.90),
    "PTM-HP-45nm": dict(node=45e-9, Rswn=80, Rswp=75.0, Rsh=5, Wr=1, Vdd=1.00),
    "PTM-LP-16nm": dict(node=16e-9, Rswn=75, Rswp=72.5, Rsh=5, Wr=1, Vdd=0.90),
    "PTM-LP-22nm": dict(node=22e-9, Rswn=75, Rswp=72.5, Rsh=5, Wr=1, Vdd=0.95),
    "PTM-LP-32nm": dict(node=32e-9, Rswn=75, Rswp=75.0, Rsh=5, Wr=1, Vdd=1.00),
    "PTM-LP-45nm": dict(node=45e-9, Rswn=80, Rswp=75.0, Rsh=5, Wr=1, Vdd=1.10),
}

# Effective area nodes for sub-28 nm lithographic nodes (minimum metal pitch table).
_MMP: dict[float, float] = {14e-9: 42e-9, 16e-9: 64e-9}


class TechModel:
    """Technology node model.

    Sets the lithographic node used for area estimates and for rescaling
    peripherals (see :class:`~loopuss.PeriphCirc`). The presets are PTM
    process corners, listed in :doc:`/presets`.

    Parameters
    ----------
    preset : str, optional
        Preset name. An unknown name raises ``ValueError``. Default: a
        custom 14 nm node.
    **kwargs
        Attribute overrides, matched by exact name and cast to float. Unknown
        names are ignored.

    Attributes
    ----------
    type : str
        Preset name, or ``"Custom"``.
    node : float
        Lithographic node [m]. Default 14 nm.
    Rswn, Rswp : float
        Sheet resistance of n-well and p-well resistors [Ω/□]. Default 300.
    Rsh : float
        Diffusion sheet resistance [Ω/□]. Default 5. Not used by the models.
    Wr : float
        Resistor form factor. Default 1.
    Vdd : float
        Nominal supply voltage [V]. Default 0.8.
    """

    def __init__(self, preset: str | None = None, **kwargs):
        self.type: str = "Custom"
        self.node: float = 14e-9
        self.Rswn: float = 300.0
        self.Rswp: float = 300.0
        self.Rsh: float = 5.0
        self.Wr: float = 1.0
        self.Vdd: float = 0.8

        if preset is not None:
            if preset not in _PRESETS:
                raise ValueError(
                    f"Unknown TechModel preset {preset!r}. Available: {', '.join(_PRESETS)}."
                )
            self.type = preset
            for k, v in _PRESETS[preset].items():
                setattr(self, k, v)

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    @property
    def effective_node(self) -> float:
        """Node used for area scaling [m].

        The 14 nm and 16 nm nodes map to their minimum metal pitch (42 nm and
        64 nm), because their cell area does not shrink with the nominal node.
        Other nodes are returned unchanged.
        """
        return _MMP.get(self.node, self.node)

    def resistor_size(self, R: float, rtype: str = "n") -> tuple[float, float, float]:
        """Size a well resistor.

        Parameters
        ----------
        R : float
            Resistance [Ω].
        rtype : {"n", "p"}, optional
            Well type; any value other than ``"n"`` uses ``Rswp``. Default ``"n"``.

        Returns
        -------
        W, L, area : float
            Width (the node) and length [m], and area [m²].
        """
        Rsw = self.Rswn if rtype == "n" else self.Rswp
        W = self.node
        L = R * self.Wr / Rsw * W
        return W, L, W * L

    def resistor_value(self, W: float, L: float, rtype: str = "n") -> float:
        """Resistance [Ω] of a well resistor of width ``W`` and length ``L`` [m]."""
        Rsw = self.Rswn if rtype == "n" else self.Rswp
        return Rsw * (L / W) / self.Wr

    def __repr__(self) -> str:
        return f"TechModel({self.type!r}, Node={self.node * 1e9:.0f}nm)"
