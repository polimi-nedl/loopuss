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
    "PTM-MG-LTSP-14nm": dict(node=14e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.80),
    "PTM-MG-LTSP-16nm": dict(node=16e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.85),
    "PTM-MG-LTSP-20nm": dict(node=20e-9, Rswn=50, Rswp=50, Rsh=0, Wr=1, Vdd=0.90),
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
    """Technology node characterisation."""

    def __init__(self, preset: str | None = None, **kwargs):
        self.type: str = "Custom"
        self.node: float = 14e-9
        self.Rswn: float = 300.0
        self.Rswp: float = 300.0
        self.Rsh: float = 5.0
        self.Wr: float = 1.0
        self.Vdd: float = 0.8

        if preset is not None:
            if preset in _PRESETS:
                self.type = preset
                for k, v in _PRESETS[preset].items():
                    setattr(self, k, v)
            else:
                self.type = preset

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    @property
    def effective_node(self) -> float:
        """Effective area node (accounts for sub-28 nm multi-patterning)."""
        return _MMP.get(self.node, self.node)

    def resistor_size(self, R: float, rtype: str = "n") -> tuple[float, float, float]:
        """Return (W, L, Area) for a resistance R [Ohm]. rtype: 'n' or 'p'."""
        Rsw = self.Rswn if rtype == "n" else self.Rswp
        W = self.node
        L = R * self.Wr / Rsw * W
        return W, L, W * L

    def resistor_value(self, W: float, L: float, rtype: str = "n") -> float:
        """Return resistance [Ohm] for width W and length L."""
        Rsw = self.Rswn if rtype == "n" else self.Rswp
        return Rsw * (L / W) / self.Wr

    def __repr__(self) -> str:
        return f"TechModel({self.type!r}, Node={self.node * 1e9:.0f}nm)"
