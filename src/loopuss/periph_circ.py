"""Abstract base for circuit peripherals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .tech_model import TechModel


class PeriphCirc(ABC):
    """Abstract base class for peripheral circuit models.

    Holds what every peripheral shares: a preset type, area, supply, leakage,
    dynamic energy, a technology node, and an energy accumulator that fills up
    during simulation. :class:`~loopuss.OAModel`, :class:`~loopuss.ADCModel`,
    :class:`~loopuss.VoltageDAC`, :class:`~loopuss.CurrentDAC`,
    :class:`~loopuss.MemWriteModel` and :class:`~loopuss.DSPModel` derive from
    it.

    Construction runs in three steps: the preset named by ``preset_type`` is
    loaded, the model is rescaled if ``technology`` has a different node, and
    the keyword arguments below override individual fields.

    Parameters
    ----------
    preset_type : str
        Preset name. A name the subclass does not know raises ``ValueError``.
    technology : TechModel, optional
        Target technology. If its node differs from the preset's, ``area`` is
        scaled by the squared ratio of the effective nodes, and the supply
        current (hence the leakage) by the ratio of the nodes.
    area : float, optional
        Area [m²]. Default 0.
    dynamic_energy : float, optional
        Energy per operation [J]. Default 0.
    Vsupply : float or array_like of 2 floats, optional
        Supply rails ``[Vss, Vdd]`` [V]; a scalar ``v`` means ``[-v, v]``.
        Default ``[-inf, inf]``, i.e. no supply limit.
    Isupply : float, optional
        Supply current [A]. Default 0. A non-zero value needs finite rails,
        from the preset or from ``Vsupply``.
    leakage_power : float, optional
        Static power [W]. With finite rails it is converted to
        ``Isupply = leakage_power / (Vdd - Vss)``, and cannot be combined with
        ``Isupply``. With unbounded rails it is stored as given.

    Raises
    ------
    ValueError
        If ``Isupply`` is non-zero without finite rails, or if ``Isupply`` and
        ``leakage_power`` are both given with finite rails.

    Notes
    -----
    With finite rails the leakage is not stored: :attr:`leakage_power` is
    always ``(Vdd - Vss) * Isupply``, so changing the rails or the supply
    current changes it.

    A subclass ``__init__`` must set its own default fields *before* calling
    ``super().__init__()``, because the base calls the subclass's
    ``_load_preset`` and those defaults would otherwise overwrite the preset.
    """

    def __init__(
        self,
        preset_type: str,
        *,
        technology=None,
        area=None,
        dynamic_energy=None,
        Vsupply=None,
        Isupply=None,
        leakage_power=None,
    ):
        # Set PeriphCirc-level defaults. Subclass __init__ must initialise
        # its own fields BEFORE calling super() so that _load_preset can
        # override them correctly.
        self.type: str = preset_type
        self.area: float = 0.0
        self._energy: dict = {"Dynamic": 0.0, "Static": 0.0}
        self.Isupply: float = 0.0
        self.Vsupply: np.ndarray = np.array([-np.inf, np.inf])
        # Leakage for unbounded rails only; with finite rails it is derived.
        self._leakage_power: float = 0.0
        self.dynamic_energy: float = 0.0
        self.technology: TechModel = TechModel()

        # Load preset values (calls overridden method in concrete subclass).
        self._load_preset(preset_type)

        # Scale to custom technology node before applying individual overrides.
        if technology is not None and technology.node != self.technology.node:
            self._scale_tech(technology)
            self.technology = technology

        # Apply PeriphCirc-level keyword overrides.
        if area is not None:
            self.area = float(area)

        if dynamic_energy is not None:
            self.dynamic_energy = float(dynamic_energy)

        if Vsupply is not None:
            v = np.asarray(Vsupply, dtype=float).ravel()
            self.Vsupply = np.array([-v[0], v[0]]) if v.size == 1 else v

        # Rails are resolved first, since they decide how Isupply and
        # leakage_power relate.
        if Isupply is not None:
            if Isupply != 0 and not self._has_rails():
                raise ValueError("Isupply needs finite Vsupply rails to define a leakage power.")
            if leakage_power is not None and self._has_rails():
                raise ValueError("Pass either Isupply or leakage_power, not both.")
            self.Isupply = float(Isupply)

        if leakage_power is not None:
            self._set_supply_from_power(float(leakage_power))

    # --- preset ---
    @abstractmethod
    def _load_preset(self, preset_type: str) -> None:
        """Load parameter values for a named preset. Called from __init__."""

    def _unknown_preset(self, name: str, known) -> ValueError:
        """Error for a preset name the subclass does not define."""
        return ValueError(
            f"Unknown {type(self).__name__} preset {name!r}. Available: {', '.join(known)}."
        )

    # --- supply ---
    def _vspan(self) -> float:
        return float(np.diff(np.asarray(self.Vsupply, dtype=float).ravel())[0])

    def _has_rails(self) -> bool:
        span = self._vspan()
        return bool(np.isfinite(span) and span > 0)

    @property
    def leakage_power(self) -> float:
        """Static power [W], read-only.

        ``(Vdd - Vss) * Isupply`` with finite rails. With unbounded rails, the
        ``leakage_power`` given at construction (default 0).
        """
        if self._has_rails():
            return self._vspan() * self.Isupply
        return self._leakage_power

    def _set_supply_from_power(self, power: float) -> None:
        """Set the supply from a static power figure [W].

        With finite rails the power is stored as ``Isupply = power / (Vdd -
        Vss)``; with unbounded rails it is stored as the leakage itself. Set
        ``Vsupply`` first, since the rails decide which one it is.
        """
        if self._has_rails():
            self.Isupply = power / self._vspan()
        else:
            self._leakage_power = power

    # --- energy ---
    @property
    def energy(self) -> dict[str, float]:
        """Accumulated energy [J]: ``Dynamic``, ``Static`` and ``Total``."""
        return {
            "Dynamic": self._energy["Dynamic"],
            "Static": self._energy["Static"],
            "Total": self._energy["Dynamic"] + self._energy["Static"],
        }

    def reset(self) -> None:
        """Clear the energy accumulator."""
        self._energy = {"Dynamic": 0.0, "Static": 0.0}

    def add_energy(self, kind: str, value: float) -> None:
        """Add ``value`` [J] to the ``"Dynamic"`` or ``"Static"`` accumulator."""
        self._energy[kind] = self._energy.get(kind, 0.0) + float(value)

    # --- tech scaling ---
    def _scale_tech(self, new_tech: TechModel) -> None:
        """Scale Area and supply current (hence leakage) to a new technology node."""
        old_eff = self.technology.effective_node
        new_eff = new_tech.effective_node
        old_node = self.technology.node
        new_node = new_tech.node

        self.area *= (new_eff / old_eff) ** 2
        # Only stored quantities are scaled: with finite rails the leakage
        # follows Isupply, and scaling both would apply the ratio twice.
        self.Isupply *= new_node / old_node
        self._leakage_power *= new_node / old_node

    def __repr__(self) -> str:
        return f"{type(self).__name__}(type={self.type!r})"
