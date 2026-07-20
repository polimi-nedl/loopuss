"""Abstract base for circuit peripherals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .tech_model import TechModel


class PeriphCirc(ABC):
    """Abstract base class for all peripheral circuit models."""

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
        self.leakage_power: float = 0.0
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

        if Isupply is not None:
            self.Isupply = float(Isupply)

        if leakage_power is not None:
            self.leakage_power = float(leakage_power)

        # LeakagePower has priority; update Isupply to match if inconsistent.
        vspan = float(np.diff(self.Vsupply)[0])
        if np.isfinite(vspan) and vspan > 0 and self.leakage_power != 0:
            expected_lp = vspan * self.Isupply
            if not np.isclose(expected_lp, self.leakage_power, rtol=1e-3):
                self.Isupply = self.leakage_power / vspan

    # --- preset ---
    @abstractmethod
    def _load_preset(self, preset_type: str) -> None:
        """Load parameter values for a named preset. Called from __init__."""

    # --- energy ---
    @property
    def energy(self) -> dict[str, float]:
        return {
            "Dynamic": self._energy["Dynamic"],
            "Static": self._energy["Static"],
            "Total": self._energy["Dynamic"] + self._energy["Static"],
        }

    def reset(self) -> None:
        self._energy = {"Dynamic": 0.0, "Static": 0.0}

    def add_energy(self, kind: str, value: float) -> None:
        self._energy[kind] = self._energy.get(kind, 0.0) + float(value)

    # --- tech scaling ---
    def _scale_tech(self, new_tech: TechModel) -> None:
        """Scale Area, Isupply, and LeakagePower to a new technology node."""
        old_eff = self.technology.effective_node
        new_eff = new_tech.effective_node
        old_node = self.technology.node
        new_node = new_tech.node

        self.area *= (new_eff / old_eff) ** 2
        self.Isupply *= new_node / old_node
        self.leakage_power *= new_node / old_node

    def __repr__(self) -> str:
        return f"{type(self).__name__}(type={self.type!r})"
