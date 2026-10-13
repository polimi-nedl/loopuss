"""Graphic Processing Unit (GPU) model for benchmarking comparison."""

from __future__ import annotations

import copy

import numpy as np

from .reports import BenchReport, _safe_div
from .tech_model import TechModel
from .utils import GroupedDisplay

_PRESETS: dict[str, dict] = {
    "NvidiaRTXA1000E_FP64": dict(
        MemorySize=4e9,
        MemoryBandwidth=224e9,
        Power=35.0,
        Throughput=72.96e9,
        Nbit=64.0,
        Ncores=2048.0,
        Fclk=1140e6,
        Area=200e-6,
        node=8e-9,
    ),
    "NvidiaJetsonXavierNX_FP16": dict(
        MemorySize=8e9,
        MemoryBandwidth=59.71e9,
        Power=15.0,
        Throughput=1.69e12,
        Nbit=16.0,
        Ncores=384.0,
        Fclk=1100e6,
        Area=350e-6,
        node=12e-9,
    ),
    "NvidiaJetsonXavierNX_INT8": dict(
        MemorySize=8e9,
        MemoryBandwidth=59.71e9,
        Power=15.0,
        Throughput=2.62e12,
        Nbit=8.0,
        Ncores=48.0,
        Fclk=854e6,
        Area=350e-6,
        node=12e-9,
    ),
    "NvidiaRTXA4500E_FP64": dict(
        MemorySize=16e9,
        MemoryBandwidth=512e9,
        Power=115.0,
        Throughput=223.6e9,
        Nbit=64.0,
        Ncores=5888.0 / 64.0,
        Fclk=1215e6,
        Area=392e-6,
        node=8e-9,
    ),
    "NvidiaRTXA4500E_FP32": dict(
        MemorySize=16e9,
        MemoryBandwidth=512e9,
        Power=115.0,
        Throughput=14.31e12,
        Nbit=32.0,
        Ncores=5888.0,
        Fclk=1215e6,
        Area=392e-6,
        node=8e-9,
    ),
    "NvidiaRTXA4500E_FP16": dict(
        MemorySize=16e9,
        MemoryBandwidth=512e9,
        Power=115.0,
        Throughput=14.31e12,
        Nbit=16.0,
        Ncores=5888.0,
        Fclk=1215e6,
        Area=392e-6,
        node=8e-9,
    ),
    "NvidiaRTXA2000E_FP64": dict(
        MemorySize=8e9,
        MemoryBandwidth=192e9,
        Power=60.0,
        Throughput=94.16e9,
        Nbit=64.0,
        Ncores=2560.0 / 64.0,
        Fclk=1177e6,
        Area=200e-6,
        node=8e-9,
    ),
    "NvidiaRTXA2000E_FP32": dict(
        MemorySize=8e9,
        MemoryBandwidth=192e9,
        Power=60.0,
        Throughput=6.026e12,
        Nbit=32.0,
        Ncores=2560.0,
        Fclk=1177e6,
        Area=200e-6,
        node=8e-9,
    ),
    "NvidiaRTXA2000E_FP16": dict(
        MemorySize=8e9,
        MemoryBandwidth=192e9,
        Power=60.0,
        Throughput=6.026e12,
        Nbit=16.0,
        Ncores=2560.0,
        Fclk=1177e6,
        Area=200e-6,
        node=8e-9,
    ),
    "NvidiaRTX5000E_FP16": dict(
        MemorySize=16e9,
        MemoryBandwidth=576e9,
        Power=120.0,
        Throughput=41.15e12,
        Nbit=16.0,
        Ncores=9728.0,
        Fclk=1680e6,
        Area=379e-6,
        node=5e-9,
    ),
    "NvidiaRTX5000E_FP32": dict(
        MemorySize=16e9,
        MemoryBandwidth=576e9,
        Power=120.0,
        Throughput=41.15e12,
        Nbit=32.0,
        Ncores=9728.0,
        Fclk=1680e6,
        Area=379e-6,
        node=5e-9,
    ),
    "NvidiaRTX5000E_FP64": dict(
        MemorySize=16e9,
        MemoryBandwidth=576e9,
        Power=120.0,
        Throughput=643e9,
        Nbit=64.0,
        Ncores=9728.0 / 64.0,
        Fclk=1680e6,
        Area=379e-6,
        node=5e-9,
    ),
    "NvidiaRTX5880_FP16": dict(
        MemorySize=48e9,
        MemoryBandwidth=864e9,
        Power=285.0,
        Throughput=69.27e12,
        Nbit=16.0,
        Ncores=14080.0,
        Fclk=2460e6,
        Area=609e-6,
        node=5e-9,
    ),
    "STM32H7R7": dict(
        MemorySize=620e3,
        MemoryBandwidth=6.4e9,
        Power=0.4125,
        Throughput=1.284e9,
        Ncores=1.0,
        Fclk=600e6,
        Nbit=16.0,
        Area=1.4629e-6,
        node=40e-9,
    ),
}


class GPUModel(GroupedDisplay):
    """Digital processor model, used as a benchmarking baseline.

    :meth:`benchmark` produces a :class:`~loopuss.BenchReport` comparable with
    :meth:`LoopUS.benchmark <loopuss.LoopUS.benchmark>`. The presets cover
    NVIDIA GPUs at several precisions plus one microcontroller; see
    :doc:`/presets`.

    Parameters
    ----------
    type_str : str, optional
        Preset name. Default ``"NvidiaRTXA4500E_FP64"``. An unknown name
        raises ``ValueError``.
    **kwargs
        Case-insensitive overrides: ``MemorySize`` [B],
        ``MemoryBandwidth`` [B/s], ``Power`` [W], ``Ncores``, ``Fclk`` [Hz],
        ``Throughput`` [OPS], ``Nbit``, ``Area`` [m²] and ``Technology``, as
        a :class:`~loopuss.TechModel` or a node in metres. Unknown names are
        ignored.
    """

    def __init__(self, type_str: str = "NvidiaRTXA4500E_FP64", **kwargs):
        self.type: str = type_str
        self.MemorySize: float = 16e9
        self.MemoryBandwidth: float = 512e9
        self.Power: float = 115.0
        self.Ncores: float = 1.0
        self.Fclk: float = 500e6
        self.Throughput: float = 552e9
        self.Nbit: float = 64.0
        self.Area: float = 392e-6
        self.technology: TechModel = TechModel(node=8e-9)

        if type_str not in _PRESETS:
            raise ValueError(
                f"Unknown GPUModel preset {type_str!r}. Available: {', '.join(_PRESETS)}."
            )
        # Load preset
        if type_str in _PRESETS:
            preset_vals = _PRESETS[type_str]
            self.MemorySize = float(preset_vals["MemorySize"])
            self.MemoryBandwidth = float(preset_vals["MemoryBandwidth"])
            self.Power = float(preset_vals["Power"])
            self.Throughput = float(preset_vals["Throughput"])
            self.Nbit = float(preset_vals["Nbit"])
            self.Ncores = float(preset_vals["Ncores"])
            self.Fclk = float(preset_vals["Fclk"])
            self.Area = float(preset_vals["Area"])
            self.technology = TechModel(node=preset_vals["node"])

        # Key-value overrides
        for k, v in kwargs.items():
            k_lower = k.lower()
            if k_lower == "memorysize":
                self.MemorySize = float(v)
            elif k_lower == "memorybandwidth":
                self.MemoryBandwidth = float(v)
            elif k_lower == "power":
                self.Power = float(v)
            elif k_lower == "ncores":
                self.Ncores = float(v)
            elif k_lower == "fclk":
                self.Fclk = float(v)
            elif k_lower == "throughput":
                self.Throughput = float(v)
            elif k_lower == "nbit":
                self.Nbit = float(v)
            elif k_lower == "area":
                self.Area = float(v)
            elif k_lower == "technology":
                if isinstance(v, TechModel):
                    self.technology = v
                else:
                    self.technology = TechModel(node=float(v))

    def benchmark(self, nops: float, ndata: float, psf: float = 1.0) -> BenchReport:
        """Benchmark a workload.

        Parameters
        ----------
        nops : float
            Operations in the workload [OPs].
        ndata : float
            Number of values moved to the device, each ``Nbit / 8`` bytes.
        psf : float, optional
            Power scaling factor of the data-transfer phase, from 0 to 1.
            Default 1.

        Returns
        -------
        BenchReport
            Report with a programming phase (data transfer,
            ``ndata * Nbit / 8 / MemoryBandwidth``) drawing ``psf * Power``
            and a computing phase (``nops / Throughput``) drawing ``Power``.
        """

        # Problem occupation
        prob_size = ndata * self.Nbit / 8.0  # [bytes]

        # Latency
        lat_prog = prob_size / self.MemoryBandwidth
        lat_comp = nops / self.Throughput
        lat_total = lat_prog + lat_comp

        latency = {
            "Programming": lat_prog,
            "Computing": lat_comp,
            "PostProcessing": 0.0,
            "Total": lat_total,
        }

        # Energy
        energy_prog = lat_prog * psf * self.Power
        energy_comp = lat_comp * self.Power
        energy_total = energy_prog + energy_comp

        bd = np.zeros((2, 6))
        bd[0, 1] = float(energy_prog)  # Programming - Dynamic
        bd[1, 3] = float(energy_comp)  # Computing - Dynamic

        energy = {
            "Breakdown": bd,
            "Labels": ["Programming", "Computing"],
            "Total": energy_total,
            "Programming": {"Static": 0.0, "Dynamic": energy_prog, "Total": energy_prog},
            "Computing": {"Static": 0.0, "Dynamic": energy_comp, "Total": energy_comp},
            "PostProcessing": {"Static": 0.0, "Dynamic": 0.0, "Total": 0.0},
        }

        # Power
        p_mean = self.Power * (lat_comp + psf * lat_prog) / lat_total if lat_total > 0 else 0.0
        power = np.array([p_mean, self.Power])

        # Area
        area = {
            "MemWrite": 0.0,
            "GPU": self.Area,
            "Total": self.Area,
        }

        # Performance Metrics
        throughput = _safe_div(nops, lat_total)
        energy_efficiency = _safe_div(nops, energy_total)
        area_efficiency = _safe_div(throughput, self.Area)

        return BenchReport(
            latency=latency,
            energy=energy,
            power=power,
            area=area,
            complexity=float(nops),
            ops=float(nops),
            throughput=throughput,
            energy_efficiency=energy_efficiency,
            area_efficiency=area_efficiency,
            circuit=copy.deepcopy(self),
        )

    def _display_header(self) -> str:
        return f"GPUModel (type={self.type!r})"

    def _property_groups(self):
        from .utils import si_format

        return [
            (
                "Hardware Specs",
                {
                    "MemorySize": f"{si_format(self.MemorySize)}B",
                    "MemoryBandwidth": f"{si_format(self.MemoryBandwidth)}B/s",
                    "Power": f"{si_format(self.Power)}W",
                    "Ncores": int(self.Ncores),
                    "Fclk": f"{si_format(self.Fclk)}Hz",
                    "Throughput": f"{si_format(self.Throughput)}OPS",
                    "Nbit": int(self.Nbit),
                    "Area": f"{si_format(self.Area)}m2",
                    "Technology": self.technology,
                },
            )
        ]
