"""Benchmark example: RRAM vs SRAM in-memory linear-system circuits.

Defines realistic peripheral models from presets (OAModel, ADCModel,
CurrentDAC, MemWriteModel, MemoryModel) and maps the same
matrix onto an RRAM and an SRAM crossbar as two LoopUS objects. With
``bench_enable=True`` each object tracks energy / area / latency continuously
during simulation; ``benchmark()`` snapshots them into a BenchReport.

The tracked metrics are visualised with ``BenchReport.barplot()``
(efficiency bar chart) and ``BenchReport.bdplot()`` (breakdown pie charts).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from loopuss import (
    ADCModel,
    BenchReport,
    CrossBar,
    CurrentDAC,
    CurrentSignal,
    LoopUS,
    MemoryModel,
    MemWriteModel,
    OAModel,
)


def main():
    rng = np.random.default_rng(0)

    # Realistic models for all peripherals (from published presets).
    OA = OAModel("Feinberg21", Vsupply=[-1, 1])
    ADC = ADCModel("Fan19")
    DAC = CurrentDAC("Ishraqul17")
    MWSRAM = MemWriteModel("SRAM", SigmaG=0)
    MWRRAM = MemWriteModel("IGVVA10", SigmaG=0)
    RRAM = MemoryModel("Woo16", Nbit=np.inf)
    SRAM = MemoryModel("SRAM", Nbit=np.inf)

    # Matrix and reference vector.
    N, V0 = 10, 1.0
    A = rng.random((N, N))
    A = A.T @ A + 2 * np.eye(N)
    A = A / A.max()
    y = -1 + 2 * rng.random(N)

    # Map the same matrix onto an RRAM and an SRAM feedback crossbar; the input
    # array is empty (a linear-system solve). LoopUS makes the empty input
    # array automatically when input_array is None.
    Gf_SRAM = CrossBar(A * SRAM.LRS, memory_model=SRAM)
    Gf_RRAM = CrossBar(A * RRAM.LRS, memory_model=RRAM)

    LSS_SRAM = LoopUS(
        None,
        Gf_SRAM,
        oa_model=OA,
        adc_model=ADC,
        current_dac=DAC,
        mem_write_model=MWSRAM,
        bench_enable=True,
    )
    LSS_RRAM = LoopUS(
        None,
        Gf_RRAM,
        oa_model=OA,
        adc_model=ADC,
        current_dac=DAC,
        mem_write_model=MWRRAM,
        bench_enable=True,
    )

    # Apply a current-vector input scaled to each technology's conductance.
    LSS_SRAM.set_inputs(CurrentSignal.ytoi(y, I0=SRAM.LRS * V0))
    LSS_RRAM.set_inputs(CurrentSignal.ytoi(y, I0=RRAM.LRS * V0))

    # Benchmark with the analytical solver.
    tgrid = np.concatenate([[0.0], np.logspace(-10, -7, 1000)])
    LSS_SRAM.tran("model", tgrid)
    LSS_RRAM.tran("model", tgrid)

    # Snapshot the tracked metrics.
    BSRAM = LSS_SRAM.benchmark()
    BRRAM = LSS_RRAM.benchmark()

    print(f"{'':12}{'SRAM':>14}{'RRAM':>14}")
    for name, attr in (
        ("Throughput [OPS]", "throughput"),
        ("EnergyEff [OPS/W]", "energy_efficiency"),
        ("AreaEff [OPS/m^2]", "area_efficiency"),
        ("Energy [J]", None),
        ("Latency [s]", None),
    ):
        if attr is not None:
            vs, vr = getattr(BSRAM, attr), getattr(BRRAM, attr)
        elif name.startswith("Energy"):
            vs, vr = BSRAM.energy["Total"], BRRAM.energy["Total"]
        else:
            vs, vr = BSRAM.latency["Total"], BRRAM.latency["Total"]
        print(f"{name:12}{vs:>14.3e}{vr:>14.3e}")

    # Efficiency bar chart comparing both technologies.
    BenchReport.barplot([BSRAM, BRRAM], "-notext", labels=["SRAM", "RRAM"])
    plt.gcf().suptitle("Efficiency metrics: SRAM vs RRAM")
    plt.gcf().tight_layout()

    # Breakdown pie charts, programming contribution excluded (Bvect.bdplot).
    fig = BenchReport.bdplot([BSRAM, BRRAM], "-noprog", labels=["SRAM", "RRAM"])

    plt.show()


if __name__ == "__main__":
    main()
