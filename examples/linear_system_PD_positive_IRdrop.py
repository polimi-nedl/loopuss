"""Positive-valued PD linear system solver with interconnect IR drop.

Reference: Z. Sun, PNAS (2019).

This example mirrors ``linear_system_PD_positive.py`` but additionally builds a
feedback crossbar with a finite line (interconnect) resistance, so the parasitic
IR drop across the array distorts the computed solution.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS
from loopuss.plotting import corrplot, matplot, tranplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    # Random positive-definite matrix with nonnegative entries.
    A = rng.random((N, N))
    A = A.T @ A + 10 * np.eye(N)
    A = A / A.max()

    # Crossbar objects: ideal feedback array and one with line resistance.
    Gin = CrossBar(np.empty((N, 0)))
    Gf = CrossBar(A * G0)
    GfIR = CrossBar(A * G0, line_resistance=1000)

    circ = LoopUS(Gin, Gf, verbose=True)
    circIR = LoopUS(Gin, GfIR, verbose=True)

    y = rng.random(N)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))
    circIR.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    Tstop = 100e-6
    time = np.concatenate([[0.0], np.logspace(-10, np.log10(Tstop), 1000)])
    Vout, time = circ.tran("model", time)
    VoutIR, timeIR = circIR.tran("model", time)
    VoutIR_spice, timeIR_spice = circIR.tran("SPICE/extreme", Tstop, dtmax=Tstop / 1e3)

    tranplot(time, Vout)
    plt.xlim(1e-10, Tstop)
    plt.title("PD linear system (positive) - ideal")

    ax = tranplot(timeIR, VoutIR)
    plt.xlim(1e-10, Tstop)
    plt.title("PD linear system (positive) - with IR drop")

    tranplot(timeIR_spice, VoutIR_spice, ax=ax, linestyle="dashed")

    ref = -np.linalg.solve(A, y)
    _, ax = plt.subplots()
    corrplot(Vout[:, -1], ref, ax=ax)
    ax.set_title(r"Ideal circuit vs $-A^{-1} y$")

    _, ax = plt.subplots()
    corrplot(VoutIR[:, -1], VoutIR_spice[:, -1], ax=ax)
    ax.set_title(r"IR-drop circuit vs SPICE co-simulation")

    _, ax = plt.subplots()
    matplot(Gf.G / 1e-6, ax=ax, c_label="Conductance [µS]")
    ax.set_title("Feedback crossbar")

    plt.show()


if __name__ == "__main__":
    main()
