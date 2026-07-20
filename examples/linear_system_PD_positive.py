"""Positive-valued PD linear system solver.

Reference: Z. Sun, PNAS (2019).
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

    A = rng.random((N, N))
    A = A.T @ A + 10 * np.eye(N)
    A = A / A.max()

    Gin = CrossBar(np.empty((N, 0)))
    Gf = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, verbose=True)

    y = rng.random(N)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    Tstop = 100e-6
    time = np.concatenate([[0.0], np.logspace(-10, np.log10(Tstop), 1000)])
    Vout, time = circ.tran("model", time)

    tranplot(time, Vout)
    plt.xlim(1e-10, Tstop)
    plt.title("PD linear system (positive) - transient")

    ref = -np.linalg.solve(A, y)
    _, ax = plt.subplots()
    corrplot(Vout[:, -1], ref, ax=ax)
    ax.set_title(r"Circuit vs $-A^{-1} y$")

    _, ax = plt.subplots()
    matplot(Gf.G / 1e-6, ax=ax, c_label="Conductance [µS]")
    ax.set_title("Feedback crossbar")

    plt.show()


if __name__ == "__main__":
    main()
