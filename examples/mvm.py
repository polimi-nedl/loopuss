"""Matrix-vector multiplication circuit example.

Computes -A @ y with a positive-valued matrix using an in-memory feedback
circuit, and plots the transient.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, LoopUS, VoltageSignal
from loopuss.plotting import corrplot, matplot, tranplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3
    V0 = 1.0

    A = rng.random((N, N))
    A = A / A.max()

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)

    circ = LoopUS(Gin, Gf, verbose=True)

    y = -1 + 2 * rng.random(N)
    circ.set_inputs(VoltageSignal.ytov(y, V0=V0))

    Vout, time = circ.tran("model", np.logspace(-10, -5, 100))

    tranplot(time, Vout)
    plt.xlim(1e-10, 10e-6)
    plt.title("MVM transient")

    _, ax = plt.subplots()
    corrplot(Vout[:, -1], -A @ y * V0, ax=ax)
    ax.set_title("Circuit vs -A*y")

    _, ax = plt.subplots()
    matplot(Gin.G / 1e-6, ax=ax, c_label="Conductance [µS]")
    ax.set_title("Input crossbar")

    plt.show()


if __name__ == "__main__":
    main()
