"""Steady-state analysis example.

Uses op('model') to find the operating point.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, LoopUS, VoltageSignal
from loopuss.plotting import corrplot, matplot


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

    y = rng.random(N)
    circ.set_inputs(VoltageSignal.ytov(y, V0=V0, stype="DC"))

    Vout, _ = circ.op("model")

    _, ax = plt.subplots()
    corrplot(Vout, -A @ y * V0, ax=ax)
    ax.set_title("Steady-state: circuit vs -A*y")

    _, ax = plt.subplots()
    matplot(Gin.G / 1e-6, ax=ax, c_label="Conductance [µS]")
    ax.set_title("Input crossbar")

    plt.show()


if __name__ == "__main__":
    main()
