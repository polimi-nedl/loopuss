"""Real-valued PD linear system solver.

Reference: Z. Sun, PNAS (2019).
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, posneg
from loopuss.plotting import corrplot, matplot, tranplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    A = -1 + 2 * rng.random((N, N))
    A = A.T @ A + np.eye(N)
    A = A / A.max()

    b = 1.0
    M, *_ = posneg(A, b)

    Gin = CrossBar(np.empty((M.shape[0], 0)))
    Gf = CrossBar(M * G0)

    circ = LoopUS(Gin, Gf, output_mask=np.arange(N))

    y = rng.random(N)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    Vout, time = circ.tran("model", np.logspace(-10, -4, 1000))

    tranplot(time, Vout)
    plt.xlim(1e-10, time[-1])
    plt.title("PD linear system (pos/neg) - transient")

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
