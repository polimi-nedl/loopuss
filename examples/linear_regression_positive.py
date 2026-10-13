"""Positive-valued linear regression circuit example.

Reference: Z. Sun, SciAdv (2020).
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, linreg
from loopuss.plotting import corrplot, matplot, tranplot


def main():
    rng = np.random.default_rng()

    M, N = 10, 4
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    A = rng.random((M, N))
    A = A / A.max()
    A = np.hstack([np.ones((M, 1)), A])  # bias column

    t = 1.0
    L, S = linreg(t, A)

    Gin = CrossBar(np.empty((L.shape[0], 0)))
    Gf = CrossBar(L * G0)

    circ = LoopUS(
        Gin,
        Gf,
        oa_model=OAModel("1P"),
        sign_vector=S,
        output_mask=M + np.arange(N + 1),
    )

    y = rng.random(M)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    time = np.concatenate([[0.0], np.logspace(-10, -4, 1000)])
    Vout, time = circ.tran("model", time)

    tranplot(time, Vout)
    plt.xlim(1e-10, time[-1])
    plt.title("Linear regression (positive) - transient")

    ref = -np.linalg.pinv(A) @ y
    _, ax = plt.subplots()
    corrplot(Vout[:, -1], ref, ax=ax)
    ax.set_title("Circuit vs -pinv(A)*y")

    _, ax = plt.subplots()
    matplot(Gf.G / 1e-6, ax=ax, c_label="Conductance [µS]")
    ax.set_title("Feedback crossbar")

    plt.show()


if __name__ == "__main__":
    main()
