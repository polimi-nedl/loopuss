"""Real-valued linear regression circuit example.

Uses pos/neg splitting with low-bandwidth compute OAs and high-bandwidth
buffer OAs.
Reference: Z. Sun, SciAdv (2020).
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, linreg, posneg
from loopuss.plotting import corrplot, matplot, tranplot


def main():
    rng = np.random.default_rng()

    M, N = 10, 4
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    A = -1 + 2 * rng.random((M, N))
    A = A / np.max(np.abs(A))
    A = np.hstack([np.ones((M, 1)), A])

    t = 1.0
    L, S = linreg(t, A)
    n_comp = L.shape[0]

    L, S, *_ = posneg(L, 1.0, sign_vector=S)
    n_buf = L.shape[0] - n_comp

    Gin = CrossBar(np.empty((L.shape[0], 0)))
    Gf = CrossBar(L * G0)

    OAs = [OAModel("1P", GBWP=1e6) for _ in range(n_comp)] + [
        OAModel("1P", GBWP=100e6) for _ in range(n_buf)
    ]

    circ = LoopUS(Gin, Gf, oa_model=OAs, sign_vector=S, output_mask=M + np.arange(N + 1))

    y = rng.random(M)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    Vout, time = circ.tran("model", np.logspace(-10, -4, 1000))

    tranplot(time, Vout)
    plt.xlim(1e-10, time[-1])
    plt.title("Linear regression (pos/neg) - transient")

    ref = -np.linalg.lstsq(A, y, rcond=None)[0]
    _, ax = plt.subplots()
    corrplot(Vout[:, -1], ref, ax=ax)
    ax.set_title(r"Circuit vs $-A \backslash y$")

    _, ax = plt.subplots()
    matplot(Gf.G / 1e-6, ax=ax, c_label="Conductance [µS]")
    ax.set_title("Feedback crossbar")

    plt.show()


if __name__ == "__main__":
    main()
