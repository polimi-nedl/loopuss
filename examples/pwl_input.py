"""Piecewise-linear (PWL) input example.

Drives a LASSO circuit with time-varying current inputs defined as PWL
waveforms via :meth:`CurrentSignal.ytoi`, then runs the transient. Each input
channel switches between two random operating points partway through the run.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel, lassoreg
from loopuss.plotting import tranplot


def main():
    rng = np.random.default_rng()

    M, N = 10, 5
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0
    VLim = 0.1

    A = rng.random((M, N))
    T, R = 1.0, 0.1
    L, S = lassoreg(T, R, A)

    oa = [OAModel("1P", GBWP=1e6) for _ in range(M + N)]
    oa += [
        OAModel("1P", A0=1e4, GBWP=1e9 + 1e9 * rng.random(), Vsupply=[-VLim, VLim])
        for _ in range(N)
    ]

    Gin = CrossBar(np.empty((M + 2 * N, 0)))
    Gf = CrossBar(L * G0)
    circ = LoopUS(Gin, Gf, sign_vector=S, oa_model=oa)

    # PWL current sources on the first M inputs: zero, then two random levels.
    y = np.hstack(
        [
            np.zeros((M, 1)),
            np.tile(-1 + 2 * rng.random((M, 1)), (1, 100)),
            np.tile(-1 + 2 * rng.random((M, 1)), (1, 100)),
        ]
    )
    tv = np.concatenate([[0.0], np.linspace(1e-10, 5e-5, 200)])
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0, stype="PWL", tv=tv))

    time = np.concatenate([[0.0], np.logspace(np.log10(tv[1]), np.log10(tv[-1]), 1000)])
    Vout, time = circ.tran("model", time)

    tranplot(time, Vout)
    plt.xlim(tv[1], tv[-1])
    plt.title("PWL input - transient")

    _, ax = plt.subplots()
    ax.plot(tv, y.T)
    ax.set_title("PWL input waveforms")
    plt.show()


if __name__ == "__main__":
    main()
