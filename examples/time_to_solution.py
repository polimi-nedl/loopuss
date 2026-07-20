"""Time-to-solution estimation example.

Uses :meth:`LoopUS.tts` to estimate how long the closed-loop circuit needs to
settle to its target error, then overlays that estimate on the transient.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS
from loopuss.plotting import tranplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    # Random positive-definite matrix with non-negative entries.
    A = rng.random((N, N))
    A = A.T @ A
    A = A / A.max()

    Gin = CrossBar(np.empty((N, 0)))
    Gf = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, verbose=True)
    circ.set_inputs(CurrentSignal.ytoi(rng.random(N), I0=I0))

    tts, tts_min, tts_max = circ.tts()
    print(f"Time to solution: {tts:.3e} s  (min {tts_min:.3e}, max {tts_max:.3e})")

    time = np.logspace(-15, np.log10(10 * tts), 1000)
    Vout, time = circ.tran("model", time)

    tranplot(time, Vout)
    plt.xlim(1e-10, 10 * tts)
    for tline in (tts, tts_min, tts_max):
        plt.axvline(tline, ls="--", color="k", lw=0.8)
    plt.title("Time to solution")
    plt.show()


if __name__ == "__main__":
    main()
