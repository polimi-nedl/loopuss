"""LASSO regression circuit example.

Maps a LASSO problem onto the block-conductance circuit with :func:`loopuss.lassoreg`
and runs the transient. The LASSO branch op-amps are supply-limited, which
introduces the soft-thresholding non-linearity that yields the sparse solution.

Reference: P. Mannocci, IEEE ISCAS (2025).
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

    # LASSO parameters.
    t, lam = 1.0, 0.1
    L, S = lassoreg(t, lam, A)

    # First M+N op-amps are ideal transconductors; the last N form the
    # supply-limited LASSO branch.
    oa = [OAModel("GM") for _ in range(M + N)]
    oa += [OAModel("1P", Vsupply=[-VLim, VLim]) for _ in range(N)]

    Gin = CrossBar(np.empty((M + 2 * N, 0)))
    Gf = CrossBar(L * G0)
    circ = LoopUS(Gin, Gf, sign_vector=S, oa_model=oa, verbose=True)

    # Current sources drive the first M inputs only.
    y = -1 + 2 * rng.random(M)
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    time = np.concatenate([[0.0], np.logspace(-10, -3, 1000)])
    Vmodel, time = circ.tran("model", time)

    o_mask = M + np.arange(N)
    tranplot(time, Vmodel)
    plt.xlim(1e-10, time[-1])
    plt.title("LASSO regression - transient")
    print("LASSO solution (circuit):")
    print(Vmodel[o_mask, -1] / V0)
    plt.show()


if __name__ == "__main__":
    main()
