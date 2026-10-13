"""Pole-zero analysis of a non-linearly operating LASSO circuit.

Sweeps every combination of linear/saturated operating regime for the LASSO
branch op-amps and reports the natural-ordered poles of each configuration via
:meth:`LoopUS.pz` and :func:`loopuss.analysis.natsortpoles`.
"""

import numpy as np

from loopuss import CrossBar, LoopUS, OAModel, lassoreg
from loopuss.analysis import natsortpoles


def main():
    rng = np.random.default_rng()

    M, N = 3, 2
    G0 = 1.0 / 10e3
    VLim = 0.1

    A = rng.random((M, N))
    t, lam = 1.0, 0.1
    L, S = lassoreg(t, lam, A)

    oa = [OAModel("1P") for _ in range(M + N)]
    oa += [OAModel("1P", Vsupply=[-VLim, VLim]) for _ in range(N)]

    Gin = CrossBar(np.empty((M + 2 * N, 0)))
    Gf = CrossBar(L * G0)
    circ = LoopUS(Gin, Gf, sign_vector=S, oa_model=oa)

    size = M + 2 * N
    poles = np.zeros((size, 2**N), dtype=complex)
    regimes = np.zeros((N, 2**N), dtype=bool)
    for col in range(2**N):
        # Bit k selects whether LASSO-branch op-amp k is linear (True).
        bits = np.array([not ((col >> (N - 1 - k)) & 1) for k in range(N)])
        regimes[:, col] = bits
        l_id = np.ones(size, dtype=bool)
        l_id[M + N : M + 2 * N] = bits
        p, *_ = circ.pz("hz", l_id)
        poles[:, col] = natsortpoles(p)

    np.set_printoptions(precision=3, suppress=True)
    print("Linear-regime mask (LASSO branch), one column per configuration:")
    print(regimes.astype(int))
    print("\nPoles [Hz]:")
    print(poles)


if __name__ == "__main__":
    main()
