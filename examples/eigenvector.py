"""Eigenvector circuit example.

Computes the leading eigenvector of a positive-definite matrix.
Reference: Z. Sun, PNAS (2019).
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel
from loopuss.plotting import tranplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    A = rng.random((N, N))
    A = A.T @ A
    A = A / A.max()

    eigvals, eigvecs = np.linalg.eig(A)
    idx0 = np.argsort(eigvals.real)[0]
    lam = float(eigvals[idx0].real)

    d = 1e-3
    M = np.block([[A, (lam + d) * np.eye(N)], [np.eye(N), np.eye(N)]])

    Gin = CrossBar(np.empty((M.shape[0], 0)))
    Gf = CrossBar(M * G0)

    y0 = -1 + 2 * rng.random(M.shape[0])
    oa = OAModel("1P", Vsupply=[-1.0, 1.0])

    circ = LoopUS(Gin, Gf, oa_model=oa, initial_condition=y0)

    # PWL current pulse on all inputs, identical waveform per channel.
    yvec = -1 + 2 * rng.random(M.shape[0])
    Iv = (
        np.column_stack(
            [np.zeros(M.shape[0]), np.zeros(M.shape[0]), yvec, yvec, np.zeros(M.shape[0])]
        )
        * I0
    )
    tv = np.array([0.0, 0.99e-9, 1e-9, 4.99e-9, 5e-9])
    sigs = [CurrentSignal("PWL", ival=Iv[i, :], time=tv) for i in range(M.shape[0])]
    circ.set_inputs(sigs)

    Tstop = 10e-3
    time = np.concatenate([[0.0], np.logspace(-10, np.log10(Tstop), 1000)])
    Vmodel, tmodel = circ.tran("model", time)

    tranplot(tmodel, Vmodel)
    plt.xlim(1e-10, Tstop)
    plt.title("Eigenvector circuit - transient")

    plt.show()


if __name__ == "__main__":
    main()
