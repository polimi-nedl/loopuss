"""Worst-case error estimation example.

Calls :meth:`LoopUS.errbound` to obtain the worst-case relative transfer error
and the input vector that realises it, then verifies the bound by applying that
input and measuring the achieved error with :func:`loopuss.relerr`.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, CurrentSignal, LoopUS, relerr
from loopuss.plotting import corrplot


def main():
    rng = np.random.default_rng()

    N = 50
    G0 = 1.0 / 10e3
    V0 = 1.0
    I0 = G0 * V0

    # Random positive-definite matrix with non-negative entries.
    A = rng.random((N, N))
    A = A.T @ A + np.eye(N)
    A = A / A.max()

    Gin = CrossBar(np.empty((N, 0)))
    Gf = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf)

    # Worst-case input vector, applied as the circuit input.
    maxerr, y, _, _ = circ.errbound()
    circ.set_inputs(CurrentSignal.ytoi(y, I0=I0))

    time = np.concatenate([[0.0], np.logspace(np.log10(np.finfo(float).eps), -4, 50)])
    Vout, time = circ.tran("model", time)

    ref = -np.linalg.solve(A, y) * V0
    _, ax = plt.subplots()
    corrplot(Vout[:, -1], ref, ax=ax)
    ax.set_title("Circuit vs ideal solution (worst-case input)")

    print(f"Projected maximum relative error: {100 * maxerr:.2f}%")
    print(f"Effective relative error:         {100 * relerr(Vout[:, -1], ref):.2f}%")
    plt.show()


if __name__ == "__main__":
    main()
