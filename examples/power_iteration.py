"""In-memory power iteration.

Combines an in-memory MVM circuit with a DSP normalisation block to compute
the dominant eigenvector by repeated application.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, DSPModel, LoopUS, VoltageSignal
from loopuss.plotting import corrplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3
    I0 = G0
    V0 = I0 / G0

    A = rng.random((N, N))
    A = A / A.max()

    Gf = CrossBar(np.eye(N) * G0)
    Gin = CrossBar(A * G0)
    circ = LoopUS(Gin, Gf, verbose=True)

    dsp = DSPModel(DSPFunction=lambda x: x / np.max(np.abs(x)), complexity=N)

    circ.set_inputs(VoltageSignal.ytov(rng.random(N), V0=V0, stype="DC"))

    y = None
    for _ in range(100):
        y, _ = circ.op("model")
        y = dsp.dsp_function(y)
        circ.set_inputs(VoltageSignal.ytov(y, V0=V0, stype="DC"))

    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    v_ref = eigvecs[:, idx].real
    v_ref = v_ref / np.max(np.abs(v_ref))

    # Align sign (power iteration converges up to sign).
    if np.dot(y, v_ref) < 0:
        v_ref = -v_ref

    _, ax = plt.subplots()
    corrplot(y, v_ref, ax=ax)
    ax.set_title("Power iteration result vs analytical eigenvector")

    plt.show()


if __name__ == "__main__":
    main()
