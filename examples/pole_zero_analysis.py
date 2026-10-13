"""Pole-zero analysis on a closed-loop in-memory circuit.

Three crossbar variants: symmetric PD, plus nonsymmetric perturbation, plus
symmetric non-PD perturbation - to see how poles split.
"""

import matplotlib.pyplot as plt
import numpy as np

from loopuss import CrossBar, LoopUS
from loopuss.plotting import reimplot


def main():
    rng = np.random.default_rng()

    N = 10
    G0 = 1.0 / 10e3

    # --- Case 1: symmetric PD ---
    A = rng.random((N, N))
    A = A.T @ A
    A = A / A.max()

    circ1 = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0))
    p1, *_ = circ1.pz()
    print("Case 1 (symmetric PD) poles [Hz]:")
    print(p1)
    _, ax = plt.subplots()
    reimplot(p1, ax=ax)
    ax.set_title("Symmetric PD - poles are real LHP")

    # --- Case 2: + nonsymmetric perturbation ---
    B = rng.random((N, N))
    B = 0.5 * (B - B.T)
    C = A + B
    C = C / C.max()
    circ2 = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(C * G0))
    p2, *_ = circ2.pz()
    print("Case 2 (nonsymmetric perturbation) poles [Hz]:")
    print(p2)
    _, ax = plt.subplots()
    reimplot(p2, ax=ax)
    ax.set_title("Nonsymmetric perturbation - pole splitting")

    # --- Case 3: + symmetric non-PD perturbation ---
    B = rng.random((N, N))
    B = 0.5 * (B + B.T)
    C = A + B
    C = C / C.max()
    circ3 = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(C * G0))
    p3, *_ = circ3.pz()
    print("Case 3 (symmetric non-PD) poles [Hz]:")
    print(p3)
    _, ax = plt.subplots()
    reimplot(p3, ax=ax)
    ax.set_title("Symmetric non-PD - poles real but not all LHP")

    plt.show()


if __name__ == "__main__":
    main()
