"""Noise analysis example.

Instantiates three noisy LoopUS circuits - noisy CrossBars only, a noisy
OAModel only, or both - and runs
a ``.noise`` analysis to obtain the output spectral noise density at each swept
frequency. PSDs are overlaid with ``psdplot``.

The MemoryAndOA circuit is solved twice: with the analytical ``model`` solver
(green) and, when LTSpice is available, with the ``SPICE/fast`` backend (grey).
The two land on top of each other, which is the point of overlaying them.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from loopuss import CrossBar, LoopUS, OAModel
from loopuss.plotting import psdplot
from loopuss.spice import SpiceAnalysis, spice_available


def main():
    rng = np.random.default_rng(0)

    # Size of data matrix and G0 conductance.
    N = 10
    G0 = 1.0 / 5e3

    # Random positive-definite matrix with nonnegative entries.
    A = rng.random((N, N))
    A = A.T @ A + np.eye(N)
    A = A / A.max()

    # Crossbar objects (noiseless=False makes them contribute thermal noise).
    Gin = CrossBar(np.empty((N, 0)))
    Gin_noisy = CrossBar(np.empty((N, 0)), noiseless=False)
    Gf = CrossBar(A * G0)
    Gf_noisy = CrossBar(A * G0, noiseless=False)

    # LoopUS objects: OA-noise only, memory-noise only, and both.
    OAOnly = LoopUS(Gin, Gf, oa_model=OAModel("1P", GBWP=20e6, noise_voltage=2.9e-9), verbose=True)
    MemoryOnly = LoopUS(Gin_noisy, Gf_noisy, verbose=True)
    MemoryAndOA = LoopUS(
        Gin_noisy,
        Gf_noisy,
        oa_model=OAModel("1P", noise_voltage=2.9e-9, noise_current=[14e-12, 3.2e-12]),
        verbose=True,
    )

    # .noise sweep up to 100 GHz.
    spAn = SpiceAnalysis("noise", Fmax=100e9)

    # Run the analyses.
    S1, freq1, R1 = MemoryOnly.noise("model", spAn)
    S2, freq2, R2 = OAOnly.noise("model", spAn)
    S3, freq3, R3 = MemoryAndOA.noise("model", spAn)

    have_spice = spice_available()
    if have_spice:
        S4, freq4, R4 = MemoryAndOA.noise("SPICE/fast", spAn)
    else:
        print("LTSpice not found - skipping the SPICE/fast PSD (S4).")

    # Overlay the PSD families, each in its own colormap (cf. brewermap).
    _, ax = plt.subplots(figsize=(9, 7))
    psdplot(freq1, S1, ax=ax, cmap="Blues")
    psdplot(freq2, S2, ax=ax, cmap="Reds")
    psdplot(freq3, S3, ax=ax, cmap="Greens")
    if have_spice:
        psdplot(freq4, S4, ax=ax, cmap="Greys")

    # One legend entry per family rather than 40 per-output lines.
    handles = [
        Line2D([], [], color=plt.get_cmap("Blues")(0.7), lw=2, label="MemoryOnly (model)"),
        Line2D([], [], color=plt.get_cmap("Reds")(0.7), lw=2, label="OAOnly (model)"),
        Line2D([], [], color=plt.get_cmap("Greens")(0.7), lw=2, label="MemoryAndOA (model)"),
    ]
    if have_spice:
        handles.append(
            Line2D([], [], color=plt.get_cmap("Greys")(0.7), lw=2, label="MemoryAndOA (SPICE/fast)")
        )
    ax.legend(handles=handles, loc="best")
    ax.set_title("Output noise PSD")
    plt.gcf().tight_layout()

    # RMS noise power (integral of S over frequency) per case.
    print(f"Memory only:  RMS(vout) = {np.linalg.norm(R1.Pout) / 1e-3:.3g} mV")
    print(f"OA only:      RMS(vout) = {np.linalg.norm(R2.Pout) / 1e-3:.3g} mV")
    print(f"MemoryAndOA:  RMS(vout) = {np.linalg.norm(R3.Pout) / 1e-3:.3g} mV")
    if have_spice:
        print(f"MemoryAndOA:  RMS(vout) = {np.linalg.norm(R4.Pout) / 1e-3:.3g} mV (SPICE)")

    plt.show()


if __name__ == "__main__":
    main()
