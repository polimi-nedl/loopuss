"""Output-noise analysis.

`noise_model` computes an analytical output-noise PSD; `noise_spice` runs an
LTSpice `.noise` analysis. `LoopUS.noise` dispatches between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from . import spice as _spice

if TYPE_CHECKING:
    from .loopus import LoopUS


def noise_model(circ: LoopUS, freq: np.ndarray):
    """Analytical output noise PSD.

    Sums, through the circuit transfer, the thermal noise of the input and
    feedback arrays at 300 K (each unless it is ``noiseless``), the current
    noise of the op-amp inputs wired to the arrays, and the op-amp voltage
    noise. The white spectrum then rolls off as a single pole at the harmonic
    mean of the circuit's pole magnitudes.

    Parameters
    ----------
    circ : LoopUS
    freq : ndarray, shape (F,)
        Frequencies [Hz].

    Returns
    -------
    Sout : ndarray, shape (len(circ.output_mask), F)
        Output noise density [V/√Hz].
    freq : ndarray
        The input frequencies.
    """
    from scipy.constants import Boltzmann

    T = circ._transfer_core()
    T1 = scipy.linalg.inv(circ.transfer("inverting"))
    T2 = T[1] + T[3]

    # feedback crossbar thermal noise
    if not circ.feedback_array.noiseless:
        SiGf = 4 * Boltzmann * 300 * np.sum(np.abs(circ.feedback_array.G), axis=1)
    else:
        SiGf = 0.0

    # input crossbar thermal noise
    if not circ.input_array.noiseless:
        SiGin = 4 * Boltzmann * 300 * np.sum(np.abs(circ.input_array.G), axis=1)
    else:
        SiGin = 0.0

    # OA input current noise (masked by sign), then squared sum
    SiOA = np.array([np.asarray(m.noise_current, float).ravel() for m in circ.oa_models])  # (n, 2)
    SiOA[circ.sign_vector > 0, 0] = 0.0
    SiOA[circ.sign_vector < 0, 1] = 0.0
    SiOA = np.sum(SiOA, axis=1) ** 2
    Sin = SiOA + SiGf + SiGin

    # OA input voltage noise
    Svn = np.array([float(m.noise_voltage) for m in circ.oa_models]) ** 2

    Sout = (T1**2) @ (Sin + (T2**2) @ Svn)
    Sout = np.sqrt(Sout)  # (n,)

    # heuristic bandwidth roll-off. p is the harmonic
    # mean of the pole magnitudes; the single-pole roll-off is |1 + i f/p|.
    poles = circ.pz()[0]
    p = poles.size / np.sum(1.0 / np.abs(poles))
    Sout = Sout[:, np.newaxis] / np.abs(1 + 1j * freq / p)

    return Sout[circ.output_mask, :], freq


def noise_spice(circ: LoopUS, solver_type: str, sp_analysis: _spice.SpiceAnalysis):
    """SPICE-based output noise PSD."""
    net, raw = circ.netlist(solver_type, sp_analysis)
    # One netlist per output, all independent -> run them across workers.
    # Only V(onoise) is read back below, so parse just that column: the
    # .raw also holds one column per device (N^2 of them), which would
    # otherwise dominate both parse time and per-worker memory.
    results = _spice.run_spice(net, raw, keep_vars=["V(onoise)"], max_workers=circ.num_workers)
    if isinstance(results, dict):
        results = [results]
    freq = np.asarray(results[0]["freq_vect"], dtype=float).ravel()
    Sout = np.vstack([_spice._spice_node_matrix(r, ["V(onoise)"])[0, :] for r in results])
    return Sout, freq
