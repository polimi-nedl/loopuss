"""Pole/singularity analysis helpers."""

from __future__ import annotations

import numpy as np


def natsortpoles(p: np.ndarray) -> np.ndarray:
    """Natural ordering of a vector of complex singularities.

    Real singularities are placed at the extremes of the returned vector in
    ascending order; complex-conjugate singularities are gathered in the
    middle, with the magnitude of the imaginary part growing towards the
    centre so that pole splitting is highlighted.
    """
    p = np.asarray(p).ravel()

    # Complex-conjugate singularities, ordered so |Im| grows towards the centre.
    ccp = p[p.imag != 0]
    order = np.argsort(ccp.imag, kind="stable")
    half = order.size // 2
    order = np.concatenate([order[:half][::-1], order[half:][::-1]])
    ccp = ccp[order]

    # Real singularities, ascending.
    rp = np.sort(p[p.imag == 0])
    nr = rp.size

    if nr > 1:
        k = -(-nr // 2)  # ceil(nr / 2)
        return np.concatenate([rp[:k], ccp, rp[k:]])
    return np.concatenate([rp, ccp])
