"""Plot helpers: tranplot / corrplot / matplot / reimplot / psdplot.

Every function draws on ``ax`` (a new figure when omitted) and returns it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def tranplot(time, V, ax=None, **kwargs):
    """Plot transient voltages on a semilog-x axis.

    Parameters
    ----------
    time : array_like, shape (T,)
        Time points [s].
    V : array_like, shape (n, T)
        One voltage trace per row [V].
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.semilogx`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    V = np.atleast_2d(V)
    for row in V:
        ax.semilogx(time, row, **kwargs)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def corrplot(y_circ, y_ref, ax=None):
    """Scatter the circuit output against a reference, with the ``y = x`` line.

    Parameters
    ----------
    y_circ : array_like
        Circuit output, plotted on the y axis.
    y_ref : array_like
        Reference solution, plotted on the x axis.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    y_circ = np.asarray(y_circ).ravel()
    y_ref = np.asarray(y_ref).ravel()
    lim = float(max(np.max(np.abs(y_circ)), np.max(np.abs(y_ref))) * 1.05)
    ax.plot([-lim, lim], [-lim, lim], "-k", linewidth=0.8)
    ax.plot(y_ref, y_circ, "o", markerfacecolor="C0", markeredgecolor="k")
    ax.set_xlabel("Reference")
    ax.set_ylabel("Circuit output")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def matplot(M, ax=None, c_label="", cmap="viridis"):
    """Heatmap of a matrix with a colorbar.

    Parameters
    ----------
    M : array_like, shape (rows, cols)
        Matrix, e.g. ``CrossBar.G``.
    ax : matplotlib.axes.Axes, optional
    c_label : str, optional
        Colorbar label.
    cmap : str, optional
        Matplotlib colormap name. Default ``"viridis"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(np.asarray(M), aspect="auto", cmap=cmap, interpolation="nearest")
    cb = ax.figure.colorbar(im, ax=ax)
    if c_label:
        cb.set_label(c_label)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return ax


def reimplot(poles, ax=None):
    """Plot complex poles on the s-plane; infinite poles are skipped.

    Parameters
    ----------
    poles : array_like
        Poles, e.g. from :meth:`LoopUS.pz <loopuss.LoopUS.pz>`.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    poles = np.asarray(poles).ravel()
    finite = poles[np.isfinite(poles)]
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.plot(finite.real, finite.imag, "x", markersize=8, markeredgewidth=1.5, color="C3")
    ax.set_xlabel("Re{p}")
    ax.set_ylabel("Im{p}")
    ax.grid(True, alpha=0.3)
    return ax


def psdplot(freq, Sout, ax=None, color_order=None, cmap="Spectral", **kwargs):
    """Plot output noise power spectral densities.

    One line per output (row of ``Sout``) on a log-log axis. Lines are coloured
    from ``color_order`` (an N-by-3/4 RGB(A) array) when given, otherwise sampled
    from the named matplotlib colormap ``cmap`` (default 'Spectral'). Pass an
    existing ``ax`` to overlay several PSD families on one
    figure, the way the noise_analysis example layers MemoryOnly/OAOnly/etc.

    Each line is labelled ``V_{out,i}`` so the caller can ``ax.legend()``.

    Parameters
    ----------
    freq : array_like, shape (F,)
        Frequencies [Hz].
    Sout : array_like, shape (n, F)
        Noise densities [V/√Hz], e.g. from :meth:`LoopUS.noise
        <loopuss.LoopUS.noise>`.
    ax : matplotlib.axes.Axes, optional
    color_order : array_like, optional
    cmap : str, optional
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.loglog`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    Sout = np.atleast_2d(np.asarray(Sout, dtype=float))
    freq = np.asarray(freq, dtype=float).ravel()
    n = Sout.shape[0]
    if color_order is None:
        color_order = plt.get_cmap(cmap)(np.linspace(0.15, 0.95, n))
    color_order = np.asarray(color_order)
    for i in range(n):
        ax.loglog(
            freq,
            Sout[i],
            color=color_order[i % len(color_order)],
            linewidth=2,
            label=rf"$V_{{out,{i + 1}}}$",
            **kwargs,
        )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Output PSD [V/$\sqrt{\mathrm{Hz}}$]")
    ax.grid(True, which="both", alpha=0.3)
    return ax
