"""Smoke tests for loopuss.plotting (non-interactive Agg backend)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from loopuss.plotting import corrplot, matplot, reimplot, tranplot


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_tranplot():
    t = np.logspace(-9, -5, 50)
    V = np.vstack([np.sin(t * 1e6), np.cos(t * 1e6)])
    ax = tranplot(t, V)
    assert ax.get_xlabel() == "Time [s]"
    assert len(ax.lines) == 2


def test_corrplot():
    ax = corrplot(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.05]))
    assert ax.get_aspect() == 1.0  # equal aspect


def test_matplot():
    ax = matplot(np.arange(12).reshape(3, 4), c_label="G")
    assert ax.images  # one imshow


def test_reimplot():
    poles = np.array([-1 + 2j, -1 - 2j, -3 + 0j, np.inf])
    ax = reimplot(poles)
    assert ax.get_xlabel() == "Re{p}"
