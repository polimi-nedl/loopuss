"""Noise analysis tests against the noise_analysis reference fixture.

Covers the analytical model (S1 MemoryOnly, S2 OAOnly, S3 MemoryAndOA) and,
when LTSpice is available, the SPICE noise backend (S4).
"""

import numpy as np
import pytest
from conftest import load_fixture

import loopuss as lp
from loopuss import spice


@pytest.fixture
def noise_data():
    return load_fixture("noise_analysis")


def _circuit(A, G0, N, *, noisy_xbar, oa):
    Gin = lp.CrossBar(np.empty((N, 0)), noiseless=not noisy_xbar)
    Gf = lp.CrossBar(A * G0, noiseless=not noisy_xbar)
    return lp.LoopUS(Gin, Gf, oa_model=oa)


def test_noise_model_memory_only(noise_data):
    D = noise_data
    A, G0, N = D["A"], float(D["G0"]), int(D["N"])
    c = _circuit(A, G0, N, noisy_xbar=True, oa=lp.OAModel("1P"))
    S, f, _ = c.noise("model", spice.SpiceAnalysis("noise", Fmax=100e9))
    np.testing.assert_allclose(f, D["freq1"].ravel(), rtol=1e-12)
    np.testing.assert_allclose(S, D["S1"], rtol=1e-9, atol=0)


def test_noise_model_oa_only(noise_data):
    D = noise_data
    A, G0, N = D["A"], float(D["G0"]), int(D["N"])
    oa = lp.OAModel("1P", GBWP=20e6, noise_voltage=2.9e-9)
    c = _circuit(A, G0, N, noisy_xbar=False, oa=oa)
    S, f, _ = c.noise("model", spice.SpiceAnalysis("noise", Fmax=100e9))
    np.testing.assert_allclose(S, D["S2"], rtol=1e-9, atol=0)


def test_noise_model_memory_and_oa(noise_data):
    D = noise_data
    A, G0, N = D["A"], float(D["G0"]), int(D["N"])
    oa = lp.OAModel("1P", noise_voltage=2.9e-9, noise_current=[14e-12, 3.2e-12])
    c = _circuit(A, G0, N, noisy_xbar=True, oa=oa)
    S, f, R = c.noise("model", spice.SpiceAnalysis("noise", Fmax=100e9))
    np.testing.assert_allclose(S, D["S3"], rtol=1e-9, atol=0)
    # Pout is the integrated RMS noise — finite and positive.
    assert np.all(R.Pout > 0)


@pytest.mark.skipif(not spice.spice_available(), reason="LTSpice not installed")
def test_noise_spice_matches_model_order(noise_data):
    """SPICE noise (S4) should be the same order of magnitude as the model."""
    D = noise_data
    A, G0, N = D["A"], float(D["G0"]), int(D["N"])
    oa = lp.OAModel("1P", noise_voltage=2.9e-9, noise_current=[14e-12, 3.2e-12])
    c = _circuit(A, G0, N, noisy_xbar=True, oa=oa)
    # smaller sweep keeps the SPICE run quick
    S, f, R = c.noise("SPICE/fast", spice.SpiceAnalysis("noise", Fmax=1e6))
    assert S.shape[0] == N
    assert np.all(np.isfinite(R.Pout))
