"""Unit tests for loopuss.GPUModel."""

import pytest

from loopuss import BenchReport, GPUModel


def test_default_gpu():
    g = GPUModel()
    assert g.type == "NvidiaRTXA4500E_FP64"
    assert g.MemorySize == 16e9
    assert g.MemoryBandwidth == 512e9
    assert g.Power == 115.0
    assert g.Nbit == 64.0
    assert g.Area == 392e-6
    assert g.technology.node == 8e-9


def test_preset_loading():
    g = GPUModel("NvidiaRTXA1000E_FP64")
    assert g.type == "NvidiaRTXA1000E_FP64"
    assert g.MemorySize == 4e9
    assert g.MemoryBandwidth == 224e9
    assert g.Power == 35.0
    assert g.Throughput == 72.96e9
    assert g.Nbit == 64.0
    assert g.Ncores == 2048.0
    assert g.Fclk == 1140e6
    assert g.Area == 200e-6
    assert g.technology.node == 8e-9


def test_kwargs_override():
    g = GPUModel("NvidiaRTXA4500E_FP64", Power=150.0, MemorySize=32e9)
    assert g.Power == 150.0
    assert g.MemorySize == 32e9


def test_repr():
    g = GPUModel("NvidiaRTXA1000E_FP64")
    rep = repr(g)
    assert "NvidiaRTXA1000E_FP64" in rep
    assert "MemorySize" in rep
    assert "Power" in rep


def test_benchmark():
    g = GPUModel("NvidiaRTXA4500E_FP64")
    nops = 1e9
    ndata = 1e6
    psf = 1.0

    report = g.benchmark(nops=nops, ndata=ndata, psf=psf)
    assert isinstance(report, BenchReport)

    # Nbit = 64
    # MemoryBandwidth = 512e9
    # Throughput = 223.6e9
    # Power = 115.0
    # prob_size = 1e6 * 64 / 8 = 8e6 bytes
    # lat_prog = 8e6 / 512e9 = 15.625e-6 seconds
    # lat_comp = 1e9 / 223.6e9 = 4.4722719e-3 seconds
    # lat_total = 1.5625e-5 + 4.4722719e-3 = 4.4878969e-3 seconds
    assert report.latency["Programming"] == pytest.approx(1.5625e-5)
    assert report.latency["Computing"] == pytest.approx(4.4722719e-3, rel=1e-6)
    assert report.latency["Total"] == pytest.approx(4.4878969e-3, rel=1e-6)

    # energy_prog = lat_prog * psf * Power = 1.5625e-5 * 1.0 * 115.0 = 1.796875e-3 Joules
    # energy_comp = lat_comp * Power = 4.4722719e-3 * 115.0 = 0.51431127 Joules
    # energy_total = 1.796875e-3 + 0.51431127 = 0.5161081 Joules
    assert report.energy["Total"] == pytest.approx(0.5161081, rel=1e-6)
    assert report.energy["Programming"]["Dynamic"] == pytest.approx(1.796875e-3)
    assert report.energy["Computing"]["Dynamic"] == pytest.approx(0.51431127, rel=1e-6)

    # Power
    assert report.power[1] == 115.0
    expected_mean = 115.0 * (4.4722719e-3 + 1.5625e-5) / 4.4878969e-3
    assert report.power[0] == pytest.approx(expected_mean, rel=1e-6)

    # Area
    assert report.area["Total"] == 392e-6
    assert report.area["GPU"] == 392e-6

    # Efficiencies
    assert report.throughput == pytest.approx(nops / 4.4878969e-3, rel=1e-6)
    assert report.energy_efficiency == pytest.approx(nops / 0.5161081, rel=1e-6)
    assert report.area_efficiency == pytest.approx((nops / 4.4878969e-3) / 392e-6, rel=1e-6)
