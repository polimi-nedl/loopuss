"""Unit tests for loopuss.reports (BenchReport / TranReport / NoiseReport)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from loopuss.reports import (
    BenchReport,
    NoiseReport,
    TranReport,
    _safe_div,
    mean_reports,
    sum_reports,
)


def _mk(scale=1.0):
    return BenchReport(
        latency={"Programming": 1.0, "Computing": 2.0, "PostProcessing": 0.5, "Total": 3.5},
        energy={
            "Breakdown": np.ones((7, 6)) * scale,
            "Labels": ["MemWrite"] * 7,
            "Total": float(np.sum(np.ones((7, 6)) * scale)),
        },
        power=np.array([2.0, 5.0]),
        area={"DAC": 1.0, "MemWrite": 1.0, "Total": 2.0},
        complexity=10.0,
        ops=100.0,
        throughput=0.0,
        energy_efficiency=0.0,
        area_efficiency=0.0,
    )


def test_safe_div():
    assert _safe_div(10, 2) == 5.0
    assert np.isinf(_safe_div(1, 0))


def test_combine_serial_adds_compute_latency():
    A, B = _mk(), _mk()
    C = BenchReport.combine(A, B, "serial", noarea=True, noprog=True)
    assert C.latency["Computing"] == 4.0  # 2 + 2
    assert C.latency["Programming"] == 1.0  # noprog keeps first
    assert C.ops == 200.0


def test_combine_parallel_takes_max_latency():
    A = _mk()
    B = _mk()
    B.latency = dict(B.latency)
    B.latency["Computing"] = 10.0
    C = BenchReport.combine(A, B, "parallel", noarea=True, noprog=True)
    assert C.latency["Computing"] == 10.0


def test_combine_noprog_zeros_memwrite_row():
    A, B = _mk(), _mk()
    C = BenchReport.combine(A, B, "serial", noarea=True, noprog=True)
    # MemWrite (row 0) of B is masked out -> equals A's row only
    np.testing.assert_array_equal(C.energy["Breakdown"][0], np.ones(6))


def test_add_operator_is_serial():
    C = _mk() + _mk()
    assert C.ops == 200.0


def test_scale_parallel_multiplies():
    A = _mk()
    C = A * 3
    assert C.ops == 300.0
    assert C.complexity == 30.0


def test_sum_reports():
    S = sum_reports([_mk(), _mk(), _mk()])
    assert S.ops == 300.0


def test_mean_reports():
    M = mean_reports([_mk(1.0), _mk(3.0)])
    # energy breakdown averaged
    assert M.energy["Breakdown"][1, 1] == pytest.approx(2.0)
    assert M.ops == 100.0
    assert M.circuit == "averaged"


def test_tran_report_power_properties():
    R = TranReport(power={"Total": np.array([1.0, 3.0, 2.0])})
    assert R.peak_power == 3.0
    assert R.mean_power == pytest.approx(2.0)


def test_noise_report_power_properties():
    R = NoiseReport(Pout=np.array([1.0, 4.0, 1.0]))
    assert R.peak_power == 4.0
    assert R.mean_power == 2.0


# --- plotting (barplot / bdplot) ---
def _mk_perf(tp=1e9, ee=1e12, ae=1e18):
    """A BenchReport with realistic, plottable efficiency/breakdown data."""
    B = _mk()
    B.throughput, B.energy_efficiency, B.area_efficiency = tp, ee, ae
    B.energy = {
        "Breakdown": np.eye(7, 6) + 0.1,
        "Labels": ["MemWrite", "DAC", "InputArray", "FeedbackArray", "OA", "ADC", "DSP"],
        "Total": float(np.sum(np.eye(7, 6) + 0.1)),
    }
    B.area = {
        "MemWrite": 1.0,
        "DAC": 2.0,
        "InputArray": 0.0,
        "FeedbackArray": 3.0,
        "OA": 4.0,
        "ADC": 5.0,
        "DSP": 0.0,
        "Total": 15.0,
    }
    return B


def test_barplot_single_and_multi():
    ax = _mk_perf().barplot()
    assert ax.get_yscale() == "log"
    assert len(ax.patches) == 3  # 3 metric bars
    # array form: 2 reports x 3 metrics
    ax2 = BenchReport.barplot([_mk_perf(), _mk_perf(tp=1e6)], "-notext", labels=["A", "B"])
    assert len(ax2.patches) == 6
    plt.close("all")


def test_barplot_metric_selection():
    ax = _mk_perf().barplot("-energy")  # single metric
    assert len(ax.patches) == 1
    plt.close("all")


def test_bdplot_grid_and_noprog():
    fig = BenchReport.bdplot([_mk_perf(), _mk_perf()], labels=["A", "B"])
    assert len(fig.axes) == 6  # 2 reports x 3 metrics
    fig2 = _mk_perf().bdplot("-area", "-noprog")
    assert len(fig2.axes) == 1
    plt.close("all")


def test_bdplot_orientation_conflict():
    with pytest.raises(ValueError):
        _mk_perf().bdplot("-vert", "-horiz")
    plt.close("all")
