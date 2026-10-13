"""Cached transfer matrices and poles follow parameter changes."""

import numpy as np
import pytest

from loopuss import CrossBar, CurrentSignal, LoopUS, OAModel

N, G0 = 3, 1e-4
A = np.array([[3.0, 1.0, 0.5], [1.0, 2.5, 0.2], [0.5, 0.2, 2.0]])


def _fresh(**kw):
    return LoopUS(None, CrossBar(A * G0), **kw)


def test_replacing_an_op_amp_updates_the_poles():
    c = _fresh()
    c.pz()
    slow = OAModel("1P", A0=1e4, GBWP=1e4)
    c.oa_models[0] = slow
    expected = _fresh(oa_model=[slow, OAModel(), OAModel()]).pz()[0]
    np.testing.assert_allclose(np.sort(c.pz()[0].real), np.sort(expected.real), rtol=1e-9)


def test_editing_an_op_amp_in_place_updates_the_poles():
    c = _fresh()
    p_before = c.pz()[0]
    c.oa_models[1].GBWP = 1e6
    c.oa_models[1].FP = np.array([1e6 / c.oa_models[1].A0])
    assert not np.allclose(np.sort(c.pz()[0].real), np.sort(p_before.real))


def test_changing_signs_updates_the_poles():
    c = _fresh()
    p_before = c.pz()[0]
    c.sign_vector[0] = 1.0
    assert not np.allclose(np.sort(c.pz()[0].real), np.sort(p_before.real))


def test_changing_orientation_updates_the_transfer():
    G = CrossBar(A * G0, line_resistance=50.0)
    c = LoopUS(None, G)
    t_before = [m.copy() for m in c._transfer_core()]
    c.xbar_orient = [["bottom", "right"], ["top", "right"]]
    t_after = c._transfer_core()
    assert any(not np.allclose(a, b) for a, b in zip(t_before, t_after, strict=True))
    ref = LoopUS(None, G, xbar_orient=[["bottom", "right"], ["top", "right"]])._transfer_core()
    for a, b in zip(t_after, ref, strict=True):
        np.testing.assert_allclose(a, b)


def test_tts_and_transient_follow_the_new_op_amps():
    c = _fresh()
    c.set_inputs(CurrentSignal.ytoi([1e-5] * N))
    tts_before = c.tts()[0]
    c.oa_models = [OAModel("1P", GBWP=1e6) for _ in range(N)]
    assert c.tts()[0] == pytest.approx(100 * tts_before, rel=1e-6)
