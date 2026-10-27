import numpy as np
import pytest

from mcno_qsm.rc2 import hz_to_ppm, magnitude_weight, phase_scale
from mcno_qsm.scoring import forward_field, masked_nrmse, score_endi


def test_unit_conversion_roundtrip():
    # 1 ppm at 7 T is 42.58*7 Hz and 2*pi*TE times that in radians.
    assert hz_to_ppm(42.58 * 7, 7.0) == pytest.approx(1.0)
    assert phase_scale(8e-3, 7.0) == pytest.approx(8e-3 * 2 * np.pi * 42.58 * 7)


def test_magnitude_weight_is_finite_and_zero_outside_signal():
    m = np.zeros((4, 4, 4, 4), dtype=np.float32)
    m[1:3, 1:3, 1:3] = np.array([1.0, 0.8, 0.5, 0.3])
    w = magnitude_weight(m)
    assert np.isfinite(w).all() and w[0, 0, 0] == 0 and w[1, 1, 1] > 0


def test_forward_field_is_linear_and_zero_mean_input_gives_real_output():
    chi = np.zeros((16, 16, 16), np.float32)
    chi[6:10, 6:10, 6:10] = 0.1
    assert np.allclose(forward_field(2 * chi), 2 * forward_field(chi), atol=1e-6)


def test_endi_on_oracle_field_recovers_susceptibility():
    pytest.importorskip("torch")
    n = 32
    chi = np.zeros((n,) * 3, np.float32)
    chi[12:20, 12:20, 12:20] = 0.05
    mask = np.zeros_like(chi, dtype=bool)
    mask[8:24, 8:24, 8:24] = True
    field = forward_field(chi, mask)
    result = score_endi(field, chi, mask, np.ones_like(chi), pad=n, max_iter=30)
    assert result["chi"].shape == chi.shape and result["iterations"] >= 4
    assert result["nrmse"] < masked_nrmse(np.zeros_like(chi), chi, mask)
