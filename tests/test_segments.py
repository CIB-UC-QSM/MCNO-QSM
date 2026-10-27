import torch

from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, image_components


def test_image_components_sum_to_the_field_and_are_real():
    torch.manual_seed(0)
    x = torch.randn(24, 28, 26, dtype=torch.float64)
    transform = DipoleletTransform(x.shape, (1.0, 1.0, 1.0), **SIM2_DIPOLELET, dtype=torch.float64)
    parts = image_components(transform, x)
    assert len(parts) == len(transform.segment_names) == 10
    assert torch.allclose(sum(parts), x, atol=1e-9)
    spectra = list(transform.analysis(x[None, None]))
    imag = max(float(torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").imag.abs().max()) for s in spectra)
    assert imag < 1e-9                      # the windows are symmetric in k, so every component is real
