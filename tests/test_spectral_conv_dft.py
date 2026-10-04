"""The real-valued DFT path of SpectralConv (used on Apple MPS) must match the
complex FFT path exactly, so models trained on either device are interchangeable."""

import copy

import pytest
import torch

from src.neuralop import FNO
from src.neuralop.conv import SpectralConv
from src.neuralop.losses import RelCombinedNormLoss

# (n_modes, spatial size): the 3-D presets on the 25x20x40 grid, odd/even
# sizes, modes exceeding the grid, and 2-D / 1-D layers.
CASES = [
    ((4, 4, 4), (25, 20, 40)),
    ((4, 6, 8), (25, 20, 40)),
    ((12, 10, 20), (25, 20, 40)),
    ((8, 6, 10), (9, 7, 10)),
    ((40, 40, 40), (9, 7, 10)),
    ((6, 8), (13, 16)),
    ((8,), (17,)),
]


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("fft_norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("n_modes,size", CASES)
def test_dft_path_matches_fft_path(n_modes, size, fft_norm):
    torch.manual_seed(0)
    conv = SpectralConv(3, 5, n_modes=n_modes, fft_norm=fft_norm).double()
    x = torch.randn(2, 3, *size, dtype=torch.float64)
    weight = conv.weight[0]._parameters["tensor"]

    x_fft = x.clone().requires_grad_()
    y_fft = conv(x_fft)
    grad_out = torch.randn_like(y_fft)
    (y_fft * grad_out).sum().backward()
    dw_fft, db_fft = weight.grad.clone(), conv.bias.grad.clone()
    conv.zero_grad()

    x_dft = x.clone().requires_grad_()
    y_dft = conv._forward_dft(x_dft)
    (y_dft * grad_out).sum().backward()

    # The FFT path computes in complex64, so agreement is at float32 precision.
    assert _rel(y_dft, y_fft) < 1e-6
    assert _rel(x_dft.grad, x_fft.grad) < 1e-6
    assert _rel(weight.grad, dw_fft) < 1e-6
    assert _rel(conv.bias.grad, db_fft) < 1e-6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_fno_on_mps_matches_cpu():
    torch.manual_seed(0)
    model = FNO(n_modes=(4, 6, 8), hidden_channels=8, in_channels=4, out_channels=2, n_layers=4)
    ref = copy.deepcopy(model).double()
    mps_model = copy.deepcopy(model).to("mps")
    x = torch.randn(4, 4, 25, 20, 40)
    y = torch.randn(4, 2, 25, 20, 40)
    criterion = RelCombinedNormLoss(dt=0.04)

    loss_ref = criterion(ref(x.double()), y.double())
    loss_ref.backward()
    loss_mps = criterion(mps_model(x.to("mps")), y.to("mps"))
    loss_mps.backward()

    assert abs(loss_mps.item() - loss_ref.item()) < 1e-5 * abs(loss_ref.item())
    for (name, p_ref), p_mps in zip(ref.named_parameters(), mps_model.parameters()):
        assert _rel(p_mps.grad.cpu().double(), p_ref.grad) < 1e-3, name
