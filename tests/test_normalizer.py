"""Unit tests for the Normalizer class."""

import tempfile
from pathlib import Path
import torch
import pytest

from src.data.normalizer import Normalizer


def test_normalizer_3d_fno_single_sample():
    """Test normalization on unbatched 3-D FNO tensor (C, T, Z, X)."""
    C, T, Z, X = 4, 25, 20, 40
    input_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    input_std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    output_mean = torch.tensor([10.0, 20.0])
    output_std = torch.tensor([2.0, 5.0])

    norm = Normalizer(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )

    # Input: (C, T, Z, X)
    x = torch.randn(C, T, Z, X)
    x_norm = norm.normalize_input(x)
    assert x_norm.shape == (C, T, Z, X)

    x_rec = norm.denormalize_input(x_norm)
    assert torch.allclose(x, x_rec, atol=1e-5)

    # Target: (C_out, T, Z, X)
    y = torch.randn(2, T, Z, X)
    y_norm = norm.normalize_output(y)
    assert y_norm.shape == (2, T, Z, X)

    y_rec = norm.denormalize_output(y_norm)
    assert torch.allclose(y, y_rec, atol=1e-5)


def test_normalizer_3d_fno_batched():
    """Test normalization on batched 3-D FNO tensor (B, C, T, Z, X)."""
    B, C, T, Z, X = 8, 4, 25, 20, 40
    input_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    input_std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    output_mean = torch.tensor([10.0, 20.0])
    output_std = torch.tensor([2.0, 5.0])

    norm = Normalizer(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )

    xb = torch.randn(B, C, T, Z, X)
    xb_norm = norm.normalize_input(xb)
    assert xb_norm.shape == (B, C, T, Z, X)

    xb_rec = norm.denormalize_input(xb_norm)
    assert torch.allclose(xb, xb_rec, atol=1e-5)

    yb = torch.randn(B, 2, T, Z, X)
    yb_norm = norm.normalize_output(yb)
    assert yb_norm.shape == (B, 2, T, Z, X)

    yb_rec = norm.denormalize_output(yb_norm)
    assert torch.allclose(yb, yb_rec, atol=1e-5)


def test_normalizer_2d_fno_single_and_batched():
    """Test normalization on 2-D FNO layouts (C, H, W) and (B, C, H, W)."""
    C, H, W = 4, 20, 40
    B = 8
    input_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    input_std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    output_mean = torch.tensor([10.0])
    output_std = torch.tensor([2.0])

    norm = Normalizer(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )

    # 3-D unbatched (C, H, W)
    x_single = torch.randn(C, H, W)
    x_single_norm = norm.normalize_input(x_single)
    assert x_single_norm.shape == (C, H, W)
    assert torch.allclose(x_single, norm.denormalize_input(x_single_norm), atol=1e-5)

    # 4-D batched (B, C, H, W) where B != C
    x_batched = torch.randn(B, C, H, W)
    x_batched_norm = norm.normalize_input(x_batched)
    assert x_batched_norm.shape == (B, C, H, W)
    assert torch.allclose(x_batched, norm.denormalize_input(x_batched_norm), atol=1e-5)


def test_normalizer_serialization():
    """Test save and load round-trip."""
    input_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    input_std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    output_mean = torch.tensor([10.0, 20.0])
    output_std = torch.tensor([2.0, 5.0])

    norm = Normalizer(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "normalizer.npz"
        norm.save(path)
        loaded = Normalizer.load(path)

        assert torch.allclose(norm.input_mean, loaded.input_mean)
        assert torch.allclose(norm.input_std, loaded.input_std)
        assert torch.allclose(norm.output_mean, loaded.output_mean)
        assert torch.allclose(norm.output_std, loaded.output_std)

