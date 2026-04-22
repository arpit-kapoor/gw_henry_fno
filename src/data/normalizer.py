"""Normalization utilities for Henry scenario data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Computes and applies mean/std normalization.

    Computes statistics on a source dataset (typically train) and applies
    the same normalization to multiple datasets (train and val).

    Parameters
    ----------
    input_mean : torch.Tensor
        Mean of input tensors, shape (C_in,).
    input_std : torch.Tensor
        Std of input tensors, shape (C_in,).
    output_mean : torch.Tensor
        Mean of output tensors, shape (C_out,).
    output_std : torch.Tensor
        Std of output tensors, shape (C_out,).
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-8.
    """

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> None:
        """Store normalization statistics used for input/output scaling."""
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.epsilon = epsilon

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        compute_output_stats: bool = True,
        epsilon: float = 1e-8,
    ) -> Normalizer:
        """Compute normalizer statistics from a dataset.

        Iterates through all samples in the dataset and computes per-channel
        mean and standard deviation.

        Parameters
        ----------
        dataset : Dataset
            Dataset to compute statistics from (typically train split).
            Expected to return (input_tensor, output_tensor) tuples.
        compute_output_stats : bool, optional
            If True, compute stats for both inputs and outputs.
            If False, only compute input stats, by default True.
        epsilon : float, optional
            Small value to avoid division by zero, by default 1e-8.

        Returns
        -------
        Normalizer
            Normalizer instance with computed statistics.
        """
        input_samples = []
        output_samples = []

        for x, y in dataset:
            # Ensure tensors and move to CPU for accumulation
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)

            x = x.cpu()
            y = y.cpu()

            input_samples.append(x)
            if compute_output_stats:
                output_samples.append(y)

        # Stack all samples and compute statistics
        # Shape: (N, C, H, W) after stacking
        input_stacked = torch.stack(input_samples, dim=0)  # (N, C, H, W)
        output_stacked = torch.stack(output_samples, dim=0) if compute_output_stats else None

        # Compute per-channel mean and std across (N, H, W)
        # Result: (C,) tensors
        input_mean = input_stacked.mean(dim=(0, 2, 3))  # Average over N, H, W
        input_std = input_stacked.std(dim=(0, 2, 3))  # Std over N, H, W

        if compute_output_stats:
            output_mean = output_stacked.mean(dim=(0, 2, 3))
            output_std = output_stacked.std(dim=(0, 2, 3))
        else:
            output_mean = torch.zeros_like(input_mean)
            output_std = torch.ones_like(input_mean)

        return cls(
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            epsilon=epsilon,
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor using computed statistics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (C, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Normalized tensor with same shape.
        """
        device = x.device
        mean = self.input_mean.to(device)
        std = (self.input_std + self.epsilon).to(device)

        # Reshape for broadcasting: (C,) -> (C, 1, 1) for 3D or (1, C, 1, 1) for 4D
        if x.ndim == 4:
            # Batch: (B, C, H, W) - expand stats to (1, C, 1, 1)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif x.ndim == 3:
            # Single sample: (C, H, W) - expand stats to (C, 1, 1)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return (x - mean) / std

    def normalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize output tensor using computed statistics.

        Parameters
        ----------
        y : torch.Tensor
            Output tensor of shape (C, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Normalized tensor with same shape.
        """
        device = y.device
        mean = self.output_mean.to(device)
        std = (self.output_std + self.epsilon).to(device)

        # Reshape for broadcasting: (C,) -> (C, 1, 1) for 3D or (1, C, 1, 1) for 4D
        if y.ndim == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif y.ndim == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return (y - mean) / std

    def denormalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor back to original scale.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input tensor of shape (C, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Denormalized tensor with same shape.
        """
        device = x.device
        mean = self.input_mean.to(device)
        std = (self.input_std + self.epsilon).to(device)

        # Reshape for broadcasting: (C,) -> (C, 1, 1) for 3D or (1, C, 1, 1) for 4D
        if x.ndim == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif x.ndim == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return x * std + mean

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor back to original scale.

        Parameters
        ----------
        y : torch.Tensor
            Normalized output tensor of shape (C, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Denormalized tensor with same shape.
        """
        device = y.device
        mean = self.output_mean.to(device)
        std = (self.output_std + self.epsilon).to(device)

        # Reshape for broadcasting: (C,) -> (C, 1, 1) for 3D or (1, C, 1, 1) for 4D
        if y.ndim == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif y.ndim == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return y * std + mean

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalizer to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary with 'input_mean', 'input_std', 'output_mean', 'output_std' keys.
        """
        return {
            "input_mean": self.input_mean.cpu().numpy(),
            "input_std": self.input_std.cpu().numpy(),
            "output_mean": self.output_mean.cpu().numpy(),
            "output_std": self.output_std.cpu().numpy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], epsilon: float = 1e-8) -> Normalizer:
        """Create normalizer from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys 'input_mean', 'input_std', 'output_mean', 'output_std'.
        epsilon : float, optional
            Small value to avoid division by zero, by default 1e-8.

        Returns
        -------
        Normalizer
            Normalizer instance.
        """
        return cls(
            input_mean=torch.from_numpy(data["input_mean"]).float(),
            input_std=torch.from_numpy(data["input_std"]).float(),
            output_mean=torch.from_numpy(data["output_mean"]).float(),
            output_std=torch.from_numpy(data["output_std"]).float(),
            epsilon=epsilon,
        )

    def save(self, path: str | Path) -> None:
        """Save normalizer statistics to NPZ file.

        Parameters
        ----------
        path : str or Path
            Path to save the NPZ file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_mean=self.input_mean.cpu().numpy(),
            input_std=self.input_std.cpu().numpy(),
            output_mean=self.output_mean.cpu().numpy(),
            output_std=self.output_std.cpu().numpy(),
        )

    @classmethod
    def load(cls, path: str | Path, epsilon: float = 1e-8) -> Normalizer:
        """Load normalizer statistics from NPZ file.

        Parameters
        ----------
        path : str or Path
            Path to the NPZ file.
        epsilon : float, optional
            Small value to avoid division by zero, by default 1e-8.

        Returns
        -------
        Normalizer
            Normalizer instance.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            return cls(
                input_mean=torch.from_numpy(data["input_mean"]).float(),
                input_std=torch.from_numpy(data["input_std"]).float(),
                output_mean=torch.from_numpy(data["output_mean"]).float(),
                output_std=torch.from_numpy(data["output_std"]).float(),
                epsilon=epsilon,
            )

    def __repr__(self) -> str:
        """Return a compact multi-line representation of normalizer stats."""
        return (
            f"Normalizer(\n"
            f"  input_mean: {self.input_mean},\n"
            f"  input_std: {self.input_std},\n"
            f"  output_mean: {self.output_mean},\n"
            f"  output_std: {self.output_std},\n"
            f"  epsilon: {self.epsilon}\n"
            f")"
        )
