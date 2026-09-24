"""Normalization utilities for Henry scenario data.

Supports tensors of any rank ≥ 3 where dimension 1 is the channel axis,
covering both the old 4-D layout ``(B, C, H, W)`` and the new 5-D layout
``(B, C, T, Z, X)`` for 3-D FNO training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Per-channel mean/std normalizer.

    Computes statistics on a source dataset (typically the train split) and
    applies the same normalisation to any split.

    Parameters
    ----------
    input_mean : torch.Tensor
        Per-channel mean of inputs, shape ``(C_in,)``.
    input_std : torch.Tensor
        Per-channel std of inputs, shape ``(C_in,)``.
    output_mean : torch.Tensor
        Per-channel mean of outputs, shape ``(C_out,)``.
    output_std : torch.Tensor
        Per-channel std of outputs, shape ``(C_out,)``.
    epsilon : float, optional
        Small value added to std to avoid division by zero, by default 1e-8.
    """

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> None:
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        compute_output_stats: bool = True,
        epsilon: float = 1e-8,
    ) -> "Normalizer":
        """Compute normaliser statistics from a dataset.

        Iterates all samples and computes per-channel mean and standard
        deviation, reducing over all non-channel dimensions.  Works for
        tensors of any shape ``(C, *spatial)``, e.g. ``(C, H, W)`` or
        ``(C, T, Z, X)``.

        Parameters
        ----------
        dataset : Dataset
            Dataset returning ``(x, y)`` tuples.
        compute_output_stats : bool, optional
            If True, compute stats for outputs as well, by default True.
        epsilon : float, optional
            Divisor epsilon, by default 1e-8.
        """
        input_samples = []
        output_samples = []

        for x, y in dataset:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            input_samples.append(x.cpu())
            if compute_output_stats:
                output_samples.append(y.cpu())

        # Stack → (N, C, *spatial)
        input_stacked = torch.stack(input_samples, dim=0)

        # Average over all axes except the channel axis (dim=1).
        reduce_dims = tuple(i for i in range(input_stacked.ndim) if i != 1)
        input_mean = input_stacked.mean(dim=reduce_dims)
        input_std = input_stacked.std(dim=reduce_dims)

        if compute_output_stats:
            output_stacked = torch.stack(output_samples, dim=0)
            reduce_dims_out = tuple(
                i for i in range(output_stacked.ndim) if i != 1
            )
            output_mean = output_stacked.mean(dim=reduce_dims_out)
            output_std = output_stacked.std(dim=reduce_dims_out)
        else:
            output_mean = torch.zeros(1)
            output_std = torch.ones(1)

        return cls(
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            epsilon=epsilon,
        )

    # ------------------------------------------------------------------
    # Internal broadcasting helper
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_stats(
        stats: torch.Tensor, ndim: int
    ) -> torch.Tensor:
        """Reshape a ``(C,)`` stats tensor to broadcast over any ``ndim`` tensor.

        For a tensor of shape ``(C, *spatial)`` (ndim-1 trailing dims after C)
        this returns shape ``(C, 1, 1, ...)``.  For a batched tensor ``(B, C,
        *spatial)`` this returns ``(1, C, 1, 1, ...)``.

        Parameters
        ----------
        stats : torch.Tensor, shape ``(C,)``
        ndim : int
            Number of dimensions in the target tensor (3, 4, or 5).
        """
        # ndim=3 → (C,1,1); ndim=4 → (1,C,1,1); ndim=5 → (1,C,1,1,1)
        if ndim == 3:
            # Single sample: (C, *spatial)
            n_trailing = ndim - 1
            return stats.view(-1, *([1] * n_trailing))
        else:
            # Batched: (B, C, *spatial)
            n_trailing = ndim - 2
            return stats.view(1, -1, *([1] * n_trailing))

    # ------------------------------------------------------------------
    # Normalise / denormalise
    # ------------------------------------------------------------------

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise input tensor channel-wise.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(C, *spatial)`` or ``(B, C, *spatial)``.

        Returns
        -------
        torch.Tensor
            Normalised tensor with the same shape.
        """
        device = x.device
        mean = self._broadcast_stats(self.input_mean.to(device), x.ndim)
        std = self._broadcast_stats(
            (self.input_std + self.epsilon).to(device), x.ndim
        )
        return (x - mean) / std

    def normalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Normalise output tensor channel-wise.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(C, *spatial)`` or ``(B, C, *spatial)``.
        """
        device = y.device
        mean = self._broadcast_stats(self.output_mean.to(device), y.ndim)
        std = self._broadcast_stats(
            (self.output_std + self.epsilon).to(device), y.ndim
        )
        return (y - mean) / std

    def denormalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation for input tensors."""
        device = x.device
        mean = self._broadcast_stats(self.input_mean.to(device), x.ndim)
        std = self._broadcast_stats(
            (self.input_std + self.epsilon).to(device), x.ndim
        )
        return x * std + mean

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation for output tensors."""
        device = y.device
        mean = self._broadcast_stats(self.output_mean.to(device), y.ndim)
        std = self._broadcast_stats(
            (self.output_std + self.epsilon).to(device), y.ndim
        )
        return y * std + mean

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/NPZ serialisation."""
        return {
            "input_mean": self.input_mean.cpu().numpy(),
            "input_std": self.input_std.cpu().numpy(),
            "output_mean": self.output_mean.cpu().numpy(),
            "output_std": self.output_std.cpu().numpy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], epsilon: float = 1e-8) -> "Normalizer":
        """Reconstruct from a dictionary (inverse of :meth:`to_dict`)."""
        return cls(
            input_mean=torch.from_numpy(data["input_mean"]).float(),
            input_std=torch.from_numpy(data["input_std"]).float(),
            output_mean=torch.from_numpy(data["output_mean"]).float(),
            output_std=torch.from_numpy(data["output_std"]).float(),
            epsilon=epsilon,
        )

    def save(self, path: str | Path) -> None:
        """Save statistics to a NPZ file."""
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
    def load(cls, path: str | Path, epsilon: float = 1e-8) -> "Normalizer":
        """Load statistics from a NPZ file."""
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                input_mean=torch.from_numpy(data["input_mean"]).float(),
                input_std=torch.from_numpy(data["input_std"]).float(),
                output_mean=torch.from_numpy(data["output_mean"]).float(),
                output_std=torch.from_numpy(data["output_std"]).float(),
                epsilon=epsilon,
            )

    def __repr__(self) -> str:
        return (
            f"Normalizer(\n"
            f"  input_mean:  {self.input_mean},\n"
            f"  input_std:   {self.input_std},\n"
            f"  output_mean: {self.output_mean},\n"
            f"  output_std:  {self.output_std},\n"
            f"  epsilon:     {self.epsilon}\n"
            f")"
        )
