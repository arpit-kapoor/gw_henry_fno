from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .normalizer import Normalizer


Split = Literal["train", "val"]


@dataclass(frozen=True)
class _SampleRef:
    """Reference to one sample identified by run and window index."""

    run_index: int
    window_index: int


class HenryScenarioDataset(Dataset):
    """Dataset for Henry coupling scenario window tensors.

    The dataset splits data at the run level and returns individual window samples.
    Each sample is returned as channel-first tensors:
    ``(C_in, H, W)`` and ``(C_out, H, W)``.

    Parameters
    ----------
    scenario_dir : str or Path
        Path to one scenario folder (e.g. ``.../scenario_01``) containing
        ``run_*/windows.npz`` directories.
    split : {'train', 'val'}
        Which split to expose.
    train_ratio : float, optional
        Fraction of runs used for training, by default 0.8.
    seed : int, optional
        Random seed used to deterministically split runs, by default 42.
    dtype : torch.dtype, optional
        Torch dtype used for returned tensors, by default ``torch.float32``.
    cache_runs : bool, optional
        If True, cache per-run arrays after first access, by default True.
    normalizer : Normalizer, optional
        Normalizer instance to apply to input/output tensors.
        If None, no normalization is applied, by default None.
    """

    def __init__(
        self,
        scenario_dir: str | Path,
        split: Split,
        train_ratio: float = 0.8,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        cache_runs: bool = True,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        super().__init__()

        # Validate high-level split configuration early so downstream errors are clearer.
        assert split in {"train", "val"}, f"Error: split must be 'train' or 'val', got {split}"
        assert 0.0 < train_ratio < 1.0, f"Error: train_ratio must be in (0, 1), got {train_ratio}"

        self.scenario_dir = Path(scenario_dir)
        if not self.scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {self.scenario_dir}")
        if not self.scenario_dir.is_dir():
            raise ValueError(f"Expected a directory for scenario_dir, got: {self.scenario_dir}")

        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.dtype = dtype
        self.cache_runs = cache_runs
        self.normalizer = normalizer

        # Discover candidate run directories and then apply deterministic run-level splitting.
        # Splitting by run (not by window) avoids leakage between train and validation.
        all_run_dirs = self._discover_run_dirs(self.scenario_dir)
        split_run_dirs = self._split_run_dirs(all_run_dirs, split=split, train_ratio=train_ratio, seed=seed)
        if len(split_run_dirs) == 0:
            raise ValueError(f"Split '{split}' has no runs in {self.scenario_dir}")

        self.run_dirs: List[Path] = split_run_dirs
        self._run_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        # Build a global index mapping dataset rows to concrete (run, window) locations.
        self._sample_refs: List[_SampleRef] = self._build_sample_refs()

        if len(self._sample_refs) == 0:
            raise ValueError(f"Split '{split}' has no samples in {self.scenario_dir}")

    @staticmethod
    def _discover_run_dirs(scenario_dir: Path) -> List[Path]:
        """Discover run directories that contain a windows.npz artifact.

        The dataset expects a layout like:
        ``scenario_dir/run_XX/windows.npz``

        Returns
        -------
        list[Path]
            Sorted run directories that include ``windows.npz``.
        """
        run_dirs = []
        for run_dir in sorted(scenario_dir.glob("run_*")):
            if not run_dir.is_dir():
                continue
            if (run_dir / "windows.npz").exists():
                run_dirs.append(run_dir)

        if not run_dirs:
            raise FileNotFoundError(
                f"No run directories with windows.npz found under: {scenario_dir}"
            )

        return run_dirs

    @staticmethod
    def _split_run_dirs(
        run_dirs: List[Path],
        split: Split,
        train_ratio: float,
        seed: int,
    ) -> List[Path]:
        """Split runs into train/validation subsets at run granularity.

        Notes
        -----
        A deterministic shuffle based on ``seed`` is used before splitting.
        At least one run is kept in each split.
        """
        if len(run_dirs) < 2:
            raise ValueError(
                "At least two runs are required for train/val splitting; "
                f"found {len(run_dirs)}"
            )

        shuffled = list(run_dirs)
        Random(seed).shuffle(shuffled)

        n_train = int(len(shuffled) * train_ratio)
        n_train = max(1, min(n_train, len(shuffled) - 1))

        train_runs = shuffled[:n_train]
        val_runs = shuffled[n_train:]

        return train_runs if split == "train" else val_runs

    @staticmethod
    def _load_windows(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load one run file and validate required keys and shapes.

        Expected arrays inside ``windows.npz``:
        - ``input_tensor``: shape ``(N, C_in, H, W)``
        - ``output_tensor``: shape ``(N, C_out, H, W)``
        """
        with np.load(path, allow_pickle=False) as data:
            required_keys = {"input_tensor", "output_tensor"}
            missing = required_keys - set(data.files)
            if missing:
                raise KeyError(
                    f"Missing keys {sorted(missing)} in {path}. "
                    f"Available keys: {data.files}"
                )

            inputs = np.asarray(data["input_tensor"], dtype=np.float32)
            outputs = np.asarray(data["output_tensor"], dtype=np.float32)

        if inputs.ndim != 4:
            raise ValueError(
                f"Expected input_tensor shape (N, C, H, W), got {inputs.shape} in {path}"
            )
        if outputs.ndim != 4:
            raise ValueError(
                f"Expected output_tensor shape (N, C, H, W), got {outputs.shape} in {path}"
            )
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError(
                f"Mismatch in window count between inputs {inputs.shape[0]} and outputs "
                f"{outputs.shape[0]} in {path}"
            )

        return inputs, outputs

    def _get_run_tensors(self, run_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return one run's tensors from cache or disk.

        When ``cache_runs`` is enabled, a run is loaded once and reused for
        subsequent sample access to reduce repeated disk I/O.
        """
        if self.cache_runs and run_index in self._run_cache:
            return self._run_cache[run_index]

        run_dir = self.run_dirs[run_index]
        windows_path = run_dir / "windows.npz"
        inputs, outputs = self._load_windows(windows_path)

        if self.cache_runs:
            self._run_cache[run_index] = (inputs, outputs)

        return inputs, outputs

    def _build_sample_refs(self) -> List[_SampleRef]:
        """Build lookup references from global index to run/window pairs.

        This flattens all windows across all selected runs into one linear index,
        so ``__getitem__`` can access samples in O(1) by first resolving the
        corresponding run and window indices.
        """
        refs: List[_SampleRef] = []
        for run_index in range(len(self.run_dirs)):
            inputs, outputs = self._get_run_tensors(run_index)
            n_windows = inputs.shape[0]
            if outputs.shape[0] != n_windows:
                raise ValueError(
                    f"Run {self.run_dirs[run_index]} has unequal input/output windows: "
                    f"{inputs.shape[0]} vs {outputs.shape[0]}"
                )

            refs.extend(_SampleRef(run_index=run_index, window_index=w) for w in range(n_windows))

        return refs

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return len(self._sample_refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one input/output tensor pair for a dataset index.

        The returned tensors are channel-first and optionally normalized
        using the dataset's ``normalizer``.
        """
        ref = self._sample_refs[index]
        inputs, outputs = self._get_run_tensors(ref.run_index)

        x = torch.from_numpy(inputs[ref.window_index]).to(self.dtype)
        y = torch.from_numpy(outputs[ref.window_index]).to(self.dtype)

        if self.normalizer is not None:
            x = self.normalizer.normalize_input(x)
            y = self.normalizer.normalize_output(y)

        return x, y

    @property
    def run_names(self) -> List[str]:
        """Return run directory names in this split."""
        return [run_dir.name for run_dir in self.run_dirs]


def create_henry_dataloaders(
    scenario_dir: str | Path,
    batch_size: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    cache_runs: bool = True,
    normalize: bool = False,
) -> Tuple[DataLoader, DataLoader] | Tuple[DataLoader, DataLoader, Normalizer]:
    """Create train and validation DataLoaders for one Henry scenario.

    Parameters
    ----------
    scenario_dir : str or Path
        Path to scenario directory.
    batch_size : int
        Batch size for dataloaders.
    train_ratio : float, optional
        Train/val split ratio, by default 0.8.
    seed : int, optional
        Random seed for reproducibility, by default 42.
    num_workers : int, optional
        Number of dataloader workers, by default 0.
    pin_memory : bool, optional
        Whether to pin memory in dataloader, by default False.
    dtype : torch.dtype, optional
        Torch dtype for tensors, by default torch.float32.
    cache_runs : bool, optional
        Whether to cache runs in memory, by default True.
    normalize : bool, optional
        If True, compute normalizer from train set and apply to both datasets,
        by default False.

    Returns
    -------
    tuple of DataLoader or tuple of (DataLoader, DataLoader, Normalizer)
        If normalize=False: ``(train_loader, val_loader)``.
        If normalize=True: ``(train_loader, val_loader, normalizer)``.
    """

    # Build the train split without normalization first so statistics are computed
    # from raw training data only.
    train_dataset_unnormalized = HenryScenarioDataset(
        scenario_dir=scenario_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_runs=cache_runs,
        normalizer=None,
    )

    # Fit normalization stats on the training split and reuse them for both splits.
    normalizer = None
    if normalize:
        normalizer = Normalizer.from_dataset(train_dataset_unnormalized)

    # Rebuild train/val datasets with optional normalization applied on access.
    train_dataset = HenryScenarioDataset(
        scenario_dir=scenario_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_runs=cache_runs,
        normalizer=normalizer,
    )
    val_dataset = HenryScenarioDataset(
        scenario_dir=scenario_dir,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_runs=cache_runs,
        normalizer=normalizer,
    )

    # Training loader is shuffled for SGD; validation loader preserves deterministic order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if normalize:
        return train_loader, val_loader, normalizer
    else:
        return train_loader, val_loader
