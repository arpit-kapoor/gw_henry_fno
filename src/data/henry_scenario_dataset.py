from __future__ import annotations

"""Dataset for Henry scenario data stored as scenario-level NPZ files.

New data layout (grid_scenarios_random_skip2_20x40)
----------------------------------------------------
Each scenario directory contains a single ``scenario.npz`` file that packs all
runs for that scenario into two 5-D arrays::

    input_tensor  : (N_runs, C_in,  T_in,  Z, X)
    output_tensor : (N_runs, C_out, T_out, Z, X)

The train/val split is applied **at the run-index level** using a shared random
partition that is identical across all scenarios: if run index k is in the
validation set it is excluded from training for **every** scenario.

Each dataset sample is a single run returned as channel-first tensors suitable
for a 3-D FNO operating over the (time, vertical, horizontal) domain::

    x : (C_in,  T_in,  Z, X)   input
    y : (C_out, T_out, Z, X)   target output
"""

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .normalizer import Normalizer


Split = Literal["train", "val"]

#: Default train fraction (70 % train, 30 % validation).
DEFAULT_TRAIN_RATIO: float = 0.7


@dataclass(frozen=True)
class _SampleRef:
    """Reference to one dataset sample: scenario and run-index within its NPZ."""

    scenario_index: int
    run_index: int


class HenryScenarioDataset(Dataset):
    """Dataset for Henry scenario tensors stored in scenario-level NPZ files.

    Parameters
    ----------
    scenarios_dir : str or Path
        Parent directory containing ``scenario_NNN/scenario.npz`` subdirectories.
    split : {'train', 'val'}
        Which split to expose.
    train_ratio : float, optional
        Fraction of runs assigned to training (default 0.7).  The same random
        partition is applied to **every** scenario so that run index k is always
        in the same split regardless of scenario.
    seed : int, optional
        RNG seed for the run-level partition, by default 42.
    dtype : torch.dtype, optional
        Dtype for returned tensors, by default ``torch.float32``.
    cache_scenarios : bool, optional
        If True, cache each scenario's full NPZ arrays after first load.
        Recommended when memory permits; avoids repeated disk I/O per epoch.
        By default True.
    normalizer : Normalizer, optional
        Applied on ``__getitem__``.  If None, no normalisation is applied.
    """

    def __init__(
        self,
        scenarios_dir: str | Path,
        split: Split,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        cache_scenarios: bool = True,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        super().__init__()

        assert split in {"train", "val"}, (
            f"split must be 'train' or 'val', got {split!r}"
        )
        assert 0.0 < train_ratio < 1.0, (
            f"train_ratio must be in (0, 1), got {train_ratio}"
        )

        self.scenarios_dir = Path(scenarios_dir)
        if not self.scenarios_dir.exists():
            raise FileNotFoundError(
                f"scenarios_dir not found: {self.scenarios_dir}"
            )

        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.dtype = dtype
        self.cache_scenarios = cache_scenarios
        self.normalizer = normalizer

        # Discover sorted scenario directories.
        self.scenario_dirs: List[Path] = self._discover_scenario_dirs(
            self.scenarios_dir
        )
        if not self.scenario_dirs:
            raise ValueError(
                f"No scenario_NNN directories found in {self.scenarios_dir}"
            )

        # Determine run count from the first scenario (assumed equal across all).
        n_runs = self._probe_n_runs(self.scenario_dirs[0])

        # Compute shared train/val run indices — same partition for every scenario.
        # Both lists are stored as attributes for external inspection.
        self.train_indices, self.val_indices = self._split_run_indices(
            n_runs=n_runs,
            train_ratio=train_ratio,
            seed=seed,
        )
        split_indices: List[int] = (
            self.train_indices if split == "train" else self.val_indices
        )

        # Build flat list: one entry per (scenario, run-in-split).
        self._sample_refs: List[_SampleRef] = [
            _SampleRef(scenario_index=s_idx, run_index=r_idx)
            for s_idx in range(len(self.scenario_dirs))
            for r_idx in split_indices
        ]

        if not self._sample_refs:
            raise ValueError(
                f"Split '{split}' produced no samples in {self.scenarios_dir}"
            )

        # Scenario-level array cache: scenario_index -> (input_arr, output_arr).
        self._scenario_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Channel names are populated on first NPZ load.
        self._input_channel_names: List[str] = []
        self._output_channel_names: List[str] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_scenario_dirs(scenarios_dir: Path) -> List[Path]:
        """Return sorted list of ``scenario_NNN`` subdirectories."""
        return sorted(
            p for p in scenarios_dir.glob("scenario_*") if p.is_dir()
        )

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_n_runs(scenario_dir: Path) -> int:
        """Read the run count from a scenario NPZ without loading full arrays."""
        npz_path = scenario_dir / "scenario.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Expected scenario NPZ not found: {npz_path}"
            )
        with np.load(npz_path, allow_pickle=False) as data:
            return int(data["input_tensor"].shape[0])

    @staticmethod
    def _split_run_indices(
        n_runs: int,
        train_ratio: float,
        seed: int,
    ) -> Tuple[List[int], List[int]]:
        """Deterministically partition ``[0, n_runs)`` into train and val sets.

        The same partition is used for every scenario so that run index k is
        always in the same split regardless of the scenario it belongs to.

        Parameters
        ----------
        n_runs : int
            Total number of runs per scenario.
        train_ratio : float
            Fraction of runs assigned to training.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        train_indices, val_indices : list[int]
        """
        if n_runs < 2:
            raise ValueError(
                f"Need at least 2 runs to split; found {n_runs}"
            )

        indices = list(range(n_runs))
        Random(seed).shuffle(indices)

        n_train = max(1, min(int(n_runs * train_ratio), n_runs - 1))
        return indices[:n_train], indices[n_train:]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_scenario_npz(
        npz_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load and validate a scenario NPZ file.

        Expected arrays
        ---------------
        ``input_tensor``  : shape ``(N_runs, C_in,  T_in,  Z, X)``, float32
        ``output_tensor`` : shape ``(N_runs, C_out, T_out, Z, X)``, float32

        Returns
        -------
        inputs, outputs, input_channel_names, output_channel_names
        """
        with np.load(npz_path, allow_pickle=True) as data:
            for key in ("input_tensor", "output_tensor"):
                if key not in data.files:
                    raise KeyError(
                        f"Missing required key '{key}' in {npz_path}. "
                        f"Available keys: {data.files}"
                    )

            inputs = np.asarray(data["input_tensor"], dtype=np.float32)
            outputs = np.asarray(data["output_tensor"], dtype=np.float32)

            in_names: List[str] = (
                [str(n) for n in data["input_channel_names"]]
                if "input_channel_names" in data.files
                else []
            )
            out_names: List[str] = (
                [str(n) for n in data["output_channel_names"]]
                if "output_channel_names" in data.files
                else []
            )

        if inputs.ndim != 5:
            raise ValueError(
                f"Expected input_tensor shape (N, C, T, Z, X); "
                f"got {inputs.shape} in {npz_path}"
            )
        if outputs.ndim != 5:
            raise ValueError(
                f"Expected output_tensor shape (N, C, T, Z, X); "
                f"got {outputs.shape} in {npz_path}"
            )
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError(
                f"Run count mismatch: inputs has {inputs.shape[0]} runs, "
                f"outputs has {outputs.shape[0]} runs in {npz_path}"
            )
        if inputs.shape[2] != outputs.shape[2]:
            raise ValueError(
                f"Time dimension mismatch: T_in={inputs.shape[2]} vs "
                f"T_out={outputs.shape[2]} in {npz_path}. "
                f"The 3-D FNO requires T_in == T_out; please regenerate the dataset "
                f"with matching input and output time steps."
            )

        return inputs, outputs, in_names, out_names

    def _get_scenario_arrays(
        self, scenario_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(input_arr, output_arr)`` for a scenario, using cache.

        Arrays have shape ``(N_runs, C, T, Z, X)``.
        """
        if self.cache_scenarios and scenario_index in self._scenario_cache:
            return self._scenario_cache[scenario_index]

        npz_path = self.scenario_dirs[scenario_index] / "scenario.npz"
        inputs, outputs, in_names, out_names = self._load_scenario_npz(npz_path)

        # Populate channel names on first successful load.
        if not self._input_channel_names and in_names:
            self._input_channel_names = in_names
        if not self._output_channel_names and out_names:
            self._output_channel_names = out_names

        if self.cache_scenarios:
            self._scenario_cache[scenario_index] = (inputs, outputs)

        return inputs, outputs

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of samples in this split."""
        return len(self._sample_refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one ``(x, y)`` pair for a given dataset index.

        Returns
        -------
        x : torch.Tensor, shape ``(C_in,  T_in,  Z, X)``
        y : torch.Tensor, shape ``(C_out, T_out, Z, X)``
        """
        ref = self._sample_refs[index]
        inputs, outputs = self._get_scenario_arrays(ref.scenario_index)

        x = torch.from_numpy(inputs[ref.run_index]).to(self.dtype)
        y = torch.from_numpy(outputs[ref.run_index]).to(self.dtype)

        if self.normalizer is not None:
            x = self.normalizer.normalize_input(x)
            y = self.normalizer.normalize_output(y)

        return x, y

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def input_channel_names(self) -> List[str]:
        """Input channel names read from the first scenario NPZ."""
        if not self._input_channel_names and self.scenario_dirs:
            self._get_scenario_arrays(0)  # trigger load to populate names
        return self._input_channel_names

    @property
    def output_channel_names(self) -> List[str]:
        """Output channel names read from the first scenario NPZ."""
        if not self._output_channel_names and self.scenario_dirs:
            self._get_scenario_arrays(0)
        return self._output_channel_names

    @property
    def scenario_names(self) -> List[str]:
        """Sorted scenario directory names discovered in ``scenarios_dir``."""
        return [d.name for d in self.scenario_dirs]

    def sample_indices_for_scenario(self, scenario_name: str) -> List[int]:
        """Return dataset indices whose samples belong to the named scenario."""
        return [
            i
            for i, ref in enumerate(self._sample_refs)
            if self.scenario_dirs[ref.scenario_index].name == scenario_name
        ]

    def run_indices_for_scenario(self, scenario_name: str) -> List[int]:
        """Return the NPZ run indices in this split for the named scenario."""
        return [
            ref.run_index
            for ref in self._sample_refs
            if self.scenario_dirs[ref.scenario_index].name == scenario_name
        ]


def create_henry_dataloaders(
    scenarios_dir: str | Path,
    batch_size: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    cache_scenarios: bool = True,
    normalize: bool = False,
) -> Tuple[DataLoader, DataLoader] | Tuple[DataLoader, DataLoader, Normalizer]:
    """Create train and validation DataLoaders for Henry scenarios.

    Parameters
    ----------
    scenarios_dir : str or Path
        Parent directory containing ``scenario_NNN/scenario.npz`` subdirs.
    batch_size : int
        Batch size for both loaders.
    train_ratio : float, optional
        Fraction of runs used for training, by default 0.7.  The same random
        partition is applied across every scenario.
    seed : int, optional
        RNG seed for the run-level partition and training reproducibility,
        by default 42.
    num_workers : int, optional
        DataLoader worker processes, by default 0.
    pin_memory : bool, optional
        Enable pinned CPU memory in DataLoaders, by default False.
    dtype : torch.dtype, optional
        Dtype for returned tensors, by default ``torch.float32``.
    cache_scenarios : bool, optional
        Cache scenario NPZ arrays in memory, by default True.
    normalize : bool, optional
        If True, compute per-channel mean/std from training data and apply
        normalisation to both splits, by default False.

    Returns
    -------
    (train_loader, val_loader) or (train_loader, val_loader, normalizer)
        The normalizer is returned only when ``normalize=True``.
    """
    # Build unnormalised train set first to compute statistics.
    train_dataset_unnorm = HenryScenarioDataset(
        scenarios_dir=scenarios_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_scenarios=cache_scenarios,
        normalizer=None,
    )

    normalizer: Optional[Normalizer] = None
    if normalize:
        normalizer = Normalizer.from_dataset(train_dataset_unnorm)

    # Rebuild datasets with optional normalisation applied on access.
    train_dataset = HenryScenarioDataset(
        scenarios_dir=scenarios_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_scenarios=cache_scenarios,
        normalizer=normalizer,
    )
    val_dataset = HenryScenarioDataset(
        scenarios_dir=scenarios_dir,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_scenarios=cache_scenarios,
        normalizer=normalizer,
    )

    loader_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    if normalize:
        return train_loader, val_loader, normalizer
    return train_loader, val_loader
