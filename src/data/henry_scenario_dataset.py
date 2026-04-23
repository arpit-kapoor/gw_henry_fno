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
    """Reference to one sample identified by scenario, run, and window index."""

    scenario_index: int
    run_index: int
    window_index: int


class HenryScenarioDataset(Dataset):
    """Dataset for Henry scenario window tensors across multiple scenarios.

    The dataset aggregates data from all scenario directories and splits at the run level.
    Each sample is returned as channel-first tensors: ``(C_in, H, W)`` and ``(C_out, H, W)``.

    Parameters
    ----------
    scenarios_dir : str or Path
        Path to parent directory containing all scenario folders (e.g., ``scenarios``)
        which contains ``scenario_01``, ``scenario_02``, etc. Each scenario folder
        contains ``run_*/windows.npz`` directories.
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
    validation_run_name : str, optional
        If provided (e.g., "run_000003"), use this specific run as validation for each scenario.
        Training will contain all other runs from all scenarios.
        If None, use random splitting based on train_ratio, by default None.
    """

    def __init__(
        self,
        scenarios_dir: str | Path,
        split: Split,
        train_ratio: float = 0.8,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        cache_runs: bool = True,
        normalizer: Optional[Normalizer] = None,
        validation_run_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Validate high-level split configuration early so downstream errors are clearer.
        assert split in {"train", "val"}, f"Error: split must be 'train' or 'val', got {split}"
        assert 0.0 < train_ratio < 1.0, f"Error: train_ratio must be in (0, 1), got {train_ratio}"

        self.scenarios_dir = Path(scenarios_dir)
        if not self.scenarios_dir.exists():
            raise FileNotFoundError(f"Scenarios directory not found: {self.scenarios_dir}")
        if not self.scenarios_dir.is_dir():
            raise ValueError(f"Expected a directory for scenarios_dir, got: {self.scenarios_dir}")

        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.dtype = dtype
        self.cache_runs = cache_runs
        self.normalizer = normalizer
        self.validation_run_name = validation_run_name

        # Discover all scenario directories and collect runs across all scenarios.
        all_scenario_dirs = self._discover_scenario_dirs(self.scenarios_dir)
        if len(all_scenario_dirs) == 0:
            raise ValueError(f"No scenario directories found in {self.scenarios_dir}")

        # Collect all runs from all scenarios with their scenario indices
        all_run_refs = self._collect_all_runs(all_scenario_dirs)
        if len(all_run_refs) == 0:
            raise ValueError(f"No runs found across all scenarios in {self.scenarios_dir}")

        # Split runs at the run level (not scenario level) to maintain data independence
        split_run_refs = self._split_run_refs(
            all_run_refs, 
            split=split, 
            train_ratio=train_ratio, 
            seed=seed,
            validation_run_name=validation_run_name,
        )
        if len(split_run_refs) == 0:
            raise ValueError(f"Split '{split}' has no runs in {self.scenarios_dir}")

        self.scenario_dirs: List[Path] = all_scenario_dirs
        self.run_refs: List[Tuple[int, Path]] = split_run_refs  # (scenario_index, run_path)
        self._run_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}  # (scenario_idx, run_idx_in_split)
        # Build a global index mapping dataset rows to concrete (scenario, run, window) locations.
        self._sample_refs: List[_SampleRef] = self._build_sample_refs()

        if len(self._sample_refs) == 0:
            raise ValueError(f"Split '{split}' has no samples in {self.scenarios_dir}")

    @staticmethod
    def _discover_scenario_dirs(scenarios_dir: Path) -> List[Path]:
        """Discover all scenario directories in the parent directory.

        Expects directories named ``scenario_NN`` where NN is a numeric index.

        Returns
        -------
        list[Path]
            Sorted list of scenario directories.
        """
        scenario_dirs = []
        for scenario_dir in sorted(scenarios_dir.glob("scenario_*")):
            if scenario_dir.is_dir():
                scenario_dirs.append(scenario_dir)

        return scenario_dirs

    @staticmethod
    def _discover_run_dirs(scenario_dir: Path) -> List[Path]:
        """Discover run directories that contain a windows.npz artifact in a single scenario.

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

        return run_dirs

    @staticmethod
    def _collect_all_runs(scenario_dirs: List[Path]) -> List[Tuple[int, Path]]:
        """Collect all runs across all scenarios.

        Parameters
        ----------
        scenario_dirs : list[Path]
            Sorted list of scenario directories.

        Returns
        -------
        list[tuple[int, Path]]
            List of (scenario_index, run_path) tuples for all runs across all scenarios.

        Raises
        ------
        FileNotFoundError
            If no runs are found in any scenario.
        """
        all_run_refs: List[Tuple[int, Path]] = []

        for scenario_idx, scenario_dir in enumerate(scenario_dirs):
            run_dirs = HenryScenarioDataset._discover_run_dirs(scenario_dir)
            for run_dir in run_dirs:
                all_run_refs.append((scenario_idx, run_dir))

        return all_run_refs

    @staticmethod
    def _split_run_refs(
        run_refs: List[Tuple[int, Path]],
        split: Split,
        train_ratio: float,
        seed: int,
        validation_run_name: Optional[str] = None,
    ) -> List[Tuple[int, Path]]:
        """Split run references into train/validation subsets at run granularity.

        Parameters
        ----------
        run_refs : list[tuple[int, Path]]
            List of (scenario_index, run_path) tuples.
        split : {'train', 'val'}
            Which split to expose.
        train_ratio : float
            Train/val split ratio (used only if validation_run_name is None).
        seed : int
            Random seed for shuffling (used only if validation_run_name is None).
        validation_run_name : str, optional
            If provided (e.g., "run_000003"), use this specific run as validation per scenario.
            Otherwise, use random splitting based on train_ratio and seed.

        Notes
        -----
        If validation_run_name is provided, implements leave-one-run-out per scenario.
        Otherwise, uses deterministic random splitting with the given seed.
        """
        if validation_run_name is not None:
            # Leave-one-run-out per scenario: designated run is validation, others are training
            train_runs = []
            val_runs = []
            for scenario_idx, run_dir in run_refs:
                if run_dir.name == validation_run_name:
                    val_runs.append((scenario_idx, run_dir))
                else:
                    train_runs.append((scenario_idx, run_dir))

            if len(val_runs) == 0:
                raise ValueError(
                    f"No runs matching validation_run_name='{validation_run_name}' found. "
                    f"Available run names: {sorted(set(run_dir.name for _, run_dir in run_refs))}"
                )

            return val_runs if split == "val" else train_runs
        else:
            # Original random splitting behavior
            if len(run_refs) < 2:
                raise ValueError(
                    "At least two runs are required for train/val splitting; "
                    f"found {len(run_refs)}"
                )

            shuffled = list(run_refs)
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

    def _get_run_tensors(self, scenario_index: int, run_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return one run's tensors from cache or disk.

        When ``cache_runs`` is enabled, a run is loaded once and reused for
        subsequent sample access to reduce repeated disk I/O.

        Parameters
        ----------
        scenario_index : int
            Index in the scenario_dirs list.
        run_index : int
            Index in the run_refs list for this split.
        """
        cache_key = (scenario_index, run_index)
        if self.cache_runs and cache_key in self._run_cache:
            return self._run_cache[cache_key]

        scenario_idx, run_dir = self.run_refs[run_index]
        windows_path = run_dir / "windows.npz"
        inputs, outputs = self._load_windows(windows_path)

        if self.cache_runs:
            self._run_cache[cache_key] = (inputs, outputs)

        return inputs, outputs

    def _build_sample_refs(self) -> List[_SampleRef]:
        """Build lookup references from global index to run/window pairs.

        This flattens all windows across all selected runs into one linear index,
        so ``__getitem__`` can access samples in O(1) by first resolving the
        corresponding run and window indices.
        """
        refs: List[_SampleRef] = []
        for run_index in range(len(self.run_refs)):
            scenario_idx, run_dir = self.run_refs[run_index]
            inputs, outputs = self._get_run_tensors(scenario_idx, run_index)
            n_windows = inputs.shape[0]
            if outputs.shape[0] != n_windows:
                raise ValueError(
                    f"Run {run_dir} has unequal input/output windows: "
                    f"{inputs.shape[0]} vs {outputs.shape[0]}"
                )

            refs.extend(
                _SampleRef(scenario_index=scenario_idx, run_index=run_index, window_index=w)
                for w in range(n_windows)
            )

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
        inputs, outputs = self._get_run_tensors(ref.scenario_index, ref.run_index)

        x = torch.from_numpy(inputs[ref.window_index]).to(self.dtype)
        y = torch.from_numpy(outputs[ref.window_index]).to(self.dtype)

        if self.normalizer is not None:
            x = self.normalizer.normalize_input(x)
            y = self.normalizer.normalize_output(y)

        return x, y

    @property
    def run_names(self) -> List[str]:
        """Return scenario and run directory names in this split."""
        return [
            f"{self.scenario_dirs[scenario_idx].name}/{run_dir.name}"
            for scenario_idx, run_dir in self.run_refs
        ]

    @property
    def scenario_names(self) -> List[str]:
        """Return scenario directory names discovered in scenarios_dir."""
        return [scenario_dir.name for scenario_dir in self.scenario_dirs]

    def sample_indices_for_scenario(self, scenario_name: str) -> List[int]:
        """Return dataset indices whose samples belong to the given scenario."""
        indices: List[int] = []
        for sample_idx, ref in enumerate(self._sample_refs):
            if self.scenario_dirs[ref.scenario_index].name == scenario_name:
                indices.append(sample_idx)
        return indices


def create_henry_dataloaders(
    scenarios_dir: str | Path,
    batch_size: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    cache_runs: bool = True,
    normalize: bool = False,
    validation_run_name: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader] | Tuple[DataLoader, DataLoader, Normalizer]:
    """Create train and validation DataLoaders for Henry scenarios.

    Parameters
    ----------
    scenarios_dir : str or Path
        Path to parent directory containing all scenario folders (e.g., ``scenarios``).
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
    validation_run_name : str, optional
        If provided (e.g., "run_000003"), use this specific run as validation for each scenario.
        Training will contain all other runs from all scenarios.
        If None, use random splitting based on train_ratio, by default None.

    Returns
    -------
    tuple of DataLoader or tuple of (DataLoader, DataLoader, Normalizer)
        If normalize=False: ``(train_loader, val_loader)``.
        If normalize=True: ``(train_loader, val_loader, normalizer)``.
    """

    # Build the train split without normalization first so statistics are computed
    # from raw training data only.
    train_dataset_unnormalized = HenryScenarioDataset(
        scenarios_dir=scenarios_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_runs=cache_runs,
        normalizer=None,
        validation_run_name=validation_run_name,
    )

    # Fit normalization stats on the training split and reuse them for both splits.
    normalizer = None
    if normalize:
        normalizer = Normalizer.from_dataset(train_dataset_unnormalized)

    # Rebuild train/val datasets with optional normalization applied on access.
    train_dataset = HenryScenarioDataset(
        scenarios_dir=scenarios_dir,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_runs=cache_runs,
        normalizer=normalizer,
        validation_run_name=validation_run_name,
    )
    val_dataset = HenryScenarioDataset(
        scenarios_dir=scenarios_dir,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
        dtype=dtype,
        cache_runs=cache_runs,
        normalizer=normalizer,
        validation_run_name=validation_run_name,
    )

    # Training loader is shuffled for SGD; validation loader preserves deterministic order.
    # Persistent workers avoid process respawn every epoch, which is costly for short epochs.
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
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
    else:
        return train_loader, val_loader
