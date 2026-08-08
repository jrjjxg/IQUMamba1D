"""Independent mixed-cardinality data loading for unknown-source experiments."""

from __future__ import annotations

import bisect
import random
import re
from collections import OrderedDict
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LazyH5SignalDataset(Dataset):
    """Read MATLAB v7.3 source/mixture frames lazily from paired files."""

    def __init__(
        self,
        records: Sequence[dict],
        native_sources: int,
        max_open_files: int = 32,
    ) -> None:
        if not records:
            raise ValueError("LazyH5SignalDataset requires at least one file pair")
        self.records = list(records)
        self.native_sources = int(native_sources)
        self.max_open_files = int(max(2, max_open_files))
        self.frame_length = int(self.records[0]["frame_length"])
        self._file_cache = OrderedDict()
        self.cumulative_frames = []
        total = 0
        for record in self.records:
            total += int(record["frame_count"])
            self.cumulative_frames.append(total)

    def __len__(self) -> int:
        return self.cumulative_frames[-1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = OrderedDict()
        return state

    def _get_h5_file(self, path: str):
        import h5py

        if path in self._file_cache:
            handle = self._file_cache.pop(path)
            self._file_cache[path] = handle
            return handle
        handle = h5py.File(path, "r")
        self._file_cache[path] = handle
        while len(self._file_cache) > self.max_open_files:
            _, old_handle = self._file_cache.popitem(last=False)
            old_handle.close()
        return handle

    def __del__(self):
        for handle in getattr(self, "_file_cache", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect.bisect_right(self.cumulative_frames, index)
        previous = self.cumulative_frames[record_index - 1] if record_index else 0
        frame_index = index - previous
        record = self.records[record_index]

        target_file = self._get_h5_file(record["target_path"])
        mixture_file = self._get_h5_file(record["mixture_path"])
        target = np.asarray(target_file["ideal_frames"][:, :, frame_index], dtype=np.float32)
        mixture = np.asarray(mixture_file["mixed_frames"][:, :, frame_index], dtype=np.float32)

        expected_channels = 2 * self.native_sources
        if target.shape[0] != expected_channels:
            raise ValueError(
                f"Expected target channels={expected_channels}, got {target.shape} in "
                f"{record['target_path']}"
            )
        if mixture.shape[0] != 2 or mixture.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"Invalid mixture/target shapes {mixture.shape}/{target.shape} in "
                f"{record['mixture_path']}"
            )
        return torch.from_numpy(mixture), torch.from_numpy(target), float(record["snr"])


def _parse_file_key(path: Path) -> Tuple[int, float]:
    match = re.search(
        r"_(?:target|mixed)_(\d+)_SNR=?([+-]?\d+(?:\.\d+)?)dB",
        path.name,
    )
    if not match:
        raise ValueError(f"Cannot parse file index/SNR from {path}")
    return int(match.group(1)), float(match.group(2))


def _resolve_choice_root(root: Optional[Union[str, Path]], data_choice: str) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "data" / "synthetic"
    if base.name == data_choice and (base / "target").is_dir():
        return base
    return base / data_choice


def _discover_records(
    data_choice: str,
    native_sources: int,
    root: Optional[Union[str, Path]],
) -> List[dict]:
    import h5py

    choice_root = _resolve_choice_root(root, data_choice)
    target_paths = sorted((choice_root / "target").glob("*Dataset_target_*_SNR*.mat"))
    mixture_paths = sorted((choice_root / "mixture").glob("*Dataset_mixed_*_SNR*.mat"))
    if not target_paths or not mixture_paths:
        raise FileNotFoundError(
            f"No paired MATLAB v7.3 files found under {choice_root / 'target'} and "
            f"{choice_root / 'mixture'}"
        )
    target_map = {_parse_file_key(path): path for path in target_paths}
    mixture_map = {_parse_file_key(path): path for path in mixture_paths}
    keys = sorted(set(target_map) & set(mixture_map), key=lambda item: (item[1], item[0]))
    if len(keys) != len(target_map) or len(keys) != len(mixture_map):
        raise ValueError(f"Target/mixture file pairing mismatch under {choice_root}")

    records = []
    expected_channels = 2 * int(native_sources)
    for file_index, snr in keys:
        target_path = target_map[(file_index, snr)]
        mixture_path = mixture_map[(file_index, snr)]
        with h5py.File(target_path, "r") as target_file:
            target_shape = tuple(target_file["ideal_frames"].shape)
        with h5py.File(mixture_path, "r") as mixture_file:
            mixture_shape = tuple(mixture_file["mixed_frames"].shape)
        if target_shape[0] != expected_channels or mixture_shape[0] != 2:
            raise ValueError(
                f"Unexpected HDF5 channel layout for {data_choice}: "
                f"target={target_shape}, mixture={mixture_shape}"
            )
        if target_shape[1:] != mixture_shape[1:]:
            raise ValueError(
                f"Target/mixture shape mismatch for {data_choice}: "
                f"target={target_shape}, mixture={mixture_shape}"
            )
        records.append(
            {
                "file_index": file_index,
                "snr": snr,
                "target_path": str(target_path),
                "mixture_path": str(mixture_path),
                "frame_length": target_shape[1],
                "frame_count": target_shape[2],
            }
        )
    return records


def _split_file_indices(
    records: Sequence[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    file_indices = sorted({int(record["file_index"]) for record in records})
    random.Random(int(seed)).shuffle(file_indices)
    count = len(file_indices)
    train_count = max(1, int(round(count * float(train_ratio))))
    val_count = max(1, int(round(count * float(val_ratio))))
    if train_count + val_count >= count:
        val_count = max(1, count - train_count - 1)
    if count - train_count - val_count < 1:
        raise ValueError(f"Need at least three file indices, got {count}")
    return (
        set(file_indices[:train_count]),
        set(file_indices[train_count : train_count + val_count]),
        set(file_indices[train_count + val_count :]),
    )


def _records_for_indices(records: Sequence[dict], indices: set) -> List[dict]:
    return [record for record in records if int(record["file_index"]) in indices]

class PaddedSourceDataset(Dataset):
    """Pad a native fixed-K dataset to a fixed maximum number of source slots."""

    def __init__(
        self,
        dataset: Dataset,
        native_sources: int,
        max_sources: int,
        dataset_name: str,
    ) -> None:
        if native_sources < 1 or native_sources > max_sources:
            raise ValueError(
                f"native_sources must be in [1, {max_sources}], got {native_sources}"
            )
        self.dataset = dataset
        self.native_sources = int(native_sources)
        self.max_sources = int(max_sources)
        self.dataset_name = str(dataset_name)
        self.frame_length = getattr(dataset, "frame_length", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            raise ValueError("Fixed-source dataset must return (mixture, target, snr, ...)")

        mixture, target, snr = sample[:3]
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        if mixture.ndim != 2 or mixture.size(0) != 2:
            raise ValueError(f"Expected mixture [2,L], got {tuple(mixture.shape)}")
        expected_channels = 2 * self.native_sources
        if target.ndim != 2 or target.size(0) != expected_channels:
            raise ValueError(
                f"Expected native target [{expected_channels},L], got {tuple(target.shape)}"
            )
        if target.size(-1) != mixture.size(-1):
            raise ValueError(
                f"Mixture/target length mismatch: {mixture.size(-1)} vs {target.size(-1)}"
            )

        padded = target.new_zeros(2 * self.max_sources, target.size(-1))
        padded[:expected_channels] = target
        valid_mask = target.new_zeros(self.max_sources)
        valid_mask[: self.native_sources] = 1.0

        return {
            "mixture": mixture,
            "target": padded,
            "snr": torch.as_tensor(snr, dtype=torch.float32),
            "valid_mask": valid_mask,
            "source_count": torch.tensor(self.native_sources, dtype=torch.long),
        }


def _validate_lengths(datasets: Sequence[Dataset]) -> int:
    lengths = set()
    for dataset in datasets:
        if len(dataset) == 0:
            raise ValueError("Unknown-source mode received an empty dataset split")
        frame_length = getattr(dataset, "frame_length", None)
        if frame_length is None:
            frame_length = int(dataset[0]["mixture"].size(-1))
        lengths.add(int(frame_length))
    if len(lengths) != 1:
        raise ValueError(
            f"All mixed-cardinality datasets must use the same frame length, got {sorted(lengths)}"
        )
    return next(iter(lengths))


class BalancedSourceBatchSampler(Sampler[List[int]]):
    """Sample K=2 and K=3 examples in every training batch."""

    def __init__(
        self,
        datasets: Sequence[PaddedSourceDataset],
        batch_size: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_count = {2: [], 3: []}
        offset = 0
        total = 0
        for dataset in datasets:
            indices = list(range(offset, offset + len(dataset)))
            self.indices_by_count[dataset.native_sources].extend(indices)
            offset += len(dataset)
            total += len(dataset)
        if not self.indices_by_count[2] or not self.indices_by_count[3]:
            raise ValueError("Balanced batches require both K=2 and K=3 datasets")
        self.num_batches = (total + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def _draw(self, values: List[int], count: int, generator: torch.Generator) -> List[int]:
        if count <= 0:
            return []
        positions = torch.randint(len(values), (count,), generator=generator).tolist()
        return [values[position] for position in positions]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for batch_index in range(self.num_batches):
            if self.batch_size == 1:
                source_count = 2 if batch_index % 2 == 0 else 3
                batch = self._draw(self.indices_by_count[source_count], 1, generator)
            else:
                count_two = self.batch_size // 2
                count_three = self.batch_size - count_two
                batch = self._draw(self.indices_by_count[2], count_two, generator)
                batch.extend(self._draw(self.indices_by_count[3], count_three, generator))
                order = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[position] for position in order]
            yield batch


def _wrap_split(
    dataset: Dataset,
    native_sources: int,
    max_sources: int,
    dataset_name: str,
) -> PaddedSourceDataset:
    return PaddedSourceDataset(
        dataset=dataset,
        native_sources=native_sources,
        max_sources=max_sources,
        dataset_name=dataset_name,
    )


def create_unknown_source_data_loaders(
    batch_size: int,
    two_source_choices: Sequence[str],
    three_source_choices: Sequence[str],
    max_sources: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
    matlab_data_root: Optional[Union[str, Path]] = None,
    public_data_root: Optional[Union[str, Path]] = None,
    seed: int = 42,
    split_strategy: str = "stratified_snr",
) -> Tuple[DataLoader, DataLoader, Dict[float, DataLoader], int]:
    """Build independent K=2/K=3 mixed loaders with a fixed K_max contract."""

    if int(max_sources) != 3:
        raise ValueError("The initial unknown-source mode requires max_sources=3")
    if not two_source_choices or not three_source_choices:
        raise ValueError("Provide at least one two-source and one three-source dataset")
    if public_data_root is not None:
        raise ValueError(
            "Independent unknown-source mode currently supports MATLAB v7.3 synthetic data only"
        )
    if split_strategy not in {"random", "stratified_snr"}:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    train_sets: List[PaddedSourceDataset] = []
    val_sets: List[PaddedSourceDataset] = []
    snr_sets: Dict[float, List[PaddedSourceDataset]] = defaultdict(list)

    source_groups: Iterable[Tuple[int, Sequence[str]]] = (
        (2, two_source_choices),
        (3, three_source_choices),
    )
    for native_sources, choices in source_groups:
        for choice_index, data_choice in enumerate(choices):
            choice_seed = int(seed) + 1000 * native_sources + choice_index
            records = _discover_records(str(data_choice), native_sources, matlab_data_root)
            train_indices, val_indices, test_indices = _split_file_indices(
                records, train_ratio, val_ratio, choice_seed
            )
            fixed_train = LazyH5SignalDataset(
                _records_for_indices(records, train_indices), native_sources
            )
            fixed_val = LazyH5SignalDataset(
                _records_for_indices(records, val_indices), native_sources
            )
            test_records = _records_for_indices(records, test_indices)
            train_sets.append(
                _wrap_split(fixed_train, native_sources, max_sources, str(data_choice))
            )
            val_sets.append(
                _wrap_split(fixed_val, native_sources, max_sources, str(data_choice))
            )
            for snr in sorted({float(record["snr"]) for record in test_records}):
                snr_records = [record for record in test_records if float(record["snr"]) == snr]
                snr_sets[float(snr)].append(
                    _wrap_split(
                        LazyH5SignalDataset(snr_records, native_sources),
                        native_sources,
                        max_sources,
                        str(data_choice),
                    )
                )

    input_size = _validate_lengths([*train_sets, *val_sets])
    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)
    batch_sampler = BalancedSourceBatchSampler(
        train_sets,
        batch_size=int(batch_size),
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
    )
    mixed_snr_loaders = {
        snr: DataLoader(
            ConcatDataset(datasets),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            persistent_workers=int(num_workers) > 0,
        )
        for snr, datasets in sorted(snr_sets.items())
    }
    return train_loader, val_loader, mixed_snr_loaders, input_size
