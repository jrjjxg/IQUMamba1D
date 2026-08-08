"""Datasets and artifact adapters for the ICASSP 2024 RF Challenge."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .protocol import (
    FRAME_LENGTH,
    SINR_DB_VALUES,
    MixtureBatch,
    build_mixtures,
    complex_to_iq,
    crop_interference_batch,
    generate_mixture_batch,
    generate_soi,
    iq_to_complex,
)

if TYPE_CHECKING:
    from .augmentation import RoundTripPairBankStore


def _h5_to_complex(value: np.ndarray) -> np.ndarray:
    """Read normal, I/Q, or compound-complex HDF5 arrays as complex64."""

    array = np.asarray(value)
    if array.dtype.fields:
        names = tuple(array.dtype.fields)
        candidate_pairs = (("r", "i"), ("real", "imag"), ("re", "im"))
        for real_name, imag_name in candidate_pairs:
            if real_name in names and imag_name in names:
                return (array[real_name] + 1j * array[imag_name]).astype(np.complex64)
        if len(names) == 2:
            return (array[names[0]] + 1j * array[names[1]]).astype(np.complex64)
        raise ValueError(f"Unsupported compound complex HDF5 dtype fields: {names}")
    if np.iscomplexobj(array):
        return array.astype(np.complex64, copy=False)
    if array.ndim == 3 and array.shape[-1] == 2:
        return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)
    raise ValueError(
        "Expected a complex HDF5 dataset or an I/Q trailing dimension, "
        f"got dtype={array.dtype}, shape={array.shape}"
    )


class RawInterferenceBank:
    """Lazily load one official raw interference bank.

    The official files contain multiple long recordings. The complete bank is
    intentionally cached per worker: random crop selection is frequent during
    training and HDF5 random slicing for each sample is much slower.

    In addition to the official HDF5 files, ``.npy`` and ``.npz`` arrays are
    accepted for the public KU-TII-style CommSignal2 augmentation. This makes
    it possible to mix an externally generated high-SNR round-trip bank into
    online training without materializing a full mixture dataset.
    """

    def __init__(self, path: str | Path, dataset_key: str = "dataset") -> None:
        self.path = Path(path)
        self.dataset_key = dataset_key
        self._frames: np.ndarray | None = None

    @property
    def frames(self) -> np.ndarray:
        if self._frames is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"Interference HDF5 not found: {self.path}")
            suffix = self.path.suffix.lower()
            if suffix == ".npy":
                frames = _h5_to_complex(np.load(self.path, allow_pickle=False))
            elif suffix == ".npz":
                with np.load(self.path, allow_pickle=False) as archive:
                    if self.dataset_key in archive:
                        value = archive[self.dataset_key]
                    elif len(archive.files) == 1:
                        value = archive[archive.files[0]]
                    else:
                        raise KeyError(
                            f"NPZ file {self.path} has no '{self.dataset_key}' array; "
                            f"available arrays: {archive.files}"
                        )
                    frames = _h5_to_complex(value)
            else:
                with h5py.File(self.path, "r") as handle:
                    if self.dataset_key not in handle:
                        raise KeyError(
                            f"HDF5 file {self.path} has no '{self.dataset_key}' dataset; "
                            f"available keys: {list(handle.keys())}"
                        )
                    frames = _h5_to_complex(np.asarray(handle[self.dataset_key]))
            if frames.ndim != 2:
                raise ValueError(
                    "The RF Challenge interference dataset must have shape (N, L), "
                    f"got {frames.shape} from {self.path}"
                )
            self._frames = np.ascontiguousarray(frames)
        return self._frames

    def __getstate__(self):
        # Windows DataLoader workers must not inherit/pickle a full cached bank.
        state = self.__dict__.copy()
        state["_frames"] = None
        return state


def resolve_interference_path(
    data_root: str | Path,
    interference_type: str,
    split: Literal["train", "test1"] = "train",
) -> Path:
    """Resolve official data-root layouts without assuming the current directory."""

    root = Path(data_root)
    if split == "train":
        filename = f"{interference_type}_raw_data.h5"
        folder = "interferenceset_frame"
    elif split == "test1":
        filename = f"{interference_type}_test1_raw_data.h5"
        folder = "testset1_frame"
    else:
        raise ValueError(f"Unsupported split '{split}'")

    candidates = (
        root / folder / filename,
        root / "dataset" / folder / filename,
        root / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _generate_online_mixture_batch(
    primary_bank: RawInterferenceBank,
    soi_type: str,
    batch_size: int,
    frame_length: int,
    nominal_sinr_db: float | np.ndarray,
    rng: np.random.Generator,
    augmentation_bank: RawInterferenceBank | None = None,
    augmentation_probability: float = 0.0,
    roundtrip_pair_bank: RoundTripPairBankStore | None = None,
    roundtrip_pair_probability: float = 0.0,
) -> MixtureBatch:
    """Generate one online batch, optionally mixing both augmentation formats.

    The public KU-TII paper reports additional CommSignal2 examples generated
    from high-SNR bit round trips. Their codec and exact generated frames were
    not released. ``augmentation_bank`` retains the legacy behavior of
    replacing only an interference crop. ``roundtrip_pair_bank`` instead
    replaces complete stored `(mixture, target, bits)` examples after the
    known SOI has passed the zero-BER round trip.
    """

    if roundtrip_pair_bank is not None and roundtrip_pair_probability > 0.0:
        if not 0.0 <= roundtrip_pair_probability <= 1.0:
            raise ValueError("roundtrip_pair_probability must be in [0, 1]")
        use_pair = rng.random(batch_size) < roundtrip_pair_probability
        pair_indices = np.flatnonzero(use_pair)
        online_indices = np.flatnonzero(~use_pair)
        if pair_indices.size:
            pair_batch = roundtrip_pair_bank.sample(
                batch_size=int(pair_indices.size),
                rng=rng,
                expected_soi_type=soi_type,
                frame_length=frame_length,
            )
            if not online_indices.size:
                return pair_batch
            nominal_values = np.broadcast_to(
                np.asarray(nominal_sinr_db, dtype=np.float32), (batch_size,)
            ).astype(np.float32, copy=False)
            online_batch = _generate_online_mixture_batch(
                primary_bank=primary_bank,
                soi_type=soi_type,
                batch_size=int(online_indices.size),
                frame_length=frame_length,
                nominal_sinr_db=nominal_values[online_indices],
                rng=rng,
                augmentation_bank=augmentation_bank,
                augmentation_probability=augmentation_probability,
            )
            if pair_batch.bits.shape[1] != online_batch.bits.shape[1]:
                raise ValueError(
                    "Round-trip pair-bank bit width does not match online generated samples: "
                    f"{pair_batch.bits.shape[1]} vs {online_batch.bits.shape[1]}"
                )
            mixture = np.empty((batch_size, frame_length), dtype=np.complex64)
            target = np.empty_like(mixture)
            bits = np.empty((batch_size, online_batch.bits.shape[1]), dtype=np.uint8)
            nominal = np.empty(batch_size, dtype=np.float32)
            actual = np.empty(batch_size, dtype=np.float32)
            scale = np.empty(batch_size, dtype=np.float32)
            phase = np.empty(batch_size, dtype=np.float32)
            mixture[online_indices] = online_batch.mixture
            target[online_indices] = online_batch.target
            bits[online_indices] = online_batch.bits
            nominal[online_indices] = online_batch.nominal_sinr_db
            actual[online_indices] = online_batch.actual_sinr_db
            scale[online_indices] = online_batch.interference_scale
            phase[online_indices] = online_batch.phase_radians
            mixture[pair_indices] = pair_batch.mixture
            target[pair_indices] = pair_batch.target
            bits[pair_indices] = pair_batch.bits
            nominal[pair_indices] = pair_batch.nominal_sinr_db
            actual[pair_indices] = pair_batch.actual_sinr_db
            scale[pair_indices] = pair_batch.interference_scale
            phase[pair_indices] = pair_batch.phase_radians
            return MixtureBatch(
                mixture=mixture,
                target=target,
                bits=bits,
                nominal_sinr_db=nominal,
                actual_sinr_db=actual,
                interference_scale=scale,
                phase_radians=phase,
            )

    if augmentation_bank is None or augmentation_probability <= 0.0:
        return generate_mixture_batch(
            soi_type=soi_type,
            interference_frames=primary_bank.frames,
            batch_size=batch_size,
            frame_length=frame_length,
            nominal_sinr_db=nominal_sinr_db,
            rng=rng,
        )

    generated = generate_soi(soi_type, batch_size, frame_length, rng)
    use_augmentation = rng.random(batch_size) < augmentation_probability
    interference = np.empty_like(generated.waveform)
    primary_indices = np.flatnonzero(~use_augmentation)
    augmented_indices = np.flatnonzero(use_augmentation)
    if primary_indices.size:
        interference[primary_indices] = crop_interference_batch(
            primary_bank.frames,
            batch_size=int(primary_indices.size),
            frame_length=frame_length,
            rng=rng,
        )
    if augmented_indices.size:
        interference[augmented_indices] = crop_interference_batch(
            augmentation_bank.frames,
            batch_size=int(augmented_indices.size),
            frame_length=frame_length,
            rng=rng,
        )
    mixed = build_mixtures(generated.waveform, interference, nominal_sinr_db, rng)
    return MixtureBatch(
        mixture=mixed.mixture,
        target=generated.waveform,
        bits=generated.bits,
        nominal_sinr_db=mixed.nominal_sinr_db,
        actual_sinr_db=mixed.actual_sinr_db,
        interference_scale=mixed.interference_scale,
        phase_radians=mixed.phase_radians,
    )


class RFChallengeOnlineDataset(Dataset):
    """Generate deterministic supervised RF Challenge mixtures on demand.

    Each sample uses one raw interference crop and a freshly generated SOI.
    This is distribution-equivalent to the official synthetic generator while
    avoiding the official starter's multi-terabyte materialized dataset.
    """

    def __init__(
        self,
        interference_path: str | Path,
        soi_type: str,
        interference_type: str,
        samples_per_epoch: int,
        frame_length: int = FRAME_LENGTH,
        sinr_mode: Literal["discrete", "continuous"] = "discrete",
        seed: int = 0,
        return_bits: bool = False,
        augmentation_interference_path: str | Path | None = None,
        augmentation_probability: float = 0.0,
        roundtrip_pair_path: str | Path | None = None,
        roundtrip_pair_probability: float = 0.0,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if sinr_mode not in {"discrete", "continuous"}:
            raise ValueError("sinr_mode must be 'discrete' or 'continuous'")
        if not 0.0 <= augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability must be in [0, 1]")
        if augmentation_probability > 0.0 and augmentation_interference_path is None:
            raise ValueError(
                "augmentation_interference_path is required when augmentation_probability is positive"
            )
        if not 0.0 <= roundtrip_pair_probability <= 1.0:
            raise ValueError("roundtrip_pair_probability must be in [0, 1]")
        if roundtrip_pair_probability > 0.0 and roundtrip_pair_path is None:
            raise ValueError(
                "roundtrip_pair_path is required when roundtrip_pair_probability is positive"
            )
        if roundtrip_pair_path is not None and str(interference_type) != "CommSignal2":
            raise ValueError("roundtrip pair augmentation is supported only for CommSignal2")
        self.bank = RawInterferenceBank(interference_path)
        self.augmentation_bank = (
            None
            if augmentation_interference_path is None
            else RawInterferenceBank(augmentation_interference_path)
        )
        self.augmentation_probability = float(augmentation_probability)
        if roundtrip_pair_path is None:
            self.roundtrip_pair_bank = None
        else:
            # Import lazily so augmentation can reuse RawInterferenceBank when
            # it builds a bank without creating a module import cycle.
            from .augmentation import RoundTripPairBankStore

            self.roundtrip_pair_bank = RoundTripPairBankStore(roundtrip_pair_path)
        self.roundtrip_pair_probability = float(roundtrip_pair_probability)
        self.soi_type = str(soi_type)
        self.interference_type = str(interference_type)
        self.samples_per_epoch = int(samples_per_epoch)
        self.frame_length = int(frame_length)
        self.sinr_mode = sinr_mode
        self.seed = int(seed)
        self.return_bits = bool(return_bits)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence((self.seed, self.epoch, int(index)))
        return np.random.default_rng(sequence)

    def _sample_sinr(self, rng: np.random.Generator) -> float:
        if self.sinr_mode == "discrete":
            return float(rng.choice(SINR_DB_VALUES))
        # Mirrors starter code: -36 * U + 3 gives [-33, 3).
        return float(-36.0 * rng.uniform() + 3.0)

    def __getitem__(self, index: int):
        rng = self._rng_for_index(index)
        batch = _generate_online_mixture_batch(
            primary_bank=self.bank,
            soi_type=self.soi_type,
            batch_size=1,
            frame_length=self.frame_length,
            nominal_sinr_db=self._sample_sinr(rng),
            rng=rng,
            augmentation_bank=self.augmentation_bank,
            augmentation_probability=self.augmentation_probability,
            roundtrip_pair_bank=self.roundtrip_pair_bank,
            roundtrip_pair_probability=self.roundtrip_pair_probability,
        )
        mixture = torch.from_numpy(complex_to_iq(batch.mixture)[0]).contiguous()
        target = torch.from_numpy(complex_to_iq(batch.target)[0]).contiguous()
        sinr = torch.tensor(batch.nominal_sinr_db[0], dtype=torch.float32)
        if self.return_bits:
            return mixture, target, sinr, torch.from_numpy(batch.bits[0].copy())
        return mixture, target, sinr


class RFChallengeOnlineBatchDataset(IterableDataset):
    """Vectorized pre-batched online generator for practical long-frame training.

    ``RFChallengeOnlineDataset`` is useful for ordinary Dataset tooling. This
    iterable variant is the default for the command-line trainer: it generates
    one complete mini-batch at a time, so RRC filtering runs once on a batch
    rather than once per 40,960-sample frame.
    """

    prebatched = True

    def __init__(
        self,
        interference_path: str | Path,
        soi_type: str,
        interference_type: str,
        samples_per_epoch: int,
        batch_size: int,
        frame_length: int = FRAME_LENGTH,
        sinr_mode: Literal["discrete", "continuous"] = "discrete",
        seed: int = 0,
        augmentation_interference_path: str | Path | None = None,
        augmentation_probability: float = 0.0,
        roundtrip_pair_path: str | Path | None = None,
        roundtrip_pair_probability: float = 0.0,
    ) -> None:
        if samples_per_epoch <= 0 or batch_size <= 0:
            raise ValueError("samples_per_epoch and batch_size must be positive")
        if sinr_mode not in {"discrete", "continuous"}:
            raise ValueError("sinr_mode must be 'discrete' or 'continuous'")
        if not 0.0 <= augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability must be in [0, 1]")
        if augmentation_probability > 0.0 and augmentation_interference_path is None:
            raise ValueError(
                "augmentation_interference_path is required when augmentation_probability is positive"
            )
        if not 0.0 <= roundtrip_pair_probability <= 1.0:
            raise ValueError("roundtrip_pair_probability must be in [0, 1]")
        if roundtrip_pair_probability > 0.0 and roundtrip_pair_path is None:
            raise ValueError(
                "roundtrip_pair_path is required when roundtrip_pair_probability is positive"
            )
        if roundtrip_pair_path is not None and str(interference_type) != "CommSignal2":
            raise ValueError("roundtrip pair augmentation is supported only for CommSignal2")
        self.bank = RawInterferenceBank(interference_path)
        self.augmentation_bank = (
            None
            if augmentation_interference_path is None
            else RawInterferenceBank(augmentation_interference_path)
        )
        self.augmentation_probability = float(augmentation_probability)
        if roundtrip_pair_path is None:
            self.roundtrip_pair_bank = None
        else:
            from .augmentation import RoundTripPairBankStore

            self.roundtrip_pair_bank = RoundTripPairBankStore(roundtrip_pair_path)
        self.roundtrip_pair_probability = float(roundtrip_pair_probability)
        self.soi_type = str(soi_type)
        self.interference_type = str(interference_type)
        self.samples_per_epoch = int(samples_per_epoch)
        self.batch_size = int(batch_size)
        self.frame_length = int(frame_length)
        self.sinr_mode = sinr_mode
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sinr_values(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if self.sinr_mode == "discrete":
            return rng.choice(SINR_DB_VALUES, size=count).astype(np.float32)
        return (-36.0 * rng.uniform(size=count) + 3.0).astype(np.float32)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        total_batches = len(self)
        for batch_index in range(worker_id, total_batches, worker_count):
            start = batch_index * self.batch_size
            count = min(self.batch_size, self.samples_per_epoch - start)
            sequence = np.random.SeedSequence((self.seed, self.epoch, batch_index))
            rng = np.random.default_rng(sequence)
            batch = _generate_online_mixture_batch(
                primary_bank=self.bank,
                soi_type=self.soi_type,
                batch_size=count,
                frame_length=self.frame_length,
                nominal_sinr_db=self._sinr_values(rng, count),
                rng=rng,
                augmentation_bank=self.augmentation_bank,
                augmentation_probability=self.augmentation_probability,
                roundtrip_pair_bank=self.roundtrip_pair_bank,
                roundtrip_pair_probability=self.roundtrip_pair_probability,
            )
            yield (
                torch.from_numpy(complex_to_iq(batch.mixture)).contiguous(),
                torch.from_numpy(complex_to_iq(batch.target)).contiguous(),
                torch.from_numpy(batch.nominal_sinr_db.copy()),
            )


class RFChallengeArrayDataset(Dataset):
    """Wrap official complex arrays for inference, validation, or local scoring."""

    # A fixed 1,100-frame validation set contains hundreds of MiB of long I/Q
    # arrays. Keep it in the parent process so Windows spawn workers do not
    # serialize a separate full copy for every validation worker.
    force_single_process_loader = True

    def __init__(
        self,
        mixtures: np.ndarray,
        targets: np.ndarray | None = None,
        bits: np.ndarray | None = None,
        nominal_sinr_db: np.ndarray | None = None,
    ) -> None:
        self.mixtures = iq_to_complex(mixtures) if np.asarray(mixtures).ndim == 3 else np.asarray(mixtures)
        self.mixtures = np.ascontiguousarray(self.mixtures.astype(np.complex64, copy=False))
        if self.mixtures.ndim != 2:
            raise ValueError(f"mixtures must have shape (B, L), got {self.mixtures.shape}")
        self.targets = None
        if targets is not None:
            self.targets = iq_to_complex(targets) if np.asarray(targets).ndim == 3 else np.asarray(targets)
            self.targets = np.ascontiguousarray(self.targets.astype(np.complex64, copy=False))
            if self.targets.shape != self.mixtures.shape:
                raise ValueError("targets must have the same shape as mixtures")
        self.bits = None if bits is None else np.asarray(bits, dtype=np.uint8)
        if self.bits is not None and self.bits.shape[0] != self.mixtures.shape[0]:
            raise ValueError("bits and mixtures must have the same batch dimension")
        self.nominal_sinr_db = None if nominal_sinr_db is None else np.asarray(nominal_sinr_db, dtype=np.float32)
        if self.nominal_sinr_db is not None and self.nominal_sinr_db.shape[0] != self.mixtures.shape[0]:
            raise ValueError("nominal_sinr_db and mixtures must have the same batch dimension")

    def __len__(self) -> int:
        return int(self.mixtures.shape[0])

    def __getitem__(self, index: int):
        mixture = torch.from_numpy(complex_to_iq(self.mixtures[index])[0]).contiguous()
        if self.targets is None:
            return mixture
        target = torch.from_numpy(complex_to_iq(self.targets[index])[0]).contiguous()
        sinr = torch.tensor(
            0.0 if self.nominal_sinr_db is None else self.nominal_sinr_db[index],
            dtype=torch.float32,
        )
        if self.bits is None:
            return mixture, target, sinr
        return mixture, target, sinr, torch.from_numpy(self.bits[index].copy())


def generate_example_evaluation_set(
    interference_path: str | Path,
    soi_type: str,
    interference_type: str,
    n_per_sinr: int = 100,
    frame_length: int = FRAME_LENGTH,
    seed: int = 0,
) -> tuple[MixtureBatch, np.ndarray]:
    """Generate a locally scoreable TestSet1Example-compatible evaluation set."""

    if n_per_sinr <= 0:
        raise ValueError("n_per_sinr must be positive")
    bank = RawInterferenceBank(interference_path)
    rng = np.random.default_rng(seed)
    batches = [
        generate_mixture_batch(
            soi_type=soi_type,
            interference_frames=bank.frames,
            batch_size=n_per_sinr,
            frame_length=frame_length,
            nominal_sinr_db=float(sinr),
            rng=rng,
        )
        for sinr in SINR_DB_VALUES
    ]
    result = MixtureBatch(
        mixture=np.concatenate([batch.mixture for batch in batches], axis=0),
        target=np.concatenate([batch.target for batch in batches], axis=0),
        bits=np.concatenate([batch.bits for batch in batches], axis=0),
        nominal_sinr_db=np.concatenate([batch.nominal_sinr_db for batch in batches], axis=0),
        actual_sinr_db=np.concatenate([batch.actual_sinr_db for batch in batches], axis=0),
        interference_scale=np.concatenate([batch.interference_scale for batch in batches], axis=0),
        phase_radians=np.concatenate([batch.phase_radians for batch in batches], axis=0),
    )
    metadata = np.empty((result.mixture.shape[0], 5), dtype=object)
    metadata[:, 0] = result.interference_scale
    metadata[:, 1] = result.nominal_sinr_db
    metadata[:, 2] = result.actual_sinr_db
    metadata[:, 3] = str(soi_type)
    metadata[:, 4] = str(interference_type)
    return result, metadata


def save_example_evaluation_set(
    output_dir: str | Path,
    identifier: str,
    soi_type: str,
    interference_type: str,
    batch: MixtureBatch,
    metadata: np.ndarray,
) -> dict[str, Path]:
    """Save files with the starter kit's TestSet1Example names and payloads."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{identifier}_testmixture_{soi_type}_{interference_type}"
    mixture_path = directory / f"{base}.npy"
    metadata_path = directory / f"{base}_metadata.npy"
    ground_truth_path = directory / f"GroundTruth_{identifier}_Dataset_{soi_type}_{interference_type}.pkl"
    np.save(mixture_path, batch.mixture.astype(np.complex64, copy=False), allow_pickle=False)
    np.save(metadata_path, metadata, allow_pickle=True)
    with ground_truth_path.open("wb") as handle:
        pickle.dump((batch.mixture, batch.target, batch.bits), handle, protocol=4)
    return {
        "mixture": mixture_path,
        "metadata": metadata_path,
        "ground_truth": ground_truth_path,
    }


def load_ground_truth(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the public starter-kit ground-truth pickle payload."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, tuple) or len(payload) < 3:
        raise ValueError(f"Unexpected RF Challenge ground-truth payload in {path}")
    mixture, target, bits = payload[:3]
    mixture = np.ascontiguousarray(np.asarray(mixture, dtype=np.complex64))
    target = np.ascontiguousarray(np.asarray(target, dtype=np.complex64))
    bits = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    if mixture.shape != target.shape or mixture.ndim != 2:
        raise ValueError("Ground-truth mixture and target must be matching (B, L) arrays")
    if bits.shape[0] != mixture.shape[0]:
        raise ValueError("Ground-truth bits and mixture must have matching batch dimensions")
    return mixture, target, bits
