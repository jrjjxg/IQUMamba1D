"""Auditable KU-TII-inspired high-SINR round-trip augmentation helpers.

The KU-TII ICASSP paper reports 22,000 additional CommSignal2 examples made
by converting high-SNR, zero-BER waveforms to bits and remodulating the bits.
It does not release the CommSignal2 codec, the qualifying-SINR rule, or the
generated examples.  This module therefore implements the reproducible part
without pretending to decode an undocumented CommSignal2 waveform:

* CommSignal2 raw frames are used as the official *interference* source;
* the known challenge SOI (QPSK or OFDM-QPSK) is generated and hard demodulated;
* only zero-BER high-SINR mixtures are retained; and
* their recovered SOI bits are remodulated into the stored supervision target.

The scalable format is a directory of memory-mappable ``.npy`` arrays.  It
lets a 22,000-example long-frame bank be read one mini-batch at a time during
training instead of loading tens of gigabytes into every data-loader worker.
``.npz`` support is kept for small portable banks and compatibility tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .protocol import (
    OFDM_DATA_SUBCARRIERS,
    OFDM_SYMBOL_LENGTH,
    QPSK_BITS_PER_SYMBOL,
    QPSK_SAMPLES_PER_SYMBOL,
    MixtureBatch,
    demodulate_soi,
    generate_mixture_batch,
    modulate_ofdm_qpsk_bits,
    modulate_qpsk_bits,
)


PAIR_BANK_FORMAT_VERSION = 1
PAIR_BANK_MANIFEST = "manifest.json"
PAIR_ARRAY_NAMES = (
    "mixtures",
    "targets",
    "bits",
    "nominal_sinr_db",
    "actual_sinr_db",
    "interference_scale",
    "phase_radians",
)


@dataclass(frozen=True)
class BitRoundTripResult:
    """Selected zero-BER examples and their regenerated waveforms."""

    waveforms: np.ndarray
    bits: np.ndarray
    bit_error_rate: np.ndarray
    selected_indices: np.ndarray


def _as_complex_batch(waveforms: np.ndarray) -> np.ndarray:
    values = np.asarray(waveforms)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError(
            "waveforms must be a complex array with shape (batch, samples), "
            f"got {values.shape}"
        )
    return np.ascontiguousarray(values.astype(np.complex64, copy=False))


def _normalize_soi_type(soi_type: str) -> str:
    normalized = str(soi_type).upper()
    if normalized not in {"QPSK", "OFDMQPSK"}:
        raise ValueError(f"Unsupported SOI type '{soi_type}'")
    return normalized


def _remodulate_soi_bits(soi_type: str, bits: np.ndarray) -> np.ndarray:
    normalized = _normalize_soi_type(soi_type)
    if normalized == "QPSK":
        return modulate_qpsk_bits(bits)
    return modulate_ofdm_qpsk_bits(bits)


def _expected_bit_width(soi_type: str, frame_length: int) -> int:
    normalized = _normalize_soi_type(soi_type)
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if normalized == "QPSK":
        if frame_length % QPSK_SAMPLES_PER_SYMBOL:
            raise ValueError("QPSK frame_length must be divisible by 16")
        return frame_length // QPSK_SAMPLES_PER_SYMBOL * QPSK_BITS_PER_SYMBOL
    if frame_length % OFDM_SYMBOL_LENGTH:
        raise ValueError("OFDMQPSK frame_length must be divisible by 80")
    return frame_length // OFDM_SYMBOL_LENGTH * OFDM_DATA_SUBCARRIERS * QPSK_BITS_PER_SYMBOL


def _as_float_vector(value: np.ndarray, name: str, count: int) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or vector.shape[0] != count:
        raise ValueError(f"{name} must have shape ({count},), got {vector.shape}")
    if not np.issubdtype(vector.dtype, np.floating):
        raise ValueError(f"{name} must have a floating-point dtype, got {vector.dtype}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float32, copy=False)


@dataclass(frozen=True)
class RoundTripPairBank:
    """Validated complete `(mixture, target, bits)` round-trip training pairs.

    Arrays may be ordinary NumPy arrays or read-only memory maps loaded from a
    directory-format bank.  Validation intentionally avoids scanning every bit
    in a large memory map, so loading a 22,000-example bank remains cheap.
    """

    mixtures: np.ndarray
    targets: np.ndarray
    bits: np.ndarray
    nominal_sinr_db: np.ndarray
    actual_sinr_db: np.ndarray
    interference_scale: np.ndarray
    phase_radians: np.ndarray
    soi_type: str | None = None
    interference_type: str = "CommSignal2"

    def __post_init__(self) -> None:
        mixtures = _as_complex_batch(self.mixtures)
        targets = _as_complex_batch(self.targets)
        if mixtures.shape != targets.shape:
            raise ValueError("mixtures and targets must have matching shapes")
        bits = np.asarray(self.bits)
        if bits.ndim != 2 or bits.shape[0] != mixtures.shape[0]:
            raise ValueError("bits must have shape (num_examples, num_bits)")
        if bits.dtype != np.uint8:
            raise ValueError(f"bits must use uint8, got {bits.dtype}")
        if mixtures.shape[0] == 0:
            raise ValueError("Round-trip pair banks must contain at least one example")
        count = int(mixtures.shape[0])
        normalized_soi = None if self.soi_type is None else _normalize_soi_type(self.soi_type)
        if normalized_soi is not None:
            expected_width = _expected_bit_width(normalized_soi, mixtures.shape[1])
            if bits.shape[1] != expected_width:
                raise ValueError(
                    "Round-trip pair bit width does not match the declared SOI protocol: "
                    f"{bits.shape[1]} vs {expected_width}"
                )
        object.__setattr__(self, "mixtures", mixtures)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(
            self,
            "nominal_sinr_db",
            _as_float_vector(self.nominal_sinr_db, "nominal_sinr_db", count),
        )
        object.__setattr__(
            self,
            "actual_sinr_db",
            _as_float_vector(self.actual_sinr_db, "actual_sinr_db", count),
        )
        object.__setattr__(
            self,
            "interference_scale",
            _as_float_vector(self.interference_scale, "interference_scale", count),
        )
        object.__setattr__(
            self,
            "phase_radians",
            _as_float_vector(self.phase_radians, "phase_radians", count),
        )
        object.__setattr__(self, "soi_type", normalized_soi)
        object.__setattr__(self, "interference_type", str(self.interference_type))

    @property
    def count(self) -> int:
        return int(self.mixtures.shape[0])

    @property
    def frame_length(self) -> int:
        return int(self.mixtures.shape[1])

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        expected_soi_type: str | None = None,
        frame_length: int | None = None,
    ) -> MixtureBatch:
        """Sample complete stored pairs with replacement for online training."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if expected_soi_type is not None and self.soi_type is not None:
            if self.soi_type != _normalize_soi_type(expected_soi_type):
                raise ValueError(
                    "Round-trip pair bank SOI type does not match the training case: "
                    f"{self.soi_type} vs {expected_soi_type}"
                )
        if frame_length is not None and self.frame_length != int(frame_length):
            raise ValueError(
                "Round-trip pair bank frame length does not match the training case: "
                f"{self.frame_length} vs {frame_length}"
            )
        indices = rng.integers(0, self.count, size=batch_size, endpoint=False)
        selected_bits = np.ascontiguousarray(self.bits[indices].astype(np.uint8, copy=False))
        if np.any((selected_bits != 0) & (selected_bits != 1)):
            raise ValueError("Round-trip pair bank contains non-binary bits")
        return MixtureBatch(
            mixture=np.ascontiguousarray(self.mixtures[indices].astype(np.complex64, copy=False)),
            target=np.ascontiguousarray(self.targets[indices].astype(np.complex64, copy=False)),
            bits=selected_bits,
            nominal_sinr_db=np.ascontiguousarray(
                self.nominal_sinr_db[indices].astype(np.float32, copy=False)
            ),
            actual_sinr_db=np.ascontiguousarray(
                self.actual_sinr_db[indices].astype(np.float32, copy=False)
            ),
            interference_scale=np.ascontiguousarray(
                self.interference_scale[indices].astype(np.float32, copy=False)
            ),
            phase_radians=np.ascontiguousarray(
                self.phase_radians[indices].astype(np.float32, copy=False)
            ),
        )

    def close(self) -> None:
        """Release directory-bank memory maps, which matters on Windows."""

        for value in (
            self.mixtures,
            self.targets,
            self.bits,
            self.nominal_sinr_db,
            self.actual_sinr_db,
            self.interference_scale,
            self.phase_radians,
        ):
            current = value
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                mapping = getattr(current, "_mmap", None)
                if mapping is not None:
                    mapping.close()
                    break
                current = getattr(current, "base", None)


class RoundTripPairBankStore:
    """Lazily load a pair bank once per worker without pickling its arrays."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._bank: RoundTripPairBank | None = None

    @property
    def bank(self) -> RoundTripPairBank:
        if self._bank is None:
            self._bank = load_roundtrip_pair_bank(self.path)
        return self._bank

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        expected_soi_type: str,
        frame_length: int,
    ) -> MixtureBatch:
        bank = self.bank
        if bank.interference_type != "CommSignal2":
            raise ValueError(
                "Round-trip pair bank must declare interference_type='CommSignal2', "
                f"got {bank.interference_type!r}"
            )
        return bank.sample(
            batch_size=batch_size,
            rng=rng,
            expected_soi_type=expected_soi_type,
            frame_length=frame_length,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_bank"] = None
        return state

    def close(self) -> None:
        """Release a lazily loaded pair bank when the caller owns its lifetime."""

        if self._bank is not None:
            self._bank.close()
            self._bank = None


@dataclass(frozen=True)
class RoundTripPairBuildReport:
    """Audit metadata from building one disk-backed pair bank."""

    output_path: Path
    source_interference_path: Path
    soi_type: str
    requested_examples: int
    accepted_examples: int
    attempted_examples: int
    max_attempts: int
    candidate_sinr_db: np.ndarray
    accepted_per_sinr_db: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": PAIR_BANK_FORMAT_VERSION,
            "output_path": str(self.output_path),
            "source_interference_path": str(self.source_interference_path),
            "soi_type": self.soi_type,
            "interference_type": "CommSignal2",
            "requested_examples": self.requested_examples,
            "accepted_examples": self.accepted_examples,
            "attempted_examples": self.attempted_examples,
            "max_attempts": self.max_attempts,
            "candidate_sinr_db": self.candidate_sinr_db.astype(float).tolist(),
            "accepted_per_sinr_db": dict(self.accepted_per_sinr_db),
        }


def regenerate_zero_ber_waveforms(
    waveforms: np.ndarray,
    expected_bits: np.ndarray,
    soi_type: str,
    max_bit_error_rate: float = 0.0,
) -> BitRoundTripResult:
    """Keep examples whose hard-demodulated BER satisfies a threshold.

    The returned waveform is reconstructed from the accepted bit sequence,
    rather than copied from its input. This mirrors the published bit-to-
    waveform round trip while ensuring a nonzero BER input cannot silently
    contaminate a supposedly clean augmentation bank.
    """

    if not 0.0 <= max_bit_error_rate <= 1.0:
        raise ValueError("max_bit_error_rate must be in [0, 1]")
    source = _as_complex_batch(waveforms)
    reference_bits = np.asarray(expected_bits, dtype=np.uint8)
    if reference_bits.ndim != 2 or reference_bits.shape[0] != source.shape[0]:
        raise ValueError("expected_bits must have shape (batch, num_bits)")
    if np.any((reference_bits != 0) & (reference_bits != 1)):
        raise ValueError("expected_bits must contain only zeros and ones")

    recovered_bits, _ = demodulate_soi(soi_type, source)
    if recovered_bits.shape != reference_bits.shape:
        raise ValueError(
            "Demodulated bit shape does not match expected_bits: "
            f"{recovered_bits.shape} vs {reference_bits.shape}"
        )
    bit_error_rate = np.mean(recovered_bits != reference_bits, axis=1)
    selected_indices = np.flatnonzero(bit_error_rate <= max_bit_error_rate)
    selected_bits = recovered_bits[selected_indices]
    regenerated = _remodulate_soi_bits(soi_type, selected_bits)
    if regenerated.shape[1] != source.shape[1]:
        raise ValueError(
            "The selected codec does not preserve waveform length: "
            f"{regenerated.shape[1]} vs {source.shape[1]}"
        )
    return BitRoundTripResult(
        waveforms=np.ascontiguousarray(regenerated.astype(np.complex64, copy=False)),
        bits=np.ascontiguousarray(selected_bits.astype(np.uint8, copy=False)),
        bit_error_rate=np.ascontiguousarray(bit_error_rate.astype(np.float32, copy=False)),
        selected_indices=np.ascontiguousarray(selected_indices.astype(np.int64, copy=False)),
    )


def save_roundtrip_bank(path: str | Path, result: BitRoundTripResult) -> Path:
    """Save reconstructed waveforms as a compact ``.npy`` interference bank.

    This is retained for backwards compatibility with the original
    ``--commsignal2-augmentation-path`` interface, which replaces only raw
    interference crops. Prefer :func:`build_commsignal2_roundtrip_pair_bank`
    for the paired mixture/target augmentation used by the new training path.
    """

    destination = Path(path)
    if destination.suffix.lower() != ".npy":
        raise ValueError("Round-trip augmentation banks must use the .npy extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, result.waveforms, allow_pickle=False)
    return destination


def save_roundtrip_pair_bank(path: str | Path, bank: RoundTripPairBank) -> Path:
    """Save a small portable pair bank as an unpickled ``.npz`` archive.

    A directory-format bank is preferable for the published 22,000-example
    scale because ZIP archives cannot be memory-mapped for random mini-batch
    reads. This portable form is useful for tests and small experiments.
    """

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Portable round-trip pair banks must use the .npz extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        format_version=np.asarray(PAIR_BANK_FORMAT_VERSION, dtype=np.int64),
        mixtures=bank.mixtures,
        targets=bank.targets,
        bits=bank.bits,
        nominal_sinr_db=bank.nominal_sinr_db,
        actual_sinr_db=bank.actual_sinr_db,
        interference_scale=bank.interference_scale,
        phase_radians=bank.phase_radians,
        soi_type=np.asarray("" if bank.soi_type is None else bank.soi_type),
        interference_type=np.asarray(bank.interference_type),
    )
    return destination


def _load_portable_pair_bank(path: Path) -> RoundTripPairBank:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in PAIR_ARRAY_NAMES if name not in archive.files]
        if missing:
            raise ValueError(f"Portable pair bank {path} is missing arrays: {missing}")
        if "format_version" not in archive.files:
            raise ValueError(f"Portable pair bank {path} has no format_version")
        version = int(np.asarray(archive["format_version"]).item())
        if version != PAIR_BANK_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported portable pair bank version {version}; "
                f"expected {PAIR_BANK_FORMAT_VERSION}"
            )
        soi_type = None
        if "soi_type" in archive.files:
            value = str(np.asarray(archive["soi_type"]).item())
            soi_type = value or None
        interference_type = "CommSignal2"
        if "interference_type" in archive.files:
            interference_type = str(np.asarray(archive["interference_type"]).item())
        return RoundTripPairBank(
            mixtures=archive["mixtures"],
            targets=archive["targets"],
            bits=archive["bits"],
            nominal_sinr_db=archive["nominal_sinr_db"],
            actual_sinr_db=archive["actual_sinr_db"],
            interference_scale=archive["interference_scale"],
            phase_radians=archive["phase_radians"],
            soi_type=soi_type,
            interference_type=interference_type,
        )


def _load_directory_pair_bank(path: Path) -> RoundTripPairBank:
    manifest_path = path / PAIR_BANK_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Round-trip pair-bank manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("format_version", -1)) != PAIR_BANK_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported round-trip pair bank version in {manifest_path}: "
            f"{manifest.get('format_version')}"
        )
    if not bool(manifest.get("complete", False)):
        raise RuntimeError(
            f"Round-trip pair bank {path} is incomplete. Rebuild it with a new output directory."
        )
    arrays: dict[str, np.ndarray] = {}
    for name in PAIR_ARRAY_NAMES:
        array_path = path / f"{name}.npy"
        if not array_path.is_file():
            raise FileNotFoundError(f"Round-trip pair bank {path} is missing {array_path.name}")
        arrays[name] = np.load(array_path, mmap_mode="r", allow_pickle=False)
    bank = RoundTripPairBank(
        mixtures=arrays["mixtures"],
        targets=arrays["targets"],
        bits=arrays["bits"],
        nominal_sinr_db=arrays["nominal_sinr_db"],
        actual_sinr_db=arrays["actual_sinr_db"],
        interference_scale=arrays["interference_scale"],
        phase_radians=arrays["phase_radians"],
        soi_type=manifest.get("soi_type"),
        interference_type=manifest.get("interference_type", "CommSignal2"),
    )
    expected_count = int(manifest.get("accepted_examples", -1))
    if bank.count != expected_count:
        raise ValueError(
            f"Round-trip pair bank count {bank.count} does not match manifest {expected_count}"
        )
    expected_length = int(manifest.get("frame_length", -1))
    if bank.frame_length != expected_length:
        raise ValueError(
            "Round-trip pair bank frame length does not match its manifest: "
            f"{bank.frame_length} vs {expected_length}"
        )
    return bank


def load_roundtrip_pair_bank(path: str | Path) -> RoundTripPairBank:
    """Load a portable ``.npz`` or scalable directory-format pair bank."""

    source = Path(path)
    if source.is_dir():
        return _load_directory_pair_bank(source)
    if not source.is_file():
        raise FileNotFoundError(f"Round-trip pair bank not found: {source}")
    if source.suffix.lower() != ".npz":
        raise ValueError(
            "Round-trip pair banks must be a directory or an .npz archive, "
            f"got {source}"
        )
    return _load_portable_pair_bank(source)


def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
    with (directory / PAIR_BANK_MANIFEST).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _open_pair_bank_arrays(
    output_dir: Path,
    num_examples: int,
    frame_length: int,
    bit_width: int,
) -> dict[str, np.memmap]:
    specifications = {
        "mixtures": (np.complex64, (num_examples, frame_length)),
        "targets": (np.complex64, (num_examples, frame_length)),
        "bits": (np.uint8, (num_examples, bit_width)),
        "nominal_sinr_db": (np.float32, (num_examples,)),
        "actual_sinr_db": (np.float32, (num_examples,)),
        "interference_scale": (np.float32, (num_examples,)),
        "phase_radians": (np.float32, (num_examples,)),
    }
    return {
        name: np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for name, (dtype, shape) in specifications.items()
    }


def _validate_candidate_sinr(values: Sequence[float]) -> np.ndarray:
    candidate = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if candidate.size == 0:
        raise ValueError("candidate_sinr_db must contain at least one value")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("candidate_sinr_db must contain only finite values")
    return candidate


def build_commsignal2_roundtrip_pair_bank(
    output_dir: str | Path,
    interference_path: str | Path,
    soi_type: str,
    num_examples: int = 22_000,
    candidate_sinr_db: Sequence[float] = (0.0, 3.0),
    frame_length: int = 40_960,
    batch_size: int = 8,
    max_attempts: int | None = None,
    seed: int = 0,
    max_bit_error_rate: float = 0.0,
) -> RoundTripPairBuildReport:
    """Build a disk-backed zero-BER CommSignal2 pair bank from train raw frames.

    ``output_dir`` must not already exist.  Incomplete output is deliberately
    retained when the acceptance budget is exhausted, making a failed run
    inspectable instead of silently publishing a short bank as complete.
    """

    normalized_soi = _normalize_soi_type(soi_type)
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= max_bit_error_rate <= 1.0:
        raise ValueError("max_bit_error_rate must be in [0, 1]")
    candidate = _validate_candidate_sinr(candidate_sinr_db)
    effective_max_attempts = num_examples * 100 if max_attempts is None else int(max_attempts)
    if effective_max_attempts < num_examples:
        raise ValueError("max_attempts must be at least num_examples")
    destination = Path(output_dir)
    if destination.suffix.lower() == ".npz":
        raise ValueError(
            "Use a new directory for --output. Directory pair banks are memory-mappable; "
            ".npz is only supported for small portable banks."
        )
    if destination.exists():
        raise FileExistsError(
            f"Round-trip pair-bank output already exists: {destination}. "
            "Choose a new directory to avoid overwriting an existing bank."
        )

    # Keep this import local: datasets can use RoundTripPairBankStore without
    # creating an import cycle while this builder still reuses its HDF5 loader.
    from .datasets import RawInterferenceBank

    source_path = Path(interference_path)
    raw_bank = RawInterferenceBank(source_path)
    raw_frames = raw_bank.frames
    if raw_frames.shape[1] < frame_length:
        raise ValueError(
            "CommSignal2 raw frames are shorter than the requested frame length: "
            f"{raw_frames.shape[1]} < {frame_length}"
        )

    bit_width = _expected_bit_width(normalized_soi, frame_length)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False, exist_ok=False)
    arrays = _open_pair_bank_arrays(destination, num_examples, frame_length, bit_width)
    partial_manifest: dict[str, object] = {
        "format_version": PAIR_BANK_FORMAT_VERSION,
        "complete": False,
        "soi_type": normalized_soi,
        "interference_type": "CommSignal2",
        "frame_length": int(frame_length),
        "requested_examples": int(num_examples),
        "accepted_examples": 0,
        "attempted_examples": 0,
        "max_attempts": int(effective_max_attempts),
        "candidate_sinr_db": candidate.astype(float).tolist(),
        "max_bit_error_rate": float(max_bit_error_rate),
        "source_interference_path": str(source_path.resolve()),
    }
    _write_manifest(destination, partial_manifest)

    rng = np.random.default_rng(seed)
    attempted = 0
    accepted = 0
    accepted_per_sinr = {f"{float(value):g}": 0 for value in np.unique(candidate)}
    try:
        while accepted < num_examples and attempted < effective_max_attempts:
            count = min(batch_size, effective_max_attempts - attempted)
            nominal_sinr = rng.choice(candidate, size=count).astype(np.float32, copy=False)
            generated = generate_mixture_batch(
                soi_type=normalized_soi,
                interference_frames=raw_frames,
                batch_size=count,
                frame_length=frame_length,
                nominal_sinr_db=nominal_sinr,
                rng=rng,
            )
            attempted += count
            recovered_bits, _ = demodulate_soi(normalized_soi, generated.mixture)
            if recovered_bits.shape != generated.bits.shape:
                raise RuntimeError(
                    "Protocol demodulator returned an unexpected bit shape while building "
                    f"the round-trip bank: {recovered_bits.shape} vs {generated.bits.shape}"
                )
            bit_error_rate = np.mean(recovered_bits != generated.bits, axis=1)
            selected = np.flatnonzero(bit_error_rate <= max_bit_error_rate)
            if selected.size == 0:
                continue
            kept_count = min(int(selected.size), num_examples - accepted)
            selected = selected[:kept_count]
            reconstructed_target = _remodulate_soi_bits(normalized_soi, recovered_bits[selected])
            if reconstructed_target.shape != generated.target[selected].shape:
                raise RuntimeError(
                    "Round-trip remodulation changed the target frame shape: "
                    f"{reconstructed_target.shape} vs {generated.target[selected].shape}"
                )
            start = accepted
            stop = accepted + kept_count
            arrays["mixtures"][start:stop] = generated.mixture[selected]
            arrays["targets"][start:stop] = reconstructed_target
            arrays["bits"][start:stop] = recovered_bits[selected]
            arrays["nominal_sinr_db"][start:stop] = generated.nominal_sinr_db[selected]
            arrays["actual_sinr_db"][start:stop] = generated.actual_sinr_db[selected]
            arrays["interference_scale"][start:stop] = generated.interference_scale[selected]
            arrays["phase_radians"][start:stop] = generated.phase_radians[selected]
            for value, value_count in zip(
                *np.unique(generated.nominal_sinr_db[selected], return_counts=True)
            ):
                key = f"{float(value):g}"
                accepted_per_sinr[key] = accepted_per_sinr.get(key, 0) + int(value_count)
            accepted = stop
    except Exception:
        partial_manifest.update(
            {
                "accepted_examples": int(accepted),
                "attempted_examples": int(attempted),
                "accepted_per_sinr_db": accepted_per_sinr,
            }
        )
        _write_manifest(destination, partial_manifest)
        for array in arrays.values():
            array.flush()
        raise

    for array in arrays.values():
        array.flush()
    if accepted != num_examples:
        partial_manifest.update(
            {
                "accepted_examples": int(accepted),
                "attempted_examples": int(attempted),
                "accepted_per_sinr_db": accepted_per_sinr,
            }
        )
        _write_manifest(destination, partial_manifest)
        raise RuntimeError(
            "Round-trip pair build exhausted its acceptance budget: "
            f"accepted {accepted}/{num_examples} zero-BER examples after {attempted} attempts. "
            f"The incomplete bank remains at {destination}. Increase --max-attempts, "
            "raise candidate SINR, or use a different seed."
        )

    report = RoundTripPairBuildReport(
        output_path=destination.resolve(),
        source_interference_path=source_path.resolve(),
        soi_type=normalized_soi,
        requested_examples=int(num_examples),
        accepted_examples=int(accepted),
        attempted_examples=int(attempted),
        max_attempts=int(effective_max_attempts),
        candidate_sinr_db=candidate.copy(),
        accepted_per_sinr_db=accepted_per_sinr,
    )
    completed_manifest = report.to_dict()
    completed_manifest.update(
        {
            "complete": True,
            "frame_length": int(frame_length),
            "max_bit_error_rate": float(max_bit_error_rate),
            "bit_width": int(bit_width),
        }
    )
    _write_manifest(destination, completed_manifest)
    return report
