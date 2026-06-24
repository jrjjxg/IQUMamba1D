from collections import defaultdict
import h5py
import numpy as np
import random as _random
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


MATLAB_DATA_CHOICE_ALIASES = {
    "QPSK16APSK-A": "QPSK+16APSK-A",
    "QPSK16APSK-B": "QPSK+16APSK-B",
    "QPSK-16APSK-A": "QPSK+16APSK-A",
    "QPSK-16APSK-B": "QPSK+16APSK-B",
}


def _normalize_matlab_data_choice(data_choice: str) -> str:
    """Normalize Kaggle-safe MATLAB dataset aliases back to canonical names."""
    return MATLAB_DATA_CHOICE_ALIASES.get(data_choice, data_choice)


def _normalize_snr_value(snr_value) -> float:
    """Normalize different SNR value types to python float."""
    if torch.is_tensor(snr_value):
        if snr_value.numel() != 1:
            raise ValueError(f"Expected scalar SNR tensor, got shape {tuple(snr_value.shape)}")
        snr_value = snr_value.item()
    return float(snr_value)


def _matlab_scalar_or_default(h5_file, key: str, default=None):
    """Read a MATLAB v7.3 scalar dataset if present, else return default."""
    if key not in h5_file:
        return default
    arr = np.asarray(h5_file[key])
    if arr.size == 0:
        return default
    return np.asarray(arr).reshape(-1)[0].item()


def _matlab_vector_or_default(h5_file, key: str, default=None):
    """Read a MATLAB v7.3 vector dataset if present, else return default."""
    if key not in h5_file:
        return default
    arr = np.asarray(h5_file[key])
    if arr.size == 0:
        return default
    return np.asarray(arr).reshape(-1).tolist()


def _read_h5_bits_vector(h5_file):
    """Read a flat bit vector from supported MATLAB v7.3 bit files."""
    if 'file_bits' in h5_file:
        key = 'file_bits'
    else:
        bit_keys = [k for k in h5_file.keys() if k.lower().startswith('bit_data')]
        if bit_keys:
            key = sorted(bit_keys)[0]
        else:
            dataset_keys = [k for k in h5_file.keys() if isinstance(h5_file[k], h5py.Dataset)]
            if not dataset_keys:
                raise KeyError("No bit dataset found in HDF5 file")
            key = sorted(dataset_keys)[0]
    return np.array(h5_file[key][:], dtype=np.uint8).reshape(-1)


def _next_multiple(value: int, factor: int) -> int:
    """Return the smallest multiple of factor that is >= value."""
    value = int(value)
    factor = max(1, int(factor))
    return ((value + factor - 1) // factor) * factor


def _pad_frame_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    """Pad frame array (N, L, C) with trailing zeros to target_length."""
    if arr.shape[1] >= target_length:
        return arr[:, :target_length, :]
    pad_width = target_length - arr.shape[1]
    pad = np.zeros((arr.shape[0], pad_width, arr.shape[2]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)

class BaseSignalDataset(Dataset, ABC):
    """Base class for signal datasets"""
    
    def __init__(self, num_sources: int = 2):
        """
        Args:
            num_sources: Number of source signals (2, 3, 4, ...)
        """
        self.num_sources = num_sources
        self.signals = []  # Store all source signals
        self.mixture = []  # Store mixed signals
        self.snrs = []     # Store SNR labels
        self.bits_per_source = []  # Optional: bits per source for BER evaluation
        self.num_samples = 0
        self.frame_length = None
        self.valid_frame_length = None
        self.model_frame_length = None
    
    @abstractmethod
    def _load_data(self, *args, **kwargs):
        """Abstract method for loading data"""
        pass
    
    def _extract_snr_from_path(self, path: str) -> float:
        """Extract SNR value from path"""
        match = re.search(r'SNR=?([+-]?\d+(?:\.\d+)?)dB', path)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Cannot extract SNR information from path {path}") 
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get mixed signal
        mixed_signal = self.mixture[idx]
        mixed_real, mixed_imag = mixed_signal[:, 0], mixed_signal[:, 1]
        input_signal = np.stack([mixed_real, mixed_imag], axis=0)
        
        # Get each source signal and organize as target format
        target_channels = []
        for source_idx in range(self.num_sources):
            signal = self.signals[source_idx][idx]
            signal_real, signal_imag = signal[:, 0], signal[:, 1]
            target_channels.extend([signal_real, signal_imag])
        
        target = np.stack(target_channels, axis=0)  # (2*num_sources, signal_length)
        snr = self.snrs[idx]
        
        return (
            torch.tensor(input_signal, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32), 
            snr,
            *(
                [tuple(
                    torch.tensor(self.bits_per_source[k][idx], dtype=torch.uint8)
                    for k in range(self.num_sources)
                )]
                if self.bits_per_source
                else []
            )
        )


class RandomSignalDataset(Dataset):
    """Lightweight debug dataset that generates random IQ mixtures on-the-fly."""

    def __init__(
        self,
        num_sources: int = 2,
        signal_length: int = 1024,
        num_samples: int = 512,
        snr_db: float = 20.0,
        seed: int = 0,
    ):
        super().__init__()
        self.num_sources = int(num_sources)
        self.signal_length = int(signal_length)
        self.num_samples = int(num_samples)
        self.snr_db = float(snr_db)
        self.gen = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        L = self.signal_length
        K = self.num_sources

        sources = torch.randn((K, 2, L), generator=self.gen, dtype=torch.float32)
        mixture = sources.sum(dim=0)  # (2, L)

        sig_pow = mixture.pow(2).mean()
        snr_lin = 10.0 ** (self.snr_db / 10.0)
        noise_pow = sig_pow / snr_lin
        noise = torch.randn((2, L), generator=self.gen, dtype=torch.float32) * torch.sqrt(noise_pow + 1e-12)
        mixture_noisy = mixture + noise

        target = sources.reshape(K * 2, L)  # (2*K, L)
        return mixture_noisy, target, torch.tensor(self.snr_db, dtype=torch.float32)


class MATLABSignalDataset(BaseSignalDataset):
    """MATLAB generated signal dataset - supports multi-source data in same file"""
    
    def __init__(self, signal_paths: List[str], mixture_paths: List[str], 
                 data_choice: str, num_sources: int = 2):
        """
        Args:
            signal_paths: Target signal path list (each file contains all sources)
            mixture_paths: Mixed signal path list
            data_choice: Data type selection ('QAM', '8PSK', etc.)
            num_sources: Number of source signals
        """
        super().__init__(num_sources)
        self.data_choice = data_choice
        self._frame_counts = []  # Track frame count per file for bits loading
        self._file_meta = []     # Per-file metadata for stream-level BER evaluation
        self._load_data(signal_paths, mixture_paths)
    
    def _load_data(self, signal_paths: List[str], mixture_paths: List[str]):
        """Load MATLAB data - separate multiple sources from same file"""
        # Determine data field name
        name_str = 'ideal_frames'
        mix_name_str = 'mixed_frames'
        
        # Initialize source signal list
        for _ in range(self.num_sources):
            self.signals.append([])
        
        sample_offset = 0
        file_frame_length = None
        file_valid_frame_length = None
        file_model_frame_length = None

        # Load target signals and separate sources
        for file_idx, path in enumerate(signal_paths):
            with h5py.File(path, 'r') as f:
                data = f[name_str][:]
                data = np.transpose(data, (2, 1, 0))  # (B, L, 2*num_sources)
                frame_length_meta = _matlab_scalar_or_default(f, 'frame_length', data.shape[1])
                valid_frame_length_meta = _matlab_scalar_or_default(f, 'valid_frame_length', frame_length_meta)
                samples_per_symbol_meta = _matlab_scalar_or_default(f, 'Fs_sps', None)
                symbols_per_frame_meta = _matlab_scalar_or_default(f, 'symbols_per_frame', None)
                bits_per_symbol_meta = _matlab_scalar_or_default(f, 'bits_per_symbol', None)
                bits_per_frame_meta = _matlab_scalar_or_default(f, 'bits_per_frame', None)
                bits_per_frame_by_source_meta = _matlab_vector_or_default(f, 'bits_per_frame_by_source', None)
                legacy_bits_per_frame_vector_meta = _matlab_vector_or_default(f, 'bits_per_frame', None)
                if (
                    bits_per_frame_by_source_meta is None
                    and legacy_bits_per_frame_vector_meta is not None
                    and len(legacy_bits_per_frame_vector_meta) > 1
                ):
                    bits_per_frame_by_source_meta = legacy_bits_per_frame_vector_meta
                    bits_per_frame_meta = None
                preamble_symbols_per_frame_meta = _matlab_scalar_or_default(f, 'preamble_symbols_per_frame', None)
                payload_symbols_per_frame_meta = _matlab_scalar_or_default(f, 'payload_symbols_per_frame', None)
                payload_bits_per_frame_meta = _matlab_scalar_or_default(f, 'payload_bits_per_frame', None)
                rrc_alpha_meta = _matlab_scalar_or_default(f, 'rrc_alpha', None)
                rrc_span_meta = _matlab_scalar_or_default(f, 'rrc_span', None)

            frame_length_meta = int(frame_length_meta)
            valid_frame_length_meta = int(valid_frame_length_meta)
            model_frame_length_meta = frame_length_meta
            if self.data_choice == '8PSK-A':
                # BER-oriented 8PSK-A variants used 4100 raw samples and padded to
                # 4112 for the model. Restore the original training pipeline by
                # adapting 8PSK-A frames back to 4096 samples at load time.
                model_frame_length_meta = 4096
                valid_frame_length_meta = min(valid_frame_length_meta, model_frame_length_meta)

            if file_frame_length is None:
                file_frame_length = frame_length_meta
                file_valid_frame_length = valid_frame_length_meta
                file_model_frame_length = model_frame_length_meta
            else:
                if frame_length_meta != file_frame_length:
                    raise ValueError(f"Inconsistent frame_length across files: {frame_length_meta} vs {file_frame_length}")
                if valid_frame_length_meta != file_valid_frame_length:
                    raise ValueError(
                        f"Inconsistent valid_frame_length across files: {valid_frame_length_meta} vs {file_valid_frame_length}"
                    )
                if model_frame_length_meta != file_model_frame_length:
                    raise ValueError(
                        f"Inconsistent model_frame_length across files: {model_frame_length_meta} vs {file_model_frame_length}"
                    )
            
            # Check if dimensions are correct
            expected_channels = 2 * self.num_sources
            if data.shape[2] != expected_channels:
                raise ValueError(
                    f"Channel count mismatch in file {path}: "
                    f"Expected {expected_channels} (2*{self.num_sources}), "
                    f"Actual {data.shape[2]}"
                )
            
            # Separate each source
            for source_idx in range(self.num_sources):
                # Extract real and imaginary parts of current source
                real_idx = source_idx * 2
                imag_idx = source_idx * 2 + 1
                source_data = data[:, :, [real_idx, imag_idx]]  # (B, L, 2)
                self.signals[source_idx].append(source_data)
            
            # Extract SNR information
            snr = self._extract_snr_from_path(path)
            n_frames = len(data)
            self.snrs.extend([snr] * n_frames)
            self._frame_counts.append(n_frames)
            self._file_meta.append({
                "file_idx": file_idx,
                "signal_path": str(path),
                "mixture_path": str(mixture_paths[file_idx]) if file_idx < len(mixture_paths) else None,
                "snr": float(snr),
                "start": sample_offset,
                "end": sample_offset + n_frames,
                "num_frames": n_frames,
                "frame_length": frame_length_meta,
                "valid_frame_length": valid_frame_length_meta,
                "model_frame_length": model_frame_length_meta,
                "samples_per_symbol": int(samples_per_symbol_meta) if samples_per_symbol_meta is not None else None,
                "symbols_per_frame": int(symbols_per_frame_meta) if symbols_per_frame_meta is not None else None,
                "bits_per_symbol": int(bits_per_symbol_meta) if bits_per_symbol_meta is not None else None,
                "bits_per_frame": int(bits_per_frame_meta) if bits_per_frame_meta is not None else None,
                "bits_per_frame_by_source": (
                    [int(x) for x in bits_per_frame_by_source_meta]
                    if bits_per_frame_by_source_meta is not None else None
                ),
                "preamble_symbols_per_frame": int(preamble_symbols_per_frame_meta) if preamble_symbols_per_frame_meta is not None else None,
                "payload_symbols_per_frame": int(payload_symbols_per_frame_meta) if payload_symbols_per_frame_meta is not None else None,
                "payload_bits_per_frame": int(payload_bits_per_frame_meta) if payload_bits_per_frame_meta is not None else None,
                "rrc_alpha": float(rrc_alpha_meta) if rrc_alpha_meta is not None else None,
                "rrc_span": int(rrc_span_meta) if rrc_span_meta is not None else None,
            })
            sample_offset += n_frames
        
        # Merge data from each source
        for source_idx in range(self.num_sources):
            self.signals[source_idx] = np.concatenate(self.signals[source_idx], axis=0)
        
        # Load mixed signals
        mixture_signals = []
        for path in mixture_paths:
            with h5py.File(path, 'r') as f:
                data = f[mix_name_str][:]
                data = np.transpose(data, (2, 1, 0))  # (B, L, 2)
            mixture_signals.append(data)
        
        self.mixture = np.concatenate(mixture_signals, axis=0)
        self.frame_length = int(file_frame_length if file_frame_length is not None else self.mixture.shape[1])
        self.valid_frame_length = int(file_valid_frame_length if file_valid_frame_length is not None else self.frame_length)
        self.model_frame_length = int(file_model_frame_length if file_model_frame_length is not None else self.frame_length)

        if self.model_frame_length != self.frame_length:
            for source_idx in range(self.num_sources):
                self.signals[source_idx] = _pad_frame_array(self.signals[source_idx], self.model_frame_length)
            self.mixture = _pad_frame_array(self.mixture, self.model_frame_length)

        self.num_samples = len(self.mixture)
        
        print(f"Successfully loaded MATLAB dataset:")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Number of sources: {self.num_sources}")
        print(f"  Target signal shapes: {[signals.shape for signals in self.signals]}")
        print(f"  Mixed signal shape: {self.mixture.shape}")
        if self.model_frame_length != self.frame_length:
            print(
                f"  Frame length metadata: valid={self.valid_frame_length}, raw={self.frame_length}, "
                f"adapted_for_model={self.model_frame_length}"
            )

    def load_bits(self, bits_paths_per_source):
        """Load bits data from per-source .mat files for BER evaluation.

        Args:
            bits_paths_per_source: list of K lists, each a list of .mat file paths
                ordered the same as signal_paths. Each file contains 'file_bits'
                with shape (1, total_bits) covering all frames in that file.
        """
        if not bits_paths_per_source:
            return

        all_bits = [[] for _ in range(self.num_sources)]

        # We need to know how many frames each signal file contributed
        # to correctly split the flat bit vector into per-frame chunks.
        # _frame_counts was set during _load_data.
        frame_counts = getattr(self, '_frame_counts', None)
        if frame_counts is None:
            print("[BER] Warning: _frame_counts not available, skipping bits loading.")
            return

        for k in range(self.num_sources):
            if k >= len(bits_paths_per_source):
                break
            paths = bits_paths_per_source[k]
            for file_idx, path in enumerate(paths):
                if not Path(path).exists():
                    print(f"[BER] Warning: bits file not found: {path}")
                    return  # Abort bits loading entirely if any file is missing
                try:
                    with h5py.File(path, 'r') as f:
                        raw = _read_h5_bits_vector(f)
                        bit_file_bits_per_frame = _matlab_scalar_or_default(f, 'source_bits_per_frame', None)
                        bit_file_bits_per_frame_vector = _matlab_vector_or_default(f, 'bits_per_frame', None)
                        if (
                            bit_file_bits_per_frame is None
                            and bit_file_bits_per_frame_vector is not None
                            and k < len(bit_file_bits_per_frame_vector)
                        ):
                            bit_file_bits_per_frame = bit_file_bits_per_frame_vector[k]
                except Exception as e:
                    print(f"[BER] Warning: failed to load bits from {path}: {e}")
                    return

                n_frames = frame_counts[file_idx]
                if n_frames <= 0:
                    continue
                bits_per_frame = None
                if bit_file_bits_per_frame is not None:
                    bits_per_frame = int(bit_file_bits_per_frame)
                if file_idx < len(self._file_meta):
                    bits_per_frame_by_source = self._file_meta[file_idx].get("bits_per_frame_by_source")
                    if bits_per_frame_by_source and k < len(bits_per_frame_by_source):
                        bits_per_frame = bits_per_frame_by_source[k]
                    elif bits_per_frame is None:
                        bits_per_frame = self._file_meta[file_idx].get("bits_per_frame")
                if bits_per_frame is not None:
                    bits_per_frame = int(bits_per_frame)
                    expected_total_bits = bits_per_frame * n_frames
                    if len(raw) != expected_total_bits:
                        print(
                            f"[BER] Warning: bits metadata mismatch for {path}: "
                            f"expected {expected_total_bits}, got {len(raw)}. Falling back to equal split."
                        )
                        bits_per_frame = None
                if bits_per_frame is None:
                    bits_per_frame = len(raw) // n_frames
                if bits_per_frame <= 0:
                    print(f"[BER] Warning: bits_per_frame=0 for {path}")
                    return
                # Split flat bit vector into per-frame chunks
                for fr in range(n_frames):
                    all_bits[k].append(raw[fr * bits_per_frame : (fr + 1) * bits_per_frame])

        # Verify all sources have same number of frames as the dataset
        for k in range(self.num_sources):
            if len(all_bits[k]) != self.num_samples:
                print(f"[BER] Warning: bits count mismatch for source {k}: "
                      f"{len(all_bits[k])} vs {self.num_samples} samples. Skipping bits.")
                return

        self.bits_per_source = [np.stack(frames, axis=0) for frames in all_bits]

    def print_metadata_summary(self):
        """Print a compact metadata summary to verify dataset/version alignment."""
        print("\n[MATLAB Meta] Dataset metadata summary")
        print(f"  data_choice: {self.data_choice}")
        print(f"  num_sources: {self.num_sources}")
        print(f"  total_samples: {self.num_samples}")
        print(f"  total_files: {len(self._file_meta)}")
        print(
            f"  lengths: valid={self.valid_frame_length}, raw={self.frame_length}, "
            f"model={self.model_frame_length}"
        )

        if not self._file_meta:
            print("  file_meta: unavailable")
            return

        first_meta = self._file_meta[0]
        print("  first_file:")
        print(f"    signal_path: {first_meta.get('signal_path')}")
        print(f"    mixture_path: {first_meta.get('mixture_path')}")
        print(f"    snr: {first_meta.get('snr')}")
        print(f"    num_frames: {first_meta.get('num_frames')}")
        print(f"    frame_length: {first_meta.get('frame_length')}")
        print(f"    valid_frame_length: {first_meta.get('valid_frame_length')}")
        print(f"    model_frame_length: {first_meta.get('model_frame_length')}")
        print(f"    Fs_sps: {first_meta.get('samples_per_symbol')}")
        print(f"    symbols_per_frame: {first_meta.get('symbols_per_frame')}")
        print(f"    bits_per_symbol: {first_meta.get('bits_per_symbol')}")
        print(f"    bits_per_frame(meta): {first_meta.get('bits_per_frame')}")
        if first_meta.get('bits_per_frame_by_source') is not None:
            print(f"    bits_per_frame_by_source: {first_meta.get('bits_per_frame_by_source')}")
        print(f"    rrc_alpha: {first_meta.get('rrc_alpha')}")
        print(f"    rrc_span: {first_meta.get('rrc_span')}")
        if first_meta.get('preamble_symbols_per_frame') is not None:
            print(f"    preamble_symbols_per_frame: {first_meta.get('preamble_symbols_per_frame')}")
        if first_meta.get('payload_symbols_per_frame') is not None:
            print(f"    payload_symbols_per_frame: {first_meta.get('payload_symbols_per_frame')}")
        if first_meta.get('payload_bits_per_frame') is not None:
            print(f"    payload_bits_per_frame: {first_meta.get('payload_bits_per_frame')}")

        if self.bits_per_source:
            first_bits_shape = tuple(self.bits_per_source[0].shape)
            inferred_bits_per_frame = (
                int(self.bits_per_source[0].shape[1]) if len(first_bits_shape) >= 2 else None
            )
            print("  bits:")
            print(f"    loaded: yes")
            print(f"    first_source_shape: {first_bits_shape}")
            print(f"    bits_per_frame(actual): {inferred_bits_per_frame}")
        else:
            print("  bits:")
            print("    loaded: no")


class PublicSignalDataset(BaseSignalDataset):
    """Public datasets (RML2016/2018, TorchSig, etc.)"""
    
    def __init__(self, signal_paths: Dict[str, List[str]], data_choice: str, 
                 num_sources: int = 2):
        """
        Args:
            signal_paths: Dictionary of signal paths for each type {'BPSK': [paths], 'QPSK': [paths], ...}
            data_choice: Data type selection ('2016', '2018', 'TorchSig')
            num_sources: Number of source signals
        """
        super().__init__(num_sources)
        self.data_choice = data_choice
        self._load_data(signal_paths)
    
    def _load_data(self, signal_paths: Dict[str, List[str]]):
        """Load public dataset"""
        # Determine data field name
        field_mapping = {
            '2018': 'name',
            'TorchSig': 'data', 
            '2016': 'MAT'
        }
        name_str = field_mapping.get(self.data_choice)
        if not name_str:
            raise NotImplementedError(f'Unimplemented data type: {self.data_choice}')
        
        # Load signals of each type
        signal_types = sorted(signal_paths.keys())[:self.num_sources]
        
        for signal_type in signal_types:
            type_signals = []
            type_snrs = []
            
            for path in signal_paths[signal_type]:
                data = loadmat(path)[name_str]
                if self.data_choice == '2016':
                    data = 100 * np.transpose(data, (0, 2, 1))  # (1000,2,128) -> (1000,128,2)
                
                type_signals.append(data)
                snr = self._extract_snr_from_path(path)
                type_snrs.extend([snr] * len(data))
            
            self.signals.append(np.concatenate(type_signals, axis=0))
            if not self.snrs:
                self.snrs = type_snrs
        
        # Generate mixed signals (public dataset needs artificial mixing)
        self._generate_mixture()
    
    def _generate_mixture(self):
        """Generate mixed signals"""
        # Ensure all source signals have same number
        min_samples = min(len(signals) for signals in self.signals)
        self.num_samples = min_samples
        
        # Truncate all signals to same length
        for i in range(len(self.signals)):
            self.signals[i] = self.signals[i][:min_samples]
        
        self.snrs = self.snrs[:min_samples]
        
        # Generate mixed signals
        mixture_list = []
        for idx in range(min_samples):
            # Sum all source signals
            mixed_real = sum(self.signals[i][idx][:, 0] for i in range(self.num_sources))
            mixed_imag = sum(self.signals[i][idx][:, 1] for i in range(self.num_sources))
            mixed_signal = np.stack([mixed_real, mixed_imag], axis=1)
            mixture_list.append(mixed_signal)
        
        self.mixture = np.array(mixture_list)


def _seed_worker(worker_id):
    """Ensure each DataLoader worker has a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


class LightweightRFTrainAugmentDataset(Dataset):
    """Train-only lightweight augmentation wrapper for communication mixtures.

    The goal is to add physically plausible diversity with near-zero overhead:
      - small independent source phase jitter
      - small independent source gain jitter
      - small common time shift
      - global complex-plane phase rotation

    We reconstruct the training mixture from the augmented targets plus the
    residual receiver noise estimated from the original sample. This keeps the
    operation cheap and avoids any disk-side dataset regeneration.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sources: int,
        source_phase_jitter_deg: float = 12.0,
        source_gain_jitter_db: float = 1.0,
        max_common_time_shift: int = 8,
        global_phase_rotation: bool = True,
        mix_enable: bool = False,
        mix_prob: float = 0.0,
        mix_sir_min_db: float = -3.0,
        mix_sir_max_db: float = 3.0,
        mix_cross_sample: bool = False,
        train_aug_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_sources = int(num_sources)
        self.source_phase_jitter_deg = float(max(0.0, source_phase_jitter_deg))
        self.source_gain_jitter_db = float(max(0.0, source_gain_jitter_db))
        self.max_common_time_shift = int(max(0, max_common_time_shift))
        self.global_phase_rotation = bool(global_phase_rotation)
        self.mix_enable = bool(mix_enable)
        self.mix_prob = float(min(max(mix_prob, 0.0), 1.0))
        self.mix_sir_min_db = float(mix_sir_min_db)
        self.mix_sir_max_db = float(mix_sir_max_db)
        self.mix_cross_sample = bool(mix_cross_sample)
        self.train_aug_warmup_epochs = int(max(0, train_aug_warmup_epochs))
        self.current_epoch = 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.current_epoch = int(max(0, epoch))

    def _augmentation_is_active(self):
        return self.current_epoch >= self.train_aug_warmup_epochs

    @staticmethod
    def _rotate_single_iq(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        real = x[0]
        imag = x[1]
        return torch.stack([
            cos_a * real - sin_a * imag,
            sin_a * real + cos_a * imag,
        ], dim=0)

    @staticmethod
    def _rotate_sources_iq(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        cos_a = torch.cos(angles).view(-1, 1)
        sin_a = torch.sin(angles).view(-1, 1)
        real = x[:, 0, :]
        imag = x[:, 1, :]
        return torch.stack([
            cos_a * real - sin_a * imag,
            sin_a * real + cos_a * imag,
        ], dim=1)

    def _split_bits_and_extras(self, extras):
        if extras and isinstance(extras[0], (tuple, list)) and len(extras[0]) == self.num_sources:
            return tuple(extras[0]), list(extras[1:])
        return None, list(extras)

    def _extract_source_from_sample(self, sample, source_idx: int, signal_length: int):
        _, donor_target, _, *donor_extras = sample
        if donor_target.dim() != 2 or donor_target.size(0) != 2 * self.num_sources:
            return None, None
        donor_sources = donor_target.view(self.num_sources, 2, signal_length)
        donor_bits, _ = self._split_bits_and_extras(donor_extras)
        donor_bits_k = donor_bits[source_idx] if donor_bits is not None else None
        return donor_sources[source_idx], donor_bits_k

    def _apply_mix_augmentation(
        self,
        sources: torch.Tensor,
        input_signal: torch.Tensor,
        bits_tuple,
    ):
        signal_length = sources.size(-1)
        augmented_sources = sources.clone()
        augmented_bits = list(bits_tuple) if bits_tuple is not None else None

        if self.mix_cross_sample:
            for source_idx in range(self.num_sources):
                donor_idx = int(torch.randint(low=0, high=len(self.dataset), size=(1,)).item())
                donor_source, donor_bits = self._extract_source_from_sample(
                    self.dataset[donor_idx],
                    source_idx=source_idx,
                    signal_length=signal_length,
                )
                if donor_source is not None:
                    augmented_sources[source_idx] = donor_source
                    if augmented_bits is not None and donor_bits is not None:
                        augmented_bits[source_idx] = donor_bits

        gain_db = torch.empty(self.num_sources, dtype=sources.dtype).uniform_(
            self.mix_sir_min_db,
            self.mix_sir_max_db,
        )
        gains = torch.pow(
            torch.tensor(10.0, dtype=sources.dtype),
            gain_db / 20.0,
        ).view(self.num_sources, 1, 1)
        augmented_sources = augmented_sources * gains

        original_clean = sources.sum(dim=0)
        remixed_clean = augmented_sources.sum(dim=0)
        original_power = original_clean.pow(2).mean().clamp_min(1e-8)
        remixed_power = remixed_clean.pow(2).mean().clamp_min(1e-8)
        augmented_sources = augmented_sources * torch.sqrt(original_power / remixed_power)

        residual_noise = input_signal - original_clean
        return augmented_sources, residual_noise, tuple(augmented_bits) if augmented_bits is not None else None

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        input_signal, target, snr, *extras = sample

        if target.dim() != 2 or input_signal.dim() != 2:
            return sample

        signal_length = target.size(-1)
        expected_channels = 2 * self.num_sources
        if target.size(0) != expected_channels or input_signal.size(0) != 2:
            return sample
        if not self._augmentation_is_active():
            return sample

        sources = target.view(self.num_sources, 2, signal_length)
        bits_tuple, passthrough_extras = self._split_bits_and_extras(extras)
        residual_noise = input_signal - sources.sum(dim=0)

        if self.mix_enable and self.mix_prob > 0.0 and torch.rand(1).item() < self.mix_prob:
            sources, residual_noise, bits_tuple = self._apply_mix_augmentation(
                sources=sources,
                input_signal=input_signal,
                bits_tuple=bits_tuple,
            )

        if self.source_phase_jitter_deg > 0:
            max_rad = np.deg2rad(self.source_phase_jitter_deg)
            phase_offsets = (
                (torch.rand(self.num_sources, dtype=sources.dtype) * 2.0 - 1.0) * max_rad
            )
            sources = self._rotate_sources_iq(sources, phase_offsets)

        if self.source_gain_jitter_db > 0:
            gain_offsets_db = (
                (torch.rand(self.num_sources, dtype=sources.dtype) * 2.0 - 1.0)
                * self.source_gain_jitter_db
            )
            gain = torch.pow(
                torch.tensor(10.0, dtype=sources.dtype),
                gain_offsets_db / 20.0,
            ).view(self.num_sources, 1, 1)
            sources = sources * gain

        if self.max_common_time_shift > 0:
            shift = int(
                torch.randint(
                    low=-self.max_common_time_shift,
                    high=self.max_common_time_shift + 1,
                    size=(1,),
                ).item()
            )
            if shift != 0:
                sources = torch.roll(sources, shifts=shift, dims=-1)
                residual_noise = torch.roll(residual_noise, shifts=shift, dims=-1)

        if self.global_phase_rotation:
            angle = (torch.rand(1, dtype=sources.dtype) * (2.0 * np.pi) - np.pi).squeeze(0)
            sources = self._rotate_sources_iq(
                sources,
                torch.full((self.num_sources,), angle, dtype=sources.dtype),
            )
            residual_noise = self._rotate_single_iq(residual_noise, angle)

        augmented_input = sources.sum(dim=0) + residual_noise
        augmented_target = sources.reshape(expected_channels, signal_length)
        output_extras = []
        if bits_tuple is not None:
            output_extras.append(bits_tuple)
        output_extras.extend(passthrough_extras)
        return augmented_input, augmented_target, snr, *output_extras


def _collect_snr_labels(dataset: Dataset) -> List[float]:
    """Collect per-sample SNR labels from dataset for stratified splitting."""
    # Fast path: datasets in this project usually expose a precomputed snrs list.
    if hasattr(dataset, "snrs"):
        snrs = getattr(dataset, "snrs")
        if snrs is not None and len(snrs) == len(dataset):
            return [_normalize_snr_value(snr) for snr in snrs]

    labels = []
    for idx in range(len(dataset)):
        _, _, snr = dataset[idx]
        labels.append(_normalize_snr_value(snr))
    return labels


def _resolve_subset_indices(dataset: Dataset):
    """Resolve nested Subset into (base_dataset, base_indices)."""
    if not isinstance(dataset, torch.utils.data.Subset):
        return dataset, None

    base_dataset = dataset.dataset
    base_indices = list(dataset.indices)
    while isinstance(base_dataset, torch.utils.data.Subset):
        parent_indices = list(base_dataset.indices)
        base_indices = [parent_indices[i] for i in base_indices]
        base_dataset = base_dataset.dataset
    return base_dataset, base_indices


def _count_snr_distribution(dataset: Dataset) -> Dict[float, int]:
    """Count SNR occurrences for a dataset split."""
    counts: Dict[float, int] = defaultdict(int)
    base_dataset, base_indices = _resolve_subset_indices(dataset)

    # Fast path for project datasets that store snrs as a list.
    if base_indices is not None and hasattr(base_dataset, "snrs"):
        snrs = getattr(base_dataset, "snrs")
        if snrs is not None and len(snrs) == len(base_dataset):
            for idx in base_indices:
                counts[_normalize_snr_value(snrs[idx])] += 1
            return dict(sorted(counts.items()))

    if base_indices is None and hasattr(dataset, "snrs"):
        snrs = getattr(dataset, "snrs")
        if snrs is not None and len(snrs) == len(dataset):
            for snr in snrs:
                counts[_normalize_snr_value(snr)] += 1
            return dict(sorted(counts.items()))

    # Fallback for datasets without cached snr labels.
    for idx in range(len(dataset)):
        _, _, snr = dataset[idx]
        counts[_normalize_snr_value(snr)] += 1
    return dict(sorted(counts.items()))


def _format_snr_counts(snr_counts: Dict[float, int]) -> str:
    """Format SNR count dict for concise logging."""
    if not snr_counts:
        return "(empty)"
    return ", ".join(f"SNR {snr:g} dB: {count}" for snr, count in sorted(snr_counts.items()))


def _largest_remainder_allocate(
    group_sizes: Dict[float, int],
    ratio: float,
    total_target: int,
    capacities: Optional[Dict[float, int]] = None,
) -> Dict[float, int]:
    """Allocate integer counts per group with deterministic largest-remainder rounding."""
    allocations: Dict[float, int] = {}
    remainders: List[Tuple[float, float]] = []

    for snr, size in group_sizes.items():
        exact = size * ratio
        base = int(exact)
        cap = capacities[snr] if capacities is not None else size
        base = min(base, cap)
        allocations[snr] = base
        remainders.append((exact - base, snr))

    remaining = total_target - sum(allocations.values())
    if remaining < 0:
        raise ValueError("Internal allocation error: base allocations exceed target.")

    # First pass: largest fractional parts.
    for _, snr in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if remaining <= 0:
            break
        cap = capacities[snr] if capacities is not None else group_sizes[snr]
        if allocations[snr] < cap:
            allocations[snr] += 1
            remaining -= 1

    # Second pass: fill any leftover by ascending SNR to guarantee completion.
    if remaining > 0:
        for snr in sorted(group_sizes.keys()):
            if remaining <= 0:
                break
            cap = capacities[snr] if capacities is not None else group_sizes[snr]
            while remaining > 0 and allocations[snr] < cap:
                allocations[snr] += 1
                remaining -= 1

    if remaining != 0:
        raise ValueError("Unable to satisfy split targets under current group capacity constraints.")

    return allocations


def _stratified_split_by_snr(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """Split dataset into train/val/test while preserving global SNR proportions."""
    dataset_size = len(dataset)
    snr_labels = _collect_snr_labels(dataset)
    if len(snr_labels) != dataset_size:
        raise ValueError("SNR label count does not match dataset size.")

    snr_to_indices: Dict[float, List[int]] = defaultdict(list)
    for idx, snr in enumerate(snr_labels):
        snr_to_indices[snr].append(idx)

    rng = np.random.default_rng(seed)
    for indices in snr_to_indices.values():
        rng.shuffle(indices)

    train_target = int(train_ratio * dataset_size)
    val_target = int(val_ratio * dataset_size)
    test_target = dataset_size - train_target - val_target

    group_sizes = {snr: len(indices) for snr, indices in snr_to_indices.items()}
    train_counts = _largest_remainder_allocate(group_sizes, train_ratio, train_target)
    remaining_caps = {snr: group_sizes[snr] - train_counts[snr] for snr in group_sizes}
    val_counts = _largest_remainder_allocate(group_sizes, val_ratio, val_target, capacities=remaining_caps)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for snr, indices in snr_to_indices.items():
        n_train = train_counts[snr]
        n_val = val_counts[snr]
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    if len(train_indices) != train_target:
        raise ValueError(f"Train split size mismatch: expected {train_target}, got {len(train_indices)}")
    if len(val_indices) != val_target:
        raise ValueError(f"Val split size mismatch: expected {val_target}, got {len(val_indices)}")
    if len(test_indices) != test_target:
        raise ValueError(f"Test split size mismatch: expected {test_target}, got {len(test_indices)}")

    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
        torch.utils.data.Subset(dataset, test_indices),
    )


def _get_matlab_file_range(data_choice: str, num_sources: int) -> List[int]:
    """Return the file index list for a MATLAB dataset.

    Must stay in sync with ``dataset_configs`` inside
    ``_create_matlab_dataset``.
    """
    data_choice = _normalize_matlab_data_choice(data_choice)
    if data_choice == "QAM" and num_sources == 2:
        return list(range(1, 21))
    # All other MATLAB configs use files 1-10
    return list(range(1, 11))


def _split_file_indices(
    file_indices: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministically split file indices into train / val / test groups."""
    indices = list(file_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n = len(indices)
    n_train = max(1, int(round(train_ratio * n)))
    n_val = max(1, int(round(val_ratio * n)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError(
            f"Not enough files ({n}) for 3 non-empty splits with "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}."
        )

    train_files = sorted(indices[:n_train])
    val_files = sorted(indices[n_train:n_train + n_val])
    test_files = sorted(indices[n_train + n_val:])
    return train_files, val_files, test_files



def create_data_loaders(
    batch_size: int,
    data_choice: str,
    num_sources: int = 2,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 16,
    pin_memory: bool = True,
    matlab_data_root: Optional[Union[str, Path]] = None,
    public_data_root: Optional[Union[str, Path]] = None,
    seed: int = 42,
    split_strategy: str = "random",
    train_aug_config: Optional[Dict[str, Union[bool, int, float]]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[float, DataLoader]]:
    """
    Unified interface for creating data loaders
    
    Args:
        batch_size: Batch size
        data_choice: Dataset selection
        num_sources: Number of source signals (2, 3, 4, ...)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        num_workers: Number of data loading threads
        pin_memory: Whether to use pin_memory
        matlab_data_root: Root path for MATLAB synthetic datasets
        public_data_root: Root path for public datasets
        seed: Random seed for reproducible splits and batch ordering
        split_strategy: Dataset split policy. "random" keeps old behavior;
            "stratified_snr" performs SNR-stratified train/val/test splitting.
        train_aug_config: Optional train-only lightweight RF augmentation config.
    
    Returns:
        train_loader, val_loader, snr_loaders
    """
    data_choice = _normalize_matlab_data_choice(data_choice)

    _MATLAB_DATA_CHOICES = {
        '8PSK_M', '8PSK_M_NS', '8PSK_Burst', '8PSK_Burst_NS',
        '8PSK_M_8192', '8PSK_M_16384', '8PSK_M_32768',
        '8PSK_M_8192_NS', '8PSK_M_16384_NS', '8PSK_M_32768_NS',
        'QPSK_16APSK', 'QPSK_16APSK_NS',
        '8PSK_Rs', '8PSK_Rs_NS',
        '16QAM_64QAM', '16QAM_128QAM', '64QAM_64QAM', '64QAM_128QAM',
        '16QAM_64QAM_128QAM',
        '8PSK-A', '8PSK-B', '8PSK-C', '8PSK-D',
        '8PSK-E', '8PSK-F', '8PSK-G',
        '8PSK-H', '8PSK-I', '8PSK-J', '8PSK-K', '8PSK-L',
        'QPSK+16APSK-A', 'QPSK+16APSK-B',
        'QAM-A', 'QAM-B', 'QAM-C', 'QAM-D', 'QAM-E',
    }
    is_matlab = data_choice in _MATLAB_DATA_CHOICES

    # Validate split ratios early
    if train_ratio < 0 or val_ratio < 0 or (train_ratio + val_ratio) > 1:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require train_ratio>=0, val_ratio>=0 and train_ratio+val_ratio<=1."
        )

    # ------------------------------------------------------------------
    # File-level split for MATLAB + stratified_snr  (prevents data leakage)
    # ------------------------------------------------------------------
    if split_strategy == "stratified_snr" and is_matlab:
        all_file_indices = _get_matlab_file_range(data_choice, num_sources)
        train_files, val_files, test_files = _split_file_indices(
            all_file_indices, train_ratio, val_ratio, seed,
        )
        print(f"[File-level split] train files={train_files}, "
              f"val files={val_files}, test files={test_files}")

        train_dataset = _create_matlab_dataset(
            data_choice, num_sources, matlab_data_root=matlab_data_root,
            file_indices=train_files,
        )
        val_dataset = _create_matlab_dataset(
            data_choice, num_sources, matlab_data_root=matlab_data_root,
            file_indices=val_files,
        )
        test_dataset = _create_matlab_dataset(
            data_choice, num_sources, matlab_data_root=matlab_data_root,
            file_indices=test_files,
        )
        dataset_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)
    else:
        # ---------- Create a single dataset, then split at sample level ----------
        if is_matlab:
            dataset = _create_matlab_dataset(
                data_choice, num_sources, matlab_data_root=matlab_data_root,
            )
        elif data_choice in ['2016', '2018', 'TorchSig']:
            dataset = _create_public_dataset(
                data_choice, num_sources, public_data_root=public_data_root,
            )
        elif str(data_choice).lower() in ['debug_random']:
            dataset = RandomSignalDataset(
                num_sources=num_sources,
                signal_length=1024,
                num_samples=512,
                snr_db=20.0,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported data type: {data_choice}")

        dataset_size = len(dataset)
        if split_strategy == "random":
            train_size = int(train_ratio * dataset_size)
            val_size = int(val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size

            split_generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=split_generator,
            )
        elif split_strategy == "stratified_snr":
            # Non-MATLAB datasets: fall back to frame-level stratified split
            train_dataset, val_dataset, test_dataset = _stratified_split_by_snr(
                dataset=dataset,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            test_size = len(test_dataset)
        else:
            raise ValueError(
                f"Unsupported split_strategy: {split_strategy}. "
                "Choose from ['random', 'stratified_snr']."
            )
    
    train_snr_counts = _count_snr_distribution(train_dataset)
    val_snr_counts = _count_snr_distribution(val_dataset)
    test_snr_counts = _count_snr_distribution(test_dataset)

    # Group test data by SNR
    snr_to_indices = defaultdict(list)
    for idx in range(len(test_dataset)):
        _, _, snr, *_rest = test_dataset[idx]
        snr_to_indices[_normalize_snr_value(snr)].append(idx)
    
    snr_loaders = {}
    for snr, indices in snr_to_indices.items():
        snr_subset = torch.utils.data.Subset(test_dataset, indices)
        snr_loader = DataLoader(snr_subset, batch_size=batch_size, shuffle=False)
        snr_loaders[snr] = snr_loader
    
    effective_train_dataset = train_dataset
    if train_aug_config and bool(train_aug_config.get("enabled", False)):
        effective_train_dataset = LightweightRFTrainAugmentDataset(
            train_dataset,
            num_sources=num_sources,
            source_phase_jitter_deg=float(train_aug_config.get("source_phase_jitter_deg", 12.0)),
            source_gain_jitter_db=float(train_aug_config.get("source_gain_jitter_db", 1.0)),
            max_common_time_shift=int(train_aug_config.get("max_common_time_shift", 8)),
            global_phase_rotation=bool(train_aug_config.get("global_phase_rotation", True)),
            mix_enable=bool(train_aug_config.get("mix_enable", False)),
            mix_prob=float(train_aug_config.get("mix_prob", 0.0)),
            mix_sir_min_db=float(train_aug_config.get("mix_sir_min_db", -3.0)),
            mix_sir_max_db=float(train_aug_config.get("mix_sir_max_db", 3.0)),
            mix_cross_sample=bool(train_aug_config.get("mix_cross_sample", False)),
            train_aug_warmup_epochs=int(train_aug_config.get("train_aug_warmup_epochs", 0)),
        )
        print(
            "  Train augmentation: lightweight_rf "
            f"(src_phase={train_aug_config.get('source_phase_jitter_deg', 12.0)}deg, "
            f"src_gain={train_aug_config.get('source_gain_jitter_db', 1.0)}dB, "
            f"shift={train_aug_config.get('max_common_time_shift', 8)}, "
            f"global_phase={bool(train_aug_config.get('global_phase_rotation', True))}, "
            f"mix_enable={bool(train_aug_config.get('mix_enable', False))}, "
            f"mix_prob={train_aug_config.get('mix_prob', 0.0)}, "
            f"mix_sir=[{train_aug_config.get('mix_sir_min_db', -3.0)}, "
            f"{train_aug_config.get('mix_sir_max_db', 3.0)}]dB, "
            f"mix_cross_sample={bool(train_aug_config.get('mix_cross_sample', False))}, "
            f"warmup_epochs={int(train_aug_config.get('train_aug_warmup_epochs', 0))})"
        )

    # Create training and validation data loaders with fixed generators
    train_g = torch.Generator().manual_seed(seed)
    val_g = torch.Generator().manual_seed(seed + 1)

    train_loader = DataLoader(
        effective_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=train_g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=val_g,
    )

    print(f"Dataset statistics:")
    print(f"  Total samples: {dataset_size}")
    print(f"  Number of sources: {num_sources}")
    _file_level = split_strategy == "stratified_snr" and is_matlab
    print(f"  Split strategy: {split_strategy}{' (file-level, no data leakage)' if _file_level else ''}")
    print(f"  Training: {train_size}, Validation: {val_size}, Test: {test_size}")
    print(f"  Train SNR counts: {_format_snr_counts(train_snr_counts)}")
    print(f"  Val SNR counts: {_format_snr_counts(val_snr_counts)}")
    print(f"  Test SNR counts: {_format_snr_counts(test_snr_counts)}")
    print(f"  Input dimension: (batch_size, 2, signal_length)")
    print(f"  Output dimension: (batch_size, {2*num_sources}, signal_length)")
    
    return train_loader, val_loader, snr_loaders


def _create_matlab_dataset(
    data_choice: str,
    num_sources: int,
    matlab_data_root: Optional[Union[str, Path]] = None,
    file_indices: Optional[List[int]] = None,
) -> MATLABSignalDataset:
    """Create MATLAB dataset.

    Args:
        file_indices: If provided, only use files with these indices instead
            of the default file_range in the config.  Used for file-level
            train/val/test splitting to prevent data leakage.
    """

    data_choice = _normalize_matlab_data_choice(data_choice)
    project_root = Path(__file__).resolve().parents[1]
    default_root = Path(matlab_data_root) if matlab_data_root else (project_root / "data" / "synthetic")
    
    # Define dataset configurations
    dataset_configs = {
        ("QAM", 2): {
            "base_subdir": "QAM_M",
            "snr_range": [30, 20],
            "file_range": range(1, 21),
            "signal_pattern": "16QAM_64QAM_Dataset_target_{i}_SNR={snr}dB.mat",
            "mixture_pattern": "16QAM_64QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK", 3): {
            "base_subdir": "8PSK",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "8PSK_Dataset_target_{i}_SNR={snr}dB.mat",
            "mixture_pattern": "8PSK_Dataset_mixed_{j}_SNR={snr}dB.mat"
        },
        ("8PSK_M", 2): {
            "base_subdir": "8PSK_M",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_NS", 2): {
            "base_subdir": "8PSK_M_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Burst", 2): {
            "base_subdir": "8PSK_Burst",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Burst_NS", 2): {
            "base_subdir": "8PSK_Burst_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_8192", 2): {
            "base_subdir": "8PSK_M_8192",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_8192_NS", 2): {
            "base_subdir": "8PSK_M_8192_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_16384", 2): {
            "base_subdir": "8PSK_M_16384",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_16384_NS", 2): {
            "base_subdir": "8PSK_M_16384_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_32768", 2): {
            "base_subdir": "8PSK_M_32768",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_32768_NS", 2): {
            "base_subdir": "8PSK_M_32768_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M", 3): {
            "base_subdir": "8PSK_M",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        # --- Table(a) Ideal AWGN Scenarios ---
        ("8PSK-A", 2): {
            "base_subdir": "8PSK-A",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-A_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-A_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-B", 2): {
            "base_subdir": "8PSK-B",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-B_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-B_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-C", 3): {
            "base_subdir": "8PSK-C",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_8PSK-C_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_8PSK-C_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-D", 3): {
            "base_subdir": "8PSK-D",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_8PSK-D_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_8PSK-D_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QPSK+16APSK-A", 2): {
            "base_subdir": "QPSK+16APSK-A",
            "base_subdir_candidates": ["QPSK+16APSK-A", "QPSK16APSK-A", "QPSK-16APSK-A"],
            "dataset_token_candidates": ["QPSK+16APSK-A", "QPSK16APSK-A", "QPSK-16APSK-A"],
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_QPSK+16APSK-A_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_QPSK+16APSK-A_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QAM-A", 2): {
            "base_subdir": "QAM-A",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_QAM-A_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_QAM-A_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QAM-B", 2): {
            "base_subdir": "QAM-B",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_QAM-B_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_QAM-B_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QAM-C", 2): {
            "base_subdir": "QAM-C",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_QAM-C_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_QAM-C_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QAM-D", 3): {
            "base_subdir": "QAM-D",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_QAM-D_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_QAM-D_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-E", 2): {
            "base_subdir": "8PSK-E",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-E_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-E_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-F", 2): {
            "base_subdir": "8PSK-F",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-F_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-F_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-G", 2): {
            "base_subdir": "8PSK-G",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-G_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-G_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-H", 2): {
            "base_subdir": "8PSK-H",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-H_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-H_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-I", 2): {
            "base_subdir": "8PSK-I",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-I_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-I_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-J", 3): {
            "base_subdir": "8PSK-J",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_8PSK-J_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_8PSK-J_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-K", 2): {
            "base_subdir": "8PSK-K",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-K_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-K_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK-L", 2): {
            "base_subdir": "8PSK-L",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK-L_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK-L_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QPSK+16APSK-B", 2): {
            "base_subdir": "QPSK+16APSK-B",
            "base_subdir_candidates": ["QPSK+16APSK-B", "QPSK16APSK-B", "QPSK-16APSK-B"],
            "dataset_token_candidates": ["QPSK+16APSK-B", "QPSK16APSK-B", "QPSK-16APSK-B"],
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_QPSK+16APSK-B_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_QPSK+16APSK-B_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QAM-E", 3): {
            "base_subdir": "QAM-E",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_QAM-E_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_QAM-E_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QPSK_16APSK", 2): {
            "base_subdir": "QPSK_16APSK",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/QPSK_16APSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/QPSK_16APSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QPSK_16APSK_NS", 2): {
            "base_subdir": "QPSK_16APSK_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/QPSK_16APSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/QPSK_16APSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Rs", 2): {
            "base_subdir": "8PSK_Rs",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Rs_NS", 2): {
            "base_subdir": "8PSK_Rs_NS",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Rs", 3): {
            "base_subdir": "8PSK_Rs",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/3Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/3Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("16QAM_64QAM", 2): {
            "base_subdir": "16QAM_64QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/16QAM_64QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/16QAM_64QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("16QAM_128QAM", 2): {
            "base_subdir": "16QAM_128QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/16QAM_128QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/16QAM_128QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("64QAM_64QAM", 2): {
            "base_subdir": "64QAM_64QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/64QAM_64QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/64QAM_64QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("64QAM_128QAM", 2): {
            "base_subdir": "64QAM_128QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/64QAM_128QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/64QAM_128QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("16QAM_64QAM_128QAM", 3): {
            "base_subdir": "16QAM_64QAM_128QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/16QAM_64QAM_128QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/16QAM_64QAM_128QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },

    }
    
    config = dataset_configs.get((data_choice, num_sources))
    if not config:
        raise NotImplementedError(f'Unimplemented {data_choice} dataset configuration for {num_sources} sources')
    
    base_subdir_candidates = config.get("base_subdir_candidates", [config["base_subdir"]])
    base_path_candidates = []
    if default_root.name in base_subdir_candidates or (
        (default_root / "target").is_dir() and (default_root / "mixture").is_dir()
    ):
        base_path_candidates.append(default_root)
    base_path_candidates.extend(default_root / subdir for subdir in base_subdir_candidates)
    base_path_candidates = list(dict.fromkeys(base_path_candidates))
    base_path = next((p for p in base_path_candidates if p.exists()), base_path_candidates[0])
    dataset_token_candidates = config.get("dataset_token_candidates", None)

    def _snr_index_key(p: Path):
        m = re.search(r'_(?:target|mixed)_(\d+)_SNR=?([+-]?\d+(?:\.\d+)?)dB', p.name)
        if not m:
            return (float('inf'), p.name)
        return (float(m.group(2)), int(m.group(1)))

    def _dataset_token_from_pattern(pattern: str) -> str:
        m = re.search(r'(?:\d+Source_)?(.+?)_(?:Dataset|BitData)_', pattern)
        if not m:
            raise ValueError(f"Cannot infer dataset token from pattern: {pattern}")
        return m.group(1)

    def _pattern_with_dataset_token(pattern: str, dataset_token: str) -> str:
        current_token = _dataset_token_from_pattern(pattern)
        return pattern.replace(current_token, dataset_token, 1)

    def _discover_matlab_paths(pattern: str, search_base_path: Path) -> List[str]:
        subdir = "target" if pattern.startswith("target/") else "mixture"
        variant = "target" if "target" in pattern else "mixed"
        dataset_token = _dataset_token_from_pattern(pattern)
        search_dir = search_base_path / subdir
        # Some datasets omit the "{num_sources}Source_" prefix in filenames
        glob_pattern = f"{num_sources}Source_{dataset_token}_Dataset_{variant}_*_SNR*.mat"
        files = sorted(search_dir.glob(glob_pattern), key=_snr_index_key)
        if not files:
            glob_pattern = f"{dataset_token}_Dataset_{variant}_*_SNR*.mat"
            files = sorted(search_dir.glob(glob_pattern), key=_snr_index_key)
        return [str(p) for p in files]

    def _resolve_paths(pattern: str, snr_range, file_range) -> List[str]:
        token_candidates = dataset_token_candidates or [_dataset_token_from_pattern(pattern)]
        for search_base_path in base_path_candidates:
            for token in token_candidates:
                candidate_pattern = _pattern_with_dataset_token(pattern, token)

                # Primary pattern
                paths = [
                    str(search_base_path / candidate_pattern.format(j=j, i=j, snr=snr))
                    for snr in snr_range for j in file_range
                ]
                if all(Path(p).exists() for p in paths):
                    return paths

                # Alternate pattern: allow "SNR-10dB" (no '=')
                if "SNR=" in candidate_pattern:
                    alt_pattern = candidate_pattern.replace("SNR=", "SNR")
                    alt_paths = [
                        str(search_base_path / alt_pattern.format(j=j, i=j, snr=snr))
                        for snr in snr_range for j in file_range
                    ]
                    if all(Path(p).exists() for p in alt_paths):
                        print("[MATLAB] Using alternate file naming: 'SNR-10dB' (no '=') pattern.")
                        return alt_paths

                # Fallback: discover available files by glob
                discovered = _discover_matlab_paths(candidate_pattern, search_base_path)
                if discovered:
                    # Filter to requested file indices
                    file_set = set(int(x) for x in file_range)
                    filtered = []
                    for p_str in discovered:
                        m = re.search(r'_(?:target|mixed)_(\d+)_SNR', Path(p_str).name)
                        if m and int(m.group(1)) in file_set:
                            filtered.append(p_str)
                    if filtered:
                        print(f"[MATLAB] Discovered {len(filtered)} files (filtered from "
                              f"{len(discovered)}) under {search_base_path} using glob fallback.")
                        return filtered
                    print(f"[MATLAB] Discovered {len(discovered)} files under {search_base_path} using glob fallback.")
                    return discovered

        # No matches, return original to trigger clear error downstream
        return [
            str(base_path / pattern.format(j=j, i=j, snr=snr))
                for snr in snr_range for j in file_range
        ]

    # Use index variable j for signal_pattern, i for mixture_pattern
    effective_file_range = file_indices if file_indices is not None else config["file_range"]
    all_signal_paths = _resolve_paths(config["signal_pattern"], config["snr_range"], effective_file_range)
    all_mixture_paths = _resolve_paths(config["mixture_pattern"], config["snr_range"], effective_file_range)
    
    dataset = MATLABSignalDataset(all_signal_paths, all_mixture_paths, data_choice, num_sources)

    # --- Auto-detect and load bits for BER evaluation ---
    bits_dir = base_path / "bits"
    if bits_dir.is_dir():
        sig_pat = config["signal_pattern"]
        # Derive bits pattern: target/...Dataset_target...mat -> bits/...BitData..._Source{k}.mat
        bits_pat = sig_pat.replace("target/", "bits/").replace("Dataset_target", "BitData")
        bits_pat = bits_pat.replace(".mat", "_Source{k}.mat")

        def _bits_pattern_candidates(source_idx: int) -> List[str]:
            token_candidates = dataset_token_candidates or [_dataset_token_from_pattern(sig_pat)]
            candidates = [
                _pattern_with_dataset_token(bits_pat, token)
                for token in token_candidates
            ]
            qpsk_apsk_choices = {
                "QPSK_16APSK",
                "QPSK_16APSK_NS",
                "QPSK+16APSK-A",
                "QPSK+16APSK-B",
            }
            if data_choice in qpsk_apsk_choices:
                prefixes = ["QPSK", "16APSK"]
                if source_idx < len(prefixes):
                    candidates.append(
                        f"bits/{prefixes[source_idx]}_BitData_{{j}}_SNR={{snr}}dB_Source{source_idx + 1}.mat"
                    )
            return list(dict.fromkeys(candidates))

        def _resolve_bit_path(patterns: List[str], snr, file_idx: int, source_idx: int) -> str:
            fallback = None
            for pattern in patterns:
                p = str(base_path / pattern.format(j=file_idx, i=file_idx, snr=snr, k=source_idx + 1))
                if fallback is None:
                    fallback = p
                if Path(p).exists():
                    return p
                if "SNR=" in p:
                    alt_p = p.replace("SNR=", "SNR")
                    if Path(alt_p).exists():
                        return alt_p
            return fallback

        bits_paths_per_source = []
        all_found = True
        for k in range(num_sources):
            source_paths = []
            candidate_patterns = _bits_pattern_candidates(k)
            for snr in config["snr_range"]:
                for j in effective_file_range:
                    p = _resolve_bit_path(candidate_patterns, snr, j, k)
                    source_paths.append(p)
            bits_paths_per_source.append(source_paths)
            if not all(Path(sp).exists() for sp in source_paths):
                all_found = False
                break

        if all_found:
            dataset.load_bits(bits_paths_per_source)
        else:
            print(f"[BER] Bits files not fully found under {bits_dir}, BER will be NaN.")

    dataset.print_metadata_summary()

    return dataset


def _create_public_dataset(
    data_choice: str,
    num_sources: int,
    public_data_root: Optional[Union[str, Path]] = None,
) -> PublicSignalDataset:
    """Create public dataset"""

    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(public_data_root) if public_data_root else (project_root / "data")
    
    if data_choice == "2018":
        signal_paths = {
            'BPSK': [str(data_root / "RML2018" / "BPSK" / f'BPSK_SNR={snr}dB.mat')
                    for snr in range(-10, 31, 4)],
            'QPSK': [str(data_root / "RML2018" / "QPSK" / f'QPSK_SNR={snr}dB.mat')
                    for snr in range(-10, 31, 4)]
        }
        
    elif data_choice == "TorchSig":
        snr_list = [-10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30]
        signal_paths = {
            'BPSK': [str(data_root / "TorchSig" / f"bpsk_SNR={snr}dB.mat")
                    for snr in snr_list],
            'QPSK': [str(data_root / "TorchSig" / f"qpsk_SNR={snr}dB.mat")
                    for snr in snr_list]
        }
        
    elif data_choice == "2016":
        snr_list = list(range(-10, 19, 2))
        signal_paths = {
            'BPSK': [str(data_root / "RML2016" / "BPSK" / f"MATBPSK_SNR={snr}dB.mat")
                    for snr in snr_list],
            'QPSK': [str(data_root / "RML2016" / "QPSK" / f"MATQPSK_SNR={snr}dB.mat")
                    for snr in snr_list]
        }
        
        # For multi-source cases, add more modulation types
        if num_sources > 2:
            signal_paths.update({
                '8PSK': [str(data_root / "RML2016" / f"MAT8PSK_SNR={snr}dB.mat")
                        for snr in snr_list],
                '16QAM': [str(data_root / "RML2016" / f"MAT16QAM_SNR={snr}dB.mat")
                         for snr in snr_list]
            })
    
    else:
        raise NotImplementedError(f'Unimplemented public dataset type: {data_choice}')
    
    return PublicSignalDataset(signal_paths, data_choice, num_sources)


def print_dataset_info(train_loader, val_loader, snr_loaders):
    """Print dataset information"""
    print("\n=== Dataset Dimension Information ===")
    
    # Training set information
    print("\nTraining set:")
    for batch_idx, (input_signal, target, snr) in enumerate(train_loader):
        print(f"  Input signal shape: {input_signal.shape}")
        print(f"  Target signal shape: {target.shape}")
        print(f"  SNR examples: {snr[:5].tolist()}")
        break
    
    # Validation set information  
    print("\nValidation set:")
    for batch_idx, (input_signal, target, snr) in enumerate(val_loader):
        print(f"  Input signal shape: {input_signal.shape}")
        print(f"  Target signal shape: {target.shape}")
        break
    
    # Test set information (grouped by SNR)
    print("\nTest set (grouped by SNR):")
    for snr, loader in list(snr_loaders.items())[:3]:  # Only show first 3 SNRs
        print(f"  SNR = {snr} dB: {len(loader.dataset)} samples")


# Example usage
if __name__ == "__main__":
    # 2-source QAM dataset
    train_loader, val_loader, snr_loaders = create_data_loaders(
        batch_size=32, 
        data_choice="QAM", 
        num_sources=2,
    )
    
    print_dataset_info(train_loader, val_loader, snr_loaders)
