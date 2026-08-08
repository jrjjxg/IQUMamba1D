"""Executable compatibility checks for the ICASSP 2024 RF Challenge pipeline.

Run ``python -m rfchallenge.compatibility`` from ``IQUMamba1D``. The default
suite needs only the project's NumPy/PyTorch environment and validates the
native implementation against an independent translation of the public
starter semantics. ``--official-runtime`` additionally invokes the original
TensorFlow/Sionna helpers when that legacy environment is actually available.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import patch

import h5py
import numpy as np
import torch
import torch.nn.functional as functional

from .legacy_reference import (
    starter_demodulate_ofdm_qpsk,
    starter_demodulate_qpsk,
    starter_evaluate,
    starter_modulate_ofdm_qpsk,
    starter_modulate_qpsk,
    starter_ofdm_data_subcarriers,
)
from .augmentation import (
    build_commsignal2_roundtrip_pair_bank,
    load_roundtrip_pair_bank,
    regenerate_zero_ber_waveforms,
    save_roundtrip_bank,
    save_roundtrip_pair_bank,
)
from .datasets import (
    RFChallengeArrayDataset,
    RFChallengeOnlineBatchDataset,
    generate_example_evaluation_set,
)
from .metrics import evaluate_case, official_case_mse_score_db
from .protocol import (
    FRAME_LENGTH,
    SINR_DB_VALUES,
    actual_sinr_db,
    build_mixtures,
    demodulate_ofdm_qpsk,
    demodulate_qpsk,
    generate_ofdm_qpsk,
    generate_qpsk,
    iq_to_complex,
    ofdm_active_subcarriers,
)


def _assert_exact(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(actual, expected):
        difference_count = int(np.count_nonzero(actual != expected))
        raise AssertionError(f"{label} differs in {difference_count} values")


def _assert_waveform_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    tolerance: float = 3e-5,
) -> float:
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance:
        raise AssertionError(
            f"{label} max absolute waveform error {error:.7g} exceeds {tolerance:.7g}"
        )
    return error


def _unit_power_random_batch(
    rng: np.random.Generator,
    batch_size: int,
    frame_length: int,
) -> np.ndarray:
    samples = (
        rng.standard_normal((batch_size, frame_length)).astype(np.float32)
        + 1j * rng.standard_normal((batch_size, frame_length)).astype(np.float32)
    ).astype(np.complex64)
    power = np.sqrt(np.mean(np.abs(samples) ** 2, axis=1, keepdims=True))
    return (samples / power).astype(np.complex64)


def _test_no_interference_self_demod(rng: np.random.Generator) -> dict[str, Any]:
    qpsk = generate_qpsk(batch_size=1, num_symbols=FRAME_LENGTH // 16, rng=rng)
    qpsk_bits, _ = demodulate_qpsk(qpsk.waveform)
    _assert_exact(qpsk_bits, qpsk.bits, "QPSK SOI self-demodulated bits")

    ofdm = generate_ofdm_qpsk(batch_size=1, num_ofdm_symbols=FRAME_LENGTH // 80, rng=rng)
    ofdm_bits, _ = demodulate_ofdm_qpsk(ofdm.waveform)
    _assert_exact(ofdm_bits, ofdm.bits, "OFDM SOI self-demodulated bits")
    return {
        "qpsk_ber": 0.0,
        "ofdm_qpsk_ber": 0.0,
        "qpsk_waveform": qpsk.waveform,
        "qpsk_bits": qpsk.bits,
        "ofdm_waveform": ofdm.waveform,
        "ofdm_bits": ofdm.bits,
    }


def _test_mixture_construction_and_sinr(rng: np.random.Generator) -> dict[str, float]:
    target = _unit_power_random_batch(rng, batch_size=4, frame_length=FRAME_LENGTH)
    interference = _unit_power_random_batch(rng, batch_size=4, frame_length=FRAME_LENGTH)
    requested_sinr = np.asarray([-30.0, -12.0, -3.0, 3.0], dtype=np.float32)
    batch = build_mixtures(target, interference, requested_sinr, rng)
    coefficient = batch.interference_scale * np.exp(1j * batch.phase_radians)
    expected_mixture = target + interference * coefficient[:, None]
    mixture_error = _assert_waveform_close(
        batch.mixture,
        expected_mixture.astype(np.complex64),
        "mixture = SOI + coefficient * interference",
        tolerance=2e-6,
    )
    manual_actual = actual_sinr_db(target, interference * coefficient[:, None])
    _assert_waveform_close(
        batch.actual_sinr_db.astype(np.complex64),
        manual_actual.astype(np.complex64),
        "reported actual SINR",
        tolerance=2e-5,
    )
    requested_error = float(np.max(np.abs(batch.actual_sinr_db - requested_sinr)))
    if requested_error > 2e-5:
        raise AssertionError(
            "Unit-power actual SINR does not match requested SINR: "
            f"max error {requested_error:.7g} dB"
        )
    return {
        "mixture_max_abs_error": mixture_error,
        "actual_vs_requested_sinr_max_abs_db": requested_error,
    }


def _test_waveform_and_carrier_compatibility(
    qpsk_bits: np.ndarray,
    qpsk_waveform: np.ndarray,
    ofdm_bits: np.ndarray,
    ofdm_waveform: np.ndarray,
) -> dict[str, float | int]:
    legacy_qpsk = starter_modulate_qpsk(qpsk_bits)
    qpsk_error = _assert_waveform_close(
        qpsk_waveform,
        legacy_qpsk,
        "native QPSK waveform versus starter-equivalent waveform",
    )
    _assert_exact(
        ofdm_active_subcarriers(),
        starter_ofdm_data_subcarriers(),
        "OFDM active data-subcarrier positions",
    )
    legacy_ofdm = starter_modulate_ofdm_qpsk(ofdm_bits)
    ofdm_error = _assert_waveform_close(
        ofdm_waveform,
        legacy_ofdm,
        "native OFDM waveform versus starter-equivalent waveform",
        tolerance=2e-6,
    )
    return {
        "qpsk_waveform_max_abs_error": qpsk_error,
        "ofdm_waveform_max_abs_error": ofdm_error,
        "ofdm_data_subcarrier_count": int(ofdm_active_subcarriers().size),
    }


def _test_demodulation_compatibility(
    qpsk_waveform: np.ndarray,
    qpsk_bits: np.ndarray,
    ofdm_waveform: np.ndarray,
    ofdm_bits: np.ndarray,
) -> dict[str, int]:
    native_qpsk, _ = demodulate_qpsk(qpsk_waveform)
    legacy_qpsk, _ = starter_demodulate_qpsk(qpsk_waveform)
    _assert_exact(native_qpsk, legacy_qpsk, "QPSK native versus starter-equivalent demod")
    _assert_exact(native_qpsk, qpsk_bits, "QPSK expected transmitted bits")

    native_ofdm, _ = demodulate_ofdm_qpsk(ofdm_waveform)
    legacy_ofdm, _ = starter_demodulate_ofdm_qpsk(ofdm_waveform)
    _assert_exact(native_ofdm, legacy_ofdm, "OFDM native versus starter-equivalent demod")
    _assert_exact(native_ofdm, ofdm_bits, "OFDM expected transmitted bits")
    return {
        "qpsk_bits_checked": int(qpsk_bits.size),
        "ofdm_bits_checked": int(ofdm_bits.size),
    }


def _test_evaluator_compatibility(rng: np.random.Generator) -> dict[str, float]:
    frame_count_per_sinr = 2
    frame_count = len(SINR_DB_VALUES) * frame_count_per_sinr
    max_mse_error = 0.0
    max_ber_error = 0.0
    for soi_type, frame_length in (("QPSK", 16 * 64), ("OFDMQPSK", 80 * 8)):
        if soi_type == "QPSK":
            generated = generate_qpsk(frame_count, frame_length // 16, rng)
        else:
            generated = generate_ofdm_qpsk(frame_count, frame_length // 80, rng)
        noise = (
            rng.standard_normal(generated.waveform.shape).astype(np.float32)
            + 1j * rng.standard_normal(generated.waveform.shape).astype(np.float32)
        ).astype(np.complex64)
        estimate = (generated.waveform + 0.03 * noise).astype(np.complex64)
        estimated_bits = generated.bits.copy()
        estimated_bits[::3, ::7] ^= np.uint8(1)
        native, _ = evaluate_case(
            estimated_soi=estimate,
            target_soi=generated.waveform,
            target_bits=generated.bits,
            estimated_bits=estimated_bits,
            soi_type=soi_type,
            interference_type="CommSignal2",
            n_per_sinr=frame_count_per_sinr,
        )
        legacy_mse_db, legacy_ber = starter_evaluate(
            estimate,
            generated.waveform,
            estimated_bits,
            generated.bits,
            frame_count_per_sinr,
        )
        _assert_exact(native.mse_db, legacy_mse_db, f"{soi_type} evaluator MSE")
        _assert_exact(native.ber, legacy_ber, f"{soi_type} evaluator BER")
        max_mse_error = max(max_mse_error, float(np.max(np.abs(native.mse_db - legacy_mse_db))))
        max_ber_error = max(max_ber_error, float(np.max(np.abs(native.ber - legacy_ber))))
    truncated_inputs = np.asarray(
        [-61.0, -50.0, -49.0, -42.0, -35.0, -28.0, -21.0, -14.0, -7.0, -3.0, 0.0],
        dtype=np.float32,
    )
    expected_truncated_score = float(np.mean(np.maximum(-50.0, truncated_inputs)))
    actual_truncated_score = official_case_mse_score_db(truncated_inputs)
    if not np.isclose(actual_truncated_score, expected_truncated_score, atol=1e-6):
        raise AssertionError(
            "Official truncated MSE aggregation differs from the public score formula: "
            f"{actual_truncated_score} vs {expected_truncated_score}"
        )
    return {
        "evaluator_mse_max_abs_error": max_mse_error,
        "evaluator_ber_max_abs_error": max_ber_error,
        "official_truncated_mse_score_db": actual_truncated_score,
    }


def _test_training_official_validation_score() -> dict[str, float]:
    """Ensure epoch validation emits the public 11-bin truncated MSE scalar."""

    from torch.utils.data import DataLoader, TensorDataset

    from .training import TrainOptions, validate_single_soi_model_with_metrics

    frame_length = 8
    frames_per_sinr = 2
    expected_mse_db = np.asarray(
        [-60.0, -48.0, -42.0, -36.0, -30.0, -24.0, -18.0, -12.0, -6.0, -3.0, 0.0],
        dtype=np.float32,
    )
    total_frames = len(SINR_DB_VALUES) * frames_per_sinr
    features = torch.zeros(total_frames, 2, frame_length)
    targets = torch.zeros_like(features)
    labels = torch.repeat_interleave(torch.from_numpy(SINR_DB_VALUES.copy()), frames_per_sinr)
    amplitudes = torch.from_numpy(np.sqrt(10.0 ** (expected_mse_db / 10.0))).float()
    for index, amplitude in enumerate(amplitudes):
        start = index * frames_per_sinr
        stop = start + frames_per_sinr
        features[start:stop, 0, :] = amplitude
    loader = DataLoader(TensorDataset(features, targets, labels), batch_size=3, shuffle=False)
    metrics = validate_single_soi_model_with_metrics(
        model=torch.nn.Identity(),
        loader=loader,
        device=torch.device("cpu"),
        options=TrainOptions(epochs=1, batch_size=3, amp=False),
    )
    expected_score = official_case_mse_score_db(expected_mse_db)
    if metrics.official_mse_score_db is None or not np.isclose(
        metrics.official_mse_score_db,
        expected_score,
        atol=2e-5,
    ):
        raise AssertionError(
            "Training validation did not emit the expected official MSE score: "
            f"{metrics.official_mse_score_db} vs {expected_score}"
        )
    if metrics.mse_db_by_sinr is None:
        raise AssertionError("Training validation did not retain per-SINR MSE values")
    _assert_waveform_close(
        np.asarray(metrics.mse_db_by_sinr, dtype=np.complex64),
        expected_mse_db.astype(np.complex64),
        "training validation per-SINR MSE",
        tolerance=2e-5,
    )
    return {
        "official_mse_score_db": float(metrics.official_mse_score_db),
        "raw_iq_validation_loss": float(metrics.loss),
    }


def _test_kutii_model_factory() -> dict[str, Any]:
    from models.kutii_learnable_dilation_wavenet import LearnableDilationConv1d
    from .models import build_single_soi_model
    from .training import _project_and_measure_dilations

    torch.manual_seed(202407)
    config_path = Path(__file__).resolve().parents[1] / "config" / "model_config_rfchallenge_kutii_wavenet.yaml"
    model, config = build_single_soi_model(
        config_path,
        frame_length=65,
        device="cpu",
        model_overrides={"dilation_cycle_length": 3},
    )
    if config.model_config["dilation_cycle_length"] != 3 or model.dilation_cycle_length != 3:
        raise AssertionError("KU-TII dilation-cycle runtime override was not applied")
    with torch.no_grad():
        output = model(torch.randn(1, 2, 65))
    if tuple(output.shape) != (1, 2, 65):
        raise AssertionError(f"KU-TII model output shape is {tuple(output.shape)}, expected (1, 2, 65)")

    conv = LearnableDilationConv1d(2, 3, initial_dilation=2.0, max_dilation=4)
    input_tensor = torch.randn(1, 2, 17)
    expected = functional.conv1d(
        input_tensor,
        conv.weight,
        conv.bias,
        padding=2,
        dilation=2,
    )
    _assert_waveform_close(
        conv(input_tensor).detach().numpy().astype(np.complex64),
        expected.detach().numpy().astype(np.complex64),
        "integer learnable dilation equals fixed Conv1d",
        tolerance=2e-6,
    )
    conv.dilation.data.fill_(2.25)
    conv(input_tensor).square().mean().backward()
    if conv.dilation.grad is None or not torch.isfinite(conv.dilation.grad):
        raise AssertionError("learnable dilation did not receive a finite gradient")
    conv.dilation.data.fill_(99.0)
    conv.project_dilation_()
    if float(conv.effective_dilation().detach()) != 4.0:
        raise AssertionError("learnable dilation projection did not enforce max_dilation")
    first_dilation = model.residual_layers[0].dilated_conv.dilation
    first_dilation.data.fill_(-5.0)
    measured_dilations = _project_and_measure_dilations(model)
    if not measured_dilations or min(measured_dilations) < 1.0:
        raise AssertionError("trainer dilation projection did not constrain the KU-TII model")
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "output_shape": list(output.shape),
        "dilation_gradient": float(conv.dilation.grad),
        "minimum_projected_dilation": min(measured_dilations),
        "selected_dilation_cycle_length": model.dilation_cycle_length,
    }


def _test_roundtrip_augmentation_and_online_dataset(rng: np.random.Generator) -> dict[str, int]:
    """Exercise the high-SNR bank writer and online augmented-data path."""

    generated = generate_qpsk(batch_size=3, num_symbols=10, rng=rng)
    roundtrip = regenerate_zero_ber_waveforms(
        generated.waveform,
        generated.bits,
        soi_type="QPSK",
    )
    _assert_exact(roundtrip.bits, generated.bits, "zero-BER round-trip bits")
    with tempfile.TemporaryDirectory(prefix="rfchallenge_compat_") as directory:
        directory_path = Path(directory)
        primary_path = directory_path / "primary.h5"
        augmented_path = directory_path / "comm_signal2_roundtrip.npy"
        with h5py.File(primary_path, "w") as handle:
            handle.create_dataset(
                "dataset",
                data=np.ones((2, generated.waveform.shape[1]), dtype=np.complex64),
            )
        save_roundtrip_bank(augmented_path, roundtrip)
        dataset = RFChallengeOnlineBatchDataset(
            interference_path=primary_path,
            augmentation_interference_path=augmented_path,
            augmentation_probability=1.0,
            soi_type="QPSK",
            interference_type="CommSignal2",
            samples_per_epoch=2,
            batch_size=2,
            frame_length=generated.waveform.shape[1],
            seed=91,
        )
        mixture, target, _ = next(iter(dataset))
        if tuple(mixture.shape) != (2, 2, generated.waveform.shape[1]):
            raise AssertionError(f"Unexpected augmented batch shape: {tuple(mixture.shape)}")
        if tuple(target.shape) != tuple(mixture.shape):
            raise AssertionError("Augmented target and mixture shapes differ")
    return {
        "roundtrip_examples": int(roundtrip.waveforms.shape[0]),
        "augmented_batch_size": int(mixture.shape[0]),
    }


def _test_fixed_stratified_validation(rng: np.random.Generator) -> dict[str, int]:
    """Check the fixed 11 x 100 TestSet1Example-compatible validation layout."""

    frame_length = 16 * 8
    raw_frames = _unit_power_random_batch(rng, batch_size=6, frame_length=frame_length * 2)
    with tempfile.TemporaryDirectory(prefix="rfchallenge_validation_") as directory:
        data_root = Path(directory) / "data_root"
        train_path = data_root / "dataset" / "interferenceset_frame" / "CommSignal2_raw_data.h5"
        raw_path = data_root / "dataset" / "testset1_frame" / "CommSignal2_test1_raw_data.h5"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(train_path, "w") as handle:
            handle.create_dataset("dataset", data=raw_frames)
        with h5py.File(raw_path, "w") as handle:
            handle.create_dataset("dataset", data=raw_frames)
        batch, metadata = generate_example_evaluation_set(
            interference_path=raw_path,
            soi_type="QPSK",
            interference_type="CommSignal2",
            n_per_sinr=100,
            frame_length=frame_length,
            seed=31,
        )
        expected_count = len(SINR_DB_VALUES) * 100
        if batch.mixture.shape != (expected_count, frame_length):
            raise AssertionError(
                f"Fixed validation shape is {batch.mixture.shape}, expected {(expected_count, frame_length)}"
            )
        values, counts = np.unique(batch.nominal_sinr_db, return_counts=True)
        _assert_exact(values, SINR_DB_VALUES, "fixed validation SINR values")
        _assert_exact(counts, np.full(len(SINR_DB_VALUES), 100), "fixed validation counts")
        demodulated_bits, _ = demodulate_qpsk(batch.target)
        _assert_exact(demodulated_bits, batch.bits, "fixed validation target bits")
        if metadata.shape != (expected_count, 5):
            raise AssertionError(f"Fixed validation metadata shape is {metadata.shape}")
        dataset = RFChallengeArrayDataset(
            mixtures=batch.mixture,
            targets=batch.target,
            bits=batch.bits,
            nominal_sinr_db=batch.nominal_sinr_db,
        )
        sample = dataset[0]
        if len(sample) != 4 or tuple(sample[0].shape) != (2, frame_length):
            raise AssertionError("Fixed validation array dataset did not preserve I/Q sample layout")

        from . import cli

        args = argparse.Namespace(
            data_root=data_root,
            samples_per_epoch=4,
            batch_size=2,
            frame_length=frame_length,
            sinr_mode="discrete",
            seed=31,
            validation_samples=1_100,
            validation_per_sinr=100,
            commsignal2_augmentation_path=None,
            commsignal2_augmentation_probability=0.0,
            commsignal2_roundtrip_pairs=None,
            commsignal2_roundtrip_probability=0.0,
        )
        _, constructed_validation, _, _, _, _, _, _, validation_metadata = cli._build_datasets(
            args,
            "QPSK",
            "CommSignal2",
            logging.getLogger("rfchallenge.compatibility"),
        )
        if len(constructed_validation) != expected_count:
            raise AssertionError("CLI dataset construction did not retain exactly 1,100 validation rows")
        if validation_metadata["validation_protocol"] != "fixed_testset1example_11x100":
            raise AssertionError(f"Unexpected validation protocol metadata: {validation_metadata}")
    return {
        "validation_examples": expected_count,
        "sinr_levels": len(SINR_DB_VALUES),
        "examples_per_sinr": 100,
    }


def _test_commsignal2_roundtrip_pair_pipeline(rng: np.random.Generator) -> dict[str, int]:
    """Build, reload, and inject zero-BER complete pair-bank examples."""

    del rng
    frame_length = 16 * 10
    with tempfile.TemporaryDirectory(prefix="rfchallenge_pair_bank_") as directory:
        directory_path = Path(directory)
        raw_path = directory_path / "CommSignal2_raw_data.h5"
        # Zero-valued raw frames make every known-SOI high-SINR mixture exactly
        # demodulable, isolating the builder and storage contracts in this test.
        with h5py.File(raw_path, "w") as handle:
            handle.create_dataset("dataset", data=np.zeros((4, frame_length), dtype=np.complex64))
        pair_path = directory_path / "pairs"
        report = build_commsignal2_roundtrip_pair_bank(
            output_dir=pair_path,
            interference_path=raw_path,
            soi_type="QPSK",
            num_examples=3,
            candidate_sinr_db=(0.0,),
            frame_length=frame_length,
            batch_size=2,
            max_attempts=6,
            seed=41,
        )
        if report.accepted_examples != 3 or report.attempted_examples < 3:
            raise AssertionError(f"Unexpected pair-builder report: {report}")
        bank = load_roundtrip_pair_bank(pair_path)
        if bank.count != 3 or bank.frame_length != frame_length:
            raise AssertionError("Round-trip pair bank did not preserve its requested dimensions")
        regenerated_bits, _ = demodulate_qpsk(bank.targets)
        _assert_exact(regenerated_bits, bank.bits, "round-trip pair target bits")

        portable_path = directory_path / "pairs.npz"
        save_roundtrip_pair_bank(portable_path, bank)
        portable = load_roundtrip_pair_bank(portable_path)
        _assert_exact(portable.bits, bank.bits, "portable pair-bank bits")
        _assert_waveform_close(
            portable.mixtures,
            bank.mixtures,
            "portable pair-bank mixtures",
            tolerance=0.0,
        )

        dataset = RFChallengeOnlineBatchDataset(
            interference_path=raw_path,
            soi_type="QPSK",
            interference_type="CommSignal2",
            samples_per_epoch=3,
            batch_size=3,
            frame_length=frame_length,
            seed=53,
            roundtrip_pair_path=pair_path,
            roundtrip_pair_probability=1.0,
        )
        mixture_iq, target_iq, _ = next(iter(dataset))
        injected_mixtures = iq_to_complex(mixture_iq.numpy())
        injected_targets = iq_to_complex(target_iq.numpy())
        for mixture, target in zip(injected_mixtures, injected_targets):
            found = any(
                np.array_equal(mixture, stored_mixture)
                and np.array_equal(target, stored_target)
                for stored_mixture, stored_target in zip(bank.mixtures, bank.targets)
            )
            if not found:
                raise AssertionError("roundtrip_pair_probability=1 did not inject a stored pair")
        pair_examples = int(bank.count)
        injected_batch_size = int(injected_mixtures.shape[0])
        if dataset.roundtrip_pair_bank is not None:
            dataset.roundtrip_pair_bank.close()
        bank.close()
    return {
        "pair_examples": pair_examples,
        "injected_batch_size": injected_batch_size,
    }


def _test_kutii_training_loop() -> dict[str, int]:
    """Run one small CPU epoch through the actual checkpointing/train hook path."""

    from torch.utils.data import TensorDataset

    from models.kutii_learnable_dilation_wavenet import KUTIIStyleLearnableDilationWaveNet
    from .training import TrainOptions, train_single_soi_model

    torch.manual_seed(51)
    model = KUTIIStyleLearnableDilationWaveNet(
        residual_channels=4,
        residual_layers=2,
        dilation_cycle_length=2,
        max_dilation=4,
    )
    features = torch.randn(2, 2, 32)
    targets = torch.randn(2, 2, 32)
    dataset = TensorDataset(features, targets)
    options = TrainOptions(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        amp=False,
        save_every_epochs=1,
        optimizer="adam",
    )
    with tempfile.TemporaryDirectory(prefix="rfchallenge_train_") as directory:
        history = train_single_soi_model(
            model=model,
            train_dataset=dataset,
            validation_dataset=dataset,
            device=torch.device("cpu"),
            output_dir=directory,
            options=options,
        )
        if not (Path(directory) / "best.pt").is_file() or not (Path(directory) / "last.pt").is_file():
            raise AssertionError("RF Challenge trainer did not write both best.pt and last.pt")
    if len(history.train_loss) != 1 or len(history.effective_dilations) != 1:
        raise AssertionError("RF Challenge trainer did not record one completed epoch")
    if len(history.effective_dilations[0]) != 2:
        raise AssertionError("RF Challenge trainer did not record all learned dilations")
    return {
        "epochs": len(history.train_loss),
        "recorded_dilations": len(history.effective_dilations[0]),
    }


def _test_train_all_sweep_orchestration() -> dict[str, int]:
    """Exercise all-case cycle selection without running the full GPU workload."""

    from . import cli

    config_path = Path(__file__).resolve().parents[1] / "config" / "model_config_rfchallenge_kutii_wavenet.yaml"
    calls: list[tuple[str, str, int]] = []

    def fake_train_one_case(
        args,
        soi: str,
        interference: str,
        logger,
        model_overrides: dict[str, object] | None = None,
        output_subdirectory: str | None = None,
    ) -> Path:
        del logger
        cycle_length = int((model_overrides or {})["dilation_cycle_length"])
        calls.append((soi, interference, cycle_length))
        checkpoint = Path(args.output_dir) / f"{soi}_{interference}" / str(output_subdirectory) / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"best_validation_loss": float(cycle_length)}, checkpoint)
        return checkpoint

    with tempfile.TemporaryDirectory(prefix="rfchallenge_all_sweep_") as directory:
        args = argparse.Namespace(
            model_config=config_path,
            dilation_cycle_lengths=[8, 9],
            seed=17,
            output_dir=Path(directory),
        )
        with patch.object(cli, "_train_one_case", side_effect=fake_train_one_case):
            result = cli.command_train_all_sweep(args, logging.getLogger("rfchallenge.compatibility"))
        if result != 0:
            raise AssertionError(f"train-all-sweep returned {result}, expected 0")
        expected_calls = len(cli.OFFICIAL_CASES) * len(args.dilation_cycle_lengths)
        if len(calls) != expected_calls:
            raise AssertionError(f"train-all-sweep ran {len(calls)} candidates, expected {expected_calls}")
        summary_path = Path(directory) / "all_cases_dilation_cycle_selection.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        cases = summary.get("cases", [])
        if len(cases) != len(cli.OFFICIAL_CASES):
            raise AssertionError("train-all-sweep summary does not contain every public case")
        if any(int(case["selected"]["dilation_cycle_length"]) != 8 for case in cases):
            raise AssertionError("train-all-sweep did not retain the best candidate for every case")
    return {
        "cases": len(cases),
        "candidate_runs": len(calls),
    }


def _official_runtime_check() -> dict[str, Any]:
    """Compare with original helper code if the legacy TensorFlow stack exists."""

    try:
        import tensorflow as tensorflow  # type: ignore[import-not-found]
        import sionna  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError as error:
        return {"status": "skipped", "reason": f"legacy dependency unavailable: {error.name}"}

    starter_root = Path(__file__).resolve().parents[2] / "icassp2024rfchallenge"
    if not starter_root.is_dir():
        return {"status": "skipped", "reason": f"starter directory not found: {starter_root}"}
    sys.path.insert(0, str(starter_root))
    try:
        import rfcutils  # type: ignore[import-not-found]

        rng = np.random.default_rng(13)
        qpsk_bits = rng.integers(0, 2, size=(1, 128), dtype=np.uint8)
        starter_qpsk, _, _, _ = rfcutils.modulate_qpsk_signal(
            tensorflow.convert_to_tensor(qpsk_bits, dtype=tensorflow.float32)
        )
        native_qpsk = starter_modulate_qpsk(qpsk_bits)
        qpsk_error = _assert_waveform_close(
            native_qpsk,
            starter_qpsk.numpy(),
            "native QPSK waveform versus executable starter",
        )
        starter_bits, _ = rfcutils.qpsk_matched_filter_demod(
            tensorflow.convert_to_tensor(native_qpsk)
        )
        native_bits, _ = demodulate_qpsk(native_qpsk)
        _assert_exact(native_bits, starter_bits.numpy().astype(np.uint8), "executable starter QPSK demod")

        ofdm_bits = rng.integers(0, 2, size=(1, 8 * 56 * 2), dtype=np.uint8)
        resource_grid = rfcutils.get_resource_grid(8)
        starter_ofdm, _, _, _ = rfcutils.modulate_ofdm_signal(
            tensorflow.convert_to_tensor(ofdm_bits.reshape(1, 1, 1, -1), dtype=tensorflow.float32),
            resource_grid,
        )
        native_ofdm = starter_modulate_ofdm_qpsk(ofdm_bits)
        ofdm_error = _assert_waveform_close(
            native_ofdm,
            starter_ofdm.numpy(),
            "native OFDM waveform versus executable starter",
            tolerance=2e-6,
        )
        starter_ofdm_bits, _ = rfcutils.ofdm_demod(
            tensorflow.convert_to_tensor(native_ofdm), resource_grid
        )
        native_ofdm_bits, _ = demodulate_ofdm_qpsk(native_ofdm)
        _assert_exact(
            native_ofdm_bits,
            starter_ofdm_bits.numpy().astype(np.uint8),
            "executable starter OFDM demod",
        )
        return {
            "status": "passed",
            "qpsk_waveform_max_abs_error": qpsk_error,
            "ofdm_waveform_max_abs_error": ofdm_error,
        }
    finally:
        try:
            sys.path.remove(str(starter_root))
        except ValueError:
            pass


def run_compatibility_suite(include_official_runtime: bool = False) -> dict[str, Any]:
    """Run every required protocol, evaluator, and KU-TII model compatibility check."""

    rng = np.random.default_rng(202407)
    self_demod = _test_no_interference_self_demod(rng)
    result: dict[str, Any] = {
        "no_interference_self_demod": {
            "qpsk_ber": self_demod["qpsk_ber"],
            "ofdm_qpsk_ber": self_demod["ofdm_qpsk_ber"],
        },
        "mixture_and_sinr": _test_mixture_construction_and_sinr(rng),
        "waveform_and_carriers": _test_waveform_and_carrier_compatibility(
            self_demod["qpsk_bits"],
            self_demod["qpsk_waveform"],
            self_demod["ofdm_bits"],
            self_demod["ofdm_waveform"],
        ),
        "demodulation": _test_demodulation_compatibility(
            self_demod["qpsk_waveform"],
            self_demod["qpsk_bits"],
            self_demod["ofdm_waveform"],
            self_demod["ofdm_bits"],
        ),
        "evaluator": _test_evaluator_compatibility(rng),
        "training_official_validation_score": _test_training_official_validation_score(),
        "kutii_model": _test_kutii_model_factory(),
        "augmentation_pipeline": _test_roundtrip_augmentation_and_online_dataset(rng),
        "fixed_stratified_validation": _test_fixed_stratified_validation(rng),
        "commsignal2_roundtrip_pair_pipeline": _test_commsignal2_roundtrip_pair_pipeline(rng),
        "kutii_training_loop": _test_kutii_training_loop(),
        "kutii_all_case_sweep": _test_train_all_sweep_orchestration(),
    }
    if include_official_runtime:
        result["official_tensorflow_sionna"] = _official_runtime_check()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ICASSP 2024 RF Challenge compatibility checks")
    parser.add_argument(
        "--official-runtime",
        action="store_true",
        help="Also invoke official TensorFlow/Sionna helpers when they are installed.",
    )
    args = parser.parse_args(argv)
    result = run_compatibility_suite(include_official_runtime=args.official_runtime)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
