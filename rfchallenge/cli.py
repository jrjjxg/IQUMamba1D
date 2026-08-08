"""Command-line entry point for the native ICASSP 2024 RF Challenge pipeline.

Run from ``IQUMamba1D``:

    python -m rfchallenge.cli smoke
    python -m rfchallenge.cli train --data-root ../dataset ...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

from .augmentation import (
    build_commsignal2_roundtrip_pair_bank,
    regenerate_zero_ber_waveforms,
    save_roundtrip_bank,
)
from .datasets import (
    RFChallengeArrayDataset,
    RFChallengeOnlineBatchDataset,
    generate_example_evaluation_set,
    resolve_interference_path,
    save_example_evaluation_set,
)
from .inference import infer_and_save_submission, predict_soi
from .metrics import (
    aggregate_case_metrics,
    evaluate_case,
    evaluate_ground_truth_file,
    save_metrics_json,
    save_submission_artifacts,
)
from .models import (
    DEFAULT_RFCHALLENGE_STAGE,
    build_single_soi_model,
    load_checkpoint,
    resolve_checkpoint_path,
    resolve_stage_config,
    select_device,
    supported_rfchallenge_stages,
)
from .protocol import (
    FRAME_LENGTH,
    INTERFERENCE_TYPES,
    OFFICIAL_CASES,
    SINR_DB_VALUES,
    SOI_TYPES,
    demodulate_ofdm_qpsk,
    demodulate_qpsk,
    generate_ofdm_qpsk,
    generate_qpsk,
    root_raised_cosine_taps,
)
from .training import TrainOptions, train_single_soi_model


def _configure_logging(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("rfchallenge")


def _add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--soi", choices=SOI_TYPES, required=True)
    parser.add_argument("--interference", choices=INTERFERENCE_TYPES, required=True)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--model-stage",
        type=int,
        choices=supported_rfchallenge_stages(),
        default=None,
        help=(
            "Use a registered single-SOI IQUMamba stage "
            f"(default: {DEFAULT_RFCHALLENGE_STAGE})."
        ),
    )
    model.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Use an explicit IQUMamba YAML instead of --model-stage.",
    )
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--frame-length", type=int, default=FRAME_LENGTH)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast")


def _resolve_model_config(args: argparse.Namespace) -> Path:
    """Resolve either the explicit YAML or the registered stage selection."""

    explicit = getattr(args, "model_config", None)
    if explicit is not None:
        return Path(explicit)
    stage = getattr(args, "model_stage", None)
    if stage is None:
        stage = DEFAULT_RFCHALLENGE_STAGE
    return resolve_stage_config(int(stage))


def _rfdemucs_case_settings(
    model_config: Path,
    soi_type: str,
    interference_type: str,
) -> tuple[dict[str, object], float | None]:
    """Return the paper's one exceptional architecture and case-wise LR."""

    from util.config import MambaConfig

    config = MambaConfig(str(model_config)).model_config
    if str(config.get("model_type", "")).lower() != "rfchallenge_rfdemucs":
        return {}, None
    if not bool(config.get("rfdemucs_paper_case_settings", True)):
        return {}, None
    if str(soi_type) == "QPSK" and str(interference_type) == "CommSignal3":
        architecture = {
            "rfdemucs_hidden": 80,
            "rfdemucs_stride": 4,
            "rfdemucs_resample": 4,
        }
    else:
        architecture = {
            "rfdemucs_hidden": int(config.get("rfdemucs_hidden", 64)),
            "rfdemucs_stride": int(config.get("rfdemucs_stride", 2)),
            "rfdemucs_resample": int(config.get("rfdemucs_resample", 2)),
        }
    case_name = f"{soi_type}_{interference_type}".lower()
    if case_name == "qpsk_commsignal2":
        learning_rate = float(
            config.get("rfdemucs_qpsk_commsignal2_learning_rate", 3e-4)
        )
    elif case_name == "qpsk_commsignal3":
        learning_rate = float(
            config.get("rfdemucs_qpsk_commsignal3_learning_rate", 3e-4)
        )
    else:
        learning_rate = float(
            config.get("rfdemucs_default_learning_rate", 3e-5)
        )
    return architecture, learning_rate


def _print_metrics(result) -> None:
    print(f"{'SINR [dB]':>12} {'MSE [dB]':>14} {'BER':>14} {'Frames':>10}")
    print("=" * 56)
    for sinr, mse, ber, count in zip(
        result.sinr_db,
        result.mse_db,
        result.ber,
        result.frame_count,
    ):
        print(f"{sinr:>12.1f} {mse:>14.5f} {ber:>14.8f} {count:>10d}")
    print(f"Official MSE score [dB] (11-bin, floor -50 dB): {result.official_mse_score_db:.5f}")
    print(f"Official BER score [dB] (lowest SINR with BER <= 1e-2): {result.official_ber_score_db:.1f}")


def _official_wavenet_case_directory(
    weights_root: str | Path,
    soi_type: str,
    interference_type: str,
) -> Path:
    """Resolve one of the eight released official WaveNet case directories."""

    root = Path(weights_root)
    folder = (
        f"dataset_{str(soi_type).lower()}_"
        f"{str(interference_type).lower()}_mixture_wavenet"
    )
    candidates = (root / folder, root / "torchmodels" / folder)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _initial_model_checkpoint_requested(args: argparse.Namespace) -> bool:
    """Return whether model-only warm-starting was requested."""

    return (
        getattr(args, "init_checkpoint", None) is not None
        or getattr(args, "init_checkpoint_root", None) is not None
    )


def _require_official_wavenet_initialization_config(
    args: argparse.Namespace,
) -> None:
    """Keep released WaveNet weights away from incompatible architectures."""

    if not _initial_model_checkpoint_requested(args):
        return
    model_config = _resolve_model_config(args)
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config}")
    from util.config import MambaConfig

    config = MambaConfig(str(model_config))
    model_type = str(config.model_config.get("model_type", "")).lower()
    if model_type != "icassp_baseline_wavenet":
        raise ValueError(
            "--init-checkpoint/--init-checkpoint-root is a model-only "
            "OneInAMillion WaveNet-ft warm start and requires "
            "model_type=icassp_baseline_wavenet; "
            f"got {model_type!r} from {model_config}"
        )


def _resolve_initial_model_checkpoint(
    args: argparse.Namespace,
    soi_type: str,
    interference_type: str,
) -> Path | None:
    """Resolve one model-only initialization checkpoint for a training case."""

    explicit = getattr(args, "init_checkpoint", None)
    root = getattr(args, "init_checkpoint_root", None)
    if explicit is not None and root is not None:
        # argparse makes this impossible for CLI callers, but retain the
        # invariant for direct Python callers and tests.
        raise ValueError(
            "Use only one of --init-checkpoint and --init-checkpoint-root"
        )
    if explicit is not None:
        return resolve_checkpoint_path(explicit)
    if root is None:
        return None
    case_directory = _official_wavenet_case_directory(
        root,
        soi_type,
        interference_type,
    )
    return resolve_checkpoint_path(case_directory)


def _preflight_all_case_initial_checkpoints(args: argparse.Namespace) -> None:
    """Resolve all eight warm-start weights before a long all-case run."""

    if not _initial_model_checkpoint_requested(args):
        return
    if getattr(args, "init_checkpoint", None) is not None:
        raise ValueError(
            "--init-checkpoint is only valid for the single-case train command; "
            "use --init-checkpoint-root for train-all or train-test-all"
        )
    _require_official_wavenet_initialization_config(args)
    for soi_type, interference_type in OFFICIAL_CASES:
        _resolve_initial_model_checkpoint(args, soi_type, interference_type)


def _print_official_leaderboard_summary(results) -> dict[str, float | int]:
    """Print the public leaderboard's eight-case Result/Average convention."""

    aggregate = aggregate_case_metrics(results)
    print("\n" + "=" * 86)
    print("LOCAL PUBLIC-TESTSET1 PROXY - OFFICIAL EIGHT-CASE SCORE FORMULA")
    print("=" * 86)
    print(
        f"{'SOI':<12} {'Interference':<16} "
        f"{'MSE score [dB]':>18} {'BER threshold SINR [dB]':>25}"
    )
    print("-" * 86)
    for result in results:
        print(
            f"{result.soi_type:<12} {result.interference_type:<16} "
            f"{result.official_mse_score_db:>18.5f} "
            f"{result.official_ber_score_db:>25.1f}"
        )
    print("-" * 86)
    print(
        "MSE Ranking-style  "
        f"Result={aggregate['official_mse_result_db']:.5f}  "
        f"Average(dB)={aggregate['official_mse_average_db']:.5f}"
    )
    print(
        "BER Ranking-style  "
        f"Result={aggregate['official_ber_result_db']:.1f}  "
        f"Average(dB)={aggregate['official_ber_average_db']:.3f}"
    )
    print(
        "Raw diagnostic     "
        f"Mean BER over all 88 case/SINR bins={aggregate['macro_ber']:.8f}"
    )
    print("=" * 86)
    print("NOTE: local TestSet1-style proxy, not the hidden TestSet2 leaderboard.")
    return aggregate


def _resolve_validation_per_sinr(args: argparse.Namespace) -> int:
    """Validate the fixed TestSet1Example-compatible validation cardinality."""

    per_sinr = int(getattr(args, "validation_per_sinr", 100))
    if per_sinr <= 0:
        raise ValueError("--validation-per-sinr must be positive")
    expected_total = len(SINR_DB_VALUES) * per_sinr
    requested_total = getattr(args, "validation_samples", None)
    if requested_total is not None and int(requested_total) != expected_total:
        raise ValueError(
            "Fixed stratified validation requires --validation-samples to equal "
            f"11 * --validation-per-sinr ({expected_total}), got {requested_total}."
        )
    return per_sinr


def _build_datasets(args, soi: str, interference: str, logger: logging.Logger):
    train_path = resolve_interference_path(args.data_root, interference, split="train")
    validation_path = resolve_interference_path(args.data_root, interference, split="test1")
    uses_testset1_frames = validation_path.is_file()
    if not uses_testset1_frames:
        logger.warning(
            "TestSet1 raw interference frames are absent; validation falls back to "
            "InterferenceSet. Extract dataset/testset1_frame for a held-out validation set."
        )
        validation_path = train_path
    augmentation_path = None
    augmentation_probability = 0.0
    requested_augmentation_path = getattr(args, "commsignal2_augmentation_path", None)
    requested_augmentation_probability = float(
        getattr(args, "commsignal2_augmentation_probability", 0.0)
    )
    roundtrip_pair_path = None
    roundtrip_pair_probability = 0.0
    requested_roundtrip_pair_path = getattr(args, "commsignal2_roundtrip_pairs", None)
    requested_roundtrip_pair_probability = float(
        getattr(args, "commsignal2_roundtrip_probability", 0.0)
    )
    if interference == "CommSignal2" and requested_augmentation_path is not None:
        augmentation_path = requested_augmentation_path
        augmentation_probability = requested_augmentation_probability
    elif requested_augmentation_path is not None:
        logger.warning(
            "CommSignal2 augmentation was provided for %s; it is ignored outside the CommSignal2 case.",
            interference,
        )
    if requested_roundtrip_pair_probability > 0.0 and requested_roundtrip_pair_path is None:
        raise ValueError(
            "--commsignal2-roundtrip-pairs is required when "
            "--commsignal2-roundtrip-probability is positive"
        )
    if interference == "CommSignal2" and requested_roundtrip_pair_path is not None:
        roundtrip_pair_path = requested_roundtrip_pair_path
        roundtrip_pair_probability = requested_roundtrip_pair_probability
    elif requested_roundtrip_pair_path is not None:
        logger.warning(
            "CommSignal2 round-trip pairs were provided for %s; they are ignored outside the CommSignal2 case.",
            interference,
        )
    validation_per_sinr = _resolve_validation_per_sinr(args)
    train_dataset = RFChallengeOnlineBatchDataset(
        interference_path=train_path,
        soi_type=soi,
        interference_type=interference,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        frame_length=args.frame_length,
        sinr_mode=args.sinr_mode,
        seed=args.seed,
        augmentation_interference_path=augmentation_path,
        augmentation_probability=augmentation_probability,
        roundtrip_pair_path=roundtrip_pair_path,
        roundtrip_pair_probability=roundtrip_pair_probability,
    )
    validation_batch, _ = generate_example_evaluation_set(
        interference_path=validation_path,
        soi_type=soi,
        interference_type=interference,
        n_per_sinr=validation_per_sinr,
        frame_length=args.frame_length,
        seed=args.seed + 10_000,
    )
    validation_dataset = RFChallengeArrayDataset(
        mixtures=validation_batch.mixture,
        targets=validation_batch.target,
        bits=validation_batch.bits,
        nominal_sinr_db=validation_batch.nominal_sinr_db,
    )
    validation_protocol = f"fixed_testset1example_11x{validation_per_sinr}"
    if not uses_testset1_frames:
        validation_protocol += "_interferenceset_fallback"
    logger.info(
        "Validation is fixed and stratified: %d frames (%d SINR levels x %d) from %s",
        len(validation_dataset),
        len(SINR_DB_VALUES),
        validation_per_sinr,
        validation_path,
    )
    return (
        train_dataset,
        validation_dataset,
        train_path,
        validation_path,
        augmentation_path,
        augmentation_probability,
        roundtrip_pair_path,
        roundtrip_pair_probability,
        {
            "validation_protocol": validation_protocol,
            "validation_per_sinr": validation_per_sinr,
            "validation_samples": len(validation_dataset),
            "validation_seed": args.seed + 10_000,
        },
    )


def _train_one_case(
    args,
    soi: str,
    interference: str,
    logger: logging.Logger,
    model_overrides: dict[str, object] | None = None,
    output_subdirectory: str | None = None,
) -> Path:
    model_config = _resolve_model_config(args)
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config}")
    paper_model_overrides, paper_learning_rate = _rfdemucs_case_settings(
        model_config, soi, interference
    )
    paper_model_overrides.update(dict(model_overrides or {}))
    model_overrides = paper_model_overrides or None
    case_dir = Path(args.output_dir) / f"{soi}_{interference}"
    if output_subdirectory is not None:
        case_dir = case_dir / output_subdirectory
    resume_checkpoint = None
    resume_metadata: dict[str, object] = {}
    if bool(getattr(args, "resume", False)):
        resume_candidates = (
            case_dir / "weights" / "latest_training_checkpoint.pth",
            case_dir / "last.pt",
        )
        resume_checkpoint = next(
            (path for path in resume_candidates if path.is_file()),
            None,
        )
        if resume_checkpoint is None:
            logger.info(
                "Resume requested but no checkpoint exists for %s + %s; "
                "starting from epoch 1",
                soi,
                interference,
            )
        else:
            try:
                resume_payload = torch.load(
                    resume_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                resume_payload = torch.load(resume_checkpoint, map_location="cpu")
            if not isinstance(resume_payload, dict):
                raise ValueError(
                    f"Not a resumable IQUMamba checkpoint: {resume_checkpoint}"
                )
            saved_metadata = resume_payload.get("metadata", {})
            if not isinstance(saved_metadata, dict):
                saved_metadata = {}
            resume_metadata = dict(saved_metadata)
            current_stage = (
                None
                if getattr(args, "model_config", None) is not None
                else (getattr(args, "model_stage", None) or DEFAULT_RFCHALLENGE_STAGE)
            )
            for key, expected in (
                ("soi_type", soi),
                ("interference_type", interference),
                ("frame_length", int(args.frame_length)),
                ("model_stage", current_stage),
            ):
                if key in saved_metadata and saved_metadata[key] != expected:
                    raise ValueError(
                        f"Resume checkpoint {key}={saved_metadata[key]!r} does "
                        f"not match current run {expected!r}"
                    )
            completed_epoch = int(resume_payload.get("epoch", 0))
            completed_best = next(
                (
                    path
                    for path in (
                        case_dir / "best.pt",
                        case_dir / "weights" / "best_training_checkpoint.pth",
                    )
                    if path.is_file()
                ),
                None,
            )
            if completed_epoch >= int(args.epochs) and completed_best is not None:
                logger.info(
                    "Training already complete for %s + %s at epoch %d/%d; "
                    "using %s",
                    soi,
                    interference,
                    completed_epoch,
                    int(args.epochs),
                    completed_best,
                )
                return completed_best
            del resume_payload
    initial_model_checkpoint = None
    initialization_mode = "random"
    initial_checkpoint_metadata = None
    if resume_checkpoint is not None:
        initialization_mode = str(
            resume_metadata.get("initialization_mode", "resumed_training_checkpoint")
        )
        saved_initial_checkpoint = resume_metadata.get("initial_checkpoint")
        if saved_initial_checkpoint is not None:
            initial_checkpoint_metadata = str(saved_initial_checkpoint)
        if _initial_model_checkpoint_requested(args):
            logger.info(
                "Local resume checkpoint takes precedence for %s + %s; "
                "the released initialization checkpoint is not loaded again",
                soi,
                interference,
            )
    else:
        _require_official_wavenet_initialization_config(args)
        initial_model_checkpoint = _resolve_initial_model_checkpoint(
            args,
            soi,
            interference,
        )
        if initial_model_checkpoint is not None:
            initialization_mode = "model_only_fresh_optimizer"
            initial_checkpoint_metadata = str(initial_model_checkpoint.resolve())
    device = select_device(args.device)
    (
        train_dataset,
        validation_dataset,
        train_path,
        validation_path,
        augmentation_path,
        augmentation_probability,
        roundtrip_pair_path,
        roundtrip_pair_probability,
        validation_metadata,
    ) = _build_datasets(
        args, soi, interference, logger
    )
    model, runtime_config = build_single_soi_model(
        model_config,
        frame_length=args.frame_length,
        device=device,
        logger=logger,
        model_overrides=model_overrides,
    )
    if initial_model_checkpoint is not None:
        runtime_model_type = str(getattr(runtime_config, "model_type", "")).lower()
        if runtime_model_type != "icassp_baseline_wavenet":
            raise ValueError(
                "Released WaveNet initialization cannot be loaded into runtime "
                f"model_type={runtime_model_type!r}"
            )
        # Deliberately omit optimizer=.  The released artifact initializes only
        # model parameters; train_single_soi_model creates a fresh optimizer,
        # scaler, scheduler, epoch counter, and best-loss state below.
        load_checkpoint(
            model,
            initial_model_checkpoint,
            # Keep the released checkpoint's unused optimizer/config payload
            # on CPU. load_state_dict copies only the selected model tensors
            # into the already-created runtime model (including CUDA models).
            device="cpu",
            strict=True,
        )
        logger.info(
            "Loaded model-only WaveNet-ft initialization for %s + %s from %s; "
            "optimizer/scaler/scheduler/epoch state starts fresh",
            soi,
            interference,
            initial_model_checkpoint,
        )
    runtime_learning_rate = (
        float(args.learning_rate)
        if paper_learning_rate is None
        else float(paper_learning_rate)
    )
    if paper_learning_rate is not None:
        logger.info(
            "RFDEMUCS paper case settings for %s + %s: overrides=%s, LR=%.2e",
            soi,
            interference,
            model_overrides,
            runtime_learning_rate,
        )
    options = TrainOptions(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=runtime_learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        gradient_clip_norm=args.gradient_clip_norm,
        save_every_epochs=args.save_every_epochs,
        loss=args.loss,
        huber_delta=args.huber_delta,
        optimizer=args.optimizer,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        minimum_learning_rate=args.minimum_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )
    train_single_soi_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
        output_dir=case_dir,
        options=options,
        metadata={
            "soi_type": soi,
            "interference_type": interference,
            "frame_length": args.frame_length,
            "model_config": str(model_config.resolve()),
            "model_stage": (
                None
                if getattr(args, "model_config", None) is not None
                else (
                    getattr(args, "model_stage", None)
                    or DEFAULT_RFCHALLENGE_STAGE
                )
            ),
            "train_interference_path": str(train_path.resolve()),
            "validation_interference_path": str(validation_path.resolve()),
            **validation_metadata,
            "sinr_mode": args.sinr_mode,
            "seed": args.seed,
            "model_overrides": dict(model_overrides or {}),
            "effective_learning_rate": runtime_learning_rate,
            "initialization_mode": initialization_mode,
            "initial_checkpoint": initial_checkpoint_metadata,
            "commsignal2_augmentation_path": (
                None if augmentation_path is None else str(Path(augmentation_path).resolve())
            ),
            "commsignal2_augmentation_probability": augmentation_probability,
            "commsignal2_roundtrip_pairs": (
                None
                if roundtrip_pair_path is None
                else str(Path(roundtrip_pair_path).resolve())
            ),
            "commsignal2_roundtrip_probability": roundtrip_pair_probability,
        },
        logger=logger,
        resume_checkpoint=resume_checkpoint,
    )
    best_checkpoint = case_dir / "best.pt"
    if not best_checkpoint.is_file():
        fallback = case_dir / "weights" / "best_training_checkpoint.pth"
        if fallback.is_file():
            best_checkpoint = fallback
        else:
            raise FileNotFoundError(
                f"Training finished without a best checkpoint for {soi} + {interference}"
            )
    logger.info("Completed %s + %s: %s", soi, interference, best_checkpoint)
    return best_checkpoint


def command_train(args: argparse.Namespace, logger: logging.Logger) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _train_one_case(args, args.soi, args.interference, logger)
    return 0


def command_train_all(args: argparse.Namespace, logger: logging.Logger) -> int:
    _preflight_all_case_initial_checkpoints(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for index, (soi, interference) in enumerate(OFFICIAL_CASES):
        logger.info("[%d/%d] Training %s + %s", index + 1, len(OFFICIAL_CASES), soi, interference)
        _train_one_case(args, soi, interference, logger)
    return 0


def _test_trained_all_cases(
    args: argparse.Namespace,
    logger: logging.Logger,
    checkpoints: dict[tuple[str, str], Path],
) -> dict[str, float | int]:
    """Test eight trained case models with the baseline's public protocol."""

    device = select_device(args.device)
    model_config = _resolve_model_config(args)
    test_root = Path(args.output_dir) / "test_results"
    test_root.mkdir(parents=True, exist_ok=True)
    results = []
    sources = []
    model_stage = (
        None
        if getattr(args, "model_config", None) is not None
        else (getattr(args, "model_stage", None) or DEFAULT_RFCHALLENGE_STAGE)
    )
    method_id = (
        "OneInAMillion_WaveNet_ft"
        if _initial_model_checkpoint_requested(args)
        else (
            f"IQUMamba_Stage{model_stage}"
            if model_stage is not None
            else f"IQUMamba_{model_config.stem}"
        )
    )

    for index, (soi_type, interference_type) in enumerate(OFFICIAL_CASES, start=1):
        checkpoint_path = checkpoints[(soi_type, interference_type)]
        interference_path = resolve_interference_path(
            args.data_root,
            interference_type,
            split="test1",
        )
        print(
            f"\n### TEST [{index}/{len(OFFICIAL_CASES)}] "
            f"{soi_type} + {interference_type} ###"
        )
        logger.info("TestSet1 interference: %s", interference_path)
        logger.info("Best trained checkpoint: %s", checkpoint_path)
        batch, _ = generate_example_evaluation_set(
            interference_path=interference_path,
            soi_type=soi_type,
            interference_type=interference_type,
            n_per_sinr=int(args.test_n_per_sinr),
            frame_length=int(args.frame_length),
            seed=int(args.test_seed),
        )
        model, _ = build_single_soi_model(
            model_config,
            frame_length=int(args.frame_length),
            device=device,
            logger=logger,
            model_overrides=_rfdemucs_case_settings(
                model_config, soi_type, interference_type
            )[0],
        )
        load_checkpoint(model, checkpoint_path, device=device, strict=True)
        estimated_soi = predict_soi(
            model=model,
            mixtures=batch.mixture,
            device=device,
            batch_size=int(args.test_batch_size),
            amp=not args.no_amp,
        )
        result, estimated_bits = evaluate_case(
            estimated_soi=estimated_soi,
            target_soi=batch.target,
            target_bits=batch.bits,
            soi_type=soi_type,
            interference_type=interference_type,
            nominal_sinr_db=batch.nominal_sinr_db,
        )
        _print_metrics(result)
        results.append(result)

        case_dir = test_root / f"{soi_type}_{interference_type}"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_json(
            case_dir / "metrics.json",
            result,
            extra={
                "checkpoint_path": str(checkpoint_path),
                "test_seed": int(args.test_seed),
                "test_n_per_sinr": int(args.test_n_per_sinr),
            },
        )
        if args.save_test_predictions:
            save_submission_artifacts(
                output_dir=case_dir,
                method_id=method_id,
                testset_identifier=f"TestSetLocal_seed{int(args.test_seed)}",
                soi_type=soi_type,
                interference_type=interference_type,
                estimated_soi=estimated_soi,
                estimated_bits=estimated_bits,
            )
        sources.append(
            {
                "soi_type": soi_type,
                "interference_type": interference_type,
                "checkpoint_path": str(checkpoint_path),
                "interference_path": str(interference_path),
            }
        )
        del model, batch, estimated_soi, estimated_bits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = _print_official_leaderboard_summary(results)
    summary = {
        "protocol": "train_test_all_local_public_testset1_proxy",
        "hidden_testset2_leaderboard_reproduction": False,
        "model_stage": model_stage,
        "model_config": str(model_config),
        "initialization_mode": (
            "model_only_fresh_optimizer"
            if _initial_model_checkpoint_requested(args)
            else "random_or_local_resume"
        ),
        "initial_checkpoint_root": (
            None
            if getattr(args, "init_checkpoint_root", None) is None
            else str(Path(args.init_checkpoint_root))
        ),
        "train_seed": int(args.seed),
        "test_seed": int(args.test_seed),
        "test_n_per_sinr": int(args.test_n_per_sinr),
        "frame_length": int(args.frame_length),
        "aggregate": aggregate,
        "cases": [result.to_dict() for result in results],
        "sources": sources,
    }
    summary_path = test_root / "train_test_all_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved train-test-all summary: %s", summary_path)
    return aggregate


def command_train_test_all(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Train/resume all eight cases and test every best checkpoint."""

    model_config = _resolve_model_config(args)
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config}")
    if int(args.test_n_per_sinr) <= 0 or int(args.test_batch_size) <= 0:
        raise ValueError("--test-n-per-sinr and --test-batch-size must be positive")
    _preflight_all_case_initial_checkpoints(args)

    # Fail before a long training run when any required public file is absent.
    for _, interference_type in OFFICIAL_CASES:
        train_path = resolve_interference_path(
            args.data_root, interference_type, split="train"
        )
        test_path = resolve_interference_path(
            args.data_root, interference_type, split="test1"
        )
        if not train_path.is_file():
            raise FileNotFoundError(f"Training interference file not found: {train_path}")
        if not test_path.is_file():
            raise FileNotFoundError(f"TestSet1 interference file not found: {test_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoints: dict[tuple[str, str], Path] = {}
    for index, (soi_type, interference_type) in enumerate(OFFICIAL_CASES, start=1):
        logger.info(
            "[%d/%d] Training %s + %s",
            index,
            len(OFFICIAL_CASES),
            soi_type,
            interference_type,
        )
        training_checkpoint = _train_one_case(
            args,
            soi_type,
            interference_type,
            logger,
        )
        weights_only_checkpoint = (
            Path(args.output_dir)
            / f"{soi_type}_{interference_type}"
            / "weights"
            / "best_model_weights.pth"
        )
        checkpoints[(soi_type, interference_type)] = (
            weights_only_checkpoint
            if weights_only_checkpoint.is_file()
            else training_checkpoint
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _test_trained_all_cases(args, logger, checkpoints)
    return 0


def _resolve_sweep_cycle_lengths(args: argparse.Namespace) -> list[int]:
    """Validate the public-information KU-TII cycle-length candidates."""

    from util.config import MambaConfig

    config = MambaConfig(str(_resolve_model_config(args)))
    model_type = str(config.model_config.get("model_type", "")).lower()
    if model_type != "kutii_learnable_dilation_wavenet":
        raise ValueError(
            "train-sweep requires model_type=kutii_learnable_dilation_wavenet; "
            f"got {model_type!r}"
        )
    cycle_lengths = sorted(set(int(value) for value in args.dilation_cycle_lengths))
    if not cycle_lengths or any(value <= 0 for value in cycle_lengths):
        raise ValueError("--dilation-cycle-lengths must contain positive integers")
    max_dilation = int(config.model_config.get("max_dilation", 1024))
    unsupported = [
        cycle_length
        for cycle_length in cycle_lengths
        if 2 ** (cycle_length - 1) > max_dilation
    ]
    if unsupported:
        raise ValueError(
            "--dilation-cycle-lengths includes values unsupported by "
            f"max_dilation={max_dilation}: {unsupported}. "
            "Increase max_dilation in the model config or remove those candidates."
        )
    return cycle_lengths


def _train_sweep_case(
    args: argparse.Namespace,
    logger: logging.Logger,
    soi: str,
    interference: str,
    cycle_lengths: list[int],
) -> dict[str, object]:
    """Train all cycle candidates for one RF mixture case and select the best."""

    candidates: list[dict[str, object]] = []
    for index, cycle_length in enumerate(cycle_lengths):
        # Every candidate must start from the same initialization and see the
        # same deterministic online samples; otherwise the cycle comparison is
        # confounded by random initialization rather than architecture choice.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(
            "[%d/%d] Training %s + %s with dilation_cycle_length=%d",
            index + 1,
            len(cycle_lengths),
            soi,
            interference,
            cycle_length,
        )
        checkpoint = _train_one_case(
            args,
            soi,
            interference,
            logger,
            model_overrides={"dilation_cycle_length": cycle_length},
            output_subdirectory=f"dilation_cycle_{cycle_length}",
        )
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        validation_loss = float(payload["best_validation_loss"])
        candidates.append(
            {
                "dilation_cycle_length": cycle_length,
                "validation_mse": validation_loss,
                "checkpoint": str(checkpoint.resolve()),
            }
        )

    best = min(candidates, key=lambda candidate: float(candidate["validation_mse"]))
    case_dir = Path(args.output_dir) / f"{soi}_{interference}"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "dilation_cycle_selection.json"
    summary: dict[str, object] = {
        "soi_type": soi,
        "interference_type": interference,
        "selection_metric": "fixed TestSet1Example-style raw-IQ validation MSE",
        "candidates": candidates,
        "selected": best,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(
        "Selected dilation_cycle_length=%d (validation_mse=%.8f); summary: %s",
        int(best["dilation_cycle_length"]),
        float(best["validation_mse"]),
        summary_path,
    )
    return summary


def command_train_sweep(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Train one KU-TII candidate per cycle length and retain the best case."""

    cycle_lengths = _resolve_sweep_cycle_lengths(args)
    _train_sweep_case(args, logger, args.soi, args.interference, cycle_lengths)
    return 0


def command_train_all_sweep(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run the KU-TII cycle search independently across all eight public cases."""

    cycle_lengths = _resolve_sweep_cycle_lengths(args)
    selections: list[dict[str, object]] = []
    for index, (soi, interference) in enumerate(OFFICIAL_CASES):
        logger.info(
            "[%d/%d] Selecting dilation cycle for %s + %s",
            index + 1,
            len(OFFICIAL_CASES),
            soi,
            interference,
        )
        selections.append(_train_sweep_case(args, logger, soi, interference, cycle_lengths))

    summary_path = Path(args.output_dir) / "all_cases_dilation_cycle_selection.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_metric": "fixed TestSet1Example-style raw-IQ validation MSE",
                "dilation_cycle_lengths": cycle_lengths,
                "cases": selections,
            },
            handle,
            indent=2,
        )
    logger.info("Completed all eight cycle searches; summary: %s", summary_path)
    return 0


def command_generate_example(args: argparse.Namespace, logger: logging.Logger) -> int:
    raw_path = resolve_interference_path(args.data_root, args.interference, split="test1")
    if not raw_path.is_file():
        raise FileNotFoundError(
            "TestSet1 raw frames are required to generate a local example set: "
            f"{raw_path}"
        )
    batch, metadata = generate_example_evaluation_set(
        interference_path=raw_path,
        soi_type=args.soi,
        interference_type=args.interference,
        n_per_sinr=args.n_per_sinr,
        frame_length=args.frame_length,
        seed=args.seed,
    )
    paths = save_example_evaluation_set(
        output_dir=args.output_dir,
        identifier=args.identifier,
        soi_type=args.soi,
        interference_type=args.interference,
        batch=batch,
        metadata=metadata,
    )
    for name, path in paths.items():
        logger.info("Saved %s: %s", name, path)
    return 0


def command_infer(args: argparse.Namespace, logger: logging.Logger) -> int:
    if not args.mixture.is_file():
        raise FileNotFoundError(f"Mixture file not found: {args.mixture}")
    mixtures = np.load(args.mixture, allow_pickle=False)
    if mixtures.ndim != 2 or not np.iscomplexobj(mixtures):
        raise ValueError("Official mixture input must be complex with shape (B, 40960)")
    if mixtures.shape[1] != args.frame_length:
        raise ValueError(
            f"Mixture frame length {mixtures.shape[1]} does not match --frame-length {args.frame_length}"
        )
    device = select_device(args.device)
    model_config = _resolve_model_config(args)
    model, _ = build_single_soi_model(
        model_config,
        frame_length=args.frame_length,
        device=device,
        logger=logger,
        model_overrides=_rfdemucs_case_settings(
            model_config, args.soi, args.interference
        )[0],
    )
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    load_checkpoint(
        model,
        checkpoint_path,
        device=device,
        strict=not args.allow_partial_checkpoint,
    )
    _, _, paths = infer_and_save_submission(
        model=model,
        mixtures=mixtures,
        device=device,
        output_dir=args.output_dir,
        method_id=args.method_id,
        testset_identifier=args.identifier,
        soi_type=args.soi,
        interference_type=args.interference,
        batch_size=args.batch_size,
        amp=not args.no_amp,
    )
    for name, path in paths.items():
        logger.info("Saved %s: %s", name, path)
    return 0


def command_evaluate(args: argparse.Namespace, logger: logging.Logger) -> int:
    estimated_soi = np.load(args.estimated_soi, allow_pickle=False)
    estimated_bits = (
        None if args.estimated_bits is None else np.load(args.estimated_bits, allow_pickle=False)
    )
    result, recovered_bits = evaluate_ground_truth_file(
        ground_truth_path=args.ground_truth,
        estimated_soi=estimated_soi,
        estimated_bits=estimated_bits,
        soi_type=args.soi,
        interference_type=args.interference,
        metadata_path=args.metadata,
    )
    _print_metrics(result)
    output_path = args.output_json
    if output_path is None:
        output_path = args.estimated_soi.with_suffix(".metrics.json")
    save_metrics_json(output_path, result)
    logger.info("Saved metrics: %s", output_path)
    if estimated_bits is None:
        logger.info("BER used protocol demodulation of estimated_soi; recovered %d frames", recovered_bits.shape[0])
    return 0


def command_benchmark_baseline_all(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    """Evaluate all eight released WaveNet checkpoints with one command."""

    if int(args.n_per_sinr) <= 0:
        raise ValueError("--n-per-sinr must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    model_config = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "model_config_icassp_baseline_wavenet.yaml"
    )

    # Fail before allocating any 40,960-sample evaluation arrays if the Kaggle
    # dataset layout is incomplete or one released case weight is missing.
    cases: list[tuple[str, str, Path, Path]] = []
    for soi_type, interference_type in OFFICIAL_CASES:
        interference_path = resolve_interference_path(
            args.data_root, interference_type, split="test1"
        )
        if not interference_path.is_file():
            raise FileNotFoundError(
                "Public TestSet1 raw interference file is required for a held-out "
                f"local benchmark: {interference_path}"
            )
        weight_directory = _official_wavenet_case_directory(
            args.weights_root, soi_type, interference_type
        )
        checkpoint_path = resolve_checkpoint_path(weight_directory)
        cases.append(
            (soi_type, interference_type, interference_path, checkpoint_path)
        )

    results = []
    case_sources = []
    for index, (
        soi_type,
        interference_type,
        interference_path,
        checkpoint_path,
    ) in enumerate(cases, start=1):
        print(
            f"\n### [{index}/{len(cases)}] "
            f"{soi_type} + {interference_type} ###"
        )
        logger.info("TestSet1 interference: %s", interference_path)
        logger.info("Official checkpoint: %s", checkpoint_path)
        batch, _ = generate_example_evaluation_set(
            interference_path=interference_path,
            soi_type=soi_type,
            interference_type=interference_type,
            n_per_sinr=int(args.n_per_sinr),
            frame_length=int(args.frame_length),
            seed=int(args.seed),
        )
        model, _ = build_single_soi_model(
            model_config,
            frame_length=int(args.frame_length),
            device=device,
            logger=logger,
        )
        load_checkpoint(model, checkpoint_path, device=device, strict=True)
        estimated_soi = predict_soi(
            model=model,
            mixtures=batch.mixture,
            device=device,
            batch_size=int(args.batch_size),
            amp=not args.no_amp,
        )
        result, estimated_bits = evaluate_case(
            estimated_soi=estimated_soi,
            target_soi=batch.target,
            target_bits=batch.bits,
            soi_type=soi_type,
            interference_type=interference_type,
            nominal_sinr_db=batch.nominal_sinr_db,
        )
        _print_metrics(result)
        results.append(result)

        case_dir = output_dir / f"{soi_type}_{interference_type}"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_json(case_dir / "metrics.json", result)
        if args.save_predictions:
            save_submission_artifacts(
                output_dir=case_dir,
                method_id="Default_Torch_WaveNet",
                testset_identifier=f"TestSetLocal_seed{int(args.seed)}",
                soi_type=soi_type,
                interference_type=interference_type,
                estimated_soi=estimated_soi,
                estimated_bits=estimated_bits,
            )
        case_sources.append({
            "soi_type": soi_type,
            "interference_type": interference_type,
            "interference_path": str(interference_path),
            "checkpoint_path": str(checkpoint_path),
        })

        del model, batch, estimated_soi, estimated_bits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = _print_official_leaderboard_summary(results)
    summary = {
        "protocol": "local_public_testset1_proxy_official_score_formula",
        "hidden_testset2_leaderboard_reproduction": False,
        "seed": int(args.seed),
        "n_per_sinr": int(args.n_per_sinr),
        "frame_length": int(args.frame_length),
        "weights_root": str(Path(args.weights_root)),
        "data_root": str(Path(args.data_root)),
        "aggregate": aggregate,
        "cases": [result.to_dict() for result in results],
        "sources": case_sources,
    }
    summary_path = output_dir / "official_baseline_all_cases_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved eight-case summary: %s", summary_path)
    return 0


def command_smoke(args: argparse.Namespace, logger: logging.Logger) -> int:
    rng = np.random.default_rng(args.seed)
    qpsk = generate_qpsk(batch_size=3, num_symbols=64, rng=rng)
    qpsk_bits, _ = demodulate_qpsk(qpsk.waveform)
    ofdm = generate_ofdm_qpsk(batch_size=3, num_ofdm_symbols=8, rng=rng)
    ofdm_bits, _ = demodulate_ofdm_qpsk(ofdm.waveform)
    if not np.array_equal(qpsk.bits, qpsk_bits):
        raise RuntimeError("QPSK protocol smoke test failed")
    if not np.array_equal(ofdm.bits, ofdm_bits):
        raise RuntimeError("OFDM protocol smoke test failed")
    taps = root_raised_cosine_taps()
    if not np.allclose(taps, taps[::-1], atol=1e-7) or not np.isclose(np.sum(taps**2), 1.0, atol=1e-6):
        raise RuntimeError("RRC tap normalization smoke test failed")
    logger.info("Protocol smoke test passed: QPSK, RRC, OFDM, and hard demodulation")
    return 0


def command_roundtrip_augment(args: argparse.Namespace, logger: logging.Logger) -> int:
    waveforms = np.load(args.waveforms, allow_pickle=False)
    bits = np.load(args.bits, allow_pickle=False)
    result = regenerate_zero_ber_waveforms(
        waveforms=waveforms,
        expected_bits=bits,
        soi_type=args.soi,
        max_bit_error_rate=args.max_bit_error_rate,
    )
    if result.waveforms.shape[0] == 0:
        raise RuntimeError("No examples met the requested BER threshold")
    destination = save_roundtrip_bank(args.output, result)
    logger.info(
        "Saved %d/%d zero-BER round-trip waveforms to %s",
        result.waveforms.shape[0],
        waveforms.shape[0],
        destination,
    )
    return 0


def command_build_commsignal2_roundtrip_pairs(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    """Create the scalable KU-TII-inspired CommSignal2 pair bank."""

    raw_path = resolve_interference_path(args.data_root, "CommSignal2", split="train")
    if not raw_path.is_file():
        raise FileNotFoundError(
            "CommSignal2 InterferenceSet raw frames are required to build round-trip pairs: "
            f"{raw_path}"
        )
    report = build_commsignal2_roundtrip_pair_bank(
        output_dir=args.output,
        interference_path=raw_path,
        soi_type=args.soi,
        num_examples=args.num_examples,
        candidate_sinr_db=args.candidate_sinr_db,
        frame_length=args.frame_length,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        seed=args.seed,
        max_bit_error_rate=args.max_bit_error_rate,
    )
    acceptance_rate = report.accepted_examples / report.attempted_examples
    logger.info(
        "Built %d/%d zero-BER %s + CommSignal2 round-trip pairs in %s "
        "(attempts=%d, acceptance=%.2f%%, per-SINR=%s)",
        report.accepted_examples,
        report.requested_examples,
        report.soi_type,
        report.output_path,
        report.attempted_examples,
        100.0 * acceptance_rate,
        report.accepted_per_sinr_db,
    )
    return 0


def command_compatibility(args: argparse.Namespace, logger: logging.Logger) -> int:
    from .compatibility import run_compatibility_suite

    result = run_compatibility_suite(include_official_runtime=args.official_runtime)
    print(json.dumps(result, indent=2, sort_keys=True))
    logger.info("RF Challenge compatibility suite passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native ICASSP 2024 RF Challenge pipeline for IQUMamba1D"
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-example", help="Generate scoreable TestSet1Example artifacts")
    generate.add_argument("--data-root", type=Path, required=True)
    _add_case_arguments(generate)
    generate.add_argument("--identifier", default="TestSet1Example")
    generate.add_argument("--output-dir", type=Path, default=Path("rfchallenge_data"))
    generate.add_argument("--n-per-sinr", type=int, default=100)
    generate.add_argument("--frame-length", type=int, default=FRAME_LENGTH)
    generate.add_argument("--seed", type=int, default=0)
    generate.set_defaults(handler=command_generate_example)

    def add_train_arguments(command_parser: argparse.ArgumentParser, include_case: bool) -> None:
        command_parser.add_argument("--data-root", type=Path, required=True)
        if include_case:
            _add_case_arguments(command_parser)
        _add_model_arguments(command_parser)
        command_parser.add_argument("--output-dir", type=Path, default=Path("results/rfchallenge"))
        command_parser.add_argument("--epochs", type=int, default=100)
        command_parser.add_argument("--samples-per-epoch", type=int, default=10_000)
        command_parser.add_argument(
            "--validation-samples",
            type=int,
            default=1_100,
            help=(
                "Compatibility total for fixed validation. It must equal "
                "11 * --validation-per-sinr; the default is 1,100."
            ),
        )
        command_parser.add_argument(
            "--validation-per-sinr",
            type=int,
            default=100,
            help="Fixed TestSet1Example-style validation frames per SINR level.",
        )
        command_parser.add_argument("--batch-size", type=int, default=1)
        command_parser.add_argument("--learning-rate", type=float, default=2e-4)
        command_parser.add_argument("--weight-decay", type=float, default=0.0)
        command_parser.add_argument("--num-workers", type=int, default=0)
        command_parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
        command_parser.add_argument("--save-every-epochs", type=int, default=1)
        command_parser.add_argument("--sinr-mode", choices=("discrete", "continuous"), default="discrete")
        command_parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
        command_parser.add_argument("--huber-delta", type=float, default=1.0)
        command_parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
        command_parser.add_argument(
            "--lr-patience",
            type=int,
            default=None,
            help="Enable ReduceLROnPlateau after this many non-improving validation epochs.",
        )
        command_parser.add_argument("--lr-factor", type=float, default=0.5)
        command_parser.add_argument("--minimum-learning-rate", type=float, default=0.0)
        command_parser.add_argument(
            "--early-stopping-patience",
            type=int,
            default=None,
            help="Stop after this many validation epochs without improvement.",
        )
        command_parser.add_argument(
            "--resume",
            action="store_true",
            help=(
                "Resume each case from weights/latest_training_checkpoint.pth "
                "and treat --epochs as the target total epoch count."
            ),
        )
        initialization = command_parser.add_mutually_exclusive_group()
        initialization.add_argument(
            "--init-checkpoint",
            type=Path,
            default=None,
            help=(
                "Initialize model parameters only from one released WaveNet "
                "checkpoint/file or case directory. Valid for single-case train; "
                "optimizer, scaler, scheduler, and epoch state start fresh."
            ),
        )
        initialization.add_argument(
            "--init-checkpoint-root",
            type=Path,
            default=None,
            help=(
                "Root containing the eight released dataset_*_mixture_wavenet "
                "directories (directly or below torchmodels). Each case loads "
                "model weights only; intended for OneInAMillion WaveNet-ft."
            ),
        )
        command_parser.add_argument(
            "--commsignal2-augmentation-path",
            type=Path,
            default=None,
            help=(
                "Legacy optional HDF5/NPY/NPZ interference-frame bank. It replaces "
                "only CommSignal2 raw crops, not complete supervised pairs."
            ),
        )
        command_parser.add_argument(
            "--commsignal2-augmentation-probability",
            type=float,
            default=0.0,
            help="Probability of replacing a CommSignal2 raw crop with a legacy augmented crop.",
        )
        command_parser.add_argument(
            "--commsignal2-roundtrip-pairs",
            type=Path,
            default=None,
            help=(
                "Directory or small .npz bank made by build-commsignal2-roundtrip-pairs. "
                "Used only for CommSignal2 training."
            ),
        )
        command_parser.add_argument(
            "--commsignal2-roundtrip-probability",
            type=float,
            default=0.0,
            help="Probability of replacing an online sample with a complete zero-BER round-trip pair.",
        )
        command_parser.add_argument("--seed", type=int, default=0)

    train = subparsers.add_parser("train", help="Train one SOI/interference case")
    add_train_arguments(train, include_case=True)
    train.set_defaults(handler=command_train)

    train_all = subparsers.add_parser("train-all", help="Train all eight public cases")
    add_train_arguments(train_all, include_case=False)
    train_all.set_defaults(handler=command_train_all)

    train_test_all = subparsers.add_parser(
        "train-test-all",
        help=(
            "Train or resume all eight cases, test every best checkpoint, "
            "and print per-case plus aggregate MSE/BER results"
        ),
    )
    add_train_arguments(train_test_all, include_case=False)
    train_test_all.add_argument("--test-n-per-sinr", type=int, default=100)
    train_test_all.add_argument("--test-seed", type=int, default=100)
    train_test_all.add_argument("--test-batch-size", type=int, default=8)
    train_test_all.add_argument(
        "--save-test-predictions",
        action="store_true",
        help="Also save all eight test estimated-SOI and estimated-bit arrays.",
    )
    train_test_all.set_defaults(handler=command_train_test_all)

    train_sweep = subparsers.add_parser(
        "train-sweep",
        help="Select a KU-TII dilation cycle separately for one mixture case",
    )
    add_train_arguments(train_sweep, include_case=True)
    train_sweep.add_argument(
        "--dilation-cycle-lengths",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11],
        help="Candidate dilation cycle lengths, all trained against the same TestSet1 validation setup.",
    )
    train_sweep.set_defaults(handler=command_train_sweep)

    train_all_sweep = subparsers.add_parser(
        "train-all-sweep",
        help="Select a KU-TII dilation cycle independently for all eight public cases",
    )
    add_train_arguments(train_all_sweep, include_case=False)
    train_all_sweep.add_argument(
        "--dilation-cycle-lengths",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11],
        help="Candidate cycle lengths evaluated independently for every public case.",
    )
    train_all_sweep.set_defaults(handler=command_train_all_sweep)

    infer = subparsers.add_parser("infer", help="Infer and save official-format submission arrays")
    _add_case_arguments(infer)
    _add_model_arguments(infer)
    infer.add_argument("--mixture", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, default=Path("outputs/rfchallenge"))
    infer.add_argument("--method-id", required=True)
    infer.add_argument("--identifier", default="TestSet1Mixture")
    infer.add_argument("--batch-size", type=int, default=1)
    infer.add_argument("--allow-partial-checkpoint", action="store_true")
    infer.set_defaults(handler=command_infer)

    evaluate = subparsers.add_parser("evaluate", help="Score locally generated TestSet1Example predictions")
    _add_case_arguments(evaluate)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--estimated-soi", type=Path, required=True)
    evaluate.add_argument("--estimated-bits", type=Path, default=None)
    evaluate.add_argument("--metadata", type=Path, default=None)
    evaluate.add_argument("--output-json", type=Path, default=None)
    evaluate.set_defaults(handler=command_evaluate)

    benchmark_baseline = subparsers.add_parser(
        "benchmark-baseline-all",
        help=(
            "Evaluate all eight released official WaveNet checkpoints on a "
            "locally generated public TestSet1-style benchmark"
        ),
    )
    benchmark_baseline.add_argument("--data-root", type=Path, required=True)
    benchmark_baseline.add_argument(
        "--weights-root",
        type=Path,
        required=True,
        help=(
            "Directory containing the eight dataset_<soi>_<interference>_"
            "mixture_wavenet folders, or its parent containing torchmodels/"
        ),
    )
    benchmark_baseline.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rfchallenge/official_baseline_all"),
    )
    benchmark_baseline.add_argument("--n-per-sinr", type=int, default=100)
    benchmark_baseline.add_argument("--seed", type=int, default=100)
    benchmark_baseline.add_argument("--batch-size", type=int, default=8)
    benchmark_baseline.add_argument("--frame-length", type=int, default=FRAME_LENGTH)
    benchmark_baseline.add_argument("--device", default=None)
    benchmark_baseline.add_argument("--no-amp", action="store_true")
    benchmark_baseline.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save all eight official-format estimated SOI/bit arrays.",
    )
    benchmark_baseline.set_defaults(handler=command_benchmark_baseline_all)

    smoke = subparsers.add_parser("smoke", help="Run protocol self-consistency tests")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(handler=command_smoke)

    augment = subparsers.add_parser(
        "roundtrip-augment",
        help="Create a clean waveform bank from high-SNR zero-BER examples",
    )
    augment.add_argument("--soi", choices=SOI_TYPES, required=True)
    augment.add_argument("--waveforms", type=Path, required=True)
    augment.add_argument("--bits", type=Path, required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--max-bit-error-rate", type=float, default=0.0)
    augment.set_defaults(handler=command_roundtrip_augment)

    build_pairs = subparsers.add_parser(
        "build-commsignal2-roundtrip-pairs",
        help="Build a memory-mappable zero-BER CommSignal2 round-trip pair bank",
    )
    build_pairs.add_argument("--data-root", type=Path, required=True)
    build_pairs.add_argument("--soi", choices=SOI_TYPES, required=True)
    build_pairs.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory. A 22,000-example bank is intentionally stored as .npy files.",
    )
    build_pairs.add_argument("--num-examples", type=int, default=22_000)
    build_pairs.add_argument(
        "--candidate-sinr-db",
        type=float,
        nargs="+",
        default=[0.0, 3.0],
        help=(
            "High-SINR values sampled while searching for zero-BER pairs. The paper "
            "does not disclose its threshold; default values are explicit, not claimed exact."
        ),
    )
    build_pairs.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum generated mixtures before failure; default is 100 times --num-examples.",
    )
    build_pairs.add_argument("--frame-length", type=int, default=FRAME_LENGTH)
    build_pairs.add_argument("--batch-size", type=int, default=8)
    build_pairs.add_argument("--seed", type=int, default=0)
    build_pairs.add_argument("--max-bit-error-rate", type=float, default=0.0)
    build_pairs.set_defaults(handler=command_build_commsignal2_roundtrip_pairs)

    compatibility = subparsers.add_parser(
        "compatibility",
        help="Run native protocol, evaluator, and KU-TII model compatibility checks",
    )
    compatibility.add_argument(
        "--official-runtime",
        action="store_true",
        help="Also invoke the original TensorFlow/Sionna starter when installed.",
    )
    compatibility.set_defaults(handler=command_compatibility)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = _configure_logging(args.verbose)
    try:
        return int(args.handler(args, logger))
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.exception("RF Challenge command failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
