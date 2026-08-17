import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from util.evaluation import (
    test_model,
    reorder_outputs_for_eval,
    extract_separation_output,
    validate_source_tensor,
    _infer_modulations_from_data_choice,
)
from util.metrics import (
    si_snr_paper,
    si_snr_repo,
    strict_ber_iq_from_bits,
)
from util.visualize import plot_losses
from util.distributed import distributed_sum
from util.output_contracts import select_finest_separation_output
from util.low_snr_training import (
    build_snr_view,
    build_cross_snr_partner,
    clean_mixture_from_targets,
    cross_snr_feature_consistency_loss,
    cross_snr_teacher_consistency_loss,
    curriculum_low_snr,
    phase_equivariance_consistency_loss,
    pit_align_sync_auxiliary,
    rotate_iq,
    sample_progressive_snr_range,
    sync_parameter_cross_snr_consistency_loss,
    sync_parameter_physical_supervision_loss,
    sync_parameter_snr_supervision_loss,
)
from util.rf_equivariance import (
    build_fixed_slot_rf_view,
    fixed_slot_rf_equivariance_consistency_loss,
    sample_fixed_slot_rf_parameters,
)
import copy
import os
import random
import time
from collections import defaultdict
from pathlib import Path


def train_model(
    model,
    scheduler, 
    train_loader, 
    val_loader, 
    snr_loaders, 
    criterion, 
    optimizer, 
    device, 
    num_epochs, 
    early_stop_patience, 
    logger, 
    results_folder, 
    data_choice,
    num_plots,
    batch_size,
    input_size,
    signal_names=None,
    num_sources=None,
    # New parameters
    gradient_clip_norm=1.0,      # Gradient clipping
    save_checkpoint_every=0,     # Save checkpoint_epoch_N snapshots only when > 0
    accumulation_steps=1,        # Gradient accumulation steps
    warmup_epochs=5,             # Learning rate warmup
    use_mixed_precision=True,    # Mixed precision training
    log_interval=100,            # Logging interval
    use_tqdm=True,               # Progress bar
    save_artifacts=True,         # Save checkpoints/plots/csv/history
    report_ber=False,            # Report BER on final evaluation
    ber_offset_search=False,     # Search sampling offset for BER
    ber_mode='file',             # 'frame' or 'file' (stream-level BER)
    ber_num_files=2,             # When ber_mode='file', evaluate this many files
    ber_compute_oracle=False,    # Also compute slow oracle BER debug upper bound
    amr_mode='sep_only',         # AMR model mode: 'sep_only', 'cls_only', 'joint'
    demod_mode='sep_only',       # SoftDemod model mode: 'sep_only', 'demod_only', 'joint'
    demod_mode_phase1=None,      # Optional phase-1 override for demod_mode
    demod_mode_phase2=None,      # Optional phase-2 override for demod_mode
    lr_phase1=None,              # Optional phase-1 learning rate override
    lr_phase2=None,              # Optional phase-2 learning rate override
    demod_teacher_weight=0.0,    # Extra clean-source demod warm-up loss weight
    demod_teacher_phase2_epochs=0,  # Number of early phase-2 epochs using teacher warm-up
    l1_sparsity_weight=0.0,      # L1 sparsity penalty for LSSG gates
    eval_pit_metric='si_snr_complex',
    report_phase_flip=False,     # Report phase-flip rate on final SNR evaluation
    phase_flip_tolerance_deg=45.0,
    phase_flip_min_sc=0.0,
    phase_flip_mode='either',
    resume_checkpoint=None,      # Full training checkpoint to resume from
    resume_allow_partial=False,  # Warm-start matching model weights across architecture variants
    init_checkpoint=None,        # Model-only initialization for a fresh fine-tuning run
    component_checkpoints=None,  # Optional 235/244/245 component checkpoints, loaded in order
    cross_snr_enable=False,
    cross_snr_probability=0.5,
    cross_snr_high_db=10.0,
    cross_snr_low_start_db=2.0,
    cross_snr_low_middle_db=-6.0,
    cross_snr_low_final_db=-10.0,
    cross_snr_first_fraction=0.2,
    cross_snr_second_fraction=0.6,
    cross_snr_pair_weight=0.5,
    cross_snr_consistency_weight=0.1,
    cross_snr_consistency_beta=0.5,
    cross_snr_eps=1e-8,
    cross_snr_shared_permutation=False,
    cross_snr_ema_teacher_enable=False,
    cross_snr_ema_decay=0.999,
    cross_snr_teacher_mode='ema',
    cross_snr_teacher_checkpoint=None,
    cross_snr_teacher_view='high_snr',
    cross_snr_pair_mode='complementary',
    cross_snr_feature_consistency_weight=0.0,
    cross_snr_feature_consistency_beta=0.5,
    cross_snr_curriculum_ranges=((10.0, 30.0), (2.0, 30.0), (-10.0, 30.0)),
    cross_snr_curriculum_boundaries=(0.2, 0.6),
    sync_snr_aux_weight=0.0,
    sync_snr_aux_min_db=-10.0,
    sync_snr_aux_max_db=30.0,
    sync_snr_aux_beta=0.1,
    sync_cross_snr_consistency_weight=0.0,
    sync_cross_snr_consistency_beta=0.1,
    sync_cfo_scale=0.25,
    sync_phase_drift_scale=0.05,
    sync_physical_require_metadata=False,
    sync_physical_supervision_weight=0.0,
    sync_physical_cfo_weight=1.0,
    sync_physical_phase_weight=1.0,
    sync_physical_timing_weight=1.0,
    sync_physical_sps_weight=1.0,
    sync_physical_drift_weight=1.0,
    sync_physical_beta=0.1,
    training_snr_floor_db=None,
    phase_equiv_enable=False,
    phase_equiv_probability=0.25,
    phase_equiv_supervised_weight=0.25,
    phase_equiv_consistency_weight=0.1,
    phase_equiv_max_degrees=180.0,
    phase_equiv_beta=0.5,
    phase_equiv_eps=1e-8,
    rf_equiv_enable=False,
    rf_equiv_probability=0.25,
    rf_equiv_supervised_weight=0.25,
    rf_equiv_consistency_weight=0.1,
    rf_equiv_max_phase_degrees=180.0,
    rf_equiv_max_cfo_cycles_per_sample=1.0e-4,
    rf_equiv_max_gain_db=2.0,
    rf_equiv_max_shift_samples=8,
    rf_equiv_conjugate_probability=0.10,
    rf_equiv_source_mode='per_source',
    rf_equiv_beta=0.5,
    rf_equiv_eps=1e-6,
    latent_mask_residual_weight=0.0,
    latent_mask_mixture_weight=0.0,
    latent_mask_residual_beta=0.5,
    stage255_snr_aux_weight=0.0,
    stage255_snr_aux_min_db=-10.0,
    stage255_snr_aux_max_db=30.0,
    stage255_snr_curriculum_enable=False,
    stage255_snr_curriculum_start_db=10.0,
    stage255_snr_curriculum_end_db=-10.0,
    stage255_snr_curriculum_fraction=0.5,
    stage255_expert_pretrain_epochs=0,
    stage255_router_warmup_epochs=0,
    stage255_router_joint_lr_scale=1.0,
):
    if num_sources is None:
        num_sources = len(signal_names) if signal_names else 2
    num_sources = int(num_sources)
    if num_sources not in (2, 3):
        raise ValueError(f"num_sources must be 2 or 3, got {num_sources}")
    if signal_names is not None and len(signal_names) != num_sources:
        raise ValueError(
            f"signal_names has {len(signal_names)} entries but num_sources={num_sources}"
        )
    cross_snr_enable = bool(cross_snr_enable)
    cross_snr_probability = min(max(float(cross_snr_probability), 0.0), 1.0)
    cross_snr_ema_teacher_enable = bool(cross_snr_ema_teacher_enable)
    cross_snr_ema_decay = float(cross_snr_ema_decay)
    if not 0.0 <= cross_snr_ema_decay < 1.0:
        raise ValueError("cross_snr_ema_decay must be in [0, 1)")
    cross_snr_teacher_mode = str(cross_snr_teacher_mode).lower()
    if cross_snr_teacher_mode not in {'ema', 'frozen'}:
        raise ValueError("cross_snr_teacher_mode must be 'ema' or 'frozen'")
    cross_snr_teacher_view = str(cross_snr_teacher_view).lower()
    if cross_snr_teacher_view not in {'high_snr', 'clean'}:
        raise ValueError("cross_snr_teacher_view must be 'high_snr' or 'clean'")
    cross_snr_pair_mode = str(cross_snr_pair_mode).lower()
    if cross_snr_pair_mode not in {'complementary', 'curriculum_student'}:
        raise ValueError(
            "cross_snr_pair_mode must be 'complementary' or 'curriculum_student'"
        )
    cross_snr_feature_consistency_weight = max(
        0.0, float(cross_snr_feature_consistency_weight)
    )
    sync_snr_aux_weight = max(0.0, float(sync_snr_aux_weight))
    sync_cross_snr_consistency_weight = max(
        0.0, float(sync_cross_snr_consistency_weight)
    )
    sync_physical_supervision_weight = max(
        0.0, float(sync_physical_supervision_weight)
    )
    latent_mask_residual_weight = max(0.0, float(latent_mask_residual_weight))
    latent_mask_mixture_weight = max(0.0, float(latent_mask_mixture_weight))
    latent_mask_residual_beta = max(1.0e-6, float(latent_mask_residual_beta))
    sync_physical_require_metadata = bool(sync_physical_require_metadata)
    training_snr_floor_db = (
        None if training_snr_floor_db is None else float(training_snr_floor_db)
    )
    if float(sync_snr_aux_max_db) <= float(sync_snr_aux_min_db):
        raise ValueError("sync_snr_aux_max_db must exceed sync_snr_aux_min_db")
    cross_snr_teacher = None
    cross_snr_teacher_updates = 0
    resume_cross_snr_teacher_state = None
    if cross_snr_enable:
        logger.info(
            "Cross-SNR paired training enabled: "
            f"prob={cross_snr_probability:.2f}, high={float(cross_snr_high_db):.1f}dB, "
            f"low={float(cross_snr_low_start_db):.1f}->{float(cross_snr_low_middle_db):.1f}"
            f"->{float(cross_snr_low_final_db):.1f}dB, pair_w={float(cross_snr_pair_weight):.3f}, "
            f"cons_w={float(cross_snr_consistency_weight):.3f}, "
            f"teacher={cross_snr_teacher_mode}/{cross_snr_teacher_view}, "
            f"pair_mode={cross_snr_pair_mode}, feature_w={cross_snr_feature_consistency_weight:.3f}"
        )
    if (
        sync_snr_aux_weight > 0.0
        or sync_cross_snr_consistency_weight > 0.0
        or sync_physical_supervision_weight > 0.0
    ):
        logger.info(
            "Synchronization parameter training enabled: "
            f"snr_w={sync_snr_aux_weight:.3f}, "
            f"cross_snr_sync_w={sync_cross_snr_consistency_weight:.3f}, "
            f"physical_w={sync_physical_supervision_weight:.3f}, "
            f"require_metadata={sync_physical_require_metadata}"
        )
    phase_equiv_enable = bool(phase_equiv_enable)
    phase_equiv_probability = min(max(float(phase_equiv_probability), 0.0), 1.0)
    if phase_equiv_enable:
        logger.info(
            "Phase-equivariant paired-view training enabled: "
            f"prob={phase_equiv_probability:.2f}, max_angle={float(phase_equiv_max_degrees):.1f}deg, "
            f"supervised_w={float(phase_equiv_supervised_weight):.3f}, "
            f"consistency_w={float(phase_equiv_consistency_weight):.3f}"
        )
    rf_equiv_enable = bool(rf_equiv_enable)
    rf_equiv_probability = min(max(float(rf_equiv_probability), 0.0), 1.0)
    rf_equiv_conjugate_probability = min(
        max(float(rf_equiv_conjugate_probability), 0.0), 1.0
    )
    rf_equiv_source_mode = str(rf_equiv_source_mode).lower()
    if rf_equiv_source_mode not in {'global', 'per_source'}:
        raise ValueError("rf_equiv_source_mode must be 'global' or 'per_source'")
    if rf_equiv_enable:
        logger.info(
            "Fixed-slot RF equivariance enabled (no PIT): "
            f"prob={rf_equiv_probability:.2f}, mode={rf_equiv_source_mode}, "
            f"phase={abs(float(rf_equiv_max_phase_degrees)):.1f}deg, "
            f"cfo={abs(float(rf_equiv_max_cfo_cycles_per_sample)):.2e}cycles/sample, "
            f"gain={abs(float(rf_equiv_max_gain_db)):.1f}dB, "
            f"shift={max(0, int(rf_equiv_max_shift_samples))}, "
            f"conj_prob={rf_equiv_conjugate_probability:.2f}, "
            f"supervised_w={float(rf_equiv_supervised_weight):.3f}, "
            f"consistency_w={float(rf_equiv_consistency_weight):.3f}"
        )
    stage255_snr_aux_weight = max(0.0, float(stage255_snr_aux_weight))
    stage255_expert_pretrain_epochs = min(
        max(0, int(stage255_expert_pretrain_epochs)), max(0, int(num_epochs) - 1)
    )
    stage255_router_warmup_epochs = min(
        max(0, int(stage255_router_warmup_epochs)),
        max(0, int(num_epochs) - stage255_expert_pretrain_epochs - 1),
    )
    stage255_router_joint_lr_scale = float(stage255_router_joint_lr_scale)
    if not 0.0 < stage255_router_joint_lr_scale <= 1.0:
        raise ValueError("stage255_router_joint_lr_scale must be in (0, 1]")
    if float(stage255_snr_aux_max_db) <= float(stage255_snr_aux_min_db):
        raise ValueError("stage255_snr_aux_max_db must exceed stage255_snr_aux_min_db")
    stage255_snr_curriculum_enable = bool(stage255_snr_curriculum_enable)
    stage255_snr_curriculum_fraction = float(stage255_snr_curriculum_fraction)
    if stage255_snr_curriculum_enable and not 0.0 < stage255_snr_curriculum_fraction <= 1.0:
        raise ValueError("stage255_snr_curriculum_fraction must be in (0, 1]")
    # Ensure output directories exist only when artifact saving is enabled
    project_root = Path(__file__).resolve().parents[1]
    if save_artifacts:
        weights_dir = project_root / "results" / results_folder / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = project_root / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        weights_dir = None
        checkpoint_dir = None

    def _clone_state_dict_to_cpu(state_dict):
        return {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }

    def _cleanup_epoch_checkpoints():
        """Remove old numbered epoch snapshots when only latest/best should be kept."""
        if not save_artifacts or weights_dir is None:
            return
        for path in weights_dir.glob("checkpoint_epoch_*.pth"):
            try:
                path.unlink()
                logger.info(f'Removed old epoch checkpoint: {path}')
            except OSError as exc:
                logger.warning(f'Could not remove old epoch checkpoint {path}: {exc}')

    def _capture_rng_state():
        rng_state = {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state['cuda_random_state_all'] = torch.cuda.get_rng_state_all()
        else:
            rng_state['cuda_random_state_all'] = None
        return rng_state

    def _restore_rng_state(rng_state):
        if not rng_state:
            return
        python_state = rng_state.get('python_random_state')
        numpy_state = rng_state.get('numpy_random_state')
        torch_state = rng_state.get('torch_random_state')
        cuda_state = rng_state.get('cuda_random_state_all')
        if python_state is not None:
            random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _load_checkpoint_file(path):
        try:
            return torch.load(str(path), map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location='cpu')

    def _load_partial_model_state(state_dict, path, aliases_only=False):
        """Warm-start matching tensors and ignore incompatible heads."""
        model_state = model.state_dict()
        alias_model = model.module if hasattr(model, 'module') else model
        prefix_aliases = (
            tuple(alias_model.checkpoint_prefix_aliases())
            if hasattr(alias_model, 'checkpoint_prefix_aliases')
            else ()
        )
        model_prefix = 'module.' if any(key.startswith('module.') for key in model_state) else ''

        def _candidate_keys(source_key):
            yield source_key
            normalized = source_key[7:] if source_key.startswith('module.') else source_key
            yield model_prefix + normalized
            for source_prefix, destination_prefix in prefix_aliases:
                if normalized.startswith(source_prefix):
                    yield model_prefix + destination_prefix + normalized[len(source_prefix):]

        compatible = {}
        skipped_unexpected = []
        skipped_shape = []
        remapped = []
        for key, value in state_dict.items():
            normalized_key = key[7:] if key.startswith('module.') else key
            if aliases_only:
                allowed_prefixes = tuple(source for source, _ in prefix_aliases) + (
                    'bottleneck_attention.',
                )
                if not normalized_key.startswith(allowed_prefixes):
                    continue
            destination = next(
                (
                    candidate for candidate in dict.fromkeys(_candidate_keys(key))
                    if candidate in model_state
                    and torch.is_tensor(value)
                    and model_state[candidate].shape == value.shape
                ),
                None,
            )
            if destination is None and not any(
                candidate in model_state for candidate in _candidate_keys(key)
            ):
                skipped_unexpected.append(key)
                continue
            if destination is None:
                skipped_shape.append(key)
                continue
            compatible[destination] = value
            if destination != key:
                remapped.append((key, destination))

        if not compatible:
            raise RuntimeError(
                f"No compatible tensors found in checkpoint {path}; cannot partial-load model weights."
            )

        missing, unexpected = model.load_state_dict(compatible, strict=False)
        logger.warning(
            f"Warm-started model weights from {path} with partial loading: "
            f"loaded={len(compatible)}, missing_in_checkpoint={len(missing)}, "
            f"unexpected_or_old_head={len(skipped_unexpected)}, shape_mismatch={len(skipped_shape)}. "
            "Optimizer/scheduler/epoch history were not restored."
        )
        if remapped:
            logger.warning(
                f"Checkpoint prefix migration remapped {len(remapped)} tensors; "
                f"first few: {remapped[:8]}"
            )
        if skipped_unexpected[:8]:
            logger.warning(f"Skipped unexpected checkpoint keys, first few: {skipped_unexpected[:8]}")

    def _load_component_checkpoint(path):
        checkpoint = _load_checkpoint_file(path)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Component checkpoint {path} does not contain a state dict")
        _load_partial_model_state(state_dict, path, aliases_only=True)
        if missing[:8]:
            logger.warning(f"New model keys left initialized, first few: {missing[:8]}")
        if unexpected[:8]:
            logger.warning(f"Unexpected keys after partial load, first few: {unexpected[:8]}")
    
    # Initialize best state
    best_val_loss = float('inf')
    no_improve_counter = 0
    best_model_weights = None
    best_epoch = None
    start_epoch = 0
    early_stop_enabled = early_stop_patience is not None and early_stop_patience > 0
    
    # Record training history
    train_losses = []
    val_losses = []
    val_si_snr_paper_vals = []
    val_si_snr_repo_vals = []
    learning_rates = []
    epoch_times = []
    model_diagnostics_history = []

    def _collect_model_diagnostics():
        """Collect optional architecture diagnostics as serialization-safe values."""

        diagnostic_model = model.module if hasattr(model, 'module') else model
        diagnostics_fn = getattr(diagnostic_model, 'diagnostics', None)
        if not callable(diagnostics_fn):
            return {}
        raw_values = diagnostics_fn()
        if not isinstance(raw_values, dict):
            logger.warning(
                "model.diagnostics() must return a dict; "
                f"got {type(raw_values).__name__}"
            )
            return {}
        values = {}
        for name, value in raw_values.items():
            if torch.is_tensor(value):
                detached = value.detach().float().cpu()
                values[str(name)] = (
                    float(detached.item())
                    if detached.numel() == 1
                    else detached.tolist()
                )
            elif isinstance(value, (int, float, bool, str)):
                values[str(name)] = value
        return values
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    
    # Learning rate warmup scheduler
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Validation/test source alignment mode: {eval_pit_metric}")
    if early_stop_enabled:
        logger.info(f"Early stopping enabled (patience={early_stop_patience})")
    else:
        logger.info("Early stopping disabled")
    
    tqdm_kwargs = dict(
        dynamic_ncols=True,
        leave=False,
        mininterval=0.3,
        smoothing=0.1,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}',
    )
    def _project_model_dilations():
        """Keep KU-TII learnable dilation scalars in their valid range."""
        base_model = model.module if hasattr(model, 'module') else model
        project = getattr(base_model, 'project_learnable_dilations_', None)
        if callable(project):
            project()

    optimizer.zero_grad(set_to_none=True)

    def _set_dataset_epoch(dataset, epoch: int):
        if dataset is None:
            return
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        children = getattr(dataset, "datasets", None)
        if children is not None:
            for child in children:
                _set_dataset_epoch(child, epoch)
        nested = getattr(dataset, "dataset", None)
        if nested is not None and nested is not dataset:
            _set_dataset_epoch(nested, epoch)

    def _criterion_uses_snr() -> bool:
        if bool(getattr(criterion, 'needs_snr', False)):
            return True
        return hasattr(criterion, 'forward') and 'snr' in criterion.forward.__code__.co_varnames

    _needs_mixture = getattr(criterion, 'needs_mixture', False)
    _is_amr = amr_mode in ('cls_only', 'joint')
    _all_demod_modes = {m for m in (demod_mode, demod_mode_phase1, demod_mode_phase2) if m is not None}
    _is_demod = any(m in ('demod_only', 'joint') for m in _all_demod_modes)
    _needs_bits = getattr(criterion, 'needs_bits', False)
    # When BER reporting is enabled, accumulate it during train/val so the
    # epoch logs and tqdm postfix expose BER trends in real time.
    _compute_epoch_ber = bool(report_ber)
    _should_load_bits = _needs_bits or _is_demod or demod_teacher_weight > 0 or report_ber
    _train_ber_stride = max(1, len(train_loader) // 16) if len(train_loader) > 0 else 1
    _val_ber_stride = max(1, len(val_loader) // 16) if len(val_loader) > 0 else 1
    _is_multitask = _is_amr or _is_demod  # model returns tuple output
    _ber_modulations_cache = {}

    if report_ber:
        logger.info(
            "BER reporting enabled: train/val loops will accumulate BER and "
            "final evaluation will also report BER."
        )
        logger.info(
            f"Train BER uses strict mode on sampled batches only "
            f"(stride={_train_ber_stride}). Validation BER is also sampled "
            f"(stride={_val_ber_stride}). Final evaluation remains full BER."
        )

    def _resolve_demod_mode_for_phase(active_phase):
        if active_phase == 1 and demod_mode_phase1 is not None:
            return demod_mode_phase1
        if active_phase == 2 and demod_mode_phase2 is not None:
            return demod_mode_phase2
        return demod_mode

    def _get_model_mode(current_amr_mode, current_demod_mode):
        """Return the mode string and flag for model forward."""
        if current_amr_mode in ('cls_only', 'joint'):
            return current_amr_mode
        if current_demod_mode in ('demod_only', 'joint'):
            return current_demod_mode
        return None

    def _resolve_lr_for_phase(active_phase):
        if active_phase == 1 and lr_phase1 is not None:
            return float(lr_phase1)
        if active_phase == 2 and lr_phase2 is not None:
            return float(lr_phase2)
        return None

    def _apply_optimizer_lr(target_lr, reason):
        if target_lr is None:
            return
        for group in optimizer.param_groups:
            group['lr'] = float(target_lr)
        logger.info(f"Optimizer LR set to {float(target_lr):.2e} ({reason})")

    def _reset_scheduler_state():
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.best = float('inf')
            scheduler.num_bad_epochs = 0
            scheduler.cooldown_counter = 0
            logger.info("Scheduler state reset for new training phase")
    
    current_amr_mode = amr_mode
    current_demod_mode = demod_mode
    _model_mode = _get_model_mode(current_amr_mode, current_demod_mode)
    _last_logged_demod_mode = None
    _last_active_phase = None
    _phase2_epoch_counter = 0
    _last_teacher_active = None

    def _model_forward(model, inputs, bits=None):
        kwargs = {}
        if current_demod_mode in ('demod_only', 'joint') and bits is not None:
            kwargs['num_bits'] = int(bits[0].shape[-1])
        if _model_mode is not None:
            model_output = model(inputs, mode=_model_mode, **kwargs)
        else:
            model_output = model(inputs, **kwargs)
        separation = extract_separation_output(model_output)
        if isinstance(separation, (list, tuple)):
            separation = select_finest_separation_output(separation)
        validate_source_tensor(separation, num_sources, 'model separation output')
        return model_output

    def _get_best_perm_and_sep(outputs, targets):
        sep_outputs = extract_separation_output(outputs)
        if isinstance(sep_outputs, (list, tuple)):
            sep_outputs = select_finest_separation_output(sep_outputs)
        validate_source_tensor(sep_outputs, num_sources, 'model separation output')
        validate_source_tensor(targets, num_sources, 'targets')
        sep_outputs = sep_outputs.float()
        targets_eval = targets.float()
        source_count_from_targets = targets_eval.shape[1] // 2
        if source_count_from_targets != num_sources:
            raise ValueError(
                f"targets encode {source_count_from_targets} sources but run expects {num_sources}"
            )
        outputs_eval, best_perm_per_sample = reorder_outputs_for_eval(
            sep_outputs,
            targets_eval,
            source_count_from_targets,
            pit_metric=eval_pit_metric,
        )
        return outputs_eval, targets_eval, source_count_from_targets, best_perm_per_sample

    dual_report_fixed_and_pit = str(eval_pit_metric).lower() == 'none'

    def _get_supplemental_pit_outputs(outputs_eval, targets_eval, source_count):
        if not dual_report_fixed_and_pit:
            return None
        pit_outputs, _ = reorder_outputs_for_eval(
            outputs_eval,
            targets_eval,
            source_count,
            pit_metric='si_snr_complex',
        )
        return pit_outputs

    def _to_fp32(obj):
        if torch.is_tensor(obj):
            return obj.float()
        if isinstance(obj, dict):
            return {key: _to_fp32(val) for key, val in obj.items()}
        if isinstance(obj, tuple):
            return tuple(_to_fp32(val) for val in obj)
        if isinstance(obj, list):
            return [_to_fp32(val) for val in obj]
        return obj

    def _extract_demod_outputs(model_output):
        if not isinstance(model_output, tuple) or len(model_output) < 2:
            return None, None
        aux = model_output[1]
        if isinstance(aux, dict):
            return aux.get('bit_logits'), aux.get('symbol_logits')
        return aux, None

    def _bits_to_symbol_labels(bits_1d, bits_per_symbol=3):
        if bits_1d is None:
            return None
        if bits_1d.numel() == 0 or (bits_1d.numel() % bits_per_symbol) != 0:
            return None
        bits_grouped = bits_1d.view(-1, bits_per_symbol).long()
        weights = (2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits_grouped.device)).view(1, -1)
        return torch.sum(bits_grouped * weights, dim=1)

    def _teacher_demod_is_active(active_phase):
        if demod_teacher_weight <= 0 or demod_teacher_phase2_epochs <= 0:
            return False
        if active_phase != 2:
            return False
        if current_demod_mode not in ('demod_only', 'joint'):
            return False
        return _phase2_epoch_counter <= demod_teacher_phase2_epochs

    def _compute_oracle_demod_loss(targets, bits):
        if bits is None:
            return None
        base_model = model.module if hasattr(model, 'module') else model
        demod_head = getattr(base_model, 'demod_head', None)
        if demod_head is None:
            return None

        num_sources = targets.shape[1] // 2
        if num_sources <= 0:
            return None

        num_bits = int(bits[0].shape[-1])
        teacher_bce = torch.tensor(0.0, device=targets.device)
        teacher_ce = torch.tensor(0.0, device=targets.device)
        teacher_ce_terms = 0

        for src_idx in range(num_sources):
            src_wave = targets[:, 2 * src_idx: 2 * src_idx + 2, :]
            bit_logits, symbol_logits = demod_head(src_wave, num_bits=num_bits)
            teacher_bce = teacher_bce + F.binary_cross_entropy_with_logits(
                bit_logits,
                bits[src_idx].float(),
                reduction='mean',
            )
            if symbol_logits is not None:
                symbol_targets = []
                max_symbols = None
                all_valid = True
                for b in range(bits[src_idx].shape[0]):
                    labels = _bits_to_symbol_labels(bits[src_idx][b], bits_per_symbol=3)
                    if labels is None:
                        all_valid = False
                        break
                    max_symbols = labels.shape[0] if max_symbols is None else min(max_symbols, labels.shape[0])
                    symbol_targets.append(labels)
                if all_valid and max_symbols is not None and max_symbols > 0:
                    stacked_targets = torch.stack([labels[:max_symbols] for labels in symbol_targets], dim=0)
                    teacher_ce = teacher_ce + F.cross_entropy(
                        symbol_logits[:, :max_symbols, :].reshape(-1, symbol_logits.shape[-1]),
                        stacked_targets.reshape(-1).to(symbol_logits.device),
                    )
                    teacher_ce_terms += 1

        teacher_bce = teacher_bce / num_sources
        if teacher_ce_terms > 0:
            teacher_ce = teacher_ce / teacher_ce_terms
        else:
            teacher_ce = torch.tensor(0.0, device=targets.device)
        return demod_teacher_weight * (teacher_bce + teacher_ce)

    def _compute_loss_fp32(outputs, targets, snr=None, inputs=None, bits=None, selected_criterion=None):
        # Keep numerically sensitive losses (e.g., SI-SNR/log/ratio) in FP32.
        # Multi-task models return (sep_output, aux_logits_list) — handle the tuple.
        active_criterion = selected_criterion if selected_criterion is not None else criterion
        # Generic waveform criteria operate on the first tensor when a model
        # also returns an auxiliary dictionary (sync, residual, or demod head).
        # This keeps complex mask metadata out of SI-SNR/Huber casting.
        if (
            isinstance(outputs, tuple)
            and len(outputs) >= 2
            and isinstance(outputs[1], dict)
            and not bool(getattr(active_criterion, 'needs_bits', False))
        ):
            out32 = _to_fp32(outputs[0])
        else:
            out32 = _to_fp32(outputs)
        tgt32 = targets.float()
        if bool(getattr(active_criterion, 'needs_bits', False)) and bits is not None:
            return active_criterion(out32, tgt32, bits=bits)
        if bool(getattr(active_criterion, 'needs_mixture', False)) and inputs is not None:
            inp32 = inputs.float()
            return active_criterion(out32, tgt32, inp32)
        if bool(getattr(active_criterion, 'needs_snr', False)):
            return active_criterion(out32, tgt32, snr)
        return active_criterion(out32, tgt32)

    def _compute_stage255_snr_aux_loss(outputs, snr):
        if stage255_snr_aux_weight <= 0.0 or snr is None or router_phase == 'expert-only':
            return None
        if not isinstance(outputs, tuple) or len(outputs) < 2 or not isinstance(outputs[1], dict):
            return None
        prediction = outputs[1].get('snr_prediction')
        if prediction is None:
            return None
        target = snr.float().reshape(-1)
        prediction = prediction.float().reshape(-1)
        if target.numel() == 1 and prediction.numel() > 1:
            target = target.expand_as(prediction)
        if target.numel() != prediction.numel():
            raise ValueError(
                f"Stage-255 SNR head produced {prediction.numel()} values for {target.numel()} targets"
            )
        snr_range = float(stage255_snr_aux_max_db) - float(stage255_snr_aux_min_db)
        target = (target - float(stage255_snr_aux_min_db)) / snr_range
        prediction = (prediction - float(stage255_snr_aux_min_db)) / snr_range
        return stage255_snr_aux_weight * F.smooth_l1_loss(prediction, target)

    def _curriculum_floor(epoch):
        curriculum_epochs = max(1, int(round(num_epochs * stage255_snr_curriculum_fraction)))
        denominator = max(1, curriculum_epochs - 1)
        progress = min(max(float(epoch) / denominator, 0.0), 1.0)
        return (
            float(stage255_snr_curriculum_start_db)
            + progress
            * (float(stage255_snr_curriculum_end_db) - float(stage255_snr_curriculum_start_db))
        )

    def _apply_stage255_snr_curriculum(inputs, targets, snr, bits, sync_metadata, epoch):
        if snr is None or inputs.size(0) <= 1:
            return inputs, targets, snr, bits, sync_metadata
        snr_values = snr.reshape(-1)
        if snr_values.numel() != inputs.size(0):
            return inputs, targets, snr, bits, sync_metadata
        keep = torch.ones_like(snr_values, dtype=torch.bool)
        if training_snr_floor_db is not None:
            keep &= snr_values >= float(training_snr_floor_db)
        if stage255_snr_curriculum_enable:
            keep &= snr_values >= _curriculum_floor(epoch)
        if bool(keep.all()):
            return inputs, targets, snr, bits, sync_metadata
        if not bool(keep.any()):
            if training_snr_floor_db is not None:
                raise RuntimeError(
                    f"Batch contains no samples at or above the required "
                    f"{training_snr_floor_db:g} dB teacher-pretraining floor"
                )
            keep[snr_values.argmax()] = True
        filtered_bits = None
        if bits is not None:
            filtered_bits = tuple(source_bits[keep] for source_bits in bits)
        filtered_metadata = sync_metadata
        if sync_metadata is not None:
            filtered_metadata = {
                key: value[keep] if torch.is_tensor(value) and value.ndim > 0
                and value.size(0) == keep.size(0) else value
                for key, value in sync_metadata.items()
            }
        return inputs[keep], targets[keep], snr[keep], filtered_bits, filtered_metadata

    def _final_separation_tensor(outputs):
        separated = extract_separation_output(outputs)
        if isinstance(separated, (list, tuple)):
            separated = select_finest_separation_output(separated)
        return separated

    def _sync_auxiliary(outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
            auxiliary = outputs[1]
            if 'snr_prediction' in auxiliary and 'sync_condition' in auxiliary:
                return auxiliary
        return None

    def _compute_sync_snr_aux_loss(outputs, snr):
        if sync_snr_aux_weight <= 0.0 or snr is None:
            return None
        auxiliary = _sync_auxiliary(outputs)
        if auxiliary is None:
            return None
        return sync_snr_aux_weight * sync_parameter_snr_supervision_loss(
            auxiliary,
            snr,
            min_db=sync_snr_aux_min_db,
            max_db=sync_snr_aux_max_db,
            beta=sync_snr_aux_beta,
        )

    def _compute_sync_physical_loss(outputs, targets, sync_metadata):
        if sync_physical_supervision_weight <= 0.0:
            return None
        auxiliary = _sync_auxiliary(outputs)
        if auxiliary is None:
            return None
        if sync_metadata is None:
            if sync_physical_require_metadata:
                raise RuntimeError(
                    "Physical synchronization supervision is required, but this batch "
                    "contains no MATLAB generator metadata. Use a supported synthetic "
                    "dataset or disable sync_physical_require_metadata explicitly."
                )
            return None
        if sync_physical_require_metadata:
            required_masks = (
                (sync_physical_cfo_weight, 'cfo_valid'),
                (sync_physical_phase_weight, 'phase_valid'),
                (sync_physical_timing_weight, 'timing_valid'),
                (sync_physical_sps_weight, 'sps_valid'),
                (sync_physical_drift_weight, 'drift_valid'),
            )
            missing = [
                name for weight, name in required_masks
                if float(weight) > 0.0
                and (
                    name not in sync_metadata
                    or not bool(sync_metadata[name].to(dtype=torch.bool).any())
                )
            ]
            if missing:
                raise RuntimeError(
                    "Physical synchronization supervision requires valid labels for "
                    + ", ".join(missing)
                )
        return sync_physical_supervision_weight * sync_parameter_physical_supervision_loss(
            auxiliary,
            sync_metadata,
            _final_separation_tensor(outputs).float(),
            targets.float(),
            num_sources=num_sources,
            cfo_weight=sync_physical_cfo_weight,
            phase_weight=sync_physical_phase_weight,
            timing_weight=sync_physical_timing_weight,
            sps_weight=sync_physical_sps_weight,
            drift_weight=sync_physical_drift_weight,
            cfo_scale=sync_cfo_scale,
            drift_scale=sync_phase_drift_scale,
            beta=sync_physical_beta,
            eps=cross_snr_eps,
        )

    def _latent_mask_auxiliary(outputs):
        if (
            isinstance(outputs, tuple)
            and len(outputs) >= 2
            and isinstance(outputs[1], dict)
        ):
            return outputs[1]
        return None

    def _compute_latent_mask_auxiliary_loss(outputs, inputs, targets):
        """Train the explicit residual slot and waveform mixture closure.

        These terms are active only for Stage373/374-style outputs. They use
        the observed mixture and clean source targets, but never generator
        parameters such as true SNR, CFO, timing, or SPS.
        """
        if latent_mask_residual_weight <= 0.0 and latent_mask_mixture_weight <= 0.0:
            return _final_separation_tensor(outputs).new_zeros(())
        auxiliary = _latent_mask_auxiliary(outputs)
        residual = auxiliary.get("residual_output") if auxiliary else None
        if residual is None:
            return _final_separation_tensor(outputs).new_zeros(())
        residual = residual.float()
        mixture = inputs.float()
        clean_mix = clean_mixture_from_targets(targets.float(), num_sources)
        target_residual = mixture - clean_mix
        if residual.shape != target_residual.shape:
            raise ValueError(
                "latent-mask residual slot must have the observed mixture shape: "
                f"got {tuple(residual.shape)}, expected {tuple(target_residual.shape)}"
            )
        loss = residual.new_zeros(())
        if latent_mask_residual_weight > 0.0:
            loss = loss + latent_mask_residual_weight * F.smooth_l1_loss(
                residual,
                target_residual,
                beta=latent_mask_residual_beta,
            )
        if latent_mask_mixture_weight > 0.0:
            separated = _final_separation_tensor(outputs).float()
            expected_channels = 2 * num_sources
            if separated.ndim != 3 or separated.size(1) != expected_channels:
                raise ValueError(
                    "latent-mask source output has an invalid channel count: "
                    f"got {tuple(separated.shape)}, expected {expected_channels} channels"
                )
            source_sum = separated.reshape(
                separated.size(0), num_sources, 2, separated.size(-1)
            ).sum(dim=1)
            loss = loss + latent_mask_mixture_weight * F.smooth_l1_loss(
                source_sum + residual,
                mixture,
                beta=latent_mask_residual_beta,
            )
        return loss

    @torch.no_grad()
    def _update_cross_snr_teacher():
        nonlocal cross_snr_teacher_updates
        if cross_snr_teacher is None or cross_snr_teacher_mode != 'ema':
            return
        cross_snr_teacher_updates += 1
        # The warm-up avoids an overly stale teacher at the beginning.
        decay = min(
            cross_snr_ema_decay,
            1.0 - 1.0 / float(cross_snr_teacher_updates + 1),
        )
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = (
            cross_snr_teacher.module
            if hasattr(cross_snr_teacher, 'module')
            else cross_snr_teacher
        )
        for teacher_parameter, student_parameter in zip(
            teacher_model.parameters(), student_model.parameters()
        ):
            teacher_parameter.mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(
            teacher_model.buffers(), student_model.buffers()
        ):
            teacher_buffer.copy_(student_buffer.detach())

    def _compute_cross_snr_extra_loss(
        outputs, inputs, targets, snr, bits, sync_metadata, epoch
    ):
        if not cross_snr_enable or snr is None or cross_snr_probability <= 0.0:
            return _final_separation_tensor(outputs).new_zeros(())
        if float(torch.rand((), device=inputs.device).item()) >= cross_snr_probability:
            return _final_separation_tensor(outputs).new_zeros(())

        with torch.no_grad():
            if cross_snr_pair_mode == 'curriculum_student':
                sampled_snr = sample_progressive_snr_range(
                    inputs.size(0),
                    epoch,
                    num_epochs,
                    cross_snr_curriculum_ranges,
                    tuple(float(value) for value in cross_snr_curriculum_boundaries),
                    device=inputs.device,
                )
                partner_inputs, partner_snr = build_snr_view(
                    inputs,
                    targets,
                    sampled_snr,
                    num_sources=num_sources,
                    eps=cross_snr_eps,
                )
            else:
                low_snr_db = curriculum_low_snr(
                    epoch,
                    num_epochs,
                    cross_snr_low_start_db,
                    cross_snr_low_middle_db,
                    cross_snr_low_final_db,
                    cross_snr_first_fraction,
                    cross_snr_second_fraction,
                )
                partner_inputs, partner_snr = build_cross_snr_partner(
                    inputs,
                    targets,
                    snr,
                    num_sources=num_sources,
                    high_snr_db=cross_snr_high_db,
                    low_snr_db=low_snr_db,
                    eps=cross_snr_eps,
                )

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                partner_outputs = _model_forward(model, partner_inputs, bits=bits)
        else:
            partner_outputs = _model_forward(model, partner_inputs, bits=bits)

        cross_snr_partner_criterion = getattr(criterion, 'cross_snr_partner_criterion', criterion)
        partner_loss = _compute_loss_fp32(
            partner_outputs,
            targets,
            partner_snr,
            inputs=partner_inputs,
            bits=bits,
            selected_criterion=cross_snr_partner_criterion,
        )
        partner_sync_snr_loss = _compute_sync_snr_aux_loss(partner_outputs, partner_snr)
        partner_sync_physical_loss = _compute_sync_physical_loss(
            partner_outputs, targets, sync_metadata
        )

        teacher_high_outputs = None
        if cross_snr_teacher is not None:
            if cross_snr_teacher_view == 'clean':
                high_inputs = clean_mixture_from_targets(targets, num_sources)
            else:
                original_is_low = (
                    snr.to(device=inputs.device, dtype=torch.float32).reshape(-1)
                    <= partner_snr.to(device=inputs.device, dtype=torch.float32).reshape(-1)
                ).view(-1, 1, 1)
                high_inputs = torch.where(original_is_low, partner_inputs, inputs)
            cross_snr_teacher.eval()
            with torch.no_grad():
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        teacher_high_outputs = _model_forward(
                            cross_snr_teacher, high_inputs, bits=bits
                        )
                else:
                    teacher_high_outputs = _model_forward(
                        cross_snr_teacher, high_inputs, bits=bits
                    )
        consistency = cross_snr_teacher_consistency_loss(
            _final_separation_tensor(outputs).float(),
            _final_separation_tensor(partner_outputs).float(),
            targets.float(),
            snr,
            partner_snr,
            num_sources=num_sources,
            beta=cross_snr_consistency_beta,
            eps=cross_snr_eps,
            shared_permutation=bool(cross_snr_shared_permutation),
            teacher_high_outputs=(
                _final_separation_tensor(teacher_high_outputs).float()
                if teacher_high_outputs is not None
                else None
            ),
            force_partner_student=(cross_snr_pair_mode == 'curriculum_student'),
        )
        feature_consistency = consistency.new_zeros(())
        if (
            cross_snr_feature_consistency_weight > 0.0
            and teacher_high_outputs is not None
            and _sync_auxiliary(partner_outputs) is not None
            and _sync_auxiliary(teacher_high_outputs) is not None
        ):
            feature_consistency = cross_snr_feature_consistency_loss(
                _sync_auxiliary(partner_outputs),
                _sync_auxiliary(teacher_high_outputs),
                beta=cross_snr_feature_consistency_beta,
                eps=cross_snr_eps,
            )
        sync_consistency = consistency.new_zeros(())
        original_sync_aux = _sync_auxiliary(outputs)
        partner_sync_aux = _sync_auxiliary(partner_outputs)
        teacher_sync_aux = _sync_auxiliary(teacher_high_outputs)
        if original_sync_aux is not None:
            original_sync_aux = pit_align_sync_auxiliary(
                original_sync_aux,
                _final_separation_tensor(outputs).float(),
                targets.float(),
                num_sources=num_sources,
            )
        if partner_sync_aux is not None:
            partner_sync_aux = pit_align_sync_auxiliary(
                partner_sync_aux,
                _final_separation_tensor(partner_outputs).float(),
                targets.float(),
                num_sources=num_sources,
            )
        if teacher_sync_aux is not None:
            teacher_sync_aux = pit_align_sync_auxiliary(
                teacher_sync_aux,
                _final_separation_tensor(teacher_high_outputs).float(),
                targets.float(),
                num_sources=num_sources,
            )
        if (
            sync_cross_snr_consistency_weight > 0.0
            and original_sync_aux is not None
            and partner_sync_aux is not None
        ):
            sync_consistency = sync_parameter_cross_snr_consistency_loss(
                original_sync_aux,
                partner_sync_aux,
                snr,
                partner_snr,
                teacher_high_auxiliary=teacher_sync_aux,
                beta=sync_cross_snr_consistency_beta,
                cfo_scale=sync_cfo_scale,
                phase_drift_scale=sync_phase_drift_scale,
                force_partner_student=(cross_snr_pair_mode == 'curriculum_student'),
            )
        pair_term = partner_loss
        if partner_sync_snr_loss is not None:
            pair_term = pair_term + partner_sync_snr_loss
        if partner_sync_physical_loss is not None:
            pair_term = pair_term + partner_sync_physical_loss
        return (
            float(cross_snr_pair_weight) * pair_term
            + float(cross_snr_consistency_weight) * consistency
            + cross_snr_feature_consistency_weight * feature_consistency
            + sync_cross_snr_consistency_weight * sync_consistency
        )

    def _compute_phase_equiv_extra_loss(outputs, inputs, targets, snr, bits):
        if not phase_equiv_enable or phase_equiv_probability <= 0.0:
            return _final_separation_tensor(outputs).new_zeros(())
        if float(torch.rand((), device=inputs.device).item()) >= phase_equiv_probability:
            return _final_separation_tensor(outputs).new_zeros(())

        max_angle = abs(float(phase_equiv_max_degrees)) * np.pi / 180.0
        angles = (torch.rand(inputs.size(0), device=inputs.device) * 2.0 - 1.0) * max_angle
        rotated_inputs = rotate_iq(inputs, angles)
        rotated_targets = rotate_iq(targets, angles)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                rotated_outputs = _model_forward(model, rotated_inputs, bits=bits)
        else:
            rotated_outputs = _model_forward(model, rotated_inputs, bits=bits)

        supervised = _compute_loss_fp32(
            rotated_outputs,
            rotated_targets,
            snr,
            inputs=rotated_inputs,
            bits=bits,
            selected_criterion=getattr(criterion, 'cross_snr_partner_criterion', criterion),
        )
        consistency = phase_equivariance_consistency_loss(
            _final_separation_tensor(outputs).float(),
            _final_separation_tensor(rotated_outputs).float(),
            targets.float(),
            angles,
            num_sources=num_sources,
            beta=phase_equiv_beta,
            eps=phase_equiv_eps,
        )
        return (
            float(phase_equiv_supervised_weight) * supervised
            + float(phase_equiv_consistency_weight) * consistency
        )

    def _compute_rf_equiv_extra_loss(outputs, inputs, targets, snr, bits):
        if not rf_equiv_enable or rf_equiv_probability <= 0.0:
            return _final_separation_tensor(outputs).new_zeros(())
        if float(torch.rand((), device=inputs.device).item()) >= rf_equiv_probability:
            return _final_separation_tensor(outputs).new_zeros(())

        with torch.no_grad():
            parameters = sample_fixed_slot_rf_parameters(
                inputs.size(0),
                num_sources,
                max_phase_degrees=rf_equiv_max_phase_degrees,
                max_cfo_cycles_per_sample=rf_equiv_max_cfo_cycles_per_sample,
                max_gain_db=rf_equiv_max_gain_db,
                max_shift_samples=rf_equiv_max_shift_samples,
                conjugate_probability=rf_equiv_conjugate_probability,
                source_mode=rf_equiv_source_mode,
                device=inputs.device,
            )
            transformed_inputs, transformed_targets = build_fixed_slot_rf_view(
                inputs,
                targets,
                parameters,
                num_sources=num_sources,
                source_mode=rf_equiv_source_mode,
            )
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                transformed_outputs = _model_forward(
                    model, transformed_inputs, bits=bits
                )
        else:
            transformed_outputs = _model_forward(
                model, transformed_inputs, bits=bits
            )
        supervised = _compute_loss_fp32(
            transformed_outputs,
            transformed_targets,
            snr,
            inputs=transformed_inputs,
            bits=bits,
            selected_criterion=getattr(
                criterion, 'cross_snr_partner_criterion', criterion
            ),
        )
        consistency = fixed_slot_rf_equivariance_consistency_loss(
            _final_separation_tensor(outputs).float(),
            _final_separation_tensor(transformed_outputs).float(),
            parameters,
            num_sources=num_sources,
            beta=rf_equiv_beta,
            eps=rf_equiv_eps,
        )
        return (
            float(rf_equiv_supervised_weight) * supervised
            + float(rf_equiv_consistency_weight) * consistency
        )

    def _get_ber_modulations(num_sources):
        if num_sources not in _ber_modulations_cache:
            _ber_modulations_cache[num_sources] = _infer_modulations_from_data_choice(data_choice, num_sources)
        return _ber_modulations_cache[num_sources]

    def _extract_bits_from_rest(rest):
        if _should_load_bits:
            for item in rest:
                if isinstance(item, (tuple, list)) and len(item) == num_sources:
                    return tuple(b.to(device, non_blocking=True) for b in item)
        return None

    def _extract_sync_metadata_from_rest(rest):
        for item in rest:
            if isinstance(item, dict) and 'sync_metadata_version' in item:
                return {
                    key: value.to(device, non_blocking=True)
                    if torch.is_tensor(value) else value
                    for key, value in item.items()
                }
        return None

    def _accumulate_batch_ber(outputs, targets, bits, strict_sum, strict_den, batch_idx=None, phase="train"):
        if not _compute_epoch_ber or bits is None:
            return strict_sum, strict_den
        if phase == "train" and batch_idx is not None and (int(batch_idx) % _train_ber_stride) != 0:
            return strict_sum, strict_den
        if phase == "val" and batch_idx is not None and (int(batch_idx) % _val_ber_stride) != 0:
            return strict_sum, strict_den

        outputs_eval, targets_eval, num_sources, _ = _get_best_perm_and_sep(outputs, targets)
        ber_modulations = _get_ber_modulations(num_sources)
        if not ber_modulations:
            return strict_sum, strict_den

        for src_idx in range(num_sources):
            if src_idx >= len(ber_modulations):
                continue
            pred_k = outputs_eval[:, 2 * src_idx: 2 * src_idx + 2, :]
            tgt_k = targets_eval[:, 2 * src_idx: 2 * src_idx + 2, :]

            ber_strict = strict_ber_iq_from_bits(
                pred_k.float(),
                tgt_k.float(),
                bits[src_idx],
                modulation=ber_modulations[src_idx],
                offset_search=ber_offset_search,
                protocol=data_choice,
            )
            if torch.isfinite(ber_strict):
                strict_sum += float(ber_strict.item())
                strict_den += 1
        return strict_sum, strict_den

    train_ber_strict_vals = []
    train_ber_oracle_vals = []
    val_ber_strict_vals = []
    val_ber_oracle_vals = []

    def _build_training_checkpoint(epoch):
        return {
            'checkpoint_version': 3,
            'num_sources': num_sources,
            'epoch': int(epoch),
            'next_epoch': int(epoch) + 1,
            'model_state_dict': model.state_dict(),
            'cross_snr_teacher_state_dict': (
                cross_snr_teacher.state_dict()
                if cross_snr_teacher is not None
                else None
            ),
            'cross_snr_teacher_updates': int(cross_snr_teacher_updates),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'warmup_scheduler_state_dict': warmup_scheduler.state_dict() if warmup_epochs > 0 else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'rng_state': _capture_rng_state(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_si_snr_paper': val_si_snr_paper_vals,
            'val_si_snr_repo': val_si_snr_repo_vals,
            'train_ber_strict': train_ber_strict_vals,
            'train_ber_oracle': train_ber_oracle_vals,
            'val_ber_strict': val_ber_strict_vals,
            'val_ber_oracle': val_ber_oracle_vals,
            'learning_rates': learning_rates,
            'epoch_times': epoch_times,
            'model_diagnostics': model_diagnostics_history,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'no_improve_counter': no_improve_counter,
            'best_model_state_dict': best_model_weights,
            'last_active_phase': _last_active_phase,
            'phase2_epoch_counter': _phase2_epoch_counter,
            'last_logged_demod_mode': _last_logged_demod_mode,
            'last_teacher_active': _last_teacher_active,
            'training_config': {
                'data_choice': data_choice,
                'batch_size': batch_size,
                'input_size': input_size,
                'num_epochs': num_epochs,
                'early_stop_patience': early_stop_patience,
                'gradient_clip_norm': gradient_clip_norm,
                'save_checkpoint_every': save_checkpoint_every,
                'init_checkpoint': str(init_checkpoint) if init_checkpoint else None,
                'accumulation_steps': accumulation_steps,
                'warmup_epochs': warmup_epochs,
                'use_mixed_precision': use_mixed_precision,
                'report_ber': report_ber,
                'ber_offset_search': ber_offset_search,
                'ber_mode': ber_mode,
                'ber_num_files': ber_num_files,
                'ber_compute_oracle': ber_compute_oracle,
                'amr_mode': amr_mode,
                'demod_mode': demod_mode,
                'demod_mode_phase1': demod_mode_phase1,
                'demod_mode_phase2': demod_mode_phase2,
                'lr_phase1': lr_phase1,
                'lr_phase2': lr_phase2,
                'demod_teacher_weight': demod_teacher_weight,
                'demod_teacher_phase2_epochs': demod_teacher_phase2_epochs,
                'eval_pit_metric': eval_pit_metric,
                'signal_names': signal_names,
                'num_sources': num_sources,
                'cross_snr_enable': cross_snr_enable,
                'cross_snr_probability': cross_snr_probability,
                'cross_snr_high_db': cross_snr_high_db,
                'cross_snr_low_start_db': cross_snr_low_start_db,
                'cross_snr_low_middle_db': cross_snr_low_middle_db,
                'cross_snr_low_final_db': cross_snr_low_final_db,
                'cross_snr_first_fraction': cross_snr_first_fraction,
                'cross_snr_second_fraction': cross_snr_second_fraction,
                'cross_snr_pair_weight': cross_snr_pair_weight,
                'cross_snr_consistency_weight': cross_snr_consistency_weight,
                'cross_snr_consistency_beta': cross_snr_consistency_beta,
                'cross_snr_eps': cross_snr_eps,
                'cross_snr_shared_permutation': bool(cross_snr_shared_permutation),
                'cross_snr_ema_teacher_enable': cross_snr_ema_teacher_enable,
                'cross_snr_ema_decay': cross_snr_ema_decay,
                'cross_snr_teacher_mode': cross_snr_teacher_mode,
                'cross_snr_teacher_checkpoint': cross_snr_teacher_checkpoint,
                'cross_snr_teacher_view': cross_snr_teacher_view,
                'cross_snr_pair_mode': cross_snr_pair_mode,
                'cross_snr_feature_consistency_weight': cross_snr_feature_consistency_weight,
                'cross_snr_feature_consistency_beta': cross_snr_feature_consistency_beta,
                'cross_snr_curriculum_ranges': cross_snr_curriculum_ranges,
                'cross_snr_curriculum_boundaries': cross_snr_curriculum_boundaries,
                'sync_snr_aux_weight': sync_snr_aux_weight,
                'sync_snr_aux_min_db': sync_snr_aux_min_db,
                'sync_snr_aux_max_db': sync_snr_aux_max_db,
                'sync_snr_aux_beta': sync_snr_aux_beta,
                'sync_cross_snr_consistency_weight': sync_cross_snr_consistency_weight,
                'sync_cross_snr_consistency_beta': sync_cross_snr_consistency_beta,
                'sync_cfo_scale': sync_cfo_scale,
                'sync_phase_drift_scale': sync_phase_drift_scale,
                'sync_physical_require_metadata': sync_physical_require_metadata,
                'sync_physical_supervision_weight': sync_physical_supervision_weight,
                'sync_physical_cfo_weight': sync_physical_cfo_weight,
                'sync_physical_phase_weight': sync_physical_phase_weight,
                'sync_physical_timing_weight': sync_physical_timing_weight,
                'sync_physical_sps_weight': sync_physical_sps_weight,
                'sync_physical_drift_weight': sync_physical_drift_weight,
                'sync_physical_beta': sync_physical_beta,
                'training_snr_floor_db': training_snr_floor_db,
                'phase_equiv_enable': phase_equiv_enable,
                'phase_equiv_probability': phase_equiv_probability,
                'phase_equiv_supervised_weight': phase_equiv_supervised_weight,
                'phase_equiv_consistency_weight': phase_equiv_consistency_weight,
                'phase_equiv_max_degrees': phase_equiv_max_degrees,
                'phase_equiv_beta': phase_equiv_beta,
                'phase_equiv_eps': phase_equiv_eps,
                'rf_equiv_enable': rf_equiv_enable,
                'rf_equiv_probability': rf_equiv_probability,
                'rf_equiv_supervised_weight': rf_equiv_supervised_weight,
                'rf_equiv_consistency_weight': rf_equiv_consistency_weight,
                'rf_equiv_max_phase_degrees': rf_equiv_max_phase_degrees,
                'rf_equiv_max_cfo_cycles_per_sample': rf_equiv_max_cfo_cycles_per_sample,
                'rf_equiv_max_gain_db': rf_equiv_max_gain_db,
                'rf_equiv_max_shift_samples': rf_equiv_max_shift_samples,
                'rf_equiv_conjugate_probability': rf_equiv_conjugate_probability,
                'rf_equiv_source_mode': rf_equiv_source_mode,
                'rf_equiv_beta': rf_equiv_beta,
                'rf_equiv_eps': rf_equiv_eps,
                'latent_mask_residual_weight': latent_mask_residual_weight,
                'latent_mask_mixture_weight': latent_mask_mixture_weight,
                'latent_mask_residual_beta': latent_mask_residual_beta,
                'stage255_snr_aux_weight': stage255_snr_aux_weight,
                'stage255_snr_aux_min_db': stage255_snr_aux_min_db,
                'stage255_snr_aux_max_db': stage255_snr_aux_max_db,
                'stage255_snr_curriculum_enable': stage255_snr_curriculum_enable,
                'stage255_snr_curriculum_start_db': stage255_snr_curriculum_start_db,
                'stage255_snr_curriculum_end_db': stage255_snr_curriculum_end_db,
                'stage255_snr_curriculum_fraction': stage255_snr_curriculum_fraction,
                'stage255_expert_pretrain_epochs': stage255_expert_pretrain_epochs,
                'stage255_router_warmup_epochs': stage255_router_warmup_epochs,
                'stage255_router_joint_lr_scale': stage255_router_joint_lr_scale,
            },
        }

    def _load_training_checkpoint(path):
        nonlocal start_epoch, best_val_loss, best_epoch, no_improve_counter, best_model_weights
        nonlocal train_losses, val_losses, val_si_snr_paper_vals, val_si_snr_repo_vals
        nonlocal learning_rates, epoch_times, model_diagnostics_history
        nonlocal train_ber_strict_vals, train_ber_oracle_vals, val_ber_strict_vals, val_ber_oracle_vals
        nonlocal _last_active_phase, _phase2_epoch_counter, _last_logged_demod_mode, _last_teacher_active
        nonlocal resume_cross_snr_teacher_state, cross_snr_teacher_updates

        checkpoint = _load_checkpoint_file(path)
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f"Checkpoint {path} does not contain model_state_dict")
        saved_num_sources = checkpoint.get('num_sources')
        if saved_num_sources is None:
            saved_num_sources = checkpoint.get('training_config', {}).get('num_sources')
        if saved_num_sources is not None and int(saved_num_sources) != num_sources:
            raise ValueError(
                f"Checkpoint {path} was created for num_sources={saved_num_sources}, "
                f"but this run uses num_sources={num_sources}."
            )
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            if not resume_allow_partial:
                raise
            _load_partial_model_state(checkpoint['model_state_dict'], path)
            return
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if warmup_epochs > 0 and checkpoint.get('warmup_scheduler_state_dict') is not None:
            warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        completed_epoch = int(checkpoint.get('epoch', -1))
        start_epoch = int(checkpoint.get('next_epoch', completed_epoch + 1))
        train_losses = list(checkpoint.get('train_losses', []))
        val_losses = list(checkpoint.get('val_losses', []))
        val_si_snr_paper_vals = list(checkpoint.get('val_si_snr_paper', []))
        val_si_snr_repo_vals = list(checkpoint.get('val_si_snr_repo', []))
        learning_rates = list(checkpoint.get('learning_rates', []))
        epoch_times = list(checkpoint.get('epoch_times', []))
        model_diagnostics_history = list(
            checkpoint.get('model_diagnostics', [])
        )
        train_ber_strict_vals = list(checkpoint.get('train_ber_strict', []))
        train_ber_oracle_vals = list(checkpoint.get('train_ber_oracle', []))
        val_ber_strict_vals = list(checkpoint.get('val_ber_strict', []))
        val_ber_oracle_vals = list(checkpoint.get('val_ber_oracle', []))
        best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        no_improve_counter = int(checkpoint.get('no_improve_counter', no_improve_counter))
        best_model_weights = checkpoint.get('best_model_state_dict', None)
        if best_model_weights is None and 'model_state_dict' in checkpoint:
            best_model_weights = _clone_state_dict_to_cpu(checkpoint['model_state_dict'])
        _last_active_phase = checkpoint.get('last_active_phase', _last_active_phase)
        _phase2_epoch_counter = int(checkpoint.get('phase2_epoch_counter', _phase2_epoch_counter))
        _last_logged_demod_mode = checkpoint.get('last_logged_demod_mode', _last_logged_demod_mode)
        _last_teacher_active = checkpoint.get('last_teacher_active', _last_teacher_active)
        resume_cross_snr_teacher_state = checkpoint.get('cross_snr_teacher_state_dict')
        cross_snr_teacher_updates = int(checkpoint.get('cross_snr_teacher_updates', 0))
        _restore_rng_state(checkpoint.get('rng_state'))

        logger.info(
            f"Resumed checkpoint from {path}: completed epoch {completed_epoch + 1}, "
            f"continuing at epoch {start_epoch + 1}/{num_epochs}, best_val_loss={best_val_loss:.4f}"
        )
        if start_epoch >= num_epochs:
            logger.warning(
                f"Resume checkpoint already reached epoch {start_epoch}; "
                f"num_epochs={num_epochs}. No additional training epochs will run."
            )

    def _load_initial_model_weights(path):
        checkpoint = _load_checkpoint_file(path)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Initialization checkpoint {path} is not a state dictionary")
        state_dict = checkpoint.get('best_model_state_dict')
        source = 'best_model_state_dict'
        if state_dict is None:
            state_dict = checkpoint.get('model_state_dict')
            source = 'model_state_dict'
        if state_dict is None:
            state_dict = checkpoint
            source = 'state_dict'
        model.load_state_dict(state_dict)
        logger.info(
            f"Initialized model from {path} ({source}); starting a fresh run at epoch 1. "
            "Optimizer, scheduler, scaler, RNG, and metric history were not restored."
        )

    if init_checkpoint:
        _load_initial_model_weights(Path(init_checkpoint))
    if resume_checkpoint:
        _load_training_checkpoint(Path(resume_checkpoint))
    for component_checkpoint in component_checkpoints or ():
        _load_component_checkpoint(Path(component_checkpoint))

    if cross_snr_enable and (
        cross_snr_ema_teacher_enable or cross_snr_teacher_mode == 'frozen'
    ):
        teacher_source = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        cross_snr_teacher = copy.deepcopy(teacher_source).to(device)
        if resume_cross_snr_teacher_state is not None:
            cross_snr_teacher.load_state_dict(resume_cross_snr_teacher_state)
        elif cross_snr_teacher_mode == 'frozen':
            if not cross_snr_teacher_checkpoint:
                raise ValueError(
                    "cross_snr_teacher_mode='frozen' requires "
                    "--cross_snr_teacher_checkpoint from the same backbone architecture"
                )
            teacher_checkpoint_path = Path(cross_snr_teacher_checkpoint)
            checkpoint = _load_checkpoint_file(teacher_checkpoint_path)
            if not isinstance(checkpoint, dict):
                raise TypeError(
                    f"Frozen teacher checkpoint {teacher_checkpoint_path} is not a state dictionary"
                )
            teacher_state = checkpoint.get('best_model_state_dict')
            if teacher_state is None:
                teacher_state = checkpoint.get('model_state_dict')
            if teacher_state is None:
                teacher_state = checkpoint
            teacher_target_state = cross_snr_teacher.state_dict()
            source_has_module = any(key.startswith('module.') for key in teacher_state)
            target_has_module = any(key.startswith('module.') for key in teacher_target_state)
            if source_has_module and not target_has_module:
                teacher_state = {
                    (key[7:] if key.startswith('module.') else key): value
                    for key, value in teacher_state.items()
                }
            elif target_has_module and not source_has_module:
                teacher_state = {
                    f'module.{key}': value for key, value in teacher_state.items()
                }
            try:
                cross_snr_teacher.load_state_dict(teacher_state)
            except RuntimeError as exc:
                raise ValueError(
                    "Frozen cross-SNR teacher checkpoint is not architecture-compatible "
                    "with the selected student stage. Use a checkpoint produced by the "
                    "same stage and source count."
                ) from exc
        for parameter in cross_snr_teacher.parameters():
            parameter.requires_grad_(False)
        cross_snr_teacher.eval()
        logger.info(
            f"Initialized cross-SNR {cross_snr_teacher_mode} teacher "
            f"(checkpoint={cross_snr_teacher_checkpoint}, "
            f"decay={cross_snr_ema_decay:.5f}, updates={cross_snr_teacher_updates})"
        )

    routing_model = model.module if hasattr(model, 'module') else model
    routing_parameters = (
        tuple(routing_model.routing_parameters())
        if hasattr(routing_model, 'routing_parameters')
        else ()
    )
    routing_parameter_ids = {id(parameter) for parameter in routing_parameters}
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
    }
    original_counterfactual_enable = getattr(
        routing_model, 'counterfactual_enable', None
    )
    router_phase = None

    def _set_stage255_router_phase(epoch):
        nonlocal router_phase
        if (
            not routing_parameters
            or stage255_expert_pretrain_epochs + stage255_router_warmup_epochs <= 0
        ):
            return
        expert_pretrain = int(epoch) < stage255_expert_pretrain_epochs
        router_warmup = (
            stage255_expert_pretrain_epochs
            <= int(epoch)
            < stage255_expert_pretrain_epochs + stage255_router_warmup_epochs
        )
        if expert_pretrain:
            next_phase = 'expert-only'
        elif router_warmup:
            next_phase = 'router-only'
        else:
            next_phase = 'joint'
        for parameter in model.parameters():
            trainable = original_requires_grad[id(parameter)]
            if expert_pretrain:
                trainable = trainable and id(parameter) not in routing_parameter_ids
            elif router_warmup:
                trainable = trainable and id(parameter) in routing_parameter_ids
            parameter.requires_grad_(trainable)
        if original_counterfactual_enable is not None:
            routing_model.counterfactual_enable = bool(
                original_counterfactual_enable and not expert_pretrain
            )
        if next_phase != router_phase:
            router_phase = next_phase
            logger.info(
                f"Stage-255 training phase: {next_phase} (epoch {epoch + 1}, "
                f"router_grad_scale={stage255_router_joint_lr_scale if next_phase == 'joint' else 1.0:.3f})"
            )

    def _scale_stage255_router_gradients(epoch):
        if (
            not routing_parameters
            or int(epoch) < stage255_expert_pretrain_epochs + stage255_router_warmup_epochs
            or stage255_router_joint_lr_scale >= 1.0
        ):
            return
        for parameter in routing_parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(stage255_router_joint_lr_scale)

    if stage255_snr_aux_weight > 0.0 and routing_parameters:
        logger.info(
            f"Stage-255 auxiliary SNR regression enabled (weight={stage255_snr_aux_weight:g}, "
            f"range=[{float(stage255_snr_aux_min_db):g}, {float(stage255_snr_aux_max_db):g}] dB)"
        )
    if stage255_snr_curriculum_enable:
        logger.info(
            "Stage-255 high-to-low SNR curriculum enabled: "
            f"floor={float(stage255_snr_curriculum_start_db):g}->"
            f"{float(stage255_snr_curriculum_end_db):g} dB over "
            f"{stage255_snr_curriculum_fraction:.2f} of training"
        )

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        train_sampler = getattr(train_loader, "sampler", None)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        _set_dataset_epoch(train_loader.dataset, epoch)
        _set_stage255_router_phase(epoch)

        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        active_loss_name = criterion.get_active_name() if hasattr(criterion, "get_active_name") else None
        active_phase = criterion.get_active_phase() if hasattr(criterion, "get_active_phase") else None
        if active_phase is not None and _last_active_phase is not None and active_phase != _last_active_phase:
            # Phase-1 and phase-2 losses are usually on different scales.
            # Carrying over the old best_val_loss/early-stop state makes the
            # second phase look artificially worse and can stop it too early.
            best_val_loss = float('inf')
            no_improve_counter = 0
            best_model_weights = None
            best_epoch = None
            logger.info(f"Validation tracking reset for loss phase {active_phase}")
            _apply_optimizer_lr(_resolve_lr_for_phase(active_phase), reason=f"phase {active_phase} switch")
            _reset_scheduler_state()
        _last_active_phase = active_phase
        if active_phase == 2:
            _phase2_epoch_counter += 1
        else:
            _phase2_epoch_counter = 0
        if active_phase is not None:
            current_demod_mode = _resolve_demod_mode_for_phase(active_phase)
        else:
            current_demod_mode = demod_mode
        _model_mode = _get_model_mode(current_amr_mode, current_demod_mode)
        if current_demod_mode != _last_logged_demod_mode:
            logger.info(f"Active Demod Mode: {current_demod_mode} (epoch {epoch + 1})")
            _last_logged_demod_mode = current_demod_mode
        teacher_active = _teacher_demod_is_active(active_phase)
        if teacher_active != _last_teacher_active:
            if teacher_active:
                logger.info(
                    f"Oracle demod warm-up active (phase2 epoch {_phase2_epoch_counter}/"
                    f"{demod_teacher_phase2_epochs}, weight={demod_teacher_weight:.2f})"
                )
            elif demod_teacher_weight > 0 and demod_teacher_phase2_epochs > 0:
                logger.info("Oracle demod warm-up inactive")
            _last_teacher_active = teacher_active
        
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        train_loss_components = defaultdict(float)  # Record loss components
        train_ber_strict_sum = 0.0
        train_ber_strict_den = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if use_tqdm:
            with tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{num_epochs}',
                unit='batch',
                colour='green',
                **tqdm_kwargs,
            ) as pbar:
                for batch_idx, (inputs, targets, snr, *_rest) in enumerate(pbar):
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    snr = snr.to(device, non_blocking=True) if snr is not None else None
                    bits = _extract_bits_from_rest(_rest)
                    sync_metadata = _extract_sync_metadata_from_rest(_rest)
                    inputs, targets, snr, bits, sync_metadata = _apply_stage255_snr_curriculum(
                        inputs, targets, snr, bits, sync_metadata, epoch
                    )

                    # Mixed precision forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = _model_forward(model, inputs, bits=bits)
                        total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    else:
                        outputs = _model_forward(model, inputs, bits=bits)
                        total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    total_loss = total_loss + _compute_cross_snr_extra_loss(
                        outputs, inputs, targets, snr, bits, sync_metadata, epoch
                    )
                    total_loss = total_loss + _compute_phase_equiv_extra_loss(
                        outputs, inputs, targets, snr, bits
                    )
                    total_loss = total_loss + _compute_rf_equiv_extra_loss(
                        outputs, inputs, targets, snr, bits
                    )
                    total_loss = total_loss + _compute_latent_mask_auxiliary_loss(
                        outputs, inputs, targets
                    )
                    snr_aux_loss = _compute_stage255_snr_aux_loss(outputs, snr)
                    if snr_aux_loss is not None:
                        total_loss = total_loss + snr_aux_loss
                    sync_snr_loss = _compute_sync_snr_aux_loss(outputs, snr)
                    if sync_snr_loss is not None:
                        total_loss = total_loss + sync_snr_loss
                    sync_physical_loss = _compute_sync_physical_loss(
                        outputs, targets, sync_metadata
                    )
                    if sync_physical_loss is not None:
                        total_loss = total_loss + sync_physical_loss
                    if teacher_active:
                        oracle_demod_loss = _compute_oracle_demod_loss(targets, bits)
                        if oracle_demod_loss is not None:
                            total_loss = total_loss + oracle_demod_loss
                            
                    if l1_sparsity_weight > 0.0:
                        base_mod = model.module if hasattr(model, 'module') else model
                        if hasattr(base_mod, 'decoder') and hasattr(base_mod.decoder, 'aux_loss'):
                            total_loss = total_loss + l1_sparsity_weight * base_mod.decoder.aux_loss
                            
                    total_loss = total_loss / accumulation_steps

                    if not torch.isfinite(total_loss):
                        raise RuntimeError(
                            f"Non-finite training loss detected at epoch={epoch+1}, batch={batch_idx+1}: "
                            f"{float(total_loss.detach().cpu()):.6f}"
                        )

                    # Backward pass
                    if scaler is not None:
                        scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()

                    # Gradient accumulation
                    if (batch_idx + 1) % accumulation_steps == 0:
                        _scale_stage255_router_gradients(epoch)
                        # Gradient clipping
                        if gradient_clip_norm > 0:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

                        # Optimizer step
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        _project_model_dilations()
                        _update_cross_snr_teacher()

                        optimizer.zero_grad()

                    train_loss += total_loss.item() * accumulation_steps

                    with torch.no_grad():
                        train_ber_strict_sum, train_ber_strict_den = _accumulate_batch_ber(
                            outputs,
                            targets,
                            bits,
                            train_ber_strict_sum,
                            train_ber_strict_den,
                            batch_idx=batch_idx,
                            phase="train",
                        )

                    # Update progress bar
                    if batch_idx % log_interval == 0:
                        postfix = {
                            'Loss': f'{total_loss.item() * accumulation_steps:.4f}',
                            'LR': f'{current_lr:.2e}'
                        }
                        if _compute_epoch_ber and train_ber_strict_den > 0:
                            postfix['BER'] = f'{train_ber_strict_sum / train_ber_strict_den:.4f}'
                        pbar.set_postfix(postfix)
        else:
            for batch_idx, (inputs, targets, snr, *_rest) in enumerate(train_loader):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                snr = snr.to(device, non_blocking=True) if snr is not None else None
                bits = _extract_bits_from_rest(_rest)
                sync_metadata = _extract_sync_metadata_from_rest(_rest)
                inputs, targets, snr, bits, sync_metadata = _apply_stage255_snr_curriculum(
                    inputs, targets, snr, bits, sync_metadata, epoch
                )
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = _model_forward(model, inputs, bits=bits)
                    total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                else:
                    outputs = _model_forward(model, inputs, bits=bits)
                    total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                total_loss = total_loss + _compute_cross_snr_extra_loss(
                    outputs, inputs, targets, snr, bits, sync_metadata, epoch
                )
                total_loss = total_loss + _compute_phase_equiv_extra_loss(
                    outputs, inputs, targets, snr, bits
                )
                total_loss = total_loss + _compute_rf_equiv_extra_loss(
                    outputs, inputs, targets, snr, bits
                )
                total_loss = total_loss + _compute_latent_mask_auxiliary_loss(
                    outputs, inputs, targets
                )
                snr_aux_loss = _compute_stage255_snr_aux_loss(outputs, snr)
                if snr_aux_loss is not None:
                    total_loss = total_loss + snr_aux_loss
                sync_snr_loss = _compute_sync_snr_aux_loss(outputs, snr)
                if sync_snr_loss is not None:
                    total_loss = total_loss + sync_snr_loss
                sync_physical_loss = _compute_sync_physical_loss(
                    outputs, targets, sync_metadata
                )
                if sync_physical_loss is not None:
                    total_loss = total_loss + sync_physical_loss
                if teacher_active:
                    oracle_demod_loss = _compute_oracle_demod_loss(targets, bits)
                    if oracle_demod_loss is not None:
                        total_loss = total_loss + oracle_demod_loss
                        
                if l1_sparsity_weight > 0.0:
                    base_mod = model.module if hasattr(model, 'module') else model
                    if hasattr(base_mod, 'decoder') and hasattr(base_mod.decoder, 'aux_loss'):
                        total_loss = total_loss + l1_sparsity_weight * base_mod.decoder.aux_loss
                        
                total_loss = total_loss / accumulation_steps

                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        f"Non-finite training loss detected at epoch={epoch+1}, batch={batch_idx+1}: "
                        f"{float(total_loss.detach().cpu()):.6f}"
                    )
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    _scale_stage255_router_gradients(epoch)
                    # Gradient clipping
                    if gradient_clip_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    
                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    _project_model_dilations()
                    _update_cross_snr_teacher()

                    optimizer.zero_grad()
                
                train_loss += total_loss.item() * accumulation_steps

                with torch.no_grad():
                    train_ber_strict_sum, train_ber_strict_den = _accumulate_batch_ber(
                        outputs,
                        targets,
                        bits,
                        train_ber_strict_sum,
                        train_ber_strict_den,
                        batch_idx=batch_idx,
                        phase="train",
                    )
                
                # No progress bar in non-tqdm mode

        # Handle remaining gradients when number of batches is not divisible by accumulation_steps.
        if len(train_loader) % accumulation_steps != 0:
            _scale_stage255_router_gradients(epoch)
            if gradient_clip_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _project_model_dilations()
            _update_cross_snr_teacher()
            optimizer.zero_grad(set_to_none=True)
        
        # Every rank sees a different training shard. Reduce logging metrics so
        # checkpoint selection and logs describe the global epoch.
        (
            train_loss,
            train_batch_count,
            train_ber_strict_sum,
            train_ber_strict_den,
        ) = distributed_sum(
            (
                train_loss,
                len(train_loader),
                train_ber_strict_sum,
                train_ber_strict_den,
            ),
            device,
        )
        avg_train_loss = train_loss / train_batch_count
        train_losses.append(avg_train_loss)
        learning_rates.append(current_lr)
        avg_train_ber_strict = (
            train_ber_strict_sum / train_ber_strict_den if train_ber_strict_den > 0 else float("nan")
        )
        train_ber_strict_vals.append(avg_train_ber_strict)
        train_ber_oracle_vals.append(float("nan"))
        
        # ========== Validation Phase ==========
        model.eval()
        validation_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        val_loss = 0.0
        val_loss_den = 0
        val_si_snr_paper_sum = 0.0
        val_si_snr_paper_den = 0
        val_si_snr_repo_sum = 0.0
        val_si_snr_repo_den = 0
        val_pit_si_snr_paper_sum = 0.0
        val_pit_si_snr_paper_den = 0
        val_pit_si_snr_repo_sum = 0.0
        val_pit_si_snr_repo_den = 0
        val_amr_correct = 0
        val_amr_total = 0
        val_demod_correct_bits = 0
        val_demod_total_bits = 0
        val_demod_correct_symbols = 0
        val_demod_total_symbols = 0
        val_ber_strict_sum = 0.0
        val_ber_strict_den = 0
        # Get mod_labels from criterion if it's an AMR criterion
        _amr_mod_labels = getattr(criterion, 'mod_labels', None)
        
        with torch.no_grad():
            if use_tqdm:
                with tqdm(
                    val_loader,
                    desc=f'Validating Epoch {epoch+1}',
                    unit='batch',
                    colour='blue',
                    **tqdm_kwargs,
                ) as pbar:

                    for batch_idx, (inputs, targets, snr, *_rest) in enumerate(pbar):
                        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                        snr = snr.to(device, non_blocking=True) if snr is not None else None
                        bits = _extract_bits_from_rest(_rest)
                        
                        # Use mixed precision during validation too
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                outputs = _model_forward(validation_model, inputs, bits=bits)
                            batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                        else:
                            outputs = _model_forward(validation_model, inputs, bits=bits)
                            batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                        
                        validation_batch_size = int(inputs.shape[0])
                        val_loss += batch_loss.item() * validation_batch_size
                        val_loss_den += validation_batch_size

                        # --- Paper SI-SNR (SI-SDR) metric on validation set ---
                        outputs_eval, targets_eval, num_sources, best_perm_per_sample = _get_best_perm_and_sep(outputs, targets)
                        outputs_pit_eval = _get_supplemental_pit_outputs(
                            outputs_eval, targets_eval, num_sources
                        )

                        for k in range(num_sources):
                            pred_k = outputs_eval[:, 2 * k: 2 * k + 2, :]
                            tgt_k = targets_eval[:, 2 * k: 2 * k + 2, :]
                            val_si_snr_paper_sum += si_snr_paper(pred_k, tgt_k).item() * validation_batch_size
                            val_si_snr_paper_den += validation_batch_size
                            val_si_snr_repo_sum += si_snr_repo(pred_k, tgt_k).item() * validation_batch_size
                            val_si_snr_repo_den += validation_batch_size
                            if outputs_pit_eval is not None:
                                pred_pit_k = outputs_pit_eval[:, 2 * k: 2 * k + 2, :]
                                val_pit_si_snr_paper_sum += si_snr_paper(pred_pit_k, tgt_k).item() * validation_batch_size
                                val_pit_si_snr_paper_den += validation_batch_size
                                val_pit_si_snr_repo_sum += si_snr_repo(pred_pit_k, tgt_k).item() * validation_batch_size
                                val_pit_si_snr_repo_den += validation_batch_size

                        # --- AMR classification accuracy ---
                        if _is_amr and isinstance(outputs, tuple) and _amr_mod_labels is not None:
                            _, cls_logits_val = outputs
                            for b, perm in enumerate(best_perm_per_sample):
                                for target_idx, pred_idx in enumerate(perm):
                                    pred_cls = cls_logits_val[pred_idx][b].argmax().item()
                                    if pred_cls == int(_amr_mod_labels[target_idx]):
                                        val_amr_correct += 1
                                    val_amr_total += 1

                        # --- Soft Demod bit accuracy ---
                        if _is_demod and isinstance(outputs, tuple) and bits is not None:
                            bit_logits_val, symbol_logits_val = _extract_demod_outputs(outputs)
                            if bit_logits_val is None:
                                continue
                            for b, perm in enumerate(best_perm_per_sample):
                                for target_idx, pred_idx in enumerate(perm):
                                    pred_bits = (bit_logits_val[pred_idx][b] > 0).long()
                                    gt_bits = bits[target_idx][b].long()
                                    val_demod_correct_bits += (pred_bits == gt_bits).sum().item()
                                    val_demod_total_bits += gt_bits.numel()
                                    if symbol_logits_val is not None and symbol_logits_val[pred_idx] is not None:
                                        gt_symbols = _bits_to_symbol_labels(gt_bits)
                                        if gt_symbols is not None:
                                            pred_symbols = symbol_logits_val[pred_idx][b].argmax(dim=-1)
                                            max_symbols = min(pred_symbols.shape[0], gt_symbols.shape[0])
                                            if max_symbols > 0:
                                                val_demod_correct_symbols += (
                                                    pred_symbols[:max_symbols] == gt_symbols[:max_symbols]
                                                ).sum().item()
                                                val_demod_total_symbols += max_symbols

                        val_ber_strict_sum, val_ber_strict_den = _accumulate_batch_ber(
                            outputs,
                            targets,
                            bits,
                            val_ber_strict_sum,
                            val_ber_strict_den,
                            batch_idx=batch_idx,
                            phase="val",
                        )

                        postfix = {'Val Loss': f'{batch_loss.item():.4f}'}
                        if _compute_epoch_ber and val_ber_strict_den > 0:
                            postfix['Val BER'] = f'{val_ber_strict_sum / val_ber_strict_den:.4f}'
                        pbar.set_postfix(postfix)
            else:
                for batch_idx, (inputs, targets, snr, *_rest) in enumerate(val_loader):
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    snr = snr.to(device, non_blocking=True) if snr is not None else None
                    bits = _extract_bits_from_rest(_rest)
                    
                    # Use mixed precision during validation too
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = _model_forward(validation_model, inputs, bits=bits)
                        batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    else:
                        outputs = _model_forward(validation_model, inputs, bits=bits)
                        batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    
                    validation_batch_size = int(inputs.shape[0])
                    val_loss += batch_loss.item() * validation_batch_size
                    val_loss_den += validation_batch_size

                    # --- Paper SI-SNR (SI-SDR) metric on validation set ---
                    outputs_eval, targets_eval, num_sources, best_perm_per_sample = _get_best_perm_and_sep(outputs, targets)
                    outputs_pit_eval = _get_supplemental_pit_outputs(
                        outputs_eval, targets_eval, num_sources
                    )

                    for k in range(num_sources):
                        pred_k = outputs_eval[:, 2 * k: 2 * k + 2, :]
                        tgt_k = targets_eval[:, 2 * k: 2 * k + 2, :]
                        val_si_snr_paper_sum += si_snr_paper(pred_k, tgt_k).item() * validation_batch_size
                        val_si_snr_paper_den += validation_batch_size
                        val_si_snr_repo_sum += si_snr_repo(pred_k, tgt_k).item() * validation_batch_size
                        val_si_snr_repo_den += validation_batch_size
                        if outputs_pit_eval is not None:
                            pred_pit_k = outputs_pit_eval[:, 2 * k: 2 * k + 2, :]
                            val_pit_si_snr_paper_sum += si_snr_paper(pred_pit_k, tgt_k).item() * validation_batch_size
                            val_pit_si_snr_paper_den += validation_batch_size
                            val_pit_si_snr_repo_sum += si_snr_repo(pred_pit_k, tgt_k).item() * validation_batch_size
                            val_pit_si_snr_repo_den += validation_batch_size

                    # --- AMR classification accuracy ---
                    if _is_amr and isinstance(outputs, tuple) and _amr_mod_labels is not None:
                        _, cls_logits_val = outputs
                        for b, perm in enumerate(best_perm_per_sample):
                            for target_idx, pred_idx in enumerate(perm):
                                pred_cls = cls_logits_val[pred_idx][b].argmax().item()
                                if pred_cls == int(_amr_mod_labels[target_idx]):
                                    val_amr_correct += 1
                                val_amr_total += 1

                    # --- Soft Demod bit accuracy ---
                    if _is_demod and isinstance(outputs, tuple) and bits is not None:
                        bit_logits_val, symbol_logits_val = _extract_demod_outputs(outputs)
                        if bit_logits_val is None:
                            continue
                        for b, perm in enumerate(best_perm_per_sample):
                            for target_idx, pred_idx in enumerate(perm):
                                pred_bits = (bit_logits_val[pred_idx][b] > 0).long()
                                gt_bits = bits[target_idx][b].long()
                                val_demod_correct_bits += (pred_bits == gt_bits).sum().item()
                                val_demod_total_bits += gt_bits.numel()
                                if symbol_logits_val is not None and symbol_logits_val[pred_idx] is not None:
                                    gt_symbols = _bits_to_symbol_labels(gt_bits)
                                    if gt_symbols is not None:
                                        pred_symbols = symbol_logits_val[pred_idx][b].argmax(dim=-1)
                                        max_symbols = min(pred_symbols.shape[0], gt_symbols.shape[0])
                                        if max_symbols > 0:
                                            val_demod_correct_symbols += (
                                                pred_symbols[:max_symbols] == gt_symbols[:max_symbols]
                                            ).sum().item()
                                            val_demod_total_symbols += max_symbols

                    val_ber_strict_sum, val_ber_strict_den = _accumulate_batch_ber(
                        outputs,
                        targets,
                        bits,
                        val_ber_strict_sum,
                        val_ber_strict_den,
                        batch_idx=batch_idx,
                        phase="val",
                    )
        
        (
            val_loss,
            val_loss_den,
            val_si_snr_paper_sum,
            val_si_snr_paper_den,
            val_si_snr_repo_sum,
            val_si_snr_repo_den,
            val_pit_si_snr_paper_sum,
            val_pit_si_snr_paper_den,
            val_pit_si_snr_repo_sum,
            val_pit_si_snr_repo_den,
            val_amr_correct,
            val_amr_total,
            val_demod_correct_bits,
            val_demod_total_bits,
            val_demod_correct_symbols,
            val_demod_total_symbols,
            val_ber_strict_sum,
            val_ber_strict_den,
        ) = distributed_sum(
            (
                val_loss,
                val_loss_den,
                val_si_snr_paper_sum,
                val_si_snr_paper_den,
                val_si_snr_repo_sum,
                val_si_snr_repo_den,
                val_pit_si_snr_paper_sum,
                val_pit_si_snr_paper_den,
                val_pit_si_snr_repo_sum,
                val_pit_si_snr_repo_den,
                val_amr_correct,
                val_amr_total,
                val_demod_correct_bits,
                val_demod_total_bits,
                val_demod_correct_symbols,
                val_demod_total_symbols,
                val_ber_strict_sum,
                val_ber_strict_den,
            ),
            device,
        )

        # Validation shards can end with different batch sizes, so aggregate by
        # sample count rather than giving the final partial batch extra weight.
        avg_val_loss = val_loss / val_loss_den
        val_losses.append(avg_val_loss)

        # Step schedulers only after the epoch's optimizer updates.  Calling
        # LinearLR.step() before the first optimizer.step() skips the first
        # warmup value and triggers PyTorch's scheduler-order warning.
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        avg_val_si_snr_paper = (val_si_snr_paper_sum / val_si_snr_paper_den) if val_si_snr_paper_den > 0 else float("nan")
        avg_val_si_snr_repo = (val_si_snr_repo_sum / val_si_snr_repo_den) if val_si_snr_repo_den > 0 else float("nan")
        avg_val_pit_si_snr_paper = (
            val_pit_si_snr_paper_sum / val_pit_si_snr_paper_den
            if val_pit_si_snr_paper_den > 0 else float("nan")
        )
        avg_val_pit_si_snr_repo = (
            val_pit_si_snr_repo_sum / val_pit_si_snr_repo_den
            if val_pit_si_snr_repo_den > 0 else float("nan")
        )
        avg_val_ber_strict = val_ber_strict_sum / val_ber_strict_den if val_ber_strict_den > 0 else float("nan")
        val_si_snr_paper_vals.append(avg_val_si_snr_paper)
        val_si_snr_repo_vals.append(avg_val_si_snr_repo)
        val_ber_strict_vals.append(avg_val_ber_strict)
        val_ber_oracle_vals.append(float("nan"))
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        epoch_model_diagnostics = _collect_model_diagnostics()
        model_diagnostics_history.append(epoch_model_diagnostics)
        
        # ========== Model saving and early stopping logic ==========
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            logger.info(f'--> Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f} '
                       f'(improvement: {improvement:.4f})')
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve_counter = 0
            best_model_weights = _clone_state_dict_to_cpu(model.state_dict())
            
            # Save best model weights
            if save_artifacts:
                best_model_path = os.path.join(weights_dir, 'best_model_weights.pth')
                torch.save(best_model_weights, best_model_path)
                logger.info(f'Best model weights saved/updated at {best_model_path}')

                best_training_checkpoint_path = os.path.join(weights_dir, 'best_training_checkpoint.pth')
                torch.save(_build_training_checkpoint(epoch), best_training_checkpoint_path)
                logger.info(f'Best training checkpoint saved/updated at {best_training_checkpoint_path}')

                # Also save a stable path for "test" mode (project/checkpoint/best_model_weights.pth)
                checkpoint_best_path = checkpoint_dir / "best_model_weights.pth"
                torch.save(best_model_weights, str(checkpoint_best_path))
                logger.info(f'Checkpoint best weights saved/updated at {checkpoint_best_path}')
        else:
            no_improve_counter += 1
            if early_stop_enabled:
                logger.info(f'No improvement for {no_improve_counter}/{early_stop_patience} epochs')
            else:
                logger.info(f'No improvement for {no_improve_counter} epochs (early stopping disabled)')

        # Save full resume checkpoint after metrics and best-state updates.
        if save_artifacts:
            latest_checkpoint_path = os.path.join(weights_dir, 'latest_training_checkpoint.pth')
            full_checkpoint = _build_training_checkpoint(epoch)
            torch.save(full_checkpoint, latest_checkpoint_path)
            logger.info(f'Latest training checkpoint saved/updated at {latest_checkpoint_path}')
            if save_checkpoint_every > 0 and (epoch + 1) % save_checkpoint_every == 0:
                checkpoint_path = os.path.join(weights_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(full_checkpoint, checkpoint_path)
                logger.info(f'Checkpoint saved: {checkpoint_path}')
            elif save_checkpoint_every <= 0:
                _cleanup_epoch_checkpoints()
        
        # Detailed logging
        # Build epoch log message
        val_metric_label = 'Val Fixed' if dual_report_fixed_and_pit else 'Val'
        epoch_msg = (
            f'Epoch [{epoch+1:3d}/{num_epochs:3d}] - '
            f'Train Loss: {avg_train_loss:8.4f}, '
            f'Val Loss: {avg_val_loss:8.4f}, '
            f'{val_metric_label} SI-SNR_paper: {avg_val_si_snr_paper:7.3f} dB, '
            f'{val_metric_label} SI-SNR_repo: {avg_val_si_snr_repo:7.3f} dB, '
            f'LR: {current_lr:.2e}, '
            f'Time: {epoch_time:5.1f}s'
        )
        if _compute_epoch_ber:
            epoch_msg += (
                f', Train BER_strict(sampled): {avg_train_ber_strict:.6f}, '
                f'Val BER_strict(sampled): {avg_val_ber_strict:.6f}'
            )
        if val_amr_total > 0:
            val_amr_acc = 100.0 * val_amr_correct / val_amr_total
            epoch_msg += f', AMR Acc: {val_amr_acc:.1f}%'
        if val_demod_total_bits > 0:
            val_demod_acc = 100.0 * val_demod_correct_bits / val_demod_total_bits
            epoch_msg += f', Demod Bit Acc: {val_demod_acc:.2f}%'
        if val_demod_total_symbols > 0:
            val_demod_symbol_acc = 100.0 * val_demod_correct_symbols / val_demod_total_symbols
            epoch_msg += f', Demod Sym Acc: {val_demod_symbol_acc:.2f}%'
        logger.info(epoch_msg)
        if epoch_model_diagnostics:
            formatted_diagnostics = []
            for name, value in sorted(epoch_model_diagnostics.items()):
                if isinstance(value, float):
                    formatted_diagnostics.append(f"{name}={value:.5g}")
                else:
                    formatted_diagnostics.append(f"{name}={value}")
            logger.info("  Model diagnostics: " + ", ".join(formatted_diagnostics))
        if dual_report_fixed_and_pit:
            logger.info(
                f'  Val PIT(si_snr_complex) SI-SNR_paper: '
                f'{avg_val_pit_si_snr_paper:7.3f} dB, '
                f'SI-SNR_repo: {avg_val_pit_si_snr_repo:7.3f} dB'
            )
        if active_loss_name is not None:
            logger.info(f'Active Loss: {active_loss_name}')
        
        # Check early stopping condition
        if early_stop_enabled and no_improve_counter >= early_stop_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs!')
            break
        
        # Memory cleanup
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # ========== Post-training processing ==========
    if save_artifacts:
        # Plot loss curves (including learning rate)
        plot_enhanced_losses(train_losses, val_losses, learning_rates, epoch_times, 
                            results_folder, signal_names=signal_names)
        # Plot loss curves
        plot_losses(train_losses, val_losses, results_folder, signal_names=signal_names)
    
    # Restore best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        logger.info("Loaded best model weights for final evaluation")
    
    # Final evaluation
    logger.info("Starting final model evaluation...")
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    snr_metrics = None
    if not distributed or torch.distributed.get_rank() == 0:
        evaluation_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        snr_metrics = test_model(
            evaluation_model, snr_loaders, criterion, device, logger, results_folder,
            num_plots=1, num_points=256, input_size=input_size,
            data_choice=data_choice, signal_names=signal_names,
            save_artifacts=save_artifacts,
            report_ber=report_ber,
            ber_offset_search=ber_offset_search,
            ber_mode=ber_mode,
            ber_num_files=ber_num_files,
            ber_compute_oracle=ber_compute_oracle,
            amr_mode=amr_mode,
            demod_mode=demod_mode_phase2 if demod_mode_phase2 is not None else demod_mode,
            eval_pit_metric=eval_pit_metric,
            report_phase_flip=report_phase_flip,
            phase_flip_tolerance_deg=phase_flip_tolerance_deg,
            phase_flip_min_sc=phase_flip_min_sc,
            phase_flip_mode=phase_flip_mode,
        )
    if distributed:
        shared_metrics = [snr_metrics]
        torch.distributed.broadcast_object_list(shared_metrics, src=0)
        snr_metrics = shared_metrics[0]
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_si_snr_paper': val_si_snr_paper_vals,
        'val_si_snr_repo': val_si_snr_repo_vals,
        'train_ber_strict': train_ber_strict_vals,
        'train_ber_oracle': train_ber_oracle_vals,
        'val_ber_strict': val_ber_strict_vals,
        'val_ber_oracle': val_ber_oracle_vals,
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'model_diagnostics': model_diagnostics_history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'no_improve_counter': no_improve_counter,
        'resumed_from_checkpoint': str(resume_checkpoint) if resume_checkpoint else None,
        'initialized_from_checkpoint': str(init_checkpoint) if init_checkpoint else None,
        'component_checkpoints': [str(path) for path in (component_checkpoints or ())],
        'start_epoch': start_epoch,
        'total_epochs': len(train_losses),
    }
    
    if save_artifacts:
        history_path = os.path.join(weights_dir, 'training_history.pth')
        torch.save(training_history, history_path)
        logger.info(f'Training history saved to {history_path}')
    
    # Training summary
    total_time = sum(epoch_times)
    avg_epoch_time = np.mean(epoch_times)
    logger.info(f'\n=== Training Summary ===')
    logger.info(f'Total training time: {total_time:.1f}s ({total_time/60:.1f}m)')
    logger.info(f'Average epoch time: {avg_epoch_time:.1f}s')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')
    logger.info(f'Final learning rate: {optimizer.param_groups[0]["lr"]:.2e}')
    training_history['snr_metrics'] = snr_metrics

    return training_history


def plot_enhanced_losses(train_losses, val_losses, learning_rates, epoch_times, 
                        results_folder, signal_names=None):
    """Plot enhanced loss curves, including learning rate and time information"""
    import matplotlib.pyplot as plt
    project_root = Path(__file__).resolve().parents[1]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate curve
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Time per epoch
    ax3.plot(epochs, epoch_times, 'orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time per Epoch')
    ax3.grid(True, alpha=0.3)
    
    # Cumulative time
    cumulative_time = np.cumsum(epoch_times) / 60  # Convert to minutes
    ax4.plot(epochs, cumulative_time, 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Cumulative Time (minutes)')
    ax4.set_title('Cumulative Training Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    save_path = project_root / "results" / results_folder / "enhanced_training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
