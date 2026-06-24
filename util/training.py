import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from util.evaluation import (
    test_model,
    reorder_outputs_for_eval,
    extract_separation_output,
    _infer_modulations_from_data_choice,
)
from util.metrics import (
    si_snr_paper,
    si_snr_repo,
    strict_ber_iq_from_bits,
)
from util.visualize import plot_losses
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
    eval_pit_metric='si_snr_complex',
    report_phase_flip=False,     # Report phase-flip rate on final SNR evaluation
    phase_flip_tolerance_deg=45.0,
    phase_flip_min_sc=0.0,
    phase_flip_mode='either',
    resume_checkpoint=None,      # Full training checkpoint to resume from
    resume_allow_partial=False,  # Warm-start matching model weights across architecture variants
):
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

    def _load_partial_model_state(state_dict, path):
        """Warm-start matching tensors and ignore incompatible heads."""
        model_state = model.state_dict()
        compatible = {}
        skipped_unexpected = []
        skipped_shape = []
        for key, value in state_dict.items():
            if key not in model_state:
                skipped_unexpected.append(key)
                continue
            if not torch.is_tensor(value) or model_state[key].shape != value.shape:
                skipped_shape.append(key)
                continue
            compatible[key] = value

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
        if skipped_unexpected[:8]:
            logger.warning(f"Skipped unexpected checkpoint keys, first few: {skipped_unexpected[:8]}")
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
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    
    # Learning rate warmup scheduler
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Validation/test PIT alignment metric: {eval_pit_metric}")
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
    optimizer.zero_grad(set_to_none=True)

    def _set_dataset_epoch(dataset, epoch: int):
        if dataset is None:
            return
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        nested = getattr(dataset, "dataset", None)
        if nested is not None and nested is not dataset:
            _set_dataset_epoch(nested, epoch)

    def _criterion_uses_snr() -> bool:
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
            return model(inputs, mode=_model_mode, **kwargs)
        return model(inputs, **kwargs)

    def _get_best_perm_and_sep(outputs, targets):
        sep_outputs = extract_separation_output(outputs)
        if isinstance(sep_outputs, (list, tuple)):
            sep_outputs = sep_outputs[-1]
        sep_outputs = sep_outputs.float()
        targets_eval = targets.float()
        num_sources = targets_eval.shape[1] // 2
        outputs_eval, best_perm_per_sample = reorder_outputs_for_eval(
            sep_outputs,
            targets_eval,
            num_sources,
            pit_metric=eval_pit_metric,
        )
        return outputs_eval, targets_eval, num_sources, best_perm_per_sample

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

    def _compute_loss_fp32(outputs, targets, snr=None, inputs=None, bits=None):
        # Keep numerically sensitive losses (e.g., SI-SNR/log/ratio) in FP32.
        # Multi-task models return (sep_output, aux_logits_list) — handle the tuple.
        out32 = _to_fp32(outputs)
        tgt32 = targets.float()
        if _needs_bits and bits is not None:
            return criterion(out32, tgt32, bits=bits)
        if _needs_mixture and inputs is not None:
            inp32 = inputs.float()
            return criterion(out32, tgt32, inp32)
        if _criterion_uses_snr():
            return criterion(out32, tgt32, snr)
        return criterion(out32, tgt32)

    def _get_ber_modulations(num_sources):
        if num_sources not in _ber_modulations_cache:
            _ber_modulations_cache[num_sources] = _infer_modulations_from_data_choice(data_choice, num_sources)
        return _ber_modulations_cache[num_sources]

    def _extract_bits_from_rest(rest):
        if _should_load_bits and rest:
            return tuple(b.to(device, non_blocking=True) for b in rest[0])
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
            'checkpoint_version': 2,
            'epoch': int(epoch),
            'next_epoch': int(epoch) + 1,
            'model_state_dict': model.state_dict(),
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
            },
        }

    def _load_training_checkpoint(path):
        nonlocal start_epoch, best_val_loss, best_epoch, no_improve_counter, best_model_weights
        nonlocal train_losses, val_losses, val_si_snr_paper_vals, val_si_snr_repo_vals
        nonlocal learning_rates, epoch_times
        nonlocal train_ber_strict_vals, train_ber_oracle_vals, val_ber_strict_vals, val_ber_oracle_vals
        nonlocal _last_active_phase, _phase2_epoch_counter, _last_logged_demod_mode, _last_teacher_active

        checkpoint = _load_checkpoint_file(path)
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f"Checkpoint {path} does not contain model_state_dict")
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

    if resume_checkpoint:
        _load_training_checkpoint(Path(resume_checkpoint))

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        _set_dataset_epoch(train_loader.dataset, epoch)

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
        
        # Learning rate warmup
        if epoch < warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
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

                    # Mixed precision forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = _model_forward(model, inputs, bits=bits)
                        total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    else:
                        outputs = _model_forward(model, inputs, bits=bits)
                        total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    if teacher_active:
                        oracle_demod_loss = _compute_oracle_demod_loss(targets, bits)
                        if oracle_demod_loss is not None:
                            total_loss = total_loss + oracle_demod_loss
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
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = _model_forward(model, inputs, bits=bits)
                    total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                else:
                    outputs = _model_forward(model, inputs, bits=bits)
                    total_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                if teacher_active:
                    oracle_demod_loss = _compute_oracle_demod_loss(targets, bits)
                    if oracle_demod_loss is not None:
                        total_loss = total_loss + oracle_demod_loss
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
            if gradient_clip_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        learning_rates.append(current_lr)
        avg_train_ber_strict = (
            train_ber_strict_sum / train_ber_strict_den if train_ber_strict_den > 0 else float("nan")
        )
        train_ber_strict_vals.append(avg_train_ber_strict)
        train_ber_oracle_vals.append(float("nan"))
        
        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0.0
        val_si_snr_paper_sum = 0.0
        val_si_snr_paper_den = 0
        val_si_snr_repo_sum = 0.0
        val_si_snr_repo_den = 0
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
                                outputs = _model_forward(model, inputs, bits=bits)
                            batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                        else:
                            outputs = _model_forward(model, inputs, bits=bits)
                            batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                        
                        val_loss += batch_loss.item()

                        # --- Paper SI-SNR (SI-SDR) metric on validation set ---
                        outputs_eval, targets_eval, num_sources, best_perm_per_sample = _get_best_perm_and_sep(outputs, targets)

                        for k in range(num_sources):
                            pred_k = outputs_eval[:, 2 * k: 2 * k + 2, :]
                            tgt_k = targets_eval[:, 2 * k: 2 * k + 2, :]
                            val_si_snr_paper_sum += si_snr_paper(pred_k, tgt_k).item()
                            val_si_snr_paper_den += 1
                            val_si_snr_repo_sum += si_snr_repo(pred_k, tgt_k).item()
                            val_si_snr_repo_den += 1

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
                            outputs = _model_forward(model, inputs, bits=bits)
                        batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    else:
                        outputs = _model_forward(model, inputs, bits=bits)
                        batch_loss = _compute_loss_fp32(outputs, targets, snr, inputs=inputs, bits=bits)
                    
                    val_loss += batch_loss.item()

                    # --- Paper SI-SNR (SI-SDR) metric on validation set ---
                    outputs_eval, targets_eval, num_sources, best_perm_per_sample = _get_best_perm_and_sep(outputs, targets)

                    for k in range(num_sources):
                        pred_k = outputs_eval[:, 2 * k: 2 * k + 2, :]
                        tgt_k = targets_eval[:, 2 * k: 2 * k + 2, :]
                        val_si_snr_paper_sum += si_snr_paper(pred_k, tgt_k).item()
                        val_si_snr_paper_den += 1
                        val_si_snr_repo_sum += si_snr_repo(pred_k, tgt_k).item()
                        val_si_snr_repo_den += 1

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
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling (after warmup)
        if epoch >= warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        avg_val_si_snr_paper = (val_si_snr_paper_sum / val_si_snr_paper_den) if val_si_snr_paper_den > 0 else float("nan")
        avg_val_si_snr_repo = (val_si_snr_repo_sum / val_si_snr_repo_den) if val_si_snr_repo_den > 0 else float("nan")
        avg_val_ber_strict = val_ber_strict_sum / val_ber_strict_den if val_ber_strict_den > 0 else float("nan")
        val_si_snr_paper_vals.append(avg_val_si_snr_paper)
        val_si_snr_repo_vals.append(avg_val_si_snr_repo)
        val_ber_strict_vals.append(avg_val_ber_strict)
        val_ber_oracle_vals.append(float("nan"))
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
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
        epoch_msg = (
            f'Epoch [{epoch+1:3d}/{num_epochs:3d}] - '
            f'Train Loss: {avg_train_loss:8.4f}, '
            f'Val Loss: {avg_val_loss:8.4f}, '
            f'Val SI-SNR_paper: {avg_val_si_snr_paper:7.3f} dB, '
            f'Val SI-SNR_repo: {avg_val_si_snr_repo:7.3f} dB, '
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
    snr_metrics = test_model(
            model, snr_loaders, criterion, device, logger, results_folder,
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
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'no_improve_counter': no_improve_counter,
        'resumed_from_checkpoint': str(resume_checkpoint) if resume_checkpoint else None,
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
