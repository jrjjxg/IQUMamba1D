import numpy as np
import torch
import torch.nn as nn
import os
import sys
import random
import json
from pathlib import Path
import shutil
from datetime import datetime
import argparse
import inspect
import csv
import copy
from torch.utils.data import DataLoader, Subset

# Project root (repo-relative, cross-platform)
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
CONFIG_ROOT = PROJECT_ROOT / "config"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoint"
DATA_ROOT = PROJECT_ROOT / "data"

# Ensure local imports work when running from outside the project folder.
sys.path.append(str(PROJECT_ROOT))

from data_loader.dataloader import create_data_loaders, create_multidataset_data_loaders
from util.logger import create_logger
from util.evaluation import test_model
from util.training import train_model
from util.utils import Create_Mamba_model, create_new_results_folder
from util.config import MambaConfig
from util.blind_cross_snr import apply_blind_cross_snr_profile
from util.stage_registry import supported_stage_ids
from util.runtime import (
    collect_accelerator_diagnostics,
    log_accelerator_diagnostics,
    resolve_training_device,
    should_pin_memory,
)
from util.loss import (
    si_snr_loss,
    pit_si_snr_loss,
    l1_loss,
    si_snr_mse_loss,
    si_snr_huber_loss,
    pit_mse_loss,
    pit_l1_loss,
    pit_huber_loss,
    pit_si_snr_mse_loss,
    pit_demod_aware_loss,
    pit_si_snr_huber_loss,
    pit_si_snr_huber_identity_anchor_loss,
    pit_si_snr_huber_phase_loss,
    pit_si_snr_huber_mc_loss,
    pit_si_snr_uric_deep_supervision_loss,
    pit_si_snr_huber_uric_deep_supervision_loss,
    pit_si_snr_huber_rms_uric_deep_supervision_loss,
    pit_si_snr_huber_rms_loss,
    pit_si_snr_huber_gpahuber_loss,
    pit_si_snr_huber_xtalk_loss,
    pit_si_snr_huber_xtalk_noiseres_loss,
    pit_si_snr_complex_huber_rms_loss,
    pit_si_snr_huber_const_loss,
    pit_si_snr_huber_rms_const_loss,
    bw_mrstft_loss,
    evm_loss,
    evm_constellation_loss,
    pit_si_snr_rms_loss,
    pit_si_snr_constellation_loss,
    pit_si_snr_rms_constellation_loss,
    multi_resolution_stft_loss,
    mixture_consistency_loss,
    pit_si_snr_huber_mrstft_loss,
    pit_si_snr_huber_mrstft_mixcons_loss,
    pit_si_snr_huber_amr_loss,
    get_mod_labels_from_data_choice,
    pit_si_snr_huber_demod_loss,
    pit_si_snr_huber_cma_loss,
    pit_si_snr_huber_mf_loss,
    pit_si_snr_huber_ind_loss,
    pit_si_snr_huber_cyclic_profile_loss,
)

LOSS_FUNCTION_CHOICES = [
    'MSE', 'L1', 'Huber',
    'SI-SNR', 'PIT-SI-SNR',
    'SI-SNR+MSE', 'SI-SNR+Huber',
    'PIT-MSE', 'PIT-L1', 'PIT-Huber',
    'PIT-SI-SNR+MSE', 'PIT-SI-SNR+Huber',
    'PIT-SI-SNR+Huber+IdentityAnchor', 'PIT-SI-SNR+Huber+Phase',
    'PIT-DEMOD-AWARE',
    'PIT-SI-SNR+URIC-DS', 'PIT-SI-SNR+Huber+URIC-DS',
    'BW-MRSTFT', 'EVM', 'EVM+CONST',
    'PIT-SI-SNR+BW-MRSTFT',
    'PIT-SI-SNR+BW-MRSTFT+EVMCONST',
    # Physics-informed losses
    'PIT-SI-SNR+RMS',
    'PIT-SI-SNR+CONST',
    'PIT-SI-SNR+RMS+CONST',
    'PIT-SI-SNR+Huber+RMS',
    'PIT-SI-SNR+Huber+GPAHuber',
    'PIT-SI-SNR+Huber+XTALK',
    'PIT-SI-SNR+Huber+XTALK+NOISERES',
    'PIT-SI-SNR+ComplexHuber+RMS',
    'PIT-SI-SNR+Huber+RMS+URIC-DS',
    'PIT-SI-SNR+Huber+CONST',
    'PIT-SI-SNR+Huber+RMS+CONST',
    # Paper-aligned complex-domain loss (CTDCRN: MSE_ave - SI-SNR_ave)
    'CTDCRN',
    'PIT-CTDCRN',
    # Multi-Resolution STFT + Mixture Consistency (SOTA, works with any model)
    'MR-STFT',
    'PIT-SI-SNR+Huber+MRSTFT',
    'PIT-SI-SNR+Huber+MRSTFT+MIXCONS',
    # AMR (Automatic Modulation Recognition) joint loss
    'PIT-SI-SNR+Huber+AMR',
    # Soft demodulation joint loss
    'PIT-SI-SNR+Huber+DEMOD',
    # Advanced Communications Losses
    'PIT-SI-SNR+Huber+SoftDemod', 'PIT-SI-SNR+Huber+SoftDemodV2', 'PIT-SI-SNR+Huber+SoftDemodV3',
    'PIT-SI-SNR+Huber+AMR',    'PIT-SI-SNR+Huber+CMA', 'PIT-SI-SNR+Huber+MF',
    'PIT-SI-SNR+Huber+Independence',
    'PIT-SI-SNR+Huber+CyclicProfile',
    'PIT-SI-SNR+Huber+MC',
    'PIT-SI-SNR+Huber+LowSNRAux'
]


def normalize_data_choice(data_choice: str) -> str:
    """Normalize Kaggle-safe aliases back to canonical dataset names."""
    aliases = {
        "QPSK16APSK-A": "QPSK+16APSK-A",
        "QPSK16APSK-B": "QPSK+16APSK-B",
        "QPSK-16APSK-A": "QPSK+16APSK-A",
        "QPSK-16APSK-B": "QPSK+16APSK-B",
        "RML2016": "2016",
        "RML2018": "2018",
    }
    return aliases.get(data_choice, data_choice)


DATA_CHOICE_CHOICES = [
    'debug_random',
    'TorchSig', '2016', '2018', 'RML2016', 'RML2018',
    '8PSK_M', '8PSK_M_NS',
    '8PSK_Burst', '8PSK_Burst_NS',
    '8PSK_M_8192', '8PSK_M_16384', '8PSK_M_32768',
    '8PSK_M_8192_NS', '8PSK_M_16384_NS', '8PSK_M_32768_NS',
    'QPSK_16APSK', 'QPSK_16APSK_NS',
    '8PSK_Rs', '8PSK_Rs_NS',
    '16QAM_64QAM', '16QAM_128QAM',
    '64QAM_64QAM', '64QAM_128QAM',
    '16QAM_64QAM_128QAM',
    '8PSK-A', '8PSK-B', '8PSK-C', '8PSK-D',
    '8PSK-E', '8PSK-F', '8PSK-G',
    '8PSK-H', '8PSK-I', '8PSK-J', '8PSK-K', '8PSK-L',
    'QPSK+16APSK-A', 'QPSK+16APSK-B',
    'QPSK16APSK-A', 'QPSK16APSK-B',
    'QPSK-16APSK-A', 'QPSK-16APSK-B',
    'QAM-A', 'QAM-B', 'QAM-C', 'QAM-D', 'QAM-E',
]

class TwoPhaseCriterion:
    """Switch between two criteria based on epoch index."""

    def __init__(self, phase1_criterion, phase2_criterion, switch_epoch, phase1_name, phase2_name, logger):
        self.phase1_criterion = phase1_criterion
        self.phase2_criterion = phase2_criterion
        self.switch_epoch = max(0, int(switch_epoch))
        self.phase1_name = phase1_name
        self.phase2_name = phase2_name
        self.logger = logger
        self._active_phase = None
        self._active_name = None
        # Training/evaluation loops inspect these flags once up front.
        # Use the union so phase-specific multitask criteria still receive
        # their required side inputs after the phase switch.
        self.needs_bits = bool(
            getattr(phase1_criterion, 'needs_bits', False)
            or getattr(phase2_criterion, 'needs_bits', False)
        )
        self.needs_mixture = bool(
            getattr(phase1_criterion, 'needs_mixture', False)
            or getattr(phase2_criterion, 'needs_mixture', False)
        )
        self.needs_snr = bool(
            getattr(phase1_criterion, 'needs_snr', False)
            or getattr(phase2_criterion, 'needs_snr', False)
        )
        self.mod_labels = getattr(phase2_criterion, 'mod_labels', getattr(phase1_criterion, 'mod_labels', None))

    @staticmethod
    def _call_criterion(criterion, outputs, targets, *args, **kwargs):
        """Forward only the extras the active criterion actually accepts."""
        sig = inspect.signature(criterion)
        params = list(sig.parameters.values())
        accepts_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        positional_capacity = sum(
            1 for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        extra_positional_capacity = max(0, positional_capacity - 2)

        filtered_args = args if accepts_var_positional else args[:extra_positional_capacity]
        if accepts_var_keyword:
            filtered_kwargs = kwargs
        else:
            accepted_kwargs = {
                p.name for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}

        return criterion(outputs, targets, *filtered_args, **filtered_kwargs)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.switch_epoch:
            phase = 1
            name = self.phase1_name
        else:
            phase = 2
            name = self.phase2_name
        if phase != self._active_phase:
            self._active_phase = phase
            self._active_name = name
            if self.logger is not None:
                self.logger.info(f"Loss phase switched to Phase {phase}: {name} (epoch {epoch + 1})")

    def __call__(self, outputs, targets, *args, **kwargs):
        if self._active_phase == 1:
            return self._call_criterion(self.phase1_criterion, outputs, targets, *args, **kwargs)
        return self._call_criterion(self.phase2_criterion, outputs, targets, *args, **kwargs)

    def get_active_name(self):
        return self._active_name

    def get_active_phase(self):
        return self._active_phase


def build_criterion(loss_name, args):
    if loss_name == 'MSE':
        return nn.MSELoss(), "MSE"
    if loss_name == 'L1':
        return l1_loss, "L1"
    if loss_name == 'Huber':
        return nn.HuberLoss(), "HuberLoss"
    if loss_name == 'SI-SNR':
        return si_snr_loss, "SI-SNR"
    if loss_name == 'PIT-SI-SNR':
        return pit_si_snr_loss, "PIT-SI-SNR"
    if loss_name == 'PIT-SI-SNR+Constellation':
        # Added for Stage 157
        from util.loss import pit_si_snr_constellation_loss
        criterion = lambda outputs, targets: pit_si_snr_constellation_loss(
            outputs,
            targets,
            alpha=getattr(args, 'constellation_alpha', 0.1),
            sigma=getattr(args, 'constellation_sigma', 0.1)
        )
        return criterion, "PIT-SI-SNR+Constellation"
    if loss_name == 'SI-SNR+MSE':
        criterion = lambda outputs, targets: si_snr_mse_loss(
            outputs,
            targets,
            alpha=args.si_snr_mse_alpha,
            beta=args.si_snr_mse_beta,
        )
        return criterion, f"SI-SNR+MSE(a={args.si_snr_mse_alpha}, b={args.si_snr_mse_beta})"
    if loss_name == 'SI-SNR+Huber':
        criterion = lambda outputs, targets: si_snr_huber_loss(
            outputs,
            targets,
            alpha=args.si_snr_huber_alpha,
            beta=args.si_snr_huber_beta,
            delta=args.si_snr_huber_delta,
        )
        return criterion, (
            f"SI-SNR+Huber(a={args.si_snr_huber_alpha}, "
            f"b={args.si_snr_huber_beta}, d={args.si_snr_huber_delta})"
        )
    if loss_name == 'PIT-MSE':
        return pit_mse_loss, "PIT-MSE"
    if loss_name == 'PIT-L1':
        return pit_l1_loss, "PIT-L1"
    if loss_name == 'PIT-Huber':
        criterion = lambda outputs, targets: pit_huber_loss(
            outputs,
            targets,
            delta=args.pit_huber_delta,
        )
        return criterion, f"PIT-Huber(d={args.pit_huber_delta})"
    if loss_name == 'PIT-SI-SNR+MSE':
        criterion = lambda outputs, targets: pit_si_snr_mse_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_mse_alpha,
            beta=args.pit_si_snr_mse_beta,
        )
        return criterion, (
            f"PIT-SI-SNR+MSE(a={args.pit_si_snr_mse_alpha}, "
            f"b={args.pit_si_snr_mse_beta})"
        )
    if loss_name == 'PIT-DEMOD-AWARE':
        criterion = lambda outputs, targets: pit_demod_aware_loss(
            outputs,
            targets,
            mse_weight=args.demod_aware_mse_weight,
            sisnr_weight=args.demod_aware_sisnr_weight,
        )
        return criterion, (
            f"PIT-DEMOD-AWARE(mse={args.demod_aware_mse_weight}, "
            f"sisnr={args.demod_aware_sisnr_weight})"
        )
    if loss_name == 'PIT-SI-SNR+Huber':
        criterion = lambda outputs, targets: pit_si_snr_huber_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber(a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, d={args.pit_si_snr_huber_delta})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+IdentityAnchor':
        criterion = lambda outputs, targets: pit_si_snr_huber_identity_anchor_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            delta=args.pit_si_snr_huber_delta,
            identity_anchor_weight=args.identity_anchor_weight,
            identity_anchor_margin=args.identity_anchor_margin,
            identity_anchor_temperature=args.identity_anchor_temperature,
        )
        return criterion, (
            "PIT-SI-SNR+Huber+IdentityAnchor("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"w={args.identity_anchor_weight}, "
            f"m={args.identity_anchor_margin}, "
            f"t={args.identity_anchor_temperature})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+Phase':
        criterion = lambda outputs, targets: pit_si_snr_huber_phase_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            delta=args.pit_si_snr_huber_delta,
            phase_weight=args.phase_increment_weight,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+Phase(a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, d={args.pit_si_snr_huber_delta}, "
            f"phase={args.phase_increment_weight})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+LowSNRAux':
        def _clean_mixture_target(targets, mixture):
            if targets.dim() != 3 or targets.size(1) % 2 != 0:
                raise ValueError(
                    f"LowSNRAux expects targets with shape (B, 2K, L), got {tuple(targets.shape)}"
                )
            num_sources = targets.size(1) // 2
            clean_mixture_target = targets.reshape(targets.size(0), num_sources, 2, targets.size(-1)).sum(dim=1)
            if clean_mixture_target.size(-1) != mixture.size(-1):
                clean_mixture_target = nn.functional.interpolate(
                    clean_mixture_target,
                    size=mixture.size(-1),
                    mode='linear',
                    align_corners=False,
                )
            return clean_mixture_target

        def _low_snr_aux_criterion(outputs, targets, mixture):
            sep_outputs = outputs
            aux = None
            if isinstance(outputs, tuple):
                sep_outputs, aux = outputs
            base = pit_si_snr_huber_loss(
                sep_outputs,
                targets,
                alpha=args.pit_si_snr_huber_alpha,
                beta=args.pit_si_snr_huber_beta,
                delta=args.pit_si_snr_huber_delta,
            )
            if not isinstance(aux, dict):
                return base

            clean_hat = aux.get("clean_mixture_hat")
            noise_hat = aux.get("noise_hat")
            if clean_hat is None or noise_hat is None:
                return base

            clean_target = _clean_mixture_target(targets, mixture)
            noise_target = mixture - clean_target
            clean_loss = nn.functional.smooth_l1_loss(
                clean_hat,
                clean_target,
                beta=getattr(args, 'low_snr_aux_huber_beta', 0.5),
            )
            noise_loss = nn.functional.smooth_l1_loss(
                noise_hat,
                noise_target,
                beta=getattr(args, 'low_snr_aux_huber_beta', 0.5),
            )
            return (
                base
                + getattr(args, 'low_snr_aux_clean_weight', 0.05) * clean_loss
                + getattr(args, 'low_snr_aux_noise_weight', 0.02) * noise_loss
            )

        _low_snr_aux_criterion.needs_mixture = True
        return _low_snr_aux_criterion, (
            f"PIT-SI-SNR+Huber+LowSNRAux("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"clean={getattr(args, 'low_snr_aux_clean_weight', 0.05)}, "
            f"noise={getattr(args, 'low_snr_aux_noise_weight', 0.02)})"
        )
    if loss_name == 'PIT-SI-SNR+URIC-DS':
        criterion = lambda outputs, targets: pit_si_snr_uric_deep_supervision_loss(
            outputs,
            targets,
            stage_weight=args.uric_ds_weight,
            include_final_stage=args.uric_ds_include_final_stage,
            stage_reduction=args.uric_ds_reduction,
        )
        return criterion, (
            f"PIT-SI-SNR+URIC-DS(lambda={args.uric_ds_weight}, "
            f"include_final={args.uric_ds_include_final_stage}, "
            f"reduction={args.uric_ds_reduction})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+URIC-DS':
        criterion = lambda outputs, targets: pit_si_snr_huber_uric_deep_supervision_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            delta=args.pit_si_snr_huber_delta,
            stage_weight=args.uric_ds_weight,
            include_final_stage=args.uric_ds_include_final_stage,
            stage_reduction=args.uric_ds_reduction,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+URIC-DS(a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, d={args.pit_si_snr_huber_delta}, "
            f"lambda={args.uric_ds_weight}, include_final={args.uric_ds_include_final_stage}, "
            f"reduction={args.uric_ds_reduction})"
        )
    if loss_name == 'BW-MRSTFT':
        criterion = lambda outputs, targets: bw_mrstft_loss(
            outputs,
            targets,
            n_ffts=args.mrstft_n_ffts,
            band_ratio=args.mrstft_band_ratio,
            inband_weight=args.mrstft_inband_weight,
            outband_weight=args.mrstft_outband_weight,
            mag_weight=args.mrstft_mag_weight,
            complex_weight=args.mrstft_complex_weight,
        )
        return criterion, "BW-MRSTFT"
    if loss_name == 'EVM':
        criterion = lambda outputs, targets: evm_loss(
            outputs,
            targets,
            eps=args.evm_eps,
        )
        return criterion, "EVM (PIT-aware)"
    if loss_name == 'EVM+CONST':
        criterion = lambda outputs, targets: evm_constellation_loss(
            outputs,
            targets,
            source_names=args.source_names,
            symbol_stride=args.evm_symbol_stride,
            evm_weight=args.evm_weight,
            const_weight=args.const_weight,
            eps=args.evm_eps,
        )
        return criterion, "EVM+CONST (PIT-aware)"
    if loss_name == 'PIT-SI-SNR+BW-MRSTFT':
        criterion = lambda outputs, targets: (
            pit_si_snr_loss(outputs, targets) +
            args.mrstft_lambda * bw_mrstft_loss(
                outputs,
                targets,
                n_ffts=args.mrstft_n_ffts,
                band_ratio=args.mrstft_band_ratio,
                inband_weight=args.mrstft_inband_weight,
                outband_weight=args.mrstft_outband_weight,
                mag_weight=args.mrstft_mag_weight,
                complex_weight=args.mrstft_complex_weight,
            )
        )
        return criterion, "PIT-SI-SNR+BW-MRSTFT"
    if loss_name == 'PIT-SI-SNR+BW-MRSTFT+EVMCONST':
        criterion = lambda outputs, targets: (
            pit_si_snr_loss(outputs, targets)
            + args.mrstft_lambda * bw_mrstft_loss(
                outputs,
                targets,
                n_ffts=args.mrstft_n_ffts,
                band_ratio=args.mrstft_band_ratio,
                inband_weight=args.mrstft_inband_weight,
                outband_weight=args.mrstft_outband_weight,
                mag_weight=args.mrstft_mag_weight,
                complex_weight=args.mrstft_complex_weight,
            )
            + args.evm_lambda * evm_constellation_loss(
                outputs,
                targets,
                source_names=args.source_names,
                symbol_stride=args.evm_symbol_stride,
                evm_weight=args.evm_weight,
                const_weight=args.const_weight,
                eps=args.evm_eps,
            )
        )
        return criterion, "PIT-SI-SNR+BW-MRSTFT+EVMCONST"
    if loss_name == 'PIT-SI-SNR+RMS':
        criterion = lambda outputs, targets: pit_si_snr_rms_loss(
            outputs,
            targets,
            rms_lambda=args.rms_lambda,
        )
        return criterion, f"PIT-SI-SNR+RMS(λ={args.rms_lambda})"
    if loss_name == 'PIT-SI-SNR+CONST':
        criterion = lambda outputs, targets: pit_si_snr_constellation_loss(
            outputs,
            targets,
            source_names=args.source_names,
            symbol_stride=args.evm_symbol_stride,
            const_lambda=args.const_lambda,
        )
        return criterion, f"PIT-SI-SNR+CONST(λ={args.const_lambda})"
    if loss_name == 'PIT-SI-SNR+RMS+CONST':
        criterion = lambda outputs, targets: pit_si_snr_rms_constellation_loss(
            outputs,
            targets,
            source_names=args.source_names,
            symbol_stride=args.evm_symbol_stride,
            rms_lambda=args.rms_lambda,
            const_lambda=args.const_lambda,
        )
        return criterion, f"PIT-SI-SNR+RMS+CONST(rms_λ={args.rms_lambda}, const_λ={args.const_lambda})"
    if loss_name == 'PIT-SI-SNR+Huber+RMS':
        criterion = lambda outputs, targets: pit_si_snr_huber_rms_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            rms_lambda=args.rms_lambda,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+RMS("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"rms_λ={args.rms_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+GPAHuber':
        criterion = lambda outputs, targets: pit_si_snr_huber_gpahuber_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            gpahuber_lambda=args.gpahuber_lambda,
            gpahuber_beta=args.gpahuber_beta,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+GPAHuber("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"gpa_lambda={args.gpahuber_lambda}, "
            f"gpa_beta={args.gpahuber_beta})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+XTALK':
        criterion = lambda outputs, targets: pit_si_snr_huber_xtalk_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            xtalk_lambda=args.xtalk_lambda,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+XTALK("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"xtalk_lambda={args.xtalk_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+CMA':
        criterion = lambda outputs, targets: pit_si_snr_huber_cma_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            cma_lambda=args.cma_lambda,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+CMA("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"cma_lambda={args.cma_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+MC':
        criterion = lambda outputs, targets, mixture: pit_si_snr_huber_mc_loss(
            outputs,
            targets,
            mixture,
            num_sources=args.num_sources,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            mc_lambda=0.05,
            eps=args.eps
        )
        criterion.needs_mixture = True
        return criterion, (
            f"PIT-SI-SNR+Huber+MC ("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"mc=0.05)"
        )
    if loss_name == 'PIT-SI-SNR+Huber+Independence':
        criterion = lambda outputs, targets: pit_si_snr_huber_ind_loss(
            outputs,
            targets,
            num_sources=args.num_sources,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            alpha_ind=0.1,
            eps=1e-8
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+Ind(a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+CyclicProfile':
        criterion = lambda outputs, targets: pit_si_snr_huber_cyclic_profile_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            delta=args.pit_si_snr_huber_delta,
            cyclic_lambda=getattr(args, 'cyclic_profile_lambda', 0.05),
            cyclic_cross_lambda=getattr(args, 'cyclic_profile_cross_lambda', 0.01),
            cyclic_alphas=getattr(args, 'cyclic_profile_alphas', None),
            cyclic_lags=getattr(args, 'cyclic_profile_lags', None),
            num_sources=getattr(args, 'num_sources', None),
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+CyclicProfile("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"cyc_lambda={getattr(args, 'cyclic_profile_lambda', 0.05)}, "
            f"cyc_cross={getattr(args, 'cyclic_profile_cross_lambda', 0.01)}, "
            f"alphas={getattr(args, 'cyclic_profile_alphas', None)}, "
            f"lags={getattr(args, 'cyclic_profile_lags', None)})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+MF':
        criterion = lambda outputs, targets: pit_si_snr_huber_mf_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            mf_lambda=args.mf_lambda,
            mf_window=args.mf_window,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+MF("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"mf_lambda={args.mf_lambda}, mf_window={args.mf_window})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+XTALK+NOISERES':
        def _xtalk_noiseres_criterion(outputs, targets, mixture):
            return pit_si_snr_huber_xtalk_noiseres_loss(
                outputs,
                targets,
                mixture,
                alpha=args.pit_si_snr_huber_alpha,
                beta=args.pit_si_snr_huber_beta,
                xtalk_lambda=args.xtalk_lambda,
                noiseres_lambda=args.noiseres_lambda,
                noiseres_corr_weight=args.noiseres_corr_weight,
                noiseres_whiteness_weight=args.noiseres_whiteness_weight,
                noiseres_max_lag=args.noiseres_max_lag,
                delta=args.pit_si_snr_huber_delta,
            )
        _xtalk_noiseres_criterion.needs_mixture = True
        return _xtalk_noiseres_criterion, (
            f"PIT-SI-SNR+Huber+XTALK+NOISERES("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"xtalk_lambda={args.xtalk_lambda}, "
            f"noiseres_lambda={args.noiseres_lambda}, "
            f"corr_w={args.noiseres_corr_weight}, "
            f"white_w={args.noiseres_whiteness_weight}, "
            f"max_lag={args.noiseres_max_lag})"
        )
    if loss_name == 'PIT-SI-SNR+ComplexHuber+RMS':
        criterion = lambda outputs, targets: pit_si_snr_complex_huber_rms_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            rms_lambda=args.rms_lambda,
            huber_beta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+ComplexHuber+RMS("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"huber_beta={args.pit_si_snr_huber_delta}, "
            f"rms_λ={args.rms_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+RMS+URIC-DS':
        criterion = lambda outputs, targets: pit_si_snr_huber_rms_uric_deep_supervision_loss(
            outputs,
            targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            rms_lambda=args.rms_lambda,
            delta=args.pit_si_snr_huber_delta,
            stage_weight=args.uric_ds_weight,
            include_final_stage=args.uric_ds_include_final_stage,
            stage_reduction=args.uric_ds_reduction,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+RMS+URIC-DS("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"rms_λ={args.rms_lambda}, "
            f"ds_λ={args.uric_ds_weight}, "
            f"include_final={args.uric_ds_include_final_stage}, "
            f"reduction={args.uric_ds_reduction})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+CONST':
        criterion = lambda outputs, targets: pit_si_snr_huber_const_loss(
            outputs,
            targets,
            source_names=args.source_names,
            symbol_stride=args.evm_symbol_stride,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            const_lambda=args.const_lambda,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+CONST("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"const_λ={args.const_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+RMS+CONST':
        criterion = lambda outputs, targets: pit_si_snr_huber_rms_const_loss(
            outputs,
            targets,
            source_names=args.source_names,
            symbol_stride=args.evm_symbol_stride,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            rms_lambda=args.rms_lambda,
            const_lambda=args.const_lambda,
            delta=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+RMS+CONST("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"rms_λ={args.rms_lambda}, const_λ={args.const_lambda})"
        )
    if loss_name == 'CTDCRN':
        from util.loss import ctdcrn_loss
        return ctdcrn_loss, "CTDCRN (MSE_ave - SI-SNR_ave)"
    if loss_name == 'PIT-CTDCRN':
        from util.loss import pit_ctdcrn_loss
        return pit_ctdcrn_loss, "PIT-CTDCRN (MSE_ave - SI-SNR_ave)"
    if loss_name == 'MR-STFT':
        criterion = lambda outputs, targets: multi_resolution_stft_loss(
            outputs, targets,
            n_ffts=args.mrstft_v2_n_ffts,
            pit=True,
        )
        return criterion, f"MR-STFT(ffts={args.mrstft_v2_n_ffts})"
    if loss_name == 'PIT-SI-SNR+Huber+MRSTFT':
        criterion = lambda outputs, targets: pit_si_snr_huber_mrstft_loss(
            outputs, targets,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            mrstft_lambda=args.mrstft_v2_lambda,
            delta=args.pit_si_snr_huber_delta,
            n_ffts=args.mrstft_v2_n_ffts,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+MRSTFT("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"mrstft_λ={args.mrstft_v2_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+MRSTFT+MIXCONS':
        # This loss needs the input mixture - handled by a special wrapper.
        # The training loop will detect `criterion.needs_mixture = True`
        # and pass inputs as the third argument.
        def _mixcons_criterion(outputs, targets, mixture):
            return pit_si_snr_huber_mrstft_mixcons_loss(
                outputs, targets, mixture,
                alpha=args.pit_si_snr_huber_alpha,
                beta=args.pit_si_snr_huber_beta,
                mrstft_lambda=args.mrstft_v2_lambda,
                mixcons_lambda=args.mixcons_lambda,
                delta=args.pit_si_snr_huber_delta,
                n_ffts=args.mrstft_v2_n_ffts,
            )
        _mixcons_criterion.needs_mixture = True
        return _mixcons_criterion, (
            f"PIT-SI-SNR+Huber+MRSTFT+MIXCONS("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"mrstft_λ={args.mrstft_v2_lambda}, "
            f"mixcons_λ={args.mixcons_lambda})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+AMR':
        # Infer modulation labels from data_choice
        NUM_SOURCES = len(args.source_names)
        mod_labels = get_mod_labels_from_data_choice(args.data_choice, NUM_SOURCES)
        if mod_labels is None:
            raise ValueError(
                f"Cannot infer modulation labels from data_choice='{args.data_choice}'. "
                f"PIT-SI-SNR+Huber+AMR requires a known MATLAB dataset."
            )

        class _AMRCriterion:
            """Callable wrapper that handles the AMR model's tuple output."""
            needs_amr_mode = True  # tells training loop to use amr_mode

            def __init__(self, mod_labels_, alpha_, beta_, cls_weight_, delta_):
                self.mod_labels = mod_labels_
                self.alpha = alpha_
                self.beta = beta_
                self.cls_weight = cls_weight_
                self.delta = delta_

            def __call__(self, model_output, targets):
                if isinstance(model_output, tuple):
                    sep_output, cls_logits = model_output
                else:
                    # sep_only mode - no classification needed, fall back
                    return pit_si_snr_huber_loss(
                        model_output, targets,
                        alpha=self.alpha, beta=self.beta, delta=self.delta,
                    )
                return pit_si_snr_huber_amr_loss(
                    sep_output, targets, cls_logits, self.mod_labels,
                    alpha=self.alpha, beta=self.beta,
                    cls_weight=self.cls_weight, delta=self.delta,
                )

        criterion = _AMRCriterion(
            mod_labels_=mod_labels,
            alpha_=args.pit_si_snr_huber_alpha,
            beta_=args.pit_si_snr_huber_beta,
            cls_weight_=args.amr_cls_weight,
            delta_=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+AMR("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"cls_w={args.amr_cls_weight}, "
            f"mods={mod_labels})"
        )
    if loss_name == 'PIT-SI-SNR+Huber+DEMOD':
        class _SoftDemodCriterion:
            """Callable wrapper for joint separation + soft demodulation loss."""
            needs_bits = True  # tells training loop to pass bits

            def __init__(self, alpha_, beta_, demod_weight_, symbol_weight_, delta_):
                self.alpha = alpha_
                self.beta = beta_
                self.demod_weight = demod_weight_
                self.symbol_weight = symbol_weight_
                self.delta = delta_

            def __call__(self, model_output, targets, bits=None):
                if isinstance(model_output, tuple) and bits is not None:
                    sep_output, demod_outputs = model_output
                    return pit_si_snr_huber_demod_loss(
                        sep_output, targets, demod_outputs, bits,
                        alpha=self.alpha, beta=self.beta,
                        demod_weight=self.demod_weight,
                        symbol_weight=self.symbol_weight,
                        delta=self.delta,
                    )
                # Fallback: sep_only mode or no bits available
                out = model_output[0] if isinstance(model_output, tuple) else model_output
                return pit_si_snr_huber_loss(
                    out, targets,
                    alpha=self.alpha, beta=self.beta, delta=self.delta,
                )

        criterion = _SoftDemodCriterion(
            alpha_=args.pit_si_snr_huber_alpha,
            beta_=args.pit_si_snr_huber_beta,
            demod_weight_=args.demod_weight,
            symbol_weight_=args.demod_symbol_weight,
            delta_=args.pit_si_snr_huber_delta,
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+DEMOD("
            f"a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta}, "
            f"d={args.pit_si_snr_huber_delta}, "
            f"demod_w={args.demod_weight}, "
            f"sym_w={args.demod_symbol_weight})"
        )
    raise ValueError(f"Unsupported loss_fun: {loss_name}")


def set_random_seeds(seed):
    """Set all random seeds to ensure reproducibility of experiments"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Must be False for reproducibility

    # Global deterministic mode (covers STFT, interpolate, scatter, etc.)
    # warn_only=True: Mamba's CUDA selective-scan kernel may not support
    # deterministic mode - this lets it fall back gracefully.
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # deterministic cuBLAS
    
    print(f"Random seeds set to: {seed}")


def save_experiment_config(args, results_folder):
    """Save experiment configuration to JSON file"""
    config = {
        'timestamp': datetime.now().isoformat(),
        'data_choice': args.data_choice,
        'pretrain_data_choices': resolve_pretrain_data_choices(args),
        'pretrain_sampling': getattr(args, 'pretrain_sampling', None),
        'pretrain_dataset_weights': getattr(args, 'pretrain_dataset_weights', None),
        'pretrain_input_size': getattr(args, 'pretrain_input_size', None),
        'pretrain_length_policy': getattr(args, 'pretrain_length_policy', None),
        'mode': args.mode,
        'loss_fun': args.loss_fun,
        'save_checkpoint_every': args.save_checkpoint_every,
        'two_phase_loss': args.two_phase_loss,
        'loss_phase1': args.loss_phase1,
        'loss_phase2': args.loss_phase2,
        'loss_phase1_ratio': args.loss_phase1_ratio,
        'loss_switch_epoch': args.loss_switch_epoch,
        'si_snr_mse_alpha': args.si_snr_mse_alpha,
        'si_snr_mse_beta': args.si_snr_mse_beta,
        'si_snr_huber_alpha': args.si_snr_huber_alpha,
        'si_snr_huber_beta': args.si_snr_huber_beta,
        'si_snr_huber_delta': args.si_snr_huber_delta,
        'pit_huber_delta': args.pit_huber_delta,
        'pit_si_snr_mse_alpha': args.pit_si_snr_mse_alpha,
        'pit_si_snr_mse_beta': args.pit_si_snr_mse_beta,
        'pit_si_snr_huber_alpha': args.pit_si_snr_huber_alpha,
        'pit_si_snr_huber_beta': args.pit_si_snr_huber_beta,
        'pit_si_snr_huber_delta': args.pit_si_snr_huber_delta,
        'xtalk_lambda': args.xtalk_lambda,
        'noiseres_lambda': args.noiseres_lambda,
        'noiseres_corr_weight': args.noiseres_corr_weight,
        'noiseres_whiteness_weight': args.noiseres_whiteness_weight,
        'noiseres_max_lag': args.noiseres_max_lag,
        'uric_ds_weight': args.uric_ds_weight,
        'uric_ds_reduction': args.uric_ds_reduction,
        'uric_ds_include_final_stage': args.uric_ds_include_final_stage,
        'ric_return_intermediate': args.ric_return_intermediate,
        'ric_update_block_type': args.ric_update_block_type,
        'ric_dilations': args.ric_dilations,
        'ric_num_heads': args.ric_num_heads,
        'ric_attention_stride': args.ric_attention_stride,
        'ric_ffn_multiplier': args.ric_ffn_multiplier,
        'report_phase_flip': args.report_phase_flip,
        'phase_flip_tolerance_deg': args.phase_flip_tolerance_deg,
        'phase_flip_min_sc': args.phase_flip_min_sc,
        'phase_flip_mode': args.phase_flip_mode,
        'stage': args.stage,
        'seed': args.seed,
        'split_strategy': args.split_strategy,
        'resume_checkpoint': args.resume_checkpoint,
        'init_checkpoint': getattr(args, 'init_checkpoint', None),
        'run_id': args.run_id if hasattr(args, 'run_id') else None,
        'multiple_runs': args.multiple_runs if hasattr(args, 'multiple_runs') else False
    }
    
    config_path = RESULTS_ROOT / results_folder / "config" / "experiment_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config


def get_model_config_path(stage):
    """Get configuration file path based on stage"""
    config_mapping = {
        2: CONFIG_ROOT / "model_config_IQ_stage2.yaml",
        1: CONFIG_ROOT / "model_config_mamba.yaml",

        161: CONFIG_ROOT / "model_config_stage161_lssg_se.yaml",
        162: CONFIG_ROOT / "model_config_stage162_lssg_swiglu.yaml",
        163: CONFIG_ROOT / "model_config_stage163_dccb_deep.yaml",
        164: CONFIG_ROOT / "model_config_stage164_dccb_cross_attn.yaml",
        165: CONFIG_ROOT / "model_config_stage165_dccb_adaptive_lags.yaml",
        166: CONFIG_ROOT / "model_config_stage166_dccb_mamba.yaml",
        167: CONFIG_ROOT / "model_config_resunet1d_legacy5.yaml",
        168: CONFIG_ROOT / "model_config_stage168_agent_attention.yaml",
        169: CONFIG_ROOT / "model_config_stage169_transnext_attention.yaml",
        170: CONFIG_ROOT / "model_config_stage170_bilevel_routing_attention.yaml",
        171: CONFIG_ROOT / "model_config_stage171_deformable_temporal_attention.yaml",
        172: CONFIG_ROOT / "model_config_stage172_gla_encoder.yaml",
        173: CONFIG_ROOT / "model_config_stage173_mega_encoder.yaml",
        174: CONFIG_ROOT / "model_config_stage174_hyena_encoder.yaml",
        175: CONFIG_ROOT / "model_config_stage175_retnet_encoder.yaml",
        176: CONFIG_ROOT / "model_config_IQ_stage4_uric.yaml",
        177: CONFIG_ROOT / "model_config_stage177_griffin_encoder.yaml",
        178: CONFIG_ROOT / "model_config_stage178_xlstm_encoder.yaml",
        179: CONFIG_ROOT / "model_config_stage179_spectral_encoder.yaml",
        180: CONFIG_ROOT / "model_config_stage180_delta_linear_encoder.yaml",
        181: CONFIG_ROOT / "model_config_stage181_lssg_dw_skip.yaml",
        182: CONFIG_ROOT / "model_config_stage182_deformable_temporal_skip.yaml",
        183: CONFIG_ROOT / "model_config_stage183_frequency_aware_skip.yaml",
        184: CONFIG_ROOT / "model_config_stage184_complex_aware_skip.yaml",
        185: CONFIG_ROOT / "model_config_stage185_spectral_lowrank_encoder.yaml",
        186: CONFIG_ROOT / "model_config_stage186_resunet_hyena_bottleneck.yaml",
        187: CONFIG_ROOT / "model_config_stage187_resunet_spectral_lowrank_bottleneck.yaml",
        188: CONFIG_ROOT / "model_config_stage188_resunet_mega_mid_encoder.yaml",
        189: CONFIG_ROOT / "model_config_stage189_spectral_encoder_regularized.yaml",
        190: CONFIG_ROOT / "model_config_IQ_stage4_psk_phase_prior.yaml",
        191: CONFIG_ROOT / "model_config_IQ_stage4_qam_lattice_prior.yaml",
        192: CONFIG_ROOT / "model_config_IQ_stage4_apsk_ring_prior.yaml",
        193: CONFIG_ROOT / "model_config_IQ_stage4_output_topology_loss.yaml",
        194: CONFIG_ROOT / "model_config_IQ_stage4_feature_topology_adapter.yaml",
        195: CONFIG_ROOT / "model_config_IQ_stage4_separation_constraint_loss.yaml",
        196: CONFIG_ROOT / "model_config_IQ_stage4_cyclic_wiener_residual.yaml",
        197: CONFIG_ROOT / "model_config_bimamba_estimated_cyclofresh.yaml",
        198: CONFIG_ROOT / "model_config_bimamba_safe_allstage.yaml",
        199: CONFIG_ROOT / "model_config_IQ_stage4_low_snr_se.yaml",
        200: CONFIG_ROOT / "model_config_bimamba_direction_gated.yaml",
        201: CONFIG_ROOT / "model_config_bimamba_local_global_allstage.yaml",
        202: CONFIG_ROOT / "model_config_IQ_stage4_asg_mamba.yaml",
        203: CONFIG_ROOT / "model_config_IQ_stage4_low_snr_aux.yaml",
        204: CONFIG_ROOT / "model_config_IQ_stage4_low_snr_snr_cond.yaml",
        205: CONFIG_ROOT / "model_config_IQ_stage4_low_snr_cyclic_cond.yaml",
        206: CONFIG_ROOT / "model_config_bimamba_diff_fusion.yaml",
        207: CONFIG_ROOT / "model_config_bimamba_adaptive_diff_fusion.yaml",
        208: CONFIG_ROOT / "model_config_IQ_stage4_neural_wiener_se.yaml",
        210: CONFIG_ROOT / "model_config_IQ_stage4_multihyp_cyclic_reliability.yaml",
        211: CONFIG_ROOT / "model_config_IQ_stage4_cross_snr_consistency.yaml",
        212: CONFIG_ROOT / "model_config_IQ_stage4_receiver_symbol.yaml",
        213: CONFIG_ROOT / "model_config_IQ_stage4_cross_snr_receiver.yaml",
        214: CONFIG_ROOT / "model_config_IQ_stage4_confidence_soft_pit.yaml",
        215: CONFIG_ROOT / "model_config_bimamba_complex_diff_shared.yaml",
        216: CONFIG_ROOT / "model_config_IQ_stage4_cumulant_prior.yaml",
        217: CONFIG_ROOT / "model_config_bimamba_time_reversal_shared.yaml",
        218: CONFIG_ROOT / "model_config_bimamba_alternating_global_local.yaml",
        219: CONFIG_ROOT / "model_config_IQ_stage219_pr_unet.yaml",
        220: CONFIG_ROOT / "model_config_IQ_stage220_pr_shared_perm.yaml",
        221: CONFIG_ROOT / "model_config_IQ_stage221_pr_restricted_skip.yaml",
        222: CONFIG_ROOT / "model_config_IQ_stage222_evidence_moe.yaml",
        223: CONFIG_ROOT / "model_config_IQ_stage223_noise_contrastive_prior.yaml",
        224: CONFIG_ROOT / "model_config_IQ_stage224_blind_sync_factorized.yaml",
        225: CONFIG_ROOT / "model_config_IQ_stage225_gaussian_residual_prior.yaml",
        226: CONFIG_ROOT / "model_config_IQ_stage226_adaptive_multiview_prior.yaml",
        227: CONFIG_ROOT / "model_config_IQ_stage227_qam_source_prior.yaml",
        228: CONFIG_ROOT / "model_config_IQ_stage228_qam_mma_unrolled.yaml",
        229: CONFIG_ROOT / "model_config_IQ_stage229_qam_density_prior.yaml",
        230: CONFIG_ROOT / "model_config_IQ_stage230_qam_timing_prior.yaml",
        231: CONFIG_ROOT / "model_config_IQ_stage231_multiview_consistent.yaml",
        232: CONFIG_ROOT / "model_config_IQ_stage232_multiview_pit_only.yaml",
        233: CONFIG_ROOT / "model_config_IQ_stage233_noise_mc_only.yaml",
        234: CONFIG_ROOT / "model_config_IQ_stage234_phase_equiv_only.yaml",
        235: CONFIG_ROOT / "model_config_bimamba_cross_scale_single.yaml",
        236: CONFIG_ROOT / "model_config_bimamba_cross_scale_multi.yaml",
        237: CONFIG_ROOT / "model_config_bimamba_cross_scale_evidence.yaml",
        238: CONFIG_ROOT / "model_config_IQ_stage238_qam_turbo_unfold.yaml",
        239: CONFIG_ROOT / "model_config_bimamba_cross_scale_estimated_cyclofresh.yaml",
        240: CONFIG_ROOT / "model_config_bimamba_cross_scale_aligned.yaml",
        241: CONFIG_ROOT / "model_config_bimamba_cross_scale_multires_kv.yaml",
        242: CONFIG_ROOT / "model_config_bimamba_cross_scale_bounded_channel.yaml",
        243: CONFIG_ROOT / "model_config_bimamba_phase_equivariant_fusion.yaml",
        244: CONFIG_ROOT / "model_config_bimamba_physical_token_cross_attention.yaml",
        245: CONFIG_ROOT / "model_config_bimamba_bottleneck_self_attention.yaml",
        246: CONFIG_ROOT / "model_config_bimamba_hymba_parallel.yaml",
        247: CONFIG_ROOT / "model_config_bimamba_rf_physical_kv.yaml",
        248: CONFIG_ROOT / "model_config_bimamba_enhanced_global_cross_attention.yaml",
        249: CONFIG_ROOT / "model_config_bimamba_dual_memory_cross_attention.yaml",
        250: CONFIG_ROOT / "model_config_bimamba_hierarchical_additive_fusion.yaml",
        251: CONFIG_ROOT / "model_config_bimamba_physical_routed_enhanced_cross_attention.yaml",
        252: CONFIG_ROOT / "model_config_bimamba_unified_physical_global_kv.yaml",
        253: CONFIG_ROOT / "model_config_bimamba_physical_film_global_memory.yaml",
        254: CONFIG_ROOT / "model_config_bimamba_scale_isolated_physical_fusion.yaml",
        255: CONFIG_ROOT / "model_config_bimamba_identity_aware_physical_moe.yaml",
        256: CONFIG_ROOT / "model_config_bimamba_cross_gated_dual_memory.yaml",
        257: CONFIG_ROOT / "model_config_stage257_wavenet_mamba.yaml",
        258: CONFIG_ROOT / "model_config_stage258_wavenet_bimamba.yaml",
        259: CONFIG_ROOT / "model_config_stage259_wavenet_multirate_mamba.yaml",
        260: CONFIG_ROOT / "model_config_stage260_wavenet_multirate_bimamba.yaml",
        261: CONFIG_ROOT / "model_config_stage261_wavenet_interleaved_mamba.yaml",
        262: CONFIG_ROOT / "model_config_stage262_wavenet_interleaved_bimamba.yaml",
        268: CONFIG_ROOT / "model_config_stage268_wavenet_interleaved_gated_bimamba.yaml",
        269: CONFIG_ROOT / "model_config_stage269_wavenet_mamba_film_controller.yaml",
        270: CONFIG_ROOT / "model_config_stage270_wavenet_mamba_dilation_skip_router.yaml",
        271: CONFIG_ROOT / "model_config_stage271_wavenet_phase_aware_reverse_mamba.yaml",
        272: CONFIG_ROOT / "model_config_stage272_wavenet_stage235_memory.yaml",
        273: CONFIG_ROOT / "model_config_stage273_wavenet20_no_mamba.yaml",
        274: CONFIG_ROOT / "model_config_stage274_wavenet15_interleaved_mamba.yaml",
        275: CONFIG_ROOT / "model_config_stage275_wavenet15_no_mamba.yaml",
        276: CONFIG_ROOT / "model_config_stage276_wavenet5_chunk_mamba_wavenet5.yaml",
        277: CONFIG_ROOT / "model_config_stage277_wavenet10_chunk_mamba_wavenet10.yaml",
        278: CONFIG_ROOT / "model_config_stage278_wavenet_antialiased_mamba.yaml",
        279: CONFIG_ROOT / "model_config_stage279_wavenet_temporal_physical_controller.yaml",
        280: CONFIG_ROOT / "model_config_stage280_symbol_clock_wavenet.yaml",
        281: CONFIG_ROOT / "model_config_stage281_complex_symbol_clock_wavenet.yaml",
        282: CONFIG_ROOT / "model_config_stage282_complex_clock_control_wavenet.yaml",
        283: CONFIG_ROOT / "model_config_stage283_complex_stem_real_wavenet20.yaml",
        284: CONFIG_ROOT / "model_config_stage284_complex5_real15_wavenet.yaml",
        285: CONFIG_ROOT / "model_config_stage285_complex10_real10_wavenet.yaml",
        286: CONFIG_ROOT / "model_config_stage286_full_complex_wavenet20.yaml",
        287: CONFIG_ROOT / "model_config_stage287_full_complex_widelylinear_adapter.yaml",
        288: CONFIG_ROOT / "model_config_stage288_full_complex_real_head.yaml",
        289: CONFIG_ROOT / "model_config_stage289_full_complex_strict_head.yaml",
        290: CONFIG_ROOT / "model_config_stage290_stage4_complex_c1_stem.yaml",
        291: CONFIG_ROOT / "model_config_stage291_stage4_complex_c2_real_mamba_real_head.yaml",
        292: CONFIG_ROOT / "model_config_stage292_stage4_complex_c3_real_mamba_strict_head.yaml",
        293: CONFIG_ROOT / "model_config_stage293_stage4_complex_c4_equivariant_mamba_real_head.yaml",
        294: CONFIG_ROOT / "model_config_stage294_stage4_complex_c5_equivariant_mamba_strict_head.yaml",
        295: CONFIG_ROOT / "model_config_stage295_stage4_complex_state_mamba.yaml",
        296: CONFIG_ROOT / "model_config_stage296_fsq_token_ce.yaml",
        297: CONFIG_ROOT / "model_config_stage297_stage79_complex_c1_stem.yaml",
        298: CONFIG_ROOT / "model_config_stage298_stage197_complex_c1_stem.yaml",
        299: CONFIG_ROOT / "model_config_stage299_stage290_stage295_hybrid.yaml",
        300: CONFIG_ROOT / "model_config_stage300_stage4_cross_scale_single.yaml",
        301: CONFIG_ROOT / "model_config_stage301_stage299_cross_scale.yaml",
        302: CONFIG_ROOT / "model_config_stage302_stage298_complex_bottleneck.yaml",
        303: CONFIG_ROOT / "model_config_stage303_complex_bimamba_cross_scale.yaml",
        304: CONFIG_ROOT / "model_config_stage304_stage298_stage299_fusion.yaml",
        305: CONFIG_ROOT / "model_config_stage305_stage299_gated_fresh.yaml",
        306: CONFIG_ROOT / "model_config_stage306_stage4_physical_canonical.yaml",
        307: CONFIG_ROOT / "model_config_stage307_stage4_symbol_delay_doppler_rf.yaml",
        308: CONFIG_ROOT / "model_config_stage308_stage4_fadc1d.yaml",
        309: CONFIG_ROOT / "model_config_stage309_stage4_fdconv1d.yaml",
        310: CONFIG_ROOT / "model_config_stage310_stage4_unireplk1d.yaml",
        311: CONFIG_ROOT / "model_config_stage311_stage4_shiftwise1d.yaml",
        312: CONFIG_ROOT / "model_config_stage312_stage4_moderntcn1d.yaml",
        313: CONFIG_ROOT / "model_config_stage313_stage4_morf1d.yaml",
        314: CONFIG_ROOT / "model_config_stage314_stage4_dcnv4_1d.yaml",
        315: CONFIG_ROOT / "model_config_stage315_stage4_wtconv1d.yaml",
        316: CONFIG_ROOT / "model_config_stage316_stage4_dcls1d.yaml",
        317: CONFIG_ROOT / "model_config_stage317_fdconv_unireplk_serial.yaml",
        318: CONFIG_ROOT / "model_config_stage318_fdconv_unireplk_hierarchical.yaml",
        319: CONFIG_ROOT / "model_config_stage319_fdconv_unireplk_parallel.yaml",
        320: CONFIG_ROOT / "model_config_stage320_stage4_no_mamba.yaml",
        321: CONFIG_ROOT / "model_config_stage321_no_mamba_fdconv.yaml",
        322: CONFIG_ROOT / "model_config_stage322_no_mamba_unireplk.yaml",
        323: CONFIG_ROOT / "model_config_stage323_complex_fdconv.yaml",
        324: CONFIG_ROOT / "model_config_stage324_complex_unireplk.yaml",
        325: CONFIG_ROOT / "model_config_stage325_stage290_fdconv.yaml",
        326: CONFIG_ROOT / "model_config_stage326_stage290_unireplk.yaml",
        327: CONFIG_ROOT / "model_config_stage327_stage290_fdconv_unireplk.yaml",
        328: CONFIG_ROOT / "model_config_stage328_stage197_unireplk.yaml",
        329: CONFIG_ROOT / "model_config_stage329_stage79_unireplk.yaml",
        330: CONFIG_ROOT / "model_config_stage330_rf_mamba3_trapezoidal.yaml",
        331: CONFIG_ROOT / "model_config_stage331_rf_mamba3_cyclo_anchor.yaml",
        332: CONFIG_ROOT / "model_config_stage332_rf_mamba3_reliability.yaml",
        333: CONFIG_ROOT / "model_config_stage333_rf_mamba3_combined.yaml",
        334: CONFIG_ROOT / "model_config_stage334_phase_folded_mamba.yaml",
        335: CONFIG_ROOT / "model_config_stage335_stage4_mamba2_ssd.yaml",
        336: CONFIG_ROOT / "model_config_stage336_stage4_s4d.yaml",
        337: CONFIG_ROOT / "model_config_stage337_stage4_shared_multiscale_mamba.yaml",
        338: CONFIG_ROOT / "model_config_stage338_stage4_fixed_role_rf_mamba.yaml",
        339: CONFIG_ROOT / "model_config_stage339_stage4_routed_role_rf_mamba.yaml",
        340: CONFIG_ROOT / "model_config_stage340_stage4_official_mamba3.yaml",
        341: CONFIG_ROOT / "model_config_stage341_rf_mamba3_cyclic_reliability_fast.yaml",
        342: CONFIG_ROOT / "model_config_stage342_fresh_complex_rf_mamba3_unireplk.yaml",
        343: CONFIG_ROOT / "model_config_stage343_official_mamba3_cyclic_anchor.yaml",
        344: CONFIG_ROOT / "model_config_stage344_official_mamba3_reliability.yaml",
        345: CONFIG_ROOT / "model_config_stage345_official_mamba3_cyclic_reliability.yaml",
        346: CONFIG_ROOT / "model_config_stage346_official_mamba3_shared_cyclo_conditioning.yaml",
        347: CONFIG_ROOT / "model_config_stage347_real_state_trap_reliability.yaml",
        348: CONFIG_ROOT / "model_config_stage348_s4d_reliability.yaml",
        349: CONFIG_ROOT / "model_config_stage349_complex_rf_mamba3_unireplk.yaml",
        350: CONFIG_ROOT / "model_config_stage350_fresh_complex_unireplk.yaml",
        351: CONFIG_ROOT / "model_config_stage351_complex_stem_rf_mamba3.yaml",
        352: CONFIG_ROOT / "model_config_stage352_complex_stem_s4d.yaml",
        353: CONFIG_ROOT / "model_config_stage353_complex_stem_s4d_unireplk.yaml",
        354: CONFIG_ROOT / "model_config_stage354_rf_mamba3_a2_a3.yaml",
        355: CONFIG_ROOT / "model_config_stage355_rf_mamba3_a2_a4.yaml",
        356: CONFIG_ROOT / "model_config_stage356_rf_mamba3_a3_a4.yaml",
        357: CONFIG_ROOT / "model_config_stage357_strict_complex_s4d.yaml",
        359: CONFIG_ROOT / "model_config_stage359_bimamba_cross_scale_unireplk.yaml",
        360: CONFIG_ROOT / "model_config_stage360_iqumamba_cross_scale_unireplk.yaml",
        361: CONFIG_ROOT / "model_config_stage361_stage12_hydra.yaml",
        362: CONFIG_ROOT / "model_config_stage362_stage12_complex_state.yaml",
        363: CONFIG_ROOT / "model_config_stage363_stage12_multiscale_bimamba.yaml",
        364: CONFIG_ROOT / "model_config_stage364_stage12_independent_complex_state.yaml",
        365: CONFIG_ROOT / "model_config_stage365_stage364_unireplk.yaml",
        371: CONFIG_ROOT / "model_config_stage371_stage365_latent_mask_real.yaml",
        372: CONFIG_ROOT / "model_config_stage372_stage365_latent_mask_complex.yaml",
        373: CONFIG_ROOT / "model_config_stage373_stage365_latent_mask_residual.yaml",
        374: CONFIG_ROOT / "model_config_stage374_stage365_latent_mask_conservation.yaml",
        375: CONFIG_ROOT / "model_config_stage375_stage365_bottleneck_mask_real.yaml",
        376: CONFIG_ROOT / "model_config_stage376_stage56_latent_mask_real.yaml",
        377: CONFIG_ROOT / "model_config_stage377_stage56_complexstate_unireplk_latent_mask_real.yaml",
        378: CONFIG_ROOT / "model_config_stage378_kutii_dual_source_wavenet.yaml",
        366: CONFIG_ROOT / "model_config_stage366_stage4_cross_snr_sync_conditioned.yaml",
        367: CONFIG_ROOT / "model_config_stage367_stage4_cross_snr_ema.yaml",
        368: CONFIG_ROOT / "model_config_stage368_stage4_sync_conditioned.yaml",
        369: CONFIG_ROOT / "model_config_stage369_physical_sync_teacher.yaml",
        370: CONFIG_ROOT / "model_config_stage370_physical_sync_clean_teacher.yaml",
        265: CONFIG_ROOT / "model_config_stage265_wavenet_interleaved_crossscale_bimamba.yaml",
        266: CONFIG_ROOT / "model_config_stage266_wavenet_interleaved_physical_moe_bimamba.yaml",
        267: CONFIG_ROOT / "model_config_stage267_wavenet_interleaved_cyclofresh.yaml",
        263: CONFIG_ROOT / "model_config_stage263_unet_compute_matched_68.yaml",
        264: CONFIG_ROOT / "model_config_stage264_iq_resdilated_unet.yaml",

        3: CONFIG_ROOT / "model_config_IQ.yaml",
        4: CONFIG_ROOT / "model_config_IQ_stage4.yaml",
        5: CONFIG_ROOT / "model_config_IQ_stage5.yaml",
        6: CONFIG_ROOT / "model_config_tfgridnet_stage4.yaml",
        7: CONFIG_ROOT / "model_config_tiger.yaml",
        8: CONFIG_ROOT / "model_config_tiger_fast.yaml",
        9: CONFIG_ROOT / "model_config_tiger_tiny.yaml",
        10: CONFIG_ROOT / "model_config_tfgridnet_fast.yaml",
        11: CONFIG_ROOT / "model_config_tfgridnet_turbo.yaml",
        12: CONFIG_ROOT / "model_config_bimamba.yaml",
        13: CONFIG_ROOT / "model_config_spmamba.yaml",
        14: CONFIG_ROOT / "model_config_conformer_gridnet.yaml",
        15: CONFIG_ROOT / "model_config_spmamba_fast.yaml",
        16: CONFIG_ROOT / "model_config_IQ_bimamba_match.yaml",
        17: CONFIG_ROOT / "model_config_dual_domain.yaml",
        18: CONFIG_ROOT / "model_config_nes2net.yaml",
        19: CONFIG_ROOT / "model_config_dual_domain_lite.yaml",
        20: CONFIG_ROOT / "model_config_ctdcrn.yaml",
        21: CONFIG_ROOT / "model_config_dual_domain_v2.yaml",
        22: CONFIG_ROOT / "model_config_dual_domain_v3.yaml",
        23: CONFIG_ROOT / "model_config_dual_domain_zeroinit.yaml",
        24: CONFIG_ROOT / "model_config_dual_domain_dualpath.yaml",
        25: CONFIG_ROOT / "model_config_icassp_baseline_unet.yaml",
        26: CONFIG_ROOT / "model_config_icassp_baseline_wavenet.yaml",
        27: CONFIG_ROOT / "model_config_dual_domain_crossmamba.yaml",
        28: CONFIG_ROOT / "model_config_dual_domain_v4.yaml",
        29: CONFIG_ROOT / "model_config_dual_domain_mamba2.yaml",
        30: CONFIG_ROOT / "model_config_dual_domain_bandsplit.yaml",
        31: CONFIG_ROOT / "model_config_dual_domain_small.yaml",
        32: CONFIG_ROOT / "model_config_bimamba_lk.yaml",
        33: CONFIG_ROOT / "model_config_bimamba_jamba.yaml",
        34: CONFIG_ROOT / "model_config_convnext.yaml",
        35: CONFIG_ROOT / "model_config_bimamba_amr.yaml",
        36: CONFIG_ROOT / "model_config_bimamba_softdemod.yaml",
        37: CONFIG_ROOT / "model_config_bimamba_softdemod_v3.yaml",
        38: CONFIG_ROOT / "model_config_bimamba_softdemod_v3.yaml",
        39: CONFIG_ROOT / "model_config_bimamba_mcproj.yaml",
        40: CONFIG_ROOT / "model_config_bimamba_uric.yaml",
        41: CONFIG_ROOT / "model_config_bimamba_uric_aug.yaml",
        42: CONFIG_ROOT / "model_config_resunet1d.yaml",
        43: CONFIG_ROOT / "model_config_transformer1d.yaml",
        44: CONFIG_ROOT / "model_config_bimamba_csb.yaml",
        45: CONFIG_ROOT / "model_config_transformer1d_patch.yaml",
        46: CONFIG_ROOT / "model_config_transformer1d_patch_rope.yaml",
        47: CONFIG_ROOT / "model_config_resunet1d_uric.yaml",
        48: CONFIG_ROOT / "model_config_bimamba_csb_uric.yaml",
        49: CONFIG_ROOT / "model_config_bimamba_admm.yaml",
        50: CONFIG_ROOT / "model_config_bimamba_pgdu.yaml",
        51: CONFIG_ROOT / "model_config_bimamba_fullcomplex.yaml",
        52: CONFIG_ROOT / "model_config_bimamba_gainphase.yaml",
        53: CONFIG_ROOT / "model_config_complex_unet1d.yaml",
        54: CONFIG_ROOT / "model_config_real_unet1d.yaml",
        55: CONFIG_ROOT / "model_config_bimamba_csb_scan.yaml",
        56: CONFIG_ROOT / "model_config_resunet1d_noasc.yaml",
        57: CONFIG_ROOT / "model_config_bimamba_csb_cag.yaml",
        58: CONFIG_ROOT / "model_config_bimamba_csb_phasediff.yaml",
        59: CONFIG_ROOT / "model_config_bimamba_csb_cmasc.yaml",
        60: CONFIG_ROOT / "model_config_bimamba_csb_constellation.yaml",
        61: CONFIG_ROOT / "model_config_bimamba_layerscale.yaml",
        62: CONFIG_ROOT / "model_config_bimamba_localglobal.yaml",
        63: CONFIG_ROOT / "model_config_bimamba_glg.yaml",
        64: CONFIG_ROOT / "model_config_bimamba_complex_mask.yaml",
        65: CONFIG_ROOT / "model_config_bimamba_stage4.yaml",
        66: CONFIG_ROOT / "model_config_bimamba_stage4.yaml",
        67: CONFIG_ROOT / "model_config_IQ_stage5_320.yaml",
        68: CONFIG_ROOT / "model_config_IQ_stage4_decodermamba.yaml",
        69: CONFIG_ROOT / "model_config_bimamba_stage4_uric.yaml",
        70: CONFIG_ROOT / "model_config_IQ_stage4_rfscan_fusion.yaml",
        71: CONFIG_ROOT / "model_config_IQ_stage4_rfmamba_scan.yaml",
        72: CONFIG_ROOT / "model_config_IQ_stage4_radmamba_scan.yaml",
        73: CONFIG_ROOT / "model_config_IQ_stage4_symbol_dualpath.yaml",
        74: CONFIG_ROOT / "model_config_IQ_stage4_complex_mask_mc.yaml",
        75: CONFIG_ROOT / "model_config_IQ_stage4_noise_aware_mc.yaml",
        76: CONFIG_ROOT / "model_config_IQ_stage4_complex_adapter.yaml",
        77: CONFIG_ROOT / "model_config_IQ_stage4_cyclofresh.yaml",
        78: CONFIG_ROOT / "model_config_IQ_stage4_blind_cyclofresh.yaml",
        79: CONFIG_ROOT / "model_config_IQ_stage4_estimated_cyclofresh.yaml",
        80: CONFIG_ROOT / "model_config_IQ_stage4_cycliccorr.yaml",
        81: CONFIG_ROOT / "model_config_IQ_stage4_cycliccorr_leakcancel.yaml",
        82: CONFIG_ROOT / "model_config_IQ_stage4_multipeak_cyclofresh.yaml",
        83: CONFIG_ROOT / "model_config_IQ_stage4_sample_cyclofresh.yaml",
        84: CONFIG_ROOT / "model_config_IQ_stage4_cyclofresh_freqbias.yaml",
        85: CONFIG_ROOT / "model_config_IQ_stage4_blindstat_film.yaml",
        86: CONFIG_ROOT / "model_config_IQ_stage4_blindstat_input.yaml",
        87: CONFIG_ROOT / "model_config_IQ_stage4_demod_aware.yaml",
        88: CONFIG_ROOT / "model_config_IQ_stage4_feature_complex_mask.yaml",
        89: CONFIG_ROOT / "model_config_rf_bandscnet.yaml",
        90: CONFIG_ROOT / "model_config_complex_dpnet.yaml",
        91: CONFIG_ROOT / "model_config_complex_convtasnet.yaml",
        92: CONFIG_ROOT / "model_config_complex_sourceslot.yaml",
        93: CONFIG_ROOT / "model_config_complex_attractor.yaml",
        94: CONFIG_ROOT / "model_config_multires_stft_mask.yaml",
        95: CONFIG_ROOT / "model_config_IQ_stage4_knowledge_esd.yaml",
        96: CONFIG_ROOT / "model_config_IQ_stage4_blind_multirate_input.yaml",
        97: CONFIG_ROOT / "model_config_resunet1d_mamba_bottleneck.yaml",
        98: CONFIG_ROOT / "model_config_resunet1d_mamba_localglobal.yaml",
        99: CONFIG_ROOT / "model_config_resunet1d_mamba_dualgate.yaml",
        100: CONFIG_ROOT / "model_config_resunet1d_phaseeq.yaml",
        101: CONFIG_ROOT / "model_config_resunet1d_corrgate.yaml",
        102: CONFIG_ROOT / "model_config_resunet1d_pco.yaml",
        103: CONFIG_ROOT / "model_config_sepbamba_unet1d.yaml",
        104: CONFIG_ROOT / "model_config_resunet1d_gated_skip.yaml",
        105: CONFIG_ROOT / "model_config_resunet1d_wl_complex.yaml",
        106: CONFIG_ROOT / "model_config_resunet1d_tf_branch.yaml",
        107: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_attention.yaml",
        108: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_uct.yaml",
        109: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_dca.yaml",
        110: CONFIG_ROOT / "model_config_resunet1d_universal_prior.yaml",
        111: CONFIG_ROOT / "model_config_resunet1d_pulse_prior.yaml",
        112: CONFIG_ROOT / "model_config_resunet1d_timing_prior.yaml",
        113: CONFIG_ROOT / "model_config_resunet1d_pulse_timing_prior.yaml",
        114: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_lssg.yaml",
        115: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_lssg_channel.yaml",
        116: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_lssg_refined.yaml",
        117: CONFIG_ROOT / "model_config_resunet1d_bottleneck_sra_tcn.yaml",
        118: CONFIG_ROOT / "model_config_resunet1d_bottleneck_caspp.yaml",
        119: CONFIG_ROOT / "model_config_resunet1d_bottleneck_dccb.yaml",
        120: CONFIG_ROOT / "model_config_resunet1d_moe_prior.yaml",
        121: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_lssg_channel_ms.yaml",
        122: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_lssg_channel_context.yaml",
        123: CONFIG_ROOT / "model_config_resunet1d_strong_prior.yaml",
        124: CONFIG_ROOT / "model_config_resunet1d_bottleneck_dccb_lssg.yaml",
        125: CONFIG_ROOT / "model_config_resunet1d_bottleneck_dccb_lssg_partial_125.yaml",
        126: CONFIG_ROOT / "model_config_resunet1d_bottleneck_dccb_lssg_partial_126.yaml",
        139: CONFIG_ROOT / "model_config_resunet1d_bottleneck_dccb_full_mamba.yaml",
        140: CONFIG_ROOT / "model_config_resunet1d_bottleneck_dccb_unidirectional_mamba.yaml",
        141: CONFIG_ROOT / "model_config_iqumamba_dwt.yaml",
        142: CONFIG_ROOT / "model_config_siren.yaml",
        143: CONFIG_ROOT / "model_config_gridnet.yaml",
        144: CONFIG_ROOT / "model_config_bandsplit.yaml",
        145: CONFIG_ROOT / "model_config_iqu_mossformer.yaml",
        146: CONFIG_ROOT / "model_config_iqu_convnext.yaml",
        147: CONFIG_ROOT / "model_config_iqu_mscan.yaml",
        148: CONFIG_ROOT / "model_config_iqu_hybrid.yaml",
        149: CONFIG_ROOT / "model_config_stage149.yaml",
        150: CONFIG_ROOT / "model_config_stage150.yaml",
        151: CONFIG_ROOT / "model_config_stage151.yaml",
        152: CONFIG_ROOT / "model_config_stage152.yaml",
        153: CONFIG_ROOT / "model_config_stage153.yaml",
        154: CONFIG_ROOT / "model_config_resunet1d_skip_enhanced_lssg_dw.yaml",
        155: CONFIG_ROOT / "model_config_stage155_crossscale_lssg.yaml",
        156: CONFIG_ROOT / "model_config_stage156_sk_lssg.yaml",
        157: CONFIG_ROOT / "model_config_stage157_freq_lssg.yaml",
        158: CONFIG_ROOT / "model_config_stage158_focal_lssg.yaml",
        159: CONFIG_ROOT / "model_config_stage159_wavelet_dccb.yaml",
        160: CONFIG_ROOT / "model_config_stage160_complex_cyclo_dccb.yaml",
        8192: CONFIG_ROOT / "model_config_IQ_stage4_8192.yaml",
        16384: CONFIG_ROOT / "model_config_IQ_stage4_16384.yaml",
        32768: CONFIG_ROOT / "model_config_IQ_stage4_32768.yaml",
    }
    

    return str(config_mapping.get(stage, config_mapping[3]))


def apply_model_config_overrides(args, cfg):
    """Apply optional CLI overrides to model_config before cfg._load_enc_config()."""
    override_fields = [
        'mc_weight_mode',
        'mc_weight_power',
        'mc_min_weight',
        'mc_eps',
        'ric_num_steps',
        'ric_hidden_channels',
        'ric_kernel_size',
        'ric_dropout',
        'ric_step_init',
        'ric_return_intermediate',
        'ric_update_block_type',
        'ric_dilations',
        'ric_num_heads',
        'ric_attention_stride',
        'ric_ffn_multiplier',
        'admm_num_steps',
        'admm_hidden_channels',
        'admm_kernel_size',
        'admm_dropout',
        'admm_rho_init',
        'admm_dual_step_init',
        'admm_prox_step_init',
        'pgdu_num_steps',
        'pgdu_hidden_channels',
        'pgdu_kernel_size',
        'pgdu_dropout',
        'pgdu_step_size_init',
        'pgdu_prox_step_init',
        'gp_hidden_channels',
        'gp_kernel_size',
        'gp_max_gain_db',
        'gp_max_phase_deg',
        'gp_weight_mode',
        'gp_min_weight',
        'gp_correction_strength_init',
        'constellation_type',
        'constellation_order',
        'cgr_hidden_channels',
        'cgr_kernel_size',
        'cgr_temperature',
        'cgr_dropout',
        'cgr_gate_init',
        'cgr_residual_scale_init',
        'mamba_residual_scale_init',
        'bimamba_apply_stages',
        'bimamba_residual_scale_init',
        'bimamba_diff_scale_init',
        'bimamba_gate_logit_init',
        'bimamba_gate_token_scale_init',
        'bimamba_gate_eps',
        'bimamba_complex_diff_gate_init',
        'bimamba_complex_diff_stride',
        'bimamba_complex_diff_eps',
        'bimamba_boundary_tau_init',
        'bimamba_shrinkage_init',
        'bimamba_fusion_eps',
        'bimamba_local_kernel_size',
        'bimamba_local_gate_init',
        'cross_scale_query_stages',
        'cross_scale_global_stage',
        'cross_scale_kv_tokens',
        'cross_scale_num_heads',
        'cross_scale_dropout',
        'cross_scale_residual_scale_init',
        'cross_scale_evidence_hidden',
        'cross_scale_evidence_eps',
        'cross_scale_aligned_window_radius',
        'cross_scale_aligned_global_tokens',
        'cross_scale_coarse_kv_tokens',
        'cross_scale_fine_kv_tokens',
        'cross_scale_multires_gate_hidden',
        'cross_scale_bounded_max_scale',
        'cross_scale_bounded_initial_scale',
        'cross_scale_channel_gate_hidden',
        'cross_scale_variant',
        'training_only_deep_supervision',
        'shared_permutation_multiscale_weight',
        'shared_permutation_multiscale_weights',
        'shallow_skip_init',
        'shallow_skip_drop_probability',
        'local_kernel_size',
        'local_global_gate_hidden',
        'mamba_embed_stages',
        'mamba_embed_d_state',
        'mamba_embed_d_conv',
        'mamba_embed_expand',
        'mamba_embed_scale_init',
        'mamba_embed_local_kernel_size',
        'mamba_embed_gate_hidden',
        'pco_phase_channels',
        'pco_phase_kernel_size',
        'pco_phase_scale_init',
        'pco_corr_lags',
        'pco_corr_window',
        'pco_corr_scale_init',
        'pco_orth_scale_init',
        'pco_orth_eps',
        'rfscan_chunk_size',
        'rfscan_shift_size',
        'rfscan_freq_bands',
        'rfscan_gate_hidden',
        'rfscan_conv_kernel_size',
        'rfscan_residual_scale_init',
        'rfscan_condition_scale_init',
        'rfscan_stft_n_fft',
        'rfscan_stft_hop_length',
        'rfscan_stft_win_length',
        'rfscan_stft_freq_bins',
        'symbol_samples',
        'dual_path_chunk_symbols',
        'dual_path_hop_symbols',
        'dual_path_residual_scale_init',
        'mask_bound',
        'mask_logit_scale_init',
        'feature_mask_channels',
        'feature_mask_kernel_size',
        'feature_mask_bound',
        'feature_mask_logit_scale_init',
        'source_slot_hidden_channels',
        'source_slot_kernel_size',
        'source_slot_residual_scale_init',
        'noise_mc_source_weight',
        'noise_mc_noise_weight',
        'noise_head_hidden_channels',
        'noise_head_kernel_size',
        'noise_mc_eps',
        'low_snr_se_hidden_channels',
        'low_snr_se_kernel_size',
        'low_snr_se_scale_init',
        'low_snr_se_source_weight',
        'low_snr_se_noise_weight',
        'low_snr_se_eps',
        'low_snr_cond_hidden_channels',
        'low_snr_cond_kernel_size',
        'low_snr_cond_gate_hidden',
        'low_snr_cond_scale_init',
        'low_snr_cond_min_freq',
        'low_snr_cond_max_freq',
        'low_snr_cond_source_weight',
        'low_snr_cond_noise_weight',
        'low_snr_cond_eps',
        'wiener_hidden_channels',
        'wiener_kernel_size',
        'wiener_signal_bias_init',
        'wiener_noise_bias_init',
        'wiener_log_power_clip',
        'wiener_source_weight',
        'wiener_noise_weight',
        'wiener_eps',
        'cross_snr_probability',
        'cross_snr_high_db',
        'cross_snr_low_start_db',
        'cross_snr_low_middle_db',
        'cross_snr_low_final_db',
        'cross_snr_first_fraction',
        'cross_snr_second_fraction',
        'cross_snr_pair_weight',
        'cross_snr_consistency_weight',
        'cross_snr_consistency_beta',
        'cross_snr_eps',
        'cross_snr_shared_permutation',
        'cross_snr_ema_decay',
        'sync_snr_aux_weight',
        'sync_snr_aux_min_db',
        'sync_snr_aux_max_db',
        'sync_snr_aux_beta',
        'sync_cross_snr_consistency_weight',
        'sync_cross_snr_consistency_beta',
        'sync_cfo_scale',
        'sync_phase_drift_scale',
        'phase_equiv_probability',
        'phase_equiv_supervised_weight',
        'phase_equiv_consistency_weight',
        'phase_equiv_max_degrees',
        'phase_equiv_beta',
        'phase_equiv_eps',
        'rf_equiv_probability',
        'rf_equiv_supervised_weight',
        'rf_equiv_consistency_weight',
        'rf_equiv_max_phase_degrees',
        'rf_equiv_max_cfo_cycles_per_sample',
        'rf_equiv_max_gain_db',
        'rf_equiv_max_shift_samples',
        'rf_equiv_conjugate_probability',
        'rf_equiv_source_mode',
        'rf_equiv_beta',
        'rf_equiv_eps',
        'receiver_symbol_weight',
        'receiver_symbol_probability',
        'receiver_symbol_batch_fraction',
        'receiver_sps_candidates',
        'receiver_rrc_rolloff',
        'receiver_rrc_span',
        'receiver_constellation_weight',
        'receiver_softmin_temperature',
        'receiver_symbol_beta',
        'receiver_symbol_eps',
        'shared_permutation_multiscale_enable',
        'shared_permutation_multiscale_weight',
        'shared_permutation_multiscale_weights',
        'evidence_moe_hidden_channels',
        'evidence_moe_max_delta',
        'evidence_moe_identity_bias',
        'evidence_moe_router_temperature',
        'evidence_moe_route_hard_eval',
        'evidence_moe_lag_bank',
        'evidence_moe_return_route_aux',
        'evidence_moe_route_supervision_enable',
        'evidence_moe_route_loss_weight',
        'evidence_moe_route_target_temperature',
        'stage255_snr_aux_weight',
        'stage255_snr_aux_min_db',
        'stage255_snr_aux_max_db',
        'stage255_snr_curriculum_enable',
        'stage255_snr_curriculum_start_db',
        'stage255_snr_curriculum_end_db',
        'stage255_snr_curriculum_fraction',
        'stage255_expert_pretrain_epochs',
        'stage255_router_warmup_epochs',
        'stage255_router_joint_lr_scale',
        'fusion_route_candidate_probability',
        'noise_prior_hidden',
        'noise_prior_embedding',
        'noise_prior_patch_size',
        'noise_prior_patch_stride',
        'noise_contrastive_prior_enable',
        'noise_contrastive_prior_weight',
        'noise_contrastive_prior_patch_size',
        'noise_contrastive_prior_patch_stride',
        'noise_contrastive_prior_temperature',
        'noise_contrastive_prior_residual_weight',
        'noise_contrastive_prior_gate_floor',
        'sync_hidden',
        'sync_kernel_size',
        'sync_scale_init',
        'sync_lags',
        'sync_eps',
        'confidence_soft_pit_enable',
        'confidence_soft_pit_temperature_min',
        'confidence_soft_pit_temperature_max',
        'confidence_soft_pit_snr_low_db',
        'confidence_soft_pit_snr_high_db',
        'confidence_soft_pit_anneal_power',
        'cumulant_prior_enable',
        'cumulant_prior_weight',
        'cumulant_prior_probability',
        'cumulant_prior_batch_fraction',
        'cumulant_prior_window_sizes',
        'cumulant_prior_self_weight',
        'cumulant_prior_cross_weight',
        'cumulant_prior_confidence_floor',
        'cumulant_prior_beta',
        'cumulant_prior_eps',
        'cumulant_residual_enable',
        'cumulant_residual_weight',
        'cumulant_residual_cross_weight',
        'cumulant_residual_beta',
        'fsq_token_ce_enable',
        'fsq_token_ce_weight',
        'fsq_token_ce_temperature',
        'fsq_token_ce_warmup_steps',
        'fsq_tokenizer_checkpoint',
        'asg_patch_size',
        'asg_stride',
        'asg_num_bands',
        'asg_gate_hidden',
        'asg_scale_init',
        'asg_apply_stages',
        'asg_eps',
        'complex_adapter_hidden_channels',
        'complex_adapter_kernel_size',
        'complex_adapter_scale_init',
        'cyclofresh_sps',
        'cyclofresh_alphas',
        'cyclofresh_hidden_channels',
        'cyclofresh_kernel_size',
        'cyclofresh_scale_init',
        'cyclofresh_gate_hidden',
        'blind_cyclofresh_freqs',
        'blind_cyclofresh_max_delta',
        'blind_cyclofresh_hidden_channels',
        'blind_cyclofresh_kernel_size',
        'blind_cyclofresh_scale_init',
        'blind_cyclofresh_gate_hidden',
        'estimated_cyclofresh_min_freq',
        'estimated_cyclofresh_max_freq',
        'estimated_cyclofresh_default_freq',
        'estimated_cyclofresh_momentum',
        'estimated_cyclofresh_hidden_channels',
        'estimated_cyclofresh_kernel_size',
        'estimated_cyclofresh_scale_init',
        'estimated_cyclofresh_gate_hidden',
        'multipeak_cyclofresh_min_freq',
        'multipeak_cyclofresh_max_freq',
        'multipeak_cyclofresh_default_freq',
        'multipeak_cyclofresh_momentum',
        'multipeak_cyclofresh_num_peaks',
        'multipeak_cyclofresh_guard_bins',
        'multipeak_cyclofresh_hidden_channels',
        'multipeak_cyclofresh_kernel_size',
        'multipeak_cyclofresh_scale_init',
        'multipeak_cyclofresh_gate_hidden',
        'multipeak_cyclofresh_reliability_floor',
        'sample_cyclofresh_min_freq',
        'sample_cyclofresh_max_freq',
        'sample_cyclofresh_default_freq',
        'sample_cyclofresh_num_peaks',
        'sample_cyclofresh_guard_bins',
        'sample_cyclofresh_hidden_channels',
        'sample_cyclofresh_kernel_size',
        'sample_cyclofresh_scale_init',
        'sample_cyclofresh_gate_hidden',
        'sample_cyclofresh_reliability_floor',
        'multihyp_cyclic_freqs',
        'multihyp_cyclic_hidden_channels',
        'multihyp_cyclic_kernel_size',
        'multihyp_cyclic_scale_init',
        'multihyp_cyclic_gate_hidden',
        'multihyp_cyclic_temperature',
        'multihyp_cyclic_null_logit_init',
        'multihyp_cyclic_local_bins',
        'multihyp_cyclic_eps',
        'freqbias_hidden_channels',
        'freqbias_kernel_size',
        'freqbias_lowpass_kernel_size',
        'freqbias_scale_init',
        'freqbias_gate_hidden',
        'cycliccorr_min_freq',
        'cycliccorr_max_freq',
        'cycliccorr_default_freq',
        'cycliccorr_momentum',
        'cycliccorr_lags',
        'cycliccorr_hidden_channels',
        'cycliccorr_kernel_size',
        'cycliccorr_scale_init',
        'cycliccorr_gate_hidden',
        'leakcancel_lags',
        'leakcancel_hidden',
        'leakcancel_scale_init',
        'leakcancel_mc_scale_init',
        'leakcancel_mc_weight_mode',
        'leakcancel_mode',
        'leakcancel_coeff_limit',
        'blindstat_hidden',
        'blindstat_kernel_size',
        'blindstat_scale_init',
        'blindstat_cyclic_min_freq',
        'blindstat_cyclic_max_freq',
        'blindstat_cyclic_default_freq',
        'multirate_hidden_channels',
        'multirate_kernel_sizes',
        'multirate_dilations',
        'multirate_scale_init',
        'psk_prior_hidden_channels',
        'psk_prior_harmonics',
        'psk_prior_kernel_size',
        'psk_prior_scale_init',
        'psk_prior_reliability_floor',
        'qam_prior_hidden_channels',
        'qam_prior_axis_level_bank',
        'qam_prior_temperature',
        'qam_prior_kernel_size',
        'qam_prior_scale_init',
        'qam_prior_reliability_floor',
        'apsk_prior_hidden_channels',
        'apsk_prior_ring_radii',
        'apsk_prior_temperature',
        'apsk_prior_kernel_size',
        'apsk_prior_scale_init',
        'apsk_prior_reliability_floor',
        'topology_aux_weight',
        'topology_aux_axis_weight',
        'topology_aux_amp_weight',
        'topology_aux_phase_weight',
        'topology_aux_kurtosis_weight',
        'feature_topology_hidden_channels',
        'feature_topology_kernel_size',
        'feature_topology_scale_init',
        'feature_topology_apply_stages',
        'sep_constraint_weight',
        'sep_constraint_mix_weight',
        'sep_constraint_corr_weight',
        'sep_constraint_energy_weight',
        'cyclic_wiener_hidden_channels',
        'cyclic_wiener_kernel_size',
        'cyclic_wiener_min_freq',
        'cyclic_wiener_max_freq',
        'cyclic_wiener_default_freq',
        'cyclic_wiener_num_harmonics',
        'cyclic_wiener_scale_init',
        'cyclic_wiener_projection_strength',
    ]
    for field in override_fields:
        value = getattr(args, field, None)
        if value is not None:
            cfg.model_config[field] = value

    loss_names = [getattr(args, 'loss_fun', '')]
    if getattr(args, 'two_phase_loss', False):
        loss_names.extend([getattr(args, 'loss_phase1', ''), getattr(args, 'loss_phase2', '')])
    if any('URIC-DS' in str(loss_name) for loss_name in loss_names):
        cfg.model_config['ric_return_intermediate'] = True

    bool_override_pairs = [
        ('mc_detach_weights', 'mc_keep_weight_grads', 'mc_detach_weights'),
        ('mc_project_deep_supervision', 'mc_project_final_only', 'mc_project_deep_supervision'),
        ('mc_apply_train', 'mc_skip_train_projection', 'mc_apply_train'),
        ('mc_apply_eval', 'mc_skip_eval_projection', 'mc_apply_eval'),
        ('mask_sum_constraint', 'mask_no_sum_constraint', 'mask_sum_constraint'),
        ('mask_apply_projection', 'mask_skip_projection', 'mask_apply_projection'),
        ('mask_project_deep_supervision', 'mask_project_final_only', 'mask_project_deep_supervision'),
        ('feature_mask_sum_constraint', 'feature_mask_no_sum_constraint', 'feature_mask_sum_constraint'),
        ('feature_mask_apply_projection', 'feature_mask_skip_projection', 'feature_mask_apply_projection'),
        ('feature_mask_project_deep_supervision', 'feature_mask_project_final_only', 'feature_mask_project_deep_supervision'),
        ('feature_mask_identity_init', 'feature_mask_no_identity_init', 'feature_mask_identity_init'),
        ('source_slot_zero_init', 'source_slot_no_zero_init', 'source_slot_zero_init'),
        ('source_slot_refine_deep_supervision', 'source_slot_final_only', 'source_slot_refine_deep_supervision'),
        ('source_slot_apply_train', 'source_slot_skip_train', 'source_slot_apply_train'),
        ('source_slot_apply_eval', 'source_slot_skip_eval', 'source_slot_apply_eval'),
        ('noise_mc_apply_projection', 'noise_mc_skip_projection', 'noise_mc_apply_projection'),
        ('noise_mc_project_during_train', 'noise_mc_skip_train_projection', 'noise_mc_project_during_train'),
        ('noise_mc_project_during_eval', 'noise_mc_skip_eval_projection', 'noise_mc_project_during_eval'),
        ('noise_head_zero_init', 'noise_head_no_zero_init', 'noise_head_zero_init'),
        ('low_snr_se_zero_init', 'low_snr_se_no_zero_init', 'low_snr_se_zero_init'),
        ('low_snr_se_use_projection', 'low_snr_se_skip_projection', 'low_snr_se_use_projection'),
        ('low_snr_se_project_during_train', 'low_snr_se_skip_train_projection', 'low_snr_se_project_during_train'),
        ('low_snr_se_project_during_eval', 'low_snr_se_skip_eval_projection', 'low_snr_se_project_during_eval'),
        ('cross_scale_evidence_gate', 'cross_scale_no_evidence_gate', 'cross_scale_evidence_gate'),
        ('low_snr_se_return_aux', 'low_snr_se_no_aux', 'low_snr_se_return_aux'),
        ('low_snr_cond_zero_init', 'low_snr_cond_no_zero_init', 'low_snr_cond_zero_init'),
        ('low_snr_cond_use_projection', 'low_snr_cond_skip_projection', 'low_snr_cond_use_projection'),
        ('low_snr_cond_project_during_train', 'low_snr_cond_skip_train_projection', 'low_snr_cond_project_during_train'),
        ('low_snr_cond_project_during_eval', 'low_snr_cond_skip_eval_projection', 'low_snr_cond_project_during_eval'),
        ('low_snr_cond_return_aux', 'low_snr_cond_no_aux', 'low_snr_cond_return_aux'),
        ('wiener_use_projection', 'wiener_skip_projection', 'wiener_use_projection'),
        ('wiener_project_during_train', 'wiener_skip_train_projection', 'wiener_project_during_train'),
        ('wiener_project_during_eval', 'wiener_skip_eval_projection', 'wiener_project_during_eval'),
        ('wiener_return_aux', 'wiener_no_aux', 'wiener_return_aux'),
        ('cross_snr_enable', 'cross_snr_disable', 'cross_snr_enable'),
        ('rf_equiv_enable', 'rf_equiv_disable', 'rf_equiv_enable'),
        (
            'cross_snr_ema_teacher_enable',
            'cross_snr_ema_teacher_disable',
            'cross_snr_ema_teacher_enable',
        ),
        (
            'evidence_moe_route_supervision_enable',
            'evidence_moe_route_supervision_disable',
            'evidence_moe_route_supervision_enable',
        ),
        (
            'stage255_snr_curriculum_enable',
            'stage255_snr_curriculum_disable',
            'stage255_snr_curriculum_enable',
        ),
        ('stage255_trust_enable', 'stage255_trust_disable', 'fusion_trust_penalty_enable'),
        (
            'stage255_condition_enable',
            'stage255_condition_disable',
            'fusion_condition_routing_enable',
        ),
        (
            'stage255_counterfactual_enable',
            'stage255_counterfactual_disable',
            'fusion_counterfactual_enable',
        ),
        ('asg_zero_init', 'asg_no_zero_init', 'asg_zero_init'),
        ('complex_adapter_use_input', 'complex_adapter_no_input', 'complex_adapter_use_input'),
        ('complex_adapter_use_output', 'complex_adapter_no_output', 'complex_adapter_use_output'),
        ('complex_adapter_zero_init', 'complex_adapter_no_zero_init', 'complex_adapter_zero_init'),
        ('cyclofresh_zero_init', 'cyclofresh_no_zero_init', 'cyclofresh_zero_init'),
        ('blind_cyclofresh_zero_init', 'blind_cyclofresh_no_zero_init', 'blind_cyclofresh_zero_init'),
        ('estimated_cyclofresh_zero_init', 'estimated_cyclofresh_no_zero_init', 'estimated_cyclofresh_zero_init'),
        ('multipeak_cyclofresh_zero_init', 'multipeak_cyclofresh_no_zero_init', 'multipeak_cyclofresh_zero_init'),
        ('sample_cyclofresh_zero_init', 'sample_cyclofresh_no_zero_init', 'sample_cyclofresh_zero_init'),
        ('multihyp_cyclic_zero_init', 'multihyp_cyclic_no_zero_init', 'multihyp_cyclic_zero_init'),
        ('multihyp_cyclic_return_aux', 'multihyp_cyclic_no_aux', 'multihyp_cyclic_return_aux'),
        ('freqbias_zero_init', 'freqbias_no_zero_init', 'freqbias_zero_init'),
        ('cycliccorr_zero_init', 'cycliccorr_no_zero_init', 'cycliccorr_zero_init'),
        ('leakcancel_zero_init', 'leakcancel_no_zero_init', 'leakcancel_zero_init'),
        ('blindstat_zero_init', 'blindstat_no_zero_init', 'blindstat_zero_init'),
        ('multirate_zero_init', 'multirate_no_zero_init', 'multirate_zero_init'),
        ('psk_prior_zero_init', 'psk_prior_no_zero_init', 'psk_prior_zero_init'),
        ('qam_prior_zero_init', 'qam_prior_no_zero_init', 'qam_prior_zero_init'),
        ('apsk_prior_zero_init', 'apsk_prior_no_zero_init', 'apsk_prior_zero_init'),
        ('feature_topology_zero_init', 'feature_topology_no_zero_init', 'feature_topology_zero_init'),
        ('cyclic_wiener_zero_init', 'cyclic_wiener_no_zero_init', 'cyclic_wiener_zero_init'),
        ('cgr_use_mixture_residual', 'cgr_no_mixture_residual', 'cgr_use_mixture_residual'),
        ('cgr_zero_init', 'cgr_no_zero_init', 'cgr_zero_init'),
        ('cgr_refine_deep_supervision', 'cgr_final_only', 'cgr_refine_deep_supervision'),
        ('cgr_apply_train', 'cgr_skip_train', 'cgr_apply_train'),
        ('cgr_apply_eval', 'cgr_skip_eval', 'cgr_apply_eval'),
    ]
    for enable_flag, disable_flag, cfg_field in bool_override_pairs:
        enable = bool(getattr(args, enable_flag, False))
        disable = bool(getattr(args, disable_flag, False))
        if enable and disable:
            raise ValueError(
                f"Conflicting CLI overrides: --{enable_flag} and --{disable_flag} cannot be used together."
            )
        if enable:
            cfg.model_config[cfg_field] = True
        elif disable:
            cfg.model_config[cfg_field] = False

    if getattr(args, 'ric_untied_steps', False):
        cfg.model_config['ric_tied_steps'] = False

    if getattr(args, 'admm_untied_steps', False):
        cfg.model_config['admm_tied_steps'] = False

    if getattr(args, 'pgdu_untied_steps', False):
        cfg.model_config['pgdu_tied_steps'] = False

    gp_apply_train = getattr(args, 'gp_apply_train', False)
    gp_skip_train = getattr(args, 'gp_skip_train', False)
    if gp_apply_train and gp_skip_train:
        raise ValueError("--gp_apply_train and --gp_skip_train cannot be used together")
    if gp_apply_train:
        cfg.model_config['gp_apply_train'] = True
    elif gp_skip_train:
        cfg.model_config['gp_apply_train'] = False

    gp_apply_eval = getattr(args, 'gp_apply_eval', False)
    gp_skip_eval = getattr(args, 'gp_skip_eval', False)
    if gp_apply_eval and gp_skip_eval:
        raise ValueError("--gp_apply_eval and --gp_skip_eval cannot be used together")
    if gp_apply_eval:
        cfg.model_config['gp_apply_eval'] = True
    elif gp_skip_eval:
        cfg.model_config['gp_apply_eval'] = False


def apply_loss_args_from_model_config(args, cfg):
    """Populate stage-owned auxiliary-loss args from the loaded model config."""
    fields = (
        'topology_aux_weight',
        'topology_aux_axis_weight',
        'topology_aux_amp_weight',
        'topology_aux_phase_weight',
        'topology_aux_kurtosis_weight',
        'sep_constraint_weight',
        'sep_constraint_mix_weight',
        'sep_constraint_corr_weight',
        'sep_constraint_energy_weight',
        'cross_snr_enable',
        'cross_snr_probability',
        'cross_snr_high_db',
        'cross_snr_low_start_db',
        'cross_snr_low_middle_db',
        'cross_snr_low_final_db',
        'cross_snr_first_fraction',
        'cross_snr_second_fraction',
        'cross_snr_pair_weight',
        'cross_snr_consistency_weight',
        'cross_snr_consistency_beta',
        'cross_snr_eps',
        'cross_snr_shared_permutation',
        'cross_snr_ema_teacher_enable',
        'cross_snr_ema_decay',
        'cross_snr_teacher_mode',
        'cross_snr_teacher_checkpoint',
        'cross_snr_teacher_view',
        'cross_snr_pair_mode',
        'cross_snr_feature_consistency_weight',
        'cross_snr_feature_consistency_beta',
        'cross_snr_curriculum_ranges',
        'cross_snr_curriculum_boundaries',
        'sync_snr_aux_weight',
        'sync_snr_aux_min_db',
        'sync_snr_aux_max_db',
        'sync_snr_aux_beta',
        'sync_cross_snr_consistency_weight',
        'sync_cross_snr_consistency_beta',
        'sync_cfo_scale',
        'sync_phase_drift_scale',
        'sync_metadata_enable',
        'sync_physical_require_metadata',
        'sync_physical_supervision_weight',
        'sync_physical_cfo_weight',
        'sync_physical_phase_weight',
        'sync_physical_timing_weight',
        'sync_physical_sps_weight',
        'sync_physical_drift_weight',
        'sync_physical_beta',
        'training_snr_floor_db',
        'validation_snr_floor_db',
        'phase_equiv_enable',
        'phase_equiv_probability',
        'phase_equiv_supervised_weight',
        'phase_equiv_consistency_weight',
        'phase_equiv_max_degrees',
        'phase_equiv_beta',
        'phase_equiv_eps',
        'rf_equiv_enable',
        'rf_equiv_probability',
        'rf_equiv_supervised_weight',
        'rf_equiv_consistency_weight',
        'rf_equiv_max_phase_degrees',
        'rf_equiv_max_cfo_cycles_per_sample',
        'rf_equiv_max_gain_db',
        'rf_equiv_max_shift_samples',
        'rf_equiv_conjugate_probability',
        'rf_equiv_source_mode',
        'rf_equiv_beta',
        'rf_equiv_eps',
        'latent_mask_residual_weight',
        'latent_mask_mixture_weight',
        'latent_mask_residual_beta',
        'confidence_soft_pit_enable',
        'confidence_soft_pit_temperature_min',
        'confidence_soft_pit_temperature_max',
        'confidence_soft_pit_snr_low_db',
        'confidence_soft_pit_snr_high_db',
        'confidence_soft_pit_anneal_power',
        'cumulant_prior_enable',
        'cumulant_prior_weight',
        'cumulant_prior_probability',
        'cumulant_prior_batch_fraction',
        'cumulant_prior_window_sizes',
        'cumulant_prior_self_weight',
        'cumulant_prior_cross_weight',
        'cumulant_prior_confidence_floor',
        'cumulant_prior_beta',
        'cumulant_prior_eps',
        'cumulant_residual_enable',
        'cumulant_residual_weight',
        'cumulant_residual_cross_weight',
        'cumulant_residual_beta',
        'fsq_token_ce_enable',
        'fsq_token_ce_weight',
        'fsq_token_ce_temperature',
        'fsq_token_ce_warmup_steps',
        'fsq_tokenizer_checkpoint',
        'receiver_symbol_weight',
        'receiver_sps_candidates',
        'receiver_rrc_rolloff',
        'receiver_rrc_span',
        'receiver_constellation_weight',
        'receiver_softmin_temperature',
        'receiver_symbol_beta',
        'receiver_symbol_eps',
        'shared_permutation_multiscale_enable',
        'shared_permutation_multiscale_weight',
        'shared_permutation_multiscale_weights',
        'evidence_moe_route_supervision_enable',
        'evidence_moe_route_loss_weight',
        'evidence_moe_route_target_temperature',
        'stage255_snr_aux_weight',
        'stage255_snr_aux_min_db',
        'stage255_snr_aux_max_db',
        'stage255_snr_curriculum_enable',
        'stage255_snr_curriculum_start_db',
        'stage255_snr_curriculum_end_db',
        'stage255_snr_curriculum_fraction',
        'stage255_expert_pretrain_epochs',
        'stage255_router_warmup_epochs',
        'stage255_router_joint_lr_scale',
        'noise_contrastive_prior_enable',
        'noise_contrastive_prior_weight',
        'noise_contrastive_prior_patch_size',
        'noise_contrastive_prior_patch_stride',
        'noise_contrastive_prior_temperature',
        'noise_contrastive_prior_residual_weight',
        'noise_contrastive_prior_gate_floor',
        'qam_turbo_joint_loss_enable',
        'qam_turbo_mixture_loss_weight',
        'qam_turbo_qam_loss_weight',
        'qam_turbo_independence_loss_weight',
        'qam_turbo_intermediate_loss_weight',
        'qam_turbo_route_entropy_weight',
    )
    for field in fields:
        if getattr(args, field, None) is None and hasattr(cfg, field):
            setattr(args, field, getattr(cfg, field))


def resolve_train_aug_config(cfg, args=None, num_epochs=None):
    """Resolve optional lightweight train-time RF augmentation from config."""
    enabled = bool(getattr(cfg, 'train_aug_enable', False))
    profile = getattr(args, 'train_aug_profile', 'config') if args is not None else 'config'
    if profile == 'gain_sir_remix':
        enabled = True
    if args is not None and getattr(args, 'disable_train_aug', False):
        enabled = False
    if args is not None and getattr(args, 'enable_train_aug', False):
        enabled = True
    if not enabled:
        return None
    config = {
        "enabled": True,
        "source_phase_jitter_deg": float(getattr(cfg, 'train_aug_source_phase_jitter_deg', 12.0)),
        "source_gain_jitter_db": float(getattr(cfg, 'train_aug_source_gain_jitter_db', 1.0)),
        "max_common_time_shift": int(getattr(cfg, 'train_aug_max_common_time_shift', 8)),
        "global_phase_rotation": bool(getattr(cfg, 'train_aug_global_phase_rotation', True)),
    }
    config["mix_enable"] = bool(getattr(cfg, 'train_mix_enable', False))
    config["mix_prob"] = float(getattr(cfg, 'train_mix_prob', 0.0))
    config["mix_sir_min_db"] = float(getattr(cfg, 'train_mix_sir_min_db', -3.0))
    config["mix_sir_max_db"] = float(getattr(cfg, 'train_mix_sir_max_db', 3.0))
    config["mix_cross_sample"] = bool(getattr(cfg, 'train_mix_cross_sample', False))
    warmup_ratio = float(getattr(cfg, 'train_aug_warmup_ratio', 0.0))
    warmup_epochs_override = getattr(cfg, 'train_aug_warmup_epochs', None)

    if profile == 'gain_sir_remix':
        config.update({
            "source_phase_jitter_deg": 0.0,
            "source_gain_jitter_db": 0.0,
            "max_common_time_shift": 0,
            "global_phase_rotation": False,
        })
        config["mix_enable"] = True
        if config["mix_prob"] <= 0.0:
            config["mix_prob"] = 0.5
        warmup_ratio = 0.1

    if args is not None:
        if getattr(args, 'train_mix_enable', False):
            config["mix_enable"] = True
        if getattr(args, 'train_mix_disable', False):
            config["mix_enable"] = False
        if getattr(args, 'train_mix_prob', None) is not None:
            config["mix_prob"] = float(args.train_mix_prob)
        if getattr(args, 'train_mix_sir_min_db', None) is not None:
            config["mix_sir_min_db"] = float(args.train_mix_sir_min_db)
        if getattr(args, 'train_mix_sir_max_db', None) is not None:
            config["mix_sir_max_db"] = float(args.train_mix_sir_max_db)
        if getattr(args, 'train_mix_cross_sample', False):
            config["mix_cross_sample"] = True
        if getattr(args, 'train_aug_warmup_ratio', None) is not None:
            warmup_ratio = float(args.train_aug_warmup_ratio)
            warmup_epochs_override = None
        if getattr(args, 'train_aug_warmup_epochs', None) is not None:
            warmup_epochs_override = int(args.train_aug_warmup_epochs)

    if config["mix_sir_min_db"] > config["mix_sir_max_db"]:
        config["mix_sir_min_db"], config["mix_sir_max_db"] = (
            config["mix_sir_max_db"],
            config["mix_sir_min_db"],
        )
    config["mix_prob"] = min(max(float(config["mix_prob"]), 0.0), 1.0)
    if warmup_epochs_override is not None:
        warmup_epochs = int(max(0, warmup_epochs_override))
    else:
        total_epochs = int(num_epochs if num_epochs is not None else getattr(args, 'num_epochs', 0))
        warmup_epochs = int(max(0, round(total_epochs * min(max(warmup_ratio, 0.0), 1.0))))
    config["train_aug_profile"] = profile
    config["train_aug_warmup_ratio"] = float(min(max(warmup_ratio, 0.0), 1.0))
    config["train_aug_warmup_epochs"] = warmup_epochs

    return config


def setup_data_parameters(data_choice, logger):
    """Setup data-related parameters"""
    data_configs = {
        'debug_random': {'input_size': 1024, 'num_points': 256, 'input_channels': 2},
        '2016': {'input_size': 128, 'num_points': 128, 'input_channels': 2},
        '2018': {'input_size': 1024, 'num_points': 256, 'input_channels': 2},
        'TorchSig': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_M': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-A': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-B': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-C': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-D': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-E': {'input_size': 8192, 'num_points': 256, 'input_channels': 2},
        '8PSK-F': {'input_size': 16384, 'num_points': 256, 'input_channels': 2},
        '8PSK-G': {'input_size': 32768, 'num_points': 256, 'input_channels': 2},
        '8PSK-H': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-I': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-J': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-K': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK-L': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QPSK+16APSK-A': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QPSK+16APSK-B': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QAM-A': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QAM-B': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QAM-C': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QAM-D': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QAM-E': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_Burst': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_Burst_NS': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_8192': {'input_size': 8192, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_16384': {'input_size': 16384, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_32768': {'input_size': 32768, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_NS': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_8192_NS': {'input_size': 8192, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_16384_NS': {'input_size': 16384, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_32768_NS': {'input_size': 32768, 'num_points': 256, 'input_channels': 2},
        'QPSK_16APSK': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        'QPSK_16APSK_NS': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_Rs': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_Rs_NS': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '16QAM_64QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '16QAM_128QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '64QAM_64QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '64QAM_128QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '16QAM_64QAM_128QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
    }
    
    if data_choice not in data_configs:
        raise ValueError(f"Unsupported data choice: {data_choice}")
    
    config = data_configs[data_choice]
    logger.info(f"Dataset: {data_choice}, DataLength: {config['input_size']}")
    
    return config['input_size'], config['num_points'], config['input_channels']


def resolve_pretrain_data_choices(args):
    """Return normalized joint-pretraining choices, or an empty list."""
    choices = [
        normalize_data_choice(choice)
        for choice in (getattr(args, 'pretrain_data_choices', None) or [])
    ]
    if choices and len(choices) < 2:
        raise ValueError("--pretrain_data_choices requires at least two datasets")
    return choices


def resolve_run_data_parameters(args, logger):
    """Resolve the model waveform size for single- or multi-dataset training."""
    input_size, num_points, input_channels = setup_data_parameters(args.data_choice, logger)
    choices = resolve_pretrain_data_choices(args)
    if not choices:
        return input_size, num_points, input_channels

    requested_length = getattr(args, 'pretrain_input_size', None)
    target_length = input_size if requested_length is None else int(requested_length)
    if target_length <= 0:
        raise ValueError(f"--pretrain_input_size must be positive, got {target_length}")
    logger.info(
        f"Joint pretraining: datasets={choices}, target_length={target_length}, "
        f"sampling={args.pretrain_sampling}, length_policy={args.pretrain_length_policy}"
    )
    return target_length, num_points, input_channels


def create_run_data_loaders(
    args, batch_size, num_sources, pin_memory, train_aug_config, seed, target_length,
):
    """Dispatch to the single-domain or joint-pretraining data pipeline."""
    common = dict(
        num_sources=num_sources,
        matlab_data_root=args.synthetic_root,
        public_data_root=args.public_root,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        split_strategy=args.split_strategy,
        train_aug_config=train_aug_config,
        return_sync_metadata=bool(getattr(args, 'sync_metadata_enable', False)),
        train_snr_floor_db=getattr(args, 'training_snr_floor_db', None),
        val_snr_floor_db=getattr(args, 'validation_snr_floor_db', None),
        seed=42 if seed is None else int(seed),
    )
    choices = resolve_pretrain_data_choices(args)
    if not choices:
        return create_data_loaders(
            batch_size,
            data_choice=args.data_choice,
            **common,
        )
    if args.mode not in {'train', 'test_data'}:
        raise ValueError("--pretrain_data_choices is only supported with --mode train/test_data")
    return create_multidataset_data_loaders(
        batch_size=batch_size,
        data_choices=choices,
        target_length=target_length,
        sampling=args.pretrain_sampling,
        dataset_weights=args.pretrain_dataset_weights,
        length_policy=args.pretrain_length_policy,
        **common,
    )


def resolve_num_sources(args):
    """Validate the fixed source cardinality for one training/test run."""
    num_sources = int(getattr(args, 'num_sources', 2))
    if num_sources not in (2, 3):
        raise ValueError(f"--num_sources must be 2 or 3, got {num_sources}")
    source_names = list(getattr(args, 'source_names', []) or [])
    if len(source_names) != num_sources:
        raise ValueError(
            f"--source_names must contain exactly --num_sources entries; "
            f"got {len(source_names)} names for {num_sources} sources"
        )
    return num_sources


def apply_source_count_to_model_config(cfg, num_sources):
    """Set the separator output width without changing the YAML on disk."""
    num_sources = int(num_sources)
    if num_sources not in (2, 3):
        raise ValueError(f"num_sources must be 2 or 3, got {num_sources}")
    cfg.model_config['num_classes'] = 2 * num_sources
    # Spectral/separation baselines keep their source count in nested model
    # sections (for example ``tfgridnet_config.n_srcs``). Keep those settings
    # synchronized with the universal 2*K output contract.
    def _override_nested_source_counts(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {'n_srcs', 'num_sources'}:
                    value[key] = num_sources
                else:
                    _override_nested_source_counts(nested)
        elif isinstance(value, list):
            for nested in value:
                _override_nested_source_counts(nested)

    _override_nested_source_counts(cfg.model_config)
    # Keep an already-materialized config object coherent for callers/tests.
    cfg.num_classes = 2 * num_sources
    cfg.num_sources = num_sources


def create_model(args, cfg, input_size, device, logger):
    """Create and return model"""
    
    apply_source_count_to_model_config(cfg, resolve_num_sources(args))
    model = Create_Mamba_model(cfg, logger, input_size_=input_size, device_override=device)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device
    logger.info(f"Model parameter device: {model_device}")
    if getattr(model_device, "type", str(model_device)) != device.type:
        logger.warning(f"Model was created on {model_device}, expected {device}; moving model to {device}")
        model = model.to(device)
    
    return model


def calculate_model_complexity(model, batch_size, input_channels, input_size, logger):
    """Calculate model complexity"""
    # calflops pulls in timm/torchvision in some environments (e.g. Kaggle),
    # and may crash due to torch/torchvision binary mismatches. Treat FLOPs as optional.
    params = sum(p.numel() for p in model.parameters())
    try:
        from calflops import calculate_flops  # local import (optional dependency)
        input_tuple = (batch_size, input_channels, input_size)
        flops, macs, _ = calculate_flops(model, input_tuple, print_detailed=False)
        logger.info(f"InputSize: {input_size}, FLOPs: {flops}, MACs: {macs}, Params: {params}")
        return flops, macs, params
    except Exception as e:
        logger.warning(f"Skipping FLOPs/MACs calculation (calflops/torchvision issue): {e}")
        logger.info(f"InputSize: {input_size}, Params: {params}")
        return None, None, params


def setup_training_components(args, model, logger):
    """Initialize criterion, optimizer, scheduler, etc."""

    def _float_arg(name, default=0.0):
        value = getattr(args, name, None)
        return float(default if value is None else value)

    def _num_sources_from_args():
        source_names = getattr(args, 'source_names', None)
        return len(source_names) if source_names else None

    def _forward_criterion_attrs(dst, src):
        for attr in ('needs_mixture', 'needs_bits', 'needs_snr', 'mod_labels'):
            if hasattr(src, attr):
                setattr(dst, attr, getattr(src, attr))

    def _wrap_sync_parameter_output(crit, crit_name):
        """Expose sync-conditioned models' separation tensor to base losses."""
        base_model = model.module if hasattr(model, 'module') else model
        if base_model.__class__.__name__ not in {
            'IQUMamba1D_SyncConditioned',
            'IQUMamba1D_PhysicalSyncRTN',
        }:
            return crit, crit_name

        def sync_parameter_output_crit(outputs, targets, *fargs, **fkwargs):
            separation = outputs
            if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
                separation = outputs[0]
            return crit(separation, targets, *fargs, **fkwargs)

        _forward_criterion_attrs(sync_parameter_output_crit, crit)
        sync_parameter_output_crit.cross_snr_partner_criterion = sync_parameter_output_crit
        return sync_parameter_output_crit, f"{crit_name}+SyncConditionedOutput"

    def _wrap_qam_turbo_joint(crit, crit_name):
        base_model = model.module if hasattr(model, 'module') else model
        is_qam_turbo_model = base_model.__class__.__name__ == 'IQUMamba1D_QAMTurboUnfold'
        joint_loss_enabled = bool(getattr(args, 'qam_turbo_joint_loss_enable', False))
        if not is_qam_turbo_model and not joint_loss_enabled:
            return crit, crit_name

        def qam_turbo_joint_crit(outputs, targets, mixture=None, *fargs, **fkwargs):
            separation = outputs
            auxiliary = None
            if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
                separation, auxiliary = outputs[0], outputs[1]
            if mixture is None:
                mixture = fkwargs.pop('mixture', None)
            if getattr(crit, 'needs_mixture', False):
                if mixture is None:
                    raise ValueError("QAM turbo joint loss requires the input mixture")
                base_loss = crit(separation, targets, mixture, *fargs, **fkwargs)
            else:
                base_loss = crit(separation, targets, *fargs, **fkwargs)
            if not joint_loss_enabled or not torch.is_grad_enabled() or auxiliary is None:
                return base_loss
            if mixture is None:
                raise ValueError("QAM turbo auxiliary loss requires the input mixture")
            from util.qam_turbo_loss import qam_turbo_auxiliary_loss

            auxiliary_loss, _ = qam_turbo_auxiliary_loss(
                separation,
                targets,
                mixture,
                auxiliary,
                mixture_weight=_float_arg('qam_turbo_mixture_loss_weight', 0.15),
                qam_weight=_float_arg('qam_turbo_qam_loss_weight', 0.03),
                independence_weight=_float_arg('qam_turbo_independence_loss_weight', 0.02),
                intermediate_weight=_float_arg('qam_turbo_intermediate_loss_weight', 0.20),
                route_entropy_weight=_float_arg('qam_turbo_route_entropy_weight', 0.002),
                intermediate_alpha=_float_arg('pit_si_snr_huber_alpha', 1.0),
                intermediate_beta=_float_arg('pit_si_snr_huber_beta', 1.0),
                intermediate_delta=_float_arg('pit_si_snr_huber_delta', 1.0),
            )
            return base_loss + auxiliary_loss

        _forward_criterion_attrs(qam_turbo_joint_crit, crit)
        qam_turbo_joint_crit.needs_mixture = True
        qam_turbo_joint_crit.cross_snr_partner_criterion = qam_turbo_joint_crit
        if not joint_loss_enabled:
            return qam_turbo_joint_crit, f"{crit_name}+QAMTurboOutput"
        return qam_turbo_joint_crit, (
            f"{crit_name}+QAMTurboJoint("
            f"mix={_float_arg('qam_turbo_mixture_loss_weight', 0.15):g},"
            f"qam={_float_arg('qam_turbo_qam_loss_weight', 0.03):g},"
            f"iter={_float_arg('qam_turbo_intermediate_loss_weight', 0.20):g})"
        )

    def _wrap_qam_reg(crit, crit_name):
        qam_weight = getattr(args, 'qam_reg_weight', 0.0)
        if qam_weight <= 0.0:
            return crit, crit_name
        
        def qam_reg_crit(outputs, targets, *fargs, **fkwargs):
            from util.loss import qam_lattice_regularizer_on_output
            base_loss = crit(outputs, targets, *fargs, **fkwargs)
            # handle case where outputs is a tuple (e.g. sep_output, demod_output)
            y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
            reg_loss = qam_lattice_regularizer_on_output(y_hat)
            return base_loss + qam_weight * reg_loss

        _forward_criterion_attrs(qam_reg_crit, crit)

        return qam_reg_crit, f"{crit_name}+QAMReg(λ={qam_weight})"

    def _wrap_topology_aux(crit, crit_name):
        topology_weight = _float_arg('topology_aux_weight', 0.0)
        if topology_weight <= 0.0:
            return crit, crit_name

        def topology_aux_crit(outputs, targets, *fargs, **fkwargs):
            from util.loss import topology_stat_loss_on_output
            base_loss = crit(outputs, targets, *fargs, **fkwargs)
            y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
            aux_loss = topology_stat_loss_on_output(
                y_hat,
                targets,
                num_sources=_num_sources_from_args(),
                axis_weight=_float_arg('topology_aux_axis_weight', 1.0),
                amp_weight=_float_arg('topology_aux_amp_weight', 1.0),
                phase_weight=_float_arg('topology_aux_phase_weight', 0.5),
                kurtosis_weight=_float_arg('topology_aux_kurtosis_weight', 0.25),
            )
            return base_loss + topology_weight * aux_loss

        _forward_criterion_attrs(topology_aux_crit, crit)
        return topology_aux_crit, f"{crit_name}+TopologyLoss(w={topology_weight})"

    def _wrap_sep_constraint(crit, crit_name):
        sep_weight = _float_arg('sep_constraint_weight', 0.0)
        if sep_weight <= 0.0:
            return crit, crit_name

        def sep_constraint_crit(outputs, targets, mixture=None, *fargs, **fkwargs):
            from util.loss import separation_mechanism_loss_on_output
            if mixture is None:
                mixture = fkwargs.get('mixture', None)
            if mixture is None:
                raise ValueError("separation constraint loss requires the input mixture")
            if getattr(crit, 'needs_mixture', False):
                base_loss = crit(outputs, targets, mixture, *fargs, **fkwargs)
            else:
                base_loss = crit(outputs, targets, *fargs, **fkwargs)
            y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
            aux_loss = separation_mechanism_loss_on_output(
                y_hat,
                mixture,
                num_sources=_num_sources_from_args(),
                mix_weight=_float_arg('sep_constraint_mix_weight', 1.0),
                corr_weight=_float_arg('sep_constraint_corr_weight', 1.0),
                energy_weight=_float_arg('sep_constraint_energy_weight', 0.1),
            )
            return base_loss + sep_weight * aux_loss

        _forward_criterion_attrs(sep_constraint_crit, crit)
        sep_constraint_crit.needs_mixture = True
        return sep_constraint_crit, f"{crit_name}+SeparationConstraintLoss(w={sep_weight})"

    def _wrap_receiver_symbol(crit, crit_name):
        symbol_weight = _float_arg('receiver_symbol_weight', 0.0)
        if symbol_weight <= 0.0:
            return crit, crit_name

        def receiver_symbol_crit(outputs, targets, *fargs, **fkwargs):
            from util.low_snr_training import receiver_domain_symbol_loss, receiver_subset_size
            from util.evaluation import _infer_modulations_from_data_choice

            base_loss = crit(outputs, targets, *fargs, **fkwargs)
            probability = min(max(_float_arg('receiver_symbol_probability', 0.25), 0.0), 1.0)
            if not torch.is_grad_enabled() or probability <= 0.0:
                return base_loss
            if float(torch.rand((), device=targets.device).item()) >= probability:
                return base_loss
            y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(y_hat, (list, tuple)):
                y_hat = y_hat[-1]
            subset_size = receiver_subset_size(
                y_hat.size(0),
                _float_arg('receiver_symbol_batch_fraction', 0.25),
            )
            y_hat = y_hat[:subset_size]
            receiver_targets = targets[:subset_size]
            receiver_num_sources = int(getattr(args, 'num_sources', _num_sources_from_args() or 2))
            inferred_modulations = _infer_modulations_from_data_choice(
                getattr(args, 'data_choice', ''),
                receiver_num_sources,
            )
            symbol_loss = receiver_domain_symbol_loss(
                y_hat,
                receiver_targets,
                source_names=inferred_modulations or getattr(args, 'source_names', None),
                num_sources=receiver_num_sources,
                sps_candidates=tuple(getattr(args, 'receiver_sps_candidates', (10, 20))),
                rrc_rolloff=_float_arg('receiver_rrc_rolloff', 0.35),
                rrc_span=int(getattr(args, 'receiver_rrc_span', 20)),
                constellation_weight=_float_arg('receiver_constellation_weight', 0.05),
                softmin_temperature=_float_arg('receiver_softmin_temperature', 0.1),
                beta=_float_arg('receiver_symbol_beta', 0.5),
                eps=_float_arg('receiver_symbol_eps', 1e-8),
            )
            return base_loss + (symbol_weight / probability) * symbol_loss

        _forward_criterion_attrs(receiver_symbol_crit, crit)
        receiver_symbol_crit.cross_snr_partner_criterion = crit
        return receiver_symbol_crit, f"{crit_name}+ReceiverSymbol(w={symbol_weight})"

    def _wrap_confidence_soft_pit(crit, crit_name):
        if not bool(getattr(args, 'confidence_soft_pit_enable', False)):
            return crit, crit_name
        from util.confidence_soft_pit import ConfidenceAdaptiveSoftPITLoss

        adaptive = ConfidenceAdaptiveSoftPITLoss(
            alpha=_float_arg('pit_si_snr_huber_alpha', 1.0),
            beta=_float_arg('pit_si_snr_huber_beta', 1.0),
            rms_lambda=_float_arg('rms_lambda', 0.5),
            temperature_min=_float_arg('confidence_soft_pit_temperature_min', 0.05),
            temperature_max=_float_arg('confidence_soft_pit_temperature_max', 2.0),
            snr_low_db=_float_arg('confidence_soft_pit_snr_low_db', -10.0),
            snr_high_db=_float_arg('confidence_soft_pit_snr_high_db', 10.0),
            anneal_power=_float_arg('confidence_soft_pit_anneal_power', 2.0),
            total_epochs=int(args.num_epochs),
            delta=_float_arg('pit_si_snr_huber_delta', 1.0),
        )
        return adaptive, (
            "ConfidenceSoftPIT("
            f"tau={adaptive.temperature_min:g}->{adaptive.temperature_max:g}, "
            f"snr={adaptive.snr_low_db:g}->{adaptive.snr_high_db:g}dB)"
        )

    def _wrap_cumulant_prior(crit, crit_name):
        source_prior_enable = bool(getattr(args, 'cumulant_prior_enable', False))
        residual_prior_enable = bool(getattr(args, 'cumulant_residual_enable', False))
        if not source_prior_enable and not residual_prior_enable:
            return crit, crit_name
        prior_weight = _float_arg('cumulant_prior_weight', 0.05)
        if prior_weight <= 0.0:
            return crit, crit_name

        def cumulant_prior_crit(outputs, targets, mixture=None, *fargs, **fkwargs):
            if mixture is None:
                mixture = fkwargs.pop('mixture', None)
            if getattr(crit, 'needs_mixture', False):
                if mixture is None:
                    raise ValueError('cumulant prior wrapper requires the input mixture')
                base_loss = crit(outputs, targets, mixture, *fargs, **fkwargs)
            else:
                base_loss = crit(outputs, targets, *fargs, **fkwargs)
            if not torch.is_grad_enabled():
                return base_loss
            probability = min(max(_float_arg('cumulant_prior_probability', 0.5), 0.0), 1.0)
            if probability <= 0.0 or float(torch.rand((), device=targets.device).item()) >= probability:
                return base_loss
            from util.cumulant_prior import (
                cumulant_prior_loss,
                gaussian_residual_prior_loss,
            )
            from util.low_snr_training import receiver_subset_size

            separation = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(separation, (list, tuple)):
                separation = separation[-1]
            subset_size = receiver_subset_size(
                separation.size(0),
                _float_arg('cumulant_prior_batch_fraction', 0.25),
            )

            prior_terms = []
            window_sizes = tuple(
                getattr(args, 'cumulant_prior_window_sizes', (256, 512, 1024))
            )
            if source_prior_enable:
                prior_terms.append(cumulant_prior_loss(
                    separation[:subset_size],
                    targets[:subset_size],
                    window_sizes=window_sizes,
                    self_weight=_float_arg('cumulant_prior_self_weight', 1.0),
                    cross_weight=_float_arg('cumulant_prior_cross_weight', 0.25),
                    confidence_floor=_float_arg('cumulant_prior_confidence_floor', 0.1),
                    beta=_float_arg('cumulant_prior_beta', 0.25),
                    eps=_float_arg('cumulant_prior_eps', 1e-8),
                ))
            if residual_prior_enable:
                if mixture is None:
                    raise ValueError(
                        'Gaussian residual prior requires the input mixture'
                    )
                prior_terms.append(gaussian_residual_prior_loss(
                    separation[:subset_size],
                    targets[:subset_size],
                    mixture[:subset_size],
                    window_sizes=window_sizes,
                    residual_weight=_float_arg('cumulant_residual_weight', 1.0),
                    cross_weight=_float_arg('cumulant_residual_cross_weight', 0.25),
                    confidence_floor=_float_arg('cumulant_prior_confidence_floor', 0.1),
                    beta=_float_arg(
                        'cumulant_residual_beta',
                        _float_arg('cumulant_prior_beta', 0.25),
                    ),
                    eps=_float_arg('cumulant_prior_eps', 1e-8),
                ))
            prior_loss = torch.stack(prior_terms).sum()
            return base_loss + (prior_weight / probability) * prior_loss

        _forward_criterion_attrs(cumulant_prior_crit, crit)
        if residual_prior_enable:
            cumulant_prior_crit.needs_mixture = True
        name = f"{crit_name}+CumulantPrior(w={prior_weight:g})"
        if residual_prior_enable:
            name += (
                f"+GaussianResidual(w={_float_arg('cumulant_residual_weight', 1.0):g},"
                f"cross={_float_arg('cumulant_residual_cross_weight', 0.25):g})"
            )
        return cumulant_prior_crit, name

    def _wrap_fsq_token_ce(crit, crit_name):
        """Stage 296: frozen-FSQ-tokenizer cross-entropy prior (training only).

        Predicted sources are encoded by a frozen pretrained tokenizer and
        pushed, via per-dim cross entropy in the FSQ lattice, to carry the
        same discrete tokens as the clean targets. Non-autoregressive analog
        of the RF Transformer's token CE: BER-aligned supervision with zero
        inference-time cost (eval path returns the base loss unchanged).
        """
        if not bool(getattr(args, 'fsq_token_ce_enable', False)):
            return crit, crit_name
        ce_weight = _float_arg('fsq_token_ce_weight', 0.3)
        if ce_weight <= 0.0:
            return crit, crit_name
        checkpoint_path = getattr(args, 'fsq_tokenizer_checkpoint', None)
        if not checkpoint_path:
            raise ValueError(
                'fsq_token_ce_enable requires fsq_tokenizer_checkpoint '
                '(pretrain one with pretrain_fsq_tokenizer.py)'
            )
        temperature = _float_arg('fsq_token_ce_temperature', 0.5)
        warmup_steps = int(_float_arg('fsq_token_ce_warmup_steps', 2000))
        state = {'step': 0, 'tokenizer': None}

        def fsq_token_ce_crit(outputs, targets, *fargs, **fkwargs):
            base_loss = crit(outputs, targets, *fargs, **fkwargs)
            if not torch.is_grad_enabled():
                return base_loss
            from util.fsq_token_prior import fsq_token_ce_loss, load_frozen_tokenizer

            separation = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(separation, (list, tuple)):
                separation = separation[-1]
            if state['tokenizer'] is None:
                state['tokenizer'] = load_frozen_tokenizer(
                    checkpoint_path, device=targets.device
                )
            tokenizer = state['tokenizer']
            tokenizer_device = next(tokenizer.parameters()).device
            if tokenizer_device != targets.device:
                tokenizer.to(targets.device)
            state['step'] += 1
            if warmup_steps > 0:
                ramp = min(1.0, state['step'] / float(warmup_steps))
            else:
                ramp = 1.0
            ce = fsq_token_ce_loss(
                tokenizer,
                separation,
                targets,
                temperature=temperature,
            )
            return base_loss + (ce_weight * ramp) * ce

        _forward_criterion_attrs(fsq_token_ce_crit, crit)
        name = (
            f"{crit_name}+FSQTokenCE(w={ce_weight:g},T={temperature:g},"
            f"warmup={warmup_steps})"
        )
        return fsq_token_ce_crit, name

    def _wrap_shared_permutation_multiscale(crit, crit_name):
        if not bool(getattr(args, 'shared_permutation_multiscale_enable', False)):
            return crit, crit_name
        auxiliary_weight = _float_arg('shared_permutation_multiscale_weight', 0.2)
        if auxiliary_weight <= 0.0:
            return crit, crit_name

        def shared_permutation_multiscale_crit(outputs, targets, *fargs, **fkwargs):
            if not isinstance(outputs, (list, tuple)):
                return crit(outputs, targets, *fargs, **fkwargs)
            final_output = outputs[0]
            base_loss = crit(final_output, targets, *fargs, **fkwargs)
            if not torch.is_grad_enabled() or len(outputs) < 2:
                return base_loss
            from util.shared_permutation_multiscale import shared_permutation_auxiliary_loss

            scale_weights = tuple(getattr(
                args,
                'shared_permutation_multiscale_weights',
                (1.0, 0.5, 0.25),
            ))
            if len(scale_weights) != len(outputs):
                scale_weights = tuple(1.0 / (2 ** index) for index in range(len(outputs)))
            auxiliary_loss = shared_permutation_auxiliary_loss(
                outputs,
                targets,
                weights=scale_weights,
                include_final=False,
            )
            return base_loss + auxiliary_weight * auxiliary_loss

        _forward_criterion_attrs(shared_permutation_multiscale_crit, crit)
        return shared_permutation_multiscale_crit, (
            f"{crit_name}+SharedPermutationMultiScale(w={auxiliary_weight:g})"
        )

    def _wrap_evidence_moe_route(crit, crit_name):
        """Teach an evidence router to choose the currently better candidate."""
        if not bool(getattr(args, 'evidence_moe_route_supervision_enable', False)):
            return crit, crit_name
        route_weight = _float_arg('evidence_moe_route_loss_weight', 0.05)
        if route_weight <= 0.0:
            return crit, crit_name
        target_temperature = _float_arg('evidence_moe_route_target_temperature', 0.25)

        def evidence_moe_route_crit(outputs, targets, *fargs, **fkwargs):
            if not isinstance(outputs, tuple) or len(outputs) < 2 or not isinstance(outputs[1], dict):
                return crit(outputs, targets, *fargs, **fkwargs)

            separation = outputs[0]
            auxiliary = outputs[1]
            base_loss = crit(separation, targets, *fargs, **fkwargs)
            if not torch.is_grad_enabled():
                return base_loss
            candidate_outputs = auxiliary.get('candidate_outputs')
            route_weights = auxiliary.get('route_weights')
            if candidate_outputs is None or route_weights is None:
                return base_loss

            from util.evidence_moe_loss import counterfactual_route_loss

            route_loss, _, _ = counterfactual_route_loss(
                candidate_outputs,
                targets,
                route_weights,
                temperature=target_temperature,
                quality_loss=(
                    'pit_si_snr_huber' if int(getattr(args, 'stage', -1)) == 255
                    else 'smooth_l1'
                ),
                si_snr_alpha=float(getattr(args, 'pit_si_snr_huber_alpha', 0.1)),
                huber_beta=float(getattr(args, 'pit_si_snr_huber_beta', 1.0)),
                huber_delta=float(getattr(args, 'pit_si_snr_huber_delta', 0.5)),
            )
            return base_loss + route_weight * route_loss

        _forward_criterion_attrs(evidence_moe_route_crit, crit)
        return evidence_moe_route_crit, f"{crit_name}+EvidenceRoute(w={route_weight:g})"

    def _wrap_noise_contrastive_prior(crit, crit_name):
        """Add Stage 223's training-only residual-noise contrastive objective."""
        if not bool(getattr(args, 'noise_contrastive_prior_enable', False)):
            return crit, crit_name
        prior_weight = _float_arg('noise_contrastive_prior_weight', 0.05)
        if prior_weight <= 0.0:
            return crit, crit_name

        patch_size = max(1, int(getattr(args, 'noise_contrastive_prior_patch_size', 64)))
        patch_stride = max(1, int(getattr(args, 'noise_contrastive_prior_patch_stride', 32)))
        temperature = _float_arg('noise_contrastive_prior_temperature', 0.2)
        residual_weight = _float_arg('noise_contrastive_prior_residual_weight', 0.1)
        gate_floor = _float_arg('noise_contrastive_prior_gate_floor', 0.1)
        projector = getattr(model, 'noise_prior_projector', None)

        def noise_contrastive_prior_crit(outputs, targets, mixture, *fargs, **fkwargs):
            base_outputs = outputs
            if isinstance(outputs, tuple) and outputs:
                base_outputs = outputs[0]
            base_loss = crit(base_outputs, targets, *fargs, **fkwargs)
            if not torch.is_grad_enabled():
                return base_loss

            separation = base_outputs
            if isinstance(separation, (list, tuple)):
                separation = separation[-1]
            from util.noise_contrastive_prior import residual_noise_contrastive_prior_loss

            prior_loss, _ = residual_noise_contrastive_prior_loss(
                separation,
                targets,
                mixture,
                patch_size=patch_size,
                patch_stride=patch_stride,
                temperature=temperature,
                residual_weight=residual_weight,
                gate_floor=gate_floor,
                projector=projector,
            )
            return base_loss + prior_weight * prior_loss

        _forward_criterion_attrs(noise_contrastive_prior_crit, crit)
        noise_contrastive_prior_crit.needs_mixture = True
        return noise_contrastive_prior_crit, (
            f"{crit_name}+ResidualNoiseContrastive(w={prior_weight:g}, "
            f"patch={patch_size}/{patch_stride})"
        )

    # Handle single or two-phase training
    if args.two_phase_loss:
        if not 0.0 <= args.loss_phase1_ratio <= 1.0:
            raise ValueError(f"loss_phase1_ratio must be in [0, 1], got {args.loss_phase1_ratio}")
        if args.loss_switch_epoch is not None and args.loss_switch_epoch < 0:
            raise ValueError(f"loss_switch_epoch must be >= 0, got {args.loss_switch_epoch}")

        if args.loss_switch_epoch is None:
            switch_epoch = int(round(args.num_epochs * args.loss_phase1_ratio))
        else:
            switch_epoch = int(args.loss_switch_epoch)
        switch_epoch = max(0, min(int(args.num_epochs), switch_epoch))

        phase1_criterion, phase1_name = build_criterion(args.loss_phase1, args)
        phase2_criterion, phase2_name = build_criterion(args.loss_phase2, args)
        phase1_criterion, phase1_name = _wrap_sync_parameter_output(phase1_criterion, phase1_name)
        phase2_criterion, phase2_name = _wrap_sync_parameter_output(phase2_criterion, phase2_name)
        phase1_criterion, phase1_name = _wrap_qam_turbo_joint(phase1_criterion, phase1_name)
        phase2_criterion, phase2_name = _wrap_qam_turbo_joint(phase2_criterion, phase2_name)
        phase1_criterion, phase1_name = _wrap_noise_contrastive_prior(phase1_criterion, phase1_name)
        phase2_criterion, phase2_name = _wrap_noise_contrastive_prior(phase2_criterion, phase2_name)
        phase1_criterion, phase1_name = _wrap_evidence_moe_route(phase1_criterion, phase1_name)
        phase2_criterion, phase2_name = _wrap_evidence_moe_route(phase2_criterion, phase2_name)
        phase1_criterion, phase1_name = _wrap_qam_reg(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_topology_aux(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_sep_constraint(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_receiver_symbol(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_confidence_soft_pit(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_shared_permutation_multiscale(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_cumulant_prior(phase1_criterion, phase1_name)
        phase1_criterion, phase1_name = _wrap_fsq_token_ce(phase1_criterion, phase1_name)
        phase2_criterion, phase2_name = _wrap_qam_reg(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_topology_aux(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_sep_constraint(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_receiver_symbol(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_confidence_soft_pit(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_shared_permutation_multiscale(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_cumulant_prior(phase2_criterion, phase2_name)
        phase2_criterion, phase2_name = _wrap_fsq_token_ce(phase2_criterion, phase2_name)

        criterion = TwoPhaseCriterion(
            phase1_criterion=phase1_criterion,
            phase2_criterion=phase2_criterion,
            switch_epoch=switch_epoch,
            phase1_name=phase1_name,
            phase2_name=phase2_name,
            logger=logger,
        )
        logger.info(
            f"Loss Function: Two-phase ({phase1_name} -> {phase2_name}), "
            f"switch_epoch={switch_epoch}/{args.num_epochs}"
        )
    else:
        criterion, loss_name = build_criterion(args.loss_fun, args)
        criterion, loss_name = _wrap_sync_parameter_output(criterion, loss_name)
        criterion, loss_name = _wrap_qam_turbo_joint(criterion, loss_name)
        criterion, loss_name = _wrap_noise_contrastive_prior(criterion, loss_name)
        criterion, loss_name = _wrap_evidence_moe_route(criterion, loss_name)
        criterion, loss_name = _wrap_qam_reg(criterion, loss_name)
        criterion, loss_name = _wrap_topology_aux(criterion, loss_name)
        criterion, loss_name = _wrap_sep_constraint(criterion, loss_name)
        criterion, loss_name = _wrap_receiver_symbol(criterion, loss_name)
        criterion, loss_name = _wrap_confidence_soft_pit(criterion, loss_name)
        criterion, loss_name = _wrap_shared_permutation_multiscale(criterion, loss_name)
        criterion, loss_name = _wrap_cumulant_prior(criterion, loss_name)
        criterion, loss_name = _wrap_fsq_token_ce(criterion, loss_name)
        logger.info(f"Loss Function: {loss_name}")

    # Optimizer defaults by model family
    # IQUMamba/BiMamba: usually stable at 1e-3
    # TFGridNet/SPMamba/Conformer-GridNet: generally prefers 3e-4
    # TIGER variants: often more sensitive, use 1e-4 by default
    if args.stage in {6, 10, 11, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 30, 31, 238, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 378}:
        default_lr = 3e-4
    elif args.stage in {7, 8, 9, 28}:
        default_lr = 1e-4
    else:
        default_lr = 1e-3
    base_lr = args.lr if args.lr is not None else default_lr
    lr = args.lr_phase1 if args.two_phase_loss and args.lr_phase1 is not None else base_lr
    
    optimizer_model = model.module if hasattr(model, "module") else model
    no_decay_names = (
        set(optimizer_model.no_weight_decay())
        if hasattr(optimizer_model, "no_weight_decay")
        else set()
    )
    if no_decay_names:
        decay_params = []
        no_decay_params = []
        for name, parameter in optimizer_model.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay_params if name in no_decay_names else decay_params).append(parameter)
        optimizer = torch.optim.Adam(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        logger.info(
            "Optimizer no-weight-decay parameters: "
            + ", ".join(sorted(no_decay_names))
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: Adam(lr={lr:.2e}, weight_decay={args.weight_decay:.2e})")
    if args.two_phase_loss:
        logger.info(
            "Two-phase LR plan: "
            f"phase1={args.lr_phase1 if args.lr_phase1 is not None else base_lr:.2e}, "
            f"phase2={args.lr_phase2 if args.lr_phase2 is not None else base_lr:.2e}"
        )
    
    # Scheduler
    scheduler_kwargs = {
        "optimizer": optimizer,
        "mode": "min",
        "factor": 0.5,
        "patience": 4,
    }
    # Some environments (older/stripped torch builds) may not accept 'verbose'
    if "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters:
        scheduler_kwargs["verbose"] = True

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**scheduler_kwargs)
    
    return criterion, optimizer, scheduler


def _torch_load_for_test(weights_path: Path, logger):
    """Load test weights, accepting both pure state_dict files and trusted training checkpoints."""
    try:
        return torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only.
        return torch.load(str(weights_path), map_location="cpu")
    except Exception as exc:
        logger.warning(
            f"weights_only=True could not load {weights_path}. Falling back to "
            "weights_only=False so this project's full training checkpoints can be "
            f"evaluated. Only do this for trusted checkpoints. Original error: {exc}"
        )
        try:
            return torch.load(str(weights_path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(weights_path), map_location="cpu")


def _candidate_weight_paths(weights_path: Path):
    """Return primary and sibling checkpoint candidates for test-mode evaluation."""
    candidates = [weights_path]
    for filename in (
        "best_training_checkpoint.pth",
        "latest_training_checkpoint.pth",
        "best_model_weights.pth",
    ):
        candidate = weights_path.with_name(filename)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _extract_model_state_for_test(loaded_obj, weights_path: Path):
    """Return a model state_dict from either a pure weights file or a full training checkpoint."""
    if not isinstance(loaded_obj, dict):
        raise TypeError(f"Unsupported weights object in {weights_path}: {type(loaded_obj).__name__}")

    for key in ("best_model_state_dict", "model_state_dict", "state_dict"):
        state_dict = loaded_obj.get(key)
        if isinstance(state_dict, dict):
            return state_dict, key

    return loaded_obj, "raw_state_dict"


def validate_checkpoint_source_count(loaded_obj, num_sources, weights_path=None):
    """Reject a full checkpoint whose declared output cardinality is incompatible."""
    if not isinstance(loaded_obj, dict):
        return
    saved_num_sources = loaded_obj.get('num_sources')
    if saved_num_sources is None:
        saved_num_sources = loaded_obj.get('training_config', {}).get('num_sources')
    if saved_num_sources is not None and int(saved_num_sources) != int(num_sources):
        location = f" {weights_path}" if weights_path is not None else ""
        raise ValueError(
            f"Checkpoint{location} was created for num_sources={saved_num_sources}, "
            f"but this run uses num_sources={num_sources}."
        )


def _load_model_state_for_test(weights_path: Path, logger, num_sources=None):
    """Load model weights for test mode, falling back to sibling checkpoints if needed."""
    errors = []
    for candidate in _candidate_weight_paths(weights_path):
        if not candidate.exists():
            errors.append(f"{candidate}: file does not exist")
            continue
        try:
            loaded_weights = _torch_load_for_test(candidate, logger)
            if num_sources is not None:
                validate_checkpoint_source_count(loaded_weights, num_sources, candidate)
            state_dict, state_source = _extract_model_state_for_test(loaded_weights, candidate)
            if candidate != weights_path:
                logger.warning(
                    f"Primary weights file {weights_path} could not be used; "
                    f"loaded fallback checkpoint {candidate}."
                )
            return state_dict, f"{state_source} from {candidate}"
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    detail = "\n  ".join(errors)
    raise RuntimeError(
        f"Could not load model weights from {weights_path} or sibling checkpoints.\n"
        f"  {detail}"
    )


def subsample_train_loader(train_loader, ratio: float, seed: int, logger):
    """Use only a ratio of the current training split while keeping val/test unchanged."""
    if ratio >= 1.0:
        return train_loader
    if ratio <= 0.0:
        raise ValueError(f"train_subset_ratio must be in (0, 1], got {ratio}")

    train_dataset = train_loader.dataset
    total_size = len(train_dataset)
    keep_size = max(1, int(total_size * ratio))

    generator = torch.Generator()
    generator.manual_seed(0 if seed is None else int(seed))
    indices = torch.randperm(total_size, generator=generator)[:keep_size].tolist()
    subset = Subset(train_dataset, indices)

    logger.info(
        f"Train subset ratio: {ratio:.2f} (using {keep_size}/{total_size} samples from current training split)"
    )

    return DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
    )


def collect_loss_run_summary(results_folder: str) -> dict:
    """Collect key metrics from one finished run."""
    summary = {"results_folder": results_folder}

    history_path = RESULTS_ROOT / results_folder / "weights" / "training_history.pth"
    if history_path.exists():
        history = torch.load(str(history_path), map_location="cpu")
        summary["best_val_loss"] = float(history.get("best_val_loss", float("nan")))
        summary["trained_epochs"] = int(history.get("total_epochs", 0))
    else:
        summary["best_val_loss"] = float("nan")
        summary["trained_epochs"] = 0

    csv_path = RESULTS_ROOT / results_folder / "detailed_metrics_summary.csv"
    if not csv_path.exists():
        summary["avg_corr_overall"] = float("nan")
        summary["avg_si_snr_complex_overall"] = float("nan")
        summary["avg_mse_overall"] = float("nan")
        summary["avg_phase_flip_rate_overall"] = float("nan")
        return summary

    corr_vals = []
    si_snr_complex_vals = []
    mse_vals = []
    phase_flip_vals = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Source") != "Overall":
                continue
            corr_vals.append(float(row["Correlation"]))
            si_snr_complex_vals.append(float(row["SI-SNR_complex"]))
            mse_vals.append(float(row["MSE"]))
            if "Phase_Flip_Rate" in row and row["Phase_Flip_Rate"] != "":
                phase_flip_vals.append(float(row["Phase_Flip_Rate"]))

    def _safe_mean(values):
        return float(np.mean(values)) if values else float("nan")

    summary["avg_corr_overall"] = _safe_mean(corr_vals)
    summary["avg_si_snr_complex_overall"] = _safe_mean(si_snr_complex_vals)
    summary["avg_mse_overall"] = _safe_mean(mse_vals)
    summary["avg_phase_flip_rate_overall"] = _safe_mean(phase_flip_vals)
    return summary


def summarize_snr_metrics(snr_metrics: dict) -> dict:
    """Aggregate overall metrics across SNR points for quick comparison."""
    if not snr_metrics:
        return {
            "avg_corr_overall": float("nan"),
            "avg_si_snr_complex_overall": float("nan"),
            "avg_mse_overall": float("nan"),
            "avg_phase_flip_rate_overall": float("nan"),
        }

    corr_vals = []
    si_snr_complex_vals = []
    mse_vals = []
    phase_flip_vals = []
    for _, metrics in snr_metrics.items():
        corr_vals.append(float(metrics.get("Correlation", float("nan"))))
        si_snr_complex_vals.append(float(metrics.get("SI-SNR_complex", float("nan"))))
        mse_vals.append(float(metrics.get("MSE", float("nan"))))
        phase_flip_vals.append(float(metrics.get("Phase_Flip_Rate", float("nan"))))

    def _nanmean(values):
        arr = np.array(values, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size > 0 else float("nan")

    return {
        "avg_corr_overall": _nanmean(corr_vals),
        "avg_si_snr_complex_overall": _nanmean(si_snr_complex_vals),
        "avg_mse_overall": _nanmean(mse_vals),
        "avg_phase_flip_rate_overall": _nanmean(phase_flip_vals),
    }


def test_data_loading(args, logger):
    """Test only data loading functionality"""
    logger.info("=" * 50)
    logger.info("Data Loading Test Mode")
    logger.info("=" * 50)
    
    # Setup data parameters
    input_size, num_points, input_channels = resolve_run_data_parameters(args, logger)
    
    # Constant settings
    SIGNAL_NAMES = args.source_names
    NUM_SOURCES = args.num_sources
    batch_size = args.batch_size
    
    logger.info(f"Signal Names: {SIGNAL_NAMES}")
    logger.info(f"Number of Sources: {NUM_SOURCES}")
    logger.info(f"Input Size: {input_size}")
    logger.info(f"Input Channels: {input_channels}")
    logger.info(f"Batch Size: {batch_size}")
    cfg = MambaConfig(get_model_config_path(args.stage), train=True)
    apply_model_config_overrides(args, cfg)
    cfg._load_enc_config()
    apply_loss_args_from_model_config(args, cfg)
    apply_blind_cross_snr_profile(args)
    train_aug_config = resolve_train_aug_config(cfg, args=args, num_epochs=args.num_epochs)
    device = resolve_training_device(torch, require_cuda=args.require_cuda)
    logger.info(f"Using device: {device}")
    log_accelerator_diagnostics(logger, collect_accelerator_diagnostics(torch))
    pin_memory = should_pin_memory(device, args.no_pin_memory)
    logger.info(f"DataLoader pin_memory: {pin_memory}")
    
    try:
        # Create data loaders
        logger.info("Starting to create data loaders...")
        train_loader, val_loader, snr_loaders = create_run_data_loaders(
            args, batch_size, NUM_SOURCES, pin_memory, train_aug_config, args.seed, input_size,
        )
        
        logger.info("OK Data loaders created successfully!")
        
        # Test training data loader
        logger.info("Testing training data loader...")
        train_iter = iter(train_loader)
        train_batch = next(train_iter)
        logger.info(
            f"OK Training batch shapes: "
            f"{[x.shape if hasattr(x, 'shape') else type(x).__name__ for x in train_batch]}"
        )
        
        # Test validation data loader
        logger.info("Testing validation data loader...")
        val_iter = iter(val_loader)
        val_batch = next(val_iter)
        logger.info(
            f"OK Validation batch shapes: "
            f"{[x.shape if hasattr(x, 'shape') else type(x).__name__ for x in val_batch]}"
        )
        
        # Test SNR data loaders
        logger.info("Testing SNR data loaders...")
        for snr, snr_loader in snr_loaders.items():
            snr_iter = iter(snr_loader)
            snr_batch = next(snr_iter)
            logger.info(
                f"OK SNR {snr}dB batch shapes: "
                f"{[x.shape if hasattr(x, 'shape') else type(x).__name__ for x in snr_batch]}"
            )
        
        logger.info("=" * 50)
        logger.info("All data loading tests passed!")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}")
        logger.error("=" * 50)
        raise e


def run_single_experiment(args, seed=None):
    """Run single experiment"""
    # Set random seeds
    if seed is not None:
        set_random_seeds(seed)
        args.seed = seed
    
    # If in data test mode, create simple temporary logger
    if args.mode == 'test_data':
        # Create console logger
        import logging
        logger = logging.getLogger('data_test')
        logger.setLevel(logging.INFO)
        
        # If no handlers, add console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Execute data loading test
        test_data_loading(args, logger)
        return "data_test_completed"
    
    # Full experiment flow (train/test)
    compact_mode = bool(getattr(args, "compact_results", False))
    results_folder = None
    if not compact_mode:
        results_folder = create_new_results_folder()
        if seed is not None:
            results_folder = f"{results_folder}_seed_{seed}"

        # Create necessary folders
        folders = ['weights', 'logs', 'saved_plots', 'config']
        for folder in folders:
            (RESULTS_ROOT / results_folder / folder).mkdir(parents=True, exist_ok=True)

        logger = create_logger(str(RESULTS_ROOT / results_folder / "logs" / "output.log"))
    else:
        logger_name = f"ablation_compact_{args.loss_fun}_{seed}_{random.randint(0, 10**6)}"
        logger = create_logger(logger_name, file_handle=False)

    logger.info(f"Starting experiment - Seed: {seed if seed is not None else 'None'}")
    
    # Save experiment configuration
    if not compact_mode:
        save_experiment_config(args, results_folder)
    
    # Setup device
    device = resolve_training_device(torch, require_cuda=args.require_cuda)
    logger.info(f"Using device: {device}")
    log_accelerator_diagnostics(logger, collect_accelerator_diagnostics(torch))
    pin_memory = should_pin_memory(device, args.no_pin_memory)
    logger.info(f"DataLoader pin_memory: {pin_memory}")
    
    # Setup data parameters
    input_size, num_points, input_channels = resolve_run_data_parameters(args, logger)
    
    # Get config file path and copy
    config_path = get_model_config_path(args.stage)
    if not compact_mode:
        shutil.copy2(config_path, str(RESULTS_ROOT / results_folder / "config"))
    
    # Create config object
    cfg = MambaConfig(config_path, train=True)
    apply_model_config_overrides(args, cfg)
    
    
    # Create model
    model = create_model(args, cfg, input_size, device, logger)
    apply_loss_args_from_model_config(args, cfg)
    apply_blind_cross_snr_profile(args)
    train_aug_config = resolve_train_aug_config(cfg, args=args)
    if train_aug_config:
        recommended_cmd = (
            f"python main.py --mode train --data_choice {args.data_choice} "
            f"--num_sources {args.num_sources} --source_names {' '.join(args.source_names)} --stage {args.stage} "
            f"--loss_fun {args.loss_fun}"
        )
        if train_aug_config.get('mix_enable', False):
            recommended_cmd += (
                " --train_mix_enable"
                f" --train_mix_prob {train_aug_config.get('mix_prob', 0.0)}"
                f" --train_mix_sir_min_db {train_aug_config.get('mix_sir_min_db', -3.0)}"
                f" --train_mix_sir_max_db {train_aug_config.get('mix_sir_max_db', 3.0)}"
            )
            if train_aug_config.get('mix_cross_sample', False):
                recommended_cmd += " --train_mix_cross_sample"
        logger.info(
            "Train augmentation enabled: "
            f"src_phase={train_aug_config['source_phase_jitter_deg']}deg, "
            f"src_gain={train_aug_config['source_gain_jitter_db']}dB, "
            f"shift={train_aug_config['max_common_time_shift']}, "
            f"global_phase={train_aug_config['global_phase_rotation']}, "
            f"mix_enable={train_aug_config.get('mix_enable', False)}, "
            f"mix_prob={train_aug_config.get('mix_prob', 0.0)}, "
            f"mix_sir=[{train_aug_config.get('mix_sir_min_db', -3.0)}, "
            f"{train_aug_config.get('mix_sir_max_db', 3.0)}]dB, "
            f"mix_cross_sample={train_aug_config.get('mix_cross_sample', False)}, "
            f"profile={train_aug_config.get('train_aug_profile', 'config')}, "
            f"warmup_epochs={train_aug_config.get('train_aug_warmup_epochs', 0)}"
        )
        logger.info(
            "Recommended training command: "
            f"{recommended_cmd}"
        )
    
    # Calculate model complexity
    batch_size = args.batch_size
    flops, macs, params = calculate_model_complexity(model, batch_size, input_channels, input_size, logger)
    
    # Setup training components
    criterion, optimizer, scheduler = setup_training_components(args, model, logger)
    
    # Constant settings
    SIGNAL_NAMES = args.source_names
    NUM_SOURCES = args.num_sources
    num_epochs = args.num_epochs
    early_stop_patience = args.early_stop_patience
    
    # Execute corresponding mode
    if args.mode in ['train', 'test']:
        train_loader, val_loader, snr_loaders = create_run_data_loaders(
            args, batch_size, NUM_SOURCES, pin_memory, train_aug_config, seed, input_size,
        )
        if args.mode == 'train':
            train_loader = subsample_train_loader(train_loader, args.train_subset_ratio, seed, logger)
    
    def _stage_arg(name, default):
        value = getattr(args, name, None)
        return default if value is None else value

    training_history = None
    if args.mode == 'train':
        logger.info("Training Mode")
        pretrain_choices = resolve_pretrain_data_choices(args)
        training_data_choice = (
            f"MULTI[{','.join(pretrain_choices)}]"
            if pretrain_choices else args.data_choice
        )
        training_history = train_model(
            model, scheduler, train_loader, val_loader, snr_loaders,
            criterion, optimizer, device, num_epochs, early_stop_patience,
            logger, results_folder if results_folder else "__compact__", data_choice=training_data_choice,
            num_plots=0 if compact_mode else 1, batch_size=batch_size, input_size=input_size,
            signal_names=SIGNAL_NAMES,
            num_sources=NUM_SOURCES,
            accumulation_steps=args.accumulation_steps,
            use_mixed_precision=not args.no_mixed_precision,
            save_artifacts=not compact_mode,
            save_checkpoint_every=args.save_checkpoint_every,
            report_ber=args.report_ber,
            ber_offset_search=args.ber_offset_search,
            ber_mode=args.ber_mode,
            ber_num_files=args.ber_num_files,
            ber_compute_oracle=args.ber_compute_oracle,
            amr_mode=args.amr_mode,
            demod_mode=args.demod_mode,
            demod_mode_phase1=args.demod_mode_phase1,
            demod_mode_phase2=args.demod_mode_phase2,
            lr_phase1=args.lr_phase1,
            lr_phase2=args.lr_phase2,
            demod_teacher_weight=args.demod_teacher_weight,
            demod_teacher_phase2_epochs=args.demod_teacher_phase2_epochs,
            l1_sparsity_weight=cfg.model_config.get("l1_sparsity_weight", 0.0),
            eval_pit_metric=args.eval_pit_metric,
            report_phase_flip=args.report_phase_flip,
            phase_flip_tolerance_deg=args.phase_flip_tolerance_deg,
            phase_flip_min_sc=args.phase_flip_min_sc,
            phase_flip_mode=args.phase_flip_mode,
            resume_checkpoint=args.resume_checkpoint,
            resume_allow_partial=args.resume_allow_partial,
            init_checkpoint=args.init_checkpoint,
            component_checkpoints=args.component_checkpoints,
            cross_snr_enable=bool(_stage_arg('cross_snr_enable', False)),
            cross_snr_probability=float(_stage_arg('cross_snr_probability', 0.5)),
            cross_snr_high_db=float(_stage_arg('cross_snr_high_db', 10.0)),
            cross_snr_low_start_db=float(_stage_arg('cross_snr_low_start_db', 2.0)),
            cross_snr_low_middle_db=float(_stage_arg('cross_snr_low_middle_db', -6.0)),
            cross_snr_low_final_db=float(_stage_arg('cross_snr_low_final_db', -10.0)),
            cross_snr_first_fraction=float(_stage_arg('cross_snr_first_fraction', 0.2)),
            cross_snr_second_fraction=float(_stage_arg('cross_snr_second_fraction', 0.6)),
            cross_snr_pair_weight=float(_stage_arg('cross_snr_pair_weight', 0.5)),
            cross_snr_consistency_weight=float(_stage_arg('cross_snr_consistency_weight', 0.1)),
            cross_snr_consistency_beta=float(_stage_arg('cross_snr_consistency_beta', 0.5)),
            cross_snr_eps=float(_stage_arg('cross_snr_eps', 1e-8)),
            cross_snr_shared_permutation=bool(_stage_arg('cross_snr_shared_permutation', False)),
            cross_snr_ema_teacher_enable=bool(
                _stage_arg('cross_snr_ema_teacher_enable', False)
            ),
            cross_snr_ema_decay=float(_stage_arg('cross_snr_ema_decay', 0.999)),
            cross_snr_teacher_mode=str(_stage_arg('cross_snr_teacher_mode', 'ema')),
            cross_snr_teacher_checkpoint=_stage_arg('cross_snr_teacher_checkpoint', None),
            cross_snr_teacher_view=str(_stage_arg('cross_snr_teacher_view', 'high_snr')),
            cross_snr_pair_mode=str(_stage_arg('cross_snr_pair_mode', 'complementary')),
            cross_snr_feature_consistency_weight=float(
                _stage_arg('cross_snr_feature_consistency_weight', 0.0)
            ),
            cross_snr_feature_consistency_beta=float(
                _stage_arg('cross_snr_feature_consistency_beta', 0.5)
            ),
            cross_snr_curriculum_ranges=_stage_arg(
                'cross_snr_curriculum_ranges', ((10.0, 30.0), (2.0, 30.0), (-10.0, 30.0))
            ),
            cross_snr_curriculum_boundaries=_stage_arg(
                'cross_snr_curriculum_boundaries', (0.2, 0.6)
            ),
            sync_snr_aux_weight=float(_stage_arg('sync_snr_aux_weight', 0.0)),
            sync_snr_aux_min_db=float(_stage_arg('sync_snr_aux_min_db', -10.0)),
            sync_snr_aux_max_db=float(_stage_arg('sync_snr_aux_max_db', 30.0)),
            sync_snr_aux_beta=float(_stage_arg('sync_snr_aux_beta', 0.1)),
            sync_cross_snr_consistency_weight=float(
                _stage_arg('sync_cross_snr_consistency_weight', 0.0)
            ),
            sync_cross_snr_consistency_beta=float(
                _stage_arg('sync_cross_snr_consistency_beta', 0.1)
            ),
            sync_cfo_scale=float(_stage_arg('sync_cfo_scale', 0.25)),
            sync_phase_drift_scale=float(
                _stage_arg('sync_phase_drift_scale', 0.05)
            ),
            sync_physical_require_metadata=bool(
                _stage_arg('sync_physical_require_metadata', False)
            ),
            sync_physical_supervision_weight=float(
                _stage_arg('sync_physical_supervision_weight', 0.0)
            ),
            sync_physical_cfo_weight=float(_stage_arg('sync_physical_cfo_weight', 1.0)),
            sync_physical_phase_weight=float(_stage_arg('sync_physical_phase_weight', 1.0)),
            sync_physical_timing_weight=float(_stage_arg('sync_physical_timing_weight', 1.0)),
            sync_physical_sps_weight=float(_stage_arg('sync_physical_sps_weight', 1.0)),
            sync_physical_drift_weight=float(_stage_arg('sync_physical_drift_weight', 1.0)),
            sync_physical_beta=float(_stage_arg('sync_physical_beta', 0.1)),
            training_snr_floor_db=_stage_arg('training_snr_floor_db', None),
            phase_equiv_enable=bool(_stage_arg('phase_equiv_enable', False)),
            phase_equiv_probability=float(_stage_arg('phase_equiv_probability', 0.25)),
            phase_equiv_supervised_weight=float(_stage_arg('phase_equiv_supervised_weight', 0.25)),
            phase_equiv_consistency_weight=float(_stage_arg('phase_equiv_consistency_weight', 0.1)),
            phase_equiv_max_degrees=float(_stage_arg('phase_equiv_max_degrees', 180.0)),
            phase_equiv_beta=float(_stage_arg('phase_equiv_beta', 0.5)),
            phase_equiv_eps=float(_stage_arg('phase_equiv_eps', 1e-8)),
            rf_equiv_enable=bool(_stage_arg('rf_equiv_enable', False)),
            rf_equiv_probability=float(_stage_arg('rf_equiv_probability', 0.25)),
            rf_equiv_supervised_weight=float(
                _stage_arg('rf_equiv_supervised_weight', 0.25)
            ),
            rf_equiv_consistency_weight=float(
                _stage_arg('rf_equiv_consistency_weight', 0.1)
            ),
            rf_equiv_max_phase_degrees=float(
                _stage_arg('rf_equiv_max_phase_degrees', 180.0)
            ),
            rf_equiv_max_cfo_cycles_per_sample=float(
                _stage_arg('rf_equiv_max_cfo_cycles_per_sample', 1.0e-4)
            ),
            rf_equiv_max_gain_db=float(_stage_arg('rf_equiv_max_gain_db', 2.0)),
            rf_equiv_max_shift_samples=int(
                _stage_arg('rf_equiv_max_shift_samples', 8)
            ),
            rf_equiv_conjugate_probability=float(
                _stage_arg('rf_equiv_conjugate_probability', 0.10)
            ),
            rf_equiv_source_mode=str(
                _stage_arg('rf_equiv_source_mode', 'per_source')
            ),
            rf_equiv_beta=float(_stage_arg('rf_equiv_beta', 0.5)),
            rf_equiv_eps=float(_stage_arg('rf_equiv_eps', 1e-6)),
            latent_mask_residual_weight=float(
                _stage_arg('latent_mask_residual_weight', 0.0)
            ),
            latent_mask_mixture_weight=float(
                _stage_arg('latent_mask_mixture_weight', 0.0)
            ),
            latent_mask_residual_beta=float(
                _stage_arg('latent_mask_residual_beta', 0.5)
            ),
            stage255_snr_aux_weight=float(_stage_arg('stage255_snr_aux_weight', 0.0)),
            stage255_snr_aux_min_db=float(_stage_arg('stage255_snr_aux_min_db', -10.0)),
            stage255_snr_aux_max_db=float(_stage_arg('stage255_snr_aux_max_db', 30.0)),
            stage255_snr_curriculum_enable=bool(
                _stage_arg('stage255_snr_curriculum_enable', False)
            ),
            stage255_snr_curriculum_start_db=float(
                _stage_arg('stage255_snr_curriculum_start_db', 10.0)
            ),
            stage255_snr_curriculum_end_db=float(
                _stage_arg('stage255_snr_curriculum_end_db', -10.0)
            ),
            stage255_snr_curriculum_fraction=float(
                _stage_arg('stage255_snr_curriculum_fraction', 0.5)
            ),
            stage255_expert_pretrain_epochs=int(
                _stage_arg('stage255_expert_pretrain_epochs', 0)
            ),
            stage255_router_warmup_epochs=int(
                _stage_arg('stage255_router_warmup_epochs', 0)
            ),
            stage255_router_joint_lr_scale=float(
                _stage_arg('stage255_router_joint_lr_scale', 1.0)
            ),
        )

    
    elif args.mode == 'test':
        logger.info("Testing Mode")
        model.eval()
        weights_path = Path(args.weights_path) if args.weights_path else (CHECKPOINT_ROOT / "best_model_weights.pth")
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. "
                f"Pass --weights_path <.../best_model_weights.pth> to evaluate a specific run."
            )
        state_dict, state_source = _load_model_state_for_test(
            weights_path, logger, num_sources=NUM_SOURCES
        )
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights ({state_source}).")
        
        snr_metrics = test_model(
            model, snr_loaders, criterion, device, logger, results_folder,
            num_plots=0 if compact_mode else 1, num_points=num_points, input_size=input_size,
            data_choice=args.data_choice, signal_names=SIGNAL_NAMES,
            save_artifacts=not compact_mode,
            report_ber=args.report_ber,
            ber_offset_search=args.ber_offset_search,
            ber_mode=args.ber_mode,
            ber_num_files=args.ber_num_files,
            ber_compute_oracle=args.ber_compute_oracle,
            amr_mode=args.amr_mode,
            demod_mode=args.demod_mode_phase2 if args.demod_mode_phase2 is not None else args.demod_mode,
            eval_pit_metric=args.eval_pit_metric,
            report_phase_flip=args.report_phase_flip,
            phase_flip_tolerance_deg=args.phase_flip_tolerance_deg,
            phase_flip_min_sc=args.phase_flip_min_sc,
            phase_flip_mode=args.phase_flip_mode,
        )
        
        # Clean up checkpoint file only when using the default checkpoint path.
        if args.weights_path is None:
            checkpoint_path = CHECKPOINT_ROOT / "best_model_weights.pth"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
    
    if compact_mode:
        logger.info("Experiment completed (compact mode, artifacts suppressed)")
    else:
        logger.info(f"Experiment completed - Results saved in: {results_folder}")
    if compact_mode and args.mode == 'train' and training_history is not None:
        summary = {
            "results_folder": None,
            "best_val_loss": float(training_history.get("best_val_loss", float("nan"))),
            "trained_epochs": int(training_history.get("total_epochs", 0)),
        }
        summary.update(summarize_snr_metrics(training_history.get("snr_metrics", {})))
        return summary

    return results_folder


def plot_ablation_ranking(summary_rows_sorted, save_dir, model_name="", num_epochs=0):
    """Generate a visual ranking chart for loss ablation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(summary_rows_sorted)
    if n == 0:
        return

    loss_names = [r["loss_fun"] for r in summary_rows_sorted]
    si_snr_vals = [r.get("avg_si_snr_complex_overall", float("nan")) for r in summary_rows_sorted]
    corr_vals = [r.get("avg_corr_overall", float("nan")) for r in summary_rows_sorted]
    mse_vals = [r.get("avg_mse_overall", float("nan")) for r in summary_rows_sorted]
    val_loss_vals = [r.get("best_val_loss", float("nan")) for r in summary_rows_sorted]

    # Color palette: rank 1 = gold, 2 = silver, 3 = bronze, rest = steel blue
    colors = []
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    for i in range(n):
        if i < 3:
            colors.append(medal_colors[i])
        else:
            colors.append("#4682B4")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Loss Ablation Ranking - {model_name} / {num_epochs} epochs",
        fontsize=16, fontweight="bold", y=0.98,
    )

    y_pos = np.arange(n)

    # --- Panel 1: SI-SNR_complex (higher is better) ---
    ax = axes[0, 0]
    bars = ax.barh(y_pos, si_snr_vals, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loss_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("SI-SNR_complex (dB)  [-> higher is better]")
    ax.set_title("SI-SNR_complex (avg over SNRs)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, si_snr_vals):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 2: Pearson Correlation (higher is better) ---
    ax = axes[0, 1]
    bars = ax.barh(y_pos, corr_vals, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loss_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson Correlation  [-> higher is better]")
    ax.set_title("Pearson Correlation (avg over SNRs)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, corr_vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 3: MSE (lower is better) ---
    ax = axes[1, 0]
    # Reverse colors for MSE (lower = better, so re-sort)
    mse_order = np.argsort(mse_vals)
    mse_colors = []
    for rank, idx in enumerate(mse_order):
        if rank < 3:
            mse_colors.append(medal_colors[rank])
        else:
            mse_colors.append("#4682B4")
    # But plot in original order, just color by MSE rank
    plot_mse_colors = ["#4682B4"] * n
    for rank, idx in enumerate(mse_order):
        if rank < 3:
            plot_mse_colors[idx] = medal_colors[rank]
    bars = ax.barh(y_pos, mse_vals, color=plot_mse_colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loss_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("MSE  [-> lower is better]")
    ax.set_title("MSE (avg over SNRs)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 4: Summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for i, row in enumerate(summary_rows_sorted):
        table_data.append([
            f"#{i+1}",
            row["loss_fun"],
            f"{row.get('avg_si_snr_complex_overall', float('nan')):.2f}",
            f"{row.get('avg_corr_overall', float('nan')):.4f}",
            f"{row.get('avg_mse_overall', float('nan')):.4f}",
            f"{row.get('best_val_loss', float('nan')):.4f}",
            f"{row.get('trained_epochs', 0)}",
        ])
    col_labels = ["Rank", "Loss Function", "SI-SNR_c", "Corr", "MSE", "ValLoss", "Epochs"]
    table = ax.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    # Color header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color medal rows
    for i in range(min(3, n)):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(medal_colors[i] + "40")  # semi-transparent
    ax.set_title("Summary Table", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = Path(save_dir) / "ablation_ranking.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nRanking chart saved to: {save_path}")
    return str(save_path)


def run_loss_ablation_experiment(args, seed=None):
    """Run multiple loss functions under the same setup and aggregate results."""
    loss_list = [loss_name.strip() for loss_name in args.ablation_losses if loss_name.strip()]
    if not loss_list:
        raise ValueError("ablation_losses is empty")

    summary_rows = []

    # Create a dedicated ablation results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_name = get_model_config_path(args.stage).split("model_config_")[-1].replace(".yaml", "")
    ablation_folder = f"ablation_{stage_name}_{args.num_epochs}e_{timestamp}"
    ablation_dir = RESULTS_ROOT / ablation_folder
    ablation_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Loss Ablation Mode | Train subset ratio={args.train_subset_ratio:.2f}, epochs={args.num_epochs}")
    print(f"Model: stage={args.stage} ({stage_name})")
    print(f"Losses: {loss_list}")
    print(f"Results dir: {ablation_dir}")
    print("=" * 80)

    for idx, loss_name in enumerate(loss_list, start=1):
        print(f"\n--- [{idx}/{len(loss_list)}] Running loss: {loss_name} ---")
        run_args = copy.deepcopy(args)
        run_args.mode = "train"
        run_args.loss_fun = loss_name
        run_args.multiple_runs = False
        run_args.compact_results = True

        run_result = run_single_experiment(run_args, seed=seed)
        if isinstance(run_result, dict):
            run_summary = run_result
        else:
            run_summary = collect_loss_run_summary(run_result)
        run_summary["loss_fun"] = loss_name
        summary_rows.append(run_summary)

    def _score_key(row):
        si_snr = row.get("avg_si_snr_complex_overall", float("nan"))
        corr = row.get("avg_corr_overall", float("nan"))
        val_loss = row.get("best_val_loss", float("inf"))

        si_snr = -1e12 if np.isnan(si_snr) else si_snr
        corr = -1e12 if np.isnan(corr) else corr
        val_loss = 1e12 if np.isnan(val_loss) else val_loss
        return (-si_snr, -corr, val_loss)

    summary_rows_sorted = sorted(summary_rows, key=_score_key)

    print("\n=== Loss Ablation Ranking ===")
    phase_header = " | PhaseFlip" if args.report_phase_flip else ""
    header = f"{'Rank':>4s} | {'Loss':>28s} | {'SI-SNR_c(avg)':>14s} | {'Corr(avg)':>10s} | {'MSE(avg)':>10s}{phase_header} | {'BestValLoss':>12s} | {'Epochs':>6s}"
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(summary_rows_sorted, start=1):
        print(
            f"{idx:>4d} | {row['loss_fun']:>28s} | "
            f"{row['avg_si_snr_complex_overall']:>14.4f} | "
            f"{row['avg_corr_overall']:>10.6f} | "
            f"{row.get('avg_mse_overall', float('nan')):>10.6f}"
            f"{(' | ' + format(row.get('avg_phase_flip_rate_overall', float('nan')), '>9.2%')) if args.report_phase_flip else ''} | "
            f"{row['best_val_loss']:>12.6f} | "
            f"{row['trained_epochs']:>6d}"
        )

    best_si_snr = max(summary_rows, key=lambda x: x.get("avg_si_snr_complex_overall", float("-inf")))
    best_corr = max(summary_rows, key=lambda x: x.get("avg_corr_overall", float("-inf")))
    best_val = min(summary_rows, key=lambda x: x.get("best_val_loss", float("inf")))
    print("\nBest by metric:")
    print(f"- SI-SNR_complex(avg): {best_si_snr['loss_fun']} ({best_si_snr['avg_si_snr_complex_overall']:.4f})")
    print(f"- Correlation(avg):    {best_corr['loss_fun']} ({best_corr['avg_corr_overall']:.6f})")
    print(f"- Best val loss:       {best_val['loss_fun']} ({best_val['best_val_loss']:.6f})")

    # Generate ranking chart
    plot_ablation_ranking(
        summary_rows_sorted,
        save_dir=str(ablation_dir),
        model_name=f"stage={args.stage} ({stage_name})",
        num_epochs=args.num_epochs,
    )

    # Save raw results as JSON for later use
    results_json = {
        "model_stage": args.stage,
        "model_name": stage_name,
        "num_epochs": args.num_epochs,
        "seed": seed,
        "train_subset_ratio": args.train_subset_ratio,
        "ranking": [
            {
                "rank": i + 1,
                "loss_fun": r["loss_fun"],
                "si_snr_complex": r.get("avg_si_snr_complex_overall"),
                "correlation": r.get("avg_corr_overall"),
                "mse": r.get("avg_mse_overall"),
                "phase_flip_rate": r.get("avg_phase_flip_rate_overall"),
                "best_val_loss": r.get("best_val_loss"),
                "trained_epochs": r.get("trained_epochs"),
            }
            for i, r in enumerate(summary_rows_sorted)
        ],
    }
    json_path = ablation_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"Results JSON saved to: {json_path}")

    return summary_rows_sorted


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='IQU Mamba 1D Training and Testing Program')
    
    # Basic parameters

    parser.add_argument('--data_choice', type=str, default='8PSK_M',
                       choices=DATA_CHOICE_CHOICES,
                       help='Dataset: 2018/8PSK_M/8PSK_M_8192/8PSK_M_16384/8PSK_M_32768/QPSK_16APSK/8PSK_Rs/16QAM_64QAM/64QAM_64QAM/64QAM_128QAM/16QAM_64QAM_128QAM/16QAM_64QAM_128QAM_256QAM')
    parser.add_argument(
        '--pretrain_data_choices',
        nargs='+',
        choices=DATA_CHOICE_CHOICES,
        default=None,
        help=(
            'Datasets used for joint pretraining. When set, --data_choice selects the '
            'model/input reference and this list controls the actual training domains.'
        ),
    )
    parser.add_argument(
        '--pretrain_sampling',
        choices=['balanced', 'proportional'],
        default='balanced',
        help='Joint-pretraining sampling: equal dataset probability or natural sample counts.',
    )
    parser.add_argument(
        '--pretrain_dataset_weights',
        nargs='+',
        type=float,
        default=None,
        help='Optional positive sampling weight for each --pretrain_data_choices entry.',
    )
    parser.add_argument(
        '--pretrain_input_size',
        type=int,
        default=None,
        help='Joint waveform length (default: the --data_choice input length).',
    )
    parser.add_argument(
        '--pretrain_length_policy',
        choices=['strict', 'crop', 'pad_crop'],
        default='strict',
        help='How joint pretraining handles datasets whose waveform length differs.',
    )
    parser.add_argument(
        '--synthetic_root',
        type=str,
        default=None,
        help='Root folder for MATLAB-generated synthetic datasets (default: <project>/data/synthetic)',
    )
    parser.add_argument(
        '--public_root',
        type=str,
        default=None,
        help='Root folder for public datasets (default: <project>/data). Useful for Kaggle: /kaggle/input/<dataset>',
    )
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'test_data', 'ablation_losses'],
                       help='Mode: train/test/test_data/ablation_losses')
    parser.add_argument(
        '--weights_path',
        type=str,
        default=None,
        help="Weights path for --mode test (default: <project>/checkpoint/best_model_weights.pth). "
             "Pure weights and full training checkpoints are both accepted. "
             "Example: IQUMamba1D/results/results_13/weights/best_model_weights.pth",
    )
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        default=None,
        help="Full training checkpoint for --mode train resume. "
             "Use results/<run>/weights/latest_training_checkpoint.pth or checkpoint_epoch_N.pth.",
    )
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help=(
            'Initialize model weights for a fresh training run. Unlike --resume_checkpoint, '
            'optimizer, scheduler, epoch, RNG, and validation history are not restored.'
        ),
    )
    parser.add_argument(
        '--resume_allow_partial',
        action='store_true',
        help="Allow --resume_checkpoint to warm-start only matching model tensors. "
             "Use when initializing a new architecture from an old checkpoint; optimizer and epoch state are not restored.",
    )
    parser.add_argument(
        '--component_checkpoints',
        nargs='+',
        default=None,
        help="Optional Stage-235/244/245 checkpoints. Only declared fusion component "
             "prefixes are transplanted; shared backbone tensors are not overwritten.",
    )
    parser.add_argument('--loss_fun', type=str, default='Huber',
                       choices=LOSS_FUNCTION_CHOICES,
                       help='Loss function name')
    parser.add_argument('--two_phase_loss', action='store_true',
                       help='Enable two-phase loss schedule (phase1 then phase2).')
    parser.add_argument('--loss_phase1', type=str, default='PIT-Huber',
                       choices=LOSS_FUNCTION_CHOICES,
                       help='Phase-1 loss name when --two_phase_loss is enabled.')
    parser.add_argument('--loss_phase2', type=str, default='PIT-SI-SNR+BW-MRSTFT',
                       choices=LOSS_FUNCTION_CHOICES,
                       help='Phase-2 loss name when --two_phase_loss is enabled.')
    parser.add_argument('--loss_phase1_ratio', type=float, default=0.3,
                       help='Fraction of epochs for phase-1 loss in [0,1] (used if --loss_switch_epoch is not set).')
    parser.add_argument('--loss_switch_epoch', type=int, default=None,
                       help='Epoch index (0-based) to switch from phase-1 to phase-2. Overrides --loss_phase1_ratio.')
    # Loss coefficient knobs (avoid source edits for combo-loss tuning)
    parser.add_argument('--si_snr_mse_alpha', type=float, default=1.0,
                       help='Alpha for SI-SNR+MSE')
    parser.add_argument('--si_snr_mse_beta', type=float, default=1.0,
                       help='Beta for SI-SNR+MSE')
    parser.add_argument('--si_snr_huber_alpha', type=float, default=1.0,
                       help='Alpha for SI-SNR+Huber')
    parser.add_argument('--si_snr_huber_beta', type=float, default=0.1,
                       help='Beta for SI-SNR+Huber')
    parser.add_argument('--si_snr_huber_delta', type=float, default=1.0,
                       help='Huber delta for SI-SNR+Huber')
    parser.add_argument('--pit_huber_delta', type=float, default=1.0,
                       help='Huber delta for PIT-Huber')
    parser.add_argument('--pit_si_snr_mse_alpha', type=float, default=1.0,
                       help='Alpha for PIT-SI-SNR+MSE')
    parser.add_argument('--pit_si_snr_mse_beta', type=float, default=0.1,
                       help='Beta for PIT-SI-SNR+MSE')
    parser.add_argument('--demod_aware_mse_weight', type=float, default=1.0,
                       help='Pointwise MSE weight for PIT-DEMOD-AWARE')
    parser.add_argument('--demod_aware_sisnr_weight', type=float, default=0.1,
                       help='Positive-projection SI-SNR weight for PIT-DEMOD-AWARE')
    parser.add_argument('--pit_si_snr_huber_alpha', type=float, default=1.0,
                       help='Alpha for PIT-SI-SNR+Huber')
    parser.add_argument('--pit_si_snr_huber_beta', type=float, default=1.0,
                       help='Beta for PIT-SI-SNR+Huber')
    parser.add_argument('--pit_si_snr_huber_delta', type=float, default=1.0,
                       help='Huber delta for PIT-SI-SNR+Huber')
    parser.add_argument('--identity_anchor_weight', type=float, default=0.05,
                       help='Weight of the fixed source-identity ranking anchor')
    parser.add_argument('--identity_anchor_margin', type=float, default=0.2,
                       help='Required identity-vs-swap cost margin')
    parser.add_argument('--identity_anchor_temperature', type=float, default=0.5,
                       help='Softplus temperature for the identity ranking anchor')
    parser.add_argument('--phase_increment_weight', type=float, default=0.05,
                       help='Phase-step phasor weight for PIT-SI-SNR+Huber+Phase')
    parser.add_argument('--low_snr_aux_clean_weight', type=float, default=0.05,
                       help='Clean-mixture auxiliary weight for PIT-SI-SNR+Huber+LowSNRAux')
    parser.add_argument('--low_snr_aux_noise_weight', type=float, default=0.02,
                       help='Residual-noise auxiliary weight for PIT-SI-SNR+Huber+LowSNRAux')
    parser.add_argument('--low_snr_aux_huber_beta', type=float, default=0.5,
                       help='Smooth-L1 beta for PIT-SI-SNR+Huber+LowSNRAux auxiliary terms')
    parser.add_argument('--uric_ds_weight', type=float, default=0.1,
                       help='Lambda for URIC intermediate deep-supervision stage losses')
    parser.add_argument('--uric_ds_reduction', type=str, default='sum',
                       choices=['sum', 'mean'],
                       help='Reduce URIC intermediate stage losses by sum (paper-style) or mean')
    parser.add_argument('--uric_ds_include_final_stage', action='store_true',
                       help='Also include the final URIC stage in the stage-loss term')
    parser.add_argument('--stage', type=int, default=4,
                       choices=supported_stage_ids(),
                       help=(
                            'Stage: 2/3/4/5=IQUMamba | 6=TFGridNet | '
                            '7=TIGER | 8=TIGER-Fast | 9=TIGER-Tiny | '
                            '10=TFGridNet-Fast | 11=TFGridNet-Turbo | '
                            '12=BiMamba | 13=SPMamba | 14=Conformer-GridNet | '
                             '15=SPMamba-Fast | 16=IQUMamba(BiMamba-matched config) | '
                             '17=DualDomainMamba | 18=NES2Net | 19=DualDomainMambaLite | 20=CTDCRN | '
                             '21=DualDomainMambaV2 | 22=DualDomainMambaV3 | '
                             '23=DualDomainMambaZeroInit | 24=DualDomainMambaDualPath | '
                             '25=ICASPBaselineUNet | 26=ICASPBaselineWaveNet | '
                             '27=DualDomainMambaCrossMamba | 28=DualDomainMambaV4 | '
                             '29=DualDomainMamba2(Mamba-2 SSD) | 30=DualDomainBandSplit | '
                             '0=Mamba Default | 1=Mamba | '
                             '161-162=LSSG SE/SwiGLU | 163-166=DCCB Deep/Attn/Lags/Mamba | 167=ResUNet1D_Legacy5(old stage42 5-stage) | '
                             '168=AgentAttentionBottleneck | 169=TransNeXtBottleneck | 170=BiLevelRoutingBottleneck | 171=DeformableTemporalBottleneck | '
                             '172=GLAEncoder(Mamba replacement) | 173=MegaEncoder(Mamba replacement) | 174=HyenaEncoder(Mamba replacement) | 175=RetNetEncoder(Mamba replacement) | '
                             '176=IQUMamba_Stage4_URIC(stage4 Mamba+ASC + URIC) | '
                             '177=GriffinEncoder(Mamba replacement) | 178=xLSTMEncoder(Mamba replacement) | 179=SpectralEncoder(FNO replacement) | 180=DeltaLinearEncoder(Mamba replacement) | '
                             '181=LSSG-DW Skip | 182=Deformable Temporal Skip | 183=Frequency-Aware Skip | 184=Complex-Aware Skip | 185=SpectralLowRankEncoder | '
                             '186=ResUNet_HyenaBottleneck(stage56 + Hyena bottleneck) | 187=ResUNet_SpectralLowRankBottleneck(stage56 + spectral bottleneck) | 188=ResUNet_MegaMidEncoder(stage56 + MEGA mid encoder) | '
                             '189=SpectralEncoderRegularized(stage179 + ordinary dropout/capacity regularization) | '
                             '190=PSKPhasePrior | 191=QAMLatticePrior | 192=APSKRingPrior | '
                             '193=TopologyLoss | 194=FeatureTopologyAdapter | 195=SeparationConstraintLoss | '
                             '196=CyclicWienerResidual | '
                             '197=IQUBiMamba_EstimatedCycloFRESH | 198=IQUBiMamba_SafeAllStage | 199=IQUMamba_LowSNRSE | 200=IQUBiMamba_DirectionGated | 201=IQUBiMamba_LocalGlobalAllStage | 202=IQUMamba_ASGMamba | 203=IQUMamba_LowSNRAux | 204=IQUMamba_LowSNRSNRCond | 205=IQUMamba_LowSNRCyclicCond | 206=IQUBiMamba_DiffFusion | 207=IQUBiMamba_AdaptiveDiffFusion | 208=IQUMamba_NeuralWienerSE | 210=IQUMamba_MultiHypCyclicReliability | 211=IQUMamba_CrossSNRConsistency | 212=IQUMamba_ReceiverSymbol | 213=IQUMamba_CrossSNRReceiver | 217=IQUBiMamba_TimeReversalShared | 218=IQUBiMamba_AlternatingGlobalLocal | 219=IQUMamba_PRUNet | 220=IQUMamba_PRSharedPerm | 221=IQUMamba_PRRestrictedSkip | 225=IQUMamba_GaussianResidualPrior | '
                             '226=IQUMamba_AdaptiveMultiViewPrior | 227=IQUMamba_QAMSourcePrior | 228=IQUMamba_QAMMMAUnrolled | 229=IQUMamba_QAMDensityPrior | 230=IQUMamba_QAMTimingPrior | 231=IQUMamba_MultiViewConsistent | 232=IQUMamba_MultiViewPITOnly | 233=IQUMamba_NoiseMCOnly | 234=IQUMamba_PhaseEquivOnly | 235=BiMamba_CrossScaleSingle | 236=BiMamba_CrossScaleMulti | 237=BiMamba_CrossScaleEvidence | 238=IQUMamba_QAMTurboUnfold | 239=BiMamba_CrossScaleEstimatedCycloFRESH | 240=BiMamba_CrossScaleAligned | 241=BiMamba_CrossScaleMultiResKV | 242=BiMamba_CrossScaleBoundedChannel | '
                             '2=IQ Stage 2 | '
                             '31=DualDomainMambaSmall | '
                             '32=BiMamba_LK(Large-Kernel Stem) | '
                             '33=BiMamba_Jamba(BiMamba+Attention Hybrid) | '
                             '34=ConvNeXt1D(Large-Kernel Pure-CNN) | '
                             '35=BiMamba_AMR(Joint BSS+AMR) | '
                             '36=BiMamba_SoftDemod(Joint BSS+SoftDemod) | '
                             '37=BiMamba_SoftDemodV2(Receiver-aware Joint BSS+SoftDemod) | '
                             '38=BiMamba_SoftDemodV3(Offset/Phase-aware Joint BSS+SoftDemod) | '
                             '39=BiMamba_MCProj(BiMamba+MixtureConsistencyProjection) | '
                             '40=BiMamba_URIC(BiMamba+Unrolled Residual IC) | '
                             '41=BiMamba_URIC_AUG(URIC+lightweight RF train augmentation) | '
                             '42=ResUNet1D(Pure Convolutional U-Net Baseline) | '
                             '43=Transformer1D(Pure Transformer U-Net Baseline) | '
                             '44=BiMamba_CSB(Complex-valued Stem+Bottleneck) | '
                             '45=Transformer1D_Patch(Fixed patch/time token) | '
                             '46=Transformer1D_Patch_RoPE(Fixed patch/time token + RoPE) | '
                             '47=ResUNet1D_URIC(ResUNet + Unrolled Residual IC) | '
                             '48=BiMamba_CSB_URIC(CSB + Unrolled Residual IC) | '
                             '49=BiMamba_ADMM(ADMM-Unfolded Communication-Prior Separator) | '
                             '50=BiMamba_PGDU(PGD-Unfolded Communication-Prior Separator) | '
                             '51=BiMamba_FullComplex(Complex feature path + complex-wrapped BiMamba) | '
                             '52=BiMamba_GainPhase(Gain/Phase channel consistency) | '
                             '53=ComplexUNet1D(Pure complex convolutional U-Net baseline) | '
                             '54=RealUNet1D(Strict real-valued mirror of ComplexUNet1D) | '
                             '55=BiMamba_CSB_Scan(CSB+complex-coupled chunk gated scans) | '
                             '56=ResUNet1D_NoASC(ResUNet with plain skip concat) | '
                             '107=SkipEnhanced_Attention(Attention U-Net style skip) | '
                             '108=SkipEnhanced_UCTransNet(UCTransNet-lite style skip) | '
                             '109=SkipEnhanced_DCALite(DCA-lite channel+temporal cross-attention) | '
                             '110=UniversalPriorAdapter(Multi-hypothesis receiver prior bank) | '
                             '111=PulsePriorAdapter(RRC initialized learnable pulse-shaping prior) | '
                             '112=TimingPriorAdapter(Multi-hypothesis FFT-based fractional delay prior) | '
                             '113=PulseTimingPriorAdapter(Pulse + Timing adapters) | '
                             '114=LeakageSuppressedSkipGate(LSSG pure temporal multiplicative inhibition) | '
                             '115=ChannelWiseLSSG(Ablation C: LSSG with channel-wise gate) | '
                             '116=RefinedLSSG(Ablation D: LSSG with conv refinement) | '
                             '121=ChannelWiseLSSG_MS(CW-LSSG + Multi-Scale Gate) | '
                             '122=ChannelWiseLSSG_Context(CW-LSSG + Multi-Scale + Global Context) | '
                             '117=SRATCNBottleneck(Symbol-Rate-Aware Dilated TCN Bottleneck) | '
                             '118=ComplexASPPBottleneck(Complex ASPP-1D Bottleneck) | '
                             '119=IQUResUNet1D_BottleneckEnhanced(DCCB) | '
                             '120=MoEPriorAdapter(Mixture of Prior Experts with Router) | '
                             '123=StrongPriorAdapter(Strong End-to-End Residual Prior Adapter) | '
                             '124=DCCB_LSSG_Combined(Dual-Domain Bottleneck + Channel LSSG Skip) | '
                             '125=DCCB_LSSG_Partial_125(DCCB_Scaled + Partial LSSG Stages 2,3) | '
                             '126=DCCB_LSSG_Partial_126(DCCB_Scaled + Partial LSSG Stage 3) | '
                             '127=EncoderBiMamba_LSSG_Channel(EncoderBiMamba + CW-LSSG) | '
                             '128=Strategy_A_EncoderMamba(Encoder BiMamba + LSSG) | '
                             '129=Strategy_B_SkipMamba(Skip BiMamba + LSSG) | '
                             '130=Strategy_C_DecoderMamba(Decoder BiMamba + LSSG) | '
                             '131=Strategy_D_DualPathMambaBottleneck(DualPathMamba Bottleneck) | '
                             '132=QAMRadiusDirectedPriorAdapter(Soft RDE Prior Adapter for QAM) | '
                             '133=Strategy_E_OriginalBiMambaLayer(Original BiMambaLayer + LSSG_Channel) | '
                             '134=Strategy_F_OriginalFullBiMamba(Original BiMambaLayer in Encoder & Decoder) | '
                             '135=BiMamba_PGD_EQ(PGD Unfolded CMA/MMA Equalization) | '
                             '136=BiMamba_PhysicalChannel(CFO & Multipath Consistency) | '
                             '137=BiMamba_PhysicalChannel_PGDEQ(CFO/Multipath + PGD CMA/MMA) | '
                             '138=IQUResUNet1D_Bottleneck_DCCB_Mamba(DCCB + Encoder Mamba) | '
                             '139=IQUResUNet1D_Bottleneck_DCCB_Full_Mamba(DCCB + Encoder & Decoder Mamba) | '
                             '140=IQUResUNet1D_Bottleneck_DCCB_Unidirectional_Mamba(DCCB + Encoder & Decoder Unidirectional Mamba) | '
                             '141=IQUMamba1D_DWT(Non-parametric Haar DWT backbone + Shallow Mamba) | '
                             '142=IQUMamba1D_SIREN(Sine Periodic Activation for QAM Phase Tracking) | '
                             '143=ConformerGridNet(TF-GridNet-lite 2D Spectrogram Separator) | '
                             '144=BandSplitSeparator(Pure Band-Split Mamba Separator) | '
                             '145=IQUResUNet1D_MossFormer(U-Net with MossFormer-lite Bottleneck) | '
                             '146=IQUResUNet1D_ConvNeXt(Stage 119 with ConvNeXt1DBlock) | '
                             '147=IQUResUNet1D_MSCAN(Stage 119 with MSCAN1DBlock) | '
                             '148=IQUResUNet1D_HybridMamba(Stage 119 with Hybrid CNN-BiMamba Block) | '
                             '149=Ablation_CASPP_LSSG_Shallow(CASPP + LSSG stage 3 + alpha 0.02) | '
                             '150=Ablation_CASPP_LSSG_All(CASPP + LSSG all stages + alpha 0.02) | '
                             '151=Ablation_SRATCN_LSSG_Shallow(SRA-TCN + LSSG stage 3) | '
                             '152=Ablation_CASPP_Attn_Shallow(CASPP + Attn stage 3) | '
                             '57=BiMamba_CSB_CAG(CSB+scaled alpha+complex-aware channel gate) | '
                             '58=BiMamba_CSB_PhaseDiff(CSB+phase-difference guided scans) | '
                             '59=BiMamba_CSB_CMASC(CSB+complex mixture-consistent ASC) | '
                             '60=BiMamba_CSB_Constellation(CSB+soft constellation-guided refinement) | '
                             '61=BiMamba_LayerScale(learnable Mamba residual scale) | '
                             '62=BiMamba_LocalGlobal(gated local conv + global BiMamba) | '
                             '63=BiMamba_GLG(LocalGlobal + LayerScale) | '
                             '64=BiMamba-ComplexMask(local complex encoder + complex mask head) | '
                             '65=BiMamba_Stage4(stage-4 matched to IQUMamba stage 4) | '
                             '66=BiMamba_Stage4(stage-4 matched to IQUMamba stage 4, alias) | '
                             '67=IQUMamba_Stage5_320(original IQUMamba + one 320-channel stage) | '
                             '68=IQUMamba_DecoderMamba_Stage4(stage-4 IQUMamba + decoder Mamba) | '
                             '69=BiMamba_Stage4_URIC(stage-4 BiMamba + lightweight URIC) | '
                             '70=IQUMamba_RFScanFusion_Stage4(temporal+chunk+frequency scan fusion) | '
                             '71=IQUMamba_RFMambaScan_Stage4(frequency amplitude/phase scan gate) | '
                             '72=IQUMamba_RadMambaScan_Stage4(Conv1D shifted-chunk scan) | '
                             '73=IQUMamba_SymbolDualPath_Stage4(symbol-aligned intra/inter Mamba adapter) | '
                             '74=IQUMamba_ComplexMaskMC_Stage4(complex mask + mixture constraint) | '
                             '75=IQUMamba_NoiseAwareMC_Stage4(source+residual-noise mixture consistency) | '
                             '76=IQUMamba_ComplexAdapter_Stage4(local complex-aware boundary adapters) | '
                             '77=IQUMamba_CycloFRESH_Stage4(cyclostationary FRESH input adapter) | '
                             '78=IQUMamba_BlindCycloFRESH_Stage4(learnable cyclic-frequency FRESH input adapter) | '
                             '79=IQUMamba_EstimatedCycloFRESH_Stage4(mixture-estimated cyclic-frequency FRESH input adapter) | '
                             '80=IQUMamba_CyclicCorr_Stage4(mixture-estimated cyclic-correlation adapter) | '
                             '81=IQUMamba_CyclicCorrLeakCancel_Stage4(cyclic-correlation leakage cancellation head) | '
                             '82=IQUMamba_MultiPeakCycloFRESH_Stage4(multi-peak mixture-estimated FRESH input adapter) | '
                             '83=IQUMamba_SampleCycloFRESH_Stage4(per-sample mixture-estimated FRESH input adapter) | '
                             '84=IQUMamba_CycloFRESHFreqBias_Stage4(stage-79 FRESH plus high-frequency residual adapter) | '
                             '197=IQUBiMamba_EstimatedCycloFRESH_Stage4(stage-12 BiMamba + stage-79 estimated FRESH input adapter) | '
                             '198=IQUBiMamba_SafeAllStage_Stage4(stage-12 follow-up: BiMamba at all four encoder stages + residual scale) | '
                             '199=IQUMamba_LowSNRSE_Stage4(low-SNR enhancement front-end + stage-4 IQUMamba) | '
                             '202=IQUMamba_ASGMamba_Stage4(local FFT adaptive spectral gate before selected Mamba layers) | '
                             '203=IQUMamba_LowSNRAux_Stage4(low-SNR enhancement + clean/noise auxiliary loss) | '
                             '204=IQUMamba_LowSNRSNRCond_Stage4(SNR-proxy conditioned low-SNR enhancement) | '
                             '205=IQUMamba_LowSNRCyclicCond_Stage4(cyclic-reliability conditioned low-SNR enhancement) | '
                             '206=IQUBiMamba_DiffFusion_Stage4(symmetric plus direction-difference fusion, no fusion MLP) | '
                              '207=IQUBiMamba_AdaptiveDiffFusion_Stage4(reliability-gated direction-difference fusion, no attention) | '
                              '330=IQUMamba_RF_Mamba3_Trapezoidal(Stage295 + exponential-trapezoidal state input) | '
                              '331=IQUMamba_RF_Mamba3_CycloAnchor(Stage295 + bounded cyclostationary pole bank) | '
                              '332=IQUMamba_RF_Mamba3_Reliability(Stage295 + reliability-conditioned state time step) | '
                              '333=IQUMamba_RF_Mamba3_Combined(trapezoidal + cyclo anchor + reliability) | '
                              '334=IQUMamba_PhaseFoldedMamba(Stage4 + blind multi-period phase trajectories) | '
                              '347=IQUMamba_RealStateTrapReliability(Stage333 without complex pole rotation) | '
                              '348=IQUMamba_ReliabilitySelectiveS4D(S4D-Lin poles + reliability-controlled state time, fused scan) | '
                              '349=IQUMamba_ComplexRFMamba3UniRepLK(Stage342 without Estimated CycloFRESH) | '
                              '350=IQUMamba_FRESHComplexStemUniRepLK(Stage79 + Stage290 + Stage310) | '
                              '351=IQUMamba_ComplexStemRFMamba3(Stage290 + Stage333) | '
                              '352=IQUMamba_ComplexStemS4D(Stage290 + Stage336) | '
                              '353=IQUMamba_ComplexStemS4DUniRepLK(Stage290 + Stage336 + Stage310) | '
                              '359=IQUBiMamba_CrossScaleUniRepLK(Stage235 + Stage310) | '
                              '360=IQUMamba_CrossScaleUniRepLK(Stage300 + Stage310) | '
                              '365=IQUBiMamba_IndependentComplexStateUniRepLK(Stage364 + Stage310) | '
                              '376=Stage56_NoASC_RealLatentMask(Stage56 + Stage371-style per-scale real latent mask) | '
                              '377=Stage56_NoASC_ComplexStateUniRepLK_RealLatentMask(Stage56 plain decoder + complex-state BiMamba + UniRepLK + simplex mask) | '
                              '366=IQUMamba Stage4 + Cross-SNR EMA + Sync-Parameter FiLM | '
                              '367=IQUMamba Stage4 + Cross-SNR EMA only | '
                              '368=IQUMamba Stage4 + Sync-Parameter FiLM only | '
                             '208=IQUMamba_NeuralWienerSE_Stage4(neural Wiener signal/noise power gate for low-SNR mixtures) | '
                             '210=IQUMamba_MultiHypCyclicReliability_Stage4(null-aware multi-hypothesis cyclic evidence adapter) | '
                             '211=IQUMamba_CrossSNRConsistency_Stage4(training-only complementary-SNR teacher consistency) | '
                             '212=IQUMamba_ReceiverSymbol_Stage4(training-only RRC matched-filter symbol supervision) | '
                             '217=IQUBiMamba_TimeReversalShared_Stage4(shared-core reversal-equivariant uncertainty-shrunk fusion) | '
                             '218=IQUBiMamba_AlternatingGlobalLocal_Stage4(one global scan plus opposite-direction depthwise local context) | '
                             '219=IQUMamba_PRUNet_Stage4(perfect-reconstruction Haar polyphase sampling) | '
                             '220=IQUMamba_PRSharedPerm_Stage4(training-only shared-permutation multi-scale supervision) | '
                             '221=IQUMamba_PRRestrictedSkip_Stage4(bounded stochastic shallow skip) | '
                             '222=IQUMamba_EvidenceRoutedMoE_Stage4(identity/local/periodic/leakage residual experts with evidence routing) | '
                             '223=IQUMamba_ResidualNoiseContrastive_Stage4(training-only source-vs-residual-noise patch prior) | '
                             '224=IQUMamba_BlindSyncFactorized_Stage4(mixture-only local synchronization factorization) | '
                             '226=IQUMamba_AdaptiveMultiViewPrior_Stage4(mixture-only evidence-routed cyclic/phase/pulse views with identity fallback) | '
                             '227=IQUMamba_QAMSourcePrior_Stage4(coarse separation followed by source-wise soft 16/64/128-QAM geometry refinement) | '
                             '228=IQUMamba_QAMMMAUnrolled_Stage4(source-wise multi-modulus QAM equalization unfolding) | '
                             '229=IQUMamba_QAMDensityPrior_Stage4(source-wise constellation-density/topology conditioning) | '
                             '230=IQUMamba_QAMTimingPrior_Stage4(blind RRC timing and SPS hypothesis routing) | 231=IQUMamba_MultiViewConsistent_Stage4(shared-IQ norm + noise MC + shared-permutation paired views) | '
                             '232=IQUMamba_MultiViewPITOnly_Stage4(shared-permutation cross-SNR views only) | 233=IQUMamba_NoiseMCOnly_Stage4(time-domain noise-aware mixture consistency only) | 234=IQUMamba_PhaseEquivOnly_Stage4(shared-IQ norm + phase equivariance only) | '
                             '235=BiMamba_CrossScaleSingle_Stage12(compressed bottleneck KV to stage-2 queries) | 236=BiMamba_CrossScaleMulti_Stage12(compressed bottleneck KV to stage-1/2 queries) | 237=BiMamba_CrossScaleEvidence_Stage12(multi-scale cross-attention + physical-evidence gates) | '
                             '238=IQUMamba_QAMTurboUnfold_Stage4(joint soft-QAM, complex-channel solve, and interference-cancellation unfolding) | '
                             '239=BiMamba_CrossScaleEstimatedCycloFRESH_Stage12(stage-235 cross-scale attention + stage-79 estimated FRESH input adapter) | '
                             '240=BiMamba_CrossScaleAligned_Stage12(stage-235 with aligned KV windows) | 241=BiMamba_CrossScaleMultiResKV_Stage12(stage-235 with gated 32/128-token KV banks) | 242=BiMamba_CrossScaleBoundedChannel_Stage12(stage-235 with bounded channel-wise residual gates) | '
                             '251=BiMamba_PhysicalRoutedEnhancedCrossAttention_Stage12(stage-245 memory -> stage-235 injection, routed by stage-244 physical evidence) | '
                             '252=BiMamba_UnifiedPhysicalGlobalKV | 253=BiMamba_PhysicalFiLMGlobalMemory | 254=BiMamba_ScaleIsolatedPhysicalFusion | 255=BiMamba_IdentityAwarePhysicalMoE | 256=BiMamba_CrossGatedDualMemory | '
                             '257=ICASSPWaveNet_Mamba(post-skip unidirectional Mamba fusion) | 258=ICASSPWaveNet_BiMamba(post-skip bidirectional Mamba fusion) | '
                             '259=ICASSPWaveNet_MultiRateMamba(WaveNet-10 then stride-4 unidirectional Mamba) | 260=ICASSPWaveNet_MultiRateBiMamba(WaveNet-10 then stride-4 bidirectional Mamba) | '
                             '261=ICASSPWaveNet_InterleavedMamba(WaveNet-10 -> stride-4 Mamba -> WaveNet-10) | 262=ICASSPWaveNet_InterleavedBiMamba(WaveNet-10 -> stride-4 BiMamba -> WaveNet-10) | '
                             '263=UNet_ComputeMatched_68(strict official UNet with k=68) | 264=IQResDilatedUNet(learned analysis, gated dilation, bottleneck BiMamba) | '
                              '265=ICASSPWaveNet_261_Stage235CrossScaleBiMamba | 266=ICASSPWaveNet_261_Stage255PhysicalMoE_BiMamba | 267=ICASSPWaveNet_261_Stage79CycloFRESH | '
                              '268=ICASSPWaveNet_261_GatedBiMamba(forward Mamba + gated reverse correction) | '
                              '269=ICASSPWaveNet_MambaFiLMController(Mamba controls late FiLM/residual/skip paths) | '
                              '270=ICASSPWaveNet_MambaDilationSkipRouter(frame-conditioned late dilation/skip routing) | '
                              '271=ICASSPWaveNet_PhaseAwareReverseMamba(Stage-261 plus conj(flip(I/Q)) reverse branch) | '
                              '272=ICASSPWaveNet_Stage235Memory(Stage-235 K/V memory replaces Stage-261 Mamba context) | '
                              '273=ICASSPWaveNet20_NoMamba(direct Stage-261 architectural control) | '
                              '274=ICASSPWaveNet15_InterleavedMamba(WaveNet-10 -> stride-4 UniMamba -> WaveNet-5) | '
                              '275=ICASSPWaveNet15_NoMamba(compute/depth control for Stage 274) | '
                              '276=ICASSPWaveNet_5_ChunkMamba_5(strong local/global fusion) | '
                              '277=ICASSPWaveNet_10_ChunkMamba_10(strong local/global fusion) | '
                              '378=KUTIIDualSourceWaveNet(shared 30-block learnable-dilation WaveNet + source slots) | '
                              '213=IQUMamba_CrossSNRReceiver_Stage4(combined cross-SNR and receiver-domain training) | '
                             '214=IQUMamba_ConfidenceSoftPIT_Stage4(SNR/epoch-adaptive probabilistic PIT, zero inference cost) | '
                             '215=IQUBiMamba_ComplexDiffShared_Stage4(bottleneck shared-core absolute/complex-difference scans) | '
                             '216=IQUMamba_CumulantPrior_Stage4(training-only modulation-agnostic fourth-order cumulant prior) | '
                             '85=IQUMamba_BlindStatFiLM_Stage4(mixture-only blind-stat feature FiLM) | '
                             '86=IQUMamba_BlindStatInput_Stage4(mixture-only blind-stat input adapter) | '
                             '87=IQUMamba_DemodAwareLoss_Stage4(original stage-4 model, use PIT-DEMOD-AWARE loss) | '
                             '88=IQUMamba_FeatureComplexMask_Stage4(learned complex feature mask output) | '
                             '89=RF-BandSCNet(complex STFT band-split spectral separator) | '
                             '90=Complex-DPNet(learned complex encoder + dual-path separator) | '
                             '91=Complex-ConvTasNet(learned complex filterbank + dilated TCN masks) | '
                             '92=Complex-SourceSlotNet(direct complex source-slot separator) | '
                             '93=Complex-AttractorNet(TF-bin embedding + source attractors) | '
                             '94=MultiRes-STFTMaskNet(multi-resolution complex spectral masks) | '
                             '95=IQUMamba_KnowledgeESD_Stage4(source-slot refinement + mixture projection) | '
                             '96=IQUMamba_BlindMultiRateInput_Stage4(mixture-only multi-rate input adapter) | '
                             '97=ResUNet_MambaBottleneck(stage42 bottleneck Mamba adapter) | '
                             '98=ResUNet_MambaLocalGlobal(stage42 gated local/global Mamba) | '
                             '99=ResUNet_MambaDualGate(stage42 temporal/channel Mamba gate) | '
                             '100=ResUNet_PhaseEq(stage42 phase-equivariant complex adapter) | '
                             '101=ResUNet_CorrGate(stage42 local complex-correlation skip gate) | '
                             '102=ResUNet_PCO(stage42 phase/correlation/orthogonal output) | '
                             '103=SepBambaUNet1D(4-stage SepMamba U-Net) | '
                             '104=ResUNet1D_GatedSkip(ResUNet with Decoder-Guided Gated Skip) | '
                             '105=ResUNet1D_WLComplex(ResUNet with Widely-Linear stem and Complex Mask) | '
                             '106=ResUNet1D_TFBranch(Time-Frequency Dual-Branch ResUNet) | '
                             '153=ESDMaskWrapper_SkipEnhanced(ESD+LSSG) | '
                             '154=resunet1d_skip_enhanced_lssg_dw(LSSG depthwise local gate) | '
                             '155=resunet1d_crossscale_lssg(UNet3+/UCTransNet cross-scale LSSG) | '
                             '156=resunet1d_sk_lssg(SKNet selective-kernel LSSG) | '
                             '157=resunet1d_freq_lssg(FcaNet/GFNet frequency-conditioned LSSG) | '
                             '158=resunet1d_focal_lssg(FocalNet gated-context LSSG) | '
                             '159=resunet1d_wavelet_dccb(WTConv-style wavelet bottleneck) | '
                             '160=resunet1d_complex_cyclo_dccb(complex autocorr/pseudo-corr DCCB)'
                        ))
    parser.add_argument('--num_sources', type=int, default=2, choices=[2, 3],
                       help='Fixed source count for this run: 2 or 3.')
    parser.add_argument('--source_names', nargs='+', type=str, default=['S1', 'S2'],
                       help='One name per source; must contain exactly --num_sources entries.')
    parser.add_argument('--eval_pit_metric', type=str, default='si_snr_complex',
                       choices=['none', 'si_snr_complex', 'si_snr_real', 'mse'],
                       help=('Source alignment used for primary validation/test metrics; '
                             'none keeps fixed output order and also prints supplemental '
                             'PIT(si_snr_complex) metrics'))
    parser.add_argument('--report_phase_flip', action='store_true',
                       help='Report source-level phase flip rate for each final SNR-bin evaluation')
    parser.add_argument('--phase_flip_tolerance_deg', type=float, default=45.0,
                       help='Count a phase flip when global phase is within this many degrees of +/-180')
    parser.add_argument('--phase_flip_min_sc', type=float, default=0.0,
                       help='Optional minimum complex similarity coefficient required before counting a phase flip')
    parser.add_argument('--phase_flip_mode', type=str, default='either',
                       choices=['phase', 'sign', 'either'],
                       help='Phase flip detector: phase=near +/-180deg, sign=negative real SI-SNR alpha, either=union')

    # Lightweight BiMamba gated variant overrides (stages 61/62/63)
    parser.add_argument('--mamba_residual_scale_init', type=float, default=None,
                       help='Override initial learnable BiMamba residual scale in stages 61/63')
    parser.add_argument('--bimamba_apply_stages', nargs='+', type=int, default=None,
                       help='Override encoder stage indices that use BiMamba in stages 198/200/201/206/207')
    parser.add_argument('--bimamba_residual_scale_init', type=float, default=None,
                       help='Override initial residual scale for safe/lightweight BiMamba stages')
    parser.add_argument('--bimamba_diff_scale_init', type=float, default=None,
                       help='Override initial direction-difference scale in stages 206/207')
    parser.add_argument('--bimamba_gate_logit_init', type=float, default=None,
                       help='Override initial per-channel direction gate logit in stage 207')
    parser.add_argument('--bimamba_gate_token_scale_init', type=float, default=None,
                       help='Override token reliability contribution to the direction gate in stage 207')
    parser.add_argument('--bimamba_gate_eps', type=float, default=None,
                       help='Override numerical epsilon for direction-reliability normalization in stage 207')
    parser.add_argument('--bimamba_complex_diff_gate_init', type=float, default=None,
                       help='Override initial differential-view gate in stage 215')
    parser.add_argument('--bimamba_complex_diff_stride', type=int, default=None,
                       help='Override differential-view temporal stride in stage 215')
    parser.add_argument('--bimamba_complex_diff_eps', type=float, default=None,
                       help='Override complex differential epsilon in stage 215')
    parser.add_argument('--bimamba_boundary_tau_init', type=float, default=None,
                       help='Override boundary reliability time constant in stage 217')
    parser.add_argument('--bimamba_shrinkage_init', type=float, default=None,
                       help='Override forward/backward disagreement shrinkage in stage 217')
    parser.add_argument('--bimamba_fusion_eps', type=float, default=None,
                       help='Override robust fusion numerical epsilon in stage 217')
    parser.add_argument('--bimamba_local_kernel_size', type=int, default=None,
                       help='Override opposite-direction local kernel size in stage 218')
    parser.add_argument('--bimamba_local_gate_init', type=float, default=None,
                       help='Override opposite-direction local gate in stage 218')
    parser.add_argument('--cross_scale_query_stages', nargs='+', type=int, default=None,
                       help='Override query skip stages for stages 235-242')
    parser.add_argument('--cross_scale_global_stage', type=int, default=None,
                       help='Override the global KV encoder stage for stages 235-242')
    parser.add_argument('--cross_scale_kv_tokens', type=int, default=None,
                       help='Override compressed global KV token count for stages 235-242')
    parser.add_argument('--cross_scale_num_heads', type=int, default=None,
                       help='Override cross-attention head count for stages 235-242')
    parser.add_argument('--cross_scale_dropout', type=float, default=None,
                       help='Override cross-attention dropout for stages 235-242')
    parser.add_argument('--cross_scale_residual_scale_init', type=float, default=None,
                       help='Override initial cross-scale residual strength for stages 235-242')
    parser.add_argument('--cross_scale_evidence_hidden', type=int, default=None,
                       help='Override physical-evidence gate hidden width in stage 237')
    parser.add_argument('--cross_scale_evidence_eps', type=float, default=None,
                       help='Override physical-evidence numerical epsilon in stage 237')
    parser.add_argument('--cross_scale_evidence_gate', action='store_true', default=None,
                       help='Enable mixture physical-evidence gates for cross-scale attention')
    parser.add_argument('--cross_scale_no_evidence_gate', action='store_true', default=None,
                       help='Disable mixture physical-evidence gates for cross-scale attention')
    parser.add_argument('--cross_scale_variant', type=str, default=None,
                       choices=['aligned', 'multires_kv', 'bounded_channel_gate'],
                       help='Override advanced Stage-235 fusion variant for stages 240-242')
    parser.add_argument('--cross_scale_aligned_window_radius', type=int, default=None,
                       help='Aligned local-KV window radius in stage 240')
    parser.add_argument('--cross_scale_aligned_global_tokens', type=int, default=None,
                       help='Always-visible global summary token count in stage 240')
    parser.add_argument('--cross_scale_coarse_kv_tokens', type=int, default=None,
                       help='Coarse KV token count in stage 241')
    parser.add_argument('--cross_scale_fine_kv_tokens', type=int, default=None,
                       help='Fine KV token count in stage 241')
    parser.add_argument('--cross_scale_multires_gate_hidden', type=int, default=None,
                       help='Resolution-routing hidden width in stage 241')
    parser.add_argument('--cross_scale_bounded_max_scale', type=float, default=None,
                       help='Maximum per-channel residual scale in stage 242')
    parser.add_argument('--cross_scale_bounded_initial_scale', type=float, default=None,
                       help='Initial per-channel residual scale in stage 242')
    parser.add_argument('--cross_scale_channel_gate_hidden', type=int, default=None,
                       help='Channel-gate hidden width in stage 242')
    parser.add_argument('--shared_permutation_multiscale_enable', action='store_true', default=None,
                       help='Enable final-output permutation locking for multi-scale decoder supervision')
    parser.add_argument('--shared_permutation_multiscale_weight', type=float, default=None,
                       help='Weight of the stage-220 shared-permutation auxiliary loss')
    parser.add_argument('--shared_permutation_multiscale_weights', nargs='+', type=float, default=None,
                       help='Per-decoder-scale weights for stage-220 supervision')
    parser.add_argument('--shallow_skip_init', type=float, default=None,
                       help='Initial bounded shallow-skip scale in stage 221')
    parser.add_argument('--shallow_skip_drop_probability', type=float, default=None,
                       help='Training-only whole-path shallow-skip drop probability in stage 221')
    parser.add_argument('--evidence_moe_hidden_channels', type=int, default=None,
                       help='Hidden width of the shared residual experts in stage 222')
    parser.add_argument('--evidence_moe_max_delta', type=float, default=None,
                       help='Bounded residual amplitude relative to mixture RMS in stage 222')
    parser.add_argument('--evidence_moe_identity_bias', type=float, default=None,
                       help='Initial router bias for the identity candidate in stage 222')
    parser.add_argument('--evidence_moe_router_temperature', type=float, default=None,
                       help='Soft route temperature during stage-222 training')
    parser.add_argument('--evidence_moe_route_hard_eval', action='store_true', default=None,
                       help='Use a single selected candidate at stage-222 evaluation')
    parser.add_argument('--evidence_moe_lag_bank', nargs='+', type=int, default=None,
                       help='Positive lag bank used by the modulation-agnostic evidence extractor')
    parser.add_argument('--evidence_moe_return_route_aux', action='store_true', default=None,
                       help='Return candidate and route diagnostics during stage-222 training')
    parser.add_argument('--evidence_moe_route_supervision_enable', action='store_true', default=None,
                       help='Enable training-only counterfactual route supervision')
    parser.add_argument('--evidence_moe_route_supervision_disable', action='store_true',
                       help='Disable counterfactual route supervision from the stage config')
    parser.add_argument('--evidence_moe_route_loss_weight', type=float, default=None,
                       help='Weight of counterfactual route supervision')
    parser.add_argument('--evidence_moe_route_target_temperature', type=float, default=None,
                       help='Temperature for the detached candidate-quality route target')
    parser.add_argument('--stage255_snr_aux_weight', type=float, default=None,
                       help='Weight of the Stage-255 mixture-only SNR regression objective')
    parser.add_argument('--stage255_snr_aux_min_db', type=float, default=None,
                       help='Lower normalization bound for Stage-255 SNR regression')
    parser.add_argument('--stage255_snr_aux_max_db', type=float, default=None,
                       help='Upper normalization bound for Stage-255 SNR regression')
    parser.add_argument('--stage255_snr_curriculum_enable', action='store_true', default=None,
                       help='Progressively admit lower-SNR training samples in Stage 255')
    parser.add_argument('--stage255_snr_curriculum_disable', action='store_true',
                       help='Disable the Stage-255 SNR curriculum from the stage config')
    parser.add_argument('--stage255_snr_curriculum_start_db', type=float, default=None,
                       help='Initial minimum admitted SNR for the Stage-255 curriculum')
    parser.add_argument('--stage255_snr_curriculum_end_db', type=float, default=None,
                       help='Final minimum admitted SNR for the Stage-255 curriculum')
    parser.add_argument('--stage255_snr_curriculum_fraction', type=float, default=None,
                       help='Fraction of training over which Stage-255 lowers its SNR floor')
    parser.add_argument('--stage255_router_warmup_epochs', type=int, default=None,
                       help='Epochs that train only the Stage-255 router and condition encoder')
    parser.add_argument('--stage255_expert_pretrain_epochs', type=int, default=None,
                       help='Epochs that pretrain Stage-255 experts with the router frozen')
    parser.add_argument('--stage255_router_joint_lr_scale', type=float, default=None,
                       help='Router gradient scale after Stage-255 joint fine-tuning starts')
    parser.add_argument('--stage255_trust_enable', action='store_true',
                       help='Enable the Stage-255 Physical/Joint trust penalty')
    parser.add_argument('--stage255_trust_disable', action='store_true',
                       help='Disable the Stage-255 Physical/Joint trust penalty')
    parser.add_argument('--stage255_condition_enable', action='store_true',
                       help='Enable Stage-255 mixture condition routing and SNR prediction')
    parser.add_argument('--stage255_condition_disable', action='store_true',
                       help='Disable Stage-255 mixture condition routing and SNR prediction')
    parser.add_argument('--stage255_counterfactual_enable', action='store_true',
                       help='Enable Stage-255 training-only candidate decoding')
    parser.add_argument('--stage255_counterfactual_disable', action='store_true',
                       help='Disable Stage-255 training-only candidate decoding')
    parser.add_argument(
        '--stage255_route_candidate_probability',
        dest='fusion_route_candidate_probability',
        type=float,
        default=None,
        help='Probability of four-candidate decoding on counterfactual training batches',
    )
    parser.add_argument('--noise_prior_hidden', type=int, default=None,
                       help='Hidden width of stage-223 training-only patch projector')
    parser.add_argument('--noise_prior_embedding', type=int, default=None,
                       help='Embedding width of stage-223 training-only patch projector')
    parser.add_argument('--noise_prior_patch_size', type=int, default=None,
                       help='Default local patch size used by the stage-223 projector')
    parser.add_argument('--noise_prior_patch_stride', type=int, default=None,
                       help='Default local patch stride used by the stage-223 projector')
    parser.add_argument('--noise_contrastive_prior_enable', action='store_true', default=None,
                       help='Enable stage-223 residual-noise contrastive training objective')
    parser.add_argument('--noise_contrastive_prior_weight', type=float, default=None,
                       help='Weight of the stage-223 residual-noise contrastive objective')
    parser.add_argument('--noise_contrastive_prior_patch_size', type=int, default=None,
                       help='Patch size for stage-223 residual-noise contrastive loss')
    parser.add_argument('--noise_contrastive_prior_patch_stride', type=int, default=None,
                       help='Patch stride for stage-223 residual-noise contrastive loss')
    parser.add_argument('--noise_contrastive_prior_temperature', type=float, default=None,
                       help='Temperature of stage-223 source-vs-residual contrastive logits')
    parser.add_argument('--noise_contrastive_prior_residual_weight', type=float, default=None,
                       help='Small residual reconstruction term in stage 223')
    parser.add_argument('--noise_contrastive_prior_gate_floor', type=float, default=None,
                       help='Residual-power gate floor for stage 223')
    parser.add_argument('--sync_hidden', type=int, default=None,
                       help='Hidden width of stage-224 local synchronization branch')
    parser.add_argument('--sync_kernel_size', type=int, default=None,
                       help='Kernel size of stage-224 local synchronization branch')
    parser.add_argument('--sync_scale_init', type=float, default=None,
                       help='Initial bounded residual scale in stage 224')
    parser.add_argument('--sync_lags', nargs='+', type=int, default=None,
                       help='Positive lag bank for stage-224 blind synchronization evidence')
    parser.add_argument('--sync_eps', type=float, default=None,
                       help='Numerical epsilon for stage-224 synchronization evidence')
    parser.add_argument('--local_kernel_size', type=int, default=None,
                       help='Override local depthwise-conv kernel size in stages 62/63')
    parser.add_argument('--local_global_gate_hidden', type=int, default=None,
                       help='Override local/global fusion gate hidden size in stages 62/63')

    # RF/Radar-inspired IQUMamba scan variant overrides (stages 70/71/72)
    parser.add_argument('--rfscan_chunk_size', type=int, default=None,
                       help='Override shifted chunk length in RFScan/RadMamba stages 70/72')
    parser.add_argument('--rfscan_shift_size', type=int, default=None,
                       help='Override shifted chunk offset in RFScan/RadMamba stages 70/72; default is half chunk')
    parser.add_argument('--rfscan_freq_bands', type=int, default=None,
                       help='Override number of pooled FFT bands in RFScan/RFMamba stages 70/71')
    parser.add_argument('--rfscan_gate_hidden', type=int, default=None,
                       help='Override RFScan frequency/branch gate hidden size in stages 70/71')
    parser.add_argument('--rfscan_conv_kernel_size', type=int, default=None,
                       help='Override Conv1D token projection kernel size in RFScan/RadMamba stages 70/72')
    parser.add_argument('--rfscan_residual_scale_init', type=float, default=None,
                       help='Override initial residual scale for RFScan layers in stages 70/71/72')
    parser.add_argument('--rfscan_condition_scale_init', type=float, default=None,
                       help='Override initial raw-IQ physics condition injection scale in stages 70/71/72')
    parser.add_argument('--rfscan_stft_n_fft', type=int, default=None,
                       help='Override raw-IQ STFT n_fft for RFScan/RadMamba physical conditioner')
    parser.add_argument('--rfscan_stft_hop_length', type=int, default=None,
                       help='Override raw-IQ STFT hop length for RFScan/RadMamba physical conditioner')
    parser.add_argument('--rfscan_stft_win_length', type=int, default=None,
                       help='Override raw-IQ STFT window length for RFScan/RadMamba physical conditioner')
    parser.add_argument('--rfscan_stft_freq_bins', type=int, default=None,
                       help='Override pooled frequency bins in raw-IQ STFT conditioner')

    # Symbol-aligned dual-path IQUMamba overrides (stage 73)
    parser.add_argument('--symbol_samples', type=int, default=None,
                       help='Override samples per symbol used by stage 73 symbol-aligned chunking')
    parser.add_argument('--dual_path_chunk_symbols', type=int, default=None,
                       help='Override number of symbols per dual-path chunk in stage 73')
    parser.add_argument('--dual_path_hop_symbols', type=int, default=None,
                       help='Override number of symbols per dual-path hop in stage 73')
    parser.add_argument('--dual_path_residual_scale_init', type=float, default=None,
                       help='Override initial residual scale for stage 73 dual-path adapter')

    # Complex-mask mixture-consistent IQUMamba overrides (stage 74)
    parser.add_argument('--mask_bound', type=float, default=None,
                       help='Override tanh mask bound for stage 74 complex masks; <=0 disables bounding')
    parser.add_argument('--mask_logit_scale_init', type=float, default=None,
                       help='Override initial scale applied to stage 74 mask logits')
    parser.add_argument('--mask_sum_constraint', action='store_true',
                       help='Enable complex mask sum constraint in stage 74')
    parser.add_argument('--mask_no_sum_constraint', action='store_true',
                       help='Disable complex mask sum constraint in stage 74')
    parser.add_argument('--mask_apply_projection', action='store_true',
                       help='Enable exact mixture-consistency projection after stage 74 masks')
    parser.add_argument('--mask_skip_projection', action='store_true',
                       help='Disable exact mixture-consistency projection after stage 74 masks')
    parser.add_argument('--mask_project_deep_supervision', action='store_true',
                       help='Project every deep-supervision output in stage 74')
    parser.add_argument('--mask_project_final_only', action='store_true',
                       help='Project only the full-resolution output in stage 74')

    # Learned complex feature-mask IQUMamba overrides (stage 88)
    parser.add_argument('--feature_mask_channels', type=int, default=None,
                       help='Override learned complex feature channels in stage 88')
    parser.add_argument('--feature_mask_kernel_size', type=int, default=None,
                       help='Override complex feature encoder/decoder kernel size in stage 88')
    parser.add_argument('--feature_mask_bound', type=float, default=None,
                       help='Override tanh mask bound for stage 88 feature masks; <=0 disables bounding')
    parser.add_argument('--feature_mask_logit_scale_init', type=float, default=None,
                       help='Override initial scale applied to stage 88 feature-mask logits')
    parser.add_argument('--feature_mask_sum_constraint', action='store_true',
                       help='Enable complex feature-mask sum constraint in stage 88')
    parser.add_argument('--feature_mask_no_sum_constraint', action='store_true',
                       help='Disable complex feature-mask sum constraint in stage 88')
    parser.add_argument('--feature_mask_apply_projection', action='store_true',
                       help='Enable waveform mixture-consistency projection after stage 88 decoding')
    parser.add_argument('--feature_mask_skip_projection', action='store_true',
                       help='Disable waveform mixture-consistency projection after stage 88 decoding')
    parser.add_argument('--feature_mask_project_deep_supervision', action='store_true',
                       help='Project every deep-supervision output in stage 88')
    parser.add_argument('--feature_mask_project_final_only', action='store_true',
                       help='Project only the full-resolution output in stage 88')
    parser.add_argument('--feature_mask_identity_init', action='store_true',
                       help='Use identity initialization for the stage 88 complex feature bank')
    parser.add_argument('--feature_mask_no_identity_init', action='store_true',
                       help='Disable identity initialization for the stage 88 complex feature bank')

    # Knowledge-embedded source-slot IQUMamba overrides (stage 95)
    parser.add_argument('--source_slot_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 95 source-slot residual refiner')
    parser.add_argument('--source_slot_kernel_size', type=int, default=None,
                       help='Override kernel size of the stage 95 source-slot residual refiner')
    parser.add_argument('--source_slot_residual_scale_init', type=float, default=None,
                       help='Override initial residual scale of the stage 95 source-slot refiner')
    parser.add_argument('--source_slot_zero_init', action='store_true',
                       help='Zero-initialize the final source-slot residual layer in stage 95')
    parser.add_argument('--source_slot_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final source-slot residual layer in stage 95')
    parser.add_argument('--source_slot_refine_deep_supervision', action='store_true',
                       help='Refine every deep-supervision output in stage 95')
    parser.add_argument('--source_slot_final_only', action='store_true',
                       help='Refine only the full-resolution output in stage 95')
    parser.add_argument('--source_slot_apply_train', action='store_true',
                       help='Enable source-slot refinement during training in stage 95')
    parser.add_argument('--source_slot_skip_train', action='store_true',
                       help='Disable source-slot refinement during training in stage 95')
    parser.add_argument('--source_slot_apply_eval', action='store_true',
                       help='Enable source-slot refinement during evaluation in stage 95')
    parser.add_argument('--source_slot_skip_eval', action='store_true',
                       help='Disable source-slot refinement during evaluation in stage 95')

    # Noise-aware mixture-consistent IQUMamba overrides (stage 75)
    parser.add_argument('--noise_mc_source_weight', type=float, default=None,
                       help='Override source residual share in stage 75 noise-aware projection')
    parser.add_argument('--noise_mc_noise_weight', type=float, default=None,
                       help='Override residual-noise share in stage 75 noise-aware projection')
    parser.add_argument('--noise_head_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 75 residual-noise head')
    parser.add_argument('--noise_head_kernel_size', type=int, default=None,
                       help='Override kernel size of the stage 75 residual-noise head')
    parser.add_argument('--noise_mc_eps', type=float, default=None,
                       help='Override numerical epsilon of the stage 75 noise-aware projection')
    parser.add_argument('--noise_mc_apply_projection', action='store_true',
                       help='Enable stage 75 noise-aware projection')
    parser.add_argument('--noise_mc_skip_projection', action='store_true',
                       help='Disable stage 75 noise-aware projection')
    parser.add_argument('--noise_mc_project_during_train', action='store_true',
                       help='Enable stage 75 noise-aware projection during training')
    parser.add_argument('--noise_mc_skip_train_projection', action='store_true',
                       help='Disable stage 75 noise-aware projection during training')
    parser.add_argument('--noise_mc_project_during_eval', action='store_true',
                       help='Enable stage 75 noise-aware projection during evaluation')
    parser.add_argument('--noise_mc_skip_eval_projection', action='store_true',
                       help='Disable stage 75 noise-aware projection during evaluation')
    parser.add_argument('--noise_head_zero_init', action='store_true',
                       help='Zero-initialize the final residual-noise head layer in stage 75')
    parser.add_argument('--noise_head_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final residual-noise head layer in stage 75')

    # Low-SNR enhancement-front-end IQUMamba overrides (stage 199)
    parser.add_argument('--low_snr_se_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 199 low-SNR enhancement front-end')
    parser.add_argument('--low_snr_se_kernel_size', type=int, default=None,
                       help='Override convolution kernel size of the stage 199 low-SNR enhancement front-end')
    parser.add_argument('--low_snr_se_scale_init', type=float, default=None,
                       help='Override initial residual scale of the stage 199 low-SNR enhancement front-end')
    parser.add_argument('--low_snr_se_source_weight', type=float, default=None,
                       help='Override source residual share in the stage 199 noisy-mixture projection')
    parser.add_argument('--low_snr_se_noise_weight', type=float, default=None,
                       help='Override residual-noise share in the stage 199 noisy-mixture projection')
    parser.add_argument('--low_snr_se_eps', type=float, default=None,
                       help='Override numerical epsilon of the stage 199 noisy-mixture projection')
    parser.add_argument('--low_snr_se_zero_init', action='store_true',
                       help='Zero-initialize the final enhancement layer in stage 199')
    parser.add_argument('--low_snr_se_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final enhancement layer in stage 199')
    parser.add_argument('--low_snr_se_use_projection', action='store_true',
                       help='Enable stage 199 noisy-mixture projection')
    parser.add_argument('--low_snr_se_skip_projection', action='store_true',
                       help='Disable stage 199 noisy-mixture projection')
    parser.add_argument('--low_snr_se_project_during_train', action='store_true',
                       help='Enable stage 199 noisy-mixture projection during training')
    parser.add_argument('--low_snr_se_skip_train_projection', action='store_true',
                       help='Disable stage 199 noisy-mixture projection during training')
    parser.add_argument('--low_snr_se_project_during_eval', action='store_true',
                       help='Enable stage 199 noisy-mixture projection during evaluation')
    parser.add_argument('--low_snr_se_skip_eval_projection', action='store_true',
                       help='Disable stage 199 noisy-mixture projection during evaluation')
    parser.add_argument('--low_snr_se_return_aux', action='store_true',
                       help='Return stage 199 enhancement diagnostics as an aux dictionary')
    parser.add_argument('--low_snr_se_no_aux', action='store_true',
                       help='Return only separated sources from stage 199')

    # Conditioned low-SNR enhancement front-end IQUMamba overrides (stages 204-205)
    parser.add_argument('--low_snr_cond_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 204/205 conditioned low-SNR front-end')
    parser.add_argument('--low_snr_cond_kernel_size', type=int, default=None,
                       help='Override convolution kernel size of the stage 204/205 conditioned low-SNR front-end')
    parser.add_argument('--low_snr_cond_gate_hidden', type=int, default=None,
                       help='Override conditioning gate hidden size of the stage 204/205 low-SNR front-end')
    parser.add_argument('--low_snr_cond_scale_init', type=float, default=None,
                       help='Override initial residual scale of the stage 204/205 conditioned low-SNR front-end')
    parser.add_argument('--low_snr_cond_min_freq', type=float, default=None,
                       help='Override minimum cyclic frequency for stage 205 cyclic-reliability conditioning')
    parser.add_argument('--low_snr_cond_max_freq', type=float, default=None,
                       help='Override maximum cyclic frequency for stage 205 cyclic-reliability conditioning')
    parser.add_argument('--low_snr_cond_source_weight', type=float, default=None,
                       help='Override source residual share in the stage 204/205 noisy-mixture projection')
    parser.add_argument('--low_snr_cond_noise_weight', type=float, default=None,
                       help='Override residual-noise share in the stage 204/205 noisy-mixture projection')
    parser.add_argument('--low_snr_cond_eps', type=float, default=None,
                       help='Override numerical epsilon in the stage 204/205 conditioned low-SNR front-end')
    parser.add_argument('--low_snr_cond_zero_init', action='store_true',
                       help='Zero-initialize the final conditioned enhancement layer in stage 204/205')
    parser.add_argument('--low_snr_cond_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final conditioned enhancement layer in stage 204/205')
    parser.add_argument('--low_snr_cond_use_projection', action='store_true',
                       help='Enable stage 204/205 noisy-mixture projection')
    parser.add_argument('--low_snr_cond_skip_projection', action='store_true',
                       help='Disable stage 204/205 noisy-mixture projection')
    parser.add_argument('--low_snr_cond_project_during_train', action='store_true',
                       help='Enable stage 204/205 noisy-mixture projection during training')
    parser.add_argument('--low_snr_cond_skip_train_projection', action='store_true',
                       help='Disable stage 204/205 noisy-mixture projection during training')
    parser.add_argument('--low_snr_cond_project_during_eval', action='store_true',
                       help='Enable stage 204/205 noisy-mixture projection during evaluation')
    parser.add_argument('--low_snr_cond_skip_eval_projection', action='store_true',
                       help='Disable stage 204/205 noisy-mixture projection during evaluation')
    parser.add_argument('--low_snr_cond_return_aux', action='store_true',
                       help='Return stage 204/205 enhancement diagnostics as an aux dictionary')
    parser.add_argument('--low_snr_cond_no_aux', action='store_true',
                       help='Return only separated sources from stage 204/205')

    # Neural Wiener low-SNR front-end IQUMamba overrides (stage 208)
    parser.add_argument('--wiener_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 208 neural Wiener front-end')
    parser.add_argument('--wiener_kernel_size', type=int, default=None,
                       help='Override convolution kernel size of the stage 208 neural Wiener front-end')
    parser.add_argument('--wiener_signal_bias_init', type=float, default=None,
                       help='Override initial signal-power bias of the stage 208 Wiener gate')
    parser.add_argument('--wiener_noise_bias_init', type=float, default=None,
                       help='Override initial noise-power bias of the stage 208 Wiener gate')
    parser.add_argument('--wiener_log_power_clip', type=float, default=None,
                       help='Override log-power clipping range for the stage 208 Wiener front-end')
    parser.add_argument('--wiener_source_weight', type=float, default=None,
                       help='Override source residual share in the stage 208 mixture projection')
    parser.add_argument('--wiener_noise_weight', type=float, default=None,
                       help='Override residual-noise share in the stage 208 mixture projection')
    parser.add_argument('--wiener_eps', type=float, default=None,
                       help='Override numerical epsilon in the stage 208 neural Wiener gate')
    parser.add_argument('--wiener_use_projection', action='store_true',
                       help='Enable stage 208 noisy-mixture projection')
    parser.add_argument('--wiener_skip_projection', action='store_true',
                       help='Disable stage 208 noisy-mixture projection')
    parser.add_argument('--wiener_project_during_train', action='store_true',
                       help='Enable stage 208 noisy-mixture projection during training')
    parser.add_argument('--wiener_skip_train_projection', action='store_true',
                       help='Disable stage 208 noisy-mixture projection during training')
    parser.add_argument('--wiener_project_during_eval', action='store_true',
                       help='Enable stage 208 noisy-mixture projection during evaluation')
    parser.add_argument('--wiener_skip_eval_projection', action='store_true',
                       help='Disable stage 208 noisy-mixture projection during evaluation')
    parser.add_argument('--wiener_return_aux', action='store_true',
                       help='Return stage 208 Wiener diagnostics as an aux dictionary')
    parser.add_argument('--wiener_no_aux', action='store_true',
                       help='Return only separated sources from stage 208')

    # Training-only low-SNR objectives (stages 211-213)
    parser.add_argument('--cross_snr_probability', type=float, default=None,
                       help='Override fraction of training batches using cross-SNR pairing')
    parser.add_argument('--cross_snr_high_db', type=float, default=None,
                       help='Override high-SNR teacher partner level')
    parser.add_argument('--cross_snr_low_start_db', type=float, default=None,
                       help='Override initial low-SNR curriculum level')
    parser.add_argument('--cross_snr_low_middle_db', type=float, default=None,
                       help='Override middle low-SNR curriculum level')
    parser.add_argument('--cross_snr_low_final_db', type=float, default=None,
                       help='Override final low-SNR curriculum level')
    parser.add_argument('--cross_snr_first_fraction', type=float, default=None,
                       help='Override first cross-SNR curriculum boundary')
    parser.add_argument('--cross_snr_second_fraction', type=float, default=None,
                       help='Override second cross-SNR curriculum boundary')
    parser.add_argument('--cross_snr_pair_weight', type=float, default=None,
                       help='Override supervised partner-mixture loss weight')
    parser.add_argument('--cross_snr_consistency_weight', type=float, default=None,
                       help='Override detached high-to-low consistency weight')
    parser.add_argument('--cross_snr_consistency_beta', type=float, default=None,
                       help='Override Smooth-L1 beta for cross-SNR consistency')
    parser.add_argument('--cross_snr_eps', type=float, default=None,
                       help='Override numerical epsilon for cross-SNR training')
    parser.add_argument('--cross_snr_enable', action='store_true', default=None,
                       help='Enable training-only cross-SNR paired consistency')
    parser.add_argument('--cross_snr_disable', action='store_true',
                       help='Disable training-only cross-SNR paired consistency')
    parser.add_argument('--blind_cross_snr_distill', action='store_true',
                       help=('Attach clean-to-noisy cross-SNR distillation to the selected stage '
                             'without synchronization or generator-parameter supervision'))
    parser.add_argument('--cross_snr_ema_teacher_enable', action='store_true', default=None,
                       help='Use an EMA high-SNR teacher for cross-SNR distillation')
    parser.add_argument('--cross_snr_ema_teacher_disable', action='store_true',
                       help='Use the online detached high-SNR branch instead of an EMA teacher')
    parser.add_argument('--cross_snr_ema_decay', type=float, default=None,
                       help='EMA decay for the cross-SNR teacher')
    parser.add_argument('--cross_snr_teacher_mode', choices=['ema', 'frozen'], default=None,
                       help='Use an EMA teacher or a separately pretrained frozen teacher')
    parser.add_argument('--cross_snr_teacher_checkpoint', type=str, default=None,
                       help=('Checkpoint for a pretrained frozen teacher with the same architecture '
                             'as the selected student stage'))
    parser.add_argument('--cross_snr_teacher_view', choices=['high_snr', 'clean'], default=None,
                       help='Input view used by the cross-SNR teacher')
    parser.add_argument('--cross_snr_pair_mode', choices=['complementary', 'curriculum_student'], default=None,
                       help='Construct a complementary partner or an explicit curriculum student view')
    parser.add_argument('--cross_snr_feature_consistency_weight', type=float, default=None,
                       help='Weight of clean/high teacher bottleneck feature distillation')
    parser.add_argument('--cross_snr_feature_consistency_beta', type=float, default=None,
                       help='Smooth-L1 beta for bottleneck feature distillation')
    parser.add_argument('--sync_snr_aux_weight', type=float, default=None,
                       help='Weight of explicit synchronization-head SNR supervision')
    parser.add_argument('--sync_snr_aux_min_db', type=float, default=None,
                       help='Minimum SNR used to normalize synchronization-head supervision')
    parser.add_argument('--sync_snr_aux_max_db', type=float, default=None,
                       help='Maximum SNR used to normalize synchronization-head supervision')
    parser.add_argument('--sync_snr_aux_beta', type=float, default=None,
                       help='Smooth-L1 beta for synchronization-head SNR supervision')
    parser.add_argument('--sync_physical_supervision_weight', type=float, default=None,
                       help='Weight of PIT-aligned per-source physical synchronization supervision')
    parser.add_argument('--training_snr_floor_db', type=float, default=None,
                       help='Train only on dataset samples at or above this SNR')
    parser.add_argument('--validation_snr_floor_db', type=float, default=None,
                       help='Validate only on dataset samples at or above this SNR')
    parser.add_argument('--sync_cross_snr_consistency_weight', type=float, default=None,
                       help='Weight of CFO/phase/timing/SPS/drift cross-SNR consistency')
    parser.add_argument('--sync_cross_snr_consistency_beta', type=float, default=None,
                       help='Smooth-L1 beta for synchronization-parameter distillation')
    parser.add_argument('--sync_cfo_scale', type=float, default=None,
                       help='CFO normalization scale in cycles/sample for sync distillation')
    parser.add_argument('--sync_phase_drift_scale', type=float, default=None,
                       help='Phase-drift normalization scale in rad/sample')
    parser.add_argument('--rf_equiv_enable', action='store_true', default=None,
                       help=('Enable fixed-slot RF transformation equivariance without PIT '
                             'or generator-parameter labels'))
    parser.add_argument('--rf_equiv_disable', action='store_true',
                       help='Disable fixed-slot RF transformation equivariance')
    parser.add_argument('--rf_equiv_probability', type=float, default=None,
                       help='Fraction of training batches receiving an RF-equivariant view')
    parser.add_argument('--rf_equiv_supervised_weight', type=float, default=None,
                       help='Weight of supervised separation on the transformed view')
    parser.add_argument('--rf_equiv_consistency_weight', type=float, default=None,
                       help='Weight of inverse-transformed fixed-slot output consistency')
    parser.add_argument('--rf_equiv_max_phase_degrees', type=float, default=None,
                       help='Maximum absolute random per-slot phase rotation')
    parser.add_argument('--rf_equiv_max_cfo_cycles_per_sample', type=float, default=None,
                       help='Maximum absolute random per-slot CFO in cycles/sample')
    parser.add_argument('--rf_equiv_max_gain_db', type=float, default=None,
                       help='Maximum absolute random per-slot complex gain in dB')
    parser.add_argument('--rf_equiv_max_shift_samples', type=int, default=None,
                       help='Maximum absolute circular time shift in samples')
    parser.add_argument('--rf_equiv_conjugate_probability', type=float, default=None,
                       help='Probability of IQ conjugation for each sampled transform')
    parser.add_argument('--rf_equiv_source_mode', choices=['global', 'per_source'], default=None,
                       help='Use one shared RF transform or an independent transform per source slot')
    parser.add_argument('--rf_equiv_beta', type=float, default=None,
                       help='Smooth-L1 beta for RF-equivariant output consistency')
    parser.add_argument('--rf_equiv_eps', type=float, default=None,
                       help='Numerical epsilon for RMS-normalized RF consistency')
    parser.add_argument('--latent_mask_residual_weight', type=float, default=None,
                       help='Weight of explicit residual/noise-slot waveform supervision')
    parser.add_argument('--latent_mask_mixture_weight', type=float, default=None,
                       help='Weight of source-plus-residual mixture closure')
    parser.add_argument('--latent_mask_residual_beta', type=float, default=None,
                       help='Smooth-L1 beta for latent-mask residual objectives')
    parser.add_argument('--receiver_symbol_weight', type=float, default=None,
                       help='Override matched-filter receiver-domain loss weight')
    parser.add_argument('--receiver_symbol_probability', type=float, default=None,
                       help='Probability of evaluating the training-only receiver loss')
    parser.add_argument('--receiver_symbol_batch_fraction', type=float, default=None,
                       help='Fraction of a selected batch used by the receiver loss')
    parser.add_argument('--receiver_sps_candidates', nargs='+', type=int, default=None,
                       help='Override samples-per-symbol hypotheses, for example 10 20')
    parser.add_argument('--receiver_rrc_rolloff', type=float, default=None,
                       help='Override receiver RRC rolloff')
    parser.add_argument('--receiver_rrc_span', type=int, default=None,
                       help='Override receiver RRC span in symbols')
    parser.add_argument('--receiver_constellation_weight', type=float, default=None,
                       help='Override constellation subweight inside receiver loss')
    parser.add_argument('--receiver_softmin_temperature', type=float, default=None,
                       help='Override SPS-hypothesis soft-min temperature')
    parser.add_argument('--receiver_symbol_beta', type=float, default=None,
                       help='Override Smooth-L1 beta for symbol EVM')
    parser.add_argument('--receiver_symbol_eps', type=float, default=None,
                       help='Override numerical epsilon for receiver-domain loss')
    parser.add_argument('--cumulant_prior_weight', type=float, default=None,
                       help='Weight of the stage-216 training-only cumulant prior')
    parser.add_argument('--cumulant_prior_probability', type=float, default=None,
                       help='Probability of evaluating the stage-216 prior per training batch')
    parser.add_argument('--cumulant_prior_batch_fraction', type=float, default=None,
                       help='Fraction of a selected batch used by the stage-216 prior')
    parser.add_argument('--cumulant_prior_window_sizes', nargs='+', type=int, default=None,
                       help='Window sizes used by the stage-216 cumulant estimator')
    parser.add_argument('--cumulant_prior_self_weight', type=float, default=None,
                       help='Within-source cumulant matching weight')
    parser.add_argument('--cumulant_prior_cross_weight', type=float, default=None,
                       help='Cross-source cumulant matching weight')
    parser.add_argument('--cumulant_prior_confidence_floor', type=float, default=None,
                       help='Minimum reliability weight for cumulant estimates')
    parser.add_argument('--cumulant_prior_beta', type=float, default=None,
                       help='Smooth-L1 beta for cumulant matching')
    parser.add_argument('--cumulant_prior_eps', type=float, default=None,
                       help='Numerical epsilon for normalized cumulants')
    parser.add_argument('--cumulant_residual_enable', action='store_true', default=None,
                       help='Enable Stage225 Gaussian-residual cumulant prior')
    parser.add_argument('--cumulant_residual_weight', type=float, default=None,
                       help='Weight of the Gaussian-residual cumulant term')
    parser.add_argument('--cumulant_residual_cross_weight', type=float, default=None,
                       help='Weight of source-residual independence term')
    parser.add_argument('--cumulant_residual_beta', type=float, default=None,
                       help='Smooth-L1 beta for the Gaussian-residual prior')

    # Stage 296: FSQ token cross-entropy prior (training-only)
    parser.add_argument('--fsq_token_ce_enable', action='store_true', default=None,
                       help='Enable the stage-296 frozen-FSQ-tokenizer token CE prior')
    parser.add_argument('--fsq_token_ce_weight', type=float, default=None,
                       help='Weight of the stage-296 FSQ token cross-entropy term')
    parser.add_argument('--fsq_token_ce_temperature', type=float, default=None,
                       help='Distance-to-logit temperature for FSQ token CE')
    parser.add_argument('--fsq_token_ce_warmup_steps', type=int, default=None,
                       help='Linear warmup steps before the token CE reaches full weight')
    parser.add_argument('--fsq_tokenizer_checkpoint', type=str, default=None,
                       help='Path to the pretrained FSQ tokenizer checkpoint '
                            '(see pretrain_fsq_tokenizer.py)')

    # Adaptive Spectral Gating IQUMamba overrides (stage 202)
    parser.add_argument('--asg_patch_size', type=int, default=None,
                       help='Override local FFT patch size in stage 202 ASG Mamba')
    parser.add_argument('--asg_stride', type=int, default=None,
                       help='Override local FFT patch stride in stage 202 ASG Mamba')
    parser.add_argument('--asg_num_bands', type=int, default=None,
                       help='Override number of spectral energy bands in stage 202 ASG Mamba')
    parser.add_argument('--asg_gate_hidden', type=int, default=None,
                       help='Override hidden size of the stage 202 ASG gate MLP')
    parser.add_argument('--asg_scale_init', type=float, default=None,
                       help='Override initial residual gate scale in stage 202 ASG Mamba')
    parser.add_argument('--asg_apply_stages', nargs='+', type=int, default=None,
                       help='Override encoder stage indices wrapped by ASG in stage 202')
    parser.add_argument('--asg_eps', type=float, default=None,
                       help='Override numerical epsilon used by stage 202 ASG band-energy normalization')
    parser.add_argument('--asg_zero_init', action='store_true',
                       help='Zero-initialize the final ASG gate layer in stage 202')
    parser.add_argument('--asg_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final ASG gate layer in stage 202')

    # Local complex-aware IQUMamba overrides (stage 76)
    parser.add_argument('--complex_adapter_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in stage 76 adapters')
    parser.add_argument('--complex_adapter_kernel_size', type=int, default=None,
                       help='Override tied complex convolution kernel size in stage 76 adapters')
    parser.add_argument('--complex_adapter_scale_init', type=float, default=None,
                       help='Override initial residual scale in stage 76 adapters')
    parser.add_argument('--complex_adapter_use_input', action='store_true',
                       help='Enable the stage 76 input complex adapter')
    parser.add_argument('--complex_adapter_no_input', action='store_true',
                       help='Disable the stage 76 input complex adapter')
    parser.add_argument('--complex_adapter_use_output', action='store_true',
                       help='Enable the stage 76 output complex adapter')
    parser.add_argument('--complex_adapter_no_output', action='store_true',
                       help='Disable the stage 76 output complex adapter')
    parser.add_argument('--complex_adapter_zero_init', action='store_true',
                       help='Zero-initialize the final tied complex adapter layer in stage 76')
    parser.add_argument('--complex_adapter_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final tied complex adapter layer in stage 76')

    # Cyclostationary/FRESH IQUMamba overrides (stage 77)
    parser.add_argument('--cyclofresh_sps', type=int, default=None,
                       help='Override samples-per-symbol normalization for stage 77 cyclic frequencies')
    parser.add_argument('--cyclofresh_alphas', nargs='+', type=float, default=None,
                       help='Override stage 77 cyclic frequencies in symbol-rate units, e.g. 0 1 -1 2 -2')
    parser.add_argument('--cyclofresh_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 77 FRESH adapter')
    parser.add_argument('--cyclofresh_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 77 FRESH adapter')
    parser.add_argument('--cyclofresh_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 77 FRESH adapter')
    parser.add_argument('--cyclofresh_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 77 FRESH adapter')
    parser.add_argument('--cyclofresh_zero_init', action='store_true',
                       help='Zero-initialize the final FRESH projection in stage 77')
    parser.add_argument('--cyclofresh_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final FRESH projection in stage 77')

    # Blind learnable-cyclic-frequency FRESH IQUMamba overrides (stage 78)
    parser.add_argument('--blind_cyclofresh_freqs', nargs='+', type=float, default=None,
                       help='Override stage 78 initial normalized cyclic frequencies in cycles/sample')
    parser.add_argument('--blind_cyclofresh_max_delta', type=float, default=None,
                       help='Override max learnable frequency offset in stage 78')
    parser.add_argument('--blind_cyclofresh_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 78 blind FRESH adapter')
    parser.add_argument('--blind_cyclofresh_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 78 blind FRESH adapter')
    parser.add_argument('--blind_cyclofresh_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 78 blind FRESH adapter')
    parser.add_argument('--blind_cyclofresh_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 78 blind FRESH adapter')
    parser.add_argument('--blind_cyclofresh_zero_init', action='store_true',
                       help='Zero-initialize the final blind FRESH projection in stage 78')
    parser.add_argument('--blind_cyclofresh_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final blind FRESH projection in stage 78')

    # Mixture-estimated cyclic-frequency FRESH IQUMamba overrides (stage 79)
    parser.add_argument('--estimated_cyclofresh_min_freq', type=float, default=None,
                       help='Override stage 79 lower cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--estimated_cyclofresh_max_freq', type=float, default=None,
                       help='Override stage 79 upper cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--estimated_cyclofresh_default_freq', type=float, default=None,
                       help='Override stage 79 fallback cyclic frequency in cycles/sample')
    parser.add_argument('--estimated_cyclofresh_momentum', type=float, default=None,
                       help='Override stage 79 EMA momentum for train-estimated cyclic frequency')
    parser.add_argument('--estimated_cyclofresh_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 79 estimated FRESH adapter')
    parser.add_argument('--estimated_cyclofresh_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 79 estimated FRESH adapter')
    parser.add_argument('--estimated_cyclofresh_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 79 estimated FRESH adapter')
    parser.add_argument('--estimated_cyclofresh_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 79 estimated FRESH adapter')
    parser.add_argument('--estimated_cyclofresh_zero_init', action='store_true',
                       help='Zero-initialize the final estimated FRESH projection in stage 79')
    parser.add_argument('--estimated_cyclofresh_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final estimated FRESH projection in stage 79')

    # CycloFRESH-plus IQUMamba overrides (stages 82/83/84)
    parser.add_argument('--multipeak_cyclofresh_min_freq', type=float, default=None,
                       help='Override stage 82 lower cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--multipeak_cyclofresh_max_freq', type=float, default=None,
                       help='Override stage 82 upper cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--multipeak_cyclofresh_default_freq', type=float, default=None,
                       help='Override stage 82 fallback cyclic frequency in cycles/sample')
    parser.add_argument('--multipeak_cyclofresh_momentum', type=float, default=None,
                       help='Override stage 82 EMA momentum for train-estimated cyclic frequencies')
    parser.add_argument('--multipeak_cyclofresh_num_peaks', type=int, default=None,
                       help='Override number of stage 82 envelope-power spectrum peaks')
    parser.add_argument('--multipeak_cyclofresh_guard_bins', type=int, default=None,
                       help='Override excluded FFT guard bins around each selected stage 82 peak')
    parser.add_argument('--multipeak_cyclofresh_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 82 FRESH adapter')
    parser.add_argument('--multipeak_cyclofresh_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 82 FRESH adapter')
    parser.add_argument('--multipeak_cyclofresh_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 82 FRESH adapter')
    parser.add_argument('--multipeak_cyclofresh_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 82 FRESH adapter')
    parser.add_argument('--multipeak_cyclofresh_reliability_floor', type=float, default=None,
                       help='Override minimum reliability scale for stage 82 adapter output')
    parser.add_argument('--multipeak_cyclofresh_zero_init', action='store_true',
                       help='Zero-initialize the final multi-peak FRESH projection in stage 82')
    parser.add_argument('--multipeak_cyclofresh_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final multi-peak FRESH projection in stage 82')
    parser.add_argument('--sample_cyclofresh_min_freq', type=float, default=None,
                       help='Override stage 83 lower cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--sample_cyclofresh_max_freq', type=float, default=None,
                       help='Override stage 83 upper cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--sample_cyclofresh_default_freq', type=float, default=None,
                       help='Override stage 83 fallback cyclic frequency in cycles/sample')
    parser.add_argument('--sample_cyclofresh_num_peaks', type=int, default=None,
                       help='Override number of per-sample stage 83 envelope-power spectrum peaks')
    parser.add_argument('--sample_cyclofresh_guard_bins', type=int, default=None,
                       help='Override excluded FFT guard bins around each selected stage 83 peak')
    parser.add_argument('--sample_cyclofresh_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 83 FRESH adapter')
    parser.add_argument('--sample_cyclofresh_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 83 FRESH adapter')
    parser.add_argument('--sample_cyclofresh_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 83 FRESH adapter')
    parser.add_argument('--sample_cyclofresh_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 83 FRESH adapter')
    parser.add_argument('--sample_cyclofresh_reliability_floor', type=float, default=None,
                       help='Override minimum reliability scale for stage 83 adapter output')
    parser.add_argument('--sample_cyclofresh_zero_init', action='store_true',
                       help='Zero-initialize the final sample-adaptive FRESH projection in stage 83')
    parser.add_argument('--sample_cyclofresh_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final sample-adaptive FRESH projection in stage 83')
    parser.add_argument('--multihyp_cyclic_freqs', nargs='+', type=float, default=None,
                       help='Override stage 210 normalized cyclic-frequency hypotheses in cycles/sample')
    parser.add_argument('--multihyp_cyclic_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 210 multi-hypothesis cyclic adapter')
    parser.add_argument('--multihyp_cyclic_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 210 multi-hypothesis cyclic adapter')
    parser.add_argument('--multihyp_cyclic_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 210 multi-hypothesis cyclic adapter')
    parser.add_argument('--multihyp_cyclic_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 210 multi-hypothesis cyclic adapter')
    parser.add_argument('--multihyp_cyclic_temperature', type=float, default=None,
                       help='Override softmax temperature for stage 210 cyclic evidence selection')
    parser.add_argument('--multihyp_cyclic_null_logit_init', type=float, default=None,
                       help='Override initial null-hypothesis logit in the stage 210 cyclic adapter')
    parser.add_argument('--multihyp_cyclic_local_bins', type=int, default=None,
                       help='Override local noise-floor bins around each stage 210 cyclic hypothesis')
    parser.add_argument('--multihyp_cyclic_eps', type=float, default=None,
                       help='Override numerical epsilon in the stage 210 cyclic evidence estimator')
    parser.add_argument('--multihyp_cyclic_zero_init', action='store_true',
                       help='Zero-initialize the final multi-hypothesis cyclic projection in stage 210')
    parser.add_argument('--multihyp_cyclic_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final multi-hypothesis cyclic projection in stage 210')
    parser.add_argument('--multihyp_cyclic_return_aux', action='store_true',
                       help='Return stage 210 cyclic-hypothesis diagnostics as an aux dictionary')
    parser.add_argument('--multihyp_cyclic_no_aux', action='store_true',
                       help='Return only separated sources from stage 210')
    parser.add_argument('--freqbias_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 84 high-frequency residual adapter')
    parser.add_argument('--freqbias_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 84 high-frequency residual adapter')
    parser.add_argument('--freqbias_lowpass_kernel_size', type=int, default=None,
                       help='Override moving-average low-pass kernel size used by the stage 84 high-pass adapter')
    parser.add_argument('--freqbias_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 84 high-frequency residual adapter')
    parser.add_argument('--freqbias_gate_hidden', type=int, default=None,
                       help='Override gate hidden channels in the stage 84 high-frequency residual adapter')
    parser.add_argument('--freqbias_zero_init', action='store_true',
                       help='Zero-initialize the final high-frequency residual projection in stage 84')
    parser.add_argument('--freqbias_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final high-frequency residual projection in stage 84')

    # Blind-statistic IQUMamba overrides (stages 85/86)
    parser.add_argument('--blindstat_hidden', type=int, default=None,
                       help='Override hidden dimension for stage 85/86 blind-statistic adapters')
    parser.add_argument('--blindstat_kernel_size', type=int, default=None,
                       help='Override input-adapter convolution kernel size for stage 86')
    parser.add_argument('--blindstat_scale_init', type=float, default=None,
                       help='Override initial residual/FiLM scale for stage 85/86')
    parser.add_argument('--blindstat_cyclic_min_freq', type=float, default=None,
                       help='Override lower cyclic-frequency bound used by stage 85/86 blind statistics')
    parser.add_argument('--blindstat_cyclic_max_freq', type=float, default=None,
                       help='Override upper cyclic-frequency bound used by stage 85/86 blind statistics')
    parser.add_argument('--blindstat_cyclic_default_freq', type=float, default=None,
                       help='Override fallback cyclic frequency used by stage 85/86 blind statistics')
    parser.add_argument('--blindstat_zero_init', action='store_true',
                       help='Zero-initialize the final blind-statistic adapter projection in stage 85/86')
    parser.add_argument('--blindstat_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final blind-statistic adapter projection in stage 85/86')
    # Blind multi-rate input adapter overrides (stage 96)
    parser.add_argument('--multirate_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 96 blind multi-rate input adapter')
    parser.add_argument('--multirate_kernel_sizes', nargs='+', type=int, default=None,
                       help='Override stage 96 branch kernel sizes, e.g. 5 9 17 33')
    parser.add_argument('--multirate_dilations', nargs='+', type=int, default=None,
                       help='Override stage 96 branch dilations, e.g. 1 2 4 8')
    parser.add_argument('--multirate_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 96 blind multi-rate input adapter')
    parser.add_argument('--multirate_zero_init', action='store_true',
                       help='Zero-initialize the final stage 96 adapter projection')
    parser.add_argument('--multirate_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final stage 96 adapter projection')
    # Modulation-structure prior IQUMamba overrides (stages 190/191/192)
    parser.add_argument('--psk_prior_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 190 PSK phase-step prior adapter')
    parser.add_argument('--psk_prior_harmonics', nargs='+', type=int, default=None,
                       help='Override stage 190 phase harmonics, e.g. 1 2 4 8 16')
    parser.add_argument('--psk_prior_kernel_size', type=int, default=None,
                       help='Override convolution kernel size in the stage 190 PSK prior adapter')
    parser.add_argument('--psk_prior_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 190 PSK prior adapter')
    parser.add_argument('--psk_prior_reliability_floor', type=float, default=None,
                       help='Override minimum reliability gate value in the stage 190 PSK prior adapter')
    parser.add_argument('--psk_prior_zero_init', action='store_true',
                       help='Zero-initialize the final stage 190 prior adapter projection')
    parser.add_argument('--psk_prior_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final stage 190 prior adapter projection')
    parser.add_argument('--qam_prior_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 191 QAM lattice prior adapter')
    parser.add_argument('--qam_prior_axis_level_bank', nargs='+', type=int, default=None,
                       help='Override stage 191 per-axis lattice candidates, e.g. 4 8 12 16')
    parser.add_argument('--qam_prior_temperature', type=float, default=None,
                       help='Override soft nearest-lattice temperature in the stage 191 QAM prior adapter')
    parser.add_argument('--qam_prior_kernel_size', type=int, default=None,
                       help='Override convolution kernel size in the stage 191 QAM prior adapter')
    parser.add_argument('--qam_prior_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 191 QAM prior adapter')
    parser.add_argument('--qam_prior_reliability_floor', type=float, default=None,
                       help='Override minimum reliability gate value in the stage 191 QAM prior adapter')
    parser.add_argument('--qam_prior_zero_init', action='store_true',
                       help='Zero-initialize the final stage 191 prior adapter projection')
    parser.add_argument('--qam_prior_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final stage 191 prior adapter projection')
    parser.add_argument('--apsk_prior_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 192 APSK ring prior adapter')
    parser.add_argument('--apsk_prior_ring_radii', nargs='+', type=float, default=None,
                       help='Override stage 192 normalized APSK ring radii, e.g. 0.40 1.13')
    parser.add_argument('--apsk_prior_temperature', type=float, default=None,
                       help='Override soft nearest-ring temperature in the stage 192 APSK prior adapter')
    parser.add_argument('--apsk_prior_kernel_size', type=int, default=None,
                       help='Override convolution kernel size in the stage 192 APSK prior adapter')
    parser.add_argument('--apsk_prior_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 192 APSK prior adapter')
    parser.add_argument('--apsk_prior_reliability_floor', type=float, default=None,
                       help='Override minimum reliability gate value in the stage 192 APSK prior adapter')
    parser.add_argument('--apsk_prior_zero_init', action='store_true',
                       help='Zero-initialize the final stage 192 prior adapter projection')
    parser.add_argument('--apsk_prior_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final stage 192 prior adapter projection')
    parser.add_argument('--feature_topology_hidden_channels', type=int, default=None,
                       help='Override hidden channels of the stage 194 feature-domain topology adapter')
    parser.add_argument('--feature_topology_kernel_size', type=int, default=None,
                       help='Override convolution kernel size in the stage 194 feature-domain topology adapter')
    parser.add_argument('--feature_topology_scale_init', type=float, default=None,
                       help='Override residual scale in the stage 194 feature-domain topology adapter')
    parser.add_argument('--feature_topology_apply_stages', nargs='+', type=int, default=None,
                       help='Override encoder stages adapted by stage 194, e.g. 1 2 3')
    parser.add_argument('--feature_topology_zero_init', action='store_true',
                       help='Zero-initialize the final stage 194 feature adapter projection')
    parser.add_argument('--feature_topology_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final stage 194 feature adapter projection')

    # PCO ResUNet overrides (stages 100/101/102)
    parser.add_argument('--pco_phase_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 100/102 phase-equivariant adapter')
    parser.add_argument('--pco_phase_kernel_size', type=int, default=None,
                       help='Override FIR kernel size in the stage 100/102 phase-equivariant adapter')
    parser.add_argument('--pco_phase_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 100/102 phase-equivariant adapter')
    parser.add_argument('--pco_corr_lags', nargs='+', type=int, default=None,
                       help='Override local complex-correlation lags for stages 101/102')
    parser.add_argument('--pco_corr_window', type=int, default=None,
                       help='Override moving-average window for stages 101/102 correlation statistics')
    parser.add_argument('--pco_corr_scale_init', type=float, default=None,
                       help='Override initial skip-gate scale for stages 101/102')
    parser.add_argument('--pco_orth_scale_init', type=float, default=None,
                       help='Override initial output orthogonalization residual scale for stage 102')
    parser.add_argument('--pco_orth_eps', type=float, default=None,
                       help='Override numerical epsilon for stage 102 source orthogonalization')

    # Mixture-estimated cyclic-correlation IQUMamba overrides (stage 80)
    parser.add_argument('--cycliccorr_min_freq', type=float, default=None,
                       help='Override stage 80 lower cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--cycliccorr_max_freq', type=float, default=None,
                       help='Override stage 80 upper cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--cycliccorr_default_freq', type=float, default=None,
                       help='Override stage 80 fallback cyclic frequency in cycles/sample')
    parser.add_argument('--cycliccorr_momentum', type=float, default=None,
                       help='Override stage 80 EMA momentum for train-estimated cyclic frequency')
    parser.add_argument('--cycliccorr_lags', nargs='+', type=int, default=None,
                       help='Override sample lags for stage 80 cyclic-correlation statistics, e.g. 0 1 2 4 8')
    parser.add_argument('--cycliccorr_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 80 cyclic-correlation adapter')
    parser.add_argument('--cycliccorr_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 80 cyclic-correlation adapter')
    parser.add_argument('--cycliccorr_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 80 cyclic-correlation adapter')
    parser.add_argument('--cycliccorr_gate_hidden', type=int, default=None,
                       help='Override gate hidden units in the stage 80 cyclic-correlation adapter')
    parser.add_argument('--cycliccorr_zero_init', action='store_true',
                       help='Zero-initialize the final cyclic-correlation projection in stage 80')
    parser.add_argument('--cycliccorr_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final cyclic-correlation projection in stage 80')

    # Output-side cyclic leakage cancellation overrides (stage 81)
    parser.add_argument('--leakcancel_lags', nargs='+', type=int, default=None,
                       help='Override sample lags for stage 81 cross cyclic-correlation statistics, e.g. 0 1 2 4 8')
    parser.add_argument('--leakcancel_hidden', type=int, default=None,
                       help='Override hidden units in the stage 81 leakage coefficient head')
    parser.add_argument('--leakcancel_scale_init', type=float, default=None,
                       help='Override initial leakage cancellation scale in stage 81')
    parser.add_argument('--leakcancel_mc_scale_init', type=float, default=None,
                       help='Override initial soft mixture-consistency scale in stage 81, in [0,1]')
    parser.add_argument('--leakcancel_mc_weight_mode', type=str, default=None, choices=['uniform', 'energy'],
                       help='Override residual redistribution weights for stage 81 soft mixture consistency')
    parser.add_argument('--leakcancel_mode', type=str, default=None, choices=['covariance', 'learned', 'hybrid'],
                       help='Override stage 81 leakage cancellation mode')
    parser.add_argument('--leakcancel_coeff_limit', type=float, default=None,
                       help='Override maximum magnitude for each stage 81 complex leakage coefficient')
    parser.add_argument('--leakcancel_zero_init', action='store_true',
                       help='Zero-initialize the stage 81 leakage coefficient head')
    parser.add_argument('--leakcancel_no_zero_init', action='store_true',
                       help='Do not zero-initialize the stage 81 leakage coefficient head')

    # Source-wise cyclic-Wiener residual overrides (stage 196)
    parser.add_argument('--cyclic_wiener_hidden_channels', type=int, default=None,
                       help='Override hidden complex channels in the stage 196 cyclic-Wiener residual head')
    parser.add_argument('--cyclic_wiener_kernel_size', type=int, default=None,
                       help='Override complex-tied FIR kernel size in the stage 196 cyclic-Wiener residual head')
    parser.add_argument('--cyclic_wiener_min_freq', type=float, default=None,
                       help='Override stage 196 lower source-wise cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--cyclic_wiener_max_freq', type=float, default=None,
                       help='Override stage 196 upper source-wise cyclic-frequency search bound in cycles/sample')
    parser.add_argument('--cyclic_wiener_default_freq', type=float, default=None,
                       help='Override stage 196 fallback source-wise cyclic frequency in cycles/sample')
    parser.add_argument('--cyclic_wiener_num_harmonics', type=int, default=None,
                       help='Override number of +/- harmonic frequency-shift branches in stage 196')
    parser.add_argument('--cyclic_wiener_scale_init', type=float, default=None,
                       help='Override initial residual scale in the stage 196 cyclic-Wiener residual head')
    parser.add_argument('--cyclic_wiener_projection_strength', type=float, default=None,
                       help='Override final soft mixture-consistency projection strength in stage 196')
    parser.add_argument('--cyclic_wiener_zero_init', action='store_true',
                       help='Zero-initialize the final cyclic-Wiener residual projection in stage 196')
    parser.add_argument('--cyclic_wiener_no_zero_init', action='store_true',
                       help='Do not zero-initialize the final cyclic-Wiener residual projection in stage 196')

    # Constellation-guided refinement optional overrides (mainly for stage 60)
    parser.add_argument('--constellation_type', type=str, default=None, choices=['psk'],
                       help='Override constellation prior type in stage 60')
    parser.add_argument('--constellation_order', type=int, default=None,
                       help='Override constellation order in stage 60, e.g. 8 for 8PSK')
    parser.add_argument('--cgr_hidden_channels', type=int, default=None,
                       help='Override constellation-guided refinement hidden channels in stage 60')
    parser.add_argument('--cgr_kernel_size', type=int, default=None,
                       help='Override constellation-guided refinement kernel size in stage 60')
    parser.add_argument('--cgr_temperature', type=float, default=None,
                       help='Override soft constellation assignment temperature in stage 60')
    parser.add_argument('--cgr_dropout', type=float, default=None,
                       help='Override constellation-guided refinement dropout in stage 60')
    parser.add_argument('--cgr_gate_init', type=float, default=None,
                       help='Override initial residual gate probability in stage 60')
    parser.add_argument('--cgr_residual_scale_init', type=float, default=None,
                       help='Override initial residual scale in stage 60')
    parser.add_argument('--cgr_use_mixture_residual', action='store_true',
                       help='Use mixture-consistency residual features in stage 60')
    parser.add_argument('--cgr_no_mixture_residual', action='store_true',
                       help='Disable mixture-consistency residual features in stage 60')
    parser.add_argument('--cgr_zero_init', action='store_true',
                       help='Zero-initialize constellation residual output in stage 60')
    parser.add_argument('--cgr_no_zero_init', action='store_true',
                       help='Do not zero-initialize constellation residual output in stage 60')
    parser.add_argument('--cgr_refine_deep_supervision', action='store_true',
                       help='Refine every deep-supervision output in stage 60')
    parser.add_argument('--cgr_final_only', action='store_true',
                       help='Refine only final/full-resolution output in stage 60')
    parser.add_argument('--cgr_apply_train', action='store_true',
                       help='Enable constellation-guided refinement during training in stage 60')
    parser.add_argument('--cgr_skip_train', action='store_true',
                       help='Disable constellation-guided refinement during training in stage 60')
    parser.add_argument('--cgr_apply_eval', action='store_true',
                       help='Enable constellation-guided refinement during evaluation in stage 60')
    parser.add_argument('--cgr_skip_eval', action='store_true',
                       help='Disable constellation-guided refinement during evaluation in stage 60')

    # Mixture-consistency projection optional overrides (mainly for stage 39)
    parser.add_argument('--mc_weight_mode', type=str, default=None, choices=['energy', 'uniform'],
                       help='Override MC projection weight mode in stage 39')
    parser.add_argument('--mc_weight_power', type=float, default=None,
                       help='Override MC projection weight exponent in stage 39')
    parser.add_argument('--mc_min_weight', type=float, default=None,
                       help='Override MC projection minimum weight floor in stage 39')
    parser.add_argument('--mc_eps', type=float, default=None,
                       help='Override MC projection numerical epsilon in stage 39')
    parser.add_argument('--mc_detach_weights', action='store_true',
                       help='Detach MC weight computation from gradients in stage 39')
    parser.add_argument('--mc_keep_weight_grads', action='store_true',
                       help='Keep gradients through MC weight computation in stage 39')
    parser.add_argument('--mc_project_deep_supervision', action='store_true',
                       help='Project every deep-supervision output in stage 39')
    parser.add_argument('--mc_project_final_only', action='store_true',
                       help='Project only the final output when deep supervision is enabled in stage 39')
    parser.add_argument('--mc_apply_train', action='store_true',
                       help='Enable MC projection during training in stage 39')
    parser.add_argument('--mc_skip_train_projection', action='store_true',
                       help='Disable MC projection during training in stage 39')
    parser.add_argument('--mc_apply_eval', action='store_true',
                       help='Enable MC projection during evaluation in stage 39')
    parser.add_argument('--mc_skip_eval_projection', action='store_true',
                       help='Disable MC projection during evaluation in stage 39')

    # URIC (Unrolled Residual Interference Cancellation) optional overrides
    parser.add_argument('--ric_num_steps', type=int, default=None,
                       help='Override URIC iteration steps in stage 40/41/47/48/69')
    parser.add_argument('--ric_hidden_channels', type=int, default=None,
                       help='Override URIC hidden channels in stage 40/41/47/48/69')
    parser.add_argument('--ric_kernel_size', type=int, default=None,
                       help='Override URIC kernel size in stage 40/41/47/48/69 (odd integer recommended)')
    parser.add_argument('--ric_dropout', type=float, default=None,
                       help='Override URIC dropout in stage 40/41/47/48/69')
    parser.add_argument('--ric_step_init', type=float, default=None,
                       help='Override URIC initial step size in stage 40/41/47/48/69, must be in (0,1)')
    parser.add_argument('--ric_untied_steps', action='store_true',
                       help='Use untied URIC update blocks instead of shared weights in stage 40/41/47/48/69')
    parser.add_argument('--ric_return_intermediate', action='store_true',
                       help='Force URIC models to return per-step intermediate estimates while training')
    parser.add_argument('--ric_update_block_type', type=str, default=None,
                       choices=['conv', 'dilated_gated', 'cross_attention'],
                       help='Override URIC update block type in stage 40/41/47/48/69')
    parser.add_argument('--ric_dilations', nargs='+', type=int, default=None,
                       help='Override dilations for dilated_gated URIC block, e.g. --ric_dilations 1 2 4')
    parser.add_argument('--ric_num_heads', type=int, default=None,
                       help='Override attention heads for cross_attention URIC block')
    parser.add_argument('--ric_attention_stride', type=int, default=None,
                       help='Override context downsampling stride for cross_attention URIC block')
    parser.add_argument('--ric_ffn_multiplier', type=int, default=None,
                       help='Override feed-forward expansion multiplier for cross_attention URIC block')

    # ADMM-unfolded communication-prior optional overrides
    parser.add_argument('--admm_num_steps', type=int, default=None,
                       help='Override ADMM unfolded iteration steps in stage 49')
    parser.add_argument('--admm_hidden_channels', type=int, default=None,
                       help='Override ADMM proximal hidden channels in stage 49')
    parser.add_argument('--admm_kernel_size', type=int, default=None,
                       help='Override ADMM proximal kernel size in stage 49 (odd integer recommended)')
    parser.add_argument('--admm_dropout', type=float, default=None,
                       help='Override ADMM proximal dropout in stage 49')
    parser.add_argument('--admm_rho_init', type=float, default=None,
                       help='Override initial ADMM penalty rho in stage 49')
    parser.add_argument('--admm_dual_step_init', type=float, default=None,
                       help='Override initial ADMM dual update step in stage 49')
    parser.add_argument('--admm_prox_step_init', type=float, default=None,
                       help='Override initial ADMM learnable proximal residual step in stage 49')
    parser.add_argument('--admm_untied_steps', action='store_true',
                       help='Use untied ADMM proximal blocks instead of shared weights in stage 49')

    # PGD-unfolded communication-prior optional overrides
    parser.add_argument('--pgdu_num_steps', type=int, default=None,
                       help='Override PGD-U unfolded iteration steps in stage 50')
    parser.add_argument('--pgdu_hidden_channels', type=int, default=None,
                       help='Override PGD-U proximal hidden channels in stage 50')
    parser.add_argument('--pgdu_kernel_size', type=int, default=None,
                       help='Override PGD-U proximal kernel size in stage 50 (odd integer recommended)')
    parser.add_argument('--pgdu_dropout', type=float, default=None,
                       help='Override PGD-U proximal dropout in stage 50')
    parser.add_argument('--pgdu_step_size_init', type=float, default=None,
                       help='Override initial PGD-U data-consistency step size in stage 50')
    parser.add_argument('--pgdu_prox_step_init', type=float, default=None,
                       help='Override initial PGD-U learnable proximal residual step in stage 50')
    parser.add_argument('--pgdu_untied_steps', action='store_true',
                       help='Use untied PGD-U proximal blocks instead of shared weights in stage 50')

    # Gain/phase channel consistency optional overrides
    parser.add_argument('--gp_hidden_channels', type=int, default=None,
                       help='Override gain/phase parameter head hidden channels in stage 52')
    parser.add_argument('--gp_kernel_size', type=int, default=None,
                       help='Override gain/phase parameter head kernel size in stage 52')
    parser.add_argument('--gp_max_gain_db', type=float, default=None,
                       help='Override maximum absolute gain range in dB for stage 52')
    parser.add_argument('--gp_max_phase_deg', type=float, default=None,
                       help='Override maximum absolute phase range in degrees for stage 52')
    parser.add_argument('--gp_weight_mode', type=str, default=None, choices=['energy', 'uniform'],
                       help='Override gain/phase residual distribution weights in stage 52')
    parser.add_argument('--gp_min_weight', type=float, default=None,
                       help='Override gain/phase residual minimum weight floor in stage 52')
    parser.add_argument('--gp_correction_strength_init', type=float, default=None,
                       help='Override initial gain/phase residual correction strength in stage 52')
    parser.add_argument('--gp_apply_train', action='store_true',
                       help='Enable gain/phase channel consistency during training in stage 52')
    parser.add_argument('--gp_skip_train', action='store_true',
                       help='Disable gain/phase channel consistency during training in stage 52')
    parser.add_argument('--gp_apply_eval', action='store_true',
                       help='Enable gain/phase channel consistency during evaluation in stage 52')
    parser.add_argument('--gp_skip_eval', action='store_true',
                       help='Disable gain/phase channel consistency during evaluation in stage 52')

    # Train-only augmentation overrides (mainly for stage 41)
    parser.add_argument('--enable_train_aug', action='store_true',
                       help='Force-enable train-time augmentation defined by the model config')
    parser.add_argument('--disable_train_aug', action='store_true',
                       help='Disable train-time augmentation even if enabled in the model config')
    parser.add_argument('--train_mix_enable', action='store_true',
                       help='Enable online mixture-level augmentation on the training split')
    parser.add_argument('--train_mix_disable', action='store_true',
                       help='Disable online mixture-level augmentation on the training split')
    parser.add_argument('--train_mix_prob', type=float, default=None,
                       help='Probability of applying mixture-level augmentation per training sample')
    parser.add_argument('--train_mix_sir_min_db', type=float, default=None,
                       help='Lower bound of per-source remix gain range in dB')
    parser.add_argument('--train_mix_sir_max_db', type=float, default=None,
                       help='Upper bound of per-source remix gain range in dB')
    parser.add_argument('--train_mix_cross_sample', action='store_true',
                       help='Use cross-sample source replacement before remixing (heavier but stronger)')
    parser.add_argument('--train_aug_profile', type=str, default='config',
                       choices=['config', 'gain_sir_remix'],
                       help='Train augmentation profile: config keeps YAML defaults; gain_sir_remix enables only per-source SIR remix')
    parser.add_argument('--train_aug_warmup_ratio', type=float, default=None,
                       help='Fraction of early epochs with train augmentation disabled, e.g. 0.1 for first 10%')
    parser.add_argument('--train_aug_warmup_epochs', type=int, default=None,
                       help='Exact number of early epochs with train augmentation disabled; overrides ratio')

    # Runtime / memory knobs (for weak GPUs)
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (reduce this first if you hit CUDA OOM)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: IQUMamba/BiMamba=1e-3, TFGridNet-family=3e-4, TIGER-family=1e-4)')
    parser.add_argument('--lr_phase1', type=float, default=None,
                       help='Optional phase-1 learning rate override when --two_phase_loss is enabled')
    parser.add_argument('--lr_phase2', type=float, default=None,
                       help='Optional phase-2 learning rate override when --two_phase_loss is enabled')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for Adam optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (increase to keep effective batch when batch_size is small)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (0 is safest on Windows)')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable AMP mixed precision (AMP usually saves VRAM)')
    parser.add_argument('--no_pin_memory', action='store_true',
                       help='Disable DataLoader pin_memory')
    parser.add_argument('--require_cuda', action='store_true',
                       help='Fail immediately if CUDA is not available')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                       help='Early stopping patience (<=0 disables early stopping)')
    parser.add_argument('--save_checkpoint_every', type=int, default=0,
                       help='Save numbered checkpoint_epoch_N.pth every N epochs; 0 keeps only latest/best checkpoints')
    parser.add_argument('--train_subset_ratio', type=float, default=1.0,
                       help='Use only this ratio of the current training split (0,1]')
    parser.add_argument('--split_strategy', type=str, default='random',
                       choices=['random', 'stratified_snr'],
                       help='Dataset split strategy: random or SNR-stratified')
    
    # Random seed related parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default not set)')
    parser.add_argument('--multiple_runs', action='store_true',
                       help='Perform multiple runs experiment')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs for multiple runs (default 5)')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting seed for multiple runs (default 42)')
    parser.add_argument('--ablation_losses', nargs='+', type=str,
                       default=['PIT-SI-SNR', 'SI-SNR', 'Huber'],
                       help='Loss list for --mode ablation_losses')
    parser.add_argument('--compact_results', action='store_true',
                       help='Reduce file outputs (mainly for ablation mode)')
    parser.add_argument('--mrstft_n_ffts', nargs='+', type=int, default=[128, 256, 512],
                       help='MRSTFT FFT sizes')
    parser.add_argument('--mrstft_band_ratio', type=float, default=0.35,
                       help='Occupied bandwidth ratio in [0,1]')
    parser.add_argument('--mrstft_inband_weight', type=float, default=2.0,
                       help='In-band spectral weight')
    parser.add_argument('--mrstft_outband_weight', type=float, default=0.5,
                       help='Out-of-band spectral weight')
    parser.add_argument('--mrstft_complex_weight', type=float, default=0.3,
                       help='MRSTFT complex-spectrum term weight')
    parser.add_argument('--mrstft_lambda', type=float, default=0.2,
                       help='Global lambda for MRSTFT in combo losses')
    parser.add_argument('--evm_eps', type=float, default=1e-8,
                       help='Epsilon for EVM computation')
    parser.add_argument('--evm_symbol_stride', type=int, default=1,
                       help='Downsampling factor for constellation EVM loss (e.g. 10 for SPS=10)')

    # Advanced Communications-Aware Losses
    parser.add_argument('--cma_lambda', type=float, default=0.1,
                       help='Lambda for CMA (Constant Modulus Algorithm) penalty')
    parser.add_argument('--mf_lambda', type=float, default=0.5,
                       help='Lambda for Matched-Filter Differentiable Demod Loss')
    parser.add_argument('--mf_window', type=int, default=5,
                       help='Moving average window size for Matched-Filter Loss')
    parser.add_argument('--evm_weight', type=float, default=1.0,
                       help='EVM term weight inside EVM+CONST')
    parser.add_argument('--const_weight', type=float, default=0.2,
                       help='Constellation term weight inside EVM+CONST')
    parser.add_argument('--evm_lambda', type=float, default=0.1,
                       help='Global lambda for EVM+CONST in combo losses')
    # Physics-informed loss knobs
    parser.add_argument('--rms_lambda', type=float, default=0.5,
                       help='Weight for RMS gain constraint in PIT-SI-SNR+RMS / PIT-SI-SNR+RMS+CONST')
    parser.add_argument('--const_lambda', type=float, default=0.1,
                       help='Weight for constellation proximity in PIT-SI-SNR+CONST / PIT-SI-SNR+RMS+CONST')
    parser.add_argument('--qam_reg_weight', type=float, default=0.0,
                       help='Weight of QAM Lattice output regularizer loss term')
    parser.add_argument('--topology_aux_weight', type=float, default=None,
                       help='Weight of the stage 193 output topology-stat auxiliary loss')
    parser.add_argument('--topology_aux_axis_weight', type=float, default=None,
                       help='Axis-stat term weight inside stage 193 topology loss')
    parser.add_argument('--topology_aux_amp_weight', type=float, default=None,
                       help='Amplitude-stat term weight inside stage 193 topology loss')
    parser.add_argument('--topology_aux_phase_weight', type=float, default=None,
                       help='Phase-step-stat term weight inside stage 193 topology loss')
    parser.add_argument('--topology_aux_kurtosis_weight', type=float, default=None,
                       help='Kurtosis-stat term weight inside stage 193 topology loss')
    parser.add_argument('--sep_constraint_weight', type=float, default=None,
                       help='Weight of the stage 195 separation-mechanism auxiliary loss')
    parser.add_argument('--sep_constraint_mix_weight', type=float, default=None,
                       help='Mixture-consistency term weight inside stage 195 separation loss')
    parser.add_argument('--sep_constraint_corr_weight', type=float, default=None,
                       help='Cross-source correlation term weight inside stage 195 separation loss')
    parser.add_argument('--sep_constraint_energy_weight', type=float, default=None,
                       help='Energy-balance term weight inside stage 195 separation loss')
    parser.add_argument('--gpahuber_lambda', type=float, default=0.1,
                       help='Weight for gain/phase-aligned complex Huber in PIT-SI-SNR+Huber+GPAHuber')
    parser.add_argument('--gpahuber_beta', type=float, default=1.0,
                       help='Huber beta for normalized gain/phase-aligned complex error')
    parser.add_argument('--xtalk_lambda', type=float, default=0.05,
                       help='Weight for cross-source leakage suppression in PIT-SI-SNR+Huber+XTALK')
    parser.add_argument('--noiseres_lambda', type=float, default=0.02,
                       help='Weight for residual-noise decorrelation in PIT-SI-SNR+Huber+XTALK+NOISERES')
    parser.add_argument('--noiseres_corr_weight', type=float, default=1.0,
                       help='Weight for residual-vs-source correlation inside NOISERES')
    parser.add_argument('--noiseres_whiteness_weight', type=float, default=0.25,
                       help='Weight for residual short-lag whiteness inside NOISERES')
    parser.add_argument('--noiseres_max_lag', type=int, default=8,
                       help='Maximum lag for residual whiteness term inside NOISERES')
    parser.add_argument('--cyclic_profile_lambda', type=float, default=0.05,
                       help='Weight for cyclic autocorrelation profile consistency in PIT-SI-SNR+Huber+CyclicProfile')
    parser.add_argument('--cyclic_profile_cross_lambda', type=float, default=0.01,
                       help='Weight for cross-source cyclic-correlation suppression in PIT-SI-SNR+Huber+CyclicProfile')
    parser.add_argument('--cyclic_profile_alphas', nargs='+', type=float,
                       default=[0.0, 0.05, 0.1, 0.15, 0.2],
                       help='Normalized cyclic frequencies in cycles/sample for CyclicProfile loss')
    parser.add_argument('--cyclic_profile_lags', nargs='+', type=int,
                       default=[0, 1, 2, 4, 8],
                       help='Sample lags for CyclicProfile loss')
    # Multi-Resolution STFT v2 (TF-GridNet / BSRNN style)
    parser.add_argument('--mrstft_v2_n_ffts', nargs='+', type=int, default=[256, 512, 1024, 2048],
                       help='FFT sizes for MR-STFT v2 loss (default: 256 512 1024 2048)')
    parser.add_argument('--mrstft_v2_lambda', type=float, default=0.3,
                       help='Weight for MR-STFT v2 term in composite losses')
    # Mixture Consistency
    parser.add_argument('--mixcons_lambda', type=float, default=0.1,
                       help='Weight for mixture consistency constraint (sources sum to mixture)')
    
    # BER evaluation parameters
    parser.add_argument('--report_ber', action='store_true',
                       help='Report BER during train/val logging and on final evaluation / in test mode')
    parser.add_argument('--ber_offset_search', action='store_true',
                       help='Search sampling offset for BER (slower but more accurate)')
    parser.add_argument('--ber_mode', type=str, default='file',
                        choices=['frame', 'file'],
                       help='BER evaluation mode: frame (per-frame) or file (stream-level, may be pessimistic when residual CFO/phase drift is present)')
    parser.add_argument('--ber_num_files', type=int, default=2,
                       help='When ber_mode=file, evaluate this many files per SNR (0=all)')
    parser.add_argument('--ber_compute_oracle', action='store_true',
                       help='Also compute oracle BER debug upper bound. This is much slower, especially for 8PSK-A.')

    # AMR (Automatic Modulation Recognition) joint training parameters
    parser.add_argument('--amr_mode', type=str, default='sep_only',
                       choices=['sep_only', 'cls_only', 'joint'],
                       help='AMR model mode: sep_only=separation only, '
                            'cls_only=classifier only (separator frozen), '
                            'joint=end-to-end')
    parser.add_argument('--amr_cls_weight', type=float, default=0.1,
                       help='Weight for AMR classification loss in joint mode')

    # Soft Demodulation joint training parameters
    parser.add_argument('--demod_mode', type=str, default='sep_only',
                       choices=['sep_only', 'demod_only', 'joint'],
                       help='SoftDemod model mode: sep_only=separation only, '
                            'demod_only=demod head only (separator frozen), '
                            'joint=end-to-end')
    parser.add_argument('--demod_mode_phase1', type=str, default=None,
                       choices=['sep_only', 'demod_only', 'joint'],
                       help='Optional phase-1 demod mode override when --two_phase_loss is enabled')
    parser.add_argument('--demod_mode_phase2', type=str, default=None,
                       choices=['sep_only', 'demod_only', 'joint'],
                       help='Optional phase-2 demod mode override when --two_phase_loss is enabled')
    parser.add_argument('--demod_weight', type=float, default=0.5,
                       help='Weight for soft demodulation BCE loss in joint mode')
    parser.add_argument('--demod_symbol_weight', type=float, default=0.5,
                       help='Weight for symbol-classification CE loss in joint demod mode')
    parser.add_argument('--demod_teacher_weight', type=float, default=0.0,
                       help='Extra clean-source demod warm-up loss weight during early phase-2 training')
    parser.add_argument('--demod_teacher_phase2_epochs', type=int, default=0,
                       help='Number of early phase-2 epochs to apply clean-source demod warm-up')

    # Other parameters
    parser.add_argument("--test_args", action="store_true",
                       help="Test command line arguments only, don't run actual experiment")
    
    parsed_args = parser.parse_args()
    parsed_args.data_choice = normalize_data_choice(parsed_args.data_choice)
    parsed_args.pretrain_data_choices = [
        normalize_data_choice(choice)
        for choice in (parsed_args.pretrain_data_choices or [])
    ] or None
    if parsed_args.resume_checkpoint and parsed_args.init_checkpoint:
        parser.error("--resume_checkpoint and --init_checkpoint are mutually exclusive")
    if parsed_args.pretrain_dataset_weights and not parsed_args.pretrain_data_choices:
        parser.error("--pretrain_dataset_weights requires --pretrain_data_choices")
    if (
        parsed_args.pretrain_dataset_weights
        and len(parsed_args.pretrain_dataset_weights) != len(parsed_args.pretrain_data_choices)
    ):
        parser.error(
            "--pretrain_dataset_weights must have one value per --pretrain_data_choices entry"
        )
    if parsed_args.pretrain_data_choices and parsed_args.report_ber:
        parser.error("BER reporting is not defined for mixed-modulation joint pretraining")
    if parsed_args.pretrain_data_choices and parsed_args.loss_fun == 'PIT-SI-SNR+Huber+AMR':
        parser.error("The current AMR auxiliary loss does not support per-sample mixed-domain labels")
    if parsed_args.pretrain_data_choices and parsed_args.train_subset_ratio != 1.0:
        parser.error(
            "--train_subset_ratio is not supported with joint pretraining because it would "
            "break dataset-balanced sampling"
        )
    parsed_args.num_sources = resolve_num_sources(parsed_args)
    loss_was_explicit = any(arg == '--loss_fun' or arg.startswith('--loss_fun=') for arg in sys.argv[1:])
    if parsed_args.stage == 87 and not loss_was_explicit:
        parsed_args.loss_fun = 'PIT-DEMOD-AWARE'
    if parsed_args.stage == 95 and not loss_was_explicit:
        parsed_args.loss_fun = 'PIT-DEMOD-AWARE'
    if parsed_args.stage == 214 and not loss_was_explicit:
        parsed_args.loss_fun = 'PIT-SI-SNR+Huber+RMS'
    if parsed_args.stage == 231 and not loss_was_explicit:
        parsed_args.loss_fun = 'PIT-SI-SNR+Huber'
    return parsed_args


def main():
    """Main function"""
    args = parse_arguments()
    
    if args.test_args:
        print(f"[Parameter testing mode]")
        print(f"Data: {args.data_choice}")
        if args.pretrain_data_choices:
            print(f"Pretrain Data: {args.pretrain_data_choices}")
            print(f"Pretrain Sampling: {args.pretrain_sampling}")
            print(f"Pretrain Input Size: {args.pretrain_input_size or 'data_choice default'}")
            print(f"Pretrain Length Policy: {args.pretrain_length_policy}")
        print(f"Mode: {args.mode}")
        print(f"Signal Names: {args.source_names}")
        print(f"Seed: {args.seed}")
        print(f"Split Strategy: {args.split_strategy}")
        print(f"Multiple Runs: {args.multiple_runs}")
        if args.multiple_runs:
            print(f"Number of Runs: {args.num_runs}")
            print(f"Starting Seed: {args.start_seed}")
        return
    
    print("=" * 80)
    print("IQU Mamba 1D Training and Testing Program")
    print("=" * 80)

    if args.mode == 'ablation_losses':
        if args.multiple_runs:
            raise ValueError("ablation_losses mode does not support --multiple_runs. Use one fixed seed.")
        args.compact_results = True
        if args.early_stop_patience < args.num_epochs:
            args.early_stop_patience = args.num_epochs

        run_loss_ablation_experiment(args, seed=args.seed)
        return
    
    if args.multiple_runs:
        # Multiple runs experiment
        if args.mode == 'test_data':
            print(f"Data loading test mode - Performing {args.num_runs} tests...")
        else:
            print(f"Starting {args.num_runs} experiments...")
        
        results_folders = []
        
        for i in range(args.num_runs):
            current_seed = args.start_seed + i
            
            if args.mode == 'test_data':
                print(f"\n--- Data loading test {i+1}/{args.num_runs} (Seed: {current_seed}) ---")
            else:
                print(f"\n--- Running {i+1}/{args.num_runs} experiment (Seed: {current_seed}) ---")
            
            try:
                results_folder = run_single_experiment(args, seed=current_seed)
                results_folders.append(results_folder)
                
                if args.mode == 'test_data':
                    print(f"Data loading test {i+1} completed")
                else:
                    print(f"Experiment {i+1} completed: {results_folder}")
            except Exception as e:
                if args.mode == 'test_data':
                    print(f"Data loading test {i+1} failed: {str(e)}")
                else:
                    print(f"Experiment {i+1} failed: {str(e)}")
                continue
        
        if args.mode == 'test_data':
            print(f"\nAll data loading tests completed!")
        else:
            print(f"\nAll experiments completed! Result folders: {results_folders}")

    
    else:
        # Single run experiment
        if args.mode == 'test_data':
            if args.seed is not None:
                print(f"Running data loading test (Seed: {args.seed})")
            else:
                print("Running data loading test (No fixed seed)")
        else:
            if args.seed is not None:
                print(f"Running single experiment (Seed: {args.seed})")
            else:
                print("Running single experiment (No fixed seed)")
        
        try:
            results_folder = run_single_experiment(args, seed=args.seed)
            
            if args.mode == 'test_data':
                print(f"Data loading test completed!")
            else:
                print(f"Experiment completed! Results saved in: {results_folder}")
        except Exception as e:
            if args.mode == 'test_data':
                print(f"Data loading test failed: {str(e)}")
            else:
                print(f"Experiment failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()

