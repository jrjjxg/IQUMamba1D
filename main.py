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

from data_loader.dataloader import create_data_loaders
from util.logger import create_logger
from util.evaluation import test_model
from util.training import train_model
from util.utils import Create_Mamba_model, create_new_results_folder
from util.config import MambaConfig
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
)

LOSS_FUNCTION_CHOICES = [
    'MSE', 'L1', 'Huber',
    'SI-SNR', 'PIT-SI-SNR',
    'SI-SNR+MSE', 'SI-SNR+Huber',
    'PIT-MSE', 'PIT-L1', 'PIT-Huber',
    'PIT-SI-SNR+MSE', 'PIT-SI-SNR+Huber',
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
    'PIT-SI-SNR+Huber+AMR', 'PIT-SI-SNR+Huber+CMA', 'PIT-SI-SNR+Huber+MF',
    'PIT-SI-SNR+Huber+Independence'
]


def normalize_data_choice(data_choice: str) -> str:
    """Normalize Kaggle-safe aliases back to canonical dataset names."""
    aliases = {
        "QPSK16APSK-A": "QPSK+16APSK-A",
        "QPSK16APSK-B": "QPSK+16APSK-B",
        "QPSK-16APSK-A": "QPSK+16APSK-A",
        "QPSK-16APSK-B": "QPSK+16APSK-B",
    }
    return aliases.get(data_choice, data_choice)


DATA_CHOICE_CHOICES = [
    'debug_random',
    'TorchSig', '2016', '2018',
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
    if loss_name == 'PIT-SI-SNR+Huber+Independence':
        criterion = lambda outputs, targets: pit_si_snr_huber_ind_loss(
            outputs,
            targets,
            num_sources=2,
            alpha=args.pit_si_snr_huber_alpha,
            beta=args.pit_si_snr_huber_beta,
            alpha_ind=0.1,
            eps=1e-8
        )
        return criterion, (
            f"PIT-SI-SNR+Huber+Ind(a={args.pit_si_snr_huber_alpha}, "
            f"b={args.pit_si_snr_huber_beta})"
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
            f"rms_位={args.rms_lambda})"
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
        # This loss needs the input mixture — handled by a special wrapper.
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
                    # sep_only mode — no classification needed, fall back
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
    # deterministic mode — this lets it fall back gracefully.
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
        37: CONFIG_ROOT / "model_config_bimamba_softdemod_v2.yaml",
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
        ('complex_adapter_use_input', 'complex_adapter_no_input', 'complex_adapter_use_input'),
        ('complex_adapter_use_output', 'complex_adapter_no_output', 'complex_adapter_use_output'),
        ('complex_adapter_zero_init', 'complex_adapter_no_zero_init', 'complex_adapter_zero_init'),
        ('cyclofresh_zero_init', 'cyclofresh_no_zero_init', 'cyclofresh_zero_init'),
        ('blind_cyclofresh_zero_init', 'blind_cyclofresh_no_zero_init', 'blind_cyclofresh_zero_init'),
        ('estimated_cyclofresh_zero_init', 'estimated_cyclofresh_no_zero_init', 'estimated_cyclofresh_zero_init'),
        ('multipeak_cyclofresh_zero_init', 'multipeak_cyclofresh_no_zero_init', 'multipeak_cyclofresh_zero_init'),
        ('sample_cyclofresh_zero_init', 'sample_cyclofresh_no_zero_init', 'sample_cyclofresh_zero_init'),
        ('freqbias_zero_init', 'freqbias_no_zero_init', 'freqbias_zero_init'),
        ('cycliccorr_zero_init', 'cycliccorr_no_zero_init', 'cycliccorr_zero_init'),
        ('leakcancel_zero_init', 'leakcancel_no_zero_init', 'leakcancel_zero_init'),
        ('blindstat_zero_init', 'blindstat_no_zero_init', 'blindstat_zero_init'),
        ('multirate_zero_init', 'multirate_no_zero_init', 'multirate_zero_init'),
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


def create_model(args, cfg, input_size, device, logger):
    """Create and return model"""

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

        # forward special attributes
        if hasattr(crit, 'needs_mixture'):
            qam_reg_crit.needs_mixture = crit.needs_mixture
        if hasattr(crit, 'needs_bits'):
            qam_reg_crit.needs_bits = crit.needs_bits

        return qam_reg_crit, f"{crit_name}+QAMReg(λ={qam_weight})"

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
        
        phase1_criterion, phase1_name = _wrap_qam_reg(phase1_criterion, phase1_name)
        phase2_criterion, phase2_name = _wrap_qam_reg(phase2_criterion, phase2_name)
        
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
        criterion, loss_name = _wrap_qam_reg(criterion, loss_name)
        logger.info(f"Loss Function: {loss_name}")

    # Optimizer defaults by model family
    # IQUMamba/BiMamba: usually stable at 1e-3
    # TFGridNet/SPMamba/Conformer-GridNet: generally prefers 3e-4
    # TIGER variants: often more sensitive, use 1e-4 by default
    if args.stage in {6, 10, 11, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 30, 31}:
        default_lr = 3e-4
    elif args.stage in {7, 8, 9, 28}:
        default_lr = 1e-4
    else:
        default_lr = 1e-3
    base_lr = args.lr if args.lr is not None else default_lr
    lr = args.lr_phase1 if args.two_phase_loss and args.lr_phase1 is not None else base_lr
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


def _load_model_state_for_test(weights_path: Path, logger):
    """Load model weights for test mode, falling back to sibling checkpoints if needed."""
    errors = []
    for candidate in _candidate_weight_paths(weights_path):
        if not candidate.exists():
            errors.append(f"{candidate}: file does not exist")
            continue
        try:
            loaded_weights = _torch_load_for_test(candidate, logger)
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
    input_size, num_points, input_channels = setup_data_parameters(args.data_choice, logger)
    
    # Constant settings
    SIGNAL_NAMES = args.source_names
    NUM_SOURCES = len(SIGNAL_NAMES)
    batch_size = args.batch_size
    
    logger.info(f"Signal Names: {SIGNAL_NAMES}")
    logger.info(f"Number of Sources: {NUM_SOURCES}")
    logger.info(f"Input Size: {input_size}")
    logger.info(f"Input Channels: {input_channels}")
    logger.info(f"Batch Size: {batch_size}")
    cfg = MambaConfig(get_model_config_path(args.stage), train=True)
    apply_model_config_overrides(args, cfg)
    cfg._load_enc_config()
    train_aug_config = resolve_train_aug_config(cfg, args=args, num_epochs=args.num_epochs)
    device = resolve_training_device(torch, require_cuda=args.require_cuda)
    logger.info(f"Using device: {device}")
    log_accelerator_diagnostics(logger, collect_accelerator_diagnostics(torch))
    pin_memory = should_pin_memory(device, args.no_pin_memory)
    logger.info(f"DataLoader pin_memory: {pin_memory}")
    
    try:
        # Create data loaders
        logger.info("Starting to create data loaders...")
        train_loader, val_loader, snr_loaders = create_data_loaders(
            batch_size,
            data_choice=args.data_choice,
            num_sources=NUM_SOURCES,
            matlab_data_root=args.synthetic_root,
            public_data_root=args.public_root,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            split_strategy=args.split_strategy,
            train_aug_config=train_aug_config,
        )
        
        logger.info("✓ Data loaders created successfully!")
        
        # Test training data loader
        logger.info("Testing training data loader...")
        train_iter = iter(train_loader)
        train_batch = next(train_iter)
        logger.info(f"✓ Training batch shapes: {[x.shape for x in train_batch]}")
        
        # Test validation data loader
        logger.info("Testing validation data loader...")
        val_iter = iter(val_loader)
        val_batch = next(val_iter)
        logger.info(f"✓ Validation batch shapes: {[x.shape for x in val_batch]}")
        
        # Test SNR data loaders
        logger.info("Testing SNR data loaders...")
        for snr, snr_loader in snr_loaders.items():
            snr_iter = iter(snr_loader)
            snr_batch = next(snr_iter)
            logger.info(f"✓ SNR {snr}dB batch shapes: {[x.shape for x in snr_batch]}")
        
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
    input_size, num_points, input_channels = setup_data_parameters(args.data_choice, logger)
    
    # Get config file path and copy
    config_path = get_model_config_path(args.stage)
    if not compact_mode:
        shutil.copy2(config_path, str(RESULTS_ROOT / results_folder / "config"))
    
    # Create config object
    cfg = MambaConfig(config_path, train=True)
    apply_model_config_overrides(args, cfg)
    
    
    # Create model
    model = create_model(args, cfg, input_size, device, logger)
    train_aug_config = resolve_train_aug_config(cfg, args=args)
    if train_aug_config:
        recommended_cmd = (
            f"python main.py --mode train --data_choice {args.data_choice} "
            f"--source_names {' '.join(args.source_names)} --stage {args.stage} "
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
    NUM_SOURCES = len(SIGNAL_NAMES)
    num_epochs = args.num_epochs
    early_stop_patience = args.early_stop_patience
    
    # Execute corresponding mode
    if args.mode in ['train', 'test']:
        train_loader, val_loader, snr_loaders = create_data_loaders(
            batch_size,
            data_choice=args.data_choice,
            num_sources=NUM_SOURCES,
            matlab_data_root=args.synthetic_root,
            public_data_root=args.public_root,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            split_strategy=args.split_strategy,
            train_aug_config=train_aug_config,
        )
        if args.mode == 'train':
            train_loader = subsample_train_loader(train_loader, args.train_subset_ratio, seed, logger)
    
    training_history = None
    if args.mode == 'train':
        logger.info("Training Mode")
        training_history = train_model(
            model, scheduler, train_loader, val_loader, snr_loaders,
            criterion, optimizer, device, num_epochs, early_stop_patience,
            logger, results_folder if results_folder else "__compact__", data_choice=args.data_choice,
            num_plots=0 if compact_mode else 1, batch_size=batch_size, input_size=input_size,
            signal_names=SIGNAL_NAMES,
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
            eval_pit_metric=args.eval_pit_metric,
            report_phase_flip=args.report_phase_flip,
            phase_flip_tolerance_deg=args.phase_flip_tolerance_deg,
            phase_flip_min_sc=args.phase_flip_min_sc,
            phase_flip_mode=args.phase_flip_mode,
            resume_checkpoint=args.resume_checkpoint,
            resume_allow_partial=args.resume_allow_partial,
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
        state_dict, state_source = _load_model_state_for_test(weights_path, logger)
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
        f"Loss Ablation Ranking — {model_name} / {num_epochs} epochs",
        fontsize=16, fontweight="bold", y=0.98,
    )

    y_pos = np.arange(n)

    # --- Panel 1: SI-SNR_complex (higher is better) ---
    ax = axes[0, 0]
    bars = ax.barh(y_pos, si_snr_vals, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loss_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("SI-SNR_complex (dB)  [↑ higher is better]")
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
    ax.set_xlabel("Pearson Correlation  [↑ higher is better]")
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
    ax.set_xlabel("MSE  [↓ lower is better]")
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
        '--resume_allow_partial',
        action='store_true',
        help="Allow --resume_checkpoint to warm-start only matching model tensors. "
             "Use when initializing a new architecture from an old checkpoint; optimizer and epoch state are not restored.",
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
    parser.add_argument('--uric_ds_weight', type=float, default=0.1,
                       help='Lambda for URIC intermediate deep-supervision stage losses')
    parser.add_argument('--uric_ds_reduction', type=str, default='sum',
                       choices=['sum', 'mean'],
                       help='Reduce URIC intermediate stage losses by sum (paper-style) or mean')
    parser.add_argument('--uric_ds_include_final_stage', action='store_true',
                       help='Also include the final URIC stage in the stage-loss term')
    parser.add_argument('--stage', type=int, default=4,
                       choices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 8192, 16384, 32768],
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
                             '106=ResUNet1D_TFBranch(Time-Frequency Dual-Branch ResUNet)'
                        ))
    parser.add_argument('--source_names', nargs='+', type=str, default=['S1', 'S2'],
                       help='Source names. Defaults to S1 S2 for two-source datasets.')
    parser.add_argument('--eval_pit_metric', type=str, default='si_snr_complex',
                       choices=['si_snr_complex', 'si_snr_real', 'mse'],
                       help='Shared PIT metric used to reorder outputs before reporting validation/test metrics')
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
    loss_was_explicit = any(arg == '--loss_fun' or arg.startswith('--loss_fun=') for arg in sys.argv[1:])
    if parsed_args.stage == 87 and not loss_was_explicit:
        parsed_args.loss_fun = 'PIT-DEMOD-AWARE'
    if parsed_args.stage == 95 and not loss_was_explicit:
        parsed_args.loss_fun = 'PIT-DEMOD-AWARE'
    return parsed_args


def main():
    """Main function"""
    args = parse_arguments()
    
    if args.test_args:
        print(f"[Parameter testing mode]")
        print(f"Data: {args.data_choice}")
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
