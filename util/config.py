import os
import yaml
from typing import Dict, Any

class MambaConfig:
    def __init__(self, config_path: str, train: bool = True):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No Config File: {config_path}")
        
        # Load Config
        with open(config_path, 'r', encoding='utf-8') as config_file:
            self.cfg = yaml.safe_load(config_file)
        self.model_config = self.cfg['model_config']
        
    
    def _load_enc_config(self):
        cfg = self.model_config
        self.model_type = cfg.get('model_type', 'iqumamba').lower()
        self.input_channels = cfg['input_channels']
        self.num_classes = cfg['num_classes']
        self.tfgridnet_config = cfg.get('tfgridnet_config', {})
        self.tiger_config = cfg.get('tiger_config', {})
        self.spmamba_config = cfg.get('spmamba_config', {})
        self.conformer_gridnet_config = cfg.get('conformer_gridnet_config', {})
        self.dual_domain_config = cfg.get('dual_domain_config', {})
        self.nes2net_config = cfg.get('nes2net_config', {})
        self.ctdcrn_config = cfg.get('ctdcrn_config', {})
        self.rf_bandscnet_config = cfg.get('rf_bandscnet_config', {})
        self.complex_dpnet_config = cfg.get('complex_dpnet_config', {})
        self.complex_convtasnet_config = cfg.get('complex_convtasnet_config', {})
        self.complex_sourceslot_config = cfg.get('complex_sourceslot_config', {})
        self.complex_attractor_config = cfg.get('complex_attractor_config', {})
        self.multires_stft_mask_config = cfg.get('multires_stft_mask_config', {})

        # Training-only low-SNR stage settings (211-216, 223, 225). Keep these on the
        # config object even though these stages use the plain IQUMamba
        # model, so main.py can propagate them into the criterion/train loop.
        low_snr_training_fields = (
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
            'latent_mask_residual_weight',
            'latent_mask_mixture_weight',
            'latent_mask_residual_beta',
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
        for field in low_snr_training_fields:
            if field in cfg:
                setattr(self, field, cfg[field])

        if self.model_type in ('tiger', 'spmamba', 'conformer_gridnet'):
            self.n_stages = cfg.get('n_stages', 0)
            self.features_per_stage = cfg.get('features_per_stage', [])
            self.kernel_sizes = cfg.get('kernel_sizes', [])
            self.strides = cfg.get('strides', [])
            self.n_conv_per_stage = cfg.get('n_conv_per_stage', [])
            self.n_conv_per_stage_decoder = cfg.get('n_conv_per_stage_decoder', [])
            self.deep_supervision = cfg.get('deep_supervision', False)
            return

        if self.model_type == 'tfgridnet':
            self.n_stages = cfg.get('n_stages', 0)
            self.features_per_stage = cfg.get('features_per_stage', [])
            self.kernel_sizes = cfg.get('kernel_sizes', [])
            self.strides = cfg.get('strides', [])
            self.n_conv_per_stage = cfg.get('n_conv_per_stage', [])
            self.n_conv_per_stage_decoder = cfg.get('n_conv_per_stage_decoder', [])
            self.deep_supervision = cfg.get('deep_supervision', False)
            return

        if self.model_type == 'icassp_baseline_unet':
            self.k_neurons = cfg.get('k_neurons', 32)
            self.k_sz = cfg.get('k_sz', 3)
            self.long_k_sz = cfg.get('long_k_sz', 101)
            self.dropout_first = cfg.get('dropout_first', 0.25)
            self.dropout_rest = cfg.get('dropout_rest', 0.5)
            return

        if self.model_type == 'iq_resdilated_unet':
            self.k_neurons = cfg.get('k_neurons', 64)
            self.k_sz = cfg.get('k_sz', 3)
            self.long_k_sz = cfg.get('long_k_sz', 101)
            self.dropout_first = cfg.get('dropout_first', 0.10)
            self.dropout_rest = cfg.get('dropout_rest', 0.10)
            self.shallow_dilated_channels = cfg.get(
                'shallow_dilated_channels', [128, 64]
            )
            self.skip_gate_groups = cfg.get('skip_gate_groups', 16)
            self.use_bottleneck_bimamba = cfg.get('use_bottleneck_bimamba', True)
            self.mamba_d_state = cfg.get('mamba_d_state', 16)
            self.mamba_d_conv = cfg.get('mamba_d_conv', 4)
            self.mamba_expand = cfg.get('mamba_expand', 2)
            self.mamba_dropout = cfg.get('mamba_dropout', 0.0)
            self.mamba_scale_init = cfg.get('mamba_scale_init', 1e-2)
            return

        if self.model_type == 'rfchallenge_rfdemucs':
            self.rfdemucs_hidden = cfg.get('rfdemucs_hidden', 64)
            self.rfdemucs_depth = cfg.get('rfdemucs_depth', 5)
            self.rfdemucs_kernel_size = cfg.get('rfdemucs_kernel_size', 8)
            self.rfdemucs_stride = cfg.get('rfdemucs_stride', 2)
            self.rfdemucs_resample = cfg.get('rfdemucs_resample', 2)
            self.rfdemucs_growth = cfg.get('rfdemucs_growth', 2.0)
            self.rfdemucs_max_hidden = cfg.get('rfdemucs_max_hidden', 10_000)
            self.rfdemucs_normalize = cfg.get('rfdemucs_normalize', False)
            self.rfdemucs_glu = cfg.get('rfdemucs_glu', True)
            self.rfdemucs_rescale = cfg.get('rfdemucs_rescale', 0.1)
            self.rfdemucs_lstm_layers = cfg.get('rfdemucs_lstm_layers', 2)
            self.rfdemucs_sinc_zeros = cfg.get('rfdemucs_sinc_zeros', 56)
            return

        if self.model_type in {
            'icassp_baseline_wavenet',
            'icassp_wavenet_mamba',
            'icassp_wavenet_bimamba',
            'icassp_wavenet_multirate_mamba',
            'icassp_wavenet_multirate_bimamba',
            'icassp_wavenet_interleaved_mamba',
            'icassp_wavenet_interleaved_bimamba',
            'icassp_wavenet_chunk_mamba_strong_fusion',
            'icassp_wavenet_interleaved_gated_bimamba',
            'icassp_wavenet_mamba_film_controller',
            'icassp_wavenet_mamba_dilation_skip_router',
            'icassp_wavenet_interleaved_phase_aware_reverse_mamba',
            'icassp_wavenet_interleaved_crossscale_bimamba',
            'icassp_wavenet_interleaved_stage235_memory',
            'icassp_wavenet_interleaved_physical_moe_bimamba',
            'icassp_wavenet_interleaved_cyclofresh',
            'icassp_wavenet_antialiased_mamba',
            'icassp_wavenet_temporal_physical_controller',
            'icassp_symbol_clock_wavenet',
            'icassp_complex_wavenet',
            'kutii_learnable_dilation_wavenet',
            'kutii_dual_source_wavenet',
        }:
            self.residual_channels = cfg.get('residual_channels', 64)
            self.residual_layers = cfg.get('residual_layers', 30)
            self.dilation_cycle_length = cfg.get('dilation_cycle_length', 10)
            self.max_dilation = cfg.get('max_dilation', 1024)
            self.mamba_d_state = cfg.get('mamba_d_state', 16)
            self.mamba_d_conv = cfg.get('mamba_d_conv', 4)
            self.mamba_expand = cfg.get('mamba_expand', 2)
            self.mamba_dropout = cfg.get('mamba_dropout', 0.0)
            self.mamba_scale_init = cfg.get('mamba_scale_init', 1e-2)
            self.mamba_channels = cfg.get('mamba_channels', 64)
            self.mamba_downsample_factor = cfg.get('mamba_downsample_factor', 4)
            self.mamba_insert_after_block = cfg.get('mamba_insert_after_block', 10)
            self.mamba_chunk_size = cfg.get('mamba_chunk_size', 64)
            self.mamba_chunk_hop = cfg.get('mamba_chunk_hop', 32)
            self.mamba_fusion_gain_init = cfg.get('mamba_fusion_gain_init', 1.0)
            self.mamba_fusion_norm_eps = cfg.get('mamba_fusion_norm_eps', 1e-6)
            self.mamba_backward_gate_init = cfg.get(
                'mamba_backward_gate_init', 0.0
            )
            # Stages 269/270: global Mamba controls for the second WaveNet cycle.
            self.mamba_controller_hidden = cfg.get('mamba_controller_hidden', 128)
            self.mamba_controller_dropout = cfg.get('mamba_controller_dropout', 0.0)
            self.mamba_control_gate_max_delta = cfg.get(
                'mamba_control_gate_max_delta', 0.5
            )
            self.mamba_control_film_max_delta = cfg.get(
                'mamba_control_film_max_delta', 0.1
            )
            self.mamba_router_strength = cfg.get('mamba_router_strength', 0.25)
            self.mamba_router_temperature = cfg.get(
                'mamba_router_temperature', 1.0
            )
            # Stage 271: raw-I/Q complex-conjugate reverse Mamba branch.
            self.phase_reverse_mamba_channels = cfg.get(
                'phase_reverse_mamba_channels', self.mamba_channels
            )
            self.phase_reverse_downsample_factor = cfg.get(
                'phase_reverse_downsample_factor', self.mamba_downsample_factor
            )
            self.phase_reverse_d_state = cfg.get(
                'phase_reverse_d_state', self.mamba_d_state
            )
            self.phase_reverse_d_conv = cfg.get(
                'phase_reverse_d_conv', self.mamba_d_conv
            )
            self.phase_reverse_expand = cfg.get(
                'phase_reverse_expand', self.mamba_expand
            )
            self.phase_reverse_dropout = cfg.get(
                'phase_reverse_dropout', self.mamba_dropout
            )
            self.phase_reverse_scale_init = cfg.get(
                'phase_reverse_scale_init', 1e-2
            )
            # Stage 265: Stage-235-style BiMamba global memory + compact K/V.
            self.cross_scale_kv_tokens = cfg.get('cross_scale_kv_tokens', 64)
            self.cross_scale_num_heads = cfg.get('cross_scale_num_heads', 4)
            self.cross_scale_dropout = cfg.get('cross_scale_dropout', 0.0)
            self.cross_scale_residual_scale_init = cfg.get(
                'cross_scale_residual_scale_init', 1e-2
            )
            # Stage 266: Stage-255-style identity/global/physical/joint MoE.
            self.fusion_global_kv_tokens = cfg.get('fusion_global_kv_tokens', 64)
            self.fusion_num_heads = cfg.get('fusion_num_heads', 4)
            self.fusion_dropout = cfg.get('fusion_dropout', 0.0)
            self.fusion_channel_scale_init = cfg.get('fusion_channel_scale_init', 0.1)
            self.fusion_channel_scale_max = cfg.get('fusion_channel_scale_max', 0.5)
            self.fusion_router_hidden = cfg.get('fusion_router_hidden', 64)
            self.fusion_expert_prior = cfg.get(
                'fusion_expert_prior', [0.7, 0.1, 0.1, 0.1]
            )
            self.fusion_condition_hidden = cfg.get('fusion_condition_hidden', 16)
            self.fusion_condition_embedding = cfg.get('fusion_condition_embedding', 16)
            self.fusion_trust_penalty_init = cfg.get('fusion_trust_penalty_init', 0.1)
            self.fusion_trust_penalty_enable = cfg.get('fusion_trust_penalty_enable', True)
            self.fusion_condition_routing_enable = cfg.get(
                'fusion_condition_routing_enable', True
            )
            self.physical_cyclic_lags = cfg.get('physical_cyclic_lags', [0, 1, 2, 4, 8])
            self.physical_polyphase_branches = cfg.get('physical_polyphase_branches', 8)
            self.physical_symbol_orders = cfg.get('physical_symbol_orders', [2, 4, 8])
            self.physical_min_cyclic_freq = cfg.get('physical_min_cyclic_freq', 1.0 / 64.0)
            self.physical_max_cyclic_freq = cfg.get('physical_max_cyclic_freq', 1.0 / 8.0)
            self.physical_cyclic_temperature = cfg.get('physical_cyclic_temperature', 0.25)
            # Stage 267: Stage-79 metadata-free estimated Cyclo-FRESH prior.
            self.estimated_cyclofresh_min_freq = cfg.get(
                'estimated_cyclofresh_min_freq', 1.0 / 64.0
            )
            self.estimated_cyclofresh_max_freq = cfg.get(
                'estimated_cyclofresh_max_freq', 1.0 / 8.0
            )
            self.estimated_cyclofresh_default_freq = cfg.get(
                'estimated_cyclofresh_default_freq', 1.0 / 32.0
            )
            self.estimated_cyclofresh_momentum = cfg.get(
                'estimated_cyclofresh_momentum', 0.05
            )
            self.estimated_cyclofresh_hidden_channels = cfg.get(
                'estimated_cyclofresh_hidden_channels', 8
            )
            self.estimated_cyclofresh_kernel_size = cfg.get(
                'estimated_cyclofresh_kernel_size', 9
            )
            self.estimated_cyclofresh_scale_init = cfg.get(
                'estimated_cyclofresh_scale_init', 1e-2
            )
            self.estimated_cyclofresh_gate_hidden = cfg.get(
                'estimated_cyclofresh_gate_hidden', 8
            )
            self.estimated_cyclofresh_zero_init = cfg.get(
                'estimated_cyclofresh_zero_init', True
            )
            # Stages 278-282: anti-aliased context, ordered physical control,
            # and symbol-clock-conditioned WaveNet variants.
            self.antialias_taps_per_phase = cfg.get(
                'antialias_taps_per_phase', 8
            )
            self.antialias_cutoff_ratio = cfg.get(
                'antialias_cutoff_ratio', 0.90
            )
            self.controller_insert_after_block = cfg.get(
                'controller_insert_after_block', 10
            )
            self.physical_token_channels = cfg.get(
                'physical_token_channels', 64
            )
            self.physical_chunk_size = cfg.get('physical_chunk_size', 64)
            self.physical_chunk_hop = cfg.get('physical_chunk_hop', 32)
            self.physical_lags = cfg.get(
                'physical_lags', [1, 2, 4, 8, 16, 32]
            )
            self.symbol_candidate_periods = cfg.get(
                'symbol_candidate_periods', [2, 4, 8, 16, 32]
            )
            self.symbol_dilation_multipliers = cfg.get(
                'symbol_dilation_multipliers', [1, 2, 4, 8, 16]
            )
            self.symbol_max_dilation = cfg.get('symbol_max_dilation', 512)
            self.pre_residual_layers = cfg.get('pre_residual_layers', 10)
            self.pre_dilation_cycle_length = cfg.get(
                'pre_dilation_cycle_length', 10
            )
            self.adaptive_layers = cfg.get('adaptive_layers', 5)
            self.use_widely_linear_stem = cfg.get(
                'use_widely_linear_stem', False
            )
            self.use_temporal_controls = cfg.get(
                'use_temporal_controls', False
            )
            self.temporal_control_gate_max_delta = cfg.get(
                'temporal_control_gate_max_delta', 0.5
            )
            self.temporal_control_film_max_delta = cfg.get(
                'temporal_control_film_max_delta', 0.1
            )
            self.symbol_evidence_strength = cfg.get(
                'symbol_evidence_strength', 1.0
            )
            self.symbol_router_temperature = cfg.get(
                'symbol_router_temperature', 0.35
            )
            # Stages 283-289: compute-conscious strict-complex WaveNet
            # ablations based only on the Stage-273 20-block backbone.
            self.complex_layers = cfg.get('complex_layers', 20)
            self.strict_complex_output = cfg.get(
                'strict_complex_output', False
            )
            self.use_conjugate_adapter = cfg.get(
                'use_conjugate_adapter', False
            )
            self.conjugate_adapter_max = cfg.get(
                'conjugate_adapter_max', 0.15
            )
            self.complex_norm_enable = cfg.get(
                'complex_norm_enable', False
            )
            self.complex_norm_eps = cfg.get('complex_norm_eps', 1e-6)
            return

        if self.model_type == 'ctdcrn':
            self.n_stages = cfg.get('n_stages', 0)
            self.features_per_stage = cfg.get('features_per_stage', [])
            self.kernel_sizes = cfg.get('kernel_sizes', [])
            self.strides = cfg.get('strides', [])
            self.n_conv_per_stage = cfg.get('n_conv_per_stage', [])
            self.n_conv_per_stage_decoder = cfg.get('n_conv_per_stage_decoder', [])
            self.deep_supervision = cfg.get('deep_supervision', False)
            return

        if self.model_type in (
            'rf_bandscnet',
            'complex_dpnet',
            'complex_convtasnet',
            'complex_sourceslot',
            'complex_attractor',
            'multires_stft_mask',
            'sepbamba_unet1d',
        ):
            self.n_stages = cfg.get('n_stages', 0)
            self.features_per_stage = cfg.get('features_per_stage', [])
            self.kernel_sizes = cfg.get('kernel_sizes', [])
            self.strides = cfg.get('strides', [])
            self.n_conv_per_stage = cfg.get('n_conv_per_stage', [])
            self.n_conv_per_stage_decoder = cfg.get('n_conv_per_stage_decoder', [])
            self.deep_supervision = cfg.get('deep_supervision', False)
            return

        self.n_stages = cfg.get('n_stages', 0)
        self.features_per_stage = cfg.get('features_per_stage', [])
        self.kernel_sizes = cfg.get('kernel_sizes', [])
        self.strides = cfg.get('strides', [])
        self.n_conv_per_stage = cfg.get('n_conv_per_stage', [])
        self.n_conv_per_stage_decoder = cfg.get('n_conv_per_stage_decoder', [])
        self.deep_supervision = cfg.get('deep_supervision', False)
        if 'evidence_moe_hidden_channels' in cfg:
            self.evidence_moe_hidden_channels = int(cfg['evidence_moe_hidden_channels'])
        if 'evidence_moe_max_delta' in cfg:
            self.evidence_moe_max_delta = float(cfg['evidence_moe_max_delta'])
        if 'evidence_moe_identity_bias' in cfg:
            self.evidence_moe_identity_bias = float(cfg['evidence_moe_identity_bias'])
        if 'evidence_moe_router_temperature' in cfg:
            self.evidence_moe_router_temperature = float(cfg['evidence_moe_router_temperature'])
        if 'evidence_moe_route_hard_eval' in cfg:
            self.evidence_moe_route_hard_eval = bool(cfg['evidence_moe_route_hard_eval'])
        if 'evidence_moe_lag_bank' in cfg:
            self.evidence_moe_lag_bank = [int(lag) for lag in cfg['evidence_moe_lag_bank']]
        if 'evidence_moe_return_route_aux' in cfg:
            self.evidence_moe_return_route_aux = bool(cfg['evidence_moe_return_route_aux'])
        if 'noise_prior_hidden' in cfg:
            self.noise_prior_hidden = int(cfg['noise_prior_hidden'])
        if 'noise_prior_embedding' in cfg:
            self.noise_prior_embedding = int(cfg['noise_prior_embedding'])
        if 'noise_prior_patch_size' in cfg:
            self.noise_prior_patch_size = int(cfg['noise_prior_patch_size'])
        if 'noise_prior_patch_stride' in cfg:
            self.noise_prior_patch_stride = int(cfg['noise_prior_patch_stride'])
        if 'sync_hidden' in cfg:
            self.sync_hidden = int(cfg['sync_hidden'])
        if 'sync_kernel_size' in cfg:
            self.sync_kernel_size = int(cfg['sync_kernel_size'])
        if 'sync_scale_init' in cfg:
            self.sync_scale_init = float(cfg['sync_scale_init'])
        if 'sync_lags' in cfg:
            self.sync_lags = [int(lag) for lag in cfg['sync_lags']]
        if 'sync_eps' in cfg:
            self.sync_eps = float(cfg['sync_eps'])
        if 'decoder_mamba_stages' in cfg:
            self.decoder_mamba_stages = [int(stage) for stage in cfg['decoder_mamba_stages']]

        # Optional MIT-inspired large-kernel stem parameters (bimamba_lk)
        if 'stem_kernel_size' in cfg:
            self.stem_kernel_size = int(cfg['stem_kernel_size'])
        if 'stem_channels' in cfg:
            self.stem_channels = int(cfg['stem_channels'])

        # Optional complex stem / bottleneck bridge parameters (bimamba_csb)
        if 'complex_stem_hidden_channels' in cfg:
            self.complex_stem_hidden_channels = int(cfg['complex_stem_hidden_channels'])
        if 'complex_stem_kernel_size' in cfg:
            self.complex_stem_kernel_size = int(cfg['complex_stem_kernel_size'])
        if 'complex_bottleneck_hidden_channels' in cfg:
            self.complex_bottleneck_hidden_channels = int(cfg['complex_bottleneck_hidden_channels'])
        if 'complex_bottleneck_num_blocks' in cfg:
            self.complex_bottleneck_num_blocks = int(cfg['complex_bottleneck_num_blocks'])
        if 'complex_bottleneck_kernel_size' in cfg:
            self.complex_bottleneck_kernel_size = int(cfg['complex_bottleneck_kernel_size'])
        if 'complex_bottleneck_dilation_growth' in cfg:
            self.complex_bottleneck_dilation_growth = int(cfg['complex_bottleneck_dilation_growth'])
        if 'complex_bottleneck_zero_init' in cfg:
            self.complex_bottleneck_zero_init = bool(cfg['complex_bottleneck_zero_init'])
        if 'complex_encoder_channels' in cfg:
            self.complex_encoder_channels = [int(c) for c in cfg['complex_encoder_channels']]
        if 'complex_encoder_num_stages' in cfg:
            self.complex_encoder_num_stages = int(cfg['complex_encoder_num_stages'])
        if 'complex_encoder_kernel_size' in cfg:
            self.complex_encoder_kernel_size = int(cfg['complex_encoder_kernel_size'])
        if 'complex_to_real_channels' in cfg:
            self.complex_to_real_channels = int(cfg['complex_to_real_channels'])
        if 'complex_mask_latent_channels' in cfg:
            self.complex_mask_latent_channels = int(cfg['complex_mask_latent_channels'])
        if 'complex_reconstruction_kernel_size' in cfg:
            self.complex_reconstruction_kernel_size = int(cfg['complex_reconstruction_kernel_size'])
        if 'complex_eps' in cfg:
            self.complex_eps = float(cfg['complex_eps'])
        if 'complex_leaky_relu_slope' in cfg:
            self.complex_leaky_relu_slope = float(cfg['complex_leaky_relu_slope'])
        if 'cs_scan_chunk_size' in cfg:
            self.cs_scan_chunk_size = int(cfg['cs_scan_chunk_size'])
        if 'cs_scan_shift_size' in cfg:
            self.cs_scan_shift_size = None if cfg['cs_scan_shift_size'] is None else int(cfg['cs_scan_shift_size'])
        if 'cs_scan_gate_hidden' in cfg:
            self.cs_scan_gate_hidden = int(cfg['cs_scan_gate_hidden'])
        if 'cag_alpha_init' in cfg:
            self.cag_alpha_init = float(cfg['cag_alpha_init'])
        if 'cag_gate_hidden' in cfg:
            self.cag_gate_hidden = int(cfg['cag_gate_hidden'])
        if 'phasediff_eps' in cfg:
            self.phasediff_eps = float(cfg['phasediff_eps'])
        if 'cmasc_gate_hidden' in cfg:
            self.cmasc_gate_hidden = int(cfg['cmasc_gate_hidden'])
        if 'cmasc_residual_scale_init' in cfg:
            self.cmasc_residual_scale_init = float(cfg['cmasc_residual_scale_init'])
        if 'cmasc_eps' in cfg:
            self.cmasc_eps = float(cfg['cmasc_eps'])
        if 'mamba_residual_scale_init' in cfg:
            self.mamba_residual_scale_init = float(cfg['mamba_residual_scale_init'])
        if 'bimamba_apply_stages' in cfg:
            self.bimamba_apply_stages = [int(stage) for stage in cfg['bimamba_apply_stages']]
        if 'bimamba_residual_scale_init' in cfg:
            self.bimamba_residual_scale_init = float(cfg['bimamba_residual_scale_init'])
        if 'bimamba_diff_scale_init' in cfg:
            self.bimamba_diff_scale_init = float(cfg['bimamba_diff_scale_init'])
        if 'bimamba_gate_logit_init' in cfg:
            self.bimamba_gate_logit_init = float(cfg['bimamba_gate_logit_init'])
        if 'bimamba_gate_token_scale_init' in cfg:
            self.bimamba_gate_token_scale_init = float(cfg['bimamba_gate_token_scale_init'])
        if 'bimamba_gate_eps' in cfg:
            self.bimamba_gate_eps = float(cfg['bimamba_gate_eps'])
        if 'bimamba_complex_diff_gate_init' in cfg:
            self.bimamba_complex_diff_gate_init = float(cfg['bimamba_complex_diff_gate_init'])
        if 'bimamba_complex_diff_stride' in cfg:
            self.bimamba_complex_diff_stride = int(cfg['bimamba_complex_diff_stride'])
        if 'bimamba_complex_diff_eps' in cfg:
            self.bimamba_complex_diff_eps = float(cfg['bimamba_complex_diff_eps'])
        if 'bimamba_boundary_tau_init' in cfg:
            self.bimamba_boundary_tau_init = float(cfg['bimamba_boundary_tau_init'])
        if 'bimamba_shrinkage_init' in cfg:
            self.bimamba_shrinkage_init = float(cfg['bimamba_shrinkage_init'])
        if 'bimamba_fusion_eps' in cfg:
            self.bimamba_fusion_eps = float(cfg['bimamba_fusion_eps'])
        if 'bimamba_local_kernel_size' in cfg:
            self.bimamba_local_kernel_size = int(cfg['bimamba_local_kernel_size'])
        if 'bimamba_local_gate_init' in cfg:
            self.bimamba_local_gate_init = float(cfg['bimamba_local_gate_init'])
        if 'training_only_deep_supervision' in cfg:
            self.training_only_deep_supervision = bool(cfg['training_only_deep_supervision'])
        if 'shallow_skip_init' in cfg:
            self.shallow_skip_init = float(cfg['shallow_skip_init'])
        if 'shallow_skip_drop_probability' in cfg:
            self.shallow_skip_drop_probability = float(cfg['shallow_skip_drop_probability'])
        if 'local_kernel_size' in cfg:
            self.local_kernel_size = int(cfg['local_kernel_size'])
        if 'local_global_gate_hidden' in cfg:
            self.local_global_gate_hidden = int(cfg['local_global_gate_hidden'])
        if 'mamba_embed_stages' in cfg:
            self.mamba_embed_stages = [int(stage) for stage in cfg['mamba_embed_stages']]
        if 'mamba_embed_d_state' in cfg:
            self.mamba_embed_d_state = int(cfg['mamba_embed_d_state'])
        if 'mamba_embed_d_conv' in cfg:
            self.mamba_embed_d_conv = int(cfg['mamba_embed_d_conv'])
        if 'mamba_embed_expand' in cfg:
            self.mamba_embed_expand = int(cfg['mamba_embed_expand'])
        if 'mamba_embed_scale_init' in cfg:
            self.mamba_embed_scale_init = float(cfg['mamba_embed_scale_init'])
        if 'mamba_embed_local_kernel_size' in cfg:
            self.mamba_embed_local_kernel_size = int(cfg['mamba_embed_local_kernel_size'])
        if 'mamba_embed_gate_hidden' in cfg:
            self.mamba_embed_gate_hidden = int(cfg['mamba_embed_gate_hidden'])
        if 'pco_phase_channels' in cfg:
            self.pco_phase_channels = int(cfg['pco_phase_channels'])
        if 'pco_phase_kernel_size' in cfg:
            self.pco_phase_kernel_size = int(cfg['pco_phase_kernel_size'])
        if 'pco_phase_scale_init' in cfg:
            self.pco_phase_scale_init = float(cfg['pco_phase_scale_init'])
        if 'pco_corr_lags' in cfg:
            self.pco_corr_lags = [int(lag) for lag in cfg['pco_corr_lags']]
        if 'pco_corr_window' in cfg:
            self.pco_corr_window = int(cfg['pco_corr_window'])
        if 'pco_corr_scale_init' in cfg:
            self.pco_corr_scale_init = float(cfg['pco_corr_scale_init'])
        if 'pco_orth_scale_init' in cfg:
            self.pco_orth_scale_init = float(cfg['pco_orth_scale_init'])
        if 'pco_orth_eps' in cfg:
            self.pco_orth_eps = float(cfg['pco_orth_eps'])
        if 'rfscan_chunk_size' in cfg:
            self.rfscan_chunk_size = int(cfg['rfscan_chunk_size'])
        if 'rfscan_shift_size' in cfg:
            self.rfscan_shift_size = None if cfg['rfscan_shift_size'] is None else int(cfg['rfscan_shift_size'])
        if 'rfscan_freq_bands' in cfg:
            self.rfscan_freq_bands = int(cfg['rfscan_freq_bands'])
        if 'rfscan_gate_hidden' in cfg:
            self.rfscan_gate_hidden = int(cfg['rfscan_gate_hidden'])
        if 'rfscan_conv_kernel_size' in cfg:
            self.rfscan_conv_kernel_size = int(cfg['rfscan_conv_kernel_size'])
        if 'rfscan_residual_scale_init' in cfg:
            self.rfscan_residual_scale_init = float(cfg['rfscan_residual_scale_init'])
        if 'rfscan_condition_scale_init' in cfg:
            self.rfscan_condition_scale_init = float(cfg['rfscan_condition_scale_init'])
        if 'rfscan_stft_n_fft' in cfg:
            self.rfscan_stft_n_fft = int(cfg['rfscan_stft_n_fft'])
        if 'rfscan_stft_hop_length' in cfg:
            self.rfscan_stft_hop_length = int(cfg['rfscan_stft_hop_length'])
        if 'rfscan_stft_win_length' in cfg:
            self.rfscan_stft_win_length = None if cfg['rfscan_stft_win_length'] is None else int(cfg['rfscan_stft_win_length'])
        if 'rfscan_stft_freq_bins' in cfg:
            self.rfscan_stft_freq_bins = int(cfg['rfscan_stft_freq_bins'])
        if 'symbol_samples' in cfg:
            self.symbol_samples = int(cfg['symbol_samples'])
        if 'dual_path_chunk_symbols' in cfg:
            self.dual_path_chunk_symbols = int(cfg['dual_path_chunk_symbols'])
        if 'dual_path_hop_symbols' in cfg:
            self.dual_path_hop_symbols = int(cfg['dual_path_hop_symbols'])
        if 'dual_path_residual_scale_init' in cfg:
            self.dual_path_residual_scale_init = float(cfg['dual_path_residual_scale_init'])
        if 'mask_bound' in cfg:
            self.mask_bound = float(cfg['mask_bound'])
        if 'mask_sum_constraint' in cfg:
            self.mask_sum_constraint = bool(cfg['mask_sum_constraint'])
        if 'mask_apply_projection' in cfg:
            self.mask_apply_projection = bool(cfg['mask_apply_projection'])
        if 'mask_project_deep_supervision' in cfg:
            self.mask_project_deep_supervision = bool(cfg['mask_project_deep_supervision'])
        if 'mask_logit_scale_init' in cfg:
            self.mask_logit_scale_init = float(cfg['mask_logit_scale_init'])
        if 'feature_mask_channels' in cfg:
            self.feature_mask_channels = int(cfg['feature_mask_channels'])
        if 'feature_mask_kernel_size' in cfg:
            self.feature_mask_kernel_size = int(cfg['feature_mask_kernel_size'])
        if 'feature_mask_bound' in cfg:
            self.feature_mask_bound = float(cfg['feature_mask_bound'])
        if 'feature_mask_sum_constraint' in cfg:
            self.feature_mask_sum_constraint = bool(cfg['feature_mask_sum_constraint'])
        if 'feature_mask_apply_projection' in cfg:
            self.feature_mask_apply_projection = bool(cfg['feature_mask_apply_projection'])
        if 'feature_mask_project_deep_supervision' in cfg:
            self.feature_mask_project_deep_supervision = bool(cfg['feature_mask_project_deep_supervision'])
        if 'feature_mask_logit_scale_init' in cfg:
            self.feature_mask_logit_scale_init = float(cfg['feature_mask_logit_scale_init'])
        if 'feature_mask_identity_init' in cfg:
            self.feature_mask_identity_init = bool(cfg['feature_mask_identity_init'])
        if 'source_slot_hidden_channels' in cfg:
            self.source_slot_hidden_channels = int(cfg['source_slot_hidden_channels'])
        if 'source_slot_kernel_size' in cfg:
            self.source_slot_kernel_size = int(cfg['source_slot_kernel_size'])
        if 'source_slot_residual_scale_init' in cfg:
            self.source_slot_residual_scale_init = float(cfg['source_slot_residual_scale_init'])
        if 'source_slot_zero_init' in cfg:
            self.source_slot_zero_init = bool(cfg['source_slot_zero_init'])
        if 'source_slot_refine_deep_supervision' in cfg:
            self.source_slot_refine_deep_supervision = bool(cfg['source_slot_refine_deep_supervision'])
        if 'source_slot_apply_train' in cfg:
            self.source_slot_apply_train = bool(cfg['source_slot_apply_train'])
        if 'source_slot_apply_eval' in cfg:
            self.source_slot_apply_eval = bool(cfg['source_slot_apply_eval'])
        if 'noise_mc_apply_projection' in cfg:
            self.noise_mc_apply_projection = bool(cfg['noise_mc_apply_projection'])
        if 'noise_mc_project_during_train' in cfg:
            self.noise_mc_project_during_train = bool(cfg['noise_mc_project_during_train'])
        if 'noise_mc_project_during_eval' in cfg:
            self.noise_mc_project_during_eval = bool(cfg['noise_mc_project_during_eval'])
        if 'noise_mc_source_weight' in cfg:
            self.noise_mc_source_weight = float(cfg['noise_mc_source_weight'])
        if 'noise_mc_noise_weight' in cfg:
            self.noise_mc_noise_weight = float(cfg['noise_mc_noise_weight'])
        if 'noise_head_hidden_channels' in cfg:
            self.noise_head_hidden_channels = int(cfg['noise_head_hidden_channels'])
        if 'noise_head_kernel_size' in cfg:
            self.noise_head_kernel_size = int(cfg['noise_head_kernel_size'])
        if 'noise_head_zero_init' in cfg:
            self.noise_head_zero_init = bool(cfg['noise_head_zero_init'])
        if 'noise_mc_eps' in cfg:
            self.noise_mc_eps = float(cfg['noise_mc_eps'])
        if 'iq_power_norm_eps' in cfg:
            self.iq_power_norm_eps = float(cfg['iq_power_norm_eps'])
        if 'iq_power_norm_detach_scale' in cfg:
            self.iq_power_norm_detach_scale = bool(cfg['iq_power_norm_detach_scale'])
        if 'low_snr_se_hidden_channels' in cfg:
            self.low_snr_se_hidden_channels = int(cfg['low_snr_se_hidden_channels'])
        if 'low_snr_se_kernel_size' in cfg:
            self.low_snr_se_kernel_size = int(cfg['low_snr_se_kernel_size'])
        if 'low_snr_se_scale_init' in cfg:
            self.low_snr_se_scale_init = float(cfg['low_snr_se_scale_init'])
        if 'low_snr_se_zero_init' in cfg:
            self.low_snr_se_zero_init = bool(cfg['low_snr_se_zero_init'])
        if 'low_snr_se_use_projection' in cfg:
            self.low_snr_se_use_projection = bool(cfg['low_snr_se_use_projection'])
        if 'low_snr_se_project_during_train' in cfg:
            self.low_snr_se_project_during_train = bool(cfg['low_snr_se_project_during_train'])
        if 'low_snr_se_project_during_eval' in cfg:
            self.low_snr_se_project_during_eval = bool(cfg['low_snr_se_project_during_eval'])
        if 'low_snr_se_source_weight' in cfg:
            self.low_snr_se_source_weight = float(cfg['low_snr_se_source_weight'])
        if 'low_snr_se_noise_weight' in cfg:
            self.low_snr_se_noise_weight = float(cfg['low_snr_se_noise_weight'])
        if 'low_snr_se_return_aux' in cfg:
            self.low_snr_se_return_aux = bool(cfg['low_snr_se_return_aux'])
        if 'low_snr_se_eps' in cfg:
            self.low_snr_se_eps = float(cfg['low_snr_se_eps'])
        if 'low_snr_cond_hidden_channels' in cfg:
            self.low_snr_cond_hidden_channels = int(cfg['low_snr_cond_hidden_channels'])
        if 'low_snr_cond_kernel_size' in cfg:
            self.low_snr_cond_kernel_size = int(cfg['low_snr_cond_kernel_size'])
        if 'low_snr_cond_gate_hidden' in cfg:
            self.low_snr_cond_gate_hidden = int(cfg['low_snr_cond_gate_hidden'])
        if 'low_snr_cond_scale_init' in cfg:
            self.low_snr_cond_scale_init = float(cfg['low_snr_cond_scale_init'])
        if 'low_snr_cond_zero_init' in cfg:
            self.low_snr_cond_zero_init = bool(cfg['low_snr_cond_zero_init'])
        if 'low_snr_cond_min_freq' in cfg:
            self.low_snr_cond_min_freq = float(cfg['low_snr_cond_min_freq'])
        if 'low_snr_cond_max_freq' in cfg:
            self.low_snr_cond_max_freq = float(cfg['low_snr_cond_max_freq'])
        if 'low_snr_cond_use_projection' in cfg:
            self.low_snr_cond_use_projection = bool(cfg['low_snr_cond_use_projection'])
        if 'low_snr_cond_project_during_train' in cfg:
            self.low_snr_cond_project_during_train = bool(cfg['low_snr_cond_project_during_train'])
        if 'low_snr_cond_project_during_eval' in cfg:
            self.low_snr_cond_project_during_eval = bool(cfg['low_snr_cond_project_during_eval'])
        if 'low_snr_cond_source_weight' in cfg:
            self.low_snr_cond_source_weight = float(cfg['low_snr_cond_source_weight'])
        if 'low_snr_cond_noise_weight' in cfg:
            self.low_snr_cond_noise_weight = float(cfg['low_snr_cond_noise_weight'])
        if 'low_snr_cond_return_aux' in cfg:
            self.low_snr_cond_return_aux = bool(cfg['low_snr_cond_return_aux'])
        if 'low_snr_cond_eps' in cfg:
            self.low_snr_cond_eps = float(cfg['low_snr_cond_eps'])
        if 'wiener_hidden_channels' in cfg:
            self.wiener_hidden_channels = int(cfg['wiener_hidden_channels'])
        if 'wiener_kernel_size' in cfg:
            self.wiener_kernel_size = int(cfg['wiener_kernel_size'])
        if 'wiener_signal_bias_init' in cfg:
            self.wiener_signal_bias_init = float(cfg['wiener_signal_bias_init'])
        if 'wiener_noise_bias_init' in cfg:
            self.wiener_noise_bias_init = float(cfg['wiener_noise_bias_init'])
        if 'wiener_log_power_clip' in cfg:
            self.wiener_log_power_clip = float(cfg['wiener_log_power_clip'])
        if 'wiener_use_projection' in cfg:
            self.wiener_use_projection = bool(cfg['wiener_use_projection'])
        if 'wiener_project_during_train' in cfg:
            self.wiener_project_during_train = bool(cfg['wiener_project_during_train'])
        if 'wiener_project_during_eval' in cfg:
            self.wiener_project_during_eval = bool(cfg['wiener_project_during_eval'])
        if 'wiener_source_weight' in cfg:
            self.wiener_source_weight = float(cfg['wiener_source_weight'])
        if 'wiener_noise_weight' in cfg:
            self.wiener_noise_weight = float(cfg['wiener_noise_weight'])
        if 'wiener_return_aux' in cfg:
            self.wiener_return_aux = bool(cfg['wiener_return_aux'])
        if 'wiener_eps' in cfg:
            self.wiener_eps = float(cfg['wiener_eps'])
        if 'asg_patch_size' in cfg:
            self.asg_patch_size = int(cfg['asg_patch_size'])
        if 'asg_stride' in cfg:
            self.asg_stride = int(cfg['asg_stride'])
        if 'asg_num_bands' in cfg:
            self.asg_num_bands = int(cfg['asg_num_bands'])
        if 'asg_gate_hidden' in cfg:
            self.asg_gate_hidden = int(cfg['asg_gate_hidden'])
        if 'asg_scale_init' in cfg:
            self.asg_scale_init = float(cfg['asg_scale_init'])
        if 'asg_zero_init' in cfg:
            self.asg_zero_init = bool(cfg['asg_zero_init'])
        if 'asg_apply_stages' in cfg:
            self.asg_apply_stages = [int(stage) for stage in cfg['asg_apply_stages']]
        if 'asg_eps' in cfg:
            self.asg_eps = float(cfg['asg_eps'])
        if 'complex_adapter_hidden_channels' in cfg:
            self.complex_adapter_hidden_channels = int(cfg['complex_adapter_hidden_channels'])
        if 'complex_adapter_kernel_size' in cfg:
            self.complex_adapter_kernel_size = int(cfg['complex_adapter_kernel_size'])
        if 'complex_adapter_scale_init' in cfg:
            self.complex_adapter_scale_init = float(cfg['complex_adapter_scale_init'])
        if 'complex_adapter_use_input' in cfg:
            self.complex_adapter_use_input = bool(cfg['complex_adapter_use_input'])
        if 'complex_adapter_use_output' in cfg:
            self.complex_adapter_use_output = bool(cfg['complex_adapter_use_output'])
        if 'complex_adapter_zero_init' in cfg:
            self.complex_adapter_zero_init = bool(cfg['complex_adapter_zero_init'])
        if 'cyclofresh_sps' in cfg:
            self.cyclofresh_sps = int(cfg['cyclofresh_sps'])
        if 'cyclofresh_alphas' in cfg:
            self.cyclofresh_alphas = [float(alpha) for alpha in cfg['cyclofresh_alphas']]
        if 'cyclofresh_hidden_channels' in cfg:
            self.cyclofresh_hidden_channels = int(cfg['cyclofresh_hidden_channels'])
        if 'cyclofresh_kernel_size' in cfg:
            self.cyclofresh_kernel_size = int(cfg['cyclofresh_kernel_size'])
        if 'cyclofresh_scale_init' in cfg:
            self.cyclofresh_scale_init = float(cfg['cyclofresh_scale_init'])
        if 'cyclofresh_gate_hidden' in cfg:
            self.cyclofresh_gate_hidden = int(cfg['cyclofresh_gate_hidden'])
        if 'cyclofresh_zero_init' in cfg:
            self.cyclofresh_zero_init = bool(cfg['cyclofresh_zero_init'])
        if 'blind_cyclofresh_freqs' in cfg:
            self.blind_cyclofresh_freqs = [float(freq) for freq in cfg['blind_cyclofresh_freqs']]
        if 'blind_cyclofresh_max_delta' in cfg:
            self.blind_cyclofresh_max_delta = float(cfg['blind_cyclofresh_max_delta'])
        if 'blind_cyclofresh_hidden_channels' in cfg:
            self.blind_cyclofresh_hidden_channels = int(cfg['blind_cyclofresh_hidden_channels'])
        if 'blind_cyclofresh_kernel_size' in cfg:
            self.blind_cyclofresh_kernel_size = int(cfg['blind_cyclofresh_kernel_size'])
        if 'blind_cyclofresh_scale_init' in cfg:
            self.blind_cyclofresh_scale_init = float(cfg['blind_cyclofresh_scale_init'])
        if 'blind_cyclofresh_gate_hidden' in cfg:
            self.blind_cyclofresh_gate_hidden = int(cfg['blind_cyclofresh_gate_hidden'])
        if 'blind_cyclofresh_zero_init' in cfg:
            self.blind_cyclofresh_zero_init = bool(cfg['blind_cyclofresh_zero_init'])
        if 'estimated_cyclofresh_min_freq' in cfg:
            self.estimated_cyclofresh_min_freq = float(cfg['estimated_cyclofresh_min_freq'])
        if 'estimated_cyclofresh_max_freq' in cfg:
            self.estimated_cyclofresh_max_freq = float(cfg['estimated_cyclofresh_max_freq'])
        if 'estimated_cyclofresh_default_freq' in cfg:
            self.estimated_cyclofresh_default_freq = float(cfg['estimated_cyclofresh_default_freq'])
        if 'estimated_cyclofresh_momentum' in cfg:
            self.estimated_cyclofresh_momentum = float(cfg['estimated_cyclofresh_momentum'])
        if 'estimated_cyclofresh_hidden_channels' in cfg:
            self.estimated_cyclofresh_hidden_channels = int(cfg['estimated_cyclofresh_hidden_channels'])
        if 'estimated_cyclofresh_kernel_size' in cfg:
            self.estimated_cyclofresh_kernel_size = int(cfg['estimated_cyclofresh_kernel_size'])
        if 'estimated_cyclofresh_scale_init' in cfg:
            self.estimated_cyclofresh_scale_init = float(cfg['estimated_cyclofresh_scale_init'])
        if 'estimated_cyclofresh_gate_hidden' in cfg:
            self.estimated_cyclofresh_gate_hidden = int(cfg['estimated_cyclofresh_gate_hidden'])
        if 'estimated_cyclofresh_zero_init' in cfg:
            self.estimated_cyclofresh_zero_init = bool(cfg['estimated_cyclofresh_zero_init'])
        if 'complex_stem_enable' in cfg:
            self.complex_stem_enable = bool(cfg['complex_stem_enable'])
        if 'complex_norm_eps' in cfg:
            self.complex_norm_eps = float(cfg['complex_norm_eps'])
        if 'multipeak_cyclofresh_min_freq' in cfg:
            self.multipeak_cyclofresh_min_freq = float(cfg['multipeak_cyclofresh_min_freq'])
        if 'multipeak_cyclofresh_max_freq' in cfg:
            self.multipeak_cyclofresh_max_freq = float(cfg['multipeak_cyclofresh_max_freq'])
        if 'multipeak_cyclofresh_default_freq' in cfg:
            self.multipeak_cyclofresh_default_freq = float(cfg['multipeak_cyclofresh_default_freq'])
        if 'multipeak_cyclofresh_momentum' in cfg:
            self.multipeak_cyclofresh_momentum = float(cfg['multipeak_cyclofresh_momentum'])
        if 'multipeak_cyclofresh_num_peaks' in cfg:
            self.multipeak_cyclofresh_num_peaks = int(cfg['multipeak_cyclofresh_num_peaks'])
        if 'multipeak_cyclofresh_guard_bins' in cfg:
            self.multipeak_cyclofresh_guard_bins = int(cfg['multipeak_cyclofresh_guard_bins'])
        if 'multipeak_cyclofresh_hidden_channels' in cfg:
            self.multipeak_cyclofresh_hidden_channels = int(cfg['multipeak_cyclofresh_hidden_channels'])
        if 'multipeak_cyclofresh_kernel_size' in cfg:
            self.multipeak_cyclofresh_kernel_size = int(cfg['multipeak_cyclofresh_kernel_size'])
        if 'multipeak_cyclofresh_scale_init' in cfg:
            self.multipeak_cyclofresh_scale_init = float(cfg['multipeak_cyclofresh_scale_init'])
        if 'multipeak_cyclofresh_gate_hidden' in cfg:
            self.multipeak_cyclofresh_gate_hidden = int(cfg['multipeak_cyclofresh_gate_hidden'])
        if 'multipeak_cyclofresh_reliability_floor' in cfg:
            self.multipeak_cyclofresh_reliability_floor = float(cfg['multipeak_cyclofresh_reliability_floor'])
        if 'multipeak_cyclofresh_zero_init' in cfg:
            self.multipeak_cyclofresh_zero_init = bool(cfg['multipeak_cyclofresh_zero_init'])
        if 'sample_cyclofresh_min_freq' in cfg:
            self.sample_cyclofresh_min_freq = float(cfg['sample_cyclofresh_min_freq'])
        if 'sample_cyclofresh_max_freq' in cfg:
            self.sample_cyclofresh_max_freq = float(cfg['sample_cyclofresh_max_freq'])
        if 'sample_cyclofresh_default_freq' in cfg:
            self.sample_cyclofresh_default_freq = float(cfg['sample_cyclofresh_default_freq'])
        if 'sample_cyclofresh_num_peaks' in cfg:
            self.sample_cyclofresh_num_peaks = int(cfg['sample_cyclofresh_num_peaks'])
        if 'sample_cyclofresh_guard_bins' in cfg:
            self.sample_cyclofresh_guard_bins = int(cfg['sample_cyclofresh_guard_bins'])
        if 'sample_cyclofresh_hidden_channels' in cfg:
            self.sample_cyclofresh_hidden_channels = int(cfg['sample_cyclofresh_hidden_channels'])
        if 'sample_cyclofresh_kernel_size' in cfg:
            self.sample_cyclofresh_kernel_size = int(cfg['sample_cyclofresh_kernel_size'])
        if 'sample_cyclofresh_scale_init' in cfg:
            self.sample_cyclofresh_scale_init = float(cfg['sample_cyclofresh_scale_init'])
        if 'sample_cyclofresh_gate_hidden' in cfg:
            self.sample_cyclofresh_gate_hidden = int(cfg['sample_cyclofresh_gate_hidden'])
        if 'sample_cyclofresh_reliability_floor' in cfg:
            self.sample_cyclofresh_reliability_floor = float(cfg['sample_cyclofresh_reliability_floor'])
        if 'sample_cyclofresh_zero_init' in cfg:
            self.sample_cyclofresh_zero_init = bool(cfg['sample_cyclofresh_zero_init'])
        if 'multihyp_cyclic_freqs' in cfg:
            self.multihyp_cyclic_freqs = [float(freq) for freq in cfg['multihyp_cyclic_freqs']]
        if 'multihyp_cyclic_hidden_channels' in cfg:
            self.multihyp_cyclic_hidden_channels = int(cfg['multihyp_cyclic_hidden_channels'])
        if 'multihyp_cyclic_kernel_size' in cfg:
            self.multihyp_cyclic_kernel_size = int(cfg['multihyp_cyclic_kernel_size'])
        if 'multihyp_cyclic_scale_init' in cfg:
            self.multihyp_cyclic_scale_init = float(cfg['multihyp_cyclic_scale_init'])
        if 'multihyp_cyclic_gate_hidden' in cfg:
            self.multihyp_cyclic_gate_hidden = int(cfg['multihyp_cyclic_gate_hidden'])
        if 'multihyp_cyclic_temperature' in cfg:
            self.multihyp_cyclic_temperature = float(cfg['multihyp_cyclic_temperature'])
        if 'multihyp_cyclic_null_logit_init' in cfg:
            self.multihyp_cyclic_null_logit_init = float(cfg['multihyp_cyclic_null_logit_init'])
        if 'multihyp_cyclic_local_bins' in cfg:
            self.multihyp_cyclic_local_bins = int(cfg['multihyp_cyclic_local_bins'])
        if 'multihyp_cyclic_zero_init' in cfg:
            self.multihyp_cyclic_zero_init = bool(cfg['multihyp_cyclic_zero_init'])
        if 'multihyp_cyclic_return_aux' in cfg:
            self.multihyp_cyclic_return_aux = bool(cfg['multihyp_cyclic_return_aux'])
        if 'multihyp_cyclic_eps' in cfg:
            self.multihyp_cyclic_eps = float(cfg['multihyp_cyclic_eps'])
        if 'freqbias_hidden_channels' in cfg:
            self.freqbias_hidden_channels = int(cfg['freqbias_hidden_channels'])
        if 'freqbias_kernel_size' in cfg:
            self.freqbias_kernel_size = int(cfg['freqbias_kernel_size'])
        if 'freqbias_lowpass_kernel_size' in cfg:
            self.freqbias_lowpass_kernel_size = int(cfg['freqbias_lowpass_kernel_size'])
        if 'freqbias_scale_init' in cfg:
            self.freqbias_scale_init = float(cfg['freqbias_scale_init'])
        if 'freqbias_gate_hidden' in cfg:
            self.freqbias_gate_hidden = int(cfg['freqbias_gate_hidden'])
        if 'freqbias_zero_init' in cfg:
            self.freqbias_zero_init = bool(cfg['freqbias_zero_init'])
        if 'cycliccorr_min_freq' in cfg:
            self.cycliccorr_min_freq = float(cfg['cycliccorr_min_freq'])
        if 'cycliccorr_max_freq' in cfg:
            self.cycliccorr_max_freq = float(cfg['cycliccorr_max_freq'])
        if 'cycliccorr_default_freq' in cfg:
            self.cycliccorr_default_freq = float(cfg['cycliccorr_default_freq'])
        if 'cycliccorr_momentum' in cfg:
            self.cycliccorr_momentum = float(cfg['cycliccorr_momentum'])
        if 'cycliccorr_lags' in cfg:
            self.cycliccorr_lags = [int(lag) for lag in cfg['cycliccorr_lags']]
        if 'cycliccorr_hidden_channels' in cfg:
            self.cycliccorr_hidden_channels = int(cfg['cycliccorr_hidden_channels'])
        if 'cycliccorr_kernel_size' in cfg:
            self.cycliccorr_kernel_size = int(cfg['cycliccorr_kernel_size'])
        if 'cycliccorr_scale_init' in cfg:
            self.cycliccorr_scale_init = float(cfg['cycliccorr_scale_init'])
        if 'cycliccorr_gate_hidden' in cfg:
            self.cycliccorr_gate_hidden = int(cfg['cycliccorr_gate_hidden'])
        if 'cycliccorr_zero_init' in cfg:
            self.cycliccorr_zero_init = bool(cfg['cycliccorr_zero_init'])
        if 'blindstat_hidden' in cfg:
            self.blindstat_hidden = int(cfg['blindstat_hidden'])
        if 'blindstat_kernel_size' in cfg:
            self.blindstat_kernel_size = int(cfg['blindstat_kernel_size'])
        if 'blindstat_scale_init' in cfg:
            self.blindstat_scale_init = float(cfg['blindstat_scale_init'])
        if 'blindstat_cyclic_min_freq' in cfg:
            self.blindstat_cyclic_min_freq = float(cfg['blindstat_cyclic_min_freq'])
        if 'blindstat_cyclic_max_freq' in cfg:
            self.blindstat_cyclic_max_freq = float(cfg['blindstat_cyclic_max_freq'])
        if 'blindstat_cyclic_default_freq' in cfg:
            self.blindstat_cyclic_default_freq = float(cfg['blindstat_cyclic_default_freq'])
        if 'blindstat_zero_init' in cfg:
            self.blindstat_zero_init = bool(cfg['blindstat_zero_init'])
        if 'multirate_hidden_channels' in cfg:
            self.multirate_hidden_channels = int(cfg['multirate_hidden_channels'])
        if 'multirate_kernel_sizes' in cfg:
            self.multirate_kernel_sizes = [int(k) for k in cfg['multirate_kernel_sizes']]
        if 'multirate_dilations' in cfg:
            self.multirate_dilations = [int(d) for d in cfg['multirate_dilations']]
        if 'multirate_scale_init' in cfg:
            self.multirate_scale_init = float(cfg['multirate_scale_init'])
        if 'multirate_zero_init' in cfg:
            self.multirate_zero_init = bool(cfg['multirate_zero_init'])
        if 'psk_prior_hidden_channels' in cfg:
            self.psk_prior_hidden_channels = int(cfg['psk_prior_hidden_channels'])
        if 'psk_prior_harmonics' in cfg:
            self.psk_prior_harmonics = [int(harmonic) for harmonic in cfg['psk_prior_harmonics']]
        if 'psk_prior_kernel_size' in cfg:
            self.psk_prior_kernel_size = int(cfg['psk_prior_kernel_size'])
        if 'psk_prior_scale_init' in cfg:
            self.psk_prior_scale_init = float(cfg['psk_prior_scale_init'])
        if 'psk_prior_reliability_floor' in cfg:
            self.psk_prior_reliability_floor = float(cfg['psk_prior_reliability_floor'])
        if 'psk_prior_zero_init' in cfg:
            self.psk_prior_zero_init = bool(cfg['psk_prior_zero_init'])
        if 'qam_prior_hidden_channels' in cfg:
            self.qam_prior_hidden_channels = int(cfg['qam_prior_hidden_channels'])
        if 'qam_prior_axis_level_bank' in cfg:
            self.qam_prior_axis_level_bank = [int(levels) for levels in cfg['qam_prior_axis_level_bank']]
        if 'qam_prior_temperature' in cfg:
            self.qam_prior_temperature = float(cfg['qam_prior_temperature'])
        if 'qam_prior_kernel_size' in cfg:
            self.qam_prior_kernel_size = int(cfg['qam_prior_kernel_size'])
        if 'qam_prior_scale_init' in cfg:
            self.qam_prior_scale_init = float(cfg['qam_prior_scale_init'])
        if 'qam_prior_reliability_floor' in cfg:
            self.qam_prior_reliability_floor = float(cfg['qam_prior_reliability_floor'])
        if 'qam_prior_zero_init' in cfg:
            self.qam_prior_zero_init = bool(cfg['qam_prior_zero_init'])
        if 'apsk_prior_hidden_channels' in cfg:
            self.apsk_prior_hidden_channels = int(cfg['apsk_prior_hidden_channels'])
        if 'apsk_prior_ring_radii' in cfg:
            self.apsk_prior_ring_radii = [float(radius) for radius in cfg['apsk_prior_ring_radii']]
        if 'apsk_prior_temperature' in cfg:
            self.apsk_prior_temperature = float(cfg['apsk_prior_temperature'])
        if 'apsk_prior_kernel_size' in cfg:
            self.apsk_prior_kernel_size = int(cfg['apsk_prior_kernel_size'])
        if 'apsk_prior_scale_init' in cfg:
            self.apsk_prior_scale_init = float(cfg['apsk_prior_scale_init'])
        if 'apsk_prior_reliability_floor' in cfg:
            self.apsk_prior_reliability_floor = float(cfg['apsk_prior_reliability_floor'])
        if 'apsk_prior_zero_init' in cfg:
            self.apsk_prior_zero_init = bool(cfg['apsk_prior_zero_init'])
        if 'topology_aux_weight' in cfg:
            self.topology_aux_weight = float(cfg['topology_aux_weight'])
        if 'topology_aux_axis_weight' in cfg:
            self.topology_aux_axis_weight = float(cfg['topology_aux_axis_weight'])
        if 'topology_aux_amp_weight' in cfg:
            self.topology_aux_amp_weight = float(cfg['topology_aux_amp_weight'])
        if 'topology_aux_phase_weight' in cfg:
            self.topology_aux_phase_weight = float(cfg['topology_aux_phase_weight'])
        if 'topology_aux_kurtosis_weight' in cfg:
            self.topology_aux_kurtosis_weight = float(cfg['topology_aux_kurtosis_weight'])
        if 'feature_topology_hidden_channels' in cfg:
            self.feature_topology_hidden_channels = int(cfg['feature_topology_hidden_channels'])
        if 'feature_topology_kernel_size' in cfg:
            self.feature_topology_kernel_size = int(cfg['feature_topology_kernel_size'])
        if 'feature_topology_scale_init' in cfg:
            self.feature_topology_scale_init = float(cfg['feature_topology_scale_init'])
        if 'feature_topology_apply_stages' in cfg:
            self.feature_topology_apply_stages = [int(stage) for stage in cfg['feature_topology_apply_stages']]
        if 'feature_topology_zero_init' in cfg:
            self.feature_topology_zero_init = bool(cfg['feature_topology_zero_init'])
        if 'sep_constraint_weight' in cfg:
            self.sep_constraint_weight = float(cfg['sep_constraint_weight'])
        if 'sep_constraint_mix_weight' in cfg:
            self.sep_constraint_mix_weight = float(cfg['sep_constraint_mix_weight'])
        if 'sep_constraint_corr_weight' in cfg:
            self.sep_constraint_corr_weight = float(cfg['sep_constraint_corr_weight'])
        if 'sep_constraint_energy_weight' in cfg:
            self.sep_constraint_energy_weight = float(cfg['sep_constraint_energy_weight'])
        if 'cyclic_wiener_hidden_channels' in cfg:
            self.cyclic_wiener_hidden_channels = int(cfg['cyclic_wiener_hidden_channels'])
        if 'cyclic_wiener_kernel_size' in cfg:
            self.cyclic_wiener_kernel_size = int(cfg['cyclic_wiener_kernel_size'])
        if 'cyclic_wiener_min_freq' in cfg:
            self.cyclic_wiener_min_freq = float(cfg['cyclic_wiener_min_freq'])
        if 'cyclic_wiener_max_freq' in cfg:
            self.cyclic_wiener_max_freq = float(cfg['cyclic_wiener_max_freq'])
        if 'cyclic_wiener_default_freq' in cfg:
            self.cyclic_wiener_default_freq = float(cfg['cyclic_wiener_default_freq'])
        if 'cyclic_wiener_num_harmonics' in cfg:
            self.cyclic_wiener_num_harmonics = int(cfg['cyclic_wiener_num_harmonics'])
        if 'cyclic_wiener_scale_init' in cfg:
            self.cyclic_wiener_scale_init = float(cfg['cyclic_wiener_scale_init'])
        if 'cyclic_wiener_projection_strength' in cfg:
            self.cyclic_wiener_projection_strength = float(cfg['cyclic_wiener_projection_strength'])
        if 'cyclic_wiener_zero_init' in cfg:
            self.cyclic_wiener_zero_init = bool(cfg['cyclic_wiener_zero_init'])
        if 'leakcancel_lags' in cfg:
            self.leakcancel_lags = [int(lag) for lag in cfg['leakcancel_lags']]
        if 'leakcancel_hidden' in cfg:
            self.leakcancel_hidden = int(cfg['leakcancel_hidden'])
        if 'leakcancel_scale_init' in cfg:
            self.leakcancel_scale_init = float(cfg['leakcancel_scale_init'])
        if 'leakcancel_mc_scale_init' in cfg:
            self.leakcancel_mc_scale_init = float(cfg['leakcancel_mc_scale_init'])
        if 'leakcancel_mc_weight_mode' in cfg:
            self.leakcancel_mc_weight_mode = str(cfg['leakcancel_mc_weight_mode'])
        if 'leakcancel_mode' in cfg:
            self.leakcancel_mode = str(cfg['leakcancel_mode'])
        if 'leakcancel_coeff_limit' in cfg:
            self.leakcancel_coeff_limit = float(cfg['leakcancel_coeff_limit'])
        if 'leakcancel_zero_init' in cfg:
            self.leakcancel_zero_init = bool(cfg['leakcancel_zero_init'])
        if 'constellation_type' in cfg:
            self.constellation_type = str(cfg['constellation_type'])
        if 'constellation_order' in cfg:
            self.constellation_order = int(cfg['constellation_order'])
        if 'cgr_hidden_channels' in cfg:
            self.cgr_hidden_channels = int(cfg['cgr_hidden_channels'])
        if 'cgr_kernel_size' in cfg:
            self.cgr_kernel_size = int(cfg['cgr_kernel_size'])
        if 'cgr_temperature' in cfg:
            self.cgr_temperature = float(cfg['cgr_temperature'])
        if 'cgr_dropout' in cfg:
            self.cgr_dropout = float(cfg['cgr_dropout'])
        if 'cgr_gate_init' in cfg:
            self.cgr_gate_init = float(cfg['cgr_gate_init'])
        if 'cgr_residual_scale_init' in cfg:
            self.cgr_residual_scale_init = float(cfg['cgr_residual_scale_init'])
        if 'cgr_use_mixture_residual' in cfg:
            self.cgr_use_mixture_residual = bool(cfg['cgr_use_mixture_residual'])
        if 'cgr_zero_init' in cfg:
            self.cgr_zero_init = bool(cfg['cgr_zero_init'])
        if 'cgr_refine_deep_supervision' in cfg:
            self.cgr_refine_deep_supervision = bool(cfg['cgr_refine_deep_supervision'])
        if 'cgr_apply_train' in cfg:
            self.cgr_apply_train = bool(cfg['cgr_apply_train'])
        if 'cgr_apply_eval' in cfg:
            self.cgr_apply_eval = bool(cfg['cgr_apply_eval'])

        # Optional Jamba-style hybrid parameters (bimamba_jamba)
        if 'attn_stages' in cfg:
            self.attn_stages = [int(s) for s in cfg['attn_stages']]
        if 'attn_n_heads' in cfg:
            self.attn_n_heads = int(cfg['attn_n_heads'])
        if 'attn_dropout' in cfg:
            self.attn_dropout = float(cfg['attn_dropout'])
        if 'attn_ffn_expand' in cfg:
            self.attn_ffn_expand = int(cfg['attn_ffn_expand'])

        # Optional compressed-KV cross-scale attention parameters (stages 235-237)
        if 'cross_scale_query_stages' in cfg:
            self.cross_scale_query_stages = [int(s) for s in cfg['cross_scale_query_stages']]
        if 'cross_scale_global_stage' in cfg:
            self.cross_scale_global_stage = int(cfg['cross_scale_global_stage'])
        if 'cross_scale_kv_tokens' in cfg:
            self.cross_scale_kv_tokens = int(cfg['cross_scale_kv_tokens'])
        if 'cross_scale_num_heads' in cfg:
            self.cross_scale_num_heads = int(cfg['cross_scale_num_heads'])
        if 'cross_scale_dropout' in cfg:
            self.cross_scale_dropout = float(cfg['cross_scale_dropout'])
        if 'cross_scale_residual_scale_init' in cfg:
            self.cross_scale_residual_scale_init = float(cfg['cross_scale_residual_scale_init'])
        if 'cross_scale_evidence_gate' in cfg:
            self.cross_scale_evidence_gate = bool(cfg['cross_scale_evidence_gate'])
        if 'cross_scale_evidence_hidden' in cfg:
            self.cross_scale_evidence_hidden = int(cfg['cross_scale_evidence_hidden'])
        if 'cross_scale_evidence_eps' in cfg:
            self.cross_scale_evidence_eps = float(cfg['cross_scale_evidence_eps'])
        if 'cross_scale_variant' in cfg:
            self.cross_scale_variant = str(cfg['cross_scale_variant'])
        if 'cross_scale_aligned_window_radius' in cfg:
            self.cross_scale_aligned_window_radius = int(cfg['cross_scale_aligned_window_radius'])
        if 'cross_scale_aligned_global_tokens' in cfg:
            self.cross_scale_aligned_global_tokens = int(cfg['cross_scale_aligned_global_tokens'])
        if 'cross_scale_coarse_kv_tokens' in cfg:
            self.cross_scale_coarse_kv_tokens = int(cfg['cross_scale_coarse_kv_tokens'])
        if 'cross_scale_fine_kv_tokens' in cfg:
            self.cross_scale_fine_kv_tokens = int(cfg['cross_scale_fine_kv_tokens'])
        if 'cross_scale_multires_gate_hidden' in cfg:
            self.cross_scale_multires_gate_hidden = int(cfg['cross_scale_multires_gate_hidden'])
        if 'cross_scale_bounded_max_scale' in cfg:
            self.cross_scale_bounded_max_scale = float(cfg['cross_scale_bounded_max_scale'])
        if 'cross_scale_bounded_initial_scale' in cfg:
            self.cross_scale_bounded_initial_scale = float(cfg['cross_scale_bounded_initial_scale'])
        if 'cross_scale_channel_gate_hidden' in cfg:
            self.cross_scale_channel_gate_hidden = int(cfg['cross_scale_channel_gate_hidden'])

        # Optional Stage-12 Mamba/KV-attention ablations (stages 243-247).
        kv_attention_int_fields = (
            'phase_fusion_kv_tokens',
            'phase_fusion_num_heads',
            'phase_fusion_head_dim',
            'physical_query_stage',
            'physical_num_heads',
            'physical_polyphase_branches',
            'bottleneck_attention_stage',
            'bottleneck_attention_num_heads',
            'hymba_stage',
            'hymba_num_heads',
            'rf_physical_query_stage',
            'rf_physical_num_heads',
            'rf_physical_stft_n_fft',
            'rf_physical_stft_hop_length',
            'rf_physical_stft_win_length',
            'rf_physical_num_subbands',
            'rf_physical_temporal_tokens',
            'rf_physical_polyphase_branches',
            'fusion_query_stage',
            'fusion_global_stage',
            'fusion_global_kv_tokens',
            'fusion_num_heads',
            'fusion_bottleneck_num_heads',
            'fusion_router_hidden',
            'fusion_condition_hidden',
            'fusion_condition_embedding',
            'fusion_film_hidden',
            'fusion_physical_stage',
            'fusion_physical_film_hidden',
        )
        kv_attention_float_fields = (
            'phase_fusion_dropout',
            'phase_fusion_scale_init',
            'physical_dropout',
            'physical_residual_scale_init',
            'physical_min_cyclic_freq',
            'physical_max_cyclic_freq',
            'physical_cyclic_temperature',
            'bottleneck_attention_dropout',
            'bottleneck_attention_scale_init',
            'hymba_dropout',
            'hymba_mamba_scale_init',
            'hymba_attention_scale_init',
            'hymba_attention_scale_max',
            'rf_physical_dropout',
            'rf_physical_residual_scale_init',
            'rf_physical_min_cyclic_freq',
            'rf_physical_max_cyclic_freq',
            'rf_physical_cyclic_temperature',
            'fusion_dropout',
            'fusion_global_scale_init',
            'fusion_physical_scale_init',
            'fusion_channel_scale_init',
            'fusion_channel_scale_max',
            'fusion_bottleneck_dropout',
            'fusion_bottleneck_scale_init',
            'fusion_router_gate_init',
            'fusion_router_gate_max',
            'fusion_trust_penalty_init',
            'fusion_route_candidate_probability',
            'fusion_film_max_delta',
            'fusion_physical_film_max_delta',
        )
        kv_attention_list_fields = (
            'physical_cyclic_lags',
            'physical_symbol_orders',
            'rf_physical_cyclic_lags',
            'rf_physical_symbol_orders',
        )
        for field in kv_attention_int_fields:
            if field in cfg:
                setattr(self, field, int(cfg[field]))
        for field in kv_attention_float_fields:
            if field in cfg:
                setattr(self, field, float(cfg[field]))
        for field in kv_attention_list_fields:
            if field in cfg:
                setattr(self, field, [int(value) for value in cfg[field]])
        if 'fusion_router_prior' in cfg:
            self.fusion_router_prior = [float(value) for value in cfg['fusion_router_prior']]
        if 'fusion_expert_prior' in cfg:
            self.fusion_expert_prior = [float(value) for value in cfg['fusion_expert_prior']]
        if 'fusion_return_route_aux' in cfg:
            self.fusion_return_route_aux = bool(cfg['fusion_return_route_aux'])
        for field in (
            'fusion_trust_penalty_enable',
            'fusion_condition_routing_enable',
            'fusion_counterfactual_enable',
        ):
            if field in cfg:
                setattr(self, field, bool(cfg[field]))

        # Optional ConvNeXt-style large-kernel parameters (convnext)
        if 'lk_kernel_size' in cfg:
            self.lk_kernel_size = int(cfg['lk_kernel_size'])
        if 'lk_expand' in cfg:
            self.lk_expand = int(cfg['lk_expand'])

        # Optional Transformer baseline parameters (transformer1d)
        if 'transformer_n_heads' in cfg:
            self.transformer_n_heads = int(cfg['transformer_n_heads'])
        if 'transformer_dropout' in cfg:
            self.transformer_dropout = float(cfg['transformer_dropout'])
        if 'transformer_ffn_expand' in cfg:
            self.transformer_ffn_expand = int(cfg['transformer_ffn_expand'])
        if 'transformer_token_layout' in cfg:
            self.transformer_token_layout = str(cfg['transformer_token_layout'])
        if 'transformer_pos_encoding' in cfg:
            self.transformer_pos_encoding = str(cfg['transformer_pos_encoding'])
        if 'complex_attention_score' in cfg:
            self.complex_attention_score = str(cfg['complex_attention_score'])

        # Optional AMR classifier head parameters (bimamba_amr)
        if 'num_mod_classes' in cfg:
            self.num_mod_classes = int(cfg['num_mod_classes'])
        if 'cls_hidden' in cfg:
            self.cls_hidden = int(cfg['cls_hidden'])
        if 'cls_mamba_dim' in cfg:
            self.cls_mamba_dim = int(cfg['cls_mamba_dim'])
        if 'cls_dropout' in cfg:
            self.cls_dropout = float(cfg['cls_dropout'])
        if 'detach_cls' in cfg:
            self.detach_cls = bool(cfg['detach_cls'])

        # Optional Soft Demodulation head parameters (bimamba_softdemod)
        if 'demod_num_bits' in cfg:
            self.demod_num_bits = int(cfg['demod_num_bits'])
        if 'demod_bits_per_symbol' in cfg:
            self.demod_bits_per_symbol = int(cfg['demod_bits_per_symbol'])
        if 'demod_hidden' in cfg:
            self.demod_hidden = int(cfg['demod_hidden'])
        if 'demod_rnn_hidden' in cfg:
            self.demod_rnn_hidden = int(cfg['demod_rnn_hidden'])
        if 'demod_dropout' in cfg:
            self.demod_dropout = float(cfg['demod_dropout'])
        if 'detach_demod' in cfg:
            self.detach_demod = bool(cfg['detach_demod'])
        if 'demod_adapter_hidden' in cfg:
            self.demod_adapter_hidden = int(cfg['demod_adapter_hidden'])
        if 'demod_symbol_hidden' in cfg:
            self.demod_symbol_hidden = int(cfg['demod_symbol_hidden'])
        if 'demod_context_layers' in cfg:
            self.demod_context_layers = int(cfg['demod_context_layers'])
        if 'demod_symbol_logit_scale' in cfg:
            self.demod_symbol_logit_scale = float(cfg['demod_symbol_logit_scale'])
        if 'demod_timing_offsets' in cfg:
            self.demod_timing_offsets = int(cfg['demod_timing_offsets'])
        if 'demod_attn_heads' in cfg:
            self.demod_attn_heads = int(cfg['demod_attn_heads'])

        # Optional mixture-consistency projection parameters (bimamba_mcproj)
        if 'mc_weight_mode' in cfg:
            self.mc_weight_mode = str(cfg['mc_weight_mode'])
        if 'mc_weight_power' in cfg:
            self.mc_weight_power = float(cfg['mc_weight_power'])
        if 'mc_min_weight' in cfg:
            self.mc_min_weight = float(cfg['mc_min_weight'])
        if 'mc_eps' in cfg:
            self.mc_eps = float(cfg['mc_eps'])
        if 'mc_detach_weights' in cfg:
            self.mc_detach_weights = bool(cfg['mc_detach_weights'])
        if 'mc_project_deep_supervision' in cfg:
            self.mc_project_deep_supervision = bool(cfg['mc_project_deep_supervision'])
        if 'mc_apply_train' in cfg:
            self.mc_apply_train = bool(cfg['mc_apply_train'])
        if 'mc_apply_eval' in cfg:
            self.mc_apply_eval = bool(cfg['mc_apply_eval'])

        # Optional unrolled residual IC head parameters (bimamba_uric)
        if 'ric_num_steps' in cfg:
            self.ric_num_steps = int(cfg['ric_num_steps'])
        if 'ric_hidden_channels' in cfg:
            self.ric_hidden_channels = int(cfg['ric_hidden_channels'])
        if 'ric_kernel_size' in cfg:
            self.ric_kernel_size = int(cfg['ric_kernel_size'])
        if 'ric_dropout' in cfg:
            self.ric_dropout = float(cfg['ric_dropout'])
        if 'ric_tied_steps' in cfg:
            self.ric_tied_steps = bool(cfg['ric_tied_steps'])
        if 'ric_step_init' in cfg:
            self.ric_step_init = float(cfg['ric_step_init'])
        if 'ric_return_intermediate' in cfg:
            self.ric_return_intermediate = bool(cfg['ric_return_intermediate'])
        if 'ric_update_block_type' in cfg:
            self.ric_update_block_type = str(cfg['ric_update_block_type'])
        if 'ric_dilations' in cfg:
            self.ric_dilations = tuple(int(d) for d in cfg['ric_dilations'])
        if 'ric_num_heads' in cfg:
            self.ric_num_heads = int(cfg['ric_num_heads'])
        if 'ric_attention_stride' in cfg:
            self.ric_attention_stride = int(cfg['ric_attention_stride'])
        if 'ric_ffn_multiplier' in cfg:
            self.ric_ffn_multiplier = int(cfg['ric_ffn_multiplier'])

        # Optional ADMM-unfolded communication-prior head parameters
        if 'admm_num_steps' in cfg:
            self.admm_num_steps = int(cfg['admm_num_steps'])
        if 'admm_hidden_channels' in cfg:
            self.admm_hidden_channels = int(cfg['admm_hidden_channels'])
        if 'admm_kernel_size' in cfg:
            self.admm_kernel_size = int(cfg['admm_kernel_size'])
        if 'admm_dropout' in cfg:
            self.admm_dropout = float(cfg['admm_dropout'])
        if 'admm_tied_steps' in cfg:
            self.admm_tied_steps = bool(cfg['admm_tied_steps'])
        if 'admm_rho_init' in cfg:
            self.admm_rho_init = float(cfg['admm_rho_init'])
        if 'admm_dual_step_init' in cfg:
            self.admm_dual_step_init = float(cfg['admm_dual_step_init'])
        if 'admm_prox_step_init' in cfg:
            self.admm_prox_step_init = float(cfg['admm_prox_step_init'])

        # Optional PGD-unfolded communication-prior head parameters
        if 'pgdu_num_steps' in cfg:
            self.pgdu_num_steps = int(cfg['pgdu_num_steps'])
        if 'pgdu_hidden_channels' in cfg:
            self.pgdu_hidden_channels = int(cfg['pgdu_hidden_channels'])
        if 'pgdu_kernel_size' in cfg:
            self.pgdu_kernel_size = int(cfg['pgdu_kernel_size'])
        if 'pgdu_dropout' in cfg:
            self.pgdu_dropout = float(cfg['pgdu_dropout'])
        if 'pgdu_tied_steps' in cfg:
            self.pgdu_tied_steps = bool(cfg['pgdu_tied_steps'])
        if 'pgdu_step_size_init' in cfg:
            self.pgdu_step_size_init = float(cfg['pgdu_step_size_init'])
        if 'pgdu_prox_step_init' in cfg:
            self.pgdu_prox_step_init = float(cfg['pgdu_prox_step_init'])

        # Optional gain/phase channel-consistency parameters
        if 'gp_hidden_channels' in cfg:
            self.gp_hidden_channels = int(cfg['gp_hidden_channels'])
        if 'gp_kernel_size' in cfg:
            self.gp_kernel_size = int(cfg['gp_kernel_size'])
        if 'gp_max_gain_db' in cfg:
            self.gp_max_gain_db = float(cfg['gp_max_gain_db'])
        if 'gp_max_phase_deg' in cfg:
            self.gp_max_phase_deg = float(cfg['gp_max_phase_deg'])
        if 'gp_weight_mode' in cfg:
            self.gp_weight_mode = str(cfg['gp_weight_mode'])
        if 'gp_min_weight' in cfg:
            self.gp_min_weight = float(cfg['gp_min_weight'])
        if 'gp_correction_strength_init' in cfg:
            self.gp_correction_strength_init = float(cfg['gp_correction_strength_init'])
        if 'gp_apply_train' in cfg:
            self.gp_apply_train = bool(cfg['gp_apply_train'])
        if 'gp_apply_eval' in cfg:
            self.gp_apply_eval = bool(cfg['gp_apply_eval'])

        # Optional lightweight train-time RF augmentation defaults
        if 'train_aug_enable' in cfg:
            self.train_aug_enable = bool(cfg['train_aug_enable'])
        if 'train_aug_source_phase_jitter_deg' in cfg:
            self.train_aug_source_phase_jitter_deg = float(cfg['train_aug_source_phase_jitter_deg'])
        if 'train_aug_source_gain_jitter_db' in cfg:
            self.train_aug_source_gain_jitter_db = float(cfg['train_aug_source_gain_jitter_db'])
        if 'train_aug_max_common_time_shift' in cfg:
            self.train_aug_max_common_time_shift = int(cfg['train_aug_max_common_time_shift'])
        if 'train_aug_global_phase_rotation' in cfg:
            self.train_aug_global_phase_rotation = bool(cfg['train_aug_global_phase_rotation'])
        if 'train_mix_enable' in cfg:
            self.train_mix_enable = bool(cfg['train_mix_enable'])
        if 'train_mix_prob' in cfg:
            self.train_mix_prob = float(cfg['train_mix_prob'])
        if 'train_mix_sir_min_db' in cfg:
            self.train_mix_sir_min_db = float(cfg['train_mix_sir_min_db'])
        if 'train_mix_sir_max_db' in cfg:
            self.train_mix_sir_max_db = float(cfg['train_mix_sir_max_db'])
        if 'train_mix_cross_sample' in cfg:
            self.train_mix_cross_sample = bool(cfg['train_mix_cross_sample'])
        if 'train_aug_warmup_ratio' in cfg:
            self.train_aug_warmup_ratio = float(cfg['train_aug_warmup_ratio'])
        if 'train_aug_warmup_epochs' in cfg:
            self.train_aug_warmup_epochs = int(cfg['train_aug_warmup_epochs'])
