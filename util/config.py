import os
import yaml
from typing import Dict, Any

class MambaConfig:
    def __init__(self, config_path: str, train: bool = True):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No Config File: {config_path}")
        
        # Load Config
        self.cfg = yaml.safe_load(open(config_path, 'r'))
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

        if self.model_type == 'icassp_baseline_wavenet':
            self.residual_channels = cfg.get('residual_channels', 64)
            self.residual_layers = cfg.get('residual_layers', 30)
            self.dilation_cycle_length = cfg.get('dilation_cycle_length', 10)
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

        self.n_stages = cfg['n_stages']
        self.features_per_stage = cfg['features_per_stage']
        self.kernel_sizes = cfg['kernel_sizes']
        self.strides = cfg['strides']
        self.n_conv_per_stage = cfg['n_conv_per_stage']
        self.n_conv_per_stage_decoder = cfg['n_conv_per_stage_decoder']
        self.deep_supervision = cfg['deep_supervision']
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
