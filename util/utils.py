import os
import torch.nn as nn
import torch
from pathlib import Path
from models.tfgridnet_v3_pytorch import TFGridNetV3Separator1D
from models.tiger_separator import TIGERSeparator1D
from models.conformer_gridnet import ConformerGridNetSeparator1D
from models.ctdcrn import CTDCRNSeparator1D, CTDCRNConfig
from torch.nn import LeakyReLU, InstanceNorm1d
from util.config import MambaConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nn_module(module_name: str):
    module_map = {
        "LeakyReLU": LeakyReLU,
        "InstanceNorm1d": InstanceNorm1d,
    }
    return module_map.get(module_name, None)

def Create_Mamba_model(config: MambaConfig, logger, input_size_, device_override=None):
    global input_size, device
    input_size = input_size_
    if device_override is not None:
        device = device_override

    config._load_enc_config()
    if config.model_type == "tfgridnet":
        if logger is not None:
            logger.info("Model Type: TFGridNetV3Separator1D")
        return _create_tfgridnet_model(config)
    if config.model_type == "tiger":
        if logger is not None:
            logger.info("Model Type: TIGERSeparator1D")
        return _create_tiger_model(config)
    if config.model_type == "bimamba":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D")
        return _create_bimamba_model(config)
    if config.model_type == "bimamba_layerscale":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_LayerScale (BiMamba + learnable residual scale)")
        return _create_bimamba_layerscale_model(config)
    if config.model_type == "bimamba_localglobal":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_LocalGlobal (BiMamba + gated local-global fusion)")
        return _create_bimamba_localglobal_model(config)
    if config.model_type == "bimamba_glg":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_GLG (BiMamba + gated local-global fusion + LayerScale)")
        return _create_bimamba_glg_model(config)
    if config.model_type == "bimamba_uric":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_URIC (BiMamba+Unrolled Residual IC)")
        return _create_bimamba_uric_model(config)
    if config.model_type == "bimamba_uric_aug":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_URIC_AUG (URIC + lightweight RF train augmentation)")
        return _create_bimamba_uric_aug_model(config)
    if config.model_type == "bimamba_admm":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_ADMM (BiMamba+ADMM-Unfolded Communication Prior)")
        return _create_bimamba_admm_model(config)
    if config.model_type == "bimamba_pgdu":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_PGDU (BiMamba+PGD-Unfolded Communication Prior)")
        return _create_bimamba_pgdu_model(config)
    if config.model_type == "bimamba_gainphase":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_GainPhase (BiMamba+Gain/Phase Channel Consistency)")
        return _create_bimamba_gainphase_model(config)
    if config.model_type == "bimamba_mcproj":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_MC (BiMamba+MixtureConsistencyProjection)")
        return _create_bimamba_mcproj_model(config)
    if config.model_type == "bimamba_lk":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_LK (Large-Kernel Stem)")
        return _create_bimamba_lk_model(config)
    if config.model_type == "bimamba_csb":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB (Complex Stem + Complex Bottleneck)")
        return _create_bimamba_csb_model(config)
    if config.model_type == "bimamba_csb_scan":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB_Scan (CSB + complex-coupled chunk gated scans)")
        return _create_bimamba_csb_scan_model(config)
    if config.model_type == "bimamba_csb_cag":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB_CAG (CSB + scaled residual alpha + complex-aware channel gate)")
        return _create_bimamba_csb_cag_model(config)
    if config.model_type == "bimamba_csb_phasediff":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB_PhaseDiff (CSB + phase-difference guided scans)")
        return _create_bimamba_csb_phasediff_model(config)
    if config.model_type == "bimamba_csb_cmasc":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB_CMASC (CSB + complex mixture-consistent ASC)")
        return _create_bimamba_csb_cmasc_model(config)
    if config.model_type == "bimamba_csb_constellation":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB_Constellation (CSB + soft constellation-guided refinement)")
        return _create_bimamba_csb_constellation_model(config)
    if config.model_type == "bimamba_fullcomplex":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_FullComplex (Complex feature path + complex-wrapped BiMamba)")
        return _create_bimamba_fullcomplex_model(config)
    if config.model_type == "bimamba_complex_mask":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_ComplexMask (Local complex encoder + real BiMamba + complex mask head)")
        return _create_bimamba_complex_mask_model(config)
    if config.model_type == "complex_unet1d":
        if logger is not None:
            logger.info("Model Type: IQUComplexUNet1D (Pure Complex Convolutional U-Net Baseline)")
        return _create_complex_unet1d_model(config)
    if config.model_type == "real_unet1d":
        if logger is not None:
            logger.info("Model Type: IQURealUNet1D (Real-valued mirror of the complex U-Net baseline)")
        return _create_real_unet1d_model(config)
    if config.model_type == "bimamba_csb_uric":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_CSB_URIC (Complex Stem + Complex Bottleneck + URIC)")
        return _create_bimamba_csb_uric_model(config)
    if config.model_type == "bimamba_jamba":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_Jamba (BiMamba+Attention Hybrid)")
        return _create_bimamba_jamba_model(config)
    if config.model_type == "convnext":
        if logger is not None:
            logger.info("Model Type: IQUConvNeXt1D (Large-Kernel ConvNeXt)")
        return _create_convnext_model(config)
    if config.model_type == "transformer1d":
        if logger is not None:
            logger.info("Model Type: IQUTransformer1D (Pure Transformer U-Net Baseline)")
        return _create_transformer1d_model(config)
    if config.model_type == "complex_transformer1d":
        if logger is not None:
            logger.info("Model Type: IQUComplexTransformer1D (Transformer U-Net + complex-valued attention)")
        return _create_complex_transformer1d_model(config)
    if config.model_type == "resunet1d":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D (Pure Convolutional U-Net Baseline)")
        return _create_resunet1d_model(config)
    if config.model_type == "resunet1d_mamba_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MambaBottleneck (stage42 ResUNet + bottleneck Mamba adapter)")
        return _create_resunet1d_mamba_bottleneck_model(config)
    if config.model_type == "resunet1d_mamba_localglobal":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MambaLocalGlobal (stage42 ResUNet + gated local/global Mamba)")
        return _create_resunet1d_mamba_localglobal_model(config)
    if config.model_type == "resunet1d_mamba_dualgate":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MambaDualGate (stage42 ResUNet + temporal/channel Mamba gate)")
        return _create_resunet1d_mamba_dualgate_model(config)
    if config.model_type == "resunet1d_phaseeq":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_PhaseEquivariant (stage42 ResUNet + phase-equivariant input adapter)")
        return _create_resunet1d_phaseeq_model(config)
    if config.model_type == "resunet1d_corrgate":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_CorrGate (stage42 ResUNet + local complex-correlation skip gates)")
        return _create_resunet1d_corrgate_model(config)
    if config.model_type == "resunet1d_pco":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_PCO (stage42 ResUNet + phase/correlation/orthogonalization)")
        return _create_resunet1d_pco_model(config)
    if config.model_type == "resunet1d_noasc":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_NoASC (ResUNet with plain skip concat)")
        return _create_resunet1d_noasc_model(config)
    if config.model_type == "resunet1d_gated_skip":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_GatedSkip (ResUNet with Decoder-Guided Gated Skip)")
        return _create_resunet1d_gated_skip_model(config)
    if config.model_type == "resunet1d_wl_complex":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_WLComplex (ResUNet with Widely-Linear stem and Complex Mask)")
        return _create_resunet1d_wl_complex_model(config)
    if config.model_type == "resunet1d_tf_branch":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_TFBranch (Time-Frequency Dual-Branch ResUNet)")
        return _create_resunet1d_tf_branch_model(config)
    
    # Advanced LSSG Skip Modes
    if config.model_type == "resunet1d_skip_enhanced_lssg":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced_LSSG (Stage 114)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced_LSSG_Channel (Stage 115)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_ms":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced_LSSG_Channel_MS (Stage 121)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel_ms")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_context":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced_LSSG_Channel_Context (Stage 122)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel_context")
    if config.model_type == "resunet1d_skip_enhanced_lssg_refined":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced_LSSG_Refined (Stage 116)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_refined")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_mamba" or config.model_type == "resunet1d_skip_enhanced_lssg_channel_encoder_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_EncoderBiMamba_LSSG_Channel (Stage 128/127)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_original_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_OriginalBiMambaLayer_LSSG_Channel (Stage 133)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_original_full_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_OriginalFullBiMamba_LSSG_Channel (Stage 134)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel")
    if config.model_type == "bimamba_pgd_eq":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_PGD_EQ (Stage 135)")
        return _create_bimamba_pgd_eq_model(config)
    if config.model_type == "bimamba_phys_channel":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_PhysicalChannel (Stage 136)")
        return _create_bimamba_phys_channel_model(config)
    if config.model_type == "bimamba_phys_channel_pgd_eq":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_PhysicalChannel_PGDEQ (Stage 137)")
        return _create_bimamba_phys_channel_pgd_eq_model(config)
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_skip_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipMamba_LSSG_Channel (Stage 129)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel")
    if config.model_type == "resunet1d_skip_enhanced_lssg_channel_decoder_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_DecoderBiMamba_LSSG_Channel (Stage 130)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_channel")

    # Advanced Bottleneck Modes
    if config.model_type == "resunet1d_bottleneck_sra_tcn":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_SRA_TCN (Stage 117)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="sra_tcn")
    if config.model_type == "resunet1d_bottleneck_caspp":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_CASPP (Stage 118)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="caspp")
    if config.model_type == "resunet1d_bottleneck_dccb":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB (Stage 119)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb")
    if config.model_type == "resunet1d_bottleneck_dccb_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_Mamba (Stage 138)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb")
    if config.model_type == "resunet1d_bottleneck_dccb_full_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_Full_Mamba (Stage 139)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb", skip_mode="lssg_channel")
    if config.model_type == "resunet1d_bottleneck_dccb_unidirectional_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_Unidirectional_Mamba (Stage 140)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb", skip_mode="lssg_channel")
    if config.model_type == "ablation_caspp_lssg_shallow":
        if logger is not None:
            logger.info("Model Type: Ablation_CASPP_LSSG_Shallow (Stage 149)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="caspp", skip_mode="lssg")
    if config.model_type == "ablation_caspp_lssg_all":
        if logger is not None:
            logger.info("Model Type: Ablation_CASPP_LSSG_All (Stage 150)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="caspp", skip_mode="lssg")
    if config.model_type == "ablation_sratcn_lssg_shallow":
        if logger is not None:
            logger.info("Model Type: Ablation_SRATCN_LSSG_Shallow (Stage 151)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="sra_tcn", skip_mode="lssg")
    if config.model_type == "ablation_caspp_attn_shallow":
        if logger is not None:
            logger.info("Model Type: Ablation_CASPP_Attn_Shallow (Stage 152)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="caspp", skip_mode="attention")
    if config.model_type == "iqumamba_dwt":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_DWT (Stage 141)")
        return _create_iqumamba_dwt(config)
    if config.model_type == "iq_conformer_gridnet":
        if logger is not None:
            logger.info("Model Type: ConformerGridNetSeparator1D (Stage 143)")
        return _create_conformer_gridnet(config)
    if config.model_type == "iq_bandsplit":
        if logger is not None:
            logger.info("Model Type: BandSplitSeparator (Stage 144)")
        return _create_bandsplit_separator(config)
    if config.model_type == "iqu_mossformer":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MossFormer (Stage 145)")
        return _create_iqu_mossformer(config)
    if config.model_type == "iqu_modern_convnext":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_ConvNeXt (Stage 146)")
        return _create_iqu_resunet_modernized(config, block_mode="convnext")
    if config.model_type == "iqu_modern_mscan":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MSCAN (Stage 147)")
        return _create_iqu_resunet_modernized(config, block_mode="mscan")
    if config.model_type == "iqu_modern_hybrid":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_HybridMamba (Stage 148)")
        return _create_iqu_resunet_modernized(config, block_mode="hybrid")
    if config.model_type == "iqumamba_siren":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_SIREN (Stage 142)")
        return _create_iqumamba_siren(config)
    if config.model_type == "resunet1d_bottleneck_dccb_lssg":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_LSSG (Stage 124)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb", skip_mode="lssg_channel")
    if config.model_type == "resunet1d_bottleneck_dccb_lssg_partial_125":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_LSSG_Partial (Stage 125)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb", skip_mode="lssg_channel", gated_decoder_stages=[2, 3])
    if config.model_type == "resunet1d_bottleneck_dccb_lssg_partial_126":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_LSSG_Partial (Stage 126)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb", skip_mode="lssg_channel", gated_decoder_stages=[3])
    if config.model_type == "resunet1d_bottleneck_dual_path_mamba":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DualPathMamba (Stage 131)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dual_path_mamba", skip_mode="lssg_channel")

    # Prior Adapters
    if config.model_type == "resunet1d_moe_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MoEPrior (Stage 120)")
        return _create_resunet1d_moe_prior_model(config)
    if config.model_type == "resunet1d_qam_lattice_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_QAMPrior (Stage 132)")
        return _create_resunet1d_qam_prior_model(config)
    if config.model_type == "resunet1d_strong_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_StrongPrior (Stage 123)")
        return _create_resunet1d_strong_prior_model(config)
    if config.model_type == "resunet1d_skip_enhanced_attention":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced (Attention U-Net style skip)")
        return _create_resunet1d_skip_enhanced_attention_model(config)
    if config.model_type == "resunet1d_skip_enhanced_uct":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced (UCTransNet-lite style skip)")
        return _create_resunet1d_skip_enhanced_uct_model(config)
    if config.model_type == "resunet1d_skip_enhanced_dca":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SkipEnhanced (DCA-lite style skip)")
        return _create_resunet1d_skip_enhanced_dca_model(config)
    if config.model_type == "resunet1d_universal_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_UniversalPrior (Universal Multi-Source Receiver-Prior Adapter)")
        return _create_resunet1d_universal_prior_model(config)
    if config.model_type == "resunet1d_pulse_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_PulsePrior (Pulse-shaping Prior Adapter)")
        return _create_resunet1d_pulse_prior_model(config)
    if config.model_type == "resunet1d_timing_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_TimingPrior (Multi-hypothesis Timing Adapter)")
        return _create_resunet1d_timing_prior_model(config)
    if config.model_type == "resunet1d_pulse_timing_prior":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_PulseTimingPrior (Pulse + Timing Adapters)")
        return _create_resunet1d_pulse_timing_prior_model(config)
    if config.model_type == "resunet1d_uric":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_URIC (ResUNet + Unrolled Residual IC)")
        return _create_resunet1d_uric_model(config)
    if config.model_type == "bimamba_amr":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_AMR (Joint BSS+AMR)")
        return _create_bimamba_amr_model(config)
    if config.model_type == "bimamba_softdemod":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_SoftDemod (Joint BSS+SoftDemod)")
        return _create_bimamba_softdemod_model(config)
    if config.model_type == "bimamba_softdemod_v2":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_SoftDemodV2 (Receiver-aware Joint BSS+SoftDemod)")
        return _create_bimamba_softdemod_v2_model(config)
    if config.model_type == "bimamba_softdemod_v3":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_SoftDemodV3 (Offset/Phase-aware Joint BSS+SoftDemod)")
        return _create_bimamba_softdemod_v3_model(config)
    if config.model_type == "spmamba":
        if logger is not None:
            logger.info("Model Type: SPMambaSeparator1D")
        return _create_spmamba_model(config)
    if config.model_type == "conformer_gridnet":
        if logger is not None:
            logger.info("Model Type: ConformerGridNetSeparator1D")
        return _create_conformer_gridnet_model(config)
    if config.model_type == "dual_domain_mamba":
        if logger is not None:
            logger.info("Model Type: DualDomainMamba")
        return _create_dual_domain_model(config)
    if config.model_type == "dual_domain_mamba2":
        if logger is not None:
            logger.info("Model Type: DualDomainMamba2 (Mamba-2 SSD)")
        return _create_dual_domain_mamba2_model(config)
    if config.model_type == "dual_domain_mamba_zeroinit":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaZeroInit")
        return _create_dual_domain_zeroinit_model(config)
    if config.model_type == "dual_domain_mamba_dualpath":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaDualPath")
        return _create_dual_domain_dualpath_model(config)
    if config.model_type == "dual_domain_mamba_crossmamba":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaCrossMamba")
        return _create_dual_domain_crossmamba_model(config)
    if config.model_type == "dual_domain_mamba_lite":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaLite")
        return _create_dual_domain_lite_model(config)
    if config.model_type == "dual_domain_mamba_small":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaSmall")
        return _create_dual_domain_small_model(config)
    if config.model_type == "dual_domain_mamba_v2":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaV2")
        return _create_dual_domain_v2_model(config)
    if config.model_type == "dual_domain_mamba_v3":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaV3")
        return _create_dual_domain_v3_model(config)
    if config.model_type == "dual_domain_mamba_v4":
        if logger is not None:
            logger.info("Model Type: DualDomainMambaV4")
        return _create_dual_domain_v4_model(config)
    if config.model_type == "dual_domain_bandsplit":
        if logger is not None:
            logger.info("Model Type: DualDomainBandSplit")
        return _create_dual_domain_bandsplit_model(config)
    if config.model_type == "nes2net":
        if logger is not None:
            logger.info("Model Type: NES2Net")
        return _create_nes2net_model(config)
    if config.model_type == "ctdcrn":
        if logger is not None:
            logger.info("Model Type: CTDCRNSeparator1D")
        return _create_ctdcrn_model(config)
    if config.model_type == "rf_bandscnet":
        if logger is not None:
            logger.info("Model Type: RFBandSCNetSeparator1D (complex STFT band-split spectral masking)")
        return _create_rf_bandscnet_model(config)
    if config.model_type == "complex_dpnet":
        if logger is not None:
            logger.info("Model Type: ComplexDPNetSeparator1D (learned complex encoder + dual-path masking)")
        return _create_complex_dpnet_model(config)
    if config.model_type == "complex_convtasnet":
        if logger is not None:
            logger.info("Model Type: ComplexConvTasNetSeparator1D (learned complex filterbank + dilated TCN masks)")
        return _create_complex_convtasnet_model(config)
    if config.model_type == "complex_sourceslot":
        if logger is not None:
            logger.info("Model Type: ComplexSourceSlotSeparator1D (direct complex source-slot separator)")
        return _create_complex_sourceslot_model(config)
    if config.model_type == "complex_attractor":
        if logger is not None:
            logger.info("Model Type: ComplexAttractorSeparator1D (TF-bin embeddings + source attractors)")
        return _create_complex_attractor_model(config)
    if config.model_type == "multires_stft_mask":
        if logger is not None:
            logger.info("Model Type: MultiResolutionSTFTMaskSeparator1D (multi-resolution complex spectral masks)")
        return _create_multires_stft_mask_model(config)
    if config.model_type == "icassp_baseline_unet":
        if logger is not None:
            logger.info("Model Type: ICASPBaselineUNet")
        return _create_icassp_baseline_unet_model(config)
    if config.model_type == "icassp_baseline_wavenet":
        if logger is not None:
            logger.info("Model Type: ICASPBaselineWaveNet")
        return _create_icassp_baseline_wavenet_model(config)
    if config.model_type == "iqumamba_decodermamba":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_DecoderMamba (stage-4 IQUMamba + decoder Mamba)")
        return _create_iqumamba_decodermamba_model(config)
    if config.model_type == "iqumamba_rfscan_fusion":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_RFScanFusion (stage-4 IQUMamba + temporal/chunk/frequency scan fusion)")
        return _create_iqumamba_rfscan_fusion_model(config)
    if config.model_type == "iqumamba_rfmamba_scan":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_RFMambaScan (stage-4 IQUMamba + RFMamba-inspired frequency scan)")
        return _create_iqumamba_rfmamba_scan_model(config)
    if config.model_type == "iqumamba_radmamba_scan":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_RadMambaScan (stage-4 IQUMamba + RadMamba-inspired chunk scan)")
        return _create_iqumamba_radmamba_scan_model(config)
    if config.model_type == "iqumamba_symbol_dualpath":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_SymbolDualPath (stage-4 IQUMamba + symbol-aligned dual-path Mamba adapter)")
        return _create_iqumamba_symbol_dualpath_model(config)
    if config.model_type == "iqumamba_complex_mask_mc":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_ComplexMaskMC (stage-4 IQUMamba + complex mask + mixture constraint)")
        return _create_iqumamba_complex_mask_mc_model(config)
    if config.model_type == "iqumamba_feature_complex_mask":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_FeatureComplexMask (stage-4 IQUMamba + learned complex feature mask)")
        return _create_iqumamba_feature_complex_mask_model(config)
    if config.model_type == "iqumamba_knowledge_esd":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_KnowledgeESD (stage-4 IQUMamba + source-slot refinement + mixture projection)")
        return _create_iqumamba_knowledge_esd_model(config)
    if config.model_type == "iqumamba_blind_multirate_input":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_BlindMultiRateInput (stage-4 IQUMamba + blind multi-rate input adapter)")
        return _create_iqumamba_blind_multirate_input_model(config)
    if config.model_type == "iqumamba_noise_aware_mc":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_NoiseAwareMC (stage-4 IQUMamba + residual-noise mixture consistency)")
        return _create_iqumamba_noise_aware_mc_model(config)
    if config.model_type == "iqumamba_complex_adapter":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_ComplexAdapter (stage-4 IQUMamba + local complex-aware adapters)")
        return _create_iqumamba_complex_adapter_model(config)
    if config.model_type == "iqumamba_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_CycloFRESH (stage-4 IQUMamba + cyclostationary FRESH input adapter)")
        return _create_iqumamba_cyclofresh_model(config)
    if config.model_type == "iqumamba_blind_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_BlindCycloFRESH (stage-4 IQUMamba + learnable cyclic-frequency FRESH input adapter)")
        return _create_iqumamba_blind_cyclofresh_model(config)
    if config.model_type == "iqumamba_estimated_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_EstimatedCycloFRESH (stage-4 IQUMamba + mixture-estimated cyclic-frequency FRESH input adapter)")
        return _create_iqumamba_estimated_cyclofresh_model(config)
    if config.model_type == "iqumamba_multipeak_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_MultiPeakCycloFRESH (stage-4 IQUMamba + multi-peak mixture-estimated FRESH input adapter)")
        return _create_iqumamba_multipeak_cyclofresh_model(config)
    if config.model_type == "iqumamba_sample_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_SampleCycloFRESH (stage-4 IQUMamba + per-sample mixture-estimated FRESH input adapter)")
        return _create_iqumamba_sample_cyclofresh_model(config)
    if config.model_type == "iqumamba_cyclofresh_freqbias":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_CycloFRESHFreqBias (stage-4 IQUMamba + estimated FRESH + high-frequency residual adapter)")
        return _create_iqumamba_cyclofresh_freqbias_model(config)
    if config.model_type == "iqumamba_blindstat_film":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_BlindStatFiLM (stage-4 IQUMamba + mixture-only blind-stat feature FiLM)")
        return _create_iqumamba_blindstat_film_model(config)
    if config.model_type == "iqumamba_blindstat_input":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_BlindStatInput (stage-4 IQUMamba + mixture-only blind-stat input adapter)")
        return _create_iqumamba_blindstat_input_model(config)
    if config.model_type == "iqumamba_cycliccorr":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_CyclicCorr (stage-4 IQUMamba + mixture-estimated cyclic-correlation adapter)")
        return _create_iqumamba_cycliccorr_model(config)
    if config.model_type == "iqumamba_cycliccorr_leakcancel":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_CyclicCorrLeakCancel (stage-4 IQUMamba + cyclic leakage cancellation)")
        return _create_iqumamba_cycliccorr_leakcancel_model(config)
    if config.model_type == "sepbamba_unet1d":
        if logger is not None:
            logger.info("Model Type: IQUSepBambaUNet1D (4-stage SepMamba U-Net)")
        return _create_sepbamba_unet1d_model(config)
    if logger is not None:
        logger.info("Model Type: IQUMamba1D")
    return _create_enc_model(config)


def _create_iqu_mossformer(config):
    from models.IQUResUNet1D_MossFormer import IQUResUNet1D_MossFormer
    import torch.nn as nn
    model_cfg = config.model_config
    return IQUResUNet1D_MossFormer(
        input_size=4096,
        input_channels=model_cfg.get("input_channels", config.input_channels),
        n_stages=model_cfg.get("n_stages", config.n_stages),
        features_per_stage=model_cfg.get("features_per_stage", config.features_per_stage),
        conv_op=nn.Conv1d,
        kernel_sizes=model_cfg.get("kernel_sizes", [3]*config.n_stages),
        strides=model_cfg.get("strides", [1]+[2]*(config.n_stages-1)),
        n_conv_per_stage=model_cfg.get("n_conv_per_stage", config.n_conv_per_stage),
        num_classes=model_cfg.get("num_classes", config.num_classes),
        n_conv_per_stage_decoder=model_cfg.get("n_conv_per_stage_decoder", config.n_conv_per_stage_decoder),
        conv_bias=model_cfg.get("conv_bias", config.conv_bias),
        n_mossformer_blocks=model_cfg.get("n_mossformer_blocks", 2),
    ).to(device)


def _create_iqu_resunet_modernized(config, block_mode="convnext"):
    from models.IQUResUNet1D_Modernized import IQUResUNet1D_BottleneckEnhanced_Modernized
    return IQUResUNet1D_BottleneckEnhanced_Modernized(
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        block_mode=block_mode
    ).to(device)


def _create_enc_model(config):
    from models.IQUMamba1D import IQUMamba1D

    return IQUMamba1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_iqumamba_decodermamba_model(config):
    from models.IQUMamba1D_DecoderMamba import IQUMamba1D_DecoderMamba

    return IQUMamba1D_DecoderMamba(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        decoder_mamba_stages=getattr(config, 'decoder_mamba_stages', (0,)),
    ).to(device)


def _rfscan_kwargs(config):
    return {
        'rfscan_chunk_size': int(getattr(config, 'rfscan_chunk_size', 256)),
        'rfscan_shift_size': getattr(config, 'rfscan_shift_size', None),
        'rfscan_freq_bands': int(getattr(config, 'rfscan_freq_bands', 16)),
        'rfscan_gate_hidden': int(getattr(config, 'rfscan_gate_hidden', 64)),
        'rfscan_conv_kernel_size': int(getattr(config, 'rfscan_conv_kernel_size', 5)),
        'rfscan_residual_scale_init': float(getattr(config, 'rfscan_residual_scale_init', 0.1)),
        'rfscan_condition_scale_init': float(getattr(config, 'rfscan_condition_scale_init', 0.1)),
        'rfscan_stft_n_fft': int(getattr(config, 'rfscan_stft_n_fft', 256)),
        'rfscan_stft_hop_length': int(getattr(config, 'rfscan_stft_hop_length', 64)),
        'rfscan_stft_win_length': getattr(config, 'rfscan_stft_win_length', None),
        'rfscan_stft_freq_bins': int(getattr(config, 'rfscan_stft_freq_bins', 32)),
    }


def _create_iqumamba_rfscan_model(config, model_class):
    return model_class(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_rfscan_kwargs(config),
    ).to(device)


def _create_iqumamba_rfscan_fusion_model(config):
    from models.IQUMamba1D_RFScan import IQUMamba1D_RFScanFusion

    return _create_iqumamba_rfscan_model(config, IQUMamba1D_RFScanFusion)


def _create_iqumamba_rfmamba_scan_model(config):
    from models.IQUMamba1D_RFScan import IQUMamba1D_RFMambaScan

    return _create_iqumamba_rfscan_model(config, IQUMamba1D_RFMambaScan)


def _create_iqumamba_radmamba_scan_model(config):
    from models.IQUMamba1D_RFScan import IQUMamba1D_RadMambaScan

    return _create_iqumamba_rfscan_model(config, IQUMamba1D_RadMambaScan)


def _create_iqumamba_symbol_dualpath_model(config):
    from models.IQUMamba1D_SymbolDualPath import IQUMamba1D_SymbolDualPath

    return IQUMamba1D_SymbolDualPath(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        symbol_samples=int(getattr(config, 'symbol_samples', 20)),
        dual_path_chunk_symbols=int(getattr(config, 'dual_path_chunk_symbols', 4)),
        dual_path_hop_symbols=int(getattr(config, 'dual_path_hop_symbols', 2)),
        dual_path_residual_scale_init=float(getattr(config, 'dual_path_residual_scale_init', 0.01)),
    ).to(device)


def _create_iqumamba_complex_mask_mc_model(config):
    from models.IQUMamba1D_ComplexMaskMC import IQUMamba1D_ComplexMaskMC

    return IQUMamba1D_ComplexMaskMC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        mask_bound=float(getattr(config, 'mask_bound', 4.0)),
        mask_sum_constraint=bool(getattr(config, 'mask_sum_constraint', True)),
        mask_apply_projection=bool(getattr(config, 'mask_apply_projection', True)),
        mask_project_deep_supervision=bool(getattr(config, 'mask_project_deep_supervision', True)),
        mask_logit_scale_init=float(getattr(config, 'mask_logit_scale_init', 0.1)),
        mc_weight_mode=str(getattr(config, 'mc_weight_mode', 'uniform')),
        mc_weight_power=float(getattr(config, 'mc_weight_power', 1.0)),
        mc_min_weight=float(getattr(config, 'mc_min_weight', 0.0)),
        mc_eps=float(getattr(config, 'mc_eps', 1e-8)),
        mc_detach_weights=bool(getattr(config, 'mc_detach_weights', False)),
        mc_apply_train=bool(getattr(config, 'mc_apply_train', True)),
        mc_apply_eval=bool(getattr(config, 'mc_apply_eval', True)),
    ).to(device)


def _create_iqumamba_feature_complex_mask_model(config):
    from models.IQUMamba1D_FeatureComplexMask import IQUMamba1D_FeatureComplexMask

    return IQUMamba1D_FeatureComplexMask(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        feature_mask_channels=int(getattr(config, 'feature_mask_channels', 8)),
        feature_mask_kernel_size=int(getattr(config, 'feature_mask_kernel_size', 9)),
        feature_mask_bound=float(getattr(config, 'feature_mask_bound', 4.0)),
        feature_mask_sum_constraint=bool(getattr(config, 'feature_mask_sum_constraint', True)),
        feature_mask_apply_projection=bool(getattr(config, 'feature_mask_apply_projection', True)),
        feature_mask_project_deep_supervision=bool(getattr(config, 'feature_mask_project_deep_supervision', True)),
        feature_mask_logit_scale_init=float(getattr(config, 'feature_mask_logit_scale_init', 0.05)),
        feature_mask_identity_init=bool(getattr(config, 'feature_mask_identity_init', True)),
        mc_weight_mode=str(getattr(config, 'mc_weight_mode', 'uniform')),
        mc_weight_power=float(getattr(config, 'mc_weight_power', 1.0)),
        mc_min_weight=float(getattr(config, 'mc_min_weight', 0.0)),
        mc_eps=float(getattr(config, 'mc_eps', 1e-8)),
        mc_detach_weights=bool(getattr(config, 'mc_detach_weights', False)),
        mc_apply_train=bool(getattr(config, 'mc_apply_train', True)),
        mc_apply_eval=bool(getattr(config, 'mc_apply_eval', True)),
    ).to(device)


def _create_iqumamba_knowledge_esd_model(config):
    from models.IQUMamba1D_KnowledgeESD import IQUMamba1D_KnowledgeESD

    return IQUMamba1D_KnowledgeESD(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        source_slot_hidden_channels=int(getattr(config, 'source_slot_hidden_channels', 32)),
        source_slot_kernel_size=int(getattr(config, 'source_slot_kernel_size', 7)),
        source_slot_residual_scale_init=float(getattr(config, 'source_slot_residual_scale_init', 0.01)),
        source_slot_zero_init=bool(getattr(config, 'source_slot_zero_init', True)),
        source_slot_refine_deep_supervision=bool(getattr(config, 'source_slot_refine_deep_supervision', True)),
        source_slot_apply_train=bool(getattr(config, 'source_slot_apply_train', True)),
        source_slot_apply_eval=bool(getattr(config, 'source_slot_apply_eval', True)),
        mc_weight_mode=str(getattr(config, 'mc_weight_mode', 'uniform')),
        mc_weight_power=float(getattr(config, 'mc_weight_power', 1.0)),
        mc_min_weight=float(getattr(config, 'mc_min_weight', 0.0)),
        mc_eps=float(getattr(config, 'mc_eps', 1e-8)),
        mc_detach_weights=bool(getattr(config, 'mc_detach_weights', False)),
        mc_project_deep_supervision=bool(getattr(config, 'mc_project_deep_supervision', True)),
        mc_apply_train=bool(getattr(config, 'mc_apply_train', True)),
        mc_apply_eval=bool(getattr(config, 'mc_apply_eval', True)),
    ).to(device)


def _create_iqumamba_blind_multirate_input_model(config):
    from models.IQUMamba1D_BlindMultiRateInput import IQUMamba1D_BlindMultiRateInput

    return IQUMamba1D_BlindMultiRateInput(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        multirate_hidden_channels=int(getattr(config, 'multirate_hidden_channels', 8)),
        multirate_kernel_sizes=tuple(getattr(config, 'multirate_kernel_sizes', (5, 9, 17, 33))),
        multirate_dilations=tuple(getattr(config, 'multirate_dilations', (1, 2, 4, 8))),
        multirate_scale_init=float(getattr(config, 'multirate_scale_init', 0.01)),
        multirate_zero_init=bool(getattr(config, 'multirate_zero_init', True)),
    ).to(device)


def _create_iqumamba_noise_aware_mc_model(config):
    from models.IQUMamba1D_NoiseAwareMC import IQUMamba1D_NoiseAwareMC

    return IQUMamba1D_NoiseAwareMC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        noise_mc_apply_projection=bool(getattr(config, 'noise_mc_apply_projection', True)),
        noise_mc_project_during_train=bool(getattr(config, 'noise_mc_project_during_train', True)),
        noise_mc_project_during_eval=bool(getattr(config, 'noise_mc_project_during_eval', True)),
        noise_mc_source_weight=float(getattr(config, 'noise_mc_source_weight', 0.25)),
        noise_mc_noise_weight=float(getattr(config, 'noise_mc_noise_weight', 1.0)),
        noise_head_hidden_channels=int(getattr(config, 'noise_head_hidden_channels', 32)),
        noise_head_kernel_size=int(getattr(config, 'noise_head_kernel_size', 7)),
        noise_head_zero_init=bool(getattr(config, 'noise_head_zero_init', True)),
        noise_mc_eps=float(getattr(config, 'noise_mc_eps', 1e-8)),
    ).to(device)


def _create_iqumamba_complex_adapter_model(config):
    from models.IQUMamba1D_ComplexAdapter import IQUMamba1D_ComplexAdapter

    return IQUMamba1D_ComplexAdapter(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_adapter_hidden_channels=int(getattr(config, 'complex_adapter_hidden_channels', 8)),
        complex_adapter_kernel_size=int(getattr(config, 'complex_adapter_kernel_size', 5)),
        complex_adapter_scale_init=float(getattr(config, 'complex_adapter_scale_init', 0.01)),
        complex_adapter_use_input=bool(getattr(config, 'complex_adapter_use_input', True)),
        complex_adapter_use_output=bool(getattr(config, 'complex_adapter_use_output', True)),
        complex_adapter_zero_init=bool(getattr(config, 'complex_adapter_zero_init', True)),
    ).to(device)


def _create_iqumamba_cyclofresh_model(config):
    from models.IQUMamba1D_CycloFRESH import IQUMamba1D_CycloFRESH

    return IQUMamba1D_CycloFRESH(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        cyclofresh_sps=int(getattr(config, 'cyclofresh_sps', 20)),
        cyclofresh_alphas=tuple(getattr(config, 'cyclofresh_alphas', (0.0, 1.0, -1.0, 2.0, -2.0))),
        cyclofresh_hidden_channels=int(getattr(config, 'cyclofresh_hidden_channels', 8)),
        cyclofresh_kernel_size=int(getattr(config, 'cyclofresh_kernel_size', 9)),
        cyclofresh_scale_init=float(getattr(config, 'cyclofresh_scale_init', 0.01)),
        cyclofresh_gate_hidden=int(getattr(config, 'cyclofresh_gate_hidden', 8)),
        cyclofresh_zero_init=bool(getattr(config, 'cyclofresh_zero_init', True)),
    ).to(device)


def _create_iqumamba_blind_cyclofresh_model(config):
    from models.IQUMamba1D_BlindCycloFRESH import IQUMamba1D_BlindCycloFRESH

    return IQUMamba1D_BlindCycloFRESH(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        blind_cyclofresh_freqs=tuple(getattr(
            config,
            'blind_cyclofresh_freqs',
            (-0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24),
        )),
        blind_cyclofresh_max_delta=float(getattr(config, 'blind_cyclofresh_max_delta', 0.03)),
        blind_cyclofresh_hidden_channels=int(getattr(config, 'blind_cyclofresh_hidden_channels', 8)),
        blind_cyclofresh_kernel_size=int(getattr(config, 'blind_cyclofresh_kernel_size', 9)),
        blind_cyclofresh_scale_init=float(getattr(config, 'blind_cyclofresh_scale_init', 0.01)),
        blind_cyclofresh_gate_hidden=int(getattr(config, 'blind_cyclofresh_gate_hidden', 8)),
        blind_cyclofresh_zero_init=bool(getattr(config, 'blind_cyclofresh_zero_init', True)),
    ).to(device)


def _create_iqumamba_estimated_cyclofresh_model(config):
    from models.IQUMamba1D_EstimatedCycloFRESH import IQUMamba1D_EstimatedCycloFRESH

    return IQUMamba1D_EstimatedCycloFRESH(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        estimated_cyclofresh_min_freq=float(getattr(config, 'estimated_cyclofresh_min_freq', 1.0 / 64.0)),
        estimated_cyclofresh_max_freq=float(getattr(config, 'estimated_cyclofresh_max_freq', 1.0 / 8.0)),
        estimated_cyclofresh_default_freq=float(getattr(config, 'estimated_cyclofresh_default_freq', 1.0 / 32.0)),
        estimated_cyclofresh_momentum=float(getattr(config, 'estimated_cyclofresh_momentum', 0.05)),
        estimated_cyclofresh_hidden_channels=int(getattr(config, 'estimated_cyclofresh_hidden_channels', 8)),
        estimated_cyclofresh_kernel_size=int(getattr(config, 'estimated_cyclofresh_kernel_size', 9)),
        estimated_cyclofresh_scale_init=float(getattr(config, 'estimated_cyclofresh_scale_init', 0.01)),
        estimated_cyclofresh_gate_hidden=int(getattr(config, 'estimated_cyclofresh_gate_hidden', 8)),
        estimated_cyclofresh_zero_init=bool(getattr(config, 'estimated_cyclofresh_zero_init', True)),
    ).to(device)


def _create_iqumamba_multipeak_cyclofresh_model(config):
    from models.IQUMamba1D_CycloFRESHPlus import IQUMamba1D_MultiPeakCycloFRESH

    return IQUMamba1D_MultiPeakCycloFRESH(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        multipeak_cyclofresh_min_freq=float(getattr(config, 'multipeak_cyclofresh_min_freq', 1.0 / 64.0)),
        multipeak_cyclofresh_max_freq=float(getattr(config, 'multipeak_cyclofresh_max_freq', 1.0 / 8.0)),
        multipeak_cyclofresh_default_freq=float(getattr(config, 'multipeak_cyclofresh_default_freq', 1.0 / 32.0)),
        multipeak_cyclofresh_momentum=float(getattr(config, 'multipeak_cyclofresh_momentum', 0.05)),
        multipeak_cyclofresh_num_peaks=int(getattr(config, 'multipeak_cyclofresh_num_peaks', 2)),
        multipeak_cyclofresh_guard_bins=int(getattr(config, 'multipeak_cyclofresh_guard_bins', 3)),
        multipeak_cyclofresh_hidden_channels=int(getattr(config, 'multipeak_cyclofresh_hidden_channels', 8)),
        multipeak_cyclofresh_kernel_size=int(getattr(config, 'multipeak_cyclofresh_kernel_size', 9)),
        multipeak_cyclofresh_scale_init=float(getattr(config, 'multipeak_cyclofresh_scale_init', 0.01)),
        multipeak_cyclofresh_gate_hidden=int(getattr(config, 'multipeak_cyclofresh_gate_hidden', 8)),
        multipeak_cyclofresh_reliability_floor=float(getattr(config, 'multipeak_cyclofresh_reliability_floor', 0.25)),
        multipeak_cyclofresh_zero_init=bool(getattr(config, 'multipeak_cyclofresh_zero_init', True)),
    ).to(device)


def _create_iqumamba_sample_cyclofresh_model(config):
    from models.IQUMamba1D_CycloFRESHPlus import IQUMamba1D_SampleCycloFRESH

    return IQUMamba1D_SampleCycloFRESH(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        sample_cyclofresh_min_freq=float(getattr(config, 'sample_cyclofresh_min_freq', 1.0 / 64.0)),
        sample_cyclofresh_max_freq=float(getattr(config, 'sample_cyclofresh_max_freq', 1.0 / 8.0)),
        sample_cyclofresh_default_freq=float(getattr(config, 'sample_cyclofresh_default_freq', 1.0 / 32.0)),
        sample_cyclofresh_num_peaks=int(getattr(config, 'sample_cyclofresh_num_peaks', 1)),
        sample_cyclofresh_guard_bins=int(getattr(config, 'sample_cyclofresh_guard_bins', 3)),
        sample_cyclofresh_hidden_channels=int(getattr(config, 'sample_cyclofresh_hidden_channels', 8)),
        sample_cyclofresh_kernel_size=int(getattr(config, 'sample_cyclofresh_kernel_size', 9)),
        sample_cyclofresh_scale_init=float(getattr(config, 'sample_cyclofresh_scale_init', 0.01)),
        sample_cyclofresh_gate_hidden=int(getattr(config, 'sample_cyclofresh_gate_hidden', 8)),
        sample_cyclofresh_reliability_floor=float(getattr(config, 'sample_cyclofresh_reliability_floor', 0.25)),
        sample_cyclofresh_zero_init=bool(getattr(config, 'sample_cyclofresh_zero_init', True)),
    ).to(device)


def _create_iqumamba_cyclofresh_freqbias_model(config):
    from models.IQUMamba1D_CycloFRESHPlus import IQUMamba1D_CycloFRESHFreqBias

    return IQUMamba1D_CycloFRESHFreqBias(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        estimated_cyclofresh_min_freq=float(getattr(config, 'estimated_cyclofresh_min_freq', 1.0 / 64.0)),
        estimated_cyclofresh_max_freq=float(getattr(config, 'estimated_cyclofresh_max_freq', 1.0 / 8.0)),
        estimated_cyclofresh_default_freq=float(getattr(config, 'estimated_cyclofresh_default_freq', 1.0 / 32.0)),
        estimated_cyclofresh_momentum=float(getattr(config, 'estimated_cyclofresh_momentum', 0.05)),
        estimated_cyclofresh_hidden_channels=int(getattr(config, 'estimated_cyclofresh_hidden_channels', 8)),
        estimated_cyclofresh_kernel_size=int(getattr(config, 'estimated_cyclofresh_kernel_size', 9)),
        estimated_cyclofresh_scale_init=float(getattr(config, 'estimated_cyclofresh_scale_init', 0.01)),
        estimated_cyclofresh_gate_hidden=int(getattr(config, 'estimated_cyclofresh_gate_hidden', 8)),
        estimated_cyclofresh_zero_init=bool(getattr(config, 'estimated_cyclofresh_zero_init', True)),
        freqbias_hidden_channels=int(getattr(config, 'freqbias_hidden_channels', 8)),
        freqbias_kernel_size=int(getattr(config, 'freqbias_kernel_size', 9)),
        freqbias_lowpass_kernel_size=int(getattr(config, 'freqbias_lowpass_kernel_size', 17)),
        freqbias_scale_init=float(getattr(config, 'freqbias_scale_init', 0.01)),
        freqbias_gate_hidden=int(getattr(config, 'freqbias_gate_hidden', 8)),
        freqbias_zero_init=bool(getattr(config, 'freqbias_zero_init', True)),
    ).to(device)


def _blindstat_common_kwargs(config):
    return {
        'blindstat_hidden': int(getattr(config, 'blindstat_hidden', 32)),
        'blindstat_scale_init': float(getattr(config, 'blindstat_scale_init', 0.01)),
        'blindstat_cyclic_min_freq': float(getattr(config, 'blindstat_cyclic_min_freq', 1.0 / 64.0)),
        'blindstat_cyclic_max_freq': float(getattr(config, 'blindstat_cyclic_max_freq', 1.0 / 8.0)),
        'blindstat_cyclic_default_freq': float(getattr(config, 'blindstat_cyclic_default_freq', 1.0 / 32.0)),
        'blindstat_zero_init': bool(getattr(config, 'blindstat_zero_init', True)),
    }


def _create_iqumamba_blindstat_film_model(config):
    from models.IQUMamba1D_BlindStatAdapters import IQUMamba1D_BlindStatFiLM

    return IQUMamba1D_BlindStatFiLM(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_blindstat_common_kwargs(config),
    ).to(device)


def _create_iqumamba_blindstat_input_model(config):
    from models.IQUMamba1D_BlindStatAdapters import IQUMamba1D_BlindStatInput

    kwargs = _blindstat_common_kwargs(config)
    kwargs['blindstat_hidden'] = int(getattr(config, 'blindstat_hidden', 16))
    kwargs['blindstat_kernel_size'] = int(getattr(config, 'blindstat_kernel_size', 9))
    return IQUMamba1D_BlindStatInput(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **kwargs,
    ).to(device)


def _create_iqumamba_cycliccorr_model(config):
    from models.IQUMamba1D_CyclicCorr import IQUMamba1D_CyclicCorr

    return IQUMamba1D_CyclicCorr(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        cycliccorr_min_freq=float(getattr(config, 'cycliccorr_min_freq', 1.0 / 64.0)),
        cycliccorr_max_freq=float(getattr(config, 'cycliccorr_max_freq', 1.0 / 8.0)),
        cycliccorr_default_freq=float(getattr(config, 'cycliccorr_default_freq', 1.0 / 32.0)),
        cycliccorr_momentum=float(getattr(config, 'cycliccorr_momentum', 0.05)),
        cycliccorr_lags=tuple(getattr(config, 'cycliccorr_lags', (0, 1, 2, 4, 8))),
        cycliccorr_hidden_channels=int(getattr(config, 'cycliccorr_hidden_channels', 8)),
        cycliccorr_kernel_size=int(getattr(config, 'cycliccorr_kernel_size', 9)),
        cycliccorr_scale_init=float(getattr(config, 'cycliccorr_scale_init', 0.01)),
        cycliccorr_gate_hidden=int(getattr(config, 'cycliccorr_gate_hidden', 16)),
        cycliccorr_zero_init=bool(getattr(config, 'cycliccorr_zero_init', True)),
    ).to(device)


def _cycliccorr_kwargs(config):
    return {
        'cycliccorr_min_freq': float(getattr(config, 'cycliccorr_min_freq', 1.0 / 64.0)),
        'cycliccorr_max_freq': float(getattr(config, 'cycliccorr_max_freq', 1.0 / 8.0)),
        'cycliccorr_default_freq': float(getattr(config, 'cycliccorr_default_freq', 1.0 / 32.0)),
        'cycliccorr_momentum': float(getattr(config, 'cycliccorr_momentum', 0.05)),
        'cycliccorr_lags': tuple(getattr(config, 'cycliccorr_lags', (0, 1, 2, 4, 8))),
        'cycliccorr_hidden_channels': int(getattr(config, 'cycliccorr_hidden_channels', 8)),
        'cycliccorr_kernel_size': int(getattr(config, 'cycliccorr_kernel_size', 9)),
        'cycliccorr_scale_init': float(getattr(config, 'cycliccorr_scale_init', 0.01)),
        'cycliccorr_gate_hidden': int(getattr(config, 'cycliccorr_gate_hidden', 16)),
        'cycliccorr_zero_init': bool(getattr(config, 'cycliccorr_zero_init', True)),
    }


def _create_iqumamba_cycliccorr_leakcancel_model(config):
    from models.IQUMamba1D_CyclicCorrLeakCancel import IQUMamba1D_CyclicCorrLeakCancel

    return IQUMamba1D_CyclicCorrLeakCancel(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        cycliccorr_min_freq=float(getattr(config, 'cycliccorr_min_freq', 1.0 / 64.0)),
        cycliccorr_max_freq=float(getattr(config, 'cycliccorr_max_freq', 1.0 / 8.0)),
        cycliccorr_default_freq=float(getattr(config, 'cycliccorr_default_freq', 1.0 / 32.0)),
        leakcancel_lags=tuple(getattr(config, 'leakcancel_lags', (0, 1, 2, 4, 8))),
        leakcancel_hidden=int(getattr(config, 'leakcancel_hidden', 16)),
        leakcancel_scale_init=float(getattr(config, 'leakcancel_scale_init', 0.2)),
        leakcancel_mc_scale_init=float(getattr(config, 'leakcancel_mc_scale_init', 0.05)),
        leakcancel_mc_weight_mode=str(getattr(config, 'leakcancel_mc_weight_mode', 'uniform')),
        leakcancel_mode=str(getattr(config, 'leakcancel_mode', 'covariance')),
        leakcancel_coeff_limit=float(getattr(config, 'leakcancel_coeff_limit', 0.25)),
        leakcancel_zero_init=bool(getattr(config, 'leakcancel_zero_init', True)),
    ).to(device)


def _create_tfgridnet_model(config):
    if config.input_channels != 2:
        raise ValueError(f"TFGridNetV3Separator1D expects input_channels=2, got {config.input_channels}")
        raise ValueError(f"TIGERSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    tc = config.tiger_config if isinstance(config.tiger_config, dict) else {}
    n_srcs = int(tc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != n_srcs * 2:
        raise ValueError(
            f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs})."
        )

    return TIGERSeparator1D(
        n_srcs=n_srcs,
        n_fft=int(tc.get("n_fft", 256)),
        hop_length=int(tc.get("hop_length", 64)),
        win_length=int(tc.get("win_length", 256)),
        center=bool(tc.get("center", True)),
        normalize_input=bool(tc.get("normalize_input", True)),
        eps=float(tc.get("eps", 1e-8)),
        out_channels=int(tc.get("out_channels", 128)),
        in_channels=int(tc.get("in_channels", 512)),
        num_blocks=int(tc.get("num_blocks", 16)),
        upsampling_depth=int(tc.get("upsampling_depth", 4)),
        att_n_head=int(tc.get("att_n_head", 4)),
        att_hid_chan=int(tc.get("att_hid_chan", 4)),
        nband=int(tc.get("nband", 8)),
    ).to(device)


def _create_bimamba_pgd_eq_model(config):
    """Factory for IQUBiMamba1D_PGD_EQ."""
    from models.IQU_DeepUnfoldedEq import IQUBiMamba1D_PGD_EQ

    return IQUBiMamba1D_PGD_EQ(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_bimamba_phys_channel_model(config):
    """Factory for IQUBiMamba1D_PhysicalChannel."""
    from models.IQU_PhysicalChannelHead import IQUBiMamba1D_PhysicalChannel

    return IQUBiMamba1D_PhysicalChannel(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_bimamba_phys_channel_pgd_eq_model(config):
    """Factory for IQUBiMamba1D_PhysicalChannel_PGDEQ."""
    from models.IQU_PhysicalChannelHead import IQUBiMamba1D_PhysicalChannel_PGDEQ

    return IQUBiMamba1D_PhysicalChannel_PGDEQ(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)

def _create_bimamba_model(config):
    from models.IQUBiMamba1D import IQUBiMamba1D

    return IQUBiMamba1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_gated_bimamba_model(config, model_class):
    return model_class(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        mamba_residual_scale_init=float(getattr(config, 'mamba_residual_scale_init', 0.1)),
        local_kernel_size=int(getattr(config, 'local_kernel_size', 7)),
        local_global_gate_hidden=int(getattr(config, 'local_global_gate_hidden', 64)),
    ).to(device)


def _create_bimamba_layerscale_model(config):
    from models.IQUBiMamba1D_GatedVariants import IQUBiMamba1D_LayerScale

    return _create_gated_bimamba_model(config, IQUBiMamba1D_LayerScale)


def _create_bimamba_localglobal_model(config):
    from models.IQUBiMamba1D_GatedVariants import IQUBiMamba1D_LocalGlobal

    return _create_gated_bimamba_model(config, IQUBiMamba1D_LocalGlobal)


def _create_bimamba_glg_model(config):
    from models.IQUBiMamba1D_GatedVariants import IQUBiMamba1D_GLG

    return _create_gated_bimamba_model(config, IQUBiMamba1D_GLG)


def _uric_kwargs(config):
    return {
        'ric_num_steps': int(getattr(config, 'ric_num_steps', 3)),
        'ric_hidden_channels': int(getattr(config, 'ric_hidden_channels', 48)),
        'ric_kernel_size': int(getattr(config, 'ric_kernel_size', 7)),
        'ric_dropout': float(getattr(config, 'ric_dropout', 0.0)),
        'ric_tied_steps': bool(getattr(config, 'ric_tied_steps', True)),
        'ric_step_init': float(getattr(config, 'ric_step_init', 0.5)),
        'ric_return_intermediate': bool(getattr(config, 'ric_return_intermediate', False)),
        'ric_update_block_type': str(getattr(config, 'ric_update_block_type', 'conv')),
        'ric_dilations': tuple(int(d) for d in getattr(config, 'ric_dilations', (1, 2, 4))),
        'ric_num_heads': int(getattr(config, 'ric_num_heads', 4)),
        'ric_attention_stride': int(getattr(config, 'ric_attention_stride', 1)),
        'ric_ffn_multiplier': int(getattr(config, 'ric_ffn_multiplier', 2)),
    }


def _create_bimamba_uric_model(config):
    from models.IQUBiMamba1D_URIC import IQUBiMamba1D_URIC

    return IQUBiMamba1D_URIC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_uric_kwargs(config),
    ).to(device)


def _create_bimamba_uric_aug_model(config):
    from models.IQUBiMamba1D_URIC_AUG import IQUBiMamba1D_URIC_AUG

    return IQUBiMamba1D_URIC_AUG(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_uric_kwargs(config),
    ).to(device)


def _create_bimamba_admm_model(config):
    from models.IQUBiMamba1D import IQUBiMamba1D_ADMM

    return IQUBiMamba1D_ADMM(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        admm_num_steps=int(getattr(config, 'admm_num_steps', 3)),
        admm_hidden_channels=int(getattr(config, 'admm_hidden_channels', 48)),
        admm_kernel_size=int(getattr(config, 'admm_kernel_size', 7)),
        admm_dropout=float(getattr(config, 'admm_dropout', 0.0)),
        admm_tied_steps=bool(getattr(config, 'admm_tied_steps', True)),
        admm_rho_init=float(getattr(config, 'admm_rho_init', 1.0)),
        admm_dual_step_init=float(getattr(config, 'admm_dual_step_init', 1.0)),
        admm_prox_step_init=float(getattr(config, 'admm_prox_step_init', 0.25)),
    ).to(device)


def _create_bimamba_pgdu_model(config):
    from models.IQUBiMamba1D import IQUBiMamba1D_PGDU

    return IQUBiMamba1D_PGDU(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        pgdu_num_steps=int(getattr(config, 'pgdu_num_steps', 3)),
        pgdu_hidden_channels=int(getattr(config, 'pgdu_hidden_channels', 48)),
        pgdu_kernel_size=int(getattr(config, 'pgdu_kernel_size', 7)),
        pgdu_dropout=float(getattr(config, 'pgdu_dropout', 0.0)),
        pgdu_tied_steps=bool(getattr(config, 'pgdu_tied_steps', True)),
        pgdu_step_size_init=float(getattr(config, 'pgdu_step_size_init', 0.5)),
        pgdu_prox_step_init=float(getattr(config, 'pgdu_prox_step_init', 0.25)),
    ).to(device)


def _create_bimamba_gainphase_model(config):
    from models.IQUBiMamba1D import IQUBiMamba1D_GainPhase

    return IQUBiMamba1D_GainPhase(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        gp_hidden_channels=int(getattr(config, 'gp_hidden_channels', 32)),
        gp_kernel_size=int(getattr(config, 'gp_kernel_size', 7)),
        gp_max_gain_db=float(getattr(config, 'gp_max_gain_db', 12.0)),
        gp_max_phase_deg=float(getattr(config, 'gp_max_phase_deg', 180.0)),
        gp_weight_mode=str(getattr(config, 'gp_weight_mode', 'energy')),
        gp_min_weight=float(getattr(config, 'gp_min_weight', 1e-3)),
        gp_correction_strength_init=float(getattr(config, 'gp_correction_strength_init', 1.0)),
        gp_apply_train=bool(getattr(config, 'gp_apply_train', True)),
        gp_apply_eval=bool(getattr(config, 'gp_apply_eval', True)),
    ).to(device)


def _create_bimamba_mcproj_model(config):
    from models.IQUBiMamba1D_MC import IQUBiMamba1D_MC

    return IQUBiMamba1D_MC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        mc_weight_mode=str(getattr(config, 'mc_weight_mode', 'energy')),
        mc_weight_power=float(getattr(config, 'mc_weight_power', 1.0)),
        mc_min_weight=float(getattr(config, 'mc_min_weight', 1e-3)),
        mc_eps=float(getattr(config, 'mc_eps', 1e-8)),
        mc_detach_weights=bool(getattr(config, 'mc_detach_weights', False)),
        mc_project_deep_supervision=bool(getattr(config, 'mc_project_deep_supervision', True)),
        mc_apply_train=bool(getattr(config, 'mc_apply_train', True)),
        mc_apply_eval=bool(getattr(config, 'mc_apply_eval', True)),
    ).to(device)


def _create_bimamba_lk_model(config):
    """Factory for IQUBiMamba1D_LK — MIT-inspired large-kernel stem."""
    from models.IQUBiMamba1D_LK import IQUBiMamba1D_LK

    return IQUBiMamba1D_LK(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        # MIT-inspired large-kernel stem
        stem_channels=int(getattr(config, 'stem_channels', 128)),
        stem_kernel_size=int(getattr(config, 'stem_kernel_size', 33)),
    ).to(device)


def _create_bimamba_csb_model(config):
    """Factory for IQUBiMamba1D_CSB - complex stem + complex bottleneck bridge."""
    from models.IQUBiMamba1D_CSB import IQUBiMamba1D_CSB

    return IQUBiMamba1D_CSB(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
    ).to(device)


def _create_bimamba_csb_scan_model(config):
    """Factory for IQUBiMamba1D_CSB_Scan - CSB plus gated communication-aware scans."""
    from models.IQUBiMamba1D_CSB_Scan import IQUBiMamba1D_CSB_Scan

    return IQUBiMamba1D_CSB_Scan(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
        cs_scan_chunk_size=int(getattr(config, 'cs_scan_chunk_size', 256)),
        cs_scan_shift_size=getattr(config, 'cs_scan_shift_size', None),
        cs_scan_gate_hidden=int(getattr(config, 'cs_scan_gate_hidden', 64)),
    ).to(device)


def _create_bimamba_csb_cag_model(config):
    """Factory for IQUBiMamba1D_CSB_CAG - CSB plus scaled gated BiMamba residuals."""
    from models.IQUBiMamba1D_CSB_CAG import IQUBiMamba1D_CSB_CAG

    return IQUBiMamba1D_CSB_CAG(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
        cag_alpha_init=float(getattr(config, 'cag_alpha_init', 0.1)),
        cag_gate_hidden=int(getattr(config, 'cag_gate_hidden', 64)),
    ).to(device)


def _create_bimamba_csb_phasediff_model(config):
    """Factory for IQUBiMamba1D_CSB_PhaseDiff - CSB plus phase-difference guided scans."""
    from models.IQUBiMamba1D_CSB_PhaseDiff import IQUBiMamba1D_CSB_PhaseDiff

    return IQUBiMamba1D_CSB_PhaseDiff(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
        phasediff_eps=float(getattr(config, 'phasediff_eps', 1e-6)),
    ).to(device)


def _create_bimamba_csb_cmasc_model(config):
    """Factory for IQUBiMamba1D_CSB_CMASC - CSB plus complex mixture-consistent ASC."""
    from models.IQUBiMamba1D_CSB_CMASC import IQUBiMamba1D_CSB_CMASC

    return IQUBiMamba1D_CSB_CMASC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
        cmasc_gate_hidden=int(getattr(config, 'cmasc_gate_hidden', 64)),
        cmasc_residual_scale_init=float(getattr(config, 'cmasc_residual_scale_init', 0.5)),
        cmasc_eps=float(getattr(config, 'cmasc_eps', 1e-6)),
    ).to(device)


def _create_bimamba_csb_constellation_model(config):
    """Factory for IQUBiMamba1D_CSB_Constellation - CSB plus soft constellation prior."""
    from models.IQUBiMamba1D_CSB_Constellation import IQUBiMamba1D_CSB_Constellation

    return IQUBiMamba1D_CSB_Constellation(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
        constellation_type=str(getattr(config, 'constellation_type', 'psk')),
        constellation_order=int(getattr(config, 'constellation_order', 8)),
        cgr_hidden_channels=int(getattr(config, 'cgr_hidden_channels', 48)),
        cgr_kernel_size=int(getattr(config, 'cgr_kernel_size', 7)),
        cgr_temperature=float(getattr(config, 'cgr_temperature', 0.25)),
        cgr_dropout=float(getattr(config, 'cgr_dropout', 0.0)),
        cgr_gate_init=float(getattr(config, 'cgr_gate_init', 0.1)),
        cgr_residual_scale_init=float(getattr(config, 'cgr_residual_scale_init', 1.0)),
        cgr_use_mixture_residual=bool(getattr(config, 'cgr_use_mixture_residual', True)),
        cgr_zero_init=bool(getattr(config, 'cgr_zero_init', True)),
        cgr_refine_deep_supervision=bool(getattr(config, 'cgr_refine_deep_supervision', False)),
        cgr_apply_train=bool(getattr(config, 'cgr_apply_train', True)),
        cgr_apply_eval=bool(getattr(config, 'cgr_apply_eval', True)),
    ).to(device)


def _create_bimamba_fullcomplex_model(config):
    """Factory for IQUBiMamba1D_FullComplex - complex path + complex-wrapped BiMamba."""
    from models.IQUBiMamba1D_FullComplex import IQUBiMamba1D_FullComplex

    return IQUBiMamba1D_FullComplex(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
    ).to(device)


def _create_bimamba_complex_mask_model(config):
    """Factory for local complex encoder + real BiMamba + complex mask head."""
    from models.IQUBiMamba1D_ComplexMask import IQUBiMamba1D_ComplexMask

    complex_encoder_channels = getattr(config, 'complex_encoder_channels', None)
    if complex_encoder_channels is None:
        num_complex_stages = int(getattr(config, 'complex_encoder_num_stages', 1))
        complex_encoder_channels = [int(config.features_per_stage[0])] * num_complex_stages

    return IQUBiMamba1D_ComplexMask(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_encoder_channels=[int(c) for c in complex_encoder_channels],
        complex_encoder_kernel_size=int(getattr(config, 'complex_encoder_kernel_size', 5)),
        complex_to_real_channels=getattr(config, 'complex_to_real_channels', None),
        complex_mask_latent_channels=int(getattr(config, 'complex_mask_latent_channels', 64)),
        complex_reconstruction_kernel_size=int(getattr(config, 'complex_reconstruction_kernel_size', 3)),
        complex_eps=float(getattr(config, 'complex_eps', 1e-8)),
        complex_leaky_relu_slope=float(getattr(config, 'complex_leaky_relu_slope', 0.01)),
    ).to(device)


def _create_complex_unet1d_model(config):
    """Factory for IQUComplexUNet1D - pure complex-convolutional U-Net baseline."""
    from models.IQUComplexUNet1D import IQUComplexUNet1D

    return IQUComplexUNet1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
    ).to(device)


def _create_real_unet1d_model(config):
    """Factory for IQURealUNet1D - strict real-valued mirror of the complex U-Net baseline."""
    from models.IQURealUNet1D import IQURealUNet1D

    return IQURealUNet1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        stem_kernel_size=int(getattr(config, 'stem_kernel_size', 5)),
    ).to(device)


def _create_sepbamba_unet1d_model(config):
    """Factory for IQUSepBambaUNet1D - 4-stage SepMamba U-Net."""
    from models.IQUSepBambaUNet1D import IQUSepBambaUNet1D

    model_cfg = config.model_config
    features_per_stage = model_cfg.get("features_per_stage", [32, 64, 128, 256])
    d_state = int(model_cfg.get("d_state", 16))
    d_conv = int(model_cfg.get("d_conv", 4))
    expand = int(model_cfg.get("expand", 2))
    fusion = model_cfg.get("fusion", "proj")
    num_layers = int(model_cfg.get("num_layers", 1))
    residual_scale_init = float(model_cfg.get("residual_scale_init", 0.1))
    use_bamba = model_cfg.get("use_bamba", [False, True, True, True])
    norm_type = model_cfg.get("norm_type", "instance")
    use_complex_mask = model_cfg.get("use_complex_mask", False)

    return IQUSepBambaUNet1D(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        features_per_stage=features_per_stage,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        fusion=fusion,
        num_layers=num_layers,
        residual_scale_init=residual_scale_init,
        use_bamba=use_bamba,
        norm_type=norm_type,
        use_complex_mask=use_complex_mask,
    ).to(device)


def _create_bimamba_csb_uric_model(config):
    """Factory for IQUBiMamba1D_CSB_URIC - CSB backbone + URIC refinement."""
    from models.IQUBiMamba1D_CSB_URIC import IQUBiMamba1D_CSB_URIC

    return IQUBiMamba1D_CSB_URIC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        complex_stem_hidden_channels=int(getattr(config, 'complex_stem_hidden_channels', 32)),
        complex_stem_kernel_size=int(getattr(config, 'complex_stem_kernel_size', 5)),
        complex_bottleneck_hidden_channels=int(getattr(config, 'complex_bottleneck_hidden_channels', 128)),
        complex_bottleneck_num_blocks=int(getattr(config, 'complex_bottleneck_num_blocks', 3)),
        complex_bottleneck_kernel_size=int(getattr(config, 'complex_bottleneck_kernel_size', 5)),
        complex_bottleneck_dilation_growth=int(getattr(config, 'complex_bottleneck_dilation_growth', 2)),
        complex_bottleneck_zero_init=bool(getattr(config, 'complex_bottleneck_zero_init', True)),
        **_uric_kwargs(config),
    ).to(device)


def _create_bimamba_jamba_model(config):
    """Factory for IQUBiMamba1D_Jamba — Jamba-style BiMamba+Attention hybrid."""
    from models.IQUBiMamba1D_Jamba import IQUBiMamba1D_Jamba

    return IQUBiMamba1D_Jamba(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        # Jamba-specific
        attn_stages=getattr(config, 'attn_stages', None),
        attn_n_heads=int(getattr(config, 'attn_n_heads', 4)),
        attn_dropout=float(getattr(config, 'attn_dropout', 0.0)),
        attn_ffn_expand=int(getattr(config, 'attn_ffn_expand', 4)),
    ).to(device)


def _create_convnext_model(config):
    """Factory for IQUConvNeXt1D — ConvNeXt-style large-kernel CNN."""
    from models.IQUConvNeXt1D import IQUConvNeXt1D

    return IQUConvNeXt1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        # ConvNeXt-specific
        lk_kernel_size=int(getattr(config, 'lk_kernel_size', 31)),
        lk_expand=int(getattr(config, 'lk_expand', 4)),
    ).to(device)


def _create_transformer1d_model(config):
    """Factory for IQUTransformer1D — pure Transformer U-Net baseline."""
    from models.IQUTransformer1D import IQUTransformer1D

    return IQUTransformer1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        transformer_n_heads=int(getattr(config, 'transformer_n_heads', 4)),
        transformer_dropout=float(getattr(config, 'transformer_dropout', 0.0)),
        transformer_ffn_expand=int(getattr(config, 'transformer_ffn_expand', 4)),
        transformer_token_layout=str(getattr(config, 'transformer_token_layout', 'adaptive')),
        transformer_pos_encoding=str(getattr(config, 'transformer_pos_encoding', 'sinusoidal')),
    ).to(device)


def _create_complex_transformer1d_model(config):
    """Factory for IQUComplexTransformer1D - Transformer U-Net with complex attention."""
    from models.IQUComplexTransformer1D import IQUComplexTransformer1D

    return IQUComplexTransformer1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        transformer_n_heads=int(getattr(config, 'transformer_n_heads', 4)),
        transformer_dropout=float(getattr(config, 'transformer_dropout', 0.0)),
        transformer_ffn_expand=int(getattr(config, 'transformer_ffn_expand', 4)),
        transformer_token_layout=str(getattr(config, 'transformer_token_layout', 'patch')),
        transformer_pos_encoding=str(getattr(config, 'transformer_pos_encoding', 'sinusoidal')),
        complex_attention_score=str(getattr(config, 'complex_attention_score', 'magnitude')),
    ).to(device)


def _create_resunet1d_model(config):
    """Factory for IQUResUNet1D — pure convolutional 1D U-Net baseline."""
    from models.IQUResUNet1D import IQUResUNet1D

    return IQUResUNet1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)

def _resunet_mamba_embed_kwargs(config):
    return {
        "mamba_embed_stages": getattr(config, "mamba_embed_stages", None),
        "mamba_embed_d_state": int(getattr(config, "mamba_embed_d_state", 16)),
        "mamba_embed_d_conv": int(getattr(config, "mamba_embed_d_conv", 4)),
        "mamba_embed_expand": int(getattr(config, "mamba_embed_expand", 2)),
        "mamba_embed_scale_init": float(getattr(config, "mamba_embed_scale_init", 0.0)),
        "mamba_embed_local_kernel_size": int(getattr(config, "mamba_embed_local_kernel_size", 7)),
        "mamba_embed_gate_hidden": int(getattr(config, "mamba_embed_gate_hidden", 64)),
    }


def _create_resunet1d_mamba_embed_model(config, model_class):
    return model_class(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_resunet_mamba_embed_kwargs(config),
    ).to(device)


def _create_resunet1d_mamba_bottleneck_model(config):
    from models.IQUResUNet1D_MambaEmbed import IQUResUNet1D_MambaBottleneck

    return _create_resunet1d_mamba_embed_model(config, IQUResUNet1D_MambaBottleneck)


def _create_resunet1d_mamba_localglobal_model(config):
    from models.IQUResUNet1D_MambaEmbed import IQUResUNet1D_MambaLocalGlobal

    return _create_resunet1d_mamba_embed_model(config, IQUResUNet1D_MambaLocalGlobal)


def _create_resunet1d_mamba_dualgate_model(config):
    from models.IQUResUNet1D_MambaEmbed import IQUResUNet1D_MambaDualGate

    return _create_resunet1d_mamba_embed_model(config, IQUResUNet1D_MambaDualGate)


def _resunet_pco_kwargs(config):
    return {
        "pco_phase_channels": int(getattr(config, "pco_phase_channels", 16)),
        "pco_phase_kernel_size": int(getattr(config, "pco_phase_kernel_size", 7)),
        "pco_phase_scale_init": float(getattr(config, "pco_phase_scale_init", 0.01)),
        "pco_corr_lags": getattr(config, "pco_corr_lags", [1, 2, 4, 8]),
        "pco_corr_window": int(getattr(config, "pco_corr_window", 33)),
        "pco_corr_scale_init": float(getattr(config, "pco_corr_scale_init", 0.01)),
        "pco_orth_scale_init": float(getattr(config, "pco_orth_scale_init", 0.01)),
        "pco_orth_eps": float(getattr(config, "pco_orth_eps", 1e-5)),
    }


def _create_resunet1d_pco_variant_model(config, model_class):
    return model_class(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_resunet_pco_kwargs(config),
    ).to(device)


def _create_resunet1d_phaseeq_model(config):
    from models.IQUResUNet1D_PCO import IQUResUNet1D_PhaseEquivariant

    return _create_resunet1d_pco_variant_model(config, IQUResUNet1D_PhaseEquivariant)


def _create_resunet1d_corrgate_model(config):
    from models.IQUResUNet1D_PCO import IQUResUNet1D_CorrGate

    return _create_resunet1d_pco_variant_model(config, IQUResUNet1D_CorrGate)


def _create_resunet1d_pco_model(config):
    from models.IQUResUNet1D_PCO import IQUResUNet1D_PCO

    return _create_resunet1d_pco_variant_model(config, IQUResUNet1D_PCO)


def _create_resunet1d_noasc_model(config):
    """Factory for IQUResUNet1D_NoASC - ResUNet with direct skip concatenation."""
    from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC
    
    use_complex_mask = config.model_config.get("use_complex_mask", False)

    return IQUResUNet1D_NoASC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        use_complex_mask=use_complex_mask,
    ).to(device)


def _create_resunet1d_gated_skip_model(config):
    """Factory for IQUResUNet1D_GatedSkip - ResUNet with Decoder-Guided Gated Skip."""
    from models.IQUResUNet1D_GatedSkip import IQUResUNet1D_GatedSkip

    model_cfg = config.model_config
    residual_scale_init = float(model_cfg.get("residual_scale_init", 0.1))
    gate_kernel_size = int(model_cfg.get("gate_kernel_size", 3))
    use_complex_mask = model_cfg.get("use_complex_mask", False)

    return IQUResUNet1D_GatedSkip(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        residual_scale_init=residual_scale_init,
        gate_kernel_size=gate_kernel_size,
        use_complex_mask=use_complex_mask,
    ).to(device)


def _create_resunet1d_skip_enhanced_model(config, skip_mode="attention"):
    """Factory for IQUResUNet1D_SkipEnhanced."""
    from models.IQUResUNet1D_SkipEnhanced import IQUResUNet1D_SkipEnhanced
    from models.IQU_BottleneckEnhanced import IQUResUNet1D_BottleneckEnhanced

    model_cfg = config.model_config
    residual_scale_init = float(model_cfg.get("residual_scale_init", 0.1))
    attn_dim = int(model_cfg.get("attn_dim", 64))
    num_heads = int(model_cfg.get("num_heads", 4))
    max_tokens = int(model_cfg.get("max_tokens", 256))
    use_complex_mask = model_cfg.get("use_complex_mask", False)
    use_mamba_stages = model_cfg.get("use_mamba_stages", None)
    mamba_residual_scale_init = float(model_cfg.get("mamba_residual_scale_init", 0.0))
    use_skip_mamba = model_cfg.get("use_skip_mamba", False)
    use_decoder_mamba = model_cfg.get("use_decoder_mamba", False)

    is_original = getattr(config, 'model_type', '') in [
        "resunet1d_skip_enhanced_lssg_channel_original_mamba",
        "resunet1d_skip_enhanced_lssg_channel_original_full_mamba"
    ]
    encoder_mamba_block_type = "original" if is_original else "safe"
    decoder_mamba_block_type = "original" if is_original else "safe"

    return IQUResUNet1D_SkipEnhanced(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        skip_mode=skip_mode,
        residual_scale_init=residual_scale_init,
        attn_dim=attn_dim,
        num_heads=num_heads,
        max_tokens=max_tokens,
        use_complex_mask=use_complex_mask,
        use_mamba_stages=use_mamba_stages,
        mamba_residual_scale_init=mamba_residual_scale_init,
        encoder_mamba_block_type=encoder_mamba_block_type,
        decoder_mamba_block_type=decoder_mamba_block_type,
        use_skip_mamba=use_skip_mamba,
        use_decoder_mamba=use_decoder_mamba,
    ).to(device)


def _create_resunet1d_skip_enhanced_attention_model(config):
    return _create_resunet1d_skip_enhanced_model(config, skip_mode="attention")

def _create_resunet1d_skip_enhanced_uct_model(config):
    return _create_resunet1d_skip_enhanced_model(config, skip_mode="uct")

def _create_resunet1d_skip_enhanced_dca_model(config):
    return _create_resunet1d_skip_enhanced_model(config, skip_mode="dca")

def _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="sra_tcn", skip_mode=None, gated_decoder_stages=None):
    from models.IQU_BottleneckEnhanced import IQUResUNet1D_BottleneckEnhanced
    if skip_mode is None:
        skip_mode = config.model_config.get("skip_mode", None)
        
    model_cfg = config.model_config
    use_mamba_stages = model_cfg.get("use_mamba_stages", None)
    
    is_original_full = getattr(config, 'model_type', '') == "resunet1d_bottleneck_dccb_full_mamba"
    is_uni = getattr(config, 'model_type', '') == "resunet1d_bottleneck_dccb_unidirectional_mamba"
    
    if is_original_full:
        encoder_mamba_block_type = "original"
        decoder_mamba_block_type = "original"
    elif is_uni:
        encoder_mamba_block_type = "unidirectional"
        decoder_mamba_block_type = "unidirectional"
    else:
        encoder_mamba_block_type = model_cfg.get("encoder_mamba_block_type", "safe")
        decoder_mamba_block_type = model_cfg.get("decoder_mamba_block_type", "safe")
        
    if gated_decoder_stages is None:
        gated_decoder_stages = model_cfg.get("gated_decoder_stages", None)
        
    use_decoder_mamba = model_cfg.get("use_decoder_mamba", False)
    mamba_residual_scale_init = model_cfg.get("mamba_residual_scale_init", 0.0)
    residual_scale_init = model_cfg.get("residual_scale_init", 0.1)
    
    return IQUResUNet1D_BottleneckEnhanced(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm1d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=config.deep_supervision,
        use_complex_mask=model_cfg.get("use_complex_mask", False),
        bottleneck_mode=bottleneck_mode,
        skip_mode=skip_mode,
        gated_decoder_stages=gated_decoder_stages,
        use_mamba_stages=use_mamba_stages,
        encoder_mamba_block_type=encoder_mamba_block_type,
        use_decoder_mamba=use_decoder_mamba,
        decoder_mamba_block_type=decoder_mamba_block_type,
        mamba_residual_scale_init=mamba_residual_scale_init,
        residual_scale_init=residual_scale_init,
    ).to(device)

def _create_resunet1d_moe_prior_model(config):
    from models.IQU_MoEPriorAdapter import IQUResUNet1D_MoEPrior
    model_cfg = config.model_config
    return IQUResUNet1D_MoEPrior(
        input_size=config.input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        identity_bias=float(model_cfg.get("identity_bias", 1.5)),
        max_scale=float(model_cfg.get("max_scale", 0.2)),
        scale_init=float(model_cfg.get("scale_init", -1.5)),
    )

def _create_resunet1d_strong_prior_model(config):
    from models.IQU_StrongPriorAdapter import IQUResUNet1D_StrongPrior
    model_cfg = config.model_config
    return IQUResUNet1D_StrongPrior(
        input_size=config.input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        adapter_channels=int(model_cfg.get("adapter_channels", 32)),
        adapter_max_scale=float(model_cfg.get("adapter_max_scale", 0.5)),
        adapter_scale_init=float(model_cfg.get("adapter_scale_init", 0.0)),
        adapter_temperature=float(model_cfg.get("adapter_temperature", 1.0)),
    )


def _create_resunet1d_universal_prior_model(config):
    """Factory for IQUResUNet1D_UniversalPrior."""
    from models.IQU_UniversalPriorAdapter import IQUResUNet1D_UniversalPrior

    model_cfg = config.model_config
    use_complex_mask = model_cfg.get("use_complex_mask", False)

    min_freq = float(model_cfg.get("universal_prior_min_freq", 1.0 / 64.0))
    max_freq = float(model_cfg.get("universal_prior_max_freq", 1.0 / 8.0))
    top_k = int(model_cfg.get("universal_prior_top_k", 3))
    rolloffs = model_cfg.get("universal_prior_rolloffs", [0.2, 0.35, 0.5])
    rrc_kernel_size = int(model_cfg.get("universal_prior_rrc_kernel_size", 31))
    fresh_kernel_size = int(model_cfg.get("universal_prior_fresh_kernel_size", 9))
    hidden_channels = int(model_cfg.get("universal_prior_hidden_channels", 16))
    gate_hidden = int(model_cfg.get("universal_prior_gate_hidden", 16))
    scale_init = float(model_cfg.get("universal_prior_scale_init", 0.01))

    return IQUResUNet1D_UniversalPrior(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        use_complex_mask=use_complex_mask,
        min_freq=min_freq,
        max_freq=max_freq,
        top_k=top_k,
        rolloffs=rolloffs,
        rrc_kernel_size=rrc_kernel_size,
        fresh_kernel_size=fresh_kernel_size,
        hidden_channels=hidden_channels,
        gate_hidden=gate_hidden,
        scale_init=scale_init,
    ).to(device)


def _create_resunet1d_pulse_prior_model(config):
    """Factory for IQUResUNet1D_PulsePrior."""
    from models.IQU_PulsePriorAdapter import IQUResUNet1D_PulsePrior
    model_cfg = config.model_config
    return IQUResUNet1D_PulsePrior(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        use_complex_mask=model_cfg.get("use_complex_mask", False),
        rolloffs=model_cfg.get("pulse_prior_rolloffs", [0.2, 0.35, 0.5]),
        rrc_kernel_size=int(model_cfg.get("pulse_prior_rrc_kernel_size", 31)),
        gate_hidden=int(model_cfg.get("pulse_prior_gate_hidden", 16)),
        scale_init=float(model_cfg.get("pulse_prior_scale_init", 0.01)),
    ).to(device)


def _create_resunet1d_timing_prior_model(config):
    """Factory for IQUResUNet1D_TimingPrior."""
    from models.IQU_TimingPriorAdapter import IQUResUNet1D_TimingPrior
    model_cfg = config.model_config
    return IQUResUNet1D_TimingPrior(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        use_complex_mask=model_cfg.get("use_complex_mask", False),
        num_hypotheses=int(model_cfg.get("timing_prior_num_hypotheses", 4)),
        gate_hidden=int(model_cfg.get("timing_prior_gate_hidden", 16)),
        scale_init=float(model_cfg.get("timing_prior_scale_init", 0.01)),
    ).to(device)


def _create_resunet1d_pulse_timing_prior_model(config):
    """Factory for IQUResUNet1D_PulseTimingPrior."""
    from models.IQU_PulseTimingPriorAdapter import IQUResUNet1D_PulseTimingPrior
    model_cfg = config.model_config
    return IQUResUNet1D_PulseTimingPrior(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        use_complex_mask=model_cfg.get("use_complex_mask", False),
        rolloffs=model_cfg.get("pulse_prior_rolloffs", [0.2, 0.35, 0.5]),
        rrc_kernel_size=int(model_cfg.get("pulse_prior_rrc_kernel_size", 31)),
        num_hypotheses=int(model_cfg.get("timing_prior_num_hypotheses", 4)),
        gate_hidden=int(model_cfg.get("prior_gate_hidden", 16)),
        scale_init=float(model_cfg.get("prior_scale_init", 0.01)),
    ).to(device)


def _create_resunet1d_qam_prior_model(config):
    from models.IQU_QAMRDEPriorAdapter import IQUResUNet1D_QAMPrior
    import torch
    import torch.nn as nn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = config.model_config

    return IQUResUNet1D_QAMPrior(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        qam_axis_level_bank=model_cfg.get("qam_axis_level_bank", (4, 8, 16)),
        qam_max_scale=float(model_cfg.get("qam_max_scale", 0.35)),
        qam_scale_init=float(model_cfg.get("qam_scale_init", -1.0)),
        return_adapter_aux=True,
    ).to(device)


def _create_resunet1d_wl_complex_model(config):
    """Factory for IQUResUNet1D_WLComplex - ResUNet with Widely-Linear stem and Complex Mask."""
    from models.IQUResUNet1D_WLComplex import IQUResUNet1D_WLComplex

    return IQUResUNet1D_WLComplex(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_resunet1d_tf_branch_model(config):
    """Factory for IQUResUNet1D_TFBranch - Time-Frequency Dual-Branch ResUNet."""
    from models.IQUResUNet1D_TFBranch import IQUResUNet1D_TFBranch

    model_cfg = config.model_config
    n_fft = int(model_cfg.get("n_fft", 256))
    hop_length = int(model_cfg.get("hop_length", 64))
    win_length = int(model_cfg.get("win_length", 256))
    freq_features_per_stage = model_cfg.get("freq_features_per_stage", [128, 256, 384, 512])

    return IQUResUNet1D_TFBranch(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        freq_features_per_stage=freq_features_per_stage,
    ).to(device)


def _create_resunet1d_uric_model(config):
    from models.IQUResUNet1D_URIC import IQUResUNet1D_URIC

    return IQUResUNet1D_URIC(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        **_uric_kwargs(config),
    ).to(device)


def _create_bimamba_amr_model(config):
    """Factory for IQUBiMamba1D_AMR — Joint BSS + AMR."""
    from models.IQUBiMamba1D_AMR import IQUBiMamba1D_AMR

    return IQUBiMamba1D_AMR(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        # AMR-specific
        num_mod_classes=int(getattr(config, 'num_mod_classes', 11)),
        cls_hidden=int(getattr(config, 'cls_hidden', 64)),
        cls_mamba_dim=int(getattr(config, 'cls_mamba_dim', 64)),
        cls_dropout=float(getattr(config, 'cls_dropout', 0.3)),
        detach_cls=bool(getattr(config, 'detach_cls', False)),
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_bimamba_softdemod_model(config):
    """Factory for IQUBiMamba1D_SoftDemod — Joint BSS + Soft Demodulation."""
    from models.IQUBiMamba1D_SoftDemod import IQUBiMamba1D_SoftDemod

    return IQUBiMamba1D_SoftDemod(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        # SoftDemod-specific
        num_bits=int(getattr(config, 'demod_num_bits', 615)),
        demod_bits_per_symbol=int(getattr(config, 'demod_bits_per_symbol', 3)),
        demod_hidden=int(getattr(config, 'demod_hidden', 64)),
        demod_rnn_hidden=int(getattr(config, 'demod_rnn_hidden', 64)),
        demod_dropout=float(getattr(config, 'demod_dropout', 0.2)),
        detach_demod=bool(getattr(config, 'detach_demod', False)),
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_bimamba_softdemod_v2_model(config):
    """Factory for IQUBiMamba1D_SoftDemodV2 — receiver-aware Joint BSS + Soft Demodulation."""
    from models.IQUBiMamba1D_SoftDemod import IQUBiMamba1D_SoftDemodV2

    return IQUBiMamba1D_SoftDemodV2(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        num_bits=int(getattr(config, 'demod_num_bits', 615)),
        demod_bits_per_symbol=int(getattr(config, 'demod_bits_per_symbol', 3)),
        demod_hidden=int(getattr(config, 'demod_hidden', 64)),
        demod_rnn_hidden=int(getattr(config, 'demod_rnn_hidden', 96)),
        demod_dropout=float(getattr(config, 'demod_dropout', 0.2)),
        detach_demod=bool(getattr(config, 'detach_demod', False)),
        demod_adapter_hidden=int(getattr(config, 'demod_adapter_hidden', 96)),
        demod_symbol_hidden=int(getattr(config, 'demod_symbol_hidden', 128)),
        demod_context_layers=int(getattr(config, 'demod_context_layers', 2)),
        demod_symbol_logit_scale=float(getattr(config, 'demod_symbol_logit_scale', 12.0)),
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_bimamba_softdemod_v3_model(config):
    """Factory for IQUBiMamba1D_SoftDemodV3 — stronger receiver-structured Joint BSS + Soft Demodulation."""
    from models.IQUBiMamba1D_SoftDemod import IQUBiMamba1D_SoftDemodV3

    return IQUBiMamba1D_SoftDemodV3(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        num_bits=int(getattr(config, 'demod_num_bits', 615)),
        demod_bits_per_symbol=int(getattr(config, 'demod_bits_per_symbol', 3)),
        demod_hidden=int(getattr(config, 'demod_hidden', 64)),
        demod_rnn_hidden=int(getattr(config, 'demod_rnn_hidden', 128)),
        demod_dropout=float(getattr(config, 'demod_dropout', 0.2)),
        detach_demod=bool(getattr(config, 'detach_demod', False)),
        demod_adapter_hidden=int(getattr(config, 'demod_adapter_hidden', 128)),
        demod_symbol_hidden=int(getattr(config, 'demod_symbol_hidden', 160)),
        demod_context_layers=int(getattr(config, 'demod_context_layers', 2)),
        demod_symbol_logit_scale=float(getattr(config, 'demod_symbol_logit_scale', 14.0)),
        demod_timing_offsets=int(getattr(config, 'demod_timing_offsets', 4)),
        demod_attn_heads=int(getattr(config, 'demod_attn_heads', 4)),
        deep_supervision=config.deep_supervision,
    ).to(device)


def _create_spmamba_model(config):
    from models.spmamba_gridnet import SPMambaSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"SPMambaSeparator1D expects input_channels=2, got {config.input_channels}")

    sc = config.spmamba_config if isinstance(config.spmamba_config, dict) else {}
    n_srcs = int(sc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != n_srcs * 2:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return SPMambaSeparator1D(
        n_srcs=n_srcs,
        n_fft=int(sc.get("n_fft", 256)),
        hop_length=int(sc.get("hop_length", 64)),
        win_length=int(sc.get("win_length", 256)),
        center=bool(sc.get("center", True)),
        normalize_input=bool(sc.get("normalize_input", True)),
        eps=float(sc.get("eps", 1e-8)),
        n_layers=int(sc.get("n_layers", 4)),
        hidden_channels=int(sc.get("hidden_channels", 128)),
        attn_n_head=int(sc.get("attn_n_head", 4)),
        attn_qk_output_channel=int(sc.get("attn_qk_output_channel", 4)),
        emb_dim=int(sc.get("emb_dim", 48)),
        emb_ks=int(sc.get("emb_ks", 4)),
        emb_hs=int(sc.get("emb_hs", 1)),
        d_state=int(sc.get("d_state", 16)),
        d_conv=int(sc.get("d_conv", 4)),
        mamba_expand=int(sc.get("mamba_expand", 2)),
    ).to(device)


def _create_conformer_gridnet_model(config):
    from models.conformer_gridnet import ConformerGridNetSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"ConformerGridNetSeparator1D expects input_channels=2, got {config.input_channels}")

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMamba requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    # Parse optional freq-tower settings with sensible defaults
    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMamba(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        # Dual-domain specific
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_mamba2_model(config):
    """Factory for DualDomainMamba2 — uses Mamba-2 (SSD) instead of Mamba-1."""
    from models.dual_domain_mamba2 import DualDomainMamba2

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMamba2 requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMamba2(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        # Dual-domain specific
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
        # Mamba-2 specific
        d_state=int(dd_cfg.get('d_state', 64)),
        headdim=int(dd_cfg.get('headdim', 32)),
    ).to(device)


def _create_dual_domain_zeroinit_model(config):
    from models.dual_domain_mamba_zeroinit import DualDomainMambaZeroInit

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaZeroInit requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaZeroInit(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_dualpath_model(config):
    from models.dual_domain_mamba_dualpath import DualDomainMambaDualPath

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaDualPath requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaDualPath(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_crossmamba_model(config):
    from models.dual_domain_mamba_crossmamba import DualDomainMambaCrossMamba

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaCrossMamba requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaCrossMamba(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_lite_model(config):
    from models.dual_domain_mamba_lite import DualDomainMambaLite

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaLite requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaLite(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_small_model(config):
    from models.dual_domain_mamba_small import DualDomainMambaSmall

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaSmall requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaSmall(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 128)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.1)),
    ).to(device)


def _create_dual_domain_v2_model(config):
    from models.dual_domain_mamba_v2 import DualDomainMambaV2

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaV2 requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaV2(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_v3_model(config):
    from models.dual_domain_mamba_v3 import DualDomainMambaV3

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaV3 requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaV3(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
    ).to(device)


def _create_dual_domain_v4_model(config):
    from models.dual_domain_mamba_v4 import DualDomainMambaV4

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainMambaV4 requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainMambaV4(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.0)),
        # V4-specific
        bottleneck_dim=int(dd_cfg.get('bottleneck_dim', 256)),
        fusion_start_stage=int(dd_cfg.get('fusion_start_stage', 2)),
    ).to(device)


def _create_dual_domain_bandsplit_model(config):
    from models.dual_domain_bandsplit import DualDomainBandSplit

    if config.input_channels != 2:
        raise ValueError(
            f"DualDomainBandSplit requires input_channels=2 (I/Q), "
            f"got {config.input_channels}"
        )

    dd_cfg = config.dual_domain_config if isinstance(config.dual_domain_config, dict) else {}

    freq_features = dd_cfg.get('freq_features_per_stage', None)
    if freq_features is not None:
        freq_features = [int(f) for f in freq_features]

    return DualDomainBandSplit(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        n_fft=int(dd_cfg.get('n_fft', 256)),
        hop_length=int(dd_cfg.get('hop_length', 64)),
        win_length=int(dd_cfg.get('win_length', 256)),
        freq_features_per_stage=freq_features,
        cross_attn_heads=int(dd_cfg.get('cross_attn_heads', 4)),
        cross_attn_dropout=float(dd_cfg.get('cross_attn_dropout', 0.1)),
        # Band-Split specific
        n_bands=int(dd_cfg.get('n_bands', 8)),
        hidden_dim=int(dd_cfg.get('hidden_dim', 128)),
        n_band_mamba_layers=int(dd_cfg.get('n_band_mamba_layers', 2)),
        fusion_start_stage=int(dd_cfg.get('fusion_start_stage', 2)),
    ).to(device)


def _create_nes2net_model(config):
    from models.nes2net import NES2Net

    nc = config.nes2net_config if isinstance(config.nes2net_config, dict) else {}

    unet_features = nc.get('unet_features', [32, 64, 128, 256])
    if unet_features is not None:
        unet_features = [int(f) for f in unet_features]

    return NES2Net(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        max_sources=int(nc.get('max_sources', 5)),
        nem_base_channels=int(nc.get('nem_base_channels', 64)),
        nem_num_blocks=int(nc.get('nem_num_blocks', 5)),
        unet_features=unet_features,
        unet_kernel_size=int(nc.get('unet_kernel_size', 3)),
        mode=str(nc.get('mode', 'separation')),
    ).to(device)


def _create_ctdcrn_model(config: MambaConfig):
    if config.input_channels != 2:
        raise ValueError(f"CTDCRNSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    cc = config.ctdcrn_config if isinstance(config.ctdcrn_config, dict) else {}
    n_srcs = int(cc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != n_srcs * 2:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    model_cfg = CTDCRNConfig(
        n_srcs=n_srcs,
        J=int(cc.get("J", 2)),
        M=int(cc.get("M", 128)),
        N=int(cc.get("N", 32)),
        U=int(cc.get("U", 128)),
        S=int(cc.get("S", 3)),
        V=int(cc.get("V", 8)),
        L=int(cc.get("L", 1)),
        H=int(cc.get("H", 32)),
        eps=float(cc.get("eps", 1e-8)),
        leaky_relu_slope=float(cc.get("leaky_relu_slope", 0.01)),
    )
    return CTDCRNSeparator1D(model_cfg).to(device)


def _create_rf_bandscnet_model(config: MambaConfig):
    from models.rf_bandscnet import RFBandSCNetSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"RFBandSCNetSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    rc = config.rf_bandscnet_config if isinstance(config.rf_bandscnet_config, dict) else {}
    n_srcs = int(rc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != 2 * n_srcs:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return RFBandSCNetSeparator1D(
        n_srcs=n_srcs,
        n_fft=int(rc.get("n_fft", 256)),
        hop_length=int(rc.get("hop_length", 64)),
        win_length=int(rc.get("win_length", 256)),
        center=bool(rc.get("center", True)),
        normalize_input=bool(rc.get("normalize_input", True)),
        eps=float(rc.get("eps", 1e-8)),
        n_bands=int(rc.get("n_bands", 16)),
        hidden_dim=int(rc.get("hidden_dim", 96)),
        rnn_hidden=int(rc.get("rnn_hidden", 96)),
        n_layers=int(rc.get("n_layers", 6)),
        dropout=float(rc.get("dropout", 0.0)),
        mask_bound=float(rc.get("mask_bound", 4.0)),
        mask_sum_constraint=bool(rc.get("mask_sum_constraint", True)),
        mask_head_zero_init=bool(rc.get("mask_head_zero_init", True)),
        apply_projection=bool(rc.get("apply_projection", True)),
        mc_weight_mode=str(rc.get("mc_weight_mode", "uniform")),
        mc_weight_power=float(rc.get("mc_weight_power", 1.0)),
        mc_min_weight=float(rc.get("mc_min_weight", 0.0)),
        mc_detach_weights=bool(rc.get("mc_detach_weights", False)),
    ).to(device)


def _create_complex_dpnet_model(config: MambaConfig):
    from models.complex_dpnet import ComplexDPNetSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"ComplexDPNetSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    dc = config.complex_dpnet_config if isinstance(config.complex_dpnet_config, dict) else {}
    n_srcs = int(dc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != 2 * n_srcs:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return ComplexDPNetSeparator1D(
        n_srcs=n_srcs,
        feature_channels=int(dc.get("feature_channels", 64)),
        kernel_size=int(dc.get("kernel_size", 9)),
        hidden_dim=int(dc.get("hidden_dim", 128)),
        rnn_hidden=int(dc.get("rnn_hidden", 128)),
        n_layers=int(dc.get("n_layers", 6)),
        chunk_size=int(dc.get("chunk_size", 128)),
        hop_size=int(dc.get("hop_size", 64)),
        dropout=float(dc.get("dropout", 0.0)),
        mask_bound=float(dc.get("mask_bound", 4.0)),
        mask_sum_constraint=bool(dc.get("mask_sum_constraint", True)),
        identity_init=bool(dc.get("identity_init", True)),
        mask_head_zero_init=bool(dc.get("mask_head_zero_init", True)),
        apply_projection=bool(dc.get("apply_projection", True)),
        mc_weight_mode=str(dc.get("mc_weight_mode", "uniform")),
        mc_weight_power=float(dc.get("mc_weight_power", 1.0)),
        mc_min_weight=float(dc.get("mc_min_weight", 0.0)),
        mc_eps=float(dc.get("mc_eps", 1e-8)),
        mc_detach_weights=bool(dc.get("mc_detach_weights", False)),
    ).to(device)


def _create_complex_convtasnet_model(config: MambaConfig):
    from models.complex_convtasnet import ComplexConvTasNetSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"ComplexConvTasNetSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    tc = config.complex_convtasnet_config if isinstance(config.complex_convtasnet_config, dict) else {}
    n_srcs = int(tc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != 2 * n_srcs:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return ComplexConvTasNetSeparator1D(
        n_srcs=n_srcs,
        feature_channels=int(tc.get("feature_channels", 64)),
        kernel_size=int(tc.get("kernel_size", 9)),
        hidden_dim=int(tc.get("hidden_dim", 128)),
        bottleneck_dim=int(tc.get("bottleneck_dim", 128)),
        num_repeats=int(tc.get("num_repeats", 3)),
        blocks_per_repeat=int(tc.get("blocks_per_repeat", 8)),
        tcn_kernel_size=int(tc.get("tcn_kernel_size", 3)),
        dropout=float(tc.get("dropout", 0.0)),
        mask_bound=float(tc.get("mask_bound", 4.0)),
        mask_sum_constraint=bool(tc.get("mask_sum_constraint", True)),
        identity_init=bool(tc.get("identity_init", True)),
        mask_head_zero_init=bool(tc.get("mask_head_zero_init", True)),
        apply_projection=bool(tc.get("apply_projection", True)),
        mc_weight_mode=str(tc.get("mc_weight_mode", "uniform")),
        mc_weight_power=float(tc.get("mc_weight_power", 1.0)),
        mc_min_weight=float(tc.get("mc_min_weight", 0.0)),
        mc_eps=float(tc.get("mc_eps", 1e-8)),
        mc_detach_weights=bool(tc.get("mc_detach_weights", False)),
    ).to(device)


def _create_complex_sourceslot_model(config: MambaConfig):
    from models.complex_sourceslot_net import ComplexSourceSlotSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"ComplexSourceSlotSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    sc = config.complex_sourceslot_config if isinstance(config.complex_sourceslot_config, dict) else {}
    n_srcs = int(sc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != 2 * n_srcs:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return ComplexSourceSlotSeparator1D(
        n_srcs=n_srcs,
        slot_channels=int(sc.get("slot_channels", 64)),
        kernel_size=int(sc.get("kernel_size", 9)),
        hidden_dim=int(sc.get("hidden_dim", 128)),
        n_layers=int(sc.get("n_layers", 8)),
        temporal_kernel_size=int(sc.get("temporal_kernel_size", 5)),
        dilation_cycle=int(sc.get("dilation_cycle", 4)),
        source_attention_heads=int(sc.get("source_attention_heads", 4)),
        dropout=float(sc.get("dropout", 0.0)),
        identity_split_init=bool(sc.get("identity_split_init", True)),
        slot_residual_scale_init=float(sc.get("slot_residual_scale_init", 0.0)),
        apply_projection=bool(sc.get("apply_projection", True)),
        mc_weight_mode=str(sc.get("mc_weight_mode", "uniform")),
        mc_weight_power=float(sc.get("mc_weight_power", 1.0)),
        mc_min_weight=float(sc.get("mc_min_weight", 0.0)),
        mc_eps=float(sc.get("mc_eps", 1e-8)),
        mc_detach_weights=bool(sc.get("mc_detach_weights", False)),
    ).to(device)


def _create_complex_attractor_model(config: MambaConfig):
    from models.complex_attractor_net import ComplexAttractorSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"ComplexAttractorSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    ac = config.complex_attractor_config if isinstance(config.complex_attractor_config, dict) else {}
    n_srcs = int(ac.get("n_srcs", config.num_classes // 2))
    if config.num_classes != 2 * n_srcs:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return ComplexAttractorSeparator1D(
        n_srcs=n_srcs,
        n_fft=int(ac.get("n_fft", 256)),
        hop_length=int(ac.get("hop_length", 64)),
        win_length=int(ac.get("win_length", 256)),
        center=bool(ac.get("center", True)),
        normalize_input=bool(ac.get("normalize_input", True)),
        embedding_dim=int(ac.get("embedding_dim", 64)),
        hidden_dim=int(ac.get("hidden_dim", 96)),
        rnn_hidden=int(ac.get("rnn_hidden", 96)),
        n_layers=int(ac.get("n_layers", 2)),
        dropout=float(ac.get("dropout", 0.0)),
        attractor_temperature=float(ac.get("attractor_temperature", 1.0)),
        logit_scale_init=float(ac.get("logit_scale_init", 0.0)),
        eps=float(ac.get("eps", 1e-8)),
        apply_projection=bool(ac.get("apply_projection", True)),
        mc_weight_mode=str(ac.get("mc_weight_mode", "uniform")),
        mc_weight_power=float(ac.get("mc_weight_power", 1.0)),
        mc_min_weight=float(ac.get("mc_min_weight", 0.0)),
        mc_detach_weights=bool(ac.get("mc_detach_weights", False)),
    ).to(device)


def _create_multires_stft_mask_model(config: MambaConfig):
    from models.multires_stft_masknet import MultiResolutionSTFTMaskSeparator1D

    if config.input_channels != 2:
        raise ValueError(f"MultiResolutionSTFTMaskSeparator1D expects input_channels=2, got {config.input_channels}")
    if config.num_classes % 2 != 0:
        raise ValueError(f"num_classes must be even for I/Q output pairs, got {config.num_classes}")

    mc = config.multires_stft_mask_config if isinstance(config.multires_stft_mask_config, dict) else {}
    n_srcs = int(mc.get("n_srcs", config.num_classes // 2))
    if config.num_classes != 2 * n_srcs:
        raise ValueError(f"num_classes ({config.num_classes}) must equal 2*n_srcs ({2 * n_srcs}).")

    return MultiResolutionSTFTMaskSeparator1D(
        n_srcs=n_srcs,
        n_ffts=[int(v) for v in mc.get("n_ffts", [128, 256, 512])],
        hop_lengths=[int(v) for v in mc.get("hop_lengths", [32, 64, 128])],
        win_lengths=[int(v) for v in mc.get("win_lengths", [128, 256, 512])],
        center=bool(mc.get("center", True)),
        normalize_input=bool(mc.get("normalize_input", True)),
        hidden_dim=int(mc.get("hidden_dim", 128)),
        n_blocks=int(mc.get("n_blocks", 6)),
        kernel_size=int(mc.get("kernel_size", 5)),
        dilation_cycle=int(mc.get("dilation_cycle", 4)),
        dropout=float(mc.get("dropout", 0.0)),
        mask_bound=float(mc.get("mask_bound", 4.0)),
        mask_sum_constraint=bool(mc.get("mask_sum_constraint", True)),
        mask_head_zero_init=bool(mc.get("mask_head_zero_init", True)),
        eps=float(mc.get("eps", 1e-8)),
        apply_projection=bool(mc.get("apply_projection", True)),
        mc_weight_mode=str(mc.get("mc_weight_mode", "uniform")),
        mc_weight_power=float(mc.get("mc_weight_power", 1.0)),
        mc_min_weight=float(mc.get("mc_min_weight", 0.0)),
        mc_detach_weights=bool(mc.get("mc_detach_weights", False)),
    ).to(device)


import os
import glob

def create_new_results_folder(base_dir='results'):
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    next_num = 0
    while True:
        base_pattern = str(results_path / f"{base_dir}_{next_num}*")
        if not glob.glob(base_pattern):
            break
        next_num += 1
    
    return f"{base_dir}_{next_num}"
    return ICASPBaselineWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, 'residual_channels', 64)),
        residual_layers=int(getattr(config, 'residual_layers', 30)),
        dilation_cycle_length=int(getattr(config, 'dilation_cycle_length', 10)),
    ).to(device)
