import math
import os
import os
import math
import torch.nn as nn
import torch
from pathlib import Path
from models.tfgridnet_v3_pytorch import TFGridNetV3Separator1D
from models.tiger_separator import TIGERSeparator1D
from models.conformer_gridnet import ConformerGridNetSeparator1D
from models.ctdcrn import CTDCRNSeparator1D, CTDCRNConfig
from models.IQU_ESD_Wrapper import ESDMaskWrapper1D
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
    if config.model_type in {
        "bimamba_hydra",
        "bimamba_complex_state",
        "bimamba_complex_state_independent",
        "bimamba_complex_state_independent_unireplk",
        "bimamba_multiscale",
        "bimamba_complex_latent_mask_real",
        "bimamba_complex_latent_mask_ratio",
        "bimamba_complex_latent_mask_residual",
        "bimamba_complex_latent_mask_conservation",
        "bimamba_bottleneck_mask_real",
    }:
        if logger is not None:
            descriptions = {
                "bimamba_hydra": "Stage 361: Stage12-Hydra (formal quasiseparable bidirectional SSM)",
                "bimamba_complex_state": "Stage 362: Stage12-ComplexState (conjugate complex state + adaptive fusion)",
                "bimamba_multiscale": "Stage 363: Stage12-MultiScaleBiMamba (global scans + local d_conv 3/7/15)",
                "bimamba_complex_state_independent": "Stage 364: Stage12-IndependentComplexState (two independent Stage-295 complex SSMs)",
                "bimamba_complex_state_independent_unireplk": "Stage 365: Stage364-IndependentComplexState + Stage310-UniRepLK",
                "bimamba_complex_latent_mask_real": "Stage 371: Stage365 + real latent mask",
                "bimamba_complex_latent_mask_ratio": "Stage 372: Stage365 + complex ratio latent mask",
                "bimamba_complex_latent_mask_residual": "Stage 373: Stage365 + complex mask + residual slot",
                "bimamba_complex_latent_mask_conservation": "Stage 374: Stage365 + complex conservation mask + residual slot",
                "bimamba_bottleneck_mask_real": "Stage 375: Stage365 + bottleneck-only real mask + batched shared decoder",
            }
            logger.info(f"Model Type: {descriptions[config.model_type]}")
        return _create_bimamba_core_upgrade_model(config)
    if config.model_type == "iqumamba_cross_scale_single":
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1D_CrossScaleAttention "
                "(Stage 300: Stage-4 unidirectional Mamba + compressed global skip attention)"
            )
        return _create_iqumamba_cross_scale_attention_model(config)
    if config.model_type in {
        "bimamba_cross_scale_unireplk",
        "iqumamba_cross_scale_unireplk",
    }:
        if logger is not None:
            direction = (
                "bidirectional"
                if config.model_type == "bimamba_cross_scale_unireplk"
                else "unidirectional"
            )
            logger.info(
                "Model Type: UniRepLK + compressed global Cross-Scale "
                f"({direction} Mamba, four-stage U-Net)"
            )
        return _create_unireplk_cross_scale_model(config)
    if config.model_type in {
        "iqumamba_physical_canonical",
        "iqumamba_symbol_delay_doppler_rf",
    }:
        if logger is not None:
            description = (
                "Stage 306: Stage-4 + physical source canonicalization"
                if config.model_type == "iqumamba_physical_canonical"
                else "Stage 307: Stage-4 + symbol-normalized delay-Doppler receptive field"
            )
            logger.info(f"Model Type: {description}")
            if config.model_type == "iqumamba_physical_canonical":
                logger.info(
                    "Stage 306 assumes target identity follows the configured physical "
                    "ordering; arbitrary exchangeable S1/S2 labels remain unidentifiable."
                )
        return _create_iqumamba_physics_receptive_field_model(config)
    if config.model_type == "iqumamba_recent_rf":
        if logger is not None:
            logger.info(
                "Model Type: Stage-4 + recent receptive-field operator "
                f"({config.model_config.get('rf_module_type')})"
            )
        return _create_iqumamba_recent_rf_model(config)
    if config.model_type in {
        "iqumamba_stage391_unireplk_post_mamba",
        "iqumamba_stage392_unireplk_parallel_delta",
        "iqumamba_stage393_adaptive_rf_post_mamba",
        "iqumamba_stage394_adaptive_rf_parallel_delta",
        "iqumamba_stage395_unireplk_delta_post",
        "iqumamba_stage396_unireplk_full_pre",
        "iqumamba_stage397_stage394_complex_bimamba",
    }:
        if logger is not None:
            descriptions = {
                "iqumamba_stage391_unireplk_post_mamba":
                    "Stage391 (Stage310 connection, UniRepLK at stages 0/2)",
                "iqumamba_stage392_unireplk_parallel_delta":
                    "Stage392 (Stage391 with Stage381 parallel-delta connection)",
                "iqumamba_stage393_adaptive_rf_post_mamba":
                    "Stage393 (Stage391 with Stage389 adaptive RF UniRepLK)",
                "iqumamba_stage394_adaptive_rf_parallel_delta":
                    "Stage394 (Stage392 with Stage389 adaptive RF UniRepLK)",
                "iqumamba_stage395_unireplk_delta_post":
                    "Stage395 (Delta-Post: post-Mamba UniRepLK residual delta)",
                "iqumamba_stage396_unireplk_full_pre":
                    "Stage396 (Full-Pre: pre-Mamba complete UniRepLK output)",
                "iqumamba_stage397_stage394_complex_bimamba":
                    "Stage397 (Stage394 with Complex-State BiMamba at stages 1/3)",
            }
            logger.info(f"Model Type: {descriptions[config.model_type]}")
        return _create_stage391_396_model(config)
    if config.model_type == "iqumamba_fdconv_unirep_ablation":
        if logger is not None:
            logger.info(
                "Model Type: Stage-4 FDConv/UniRepLK ablation "
                f"({config.model_config.get('rf_variant')})"
            )
        return _create_iqumamba_fdconv_unirep_ablation_model(config)
    if config.model_type == "iqumamba_complex_recent_rf":
        if logger is not None:
            logger.info(
                "Model Type: Stage-4 + complex IQ receptive-field front-end "
                f"({config.model_config.get('complex_rf_type')})"
            )
        return _create_iqumamba_complex_recent_rf_model(config)
    if config.model_type == "iqumamba_strong_rf_combination":
        if logger is not None:
            logger.info(
                "Model Type: strong backbone + recent RF combination "
                f"({config.model_config.get('combination_variant')})"
            )
        return _create_iqumamba_strong_rf_combination_model(config)
    if config.model_type in {
        "bimamba_cross_scale_single",
        "bimamba_cross_scale_multi",
        "bimamba_cross_scale_evidence",
    }:
        if logger is not None:
            logger.info(
                "Model Type: IQUBiMamba1D_CrossScaleAttention "
                f"({config.model_type})"
            )
        return _create_bimamba_cross_scale_attention_model(config)
    if config.model_type == "bimamba_cross_scale_estimated_cyclofresh":
        if logger is not None:
            logger.info(
                "Model Type: IQUBiMamba1D_CrossScaleEstimatedCycloFRESH "
                "(stage 235 + stage-79 estimated Cyclo-FRESH adapter)"
            )
        return _create_bimamba_cross_scale_estimated_cyclofresh_model(config)
    if config.model_type in {
        "bimamba_cross_scale_aligned",
        "bimamba_cross_scale_multires_kv",
        "bimamba_cross_scale_bounded_channel",
    }:
        if logger is not None:
            logger.info(
                "Model Type: IQUBiMamba1D_AdvancedCrossScaleAttention "
                f"({config.model_type})"
            )
        return _create_bimamba_advanced_cross_scale_attention_model(config)
    if config.model_type in {
        "bimamba_phase_equivariant_fusion",
        "bimamba_physical_token_cross_attention",
        "bimamba_bottleneck_self_attention",
        "bimamba_hymba_parallel",
        "bimamba_rf_physical_kv",
    }:
        if logger is not None:
            logger.info(f"Model Type: Stage-12 Mamba/KV attention ablation ({config.model_type})")
        return _create_bimamba_kv_attention_ablation_model(config)
    if config.model_type in {
        "bimamba_enhanced_global_cross_attention",
        "bimamba_dual_memory_cross_attention",
        "bimamba_hierarchical_additive_fusion",
        "bimamba_hierarchical_routed_fusion",
        "bimamba_physical_routed_enhanced_cross_attention",
        "bimamba_unified_physical_global_kv",
        "bimamba_physical_film_global_memory",
        "bimamba_scale_isolated_physical_fusion",
        "bimamba_identity_aware_physical_moe",
        "bimamba_cross_gated_dual_memory",
    }:
        if logger is not None:
            logger.info(f"Model Type: hierarchical Stage-12 KV fusion ({config.model_type})")
        return _create_bimamba_hierarchical_kv_fusion_model(config)
    if config.model_type == "bimamba_safe_allstage":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_SafeAllStages (all-stage BiMamba with learnable residual scale)")
        return _create_bimamba_safe_allstage_model(config)
    if config.model_type == "bimamba_direction_gated":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_DirectionGated (all-stage BiMamba with adaptive direction gate)")
        return _create_bimamba_direction_gated_model(config)
    if config.model_type == "bimamba_local_global_allstage":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_LocalGlobalAllStages (all-stage BiMamba with local/global gated fusion)")
        return _create_bimamba_local_global_allstage_model(config)
    if config.model_type == "bimamba_diff_fusion":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_DiffFusion (lightweight symmetric/difference BiMamba fusion)")
        return _create_bimamba_diff_fusion_model(config)
    if config.model_type == "bimamba_adaptive_diff_fusion":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_AdaptiveDiffFusion (lightweight reliability-gated BiMamba fusion)")
        return _create_bimamba_adaptive_diff_fusion_model(config)
    if config.model_type == "bimamba_complex_diff_shared":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_ComplexDiffShared (bottleneck heterogeneous shared-core BiMamba)")
        return _create_bimamba_complex_diff_shared_model(config)
    if config.model_type == "bimamba_time_reversal_shared":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_TimeReversalShared (shared-core reversal-equivariant BiMamba)")
        return _create_bimamba_time_reversal_shared_model(config)
    if config.model_type == "bimamba_alternating_global_local":
        if logger is not None:
            logger.info("Model Type: IQUBiMamba1D_AlternatingGlobalLocal (one global scan plus opposite local context)")
        return _create_bimamba_alternating_global_local_model(config)
    if config.model_type == "iqumamba_pr_unet":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_PerfectReconstruction (Haar polyphase U-Net)")
        return _create_iqumamba_pr_unet_model(config)
    if config.model_type == "iqumamba_pr_shared_perm":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_PerfectReconstruction (training-only shared-permutation deep supervision)")
        return _create_iqumamba_pr_unet_model(config)
    if config.model_type == "iqumamba_pr_restricted_skip":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_RestrictedShallowSkip (bounded stochastic shallow skip)")
        return _create_iqumamba_pr_restricted_skip_model(config)
    if config.model_type == "iqumamba_evidence_moe":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_EvidenceRoutedMoE (Stage-4 backbone + evidence-routed residual experts)")
        return _create_iqumamba_evidence_moe_model(config)
    if config.model_type == "iqumamba_adaptive_multiview_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_AdaptiveMultiViewPrior (Stage-4 backbone + modulation-agnostic multi-view evidence routing)")
        return _create_iqumamba_adaptive_multiview_prior_model(config)
    if config.model_type == "iqumamba_qam_source_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_QAMSourcePrior (Stage-4 backbone + source-wise QAM geometry refinement)")
        return _create_iqumamba_qam_source_prior_model(config)
    if config.model_type == "iqumamba_qam_mma_unrolled":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_QAMMMAUnrolled (Stage-4 backbone + source-wise QAM MMA unfolding)")
        return _create_iqumamba_qam_mma_unrolled_model(config)
    if config.model_type == "iqumamba_qam_density_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_QAMDensityPrior (Stage-4 backbone + source-wise constellation-density prior)")
        return _create_iqumamba_qam_density_prior_model(config)
    if config.model_type == "iqumamba_qam_timing_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_QAMTimingPrior (Stage-4 backbone + blind QAM timing/SPS prior)")
        return _create_iqumamba_qam_timing_prior_model(config)
    if config.model_type == "iqumamba_qam_turbo_unfold":
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1D_QAMTurboUnfold "
                "(Stage-4 + joint soft-QAM/channel/interference-cancellation unfolding)"
            )
        return _create_iqumamba_qam_turbo_unfold_model(config)
    if config.model_type == "iqumamba_noise_contrastive_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_NoiseContrastivePrior (Stage-4 backbone + training-only residual-noise projector)")
        return _create_iqumamba_noise_contrastive_prior_model(config)
    if config.model_type == "iqumamba_blind_sync_factorized":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_BlindSyncFactorized (Stage-4 backbone + mixture-only sync factorization)")
        return _create_iqumamba_blind_sync_factorized_model(config)
    if config.model_type == "iqumamba_sync_conditioned":
        if logger is not None:
            logger.info(
                "Model Type: Stage 366 IQUMamba1D_SyncConditioned "
                "(four-level Stage-4 + explicit sync head + four-scale FiLM; "
                "cross-SNR EMA distillation is training-only)"
            )
        return _create_iqumamba_sync_conditioned_model(config)
    if config.model_type == "iqumamba_physical_sync_rtn":
        if logger is not None:
            logger.info(
                "Model Type: Stages 369/370 IQUMamba1D_PhysicalSyncRTN "
                "(per-source supervised synchronization + differentiable radio views + FiLM)"
            )
        return _create_iqumamba_physical_sync_rtn_model(config)
    if config.model_type == "bimamba_estimated_cyclofresh":
        if logger is not None:
            suffix = (
                " + Stage-290 C1 strict-complex stem"
                if bool(getattr(config, "complex_stem_enable", False))
                else ""
            )
            logger.info(
                "Model Type: IQUBiMamba1D_EstimatedCycloFRESH "
                f"(stage-12 BiMamba + stage-79 estimated FRESH input adapter{suffix})"
            )
        return _create_bimamba_estimated_cyclofresh_model(config)
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
    if config.model_type == "resunet1d_noasc":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_NoASC (ResUNet with plain skip concat)")
        return _create_resunet1d_noasc_model(config)
    if config.model_type == "resunet1d_noasc_latent_mask_real":
        if logger is not None:
            logger.info(
                "Model Type: Stage376 (Stage56 pure 1D ResUNet + "
                "Stage371-style real latent mask)"
            )
        return _create_resunet1d_noasc_latent_mask_model(config)
    if config.model_type == "resunet1d_complexstate_unireplk_latent_mask_real":
        if logger is not None:
            logger.info(
                "Model Type: Stage377 (Stage56 plain decoder + independent "
                "complex-state BiMamba + UniRepLK + real latent mask)"
            )
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_complexstate_unireplk_latent_mask_sigmoid":
        if logger is not None:
            logger.info(
                "Model Type: Stage379 (Stage377 with independent sigmoid "
                "latent masks instead of source-wise softmax)"
            )
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_complexstate_unireplk_light_separator":
        if logger is not None:
            logger.info(
                "Model Type: Stage380 (Stage377 + lightweight dilated "
                "depthwise temporal mask separator)"
            )
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_complexstate_unireplk_no_stage1":
        if logger is not None:
            logger.info(
                "Model Type: Stage381 (Stage377 without encoder-stage1 UniRepLK)"
            )
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_complexstate_unireplk_no_stage1_xl":
        if logger is not None:
            logger.info(
                "Model Type: Stage385 (Stage381-XL without encoder-stage1 "
                "UniRepLK; widths 96/192/384/768)"
            )
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_stage386_unireplk_backbone":
        if logger is not None:
            logger.info("Model Type: Stage386 (Stage381 with replaceable residual cores replaced by UniRepLK)")
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_stage387_integrated_unireplk":
        if logger is not None:
            logger.info("Model Type: Stage387 (one integrated UniRepLK block per encoder/decoder level)")
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_stage388_adaptive_complex_unireplk":
        if logger is not None:
            logger.info("Model Type: Stage388 (adaptive RF-routed complex UniRepLK)")
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_stage389_adaptive_real_unireplk":
        if logger is not None:
            logger.info("Model Type: Stage389 (Stage388 routing-only ablation)")
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_stage390_fixed_complex_unireplk":
        if logger is not None:
            logger.info("Model Type: Stage390 (Stage388 fixed-complex ablation)")
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_complexstate_unireplk_separator":
        if logger is not None:
            logger.info(
                "Model Type: Stage382 (Stage377 UniRepLK adapters relocated "
                "from encoder to mask separator)"
            )
        return _create_resunet1d_complexstate_unireplk_latent_mask_model(config)
    if config.model_type == "resunet1d_hyena_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_HyenaBottleneck (Stage 186, stage56 + Hyena bottleneck)")
        return _create_resunet1d_hyena_bottleneck_model(config)
    if config.model_type == "resunet1d_spectral_lowrank_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SpectralLowRankBottleneck (Stage 187, stage56 + low-rank spectral bottleneck)")
        return _create_resunet1d_spectral_lowrank_bottleneck_model(config)
    if config.model_type == "resunet1d_mega_mid_encoder":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_MegaMidEncoder (Stage 188, stage56 + MEGA mid encoder)")
        return _create_resunet1d_mega_mid_encoder_model(config)
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
    if config.model_type == "resunet1d_skip_enhanced_lssg_dw":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Skip_Enhanced_LSSG_DW (Stage 154)")
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_dw")
    if config.model_type == "resunet1d_crossscale_lssg":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_CrossScaleLSSG (Stage 155)")
        return _create_resunet1d_crossscale_lssg_model(config)
    if config.model_type == "resunet1d_sk_lssg":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_SKLSSG (Stage 156)")
        return _create_resunet1d_sk_lssg_model(config)
    if config.model_type == "resunet1d_freq_lssg":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_FreqLSSG (Stage 157)")
        return _create_resunet1d_freq_lssg_model(config)
    if config.model_type == "resunet1d_focal_lssg":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_FocalLSSG (Stage 158)")
        return _create_resunet1d_focal_lssg_model(config)
    if config.model_type == "resunet1d_wavelet_dccb":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_WaveletDCCB (Stage 159)")
        return _create_resunet1d_wavelet_dccb_model(config)
    if config.model_type == "resunet1d_complex_cyclo_dccb":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_ComplexCycloDCCB (Stage 160)")
        return _create_resunet1d_complex_cyclo_dccb_model(config)
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
    if config.model_type == "resunet1d_skip_enhanced_lssg_se":
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_se")
    if config.model_type == "resunet1d_skip_enhanced_lssg_dw":
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_dw")
    if config.model_type == "resunet1d_skip_enhanced_lssg_swiglu":
        return _create_resunet1d_skip_enhanced_model(config, skip_mode="lssg_swiglu")
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
    if config.model_type == "resunet1d_bottleneck_dccb_deep":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_Deep (Stage 163)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb_deep")
    if config.model_type == "resunet1d_bottleneck_dccb_cross_attn":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_CrossAttn (Stage 164)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb_cross_attn")
    if config.model_type == "resunet1d_bottleneck_dccb_adaptive_lags":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_AdaptiveLags (Stage 165)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb_adaptive_lags")
    if config.model_type == "resunet1d_bottleneck_dccb_mamba_v2":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_Bottleneck_DCCB_MambaV2 (Stage 166)")
        return _create_resunet1d_bottleneck_enhanced_model(config, bottleneck_mode="dccb_mamba")
    if config.model_type == "resunet1d_agent_attention_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_AgentAttentionBottleneck (Stage 168)")
        return _create_resunet1d_agent_attention_bottleneck_model(config)
    if config.model_type == "resunet1d_transnext_attention_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_TransNeXtBottleneck (Stage 169)")
        return _create_resunet1d_transnext_attention_bottleneck_model(config)
    if config.model_type == "resunet1d_bilevel_routing_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_BiLevelRoutingBottleneck (Stage 170)")
        return _create_resunet1d_bilevel_routing_bottleneck_model(config)
    if config.model_type == "resunet1d_deformable_temporal_bottleneck":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_DeformableTemporalBottleneck (Stage 171)")
        return _create_resunet1d_deformable_temporal_bottleneck_model(config)
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
    if config.model_type == "esd_mask_wrapper":
        if logger is not None:
            logger.info("Model Type: ESDMaskWrapper_SkipEnhanced (Stage 153)")
        return _create_esd_mask_model(config)
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
            logger.info("Model Type: Stage 124 - DCCB + Channel-LSSG")
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
    if config.model_type == "iq_resdilated_unet":
        if logger is not None:
            logger.info(
                "Model Type: IQResDilatedUNet "
                "(Stage 264, learned analysis + gated dilation + bottleneck BiMamba)"
            )
        return _create_iq_resdilated_unet_model(config)
    if config.model_type == "rfchallenge_rfdemucs":
        if logger is not None:
            logger.info(
                "Model Type: RFDEMUCS (Stage 358, TUB five-level U-Net + BLSTM)"
            )
        return _create_rfdemucs_model(config)
    if config.model_type == "icassp_baseline_wavenet":
        if logger is not None:
            logger.info("Model Type: ICASPBaselineWaveNet")
        return _create_icassp_baseline_wavenet_model(config)
    if config.model_type == "icassp_wavenet_mamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetMamba "
                "(Stage 257, post-skip unidirectional Mamba fusion)"
            )
        return _create_icassp_wavenet_mamba_model(config, bidirectional=False)
    if config.model_type == "icassp_wavenet_bimamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetBiMamba "
                "(Stage 258, post-skip bidirectional Mamba fusion)"
            )
        return _create_icassp_wavenet_mamba_model(config, bidirectional=True)
    if config.model_type == "icassp_wavenet_multirate_mamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetMultiRateMamba "
                "(Stage 259, WaveNet-10 + stride-4 unidirectional Mamba)"
            )
        return _create_icassp_wavenet_multirate_mamba_model(config, bidirectional=False)
    if config.model_type == "icassp_wavenet_multirate_bimamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetMultiRateBiMamba "
                "(Stage 260, WaveNet-10 + stride-4 bidirectional Mamba)"
            )
        return _create_icassp_wavenet_multirate_mamba_model(config, bidirectional=True)
    if config.model_type == "icassp_wavenet_interleaved_mamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedMamba "
                "(Stage 261, WaveNet-10 -> stride-4 Mamba -> WaveNet-10)"
            )
        return _create_icassp_wavenet_interleaved_mamba_model(config, bidirectional=False)
    if config.model_type == "icassp_wavenet_interleaved_bimamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedBiMamba "
                "(Stage 262, WaveNet-10 -> stride-4 BiMamba -> WaveNet-10)"
            )
        return _create_icassp_wavenet_interleaved_mamba_model(config, bidirectional=True)
    if config.model_type == "icassp_wavenet_chunk_mamba_strong_fusion":
        if logger is not None:
            insert_after = int(getattr(config, "mamba_insert_after_block", 5))
            residual_layers = int(getattr(config, "residual_layers", 10))
            logger.info(
                "Model Type: ICASPBaselineWaveNetChunkMambaStrongFusion "
                f"(WaveNet-{insert_after} -> chunk Mamba strong fusion -> "
                f"WaveNet-{residual_layers - insert_after})"
            )
        return _create_icassp_wavenet_chunk_mamba_strong_fusion_model(config)
    if config.model_type == "icassp_wavenet_mamba_film_controller":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetMambaFiLMController "
                "(Stage 269, Mamba-controlled FiLM/residual/skip gates)"
            )
        return _create_icassp_wavenet_mamba_film_controller_model(config)
    if config.model_type == "icassp_wavenet_mamba_dilation_skip_router":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetMambaDilationSkipRouter "
                "(Stage 270, frame-conditioned dilation-scale/skip routing)"
            )
        return _create_icassp_wavenet_mamba_dilation_skip_router_model(config)
    if config.model_type == "icassp_wavenet_interleaved_phase_aware_reverse_mamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedPhaseAwareReverseMamba "
                "(Stage 271, Stage-261 + conj(flip(I/Q)) reverse branch)"
            )
        return _create_icassp_wavenet_interleaved_phase_aware_reverse_mamba_model(config)
    if config.model_type == "icassp_wavenet_interleaved_gated_bimamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedGatedBiMamba "
                "(Stage 268, Stage-261 forward Mamba + gated reverse correction)"
            )
        return _create_icassp_wavenet_interleaved_gated_bimamba_model(config)
    if config.model_type == "icassp_wavenet_interleaved_crossscale_bimamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedCrossScaleBiMamba "
                "(Stage 265, Stage-261 + Stage-235-style BiMamba global memory)"
            )
        return _create_icassp_wavenet_interleaved_crossscale_bimamba_model(config)
    if config.model_type == "icassp_wavenet_interleaved_stage235_memory":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedStage235Memory "
                "(Stage 272, Stage-235-style K/V memory replaces Stage-261 context)"
            )
        return _create_icassp_wavenet_interleaved_stage235_memory_model(config)
    if config.model_type == "icassp_wavenet_interleaved_physical_moe_bimamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedPhysicalMoEBiMamba "
                "(Stage 266, Stage-261 + Stage-255-style physical MoE)"
            )
        return _create_icassp_wavenet_interleaved_physical_moe_bimamba_model(config)
    if config.model_type == "icassp_wavenet_interleaved_cyclofresh":
        if logger is not None:
            logger.info(
                "Model Type: ICASPBaselineWaveNetInterleavedMambaCycloFRESH "
                "(Stage 267, Stage-261 + Stage-79 estimated Cyclo-FRESH prior)"
            )
        return _create_icassp_wavenet_interleaved_cyclofresh_model(config)
    if config.model_type == "icassp_wavenet_antialiased_mamba":
        if logger is not None:
            logger.info(
                "Model Type: ICASPAntiAliasedInterleavedMamba "
                "(Stage 278, anti-aliased Stage-261 context)"
            )
        return _create_icassp_wavenet_antialiased_mamba_model(config)
    if config.model_type == "icassp_wavenet_temporal_physical_controller":
        if logger is not None:
            logger.info(
                "Model Type: ICASPTemporalPhysicalControllerWaveNet "
                "(Stage 279, ordered physical tokens and time-varying controls)"
            )
        return _create_icassp_wavenet_temporal_physical_controller_model(config)
    if config.model_type == "icassp_symbol_clock_wavenet":
        if logger is not None:
            logger.info(
                "Model Type: ICASPSymbolClockWaveNet "
                "(Stages 280-282, physical symbol-clock dilation routing)"
            )
        return _create_icassp_symbol_clock_wavenet_model(config)
    if config.model_type == "icassp_complex_wavenet":
        if logger is not None:
            logger.info(
                "Model Type: ICASPComplexWaveNet "
                "(Stages 283-289, Stage-273 strict-complex ablations)"
            )
        return _create_icassp_complex_wavenet_model(config)
    if config.model_type in {
        "iqumamba_stage4_complex_c1",
        "iqumamba_stage4_complex",
    }:
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1DComplexStage4 "
                "(Stages 290-294, controlled Stage-4 complex ablations)"
            )
        return _create_iqumamba_stage4_complex_model(config)
    if config.model_type in {
        "iqumamba_stage4_complex_state",
        "iqumamba_stage4_complex_stem_complex_state",
        "iqumamba_rf_mamba3",
        "iqumamba_rf_mamba3_fast",
        "iqumamba_real_state_trap_reliability",
    }:
        if logger is not None:
            if config.model_type == "iqumamba_stage4_complex_stem_complex_state":
                stage_label = (
                    "Stage 299, Stage-290 complex stem + Stage-295 "
                    "complex-state SSM"
                )
            elif config.model_type in {
                "iqumamba_rf_mamba3",
                "iqumamba_rf_mamba3_fast",
            }:
                stage_label = (
                    "Stages 330-333/341/351/354-356, controlled RF Mamba-3 "
                    "recurrence and complex-stem ablations"
                )
            elif config.model_type == "iqumamba_real_state_trap_reliability":
                stage_label = (
                    "Stage 347, Stage-333 real-state trapezoidal + reliability "
                    "ablation"
                )
            else:
                stage_label = "Stage 295, complex-state / oscillatory selective SSM"
            logger.info(
                "Model Type: IQUMamba1DComplexStateMamba "
                f"({stage_label}; "
                f"scan_backend={config.model_config.get('scan_backend', 'auto')})"
            )
        return _create_iqumamba_stage4_complex_state_model(config)
    if config.model_type in {
        "iqumamba_stage4_official_mamba3",
        "iqumamba_full_rf_mamba3_combination",
    }:
        if logger is not None:
            descriptions = {
                "iqumamba_stage4_official_mamba3":
                    "Stage 340, official Mamba-3 fused-kernel reproduction",
                "iqumamba_full_rf_mamba3_combination":
                    "Stages 342/349, complex stem + full RF Mamba-3 + UniRepLK",
            }
            logger.info(
                "Model Type: IQUMamba1D Mamba-3 extension "
                f"({descriptions[config.model_type]})"
            )
        return _create_iqumamba_mamba3_extension_model(config)
    if config.model_type == "iqumamba_official_rf_mamba3":
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1D official fused RF-aware Mamba-3 "
                f"(Stages 343-346, variant="
                f"{config.model_config.get('rf_mamba3_variant')})"
            )
        return _create_iqumamba_official_rf_mamba3_model(config)
    if config.model_type == "iqumamba_phase_folded_mamba":
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1DPhaseFoldedMamba "
                "(Stage 334, blind multi-period phase trajectories)"
            )
        return _create_iqumamba_phase_folded_mamba_model(config)
    if config.model_type in {
        "iqumamba_stage4_mamba2_ssd",
        "iqumamba_stage4_s4d",
        "iqumamba_stage4_complex_s4d",
        "iqumamba_stage4_s4d_unireplk",
        "iqumamba_stage4_s4d_reliability",
        "iqumamba_stage4_role_rf",
    }:
        if logger is not None:
            descriptions = {
                "iqumamba_stage4_mamba2_ssd":
                    "Stage 335, official Mamba-2/SSD reproduction",
                "iqumamba_stage4_s4d":
                    "Stages 336/352, S4D-Lin with optional strict-complex stem",
                "iqumamba_stage4_complex_s4d":
                    "Stage 357, full strict-complex U-Net and S4D memory",
                "iqumamba_stage4_s4d_unireplk":
                    "Stage 353, strict-complex stem + S4D-Lin + UniRepLK",
                "iqumamba_stage4_s4d_reliability":
                    "Stage 348, S4D-Lin poles + reliability-controlled state time",
                "iqumamba_stage4_role_rf":
                    "Stages 337-339, delta/B/C receptive-field ablations",
            }
            logger.info(
                "Model Type: IQUMamba1D memory/RF experiment "
                f"({descriptions[config.model_type]})"
            )
        return _create_iqumamba_memory_rf_model(config)
    if config.model_type in {
        "iqumamba_stage299_cross_scale",
        "bimamba_stage298_complex_bottleneck",
        "bimamba_complex_stem_cross_scale",
        "stage298_stage299_output_fusion",
        "iqumamba_stage299_gated_fresh",
    }:
        if logger is not None:
            stage_labels = {
                "iqumamba_stage299_cross_scale":
                    "Stage 301: Stage 299 + bottleneck cross-scale attention",
                "bimamba_stage298_complex_bottleneck":
                    "Stage 302: Stage 298 + one complex-state bottleneck",
                "bimamba_complex_stem_cross_scale":
                    "Stage 303: complex stem + BiMamba + cross-scale, no FRESH",
                "stage298_stage299_output_fusion":
                    "Stage 304: learnable Stage 298/299 output fusion",
                "iqumamba_stage299_gated_fresh":
                    "Stage 305: Stage 299 + conservative gated FRESH residual",
            }
            logger.info(
                "Model Type: IQUMamba1D combined experiment "
                f"({stage_labels[config.model_type]})"
            )
        return _create_iqumamba_combined_stage_model(config)
    if config.model_type == "kutii_learnable_dilation_wavenet":
        if logger is not None:
            logger.info(
                "Model Type: KUTIIStyleLearnableDilationWaveNet "
                "(public KU-TII reproduction)"
            )
        return _create_kutii_learnable_dilation_wavenet_model(config)
    if config.model_type == "kutii_dual_source_wavenet":
        if logger is not None:
            variant = str(getattr(config, "comparison_variant", "full"))
            variant_labels = {
                "full": "Stage 378 full (256 channels, 30 blocks)",
                "param_match": "Stage 383 ParamMatch (108 channels, 30 blocks)",
                "flops_match": "Stage 384 legacy FLOPsMatch (56 channels, 30 blocks)",
                "flops_match_stage381_enhanced": (
                    "Stage 384 Stage4-FLOPsMatch + Stage381 modules "
                    "(52 channels, 30 blocks)"
                ),
            }
            variant_label = variant_labels.get(
                variant,
                f"comparison variant={variant}",
            )
            logger.info(
                "Model Type: KUTIIDualSourceWaveNet "
                f"({variant_label}, shared learnable-dilation trunk + source slots)"
            )
        return _create_kutii_dual_source_wavenet_model(config)
    if config.model_type == "iqumamba_decodermamba":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_DecoderMamba (stage-4 IQUMamba + decoder Mamba)")
        return _create_iqumamba_decodermamba_model(config)
    if config.model_type == "iqumamba_uric":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_URIC (Stage 176, stage4 IQUMamba + URIC)")
        return _create_iqumamba_uric_model(config)
    if config.model_type == "iqumamba_gla_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_GLAEncoder (Stage 172, encoder Mamba replacement)")
        return _create_iqumamba_gla_encoder_model(config)
    if config.model_type == "iqumamba_mega_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_MegaEncoder (Stage 173, encoder Mamba replacement)")
        return _create_iqumamba_mega_encoder_model(config)
    if config.model_type == "iqumamba_hyena_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_HyenaEncoder (Stage 174, encoder Mamba replacement)")
        return _create_iqumamba_hyena_encoder_model(config)
    if config.model_type == "iqumamba_retnet_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_RetNetEncoder (Stage 175, encoder Mamba replacement)")
        return _create_iqumamba_retnet_encoder_model(config)
    if config.model_type == "iqumamba_griffin_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_GriffinEncoder (Stage 177, encoder Mamba replacement)")
        return _create_iqumamba_griffin_encoder_model(config)
    if config.model_type == "iqumamba_xlstm_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_xLSTMEncoder (Stage 178, encoder Mamba replacement)")
        return _create_iqumamba_xlstm_encoder_model(config)
    if config.model_type == "iqumamba_spectral_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_SpectralEncoder (Stage 179, encoder Mamba replacement)")
        return _create_iqumamba_spectral_encoder_model(config)
    if config.model_type == "iqumamba_spectral_lowrank_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_SpectralLowRankEncoder (Stage 185, low-rank spectral replacement)")
        return _create_iqumamba_spectral_lowrank_encoder_model(config)
    if config.model_type == "iqumamba_delta_linear_encoder":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_DeltaLinearEncoder (Stage 180, encoder Mamba replacement)")
        return _create_iqumamba_delta_linear_encoder_model(config)
    if config.model_type == "resunet1d_lssg_dw_skip":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_LSSGDWSkip (Stage 181, clean LSSG-DW skip fusion)")
        return _create_resunet1d_lssg_dw_skip_model(config)
    if config.model_type == "resunet1d_deformable_temporal_skip":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_DeformableTemporalSkip (Stage 182, learned temporal skip alignment)")
        return _create_resunet1d_deformable_temporal_skip_model(config)
    if config.model_type == "resunet1d_frequency_aware_skip":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_FrequencyAwareSkip (Stage 183, FFT-conditioned skip fusion)")
        return _create_resunet1d_frequency_aware_skip_model(config)
    if config.model_type == "resunet1d_complex_aware_skip":
        if logger is not None:
            logger.info("Model Type: IQUResUNet1D_ComplexAwareSkip (Stage 184, I/Q-pair-aware skip fusion)")
        return _create_resunet1d_complex_aware_skip_model(config)
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
    if config.model_type == "iqumamba_feature_topology_adapter":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_FeatureTopologyAdapter (stage-4 IQUMamba + feature-domain topology adapter)")
        return _create_iqumamba_feature_topology_adapter_model(config)
    if config.model_type == "iqumamba_cyclic_wiener_residual":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_CyclicWienerResidual (stage-4 IQUMamba + source-wise cyclic-Wiener residual head)")
        return _create_iqumamba_cyclic_wiener_residual_model(config)
    if config.model_type == "iqumamba_psk_phase_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_PSKPhasePrior (stage-4 IQUMamba + phase-step input adapter)")
        return _create_iqumamba_psk_phase_prior_model(config)
    if config.model_type == "iqumamba_qam_lattice_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_QAMLatticePrior (stage-4 IQUMamba + QAM lattice input adapter)")
        return _create_iqumamba_qam_lattice_prior_model(config)
    if config.model_type == "iqumamba_apsk_ring_prior":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_APSKRingPrior (stage-4 IQUMamba + APSK ring-radius input adapter)")
        return _create_iqumamba_apsk_ring_prior_model(config)
    if config.model_type == "iqumamba_noise_aware_mc":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_NoiseAwareMC (stage-4 IQUMamba + residual-noise mixture consistency)")
        return _create_iqumamba_noise_aware_mc_model(config)
    if config.model_type == "iqumamba_multiview_consistent":
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1D_MultiViewConsistent "
                "(stage-4 + shared-IQ power norm + noise-aware time-domain projection)"
            )
        return _create_iqumamba_multiview_consistent_model(config)
    if config.model_type == "iqumamba_shared_iq_norm":
        if logger is not None:
            logger.info(
                "Model Type: IQUMamba1D_SharedIQNorm "
                "(stage-4 + shared-IQ power normalization only)"
            )
        return _create_iqumamba_shared_iq_norm_model(config)
    if config.model_type == "iqumamba_low_snr_se":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_LowSNRSE (stage-4 IQUMamba + low-SNR enhancement front-end)")
        return _create_iqumamba_low_snr_se_model(config)
    if config.model_type == "iqumamba_low_snr_snr_cond":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_LowSNRSNRConditioned (stage-4 IQUMamba + SNR-proxy conditioned low-SNR front-end)")
        return _create_iqumamba_low_snr_snr_cond_model(config)
    if config.model_type == "iqumamba_low_snr_cyclic_cond":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_LowSNRCyclicConditioned (stage-4 IQUMamba + cyclic-reliability conditioned low-SNR front-end)")
        return _create_iqumamba_low_snr_cyclic_cond_model(config)
    if config.model_type == "iqumamba_neural_wiener_se":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_NeuralWienerSE (stage-4 IQUMamba + neural Wiener low-SNR front-end)")
        return _create_iqumamba_neural_wiener_se_model(config)
    if config.model_type == "iqumamba_asg_mamba":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_ASGMamba (stage-4 IQUMamba + adaptive spectral gating before Mamba)")
        return _create_iqumamba_asg_mamba_model(config)
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
            suffix = (
                " + Stage-290 C1 strict-complex stem"
                if bool(getattr(config, "complex_stem_enable", False))
                else ""
            )
            logger.info(
                "Model Type: IQUMamba1D_EstimatedCycloFRESH "
                f"(stage-4 IQUMamba + mixture-estimated cyclic-frequency FRESH input adapter{suffix})"
            )
        return _create_iqumamba_estimated_cyclofresh_model(config)
    if config.model_type == "iqumamba_multipeak_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_MultiPeakCycloFRESH (stage-4 IQUMamba + multi-peak mixture-estimated FRESH input adapter)")
        return _create_iqumamba_multipeak_cyclofresh_model(config)
    if config.model_type == "iqumamba_sample_cyclofresh":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_SampleCycloFRESH (stage-4 IQUMamba + per-sample mixture-estimated FRESH input adapter)")
        return _create_iqumamba_sample_cyclofresh_model(config)
    if config.model_type == "iqumamba_multihyp_cyclic_reliability":
        if logger is not None:
            logger.info("Model Type: IQUMamba1D_MultiHypCyclicReliability (stage-4 IQUMamba + null-aware cyclic evidence selection)")
        return _create_iqumamba_multihyp_cyclic_reliability_model(config)
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


def _create_iqumamba_physics_receptive_field_model(config):
    from models.IQUMamba1D_PhysicsReceptiveField import (
        IQUMamba1DPhysicalCanonical,
        IQUMamba1DSymbolDelayDopplerRF,
    )

    cfg = config.model_config
    common = dict(
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
    )
    if config.model_type == "iqumamba_physical_canonical":
        return IQUMamba1DPhysicalCanonical(
            **common,
            canonical_cfo_weight=float(cfg.get("canonical_cfo_weight", 1.0)),
            canonical_bandwidth_weight=float(
                cfg.get("canonical_bandwidth_weight", 0.0)
            ),
            canonical_ascending=bool(cfg.get("canonical_ascending", True)),
            canonical_symbol_orders=cfg.get("canonical_symbol_orders", [2, 4, 8]),
            canonical_order_temperature=float(
                cfg.get("canonical_order_temperature", 0.1)
            ),
            canonical_eps=float(cfg.get("canonical_eps", 1e-8)),
        ).to(device)
    return IQUMamba1DSymbolDelayDopplerRF(
        **common,
        rf_sps_candidates=cfg.get("rf_sps_candidates", [8, 10, 16, 20, 32, 40]),
        rf_symbol_spans=cfg.get("rf_symbol_spans", [1, 2, 4, 8, 16, 32, 64]),
        rf_default_sps=float(cfg.get("rf_default_sps", 20.0)),
        rf_sps_temperature=float(cfg.get("rf_sps_temperature", 0.25)),
        rf_max_phase_step=float(cfg.get("rf_max_phase_step", 0.25)),
        rf_max_doppler_offset=float(cfg.get("rf_max_doppler_offset", 0.05)),
        rf_gate_hidden=int(cfg.get("rf_gate_hidden", 24)),
        rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.01)),
        rf_eps=float(cfg.get("rf_eps", 1e-6)),
    ).to(device)


def _create_iqumamba_recent_rf_model(config):
    from models.IQUMamba1D_RecentRFModules import IQUMamba1DRecentRF

    cfg = config.model_config
    common = dict(
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
    )
    return IQUMamba1DRecentRF(
        **common,
        rf_module_type=cfg["rf_module_type"],
        rf_hidden_channels=int(cfg.get("rf_hidden_channels", 16)),
        rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
        rf_module_config=cfg,
    ).to(device)


def _create_stage391_396_model(config):
    from models.IQUMamba1D_Stage391Ablations import (
        IQUMamba1D_Stage391,
        IQUMamba1D_Stage392,
        IQUMamba1D_Stage393,
        IQUMamba1D_Stage394,
        IQUMamba1D_Stage395,
        IQUMamba1D_Stage396,
        IQUMamba1D_Stage397,
    )

    cfg = config.model_config
    model_classes = {
        "iqumamba_stage391_unireplk_post_mamba": IQUMamba1D_Stage391,
        "iqumamba_stage392_unireplk_parallel_delta": IQUMamba1D_Stage392,
        "iqumamba_stage393_adaptive_rf_post_mamba": IQUMamba1D_Stage393,
        "iqumamba_stage394_adaptive_rf_parallel_delta": IQUMamba1D_Stage394,
        "iqumamba_stage395_unireplk_delta_post": IQUMamba1D_Stage395,
        "iqumamba_stage396_unireplk_full_pre": IQUMamba1D_Stage396,
        "iqumamba_stage397_stage394_complex_bimamba": IQUMamba1D_Stage397,
    }
    model_class = model_classes[config.model_type]
    model_kwargs = dict(
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
        rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
        rf_large_kernel=int(cfg.get("rf_large_kernel", 17)),
        rf_kernels=tuple(int(value) for value in cfg.get("rf_kernels", (9, 17))),
        rf_ffn_factor=int(cfg.get("rf_ffn_factor", 4)),
        rf_layer_scale=float(cfg.get("rf_layer_scale", 1e-6)),
    )
    if config.model_type == "iqumamba_stage397_stage394_complex_bimamba":
        model_kwargs.update(
            complex_state_d_state=int(cfg.get("complex_state_d_state", 8)),
            complex_state_d_conv=int(cfg.get("complex_state_d_conv", 4)),
            complex_state_expand=int(cfg.get("complex_state_expand", 2)),
            complex_state_scan_checkpoint=bool(cfg.get("complex_state_scan_checkpoint", True)),
            complex_state_scan_backend=str(cfg.get("complex_state_scan_backend", "auto")),
            complex_state_fusion_hidden=int(cfg.get("complex_state_fusion_hidden", 64)),
            bimamba_residual_scale_init=float(cfg.get("bimamba_residual_scale_init", 1.0)),
        )
    return model_class(**model_kwargs).to(device)


def _create_iqumamba_fdconv_unirep_ablation_model(config):
    from models.IQUMamba1D_RecentRFModules import IQUMamba1DFDConvUniRepAblation

    cfg = config.model_config
    return IQUMamba1DFDConvUniRepAblation(
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
        rf_variant=cfg["rf_variant"],
        rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
        rf_module_config=cfg,
    ).to(device)


def _create_iqumamba_strong_rf_combination_model(config):
    from models.IQUMamba1D_StrongRFCombinations import (
        IQUMamba1DStrongRFCombination,
    )

    cfg = config.model_config
    return IQUMamba1DStrongRFCombination(
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
        combination_variant=cfg["combination_variant"],
        complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
        rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
        rf_module_config=cfg,
        estimated_cyclofresh_config=cfg,
    ).to(device)


def _create_iqumamba_complex_recent_rf_model(config):
    from models.IQUMamba1D_ComplexRecentRF import IQUMamba1DComplexRecentRF

    cfg = config.model_config
    return IQUMamba1DComplexRecentRF(
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
        complex_rf_type=cfg["complex_rf_type"],
        complex_hidden_channels=int(cfg.get("complex_hidden_channels", 8)),
        complex_residual_scale_init=float(
            cfg.get("complex_residual_scale_init", 0.05)
        ),
        complex_rf_config=cfg,
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


def _create_iqumamba_uric_model(config):
    from models.IQUMamba1D_URIC import IQUMamba1D_URIC

    return IQUMamba1D_URIC(
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


def _iqumamba_replacement_common_kwargs(config):
    return {
        "input_size": input_size,
        "input_channels": config.input_channels,
        "n_stages": config.n_stages,
        "features_per_stage": config.features_per_stage,
        "conv_op": nn.Conv1d,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "n_conv_per_stage": config.n_conv_per_stage,
        "num_classes": config.num_classes,
        "n_conv_per_stage_decoder": config.n_conv_per_stage_decoder,
        "deep_supervision": config.deep_supervision,
    }


def _create_iqumamba_gla_encoder_model(config):
    from models.IQUMamba1D_GLAEncoder import IQUMamba1D_GLAEncoder

    model_cfg = config.model_config
    return IQUMamba1D_GLAEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _create_iqumamba_mega_encoder_model(config):
    from models.IQUMamba1D_MegaEncoder import IQUMamba1D_MegaEncoder

    model_cfg = config.model_config
    return IQUMamba1D_MegaEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        ema_kernel_size=int(model_cfg.get("ema_kernel_size", 63)),
        expansion=float(model_cfg.get("expansion", 2.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _create_iqumamba_hyena_encoder_model(config):
    from models.IQUMamba1D_HyenaEncoder import IQUMamba1D_HyenaEncoder

    model_cfg = config.model_config
    return IQUMamba1D_HyenaEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        filter_hidden=int(model_cfg.get("filter_hidden", 64)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _create_iqumamba_retnet_encoder_model(config):
    from models.IQUMamba1D_RetNetEncoder import IQUMamba1D_RetNetEncoder

    model_cfg = config.model_config
    return IQUMamba1D_RetNetEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        retention_kernel_size=int(model_cfg.get("retention_kernel_size", 128)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _create_iqumamba_griffin_encoder_model(config):
    from models.IQUMamba1D_GriffinEncoder import IQUMamba1D_GriffinEncoder

    model_cfg = config.model_config
    return IQUMamba1D_GriffinEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        recurrence_kernel_size=int(model_cfg.get("recurrence_kernel_size", 128)),
        local_kernel_size=int(model_cfg.get("local_kernel_size", 5)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _create_iqumamba_xlstm_encoder_model(config):
    from models.IQUMamba1D_xLSTMEncoder import IQUMamba1D_xLSTMEncoder

    model_cfg = config.model_config
    return IQUMamba1D_xLSTMEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
        forget_bias=float(model_cfg.get("forget_bias", 1.0)),
    ).to(device)


def _create_iqumamba_spectral_encoder_model(config):
    from models.IQUMamba1D_SpectralEncoder import IQUMamba1D_SpectralEncoder

    model_cfg = config.model_config
    return IQUMamba1D_SpectralEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        mode_count=int(model_cfg.get("mode_count", 128)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _create_iqumamba_spectral_lowrank_encoder_model(config):
    from models.IQUMamba1D_SpectralLowRankEncoder import IQUMamba1D_SpectralLowRankEncoder

    model_cfg = config.model_config
    return IQUMamba1D_SpectralLowRankEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        mode_count=int(model_cfg.get("mode_count", 32)),
        spectral_rank=int(model_cfg.get("spectral_rank", 4)),
        dropout=float(model_cfg.get("dropout", 0.10)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.02)),
    ).to(device)


def _create_iqumamba_delta_linear_encoder_model(config):
    from models.IQUMamba1D_DeltaLinearEncoder import IQUMamba1D_DeltaLinearEncoder

    model_cfg = config.model_config
    return IQUMamba1D_DeltaLinearEncoder(
        **_iqumamba_replacement_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
    ).to(device)


def _resunet_skip_fusion_common_kwargs(config):
    return {
        "input_size": input_size,
        "input_channels": config.input_channels,
        "n_stages": config.n_stages,
        "features_per_stage": config.features_per_stage,
        "conv_op": nn.Conv1d,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "n_conv_per_stage": config.n_conv_per_stage,
        "num_classes": config.num_classes,
        "n_conv_per_stage_decoder": config.n_conv_per_stage_decoder,
        "deep_supervision": config.deep_supervision,
    }


def _create_resunet1d_lssg_dw_skip_model(config):
    from models.IQUResUNet1D_SkipFusionReplacements import IQUResUNet1D_LSSGDWSkip

    model_cfg = config.model_config
    return IQUResUNet1D_LSSGDWSkip(
        **_resunet_skip_fusion_common_kwargs(config),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        depthwise_kernel_size=int(model_cfg.get("depthwise_kernel_size", 5)),
    ).to(device)


def _create_resunet1d_deformable_temporal_skip_model(config):
    from models.IQUResUNet1D_SkipFusionReplacements import IQUResUNet1D_DeformableTemporalSkip

    model_cfg = config.model_config
    return IQUResUNet1D_DeformableTemporalSkip(
        **_resunet_skip_fusion_common_kwargs(config),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        sampling_offsets=int(model_cfg.get("sampling_offsets", 4)),
        offset_kernel_size=int(model_cfg.get("offset_kernel_size", 5)),
        offset_range=float(model_cfg.get("offset_range", 8.0)),
    ).to(device)


def _create_resunet1d_frequency_aware_skip_model(config):
    from models.IQUResUNet1D_SkipFusionReplacements import IQUResUNet1D_FrequencyAwareSkip

    model_cfg = config.model_config
    return IQUResUNet1D_FrequencyAwareSkip(
        **_resunet_skip_fusion_common_kwargs(config),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        freq_bins=model_cfg.get("freq_bins", [1, 2, 4, 8, 16, 32]),
    ).to(device)


def _create_resunet1d_complex_aware_skip_model(config):
    from models.IQUResUNet1D_SkipFusionReplacements import IQUResUNet1D_ComplexAwareSkip

    model_cfg = config.model_config
    return IQUResUNet1D_ComplexAwareSkip(
        **_resunet_skip_fusion_common_kwargs(config),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        complex_eps=float(model_cfg.get("complex_eps", 1e-6)),
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


def _create_iqumamba_psk_phase_prior_model(config):
    from models.IQUMamba1D_ModulationPriors import IQUMamba1D_PSKPhasePrior

    return IQUMamba1D_PSKPhasePrior(
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
        psk_prior_hidden_channels=int(getattr(config, 'psk_prior_hidden_channels', 8)),
        psk_prior_harmonics=tuple(getattr(config, 'psk_prior_harmonics', (1, 2, 4, 8))),
        psk_prior_kernel_size=int(getattr(config, 'psk_prior_kernel_size', 9)),
        psk_prior_scale_init=float(getattr(config, 'psk_prior_scale_init', 0.01)),
        psk_prior_reliability_floor=float(getattr(config, 'psk_prior_reliability_floor', 0.05)),
        psk_prior_zero_init=bool(getattr(config, 'psk_prior_zero_init', True)),
    ).to(device)


def _create_iqumamba_feature_topology_adapter_model(config):
    from models.IQUMamba1D_FeatureTopologyAdapter import IQUMamba1D_FeatureTopologyAdapter

    return IQUMamba1D_FeatureTopologyAdapter(
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
        feature_topology_hidden_channels=int(getattr(config, 'feature_topology_hidden_channels', 16)),
        feature_topology_kernel_size=int(getattr(config, 'feature_topology_kernel_size', 7)),
        feature_topology_scale_init=float(getattr(config, 'feature_topology_scale_init', 0.01)),
        feature_topology_zero_init=bool(getattr(config, 'feature_topology_zero_init', True)),
        feature_topology_apply_stages=tuple(getattr(config, 'feature_topology_apply_stages', ())),
    ).to(device)


def _create_iqumamba_cyclic_wiener_residual_model(config):
    from models.IQUMamba1D_CyclicWienerResidual import IQUMamba1D_CyclicWienerResidual

    return IQUMamba1D_CyclicWienerResidual(
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
        cyclic_wiener_hidden_channels=int(getattr(config, 'cyclic_wiener_hidden_channels', 16)),
        cyclic_wiener_kernel_size=int(getattr(config, 'cyclic_wiener_kernel_size', 9)),
        cyclic_wiener_min_freq=float(getattr(config, 'cyclic_wiener_min_freq', 1.0 / 128.0)),
        cyclic_wiener_max_freq=float(getattr(config, 'cyclic_wiener_max_freq', 1.0 / 4.0)),
        cyclic_wiener_default_freq=float(getattr(config, 'cyclic_wiener_default_freq', 1.0 / 32.0)),
        cyclic_wiener_num_harmonics=int(getattr(config, 'cyclic_wiener_num_harmonics', 2)),
        cyclic_wiener_scale_init=float(getattr(config, 'cyclic_wiener_scale_init', 0.01)),
        cyclic_wiener_projection_strength=float(getattr(config, 'cyclic_wiener_projection_strength', 0.5)),
        cyclic_wiener_zero_init=bool(getattr(config, 'cyclic_wiener_zero_init', True)),
    ).to(device)


def _create_iqumamba_qam_lattice_prior_model(config):
    from models.IQUMamba1D_ModulationPriors import IQUMamba1D_QAMLatticePrior

    return IQUMamba1D_QAMLatticePrior(
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
        qam_prior_hidden_channels=int(getattr(config, 'qam_prior_hidden_channels', 8)),
        qam_prior_axis_level_bank=tuple(getattr(config, 'qam_prior_axis_level_bank', (4, 8, 12, 16))),
        qam_prior_temperature=float(getattr(config, 'qam_prior_temperature', 24.0)),
        qam_prior_kernel_size=int(getattr(config, 'qam_prior_kernel_size', 9)),
        qam_prior_scale_init=float(getattr(config, 'qam_prior_scale_init', 0.01)),
        qam_prior_reliability_floor=float(getattr(config, 'qam_prior_reliability_floor', 0.05)),
        qam_prior_zero_init=bool(getattr(config, 'qam_prior_zero_init', True)),
    ).to(device)


def _create_iqumamba_apsk_ring_prior_model(config):
    from models.IQUMamba1D_ModulationPriors import IQUMamba1D_APSKRingPrior

    return IQUMamba1D_APSKRingPrior(
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
        apsk_prior_hidden_channels=int(getattr(config, 'apsk_prior_hidden_channels', 8)),
        apsk_prior_ring_radii=tuple(getattr(config, 'apsk_prior_ring_radii', (0.40, 1.13))),
        apsk_prior_temperature=float(getattr(config, 'apsk_prior_temperature', 18.0)),
        apsk_prior_kernel_size=int(getattr(config, 'apsk_prior_kernel_size', 9)),
        apsk_prior_scale_init=float(getattr(config, 'apsk_prior_scale_init', 0.01)),
        apsk_prior_reliability_floor=float(getattr(config, 'apsk_prior_reliability_floor', 0.05)),
        apsk_prior_zero_init=bool(getattr(config, 'apsk_prior_zero_init', True)),
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


def _create_iqumamba_multiview_consistent_model(config):
    from models.IQUMamba1D_MultiViewConsistent import IQUMamba1D_MultiViewConsistent

    return IQUMamba1D_MultiViewConsistent(
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
        iq_power_norm_eps=float(getattr(config, 'iq_power_norm_eps', 1e-6)),
        iq_power_norm_detach_scale=bool(getattr(config, 'iq_power_norm_detach_scale', False)),
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


def _create_iqumamba_shared_iq_norm_model(config):
    from models.IQUMamba1D_MultiViewConsistent import IQUMamba1D_SharedIQNorm

    return IQUMamba1D_SharedIQNorm(
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
        iq_power_norm_eps=float(getattr(config, 'iq_power_norm_eps', 1e-6)),
        iq_power_norm_detach_scale=bool(getattr(config, 'iq_power_norm_detach_scale', False)),
    ).to(device)


def _create_iqumamba_low_snr_se_model(config):
    from models.IQUMamba1D_LowSNRSE import IQUMamba1D_LowSNRSE

    return IQUMamba1D_LowSNRSE(
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
        low_snr_se_hidden_channels=int(getattr(config, 'low_snr_se_hidden_channels', 24)),
        low_snr_se_kernel_size=int(getattr(config, 'low_snr_se_kernel_size', 9)),
        low_snr_se_scale_init=float(getattr(config, 'low_snr_se_scale_init', 0.01)),
        low_snr_se_zero_init=bool(getattr(config, 'low_snr_se_zero_init', True)),
        low_snr_se_use_projection=bool(getattr(config, 'low_snr_se_use_projection', True)),
        low_snr_se_project_during_train=bool(getattr(config, 'low_snr_se_project_during_train', True)),
        low_snr_se_project_during_eval=bool(getattr(config, 'low_snr_se_project_during_eval', True)),
        low_snr_se_source_weight=float(getattr(config, 'low_snr_se_source_weight', 0.25)),
        low_snr_se_noise_weight=float(getattr(config, 'low_snr_se_noise_weight', 1.0)),
        low_snr_se_return_aux=bool(getattr(config, 'low_snr_se_return_aux', False)),
        low_snr_se_eps=float(getattr(config, 'low_snr_se_eps', 1e-8)),
    ).to(device)


def _low_snr_conditioned_kwargs(config):
    return dict(
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
        low_snr_cond_hidden_channels=int(getattr(config, 'low_snr_cond_hidden_channels', 24)),
        low_snr_cond_kernel_size=int(getattr(config, 'low_snr_cond_kernel_size', 9)),
        low_snr_cond_gate_hidden=int(getattr(config, 'low_snr_cond_gate_hidden', 12)),
        low_snr_cond_scale_init=float(getattr(config, 'low_snr_cond_scale_init', 0.01)),
        low_snr_cond_zero_init=bool(getattr(config, 'low_snr_cond_zero_init', True)),
        low_snr_cond_use_projection=bool(getattr(config, 'low_snr_cond_use_projection', True)),
        low_snr_cond_project_during_train=bool(getattr(config, 'low_snr_cond_project_during_train', False)),
        low_snr_cond_project_during_eval=bool(getattr(config, 'low_snr_cond_project_during_eval', True)),
        low_snr_cond_source_weight=float(getattr(config, 'low_snr_cond_source_weight', 0.25)),
        low_snr_cond_noise_weight=float(getattr(config, 'low_snr_cond_noise_weight', 1.0)),
        low_snr_cond_return_aux=bool(getattr(config, 'low_snr_cond_return_aux', False)),
        low_snr_cond_eps=float(getattr(config, 'low_snr_cond_eps', 1e-8)),
    )


def _create_iqumamba_low_snr_snr_cond_model(config):
    from models.IQUMamba1D_LowSNRConditioned import IQUMamba1D_LowSNRSNRConditioned

    return IQUMamba1D_LowSNRSNRConditioned(**_low_snr_conditioned_kwargs(config)).to(device)


def _create_iqumamba_low_snr_cyclic_cond_model(config):
    from models.IQUMamba1D_LowSNRConditioned import IQUMamba1D_LowSNRCyclicConditioned

    kwargs = _low_snr_conditioned_kwargs(config)
    kwargs.update(
        min_freq=float(getattr(config, 'low_snr_cond_min_freq', 0.01)),
        max_freq=float(getattr(config, 'low_snr_cond_max_freq', 0.45)),
    )
    return IQUMamba1D_LowSNRCyclicConditioned(**kwargs).to(device)


def _create_iqumamba_neural_wiener_se_model(config):
    from models.IQUMamba1D_NeuralWienerSE import IQUMamba1D_NeuralWienerSE

    return IQUMamba1D_NeuralWienerSE(
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
        wiener_hidden_channels=int(getattr(config, 'wiener_hidden_channels', 16)),
        wiener_kernel_size=int(getattr(config, 'wiener_kernel_size', 9)),
        wiener_signal_bias_init=float(getattr(config, 'wiener_signal_bias_init', 3.0)),
        wiener_noise_bias_init=float(getattr(config, 'wiener_noise_bias_init', -3.0)),
        wiener_log_power_clip=float(getattr(config, 'wiener_log_power_clip', 8.0)),
        wiener_use_projection=bool(getattr(config, 'wiener_use_projection', True)),
        wiener_project_during_train=bool(getattr(config, 'wiener_project_during_train', False)),
        wiener_project_during_eval=bool(getattr(config, 'wiener_project_during_eval', True)),
        wiener_source_weight=float(getattr(config, 'wiener_source_weight', 0.25)),
        wiener_noise_weight=float(getattr(config, 'wiener_noise_weight', 1.0)),
        wiener_return_aux=bool(getattr(config, 'wiener_return_aux', False)),
        wiener_eps=float(getattr(config, 'wiener_eps', 1e-8)),
    ).to(device)


def _create_iqumamba_asg_mamba_model(config):
    from models.IQUMamba1D_ASGMamba import IQUMamba1D_ASGMamba

    return IQUMamba1D_ASGMamba(
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
        asg_patch_size=int(getattr(config, 'asg_patch_size', 32)),
        asg_stride=int(getattr(config, 'asg_stride', 16)),
        asg_num_bands=int(getattr(config, 'asg_num_bands', 3)),
        asg_gate_hidden=int(getattr(config, 'asg_gate_hidden', 8)),
        asg_scale_init=float(getattr(config, 'asg_scale_init', 0.01)),
        asg_zero_init=bool(getattr(config, 'asg_zero_init', True)),
        asg_apply_stages=tuple(getattr(config, 'asg_apply_stages', (1, 3))),
        asg_eps=float(getattr(config, 'asg_eps', 1e-8)),
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
        complex_stem_enable=bool(getattr(config, 'complex_stem_enable', False)),
        complex_norm_eps=float(getattr(config, 'complex_norm_eps', 1e-6)),
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


def _create_iqumamba_multihyp_cyclic_reliability_model(config):
    from models.IQUMamba1D_MultiHypCyclicReliability import IQUMamba1D_MultiHypCyclicReliability

    return IQUMamba1D_MultiHypCyclicReliability(
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
        multihyp_cyclic_freqs=tuple(getattr(config, 'multihyp_cyclic_freqs', (0.015625, 0.03125, 0.0625, 0.125))),
        multihyp_cyclic_hidden_channels=int(getattr(config, 'multihyp_cyclic_hidden_channels', 8)),
        multihyp_cyclic_kernel_size=int(getattr(config, 'multihyp_cyclic_kernel_size', 9)),
        multihyp_cyclic_scale_init=float(getattr(config, 'multihyp_cyclic_scale_init', 0.01)),
        multihyp_cyclic_gate_hidden=int(getattr(config, 'multihyp_cyclic_gate_hidden', 8)),
        multihyp_cyclic_temperature=float(getattr(config, 'multihyp_cyclic_temperature', 0.5)),
        multihyp_cyclic_null_logit_init=float(getattr(config, 'multihyp_cyclic_null_logit_init', 2.0)),
        multihyp_cyclic_local_bins=int(getattr(config, 'multihyp_cyclic_local_bins', 5)),
        multihyp_cyclic_zero_init=bool(getattr(config, 'multihyp_cyclic_zero_init', True)),
        multihyp_cyclic_return_aux=bool(getattr(config, 'multihyp_cyclic_return_aux', False)),
        multihyp_cyclic_eps=float(getattr(config, 'multihyp_cyclic_eps', 1e-8)),
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


def _create_bimamba_core_upgrade_model(config):
    from models.IQUBiMamba1D_CoreUpgrades import (
        IQUBiMamba1D_ComplexState,
        IQUBiMamba1D_Hydra,
        IQUBiMamba1D_IndependentComplexState,
        IQUBiMamba1D_IndependentComplexStateUniRepLK,
        IQUBiMamba1D_MultiScale,
    )
    from models.IQUBiMamba1D_LatentMask import IQUBiMamba1D_ComplexLatentMask
    from models.IQUBiMamba1D_BottleneckMask import IQUBiMamba1D_BottleneckRealMask

    model_classes = {
        "bimamba_hydra": IQUBiMamba1D_Hydra,
        "bimamba_complex_state": IQUBiMamba1D_ComplexState,
        "bimamba_complex_state_independent": IQUBiMamba1D_IndependentComplexState,
        "bimamba_complex_state_independent_unireplk": IQUBiMamba1D_IndependentComplexStateUniRepLK,
        "bimamba_complex_latent_mask_real": IQUBiMamba1D_ComplexLatentMask,
        "bimamba_complex_latent_mask_ratio": IQUBiMamba1D_ComplexLatentMask,
        "bimamba_complex_latent_mask_residual": IQUBiMamba1D_ComplexLatentMask,
        "bimamba_complex_latent_mask_conservation": IQUBiMamba1D_ComplexLatentMask,
        "bimamba_bottleneck_mask_real": IQUBiMamba1D_BottleneckRealMask,
        "bimamba_multiscale": IQUBiMamba1D_MultiScale,
    }
    cfg = config.model_config
    common = dict(
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
        bimamba_apply_stages=tuple(cfg.get("bimamba_apply_stages", (1, 3))),
        bimamba_residual_scale_init=float(
            cfg.get("bimamba_residual_scale_init", 1.0)
        ),
    )
    if config.model_type == "bimamba_hydra":
        extra = dict(
            hydra_d_state=int(cfg.get("hydra_d_state", 64)),
            hydra_d_conv=int(cfg.get("hydra_d_conv", 7)),
            hydra_expand=int(cfg.get("hydra_expand", 2)),
            hydra_headdim=int(cfg.get("hydra_headdim", 64)),
            hydra_ngroups=int(cfg.get("hydra_ngroups", 1)),
            hydra_chunk_size=int(cfg.get("hydra_chunk_size", 256)),
            hydra_prefer_fused_scan=bool(
                cfg.get("hydra_prefer_fused_scan", True)
            ),
        )
    elif config.model_type in {
        "bimamba_complex_state",
        "bimamba_complex_state_independent",
        "bimamba_complex_state_independent_unireplk",
        "bimamba_complex_latent_mask_real",
        "bimamba_complex_latent_mask_ratio",
        "bimamba_complex_latent_mask_residual",
        "bimamba_complex_latent_mask_conservation",
        "bimamba_bottleneck_mask_real",
    }:
        extra = dict(
            complex_state_d_state=int(cfg.get("complex_state_d_state", 8)),
            complex_state_d_conv=int(cfg.get("complex_state_d_conv", 4)),
            complex_state_expand=int(cfg.get("complex_state_expand", 2)),
            complex_state_scan_checkpoint=bool(
                cfg.get("complex_state_scan_checkpoint", True)
            ),
            complex_state_scan_backend=str(
                cfg.get("complex_state_scan_backend", "auto")
            ),
            complex_state_fusion_hidden=int(
                cfg.get("complex_state_fusion_hidden", 64)
            ),
        )
    else:
        extra = dict(
            multiscale_d_state=int(cfg.get("multiscale_d_state", 16)),
            multiscale_global_d_conv=int(
                cfg.get("multiscale_global_d_conv", 4)
            ),
            multiscale_expand=int(cfg.get("multiscale_expand", 2)),
            multiscale_local_kernels=tuple(
                int(value)
                for value in cfg.get("multiscale_local_kernels", (3, 7, 15))
            ),
            multiscale_local_scale_init=float(
                cfg.get("multiscale_local_scale_init", 0.1)
            ),
        )
    if config.model_type in {
        "bimamba_complex_state_independent_unireplk",
        "bimamba_complex_latent_mask_real",
        "bimamba_complex_latent_mask_ratio",
        "bimamba_complex_latent_mask_residual",
        "bimamba_complex_latent_mask_conservation",
        "bimamba_bottleneck_mask_real",
    }:
        extra.update(
            rf_apply_stages=tuple(
                int(stage)
                for stage in cfg.get("rf_apply_stages", (0, 1, 2))
            ),
            rf_residual_scale_init=float(
                cfg.get("rf_residual_scale_init", 0.05)
            ),
            rf_large_kernel=int(cfg.get("rf_large_kernel", 17)),
            rf_ffn_factor=int(cfg.get("rf_ffn_factor", 4)),
            rf_layer_scale=float(cfg.get("rf_layer_scale", 1e-6)),
        )
    if config.model_type in {
        "bimamba_complex_latent_mask_real",
        "bimamba_complex_latent_mask_ratio",
        "bimamba_complex_latent_mask_residual",
        "bimamba_complex_latent_mask_conservation",
    }:
        mode = {
            "bimamba_complex_latent_mask_real": "real",
            "bimamba_complex_latent_mask_ratio": "complex_ratio",
            "bimamba_complex_latent_mask_residual": "complex_residual",
            "bimamba_complex_latent_mask_conservation": "complex_conservation",
        }[config.model_type]
        extra.update(
            latent_mask_mode=str(cfg.get("latent_mask_mode", mode)),
            latent_mask_phase_limit=float(
                cfg.get("latent_mask_phase_limit", math.pi)
            ),
            latent_mask_eps=float(cfg.get("latent_mask_eps", 1.0e-6)),
            latent_mask_residual_weight=float(
                cfg.get("latent_mask_residual_weight", 0.1)
            ),
            latent_mask_mixture_weight=float(
                cfg.get("latent_mask_mixture_weight", 0.1)
            ),
            latent_mask_residual_beta=float(
                cfg.get("latent_mask_residual_beta", 0.5)
            ),
        )
    return model_classes[config.model_type](**common, **extra).to(device)


def _create_bimamba_cross_scale_attention_model(config):
    from models.IQUBiMamba1D_CrossScaleAttention import IQUBiMamba1D_CrossScaleAttention

    return IQUBiMamba1D_CrossScaleAttention(
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
        cross_scale_query_stages=getattr(config, 'cross_scale_query_stages', [2]),
        cross_scale_global_stage=int(getattr(config, 'cross_scale_global_stage', 3)),
        cross_scale_kv_tokens=int(getattr(config, 'cross_scale_kv_tokens', 64)),
        cross_scale_num_heads=int(getattr(config, 'cross_scale_num_heads', 4)),
        cross_scale_dropout=float(getattr(config, 'cross_scale_dropout', 0.0)),
        cross_scale_residual_scale_init=float(
            getattr(config, 'cross_scale_residual_scale_init', 0.01)
        ),
        cross_scale_evidence_gate=bool(getattr(config, 'cross_scale_evidence_gate', False)),
        cross_scale_evidence_hidden=int(getattr(config, 'cross_scale_evidence_hidden', 32)),
        cross_scale_evidence_eps=float(getattr(config, 'cross_scale_evidence_eps', 1e-6)),
    ).to(device)


def _create_iqumamba_cross_scale_attention_model(config):
    from models.IQUMamba1D_CrossScaleAttention import (
        IQUMamba1D_CrossScaleAttention,
    )

    return IQUMamba1D_CrossScaleAttention(
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
        cross_scale_query_stages=getattr(
            config, "cross_scale_query_stages", [2]
        ),
        cross_scale_global_stage=int(
            getattr(config, "cross_scale_global_stage", 3)
        ),
        cross_scale_kv_tokens=int(
            getattr(config, "cross_scale_kv_tokens", 64)
        ),
        cross_scale_num_heads=int(
            getattr(config, "cross_scale_num_heads", 4)
        ),
        cross_scale_dropout=float(
            getattr(config, "cross_scale_dropout", 0.0)
        ),
        cross_scale_residual_scale_init=float(
            getattr(config, "cross_scale_residual_scale_init", 0.01)
        ),
        cross_scale_evidence_gate=bool(
            getattr(config, "cross_scale_evidence_gate", False)
        ),
        cross_scale_evidence_hidden=int(
            getattr(config, "cross_scale_evidence_hidden", 32)
        ),
        cross_scale_evidence_eps=float(
            getattr(config, "cross_scale_evidence_eps", 1e-6)
        ),
    ).to(device)


def _create_unireplk_cross_scale_model(config):
    from models.IQUMamba1D_UniRepLKCrossScale import (
        IQUBiMamba1D_UniRepLKCrossScale,
        IQUMamba1D_UniRepLKCrossScale,
    )

    model_class = (
        IQUBiMamba1D_UniRepLKCrossScale
        if config.model_type == "bimamba_cross_scale_unireplk"
        else IQUMamba1D_UniRepLKCrossScale
    )
    cfg = config.model_config
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
        cross_scale_query_stages=getattr(
            config, "cross_scale_query_stages", [2]
        ),
        cross_scale_global_stage=int(
            getattr(config, "cross_scale_global_stage", 3)
        ),
        cross_scale_kv_tokens=int(
            getattr(config, "cross_scale_kv_tokens", 64)
        ),
        cross_scale_num_heads=int(
            getattr(config, "cross_scale_num_heads", 4)
        ),
        cross_scale_dropout=float(
            getattr(config, "cross_scale_dropout", 0.0)
        ),
        cross_scale_residual_scale_init=float(
            getattr(config, "cross_scale_residual_scale_init", 0.01)
        ),
        cross_scale_evidence_gate=bool(
            getattr(config, "cross_scale_evidence_gate", False)
        ),
        cross_scale_evidence_hidden=int(
            getattr(config, "cross_scale_evidence_hidden", 32)
        ),
        cross_scale_evidence_eps=float(
            getattr(config, "cross_scale_evidence_eps", 1e-6)
        ),
        rf_apply_stages=tuple(int(stage) for stage in cfg.get(
            "rf_apply_stages", (0, 1, 2)
        )),
        rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
        rf_large_kernel=int(cfg.get("rf_large_kernel", 17)),
        rf_ffn_factor=int(cfg.get("rf_ffn_factor", 4)),
        rf_layer_scale=float(cfg.get("rf_layer_scale", 1e-6)),
    ).to(device)


def _create_bimamba_cross_scale_estimated_cyclofresh_model(config):
    from models.IQUBiMamba1D_CrossScaleEstimatedCycloFRESH import (
        IQUBiMamba1D_CrossScaleEstimatedCycloFRESH,
    )

    return IQUBiMamba1D_CrossScaleEstimatedCycloFRESH(
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
        cross_scale_query_stages=getattr(config, 'cross_scale_query_stages', [2]),
        cross_scale_global_stage=int(getattr(config, 'cross_scale_global_stage', 3)),
        cross_scale_kv_tokens=int(getattr(config, 'cross_scale_kv_tokens', 64)),
        cross_scale_num_heads=int(getattr(config, 'cross_scale_num_heads', 4)),
        cross_scale_dropout=float(getattr(config, 'cross_scale_dropout', 0.0)),
        cross_scale_residual_scale_init=float(
            getattr(config, 'cross_scale_residual_scale_init', 0.01)
        ),
        estimated_cyclofresh_min_freq=float(
            getattr(config, 'estimated_cyclofresh_min_freq', 1.0 / 64.0)
        ),
        estimated_cyclofresh_max_freq=float(
            getattr(config, 'estimated_cyclofresh_max_freq', 1.0 / 8.0)
        ),
        estimated_cyclofresh_default_freq=float(
            getattr(config, 'estimated_cyclofresh_default_freq', 1.0 / 32.0)
        ),
        estimated_cyclofresh_momentum=float(
            getattr(config, 'estimated_cyclofresh_momentum', 0.05)
        ),
        estimated_cyclofresh_hidden_channels=int(
            getattr(config, 'estimated_cyclofresh_hidden_channels', 8)
        ),
        estimated_cyclofresh_kernel_size=int(
            getattr(config, 'estimated_cyclofresh_kernel_size', 9)
        ),
        estimated_cyclofresh_scale_init=float(
            getattr(config, 'estimated_cyclofresh_scale_init', 0.01)
        ),
        estimated_cyclofresh_gate_hidden=int(
            getattr(config, 'estimated_cyclofresh_gate_hidden', 8)
        ),
        estimated_cyclofresh_zero_init=bool(
            getattr(config, 'estimated_cyclofresh_zero_init', True)
        ),
    ).to(device)


def _create_bimamba_advanced_cross_scale_attention_model(config):
    from models.IQUBiMamba1D_CrossScaleAdvanced import (
        IQUBiMamba1D_AdvancedCrossScaleAttention,
    )

    return IQUBiMamba1D_AdvancedCrossScaleAttention(
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
        cross_scale_variant=str(getattr(config, 'cross_scale_variant', 'aligned')),
        cross_scale_query_stages=getattr(config, 'cross_scale_query_stages', [2]),
        cross_scale_global_stage=int(getattr(config, 'cross_scale_global_stage', 3)),
        cross_scale_kv_tokens=int(getattr(config, 'cross_scale_kv_tokens', 64)),
        cross_scale_num_heads=int(getattr(config, 'cross_scale_num_heads', 4)),
        cross_scale_dropout=float(getattr(config, 'cross_scale_dropout', 0.0)),
        cross_scale_residual_scale_init=float(
            getattr(config, 'cross_scale_residual_scale_init', 0.01)
        ),
        cross_scale_aligned_window_radius=int(
            getattr(config, 'cross_scale_aligned_window_radius', 4)
        ),
        cross_scale_aligned_global_tokens=int(
            getattr(config, 'cross_scale_aligned_global_tokens', 1)
        ),
        cross_scale_coarse_kv_tokens=int(
            getattr(config, 'cross_scale_coarse_kv_tokens', 32)
        ),
        cross_scale_fine_kv_tokens=int(
            getattr(config, 'cross_scale_fine_kv_tokens', 128)
        ),
        cross_scale_multires_gate_hidden=int(
            getattr(config, 'cross_scale_multires_gate_hidden', 32)
        ),
        cross_scale_bounded_max_scale=float(
            getattr(config, 'cross_scale_bounded_max_scale', 0.1)
        ),
        cross_scale_bounded_initial_scale=float(
            getattr(config, 'cross_scale_bounded_initial_scale', 0.01)
        ),
        cross_scale_channel_gate_hidden=int(
            getattr(config, 'cross_scale_channel_gate_hidden', 64)
        ),
    ).to(device)


def _create_bimamba_kv_attention_ablation_model(config):
    from models.IQUBiMamba1D_KVAttentionAblations import (
        IQUBiMamba1D_BottleneckSelfAttention,
        IQUBiMamba1D_HymbaParallel,
        IQUBiMamba1D_PhaseEquivariantFusion,
        IQUBiMamba1D_PhysicalTokenCrossAttention,
        IQUBiMamba1D_RFPhysicalKVCrossAttention,
    )

    common = {
        "input_size": input_size,
        "input_channels": config.input_channels,
        "n_stages": config.n_stages,
        "features_per_stage": config.features_per_stage,
        "conv_op": nn.Conv1d,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "n_conv_per_stage": config.n_conv_per_stage,
        "num_classes": config.num_classes,
        "n_conv_per_stage_decoder": config.n_conv_per_stage_decoder,
        "deep_supervision": config.deep_supervision,
    }
    model_type = str(config.model_type)
    if model_type == "bimamba_phase_equivariant_fusion":
        model = IQUBiMamba1D_PhaseEquivariantFusion(
            **common,
            phase_fusion_kv_tokens=int(getattr(config, "phase_fusion_kv_tokens", 32)),
            phase_fusion_num_heads=int(getattr(config, "phase_fusion_num_heads", 4)),
            phase_fusion_head_dim=int(getattr(config, "phase_fusion_head_dim", 8)),
            phase_fusion_dropout=float(getattr(config, "phase_fusion_dropout", 0.0)),
            phase_fusion_scale_init=float(getattr(config, "phase_fusion_scale_init", 0.01)),
        )
    elif model_type == "bimamba_physical_token_cross_attention":
        model = IQUBiMamba1D_PhysicalTokenCrossAttention(
            **common,
            physical_query_stage=int(getattr(config, "physical_query_stage", 2)),
            physical_num_heads=int(getattr(config, "physical_num_heads", 4)),
            physical_dropout=float(getattr(config, "physical_dropout", 0.0)),
            physical_residual_scale_init=float(
                getattr(config, "physical_residual_scale_init", 0.01)
            ),
            physical_cyclic_lags=getattr(config, "physical_cyclic_lags", [0, 1, 2, 4, 8]),
            physical_polyphase_branches=int(
                getattr(config, "physical_polyphase_branches", 8)
            ),
            physical_symbol_orders=getattr(config, "physical_symbol_orders", [2, 4, 8]),
            physical_min_cyclic_freq=float(
                getattr(config, "physical_min_cyclic_freq", 1.0 / 64.0)
            ),
            physical_max_cyclic_freq=float(
                getattr(config, "physical_max_cyclic_freq", 1.0 / 8.0)
            ),
            physical_cyclic_temperature=float(
                getattr(config, "physical_cyclic_temperature", 0.25)
            ),
        )
    elif model_type == "bimamba_bottleneck_self_attention":
        model = IQUBiMamba1D_BottleneckSelfAttention(
            **common,
            bottleneck_attention_stage=int(
                getattr(config, "bottleneck_attention_stage", 3)
            ),
            bottleneck_attention_num_heads=int(
                getattr(config, "bottleneck_attention_num_heads", 4)
            ),
            bottleneck_attention_dropout=float(
                getattr(config, "bottleneck_attention_dropout", 0.0)
            ),
            bottleneck_attention_scale_init=float(
                getattr(config, "bottleneck_attention_scale_init", 1.0)
            ),
        )
    elif model_type == "bimamba_hymba_parallel":
        model = IQUBiMamba1D_HymbaParallel(
            **common,
            hymba_stage=int(getattr(config, "hymba_stage", 3)),
            hymba_num_heads=int(getattr(config, "hymba_num_heads", 4)),
            hymba_dropout=float(getattr(config, "hymba_dropout", 0.0)),
            hymba_mamba_scale_init=float(getattr(config, "hymba_mamba_scale_init", 1.0)),
            hymba_attention_scale_init=float(
                getattr(config, "hymba_attention_scale_init", 0.01)
            ),
            hymba_attention_scale_max=float(
                getattr(config, "hymba_attention_scale_max", 1.0)
            ),
        )
    elif model_type == "bimamba_rf_physical_kv":
        model = IQUBiMamba1D_RFPhysicalKVCrossAttention(
            **common,
            rf_physical_query_stage=int(getattr(config, "rf_physical_query_stage", 2)),
            rf_physical_num_heads=int(getattr(config, "rf_physical_num_heads", 4)),
            rf_physical_dropout=float(getattr(config, "rf_physical_dropout", 0.0)),
            rf_physical_residual_scale_init=float(
                getattr(config, "rf_physical_residual_scale_init", 0.01)
            ),
            rf_physical_stft_n_fft=int(getattr(config, "rf_physical_stft_n_fft", 256)),
            rf_physical_stft_hop_length=int(
                getattr(config, "rf_physical_stft_hop_length", 64)
            ),
            rf_physical_stft_win_length=int(
                getattr(config, "rf_physical_stft_win_length", 256)
            ),
            rf_physical_num_subbands=int(getattr(config, "rf_physical_num_subbands", 8)),
            rf_physical_temporal_tokens=int(
                getattr(config, "rf_physical_temporal_tokens", 4)
            ),
            rf_physical_cyclic_lags=getattr(
                config, "rf_physical_cyclic_lags", [0, 1, 2, 4, 8]
            ),
            rf_physical_polyphase_branches=int(
                getattr(config, "rf_physical_polyphase_branches", 8)
            ),
            rf_physical_symbol_orders=getattr(
                config, "rf_physical_symbol_orders", [2, 4, 8]
            ),
            rf_physical_min_cyclic_freq=float(
                getattr(config, "rf_physical_min_cyclic_freq", 1.0 / 64.0)
            ),
            rf_physical_max_cyclic_freq=float(
                getattr(config, "rf_physical_max_cyclic_freq", 1.0 / 8.0)
            ),
            rf_physical_cyclic_temperature=float(
                getattr(config, "rf_physical_cyclic_temperature", 0.25)
            ),
        )
    else:
        raise ValueError(f"Unsupported Stage-12 Mamba/KV attention model type: {model_type}")
    return model.to(device)


def _create_bimamba_hierarchical_kv_fusion_model(config):
    from models.IQUBiMamba1D_HierarchicalKVFusion import (
        IQUBiMamba1D_DualMemoryCrossAttention,
        IQUBiMamba1D_EnhancedGlobalCrossAttention,
        IQUBiMamba1D_HierarchicalAdditiveFusion,
        IQUBiMamba1D_PhysicalRoutedEnhancedCrossAttention,
        IQUBiMamba1D_UnifiedPhysicalGlobalKV,
        IQUBiMamba1D_PhysicalFiLMGlobalMemory,
        IQUBiMamba1D_ScaleIsolatedPhysicalFusion,
        IQUBiMamba1D_IdentityAwarePhysicalMoE,
        IQUBiMamba1D_CrossGatedDualMemory,
    )

    common = {
        "input_size": input_size,
        "input_channels": config.input_channels,
        "n_stages": config.n_stages,
        "features_per_stage": config.features_per_stage,
        "conv_op": nn.Conv1d,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "n_conv_per_stage": config.n_conv_per_stage,
        "num_classes": config.num_classes,
        "n_conv_per_stage_decoder": config.n_conv_per_stage_decoder,
        "deep_supervision": config.deep_supervision,
    }
    fusion = {
        "fusion_query_stage": int(getattr(config, "fusion_query_stage", 2)),
        "fusion_global_stage": int(getattr(config, "fusion_global_stage", 3)),
        "fusion_global_kv_tokens": int(getattr(config, "fusion_global_kv_tokens", 64)),
        "fusion_num_heads": int(getattr(config, "fusion_num_heads", 4)),
        "fusion_dropout": float(getattr(config, "fusion_dropout", 0.0)),
    }
    physical = {
        "physical_cyclic_lags": getattr(config, "physical_cyclic_lags", [0, 1, 2, 4, 8]),
        "physical_polyphase_branches": int(
            getattr(config, "physical_polyphase_branches", 8)
        ),
        "physical_symbol_orders": getattr(config, "physical_symbol_orders", [2, 4, 8]),
        "physical_min_cyclic_freq": float(
            getattr(config, "physical_min_cyclic_freq", 1.0 / 64.0)
        ),
        "physical_max_cyclic_freq": float(
            getattr(config, "physical_max_cyclic_freq", 1.0 / 8.0)
        ),
        "physical_cyclic_temperature": float(
            getattr(config, "physical_cyclic_temperature", 0.25)
        ),
    }
    model_type = str(config.model_type)
    if model_type == "bimamba_enhanced_global_cross_attention":
        model = IQUBiMamba1D_EnhancedGlobalCrossAttention(
            **common,
            **fusion,
            fusion_global_scale_init=float(
                getattr(config, "fusion_global_scale_init", 0.01)
            ),
            fusion_bottleneck_num_heads=int(
                getattr(config, "fusion_bottleneck_num_heads", 4)
            ),
            fusion_bottleneck_dropout=float(
                getattr(config, "fusion_bottleneck_dropout", 0.0)
            ),
            fusion_bottleneck_scale_init=float(
                getattr(config, "fusion_bottleneck_scale_init", 1.0)
            ),
        )
    elif model_type == "bimamba_dual_memory_cross_attention":
        model = IQUBiMamba1D_DualMemoryCrossAttention(
            **common,
            **fusion,
            **physical,
            fusion_global_scale_init=float(
                getattr(config, "fusion_global_scale_init", 0.01)
            ),
            fusion_physical_scale_init=float(
                getattr(config, "fusion_physical_scale_init", 0.01)
            ),
        )
    elif model_type in {
        "bimamba_hierarchical_additive_fusion",
        "bimamba_hierarchical_routed_fusion",
    }:
        model = IQUBiMamba1D_HierarchicalAdditiveFusion(
            **common,
            **fusion,
            **physical,
            fusion_global_scale_init=float(
                getattr(config, "fusion_global_scale_init", 0.01)
            ),
            fusion_physical_scale_init=float(
                getattr(config, "fusion_physical_scale_init", 0.01)
            ),
            fusion_bottleneck_num_heads=int(
                getattr(config, "fusion_bottleneck_num_heads", 4)
            ),
            fusion_bottleneck_dropout=float(
                getattr(config, "fusion_bottleneck_dropout", 0.0)
            ),
            fusion_bottleneck_scale_init=float(
                getattr(config, "fusion_bottleneck_scale_init", 1.0)
            ),
        )
    elif model_type == "bimamba_physical_routed_enhanced_cross_attention":
        model = IQUBiMamba1D_PhysicalRoutedEnhancedCrossAttention(
            **common,
            **fusion,
            **physical,
            fusion_channel_scale_init=float(
                getattr(config, "fusion_channel_scale_init", 0.1)
            ),
            fusion_channel_scale_max=float(
                getattr(config, "fusion_channel_scale_max", 0.5)
            ),
            fusion_bottleneck_num_heads=int(
                getattr(config, "fusion_bottleneck_num_heads", 4)
            ),
            fusion_bottleneck_dropout=float(
                getattr(config, "fusion_bottleneck_dropout", 0.0)
            ),
            fusion_bottleneck_scale_init=float(
                getattr(config, "fusion_bottleneck_scale_init", 0.1)
            ),
            fusion_router_hidden=int(getattr(config, "fusion_router_hidden", 64)),
            fusion_router_gate_init=float(
                getattr(config, "fusion_router_gate_init", 1.0)
            ),
            fusion_router_gate_max=float(
                getattr(config, "fusion_router_gate_max", 2.0)
            ),
        )
    elif model_type in {
        "bimamba_unified_physical_global_kv",
        "bimamba_physical_film_global_memory",
        "bimamba_scale_isolated_physical_fusion",
        "bimamba_identity_aware_physical_moe",
        "bimamba_cross_gated_dual_memory",
    }:
        model_classes = {
            "bimamba_unified_physical_global_kv": IQUBiMamba1D_UnifiedPhysicalGlobalKV,
            "bimamba_physical_film_global_memory": IQUBiMamba1D_PhysicalFiLMGlobalMemory,
            "bimamba_scale_isolated_physical_fusion": IQUBiMamba1D_ScaleIsolatedPhysicalFusion,
            "bimamba_identity_aware_physical_moe": IQUBiMamba1D_IdentityAwarePhysicalMoE,
            "bimamba_cross_gated_dual_memory": IQUBiMamba1D_CrossGatedDualMemory,
        }
        extra = {}
        if model_type == "bimamba_physical_film_global_memory":
            extra.update(
                fusion_film_hidden=int(getattr(config, "fusion_film_hidden", 64)),
                fusion_film_max_delta=float(
                    getattr(config, "fusion_film_max_delta", 0.1)
                ),
            )
        elif model_type == "bimamba_scale_isolated_physical_fusion":
            extra.update(
                fusion_physical_stage=int(getattr(config, "fusion_physical_stage", 1)),
                fusion_physical_film_hidden=int(
                    getattr(config, "fusion_physical_film_hidden", 64)
                ),
                fusion_physical_film_max_delta=float(
                    getattr(config, "fusion_physical_film_max_delta", 0.1)
                ),
            )
        elif model_type == "bimamba_identity_aware_physical_moe":
            extra.update(
                fusion_router_hidden=int(getattr(config, "fusion_router_hidden", 64)),
                fusion_expert_prior=getattr(
                    config, "fusion_expert_prior", [0.7, 0.1, 0.1, 0.1]
                ),
                fusion_condition_hidden=int(
                    getattr(config, "fusion_condition_hidden", 16)
                ),
                fusion_condition_embedding=int(
                    getattr(config, "fusion_condition_embedding", 16)
                ),
                fusion_trust_penalty_init=float(
                    getattr(config, "fusion_trust_penalty_init", 0.1)
                ),
                fusion_trust_penalty_enable=bool(
                    getattr(config, "fusion_trust_penalty_enable", False)
                ),
                fusion_condition_routing_enable=bool(
                    getattr(config, "fusion_condition_routing_enable", False)
                ),
                fusion_counterfactual_enable=bool(
                    getattr(config, "fusion_counterfactual_enable", False)
                ),
                fusion_return_route_aux=bool(
                    getattr(config, "fusion_return_route_aux", False)
                ),
                fusion_route_candidate_probability=float(
                    getattr(config, "fusion_route_candidate_probability", 1.0)
                ),
            )
        elif model_type == "bimamba_cross_gated_dual_memory":
            extra.update(
                fusion_router_hidden=int(getattr(config, "fusion_router_hidden", 64))
            )
        model = model_classes[model_type](
            **common,
            **fusion,
            **physical,
            fusion_channel_scale_init=float(
                getattr(config, "fusion_channel_scale_init", 0.1)
            ),
            fusion_channel_scale_max=float(
                getattr(config, "fusion_channel_scale_max", 0.5)
            ),
            fusion_bottleneck_num_heads=int(
                getattr(config, "fusion_bottleneck_num_heads", 4)
            ),
            fusion_bottleneck_dropout=float(
                getattr(config, "fusion_bottleneck_dropout", 0.0)
            ),
            fusion_bottleneck_scale_init=float(
                getattr(config, "fusion_bottleneck_scale_init", 0.1)
            ),
            **extra,
        )
    else:
        raise ValueError(f"Unsupported hierarchical KV fusion model type: {model_type}")
    return model.to(device)


def _create_bimamba_safe_allstage_model(config):
    from models.IQUBiMamba1D_SafeAllStages import IQUBiMamba1D_SafeAllStages

    return IQUBiMamba1D_SafeAllStages(
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
        bimamba_apply_stages=getattr(config, 'bimamba_apply_stages', None),
        bimamba_residual_scale_init=float(getattr(config, 'bimamba_residual_scale_init', 0.01)),
    ).to(device)


def _create_bimamba_direction_gated_model(config):
    from models.IQUBiMamba1D_DirectionGated import IQUBiMamba1D_DirectionGated

    return IQUBiMamba1D_DirectionGated(
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
        bimamba_apply_stages=getattr(config, 'bimamba_apply_stages', None),
        bimamba_residual_scale_init=float(getattr(config, 'bimamba_residual_scale_init', 0.01)),
    ).to(device)


def _create_bimamba_local_global_allstage_model(config):
    from models.IQUBiMamba1D_LocalGlobalAllStages import IQUBiMamba1D_LocalGlobalAllStages

    return IQUBiMamba1D_LocalGlobalAllStages(
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
        bimamba_apply_stages=getattr(config, 'bimamba_apply_stages', None),
        bimamba_residual_scale_init=float(getattr(config, 'bimamba_residual_scale_init', 0.01)),
        local_kernel_size=int(getattr(config, 'local_kernel_size', 7)),
        local_global_gate_hidden=int(getattr(config, 'local_global_gate_hidden', 64)),
    ).to(device)


def _create_light_fusion_bimamba_model(config, model_class):
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
        bimamba_apply_stages=getattr(config, 'bimamba_apply_stages', None),
        bimamba_residual_scale_init=float(getattr(config, 'bimamba_residual_scale_init', 0.01)),
        bimamba_diff_scale_init=float(getattr(config, 'bimamba_diff_scale_init', 0.0)),
        bimamba_gate_logit_init=float(getattr(config, 'bimamba_gate_logit_init', -1.5)),
        bimamba_gate_token_scale_init=float(getattr(config, 'bimamba_gate_token_scale_init', 1.0)),
        bimamba_gate_eps=float(getattr(config, 'bimamba_gate_eps', 1e-6)),
    ).to(device)


def _create_bimamba_diff_fusion_model(config):
    from models.IQUBiMamba1D_LightFusion import IQUBiMamba1D_DiffFusion

    return _create_light_fusion_bimamba_model(config, IQUBiMamba1D_DiffFusion)


def _create_bimamba_adaptive_diff_fusion_model(config):
    from models.IQUBiMamba1D_LightFusion import IQUBiMamba1D_AdaptiveDiffFusion

    return _create_light_fusion_bimamba_model(config, IQUBiMamba1D_AdaptiveDiffFusion)


def _create_bimamba_complex_diff_shared_model(config):
    from models.IQUBiMamba1D_ComplexDiffShared import IQUBiMamba1D_ComplexDiffShared

    return IQUBiMamba1D_ComplexDiffShared(
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
        bimamba_apply_stages=getattr(config, 'bimamba_apply_stages', [3]),
        bimamba_residual_scale_init=float(getattr(config, 'bimamba_residual_scale_init', 0.1)),
        bimamba_complex_diff_gate_init=float(getattr(config, 'bimamba_complex_diff_gate_init', 0.2)),
        bimamba_complex_diff_stride=int(getattr(config, 'bimamba_complex_diff_stride', 2)),
        bimamba_complex_diff_eps=float(getattr(config, 'bimamba_complex_diff_eps', 1e-6)),
    ).to(device)


def _create_robust_fusion_bimamba_model(config, model_class):
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
        bimamba_apply_stages=getattr(config, 'bimamba_apply_stages', [1, 3]),
        bimamba_residual_scale_init=float(getattr(config, 'bimamba_residual_scale_init', 1.0)),
        bimamba_boundary_tau_init=float(getattr(config, 'bimamba_boundary_tau_init', 16.0)),
        bimamba_shrinkage_init=float(getattr(config, 'bimamba_shrinkage_init', 0.02)),
        bimamba_fusion_eps=float(getattr(config, 'bimamba_fusion_eps', 1e-6)),
        bimamba_local_kernel_size=int(getattr(config, 'bimamba_local_kernel_size', 5)),
        bimamba_local_gate_init=float(getattr(config, 'bimamba_local_gate_init', 0.1)),
    ).to(device)


def _create_bimamba_time_reversal_shared_model(config):
    from models.IQUBiMamba1D_RobustFusion import IQUBiMamba1D_TimeReversalShared

    return _create_robust_fusion_bimamba_model(config, IQUBiMamba1D_TimeReversalShared)


def _create_bimamba_alternating_global_local_model(config):
    from models.IQUBiMamba1D_RobustFusion import IQUBiMamba1D_AlternatingGlobalLocal

    return _create_robust_fusion_bimamba_model(config, IQUBiMamba1D_AlternatingGlobalLocal)


def _pr_unet_common_kwargs(config):
    return {
        'input_size': input_size,
        'input_channels': config.input_channels,
        'n_stages': config.n_stages,
        'features_per_stage': config.features_per_stage,
        'conv_op': nn.Conv1d,
        'kernel_sizes': config.kernel_sizes,
        'strides': config.strides,
        'n_conv_per_stage': config.n_conv_per_stage,
        'num_classes': config.num_classes,
        'n_conv_per_stage_decoder': config.n_conv_per_stage_decoder,
        'deep_supervision': config.deep_supervision,
        'training_only_deep_supervision': bool(getattr(config, 'training_only_deep_supervision', True)),
    }


def _create_iqumamba_pr_unet_model(config):
    from models.IQUMamba1D_PerfectReconstruction import IQUMamba1D_PerfectReconstruction

    return IQUMamba1D_PerfectReconstruction(**_pr_unet_common_kwargs(config)).to(device)


def _create_iqumamba_pr_restricted_skip_model(config):
    from models.IQUMamba1D_PerfectReconstruction import IQUMamba1D_RestrictedShallowSkip

    return IQUMamba1D_RestrictedShallowSkip(
        **_pr_unet_common_kwargs(config),
        shallow_skip_init=float(getattr(config, 'shallow_skip_init', 0.25)),
        shallow_skip_drop_probability=float(getattr(config, 'shallow_skip_drop_probability', 0.25)),
    ).to(device)


def _create_iqumamba_evidence_moe_model(config):
    from models.IQUMamba1D_EvidenceRoutedMoE import IQUMamba1D_EvidenceRoutedMoE

    model_cfg = config.model_config
    return IQUMamba1D_EvidenceRoutedMoE(
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
        evidence_moe_hidden_channels=int(model_cfg.get('evidence_moe_hidden_channels', 12)),
        evidence_moe_max_delta=float(model_cfg.get('evidence_moe_max_delta', 0.15)),
        evidence_moe_identity_bias=float(model_cfg.get('evidence_moe_identity_bias', 1.5)),
        evidence_moe_router_temperature=float(model_cfg.get('evidence_moe_router_temperature', 1.0)),
        evidence_moe_route_hard_eval=bool(model_cfg.get('evidence_moe_route_hard_eval', True)),
        evidence_moe_lag_bank=tuple(model_cfg.get('evidence_moe_lag_bank', (1, 2, 4, 8, 16, 32, 64, 128))),
        evidence_moe_return_route_aux=bool(model_cfg.get('evidence_moe_return_route_aux', True)),
    ).to(device)


def _create_iqumamba_adaptive_multiview_prior_model(config):
    from models.IQUMamba1D_AdaptiveMultiViewPrior import IQUMamba1D_AdaptiveMultiViewPrior

    model_cfg = config.model_config
    return IQUMamba1D_AdaptiveMultiViewPrior(
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
        adaptive_view_evidence_lags=tuple(model_cfg.get("adaptive_view_evidence_lags", (1, 2, 4, 8, 16, 32, 64))),
        adaptive_view_cyclic_lags=tuple(model_cfg.get("adaptive_view_cyclic_lags", (1, 2, 4, 8, 16, 32))),
        adaptive_view_cyclic_top_k=int(model_cfg.get("adaptive_view_cyclic_top_k", 2)),
        adaptive_view_cyclic_min_freq=float(model_cfg.get("adaptive_view_cyclic_min_freq", 1.0 / 128.0)),
        adaptive_view_cyclic_max_freq=float(model_cfg.get("adaptive_view_cyclic_max_freq", 0.25)),
        adaptive_view_pulse_rolloffs=tuple(model_cfg.get("adaptive_view_pulse_rolloffs", (0.2, 0.35, 0.5))),
        adaptive_view_pulse_kernel_size=int(model_cfg.get("adaptive_view_pulse_kernel_size", 31)),
        adaptive_view_hidden_channels=int(model_cfg.get("adaptive_view_hidden_channels", 8)),
        adaptive_view_router_hidden_channels=int(model_cfg.get("adaptive_view_router_hidden_channels", 16)),
        adaptive_view_identity_bias=float(model_cfg.get("adaptive_view_identity_bias", 2.0)),
        adaptive_view_router_temperature=float(model_cfg.get("adaptive_view_router_temperature", 1.0)),
        adaptive_view_max_delta=float(model_cfg.get("adaptive_view_max_delta", 0.15)),
        adaptive_view_max_scale=float(model_cfg.get("adaptive_view_max_scale", 0.2)),
        adaptive_view_scale_init=float(model_cfg.get("adaptive_view_scale_init", 0.01)),
    ).to(device)


def _create_iqumamba_qam_source_prior_model(config):
    from models.IQUMamba1D_QAMSourcePrior import IQUMamba1D_QAMSourcePrior

    model_cfg = config.model_config
    return IQUMamba1D_QAMSourcePrior(
        **_stage4_common_kwargs(config),
        qam_source_hidden_channels=int(model_cfg.get("qam_source_hidden_channels", 16)),
        qam_source_axis_level_bank=tuple(model_cfg.get("qam_source_axis_level_bank", (4, 8))),
        qam_source_include_cross_128=bool(model_cfg.get("qam_source_include_cross_128", True)),
        qam_source_phase_bank=tuple(model_cfg.get(
            "qam_source_phase_bank",
            (-3.141592653589793 / 8.0, -3.141592653589793 / 16.0, 0.0,
             3.141592653589793 / 16.0, 3.141592653589793 / 8.0,
             3.141592653589793 / 4.0),
        )),
        qam_source_projection_temperature=float(model_cfg.get("qam_source_projection_temperature", 24.0)),
        qam_source_route_temperature=float(model_cfg.get("qam_source_route_temperature", 10.0)),
        qam_source_null_bias=float(model_cfg.get("qam_source_null_bias", 0.0)),
        qam_source_reliability_floor=float(model_cfg.get("qam_source_reliability_floor", 0.05)),
        qam_source_max_scale=float(model_cfg.get("qam_source_max_scale", 0.15)),
        qam_source_scale_init=float(model_cfg.get("qam_source_scale_init", 0.0)),
        qam_source_max_refine=float(model_cfg.get("qam_source_max_refine", 0.10)),
        qam_source_kernel_size=int(model_cfg.get("qam_source_kernel_size", 9)),
    ).to(device)


def _create_iqumamba_qam_mma_unrolled_model(config):
    from models.IQUMamba1D_QAMIndependentPriors import IQUMamba1D_QAMMMAUnrolled

    model_cfg = config.model_config
    return IQUMamba1D_QAMMMAUnrolled(
        **_stage4_common_kwargs(config),
        qam_mma_hidden_channels=int(model_cfg.get("qam_mma_hidden_channels", 16)),
        qam_mma_axis_level_bank=tuple(model_cfg.get("qam_mma_axis_level_bank", (4, 8))),
        qam_mma_include_cross_128=bool(model_cfg.get("qam_mma_include_cross_128", True)),
        qam_mma_phase_bank=tuple(model_cfg.get("qam_mma_phase_bank", (-3.141592653589793 / 8.0, 0.0, 3.141592653589793 / 8.0, 3.141592653589793 / 4.0))),
        qam_mma_num_unroll_steps=int(model_cfg.get("qam_mma_num_unroll_steps", 3)),
        qam_mma_step_init=float(model_cfg.get("qam_mma_step_init", 0.05)),
        qam_mma_route_temperature=float(model_cfg.get("qam_mma_route_temperature", 8.0)),
        qam_mma_null_bias=float(model_cfg.get("qam_mma_null_bias", 0.0)),
        qam_mma_reliability_floor=float(model_cfg.get("qam_mma_reliability_floor", 0.05)),
        qam_mma_max_scale=float(model_cfg.get("qam_mma_max_scale", 0.12)),
        qam_mma_scale_init=float(model_cfg.get("qam_mma_scale_init", 0.005)),
        qam_mma_kernel_size=int(model_cfg.get("qam_mma_kernel_size", 7)),
    ).to(device)


def _create_iqumamba_qam_density_prior_model(config):
    from models.IQUMamba1D_QAMIndependentPriors import IQUMamba1D_QAMDensityPrior

    model_cfg = config.model_config
    return IQUMamba1D_QAMDensityPrior(
        **_stage4_common_kwargs(config),
        qam_density_hidden_channels=int(model_cfg.get("qam_density_hidden_channels", 16)),
        qam_density_axis_level_bank=tuple(model_cfg.get("qam_density_axis_level_bank", (4, 8))),
        qam_density_include_cross_128=bool(model_cfg.get("qam_density_include_cross_128", True)),
        qam_density_phase_bank=tuple(model_cfg.get("qam_density_phase_bank", (-3.141592653589793 / 8.0, 0.0, 3.141592653589793 / 8.0, 3.141592653589793 / 4.0))),
        qam_density_temperature=float(model_cfg.get("qam_density_temperature", 12.0)),
        qam_density_route_temperature=float(model_cfg.get("qam_density_route_temperature", 4.0)),
        qam_density_null_bias=float(model_cfg.get("qam_density_null_bias", 0.0)),
        qam_density_max_scale=float(model_cfg.get("qam_density_max_scale", 0.12)),
        qam_density_scale_init=float(model_cfg.get("qam_density_scale_init", 0.02)),
        qam_density_kernel_size=int(model_cfg.get("qam_density_kernel_size", 9)),
    ).to(device)


def _create_iqumamba_qam_timing_prior_model(config):
    from models.IQUMamba1D_QAMIndependentPriors import IQUMamba1D_QAMTimingPrior

    model_cfg = config.model_config
    return IQUMamba1D_QAMTimingPrior(
        **_stage4_common_kwargs(config),
        qam_timing_hidden_channels=int(model_cfg.get("qam_timing_hidden_channels", 16)),
        qam_timing_sps_candidates=tuple(model_cfg.get("qam_timing_sps_candidates", (10, 20))),
        qam_timing_rrc_rolloff=float(model_cfg.get("qam_timing_rrc_rolloff", 0.35)),
        qam_timing_rrc_span=int(model_cfg.get("qam_timing_rrc_span", 12)),
        qam_timing_axis_level_bank=tuple(model_cfg.get("qam_timing_axis_level_bank", (4, 8))),
        qam_timing_route_temperature=float(model_cfg.get("qam_timing_route_temperature", 12.0)),
        qam_timing_null_bias=float(model_cfg.get("qam_timing_null_bias", 0.0)),
        qam_timing_reliability_floor=float(model_cfg.get("qam_timing_reliability_floor", 0.05)),
        qam_timing_max_scale=float(model_cfg.get("qam_timing_max_scale", 0.12)),
        qam_timing_scale_init=float(model_cfg.get("qam_timing_scale_init", 0.02)),
        qam_timing_kernel_size=int(model_cfg.get("qam_timing_kernel_size", 9)),
    ).to(device)


def _create_iqumamba_qam_turbo_unfold_model(config):
    from models.IQUMamba1D_QAMTurboUnfold import IQUMamba1D_QAMTurboUnfold

    model_cfg = config.model_config
    return IQUMamba1D_QAMTurboUnfold(
        **_stage4_common_kwargs(config),
        qam_turbo_iterations=int(model_cfg.get("qam_turbo_iterations", 3)),
        qam_turbo_hidden_channels=int(model_cfg.get("qam_turbo_hidden_channels", 32)),
        qam_turbo_kernel_size=int(model_cfg.get("qam_turbo_kernel_size", 7)),
        qam_turbo_orders=tuple(model_cfg.get("qam_turbo_orders", (16, 64, 128))),
        qam_turbo_sps=int(model_cfg.get("qam_turbo_sps", 20)),
        qam_turbo_rrc_rolloff=float(model_cfg.get("qam_turbo_rrc_rolloff", 0.35)),
        qam_turbo_rrc_span=int(model_cfg.get("qam_turbo_rrc_span", 20)),
        qam_turbo_posterior_temperature=float(
            model_cfg.get("qam_turbo_posterior_temperature", 18.0)
        ),
        qam_turbo_route_temperature=float(
            model_cfg.get("qam_turbo_route_temperature", 10.0)
        ),
        qam_turbo_channel_taps=int(model_cfg.get("qam_turbo_channel_taps", 3)),
        qam_turbo_channel_ridge=float(model_cfg.get("qam_turbo_channel_ridge", 1e-3)),
        qam_turbo_detach_channel_solve=bool(
            model_cfg.get("qam_turbo_detach_channel_solve", True)
        ),
        qam_turbo_data_step_init=float(model_cfg.get("qam_turbo_data_step_init", 0.15)),
        qam_turbo_prior_step_init=float(model_cfg.get("qam_turbo_prior_step_init", 0.08)),
        qam_turbo_learned_step_init=float(
            model_cfg.get("qam_turbo_learned_step_init", 0.05)
        ),
        qam_turbo_eps=float(model_cfg.get("qam_turbo_eps", 1e-6)),
    ).to(device)


def _stage4_common_kwargs(config):
    return {
        'input_size': input_size,
        'input_channels': config.input_channels,
        'n_stages': config.n_stages,
        'features_per_stage': config.features_per_stage,
        'conv_op': nn.Conv1d,
        'kernel_sizes': config.kernel_sizes,
        'strides': config.strides,
        'n_conv_per_stage': config.n_conv_per_stage,
        'num_classes': config.num_classes,
        'n_conv_per_stage_decoder': config.n_conv_per_stage_decoder,
        'deep_supervision': config.deep_supervision,
    }


def _create_iqumamba_noise_contrastive_prior_model(config):
    from models.IQUMamba1D_NoiseContrastivePrior import IQUMamba1D_NoiseContrastivePrior

    model_cfg = config.model_config
    return IQUMamba1D_NoiseContrastivePrior(
        **_stage4_common_kwargs(config),
        noise_prior_hidden=int(model_cfg.get('noise_prior_hidden', 12)),
        noise_prior_embedding=int(model_cfg.get('noise_prior_embedding', 16)),
        noise_prior_patch_size=int(model_cfg.get('noise_prior_patch_size', 64)),
        noise_prior_patch_stride=int(model_cfg.get('noise_prior_patch_stride', 32)),
    ).to(device)


def _create_iqumamba_blind_sync_factorized_model(config):
    from models.IQUMamba1D_BlindSyncFactorized import IQUMamba1D_BlindSyncFactorized

    model_cfg = config.model_config
    return IQUMamba1D_BlindSyncFactorized(
        **_stage4_common_kwargs(config),
        sync_hidden=int(model_cfg.get('sync_hidden', 12)),
        sync_kernel_size=int(model_cfg.get('sync_kernel_size', 5)),
        sync_scale_init=float(model_cfg.get('sync_scale_init', 0.01)),
        sync_lags=tuple(model_cfg.get('sync_lags', (1, 2, 4, 8))),
        sync_eps=float(model_cfg.get('sync_eps', 1e-6)),
    ).to(device)


def _create_iqumamba_sync_conditioned_model(config):
    from models.IQUMamba1D_SyncConditioned import IQUMamba1D_SyncConditioned

    model_cfg = config.model_config
    return IQUMamba1D_SyncConditioned(
        **_stage4_common_kwargs(config),
        sync_hidden=int(model_cfg.get('sync_hidden', 48)),
        sync_lags=tuple(model_cfg.get('sync_lags', (1, 2, 4, 8, 16))),
        sync_sps_candidates=tuple(
            model_cfg.get('sync_sps_candidates', (8, 10, 16, 20, 32, 40))
        ),
        sync_snr_min_db=float(model_cfg.get('sync_snr_min_db', -10.0)),
        sync_snr_max_db=float(model_cfg.get('sync_snr_max_db', 30.0)),
        sync_max_cfo_cycles_per_sample=float(
            model_cfg.get('sync_max_cfo_cycles_per_sample', 0.25)
        ),
        sync_max_phase_drift_rad_per_sample=float(
            model_cfg.get('sync_max_phase_drift_rad_per_sample', 0.05)
        ),
        sync_sps_temperature=float(model_cfg.get('sync_sps_temperature', 1.0)),
        sync_film_max_delta=float(model_cfg.get('sync_film_max_delta', 0.10)),
        sync_eps=float(model_cfg.get('sync_eps', 1e-6)),
    ).to(device)


def _create_iqumamba_physical_sync_rtn_model(config):
    from models.IQUMamba1D_PhysicalSyncRTN import IQUMamba1D_PhysicalSyncRTN

    model_cfg = config.model_config
    return IQUMamba1D_PhysicalSyncRTN(
        **_stage4_common_kwargs(config),
        sync_hidden=int(model_cfg.get('sync_hidden', 64)),
        sync_lags=tuple(model_cfg.get('sync_lags', (1, 2, 4, 8, 16))),
        sync_sps_candidates=tuple(
            model_cfg.get('sync_sps_candidates', (8, 10, 14, 16, 20, 25, 32, 40))
        ),
        sync_snr_min_db=float(model_cfg.get('sync_snr_min_db', -10.0)),
        sync_snr_max_db=float(model_cfg.get('sync_snr_max_db', 30.0)),
        sync_max_cfo_cycles_per_sample=float(
            model_cfg.get('sync_max_cfo_cycles_per_sample', 1e-4)
        ),
        sync_max_phase_drift_rad_per_sample=float(
            model_cfg.get('sync_max_phase_drift_rad_per_sample', 1e-4)
        ),
        sync_sps_temperature=float(model_cfg.get('sync_sps_temperature', 1.0)),
        sync_film_max_delta=float(model_cfg.get('sync_film_max_delta', 0.10)),
        rtn_residual_scale_init=float(model_cfg.get('rtn_residual_scale_init', 0.10)),
        sync_eps=float(model_cfg.get('sync_eps', 1e-6)),
    ).to(device)


def _create_bimamba_estimated_cyclofresh_model(config):
    from models.IQUBiMamba1D_EstimatedCycloFRESH import IQUBiMamba1D_EstimatedCycloFRESH

    return IQUBiMamba1D_EstimatedCycloFRESH(
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
        complex_stem_enable=bool(getattr(config, 'complex_stem_enable', False)),
        complex_norm_eps=float(getattr(config, 'complex_norm_eps', 1e-6)),
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

def _create_resunet1d_noasc_model(config):
    """Factory for IQUResUNet1D_NoASC with raw decoder skip concatenation."""
    from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC

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
    ).to(device)


def _create_resunet1d_noasc_latent_mask_model(config):
    """Factory for Stage376: Stage56 plus real latent simplex masks."""
    from models.IQUResUNet1D_LatentMask import IQUResUNet1D_NoASC_LatentMask

    cfg = config.model_config
    return IQUResUNet1D_NoASC_LatentMask(
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
        latent_mask_mode=str(cfg.get("latent_mask_mode", "real")),
        latent_mask_eps=float(cfg.get("latent_mask_eps", 1.0e-6)),
    ).to(device)


def _create_resunet1d_complexstate_unireplk_latent_mask_model(config):
    """Factory for Stage377 and its mask-estimator ablations."""
    from models.IQUResUNet1D_ComplexStateUniRepLK_LatentMask import (
        IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
    )

    cfg = config.model_config
    model_class = IQUResUNet1D_ComplexStateUniRepLK_LatentMask
    separator_kwargs = {}
    backbone_kwargs = {}
    if config.model_type == "resunet1d_stage386_unireplk_backbone":
        from models.IQUResUNet1D_UniRepLKBackbone import IQUResUNet1D_Stage386
        model_class = IQUResUNet1D_Stage386
    elif config.model_type == "resunet1d_stage387_integrated_unireplk":
        from models.IQUResUNet1D_UniRepLKBackbone import IQUResUNet1D_Stage387
        model_class = IQUResUNet1D_Stage387
    elif config.model_type == "resunet1d_stage388_adaptive_complex_unireplk":
        from models.IQUResUNet1D_UniRepLKBackbone import IQUResUNet1D_Stage388
        model_class = IQUResUNet1D_Stage388
    elif config.model_type == "resunet1d_stage389_adaptive_real_unireplk":
        from models.IQUResUNet1D_UniRepLKBackbone import IQUResUNet1D_Stage389
        model_class = IQUResUNet1D_Stage389
    elif config.model_type == "resunet1d_stage390_fixed_complex_unireplk":
        from models.IQUResUNet1D_UniRepLKBackbone import IQUResUNet1D_Stage390
        model_class = IQUResUNet1D_Stage390
    if config.model_type == "resunet1d_complexstate_unireplk_light_separator":
        from models.IQUResUNet1D_LightMaskSeparator import (
            IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator,
        )

        model_class = IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator
        separator_kwargs = {
            "separator_kernel_size": int(
                cfg.get("separator_kernel_size", 5)
            ),
            "separator_dilations": tuple(
                int(value)
                for value in cfg.get("separator_dilations", (1, 2))
            ),
            "separator_residual_scale_init": float(
                cfg.get("separator_residual_scale_init", 0.1)
            ),
        }
    elif config.model_type == "resunet1d_complexstate_unireplk_separator":
        from models.IQUResUNet1D_UniRepLKSeparator import (
            IQUResUNet1D_ComplexStateUniRepLKSeparator,
        )

        model_class = IQUResUNet1D_ComplexStateUniRepLKSeparator
        separator_kwargs = {
            "separator_unireplk_stages": tuple(
                int(value)
                for value in cfg.get("separator_unireplk_stages", (0, 1, 2))
            ),
        }
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
        bimamba_apply_stages=tuple(cfg.get("bimamba_apply_stages", (1, 3))),
        bimamba_residual_scale_init=float(
            cfg.get("bimamba_residual_scale_init", 1.0)
        ),
        complex_state_d_state=int(cfg.get("complex_state_d_state", 8)),
        complex_state_d_conv=int(cfg.get("complex_state_d_conv", 4)),
        complex_state_expand=int(cfg.get("complex_state_expand", 2)),
        complex_state_scan_checkpoint=bool(
            cfg.get("complex_state_scan_checkpoint", True)
        ),
        complex_state_scan_backend=str(
            cfg.get("complex_state_scan_backend", "auto")
        ),
        complex_state_fusion_hidden=int(
            cfg.get("complex_state_fusion_hidden", 64)
        ),
        rf_apply_stages=tuple(
            int(stage) for stage in cfg.get("rf_apply_stages", (0, 1, 2))
        ),
        rf_residual_scale_init=float(
            cfg.get("rf_residual_scale_init", 0.05)
        ),
        rf_large_kernel=int(cfg.get("rf_large_kernel", 17)),
        rf_ffn_factor=int(cfg.get("rf_ffn_factor", 4)),
        rf_layer_scale=float(cfg.get("rf_layer_scale", 1e-6)),
        latent_mask_mode=str(cfg.get("latent_mask_mode", "real")),
        latent_mask_eps=float(cfg.get("latent_mask_eps", 1.0e-6)),
        **backbone_kwargs,
        **separator_kwargs,
    ).to(device)


def _create_resunet1d_hyena_bottleneck_model(config):
    from models.IQUResUNet1D_TaskSpecificLongContext import IQUResUNet1D_HyenaBottleneck

    model_cfg = config.model_config
    return IQUResUNet1D_HyenaBottleneck(
        **_resunet_innovation_common_kwargs(config),
        hyena_filter_hidden=int(model_cfg.get("hyena_filter_hidden", 64)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
    ).to(device)


def _create_resunet1d_spectral_lowrank_bottleneck_model(config):
    from models.IQUResUNet1D_TaskSpecificLongContext import IQUResUNet1D_SpectralLowRankBottleneck

    model_cfg = config.model_config
    return IQUResUNet1D_SpectralLowRankBottleneck(
        **_resunet_innovation_common_kwargs(config),
        mode_count=int(model_cfg.get("mode_count", 32)),
        spectral_rank=int(model_cfg.get("spectral_rank", 4)),
        dropout=float(model_cfg.get("dropout", 0.10)),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.02)),
    ).to(device)


def _create_resunet1d_mega_mid_encoder_model(config):
    from models.IQUResUNet1D_TaskSpecificLongContext import IQUResUNet1D_MegaMidEncoder

    model_cfg = config.model_config
    return IQUResUNet1D_MegaMidEncoder(
        **_resunet_innovation_common_kwargs(config),
        mega_stages=model_cfg.get("mega_stages", [1, 2]),
        ema_kernel_size=int(model_cfg.get("ema_kernel_size", 63)),
        expansion=float(model_cfg.get("expansion", 2.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.05)),
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
    use_vector_alpha = model_cfg.get("use_vector_alpha", False)
    global_ctx_channels = model_cfg.get("global_ctx_channels", None)
    if global_ctx_channels is not None:
        global_ctx_channels = int(global_ctx_channels)

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
        global_ctx_channels=global_ctx_channels,
        use_complex_mask=use_complex_mask,
        use_mamba_stages=use_mamba_stages,
        mamba_residual_scale_init=mamba_residual_scale_init,
        encoder_mamba_block_type=encoder_mamba_block_type,
        decoder_mamba_block_type=decoder_mamba_block_type,
        use_skip_mamba=use_skip_mamba,
        use_decoder_mamba=use_decoder_mamba,
        use_vector_alpha=use_vector_alpha,
    ).to(device)


def _resunet_innovation_common_kwargs(config):
    model_cfg = config.model_config
    return {
        "input_size": input_size,
        "input_channels": config.input_channels,
        "n_stages": config.n_stages,
        "features_per_stage": config.features_per_stage,
        "conv_op": nn.Conv1d,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "n_conv_per_stage": config.n_conv_per_stage,
        "num_classes": config.num_classes,
        "n_conv_per_stage_decoder": config.n_conv_per_stage_decoder,
        "conv_bias": bool(model_cfg.get("conv_bias", True)),
        "deep_supervision": config.deep_supervision,
    }


def _create_resunet1d_crossscale_lssg_model(config):
    from models.IQUResUNet1D_CrossScaleLSSG import IQUResUNet1D_CrossScaleLSSG

    model_cfg = config.model_config
    return IQUResUNet1D_CrossScaleLSSG(
        **_resunet_innovation_common_kwargs(config),
        gated_decoder_stages=model_cfg.get("gated_decoder_stages", None),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
    ).to(device)


def _create_resunet1d_sk_lssg_model(config):
    from models.IQUResUNet1D_SKLSSG import IQUResUNet1D_SKLSSG

    model_cfg = config.model_config
    return IQUResUNet1D_SKLSSG(
        **_resunet_innovation_common_kwargs(config),
        gated_decoder_stages=model_cfg.get("gated_decoder_stages", None),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        branch_kernels=model_cfg.get("branch_kernels", [1, 5, 9, 17]),
    ).to(device)


def _create_resunet1d_freq_lssg_model(config):
    from models.IQUResUNet1D_FreqLSSG import IQUResUNet1D_FreqLSSG

    model_cfg = config.model_config
    return IQUResUNet1D_FreqLSSG(
        **_resunet_innovation_common_kwargs(config),
        gated_decoder_stages=model_cfg.get("gated_decoder_stages", None),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        frequency_indices=model_cfg.get("frequency_indices", [1, 2, 4, 8, 16, 32]),
    ).to(device)


def _create_resunet1d_focal_lssg_model(config):
    from models.IQUResUNet1D_FocalLSSG import IQUResUNet1D_FocalLSSG

    model_cfg = config.model_config
    return IQUResUNet1D_FocalLSSG(
        **_resunet_innovation_common_kwargs(config),
        gated_decoder_stages=model_cfg.get("gated_decoder_stages", None),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        focal_levels=model_cfg.get("focal_levels", [3, 7, 15]),
    ).to(device)


def _create_resunet1d_wavelet_dccb_model(config):
    from models.IQUResUNet1D_WaveletDCCB import IQUResUNet1D_WaveletDCCB

    model_cfg = config.model_config
    return IQUResUNet1D_WaveletDCCB(
        **_resunet_innovation_common_kwargs(config),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
        wavelet_scale=float(model_cfg.get("wavelet_scale", 0.05)),
    ).to(device)


def _create_resunet1d_complex_cyclo_dccb_model(config):
    from models.IQUResUNet1D_ComplexCycloDCCB import IQUResUNet1D_ComplexCycloDCCB

    model_cfg = config.model_config
    return IQUResUNet1D_ComplexCycloDCCB(
        **_resunet_innovation_common_kwargs(config),
        cyclo_lags=model_cfg.get("cyclo_lags", [0, 1, 2, 4, 8, 16]),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
        cyclo_scale_init=float(model_cfg.get("cyclo_scale_init", 0.05)),
    ).to(device)


def _create_resunet1d_agent_attention_bottleneck_model(config):
    from models.IQUResUNet1D_AgentAttentionBottleneck import IQUResUNet1D_AgentAttentionBottleneck

    model_cfg = config.model_config
    return IQUResUNet1D_AgentAttentionBottleneck(
        **_resunet_innovation_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        agent_tokens=int(model_cfg.get("agent_tokens", 64)),
        attn_drop=float(model_cfg.get("attention_dropout", 0.05)),
        proj_drop=float(model_cfg.get("projection_dropout", 0.0)),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
    ).to(device)


def _create_resunet1d_transnext_attention_bottleneck_model(config):
    from models.IQUResUNet1D_TransNeXtBottleneck import IQUResUNet1D_TransNeXtBottleneck

    model_cfg = config.model_config
    return IQUResUNet1D_TransNeXtBottleneck(
        **_resunet_innovation_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        sr_ratio=int(model_cfg.get("sr_ratio", 4)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 2.0)),
        attn_drop=float(model_cfg.get("attention_dropout", 0.05)),
        proj_drop=float(model_cfg.get("projection_dropout", 0.0)),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
    ).to(device)


def _create_resunet1d_bilevel_routing_bottleneck_model(config):
    from models.IQUResUNet1D_BiLevelRoutingBottleneck import IQUResUNet1D_BiLevelRoutingBottleneck

    model_cfg = config.model_config
    return IQUResUNet1D_BiLevelRoutingBottleneck(
        **_resunet_innovation_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        routing_segments=int(model_cfg.get("routing_segments", 16)),
        routing_topk=int(model_cfg.get("routing_topk", 4)),
        attn_drop=float(model_cfg.get("attention_dropout", 0.05)),
        proj_drop=float(model_cfg.get("projection_dropout", 0.0)),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
    ).to(device)


def _create_resunet1d_deformable_temporal_bottleneck_model(config):
    from models.IQUResUNet1D_DeformableTemporalBottleneck import IQUResUNet1D_DeformableTemporalBottleneck

    model_cfg = config.model_config
    return IQUResUNet1D_DeformableTemporalBottleneck(
        **_resunet_innovation_common_kwargs(config),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        deform_points=int(model_cfg.get("deform_points", 8)),
        offset_kernel_size=int(model_cfg.get("offset_kernel_size", 5)),
        offset_range=float(model_cfg.get("offset_range", 8.0)),
        attn_drop=float(model_cfg.get("attention_dropout", 0.05)),
        proj_drop=float(model_cfg.get("projection_dropout", 0.0)),
        bottleneck_scale_init=float(model_cfg.get("bottleneck_scale_init", 0.05)),
    ).to(device)


def _create_resunet1d_skip_enhanced_attention_model(config):
    return _create_resunet1d_skip_enhanced_model(config, skip_mode="attention")

def _create_resunet1d_skip_enhanced_uct_model(config):
    return _create_resunet1d_skip_enhanced_model(config, skip_mode="uct")

def _create_resunet1d_skip_enhanced_dca_model(config):
    return _create_resunet1d_skip_enhanced_model(config, skip_mode="dca")

def _create_esd_mask_model(config):
    from models.IQUResUNet1D_SkipEnhanced import IQUResUNet1D_SkipEnhanced
    enc_channels = getattr(config, "enc_channels", 256)
    kernel_size = getattr(config, "kernel_size", 16)
    stride = getattr(config, "stride", 8)
    mask_act = getattr(config, "mask_act", "sigmoid")
    num_sources = getattr(config, "num_sources", config.num_classes // 2)
    enc_T = (config.input_size - kernel_size) // stride + 1
    
    separator = IQUResUNet1D_SkipEnhanced(
        input_size=enc_T,
        input_channels=enc_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=get_nn_module(config.conv_op) if getattr(config, 'conv_op', None) else nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=enc_channels * num_sources,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
        skip_mode="lssg",
        gated_decoder_stages=[config.n_stages - 2],
        residual_scale_init=getattr(config, "residual_scale_init", 0.02),
        use_complex_mask=False
    )
    
    model = ESDMaskWrapper1D(
        separator=separator,
        num_sources=num_sources,
        enc_channels=enc_channels,
        kernel_size=kernel_size,
        stride=stride,
        mask_act=mask_act
    )
    return model


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
    
    use_sk_routing = model_cfg.get("use_sk_routing", False)
    use_phase_aware_context = model_cfg.get("use_phase_aware_context", False)
    use_bottom_up_leakage = model_cfg.get("use_bottom_up_leakage", False)
    
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
        use_sk_routing=use_sk_routing,
        use_phase_aware_context=use_phase_aware_context,
        use_bottom_up_leakage=use_bottom_up_leakage,
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


def _create_icassp_baseline_unet_model(config: MambaConfig):
    """Build the official TensorFlow-UNet-inspired PyTorch baseline."""

    from models.icassp_baseline_unet import ICASPBaselineUNet

    return ICASPBaselineUNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        k_neurons=int(getattr(config, "k_neurons", 32)),
        k_sz=int(getattr(config, "k_sz", 3)),
        long_k_sz=int(getattr(config, "long_k_sz", 101)),
        dropout_first=float(getattr(config, "dropout_first", 0.25)),
        dropout_rest=float(getattr(config, "dropout_rest", 0.5)),
    ).to(device)


def _create_iq_resdilated_unet_model(config: MambaConfig):
    """Build Stage 264: an RF-oriented compute-matched U-Net."""

    from models.iq_resdilated_unet import IQResDilatedUNet

    return IQResDilatedUNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        k_neurons=int(getattr(config, "k_neurons", 64)),
        k_sz=int(getattr(config, "k_sz", 3)),
        long_k_sz=int(getattr(config, "long_k_sz", 101)),
        dropout_first=float(getattr(config, "dropout_first", 0.10)),
        dropout_rest=float(getattr(config, "dropout_rest", 0.10)),
        shallow_dilated_channels=tuple(
            int(channels)
            for channels in getattr(config, "shallow_dilated_channels", (128, 64))
        ),
        skip_gate_groups=int(getattr(config, "skip_gate_groups", 16)),
        use_bottleneck_bimamba=bool(
            getattr(config, "use_bottleneck_bimamba", True)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
    ).to(device)


def _create_icassp_baseline_wavenet_model(config: MambaConfig):
    """Build the public ICASSP 2024 PyTorch WaveNet baseline."""

    from models.icassp_baseline_wavenet import ICASPBaselineWaveNet

    return ICASPBaselineWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 128)),
        residual_layers=int(getattr(config, "residual_layers", 30)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
    ).to(device)


def _create_rfdemucs_model(config: MambaConfig):
    """Build the TUB RFDEMUCS waveform estimator."""

    from models.rfdemucs import RFDEMUCS

    return RFDEMUCS(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        hidden=int(getattr(config, "rfdemucs_hidden", 64)),
        depth=int(getattr(config, "rfdemucs_depth", 5)),
        kernel_size=int(getattr(config, "rfdemucs_kernel_size", 8)),
        stride=int(getattr(config, "rfdemucs_stride", 2)),
        resample=int(getattr(config, "rfdemucs_resample", 2)),
        growth=float(getattr(config, "rfdemucs_growth", 2.0)),
        max_hidden=int(getattr(config, "rfdemucs_max_hidden", 10_000)),
        normalize=bool(getattr(config, "rfdemucs_normalize", False)),
        glu=bool(getattr(config, "rfdemucs_glu", True)),
        rescale=float(getattr(config, "rfdemucs_rescale", 0.1)),
        lstm_layers=int(getattr(config, "rfdemucs_lstm_layers", 2)),
        sinc_zeros=int(getattr(config, "rfdemucs_sinc_zeros", 56)),
    ).to(device)


def _create_icassp_wavenet_mamba_model(
    config: MambaConfig,
    *,
    bidirectional: bool,
):
    """Build a Stage-257/258 WaveNet with post-skip Mamba fusion."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetBiMamba,
        ICASPBaselineWaveNetMamba,
    )

    model_class = ICASPBaselineWaveNetBiMamba if bidirectional else ICASPBaselineWaveNetMamba
    return model_class(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 30)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
    ).to(device)


def _create_icassp_wavenet_multirate_mamba_model(
    config: MambaConfig,
    *,
    bidirectional: bool,
):
    """Build Stage 259/260: a 10-block WaveNet with low-rate Mamba context."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetMultiRateBiMamba,
        ICASPBaselineWaveNetMultiRateMamba,
    )

    model_class = (
        ICASPBaselineWaveNetMultiRateBiMamba
        if bidirectional
        else ICASPBaselineWaveNetMultiRateMamba
    )
    return model_class(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 10)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
    ).to(device)


def _create_icassp_wavenet_interleaved_mamba_model(
    config: MambaConfig,
    *,
    bidirectional: bool,
):
    """Build Stage 261/262: multi-rate Mamba between two WaveNet cycles."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedBiMamba,
        ICASPBaselineWaveNetInterleavedMamba,
    )

    model_class = (
        ICASPBaselineWaveNetInterleavedBiMamba
        if bidirectional
        else ICASPBaselineWaveNetInterleavedMamba
    )
    return model_class(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
    ).to(device)


def _create_icassp_wavenet_chunk_mamba_strong_fusion_model(
    config: MambaConfig,
):
    """Build Stages 276/277: local WaveNet around chunk-token Mamba fusion."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetChunkMambaStrongFusion,
    )

    return ICASPBaselineWaveNetChunkMambaStrongFusion(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 10)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 5)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 5)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_chunk_size=int(getattr(config, "mamba_chunk_size", 64)),
        mamba_chunk_hop=int(getattr(config, "mamba_chunk_hop", 32)),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_fusion_gain_init=float(
            getattr(config, "mamba_fusion_gain_init", 1.0)
        ),
        mamba_fusion_norm_eps=float(
            getattr(config, "mamba_fusion_norm_eps", 1e-6)
        ),
    ).to(device)


def _create_icassp_wavenet_mamba_film_controller_model(
    config: MambaConfig,
):
    """Build Stage 269: Mamba controls the late WaveNet block paths."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetMambaFiLMController,
    )

    return ICASPBaselineWaveNetMambaFiLMController(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_controller_hidden=int(
            getattr(config, "mamba_controller_hidden", 128)
        ),
        mamba_controller_dropout=float(
            getattr(config, "mamba_controller_dropout", 0.0)
        ),
        mamba_control_gate_max_delta=float(
            getattr(config, "mamba_control_gate_max_delta", 0.5)
        ),
        mamba_control_film_max_delta=float(
            getattr(config, "mamba_control_film_max_delta", 0.1)
        ),
    ).to(device)


def _create_icassp_wavenet_mamba_dilation_skip_router_model(
    config: MambaConfig,
):
    """Build Stage 270: normalized routing across late dilation paths."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetMambaDilationSkipRouter,
    )

    return ICASPBaselineWaveNetMambaDilationSkipRouter(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_controller_hidden=int(
            getattr(config, "mamba_controller_hidden", 128)
        ),
        mamba_controller_dropout=float(
            getattr(config, "mamba_controller_dropout", 0.0)
        ),
        mamba_router_strength=float(getattr(config, "mamba_router_strength", 0.25)),
        mamba_router_temperature=float(
            getattr(config, "mamba_router_temperature", 1.0)
        ),
    ).to(device)


def _create_icassp_wavenet_interleaved_phase_aware_reverse_mamba_model(
    config: MambaConfig,
):
    """Build Stage 271: Stage 261 with a physical complex reverse branch."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedPhaseAwareReverseMamba,
    )

    return ICASPBaselineWaveNetInterleavedPhaseAwareReverseMamba(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
        phase_reverse_mamba_channels=int(
            getattr(config, "phase_reverse_mamba_channels", 64)
        ),
        phase_reverse_downsample_factor=int(
            getattr(config, "phase_reverse_downsample_factor", 4)
        ),
        phase_reverse_d_state=int(getattr(config, "phase_reverse_d_state", 16)),
        phase_reverse_d_conv=int(getattr(config, "phase_reverse_d_conv", 4)),
        phase_reverse_expand=int(getattr(config, "phase_reverse_expand", 2)),
        phase_reverse_dropout=float(
            getattr(config, "phase_reverse_dropout", 0.0)
        ),
        phase_reverse_scale_init=float(
            getattr(config, "phase_reverse_scale_init", 1e-2)
        ),
    ).to(device)


def _create_icassp_wavenet_interleaved_gated_bimamba_model(
    config: MambaConfig,
):
    """Build Stage 268: Stage-261 plus a gated reverse Mamba correction."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedGatedBiMamba,
    )

    return ICASPBaselineWaveNetInterleavedGatedBiMamba(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
        mamba_backward_gate_init=float(
            getattr(config, "mamba_backward_gate_init", 1e-2)
        ),
    ).to(device)


def _create_icassp_wavenet_interleaved_crossscale_bimamba_model(
    config: MambaConfig,
):
    """Build Stage 265: Stage-261 plus Stage-235-style BiMamba memory."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedCrossScaleBiMamba,
    )

    return ICASPBaselineWaveNetInterleavedCrossScaleBiMamba(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
        cross_scale_kv_tokens=int(getattr(config, "cross_scale_kv_tokens", 64)),
        cross_scale_num_heads=int(getattr(config, "cross_scale_num_heads", 4)),
        cross_scale_dropout=float(getattr(config, "cross_scale_dropout", 0.0)),
        cross_scale_residual_scale_init=float(
            getattr(config, "cross_scale_residual_scale_init", 1e-2)
        ),
    ).to(device)


def _create_icassp_wavenet_interleaved_stage235_memory_model(
    config: MambaConfig,
):
    """Build Stage 272: Stage-235 K/V memory in place of Stage-261 Mamba."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedStage235Memory,
    )

    return ICASPBaselineWaveNetInterleavedStage235Memory(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        cross_scale_kv_tokens=int(getattr(config, "cross_scale_kv_tokens", 64)),
        cross_scale_num_heads=int(getattr(config, "cross_scale_num_heads", 4)),
        cross_scale_dropout=float(getattr(config, "cross_scale_dropout", 0.0)),
        cross_scale_residual_scale_init=float(
            getattr(config, "cross_scale_residual_scale_init", 1e-2)
        ),
    ).to(device)


def _create_icassp_wavenet_interleaved_physical_moe_bimamba_model(
    config: MambaConfig,
):
    """Build Stage 266: Stage-261 plus Stage-255-style physical MoE."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedPhysicalMoEBiMamba,
    )

    return ICASPBaselineWaveNetInterleavedPhysicalMoEBiMamba(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
        fusion_global_kv_tokens=int(getattr(config, "fusion_global_kv_tokens", 64)),
        fusion_num_heads=int(getattr(config, "fusion_num_heads", 4)),
        fusion_dropout=float(getattr(config, "fusion_dropout", 0.0)),
        fusion_channel_scale_init=float(
            getattr(config, "fusion_channel_scale_init", 0.1)
        ),
        fusion_channel_scale_max=float(
            getattr(config, "fusion_channel_scale_max", 0.5)
        ),
        fusion_router_hidden=int(getattr(config, "fusion_router_hidden", 64)),
        fusion_expert_prior=getattr(
            config, "fusion_expert_prior", [0.7, 0.1, 0.1, 0.1]
        ),
        fusion_condition_hidden=int(
            getattr(config, "fusion_condition_hidden", 16)
        ),
        fusion_condition_embedding=int(
            getattr(config, "fusion_condition_embedding", 16)
        ),
        fusion_trust_penalty_init=float(
            getattr(config, "fusion_trust_penalty_init", 0.1)
        ),
        fusion_trust_penalty_enable=bool(
            getattr(config, "fusion_trust_penalty_enable", True)
        ),
        fusion_condition_routing_enable=bool(
            getattr(config, "fusion_condition_routing_enable", True)
        ),
        physical_cyclic_lags=getattr(
            config, "physical_cyclic_lags", [0, 1, 2, 4, 8]
        ),
        physical_polyphase_branches=int(
            getattr(config, "physical_polyphase_branches", 8)
        ),
        physical_symbol_orders=getattr(config, "physical_symbol_orders", [2, 4, 8]),
        physical_min_cyclic_freq=float(
            getattr(config, "physical_min_cyclic_freq", 1.0 / 64.0)
        ),
        physical_max_cyclic_freq=float(
            getattr(config, "physical_max_cyclic_freq", 1.0 / 8.0)
        ),
        physical_cyclic_temperature=float(
            getattr(config, "physical_cyclic_temperature", 0.25)
        ),
    ).to(device)


def _create_icassp_wavenet_interleaved_cyclofresh_model(
    config: MambaConfig,
):
    """Build Stage 267: Stage-261 plus Stage-79 estimated Cyclo-FRESH."""

    from models.icassp_wavenet_mamba import (
        ICASPBaselineWaveNetInterleavedMambaCycloFRESH,
    )

    return ICASPBaselineWaveNetInterleavedMambaCycloFRESH(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 64)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 1e-2)),
        estimated_cyclofresh_min_freq=float(
            getattr(config, "estimated_cyclofresh_min_freq", 1.0 / 64.0)
        ),
        estimated_cyclofresh_max_freq=float(
            getattr(config, "estimated_cyclofresh_max_freq", 1.0 / 8.0)
        ),
        estimated_cyclofresh_default_freq=float(
            getattr(config, "estimated_cyclofresh_default_freq", 1.0 / 32.0)
        ),
        estimated_cyclofresh_momentum=float(
            getattr(config, "estimated_cyclofresh_momentum", 0.05)
        ),
        estimated_cyclofresh_hidden_channels=int(
            getattr(config, "estimated_cyclofresh_hidden_channels", 8)
        ),
        estimated_cyclofresh_kernel_size=int(
            getattr(config, "estimated_cyclofresh_kernel_size", 9)
        ),
        estimated_cyclofresh_scale_init=float(
            getattr(config, "estimated_cyclofresh_scale_init", 1e-2)
        ),
        estimated_cyclofresh_gate_hidden=int(
            getattr(config, "estimated_cyclofresh_gate_hidden", 8)
        ),
        estimated_cyclofresh_zero_init=bool(
            getattr(config, "estimated_cyclofresh_zero_init", True)
        ),
    ).to(device)


def _create_icassp_wavenet_antialiased_mamba_model(config: MambaConfig):
    """Build Stage 278: Stage 261 with deterministic anti-alias analysis."""

    from models.icassp_symbol_clock_wavenet import (
        ICASPAntiAliasedInterleavedMamba,
    )

    return ICASPAntiAliasedInterleavedMamba(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 128)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        mamba_insert_after_block=int(
            getattr(config, "mamba_insert_after_block", 10)
        ),
        mamba_channels=int(getattr(config, "mamba_channels", 64)),
        mamba_downsample_factor=int(
            getattr(config, "mamba_downsample_factor", 4)
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        mamba_scale_init=float(getattr(config, "mamba_scale_init", 0.01)),
        antialias_taps_per_phase=int(
            getattr(config, "antialias_taps_per_phase", 8)
        ),
        antialias_cutoff_ratio=float(
            getattr(config, "antialias_cutoff_ratio", 0.90)
        ),
    ).to(device)


def _create_icassp_wavenet_temporal_physical_controller_model(
    config: MambaConfig,
):
    """Build Stage 279: fixed dilations with ordered physical controls."""

    from models.icassp_symbol_clock_wavenet import (
        ICASPTemporalPhysicalControllerWaveNet,
    )

    return ICASPTemporalPhysicalControllerWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 128)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        controller_insert_after_block=int(
            getattr(config, "controller_insert_after_block", 10)
        ),
        token_channels=int(getattr(config, "physical_token_channels", 64)),
        chunk_size=int(getattr(config, "physical_chunk_size", 64)),
        chunk_hop=int(getattr(config, "physical_chunk_hop", 32)),
        physical_lags=tuple(
            int(x) for x in getattr(config, "physical_lags", (1, 2, 4, 8, 16, 32))
        ),
        candidate_periods=tuple(
            int(x)
            for x in getattr(
                config, "symbol_candidate_periods", (2, 4, 8, 16, 32)
            )
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        control_gate_max_delta=float(
            getattr(config, "temporal_control_gate_max_delta", 0.5)
        ),
        control_film_max_delta=float(
            getattr(config, "temporal_control_film_max_delta", 0.1)
        ),
        evidence_strength=float(
            getattr(config, "symbol_evidence_strength", 1.0)
        ),
        router_temperature=float(
            getattr(config, "symbol_router_temperature", 0.35)
        ),
    ).to(device)


def _create_icassp_symbol_clock_wavenet_model(config: MambaConfig):
    """Build Stages 280-282 from one explicitly ablated implementation."""

    from models.icassp_symbol_clock_wavenet import ICASPSymbolClockWaveNet

    return ICASPSymbolClockWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 128)),
        pre_residual_layers=int(getattr(config, "pre_residual_layers", 10)),
        pre_dilation_cycle_length=int(
            getattr(config, "pre_dilation_cycle_length", 10)
        ),
        adaptive_layers=int(getattr(config, "adaptive_layers", 5)),
        candidate_periods=tuple(
            int(x)
            for x in getattr(
                config, "symbol_candidate_periods", (2, 4, 8, 16, 32)
            )
        ),
        dilation_multipliers=tuple(
            float(x)
            for x in getattr(
                config, "symbol_dilation_multipliers", (1, 2, 4, 8, 16)
            )
        ),
        max_dilation=int(getattr(config, "symbol_max_dilation", 512)),
        use_widely_linear_stem=bool(
            getattr(config, "use_widely_linear_stem", False)
        ),
        use_temporal_controls=bool(
            getattr(config, "use_temporal_controls", False)
        ),
        token_channels=int(getattr(config, "physical_token_channels", 64)),
        chunk_size=int(getattr(config, "physical_chunk_size", 64)),
        chunk_hop=int(getattr(config, "physical_chunk_hop", 32)),
        physical_lags=tuple(
            int(x) for x in getattr(config, "physical_lags", (1, 2, 4, 8, 16, 32))
        ),
        mamba_d_state=int(getattr(config, "mamba_d_state", 16)),
        mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
        mamba_expand=int(getattr(config, "mamba_expand", 2)),
        mamba_dropout=float(getattr(config, "mamba_dropout", 0.0)),
        control_gate_max_delta=float(
            getattr(config, "temporal_control_gate_max_delta", 0.5)
        ),
        control_film_max_delta=float(
            getattr(config, "temporal_control_film_max_delta", 0.1)
        ),
        evidence_strength=float(
            getattr(config, "symbol_evidence_strength", 1.0)
        ),
        router_temperature=float(
            getattr(config, "symbol_router_temperature", 0.35)
        ),
    ).to(device)


def _create_icassp_complex_wavenet_model(config: MambaConfig):
    """Build Stages 283-289 from one Stage-273 complex-ablation backbone."""

    from models.icassp_complex_wavenet import ICASPComplexWaveNet

    return ICASPComplexWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 128)),
        residual_layers=int(getattr(config, "residual_layers", 20)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        complex_layers=int(getattr(config, "complex_layers", 20)),
        strict_complex_output=bool(
            getattr(config, "strict_complex_output", False)
        ),
        use_conjugate_adapter=bool(
            getattr(config, "use_conjugate_adapter", False)
        ),
        conjugate_adapter_max=float(
            getattr(config, "conjugate_adapter_max", 0.15)
        ),
        complex_norm_enable=bool(
            getattr(config, "complex_norm_enable", False)
        ),
        complex_norm_eps=float(getattr(config, "complex_norm_eps", 1e-6)),
    ).to(device)


def _create_iqumamba_stage4_complex_model(config: MambaConfig):
    """Build the controlled C1-C5 complex-valued Stage-4 ablations."""

    from models.IQUMamba1D_ComplexStage4 import (
        IQUMamba1DComplexStage4,
        IQUMamba1DComplexStemC1,
    )

    model_class = (
        IQUMamba1DComplexStemC1
        if config.model_type == "iqumamba_stage4_complex_c1"
        else IQUMamba1DComplexStage4
    )
    model_cfg = config.model_config
    return model_class(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        strict_complex_output=bool(
            model_cfg.get("strict_complex_output", False)
        ),
        use_equivariant_mamba=bool(
            model_cfg.get("use_equivariant_mamba", False)
        ),
        complex_norm_eps=float(model_cfg.get("complex_norm_eps", 1e-6)),
        mamba_d_state=int(model_cfg.get("mamba_d_state", 16)),
        mamba_d_conv=int(model_cfg.get("mamba_d_conv", 4)),
        mamba_expand=int(model_cfg.get("mamba_expand", 2)),
        mamba_max_gain_delta=float(
            model_cfg.get("mamba_max_gain_delta", 0.25)
        ),
        mamba_max_rotation=float(
            model_cfg.get("mamba_max_rotation", math.pi / 2)
        ),
    ).to(device)


def _create_iqumamba_stage4_complex_state_model(config: MambaConfig):
    """Build Stage 295/299 and controlled RF Mamba-3 recurrence variants."""

    from models.IQUMamba1D_ComplexStateMamba import (
        IQUMamba1DComplexStateMamba,
        IQUMamba1DRealStateTrapReliability,
    )

    model_cfg = config.model_config
    model_class = (
        IQUMamba1DRealStateTrapReliability
        if config.model_type == "iqumamba_real_state_trap_reliability"
        else IQUMamba1DComplexStateMamba
    )
    return model_class(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        mamba_d_state=int(model_cfg.get("mamba_d_state", 8)),
        mamba_d_conv=int(model_cfg.get("mamba_d_conv", 4)),
        mamba_expand=int(model_cfg.get("mamba_expand", 2)),
        scan_checkpoint=bool(model_cfg.get("scan_checkpoint", True)),
        scan_backend=str(model_cfg.get("scan_backend", "auto")),
        require_mamba_fused_scan=bool(
            model_cfg.get("require_mamba_fused_scan", False)
        ),
        mamba_discretization=str(
            model_cfg.get("mamba_discretization", "exponential_euler")
        ),
        trapezoid_lambda_init=float(
            model_cfg.get("trapezoid_lambda_init", 0.5)
        ),
        cyclic_theta_enable=bool(
            model_cfg.get("cyclic_theta_enable", False)
        ),
        cyclic_frequencies=model_cfg.get("cyclic_frequencies", []),
        cyclic_max_frequency_delta=float(
            model_cfg.get("cyclic_max_frequency_delta", 0.01)
        ),
        reliability_enable=bool(
            model_cfg.get("reliability_enable", False)
        ),
        reliability_hidden=int(model_cfg.get("reliability_hidden", 8)),
        reliability_floor=float(model_cfg.get("reliability_floor", 0.05)),
        reliability_init=float(model_cfg.get("reliability_init", 0.995)),
        complex_stem_enable=bool(
            model_cfg.get("complex_stem_enable", False)
        ),
        complex_norm_eps=float(model_cfg.get("complex_norm_eps", 1e-6)),
    ).to(device)


def _create_iqumamba_mamba3_extension_model(config: MambaConfig):
    """Build Stage 340 official Mamba-3 or the Stage 342 full RF model."""

    from models.IQUMamba1D_Mamba3Extensions import (
        IQUMamba1DFullRFCombination,
        IQUMamba1DOfficialMamba3,
    )

    cfg = config.model_config
    common = dict(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
    )
    if config.model_type == "iqumamba_stage4_official_mamba3":
        model = IQUMamba1DOfficialMamba3(
            **common,
            deep_supervision=bool(getattr(config, "deep_supervision", False)),
            d_state=int(cfg.get("memory_d_state", 128)),
            expand=int(cfg.get("memory_expand", 2)),
            headdim=int(cfg.get("memory_headdim", 64)),
            ngroups=int(cfg.get("memory_ngroups", 1)),
            rope_fraction=float(cfg.get("memory_rope_fraction", 0.5)),
            is_outproj_norm=bool(cfg.get("memory_is_outproj_norm", False)),
            is_mimo=bool(cfg.get("memory_is_mimo", False)),
            mimo_rank=int(cfg.get("memory_mimo_rank", 4)),
            chunk_size=int(cfg.get("memory_chunk_size", 64)),
        )
    elif config.model_type == "iqumamba_full_rf_mamba3_combination":
        fresh = {
            name: cfg[name]
            for name in (
                "estimated_cyclofresh_min_freq",
                "estimated_cyclofresh_max_freq",
                "estimated_cyclofresh_default_freq",
                "estimated_cyclofresh_momentum",
                "estimated_cyclofresh_hidden_channels",
                "estimated_cyclofresh_kernel_size",
                "estimated_cyclofresh_scale_init",
                "estimated_cyclofresh_gate_hidden",
                "estimated_cyclofresh_zero_init",
            )
            if name in cfg
        }
        model = IQUMamba1DFullRFCombination(
            **common,
            mamba_d_state=int(cfg.get("mamba_d_state", 8)),
            mamba_d_conv=int(cfg.get("mamba_d_conv", 4)),
            mamba_expand=int(cfg.get("mamba_expand", 2)),
            scan_checkpoint=bool(cfg.get("scan_checkpoint", True)),
            scan_backend=str(cfg.get("scan_backend", "auto")),
            trapezoid_lambda_init=float(cfg.get("trapezoid_lambda_init", 0.5)),
            cyclic_frequencies=cfg.get("cyclic_frequencies", []),
            cyclic_max_frequency_delta=float(
                cfg.get("cyclic_max_frequency_delta", 0.01)
            ),
            reliability_hidden=int(cfg.get("reliability_hidden", 8)),
            reliability_floor=float(cfg.get("reliability_floor", 0.05)),
            reliability_init=float(cfg.get("reliability_init", 0.995)),
            complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
            estimated_cyclofresh_enable=bool(
                cfg.get("estimated_cyclofresh_enable", True)
            ),
            estimated_cyclofresh_config=fresh,
            rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
            unireplk_large_kernel=int(cfg.get("unireplk_large_kernel", 17)),
            unireplk_ffn_factor=int(cfg.get("unireplk_ffn_factor", 4)),
            unireplk_layer_scale=float(cfg.get("unireplk_layer_scale", 1e-6)),
        )
    else:
        raise ValueError(f"Unsupported Mamba-3 extension type: {config.model_type}")
    return model.to(device)


def _create_iqumamba_official_rf_mamba3_model(config: MambaConfig):
    """Build controlled Stage 343-346 official fused RF-Mamba-3 variants."""

    from models.IQUMamba1D_OfficialRFMamba3 import (
        IQUMamba1DOfficialRFMamba3,
    )

    cfg = config.model_config
    fresh = {
        name: cfg[name]
        for name in (
            "estimated_cyclofresh_min_freq",
            "estimated_cyclofresh_max_freq",
            "estimated_cyclofresh_default_freq",
            "estimated_cyclofresh_momentum",
            "estimated_cyclofresh_hidden_channels",
            "estimated_cyclofresh_kernel_size",
            "estimated_cyclofresh_scale_init",
            "estimated_cyclofresh_gate_hidden",
            "estimated_cyclofresh_zero_init",
        )
        if name in cfg
    }
    return IQUMamba1DOfficialRFMamba3(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=bool(getattr(config, "deep_supervision", False)),
        d_state=int(cfg.get("memory_d_state", 128)),
        expand=int(cfg.get("memory_expand", 2)),
        headdim=int(cfg.get("memory_headdim", 64)),
        ngroups=int(cfg.get("memory_ngroups", 1)),
        rope_fraction=float(cfg.get("memory_rope_fraction", 0.5)),
        is_outproj_norm=bool(cfg.get("memory_is_outproj_norm", False)),
        chunk_size=int(cfg.get("memory_chunk_size", 64)),
        force_real_state=bool(cfg.get("force_real_state", False)),
        cyclic_anchor_enable=bool(cfg.get("cyclic_anchor_enable", False)),
        cyclic_frequencies=cfg.get("cyclic_frequencies", []),
        cyclic_max_frequency_delta=float(
            cfg.get("cyclic_max_frequency_delta", 0.01)
        ),
        dynamic_cyclic_enable=bool(cfg.get("dynamic_cyclic_enable", False)),
        reliability_enable=bool(cfg.get("reliability_enable", False)),
        reliability_hidden=int(cfg.get("reliability_hidden", 8)),
        reliability_floor=float(cfg.get("reliability_floor", 0.05)),
        reliability_init=float(cfg.get("reliability_init", 0.995)),
        shared_conditioning_enable=bool(
            cfg.get("shared_conditioning_enable", False)
        ),
        estimated_cyclofresh_config=fresh,
    ).to(device)


def _create_iqumamba_phase_folded_mamba_model(config: MambaConfig):
    """Build Stage 334: Stage 4 plus a blind phase-folded complex SSM."""

    from models.IQUMamba1D_PhaseFoldedMamba import (
        IQUMamba1DPhaseFoldedMamba,
    )

    cfg = config.model_config
    return IQUMamba1DPhaseFoldedMamba(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=bool(getattr(config, "deep_supervision", False)),
        hidden_channels=int(cfg.get("phase_fold_hidden_channels", 16)),
        candidate_periods=cfg.get(
            "phase_fold_candidate_periods", [8, 12, 16, 24, 32]
        ),
        local_frequency_radius=float(
            cfg.get("phase_fold_local_frequency_radius", 0.20)
        ),
        evidence_temperature=float(
            cfg.get("phase_fold_evidence_temperature", 0.10)
        ),
        frequency_temperature=float(
            cfg.get("phase_fold_frequency_temperature", 0.25)
        ),
        null_logit_init=float(cfg.get("phase_fold_null_logit_init", 1.0)),
        num_routers=int(cfg.get("phase_fold_num_routers", 2)),
        d_state=int(cfg.get("phase_fold_d_state", 4)),
        d_conv=int(cfg.get("phase_fold_d_conv", 3)),
        expand=int(cfg.get("phase_fold_expand", 1)),
        scan_checkpoint=bool(cfg.get("phase_fold_scan_checkpoint", True)),
        scan_backend=str(cfg.get("phase_fold_scan_backend", "auto")),
        candidate_top_k=int(cfg.get("phase_fold_candidate_top_k", 3)),
        zero_init=bool(cfg.get("phase_fold_zero_init", True)),
    ).to(device)


def _create_iqumamba_memory_rf_model(config: MambaConfig):
    """Build Stage 335-339 memory/RF experiments and Stage 348."""

    from models.IQUMamba1D_MemoryRFStages import (
        IQUMamba1DMamba2SSD,
        IQUMamba1DRoleRF,
        IQUMamba1DReliabilityS4D,
        IQUMamba1DS4D,
        IQUMamba1DS4DUniRepLK,
        IQUMamba1DStrictComplexS4D,
    )

    cfg = config.model_config
    common = dict(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=bool(getattr(config, "deep_supervision", False)),
    )
    if config.model_type == "iqumamba_stage4_mamba2_ssd":
        model = IQUMamba1DMamba2SSD(
            **common,
            d_state=int(cfg.get("memory_d_state", 64)),
            d_conv=int(cfg.get("memory_d_conv", 4)),
            expand=int(cfg.get("memory_expand", 2)),
            headdim=int(cfg.get("memory_headdim", 32)),
            ngroups=int(cfg.get("memory_ngroups", 1)),
            chunk_size=int(cfg.get("memory_chunk_size", 256)),
        )
    elif config.model_type == "iqumamba_stage4_s4d":
        model = IQUMamba1DS4D(
            **common,
            d_state=int(cfg.get("memory_d_state", 64)),
            dropout=float(cfg.get("memory_dropout", 0.0)),
            dt_min=float(cfg.get("memory_dt_min", 1e-3)),
            dt_max=float(cfg.get("memory_dt_max", 1e-1)),
            complex_stem_enable=bool(cfg.get("complex_stem_enable", False)),
            complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
        )
    elif config.model_type == "iqumamba_stage4_complex_s4d":
        model = IQUMamba1DStrictComplexS4D(
            **common,
            d_state=int(cfg.get("memory_d_state", 64)),
            dropout=float(cfg.get("memory_dropout", 0.0)),
            dt_min=float(cfg.get("memory_dt_min", 1e-3)),
            dt_max=float(cfg.get("memory_dt_max", 1e-1)),
            complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
        )
    elif config.model_type == "iqumamba_stage4_s4d_unireplk":
        model = IQUMamba1DS4DUniRepLK(
            **common,
            d_state=int(cfg.get("memory_d_state", 64)),
            dropout=float(cfg.get("memory_dropout", 0.0)),
            dt_min=float(cfg.get("memory_dt_min", 1e-3)),
            dt_max=float(cfg.get("memory_dt_max", 1e-1)),
            complex_stem_enable=bool(cfg.get("complex_stem_enable", True)),
            complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
            rf_residual_scale_init=float(cfg.get("rf_residual_scale_init", 0.05)),
            unireplk_large_kernel=int(cfg.get("unireplk_large_kernel", 17)),
            unireplk_ffn_factor=int(cfg.get("unireplk_ffn_factor", 4)),
            unireplk_layer_scale=float(cfg.get("unireplk_layer_scale", 1e-6)),
        )
    elif config.model_type == "iqumamba_stage4_s4d_reliability":
        model = IQUMamba1DReliabilityS4D(
            **common,
            d_state=int(cfg.get("memory_d_state", 64)),
            dropout=float(cfg.get("memory_dropout", 0.0)),
            dt_min=float(cfg.get("memory_dt_min", 1e-3)),
            dt_max=float(cfg.get("memory_dt_max", 1e-1)),
            reliability_hidden=int(cfg.get("reliability_hidden", 8)),
            reliability_floor=float(cfg.get("reliability_floor", 0.05)),
            reliability_init=float(cfg.get("reliability_init", 0.995)),
        )
    else:
        model = IQUMamba1DRoleRF(
            **common,
            variant=str(cfg.get("role_rf_variant", "shared")),
            d_state=int(cfg.get("role_rf_d_state", 16)),
            expand=int(cfg.get("role_rf_expand", 2)),
            context_kernels=cfg.get("role_rf_context_kernels", [3, 15, 63]),
        )
    return model.to(device)


def _create_iqumamba_combined_stage_model(config: MambaConfig):
    """Build controlled Stage 301-305 combinations."""

    from models.IQUBiMamba1D_EstimatedCycloFRESH import (
        IQUBiMamba1D_EstimatedCycloFRESH,
    )
    from models.IQUMamba1D_CombinedStages import (
        Stage301ComplexStateCrossScale,
        Stage302BiFRESHComplexBottleneck,
        Stage303ComplexStemBiMambaCrossScale,
        Stage304Stage298299Fusion,
        Stage305GatedFRESHComplexState,
    )
    from models.IQUMamba1D_ComplexStateMamba import (
        IQUMamba1DComplexStateMamba,
    )

    cfg = config.model_config
    common = dict(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
    )
    state = dict(
        mamba_d_state=int(cfg.get("mamba_d_state", 8)),
        mamba_d_conv=int(cfg.get("mamba_d_conv", 4)),
        mamba_expand=int(cfg.get("mamba_expand", 2)),
        scan_checkpoint=bool(cfg.get("scan_checkpoint", True)),
        scan_backend=str(cfg.get("scan_backend", "auto")),
        complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
    )
    fresh = dict(
        estimated_cyclofresh_min_freq=float(
            cfg.get("estimated_cyclofresh_min_freq", 1.0 / 64.0)
        ),
        estimated_cyclofresh_max_freq=float(
            cfg.get("estimated_cyclofresh_max_freq", 1.0 / 8.0)
        ),
        estimated_cyclofresh_default_freq=float(
            cfg.get("estimated_cyclofresh_default_freq", 1.0 / 32.0)
        ),
        estimated_cyclofresh_momentum=float(
            cfg.get("estimated_cyclofresh_momentum", 0.05)
        ),
        estimated_cyclofresh_hidden_channels=int(
            cfg.get("estimated_cyclofresh_hidden_channels", 8)
        ),
        estimated_cyclofresh_kernel_size=int(
            cfg.get("estimated_cyclofresh_kernel_size", 9)
        ),
        estimated_cyclofresh_scale_init=float(
            cfg.get("estimated_cyclofresh_scale_init", 0.01)
        ),
        estimated_cyclofresh_gate_hidden=int(
            cfg.get("estimated_cyclofresh_gate_hidden", 8)
        ),
        estimated_cyclofresh_zero_init=bool(
            cfg.get("estimated_cyclofresh_zero_init", True)
        ),
    )
    cross_scale = dict(
        cross_scale_query_stages=cfg.get("cross_scale_query_stages", [2]),
        cross_scale_global_stage=int(cfg.get("cross_scale_global_stage", 3)),
        cross_scale_kv_tokens=int(cfg.get("cross_scale_kv_tokens", 64)),
        cross_scale_num_heads=int(cfg.get("cross_scale_num_heads", 4)),
        cross_scale_dropout=float(cfg.get("cross_scale_dropout", 0.0)),
        cross_scale_residual_scale_init=float(
            cfg.get("cross_scale_residual_scale_init", 0.01)
        ),
    )

    if config.model_type == "iqumamba_stage299_cross_scale":
        model = Stage301ComplexStateCrossScale(
            **common,
            **state,
            **cross_scale,
        )
    elif config.model_type == "bimamba_stage298_complex_bottleneck":
        model = Stage302BiFRESHComplexBottleneck(
            **common,
            conv_op=nn.Conv1d,
            deep_supervision=config.deep_supervision,
            **state,
            **fresh,
        )
    elif config.model_type == "bimamba_complex_stem_cross_scale":
        model = Stage303ComplexStemBiMambaCrossScale(
            **common,
            conv_op=nn.Conv1d,
            deep_supervision=config.deep_supervision,
            complex_norm_eps=float(cfg.get("complex_norm_eps", 1e-6)),
            cross_scale_evidence_gate=bool(
                cfg.get("cross_scale_evidence_gate", False)
            ),
            **cross_scale,
        )
    elif config.model_type == "stage298_stage299_output_fusion":
        stage298 = IQUBiMamba1D_EstimatedCycloFRESH(
            **common,
            conv_op=nn.Conv1d,
            deep_supervision=False,
            complex_stem_enable=True,
            complex_norm_eps=state["complex_norm_eps"],
            **fresh,
        )
        stage299 = IQUMamba1DComplexStateMamba(
            **common,
            **state,
            complex_stem_enable=True,
        )
        model = Stage304Stage298299Fusion(
            stage298=stage298,
            stage299=stage299,
            fusion_logit_init=float(cfg.get("fusion_logit_init", 0.0)),
        )
    elif config.model_type == "iqumamba_stage299_gated_fresh":
        model = Stage305GatedFRESHComplexState(
            **common,
            **state,
            **fresh,
            fresh_gate_logit_init=float(
                cfg.get("fresh_gate_logit_init", -3.0)
            ),
        )
    else:
        raise ValueError(f"Unsupported combined model type: {config.model_type}")
    return model.to(device)


def _create_kutii_learnable_dilation_wavenet_model(config: MambaConfig):
    """Build the transparent reproduction of the public KU-TII entry."""

    from models.kutii_learnable_dilation_wavenet import (
        KUTIIStyleLearnableDilationWaveNet,
    )

    return KUTIIStyleLearnableDilationWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 256)),
        residual_layers=int(getattr(config, "residual_layers", 30)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        max_dilation=int(getattr(config, "max_dilation", 1024)),
    ).to(device)


def _create_kutii_dual_source_wavenet_model(config: MambaConfig):
    """Build Stage 378's source-slot KU-TII-style WaveNet."""

    from models.kutii_learnable_dilation_wavenet import KUTIIDualSourceWaveNet

    return KUTIIDualSourceWaveNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        residual_channels=int(getattr(config, "residual_channels", 256)),
        residual_layers=int(getattr(config, "residual_layers", 30)),
        dilation_cycle_length=int(getattr(config, "dilation_cycle_length", 10)),
        max_dilation=int(getattr(config, "max_dilation", 1024)),
        enhanced_stage381=bool(
            config.model_config.get("enhanced_stage381", False)
        ),
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
