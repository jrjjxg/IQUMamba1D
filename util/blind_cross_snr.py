"""Architecture-agnostic blind cross-SNR distillation configuration."""


def apply_blind_cross_snr_profile(args):
    """Attach cross-SNR distillation without physical-parameter supervision.

    The selected stage remains the student/teacher backbone.  A frozen teacher
    therefore uses a checkpoint from the same stage and no extra model stage is
    registered.
    """

    if not bool(getattr(args, "blind_cross_snr_distill", False)):
        return args

    checkpoint = getattr(args, "cross_snr_teacher_checkpoint", None)
    teacher_mode = getattr(args, "cross_snr_teacher_mode", None)
    if teacher_mode is None:
        teacher_mode = "frozen" if checkpoint else "ema"
    teacher_mode = str(teacher_mode).lower()
    if teacher_mode not in {"ema", "frozen"}:
        raise ValueError("blind cross-SNR teacher mode must be 'ema' or 'frozen'")
    if teacher_mode == "frozen" and not checkpoint:
        raise ValueError(
            "Blind frozen cross-SNR distillation requires "
            "--cross_snr_teacher_checkpoint from the same backbone stage"
        )

    args.cross_snr_enable = True
    args.cross_snr_teacher_mode = teacher_mode
    args.cross_snr_ema_teacher_enable = teacher_mode == "ema"
    args.cross_snr_teacher_view = "clean"
    args.cross_snr_pair_mode = "curriculum_student"
    args.cross_snr_shared_permutation = teacher_mode == "ema"

    defaults = {
        "cross_snr_probability": 1.0,
        "cross_snr_pair_weight": 0.5,
        "cross_snr_consistency_weight": 0.1,
        "cross_snr_consistency_beta": 0.5,
        "cross_snr_feature_consistency_weight": 0.0,
        "cross_snr_feature_consistency_beta": 0.5,
        "cross_snr_eps": 1.0e-8,
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    # These are forced off even when the selected stage config enables them.
    # Clean/noisy views and clean source targets are the only extra supervision.
    args.sync_snr_aux_weight = 0.0
    args.sync_cross_snr_consistency_weight = 0.0
    args.sync_physical_supervision_weight = 0.0
    args.sync_physical_require_metadata = False
    args.sync_metadata_enable = False
    return args


__all__ = ["apply_blind_cross_snr_profile"]
