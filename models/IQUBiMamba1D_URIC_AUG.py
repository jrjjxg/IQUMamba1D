"""IQUBiMamba1D_URIC_AUG - stage-40 URIC model paired with lightweight RF augmentation.

This model intentionally keeps the separator architecture identical to
``IQUBiMamba1D_URIC`` so experiments remain directly comparable. The only
intended difference for this stage is the train-time data augmentation
configured in ``model_config_bimamba_uric_aug.yaml`` and wired up in
``main.py`` / ``data_loader.dataloader``.
"""

from __future__ import annotations

from models.IQUBiMamba1D_URIC import IQUBiMamba1D_URIC


class IQUBiMamba1D_URIC_AUG(IQUBiMamba1D_URIC):
    """URIC baseline exposed as a separate stage for augmentation experiments."""

    pass
