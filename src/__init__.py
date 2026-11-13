"""
Apex Sovereign: Multi-Annotator Boundary-Calibrated Deep Learning Framework
Official Source Package for the Solar Filament Segmentation Challenge 2026 (Team Tempest).
"""

from .preprocessing import (
    detect_solar_disk,
    limb_darkening_correction,
    normalize_solar_disk,
    PositiveBiasedSolarDataset,
)
from .models import (
    SoftDiceLoss,
    AsymmetricTverskyLoss,
    LovaszHingeLoss,
    ApexCompoundLoss,
    build_segmentation_model,
    predict_d4_tta,
    logit_space_ensemble,
)
from .postprocess import (
    topological_grouping_decoupled,
    extract_component_features,
    GBDTComponentFilter,
    VennSDFBoundaryCalibrator,
)
from .evaluation import (
    rle_encode,
    rle_decode,
    compute_panoptic_quality,
    verify_submission_contract,
)

__all__ = [
    "detect_solar_disk",
    "limb_darkening_correction",
    "normalize_solar_disk",
    "PositiveBiasedSolarDataset",
    "SoftDiceLoss",
    "AsymmetricTverskyLoss",
    "LovaszHingeLoss",
    "ApexCompoundLoss",
    "build_segmentation_model",
    "predict_d4_tta",
    "logit_space_ensemble",
    "topological_grouping_decoupled",
    "extract_component_features",
    "GBDTComponentFilter",
    "VennSDFBoundaryCalibrator",
    "rle_encode",
    "rle_decode",
    "compute_panoptic_quality",
    "verify_submission_contract",
]
