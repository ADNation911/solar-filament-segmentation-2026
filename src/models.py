"""
Multi-Backbone U-Net Architectures, Compound Losses (Dice, Asymmetric Tversky, Lovasz),
and 8-Way Dihedral (D4) Logit-Space Test-Time Augmentation.
"""
from typing import Dict, List, Optional
import numpy as np
from scipy.special import expit, logit
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None


class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice loss for continuous foreground overlap optimization."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class AsymmetricTverskyLoss(nn.Module):
    """
    Recall-weighted Asymmetric Tversky loss (alpha=0.3, beta=0.7) that penalizes
    False Negatives 2.33x more heavily than False Positives to preserve faint filament spines.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class LovaszHingeLoss(nn.Module):
    """Submodular Lovasz hinge surrogate for direct continuous Jaccard (IoU) optimization."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)
        signs = 2.0 * targets_flat - 1.0
        errors = 1.0 - logits_flat * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = targets_flat[perm]
        grad = self._lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)


class ApexCompoundLoss(nn.Module):
    """Combined BCE + Soft Dice / Asymmetric Tversky / Lovasz loss."""

    def __init__(self, mode: str = "dice", alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.mode = mode
        self.dice = SoftDiceLoss()
        self.tversky = AsymmetricTverskyLoss(alpha=alpha, beta=beta)
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        if self.mode == "tversky":
            return 0.5 * bce + 0.5 * self.tversky(logits, targets)
        if self.mode == "lovasz":
            return 0.5 * bce + 0.5 * self.lovasz(logits, targets)
        return 0.5 * bce + 0.5 * self.dice(logits, targets)


def build_segmentation_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = None,
    in_channels: int = 1,
    classes: int = 1,
) -> nn.Module:
    """
    Constructs a U-Net architecture with the specified backbone:
      - 'resnet34'
      - 'timm-efficientnet-b3' / 'efficientnet-b3'
      - 'densenet121'
    """
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required to build U-Net backbones.")
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


@torch.no_grad()
def predict_d4_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluates full 8-way dihedral group (D4) test-time augmentation (4 rotations x 2 flips)
    and returns the calibrated posterior foreground probability map.
    """
    model.eval()
    prob_sum = torch.zeros_like(x)
    for k in range(4):
        x_rot = torch.rot90(x, k, dims=(2, 3))
        p_rot = torch.sigmoid(model(x_rot))
        prob_sum += torch.rot90(p_rot, -k, dims=(2, 3))

        x_flip = torch.flip(x_rot, dims=(3,))
        p_flip = torch.sigmoid(model(x_flip))
        p_unflip = torch.flip(p_flip, dims=(3,))
        prob_sum += torch.rot90(p_unflip, -k, dims=(2, 3))

    return prob_sum / 8.0


def logit_space_ensemble(
    prob_maps: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Blends multi-backbone probability predictions in uncompressed log-odds (logit) space:
      P_grand = sigmoid( sum_m w_m * logit(P_m) )
    Default weights correspond to the 20-model 4-architecture Grand Ensemble:
      - ResNet-34 (Dice+BCE):       0.20
      - EfficientNet-B3 (Dice+BCE): 0.35
      - EfficientNet-B3 (Tversky):  0.30
      - DenseNet-121 (Lovasz):      0.15
    """
    if weights is None:
        weights = {
            "resnet34_dice": 0.20,
            "effnet_b3_dice": 0.35,
            "effnet_b3_tversky": 0.30,
            "densenet121_lovasz": 0.15,
        }
    total_w = 0.0
    logit_acc = None
    for name, p_map in prob_maps.items():
        w = weights.get(name, 1.0 / len(prob_maps))
        p_clipped = np.clip(p_map.astype(np.float32), eps, 1.0 - eps)
        l_map = logit(p_clipped)
        if logit_acc is None:
            logit_acc = w * l_map
        else:
            logit_acc += w * l_map
        total_w += w
    return expit(logit_acc / max(total_w, 1e-8)).astype(np.float32)
