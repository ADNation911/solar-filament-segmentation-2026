"""
Solar H-alpha Preprocessing, Limb Darkening Correction, and Positive-Biased Sampling.
"""
from typing import Dict, List, Optional, Tuple
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def detect_solar_disk(img_u8: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimates solar disk center (cx, cy) and physical radius R from a 2048x2048
    GONG H-alpha filtergram using Gaussian smoothing and Otsu thresholding.
    """
    h, w = img_u8.shape[:2]
    blurred = cv2.GaussianBlur(img_u8, (9, 9), 2.0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return w / 2.0, h / 2.0, min(h, w) * 0.45
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return float(cx), float(cy), float(radius)


def limb_darkening_correction(
    img: np.ndarray,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    radius: Optional[float] = None,
    u1: float = 0.62,
    u2: float = -0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies radial quadratic limb darkening correction (LDC) across the solar disk:
        I_corr(r) = I(r) / (1 - u1*(1 - mu) - u2*(1 - mu)^2),  mu = sqrt(1 - (r/R)^2)
    Returns the corrected float32 image and boolean chromospheric disk mask.
    """
    h, w = img.shape[:2]
    if cx is None or cy is None or radius is None:
        cx, cy, radius = detect_solar_disk(img)
    y, x = np.ogrid[:h, :w]
    r2 = ((x - cx) ** 2 + (y - cy) ** 2) / max(radius ** 2, 1.0)
    disk_mask = r2 <= 0.985
    mu = np.sqrt(np.clip(1.0 - np.minimum(r2, 0.985), 1e-4, 1.0))
    profile = np.clip(1.0 - u1 * (1.0 - mu) - u2 * ((1.0 - mu) ** 2), 0.35, 1.0)
    corrected = (img.astype(np.float32) / profile) * disk_mask.astype(np.float32)
    return corrected, disk_mask


def normalize_solar_disk(
    img: np.ndarray,
    apply_ldc: bool = False,
    clip_sigma: float = 6.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalizes on-disk H-alpha intensities to zero mean and unit variance while
    suppressing off-limb sky background noise.
    """
    cx, cy, radius = detect_solar_disk(img)
    if apply_ldc:
        work_img, disk_mask = limb_darkening_correction(img, cx, cy, radius)
    else:
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        disk_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= ((radius * 0.985) ** 2)
        work_img = img.astype(np.float32)

    disk_pixels = work_img[disk_mask & (img > 10)]
    if disk_pixels.size < 1000:
        disk_pixels = work_img.ravel()
    mean = float(disk_pixels.mean())
    std = float(disk_pixels.std() + 1e-4)
    norm = np.clip((work_img - mean) / std, -clip_sigma, clip_sigma)
    norm[~disk_mask] = 0.0
    return norm.astype(np.float32), {"cx": cx, "cy": cy, "radius": radius, "mean": mean, "std": std}


class PositiveBiasedSolarDataset(Dataset):
    """
    PyTorch Dataset for full-disk 2048x2048 H-alpha observations that samples
    75% of training crops centered on filament centroids (with spatial jitter)
    and 25% uniformly across the solar disk to overcome extreme (<0.8%) foreground sparsity.
    """

    def __init__(
        self,
        stems: List[str],
        images: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        crop_size: int = 1024,
        crops_per_image: int = 2,
        p_pos: float = 0.75,
        transform=None,
    ):
        self.stems = stems
        self.images = images
        self.targets = targets
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.p_pos = p_pos
        self.transform = transform

        self.stats: Dict[str, Tuple[float, float]] = {}
        self.centers: Dict[str, List[Tuple[int, int]]] = {}
        for s in stems:
            im = images[s]
            disk = im[im > 10]
            self.stats[s] = (float(disk.mean()), float(disk.std() + 1e-3)) if disk.size > 0 else (128.0, 32.0)
            tgt = targets[s]
            contours, _ = cv2.findContours((tgt > 30).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.centers[s] = [
                (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2,
                 cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2)
                for c in contours if cv2.contourArea(c) >= 50
            ]

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx % len(self.stems)]
        img = self.images[stem]
        tgt = self.targets[stem]
        h, w = img.shape[:2]
        c = self.crop_size
        bxs = self.centers[stem]

        if bxs and random.random() < self.p_pos:
            xc, yc = random.choice(bxs)
            dx = random.randint(-c // 4, c // 4)
            dy = random.randint(-c // 4, c // 4)
            x0 = int(np.clip(xc + dx - c // 2, 0, w - c))
            y0 = int(np.clip(yc + dy - c // 2, 0, h - c))
        else:
            x0 = random.randint(0, w - c)
            y0 = random.randint(0, h - c)

        im_crop = img[y0 : y0 + c, x0 : x0 + c]
        tg_crop = tgt[y0 : y0 + c, x0 : x0 + c]

        if self.transform is not None:
            augmented = self.transform(image=im_crop, mask=tg_crop)
            im_crop, tg_crop = augmented["image"], augmented["mask"]

        m, s = self.stats[stem]
        x_norm = np.clip((im_crop.astype(np.float32) - m) / s, -6.0, 6.0)[None, ...]
        y_norm = (tg_crop.astype(np.float32) / 255.0)[None, ...]
        return torch.from_numpy(x_norm), torch.from_numpy(y_norm)
