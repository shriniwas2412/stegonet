"""Image quality metrics computed on 8-bit images, exactly as saved to disk."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """PSNR in dB between two uint8 HxWx3 images (inf if identical)."""
    return float(peak_signal_noise_ratio(reference, test, data_range=255))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(reference, test, channel_axis=2, data_range=255))


def apd(reference: np.ndarray, test: np.ndarray) -> float:
    """Average absolute pixel difference on the 0-255 scale."""
    return float(np.mean(np.abs(reference.astype(np.float32) - test.astype(np.float32))))
