"""Classical least-significant-bit (LSB) steganography, used as a baseline.

The top ``bits`` bits of each secret pixel replace the bottom ``bits`` bits of
the cover pixel. This hides an image of the same size as the cover, like
StegoNet, but it's trivially detectable by statistical steganalysis.
"""

import numpy as np


def hide(cover: np.ndarray, secret: np.ndarray, bits: int = 4) -> np.ndarray:
    if cover.shape != secret.shape:
        raise ValueError("cover and secret must have the same shape")
    keep = (0xFF << bits) & 0xFF
    return (cover & keep) | (secret >> (8 - bits))


def reveal(stego: np.ndarray, bits: int = 4) -> np.ndarray:
    low = stego & ((1 << bits) - 1)
    # Shift back up and fill the lost bits with their mid value to reduce error.
    return (low << (8 - bits)) | (1 << (7 - bits))
