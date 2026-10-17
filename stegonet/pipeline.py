"""High-level helpers: load the trained model and hide/reveal PIL images."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .models import StegoNet

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "stegonet.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights=DEFAULT_WEIGHTS, device=None) -> StegoNet:
    """Load trained weights. Fails loudly if any weight is missing or extra."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(weights, map_location=device, weights_only=True)
    model = StegoNet()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def fit_size(image: Image.Image, max_side: int = 512) -> Image.Image:
    """Shrink large images and crop sides to a multiple of 4 (needed by the U-Net)."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    w, h = image.size
    return image.crop((0, 0, w - w % 4, h - h % 4))


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8))


@torch.no_grad()
def hide(model: StegoNet, cover: Image.Image, secret: Image.Image) -> Image.Image:
    """Hide ``secret`` in ``cover``. The secret is resized to the cover's size."""
    cover = fit_size(cover)
    secret = secret.convert("RGB").resize(cover.size, Image.LANCZOS)
    device = next(model.parameters()).device
    stego = model.hide(to_tensor(cover).to(device), to_tensor(secret).to(device))
    return to_image(stego)


@torch.no_grad()
def reveal(model: StegoNet, stego: Image.Image) -> Image.Image:
    device = next(model.parameters()).device
    return to_image(model.reveal(to_tensor(fit_size(stego, max_side=10**6)).to(device)))
