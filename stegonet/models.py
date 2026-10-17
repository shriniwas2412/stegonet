"""Neural networks that hide a full RGB image inside another RGB image.

The design follows the hide/reveal split of Baluja, "Hiding Images in Plain
Sight: Deep Steganography" (NeurIPS 2017):

* ``HidingNetwork`` sees the cover and the secret and predicts a small residual
  that is added to the cover to produce the stego image.
* ``RevealNetwork`` sees *only* the stego image and reconstructs the secret.

Because the reveal network never receives the cover or the secret, the only way
for it to recover the secret is from information actually embedded in the stego
pixels. Both networks are fully convolutional, so they run on any image size
whose sides are divisible by 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class HidingNetwork(nn.Module):
    """Small U-Net: (cover, secret) -> stego = cover + residual."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1 = nn.Sequential(conv_block(6, w), conv_block(w, w))
        self.enc2 = nn.Sequential(conv_block(w, 2 * w), conv_block(2 * w, 2 * w))
        self.bottleneck = nn.Sequential(conv_block(2 * w, 4 * w), conv_block(4 * w, 4 * w))
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = nn.Sequential(conv_block(4 * w, 2 * w), conv_block(2 * w, 2 * w))
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = nn.Sequential(conv_block(2 * w + 6, w), conv_block(w, w))
        self.head = nn.Conv2d(w, 3, 1)

    def forward(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cover, secret], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1, x], dim=1))
        residual = self.head(d1)
        return (cover + residual).clamp(0.0, 1.0)


class RevealNetwork(nn.Module):
    """Plain CNN: stego -> secret. Never sees the cover or the secret."""

    def __init__(self, width: int = 64, depth: int = 6):
        super().__init__()
        layers = [conv_block(3, width)]
        layers += [conv_block(width, width) for _ in range(depth - 2)]
        layers += [nn.Conv2d(width, 3, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        return self.net(stego)


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Round to 8-bit levels with a straight-through gradient.

    Real stego images are saved as 8-bit PNGs, so the reveal network is trained
    on quantized pixels rather than on floating-point values it will never see.
    """
    return x + (torch.round(x * 255.0) / 255.0 - x).detach()


class StegoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hider = HidingNetwork()
        self.revealer = RevealNetwork()

    def hide(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        return self.hider(cover, secret)

    def reveal(self, stego: torch.Tensor) -> torch.Tensor:
        return self.revealer(stego)

    def forward(self, cover: torch.Tensor, secret: torch.Tensor):
        stego = quantize(self.hide(cover, secret))
        return stego, self.reveal(stego)
