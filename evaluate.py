"""Evaluate StegoNet against the 4-bit LSB baseline on held-out test images.

    python evaluate.py --data dataset/test --pairs 500

Every number is computed on 8-bit images exactly as they would be saved and
transmitted. Writes ``results/metrics.json`` and ``results/samples.png``.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from stegonet import lsb
from stegonet.data import list_images
from stegonet.metrics import apd, psnr, ssim
from stegonet.pipeline import hide, load_model, pick_device, reveal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path, size):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    image = image.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    return image.resize((size, size), Image.LANCZOS)


def summarize(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in keys}


def score(cover, stego, secret, revealed):
    return {
        "cover_psnr": psnr(cover, stego), "cover_ssim": ssim(cover, stego), "cover_apd": apd(cover, stego),
        "secret_psnr": psnr(secret, revealed), "secret_ssim": ssim(secret, revealed),
        "secret_apd": apd(secret, revealed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset/test")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--weights", default="weights/stegonet.pt")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    # Fixed random pairing so results are reproducible. The dataset contains the same
    # icon several times (clean and with different payloads), so skip near-identical pairs.
    files = list_images(args.data)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    model = load_model(args.weights, pick_device())

    rows = {"stegonet": [], "lsb4": []}
    samples = []
    for a, b in zip(order[0::2], order[1::2], strict=False):
        if len(rows["stegonet"]) >= args.pairs:
            break
        cover_img, secret_img = load(files[a], args.size), load(files[b], args.size)
        cover, secret = np.asarray(cover_img), np.asarray(secret_img)
        if apd(cover, secret) < 10:
            continue
        i = len(rows["stegonet"])

        stego = hide(model, cover_img, secret_img)
        revealed = np.asarray(reveal(model, stego))
        stego = np.asarray(stego)
        rows["stegonet"].append(score(cover, stego, secret, revealed))

        lsb_stego = lsb.hide(cover, secret, 4)
        rows["lsb4"].append(score(cover, lsb_stego, secret, lsb.reveal(lsb_stego, 4)))

        if i < 4:
            samples.append((cover, stego, secret, revealed))

    n = len(rows["stegonet"])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    report = {"pairs": n, "image_size": args.size, "methods": {k: summarize(v) for k, v in rows.items()}}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"{n} test pairs at {args.size}x{args.size}")
    print(f"{'method':10s} {'cover PSNR':>11s} {'cover SSIM':>11s} {'secret PSNR':>12s} {'secret SSIM':>12s}")
    for name, stats in report["methods"].items():
        print(f"{name:10s} {stats['cover_psnr']['mean']:11.2f} {stats['cover_ssim']['mean']:11.4f} "
              f"{stats['secret_psnr']['mean']:12.2f} {stats['secret_ssim']['mean']:12.4f}")

    titles = ["Cover", "Stego", "Difference ×10", "Secret", "Revealed"]
    fig, axes = plt.subplots(len(samples), 5, figsize=(12.5, 2.6 * len(samples)))
    for r, (cover, stego, secret, revealed) in enumerate(samples):
        diff = np.clip(np.abs(stego.astype(int) - cover.astype(int)) * 10, 0, 255).astype(np.uint8)
        for c, image in enumerate([cover, stego, diff, secret, revealed]):
            ax = axes[r, c]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=110)
    print(f"wrote {out / 'metrics.json'} and {out / 'samples.png'}")


if __name__ == "__main__":
    main()
