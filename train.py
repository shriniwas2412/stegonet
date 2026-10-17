"""Train StegoNet on a folder of images.

    python train.py --data dataset --epochs 20

Expects ``<data>/train`` and ``<data>/val`` folders of images. Saves the best
weights (by validation loss) to ``weights/stegonet.pt`` and the per-epoch
history to ``results/training_history.json``.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stegonet.data import PairDataset
from stegonet.models import StegoNet
from stegonet.pipeline import pick_device


def batch_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b, reduction="none").mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "cover_psnr": 0.0, "secret_psnr": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for cover, secret in loader:
            cover, secret = cover.to(device), secret.to(device)
            stego, revealed = model(cover, secret)
            loss = F.mse_loss(stego, cover) + beta * F.mse_loss(revealed, secret)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals["loss"] += loss.item()
            totals["cover_psnr"] += batch_psnr(stego.detach(), cover)
            totals["secret_psnr"] += batch_psnr(revealed.detach(), secret)
            n += 1
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.75, help="weight of the reveal loss")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="weights/stegonet.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    print(f"Training on {device}")

    train_set = PairDataset(Path(args.data) / "train", args.size, train=True, limit=args.train_limit)
    val_set = PairDataset(Path(args.data) / "val", args.size, train=False, limit=args.val_limit)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.workers)

    model = StegoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_stats = run_epoch(model, train_loader, device, args.beta, optimizer)
        val_stats = run_epoch(model, val_loader, device, args.beta)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(f"epoch {epoch:3d} | {time.time() - start:5.0f}s | "
              f"val loss {val_stats['loss']:.5f} | cover PSNR {val_stats['cover_psnr']:.2f} dB | "
              f"secret PSNR {val_stats['secret_psnr']:.2f} dB", flush=True)
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "image_size": args.size,
                        "val": val_stats}, out)
        Path("results").mkdir(exist_ok=True)
        Path("results/training_history.json").write_text(json.dumps(history, indent=2))

    print(f"Best validation loss {best:.5f}; weights saved to {out}")


if __name__ == "__main__":
    main()
