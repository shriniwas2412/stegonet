"""Build ./dataset from the Kaggle Stego-Images download.

    python prepare_data.py --source archive --out dataset

Download and unzip the dataset into ``archive/`` first (it contains
``train/train``, ``val/val`` and ``test/test`` folders). Images are converted
to RGB, resized to 256x256 and written to ``dataset/{train,val,test}``.
Train and val use the ``clean`` and ``stego`` folders. The test split also ships
``stego_zip`` and ``stego_b64`` variants of the same images; the published results
in README.md were produced with ``stego_zip`` (the default for ``--test-stego``).
"""

import argparse
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="archive")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--test-stego", default="stego_zip", choices=["stego", "stego_zip", "stego_b64"])
    args = parser.parse_args()

    for split in SPLITS:
        out = Path(args.out) / split
        out.mkdir(parents=True, exist_ok=True)
        count = 0
        for sub in ("clean", args.test_stego if split == "test" else "stego"):
            folder = Path(args.source) / split / split / sub
            if not folder.is_dir():
                raise SystemExit(f"Missing {folder}. Is the Kaggle archive unzipped into '{args.source}'?")
            for path in sorted(folder.glob("*.png")):
                image = Image.open(path).convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
                image.save(out / path.name)
                count += 1
        print(f"{split}: {count} images -> {out}")


if __name__ == "__main__":
    main()
