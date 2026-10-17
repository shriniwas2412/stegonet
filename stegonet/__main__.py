"""Command-line interface.

    python -m stegonet hide    --cover cover.png --secret secret.png --out stego.png
    python -m stegonet reveal  --stego stego.png --out revealed.png
    python -m stegonet receive --port 5000                 # receiver (mutual TLS)
    python -m stegonet send    --stego stego.png --port 5000
    python -m stegonet demo    --cover cover.png --secret secret.png
"""

import argparse
import logging
import tempfile
import threading

from PIL import Image

from . import transfer
from .metrics import psnr, ssim
from .pipeline import DEFAULT_WEIGHTS, fit_size, hide, load_model, reveal


def _report(cover, stego, secret, revealed):
    import numpy as np
    c, s = np.asarray(cover.convert("RGB")), np.asarray(stego)
    print(f"cover  vs stego    : PSNR {psnr(c, s):6.2f} dB  SSIM {ssim(c, s):.4f}")
    if secret is not None:
        sec = np.asarray(secret.convert("RGB").resize(revealed.size, Image.LANCZOS))
        r = np.asarray(revealed)
        print(f"secret vs revealed : PSNR {psnr(sec, r):6.2f} dB  SSIM {ssim(sec, r):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m stegonet", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hide", help="hide a secret image inside a cover image")
    p.add_argument("--cover", required=True)
    p.add_argument("--secret", required=True)
    p.add_argument("--out", default="stego.png")

    p = sub.add_parser("reveal", help="recover the secret from a stego image")
    p.add_argument("--stego", required=True)
    p.add_argument("--out", default="revealed.png")

    p = sub.add_parser("receive", help="run the receiver: accept stego images over mTLS and reveal them")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--out-dir", default="received")

    p = sub.add_parser("send", help="send a stego image to a receiver over mTLS")
    p.add_argument("--stego", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)

    p = sub.add_parser("demo", help="hide, send over mTLS to a local receiver, reveal, and report metrics")
    p.add_argument("--cover", required=True)
    p.add_argument("--secret", required=True)
    p.add_argument("--port", type=int, default=5000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "send":
        receipt = transfer.send(Image.open(args.stego).convert("RGB"), args.host, args.port)
        print(f"delivered, integrity check {'passed' if receipt['intact'] else 'FAILED'}: {receipt}")
        return

    model = load_model(args.weights)

    if args.command == "hide":
        cover = Image.open(args.cover)
        stego = hide(model, cover, Image.open(args.secret))
        stego.save(args.out)
        print(f"stego image written to {args.out}")
        _report(fit_size(cover), stego, None, None)
    elif args.command == "reveal":
        reveal(model, Image.open(args.stego)).save(args.out)
        print(f"revealed secret written to {args.out}")
    elif args.command == "receive":
        transfer.serve(model, args.host, args.port, args.out_dir)
    elif args.command == "demo":
        cover, secret = Image.open(args.cover), Image.open(args.secret)
        stego = hide(model, cover, secret)
        ready = threading.Event()
        out_dir = tempfile.mkdtemp(prefix="stegonet-")
        receiver = threading.Thread(target=transfer.serve, daemon=True,
                                    args=(model, "127.0.0.1", args.port, out_dir),
                                    kwargs={"max_messages": 1, "ready": ready})
        receiver.start()
        ready.wait(10)
        receipt = transfer.send(stego, "127.0.0.1", args.port)
        receiver.join(10)
        revealed = Image.open(receipt["revealed_path"])
        print(f"sent {stego.size[0]}x{stego.size[1]} stego over mutual TLS; "
              f"integrity {'OK' if receipt['intact'] else 'FAILED'}; receiver saved files in {out_dir}")
        _report(fit_size(cover), stego, secret, revealed)


if __name__ == "__main__":
    main()
