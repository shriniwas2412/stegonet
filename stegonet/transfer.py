"""Sender and receiver for stego images over the mutual-TLS channel.

Only the stego image crosses the network. The receiver runs the reveal network
locally, so neither the cover nor the secret is ever transmitted.
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from PIL import Image

from . import channel, protocol
from .pipeline import reveal

log = logging.getLogger("stegonet")


def _png_digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def send(stego: Image.Image, host: str, port: int, cert_dir=channel.CERT_DIR) -> dict:
    """Send a stego image and return the receiver's receipt."""
    with channel.connect(host, port, cert_dir) as sock:
        protocol.send_image(sock, stego)
        receipt = json.loads(protocol.recv_bytes(sock))
    receipt["intact"] = receipt.get("pixel_sha256") == _png_digest(stego.convert("RGB"))
    return receipt


def handle(conn, model, out_dir: Path) -> dict:
    stego = protocol.recv_image(conn)
    secret = reveal(model, stego)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stego_path = out_dir / f"{stamp}_stego.png"
    secret_path = out_dir / f"{stamp}_revealed.png"
    stego.save(stego_path)
    secret.save(secret_path)
    receipt = {
        "pixel_sha256": _png_digest(stego),
        "size": list(stego.size),
        "revealed_path": str(secret_path),
        "peer": conn.getpeercert()["subject"][-1][0][1],
    }
    protocol.send_bytes(conn, json.dumps(receipt).encode())
    log.info("received %sx%s stego from %s -> %s", *stego.size, receipt["peer"], secret_path)
    return receipt


def serve(model, host: str, port: int, out_dir="received", cert_dir=channel.CERT_DIR,
          max_messages: int | None = None, ready: threading.Event | None = None) -> None:
    """Accept stego images until ``max_messages`` have been handled (forever if None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listener, ctx = channel.listen(host, port, cert_dir)
    log.info("receiver listening on %s:%s (mutual TLS)", host, port)
    if ready:
        ready.set()
    handled = 0
    with listener:
        while max_messages is None or handled < max_messages:
            raw, addr = listener.accept()
            raw.settimeout(30)
            try:
                with ctx.wrap_socket(raw, server_side=True) as conn:
                    handle(conn, model, out_dir)
                    handled += 1
            except Exception as exc:  # a bad client must not stop the receiver
                log.warning("rejected connection from %s: %s", addr, exc)
                raw.close()

