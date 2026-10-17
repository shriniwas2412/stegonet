import socket
import ssl
import threading

import numpy as np
import pytest
from PIL import Image

from generate_certificates import generate
from stegonet import channel, transfer
from stegonet.models import StegoNet


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def receiver(tmp_path):
    certs = generate(tmp_path / "certs")
    port = free_port()
    ready = threading.Event()
    thread = threading.Thread(
        target=transfer.serve, daemon=True,
        args=(StegoNet().eval(), "127.0.0.1", port, tmp_path / "inbox", certs),
        kwargs={"max_messages": 1, "ready": ready},
    )
    thread.start()
    assert ready.wait(10)
    return port, certs, thread


def test_send_over_mutual_tls(receiver):
    port, certs, thread = receiver
    stego = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    receipt = transfer.send(stego, "127.0.0.1", port, certs)
    thread.join(10)
    assert receipt["intact"]
    assert receipt["peer"] == "stegonet-client"
    assert Image.open(receipt["revealed_path"]).size == (32, 32)


def test_client_without_certificate_is_rejected(receiver):
    port, certs, _ = receiver
    ctx = ssl.create_default_context(cafile=str(certs / "ca.crt"))  # trusts server, presents no cert
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with pytest.raises((ssl.SSLError, ConnectionError, OSError)):
            with ctx.wrap_socket(raw, server_hostname="127.0.0.1") as conn:
                conn.sendall(b"x" * 16)
                conn.recv(1)  # TLS 1.3 reports the rejected client cert on first read
                raise ConnectionError("server closed without a response")


def test_server_from_another_ca_is_rejected(receiver, tmp_path):
    port, _, _ = receiver
    other = generate(tmp_path / "other")  # a different CA: e.g. a man-in-the-middle
    with pytest.raises(ssl.SSLCertVerificationError):
        channel.connect("127.0.0.1", port, other, timeout=5)
