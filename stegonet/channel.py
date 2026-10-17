"""Mutual-TLS channel between a sender and a receiver.

Both sides present certificates signed by the project's private CA (see
``generate_certificates.py``) and reject any peer that can't. This gives:

* confidentiality and integrity: TLS 1.2+ with forward-secret cipher suites
* server authentication: the client verifies the server's certificate and hostname
* client authentication: the server requires a client certificate (mTLS)

Steganography hides *that* a secret exists; TLS protects the stego image while
it travels, so an eavesdropper can't even get the carrier image to analyze.
"""

import socket
import ssl
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent.parent / "certificates"


def server_context(cert_dir=CERT_DIR) -> ssl.SSLContext:
    cert_dir = Path(cert_dir)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert_dir / "server.crt", cert_dir / "server.key")
    ctx.load_verify_locations(cert_dir / "ca.crt")
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def client_context(cert_dir=CERT_DIR) -> ssl.SSLContext:
    cert_dir = Path(cert_dir)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # verifies cert + hostname by default
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_verify_locations(cert_dir / "ca.crt")
    ctx.load_cert_chain(cert_dir / "client.crt", cert_dir / "client.key")
    return ctx


def connect(host: str, port: int, cert_dir=CERT_DIR, timeout: float = 30.0) -> ssl.SSLSocket:
    raw = socket.create_connection((host, port), timeout=timeout)
    try:
        return client_context(cert_dir).wrap_socket(raw, server_hostname=host)
    except Exception:
        raw.close()
        raise


def listen(host: str, port: int, cert_dir=CERT_DIR) -> tuple:
    """Return ``(listening_socket, ssl_context)``; wrap each accepted socket yourself
    so one failed handshake doesn't take the server down."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    return sock, server_context(cert_dir)
