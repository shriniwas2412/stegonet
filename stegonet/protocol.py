"""Wire format for sending images over a socket.

Each message is an 8-byte big-endian length followed by that many bytes of PNG
data. PNG is lossless, so the stego pixels arrive bit-for-bit intact. Unlike
pickle, decoding a PNG can't execute code sent by the peer, and the size cap
stops a peer from making us allocate unbounded memory.
"""

import io
import socket
import struct

from PIL import Image

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
HEADER = struct.Struct("!Q")


class ProtocolError(Exception):
    pass


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes; ``sock.recv`` alone may return fewer."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ProtocolError("connection closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def send_bytes(sock: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    sock.sendall(HEADER.pack(len(payload)) + payload)


def recv_bytes(sock: socket.socket) -> bytes:
    (length,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    if length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"peer announced {length} bytes, limit is {MAX_MESSAGE_BYTES}")
    return recv_exact(sock, length)


def send_image(sock: socket.socket, image: Image.Image) -> None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    send_bytes(sock, buf.getvalue())


def recv_image(sock: socket.socket) -> Image.Image:
    data = recv_bytes(sock)
    try:
        image = Image.open(io.BytesIO(data))
        if image.format != "PNG":
            raise ProtocolError("expected a PNG image")
        image.load()
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(f"invalid image: {exc}") from exc
    return image.convert("RGB")
