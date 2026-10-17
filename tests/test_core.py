import socket
import threading

import numpy as np
import pytest
import torch
from PIL import Image

from stegonet import lsb, protocol
from stegonet.models import StegoNet, quantize
from stegonet.pipeline import fit_size, hide, load_model, reveal


def random_image(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_model_shapes_and_range():
    model = StegoNet().eval()
    cover, secret = torch.rand(2, 3, 32, 40), torch.rand(2, 3, 32, 40)
    with torch.no_grad():
        stego, revealed = model(cover, secret)
    assert stego.shape == revealed.shape == cover.shape
    assert stego.min() >= 0 and stego.max() <= 1
    assert revealed.min() >= 0 and revealed.max() <= 1


def test_quantize_produces_8bit_levels_and_passes_gradient():
    x = torch.rand(10, requires_grad=True)
    q = quantize(x)
    assert torch.allclose(q * 255, torch.round(q * 255), atol=1e-4)
    q.sum().backward()
    assert torch.allclose(x.grad, torch.ones(10))


def test_reveal_only_sees_stego():
    # The reveal network takes a single 3-channel input: it can't peek at the secret.
    assert StegoNet().revealer.net[0][0].in_channels == 3


def test_load_model_is_strict(tmp_path):
    path = tmp_path / "w.pt"
    state = StegoNet().state_dict()
    torch.save({"state_dict": state}, path)
    load_model(path)
    state.pop(next(iter(state)))
    torch.save({"state_dict": state}, path)
    with pytest.raises(RuntimeError):
        load_model(path)


def test_pipeline_roundtrip_sizes():
    model = StegoNet().eval()
    cover = random_image(70, 50)
    stego = hide(model, cover, random_image(30, 30, seed=1))
    assert stego.size == (68, 48) == fit_size(cover).size
    assert reveal(model, stego).size == stego.size


def test_lsb_roundtrip_error_is_bounded():
    cover = np.asarray(random_image(seed=2))
    secret = np.asarray(random_image(seed=3))
    stego = lsb.hide(cover, secret, bits=4)
    assert np.abs(stego.astype(int) - cover.astype(int)).max() <= 15
    assert np.abs(lsb.reveal(stego, 4).astype(int) - secret.astype(int)).max() <= 8


def test_protocol_image_roundtrip():
    a, b = socket.socketpair()
    with a, b:
        image = random_image()
        # The socketpair buffer is smaller than the PNG, so send from another thread.
        sender = threading.Thread(target=protocol.send_image, args=(a, image))
        sender.start()
        received = protocol.recv_image(b)
        sender.join()
        assert np.array_equal(np.asarray(received), np.asarray(image))


def test_protocol_rejects_oversized_and_non_png():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(protocol.HEADER.pack(protocol.MAX_MESSAGE_BYTES + 1))
        with pytest.raises(protocol.ProtocolError):
            protocol.recv_bytes(b)
    a, b = socket.socketpair()
    with a, b:
        protocol.send_bytes(a, b"definitely not a png")
        with pytest.raises(protocol.ProtocolError):
            protocol.recv_image(b)
