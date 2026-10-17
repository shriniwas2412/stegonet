"""Streamlit web app:  streamlit run app.py"""

import io
import socket
import tempfile
import threading
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from generate_certificates import generate
from stegonet import lsb, transfer
from stegonet.metrics import psnr, ssim
from stegonet.pipeline import DEFAULT_WEIGHTS, fit_size, hide, load_model, reveal

SAMPLES = Path(__file__).resolve().parent / "assets" / "samples"

st.set_page_config(page_title="StegoNet", page_icon="🕵️", layout="wide")


@st.cache_resource
def get_model():
    return load_model(DEFAULT_WEIGHTS)


@st.cache_resource
def get_receiver():
    """Start one local mutual-TLS receiver with throwaway certificates."""
    certs = generate(Path(tempfile.mkdtemp(prefix="stegonet-certs-")))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ready = threading.Event()
    threading.Thread(target=transfer.serve, daemon=True,
                     args=(get_model(), "127.0.0.1", port, tempfile.mkdtemp(prefix="stegonet-inbox-"), certs),
                     kwargs={"ready": ready}).start()
    ready.wait(10)
    return port, certs


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def pick_image(label: str, default: str, key: str) -> Image.Image:
    upload = st.file_uploader(label, type=["png", "jpg", "jpeg", "bmp"], key=key)
    image = Image.open(upload) if upload else Image.open(SAMPLES / default)
    return image.convert("RGB")


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(a.astype(int) - b.astype(int)) * 10, 0, 255).astype(np.uint8)


st.title("🕵️ StegoNet")
st.caption("Hide an entire image inside another with a neural network, then deliver it over mutual TLS.")

if not DEFAULT_WEIGHTS.exists():
    st.error(f"Model weights not found at `{DEFAULT_WEIGHTS}`. Train with `python train.py`.")
    st.stop()

model = get_model()
# A radio keeps the selected page across reruns (st.tabs resets to the first tab on every button click).
page = st.radio("Page", ["Hide", "Reveal", "Secure transfer", "How it works"], horizontal=True,
                label_visibility="collapsed")
st.divider()

if page == "Hide":
    left, right = st.columns(2)
    with left:
        cover = fit_size(pick_image("Cover image (what everyone sees)", "cover.png", "cover"))
        st.image(cover, caption="Cover", width=256)
    with right:
        secret = pick_image("Secret image (what you want to hide)", "secret.png", "secret")
        st.image(secret, caption="Secret", width=256)
    method = st.radio("Method", ["StegoNet (neural network)", "4-bit LSB (classical baseline)"], horizontal=True)

    secret_fit = secret.resize(cover.size, Image.LANCZOS)
    c, s = np.asarray(cover), np.asarray(secret_fit)
    if method.startswith("StegoNet"):
        stego_img = hide(model, cover, secret_fit)
        revealed = np.asarray(reveal(model, stego_img))
    else:
        stego_img = Image.fromarray(lsb.hide(c, s, 4))
        revealed = lsb.reveal(np.asarray(stego_img), 4)
    g = np.asarray(stego_img)
    st.session_state["stego"] = stego_img

    cols = st.columns(4)
    cols[0].image(cover, caption="Cover", use_container_width=True)
    cols[1].image(stego_img, caption="Stego (cover + hidden secret)", use_container_width=True)
    cols[2].image(difference(g, c), caption="|Stego − Cover| × 10", use_container_width=True)
    cols[3].image(revealed, caption="Revealed from stego only", use_container_width=True)

    m = st.columns(4)
    m[0].metric("Cover PSNR", f"{psnr(c, g):.2f} dB")
    m[1].metric("Cover SSIM", f"{ssim(c, g):.4f}")
    m[2].metric("Secret PSNR", f"{psnr(s, revealed):.2f} dB")
    m[3].metric("Secret SSIM", f"{ssim(s, revealed):.4f}")
    st.download_button("Download stego PNG", png_bytes(stego_img), "stego.png", "image/png")
    st.caption("Save as PNG. Lossy formats like JPEG destroy the hidden image.")

elif page == "Reveal":
    upload = st.file_uploader("Stego image made with StegoNet (PNG)", type=["png"], key="stego_up")
    if upload:
        stego_up = Image.open(upload).convert("RGB")
        a, b, _ = st.columns([1, 1, 2])
        a.image(stego_up, caption="Stego", use_container_width=True)
        b.image(reveal(model, stego_up), caption="Revealed secret", use_container_width=True)

elif page == "Secure transfer":
    st.markdown(
        "Sends the stego image from the **Hide** page to a local receiver over **mutual TLS**. "
        "Only the stego image crosses the network; the receiver reveals the secret on its side.")
    if st.button("Send stego image", type="primary"):
        port, certs = get_receiver()
        stego_img = st.session_state.get("stego")
        if stego_img is None:
            st.warning("Create a stego image on the Hide page first.")
            st.stop()
        receipt = transfer.send(stego_img, "127.0.0.1", port, certs)
        if receipt["intact"]:
            st.success(f"Delivered over TLS. The receiver authenticated client `{receipt['peer']}` "
                       f"and the SHA-256 of the received pixels matches.")
        else:
            st.error("Integrity check failed.")
        a, b, _ = st.columns([1, 1, 2])
        a.image(stego_img, caption="Sent", use_container_width=True)
        b.image(Image.open(receipt["revealed_path"]), caption="Revealed by the receiver",
                use_container_width=True)
        st.json(receipt)

else:
    st.markdown(Path(__file__).with_name("docs").joinpath("how_it_works.md").read_text())
