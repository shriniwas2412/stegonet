## How it works

```
  SENDER                                              RECEIVER

  cover  ─┐                                       ┌──────────────┐
          ├─▶ Hiding U-Net ─▶ stego.png ─▶ mTLS ─▶│ Reveal CNN   │─▶ revealed secret
  secret ─┘                                       │ (stego only) │
                                                  └──────────────┘
```

**1. Hiding network.** A small U-Net receives the cover and the secret (6 input channels) and
predicts a residual. The stego image is `cover + residual`, so the network only has to learn a
small, hard-to-see perturbation that encodes the secret.

**2. Quantization.** Real images are stored with 8 bits per channel. During training the stego
image is rounded to 8-bit levels (with a straight-through gradient), so the reveal network learns
to decode exactly what a saved PNG contains.

**3. Reveal network.** A 6-layer CNN that receives *only the stego image*. It never sees the cover
or the secret, so everything it recovers must have been carried in the stego pixels.

**4. Training objective.** Both networks are trained jointly:

`loss = MSE(stego, cover) + β · MSE(revealed, secret)` with `β = 0.75`

The first term keeps the stego image close to the cover; the second forces the secret to survive.

**5. Transport.** Only the stego PNG travels over the network. The channel uses TLS 1.2+ with
**mutual authentication**: both sides hold certificates from a private CA, the client checks the
server's certificate and hostname, and the server rejects clients without a valid certificate.
Images are sent as length-prefixed PNG bytes (never pickle), with a size cap. The receiver returns a
SHA-256 of the pixels it got, so the sender can confirm they arrived bit-for-bit intact.

**Capacity.** The secret is the same size as the cover, so the payload is 24 bits per pixel. That's
far more than classical methods hide without visible damage.

**Limits.** The hidden image doesn't survive lossy compression (JPEG), resizing or heavy editing.
Training to resist these is listed as future work in the README.
