"""Create a private CA plus server and client certificates for mutual TLS.

    python generate_certificates.py            # writes ./certificates
    python generate_certificates.py --out DIR

The server certificate is valid for ``localhost`` and ``127.0.0.1``; add more
names with ``--host``. Private keys are written with 0600 permissions and are
git-ignored. Never commit them.
"""

import argparse
import datetime
import ipaddress
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _name(common_name: str) -> x509.Name:
    return x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "StegoNet"),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])


def _write_key(path: Path, key) -> None:
    path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    os.chmod(path, 0o600)


def _issue(subject, key, issuer, issuer_key, days, *, ca=False, usage=None, sans=None):
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if ca:
        builder = builder.add_extension(x509.KeyUsage(
            digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
            key_encipherment=False, data_encipherment=False, key_agreement=False,
            encipher_only=False, decipher_only=False), critical=True)
    if usage:
        builder = builder.add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return builder.sign(issuer_key, hashes.SHA256())


def _san(host: str):
    try:
        return x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        return x509.DNSName(host)


def generate(out_dir, hosts=("localhost", "127.0.0.1"), days: int = 365) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name("StegoNet Root CA")
    ca_cert = _issue(ca_name, ca_key, ca_name, ca_key, days, ca=True)

    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = _issue(_name(hosts[0]), server_key, ca_name, ca_key, days,
                         usage=ExtendedKeyUsageOID.SERVER_AUTH, sans=[_san(h) for h in hosts])

    client_key = ec.generate_private_key(ec.SECP256R1())
    client_cert = _issue(_name("stegonet-client"), client_key, ca_name, ca_key, days,
                         usage=ExtendedKeyUsageOID.CLIENT_AUTH)

    for name, cert in [("ca", ca_cert), ("server", server_cert), ("client", client_cert)]:
        (out / f"{name}.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_key(out / "server.key", server_key)
    _write_key(out / "client.key", client_key)
    # The CA key is only needed to issue new certificates; it isn't written to disk.
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="certificates")
    parser.add_argument("--host", action="append", help="extra hostname or IP for the server certificate")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    hosts = ["localhost", "127.0.0.1"] + (args.host or [])
    out = generate(args.out, hosts, args.days)
    print(f"Wrote ca.crt, server.crt/key and client.crt/key to {out}/ (valid for {', '.join(hosts)})")


if __name__ == "__main__":
    main()
