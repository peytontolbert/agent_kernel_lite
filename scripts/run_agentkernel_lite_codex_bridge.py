#!/usr/bin/env python3
"""Local encrypted computer-use bridge for Agent Kernel Lite.

The bridge binds to 127.0.0.1 by default. Pairing uses a short code printed in
the terminal plus explicit local approval on the computer, then every
post-pairing message is encrypted with P-256 ECDH, HKDF-SHA256, and
AES-256-GCM. Use --host 0.0.0.0 only when pairing from another device on a
trusted LAN.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError:
    print("Missing dependency: cryptography", file=sys.stderr)
    print("Install it with: python -m pip install cryptography", file=sys.stderr)
    raise


PROTOCOL = "agent-kernel-computer-bridge/v1"
LEGACY_PROTOCOL = "agent-kernel-codex-bridge/v1"
MAX_JSON_BODY_BYTES = 128 * 1024
DEFAULT_ALLOWED_ORIGINS = {
    "https://peytontolbert.com",
    "http://localhost:8797",
    "http://127.0.0.1:8797",
}
ALLOWED_SANDBOXES = {"danger-full-access", "read-only", "workspace-write"}
ALLOWED_APPROVAL_POLICIES = {"never", "on-request"}
PROVIDER_ALIASES = {
    "codex": "codex",
    "openai_codex": "codex",
    "claude": "claude_code",
    "claude_code": "claude_code",
    "cursor": "cursor",
    "cursor_agent": "cursor",
}


def b64url_uint(value: int) -> str:
    raw = value.to_bytes(32, "big")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64url_to_int(value: str) -> int:
    padded = value + ("=" * (-len(value) % 4))
    return int.from_bytes(base64.urlsafe_b64decode(padded.encode("ascii")), "big")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(value: str) -> bytes:
    return base64.b64decode(str(value).encode("ascii"), validate=True)


def public_key_to_jwk(public_key: ec.EllipticCurvePublicKey) -> dict[str, str]:
    numbers = public_key.public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": b64url_uint(numbers.x),
        "y": b64url_uint(numbers.y),
        "ext": True,
    }


def public_key_from_jwk(jwk: dict[str, Any]) -> ec.EllipticCurvePublicKey:
    if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        raise ValueError("browser key must be an EC P-256 JWK")
    numbers = ec.EllipticCurvePublicNumbers(
        b64url_to_int(str(jwk["x"])),
        b64url_to_int(str(jwk["y"])),
        ec.SECP256R1(),
    )
    return numbers.public_key()


def load_or_create_private_key(path: Path) -> ec.EllipticCurvePrivateKey:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    private_key = ec.generate_private_key(ec.SECP256R1())
    path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return private_key


def request_origin(handler: BaseHTTPRequestHandler) -> str:
    return str(handler.headers.get("Origin") or "")


def add_bridge_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    origin = request_origin(handler)
    if not handler.state.origin_allowed(origin):
        return
    handler.send_header("Access-Control-Allow-Origin", origin)
    handler.send_header("Vary", "Origin")
    handler.send_header("Access-Control-Allow-Headers", "content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    if str(handler.headers.get("Access-Control-Request-Private-Network") or "").lower() == "true":
        handler.send_header("Access-Control-Allow-Private-Network", "true")


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    add_bridge_cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(raw)


def html_response(handler: BaseHTTPRequestHandler, status: int, html_text: str) -> None:
    raw = html_text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def options_response(handler: BaseHTTPRequestHandler, status: int = 204, payload: dict[str, Any] | None = None) -> None:
    raw = b"" if status == 204 else json.dumps(payload or {}, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    if status != 204:
        handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    add_bridge_cors_headers(handler)
    if handler.state.origin_allowed(request_origin(handler)):
        handler.send_header("Access-Control-Max-Age", "600")
    handler.end_headers()
    if raw:
        handler.wfile.write(raw)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    if length > MAX_JSON_BODY_BYTES:
        raise ValueError("request body is too large")
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: int = 35) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return int(response.status), json.loads(body) if body else {}
    except HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {"status": "error", "error": body or str(error)}
        return int(error.code), payload


def local_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
        probe.close()
        if address and not address.startswith("127."):
            addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


class BridgeState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.host = str(args.host or "127.0.0.1")
        self.port = int(args.port)
        self.config_dir = Path(args.config_dir).expanduser()
        self.private_key = load_or_create_private_key(self.config_dir / "bridge-device-key.pem")
        self.grants_path = self.config_dir / "pairing-grants.json"
        self.allowed_workspaces = [Path(item).expanduser().resolve() for item in args.workspace]
        if not self.allowed_workspaces:
            self.allowed_workspaces = [Path.cwd().resolve()]
        self.provider_bins = {
            "codex": shutil.which(args.codex_bin) or args.codex_bin,
            "claude_code": shutil.which(args.claude_bin) or args.claude_bin,
            "cursor": shutil.which(args.cursor_bin) or args.cursor_bin,
        }
        self.timeout = int(args.timeout)
        self.sandbox = str(args.sandbox)
        self.approval_policy = str(args.approval_policy)
        self.allowed_origins = set(DEFAULT_ALLOWED_ORIGINS)
        for origin in args.allow_origin:
            self.allowed_origins.add(str(origin).rstrip("/"))
        self.mobile_token = secrets.token_urlsafe(24)
        self.mobile_grants: dict[str, dict[str, Any]] = {}
        self.pairing: dict[str, dict[str, Any]] = {}
        self.pairing_approval_lock = threading.Lock()
        self.grants = self.load_grants()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.sessions_lock = threading.Lock()
        if self.sandbox not in ALLOWED_SANDBOXES:
            raise ValueError(f"unsupported sandbox: {self.sandbox}")
        if self.approval_policy not in ALLOWED_APPROVAL_POLICIES:
            raise ValueError(f"unsupported approval policy: {self.approval_policy}")

    def health_payload(self) -> dict[str, Any]:
        providers = self.provider_catalog()
        codex = next((provider for provider in providers if provider["id"] == "codex"), {})
        return {
            "status": "ok",
            "protocol": PROTOCOL,
            "legacy_protocol": LEGACY_PROTOCOL,
            "paired": bool(self.grants),
            "bridge_public_jwk": public_key_to_jwk(self.private_key.public_key()),
            "providers": providers,
            "codex_bin": codex.get("binary", ""),
            "codex_available": bool(codex.get("available")),
            "allowed_workspaces": [str(item) for item in self.allowed_workspaces],
            "workspace_policy": "selected workspace must equal an allowed root or be inside it",
            "sandbox": self.sandbox,
            "approval_policy": self.approval_policy,
        }

    def start_pairing_request(self, origin: str, body: dict[str, Any]) -> dict[str, Any]:
        origin = self.require_allowed_origin(origin)
        self.cleanup_pairing_requests()
        browser_public_jwk = body.get("browser_public_jwk")
        public_key_from_jwk(browser_public_jwk)
        pairing_id = f"pair_{secrets.token_urlsafe(12)}"
        code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = time.time() + 300
        self.pairing[pairing_id] = {
            "pairing_id": pairing_id,
            "code": code,
            "origin": origin,
            "browser_public_jwk": browser_public_jwk,
            "expires_at": expires_at,
            "attempts": 0,
        }
        print("", flush=True)
        print("Agent Kernel Lite computer-use pairing request", flush=True)
        print(f"Origin: {origin}", flush=True)
        print(f"Pairing code: {code}", flush=True)
        print(f"Browser key fingerprint: {self.pairing_fingerprint(self.pairing[pairing_id])}", flush=True)
        print("Enter this code in the Agent Kernel Lite app within 5 minutes.", flush=True)
        print("The computer must also approve the pairing after the code is entered.", flush=True)
        print("", flush=True)
        return {
            "status": "pairing_code_required",
            "pairing_id": pairing_id,
            "protocol": PROTOCOL,
            "origin": origin,
            "bridge_public_jwk": public_key_to_jwk(self.private_key.public_key()),
            "expires_at": expires_at,
            "code_length": 6,
        }

    def confirm_pairing_request(self, origin: str, body: dict[str, Any]) -> dict[str, Any]:
        origin = self.require_allowed_origin(origin)
        pairing_id = str(body.get("pairing_id") or "")
        code = str(body.get("code") or "").strip()
        request = self.pairing.get(pairing_id)
        if not request or float(request["expires_at"]) < time.time():
            raise ValueError("pairing request expired or missing")
        if origin != request.get("origin"):
            raise ValueError("pairing origin does not match request origin")
        request["attempts"] = int(request.get("attempts") or 0) + 1
        if int(request["attempts"]) > 5:
            self.pairing.pop(pairing_id, None)
            raise ValueError("too many pairing attempts")
        if not secrets.compare_digest(code, str(request["code"])):
            raise ValueError("pairing code did not match")
        self.approve_pairing_on_computer(request)
        grant_id = f"grant_{secrets.token_urlsafe(18)}"
        grant = {
            "grant_id": grant_id,
            "origin": request["origin"],
            "browser_public_jwk": request["browser_public_jwk"],
            "created_at": time.time(),
            "expires_at": time.time() + 60 * 60 * 24 * 30,
            "last_seq": 0,
        }
        self.grants[grant_id] = grant
        self.pairing.pop(pairing_id, None)
        self.save_grants()
        return {"status": "paired", "grant_id": grant_id, "expires_at": grant["expires_at"]}

    def encrypted_message_response(self, origin: str, envelope: dict[str, Any]) -> dict[str, Any]:
        origin = self.require_allowed_origin(origin)
        if envelope.get("protocol") not in {PROTOCOL, LEGACY_PROTOCOL}:
            raise ValueError("unsupported protocol")
        seq = int(envelope.get("seq") or 0)
        payload, grant = self.decrypt_message(envelope)
        if origin != grant.get("origin"):
            raise ValueError("request origin does not match pairing grant")
        message_type = str(payload.get("type") or "")
        if message_type in {"computer.session.start", "codex.session.start"}:
            result = self.start_codex_session(payload)
        elif message_type in {"computer.session.send", "codex.session.send"}:
            result = self.send_codex_followup(payload)
        elif message_type in {"computer.session.status", "codex.session.status"}:
            session_id = str(payload.get("session_id") or "")
            result = self.session_snapshot(session_id, int(payload.get("since") or 0)) if session_id else {
                "status": "ok",
                "message": "bridge is ready",
                "providers": self.provider_catalog(),
                "active_sessions": [
                    self.session_snapshot(session_id, 0)
                    for session_id in list(self.sessions.keys())
                    if self.sessions.get(session_id, {}).get("status") == "running"
                ],
            }
        elif message_type in {"computer.session.cancel", "codex.session.cancel"}:
            result = self.cancel_codex_session(payload)
        elif message_type in {"computer.diff.read", "codex.diff.read"}:
            result = self.read_diff(payload)
        elif message_type in {"computer.grant.revoke", "codex.grant.revoke"}:
            removed = self.grants.pop(str(grant["grant_id"]), None)
            self.save_grants()
            result = {"status": "revoked" if removed else "missing", "grant_id": grant["grant_id"]}
        else:
            raise ValueError(f"unsupported encrypted message type: {message_type}")
        return self.encrypt_response(grant, seq, {"type": f"{message_type}.result", "result": result})

    def handle_relay_request(self, request: dict[str, Any]) -> dict[str, Any]:
        path = str(request.get("path") or "")
        origin = str(request.get("origin") or "").rstrip("/")
        body = request.get("body") if isinstance(request.get("body"), dict) else {}
        try:
            if path == "/health":
                return {"status_code": 200, "payload": self.health_payload()}
            if path == "/pairing/start":
                return {"status_code": 200, "payload": self.start_pairing_request(origin, body)}
            if path == "/pairing/confirm":
                return {"status_code": 200, "payload": self.confirm_pairing_request(origin, body)}
            if path == "/v1/message":
                return {"status_code": 200, "payload": self.encrypted_message_response(origin, body)}
            raise ValueError("not found")
        except Exception as exc:
            return {"status_code": 400, "payload": {"status": "error", "error": str(exc)}}

    def normalize_provider(self, provider: str) -> str:
        normalized = PROVIDER_ALIASES.get(str(provider or "codex").strip().lower().replace("-", "_"), "")
        if not normalized:
            raise ValueError(f"unsupported computer provider: {provider}")
        return normalized

    def provider_bin(self, provider: str) -> str:
        provider = self.normalize_provider(provider)
        binary = str(self.provider_bins.get(provider) or "")
        if not (shutil.which(binary) or Path(binary).exists()):
            raise ValueError(f"{provider} provider binary is not available: {binary}")
        return binary

    def provider_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "codex",
                "name": "Codex",
                "available": bool(shutil.which(str(self.provider_bins["codex"])) or Path(str(self.provider_bins["codex"])).exists()),
                "binary": str(self.provider_bins["codex"]),
                "capabilities": ["session.start", "session.send", "session.status", "session.cancel", "diff.read"],
            },
            {
                "id": "claude_code",
                "name": "Claude Code",
                "available": bool(shutil.which(str(self.provider_bins["claude_code"])) or Path(str(self.provider_bins["claude_code"])).exists()),
                "binary": str(self.provider_bins["claude_code"]),
                "capabilities": ["session.start"],
                "status": "adapter placeholder; enable after provider command contract is validated",
            },
            {
                "id": "cursor",
                "name": "Cursor",
                "available": bool(shutil.which(str(self.provider_bins["cursor"])) or Path(str(self.provider_bins["cursor"])).exists()),
                "binary": str(self.provider_bins["cursor"]),
                "capabilities": ["session.start"],
                "status": "adapter placeholder; enable after provider command contract is validated",
            },
        ]

    def origin_allowed(self, origin: str) -> bool:
        if not origin:
            return False
        origin = origin.rstrip("/")
        if origin in self.allowed_origins:
            return True
        return origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")

    def origin_allowed_for_host(self, origin: str, host_header: str) -> bool:
        if self.origin_allowed(origin):
            return True
        host = str(host_header or "").strip()
        if not origin or not host:
            return False
        origin = origin.rstrip("/")
        return origin in {f"http://{host}", f"https://{host}"}

    def require_allowed_origin(self, origin: str) -> str:
        origin = origin.rstrip("/")
        if not self.origin_allowed(origin):
            allowed = ", ".join(sorted(self.allowed_origins))
            raise ValueError(f"origin is not allowed: {origin or 'missing'} (allowed: {allowed}, plus localhost loopback ports)")
        return origin

    def load_grants(self) -> dict[str, dict[str, Any]]:
        if not self.grants_path.exists():
            return {}
        try:
            value = json.loads(self.grants_path.read_text())
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def save_grants(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.grants_path.write_text(json.dumps(self.grants, indent=2, sort_keys=True))
        try:
            self.grants_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def cleanup_pairing_requests(self) -> None:
        now = time.time()
        self.pairing = {
            pairing_id: request
            for pairing_id, request in self.pairing.items()
            if float(request.get("expires_at") or 0) >= now
        }

    def pairing_fingerprint(self, request: dict[str, Any]) -> str:
        raw = json.dumps(request.get("browser_public_jwk") or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return ":".join(digest[index:index + 2] for index in range(0, 16, 2))

    def approve_pairing_on_computer(self, request: dict[str, Any]) -> None:
        if not sys.stdin.isatty():
            raise ValueError("local computer approval is required, but this bridge has no interactive terminal")
        with self.pairing_approval_lock:
            print("", flush=True)
            print("Approve Agent Kernel computer-use pairing?", flush=True)
            print(f"Origin: {request.get('origin')}", flush=True)
            print(f"Pairing code: {request.get('code')}", flush=True)
            print(f"Browser key fingerprint: {self.pairing_fingerprint(request)}", flush=True)
            print("Type APPROVE to complete pairing, or anything else to reject.", flush=True)
            answer = input("> ").strip()
        if answer != "APPROVE":
            raise ValueError("pairing was rejected on the computer")

    def approve_mobile_console(self, token: str) -> dict[str, Any]:
        if not sys.stdin.isatty():
            raise ValueError("local computer approval is required, but this bridge has no interactive terminal")
        grant_id = f"mobile_{secrets.token_urlsafe(18)}"
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self.pairing_approval_lock:
            print("", flush=True)
            print("Approve Agent Kernel mobile console?", flush=True)
            print(f"Mobile code: {code}", flush=True)
            print("Type APPROVE to allow this phone to use the local Computer Use console.", flush=True)
            answer = input("> ").strip()
        if answer != "APPROVE":
            raise ValueError("mobile console was rejected on the computer")
        grant = {
            "grant_id": grant_id,
            "created_at": time.time(),
            "expires_at": time.time() + 60 * 60 * 12,
        }
        self.mobile_grants[grant_id] = grant
        return {"status": "approved", "grant_id": grant_id, "code": code, "expires_at": grant["expires_at"]}

    def require_mobile_grant(self, body: dict[str, Any]) -> dict[str, Any]:
        grant_id = str(body.get("grant_id") or "")
        grant = self.mobile_grants.get(grant_id)
        if not grant or float(grant.get("expires_at") or 0) < time.time():
            self.mobile_grants.pop(grant_id, None)
            raise ValueError("mobile console is not approved")
        return grant

    def active_mobile_sessions(self) -> list[dict[str, Any]]:
        with self.sessions_lock:
            session_ids = list(self.sessions.keys())
        return [self.session_snapshot(session_id, 0) for session_id in session_ids]

    def derive_key(self, grant: dict[str, Any]) -> bytes:
        browser_public = public_key_from_jwk(grant["browser_public_jwk"])
        shared = self.private_key.exchange(ec.ECDH(), browser_public)
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=str(grant["grant_id"]).encode("utf-8"),
            info=str(grant["origin"]).encode("utf-8"),
        ).derive(shared)

    def decrypt_message(self, envelope: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        grant_id = str(envelope.get("grant_id") or "")
        grant = self.grants.get(grant_id)
        if not grant:
            raise ValueError("unknown pairing grant")
        if float(grant.get("expires_at") or 0) < time.time():
            raise ValueError("pairing grant expired")
        seq = int(envelope.get("seq") or 0)
        if seq <= int(grant.get("last_seq") or 0):
            raise ValueError("stale or duplicate message sequence")
        aad = f"{grant_id}:{seq}".encode("utf-8")
        key = self.derive_key(grant)
        plaintext = AESGCM(key).decrypt(unb64(envelope["nonce"]), unb64(envelope["ciphertext"]), aad)
        grant["last_seq"] = seq
        self.save_grants()
        return json.loads(plaintext.decode("utf-8")), grant

    def encrypt_response(self, grant: dict[str, Any], seq: int, payload: dict[str, Any]) -> dict[str, Any]:
        nonce = secrets.token_bytes(12)
        aad = f"{grant['grant_id']}:{seq}".encode("utf-8")
        ciphertext = AESGCM(self.derive_key(grant)).encrypt(
            nonce,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            aad,
        )
        return {
            "protocol": PROTOCOL,
            "grant_id": grant["grant_id"],
            "seq": seq,
            "nonce": b64(nonce),
            "ciphertext": b64(ciphertext),
        }

    def workspace_allowed(self, raw_workspace: str) -> Path:
        workspace = Path(raw_workspace or "").expanduser().resolve()
        for allowed in self.allowed_workspaces:
            if workspace == allowed or allowed in workspace.parents:
                return workspace
        allowed_text = ", ".join(str(item) for item in self.allowed_workspaces)
        raise ValueError(f"workspace is not allowed by this bridge: {workspace} (allowed: {allowed_text})")

    def codex_base_command(self, workspace: Path, model: str = "") -> list[str]:
        # `codex exec` is the non-interactive CLI surface. The bridge owns pairing,
        # allowed workspace roots, and provider launch policy; the browser cannot
        # override sandbox or approval behavior per request.
        command = [
            self.provider_bin("codex"),
            "exec",
            "--json",
            "--cd",
            str(workspace),
            "--sandbox",
            self.sandbox,
        ]
        if model:
            command.extend(["--model", model])
        return command

    def append_session_event(self, session_id: str, event: dict[str, Any]) -> None:
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            session["events"].append(event)
            session["events"] = session["events"][-500:]
            parsed = event.get("parsed")
            if isinstance(parsed, dict):
                thread_id = self.extract_codex_thread_id(parsed)
                if thread_id and not session.get("codex_session_id"):
                    session["codex_session_id"] = thread_id
                text = self.extract_codex_event_text(parsed)
                if text:
                    session["summary"] = text[-4000:]

    def session_snapshot(self, session_id: str, since: int = 0) -> dict[str, Any]:
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError("unknown Codex bridge session")
            events = session["events"][max(0, int(since)) :]
            return {
                "session_id": session_id,
                "codex_session_id": session.get("codex_session_id") or "",
                "status": session["status"],
                "workspace": str(session["workspace"]),
                "model": session.get("model") or "",
                "started_at": session["started_at"],
                "completed_at": session.get("completed_at"),
                "exit_code": session.get("exit_code"),
                "elapsed_ms": round((time.time() - float(session["started_at"])) * 1000),
                "event_count": len(session["events"]),
                "events": events,
                "summary": session.get("summary") or "",
                "error": session.get("error") or "",
            }

    def read_stream(self, session_id: str, stream_name: str, pipe: Any) -> None:
        try:
            for line in pipe:
                text = str(line).rstrip("\n")
                if not text:
                    continue
                parsed = None
                if stream_name == "stdout":
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        parsed = None
                self.append_session_event(session_id, {
                    "index": len(self.sessions.get(session_id, {}).get("events", [])),
                    "stream": stream_name,
                    "time": time.time(),
                    "text": text[-8000:],
                    "parsed": parsed,
                })
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def wait_for_process(self, session_id: str, process: subprocess.Popen[str]) -> None:
        exit_code = process.wait()
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            session["exit_code"] = exit_code
            session["completed_at"] = time.time()
            if session["status"] == "cancelled":
                session["summary"] = "Codex session cancelled."
            elif exit_code == 0:
                session["status"] = "completed"
                session["summary"] = self.summarize_events(session)
            else:
                session["status"] = "failed"
                session["error"] = self.summarize_events(session, prefer_stderr=True)

    def summarize_events(self, session: dict[str, Any], prefer_stderr: bool = False) -> str:
        events = session.get("events", [])
        candidates = [event for event in events if (event.get("stream") == "stderr") == prefer_stderr]
        if not candidates:
            candidates = events
        for event in reversed(candidates):
            parsed = event.get("parsed")
            if isinstance(parsed, dict):
                text = self.extract_codex_event_text(parsed)
                if text:
                    return text[-4000:]
            text = str(event.get("text") or "").strip()
            if text and not text.startswith("{"):
                return text[-4000:]
        return ""

    @staticmethod
    def extract_codex_thread_id(parsed: dict[str, Any]) -> str:
        for key in ("session_id", "conversation_id", "thread_id", "task_id"):
            value = parsed.get(key)
            if value:
                return str(value)
        if parsed.get("type") == "thread.started":
            value = parsed.get("thread_id")
            return str(value) if value else ""
        msg = parsed.get("msg")
        if isinstance(msg, dict):
            for key in ("session_id", "conversation_id", "thread_id", "task_id"):
                value = msg.get(key)
                if value:
                    return str(value)
        return ""

    @staticmethod
    def extract_codex_event_text(parsed: dict[str, Any]) -> str:
        def text_from_item(item: Any) -> str:
            if not isinstance(item, dict):
                return ""
            item_type = str(item.get("type") or "")
            if item_type == "agent_message":
                return str(item.get("text") or item.get("message") or "").strip()
            if item_type == "error":
                return str(item.get("message") or "").strip()
            return ""

        text = text_from_item(parsed.get("item"))
        if text:
            return text
        msg = parsed.get("msg")
        if isinstance(msg, dict):
            text = text_from_item(msg.get("item"))
            if text:
                return text
            if str(msg.get("type") or "") == "agent_message":
                return str(msg.get("text") or msg.get("message") or "").strip()
            if msg.get("last_agent_message"):
                return str(msg.get("last_agent_message") or "").strip()
            if str(msg.get("type") or "") in {"turn.failed", "error"}:
                error = msg.get("error")
                if isinstance(error, dict):
                    return str(error.get("message") or "").strip()
                return str(msg.get("message") or "").strip()
        if parsed.get("last_agent_message"):
            return str(parsed.get("last_agent_message") or "").strip()
        if str(parsed.get("type") or "") in {"turn.failed", "error"}:
            error = parsed.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or "").strip()
            return str(parsed.get("message") or "").strip()
        return ""

    def start_codex_process(
        self,
        workspace: Path,
        command: list[str],
        action_id: str = "",
        model: str = "",
        session_id: str = "",
        initial_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        session_id = session_id or f"computer_{secrets.token_urlsafe(12)}"
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        with self.sessions_lock:
            self.sessions[session_id] = {
                "session_id": session_id,
                "action_id": action_id,
                "provider": "codex",
                "model": model,
                "workspace": workspace,
                "status": "running",
                "started_at": time.time(),
                "process": process,
                "events": list(initial_events or []),
                "command": ["codex", *command[1:-1], "..."],
            }
        for stream_name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            if pipe is not None:
                threading.Thread(target=self.read_stream, args=(session_id, stream_name, pipe), daemon=True).start()
        threading.Thread(target=self.wait_for_process, args=(session_id, process), daemon=True).start()
        return self.session_snapshot(session_id)

    def create_ready_codex_session(self, workspace: Path, action_id: str = "", model: str = "") -> dict[str, Any]:
        session_id = f"computer_{secrets.token_urlsafe(12)}"
        with self.sessions_lock:
            self.sessions[session_id] = {
                "session_id": session_id,
                "action_id": action_id,
                "provider": "codex",
                "model": model,
                "workspace": workspace,
                "status": "ready",
                "started_at": time.time(),
                "process": None,
                "events": [{
                    "index": 0,
                    "stream": "system",
                    "time": time.time(),
                    "text": f"Codex terminal ready in {workspace}.",
                    "parsed": None,
                }],
                "command": ["codex", "exec", "..."],
            }
        return self.session_snapshot(session_id)

    def start_codex_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = self.normalize_provider(str(payload.get("provider") or "codex"))
        if provider != "codex":
            raise ValueError(f"{provider} session orchestration is not implemented yet; Codex is the completed provider")
        workspace = self.workspace_allowed(str(payload.get("workspace") or self.allowed_workspaces[0]))
        prompt = str(payload.get("prompt") or "").strip()
        model = str(payload.get("model") or "").strip()
        if not prompt:
            return self.create_ready_codex_session(workspace, str(payload.get("action_id") or ""), model)
        command = [*self.codex_base_command(workspace, model), prompt]
        return self.start_codex_process(
            workspace,
            command,
            str(payload.get("action_id") or ""),
            model,
            initial_events=[{
                "index": 0,
                "stream": "user",
                "time": time.time(),
                "text": prompt,
                "parsed": None,
            }],
        )

    def send_codex_followup(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = self.normalize_provider(str(payload.get("provider") or "codex"))
        if provider != "codex":
            raise ValueError(f"{provider} follow-up orchestration is not implemented yet")
        parent_id = str(payload.get("session_id") or "")
        prompt = str(payload.get("prompt") or "").strip()
        if not parent_id:
            raise ValueError("session_id is required")
        if not prompt:
            raise ValueError("prompt is required")
        with self.sessions_lock:
            parent = self.sessions.get(parent_id)
            if not parent:
                raise ValueError("unknown Codex bridge session")
            if parent.get("status") == "running":
                raise ValueError("cannot send follow-up while parent session is running")
            workspace = Path(parent["workspace"])
            model = str(payload.get("model") or parent.get("model") or "").strip()
            codex_session_id = str(parent.get("codex_session_id") or "")
            is_ready = parent.get("status") == "ready"
            existing_events = list(parent.get("events") or [])
        user_event = {
            "index": len(existing_events),
            "stream": "user",
            "time": time.time(),
            "text": prompt,
            "parsed": None,
        }
        if is_ready:
            command = [*self.codex_base_command(workspace, model), prompt]
            return self.start_codex_process(
                workspace,
                command,
                str(payload.get("action_id") or ""),
                model,
                session_id=parent_id,
                initial_events=[*existing_events, user_event],
            )
        command = [*self.codex_base_command(workspace, model), "resume"]
        if codex_session_id:
            command.append(codex_session_id)
        else:
            command.append("--last")
        command.append(prompt)
        return self.start_codex_process(
            workspace,
            command,
            str(payload.get("action_id") or ""),
            model,
            session_id=parent_id,
            initial_events=[*existing_events, user_event],
        )

    def cancel_codex_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError("unknown Codex bridge session")
            process = session.get("process")
            if session.get("status") == "running" and process and process.poll() is None:
                session["status"] = "cancelled"
                process.terminate()
        return self.session_snapshot(session_id)

    def read_diff(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        workspace_raw = str(payload.get("workspace") or "")
        if session_id:
            with self.sessions_lock:
                session = self.sessions.get(session_id)
                if not session:
                    raise ValueError("unknown Codex bridge session")
                workspace = Path(session["workspace"])
        else:
            workspace = self.workspace_allowed(workspace_raw)
        completed = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        diff = completed.stdout
        truncated = len(diff) > 200_000
        if truncated:
            diff = diff[:200_000]
        return {
            "status": "ok" if completed.returncode == 0 else "failed",
            "workspace": str(workspace),
            "exit_code": completed.returncode,
            "diff": diff,
            "truncated": truncated,
            "stderr": completed.stderr[-4000:],
        }


MOBILE_PAGE_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Kernel Mobile Computer Use</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101418; color: #eef4f2; }
    header, main, form { padding: 14px; }
    header { border-bottom: 1px solid #26323a; position: sticky; top: 0; background: #101418; z-index: 2; }
    h1 { font-size: 18px; margin: 0 0 4px; }
    h2 { font-size: 15px; margin: 18px 0 8px; }
    .muted { color: #aebbc2; font-size: 13px; }
    button, input, select, textarea { width: 100%; font: inherit; border-radius: 8px; border: 1px solid #33434c; padding: 10px; margin-top: 8px; }
    button { background: #76d69d; color: #08120c; font-weight: 700; border: 0; }
    button.secondary { background: #24313a; color: #eef4f2; }
    textarea { min-height: 88px; resize: vertical; background: #182128; color: #eef4f2; }
    input, select { background: #182128; color: #eef4f2; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .session { border: 1px solid #2c3a43; border-radius: 8px; padding: 10px; margin: 10px 0; background: #151d23; }
    .session strong { display: block; margin-bottom: 4px; }
    .hidden { display: none !important; }
    .messages { padding-bottom: 130px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #0b1014; border: 1px solid #26323a; padding: 10px; border-radius: 8px; }
    form { position: fixed; left: 0; right: 0; bottom: 0; background: #101418; border-top: 1px solid #26323a; }
  </style>
</head>
<body>
  <header>
    <h1>Computer Use</h1>
    <div id="status" class="muted">Not approved.</div>
    <button id="approveButton" type="button">Approve This Phone</button>
  </header>
  <main>
    <section id="listView">
      <button id="newSessionButton" type="button">New Terminal</button>
      <h2>Active Sessions</h2>
      <div id="sessionList"></div>
    </section>
    <section id="newView" class="hidden">
      <button id="cancelNewButton" class="secondary" type="button">Back</button>
      <h2>New Terminal</h2>
      <label>Provider
        <select id="providerSelect"><option value="codex">Codex</option></select>
      </label>
      <label>Workspace root <select id="workspaceRootSelect"></select></label>
      <label>Repo or workspace path <input id="workspaceInput" type="text" autocomplete="off" autocapitalize="off" spellcheck="false"></label>
      <button id="startTerminalButton" type="button">Start Terminal</button>
    </section>
    <section id="chatView" class="hidden">
      <button id="backButton" class="secondary" type="button">Back</button>
      <h2 id="chatTitle">Session</h2>
      <div class="messages" id="messages"></div>
    </section>
  </main>
  <form id="messageForm" class="hidden">
    <textarea id="promptInput" placeholder="Message Codex..."></textarea>
    <button id="sendButton" type="submit">Send</button>
  </form>
  <script>
    const TOKEN = __TOKEN__;
    const WORKSPACES = __WORKSPACES__;
    let grantId = localStorage.getItem('agent-kernel-mobile-grant') || '';
    let activeSessionId = localStorage.getItem('agent-kernel-mobile-session') || '';
    let mode = 'list';
    const statusEl = document.getElementById('status');
    const approveButton = document.getElementById('approveButton');
    const listView = document.getElementById('listView');
    const newView = document.getElementById('newView');
    const chatView = document.getElementById('chatView');
    const messageForm = document.getElementById('messageForm');
    const providerSelect = document.getElementById('providerSelect');
    const workspaceRootSelect = document.getElementById('workspaceRootSelect');
    const workspaceInput = document.getElementById('workspaceInput');
    const sessionList = document.getElementById('sessionList');
    const messages = document.getElementById('messages');
    const promptInput = document.getElementById('promptInput');
    const chatTitle = document.getElementById('chatTitle');
    for (const workspace of WORKSPACES) {
      const option = document.createElement('option');
      option.value = workspace;
      option.textContent = workspace;
      workspaceRootSelect.appendChild(option);
    }
    workspaceInput.value = localStorage.getItem('agent-kernel-mobile-workspace') || WORKSPACES[0] || '';
    workspaceRootSelect.value = WORKSPACES.find((root) => workspaceInput.value === root || workspaceInput.value.startsWith(`${root}/`)) || WORKSPACES[0] || '';
    workspaceRootSelect.addEventListener('change', () => {
      workspaceInput.value = workspaceRootSelect.value;
      localStorage.setItem('agent-kernel-mobile-workspace', workspaceInput.value);
    });
    workspaceInput.addEventListener('change', () => {
      localStorage.setItem('agent-kernel-mobile-workspace', workspaceInput.value.trim());
    });
    function setMode(next) {
      mode = next;
      listView.classList.toggle('hidden', next !== 'list');
      newView.classList.toggle('hidden', next !== 'new');
      chatView.classList.toggle('hidden', next !== 'chat');
      messageForm.classList.toggle('hidden', next !== 'chat');
      promptInput.placeholder = 'Message Codex...';
      if (next === 'new') {
        activeSessionId = '';
        localStorage.removeItem('agent-kernel-mobile-session');
        messages.innerHTML = '';
        promptInput.focus();
      }
    }
    function clearMobileGrant(message = 'Not approved.') {
      grantId = '';
      activeSessionId = '';
      localStorage.removeItem('agent-kernel-mobile-grant');
      localStorage.removeItem('agent-kernel-mobile-session');
      approveButton.classList.remove('hidden');
      statusEl.textContent = message;
      sessionList.innerHTML = '';
      const note = document.createElement('p');
      note.className = 'muted';
      note.textContent = 'Approve this phone from the computer bridge to start using Computer Use.';
      sessionList.appendChild(note);
      setMode('list');
    }
    async function api(path, body = {}) {
      const response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: TOKEN, grant_id: grantId, ...body }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.status || response.status);
      return payload;
    }
    function eventText(event) {
      if (event.text) return event.text;
      if (event.parsed?.msg?.text) return event.parsed.msg.text;
      if (event.parsed?.msg?.message) return event.parsed.msg.message;
      if (event.parsed?.item?.text) return event.parsed.item.text;
      if (event.parsed?.item?.message) return event.parsed.item.message;
      return '';
    }
    function renderSession(session) {
      if (!session) return;
      statusEl.textContent = `Approved. ${session.status || 'session'}`;
      chatTitle.textContent = `${session.workspace || 'Workspace'} - ${session.status || ''}`;
      messages.innerHTML = '';
      for (const event of session.events || []) {
        const text = eventText(event);
        if (!text) continue;
        const pre = document.createElement('pre');
        pre.textContent = text;
        messages.appendChild(pre);
      }
      setMode('chat');
    }
    async function openSession(sessionId) {
      activeSessionId = sessionId;
      localStorage.setItem('agent-kernel-mobile-session', activeSessionId);
      const detail = await api('/mobile/api/status', { session_id: activeSessionId, since: 0 });
      renderSession(detail);
    }
    async function refresh() {
      approveButton.classList.toggle('hidden', Boolean(grantId));
      if (!grantId) {
        clearMobileGrant();
        return;
      }
      let payload;
      try {
        payload = await api('/mobile/api/sessions');
      } catch (error) {
        clearMobileGrant(error.message || 'Approval expired. Approve this phone again.');
        return;
      }
      const sessions = payload.sessions || [];
      sessionList.innerHTML = '';
      if (!sessions.length) {
        const empty = document.createElement('p');
        empty.className = 'muted';
        empty.textContent = 'No sessions yet.';
        sessionList.appendChild(empty);
      }
      for (const session of sessions) {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'session';
        const title = document.createElement('strong');
        title.textContent = session.workspace || 'Workspace';
        const detail = document.createElement('span');
        detail.textContent = `${session.status || 'session'} - ${session.model || 'default model'}`;
        card.append(title, detail);
        card.addEventListener('click', () => openSession(session.session_id));
        sessionList.appendChild(card);
      }
      statusEl.textContent = `Approved. ${sessions.length} session${sessions.length === 1 ? '' : 's'}.`;
      if (mode === 'chat' && activeSessionId) {
        const detail = await api('/mobile/api/status', { session_id: activeSessionId, since: 0 });
        renderSession(detail);
      }
    }
    approveButton.addEventListener('click', async () => {
      statusEl.textContent = 'Waiting for desktop approval...';
      try {
        const payload = await api('/mobile/api/approve');
        grantId = payload.grant_id;
        localStorage.setItem('agent-kernel-mobile-grant', grantId);
        statusEl.textContent = `Approved. Code ${payload.code}`;
        refresh();
      } catch (error) {
        clearMobileGrant(error.message || 'Approval failed. Try again.');
      }
    });
    document.getElementById('newSessionButton').addEventListener('click', () => setMode('new'));
    document.getElementById('cancelNewButton').addEventListener('click', () => setMode('list'));
    document.getElementById('startTerminalButton').addEventListener('click', async () => {
      const workspace = workspaceInput.value.trim() || workspaceRootSelect.value;
      localStorage.setItem('agent-kernel-mobile-workspace', workspace);
      const result = await api('/mobile/api/start', { workspace, provider: providerSelect.value || 'codex' });
      activeSessionId = result.session_id;
      localStorage.setItem('agent-kernel-mobile-session', activeSessionId);
      renderSession(result);
      window.setTimeout(refresh, 500);
    });
    document.getElementById('backButton').addEventListener('click', () => {
      activeSessionId = '';
      localStorage.removeItem('agent-kernel-mobile-session');
      setMode('list');
      refresh();
    });
    document.getElementById('messageForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const prompt = promptInput.value.trim();
      if (!prompt) return;
      promptInput.value = '';
      const workspace = workspaceInput.value.trim() || workspaceRootSelect.value;
      localStorage.setItem('agent-kernel-mobile-workspace', workspace);
      const body = { workspace, provider: providerSelect.value || 'codex', prompt };
      const result = activeSessionId
        ? await api('/mobile/api/send', { ...body, session_id: activeSessionId })
        : await api('/mobile/api/start', body);
      activeSessionId = result.session_id;
      localStorage.setItem('agent-kernel-mobile-session', activeSessionId);
      renderSession(result);
      window.setTimeout(refresh, 1200);
    });
    window.setInterval(() => { if (grantId) refresh().catch(() => {}); }, 3000);
    setMode('list');
    refresh().catch((error) => { clearMobileGrant(error.message || String(error)); });
  </script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    state: BridgeState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[bridge] {self.address_string()} {fmt % args}")

    def require_request_origin(self) -> str:
        origin = request_origin(self).rstrip("/")
        if self.state.origin_allowed_for_host(origin, self.headers.get("Host", "")):
            self.state.allowed_origins.add(origin)
            return origin
        return self.state.require_allowed_origin(origin)

    def do_OPTIONS(self) -> None:
        try:
            self.require_request_origin()
            options_response(self, 204)
        except Exception as exc:
            options_response(self, 403, {"status": "error", "error": str(exc)})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"", "/", "/mobile"}:
            self.handle_mobile_page()
            return
        if path == "/pair":
            self.handle_pair_page()
            return
        if path != "/health":
            json_response(self, 404, {"status": "error", "error": "not found"})
            return
        json_response(self, 200, self.state.health_payload())

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path.startswith("/mobile/api/"):
                self.handle_mobile_api(path, read_json(self))
                return
            origin = self.require_request_origin()
            if path == "/pairing/start":
                json_response(self, 200, self.state.start_pairing_request(origin, read_json(self)))
            elif path == "/pairing/confirm":
                json_response(self, 200, self.state.confirm_pairing_request(origin, read_json(self)))
            elif path == "/v1/message":
                json_response(self, 200, self.state.encrypted_message_response(origin, read_json(self)))
            elif path == "/v1/revoke":
                self.handle_revoke(origin)
            else:
                json_response(self, 404, {"status": "error", "error": "not found"})
        except Exception as exc:
            json_response(self, 400, {"status": "error", "error": str(exc)})

    def handle_mobile_api(self, path: str, body: dict[str, Any]) -> None:
        if path == "/mobile/api/approve":
            json_response(self, 200, self.state.approve_mobile_console(str(body.get("token") or "")))
            return
        self.state.require_mobile_grant(body)
        if path == "/mobile/api/health":
            json_response(self, 200, self.state.health_payload())
        elif path == "/mobile/api/sessions":
            json_response(self, 200, {"status": "ok", "sessions": self.state.active_mobile_sessions()})
        elif path == "/mobile/api/start":
            json_response(self, 200, self.state.start_codex_session({
                "provider": body.get("provider") or "codex",
                "workspace": body.get("workspace") or "",
                "model": body.get("model") or "",
                "prompt": body.get("prompt") or "",
            }))
        elif path == "/mobile/api/send":
            json_response(self, 200, self.state.send_codex_followup({
                "provider": body.get("provider") or "codex",
                "session_id": body.get("session_id") or "",
                "model": body.get("model") or "",
                "prompt": body.get("prompt") or "",
            }))
        elif path == "/mobile/api/status":
            json_response(self, 200, self.state.session_snapshot(str(body.get("session_id") or ""), int(body.get("since") or 0)))
        else:
            json_response(self, 404, {"status": "error", "error": "not found"})

    def handle_mobile_page(self) -> None:
        workspaces = [str(item) for item in self.state.allowed_workspaces]
        page = MOBILE_PAGE_HTML.replace("__TOKEN__", json.dumps("")).replace("__WORKSPACES__", json.dumps(workspaces))
        html_response(self, 200, page)

    def handle_pair_page(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        requested_app = str((query.get("app") or [""])[0]).strip()
        app_url = requested_app or "https://peytontolbert.com/agent_kernel/"
        try:
            parsed = urlparse(app_url)
            if parsed.scheme != "https":
                raise ValueError("app URL must be https")
            query_items = parse_qs(parsed.query)
            token = secrets.token_urlsafe(18)
            query_items["computerBroker"] = ["1"]
            query_items["computerBrokerToken"] = [token]
            app_url = parsed._replace(
                query=urlencode(query_items, doseq=True),
                fragment=f"computerBroker=1&computerBrokerToken={token}",
            ).geturl()
        except Exception:
            token = secrets.token_urlsafe(18)
            app_url = f"https://peytontolbert.com/agent_kernel/?computerBroker=1&computerBrokerToken={token}#computerBroker=1&computerBrokerToken={token}"
        app_origin = f"{urlparse(app_url).scheme}://{urlparse(app_url).netloc}"
        app_url_js = json.dumps(app_url)
        app_origin_js = json.dumps(app_origin)
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Kernel Computer Pairing</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; line-height: 1.45; background: #101418; color: #f4f7f8; }}
    main {{ max-width: 620px; margin: 0 auto; padding: 20px; }}
    button {{ font: inherit; padding: 12px 16px; border-radius: 8px; border: 0; background: #72d6a0; color: #07120b; font-weight: 700; }}
    code, pre {{ background: #1f2930; border-radius: 6px; padding: 2px 5px; }}
    pre {{ padding: 12px; overflow-wrap: anywhere; white-space: pre-wrap; }}
    .muted {{ color: #b8c2c8; }}
    iframe {{ width: 100%; min-height: 72vh; border: 1px solid #33414a; border-radius: 8px; background: white; }}
  </style>
</head>
<body>
  <main>
    <h1>Pair Agent Kernel</h1>
    <p class="muted">This local page lets the HTTPS Agent Kernel app talk to the desktop bridge without a hosted relay.</p>
    <p><button id="openApp" type="button">Open Agent Kernel</button></p>
    <p><button id="embedApp" type="button">Open Here</button></p>
    <p>Status: <span id="status">waiting</span></p>
    <pre id="log"></pre>
    <div id="frameHost"></div>
  </main>
  <script>
    const appUrl = {app_url_js};
    const appOrigin = {app_origin_js};
    let appWindow = null;
    const statusEl = document.getElementById('status');
    const logEl = document.getElementById('log');
    function log(line) {{
      logEl.textContent = `${{new Date().toLocaleTimeString()}} ${{line}}\\n${{logEl.textContent}}`;
    }}
    function sendReadyTo(target) {{
      if (!target) return;
      target.postMessage({{
        type: 'agent-kernel-computer-broker-ready',
        bridgeUrl: window.location.origin,
        token: new URL(appUrl).searchParams.get('computerBrokerToken') || '',
      }}, appOrigin);
    }}
    function sendReady() {{
      if (appWindow && !appWindow.closed) sendReadyTo(appWindow);
      const frame = document.getElementById('agentKernelFrame');
      if (frame && frame.contentWindow) sendReadyTo(frame.contentWindow);
    }}
    document.getElementById('openApp').addEventListener('click', () => {{
      appWindow = window.open(appUrl, 'agent-kernel-lite');
      statusEl.textContent = appWindow ? 'app opened' : 'popup blocked';
      log(appWindow ? 'Opened Agent Kernel app.' : 'Popup blocked. Allow popups for this local pairing page.');
      sendReady();
    }});
    document.getElementById('embedApp').addEventListener('click', () => {{
      let frame = document.getElementById('agentKernelFrame');
      if (!frame) {{
        frame = document.createElement('iframe');
        frame.id = 'agentKernelFrame';
        frame.allow = 'clipboard-read; clipboard-write';
        document.getElementById('frameHost').appendChild(frame);
      }}
      frame.src = appUrl;
      appWindow = frame.contentWindow;
      statusEl.textContent = 'app embedded';
      log('Embedded Agent Kernel app.');
      frame.addEventListener('load', sendReady);
    }});
    window.addEventListener('message', async (event) => {{
      if (event.origin !== appOrigin) return;
      const message = event.data || {{}};
      if (message.type === 'agent-kernel-computer-broker-hello') {{
        appWindow = event.source;
        statusEl.textContent = 'connected';
        log('Agent Kernel app connected to local broker.');
        sendReady();
        return;
      }}
      if (message.type !== 'agent-kernel-computer-broker-request') return;
      const requestId = message.requestId;
      try {{
        const response = await fetch(message.path, {{
          method: message.method || 'GET',
          headers: message.body ? {{ 'Content-Type': 'application/json' }} : undefined,
          body: message.body ? JSON.stringify(message.body) : undefined,
          cache: 'no-store',
        }});
        let payload = null;
        try {{ payload = await response.json(); }} catch (_) {{ payload = {{ status: response.statusText || 'empty' }}; }}
        event.source.postMessage({{
          type: 'agent-kernel-computer-broker-response',
          requestId,
          ok: response.ok,
          status: response.status,
          payload,
        }}, event.origin);
      }} catch (error) {{
        event.source.postMessage({{
          type: 'agent-kernel-computer-broker-response',
          requestId,
          ok: false,
          status: 0,
          error: error.message || String(error),
        }}, event.origin);
      }}
    }});
    window.setInterval(sendReady, 1000);
  </script>
</body>
</html>
"""
        raw = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def handle_pairing_start(self, origin: str) -> None:
        self.state.cleanup_pairing_requests()
        body = read_json(self)
        browser_public_jwk = body.get("browser_public_jwk")
        public_key_from_jwk(browser_public_jwk)
        pairing_id = f"pair_{secrets.token_urlsafe(12)}"
        code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = time.time() + 300
        self.state.pairing[pairing_id] = {
            "pairing_id": pairing_id,
            "code": code,
            "origin": origin,
            "browser_public_jwk": browser_public_jwk,
            "expires_at": expires_at,
            "attempts": 0,
        }
        print("", flush=True)
        print("Agent Kernel Lite computer-use pairing request", flush=True)
        print(f"Origin: {origin}", flush=True)
        print(f"Pairing code: {code}", flush=True)
        print(f"Browser key fingerprint: {self.state.pairing_fingerprint(self.state.pairing[pairing_id])}", flush=True)
        print("Enter this code in the Agent Kernel Lite app within 5 minutes.", flush=True)
        print("The computer must also approve the pairing after the code is entered.", flush=True)
        print("", flush=True)
        json_response(
            self,
            200,
            {
                "status": "pairing_code_required",
                "pairing_id": pairing_id,
                "protocol": PROTOCOL,
                "bridge_public_jwk": public_key_to_jwk(self.state.private_key.public_key()),
                "expires_at": expires_at,
                "code_length": 6,
            },
        )

    def handle_pairing_confirm(self, origin: str) -> None:
        body = read_json(self)
        pairing_id = str(body.get("pairing_id") or "")
        code = str(body.get("code") or "").strip()
        request = self.state.pairing.get(pairing_id)
        if not request or float(request["expires_at"]) < time.time():
            raise ValueError("pairing request expired or missing")
        if origin != request.get("origin"):
            raise ValueError("pairing origin does not match request origin")
        request["attempts"] = int(request.get("attempts") or 0) + 1
        if int(request["attempts"]) > 5:
            self.state.pairing.pop(pairing_id, None)
            raise ValueError("too many pairing attempts")
        if not secrets.compare_digest(code, str(request["code"])):
            raise ValueError("pairing code did not match")
        self.state.approve_pairing_on_computer(request)
        grant_id = f"grant_{secrets.token_urlsafe(18)}"
        grant = {
            "grant_id": grant_id,
            "origin": request["origin"],
            "browser_public_jwk": request["browser_public_jwk"],
            "created_at": time.time(),
            "expires_at": time.time() + 60 * 60 * 24 * 30,
            "last_seq": 0,
        }
        self.state.grants[grant_id] = grant
        self.state.pairing.pop(pairing_id, None)
        self.state.save_grants()
        json_response(self, 200, {"status": "paired", "grant_id": grant_id, "expires_at": grant["expires_at"]})

    def handle_encrypted_message(self, origin: str) -> None:
        envelope = read_json(self)
        if envelope.get("protocol") not in {PROTOCOL, LEGACY_PROTOCOL}:
            raise ValueError("unsupported protocol")
        seq = int(envelope.get("seq") or 0)
        payload, grant = self.state.decrypt_message(envelope)
        if origin != grant.get("origin"):
            raise ValueError("request origin does not match pairing grant")
        message_type = str(payload.get("type") or "")
        if message_type in {"computer.session.start", "codex.session.start"}:
            result = self.state.start_codex_session(payload)
        elif message_type in {"computer.session.send", "codex.session.send"}:
            result = self.state.send_codex_followup(payload)
        elif message_type in {"computer.session.status", "codex.session.status"}:
            session_id = str(payload.get("session_id") or "")
            result = self.state.session_snapshot(session_id, int(payload.get("since") or 0)) if session_id else {
                "status": "ok",
                "message": "bridge is ready",
                "providers": self.state.provider_catalog(),
                "active_sessions": [
                    self.state.session_snapshot(session_id, 0)
                    for session_id in list(self.state.sessions.keys())
                    if self.state.sessions.get(session_id, {}).get("status") == "running"
                ],
            }
        elif message_type in {"computer.session.cancel", "codex.session.cancel"}:
            result = self.state.cancel_codex_session(payload)
        elif message_type in {"computer.diff.read", "codex.diff.read"}:
            result = self.state.read_diff(payload)
        elif message_type in {"computer.grant.revoke", "codex.grant.revoke"}:
            removed = self.state.grants.pop(str(grant["grant_id"]), None)
            self.state.save_grants()
            result = {"status": "revoked" if removed else "missing", "grant_id": grant["grant_id"]}
        else:
            raise ValueError(f"unsupported encrypted message type: {message_type}")
        json_response(self, 200, self.state.encrypt_response(grant, seq, {"type": f"{message_type}.result", "result": result}))

    def handle_revoke(self, _origin: str) -> None:
        raise ValueError("plaintext revoke is disabled; use encrypted codex.grant.revoke")


def run_relay_client(state: BridgeState, relay_url: str, public_base_url: str, ttl_seconds: int = 86400) -> None:
    relay_url = relay_url.rstrip("/")
    route_id = f"route_{secrets.token_urlsafe(32)}"
    pairing_code = f"{secrets.randbelow(1_000_000):06d}"
    device_id = f"desktop_{secrets.token_urlsafe(12)}"
    token = secrets.token_urlsafe(32)
    status, registered = post_json(
        f"{relay_url}/desktop/register",
        {
            "route_id": route_id,
            "pairing_code": pairing_code,
            "device_id": device_id,
            "token": token,
            "ttl_seconds": ttl_seconds,
            "label": "Agent Kernel Desktop",
        },
    )
    if status != 200 or registered.get("status") != "registered":
        raise RuntimeError(registered.get("error") or f"relay registration failed: {status}")
    bridge_url = f"{public_base_url.rstrip('/')}/bridge/{route_id}"
    print("", flush=True)
    print("Agent Kernel Lite computer-use relay connected", flush=True)
    print(f"Relay: {relay_url}", flush=True)
    print(f"Phone bridge URL: {bridge_url}", flush=True)
    print(f"Desktop pairing code: {pairing_code}", flush=True)
    print("Enter the Phone bridge URL in the app, then pair with the code shown here.", flush=True)
    print("Keep this terminal open while using the Computer Use extension.", flush=True)
    print("", flush=True)
    while True:
        status, poll = post_json(
            f"{relay_url}/desktop/poll",
            {"device_id": device_id, "token": token, "timeout_seconds": 25},
            timeout=35,
        )
        if status != 200:
            print(f"[relay] poll failed: {status} {poll.get('error') or poll}", flush=True)
            time.sleep(2)
            continue
        if poll.get("status") == "idle":
            continue
        request = poll.get("request")
        if not isinstance(request, dict):
            time.sleep(1)
            continue
        request_id = str(request.get("request_id") or "")
        response = state.handle_relay_request(request)
        post_status, posted = post_json(
            f"{relay_url}/desktop/respond",
            {
                "device_id": device_id,
                "token": token,
                "request_id": request_id,
                "status_code": int(response.get("status_code") or 200),
                "payload": response.get("payload") if isinstance(response.get("payload"), dict) else {},
            },
        )
        if post_status != 200:
            print(f"[relay] response failed: {post_status} {posted.get('error') or posted}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent Kernel Lite computer-use bridge.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Address to bind. Use 127.0.0.1 for same-computer browser use, or "
            "0.0.0.0 / the computer LAN IP for phone pairing on trusted Wi-Fi."
        ),
    )
    parser.add_argument("--port", type=int, default=45731)
    parser.add_argument("--workspace", action="append", default=[], help="Allowed workspace root. Repeat for more roots.")
    parser.add_argument("--config-dir", default="~/.agent-kernel-lite/codex-bridge")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--cursor-bin", default="cursor-agent")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--sandbox",
        default="danger-full-access",
        choices=sorted(ALLOWED_SANDBOXES),
        help=(
            "Codex CLI sandbox mode. The default avoids bubblewrap failures in "
            "browser/desktop bridge environments; use workspace-write or read-only "
            "when those sandboxes work on the host."
        ),
    )
    parser.add_argument("--approval-policy", default="never", choices=sorted(ALLOWED_APPROVAL_POLICIES))
    parser.add_argument("--allow-origin", action="append", default=[], help="Allowed browser origin. Repeat for more origins.")
    parser.add_argument("--relay-url", default="", help="Internal relay API base URL, for example https://peytontolbert.com/agent_kernel/api/relay")
    parser.add_argument("--relay-public-url", default="", help="Public relay API base URL used by the phone. Defaults to --relay-url.")
    parser.add_argument("--relay-ttl", type=int, default=86400, help="Relay route lifetime in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = BridgeState(args)
    if str(args.relay_url or "").strip():
        run_relay_client(state, str(args.relay_url).strip(), str(args.relay_public_url or args.relay_url).strip(), int(args.relay_ttl))
        return
    Handler.state = state
    server = ThreadingHTTPServer((state.host, state.port), Handler)
    print(f"Agent Kernel Lite computer-use bridge listening on http://{state.host}:{state.port}", flush=True)
    if state.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: bridge is reachable beyond loopback. Use trusted Wi-Fi and approve pairings only from your own devices.", flush=True)
    print(f"Allowed workspace roots: {', '.join(str(item) for item in state.allowed_workspaces)}", flush=True)
    if state.host == "0.0.0.0":
        addresses = local_ipv4_addresses()
        if addresses:
            print("Mobile Computer Use:", flush=True)
            for address in addresses:
                print(f"  http://{address}:{state.port}/", flush=True)
        else:
            print(f"Mobile Computer Use: http://<computer-lan-ip>:{state.port}/", flush=True)
    else:
        print(f"Mobile Computer Use: http://{state.host}:{state.port}/", flush=True)
    print("Keep this terminal open while using the Computer Use extension.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
