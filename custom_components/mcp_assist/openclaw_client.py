"""OpenClaw Gateway WebSocket client for MCP Assist."""

import asyncio
import base64
import hashlib
import json
import logging
import ssl
import time
import uuid
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

_LOGGER = logging.getLogger(__name__)

# Protocol constants
PROTOCOL_VERSION = 3
CLIENT_ID = "gateway-client"
CLIENT_DISPLAY_NAME = "Home Assistant MCP Assist"
CLIENT_VERSION = "1.0.0"
CLIENT_MODE = "backend"
DEVICE_ROLE = "operator"
DEVICE_SCOPES = ["operator.read", "operator.write"]
CHALLENGE_TIMEOUT = 2.0
HANDSHAKE_TIMEOUT = 10.0
KEEPALIVE_INTERVAL = 30
RECONNECT_DELAY = 5

# Storage
STORAGE_KEY = "mcp_assist.openclaw_device"
STORAGE_VERSION = 1


# --- Exceptions ---

class OpenClawError(Exception):
    """Base exception for OpenClaw errors."""


class OpenClawConnectionError(OpenClawError):
    """Transient connection error (retriable)."""


class OpenClawAuthError(OpenClawError):
    """Authentication error (bad token, permanent)."""


class DevicePairingRequiredError(OpenClawError):
    """Device not yet approved on the OpenClaw server."""

    def __init__(self, message: str, device_id: str = ""):
        super().__init__(message)
        self.device_id = device_id


class OpenClawTimeoutError(OpenClawError):
    """Agent response timed out."""


# --- Device Auth ---

def _base64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class OpenClawDeviceAuth:
    """Manages Ed25519 device keypair for OpenClaw Gateway authentication."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize device auth."""
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._private_key: Optional[Ed25519PrivateKey] = None
        self._device_id: Optional[str] = None
        self._public_key_b64: Optional[str] = None

    @property
    def device_id(self) -> str:
        """Return device ID."""
        if not self._device_id:
            raise RuntimeError("Device auth not loaded — call async_load() first")
        return self._device_id

    @property
    def public_key_b64(self) -> str:
        """Return base64url-encoded public key."""
        if not self._public_key_b64:
            raise RuntimeError("Device auth not loaded — call async_load() first")
        return self._public_key_b64

    async def async_load(self) -> None:
        """Load existing keypair from storage, or generate a new one."""
        data = await self._store.async_load()

        if data and "private_key_hex" in data:
            try:
                pk_bytes = bytes.fromhex(data["private_key_hex"])
                self._private_key = Ed25519PrivateKey.from_private_bytes(pk_bytes)
                _LOGGER.debug("Loaded existing OpenClaw device keypair")
            except Exception:
                _LOGGER.warning("Failed to load stored keypair, generating new one")
                self._private_key = None

        if not self._private_key:
            self._private_key = Ed25519PrivateKey.generate()
            await self._save()
            _LOGGER.info("Generated new OpenClaw device keypair")

        # Derive public key and device ID
        pub_bytes = self._private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self._device_id = hashlib.sha256(pub_bytes).hexdigest()
        self._public_key_b64 = _base64url_encode(pub_bytes)

        _LOGGER.debug("OpenClaw device ID: %s", self._device_id)

    async def _save(self) -> None:
        """Save keypair to storage."""
        pk_bytes = self._private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        await self._store.async_save({"private_key_hex": pk_bytes.hex()})

    def sign_challenge(self, nonce: str, token: str, timestamp_ms: int) -> str:
        """Sign a challenge nonce and return base64url-encoded signature."""
        scopes_str = ",".join(DEVICE_SCOPES)
        payload = (
            f"v2|{self._device_id}|{CLIENT_ID}|{CLIENT_MODE}|"
            f"{DEVICE_ROLE}|{scopes_str}|{timestamp_ms}|{token}|{nonce}"
        )
        signature = self._private_key.sign(payload.encode("utf-8"))
        return _base64url_encode(signature)

    def build_device_dict(self, nonce: str, token: str) -> Dict[str, Any]:
        """Build the device auth dictionary for the connect handshake."""
        timestamp_ms = int(time.time() * 1000)
        signature_b64 = self.sign_challenge(nonce, token, timestamp_ms)
        return {
            "id": self._device_id,
            "publicKey": self._public_key_b64,
            "signature": signature_b64,
            "signedAt": timestamp_ms,
            "nonce": nonce,
        }


# --- WebSocket Client ---

class OpenClawClient:
    """Persistent WebSocket client for OpenClaw Gateway."""

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        use_ssl: bool,
        device_auth: OpenClawDeviceAuth,
        timeout: int = 60,
    ) -> None:
        """Initialize OpenClaw client."""
        self._host = host
        self._port = port
        self._token = token
        self._use_ssl = use_ssl
        self._device_auth = device_auth
        self._timeout = timeout

        self._ws = None
        self._connected = False
        self._keepalive_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._agent_runs: Dict[str, "_AgentRun"] = {}
        self._event_handlers: Dict[str, list] = {}

    @property
    def is_connected(self) -> bool:
        """Return whether the client is connected."""
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        """Connect to the OpenClaw Gateway and complete handshake."""
        from websockets.asyncio.client import connect

        scheme = "wss" if self._use_ssl else "ws"
        url = f"{scheme}://{self._host}:{self._port}/?token={self._token}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-OpenClaw-Token": self._token,
        }

        # SSL context for self-signed certs (tailscale, etc.)
        ssl_ctx = None
        if self._use_ssl:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        _LOGGER.info("Connecting to OpenClaw Gateway at %s://%s:%d", scheme, self._host, self._port)

        try:
            self._ws = await connect(
                url,
                additional_headers=headers,
                ssl=ssl_ctx,
                ping_interval=KEEPALIVE_INTERVAL,
                ping_timeout=10,
                open_timeout=10,
            )
        except Exception as err:
            raise OpenClawConnectionError(f"Failed to connect to OpenClaw Gateway: {err}") from err

        # Complete the handshake
        try:
            await self._handshake()
        except (OpenClawAuthError, DevicePairingRequiredError):
            await self._close_ws()
            raise
        except Exception as err:
            await self._close_ws()
            raise OpenClawConnectionError(f"Handshake failed: {err}") from err

        self._connected = True

        # Start background tasks
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        _LOGGER.info("✅ Connected to OpenClaw Gateway")

    async def _handshake(self) -> None:
        """Complete the WebSocket handshake with device auth."""
        # Wait for optional connect.challenge event
        nonce = None
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=CHALLENGE_TIMEOUT)
            msg = json.loads(raw)
            if msg.get("type") == "event" and msg.get("event") == "connect.challenge":
                nonce = msg.get("payload", {}).get("nonce")
                _LOGGER.debug("Received connect challenge")
        except asyncio.TimeoutError:
            _LOGGER.debug("No connect challenge received (legacy mode)")

        # Build connect request
        connect_id = str(uuid.uuid4())
        connect_params = {
            "minProtocol": PROTOCOL_VERSION,
            "maxProtocol": PROTOCOL_VERSION,
            "client": {
                "id": CLIENT_ID,
                "displayName": CLIENT_DISPLAY_NAME,
                "version": CLIENT_VERSION,
                "platform": "python",
                "mode": CLIENT_MODE,
            },
            "caps": [],
            "locale": "en-US",
            "userAgent": f"{CLIENT_DISPLAY_NAME}/{CLIENT_VERSION}",
            "auth": {"token": self._token},
            "role": DEVICE_ROLE,
            "scopes": list(DEVICE_SCOPES),
        }

        # Add device auth if we got a challenge
        if nonce:
            connect_params["device"] = self._device_auth.build_device_dict(nonce, self._token)

        await self._ws.send(json.dumps({
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": connect_params,
        }))

        # Wait for response
        raw = await asyncio.wait_for(self._ws.recv(), timeout=HANDSHAKE_TIMEOUT)
        resp = json.loads(raw)

        if resp.get("ok"):
            _LOGGER.debug("Handshake successful")
            return

        # Handle errors
        error = resp.get("error", {})
        code = error.get("code", "")
        message = error.get("message", "")

        if code == "NOT_PAIRED" or "pairing" in message.lower():
            raise DevicePairingRequiredError(
                f"Device not paired: {message}",
                device_id=self._device_auth.device_id,
            )

        if code in ("UNAUTHORIZED", "FORBIDDEN") or "auth" in message.lower() or "token" in message.lower():
            raise OpenClawAuthError(f"Authentication failed: {message}")

        raise OpenClawConnectionError(f"Handshake error ({code}): {message}")

    async def disconnect(self) -> None:
        """Disconnect from the OpenClaw Gateway."""
        _LOGGER.info("Disconnecting from OpenClaw Gateway")
        self._connected = False

        # Cancel background tasks
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Fail pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(OpenClawConnectionError("Disconnected"))
        self._pending_requests.clear()

        # Complete active runs
        for run in self._agent_runs.values():
            run.set_complete("error", "Disconnected")
        self._agent_runs.clear()

        await self._close_ws()

    async def _close_ws(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def send_message(self, text: str, session_key: str) -> str:
        """Send a message to the OpenClaw agent and return the response.

        Args:
            text: User message text
            session_key: OpenClaw session key (e.g., "main")

        Returns:
            The agent's response text

        Raises:
            OpenClawConnectionError: Not connected
            OpenClawTimeoutError: Response timed out
        """
        if not self.is_connected:
            raise OpenClawConnectionError("Not connected to OpenClaw Gateway")

        request_id = str(uuid.uuid4())
        idempotency_key = str(uuid.uuid4())

        # Send agent request
        await self._ws.send(json.dumps({
            "type": "req",
            "id": request_id,
            "method": "agent",
            "params": {
                "message": text,
                "sessionKey": session_key,
                "idempotencyKey": idempotency_key,
            },
        }))

        _LOGGER.debug("Sent agent request: %s", request_id[:8])

        # Wait for the initial response (contains runId)
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            resp = await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise OpenClawTimeoutError("Timeout waiting for agent run acknowledgment")

        if not resp.get("ok", True):
            error = resp.get("error", {})
            raise OpenClawError(f"Agent request failed: {error.get('message', error)}")

        run_id = resp.get("payload", {}).get("runId")
        if not run_id:
            raise OpenClawError("No runId in agent response")

        # Track this run and wait for completion
        run = _AgentRun(run_id)
        self._agent_runs[run_id] = run

        try:
            await asyncio.wait_for(run.complete_event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise OpenClawTimeoutError(
                f"OpenClaw agent timed out after {self._timeout}s"
            )
        finally:
            self._agent_runs.pop(run_id, None)

        if run.status == "error":
            raise OpenClawError(f"Agent error: {run.summary or 'unknown error'}")

        return run.summary or run.full_text

    async def _receive_loop(self) -> None:
        """Background task to receive and dispatch WebSocket messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_message(msg)
                except json.JSONDecodeError:
                    _LOGGER.debug("Invalid JSON from gateway")
                except Exception as err:
                    _LOGGER.debug("Error handling message: %s", err)
        except asyncio.CancelledError:
            return
        except Exception as err:
            _LOGGER.warning("WebSocket receive loop ended: %s", err)
            self._connected = False

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        """Handle an incoming WebSocket message."""
        msg_type = msg.get("type")

        if msg_type == "res":
            # Response to a pending request
            req_id = msg.get("id")
            future = self._pending_requests.pop(req_id, None)
            if future and not future.done():
                future.set_result(msg)

        elif msg_type == "event":
            event_name = msg.get("event")
            if event_name == "agent":
                self._handle_agent_event(msg.get("payload", {}))

        elif msg_type == "ping":
            # Respond to server ping
            try:
                await self._ws.send(json.dumps({"type": "pong"}))
            except Exception:
                pass

    def _handle_agent_event(self, payload: Dict[str, Any]) -> None:
        """Handle an agent event (streaming output or completion)."""
        run_id = payload.get("runId")
        if not run_id:
            return

        run = self._agent_runs.get(run_id)
        if not run:
            return

        # Accumulate output (gateway sends cumulative text)
        output = payload.get("output")
        if output:
            run.update_text(output)
        elif "data" in payload and "text" in payload.get("data", {}):
            run.update_text(payload["data"]["text"])

        # Check for completion
        status = payload.get("status")
        phase = payload.get("data", {}).get("phase", "")

        if status in ("ok", "error") or phase in ("end", "complete"):
            summary = payload.get("summary")
            run.set_complete(status or "ok", summary)

    async def _keepalive_loop(self) -> None:
        """Background task to send keepalive pings."""
        try:
            while self._connected:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                if self._ws and self._connected:
                    try:
                        await self._ws.send(json.dumps({"type": "ping"}))
                    except Exception:
                        _LOGGER.debug("Keepalive ping failed")
                        self._connected = False
                        break
        except asyncio.CancelledError:
            return


class _AgentRun:
    """Tracks a single agent run (request → response events → completion)."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.full_text = ""
        self.summary: Optional[str] = None
        self.status: Optional[str] = None
        self.complete_event = asyncio.Event()

    def update_text(self, cumulative_text: str) -> None:
        """Update with cumulative text from the gateway."""
        self.full_text = cumulative_text

    def set_complete(self, status: str, summary: Optional[str] = None) -> None:
        """Mark the run as complete."""
        self.status = status
        if summary:
            self.summary = summary
        self.complete_event.set()
