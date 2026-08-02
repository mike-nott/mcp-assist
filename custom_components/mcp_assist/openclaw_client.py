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
from urllib.parse import quote

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

_LOGGER = logging.getLogger(__name__)

# Protocol constants
PROTOCOL_VERSION = 4
CLIENT_ID = "gateway-client"
CLIENT_DISPLAY_NAME = "Home Assistant MCP Assist"
CLIENT_VERSION = "1.0.0"
CLIENT_MODE = "backend"
DEVICE_ROLE = "operator"
DEVICE_SCOPES = ["operator.read", "operator.write"]
# Matches the reference client's preauth handshake budget (15s). The challenge
# is mandatory on protocol 4: a connect without device auth is rejected
# without ever creating a pending pairing request on the gateway.
CHALLENGE_TIMEOUT = 15.0
HANDSHAKE_TIMEOUT = 10.0
KEEPALIVE_INTERVAL = 30
RECONNECT_DELAY = 5

# Storage
STORAGE_KEY = "mcp_assist.openclaw_device"
STORAGE_VERSION = 1


def _normalize_locale(locale: str | None) -> str:
    """Return a compact BCP-47-ish locale for the OpenClaw handshake."""
    normalized = str(locale or "").strip().replace("_", "-")
    return normalized or "en-US"


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
        """Sign a challenge nonce and return base64url-encoded signature.

        v3 payload: v2 fields plus lowercased platform and deviceFamily.
        platform must byte-match the connect params' client.platform.
        """
        scopes_str = ",".join(DEVICE_SCOPES)
        payload = (
            f"v3|{self._device_id}|{CLIENT_ID}|{CLIENT_MODE}|"
            f"{DEVICE_ROLE}|{scopes_str}|{timestamp_ms}|{token}|{nonce}|python|"
        )
        signature = self._private_key.sign(payload.encode("utf-8"))
        return _base64url_encode(signature)

    def build_device_dict(
        self, nonce: str, token: str, signed_at_ms: int | None = None
    ) -> Dict[str, Any]:
        """Build the device auth dictionary for the connect handshake.

        signed_at_ms should be the challenge's ts: the gateway enforces
        |now - signedAt| <= 120s against ITS clock, so signing our local time
        fails silently (no pending pairing) when the HA host clock drifts.
        """
        timestamp_ms = signed_at_ms or int(time.time() * 1000)
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
        locale: str | None = None,
    ) -> None:
        """Initialize OpenClaw client."""
        self._host = host
        self._port = port
        self._token = token
        self._use_ssl = use_ssl
        self._device_auth = device_auth
        self._timeout = timeout
        self._locale = _normalize_locale(locale)

        self._ws = None
        self._connected = False
        self._keepalive_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._agent_runs: Dict[str, "_AgentRun"] = {}
        # Agent events that arrived before send_message registered the run.
        self._early_agent_events: Dict[str, list] = {}
        self._event_handlers: Dict[str, list] = {}
        self._connect_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Return whether the client is connected."""
        return self._connected and self._ws is not None

    def _sanitized_host(self) -> str:
        """Return the host with any protocol prefix and trailing slash removed."""
        host = self._host.strip().rstrip("/")
        for prefix in ("https://", "http://", "wss://", "ws://"):
            if host.lower().startswith(prefix):
                return host[len(prefix):]
        return host

    def _build_ws_url(self) -> str:
        """Build the gateway websocket URL with the token percent-encoded.

        The token is also sent in the Authorization/X-OpenClaw-Token headers,
        but it is kept in the query string for gateway compatibility. Encoding
        it avoids a malformed URL (or a silently wrong token) when the token
        contains characters such as ``&``, ``#``, ``/`` or spaces.
        """
        scheme = "wss" if self._use_ssl else "ws"
        return f"{scheme}://{self._sanitized_host()}:{self._port}/?token={quote(self._token, safe='')}"

    async def connect(self) -> None:
        """Connect to the OpenClaw Gateway and complete handshake."""
        async with self._connect_lock:
            if self.is_connected:
                return
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        """Connect and complete the handshake; caller holds the connect lock."""
        from websockets.asyncio.client import connect

        # Tear down any previous connection first. A keepalive failure only
        # flips _connected without closing the socket or cancelling the
        # receive task, so without this a stale socket and receive loop would
        # linger past the reconnect. Preserve buffered early events so a
        # concurrent request still waiting to replay its completion is not
        # stranded by this teardown.
        await self._teardown_connection(
            "Reconnecting to OpenClaw Gateway", clear_early_events=False
        )

        scheme = "wss" if self._use_ssl else "ws"
        url = self._build_ws_url()
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-OpenClaw-Token": self._token,
        }

        # SSL context for self-signed certs (tailscale, etc.)
        # Avoid ssl.create_default_context() as it blocks the event loop loading certs
        ssl_ctx = None
        if self._use_ssl:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
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

        # Start background tasks, binding the receive loop to this socket so a
        # later reconnect can tell a stale loop apart from the live one.
        self._receive_task = asyncio.create_task(self._receive_loop(self._ws))
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        _LOGGER.info("✅ Connected to OpenClaw Gateway")

    async def _handshake(self) -> None:
        """Complete the WebSocket handshake with device auth."""
        # Wait for the connect.challenge event. Mandatory on protocol 4: a
        # device-less connect is rejected without registering a pending
        # pairing request, leaving nothing for `openclaw devices approve`.
        nonce = None
        challenge_ts = None
        deadline = asyncio.get_running_loop().time() + CHALLENGE_TIMEOUT
        while nonce is None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if msg.get("type") == "event" and msg.get("event") == "connect.challenge":
                payload = msg.get("payload", {})
                nonce = payload.get("nonce")
                challenge_ts = payload.get("ts")
                _LOGGER.debug("Received connect challenge")

        if not nonce:
            raise OpenClawConnectionError(
                "No connect.challenge received from gateway. "
                "OpenClaw >= 2026.5.12 (protocol 4) is required."
            )

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
            "locale": self._locale,
            "userAgent": f"{CLIENT_DISPLAY_NAME}/{CLIENT_VERSION}",
            "auth": {"token": self._token},
            "role": DEVICE_ROLE,
            "scopes": list(DEVICE_SCOPES),
        }

        # Device auth is always attached; sign the challenge's ts so validity
        # is judged against the gateway's clock, not ours
        connect_params["device"] = self._device_auth.build_device_dict(
            nonce, self._token, signed_at_ms=challenge_ts
        )

        await self._ws.send(json.dumps({
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": connect_params,
        }))

        # Wait for OUR response frame, skipping stray events — a late event
        # parsed as the connect response would produce a garbage error
        resp = None
        deadline = asyncio.get_running_loop().time() + HANDSHAKE_TIMEOUT
        while resp is None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError("Timeout waiting for connect response")
            raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("type") == "res" and msg.get("id") == connect_id:
                resp = msg

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
        await self._teardown_connection("Disconnected")

    async def _teardown_connection(
        self, reason: str, *, clear_early_events: bool = True
    ) -> None:
        """Cancel background tasks, close the socket, and fail in-flight work.

        Detaches the current socket/tasks first so a cancelled receive loop
        sees it no longer owns ``self._ws`` and skips its own cleanup, leaving
        this method as the single place that fails in-flight work.

        ``clear_early_events`` is False when tearing down before a reconnect:
        a completion may already be buffered for a request whose ``send_message``
        is between its resolved ack and registering the run, and clearing it
        would make that caller wait out its full timeout despite the answer.
        """
        self._connected = False
        ws = self._ws
        keepalive_task = self._keepalive_task
        receive_task = self._receive_task
        self._ws = None
        self._keepalive_task = None
        self._receive_task = None

        current = asyncio.current_task()
        for task in (keepalive_task, receive_task):
            if task and task is not current and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._fail_inflight(reason, clear_early_events=clear_early_events)

        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    def _fail_inflight(self, reason: str, *, clear_early_events: bool = True) -> None:
        """Fail pending requests and complete active runs with an error.

        ``clear_early_events`` is False for connection-loss cleanup from the
        receive loop: a completion may have arrived and been buffered for a run
        that ``send_message`` has not registered yet, and dropping it would make
        the caller wait out its full timeout despite having the answer. Runs
        that already completed are left untouched so an ok result is not
        overwritten with an error.
        """
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(OpenClawConnectionError(reason))
        self._pending_requests.clear()

        for run in self._agent_runs.values():
            if not run.complete_event.is_set():
                run.set_complete("error", reason)
        self._agent_runs.clear()

        if clear_early_events:
            self._early_agent_events.clear()

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
            _LOGGER.info("OpenClaw not connected, attempting reconnect...")
            try:
                await self.connect()
            except Exception as err:
                raise OpenClawConnectionError(
                    f"Not connected and reconnect failed: {err}"
                ) from err

        request_id = str(uuid.uuid4())
        idempotency_key = str(uuid.uuid4())

        # The message is sent unmodified: the OpenClaw agent's own prompts
        # (soul/agent/memory) govern style, and TTS formatting is handled
        # downstream by the clean_responses option

        # Register the response future before sending so a fast acknowledgment
        # processed during the send await cannot be dropped.
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future

        try:
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
        except Exception as err:
            self._pending_requests.pop(request_id, None)
            raise OpenClawConnectionError(f"Failed to send agent request: {err}") from err

        _LOGGER.debug("Sent agent request: %s", request_id[:8])

        # Wait for the initial response (contains runId)
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

        # Track this run, replaying any events that beat the registration.
        run = _AgentRun(run_id)
        self._agent_runs[run_id] = run
        for payload in self._early_agent_events.pop(run_id, []):
            self._apply_agent_event(run, payload)

        # If the socket dropped between the ack resolving and now, no receive
        # loop is left to complete this run — fail fast instead of waiting the
        # full timeout. A completion that already arrived was replayed above,
        # so only bail when the run is still pending.
        if not run.complete_event.is_set() and not self.is_connected:
            self._agent_runs.pop(run_id, None)
            raise OpenClawConnectionError("Connection to OpenClaw Gateway lost")

        try:
            await asyncio.wait_for(run.complete_event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise OpenClawTimeoutError(
                f"OpenClaw agent timed out after {self._timeout}s"
            )
        finally:
            self._agent_runs.pop(run_id, None)

        _LOGGER.debug(
            "Run complete: status=%s summary_len=%s full_text_len=%s",
            run.status, len(run.summary or ""), len(run.full_text),
        )

        if run.status == "error":
            raise OpenClawError(f"Agent error: {run.summary or 'unknown error'}")

        return run.summary or run.full_text

    async def _receive_loop(self, ws) -> None:
        """Background task to receive and dispatch WebSocket messages."""
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_message(msg)
                except json.JSONDecodeError:
                    _LOGGER.debug("Invalid JSON from gateway")
                except Exception as err:
                    _LOGGER.debug("Error handling message: %s", err)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _LOGGER.warning("WebSocket receive loop ended: %s", err)
        finally:
            # Only tear down shared state if this loop still owns the active
            # socket. A stale loop whose socket was already replaced by a
            # reconnect must not disconnect or fail the new connection.
            if self._ws is ws:
                self._connected = False
                # Preserve buffered early events: a completion may have arrived
                # for a run that send_message has not registered yet, and it
                # still needs to replay it.
                self._fail_inflight(
                    "Connection to OpenClaw Gateway lost",
                    clear_early_events=False,
                )

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

    # Bounds for events buffered before their run is registered.
    _MAX_EARLY_EVENT_RUNS = 8
    _MAX_EARLY_EVENTS_PER_RUN = 32

    def _handle_agent_event(self, payload: Dict[str, Any]) -> None:
        """Handle an agent event (streaming output or completion)."""
        run_id = payload.get("runId")
        if not run_id:
            return

        run = self._agent_runs.get(run_id)
        if not run:
            # The acknowledgment and first events can be processed back to
            # back, before send_message registers the run. Buffer them so
            # send_message can replay them instead of dropping completions.
            _LOGGER.debug(
                "Buffering agent event for unregistered run %s, keys: %s",
                run_id[:8], list(payload.keys()),
            )
            events = self._early_agent_events.setdefault(run_id, [])
            events.append(payload)
            del events[:-self._MAX_EARLY_EVENTS_PER_RUN]
            while len(self._early_agent_events) > self._MAX_EARLY_EVENT_RUNS:
                self._early_agent_events.pop(next(iter(self._early_agent_events)))
            return

        self._apply_agent_event(run, payload)

    def _apply_agent_event(self, run: "_AgentRun", payload: Dict[str, Any]) -> None:
        """Apply a single agent event to its run."""
        _LOGGER.debug(
            "Agent event: run=%s keys=%s output_len=%s status=%s phase=%s",
            run.run_id[:8],
            list(payload.keys()),
            len(payload.get("output", "")) if payload.get("output") else 0,
            payload.get("status"),
            payload.get("data", {}).get("phase"),
        )

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
