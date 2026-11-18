import asyncio
import base64
import re
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

import httpx
import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from channel_config import ChannelRegistry, RegistryError
from channel_catalog import CatalogCache
from provider_wire import (account_api_key, account_proxy_url, build_target_url,
                           build_upstream_headers, normalize_base_url,
                           parse_json_body_for_model, prepare_upstream_body)
from provider_capacity import (fetch_gproxy_claudecode_capacity, gproxy_five_hour_window,
                               limited_recovery_probe)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("ACCOUNT_POOL_CONFIG", APP_DIR / "pools.json"))
REQUEST_TIMEOUT_RAW = float(os.environ.get("ACCOUNT_POOL_TIMEOUT", "600"))
REQUEST_TIMEOUT: float | None = None if REQUEST_TIMEOUT_RAW <= 0 else REQUEST_TIMEOUT_RAW

app = FastAPI(title="account-pool-proxy", version="0.1.0")
LOG = logging.getLogger("account-pool-proxy")
catalog_cache = CatalogCache()


async def _fetch_channel_catalog(account, spec, caller_authorization):
    headers = [(b'accept', b'application/json')]
    if caller_authorization:
        headers.append((b'x-acc-passthrough-authorization', caller_authorization.encode('latin-1')))
    request = Request({'type': 'http', 'method': 'GET', 'path': '/v1/models',
                       'query_string': b'', 'headers': headers, 'scheme': 'http',
                       'server': ('catalog.local', 80)})
    path = spec.get('path', 'models')
    async with httpx.AsyncClient(timeout=spec.get('timeout_seconds', 4),
            proxy=account_proxy_url(account) or None, trust_env=False, follow_redirects=False) as client:
        response = await client.get(build_target_url(account, request, path),
                                    headers=build_upstream_headers(account, request, path))
        response.raise_for_status()
        return response.json()


async def discovered_registry(request, *, allowed_ids=None, config=None):
    # A request reads one local snapshot; discovery cannot mutate live config.
    snapshot = copy.deepcopy(state.config) if config is None else config
    return await catalog_cache.compile(snapshot,
        caller_authorization=str(request.headers.get('x-acc-passthrough-authorization') or ''),
        fetch=_fetch_channel_catalog, log=LOG.info, allowed_ids=allowed_ids)


async def registry_for_model(request, model):
    snapshot = copy.deepcopy(state.config)
    registry = ChannelRegistry(snapshot)
    # Explicit static names/prefixes never fetch any provider's model catalog.
    # Only an unknown name can need the one configured dynamic source (Mio).
    if registry.resolve(model) is not None:
        return registry
    return await discovered_registry(request, config=snapshot)


class UpstreamStreamingResponse(StreamingResponse):
    """Own the upstream sockets even before the first body iteration starts."""
    def __init__(self, content, *, close_upstream, **kwargs):
        super().__init__(content, **kwargs)
        self.close_upstream = close_upstream

    async def aclose(self):
        with anyio.CancelScope(shield=True):
            await self.close_upstream()
            close = getattr(self.body_iterator, 'aclose', None)
            if callable(close):
                await close()

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()


async def _until_disconnect(request: Request, operation):
    """Cancel a header wait/preflight/JSON call when its client has gone away.

    The request body has already been consumed. Ownership of receive returns to
    StreamingResponse only after both request-owned tasks are settled.
    """
    async def watch():
        try:
            while True:
                event = await request.receive()
                if event['type'] == 'http.disconnect':
                    return True
        except RuntimeError:
            return False  # Direct unit-test Requests may not have a receive.

    work = asyncio.create_task(operation)
    monitor = asyncio.create_task(watch())
    response = None
    handed_off = False
    try:
        done, _ = await asyncio.wait({work, monitor}, return_when=asyncio.FIRST_COMPLETED)
        if monitor in done and monitor.result():
            if not work.done():
                work.cancel()
            LOG.info('upstream_cancel reason=downstream_disconnected phase=before_response')
            return Response(status_code=499)
        response = await work
        handed_off = True
        return response
    finally:
        with anyio.CancelScope(shield=True):
            if not monitor.done():
                monitor.cancel()
            if not handed_off and not work.done():
                work.cancel()
            results = await asyncio.gather(work, monitor, return_exceptions=True)
            if not handed_off:
                candidate = results[0]
                if isinstance(candidate, UpstreamStreamingResponse):
                    await candidate.aclose()


def now_ts() -> float:
    return time.time()


def mask_key(key: str) -> str:
    key = str(key or "")
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def request_client_api_key(request: Request) -> str:
    auth = str(request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return (
        str(request.headers.get("x-api-key") or "").strip()
        or str(request.headers.get("x-goog-api-key") or "").strip()
    )


def account_client_keys(account: dict[str, Any]) -> set[str]:
    raw = account.get("client_api_keys")
    if raw is None:
        raw = account.get("client_api_key")
    if raw is None:
        raw = account.get("local_api_key")
    if raw is None:
        return set()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return set()
    return {str(x).strip() for x in items if str(x).strip()}


def is_pioneer_account(account: dict[str, Any]) -> bool:
    channel = str(account.get("channel") or "").lower()
    base_url = str(account.get("base_url") or "").lower()
    account_id = str(account.get("id") or "").lower()
    return channel in {"pio", "pioneer"} or "pioneer.ai" in base_url or account_id.startswith("pio")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"accounts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    previous_stat = None
    try:
        previous_stat = path.stat()
    except FileNotFoundError:
        pass
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError as e:
        # Docker single-file bind mounts make /app/pools.json a mount point;
        # replacing that path can fail with EBUSY and abort an active stream.
        # Fall back to in-place write so status bookkeeping never kills SSE.
        if getattr(e, "errno", None) != 16:
            raise
        path.write_text(text, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    if previous_stat is not None:
        try:
            os.chmod(path, previous_stat.st_mode & 0o777)
        except Exception:
            pass
        try:
            os.chown(path, previous_stat.st_uid, previous_stat.st_gid)
        except Exception:
            pass


def parse_reset(reset_time: str) -> dtime:
    hh, mm = str(reset_time or "16:00").split(":", 1)
    return dtime(hour=int(hh), minute=int(mm))


def reset_due(account: dict[str, Any], ts: float | None = None) -> bool:
    if account.get("auto_reset_enabled") is False:
        return False
    status = str(account.get("status") or "ok")
    # Pool only records quota/limit state.  Generic upstream errors must not
    # create sticky account states.
    if status != "limited":
        return False
    reset_time = str(account.get("reset_time") or "16:00")
    tz_name = str(account.get("timezone") or "Asia/Shanghai")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("Asia/Shanghai")
    now = datetime.fromtimestamp(ts or now_ts(), tz)
    rt = parse_reset(reset_time)
    today_reset = datetime.combine(now.date(), rt, tzinfo=tz)
    # If now is before today's reset clock time, the most recent reset window is
    # yesterday's reset, not "not due".  Example: reset=20:00 UTC, now=04:00 UTC;
    # the account should have reset at yesterday 20:00 UTC.
    current_reset = today_reset if now >= today_reset else today_reset - timedelta(days=1)
    last_reset_at = float(account.get("last_reset_at") or 0)
    last_reset_dt = datetime.fromtimestamp(last_reset_at, tz) if last_reset_at > 0 else None
    if last_reset_dt is not None and last_reset_dt >= current_reset:
        return False
    return True


def seconds_until_next_reset(account: dict[str, Any], ts: float | None = None) -> float:
    """Return seconds from now until this account's next configured reset.

    Used for "use the key closest to refresh first" rotation.  If a reset config
    is malformed, fall back to a large value so the account is not preferred.
    """
    reset_time = str(account.get("reset_time") or "16:00")
    tz_name = str(account.get("timezone") or "Asia/Shanghai")
    try:
        tz = ZoneInfo(tz_name)
        rt = parse_reset(reset_time)
    except Exception:
        return 365.0 * 24.0 * 3600.0

    now = datetime.fromtimestamp(ts or now_ts(), tz)
    next_reset = datetime.combine(now.date(), rt, tzinfo=tz)
    if now >= next_reset:
        next_reset = next_reset + timedelta(days=1)
    return max(0.0, (next_reset - now).total_seconds())


def account_sort_enabled(account: dict[str, Any]) -> bool:
    """Whether the account participates in reset-distance sorting.

    Default is true for backwards compatibility.  Set sort_enabled=false on
    emergency/long-term backup keys; they will only be used when all sorted keys
    are unavailable.
    """
    return bool(account.get("sort_enabled", True))


@dataclass
class PoolState:
    config_mtime: float = 0.0
    config: dict[str, Any] = field(default_factory=lambda: {"accounts": []})
    rr_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    probe_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

state = PoolState()


def account_public(account: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in account.items() if k not in {"api_key", "client_api_key", "client_api_keys", "local_api_key"}}
    out["key"] = mask_key(account_api_key(account))
    return out


async def reload_config_if_needed(force: bool = False) -> None:
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    if not force and mtime == state.config_mtime:
        return
    async with state.lock:
        try:
            mtime2 = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            mtime2 = 0.0
        if not force and mtime2 == state.config_mtime:
            return
        cfg = load_json(CONFIG_PATH)
        accounts = cfg.get("accounts")
        if not isinstance(accounts, list):
            accounts = []
        for idx, acc in enumerate(accounts):
            if not isinstance(acc, dict):
                continue
            acc.setdefault("id", f"account_{idx+1}")
            acc.setdefault("status", "ok")
            # Historical versions could persist status=error.  Treat that as
            # healthy on load; only limited/disabled are sticky pool states.
            if str(acc.get("status") or "ok") == "error":
                acc["status"] = "ok"
            acc.setdefault("timezone", "Asia/Shanghai")
            acc.setdefault("reset_time", "16:00")
            if not acc.get("protocol") and str(acc.get("channel") or "").lower() in {"pio", "pioneer"}:
                acc["protocol"] = "anthropic"
            acc.setdefault("primary", False)
            if acc.get("base_url"):
                acc["base_url"] = normalize_base_url(str(acc["base_url"]))
        cfg["accounts"] = [a for a in accounts if isinstance(a, dict)]
        state.config = cfg
        state.config_mtime = mtime2


async def persist_config() -> None:
    async with state.lock:
        save_json_atomic(CONFIG_PATH, state.config)
        try:
            state.config_mtime = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            state.config_mtime = 0.0


async def apply_daily_resets() -> list[str]:
    await reload_config_if_needed()
    changed: list[str] = []
    ts = now_ts()
    for acc in state.config.get("accounts", []):
        if reset_due(acc, ts):
            old = acc.get("status")
            acc["status"] = "ok"
            acc["last_reset_at"] = ts
            acc["last_status_change_at"] = ts
            acc["last_status_reason"] = f"daily_reset_from_{old}"
            changed.append(str(acc.get("id")))
    if changed:
        await persist_config()
    return changed


async def apply_limited_recovery_probes() -> list[str]:
    """Recover configured limited accounts from authoritative upstream usage."""
    await reload_config_if_needed()
    recovered: list[str] = []
    changed = False
    async with state.probe_lock:
        current = now_ts()
        for account in state.config.get("accounts", []):
            if str(account.get("status") or "ok") != "limited":
                continue
            probe = limited_recovery_probe(account)
            if probe is None:
                continue
            if current < float(account.get("next_limited_probe_at") or 0):
                continue
            interval = max(15.0, min(3600.0, float(probe.get("interval_seconds") or 60.0)))
            account["last_limited_probe_at"] = current
            try:
                available, used_percent, reset_timestamp = await fetch_gproxy_claudecode_capacity(
                    account,
                    probe,
                )
                account["last_limited_probe_used_percent"] = used_percent
                account["last_limited_probe_error"] = ""
                if available:
                    account["status"] = "ok"
                    account["last_status_change_at"] = current
                    account["last_status_reason"] = "gproxy_usage_probe_recovered"
                    account.pop("next_limited_probe_at", None)
                    recovered.append(str(account.get("id")))
                else:
                    account["next_limited_probe_at"] = max(
                        current + interval,
                        reset_timestamp or 0,
                    )
            except Exception as exc:
                account["last_limited_probe_error"] = f"{type(exc).__name__}: {exc}"[:300]
                account["next_limited_probe_at"] = current + interval
            changed = True
    if changed:
        await persist_config()
    return recovered


def account_matches_model(account: dict[str, Any], model: str) -> bool:
    mapping = account.get("upstream_model_map") or {}
    if model in mapping:
        return True
    models = account.get("models")
    if not models:
        return True
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, list):
        return True
    model_l = str(model or "").lower()
    for item in models:
        pat = str(item or "").lower().strip()
        if not pat:
            continue
        if pat == "*":
            return True
        if not model_l:
            continue
        if pat == model_l or (
            account.get("model_match") != "exact" and (pat in model_l or model_l in pat)
        ):
            return True
    return False


def account_supports_path(account: dict[str, Any], path: str) -> bool:
    allowed = account.get("allowed_paths")
    return not allowed or not path or path.strip("/").removeprefix("v1/") in allowed


def model_verification_enabled(account: dict[str, Any], requested_model: str) -> bool:
    mode = str(account.get("model_verification_mode") or "strict").strip().lower()
    return bool(
        account.get("verify_upstream_model") is True
        and mode == "strict"
        and str(requested_model or "").strip()
    )


def canonical_model_identity(account: dict[str, Any], model: str) -> str:
    identity = str(model or "").strip().lower()
    aliases = account.get("model_aliases")
    if not identity or not isinstance(aliases, dict):
        return identity

    for canonical, configured_aliases in aliases.items():
        canonical_id = str(canonical or "").strip().lower()
        if not canonical_id:
            continue
        if identity == canonical_id:
            return canonical_id
        if isinstance(configured_aliases, str):
            alias_ids = [configured_aliases]
        elif isinstance(configured_aliases, list):
            alias_ids = configured_aliases
        else:
            continue
        if identity in {str(alias or "").strip().lower() for alias in alias_ids}:
            return canonical_id

    return identity


def model_identities_match(account: dict[str, Any], requested_model: str, reported_model: str) -> bool:
    return canonical_model_identity(account, requested_model) == canonical_model_identity(account, reported_model)


def extract_reported_model(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    direct = payload.get("model")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    for container_name in ("message", "response", "data"):
        container = payload.get(container_name)
        if not isinstance(container, dict):
            continue
        nested = container.get("model")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()

    return ""


def payload_contains_generated_content(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    event_type = str(payload.get("type") or "").lower()
    if event_type in {"content_block_start", "content_block_delta"}:
        return True
    if event_type.startswith("response.") and any(marker in event_type for marker in ("delta", "output", "text")):
        return True

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for part_name in ("delta", "message"):
                part = choice.get(part_name)
                if not isinstance(part, dict):
                    continue
                for key, value in part.items():
                    if key == "role":
                        continue
                    if value not in (None, "", [], {}):
                        return True
            if choice.get("text") not in (None, ""):
                return True

    for key in ("content", "output", "output_text", "text"):
        if payload.get(key) not in (None, "", [], {}):
            return True

    return False


class SseModelIdentityProbe:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.reported_model = ""
        self.saw_generated_content = False
        self.saw_done = False

    def feed(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        while True:
            frame = self._pop_frame()
            if frame is None:
                return
            self._inspect_frame(frame)

    def _pop_frame(self) -> bytes | None:
        boundaries: list[tuple[int, int]] = []
        lf_index = self.buffer.find(b"\n\n")
        if lf_index >= 0:
            boundaries.append((lf_index, 2))
        crlf_index = self.buffer.find(b"\r\n\r\n")
        if crlf_index >= 0:
            boundaries.append((crlf_index, 4))
        if not boundaries:
            return None

        index, delimiter_length = min(boundaries, key=lambda item: item[0])
        frame = bytes(self.buffer[:index])
        del self.buffer[: index + delimiter_length]
        return frame

    def _inspect_frame(self, frame: bytes) -> None:
        data_lines: list[bytes] = []
        for line in frame.replace(b"\r\n", b"\n").split(b"\n"):
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return

        raw_data = b"\n".join(data_lines).strip()
        if raw_data == b"[DONE]":
            self.saw_done = True
            return
        try:
            payload = json.loads(raw_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        reported_model = extract_reported_model(payload)
        if reported_model and not self.reported_model:
            self.reported_model = reported_model
        if payload_contains_generated_content(payload):
            self.saw_generated_content = True


@dataclass
class StreamModelPreflight:
    iterator: Any
    buffered_chunks: list[bytes] = field(default_factory=list)
    reported_model: str = ""
    error_type: str = ""
    error_detail: str = ""


async def preflight_stream_model_identity(
    resp: httpx.Response,
    account: dict[str, Any],
    requested_model: str,
) -> StreamModelPreflight:
    iterator = resp.aiter_bytes().__aiter__()
    result = StreamModelPreflight(iterator=iterator)
    if not model_verification_enabled(account, requested_model):
        return result

    try:
        timeout = max(0.1, min(120.0, float(account.get("model_probe_timeout_seconds") or 15.0)))
        max_bytes = max(1024, min(1024 * 1024, int(account.get("model_probe_max_bytes") or 65536)))
    except (TypeError, ValueError):
        timeout = 15.0
        max_bytes = 65536

    deadline = asyncio.get_running_loop().time() + timeout
    buffered_size = 0
    probe = SseModelIdentityProbe()

    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = "timed out before the upstream declared its model identity"
            return result
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = "upstream ended before declaring its model identity"
            return result
        except TimeoutError:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = "timed out before the upstream declared its model identity"
            return result
        except Exception as exc:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = f"upstream failed before declaring its model identity: {type(exc).__name__}"
            return result

        if not chunk:
            continue
        result.buffered_chunks.append(chunk)
        buffered_size += len(chunk)
        probe.feed(chunk)

        if probe.reported_model:
            result.reported_model = probe.reported_model
            if not model_identities_match(account, requested_model, probe.reported_model):
                result.error_type = "upstream_model_mismatch"
                result.error_detail = "upstream reported a different model"
            return result
        if probe.saw_generated_content:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = "upstream emitted generated content before declaring its model identity"
            return result
        if probe.saw_done:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = "upstream completed before declaring its model identity"
            return result
        if buffered_size >= max_bytes:
            result.error_type = "upstream_model_identity_unverifiable"
            result.error_detail = "model identity probe exceeded its byte limit"
            return result


def verify_non_stream_model_identity(
    account: dict[str, Any],
    requested_model: str,
    content: bytes,
) -> tuple[str, str, str] | None:
    if not model_verification_enabled(account, requested_model):
        return None
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            "upstream_model_identity_unverifiable",
            "",
            "upstream returned a non-JSON response without a model identity",
        )

    reported_model = extract_reported_model(payload)
    if not reported_model:
        return (
            "upstream_model_identity_unverifiable",
            "",
            "upstream response did not declare its model identity",
        )
    if not model_identities_match(account, requested_model, reported_model):
        return ("upstream_model_mismatch", reported_model, "upstream reported a different model")
    return None


async def note_model_verification_failure(
    account: dict[str, Any],
    requested_model: str,
    reported_model: str,
    error_type: str,
) -> None:
    account["model_verification_failure_count"] = int(account.get("model_verification_failure_count") or 0) + 1
    account["last_model_verification_failure_at"] = now_ts()
    account["last_requested_model"] = requested_model
    account["last_reported_model"] = reported_model
    account["last_model_verification_error"] = error_type
    LOG.warning(
        "upstream_model_verification_failed account=%s requested=%s reported=%s type=%s",
        account.get("id"),
        requested_model,
        reported_model or "<missing>",
        error_type,
    )
    await persist_config()


def model_verification_error_response(
    account: dict[str, Any],
    requested_model: str,
    reported_model: str,
    error_type: str,
    detail: str,
) -> JSONResponse:
    account_id = str(account.get("id") or "")
    return JSONResponse(
        {
            "error": {
                "message": detail,
                "type": error_type,
                "account_id": account_id,
                "requested_model": requested_model,
                "reported_model": reported_model or None,
            }
        },
        status_code=502,
        headers={
            "x-account-pool-id": account_id,
            "x-account-pool-status": str(account.get("status") or "ok"),
        },
    )


def _registry_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"type": "channel_routing_error", "message": message}}, status_code=status)


def _request_model(path: str, body: bytes) -> tuple[str, bool]:
    model, streaming = parse_json_body_for_model(body)
    native = re.fullmatch(r"(?:v1beta/)?models/(.+):(generateContent|streamGenerateContent)", path)
    if native:
        return unquote(native.group(1)), native.group(2) == "streamGenerateContent"
    return model, streaming


def _rewrite_public_model(path: str, body: bytes, upstream_model: str) -> tuple[str, bytes]:
    native = re.fullmatch(r"((?:v1beta/)?models/).+(:generateContent|:streamGenerateContent)", path)
    if native:
        return native.group(1) + quote(upstream_model, safe="-._~") + native.group(2), body
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Expected an object")
    payload["model"] = upstream_model
    return path, json.dumps(payload, ensure_ascii=False).encode()


async def forward_raw(path: str, request: Request) -> Response | StreamingResponse | JSONResponse:
    body_bytes = await request.body()
    model, wants_stream = _request_model(path, body_bytes)
    client_api_key = request_client_api_key(request)
    await reload_config_if_needed()
    try:
        registry = ChannelRegistry(state.config)
    except (ValueError, TypeError, KeyError):
        LOG.error("channel_registry_invalid")
        return _registry_error("Invalid channel configuration", 503)
    resolution = None
    if registry.enabled and registry.authorized(client_api_key):
        if request.method != "POST":
            return _registry_error("Managed channels only accept inference POSTs", 405)
        registry = await registry_for_model(request, model)
        resolution = registry.resolve(model)
        if resolution is None:
            return _registry_error("Unknown model in channel registry", 404)
        if resolution.transport == "local":
            return _registry_error("Local diagnostic model is served by Stream", 400)
        try:
            path, body_bytes = _rewrite_public_model(path, body_bytes, resolution.upstream_model)
        except (ValueError, TypeError):
            return _registry_error("Invalid inference body")
        model = resolution.upstream_model
        LOG.info("channel_route channel=%s route=%s public_model=%s upstream_model=%s revision=%s",
                 resolution.channel, resolution.route, resolution.public_model, model, resolution.revision)
    response = await _until_disconnect(request, _forward_raw(
        path, request, body_bytes=body_bytes, model=model, wants_stream=wants_stream,
        client_api_key=client_api_key,
        allowed_ids=resolution.account_ids if resolution else None,
        account_snapshot=registry.accounts if resolution else None,
    ))
    if resolution:
        if registry.schema == 1:  # Old Stream rollout compatibility only.
            account = registry.accounts.get(response.headers.get("x-account-pool-id"), {})
            policy = registry.selected_policy(resolution, account)
            response.headers["x-acc-early-stop"] = base64.urlsafe_b64encode(json.dumps(
                policy.get("early_stop", {}), separators=(",", ":")).encode()).decode()
        response.headers["x-acc-channel"] = resolution.channel
        response.headers["x-acc-route"] = resolution.route
        response.headers["x-acc-revision"] = resolution.revision
    return response


async def _forward_raw(path: str, request: Request, *, body_bytes: bytes, model: str,
                       wants_stream: bool, client_api_key: str,
                       allowed_ids: frozenset[str] | None = None,
                       account_snapshot: dict[str, dict] | None = None) -> Response | StreamingResponse | JSONResponse:
    method = request.method.upper()
    tried: set[str] = set()
    last_limit_response: Response | None = None

    while True:
        acc = await choose_account(model, exclude_ids=tried, client_api_key=client_api_key, path=path, allowed_ids=allowed_ids, account_snapshot=account_snapshot)
        if acc is None:
            if last_limit_response is not None:
                return last_limit_response
            return JSONResponse({"error": {"message": "No ok account available", "type": "account_pool_empty"}}, status_code=503)

        account_id = str(acc.get("id"))
        tried.add(account_id)
        target = build_target_url(acc, request, path)
        headers = build_upstream_headers(acc, request, path)

        if wants_stream:
            return await forward_raw_stream_with_account(
                account=acc,
                account_id=account_id,
                target=target,
                method=method,
                body_bytes=body_bytes,
                headers=headers,
                tried=tried,
                model=model,
                path=path,
                request=request,
                client_api_key=client_api_key,
                allowed_ids=allowed_ids,
                account_snapshot=account_snapshot,
            )

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, proxy=account_proxy_url(acc) or None, trust_env=False) as client:
                resp = await client.request(
                    method, target, content=prepare_upstream_body(acc, path, body_bytes), headers=headers
                )
                content = resp.content
                media = resp.headers.get("content-type", "application/json")
                if 200 <= resp.status_code < 300:
                    verification_error = verify_non_stream_model_identity(acc, model, content)
                    if verification_error is not None:
                        error_type, reported_model, detail = verification_error
                        await note_model_verification_failure(acc, model, reported_model, error_type)
                        return model_verification_error_response(
                            acc,
                            model,
                            reported_model,
                            error_type,
                            detail,
                        )
                    note_success(account_id)
                    await persist_config()
                    return Response(
                        content=content,
                        status_code=resp.status_code,
                        media_type=media,
                        headers={"x-account-pool-id": account_id, "x-account-pool-status": str(acc.get("status") or "ok")},
                    )
                if note_limit(account_id, resp.status_code, content.decode(errors="replace")[:500]):
                    await persist_config()
                    last_limit_response = Response(
                        content=content,
                        status_code=resp.status_code,
                        media_type=media,
                        headers={"x-account-pool-id": account_id, "x-account-pool-status": str(acc.get("status") or "ok")},
                    )
                    continue
                return Response(
                    content=content,
                    status_code=resp.status_code,
                    media_type=media,
                    headers={"x-account-pool-id": account_id, "x-account-pool-status": str(acc.get("status") or "ok")},
                )
        except Exception as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "account_pool_upstream_error", "account_id": account_id}},
                status_code=502,
            )


async def forward_raw_stream_with_account(
    *,
    account: dict[str, Any],
    account_id: str,
    target: str,
    method: str,
    body_bytes: bytes,
    headers: dict[str, str],
    tried: set[str],
    model: str,
    path: str,
    request: Request,
    client_api_key: str = "",
    allowed_ids: frozenset[str] | None = None,
    account_snapshot: dict[str, dict] | None = None,
) -> Response | StreamingResponse | JSONResponse:
    transient_attempt = 0

    while True:
        retries = max(0, min(5, int(account.get("transient_5xx_retries") or 0)))
        delay = max(0.0, min(10.0, float(account.get("transient_retry_delay_seconds") or 0.5)))
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT, read=REQUEST_TIMEOUT),
            proxy=account_proxy_url(account) or None,
            trust_env=False,
        )
        try:
            upstream_request = client.build_request(
                method, target, content=prepare_upstream_body(account, path, body_bytes), headers=headers
            )
            resp = await client.send(upstream_request, stream=True)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                await client.aclose()
            raise
        except Exception as exc:
            await client.aclose()
            if transient_attempt < retries:
                transient_attempt += 1
                LOG.warning(
                    "upstream_stream_retry account=%s reason=transport attempt=%d/%d error=%s",
                    account_id,
                    transient_attempt,
                    retries,
                    type(exc).__name__,
                )
                if delay:
                    await asyncio.sleep(delay * transient_attempt)
                continue
            return JSONResponse(
                {"error": {"message": str(exc), "type": "account_pool_upstream_error", "account_id": account_id}},
                status_code=502,
            )

        if not 200 <= resp.status_code < 300:
            status_code = resp.status_code
            media_type = resp.headers.get("content-type", "application/json")
            data = await resp.aread()
            await resp.aclose()
            await client.aclose()

            if note_limit(account_id, status_code, data.decode(errors="replace")[:500]):
                await persist_config()
                next_acc = await choose_account(model, exclude_ids=tried, client_api_key=client_api_key, path=path, allowed_ids=allowed_ids, account_snapshot=account_snapshot)
                if next_acc is None:
                    return Response(
                        content=data,
                        status_code=status_code,
                        media_type=media_type,
                        headers={"x-account-pool-id": account_id, "x-account-pool-status": str(account.get("status") or "ok")},
                    )
                account = next_acc
                account_id = str(account.get("id"))
                tried.add(account_id)
                target = build_target_url(account, request, path)
                headers = build_upstream_headers(account, request, path)
                transient_attempt = 0
                continue

            if 500 <= status_code < 600 and transient_attempt < retries:
                transient_attempt += 1
                LOG.warning(
                    "upstream_stream_retry account=%s reason=http_%d attempt=%d/%d",
                    account_id,
                    status_code,
                    transient_attempt,
                    retries,
                )
                if delay:
                    await asyncio.sleep(delay * transient_attempt)
                continue

            return Response(
                content=data,
                status_code=status_code,
                media_type=media_type,
                headers={"x-account-pool-id": account_id, "x-account-pool-status": str(account.get("status") or "ok")},
            )

        try:
            preflight = await preflight_stream_model_identity(resp, account, model)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await resp.aclose()
                await client.aclose()
            raise
        if preflight.error_type:
            await resp.aclose()
            await client.aclose()
            await note_model_verification_failure(
                account,
                model,
                preflight.reported_model,
                preflight.error_type,
            )
            return model_verification_error_response(
                account,
                model,
                preflight.reported_model,
                preflight.error_type,
                preflight.error_detail,
            )

        note_success(account_id)
        await persist_config()
        response_headers = {
            "x-account-pool-id": account_id,
            "x-account-pool-status": str(account.get("status") or "ok"),
            "content-type": resp.headers.get("content-type", "text/event-stream"),
        }

        async def gen():
            first_chunk = False
            try:
                for chunk in preflight.buffered_chunks:
                    if chunk:
                        first_chunk = True
                        yield chunk
                async for chunk in preflight.iterator:
                    if chunk:
                        first_chunk = True
                        yield chunk
            except Exception as exc:
                if is_non_degrading_stream_error(exc, first_chunk=first_chunk, upstream_2xx=True):
                    note_stream_transport_error(account_id, f"{type(exc).__name__}: {exc}")
                    await persist_config()
                elif not first_chunk:
                    yield (json.dumps({
                        "error": {
                            "message": str(exc),
                            "type": "account_pool_upstream_error",
                            "account_id": account_id,
                        }
                    }, ensure_ascii=False) + "\n").encode()
            finally:
                await resp.aclose()
                await client.aclose()

        async def close_upstream():
            await resp.aclose()
            await client.aclose()

        return UpstreamStreamingResponse(gen(), close_upstream=close_upstream,
                                         status_code=resp.status_code, headers=response_headers)


async def choose_account(model: str = "", exclude_ids: set[str] | None = None, client_api_key: str = "", path: str = "", allowed_ids: frozenset[str] | None = None, account_snapshot: dict[str, dict] | None = None) -> dict[str, Any] | None:
    await apply_daily_resets()
    await apply_limited_recovery_probes()
    accounts = state.config.get("accounts", [])
    excluded = {str(x) for x in (exclude_ids or set())}

    # If the client presents a local account key, replace it by selecting only
    # accounts bound to that local key.  Otherwise, keep the existing pool
    # behavior and ignore local-key-bound CLI accounts.
    client_key = str(client_api_key or "").strip()
    matching_bound = [a for a in accounts if client_key and client_key in account_client_keys(a)]
    candidate_accounts = matching_bound if matching_bound else [a for a in accounts if not account_client_keys(a)]
    if allowed_ids is not None:
        # Only the authenticated registry resolver may pass an explicit scope.
        # Keep quota/status/priority/rotation logic identical within this scope.
        candidate_accounts = [a for a in accounts if str(a.get("id")) in allowed_ids]
        if account_snapshot is not None:
            candidate_accounts = [dict(account_snapshot[str(a['id'])], status=a.get('status', 'ok'),
                                       last_reset_at=a.get('last_reset_at', 0))
                                  for a in candidate_accounts if str(a['id']) in account_snapshot]

    # Pioneer router requests must stay on Pioneer keys.  If Pioneer keys are
    # unavailable, return pool-empty instead of falling through to other
    # Anthropic-compatible accounts that do not support model=pioneer/auto.
    if str(model or "").strip().lower() in {"pioneer/auto", "anthropic/pioneer-auto"}:
        candidate_accounts = [a for a in candidate_accounts if is_pioneer_account(a)]

    usable = [
        a for a in candidate_accounts
        if str(a.get("status") or "ok") == "ok"
        and str(a.get("id")) not in excluded
        and (account_api_key(a) or a.get("auth_mode") in {"none", "caller"})
        and a.get("base_url")
        and account_matches_model(a, model)
        and account_supports_path(a, path)
    ]
    if not usable:
        return None

    def priority_of(account: dict[str, Any]) -> int:
        try:
            return int(account.get("priority"))
        except Exception:
            return 50

    # One-shot Pioneer keys should be drained sequentially: keep using the
    # lowest-priority OK Pioneer account until it is marked limited, then move
    # to the next one.  Other account types keep the historical rotation logic.
    pioneer_usable = [a for a in usable if is_pioneer_account(a)]
    if pioneer_usable:
        best_priority = min(priority_of(a) for a in pioneer_usable)
        pioneer_pool = [a for a in pioneer_usable if priority_of(a) == best_priority]
        pioneer_pool.sort(key=lambda a: str(a.get("id") or ""))
        return pioneer_pool[0]

    # Selection order for non-Pioneer pools:
    #   1. OK accounts with sort_enabled=true, ordered by nearest next reset
    #   2. OK accounts with sort_enabled=false as fallback only
    # Within the chosen bucket, exact ties use round-robin.
    sorted_usable = [a for a in usable if account_sort_enabled(a)]
    if sorted_usable:
        ts = now_ts()
        best_seconds = min(seconds_until_next_reset(a, ts) for a in sorted_usable)
        pool = [
            a for a in sorted_usable
            if abs(seconds_until_next_reset(a, ts) - best_seconds) < 1.0
        ]
    else:
        fallback = [a for a in usable if not account_sort_enabled(a)]
        best_priority = min(priority_of(a) for a in fallback)
        pool = [a for a in fallback if priority_of(a) == best_priority]

    idx = state.rr_index % len(pool)
    state.rr_index += 1
    return pool[idx]


def update_account_status(account_id: str, status: str, reason: str, extra: dict[str, Any] | None = None) -> None:
    ts = now_ts()
    for acc in state.config.get("accounts", []):
        if str(acc.get("id")) == str(account_id):
            acc["status"] = status
            acc["last_status_change_at"] = ts
            acc["last_status_reason"] = reason
            if extra:
                acc.update(extra)
            return


def status_tracking_enabled(account: dict[str, Any]) -> bool:
    return account.get("status_tracking_enabled") is not False


def note_success(account_id: str) -> None:
    ts = now_ts()
    for acc in state.config.get("accounts", []):
        if str(acc.get("id")) == str(account_id):
            if not status_tracking_enabled(acc):
                return
            acc["success_count"] = int(acc.get("success_count") or 0) + 1
            acc["last_success_at"] = ts
            acc["last_error"] = ""
            return


def is_limit_response(status_code: int | None, message: str) -> bool:
    """Return true only for quota/credit exhaustion responses.

    Pioneer may return HTTP 402 for included-credit exhaustion instead of 429.
    Treat those as temporary pool limits, but do not turn unrelated billing,
    auth, provider, or transport errors into sticky account state.
    """
    if status_code == 429:
        return True
    if status_code != 402:
        return False
    text = (message or "").lower()
    if any(
        token in text
        for token in (
            "direct_model_requires_credits",
            "can only be used with the pioneer router",
            "route it via 'pioneer/auto'",
            'route it via "pioneer/auto"',
            '"is_router": false',
            "'is_router': false",
        )
    ):
        return False
    return any(
        token in text
        for token in (
            "out_of_credits",
            "quota_exhausted",
            "daily credit limit",
            "included credits",
            "used your included credits",
            "billing-quota",
        )
    )


def note_limit(account_id: str, status_code: int | None, message: str) -> bool:
    """Record quota telemetry and request an in-flight account fallback.

    Account pool is a key provider/rotator, not an upstream health judge.  Do
    not mark accounts as error/disabled for provider errors, 5xx, transport
    errors, malformed SSE, etc. Quota/credit exhaustion may switch accounts for
    the current request, but must not persist status=limited automatically.
    """
    for acc in state.config.get("accounts", []):
        if str(acc.get("id")) == str(account_id):
            configured_quota_error = (
                status_code in {400, 402, 403}
                and any(token.lower() in (message or "").lower()
                        for token in acc.get("quota_error_keywords", []) if token)
            )
            if not is_limit_response(status_code, message) and not configured_quota_error:
                return False
            if not status_tracking_enabled(acc):
                return False
            acc["limited_count"] = int(acc.get("limited_count") or 0) + 1
            acc["last_limit_at"] = now_ts()
            acc["last_limit_message"] = (message or "")[:500]
            return True
    return False


def note_stream_transport_error(account_id: str, message: str) -> None:
    ts = now_ts()
    for acc in state.config.get("accounts", []):
        if str(acc.get("id")) == str(account_id):
            acc["stream_transport_error_count"] = int(acc.get("stream_transport_error_count") or 0) + 1
            acc["last_stream_transport_error_at"] = ts
            acc["last_stream_transport_error"] = (message or "")[:500]
            return


def is_non_degrading_stream_error(exc: Exception, first_chunk: bool, upstream_2xx: bool) -> bool:
    if not upstream_2xx:
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        first_chunk
        or "incomplete chunked read" in text
        or "peer closed connection without sending complete message body" in text
        or "remoteprotocolerror" in text
    )


async def forward_non_stream(path: str, body: dict[str, Any], request: Request) -> Response:
    model = str(body.get("model") or "")
    client_api_key = request_client_api_key(request)
    tried: set[str] = set()
    last_limit_response: Response | None = None

    while True:
        acc = await choose_account(model, exclude_ids=tried, client_api_key=client_api_key, path=path)
        if acc is None:
            if last_limit_response is not None:
                return last_limit_response
            return JSONResponse({"error": {"message": "No ok account available", "type": "account_pool_empty"}}, status_code=503)
        account_id = str(acc.get("id"))
        tried.add(account_id)
        target = build_target_url(acc, request, path)
        headers = build_upstream_headers(acc, request, path)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, proxy=account_proxy_url(acc) or None, trust_env=False) as client:
                resp = await client.post(target, json=body, headers=headers)
                content = resp.content
                media = resp.headers.get("content-type", "application/json")
                if 200 <= resp.status_code < 300:
                    note_success(account_id)
                    await persist_config()
                    out_headers = {"x-account-pool-id": account_id, "x-account-pool-status": str(acc.get("status") or "ok")}
                    return Response(content=content, status_code=resp.status_code, media_type=media, headers=out_headers)
                if note_limit(account_id, resp.status_code, content.decode(errors="replace")[:500]):
                    await persist_config()
                    out_headers = {"x-account-pool-id": account_id, "x-account-pool-status": str(acc.get("status") or "ok")}
                    last_limit_response = Response(content=content, status_code=resp.status_code, media_type=media, headers=out_headers)
                    continue
                out_headers = {"x-account-pool-id": account_id, "x-account-pool-status": str(acc.get("status") or "ok")}
                return Response(content=content, status_code=resp.status_code, media_type=media, headers=out_headers)
        except Exception as e:
            # Do not persist generic upstream/transport errors into account state.
            return JSONResponse({"error": {"message": str(e), "type": "account_pool_upstream_error", "account_id": account_id}}, status_code=502)


async def forward_stream(path: str, body: dict[str, Any], request: Request) -> StreamingResponse | JSONResponse:
    model = str(body.get("model") or "")
    client_api_key = request_client_api_key(request)

    async def gen():
        tried: set[str] = set()
        last_limit_data: bytes | None = None
        while True:
            acc = await choose_account(model, exclude_ids=tried, client_api_key=client_api_key, path=path)
            if acc is None:
                if last_limit_data is not None:
                    yield last_limit_data
                else:
                    yield (json.dumps({"error": {"message": "No ok account available", "type": "account_pool_empty"}}, ensure_ascii=False) + "\n").encode()
                return

            account_id = str(acc.get("id"))
            tried.add(account_id)
            target = build_target_url(acc, request, path)
            headers = build_upstream_headers(acc, request, path)
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(REQUEST_TIMEOUT, read=REQUEST_TIMEOUT),
                proxy=account_proxy_url(acc) or None,
                trust_env=False,
            )
            first_chunk = False
            upstream_2xx = False
            try:
                async with client.stream("POST", target, json=body, headers=headers) as resp:
                    upstream_2xx = 200 <= resp.status_code < 300
                    if not upstream_2xx:
                        data = await resp.aread()
                        if note_limit(account_id, resp.status_code, data.decode(errors="replace")[:500]):
                            await persist_config()
                            last_limit_data = data
                            continue
                        yield data
                        return
                    note_success(account_id)
                    await persist_config()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            first_chunk = True
                            yield chunk
                    return
            except Exception as e:
                if is_non_degrading_stream_error(e, first_chunk=first_chunk, upstream_2xx=upstream_2xx):
                    note_stream_transport_error(account_id, f"{type(e).__name__}: {e}")
                    await persist_config()
                    return
                # Do not persist generic upstream/transport errors into account state.
                if not first_chunk:
                    yield (json.dumps({"error": {"message": str(e), "type": "account_pool_upstream_error", "account_id": account_id}}, ensure_ascii=False) + "\n").encode()
                return
            finally:
                await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.on_event("startup")
async def startup() -> None:
    await reload_config_if_needed(force=True)
    await apply_daily_resets()
    await apply_limited_recovery_probes()


@app.get("/ready")
async def ready():
    # Local schema validation only: no reset, discovery, probe or inference.
    await reload_config_if_needed()
    try:
        registry = ChannelRegistry(state.config)
        return {"ok": True, "routing_enabled": registry.enabled, "selection_schema": registry.schema, "revision": registry.revision}
    except (ValueError, TypeError, KeyError):
        return _registry_error("Invalid channel configuration", 503)


@app.get("/health")
async def health() -> dict[str, Any]:
    await apply_daily_resets()
    await apply_limited_recovery_probes()
    accounts = state.config.get("accounts", [])
    return {"ok": True, "accounts": len(accounts), "ok_accounts": sum(1 for a in accounts if str(a.get("status") or "ok") == "ok")}


@app.get("/admin/status")
async def admin_status() -> dict[str, Any]:
    await apply_daily_resets()
    await apply_limited_recovery_probes()
    return {
        "config": str(CONFIG_PATH),
        "now": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "accounts": [account_public(a) for a in state.config.get("accounts", [])],
    }


@app.post("/admin/reload")
async def admin_reload() -> dict[str, Any]:
    await reload_config_if_needed(force=True)
    changed = await apply_daily_resets()
    recovered = await apply_limited_recovery_probes()
    return {"ok": True, "reset": changed, "recovered": recovered, "accounts": [account_public(a) for a in state.config.get("accounts", [])]}


@app.post("/admin/account/{account_id}/status/{status}")
async def set_status(account_id: str, status: str) -> dict[str, Any]:
    if status not in {"ok", "limited", "disabled"}:
        return {"ok": False, "error": "bad_status"}
    await reload_config_if_needed()
    update_account_status(account_id, status, "manual")
    await persist_config()
    return {"ok": True, "accounts": [account_public(a) for a in state.config.get("accounts", [])]}



@app.get("/internal/channels/resolve")
async def resolve_channel(request: Request, model: str = ""):
    # Reading metadata must not run daily-reset probes or rotate any account.
    await reload_config_if_needed()
    try:
        registry = ChannelRegistry(state.config)
    except (ValueError, TypeError, KeyError):
        return _registry_error("Invalid channel configuration", 503)
    if not registry.enabled or not registry.authorized(request_client_api_key(request)):
        return _registry_error("Unauthorized", 401)
    registry = await registry_for_model(request, model)
    resolution = registry.resolve(model)
    if resolution is None:
        return _registry_error("Unknown model in channel registry", 404)
    return resolution.descriptor()


@app.get("/v1/models")
async def channel_models(request: Request):
    await reload_config_if_needed()
    try:
        registry = ChannelRegistry(state.config)
    except (ValueError, TypeError, KeyError):
        return _registry_error("Invalid channel configuration", 503)
    if not registry.enabled:
        return await forward_raw("models", request)
    key = request_client_api_key(request)
    if registry.authorized(key):
        registry = await discovered_registry(request)
        models = registry.catalog
    else:
        allowed = {str(a.get("id")) for a in state.config.get("accounts", [])
                   if key and key in account_client_keys(a)}
        if not allowed:
            return _registry_error("Unauthorized", 401)
        registry = await discovered_registry(request, allowed_ids=allowed)
        models = [item for item in registry.catalog
                  if registry.resolve(item["id"]).account_ids.intersection(allowed)]
    return {"object": "list", "data": models, "revision": registry.revision}


@app.post("/internal/channels/dedupe-key")
async def channel_dedupe_key(request: Request):
    await reload_config_if_needed()
    try:
        registry = ChannelRegistry(state.config)
    except (ValueError, TypeError, KeyError):
        return _registry_error("Invalid channel configuration", 503)
    if not registry.enabled or not registry.authorized(request_client_api_key(request)):
        return _registry_error("Unauthorized", 401)
    if registry.schema != 1:
        return _registry_error("Request coalescing is owned by Stream", 404)
    try:
        body = await request.json()
        model = str(body.get('model') or '')
        registry = await registry_for_model(request, model)
        resolution = registry.resolve(model)
        if resolution is None:
            return _registry_error("Unknown model in channel registry", 404)
        if not isinstance(body.get('request'), dict) or not isinstance(body.get('headers'), dict):
            raise ValueError()
        return registry.dedupe_identity(resolution, body['request'], body['headers'])
    except (ValueError, TypeError, KeyError):
        return _registry_error("Invalid coalescing identity request")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def v1_passthrough(path: str, request: Request):
    return await forward_raw(path, request)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def transparent_passthrough(path: str, request: Request):
    return await forward_raw(path, request)
