"""Outbound protocol after selection. No Pool state, routing, HTTP client or retries."""
from __future__ import annotations
import json
import logging
import os
from typing import Any, TYPE_CHECKING
from urllib.parse import urlsplit
from model_rules import apply_fields, matches

if TYPE_CHECKING:
    from starlette.requests import Request

LOG = logging.getLogger("account-pool-proxy")


def account_api_key(account: dict[str, Any]) -> str:
    key = str(account.get("api_key") or "").strip()
    if key:
        return key
    env_name = str(account.get("api_key_env") or "").strip()
    if env_name:
        return os.environ.get(env_name, "").strip()
    return ""


def account_proxy_url(account: dict[str, Any]) -> str:
    """Optional per-account outbound proxy URL for upstream requests."""
    proxy = str(
        account.get("proxy_url")
        or account.get("proxy")
        or account.get("http_proxy")
        or ""
    ).strip()
    if proxy:
        return proxy
    env_name = str(account.get("proxy_env") or "").strip()
    if env_name:
        return os.environ.get(env_name, "").strip()
    return ""


def normalize_base_url(url: str) -> str:
    url = str(url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def account_protocol(account: dict[str, Any], path: str = "") -> str:
    """Infer auth protocol from request path.

    This proxy is intentionally transparent: it does not convert protocols, it
    only selects an account and injects the credential style required by the
    chosen API surface.
    """
    path_key = path.lstrip("/").lower()
    if path_key.startswith("v1/"):
        path_key = path_key[3:]
    if path_key.startswith("messages"):
        return "anthropic"
    if path_key.startswith("v1beta/") or "generatecontent" in path_key:
        return "gemini"
    return "openai"


HOP_BY_HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def build_upstream_headers(account: dict[str, Any], request: Request, path: str) -> dict[str, str]:
    protocol = account_protocol(account, path)
    api_key = account_api_key(account)
    headers: dict[str, str] = {}

    for name, value in request.headers.items():
        lname = name.lower()
        if lname in HOP_BY_HOP_HEADERS:
            continue
        if lname in {"authorization", "x-api-key", "x-goog-api-key", "x-acc-passthrough-authorization"} or lname.startswith("x-acc-"):
            continue
        headers[name] = value

    copied_header_names = {name.lower() for name in headers}
    if "content-type" not in copied_header_names:
        headers["Content-Type"] = request.headers.get("content-type") or "application/json"
    if "accept" not in copied_header_names:
        headers["Accept"] = request.headers.get("accept") or "application/json"

    auth_mode = str(account.get("auth_mode") or "configured")
    if auth_mode == "none":
        pass
    elif auth_mode == "caller":
        # ACC authenticates its caller separately. This opt-in is useful for
        # existing bring-your-own-key channels; never forward it to other ones.
        incoming = str(request.headers.get("x-acc-passthrough-authorization") or "")
        if incoming:
            headers["Authorization"] = incoming
    elif protocol == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = request.headers.get("anthropic-version") or "2023-06-01"
    elif protocol == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    user_agent = str(account.get("user_agent") or account.get("upstream_user_agent") or "").strip()
    if user_agent:
        for existing in [k for k in headers if k.lower() == "user-agent"]:
            headers.pop(existing, None)
        headers["User-Agent"] = user_agent

    extra_headers = account.get("upstream_headers")
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            header_name = str(key or "").strip()
            if not header_name:
                continue
            for existing in [k for k in headers if k.lower() == header_name.lower()]:
                headers.pop(existing, None)
            headers[header_name] = str(value)

    return headers

def build_target_url(account: dict[str, Any], request: Request, path: str) -> str:
    base = str(account['base_url']).rstrip('/')
    clean_path = path.lstrip('/')

    # Most Pioneer/OpenAI-compatible accounts are configured with base_url=/v1.
    # For explicit versioned paths sent to the transparent catch-all, avoid
    # producing /v1/v1beta or /v1/v1.
    if clean_path.startswith(('v1/', 'v1beta/')) and base.endswith('/v1'):
        target = f"{base[:-3]}/{clean_path}"
    else:
        target = f"{base}/{clean_path}"

    if request.url.query:
        target = f"{target}?{request.url.query}"
    return target


def parse_json_body_for_model(body_bytes: bytes) -> tuple[str, bool]:
    if not body_bytes:
        return "", False
    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return "", False
    if not isinstance(body, dict):
        return "", False
    return str(body.get("model") or ""), body.get("stream") is True


def prepare_upstream_body(account: dict[str, Any], path: str, body_bytes: bytes) -> bytes:
    """Apply account-specific chat compatibility to a fresh original body."""
    host = (urlsplit(str(account.get("base_url") or "")).hostname or "").lower()
    if path.strip("/").removeprefix("v1/") != "chat/completions":
        return body_bytes
    model, _ = parse_json_body_for_model(body_bytes)
    mapping = account.get("upstream_model_map") or {}
    drop_fields = account.get("drop_request_fields") or []
    if model and (model in mapping or drop_fields):
        body = json.loads(body_bytes)
        if model in mapping:
            body["model"] = mapping[model]
        for field_name in drop_fields:
            body.pop(field_name, None)
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        model = str(body["model"])
    if "request_rules" in account:
        # Target-specific last-mile rules are data, not provider-name branches.
        rules = account.get("request_rules") or []
        for rule in rules:
            if matches(model, rule.get("match", {})):
                body = json.loads(body_bytes)
                apply_fields(body, rule)
                body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        return body_bytes
    if host != "himodels.ai" and not host.endswith(".himodels.ai"):
        return body_bytes
    if model.strip().lower() != "kimi-k3":
        return body_bytes

    body = json.loads(body_bytes)
    template = body.get("chat_template_kwargs")
    template = dict(template) if isinstance(template, dict) else {}
    template.pop("enable_thinking", None)
    template["thinking"] = False
    body["chat_template_kwargs"] = template
    body.pop("thinking", None)
    body["reasoning_effort"] = "low"
    body.pop("effort", None)
    if isinstance(body.get("output_config"), dict):
        body["output_config"].pop("effort", None)
        if not body["output_config"]:
            body.pop("output_config")
    body["temperature"] = 0.6
    body["top_p"] = 0.95
    LOG.info("himodels_kimi_request_settings account=%s model=%s thinking=false effort=low", account.get("id"), model)
    return json.dumps(body, ensure_ascii=False).encode("utf-8")
