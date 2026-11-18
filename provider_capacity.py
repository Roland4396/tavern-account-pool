"""Translate the configured provider usage API; the caller owns scheduling and all state."""
import os
from datetime import datetime
from typing import Any
import httpx


def gproxy_five_hour_window(payload: Any) -> tuple[float, float | None] | None:
    """Return (used_percent, reset_timestamp) from normalized GPROXY usage."""
    if not isinstance(payload, dict):
        return None
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return None
    for window in windows:
        if not isinstance(window, dict) or str(window.get("name") or "") != "five_hour":
            continue
        try:
            used_percent = float(window["used_percent"])
        except (KeyError, TypeError, ValueError):
            return None
        reset_timestamp = None
        reset_value = str(window.get("resets_at") or "").strip()
        if reset_value:
            try:
                reset_timestamp = datetime.fromisoformat(reset_value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        return used_percent, reset_timestamp
    return None


def limited_recovery_probe(account: dict[str, Any]) -> dict[str, Any] | None:
    probe = account.get("limited_recovery_probe")
    if not isinstance(probe, dict) or probe.get("enabled") is not True:
        return None
    if str(probe.get("type") or "") != "gproxy_claudecode_usage":
        return None
    return probe


async def fetch_gproxy_claudecode_capacity(
    account: dict[str, Any],
    probe: dict[str, Any],
) -> tuple[bool, float | None, float | None]:
    """Return whether any enabled Claude credential has 5h capacity."""
    base_url = str(probe.get("base_url") or account.get("base_url") or "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    provider_id = int(probe.get("provider_id") or 0)
    user_env = str(probe.get("admin_user_env") or "GPROXY_ADMIN_USER")
    password_env = str(probe.get("admin_password_env") or "GPROXY_ADMIN_PASSWORD")
    username = os.environ.get(user_env, "admin").strip()
    password = os.environ.get(password_env, "")
    if not base_url or provider_id <= 0 or not username or not password:
        raise RuntimeError("GPROXY usage probe is missing its base URL, provider, or admin login")

    timeout = max(1.0, min(30.0, float(probe.get("timeout_seconds") or 10.0)))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        login = await client.post(
            f"{base_url}/admin/login",
            json={"username": username, "password": password},
        )
        if login.status_code != 200:
            raise RuntimeError(f"GPROXY usage probe login returned HTTP {login.status_code}")
        credentials = await client.get(f"{base_url}/admin/providers/{provider_id}/credentials")
        if credentials.status_code != 200:
            raise RuntimeError(f"GPROXY credential list returned HTTP {credentials.status_code}")
        rows = credentials.json()
        if not isinstance(rows, list):
            raise RuntimeError("GPROXY credential list returned an invalid payload")

        saw_window = False
        earliest_reset: float | None = None
        lowest_used: float | None = None
        for credential in rows:
            if not isinstance(credential, dict) or credential.get("enabled") is not True:
                continue
            credential_id = credential.get("id")
            if credential_id is None:
                continue
            usage = await client.get(f"{base_url}/admin/credentials/{credential_id}/usage")
            if usage.status_code != 200:
                continue
            window = gproxy_five_hour_window(usage.json())
            if window is None:
                continue
            saw_window = True
            used_percent, reset_timestamp = window
            lowest_used = used_percent if lowest_used is None else min(lowest_used, used_percent)
            if used_percent < 100.0:
                return True, used_percent, reset_timestamp
            if reset_timestamp is not None:
                earliest_reset = (
                    reset_timestamp
                    if earliest_reset is None
                    else min(earliest_reset, reset_timestamp)
                )

        if not saw_window:
            raise RuntimeError("GPROXY usage probe found no readable five-hour window")
        return False, lowest_used, earliest_reset
