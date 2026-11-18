"""Pool-owned channel selection: prefixes, aliases, protocols and account groups.

Schema 2 contains selection data only. Schema 1 is a temporary rolling-release
compatibility reader for old Stream instances; it must not be used for new config.

Resolution never chooses an account, rotates a key, probes a provider or wakes a
worker. Account selection happens once per attempt in app.py. Public names and
wire names are deliberately separate; only ACC rewrites the latter.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
import os
from dataclasses import dataclass
from typing import Any

from model_rules import apply_fields, matches


TRANSPORTS = frozenset({"chat", "messages", "responses", "gemini", "local"})
POLICY_KEYS = frozenset({
    "streaming", "early_stop", "injections", "filters", "request",
    "runtime", "session", "latest_wins", "adapter", "terminal_tool",
})


class RegistryError(ValueError):
    pass


def account_transport(account: dict, routes: list[tuple[dict, dict]]) -> str:
    """Use existing Pool protocol/path metadata, never model-name heuristics."""
    allowed = account.get('allowed_paths') or []
    if isinstance(allowed, str):
        allowed = [allowed]
    paths = [str(p).strip('/').removeprefix('v1/') for p in allowed]
    path_types = list(dict.fromkeys({'chat/completions': 'chat', 'messages': 'messages',
                                    'responses': 'responses'}[p]
                                   for p in paths if p in {'chat/completions', 'messages', 'responses'}))
    if len(path_types) == 1:
        return path_types[0]
    route_types = list(dict.fromkeys(r['transport'] for _, r in routes if r['transport'] != 'local'))
    if len(route_types) == 1:
        return route_types[0]
    declared = {'openai': 'chat', 'chat': 'chat', 'anthropic': 'messages',
                'messages': 'messages', 'responses': 'responses', 'gemini': 'gemini'}.get(
                    str(account.get('protocol') or '').lower())
    if declared:
        return declared
    choices = path_types or route_types
    return 'chat' if not choices or 'chat' in choices else choices[0]


def merge(base: dict, override: dict) -> dict:
    """Objects merge; scalars and arrays replace (including false/empty)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def matches_explicit(model: str, selector: dict) -> bool:
    """An exact-name selector is intentional; a matching prefix/regex is not."""
    return matches(model, selector) and (
        'exact' in selector
        or any(matches_explicit(model, child) for child in selector.get('any', []))
    )


def validate_selector(selector: Any) -> None:
    if not isinstance(selector, dict) or set(selector) - {"any", "exact", "prefix", "regex"}:
        raise RegistryError("invalid model selector")
    if "regex" in selector:
        re.compile(selector["regex"], re.IGNORECASE)
    if "prefix" in selector and not isinstance(selector["prefix"], str):
        raise RegistryError("model prefix must be a string")
    if "exact" in selector and (not isinstance(selector["exact"], list)
                                or not all(isinstance(x, str) and x for x in selector["exact"])):
        raise RegistryError("exact models must be a list of names")
    if "any" in selector:
        if not isinstance(selector["any"], list) or not selector["any"]:
            raise RegistryError("any selector must be a nonempty list")
        for child in selector["any"]:
            validate_selector(child)


def validate_policy(policy: Any) -> None:
    if not isinstance(policy, dict) or set(policy) - POLICY_KEYS:
        raise RegistryError("unknown channel policy field")
    stop = policy.get("early_stop", {})
    if not isinstance(stop, dict) or set(stop) - {"enabled", "action", "tags", "case_sensitive", "content_guard"}:
        raise RegistryError("invalid early_stop policy")
    if "action" in stop and stop["action"] != "disconnect":
        raise RegistryError("early_stop action must be disconnect")
    for key in ("enabled", "case_sensitive", "content_guard"):
        if key in stop and not isinstance(stop[key], bool):
            raise RegistryError("early_stop switches must be booleans")
    if "tags" in stop and (not isinstance(stop["tags"], list) or not all(
            isinstance(tag, str) and 0 < len(tag) <= 256 for tag in stop["tags"])):
        raise RegistryError("invalid early_stop tags")
    if len(json.dumps(stop)) > 4096:
        raise RegistryError("early_stop policy is too large")
    streaming = policy.get("streaming", {})
    if not isinstance(streaming, dict) or set(streaming) - {
            "nonstream_to_stream", "force_upstream", "json_keepalive", "deadline_seconds", "exact_dedupe"}:
        raise RegistryError("invalid streaming policy")
    if streaming.get("force_upstream", "auto") not in {"auto", "stream", "json"}:
        raise RegistryError("invalid upstream stream mode")
    for key in ("nonstream_to_stream", "json_keepalive", "exact_dedupe"):
        if key in streaming and not isinstance(streaming[key], bool):
            raise RegistryError("streaming switches must be booleans")
    if "deadline_seconds" in streaming and (
            isinstance(streaming["deadline_seconds"], bool)
            or not isinstance(streaming["deadline_seconds"], (int, float))
            or not 0 <= streaming["deadline_seconds"] <= 3600):
        raise RegistryError("invalid stream deadline")
    injections = policy.get("injections", {})
    if not isinstance(injections, dict) or not all(isinstance(x, str) and isinstance(v, bool)
                                                 for x, v in injections.items()):
        raise RegistryError("injections must be named boolean switches")


@dataclass(frozen=True)
class Resolution:
    public_model: str
    upstream_model: str
    channel: str
    route: str
    transport: str
    account_ids: frozenset[str]
    policy: dict[str, Any]
    revision: str
    dedupe: dict[str, Any]

    def descriptor(self) -> dict[str, Any]:
        # Explicit projection: never leak credentials, targets or wire aliases.
        return {"model": self.public_model, "channel": self.channel,
                "route": self.route, "transport": self.transport,
                "upstream_model": self.upstream_model,
                **({"policy": copy.deepcopy(self.policy)} if self.policy else {}), "revision": self.revision,
                "dedupe_identity": bool(self.dedupe)}


class ChannelRegistry:
    def __init__(self, config: dict[str, Any]):
        routing = config.get("routing") or {}
        self.schema = routing.get("schema", 1)
        self.enabled = routing.get("enabled") is True
        self.client_keys = tuple(str(x) for x in routing.get("client_keys", []) if x)
        self.revision = hashlib.sha256(json.dumps(
            routing, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()[:20]
        self.defaults = routing.get("defaults", {}) if self.schema == 1 else {}
        self.profiles = routing.get("profiles", {}) if self.schema == 1 else {}
        self.dedupe_credentials = routing.get("dedupe_credentials", {}) if self.schema == 1 else {}
        self.rules = routing.get("model_rules", []) if self.schema == 1 else []
        self.routes: list[tuple[dict, dict]] = []
        self.exact: dict[str, tuple[dict, dict]] = {}
        self.account_exact: dict[str, tuple[dict, dict]] = {}
        self.catalog: list[dict] = []
        self.accounts = {str(a["id"]): a for a in config.get("accounts", []) if isinstance(a, dict) and a.get("id")}
        if not self.enabled:
            return
        if self.schema not in {1, 2} or not self.client_keys:
            raise RegistryError("channel schema=1/2 and client_keys are required")
        if self.schema == 2:
            if set(routing) - {'schema','enabled','client_keys','channels'}:
                raise RegistryError('Pool schema 2 accepts selection fields only')
            if any('policy' in a for a in self.accounts.values()):
                raise RegistryError('Account feature policies belong to Stream')
        validate_policy(self.defaults)
        for profile in self.profiles.values():
            validate_policy(profile)
        if self.schema == 1:
            for account in self.accounts.values():
                validate_policy(account.get("policy", {}))
        for rule in self.rules:
            validate_selector(rule.get("match"))
            validate_policy(rule.get("policy", {}))
        channel_ids: set[str] = set()
        route_ids: set[str] = set()
        for channel in routing.get("channels", []):
            if self.schema == 2 and set(channel) - {'id','enabled','accounts','routes'}:
                raise RegistryError('Pool channel accepts selection fields only')
            cid = channel.get("id")
            if not isinstance(cid, str) or not cid or cid in channel_ids:
                raise RegistryError("duplicate or empty channel id")
            channel_ids.add(cid)
            validate_policy(channel.get("policy", {}))
            for route in channel.get("routes", []):
                if self.schema == 2 and set(route) - {'id','enabled','accounts','transport','match','models','strip_prefix','visible','discovery'}:
                    raise RegistryError('Pool route accepts selection fields only')
                rid = route.get("id")
                if not isinstance(rid, str) or not rid or rid in route_ids:
                    raise RegistryError("duplicate or empty route id")
                route_ids.add(rid)
                if route.get("transport") not in TRANSPORTS:
                    raise RegistryError("unsupported channel transport")
                validate_policy(route.get("policy", {}))
                for profile in [*channel.get("profiles", []), *route.get("profiles", [])]:
                    if profile not in self.profiles:
                        raise RegistryError("unknown policy profile")
                if "match" in route:
                    validate_selector(route["match"])
                ids = route.get("accounts", channel.get("accounts", []))
                if not isinstance(ids, list) or set(ids) - self.accounts.keys():
                    raise RegistryError("unknown channel account")
                if route["transport"] != "local" and not ids:
                    raise RegistryError("channel requires an account")
                if 'discovery' in route:
                    from channel_catalog import validate_discovery
                    validate_discovery(route['discovery'], self.accounts)
                    if route['discovery']['account'] not in ids:
                        raise RegistryError('catalog source must belong to the route')
                models = route.get("models", {})
                if not isinstance(models, dict) or not all(
                        isinstance(k, str) and k and isinstance(v, str) and v for k, v in models.items()):
                    raise RegistryError("models must map public names to upstream names")
                pair = (channel, route)
                if channel.get("enabled", True) is False or route.get("enabled", True) is False:
                    continue
                self.routes.append(pair)
                for name in models:
                    if name in self.exact:
                        raise RegistryError("duplicate public model name")
                    self.exact[name] = pair
                    visible = route.get("visible", list(models))
                    if name in visible:
                        self.catalog.append({"id": name, "object": "model", "owned_by": cid})

        # Explicit aliases/exact selectors remain intentional overrides. A
        # generic selector must not steal a literal model from another account.
        # Matching selectors within that account's own scope keep their route,
        # protocol and wire-name transforms (e.g. an explicit public prefix).
        declared_pairs = [(c, r) for c in routing.get('channels', []) for r in c.get('routes', [])]
        declared_names = {name for _, r in declared_pairs for name in r.get('models', {})}
        for account_id, account in (self.accounts.items() if self.schema == 2 else []):
            if str(account.get('status') or 'ok') == 'disabled':
                continue
            belongs = lambda pair: account_id in pair[1].get('accounts', pair[0].get('accounts', []))
            pairs = [pair for pair in self.routes if belongs(pair)]
            if any(belongs(pair) for pair in declared_pairs) and not pairs:
                continue  # Preserve explicitly disabled channel/route scopes.
            transport = account_transport(account, pairs)
            compatible = [(c, r) for c, r in pairs if r['transport'] == transport]
            channel = compatible[0][0] if compatible else {'id': str(account.get('channel') or 'account-pool')}
            route = {'id': f"account-models:{channel['id']}:{transport}", 'transport': transport,
                     'accounts': [account_id], 'models': {}}
            names = account.get('models', [])
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, list):
                continue
            for name in names:
                if not isinstance(name, str) or not name.strip() or name.strip() == '*':
                    continue
                name = name.strip()
                if name in declared_names or any(
                        matches_explicit(name, r.get('match', {})) for _, r in declared_pairs):
                    continue
                own_match = next((pair for pair in pairs if matches(name, pair[1].get('match', {}))), None)
                owner_channel, owner_route = own_match if own_match is not None else (channel, route)
                existing = self.account_exact.get(name)
                if existing is None:
                    selected_route = copy.deepcopy(owner_route)
                    selected_route['accounts'] = list(owner_route.get('accounts', owner_channel.get('accounts', [])))
                    self.account_exact[name] = (owner_channel, selected_route)
                    self.catalog.append({'id': name, 'object': 'model', 'owned_by': owner_channel['id']})
                elif (existing[0]['id'] == owner_channel['id']
                      and existing[1]['id'] == owner_route['id']
                      and account_id not in existing[1]['accounts']):
                    existing[1]['accounts'].append(account_id)
        if self.account_exact:
            # Account-list edits change the effective routing revision too.
            self.revision = hashlib.sha256(json.dumps({
                'routing': routing,
                'account_models': {name: [c['id'], r['id'], r['accounts']]
                                   for name, (c, r) in self.account_exact.items()},
            }, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]

    def authorized(self, key: str) -> bool:
        return bool(key) and any(hmac.compare_digest(key, candidate) for candidate in self.client_keys)

    def resolve(self, model: str) -> Resolution | None:
        pair = self.exact.get(model)
        if pair is None and self.schema == 2:
            pair = next((pair for pair in self.routes if matches_explicit(model, pair[1].get('match', {}))), None)
        if pair is None:
            pair = self.account_exact.get(model)
        if pair is None:
            pair = next((pair for pair in self.routes if matches(model, pair[1].get("match", {}))), None)
        if pair is None:
            return None
        channel, route = pair
        policy = copy.deepcopy(self.defaults)
        for profile in channel.get("profiles", []):
            policy = merge(policy, self.profiles[profile])
        policy = merge(policy, channel.get("policy", {}))
        for profile in route.get("profiles", []):
            policy = merge(policy, self.profiles[profile])
        policy = merge(policy, route.get("policy", {}))
        upstream = route.get("models", {}).get(model)
        if upstream is None:
            prefix = route.get("strip_prefix", "")
            upstream = model[len(prefix):] if prefix and model.startswith(prefix) else model
        for rule in self.rules:
            if rule.get("transports") and route["transport"] not in rule["transports"]:
                continue
            if rule.get("channels") and channel["id"] not in rule["channels"]:
                continue
            if matches(upstream, rule["match"]):
                policy = merge(policy, rule.get("policy", {}))
        if policy.get("session", {}).get("model") == "$upstream":
            policy["session"]["model"] = upstream
        dedupe = copy.deepcopy(route.get("dedupe", {}))
        if dedupe.get("model") == "$upstream":
            dedupe["model"] = upstream
        elif dedupe.get("model") == "$public":
            dedupe["model"] = model
        if dedupe.get('gemini_payload') and 'gemini' in upstream.lower():
            dedupe.update(kind='payload', shape='chat')
        return Resolution(model, upstream, channel["id"], route["id"], route["transport"],
                          frozenset(route.get("accounts", channel.get("accounts", []))), policy, self.revision, dedupe)

    def selected_policy(self, resolution: Resolution, account: dict) -> dict:
        return merge(resolution.policy, account.get("policy", {})) if self.schema == 1 else {}

    def dedupe_identity(self, resolution: Resolution, payload: dict, headers: dict) -> dict:
        """Preserve the existing shared lock/cache key across the rollout.

        Secrets used by the previous key algorithm remain here in ACC. Stream
        receives only the digest and the cache envelope type. No account is
        selected and no upstream operation is performed by this method.
        """
        cfg = resolution.dedupe
        if not cfg:
            identity = {"scope": "acc-channel-json-v1", "route": resolution.route,
                        "upstream_model": resolution.upstream_model,
                        "accounts": sorted(resolution.account_ids),
                        "policy": resolution.policy, "request": payload}
            shape = 'chat'
        else:
            body = copy.deepcopy(payload)
            body['model'] = cfg.get('model', resolution.public_model)
            shape = cfg['shape']
            if cfg.get('kind') == 'payload':
                identity = body
            else:
                original_headers = {key: value for key, value in headers.items()
                                    if not key.lower().startswith('x-acc-')}
                auth_name = cfg.get('auth_header', 'Authorization')
                for key in list(original_headers):
                    if key.lower() in {'authorization', 'x-api-key'}:
                        original_headers.pop(key)
                credential = self.dedupe_credentials.get(cfg.get('credential'), '')
                if isinstance(credential, dict):
                    credential = os.environ.get(credential.get('env', ''), '')
                if cfg.get('caller_authorization'):
                    credential = next((v for k, v in headers.items()
                                       if k.lower() == 'x-acc-passthrough-authorization'), '')
                    prefix = ''
                else:
                    prefix = cfg.get('auth_prefix', 'Bearer ')
                if credential:
                    original_headers[auth_name] = prefix + credential
                for key, value in cfg.get('headers', {}).items():
                    original_headers[key] = value
                identity = {"scope": cfg['scope'], "url": cfg['url'],
                            "headers": original_headers, "request": body}
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                          separators=(',', ':')).encode('utf-8', errors='replace')).hexdigest()
        return {'key': digest, 'shape': shape, 'revision': resolution.revision}
