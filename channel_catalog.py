"""Lazy, authenticated model discovery; never selects an inference account.

Only explicitly declared catalog sources are contacted, only with GET, and only
from a catalog/route request. Readiness never calls this module. The cache is
separate from account health/RR state and caller credentials are never shared.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from dataclasses import dataclass

from channel_config import ChannelRegistry, RegistryError, matches, validate_selector


def validate_discovery(value, accounts):
    if not isinstance(value, dict) or set(value) - {
            'account', 'path', 'ttl_seconds', 'timeout_seconds', 'include', 'exclude',
            'public_prefix', 'strip_prefix', 'exclude_models'}:
        raise RegistryError('invalid catalog discovery')
    if value.get('account') not in accounts:
        raise RegistryError('unknown catalog account')
    # Discovery cannot become a generic URL fetch or an inference request.
    if value.get('path', 'models') not in {'models', 'v1/models'}:
        raise RegistryError('catalog path must be models or v1/models')
    for key, default, low, high in [('ttl_seconds', 300, 1, 86400), ('timeout_seconds', 4, 0.1, 30)]:
        number = value.get(key, default)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not low <= number <= high:
            raise RegistryError('invalid catalog timing')
    for key in ('include', 'exclude'):
        if key in value:
            validate_selector(value[key])
    for key in ('public_prefix', 'strip_prefix'):
        if key in value and not isinstance(value[key], str):
            raise RegistryError('invalid catalog model prefix')
    if 'exclude_models' in value and (not isinstance(value['exclude_models'], list)
            or not all(isinstance(x, str) for x in value['exclude_models'])):
        raise RegistryError('invalid catalog exclusions')


def parse_models(payload):
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError('catalog data must be a list')
    result = []
    for item in data:
        if isinstance(item, dict):
            value = item.get('id') or item.get('name') or item.get('model')
        else:
            value = item
        if isinstance(value, str) and value.strip() and len(value) <= 512:
            result.append(value.strip())
    return list(dict.fromkeys(result))


def projected_models(names, spec):
    output = {}
    for name in names:
        if name in spec.get('exclude_models', []):
            continue
        if spec.get('include') and not matches(name, spec['include']):
            continue
        if spec.get('exclude') and matches(name, spec['exclude']):
            continue
        strip = spec.get('strip_prefix', '')
        wire = name[len(strip):] if strip and name.startswith(strip) else name
        output[spec.get('public_prefix', '') + name] = wire
    return output


@dataclass
class Entry:
    checked_at: float
    names: list[str] | None


class CatalogCache:
    def __init__(self, *, clock=time.monotonic, limit=256):
        self.clock = clock
        self.limit = limit
        self.entries = {}
        self.lock = asyncio.Lock()

    async def compile(self, config, *, caller_authorization='', fetch, log=lambda _: None,
                      allowed_ids=None):
        # Pure validation/authentication happens before any network operation.
        baseline = ChannelRegistry(config)
        if not baseline.enabled:
            return baseline
        candidate = copy.deepcopy(config)
        pairs = [(c, r) for c in candidate['routing'].get('channels', [])
                 if c.get('enabled', True) is not False
                 for r in c.get('routes', []) if r.get('enabled', True) is not False]
        pinned = ({m for _, r in pairs if not r.get('discovery') for m in r.get('models', {})}
                  | set(baseline.account_exact))
        discovered_owners = {}
        # Lock coalesces simultaneous refreshes without any detached task. The
        # invoking HTTP request owns and cancels a fetch if it disconnects.
        async with self.lock:
            for channel, route in pairs:
                spec = route.get('discovery')
                if not spec:
                    continue
                account = baseline.accounts[spec['account']]
                if (str(account.get('status') or 'ok') == 'disabled'
                        or (allowed_ids is not None and spec['account'] not in allowed_ids)):
                    route['models'] = {}
                    continue
                needs_caller = account.get('auth_mode') == 'caller'
                auth = caller_authorization if needs_caller else ''
                if needs_caller and not auth:
                    route['models'] = {}
                    continue
                identity = {'account': {k: account.get(k) for k in (
                    'id', 'base_url', 'api_key', 'api_key_env', 'auth_mode', 'protocol',
                    'proxy_url', 'proxy', 'http_proxy', 'proxy_env', 'upstream_headers',
                    'user_agent', 'upstream_user_agent')},
                    'path': spec.get('path', 'models'), 'caller': auth}
                key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                now = self.clock()
                cached = self.entries.get(key)
                if cached is None or now - cached.checked_at >= spec.get('ttl_seconds', 300):
                    try:
                        payload = await fetch(account, spec, auth)
                        names = parse_models(payload)
                        cached = Entry(now, names)
                        log(f'catalog_refresh route={route["id"]} models={len(names)} result=ok')
                    except Exception as exc:
                        # Keep the last known good catalog on transient failure;
                        # only the redacted exception type enters logs.
                        cached = Entry(now, cached.names if cached else None)
                        log(f'catalog_refresh route={route["id"]} result=stale type={type(exc).__name__}')
                    self.entries[key] = cached
                    while len(self.entries) > self.limit:
                        self.entries.pop(next(iter(self.entries)))
                if cached.names is not None:
                    route['models'] = projected_models(cached.names, spec)
                # Static declarations take precedence. Dynamic sources retain
                # the old ordered last-source-wins behavior for collisions.
                for name in list(route.get('models', {})):
                    if name in pinned:
                        route['models'].pop(name)
                        continue
                    old = discovered_owners.get(name)
                    if old is not None:
                        old['models'].pop(name, None)
                    discovered_owners[name] = route
        return ChannelRegistry(candidate)
