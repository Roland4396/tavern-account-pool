"""Small, state-free model selector and field-edit language shared at the wire boundary."""
import copy
import re


def apply_fields(payload: dict, rule: dict) -> None:
    """Generic dotted-field changes applied after selecting the actual target."""
    for path in rule.get("drop", []):
        parts = path.split('.')
        value = payload
        parents = []
        for part in parts[:-1]:
            if not isinstance(value.get(part), dict):
                value = None
                break
            parents.append((value, part))
            value = value[part]
        if value is not None:
            value.pop(parts[-1], None)
            for parent, key in reversed(parents):
                if not parent[key]:
                    parent.pop(key)
    for path, new_value in rule.get("set", {}).items():
        parts = path.split('.')
        value = payload
        for part in parts[:-1]:
            if not isinstance(value.get(part), dict):
                value[part] = {}
            value = value[part]
        value[parts[-1]] = copy.deepcopy(new_value)


def matches(model: str, selector: dict) -> bool:
    if not selector:
        return False
    # Every specified condition must match. An explicit 'any' is OR.
    if "any" in selector and not any(matches(model, s) for s in selector["any"]):
        return False
    if "exact" in selector and model not in selector["exact"]:
        return False
    if "prefix" in selector and not model.startswith(selector["prefix"]):
        return False
    if "regex" in selector and re.search(selector["regex"], model, re.IGNORECASE) is None:
        return False
    return True
