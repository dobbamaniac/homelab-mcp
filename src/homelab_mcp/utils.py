import os
from typing import Any


def require_env(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(f"{name} is not configured")

    return value


def pick(
    item: dict[str, Any],
    *keys: str,
) -> dict[str, Any]:
    return {key: item[key] for key in keys if key in item}


def pick_all(
    items: list[dict[str, Any]],
    *keys: str
) -> list[dict[str, Any]]:
    return [pick(item, *keys) for item in items]
