from typing import Any

import httpx


class RequestError(RuntimeError):
    pass


async def get(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, str] | None = None,
    source: str,
) -> httpx.Response:
    try:
        response = await client.get(path, params=params)
        response.raise_for_status()
        return response
    
    except httpx.HTTPError as exc:
        raise RequestError(f"{source} failed GET {path}: {exc}") from exc


def json_data(
    response: httpx.Response,
    *,
    source: str,
    path: str,
    key: str | None = None,
) -> Any:
    try:
        data = response.json()

    except ValueError as exc:
        raise RequestError(f"{source} returned invalid JSON for GET {path}") from exc

    if key is not None:
        if not isinstance(data, dict) or key not in data:
            raise RequestError(f"{source} returned an unexpected response for GET {path}")
        
        return data[key]
    
    return data


async def get_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, str] | None = None,
    source: str,
    key: str | None = None,
) -> Any:
    response = await get(
        client,
        path,
        params=params,
        source=source,
    )

    return json_data(
        response,
        source=source,
        path=path,
        key=key,
    )
