from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from homelab_mcp.redact import redact_text
from homelab_mcp.utils import require_env


class RequestError(RuntimeError):
    pass


_hosts: dict[str, str] = {}

_TIMEOUT = 15.0
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024

_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _configured_hosts() -> dict[str, str]:
    hosts: dict[str, str] = {}

    for entry in require_env("SSH_HOSTS").split(","):
        fqdn = entry.strip().lower().removesuffix(".")
        labels = fqdn.split(".")

        if (
            len(fqdn) > 253
            or len(labels) < 2
            or not all(_DNS_LABEL.fullmatch(label) for label in labels)
        ):
            raise RuntimeError(f"Invalid SSH hostname {entry!r}")

        name = labels[0]

        if name in hosts:
            raise RuntimeError(f"Duplicate SSH hostname {name!r}")

        hosts[name] = fqdn

    return hosts


def _host(name: str) -> str:
    target = _hosts.get(name)

    if target is None:
        raise RuntimeError(f"SSH host {name!r} is not configured")

    return target


def host_names() -> tuple[str, ...]:
    return tuple(sorted(_hosts))


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Validate shared SSH configuration for the server lifetime."""
    if _hosts:
        raise RuntimeError("SSH configuration is already initialized")

    if shutil.which("ssh") is None:
        raise RuntimeError("OpenSSH client is not installed")

    key = Path(require_env("SSH_KEY"))
    known_hosts = Path(require_env("SSH_KNOWN_HOSTS"))

    require_env("SSH_USER")

    if not key.is_file():
        raise RuntimeError(f"SSH key does not exist: {key}")

    if not known_hosts.is_file():
        raise RuntimeError(f"SSH known_hosts file does not exist: {known_hosts}")

    _hosts.update(_configured_hosts())

    try:
        yield

    finally:
        _hosts.clear()


async def run(
    host: str,
    *command: str,
    check: bool = True,
) -> str:
    if not command:
        raise RuntimeError("SSH command cannot be empty")

    target = _host(host)
    user = require_env("SSH_USER")
    key = require_env("SSH_KEY")
    known_hosts = require_env("SSH_KNOWN_HOSTS")

    remote_command = shlex.join((
        "env",
        "LC_ALL=C",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        *command,
    ))

    process = await asyncio.create_subprocess_exec(
        "ssh",
        "-F", "/dev/null",
        "-T",
        "-i", key,
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "UpdateHostKeys=no",
        "-o", f"HostKeyAlias={host}",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ConnectionAttempts=1",
        "-o", "LogLevel=ERROR",
        f"{user}@{target}",
        remote_command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=_TIMEOUT,
        )

    except TimeoutError as exc:
        process.kill()
        await process.communicate()

        raise RequestError(
            f"SSH host {host!r} timed out running {command[0]!r}"
        ) from exc

    if check and process.returncode:
        message = redact_text(
            stderr.decode(errors="replace").strip()
        )

        if len(message) > 500:
            message = message[:500] + "..."

        raise RequestError(
            f"SSH host {host!r} returned exit status {process.returncode} for {command[0]!r}"
            + (f": {message}" if message else "")
        )

    if len(stdout) > _MAX_OUTPUT_BYTES:
        raise RequestError(f"SSH host {host!r} returned too much output for {command[0]!r}")

    try:
        return stdout.decode()

    except UnicodeDecodeError as exc:
        raise RequestError(
            f"SSH host {host!r} returned invalid text "
            f"for {command[0]!r}"
        ) from exc


async def run_json(
    host: str,
    *command: str,
) -> Any:
    output = await run(host, *command)

    try:
        return json.loads(output)

    except json.JSONDecodeError as exc:
        raise RequestError(
            f"SSH host {host!r} returned invalid JSON for {command[0]!r}"
        ) from exc
