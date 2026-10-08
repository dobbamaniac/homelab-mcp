import re
from typing import Any


REDACTED = "<redacted>"

_SENSITIVE_SUFFIXES = (
    "password",
    "passwd",
    "passphrase",
    "secret",
    "token",
    "api_key",
    "authorization",
    "cookie",
    "private_key",
    "psk",
)

_SENSITIVE_KEYS = {
    "aws_secret_access_key",
    "cipassword",
    "secret_access_key",
    "sshkeys",
}

_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?
        [a-z0-9_.-]*
        (?:
            password
            | passwd
            | passphrase
            | secret
            | token
            | api[_-]?key
            | client[_-]?secret
            | private[_-]?key
            | psk
        )
        ["']?
        \s*[:=]\s*
    )
    (?P<value>
        "(?:\\.|[^"])*"
        |
        '(?:\\.|[^'])*'
        |
        [^\s,;]+
    )
    """
)

_AUTH_RE = re.compile(r"(?im)^(\s*authorization\s*:\s*(?:bearer|basic)\s+)\S+")

_COOKIE_RE = re.compile(r"(?im)^(\s*(?:cookie|set-cookie)\s*:\s*).+$")

_URL_PASSWORD_RE = re.compile(r"(?i)(://[^/\s:@]+:)[^@\s/]+@")

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)


def _sensitive_key(value: str) -> bool:
    value = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")

    return (
        value in _SENSITIVE_KEYS
        or any(
            value == suffix or value.endswith(f"_{suffix}")
            for suffix in _SENSITIVE_SUFFIXES
        )
    )


def _redacted_assignment(match: re.Match[str]) -> str:
    value = match.group("value")
    quote = (
        value[:1]
        if value[:1] in {'"', "'"} and value[-1:] == value[:1]
        else ""
    )

    return f"{match.group('prefix')}{quote}{REDACTED}{quote}"


def redact_text(value: str) -> str:
    value = _PRIVATE_KEY_RE.sub(REDACTED, value)
    value = _AUTH_RE.sub(rf"\1{REDACTED}", value)
    value = _COOKIE_RE.sub(rf"\1{REDACTED}", value)
    value = _URL_PASSWORD_RE.sub(rf"\1{REDACTED}@", value)

    return _ASSIGNMENT_RE.sub(_redacted_assignment, value)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return redact_mapping(value)

    if isinstance(value, list):
        return [_redact(item) for item in value]

    if isinstance(value, str):
        return redact_text(value)

    return value


def redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            REDACTED
            if _sensitive_key(key)
            else _redact(item)
        )
        for key, item in value.items()
    }

def redact_value(value: Any) -> Any:
    """Recursively redact known secret-bearing keys and text patterns."""
    return _redact(value)
