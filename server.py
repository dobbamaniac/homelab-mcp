from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


ROOT = Path(os.environ.get("PEEK_ROOT", "/workspace")).resolve(strict=True)

MAX_READ_LINES = 400
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_OUTPUT_LINE_CHARS = 500


# This deny list is an additional precaution, not a replacement for mounting
# only files that are safe to disclose.
DENY_PATTERNS = tuple(
    pattern.strip()
    for pattern in os.environ.get(
        "PEEK_DENY",
        ".git,.git/**,.env,.env.*,secrets,secrets/**,*.pem,*.key",
    ).split(",")
    if pattern.strip()
)


# These annotations tell MCP clients that every exposed operation is local
# and read-only. The actual read-only guarantees still come from the code,
# mount configuration, and container restrictions.
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    openWorldHint=False,
)


mcp = FastMCP(
    "Peek",
    instructions=(
        "Read-only access to a mounted source tree. "
        "Prefer narrow reads and searches. "
        "Never claim that Peek can modify files or execute commands."
    ),
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
    json_response=True,
)


def _relative(path: Path) -> PurePosixPath:
    """Return a path relative to the mounted project root."""
    return PurePosixPath(path.relative_to(ROOT).as_posix())


def _is_denied(relative_path: PurePosixPath) -> bool:
    """Return whether a project-relative path matches a denied pattern."""
    value = relative_path.as_posix()

    return any(
        fnmatch.fnmatchcase(value, pattern)
        or relative_path.match(pattern)
        for pattern in DENY_PATTERNS
    )


def _resolve(user_path: str, *, must_exist: bool = True) -> Path:
    """
    Resolve a user-supplied project-relative path.

    Absolute paths, parent traversal, symlink escapes, and denied paths are
    rejected.
    """
    if not user_path or "\x00" in user_path:
        raise ValueError("path must be a non-empty string")

    raw_path = Path(user_path)

    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError("path must be relative and may not contain '..'")

    candidate = (ROOT / raw_path).resolve(strict=must_exist)

    try:
        relative = _relative(candidate)
    except ValueError as exc:
        raise ValueError("path escapes the mounted project root") from exc

    if _is_denied(relative):
        raise PermissionError(
            f"access denied by PEEK_DENY: {relative}"
        )

    return candidate


def _display(path: Path) -> str:
    """Return a user-facing project-relative path."""
    relative = _relative(path).as_posix()
    return relative if relative != "." else "."


def _looks_binary(path: Path) -> bool:
    """Use a small NUL-byte check to reject probable binary files."""
    with path.open("rb") as handle:
        return b"\x00" in handle.read(4096)


@mcp.tool(
    title="List source files",
    description=(
        "List files and directories below a project-relative path. "
        "Results are bounded and never follow directory symlinks."
    ),
    annotations=READ_ONLY,
)
def list_files(
    path: str = ".",
    max_depth: int = 2,
    limit: int = 200,
) -> str:
    """List a bounded portion of the mounted project tree."""
    base = _resolve(path)

    if not base.is_dir():
        raise ValueError(f"not a directory: {_display(base)}")

    if not 0 <= max_depth <= 5:
        raise ValueError("max_depth must be between 0 and 5")

    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    results: list[str] = []

    for current_text, directories, files in os.walk(
        base,
        followlinks=False,
    ):
        current = Path(current_text)
        depth = len(current.relative_to(base).parts)

        allowed_directories: list[str] = []

        for name in sorted(directories):
            child = current / name
            relative = _relative(child)

            if not _is_denied(relative):
                allowed_directories.append(name)

        # Prevent os.walk from descending beyond the requested depth.
        directories[:] = (
            allowed_directories
            if depth < max_depth
            else []
        )

        if current != base:
            results.append(f"{_display(current)}/")

            if len(results) >= limit:
                break

        for name in sorted(files):
            child = current / name

            if _is_denied(_relative(child)):
                continue

            # A listed symlink is marked, but directory symlinks are never
            # traversed. Reads still pass through _resolve().
            suffix = "@" if child.is_symlink() else ""
            results.append(f"{_display(child)}{suffix}")

            if len(results) >= limit:
                break

        if len(results) >= limit:
            break

    if not results:
        return "(no visible entries)"

    if len(results) == limit:
        results.append(f"... result limit reached ({limit})")

    return "\n".join(results)


@mcp.tool(
    title="Read source file",
    description=(
        f"Read a project-relative text file with line numbers. "
        f"At most {MAX_READ_LINES} lines are returned per call."
    ),
    annotations=READ_ONLY,
)
def read_file(
    path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    """Read a bounded inclusive line range from a text file."""
    target = _resolve(path)

    if not target.is_file():
        raise ValueError(f"not a file: {_display(target)}")

    if start_line < 1:
        raise ValueError("start_line must be at least 1")

    if end_line < start_line:
        raise ValueError(
            "end_line must be greater than or equal to start_line"
        )

    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ValueError(
            f"a read may contain at most {MAX_READ_LINES} lines"
        )

    if _looks_binary(target):
        raise ValueError("binary files are not supported")

    output: list[str] = []

    with target.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number < start_line:
                continue

            if line_number > end_line:
                break

            output.append(
                f"{line_number:>6} | {line.rstrip()}"
            )

    if not output:
        return (
            f"(no lines in requested range; "
            f"file: {_display(target)})"
        )

    return (
        f"# {_display(target)}\n"
        + "\n".join(output)
    )


@mcp.tool(
    title="Search source text",
    description=(
        "Search for a literal string in bounded-size text files "
        "under a project-relative directory. "
        "This is not a regular-expression search."
    ),
    annotations=READ_ONLY,
)
def search_text(
    query: str,
    path: str = ".",
    file_glob: str = "*",
    case_sensitive: bool = False,
    max_results: int = 20,
) -> str:
    """Search project text files for a literal string."""
    if not query:
        raise ValueError("query must be non-empty")

    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"max_results must be between 1 "
            f"and {MAX_SEARCH_RESULTS}"
        )

    base = _resolve(path)

    if not base.is_dir():
        raise ValueError(f"not a directory: {_display(base)}")

    needle = (
        query
        if case_sensitive
        else query.casefold()
    )

    matches: list[str] = []

    for current_text, directories, files in os.walk(
        base,
        followlinks=False,
    ):
        current = Path(current_text)

        # Prune denied directories before os.walk descends.
        directories[:] = [
            name
            for name in sorted(directories)
            if not _is_denied(_relative(current / name))
        ]

        for name in sorted(files):
            target = current / name
            relative = _relative(target)

            if _is_denied(relative):
                continue

            if not relative.match(file_glob):
                continue

            try:
                resolved = target.resolve(strict=True)

                # Reject file symlinks that escape the project root.
                resolved_relative = _relative(resolved)

                if _is_denied(resolved_relative):
                    continue

                if not resolved.is_file():
                    continue

                if resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue

                if _looks_binary(resolved):
                    continue

                with resolved.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as handle:
                    for line_number, line in enumerate(
                        handle,
                        start=1,
                    ):
                        haystack = (
                            line
                            if case_sensitive
                            else line.casefold()
                        )

                        if needle not in haystack:
                            continue

                        excerpt = (
                            line.rstrip()
                            .replace("\t", "    ")
                        )

                        excerpt = excerpt[
                            :MAX_OUTPUT_LINE_CHARS
                        ]

                        matches.append(
                            f"{relative.as_posix()}:"
                            f"{line_number}: {excerpt}"
                        )

                        if len(matches) >= max_results:
                            return (
                                "\n".join(matches)
                                + "\n... result limit reached"
                            )

            except (
                OSError,
                UnicodeError,
                ValueError,
            ):
                # Files may disappear, be unreadable, or resolve outside
                # the root while walking a concurrently changing tree.
                continue

    return (
        "\n".join(matches)
        if matches
        else "(no matches)"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
