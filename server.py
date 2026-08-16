from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

# Get PEEK_ROOT environment variable, default to "/workspace"
# Resolve to a file path which strictly must exist
ROOT = Path(os.getenv("PEEK_ROOT", "/workspace")).resolve(strict=True)

MAX_SEARCH_DEPTH = 5                # Max depth of directory search
MAX_FILES_LISTED=500                # Max number of files to be listed from a directory search
MAX_SEARCH_FILE_BYTES = 1_000_000   # Max size of searchable files
MAX_SEARCH_RESULTS = 50             # Max number of search results within a file
MAX_READ_LINES = 400                # Max number of lines to be read from a file at once
MAX_OUTPUT_LINE_CHARS = 500         # Max characters per output line

# File path patterns that will be denied
DENY_PATTERNS = (
    ".git",
    ".git/**",
    ".env",
    "*.env",
    "secrets",
    "secrets/**",
    "*.pem",
    "*.key",
)

# Describes tools to MCP clients
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,          # Tool is read-only
    destructiveHint=False,      # Tool is non-destructive
    idempotentHint=True,        # Repeatedly calling on same input has no side effects
    openWorldHint=False,        # Tool doesn't interact with external systems or public internet
)

# Create MCP server
mcp = MCPServer(
    "Peek",
    instructions=(
        "Read-only access to a mounted source tree. "
        "Prefer narrow reads and searches."
    ),
)


def _relative(path: Path) -> Path:
    """Return a path relative to the workspace root."""
    return path.relative_to(ROOT)


def _is_denied(path: Path) -> bool:
    """Return True if a relative path matches the deny list."""
    return any(path.match(pattern) for pattern in DENY_PATTERNS)


def _resolve(user_path: str) -> Path:
    """Resolve a project-relative path without allowing escape from ROOT."""
    # Check if path is empty or contains NUL-byte
    if not user_path or "\0" in user_path:
        raise ValueError("path must be non-empty")

    # Construct path relative to /workspace
    relative_path = Path(user_path)

    # Check if path is absolute or contains '..'
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("path must be relative and may not contain '..'")

    # Check if path is denied
    if _is_denied(relative_path):
        raise PermissionError(f"access denied: {relative_path}")

    # Attempt to resolve path against /workspace
    resolved_path = (ROOT / relative_path).resolve(strict=True)
    
    # Check if resolved path escapes the workspace
    try:
        resolved_relative = _relative(resolved_path)
    except ValueError:
        raise ValueError("path escapes the workspace") from None

    # Check if resolved path is denied to safeguard against escaping symlinks
    if _is_denied(resolved_relative):
        raise PermissionError(f"access denied: {resolved_relative}")

    return resolved_path


def _display(path: Path) -> str:
    """Return a project-relative path for tool output."""
    relative_path = _relative(path).as_posix()

    # Return "." if path is empty
    return relative_path or "."     


def _binary(path: Path) -> bool:
    """Reject probable binary files using a small NUL-byte check."""
    # Use NUL-byte appearing in first 4KiB as heuristic for binary files
    with path.open("rb") as file:
        return b"\0" in file.read(4096)

# Expose this function as a read-only tool to MCP server
@mcp.tool(annotations=READ_ONLY)
def list_files(
    path: str = ".",
    max_depth: int = 2,
    limit: int = 200,
) -> str:
    """List files and directories below a project-relative path.

    Results are bounded and directory symlinks are never followed.
    """
    # Safely resolve path
    base_dir = _resolve(path)

    # Assert resolved path is a directory
    if not base_dir.is_dir():
        raise ValueError(f"not a directory: {_display(base_dir)}")

    # Assert search depth is within bounds
    if not 0 <= max_depth <= MAX_SEARCH_DEPTH:
        raise ValueError(f"max_depth must be between 0 and {MAX_SEARCH_DEPTH}")

    # Assert file limit is within bounds
    if not 1 <= limit <= MAX_FILES_LISTED:
        raise ValueError(f"limit must be between 1 and {MAX_FILES_LISTED}")

    # Allocate results list
    results: list[str] = []

    # Recursively traverse directory, ignoring symlinks 
    for current, directories, files in base_dir.walk(follow_symlinks=False):
        depth = len(current.relative_to(base_dir).parts)

        # Prevent traversal into denied directories
        # We have to modify `directories` in place since `Path.walk()` holds a reference
        directories[:] = sorted(
            name
            for name in directories
            if not _is_denied(_relative(current / name))
        )
        
        # Prevent traversal past `max_depth`
        if depth >= max_depth:
            directories.clear()

        # Append current directory to results
        if current != base_dir:
            results.append(f"{_display(current)}/")

            if len(results) >= limit:
                break

        # Append files to results
        for name in sorted(files):
            child = current / name
            relative = _relative(child)

            if _is_denied(relative):
                continue

            suffix = "@" if child.is_symlink() else ""
            results.append(f"{relative.as_posix()}{suffix}")

            if len(results) >= limit:
                break

        if len(results) >= limit:
            break

    if not results:
        return "(no visible entries)"

    if len(results) >= limit:
        results.append(f"... result limit reached ({limit})")

    return "\n".join(results)


@mcp.tool(annotations=READ_ONLY)
def read_file(
    path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    """Read a project-relative text file with line numbers.

    At most 400 lines are returned per call.
    """
    target = _resolve(path)

    if not target.is_file():
        raise ValueError(f"not a file: {_display(target)}")

    if start_line < 1:
        raise ValueError("start_line must be at least 1")

    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ValueError(
            f"a read may contain at most {MAX_READ_LINES} lines"
        )

    if _binary(target):
        raise ValueError("binary files are not supported")

    output: list[str] = []

    with target.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number < start_line:
                continue

            if line_number > end_line:
                break

            output.append(
                f"{line_number:>6} | {line.rstrip()}"
            )

    if not output:
        return "(no lines in requested range)"

    return f"# {_display(target)}\n" + "\n".join(output)


@mcp.tool(annotations=READ_ONLY)
def search_text(
    query: str,
    path: str = ".",
    file_glob: str = "*",
    case_sensitive: bool = False,
    max_results: int = 20,
) -> str:
    """Search for a literal string in bounded-size text files.

    This is not a regular-expression search.
    """
    if not query:
        raise ValueError("query must be non-empty")

    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"max_results must be between 1 and {MAX_SEARCH_RESULTS}"
        )

    base = _resolve(path)

    if not base.is_dir():
        raise ValueError(f"not a directory: {_display(base)}")

    needle = query if case_sensitive else query.casefold()
    matches: list[str] = []

    for current, directories, files in base.walk(follow_symlinks=False):
        # Prevent traversal into denied directories.
        directories[:] = sorted(
            name
            for name in directories
            if not _is_denied(_relative(current / name))
        )

        for name in sorted(files):
            relative = _relative(current / name)

            if _is_denied(relative):
                continue

            if not relative.match(file_glob):
                continue

            try:
                # Re-resolve every candidate. This rejects file symlinks
                # pointing outside /workspace or into denied paths.
                target = _resolve(relative.as_posix())

                if not target.is_file():
                    continue

                if target.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue

                if _binary(target):
                    continue

                with target.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as file:
                    for line_number, line in enumerate(file, start=1):
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
                            [:MAX_OUTPUT_LINE_CHARS]
                        )

                        matches.append(
                            f"{relative.as_posix()}:"
                            f"{line_number}: {excerpt}"
                        )

                        if len(matches) >= max_results:
                            return (
                                "\n".join(matches)
                                + "\n... result limit reached"
                            )

            except (OSError, UnicodeError, ValueError, PermissionError):
                # Ignore files that disappear, are unreadable, or resolve
                # somewhere Peek is not permitted to access.
                continue

    return "\n".join(matches) if matches else "(no matches)"


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        stateless_http=True,
        json_response=True,
    )
