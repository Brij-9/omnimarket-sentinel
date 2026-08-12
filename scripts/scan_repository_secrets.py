"""Fail-closed repository secret-pattern scan without disclosing matched content."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_EXCLUDED_TRACKED_PATHS = frozenset(
    {
        "src/market_sentinel/security.py",
        "tests/test_security.py",
        "tests/operations/test_control_center_hardening.py",
    }
)
_SECRET_PATTERNS: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(
        rb"-----BEGIN" + rb"[^\r\n]{0,32}" + rb"PRIVATE" + rb" KEY-----"
    ),
)


def _tracked_paths() -> tuple[Path, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    )


def _match_count(paths: tuple[Path, ...]) -> int:
    matches = 0
    for path in paths:
        if path.as_posix() in _EXCLUDED_TRACKED_PATHS:
            continue
        data = path.read_bytes()
        matches += sum(
            1
            for pattern in _SECRET_PATTERNS
            if pattern.search(data) is not None
        )
    return matches


def main(argv: Sequence[str] | None = None) -> int:
    """Scan explicit paths, or every tracked path when none are supplied."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    paths = tuple(Path(argument) for argument in arguments) or _tracked_paths()
    matches = _match_count(paths)
    if matches:
        print(f"secret-pattern matches detected: {matches}", file=sys.stderr)
        return 1
    print("secret-pattern scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
