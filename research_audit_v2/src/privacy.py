"""Public-output guardrail scanner."""
from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = [re.compile(r"https?://", re.I), re.compile(r"[A-Z]:\\\\"), re.compile(r"(?:^|[\\/])(?:photos|aligned)(?:[\\/])", re.I), re.compile(r"@\w+\.", re.I)]


def scan_public_outputs(root: Path) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".csv", ".md", ".json", ".svg", ".tex"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"Unreadable public text output: {path.name}")
            continue
        if any(pattern.search(content) for pattern in FORBIDDEN):
            violations.append(f"Forbidden sensitive pattern in {path.name}")
    return violations
