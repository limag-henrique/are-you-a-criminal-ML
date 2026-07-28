"""Second-phase scanner for accidental disclosure in public artifacts."""
from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_PATTERNS = {
    "url": re.compile(r"https?://", re.I),
    "windows_path": re.compile(r"[A-Z]:\\\\", re.I),
    "unix_path": re.compile(r"/(?:home|users|photos|artifacts|aligned)/", re.I),
    "email": re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
    "image_reference": re.compile(r"\.(?:jpg|jpeg|png|webp)\b", re.I),
}
FORBIDDEN_COLUMNS = {"path", "resolved_path", "aligned_path", "subject_id", "name", "url", "embedding"}


def scan_text(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path.name}: not valid UTF-8 public text"]
    # Vector graphics conventionally include W3C and RDF namespace URLs. They
    # are not individual/source URLs; scan SVGs only for file/image references.
    if path.suffix.lower() == ".svg":
        return [f"{path.name}: image_reference"] if FORBIDDEN_PATTERNS["image_reference"].search(content) else []
    return [f"{path.name}: {name}" for name, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(content)]


def scan_public_tree(root: Path) -> list[str]:
    findings: list[str] = []
    for artifact in root.rglob("*"):
        if artifact.suffix.lower() not in {".csv", ".md", ".json", ".svg", ".tex"}:
            continue
        findings.extend(scan_text(artifact))
        if artifact.suffix.lower() == ".csv":
            header = artifact.read_text(encoding="utf-8").splitlines()[0].lower().split(",") if artifact.stat().st_size else []
            findings.extend(f"{artifact.name}: forbidden column {column}" for column in header if column in FORBIDDEN_COLUMNS)
    return findings
