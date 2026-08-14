"""Fail-closed scanner for accidental disclosure in public artifacts."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .io import atomic_write_json


TEXT_EXTENSIONS = {".csv", ".md", ".json", ".svg", ".tex"}
PROHIBITED_BINARY_EXTENSIONS = {
    ".bmp",
    ".feather",
    ".gif",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".png",
    ".webp",
}
FORBIDDEN_PATTERNS = {
    "url": re.compile(r"https?://", re.I),
    "windows_path": re.compile(r"(?:[A-Z]:\\|\\\\[^\\\s]+\\)", re.I),
    "unix_path": re.compile(r"/(?:home|users|photos|artifacts|aligned)(?:/|\b)", re.I),
    "email": re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
    "image_reference": re.compile(r"\.(?:jpg|jpeg|png|webp|gif|bmp)\b", re.I),
}
FORBIDDEN_COLUMNS = {
    "aligned_path",
    "email",
    "full_name",
    "image",
    "name",
    "path",
    "resolved_path",
    "subject_id",
    "url",
    "vector",
}


def _normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _forbidden_column(value: object) -> bool:
    normalized = _normalized_column(value)
    if normalized in FORBIDDEN_COLUMNS or normalized in {"embedding", "embeddings"}:
        return True
    if normalized.startswith("embedding_"):
        suffix = normalized.removeprefix("embedding_")
        return suffix.isdigit() or suffix in {"data", "values", "vector"}
    return False


def _json_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [key for child in value.values() for key in _json_keys(child)]
    if isinstance(value, list):
        return [key for child in value for key in _json_keys(child)]
    return []


def _text_rules(path: Path, content: str) -> list[str]:
    if path.suffix.lower() == ".svg":
        return ["image_reference"] if FORBIDDEN_PATTERNS["image_reference"].search(content) else []
    rules = [name for name, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(content)]
    if path.suffix.lower() == ".csv" and content:
        try:
            header = next(csv.reader(content.splitlines()))
        except (csv.Error, StopIteration):
            rules.append("invalid_csv_header")
        else:
            if any(_forbidden_column(column) for column in header):
                rules.append("forbidden_column")
    if path.suffix.lower() == ".json" and content:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            rules.append("invalid_json")
        else:
            if any(_forbidden_column(key) for key in _json_keys(payload)):
                rules.append("forbidden_key")
    return sorted(set(rules))


def scan_public_tree(root: Path) -> list[dict[str, str]]:
    """Return only artifact-relative metadata, never matched sensitive content."""
    findings: list[dict[str, str]] = []
    for artifact in sorted(root.rglob("*")):
        if not artifact.is_file():
            continue
        relative = artifact.relative_to(root).as_posix()
        suffix = artifact.suffix.lower()
        if suffix in PROHIBITED_BINARY_EXTENSIONS:
            findings.append({"artifact": relative, "rule": "prohibited_binary_artifact"})
            continue
        if suffix not in TEXT_EXTENSIONS:
            continue
        try:
            content = artifact.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"artifact": relative, "rule": "invalid_utf8_public_text"})
            continue
        findings.extend({"artifact": relative, "rule": rule} for rule in _text_rules(artifact, content))
    return findings


def write_privacy_report(root: Path, destination: Path) -> list[dict[str, str]]:
    findings = scan_public_tree(root)
    atomic_write_json(
        destination,
        {
            "status": "passed" if not findings else "failed",
            "finding_count": len(findings),
            "findings": findings,
        },
    )
    return findings
