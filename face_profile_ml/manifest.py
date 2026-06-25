from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"path", "subject_id", "quality", "split", "weight"}

QUALITY_ALIASES = {
    "high": "high",
    "best": "high",
    "best quality": "high",
    "mid": "mid",
    "medium": "mid",
    "mid quality": "mid",
    "low": "low",
    "low quality": "low",
}


def read_manifest(path: str | Path, root_dir: str | Path | None = None) -> pd.DataFrame:
    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"manifest.csv is missing required columns: {missing_text}")

    frame = frame.copy()
    frame["path"] = frame["path"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    original_quality = frame["quality"].astype(str)
    frame["quality"] = frame["quality"].map(_normalize_quality)
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0).astype(float)

    if (frame["weight"] <= 0).any():
        raise ValueError("All manifest weights must be positive numbers.")
    if frame["quality"].isna().any():
        bad = sorted(set(original_quality.loc[frame["quality"].isna()].astype(str)))
        raise ValueError(f"Unknown quality values in manifest: {bad}")

    base = Path(root_dir) if root_dir else manifest_path.parent
    frame["resolved_path"] = [
        str((base / item).resolve()) if not Path(item).is_absolute() else str(Path(item).resolve())
        for item in frame["path"]
    ]
    frame["exists"] = [Path(item).exists() for item in frame["resolved_path"]]
    frame.insert(0, "row_id", range(len(frame)))
    return frame


def split_mask(frame: pd.DataFrame, split_names: list[str]) -> pd.Series:
    normalized = {item.strip().lower() for item in split_names}
    return frame["split"].astype(str).str.lower().isin(normalized)


def _normalize_quality(value: object) -> str | None:
    key = str(value).strip().lower().replace("_", " ")
    return QUALITY_ALIASES.get(key)
