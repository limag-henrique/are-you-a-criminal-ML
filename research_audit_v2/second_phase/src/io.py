"""Atomic public-output writers."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

import pandas as pd


@contextmanager
def atomic_target(path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        yield temporary
        if not temporary.exists():
            raise RuntimeError("Atomic writer did not create its temporary file.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, content: str) -> None:
    with atomic_target(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_json(path: str | Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    with atomic_target(path) as temporary:
        frame.to_csv(temporary, index=False)
