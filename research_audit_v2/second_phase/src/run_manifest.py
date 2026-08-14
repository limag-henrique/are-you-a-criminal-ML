"""Incremental run-manifest contract."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from research_audit_v2.src.common import sha256_file

from .io import atomic_write_json


class ManifestCompatibilityError(ValueError):
    """Raised when a checkpoint belongs to different inputs or parameters."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_metadata(input_files: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256_file(path), "size_bytes": Path(path).stat().st_size}
        for name, path in sorted(input_files.items())
    }


def _public_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    def public_key(key: str) -> str:
        if key.lower() == "path":
            return "path_redacted"
        if key.lower() == "name":
            return "method_name"
        return key

    def sanitize(key: str, value: Any) -> Any:
        normalized = key.lower()
        if (
            normalized.endswith("_path")
            or normalized.endswith("_root")
            or normalized.endswith("_destination")
            or normalized in {"path", "output_root"}
        ):
            return f"<redacted-path:{key}>"
        if isinstance(value, Mapping):
            return {
                public_key(str(child)): sanitize(str(child), item)
                for child, item in value.items()
            }
        if isinstance(value, list):
            return [sanitize(key, item) for item in value]
        return value

    return {
        public_key(str(key)): sanitize(str(key), value)
        for key, value in config.items()
    }


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


class RunManifest:
    def __init__(self, path: str | Path, payload: dict[str, Any]):
        self.path = Path(path)
        self.payload = payload

    def _save(self) -> None:
        atomic_write_json(self.path, self.payload)

    @classmethod
    def start(
        cls,
        path: str | Path,
        *,
        config_name: str,
        config: Mapping[str, Any],
        seeds: Sequence[int],
        input_files: Mapping[str, str | Path],
        parameters: Mapping[str, Any],
    ) -> "RunManifest":
        config_sha256 = _config_hash(config)
        started = _utc_now()
        payload: dict[str, Any] = {
            "schema_version": 2,
            "run_id": f"{config_name}-{started.replace(':', '').replace('+00:00', 'Z')}-{config_sha256[:8]}",
            "status": "initializing",
            "completion_status": "initializing",
            "config_name": config_name,
            "config_sha256": config_sha256,
            "configuration": _public_configuration(config),
            "seeds": [int(seed) for seed in seeds],
            "parameters": dict(parameters),
            "inputs": {},
            "git_commit": _git_value("rev-parse", "HEAD"),
            "worktree_dirty": bool(_git_value("status", "--porcelain")),
            "versions": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine() or "unavailable",
                "processor": platform.processor() or "unavailable",
            },
            "started_utc": started,
            "finished_utc": None,
            "duration_seconds": None,
            "outputs": [],
        }
        manifest = cls(path, payload)
        manifest._save()
        manifest.payload["inputs"] = _input_metadata(input_files)
        manifest.payload["status"] = "running"
        manifest.payload["completion_status"] = "running"
        manifest._save()
        return manifest

    @classmethod
    def resume(
        cls,
        path: str | Path,
        *,
        config: Mapping[str, Any],
        input_files: Mapping[str, str | Path],
    ) -> "RunManifest":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("config_sha256") != _config_hash(config):
            raise ManifestCompatibilityError("Cannot resume with a different configuration.")
        if payload.get("inputs") != _input_metadata(input_files):
            raise ManifestCompatibilityError("Cannot resume with different input hashes.")
        payload["schema_version"] = 2
        payload["configuration"] = _public_configuration(config)
        if payload.get("duration_seconds") is None and payload.get("finished_utc"):
            payload["duration_seconds"] = max(
                0.0,
                (
                    datetime.fromisoformat(payload["finished_utc"])
                    - datetime.fromisoformat(payload["started_utc"])
                ).total_seconds(),
            )
        manifest = cls(manifest_path, payload)
        manifest._save()
        return manifest

    def record_output(self, path: str | Path, *, logical_name: str | None = None) -> None:
        output = Path(path)
        item = {
            "artifact": logical_name or output.name,
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        }
        existing = [
            value
            for value in self.payload["outputs"]
            if value.get("artifact", value.get("name")) != item["artifact"]
        ]
        self.payload["outputs"] = sorted(
            [*existing, item], key=lambda value: value.get("artifact", value.get("name", ""))
        )
        self._save()

    def complete(self) -> None:
        finished = _utc_now()
        self.payload["status"] = "complete"
        self.payload["completion_status"] = "complete"
        self.payload["finished_utc"] = finished
        self.payload["duration_seconds"] = max(
            0.0,
            (datetime.fromisoformat(finished) - datetime.fromisoformat(self.payload["started_utc"])).total_seconds(),
        )
        self._save()

    def fail(self, reason: str) -> None:
        finished = _utc_now()
        self.payload["status"] = "failed"
        self.payload["completion_status"] = "failed"
        self.payload["failure"] = reason
        self.payload["finished_utc"] = finished
        self.payload["duration_seconds"] = max(
            0.0,
            (datetime.fromisoformat(finished) - datetime.fromisoformat(self.payload["started_utc"])).total_seconds(),
        )
        self._save()
