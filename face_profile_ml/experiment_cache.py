"""Atomic, integrity-checked artifact bundles for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PureWindowsPath
import pickle
import re
import tempfile
from typing import Any, Mapping

import pandas as pd

from .experiment_specs import canonical_json


@dataclass(frozen=True)
class BundleValidation:
    """The integrity status of an artifact bundle."""

    status: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompletionResult:
    """The recorded completion state for a fully classified experiment."""

    status: str
    content_hash: str


class ArtifactBundle:
    """Persist an experiment's checkpoints and tables as one auditable bundle."""

    _MANIFEST_NAME = "artifact_manifest.json"

    def __init__(
        self,
        root: str | Path,
        experiment_name: str,
        config: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> None:
        if not experiment_name or Path(experiment_name).name != experiment_name:
            raise ValueError("experiment_name must be a single directory name")
        self.root = Path(root) / experiment_name
        self.root.mkdir(parents=True, exist_ok=True)
        self._assert_relative_paths(config)
        self._assert_relative_paths(inputs)
        self._config = dict(config)
        self._inputs = dict(inputs)
        self._config_hash = self._hash_json(self._config)
        self._inputs_hash = self._hash_json(self._inputs)
        manifest_path = self.root / self._MANIFEST_NAME
        if manifest_path.exists():
            validation, manifest = self._load_validated_manifest()
            if validation.status != "valid":
                raise ValueError("; ".join(validation.errors))
            self._manifest = manifest
            self._verify_resume_identity(manifest)
        else:
            self._atomic_json(self.root / "config.json", self._config)
            self._atomic_json(self.root / "inputs.json", self._inputs)
            self._manifest: dict[str, Any] = {
                "config_hash": self._config_hash,
                "inputs_hash": self._inputs_hash,
                "fits": {},
                "tables": {},
                "failure_log": None,
            }
            self._write_manifest()

    def fit_path(self, fit_id: str) -> Path:
        """Return the checkpoint path assigned to a declared fit identity."""
        return self.root / "fits" / f"{self._safe_id(fit_id)}.pkl"

    def write_fit(self, fit_id: str, fit: Any) -> Path:
        """Atomically persist a fitted object and its content hash."""
        self._prepare_mutation()
        path = self.fit_path(fit_id)
        payload = pickle.dumps(fit, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_bytes(path, payload)
        self._manifest["fits"][fit_id] = {
            "path": self._relative_path(path),
            "sha256": self._hash_bytes(payload),
        }
        self._write_manifest()
        return path

    def read_fit(self, fit_id: str) -> Any:
        """Read a checkpoint only after verifying the recorded digest."""
        entry = self._manifest["fits"].get(fit_id)
        if entry is None:
            raise KeyError(f"unknown fit_id: {fit_id}")
        path = self._path_from_manifest(entry["path"])
        payload = self._read_checked(path, entry["sha256"])
        return pickle.loads(payload)

    def record_failure(
        self, failure: Mapping[str, Any] | None = None, **details: Any
    ) -> Path:
        """Append an explicit, hashed failure classification to ``failures.csv``."""
        record = dict(failure or {})
        record.update(details)
        if not record:
            raise ValueError("failure details must not be empty")
        self._prepare_mutation()
        path = self.root / "failures.csv"
        rows = (
            pd.read_csv(path, dtype=str).fillna("").to_dict("records")
            if path.exists()
            else []
        )
        rows.append({key: str(value) for key, value in record.items()})
        columns = sorted({key for row in rows for key in row})
        frame = pd.DataFrame(rows, columns=columns).fillna("")
        csv_payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        self._atomic_bytes(path, csv_payload)
        self._manifest["failure_log"] = {
            "path": self._relative_path(path),
            "sha256": self._hash_bytes(csv_payload),
        }
        self._write_manifest()
        return path

    def write_tables(
        self,
        tables: Mapping[str, pd.DataFrame] | None = None,
        **named_tables: pd.DataFrame,
    ) -> dict[str, Path]:
        """Atomically write named result tables as Parquet files."""
        supplied = dict(tables or {})
        overlap = set(supplied) & set(named_tables)
        if overlap:
            raise ValueError(f"duplicate table names: {sorted(overlap)}")
        supplied.update(named_tables)
        if not supplied:
            raise ValueError("at least one table is required")
        self._prepare_mutation()
        paths: dict[str, Path] = {}
        for name, table in supplied.items():
            safe_name = self._safe_id(name)
            if not isinstance(table, pd.DataFrame):
                raise TypeError(f"table {name!r} must be a pandas DataFrame")
            path = self.root / f"{safe_name}.parquet"
            self._atomic_parquet(path, table)
            self._manifest["tables"][name] = {
                "path": self._relative_path(path),
                "sha256": self._hash_file(path),
            }
            paths[name] = path
        self._write_manifest()
        return paths

    def validate(self) -> BundleValidation:
        """Validate every content-addressed file referenced by the manifest."""
        validation, _ = self._load_validated_manifest()
        return validation

    def _load_validated_manifest(self) -> tuple[BundleValidation, dict[str, Any]]:
        """Load one manifest version and validate all files it claims to own."""
        errors: list[str] = []
        try:
            manifest = self._read_json(self.root / self._MANIFEST_NAME)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return BundleValidation("invalid", (f"manifest unreadable: {exc}",)), {}
        if not isinstance(manifest, dict):
            return BundleValidation("invalid", ("manifest must be a JSON object",)), {}
        if not isinstance(manifest.get("fits"), dict) or not isinstance(
            manifest.get("tables"), dict
        ):
            return BundleValidation(
                "invalid", ("manifest has invalid artifact indexes",)
            ), {}
        for field in ("config_hash", "inputs_hash"):
            if not self._is_sha256(manifest.get(field)):
                errors.append(f"manifest has invalid {field}")
        if "failure_log" not in manifest:
            errors.append("manifest is missing failure_log")
        for name, path, expected in (
            ("config", self.root / "config.json", manifest.get("config_hash")),
            ("inputs", self.root / "inputs.json", manifest.get("inputs_hash")),
        ):
            try:
                value = self._read_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"unreadable {name}: {exc}")
                continue
            if expected != self._hash_json(value):
                errors.append(f"hash mismatch: {name}")
        entries: list[tuple[str, Any]] = []
        entries.extend(
            (f"fit {fit_id}", entry) for fit_id, entry in manifest["fits"].items()
        )
        entries.extend(
            (f"table {name}", entry) for name, entry in manifest["tables"].items()
        )
        if manifest.get("failure_log") is not None:
            entries.append(("failure log", manifest["failure_log"]))
        declared_paths = {
            "config.json",
            "inputs.json",
            self._MANIFEST_NAME,
            "completion.json",
        }
        for label, entry in entries:
            if not isinstance(entry, Mapping):
                errors.append(f"invalid manifest entry: {label}")
                continue
            try:
                relative_path = entry["path"]
                digest = entry["sha256"]
                if not isinstance(relative_path, str) or not self._is_sha256(digest):
                    raise ValueError("entry must contain a relative path and SHA-256")
                path = self._path_from_manifest(relative_path)
                digest = self._hash_file(path)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"missing {label}: {exc}")
                continue
            declared_paths.add(self._relative_path(path))
            if digest != entry["sha256"]:
                errors.append(f"hash mismatch: {label}")
        errors.extend(self._orphan_errors(declared_paths))
        completion_path = self.root / "completion.json"
        if completion_path.exists():
            errors.extend(self._completion_errors(completion_path, manifest))
        return (
            BundleValidation("valid" if not errors else "invalid", tuple(errors)),
            manifest,
        )

    def complete(
        self,
        *,
        expected_fit_ids: set[str],
        expected_spec_ids: set[str],
    ) -> CompletionResult:
        """Record completion after integrity and declared-cell validation."""
        validation, manifest = self._load_validated_manifest()
        if validation.status != "valid":
            raise ValueError("; ".join(validation.errors))
        actual_fit_ids = set(manifest["fits"])
        if actual_fit_ids != set(expected_fit_ids):
            raise ValueError(
                "fit IDs do not match expected set: "
                f"expected={sorted(expected_fit_ids)}, actual={sorted(actual_fit_ids)}"
            )
        self._validate_specification_cells(manifest, set(expected_spec_ids))
        content_hash = self._content_hash(manifest)
        completion = {
            "status": "complete",
            "expected_fit_ids": sorted(expected_fit_ids),
            "expected_spec_ids": sorted(expected_spec_ids),
            "content_hash": content_hash,
        }
        self._atomic_json(self.root / "completion.json", completion)
        return CompletionResult("complete", content_hash)

    def _validate_specification_cells(
        self, manifest: Mapping[str, Any], expected_spec_ids: set[str]
    ) -> None:
        entry = manifest["tables"].get("specification_metrics")
        if entry is None:
            raise ValueError("missing specification_metrics table")
        path = self._path_from_manifest(entry["path"])
        metrics = pd.read_parquet(path)
        required = {"spec_id", "status"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"specification_metrics is missing columns: {sorted(missing)}")
        if metrics["spec_id"].duplicated().any():
            raise ValueError("specification_metrics contains duplicate spec_id values")
        observed = set(metrics["spec_id"].astype(str))
        if observed != expected_spec_ids:
            raise ValueError(
                "spec IDs do not match expected set: "
                f"expected={sorted(expected_spec_ids)}, actual={sorted(observed)}"
            )
        statuses = metrics["status"].astype(str)
        invalid = sorted(
            set(statuses[~(statuses.eq("complete") | statuses.str.startswith("ineligible_"))])
        )
        if invalid:
            raise ValueError(f"unclassified specification cells: {invalid}")

    def _verify_resume_identity(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("config_hash") != self._config_hash:
            raise ValueError("cannot resume bundle: config hash differs")
        if manifest.get("inputs_hash") != self._inputs_hash:
            raise ValueError("cannot resume bundle: inputs hash differs")

    def _prepare_mutation(self) -> None:
        """Refuse to extend a corrupt bundle and invalidate an old completion."""
        validation, manifest = self._load_validated_manifest()
        if validation.status != "valid":
            raise ValueError("; ".join(validation.errors))
        self._manifest = manifest
        (self.root / "completion.json").unlink(missing_ok=True)

    def _orphan_errors(self, declared_paths: set[str]) -> list[str]:
        errors: list[str] = []
        for path in self.root.rglob("*"):
            if path.is_file() and self._relative_path(path) not in declared_paths:
                errors.append(f"orphan artifact: {self._relative_path(path)}")
        return errors

    def _completion_errors(
        self, path: Path, manifest: Mapping[str, Any]
    ) -> list[str]:
        try:
            completion = self._read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return [f"completion unreadable: {exc}"]
        if not isinstance(completion, Mapping):
            return ["completion must be a JSON object"]
        if completion.get("status") != "complete":
            return ["completion has invalid status"]
        if completion.get("content_hash") != self._content_hash(manifest):
            return ["completion content hash does not match manifest"]
        return []

    def _write_manifest(self) -> None:
        self._atomic_json(self.root / self._MANIFEST_NAME, self._manifest)

    @staticmethod
    def _safe_id(value: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("artifact identifiers must be simple non-empty names")
        return value

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _path_from_manifest(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or PureWindowsPath(relative_path).is_absolute() or ".." in candidate.parts:
            raise ValueError("manifest paths must be relative to the bundle")
        resolved = (self.root / candidate).resolve()
        if self.root.resolve() not in resolved.parents and resolved != self.root.resolve():
            raise ValueError("manifest path escapes the bundle")
        return resolved

    @staticmethod
    def _assert_relative_paths(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key.endswith("path") and isinstance(child, str):
                    if Path(child).is_absolute() or PureWindowsPath(child).is_absolute():
                        raise ValueError("artifact metadata paths must be relative")
                ArtifactBundle._assert_relative_paths(child)
        elif isinstance(value, list):
            for child in value:
                ArtifactBundle._assert_relative_paths(child)

    @staticmethod
    def _hash_bytes(payload: bytes) -> str:
        return sha256(payload).hexdigest()

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    @classmethod
    def _hash_file(cls, path: Path) -> str:
        return cls._hash_bytes(path.read_bytes())

    @classmethod
    def _hash_json(cls, value: Any) -> str:
        return cls._hash_bytes(canonical_json(value).encode("utf-8"))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)

    @classmethod
    def _atomic_json(cls, path: Path, value: Any) -> None:
        cls._atomic_bytes(path, canonical_json(value).encode("utf-8"))

    @classmethod
    def _atomic_parquet(cls, path: Path, table: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            table.to_parquet(temporary, index=False)
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _read_checked(self, path: Path, expected_hash: str) -> bytes:
        payload = path.read_bytes()
        if self._hash_bytes(payload) != expected_hash:
            raise ValueError(f"hash mismatch: {self._relative_path(path)}")
        return payload

    def _content_hash(self, manifest: Mapping[str, Any]) -> str:
        records: list[dict[str, str]] = []
        for entry in manifest["fits"].values():
            records.append({"path": entry["path"], "sha256": entry["sha256"]})
        for entry in manifest["tables"].values():
            records.append({"path": entry["path"], "sha256": entry["sha256"]})
        if manifest["failure_log"] is not None:
            records.append(dict(manifest["failure_log"]))
        return self._hash_json(sorted(records, key=lambda record: record["path"]))
