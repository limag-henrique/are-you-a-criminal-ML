"""Private, resumable embedding extraction for the FairFace experiment."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from face_profile_ml.extractor import ArcFaceEmbedder, read_bgr_image, select_available_providers
from research_audit_v2.second_phase.src.io import atomic_write_json


def _extractor_key(config: Mapping[str, object]) -> str:
    payload = {
        key: config.get(key)
        for key in (
            "model_name",
            "ctx_id",
            "det_size",
            "providers",
            "preprocessing_mode",
            "embedding_batch_size",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_cache(root: Path, key: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    metadata_path = root / "embedding_cache.json"
    arrays_path = root / "embedding_cache.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        return {}, {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("extractor_key") != key:
            return {}, {}
        with np.load(arrays_path, allow_pickle=False) as stored:
            ids = stored["record_ids"].astype(str)
            vectors = np.asarray(stored["vectors"], dtype=np.float32)
        if len(ids) != len(vectors) or vectors.ndim != 2:
            return {}, {}
        successes = {record_id: vector for record_id, vector in zip(ids, vectors)}
        failures = {
            str(item["record_id"]): {
                "record_id": str(item["record_id"]),
                "source_race_label": str(item["source_race_label"]),
                "error_type": str(item["error_type"]),
            }
            for item in metadata.get("failures", [])
        }
        return successes, failures
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {}, {}


def _write_cache(
    root: Path,
    key: str,
    successes: Mapping[str, np.ndarray],
    failures: Mapping[str, Mapping[str, str]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ids = sorted(successes)
    vectors = np.vstack([successes[record_id] for record_id in ids]).astype(np.float32)
    handle, temporary_name = tempfile.mkstemp(prefix="embedding_cache_", suffix=".npz", dir=root)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, record_ids=np.asarray(ids), vectors=vectors)
        os.replace(temporary, root / "embedding_cache.npz")
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write_json(
        root / "embedding_cache.json",
        {
            "extractor_key": key,
            "successful_records": len(ids),
            "failures": [failures[record_id] for record_id in sorted(failures)],
        },
    )


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must be finite and non-zero.")
    return value / norm


class FairFaceAlignedCropEmbedder:
    """Batch ArcFace inference over FairFace's upstream dlib-aligned crops."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        del det_size
        try:
            import onnxruntime  # type: ignore
            from insightface import model_zoo  # type: ignore
            from insightface.utils.storage import ensure_available  # type: ignore
        except Exception as exc:
            raise RuntimeError("InsightFace is required for aligned-crop ArcFace inference.") from exc
        onnxruntime.set_default_logger_severity(3)
        model_dir = Path(ensure_available("models", model_name)).expanduser()
        self.recognizer = None
        for model_path in sorted(model_dir.glob("*.onnx")):
            candidate = model_zoo.get_model(
                str(model_path), providers=select_available_providers(providers)
            )
            if getattr(candidate, "taskname", None) == "recognition":
                self.recognizer = candidate
                break
        if self.recognizer is None:
            raise RuntimeError(f"Recognition model not found in InsightFace pack {model_name}.")
        self.recognizer.prepare(ctx_id=ctx_id)

    def extract_paths(self, paths: list[Path]) -> list[object]:
        results: list[object] = [ValueError("Unreadable image") for _ in paths]
        images = []
        positions = []
        for position, path in enumerate(paths):
            image = read_bgr_image(path)
            if image is not None:
                images.append(image)
                positions.append(position)
        if images:
            features = np.asarray(self.recognizer.get_feat(images), dtype=np.float32)
            for position, feature in zip(positions, features):
                results[position] = SimpleNamespace(embedding=_normalized(feature))
        return results


def extract_union_embeddings(
    selected: pd.DataFrame,
    reserves: pd.DataFrame,
    image_root: str | Path,
    private_root: str | Path,
    config: Mapping[str, object],
    *,
    embedder_factory: Callable[..., object] = ArcFaceEmbedder,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Extract the shared union and rebuild every scenario from usable prefixes."""
    group_column = str(config["group_column"])
    required = {"scenario", "record_id", group_column, "relative_path", "selection_rank"}
    if missing := required.difference(selected.columns):
        raise ValueError(f"Selection is missing columns: {sorted(missing)}")
    quota_frame = selected.groupby(["scenario", group_column]).size().unstack(fill_value=0)
    candidates = pd.concat(
        [
            selected.loc[:, ["record_id", group_column, "relative_path", "selection_rank"]],
            reserves.loc[:, ["record_id", group_column, "relative_path", "selection_rank"]],
        ],
        ignore_index=True,
    ).drop_duplicates("record_id")
    candidates = candidates.sort_values([group_column, "selection_rank", "record_id"])
    max_quotas = quota_frame.max(axis=0).astype(int).to_dict()

    root = Path(private_root)
    key = _extractor_key(config)
    successes, failures = _load_cache(root, key)
    embedder: object | None = None
    usable: dict[str, list[dict[str, object]]] = {}
    image_base = Path(image_root)
    batch_size = int(config.get("embedding_batch_size", 1))
    if batch_size < 1:
        raise ValueError("embedding_batch_size must be positive.")
    for group in sorted(max_quotas):
        needed = int(max_quotas[group])
        usable[group] = []
        group_candidates = candidates[candidates[group_column].eq(group)].to_dict("records")
        cursor = 0
        while len(usable[group]) < needed and cursor < len(group_candidates):
            chunk = group_candidates[cursor : cursor + batch_size]
            cursor += len(chunk)
            pending = [
                row
                for row in chunk
                if str(row["record_id"]) not in successes and str(row["record_id"]) not in failures
            ]
            if pending:
                if embedder is None:
                    embedder = embedder_factory(
                        model_name=str(config["model_name"]),
                        ctx_id=int(config["ctx_id"]),
                        det_size=int(config["det_size"]),
                        providers=config.get("providers"),
                    )
                paths = [image_base / str(row["relative_path"]) for row in pending]
                if hasattr(embedder, "extract_paths"):
                    extracted = embedder.extract_paths(paths)
                else:
                    extracted = []
                    for path in paths:
                        try:
                            extracted.append(embedder.extract_path(path))
                        except Exception as exc:
                            extracted.append(exc)
                if len(extracted) != len(pending):
                    raise RuntimeError("Batch embedder returned a misaligned result count.")
            else:
                extracted = []
            for row, result in zip(pending, extracted):
                record_id = str(row["record_id"])
                if isinstance(result, Exception):
                    failures[record_id] = {
                        "record_id": record_id,
                        "source_race_label": group,
                        "error_type": type(result).__name__,
                    }
                else:
                    try:
                        successes[record_id] = _normalized(result.embedding)
                    except Exception as exc:
                        failures[record_id] = {
                            "record_id": record_id,
                            "source_race_label": group,
                            "error_type": type(exc).__name__,
                        }
            for row in chunk:
                if str(row["record_id"]) in successes:
                    usable[group].append(row)
                    if len(usable[group]) == needed:
                        break
        if len(usable[group]) != needed:
            raise RuntimeError(
                f"Insufficient usable FairFace records for {group}: {len(usable[group])}/{needed}."
            )
        _write_cache(root, key, successes, failures)

    ordered_records = [row for group in sorted(usable) for row in usable[group]]
    index_by_id = {str(row["record_id"]): index for index, row in enumerate(ordered_records)}
    vectors = np.vstack([successes[str(row["record_id"])] for row in ordered_records]).astype(np.float32)
    final_frames = []
    for scenario in quota_frame.index:
        for group in quota_frame.columns:
            quota = int(quota_frame.loc[scenario, group])
            frame = pd.DataFrame(usable[str(group)][:quota])
            frame.insert(0, "scenario", str(scenario))
            final_frames.append(frame)
    final = pd.concat(final_frames, ignore_index=True)
    final["embedding_index"] = final["record_id"].map(index_by_id).astype(int)
    final = final.sort_values(["scenario", group_column, "selection_rank"]).reset_index(drop=True)
    _write_cache(root, key, successes, failures)
    failure_frame = pd.DataFrame(
        [failures[record_id] for record_id in sorted(failures)],
        columns=["record_id", "source_race_label", "error_type"],
    )
    return final, vectors, failure_frame
