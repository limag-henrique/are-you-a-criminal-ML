from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_audit_v2.demographic_composition.embeddings import extract_union_embeddings


def _row(scenario: str, record_id: str, group: str, rank: int) -> dict[str, object]:
    return {
        "scenario": scenario,
        "record_id": record_id,
        "source_race_label": group,
        "relative_path": f"{record_id}.jpg",
        "selection_rank": rank,
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.DataFrame(
        [
            _row("A", "x0", "X", 0),
            _row("A", "x1", "X", 1),
            _row("A", "y0", "Y", 0),
            _row("A", "y1", "Y", 1),
            _row("B", "x0", "X", 0),
            _row("B", "x1", "X", 1),
            _row("B", "y0", "Y", 0),
            _row("B", "y1", "Y", 1),
        ]
    )
    reserves = pd.DataFrame(
        [
            {key: value for key, value in _row("", "x2", "X", 2).items() if key != "scenario"},
            {key: value for key, value in _row("", "y2", "Y", 2).items() if key != "scenario"},
        ]
    )
    return selected, reserves


class FakeEmbedder:
    calls: list[str] = []

    def __init__(self, **_: object) -> None:
        pass

    def extract_path(self, path):
        record = path.stem
        self.calls.append(record)
        if record == "x0":
            raise ValueError("synthetic detection failure")
        value = float(sum(ord(char) for char in record))
        return SimpleNamespace(embedding=np.array([value, 1.0], dtype=np.float32))


def _config(model_name: str = "test-model") -> dict[str, object]:
    return {
        "group_column": "source_race_label",
        "model_name": model_name,
        "ctx_id": -1,
        "det_size": 64,
        "preprocessing_mode": "aligned_crop_direct",
        "embedding_batch_size": 2,
    }


def test_extracts_each_union_record_once_and_replaces_failures_with_same_group_reserve(tmp_path):
    selected, reserves = _inputs()
    FakeEmbedder.calls = []

    final, vectors, failures = extract_union_embeddings(
        selected,
        reserves,
        tmp_path / "images",
        tmp_path / "private",
        _config(),
        embedder_factory=FakeEmbedder,
    )

    assert FakeEmbedder.calls == ["x0", "x1", "x2", "y0", "y1"]
    assert failures[["record_id", "source_race_label"]].to_dict("records") == [
        {"record_id": "x0", "source_race_label": "X"}
    ]
    assert set(final.loc[final["source_race_label"].eq("X"), "record_id"]) == {"x1", "x2"}
    assert final.groupby(["scenario", "source_race_label"]).size().eq(2).all()
    assert final.groupby("scenario")["record_id"].nunique().eq(4).all()
    assert sorted(final["embedding_index"].unique()) == list(range(len(vectors)))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_compatible_cache_resumes_without_constructing_an_embedder(tmp_path):
    selected, reserves = _inputs()
    extract_union_embeddings(
        selected, reserves, tmp_path / "images", tmp_path / "private", _config(), embedder_factory=FakeEmbedder
    )

    def forbidden_factory(**_: object):
        raise AssertionError("Compatible cache should avoid extractor construction")

    final, vectors, failures = extract_union_embeddings(
        selected,
        reserves,
        tmp_path / "images",
        tmp_path / "private",
        _config(),
        embedder_factory=forbidden_factory,
    )

    assert len(final) == 8
    assert vectors.shape == (4, 2)
    assert failures["record_id"].tolist() == ["x0"]


def test_changed_extractor_configuration_invalidates_private_cache(tmp_path):
    selected, reserves = _inputs()
    extract_union_embeddings(
        selected, reserves, tmp_path / "images", tmp_path / "private", _config("one"), embedder_factory=FakeEmbedder
    )
    FakeEmbedder.calls = []

    extract_union_embeddings(
        selected, reserves, tmp_path / "images", tmp_path / "private", _config("two"), embedder_factory=FakeEmbedder
    )

    assert FakeEmbedder.calls == ["x0", "x1", "x2", "y0", "y1"]


def test_batch_capable_embedder_processes_aligned_crops_in_group_batches(tmp_path):
    selected, reserves = _inputs()

    class BatchEmbedder:
        calls: list[list[str]] = []

        def __init__(self, **_: object) -> None:
            pass

        def extract_path(self, path):
            raise AssertionError("Aligned-crop extraction should use the batch interface")

        def extract_paths(self, paths):
            self.calls.append([path.stem for path in paths])
            return [SimpleNamespace(embedding=np.array([index + 1.0, 1.0])) for index, _ in enumerate(paths)]

    BatchEmbedder.calls = []
    final, vectors, failures = extract_union_embeddings(
        selected,
        reserves,
        tmp_path / "images",
        tmp_path / "private",
        _config(),
        embedder_factory=BatchEmbedder,
    )

    assert BatchEmbedder.calls == [["x0", "x1"], ["y0", "y1"]]
    assert len(final) == 8
    assert vectors.shape == (4, 2)
    assert failures.empty


def test_completed_groups_are_checkpointed_before_a_later_group_failure(tmp_path):
    selected, reserves = _inputs()

    class LaterGroupFailureEmbedder(FakeEmbedder):
        def extract_path(self, path):
            if path.stem.startswith("y"):
                raise ValueError("synthetic later-group failure")
            return SimpleNamespace(embedding=np.array([2.0, 1.0], dtype=np.float32))

    with pytest.raises(RuntimeError, match="Insufficient usable FairFace records for Y"):
        extract_union_embeddings(
            selected,
            reserves,
            tmp_path / "images",
            tmp_path / "private",
            _config(),
            embedder_factory=LaterGroupFailureEmbedder,
        )

    with np.load(tmp_path / "private" / "embedding_cache.npz", allow_pickle=False) as stored:
        assert stored["record_ids"].astype(str).tolist() == ["x0", "x1"]
