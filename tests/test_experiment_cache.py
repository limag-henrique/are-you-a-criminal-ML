from hashlib import sha256
import json

import pandas as pd
import pytest

from face_profile_ml.experiment_cache import ArtifactBundle


def config() -> dict[str, object]:
    return {"seed": 7, "backend": "minibatch"}


def inputs() -> dict[str, object]:
    return {"embeddings": {"path": "inputs/embeddings.npy", "sha256": "abc"}}


def valid_fold_fit() -> dict[str, object]:
    return {"centers": [[0.0, 1.0]], "labels": [0, 0]}


def test_corrupt_checkpoint_is_not_complete(tmp_path) -> None:
    """Changing a stored checkpoint must invalidate the entire bundle."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    bundle.write_fit("fit-1", valid_fold_fit())
    bundle.fit_path("fit-1").write_bytes(b"corrupt")

    assert bundle.validate().status == "invalid"
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.complete(expected_fit_ids={"fit-1"}, expected_spec_ids=set())


def test_corrupt_config_is_reported_as_an_invalid_bundle(tmp_path) -> None:
    """A damaged manifest input is corruption, not an uncaught cache exception."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    (tmp_path / "ablation" / "config.json").write_bytes(b"not json")

    assert bundle.validate().status == "invalid"


def test_structurally_invalid_manifest_is_reported_as_invalid(tmp_path) -> None:
    """A valid JSON value still must have the artifact-manifest structure."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    (tmp_path / "ablation" / "artifact_manifest.json").write_text("[]", encoding="utf-8")

    assert bundle.validate().status == "invalid"


def test_complete_accepts_only_complete_or_explicitly_ineligible_specs(tmp_path) -> None:
    """A declared ineligible cell is a valid, auditable completion state."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    bundle.write_fit("fit-1", valid_fold_fit())
    bundle.write_tables(
        specification_metrics=pd.DataFrame(
            [
                {"spec_id": "spec-1", "status": "complete"},
                {"spec_id": "spec-2", "status": "ineligible_single_class"},
            ]
        )
    )

    completion = bundle.complete(
        expected_fit_ids={"fit-1"}, expected_spec_ids={"spec-1", "spec-2"}
    )

    assert completion.status == "complete"
    assert (tmp_path / "ablation" / "completion.json").is_file()


def test_complete_rejects_unclassified_specification_cells(tmp_path) -> None:
    """A partial or failed cell cannot be silently labelled as complete."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    bundle.write_fit("fit-1", valid_fold_fit())
    bundle.write_tables(
        specification_metrics=pd.DataFrame(
            [{"spec_id": "spec-1", "status": "partial_failed_folds"}]
        )
    )

    with pytest.raises(ValueError, match="unclassified specification cells"):
        bundle.complete(expected_fit_ids={"fit-1"}, expected_spec_ids={"spec-1"})


def test_resume_rejects_changed_input_identity(tmp_path) -> None:
    """A cache cannot be reused for a different input fingerprint."""
    ArtifactBundle(tmp_path, "ablation", config(), inputs())

    changed_inputs = {"embeddings": {"path": "inputs/embeddings.npy", "sha256": "def"}}
    with pytest.raises(ValueError, match="inputs hash differs"):
        ArtifactBundle(tmp_path, "ablation", config(), changed_inputs)


def test_complete_requires_declared_specification_coverage(tmp_path) -> None:
    """A caller cannot finalize a fit-only bundle by omitting the spec grid."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    bundle.write_fit("fit-1", valid_fold_fit())

    with pytest.raises(TypeError, match="expected_spec_ids"):
        bundle.complete(expected_fit_ids={"fit-1"})


def test_orphan_artifact_invalidates_bundle(tmp_path) -> None:
    """An interrupted publication must not leave an unknown artifact acceptable."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    orphan = tmp_path / "ablation" / "fits" / "interrupted.pkl"
    orphan.parent.mkdir()
    orphan.write_bytes(b"published before manifest")

    validation = bundle.validate()

    assert validation.status == "invalid"
    assert any("orphan artifact" in error for error in validation.errors)


def test_invalid_manifest_entry_is_reported_as_invalid(tmp_path) -> None:
    """An entry without a path/hash schema is corruption, never a TypeError."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    manifest_path = tmp_path / "ablation" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fits"] = {"fit-1": []}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert bundle.validate().status == "invalid"


def test_resume_rejects_persisted_config_hash_mismatch(tmp_path) -> None:
    """Resume checks the persisted config file, not just the manifest claim."""
    ArtifactBundle(tmp_path, "ablation", config(), inputs())
    (tmp_path / "ablation" / "config.json").write_text(
        json.dumps({"seed": 8, "backend": "minibatch"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hash mismatch: config"):
        ArtifactBundle(tmp_path, "ablation", config(), inputs())


def test_complete_uses_the_validated_manifest_from_disk(tmp_path) -> None:
    """Completion must reject a fit added to disk after this instance was created."""
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    bundle.write_fit("fit-1", valid_fold_fit())
    bundle.write_tables(
        specification_metrics=pd.DataFrame(
            [{"spec_id": "spec-1", "status": "complete"}]
        )
    )
    manifest_path = tmp_path / "ablation" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_path = tmp_path / "ablation" / manifest["fits"]["fit-1"]["path"]
    second_path = tmp_path / "ablation" / "fits" / "fit-2.pkl"
    second_path.write_bytes(first_path.read_bytes())
    manifest["fits"]["fit-2"] = {
        "path": "fits/fit-2.pkl",
        "sha256": sha256(second_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fit IDs do not match expected set"):
        bundle.complete(expected_fit_ids={"fit-1"}, expected_spec_ids={"spec-1"})
