import json

import pytest

from research_audit_v2.second_phase.src.run_manifest import ManifestCompatibilityError, RunManifest
from research_audit_v2.second_phase.src.privacy_scan import scan_public_tree


def test_run_manifest_exists_as_running_before_outputs_and_records_required_metadata(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"synthetic-input")
    path = tmp_path / "run_manifest.json"

    manifest = RunManifest.start(
        path,
        config_name="development",
        config={"mode": "development", "random_seed": 7},
        seeds=[7, 8],
        input_files={"embeddings": source},
        parameters={"k_values": [2, 3]},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["config_name"] == "development"
    assert payload["seeds"] == [7, 8]
    assert payload["parameters"] == {"k_values": [2, 3]}
    assert payload["inputs"]["embeddings"]["sha256"]
    assert payload["inputs"]["embeddings"]["size_bytes"] == len(b"synthetic-input")
    assert payload["git_commit"]
    assert payload["versions"]["python"]
    assert payload["system"]["platform"]
    assert payload["started_utc"]
    assert payload["outputs"] == []
    assert manifest.path == path


def test_run_manifest_hashes_outputs_and_records_completion_or_failure(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    path = tmp_path / "run_manifest.json"
    output = tmp_path / "table.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    manifest = RunManifest.start(
        path,
        config_name="development",
        config={"mode": "development"},
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )

    manifest.record_output(output, logical_name="tables/table.csv")
    manifest.complete()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["finished_utc"]
    assert payload["duration_seconds"] >= 0
    assert payload["outputs"] == [
        {
            "artifact": "tables/table.csv",
            "sha256": payload["outputs"][0]["sha256"],
            "size_bytes": len(output.read_bytes()),
        }
    ]

    failed = RunManifest.start(
        tmp_path / "failed.json",
        config_name="development",
        config={"mode": "development"},
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )
    failed.fail("synthetic failure")
    failed_payload = json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))
    assert failed_payload["status"] == "failed"
    assert failed_payload["failure"] == "synthetic failure"


def test_run_manifest_rejects_resume_with_incompatible_configuration_or_input(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input-v1")
    path = tmp_path / "run_manifest.json"
    RunManifest.start(
        path,
        config_name="development",
        config={"mode": "development"},
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )

    source.write_bytes(b"input-v2")
    with pytest.raises(ManifestCompatibilityError, match="input hashes"):
        RunManifest.resume(
            path,
            config={"mode": "development"},
            input_files={"manifest": source},
        )

    source.write_bytes(b"input-v1")
    with pytest.raises(ManifestCompatibilityError, match="configuration"):
        RunManifest.resume(
            path,
            config={"mode": "final"},
            input_files={"manifest": source},
        )


def test_run_manifest_redacts_absolute_input_and_output_paths_from_public_configuration(tmp_path):
    source = tmp_path / "private" / "input.bin"
    source.parent.mkdir()
    source.write_bytes(b"input")
    path = tmp_path / "run_manifest.json"

    RunManifest.start(
        path,
        config_name="development",
        config={
            "mode": "development",
            "manifest_path": str(source),
            "output_root": str(tmp_path / "private-output"),
            "report_destination": str(tmp_path / "private-reports"),
        },
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert payload["configuration"]["manifest_path"] == "<redacted-path:manifest_path>"
    assert payload["configuration"]["output_root"] == "<redacted-path:output_root>"
    assert payload["configuration"]["report_destination"] == "<redacted-path:report_destination>"


def test_run_manifest_is_created_before_input_hashing_can_fail(tmp_path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    path = tmp_path / "run_manifest.json"
    from research_audit_v2.second_phase.src import run_manifest as module

    def fail_hashing(_):
        raise OSError("synthetic hashing interruption")

    monkeypatch.setattr(module, "_input_metadata", fail_hashing)

    with pytest.raises(OSError, match="hashing interruption"):
        RunManifest.start(
            path,
            config_name="development",
            config={"mode": "development"},
            seeds=[1],
            input_files={"manifest": source},
            parameters={},
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "initializing"
    assert payload["inputs"] == {}


def test_public_manifest_schema_renames_ambiguous_nested_path_and_name_keys(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    path = tmp_path / "run_manifest.json"
    RunManifest.start(
        path,
        config_name="development",
        config={
            "mode": "development",
            "historical_evidence": {"5": {"path": str(source), "method": "csv_rows"}},
            "pca_64": {"name": "pca_64", "n_components": 64},
        },
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["configuration"]["historical_evidence"]["5"]["path_redacted"] == "<redacted-path:path>"
    assert payload["configuration"]["pca_64"]["method_name"] == "pca_64"
    assert scan_public_tree(tmp_path) == []


def test_resume_migrates_legacy_public_configuration_schema_after_hash_validation(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    path = tmp_path / "run_manifest.json"
    config = {
        "mode": "development",
        "historical_evidence": {"5": {"path": str(source), "method": "csv_rows"}},
        "pca_64": {"name": "pca_64", "n_components": 64},
    }
    RunManifest.start(
        path,
        config_name="development",
        config=config,
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["configuration"] = config
    path.write_text(json.dumps(legacy), encoding="utf-8")

    resumed = RunManifest.resume(path, config=config, input_files={"manifest": source})

    assert resumed.payload["configuration"]["historical_evidence"]["5"]["path_redacted"]
    assert resumed.payload["configuration"]["pca_64"]["method_name"] == "pca_64"
    assert scan_public_tree(tmp_path) == []


def test_resume_derives_missing_duration_from_preserved_start_and_finish_times(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    path = tmp_path / "run_manifest.json"
    config = {"mode": "development"}
    manifest = RunManifest.start(
        path,
        config_name="development",
        config=config,
        seeds=[1],
        input_files={"manifest": source},
        parameters={},
    )
    manifest.complete()
    legacy = json.loads(path.read_text(encoding="utf-8"))
    expected_duration = legacy.pop("duration_seconds")
    path.write_text(json.dumps(legacy), encoding="utf-8")

    resumed = RunManifest.resume(path, config=config, input_files={"manifest": source})

    assert resumed.payload["duration_seconds"] == expected_duration
