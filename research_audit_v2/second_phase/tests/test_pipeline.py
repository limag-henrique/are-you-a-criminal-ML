import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_audit_v2.second_phase.src.run_second_phase import PrivacyGateError, run_audit
from research_audit_v2.second_phase.src.privacy_scan import scan_public_tree


def _synthetic_config(tmp_path: Path, *, unsafe_output: bool = False) -> Path:
    rng = np.random.default_rng(101)
    vectors = np.vstack(
        [rng.normal(2, 0.2, size=(24, 8)), rng.normal(-2, 0.2, size=(16, 8))]
    ).astype("float32")
    embeddings = tmp_path / "restricted" / "embeddings.npy"
    manifest = tmp_path / "restricted" / "manifest.csv"
    embeddings.parent.mkdir()
    np.save(embeddings, vectors)
    pd.DataFrame(
        {
            "embedding_index": np.arange(len(vectors)),
            "embedding_status": ["ok"] * len(vectors),
            "quality": ["synthetic"] * len(vectors),
            "subject_id": [f"private-{index}" for index in range(len(vectors))],
            "path": [str(tmp_path / "private" / f"face-{index}.jpg") for index in range(len(vectors))],
        }
    ).to_csv(manifest, index=False)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    if unsafe_output:
        (output_root / "unsafe.md").write_text("private@example.org", encoding="utf-8")
    config = {
        "mode": "synthetic_development",
        "random_seed": 101,
        "seeds": [101, 102],
        "k_values": [2],
        "batch_sizes": [8, 16],
        "orderings": ["original", "reversed"],
        "representations": ["original_l2"],
        "primary_k": 2,
        "primary_batch_size": 8,
        "manifest_path": str(manifest),
        "embeddings_path": str(embeddings),
        "output_root": str(output_root),
        "report_destination": str(tmp_path / "deliverables"),
        "max_records": 40,
        "cluster_algorithm": "minibatch",
        "cluster_max_iter": 10,
        "cluster_batch_size": 8,
        "max_iter": 10,
        "batch_size": 8,
        "n_init": 2,
        "outer_folds": 2,
        "k": 2,
        "target_rule": "largest_cluster",
        "calibration": "logistic",
        "representation": "l2_normalized_float32",
        "dedup": {
            "method": "cosine_similarity",
            "embedding_thresholds": [0.999],
            "primary_threshold": 0.999,
        },
        "claimed_counts": [50, 45, 42, 40],
        "historical_evidence": {},
        "public_id_salt": "synthetic-public-salt",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_synthetic_pipeline_generates_all_required_outputs_and_private_data_never_escapes(tmp_path):
    config = _synthetic_config(tmp_path)

    result = run_audit(config)

    output = Path(result["output_root"])
    required = [
        "tables/data_lineage.csv",
        "tables/group_id_statistics.json",
        "tables/group_size_distribution.csv",
        "tables/threshold_review_sample.csv",
        "tables/cross_fitted_metrics.csv",
        "tables/stability_runs.csv",
        "tables/stability_summary.csv",
        "tables/stability_pairwise.csv",
        "tables/synthetic_geometry_control.csv",
        "manifests/run_manifest.json",
        "reports/privacy_scan.json",
        "reports/FINAL_REPRODUCTION_REPORT.md",
        "reports/MANUSCRIPT_UPDATE.md",
    ]
    assert all((output / relative).exists() for relative in required)
    manifest = json.loads((output / "manifests/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["outputs"]
    assert json.loads((output / "reports/privacy_scan.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert scan_public_tree(output) == []
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}
    )
    assert "private-" not in public_text
    assert str(tmp_path) not in public_text
    assert "face-" not in public_text
    assert (tmp_path / "deliverables" / "FINAL_REPRODUCTION_REPORT.md").exists()
    assert (tmp_path / "deliverables" / "MANUSCRIPT_UPDATE.md").exists()


def test_pipeline_fails_closed_and_never_marks_complete_when_privacy_scan_finds_a_leak(tmp_path):
    config = _synthetic_config(tmp_path, unsafe_output=True)

    with pytest.raises(PrivacyGateError):
        run_audit(config)

    manifest = json.loads(
        (tmp_path / "outputs" / "manifests" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["completion_status"] == "failed"


def test_pipeline_rescans_after_manifest_output_recording_before_accepting_final_state(tmp_path, monkeypatch):
    config = _synthetic_config(tmp_path)
    from research_audit_v2.second_phase.src import run_second_phase as runner

    original = runner._record_outputs

    def record_then_inject(manifest, output_root):
        original(manifest, output_root)
        (output_root / "late-leak.md").write_text("late@example.org", encoding="utf-8")

    monkeypatch.setattr(runner, "_record_outputs", record_then_inject)

    with pytest.raises(PrivacyGateError):
        run_audit(config)

    manifest = json.loads(
        (tmp_path / "outputs" / "manifests" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
