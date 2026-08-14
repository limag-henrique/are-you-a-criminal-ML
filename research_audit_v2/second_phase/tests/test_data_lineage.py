import numpy as np
import pandas as pd

from research_audit_v2.second_phase.src.data_lineage import audit_data_lineage


def test_lineage_verifies_current_counts_and_keeps_unsupported_history_unresolved(tmp_path):
    manifest = tmp_path / "manifest.csv"
    embeddings = tmp_path / "embeddings.npy"
    pd.DataFrame({"status": ["ok"] * 4}).to_csv(manifest, index=False)
    np.save(embeddings, np.ones((3, 2), dtype="float32"))

    result = audit_data_lineage(
        manifest,
        embeddings,
        claimed_counts=[5, 2, 6, 4, 3],
        historical_evidence={},
        disputed_pair=(6, 4),
    )

    by_count = result[result["claim_type"].eq("count")].set_index("claimed_count")
    assert by_count.loc[4, "classification"] == "reproduced"
    assert by_count.loc[4, "observed_count"] == 4
    assert by_count.loc[3, "classification"] == "reproduced"
    assert by_count.loc[3, "observed_count"] == 3
    assert by_count.loc[5, "classification"] == "information_not_recovered"
    assert by_count.loc[2, "classification"] == "information_not_recovered"
    assert by_count.loc[6, "classification"] == "information_not_recovered"
    assert not {"input_stage", "output_stage", "losses"}.intersection(result.columns)

    disputed = result[result["claim_type"].eq("historical_difference")].iloc[0]
    assert disputed["classification"] == "information_not_recovered"
    assert disputed["status"] == "unresolved_historical_gap"


def test_historical_count_is_preserved_only_with_explicit_artifact_evidence(tmp_path):
    manifest = tmp_path / "manifest.csv"
    embeddings = tmp_path / "embeddings.npy"
    evidence = tmp_path / "historical_report.txt"
    pd.DataFrame({"status": ["ok"]}).to_csv(manifest, index=False)
    np.save(embeddings, np.ones((1, 2), dtype="float32"))
    evidence.write_text("documented aggregate: 5", encoding="utf-8")

    result = audit_data_lineage(
        manifest,
        embeddings,
        claimed_counts=[5, 1],
        historical_evidence={5: evidence},
        disputed_pair=None,
    )

    preserved = result[result["claimed_count"].eq(5)].iloc[0]
    assert preserved["classification"] == "historical_preserved"
    assert preserved["artifact_sha256"]
    assert preserved["artifact"] == "historical_report.txt"


def test_historical_csv_counts_are_derived_by_declared_row_and_value_count_methods(tmp_path):
    manifest = tmp_path / "manifest.csv"
    embeddings = tmp_path / "embeddings.npy"
    preprocess = tmp_path / "preprocess.csv"
    processing = tmp_path / "processing.csv"
    pd.DataFrame({"status": ["ok", "ok"]}).to_csv(manifest, index=False)
    np.save(embeddings, np.ones((1, 2), dtype="float32"))
    pd.DataFrame({"decision": ["a", "b", "c", "d", "e"]}).to_csv(preprocess, index=False)
    pd.DataFrame({"status": ["success", "success", "success", "rejected"]}).to_csv(
        processing, index=False
    )

    result = audit_data_lineage(
        manifest,
        embeddings,
        claimed_counts=[5, 4, 3, 2, 1],
        historical_evidence={
            5: {"path": preprocess, "method": "csv_rows"},
            4: {"path": processing, "method": "csv_rows"},
            3: {
                "path": processing,
                "method": "csv_value_count",
                "column": "status",
                "value": "success",
            },
        },
        disputed_pair=(3, 2),
    )

    by_count = result[result["claim_type"].eq("count")].set_index("claimed_count")
    assert by_count.loc[5, "classification"] == "historical_preserved"
    assert by_count.loc[5, "observed_count"] == 5
    assert by_count.loc[5, "evidence_basis"] == "csv_rows"
    assert by_count.loc[4, "observed_count"] == 4
    assert by_count.loc[3, "observed_count"] == 3
    assert by_count.loc[3, "evidence_basis"] == "csv_value_count:status=success"
