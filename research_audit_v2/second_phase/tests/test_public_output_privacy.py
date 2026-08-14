import json

import pytest

from research_audit_v2.second_phase.src.privacy_scan import scan_public_tree, write_privacy_report


def test_scanner_rejects_sensitive_column_and_image_reference(tmp_path):
    (tmp_path / "bad.csv").write_text("subject_id,value\nx,1\n", encoding="utf-8")
    (tmp_path / "image.md").write_text("photo.jpg", encoding="utf-8")
    assert scan_public_tree(tmp_path)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("url.md", "https://restricted.example/record"),
        ("windows.md", r"C:\\Users\\person\\record.csv"),
        ("unix.md", "/home/person/record.csv"),
        ("email.md", "person@example.org"),
        ("vector.csv", "Embedding 0,value\n0.123,1\n"),
        ("names.csv", " Full Name ,value\nprivate,1\n"),
        ("json.json", json.dumps({"embedding": [0.1, 0.2]})),
    ],
)
def test_scanner_rejects_sensitive_text_and_normalized_columns(tmp_path, filename, content):
    (tmp_path / filename).write_text(content, encoding="utf-8")
    assert scan_public_tree(tmp_path)


@pytest.mark.parametrize("filename", ["face.jpg", "vectors.npy", "records.parquet"])
def test_scanner_rejects_prohibited_binary_public_artifacts(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"synthetic")
    assert scan_public_tree(tmp_path)


def test_scanner_allows_self_contained_svg_namespace_but_rejects_linked_raster(tmp_path):
    safe = tmp_path / "safe.svg"
    safe.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle r="2"/></svg>', encoding="utf-8")
    assert scan_public_tree(tmp_path) == []

    safe.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><image href="face.jpg"/></svg>',
        encoding="utf-8",
    )
    assert scan_public_tree(tmp_path)


def test_privacy_report_is_structured_and_contains_no_scanned_content(tmp_path):
    (tmp_path / "bad.csv").write_text("subject id,value\nprivate,1\n", encoding="utf-8")
    report = tmp_path / "privacy_scan.json"

    findings = write_privacy_report(tmp_path, report)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert findings
    assert payload["status"] == "failed"
    assert payload["finding_count"] == len(findings)
    assert payload["findings"][0]["artifact"] == "bad.csv"
    assert "private" not in report.read_text(encoding="utf-8")


def test_scanner_allows_aggregate_embedding_parameters_but_not_vector_components(tmp_path):
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps({"configuration": {"embedding_thresholds": [0.99, 0.999]}}),
        encoding="utf-8",
    )
    assert scan_public_tree(tmp_path) == []

    aggregate.write_text(json.dumps({"embedding_0": 0.25}), encoding="utf-8")
    assert scan_public_tree(tmp_path)
