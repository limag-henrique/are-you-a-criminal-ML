from pathlib import Path

from research_audit_v2.src.privacy import scan_public_outputs


def test_privacy_scanner_rejects_url(tmp_path: Path):
    (tmp_path / "report.md").write_text("https://restricted.example/person", encoding="utf-8")
    assert scan_public_outputs(tmp_path)


def test_privacy_scanner_accepts_pseudonymous_aggregate(tmp_path: Path):
    (tmp_path / "table.csv").write_text("record_id,count\nrec_abcd,3\n", encoding="utf-8")
    assert not scan_public_outputs(tmp_path)
