from research_audit_v2.second_phase.src.privacy_scan import scan_public_tree


def test_scanner_rejects_sensitive_column_and_image_reference(tmp_path):
    (tmp_path / "bad.csv").write_text("subject_id,value\nx,1\n", encoding="utf-8")
    (tmp_path / "image.md").write_text("photo.jpg", encoding="utf-8")
    assert scan_public_tree(tmp_path)
