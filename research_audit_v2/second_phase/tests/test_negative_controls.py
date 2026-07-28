from research_audit_v2.second_phase.src.controls import negative_controls


def test_negative_controls_pass_on_generated_data(tmp_path):
    tables, reports = tmp_path / "tables", tmp_path / "reports"; tables.mkdir(); reports.mkdir()
    assert negative_controls(7, tables, reports)["pass"].all()
