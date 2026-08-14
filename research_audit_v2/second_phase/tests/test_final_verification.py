import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_repository_test_verification_records_the_executed_command_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from research_audit_v2.second_phase.src import final_verification

    completed = SimpleNamespace(
        returncode=0,
        stdout="................................................................ 73 passed in 21.06s\n",
        stderr="",
    )
    monkeypatch.setattr(final_verification.subprocess, "run", lambda *args, **kwargs: completed)

    payload = final_verification.run_repository_tests(tmp_path)

    assert payload["status"] == "passed"
    assert payload["command"] == "python -m pytest -q research_audit_v2"
    assert payload["exit_code"] == 0
    assert payload["summary"] == "73 passed in 21.06s"
    assert json.loads(
        (tmp_path / "reports" / "final_verification.json").read_text(encoding="utf-8")
    ) == payload


def test_repository_test_verification_fails_if_success_output_has_no_pytest_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from research_audit_v2.second_phase.src import final_verification

    completed = SimpleNamespace(returncode=0, stdout="no test summary\n", stderr="")
    monkeypatch.setattr(final_verification.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="pytest summary"):
        final_verification.run_repository_tests(tmp_path)


def test_report_verification_text_uses_only_a_structured_passed_record(tmp_path: Path):
    from research_audit_v2.second_phase.src.report_generator import _test_verification_text

    reports = tmp_path / "reports"
    reports.mkdir()
    record = {
        "status": "passed",
        "command": "python -m pytest -q research_audit_v2",
        "exit_code": 0,
        "summary": "73 passed in 21.06s",
    }
    (reports / "final_verification.json").write_text(json.dumps(record), encoding="utf-8")

    text = _test_verification_text(tmp_path)

    assert "73 passed in 21.06s" in text
    assert "python -m pytest -q research_audit_v2" in text


def test_report_verification_text_does_not_claim_unverified_success(tmp_path: Path):
    from research_audit_v2.second_phase.src.report_generator import _test_verification_text

    assert "not yet recorded" in _test_verification_text(tmp_path)
