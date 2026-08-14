import json

import pandas as pd
import pytest

from research_audit_v2.second_phase.src.io import (
    atomic_target,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)


def test_atomic_target_preserves_previous_file_and_cleans_temporary_on_failure(tmp_path):
    target = tmp_path / "result.txt"
    target.write_text("previous", encoding="utf-8")

    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_target(target) as temporary:
            temporary.write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted")

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(".result.txt.*.tmp")) == []


def test_atomic_writers_replace_outputs_without_leaving_temporary_files(tmp_path):
    text_path = tmp_path / "note.md"
    json_path = tmp_path / "manifest.json"
    csv_path = tmp_path / "table.csv"

    atomic_write_text(text_path, "ok\n")
    atomic_write_json(json_path, {"status": "complete", "value": 2})
    atomic_write_csv(csv_path, pd.DataFrame([{"b": 2, "a": 1}]))

    assert text_path.read_text(encoding="utf-8") == "ok\n"
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert pd.read_csv(csv_path).to_dict("records") == [{"b": 2, "a": 1}]
    assert list(tmp_path.glob(".*.tmp")) == []
