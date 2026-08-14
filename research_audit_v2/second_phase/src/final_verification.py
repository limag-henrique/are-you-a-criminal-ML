"""Post-run verification that records tests without inventing their outcome."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from research_audit_v2.src.common import sha256_file

from .io import atomic_write_json
from .privacy_scan import scan_public_tree, write_privacy_report
from .report_generator import generate_public_reports
from .run_manifest import RunManifest


TEST_COMMAND = "python -m pytest -q research_audit_v2"
PYTEST_SUMMARY = re.compile(r"\b\d+ passed\b.*\bin \d+(?:\.\d+)?s\b")


def _summary(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        match = PYTEST_SUMMARY.search(line.strip())
        if match:
            return match.group(0)
    return None


def run_repository_tests(output_root: str | Path) -> dict[str, Any]:
    """Execute the declared suite and atomically persist its real exit status."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "research_audit_v2"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    summary = _summary(combined)
    if completed.returncode == 0 and summary is None:
        raise RuntimeError("Successful test command did not emit a recognizable pytest summary.")
    payload: dict[str, Any] = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": TEST_COMMAND,
        "exit_code": int(completed.returncode),
        "summary": summary or "pytest did not report a passing summary",
    }
    destination = Path(output_root) / "reports" / "final_verification.json"
    atomic_write_json(destination, payload)
    return payload


def _record_outputs(manifest: RunManifest, output_root: Path) -> None:
    for artifact in sorted(output_root.rglob("*")):
        if artifact.is_file() and artifact != manifest.path:
            manifest.record_output(
                artifact,
                logical_name=artifact.relative_to(output_root).as_posix(),
            )


def _assert_recorded_hashes(manifest: RunManifest, output_root: Path) -> None:
    mismatches = [
        item["artifact"]
        for item in manifest.payload["outputs"]
        if sha256_file(output_root / item["artifact"]) != item["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"Output hash verification failed for {len(mismatches)} artifact(s).")


def finalize_completed_run(config_path: str | Path) -> dict[str, Any]:
    """Run tests, refresh reports, hashes and privacy gates for a completed run."""
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    manifest_path = output_root / "manifests" / "run_manifest.json"
    input_files = {
        "manifest": Path(config["manifest_path"]),
        "embedding_matrix": Path(config["embeddings_path"]),
        "configuration": config_file,
    }
    manifest = RunManifest.resume(manifest_path, config=config, input_files=input_files)
    if manifest.payload.get("status") != "complete":
        raise RuntimeError("Final verification requires a completed run manifest.")

    verification = run_repository_tests(output_root)
    if verification["status"] != "passed":
        raise RuntimeError("Repository test suite failed; final verification was not accepted.")
    generate_public_reports(output_root, config, config["report_destination"])
    privacy_path = output_root / "reports" / "privacy_scan.json"
    findings = write_privacy_report(output_root, privacy_path)
    if findings:
        raise RuntimeError("Post-run privacy verification failed.")
    _record_outputs(manifest, output_root)
    if scan_public_tree(output_root):
        raise RuntimeError("Privacy verification failed after manifest output recording.")
    _assert_recorded_hashes(manifest, output_root)
    return {
        "status": "verified",
        "test_summary": verification["summary"],
        "artifact_count": len(manifest.payload["outputs"]),
        "privacy_findings": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(finalize_completed_run(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
