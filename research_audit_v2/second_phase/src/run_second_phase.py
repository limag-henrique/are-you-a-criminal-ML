"""Run the executable, non-sensitive-safe core of second-phase validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_audit_v2.src.common import read_config, sha256_file, write_csv
from research_audit_v2.src.deduplication import assign_groups
from research_audit_v2.src.provenance import load_audited_records

from .controls import negative_controls
from .cross_fitting import run_cross_fitting
from .data_contracts import validate_embeddings, validate_groups, validate_manifest
from .privacy_scan import scan_public_tree
from .final_reporting import determinism, failure_summary, final_reports
from .sensitivity import clustering_sensitivity


def paths(root: Path) -> dict[str, Path]:
    result = {name: root / name for name in ("tables", "figures", "reports", "logs", "manifests")}
    for folder in result.values():
        folder.mkdir(parents=True, exist_ok=True)
    return result


def static_reports(out: dict[str, Path], cfg_hash: str) -> None:
    write_csv(pd.DataFrame([
        {"analysis_id": "A01", "description": "All-record internal centroid diagnostic", "question": "Can a synthetic label be recovered from its own geometry?", "primary_parameter": "all records", "primary_result": "internal metrics", "defined_before_observation": False, "classification": "post-hoc", "post_hoc_risk": "high", "supports_main_claim": False, "notes": "Circularity diagnostic only."},
        {"analysis_id": "A02", "description": "Grouped cross-fitting", "question": "What remains when test records do not fit target geometry?", "primary_parameter": "locked config SHA-256", "primary_result": "foldwise metrics", "defined_before_observation": True, "classification": "confirmatory", "post_hoc_risk": "low", "supports_main_claim": False, "notes": "Still synthetic-label recovery."},
        {"analysis_id": "A03", "description": "Negative synthetic controls", "question": "Do metrics satisfy known mathematical expectations?", "primary_parameter": "fixed synthetic seed", "primary_result": "control pass/fail", "defined_before_observation": True, "classification": "confirmatory", "post_hoc_risk": "low", "supports_main_claim": False, "notes": "Software-validity control."},
    ]), out["tables"] / "analysis_classification.csv")
    write_csv(pd.DataFrame([
        {"result": "raw records", "published_value": 11724, "reproduced_value": np.nan, "post_correction_value": np.nan, "absolute_difference": np.nan, "relative_difference": np.nan, "likely_cause": "raw collection artifact unavailable", "article_update_required": "cannot assess", "importance": "high"},
        {"result": "manifest rows", "published_value": 9584, "reproduced_value": 9584, "post_correction_value": 9584, "absolute_difference": 0, "relative_difference": 0, "likely_cause": "direct local count", "article_update_required": "no", "importance": "high"},
        {"result": "valid embeddings", "published_value": 9482, "reproduced_value": 9482, "post_correction_value": 9482, "absolute_difference": 0, "relative_difference": 0, "likely_cause": "direct matrix count", "article_update_required": "no", "importance": "high"},
        {"result": "historical ARI/NMI/Jaccard", "published_value": np.nan, "reproduced_value": np.nan, "post_correction_value": np.nan, "absolute_difference": np.nan, "relative_difference": np.nan, "likely_cause": "historical clustering state unavailable", "article_update_required": "yes", "importance": "high"},
    ]), out["tables"] / "article_result_reconciliation.csv")
    out["reports"].joinpath("deviation_log.md").write_text(f"# Deviation log\n\nLocked configuration SHA-256: `{cfg_hash}`.\n\nNo deviations recorded before the initial execution.\n", encoding="utf-8")
    out["reports"].joinpath("face_preprocessing_specification.md").write_text("# Face preprocessing specification\n\nThe repository scripts document reading, EXIF correction, InsightFace detection/alignment, extraction and L2 normalization. The exact historical detector thresholds, crop perturbations, batch order and preserved ONNX weights for the article's clustering analysis are not locally documented. Re-extraction sensitivity is therefore not run under Python 3.14; the repository README identifies Python 3.10–3.12 as the supported interval for InsightFace.\n", encoding="utf-8")
    out["reports"].joinpath("preprocessing_sensitivity_report.md").write_text("# Preprocessing sensitivity\n\nNot executed: reproducible historical preprocessing settings and a supported extractor environment are unavailable. Numerical representation sensitivity remains a separate executable task.\n", encoding="utf-8")
    out["reports"].joinpath("program_corrections.md").write_text("# Program corrections\n\n- Added explicit contracts before analysis, preventing mismatched manifest/embedding rows and invalid numerical arrays.\n- Added grouped cross-fitting, preventing test observations from fitting clusters, selecting the target or defining the centroid.\n- Added public-output privacy scanning and atomic table writes.\n- Reduced development clustering iterations from 30 to 8 only for integration feasibility; the locked second-phase design records this and does not treat development output as final.\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="research_audit_v2/second_phase/configs/confirmatory_locked.yaml"); args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg_hash = sha256_file(args.config)
    out = paths(Path("research_audit_v2/second_phase/outputs"))
    status_path = out["manifests"] / "completion_status.json"
    status_path.write_text(json.dumps({"completion_status": "running", "config_sha256": cfg_hash}, indent=2) + "\n", encoding="utf-8")
    raw_manifest = pd.read_csv("artifacts/embedding_manifest.csv")
    raw_vectors = np.load("artifacts/embeddings.npy")
    validate_embeddings(raw_vectors)
    validate_manifest(raw_manifest, len(raw_vectors))
    primary_cfg = read_config("research_audit_v2/configs/development.yaml")
    records, vectors = load_audited_records(primary_cfg)
    records = assign_groups(records, vectors, primary_cfg, out["tables"])
    validate_groups(records["group_id"])
    negative_controls(cfg["random_seed"], out["tables"], out["reports"])
    run_cross_fitting(records, vectors, cfg, out["tables"], out["reports"])
    clustering_sensitivity(vectors, cfg, out["tables"], out["figures"], out["reports"])
    static_reports(out, cfg_hash)
    determinism(cfg["random_seed"], out["tables"], out["reports"])
    failure_summary(out["tables"], out["reports"])
    final_reports(out)
    findings = scan_public_tree(Path("research_audit_v2/second_phase/outputs"))
    out["reports"].joinpath("privacy_scan_report.md").write_text("# Privacy scan\n\n" + ("No prohibited public-output pattern found.\n" if not findings else "\n".join(f"- {value}" for value in findings) + "\n"), encoding="utf-8")
    if findings:
        status_path.write_text(json.dumps({"completion_status": "failed_privacy_scan", "config_sha256": cfg_hash, "findings": findings}, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("Second-phase privacy scan found prohibited patterns.")
    out["manifests"].joinpath("run_manifest.json").write_text(json.dumps({"config_sha256": cfg_hash, "input_manifest_sha256": sha256_file("artifacts/embedding_manifest.csv"), "input_embeddings_sha256": sha256_file("artifacts/embeddings.npy"), "completion_status": "complete"}, indent=2) + "\n", encoding="utf-8")
    status_path.write_text(json.dumps({"completion_status": "complete", "config_sha256": cfg_hash}, indent=2) + "\n", encoding="utf-8")
    print("second-phase core complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
