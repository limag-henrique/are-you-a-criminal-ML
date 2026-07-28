"""Create public, aggregate second-phase evidence reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from research_audit_v2.src.common import write_csv


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def determinism(seed: int, tables: Path, reports: Path) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(600, 24)).astype("float32")
    models = [MiniBatchKMeans(8, random_state=seed, n_init=1, batch_size=128, max_iter=8).fit(values) for _ in range(2)]
    labels = [model.labels_ for model in models]
    equal = np.array_equal(labels[0], labels[1])
    write_csv(pd.DataFrame([{"comparison": "two_consecutive_cpu_runs_synthetic", "input_hash": _hash(values), "output_hash_run_1": _hash(labels[0]), "output_hash_run_2": _hash(labels[1]), "max_absolute_difference": int(np.max(np.abs(labels[0]-labels[1]))), "classification": "bit_for_bit_identical" if equal else "divergent", "threads": "environment default", "limitation": "CPU-only synthetic check; no GPU or clean environment tested"}]), tables / "determinism_comparison.csv")
    reports.joinpath("numerical_reproducibility_report.md").write_text("# Numerical reproducibility\n\nTwo fixed-seed consecutive CPU runs on a synthetic matrix were compared by output hash. This is not a claim of absolute reproducibility across BLAS implementations, CPUs, GPUs or environments, which were not available for comparison.\n", encoding="utf-8")


def failure_summary(tables: Path, reports: Path) -> None:
    manifest = pd.read_csv("artifacts/embedding_manifest.csv")
    summary = manifest.assign(success=manifest["embedding_status"].eq("ok")).groupby("quality", dropna=False).agg(inputs=("success", "size"), successful=("success", "sum")).reset_index()
    summary["failures"] = summary["inputs"] - summary["successful"]; summary["failure_proportion"] = summary["failures"] / summary["inputs"]
    summary["source"] = "unresolved"; write_csv(summary[["source", "quality", "inputs", "successful", "failures", "failure_proportion"]], tables / "failure_rates_by_source.csv")
    write_csv(pd.DataFrame([{"status": "not_estimated", "reason": "source is not documented; a source-based failure model would not be identified"}]), tables / "failure_predictability.csv")
    reports.joinpath("missingness_and_failure_report.md").write_text("# Missingness and technical failure\n\nFailure rates are descriptive by recorded quality only. Source-specific technical selectivity cannot be estimated because the embedding manifest contains no documented source field. This absence must not be replaced by inference from names or URLs.\n", encoding="utf-8")


def final_reports(out: dict[str, Path]) -> None:
    metrics = pd.read_csv(out["tables"] / "cross_fitted_metrics.csv")
    means = metrics[["roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall", "f1", "brier"]].mean().round(3).to_dict()
    write_csv(pd.DataFrame([
        {"claim_id": "R01", "conclusion": "Synthetic-label recovery remains measurable after grouped cross-fitting", "original_analysis": "all-record internal diagnostic", "additional_test": "five-fold grouped cross-fitting", "additional_result": json.dumps(means), "remains_valid": "only as internal synthetic-label recovery", "needs_moderation": True, "weakened": True, "strengthened": False, "not_testable": False, "justification": "Target remains reconstructed and no test observation fitted its centroid", "evidence_files": "cross_fitted_metrics.csv; leakage_audit.csv", "robustness": "dependent_on_configuration"},
        {"claim_id": "R02", "conclusion": "Source enrichment describes institutional selection", "original_analysis": "source enrichment", "additional_test": "source-variable audit", "additional_result": "source absent from embedding manifest", "remains_valid": False, "needs_moderation": True, "weakened": True, "strengthened": False, "not_testable": True, "justification": "No documented source assignment", "evidence_files": "failure_rates_by_source.csv", "robustness": "not_identifiable"},
        {"claim_id": "R03", "conclusion": "Historical clustering metrics reproduce", "original_analysis": "historical artifacts", "additional_test": "artifact inventory", "additional_result": "historic target rule/state absent", "remains_valid": False, "needs_moderation": True, "weakened": True, "strengthened": False, "not_testable": True, "justification": "No controlled historical state", "evidence_files": "article_result_reconciliation.csv", "robustness": "not_testable_with_available_data"},
    ]), out["tables"] / "claim_robustness_matrix.csv")
    out["reports"].joinpath("SECOND_PHASE_FINAL_REPORT.md").write_text(f"# Second-phase final report\n\n## Executive result\n\nThe grouped cross-fitted design completed with zero duplicate-group overlap. Foldwise mean metrics were {means}. They quantify recovery of a reconstructed synthetic target only. The dispersion across folds is material and should be reported rather than optimized away.\n\n## Leakage\n\nThe all-record diagnostic has high leakage risk because target construction and centroid scoring share observations. The grouped cross-fitted design removes this direct reuse: clusters, target selection and centroid are fit in training folds only.\n\n## Limitations\n\nNo historical clustering state, documented source attribution, reliable temporal field, supported historical extraction environment, GPU comparison or clean-environment comparison was available. No social, biometric, identity, criminal or causal inference is supported.\n", encoding="utf-8")
    out["reports"].joinpath("MANUSCRIPT_REQUIRED_CHANGES.md").write_text("# Required manuscript changes\n\n1. Label all centroid/cluster prediction metrics as internal recovery of a reconstructed synthetic label.\n2. Report cross-fitted fold dispersion and zero group overlap, not only a favorable aggregate.\n3. Remove or suspend source-enrichment claims until a documented source variable is supplied.\n4. State that historical ARI, NMI and target Jaccard could not be controlledly reproduced because the historical clustering state and rule were unavailable.\n5. Add the pipeline-loss reconciliation and the unsupported historical preprocessing environment as limitations.\n", encoding="utf-8")
    out["reports"].joinpath("EDITORIAL_IMPACT_SUMMARY.md").write_text("# Editorial impact summary\n\nCritical leakage in all-record internal scoring is now exposed and a grouped cross-fitted alternative is available. The principal remaining rejection risks are non-identifiability of source effects, lack of preserved historical analysis state, and the impossibility of external validity claims from a synthetic embedding-derived target.\n", encoding="utf-8")
    out["reports"].joinpath("source_influence_report.md").write_text("# Source influence\n\nNot executed: source assignment is not a documented field in the audit manifest. Leave-one-source-out, source balancing and source weighting would require inventing an assignment and are therefore not identified.\n", encoding="utf-8")
    out["reports"].joinpath("minibatch_sensitivity_report.md").write_text("# MiniBatch sensitivity\n\nDeferred until the first-phase 20-seed stability run completes under a resource-bounded development configuration. The locked cross-fitting analysis fixes batch size, n_init and max_iter; it does not select these post hoc.\n", encoding="utf-8")
    out["reports"].joinpath("embedding_representation_report.md").write_text("# Embedding representation sensitivity\n\nDeferred as a separate predeclared sensitivity. Learned representations (PCA, standardization, whitening) must be fitted only within each cross-fitting training fold. They must not replace the locked primary L2-normalized representation based on observed metrics.\n", encoding="utf-8")
