"""Generate manuscript-facing reports exclusively from structured run outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import atomic_write_text


def _test_verification_text(output_root: str | Path) -> str:
    verification_path = Path(output_root) / "reports" / "final_verification.json"
    if not verification_path.exists():
        return (
            "Final test verification is not yet recorded. Run "
            "`python -m pytest -q research_audit_v2` before accepting the report."
        )
    try:
        record = json.loads(verification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Final test verification is not yet recorded in a valid structured artifact."
    if record.get("status") != "passed" or record.get("exit_code") != 0:
        return "Final test verification is not yet recorded as passed."
    return (
        f"Executed `{record.get('command', 'unrecorded command')}`: "
        f"**{record.get('summary', 'passing summary unavailable')}**. "
        "Evidence: `reports/final_verification.json`."
    )


def _metric_means(metrics: pd.DataFrame) -> dict[str, float]:
    columns = [
        "roc_auc",
        "pr_auc",
        "pr_auc_baseline",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
    ]
    return {
        column: round(float(metrics[column].mean()), 6)
        for column in columns
        if column in metrics and metrics[column].notna().any()
    }


def _lineage_items(lineage: pd.DataFrame, classification: str) -> list[str]:
    selected = lineage[
        lineage["claim_type"].eq("count") & lineage["classification"].eq(classification)
    ]
    return [
        f"- {int(row.claimed_count):,}: {row.status}; evidence `tables/data_lineage.csv`."
        for row in selected.itertuples()
    ]


def generate_public_reports(
    output_root: str | Path,
    config: Mapping[str, Any],
    destination_root: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_root)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lineage = pd.read_csv(output / "tables" / "data_lineage.csv")
    group_stats = json.loads((output / "tables" / "group_id_statistics.json").read_text(encoding="utf-8"))
    cross_metrics = pd.read_csv(output / "tables" / "cross_fitted_metrics.csv")
    stability_runs = pd.read_csv(output / "tables" / "stability_runs.csv")
    synthetic = pd.read_csv(output / "tables" / "synthetic_geometry_control.csv")
    means = _metric_means(cross_metrics)
    mode = str(config["mode"])
    max_records = config.get("max_records")
    scope = (
        f"resource-bounded integration run over at most {int(max_records)} records"
        if max_records is not None
        else "complete configured scientific run"
    )
    reproduced = _lineage_items(lineage, "reproduced")
    preserved = _lineage_items(lineage, "historical_preserved")
    unresolved = _lineage_items(lineage, "information_not_recovered")
    metrics_text = json.dumps(means, sort_keys=True)
    test_verification = _test_verification_text(output)
    manifest_path = output / "manifests" / "run_manifest.json"
    execution_status = "not_recorded"
    if manifest_path.exists():
        try:
            execution_status = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("status", "not_recorded"))
        except (OSError, json.JSONDecodeError):
            execution_status = "invalid_manifest"

    final_report = f"""# Final reproduction report

## Execution status

- Configuration: `{mode}`.
- Scope: {scope}.
- Run-manifest status at report generation: `{execution_status}`.
- Public-output privacy gate: required before the run manifest can be marked complete.

## Changes implemented

- Strict aligned manifest/embedding contracts, atomic writers, incremental hashed run manifest and safe resume checks.
- Grouped cross-fitting with train-only transformation, clustering, target selection, centroid, calibration and threshold.
- Separate stochastic, operational and representation stability analyses with pairwise ARI/Jaccard.
- Fully specified PCA-64 reconstruction, probable-duplicate group audit, evidence-classified count lineage and synthetic methodological control.
- Fail-closed public-output scanner and synthetic-only CI.

## Tests

{test_verification}

## Experiments effectively executed

- Grouped cross-fitting: {len(cross_metrics)} folds; mean metrics `{metrics_text}`. Evidence: `tables/cross_fitted_metrics.csv`.
- Stability: {len(stability_runs)} runs across `{', '.join(sorted(stability_runs['instability_type'].unique()))}`. Evidence: `tables/stability_runs.csv`, `tables/stability_summary.csv`, `tables/stability_pairwise.csv`.
- Synthetic geometry control: {int(synthetic['pass'].sum())}/{len(synthetic)} demonstrations passed. Evidence: `tables/synthetic_geometry_control.csv`.
- Probable-duplicate grouping: {group_stats['groups']} groups, {group_stats['non_singleton_groups']} non-singleton groups and grouped-record proportion {group_stats['grouped_record_proportion']:.6f}. Evidence: `tables/group_id_statistics.json`.

## Historical results preserved

{chr(10).join(preserved) if preserved else '- None supported only as a preserved historical artifact in this run.'}

## Results reproduced

{chr(10).join(reproduced) if reproduced else '- None of the claimed historical counts matched a current verifiable artifact in this run.'}

## New methodological reconstructions

- `group_id` is a probable-duplicate grouping based on declared cosine similarity, never confirmed identity.
- Cluster target, grouped cross-fitting, PCA-64, thresholds/calibration and stability analyses are new reconstructions, not attributed to the historical pipeline.
- All predictive metrics are internal recovery of a synthetic target only.

## Historical information not recovered

{chr(10).join(unresolved) if unresolved else '- No unresolved count claim.'}
- The relationship between 9,546 and 9,584 remains an unresolved historical gap unless an explicit documentary link is supplied. Evidence: `tables/data_lineage.csv`.

## Remaining limitations

- No result measures criminality, guilt, race/color, identity or biometric equity.
- Source, dates and historical clustering rules remain unavailable where not explicitly documented.
- Development outputs are integration/reproducibility evidence and must not be used as final scientific estimates.

## Exact reproduction commands

```powershell
python -m pytest -q research_audit_v2
python -m research_audit_v2.second_phase.src.run_second_phase --config research_audit_v2/configs/development.yaml
python -m research_audit_v2.second_phase.src.run_second_phase --config research_audit_v2/configs/final.yaml --resume
```
"""

    manuscript = f"""# Manuscript update

Only facts generated by the `{mode}` execution are listed. Metrics describe internal recovery of a reconstructed synthetic target only.

## Numbers that remain supportable

{chr(10).join([*preserved, *reproduced]) if preserved or reproduced else '- No claimed historical count was supported in this run.'}

## Numbers to replace

- Do not replace a historical number merely to match this reconstruction. Development results are not final scientific estimates. Evidence: `manifests/run_manifest.json`.

## New results

- Grouped cross-fitting mean metrics: `{metrics_text}`. Evidence: `tables/cross_fitted_metrics.csv`.
- Group audit: {group_stats['groups']} probable-duplicate groups; grouped-record proportion {group_stats['grouped_record_proportion']:.6f}. Evidence: `tables/group_id_statistics.json`.
- Stability distributions and non-reference pairwise comparisons are in `tables/stability_summary.csv` and `tables/stability_pairwise.csv`.
- PCA-64 parameters and explained variance, when executed, are in `tables/pca_specification.json`.
- The synthetic control is a methodological demonstration only. Evidence: `tables/synthetic_geometry_control.csv`.

## Claims that must be weakened

- Any predictive metric must be described as internal recovery of a clustering-derived synthetic target.
- `group_id` must be described only as probable duplication, never identity.
- Remove or suspend claims requiring undocumented source, date, race/color, identity or historical rules.

## Claims that can be strengthened

- The revised computation prevents direct train/test leakage by grouped splitting and train-only fitting. Evidence: `tables/fit_audit_events.csv`.
- Stability conclusions can use pairwise ARI/Jaccard rather than one arbitrary reference. Evidence: `tables/stability_pairwise.csv`.

## Gaps that remain non-identifiable

{chr(10).join(unresolved) if unresolved else '- No unresolved count claim.'}
- The documentary relationship between 9,546 and 9,584 is not identified. Evidence: `tables/data_lineage.csv`.
"""
    internal_final = reports / "FINAL_REPRODUCTION_REPORT.md"
    internal_manuscript = reports / "MANUSCRIPT_UPDATE.md"
    atomic_write_text(internal_final, final_report)
    atomic_write_text(internal_manuscript, manuscript)
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    external_final = destination / internal_final.name
    external_manuscript = destination / internal_manuscript.name
    atomic_write_text(external_final, final_report)
    atomic_write_text(external_manuscript, manuscript)
    return external_final, external_manuscript
