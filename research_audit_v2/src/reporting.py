"""Assemble public reports and claim traceability."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import write_csv


def write_final_reports(reports: Path, tables: Path) -> None:
    claims = pd.DataFrame([
        {"claim_id": "C01", "claim": "Final manifest and embedding counts reconcile locally", "variable": "rows", "unit": "record", "denominator": "embedding manifest", "method": "direct count", "result_file": "provenance_reconciliation.csv", "limitations": "Earlier stages unresolved", "classification": "diagnosticable"},
        {"claim_id": "C02", "claim": "Synthetic target can be recovered internally", "variable": "centroid score", "unit": "embedding", "denominator": "audited embeddings", "method": "internal label recovery", "result_file": "circularity_ablation.csv", "limitations": "Circular by construction; no external validity", "classification": "diagnosticable"},
        {"claim_id": "C03", "claim": "Institutional source enrichment", "variable": "source", "unit": "record", "denominator": "audited embeddings", "method": "descriptive enrichment", "result_file": "source_enrichment_detailed.csv", "limitations": "Source not documented in manifest", "classification": "not_identifiable"},
    ])
    write_csv(claims, tables / "results_claim_matrix.csv")
    reports.joinpath("full_research_audit.md").write_text("# Full research audit\n\n## Scope\n\nThis audit examines computational provenance, dependence and internal stability of a restricted dataset. It does not infer identity, race, criminality, lawfulness, biometric validity or social validity.\n\n## Principal limitations\n\nThe historic clustering rule and complete historical state were not preserved locally. The observed embedding manifest has no documented source field. All resampling statements are conditional to audited records.\n", encoding="utf-8")
    reports.joinpath("manuscript_update_notes.md").write_text("# Proposed manuscript updates\n\nSeparate original results from reconstructed analyses. Describe internal-label metrics as circularity diagnostics, state the unavailable historical clustering implementation, and remove any claim that public wanted lists measure criminality or that embedding patterns imply social groups.\n", encoding="utf-8")
