"""Reconcile observable pipeline stages and produce pseudonymous lineage."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import public_lineage, stable_id, write_csv

HISTORICAL_STAGES = [
    ("raw_collection", 11724, "Historical count supplied in study request; source record unavailable locally."),
    ("standardization_input", 9764, "Historical count supplied in study request; transition evidence unavailable locally."),
    ("standardization_success", 9546, "Historical count supplied in study request; transition evidence unavailable locally."),
]


def load_audited_records(config: dict[str, Any]) -> tuple[pd.DataFrame, object]:
    """Load successful embeddings and create non-reversible public row labels."""
    import numpy as np
    manifest = pd.read_csv(config["manifest_path"])
    vectors = np.load(config["embeddings_path"])
    required = {"embedding_index", "embedding_status", "quality"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Embedding manifest lacks required columns: {sorted(missing)}")
    selected = manifest.loc[(manifest["embedding_status"] == "ok") & manifest["embedding_index"].notna()].copy()
    selected["embedding_index"] = selected["embedding_index"].astype(int)
    selected = selected.loc[selected["embedding_index"].between(0, len(vectors) - 1)].copy()
    selected["record_id"] = [stable_id(f"embedding:{index}", config["public_id_salt"]) for index in selected["embedding_index"]]
    # The observed manifest has no documented source field. Do not infer one from names or paths.
    selected["source"] = selected["source"] if "source" in selected else "unresolved"
    selected["source"] = selected["source"].fillna("unresolved").astype(str)
    selected["quality"] = selected["quality"].fillna("unknown").astype(str)
    if selected["record_id"].duplicated().any():
        raise ValueError("Pseudonymous record identifiers are not unique.")
    return selected.sort_values("embedding_index").reset_index(drop=True), vectors[selected.sort_values("embedding_index")["embedding_index"].to_numpy()]


def reconcile(config: dict[str, Any], tables: Path, reports: Path, manifests: Path) -> pd.DataFrame:
    records, vectors = load_audited_records(config)
    manifest_rows = len(pd.read_csv(config["manifest_path"]))
    rows: list[dict[str, object]] = []
    sequence = [*HISTORICAL_STAGES, ("embedding_manifest", manifest_rows, "Documented by current embedding manifest."), ("valid_embeddings", len(vectors), "Documented by current embedding matrix and successful manifest rows.")]
    for (stage, incoming, reason), (_, outgoing, _) in zip(sequence, sequence[1:]):
        rows.append({"input_stage": stage, "output_stage": _[0], "input_count": incoming, "output_count": outgoing, "losses": max(incoming - outgoing, 0), "additions": max(outgoing - incoming, 0), "explained_records": 0, "unexplained_records": abs(outgoing - incoming), "documented_reason": reason, "evidence_file": "restricted input; see inventory.json", "transition_status": "documented" if stage in {"embedding_manifest"} else "unresolved"})
    table = pd.DataFrame(rows)
    write_csv(table, tables / "provenance_reconciliation.csv")
    unresolved = table.loc[table["transition_status"] != "documented", ["input_stage", "output_stage", "unexplained_records", "documented_reason"]]
    write_csv(unresolved, tables / "provenance_unresolved.csv")
    lineage = public_lineage(records)
    manifests.mkdir(parents=True, exist_ok=True)
    lineage.to_parquet(manifests / "record_lineage.parquet", index=False)
    reports.mkdir(parents=True, exist_ok=True)
    reports.joinpath("provenance_report.md").write_text(
        "# Provenance reconciliation\n\nOnly the final manifest-to-embedding transition is documented by local artifacts. Earlier supplied counts are retained as unresolved historical transitions; no explanation was inferred. Public lineage contains only pseudonymous identifiers and non-sensitive aggregate fields.\n",
        encoding="utf-8",
    )
    return records
