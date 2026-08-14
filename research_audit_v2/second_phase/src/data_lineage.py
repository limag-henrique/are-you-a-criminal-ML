"""Evidence-classified count lineage without assumed sequential transitions."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research_audit_v2.src.common import sha256_file


DEFAULT_CLAIMED_COUNTS = [11724, 9764, 9546, 9584, 9482]


def _artifact_supports_count(path: Path, count: int) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return bool(re.search(rf"(?<!\d){count}(?!\d)", content.replace(".", "")))


def _evaluate_historical_evidence(specification: Any, claimed: int) -> tuple[Path, int | None, str]:
    if isinstance(specification, Mapping):
        artifact = Path(specification["path"])
        method = str(specification["method"])
    else:
        artifact = Path(specification)
        method = "text_count"
        specification = {}
    if not artifact.is_file():
        return artifact, None, method
    if method == "text_count":
        return (
            artifact,
            claimed if _artifact_supports_count(artifact, claimed) else None,
            "explicit_count_in_preserved_artifact",
        )
    if method == "csv_rows":
        return artifact, int(len(pd.read_csv(artifact))), method
    if method == "csv_value_count":
        column = str(specification["column"])
        value = str(specification["value"])
        frame = pd.read_csv(artifact, usecols=[column])
        observed = int(frame[column].astype(str).eq(value).sum())
        return artifact, observed, f"csv_value_count:{column}={value}"
    raise ValueError(f"Unknown historical evidence method: {method}")


def audit_data_lineage(
    manifest_path: str | Path,
    embeddings_path: str | Path,
    *,
    claimed_counts: Sequence[int] = DEFAULT_CLAIMED_COUNTS,
    historical_evidence: Mapping[int, Any] | None = None,
    disputed_pair: tuple[int, int] | None = (9546, 9584),
) -> pd.DataFrame:
    """Classify each claimed aggregate independently from all other claims."""
    manifest = Path(manifest_path)
    embeddings = Path(embeddings_path)
    observed = {
        int(len(pd.read_csv(manifest))): ("current_embedding_manifest", manifest),
        int(np.load(embeddings, mmap_mode="r").shape[0]): ("current_embedding_matrix", embeddings),
    }
    evidence = {int(count): specification for count, specification in (historical_evidence or {}).items()}
    rows: list[dict[str, object]] = []
    for claimed in claimed_counts:
        claimed = int(claimed)
        if claimed in observed:
            label, artifact = observed[claimed]
            rows.append(
                {
                    "claim_type": "count",
                    "claimed_count": claimed,
                    "observed_count": claimed,
                    "classification": "reproduced",
                    "status": "verified_current_artifact",
                    "artifact": artifact.name,
                    "artifact_sha256": sha256_file(artifact),
                    "evidence_basis": label,
                }
            )
        elif claimed in evidence:
            artifact, evidence_count, basis = _evaluate_historical_evidence(evidence[claimed], claimed)
            if evidence_count == claimed:
                rows.append(
                    {
                        "claim_type": "count",
                        "claimed_count": claimed,
                        "observed_count": evidence_count,
                        "classification": "historical_preserved",
                        "status": "documented_historical_artifact",
                        "artifact": artifact.name,
                        "artifact_sha256": sha256_file(artifact),
                        "evidence_basis": basis,
                    }
                )
            else:
                rows.append(
                    {
                        "claim_type": "count",
                        "claimed_count": claimed,
                        "observed_count": evidence_count if evidence_count is not None else np.nan,
                        "classification": "information_not_recovered",
                        "status": "supporting_artifact_does_not_verify_claim",
                        "artifact": artifact.name,
                        "artifact_sha256": sha256_file(artifact) if artifact.is_file() else "not_available",
                        "evidence_basis": basis,
                    }
                )
        else:
            rows.append(
                {
                    "claim_type": "count",
                    "claimed_count": claimed,
                    "observed_count": np.nan,
                    "classification": "information_not_recovered",
                    "status": "supporting_artifact_not_identified",
                    "artifact": "not_identified",
                    "artifact_sha256": "not_available",
                    "evidence_basis": "no_explicit_local_artifact",
                }
            )
    if disputed_pair is not None:
        left, right = map(int, disputed_pair)
        rows.append(
            {
                "claim_type": "historical_difference",
                "claimed_count": np.nan,
                "observed_count": np.nan,
                "classification": "information_not_recovered",
                "status": "unresolved_historical_gap",
                "artifact": "not_identified",
                "artifact_sha256": "not_available",
                "evidence_basis": f"no_documented_relationship_between_{left}_and_{right}",
            }
        )
    return pd.DataFrame(rows)
