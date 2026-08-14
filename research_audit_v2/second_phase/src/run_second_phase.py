"""Run the tested, privacy-preserving scientific audit pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_audit_v2.src.common import stable_id, write_csv
from research_audit_v2.src.deduplication import assign_groups

from .controls import synthetic_geometry_control
from .cross_fitting import run_cross_fitting
from .data_contracts import validate_audit_inputs
from .data_lineage import DEFAULT_CLAIMED_COUNTS, audit_data_lineage
from .group_audit import safe_threshold_review_sample, summarize_probable_duplicate_groups
from .io import atomic_write_json
from .privacy_scan import scan_public_tree, write_privacy_report
from .report_generator import generate_public_reports
from .run_manifest import RunManifest
from .stability import run_stability_analysis


class PrivacyGateError(RuntimeError):
    """Raised when a public artifact fails the disclosure scanner."""


def _folders(root: Path) -> dict[str, Path]:
    result = {name: root / name for name in ("tables", "figures", "reports", "logs", "manifests")}
    for folder in result.values():
        folder.mkdir(parents=True, exist_ok=True)
    return result


def _public_records(manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    records = manifest[["embedding_index", "quality"]].copy()
    records["record_id"] = [
        stable_id(f"embedding:{int(index)}", config["public_id_salt"])
        for index in records["embedding_index"]
    ]
    records["source"] = "unresolved"
    records["quality"] = records["quality"].fillna("unknown").astype(str)
    return records[["record_id", "source", "quality", "embedding_index"]]


def _record_outputs(manifest: RunManifest, output_root: Path) -> None:
    for artifact in sorted(output_root.rglob("*")):
        if not artifact.is_file() or artifact == manifest.path:
            continue
        manifest.record_output(artifact, logical_name=artifact.relative_to(output_root).as_posix())


def run_audit(config_path: str | Path, *, resume: bool = False) -> dict[str, object]:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    out = _folders(output_root)
    manifest_path = out["manifests"] / "run_manifest.json"
    input_files = {
        "manifest": Path(config["manifest_path"]),
        "embedding_matrix": Path(config["embeddings_path"]),
        "configuration": config_file,
    }
    if resume and manifest_path.exists():
        manifest = RunManifest.resume(manifest_path, config=config, input_files=input_files)
        if manifest.payload.get("status") == "complete":
            return {"status": "complete", "output_root": str(output_root), "resumed": True}
        manifest.payload["status"] = "running"
        manifest.payload["completion_status"] = "running"
        manifest.payload.pop("failure", None)
        manifest._save()
    else:
        manifest = RunManifest.start(
            manifest_path,
            config_name=str(config["mode"]),
            config=config,
            seeds=config["seeds"],
            input_files=input_files,
            parameters={
                "k_values": config["k_values"],
                "batch_sizes": config["batch_sizes"],
                "orderings": config["orderings"],
                "representations": config["representations"],
                "max_records": config.get("max_records"),
            },
        )

    try:
        raw_manifest = pd.read_csv(config["manifest_path"])
        raw_vectors = np.load(config["embeddings_path"])
        aligned = validate_audit_inputs(raw_manifest, raw_vectors)
        records = _public_records(aligned.manifest, config)
        vectors = aligned.embeddings
        max_records = config.get("max_records")
        if max_records is not None:
            limit = min(int(max_records), len(records))
            records = records.iloc[:limit].reset_index(drop=True)
            vectors = vectors[:limit]

        records = assign_groups(records, vectors, config, out["tables"])
        threshold = float(config["dedup"].get("primary_threshold", max(config["dedup"]["embedding_thresholds"])))
        group_stats, group_distribution = summarize_probable_duplicate_groups(
            records["group_id"], metric=str(config["dedup"].get("method", "cosine_similarity")), threshold=threshold
        )
        group_stats["scope_records"] = int(len(records))
        group_stats["completion_scope"] = "development_subset" if max_records is not None else "full_available_embeddings"
        atomic_write_json(out["tables"] / "group_id_statistics.json", group_stats)
        write_csv(group_distribution, out["tables"] / "group_size_distribution.csv")
        review = safe_threshold_review_sample(
            records,
            vectors,
            threshold=threshold,
            window=float(config.get("threshold_review_window", 0.0005)),
            max_pairs=int(config.get("threshold_review_max_pairs", 100)),
            salt=config["public_id_salt"],
        )
        write_csv(review, out["tables"] / "threshold_review_sample.csv")

        lineage = audit_data_lineage(
            config["manifest_path"],
            config["embeddings_path"],
            claimed_counts=config.get("claimed_counts", DEFAULT_CLAIMED_COUNTS),
            historical_evidence=config.get("historical_evidence", {}),
            disputed_pair=tuple(config.get("disputed_pair", [9546, 9584]))
            if config.get("disputed_pair", [9546, 9584]) is not None
            else None,
        )
        write_csv(lineage, out["tables"] / "data_lineage.csv")
        synthetic_geometry_control(int(config["random_seed"]), out["tables"] / "synthetic_geometry_control.csv")
        run_cross_fitting(records, vectors, config, out["tables"], out["reports"])
        checkpoint_root = Path(
            config.get(
                "checkpoint_root",
                output_root.parent / ".checkpoints" / str(config["mode"]),
            )
        )
        run_stability_analysis(
            vectors,
            config,
            out["tables"],
            checkpoint_root=checkpoint_root,
            resume=resume,
        )
        generate_public_reports(
            output_root,
            config,
            config.get("report_destination", "research_audit_v2"),
        )

        findings = write_privacy_report(output_root, out["reports"] / "privacy_scan.json")
        if findings:
            manifest.fail("privacy_gate_failed")
            raise PrivacyGateError("Public output privacy gate failed; see privacy_scan.json.")
        _record_outputs(manifest, output_root)
        manifest.complete()
        final_findings = scan_public_tree(output_root)
        if final_findings:
            write_privacy_report(output_root, out["reports"] / "privacy_scan.json")
            manifest.fail("final_privacy_gate_failed")
            raise PrivacyGateError("Final public output privacy gate failed; see privacy_scan.json.")
        return {"status": "complete", "output_root": str(output_root), "resumed": False}
    except Exception as error:
        if manifest.payload.get("status") != "failed":
            manifest.fail(type(error).__name__)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_audit(args.config, resume=args.resume)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
