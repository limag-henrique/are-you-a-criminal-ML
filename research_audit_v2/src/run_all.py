"""Single entry point for a traceable privacy-preserving audit run."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifact_comparison import compare_artifacts
from .circularity_analysis import circularity
from .clustering_stability import fit_clusters, run_stability
from .common import read_config, sha256_file, write_csv
from .deduplication import assign_groups
from .inventory import build_inventory, git_value
from .predictive_models import predictive_models
from .privacy import scan_public_outputs
from .provenance import reconcile, load_audited_records
from .reporting import write_final_reports
from .source_enrichment import source_enrichment
from .target_heuristic import choose_target_cluster, target_membership


def prepare(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("tables", "figures", "logs", "manifests", "reports")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_environment(path: Path) -> None:
    payload = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "git_commit": git_value("rev-parse", "HEAD"), "worktree_dirty": bool(git_value("status", "--porcelain"))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = read_config(args.config)
    outputs = prepare(Path(config["output_root"]))
    config_hash = sha256_file(args.config)
    run_id = f"{config['mode']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}"
    build_inventory(config, Path("research_audit_v2/inventory.json"))
    write_environment(Path("research_audit_v2/environment/environment.json"))
    records = reconcile(config, outputs["tables"], outputs["reports"], outputs["manifests"])
    records, vectors = load_audited_records(config)
    records = assign_groups(records, vectors, config, outputs["tables"])
    # Keep the public lineage synchronized with the actual duplicate grouping.
    records[["record_id", "group_id", "source", "quality", "embedding_index"]].sort_values("record_id").to_parquet(outputs["manifests"] / "record_lineage.parquet", index=False)
    stability = run_stability(records, vectors, config, outputs["tables"], outputs["figures"], outputs["reports"])
    seed = int(stability.loc[stability["k"].eq(64), "seed"].iloc[0]) if (stability["k"] == 64).any() else int(stability["seed"].iloc[0])
    k = 64 if 64 in config["k_values"] else int(config["k_values"][0])
    _, labels = fit_clusters(vectors, k, seed, config)
    target = choose_target_cluster(labels, config["target_rule"])
    membership = target_membership(labels, target)
    counts = np.bincount(labels)
    ranked = np.argsort(-counts)
    sensitivity = []
    for rank, candidate in enumerate(ranked[:3], start=1):
        candidate_membership = labels == candidate
        sensitivity.append({"rule": config["target_rule"], "rank": rank, "cluster": int(candidate), "cluster_size": int(candidate_membership.sum()), "target_jaccard_vs_primary": float((candidate_membership & membership).sum() / max((candidate_membership | membership).sum(), 1))})
    write_csv(__import__("pandas").DataFrame(sensitivity), outputs["tables"] / "target_selection_sensitivity.csv")
    outputs["reports"].joinpath("target_heuristic_specification.md").write_text("# Target heuristic specification\n\n**Rule:** select the cluster with the greatest member count; ties go to the lowest numeric cluster label. The rule uses the clustering labels, which depend on embeddings. It is an explicit reconstruction because no historical target-selection source was located. Sensitivity reports the next two size-ranked clusters.\n", encoding="utf-8")
    circularity(vectors, membership, config, outputs["tables"], outputs["figures"], outputs["reports"])
    predictive_models(records, membership, config, outputs["tables"], outputs["reports"])
    source_enrichment(records, membership, config, outputs["tables"], outputs["figures"], outputs["reports"])
    compare_artifacts(outputs["tables"], outputs["reports"])
    write_final_reports(outputs["reports"], outputs["tables"])
    violations = scan_public_outputs(Path(config["output_root"]))
    outputs["reports"].joinpath("privacy_scan_report.md").write_text("# Privacy scan\n\n" + ("No forbidden patterns found.\n" if not violations else "\n".join(f"- {item}" for item in violations) + "\n"), encoding="utf-8")
    if violations:
        raise RuntimeError("Privacy scanner found forbidden patterns in public outputs.")
    manifest = {"run_id": run_id, "config": str(args.config), "config_sha256": config_hash, "input_hashes": {key: sha256_file(config[key]) for key in ("manifest_path", "embeddings_path")}, "git_commit": git_value("rev-parse", "HEAD"), "started_utc": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.perf_counter() - started, "completion_status": "complete"}
    (outputs["manifests"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": "complete", "duration_seconds": manifest["duration_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
