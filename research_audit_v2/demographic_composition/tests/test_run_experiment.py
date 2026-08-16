from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research_audit_v2.demographic_composition.cohorts import FAIRFACE_GROUPS
from research_audit_v2.demographic_composition.run_experiment import run_experiment


class SyntheticEmbedder:
    def __init__(self, **_: object) -> None:
        pass

    def extract_path(self, path):
        index = int(path.stem)
        group = FAIRFACE_GROUPS.index(path.parent.name)
        vector = np.zeros(8, dtype=np.float32)
        vector[group] = 2.0
        vector[7] = (index % 5) / 10
        return SimpleNamespace(embedding=vector)


def test_synthetic_end_to_end_run_is_complete_private_and_resumable(tmp_path):
    catalog = pd.DataFrame(
        [
            {"relative_path": f"{group}/{index}.jpg", "source_race_label": group}
            for group in FAIRFACE_GROUPS
            for index in range(30)
        ]
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    config = {
        "random_seed": 17,
        "sample_size": 84,
        "group_column": "source_race_label",
        "perturbed_group": "Middle Eastern",
        "catalog_path": str(catalog_path),
        "image_root": str(tmp_path / "images"),
        "private_root": str(tmp_path / "private"),
        "output_root": str(tmp_path / "public"),
        "model_name": "synthetic",
        "ctx_id": -1,
        "det_size": 64,
        "preprocessing_mode": "upstream_dlib_aligned_crop_direct_arcface",
        "embedding_batch_size": 16,
        "seeds": [17],
        "k_values": [2],
        "primary_k": 2,
        "batch_size": 16,
        "max_iter": 20,
        "n_init": 1,
        "outer_folds": 2,
        "target_rule": "largest_cluster",
        "relevance_thresholds": {
            "ari": .90,
            "target_jaccard": .80,
            "target_prevalence_delta": .02,
            "auc_delta": .03,
        },
    }
    report = tmp_path / "DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md"

    run_experiment(config, destination=report, embedder_factory=SyntheticEmbedder)

    manifest = json.loads((tmp_path / "public" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["privacy_status"] == "passed"
    assert len(set(manifest["parameters_by_scenario"].values())) == 1
    composition = pd.read_csv(tmp_path / "public" / "tables" / "scenario_composition.csv")
    assert composition.groupby("scenario")["count"].sum().eq(84).all()
    assert report.exists()
    assert (tmp_path / "private" / "final_selection.csv").exists()
    assert not (tmp_path / "public" / "final_selection.csv").exists()

    def forbidden_factory(**_: object):
        raise AssertionError("Resume should use the compatible private embedding cache")

    run_experiment(config, destination=report, resume=True, embedder_factory=forbidden_factory)
    resumed = json.loads((tmp_path / "public" / "run_manifest.json").read_text(encoding="utf-8"))
    assert resumed["status"] == "complete"
