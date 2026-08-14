import numpy as np
import pandas as pd
import json
from pathlib import Path

from research_audit_v2.second_phase.src.stability import (
    build_stability_design,
    pairwise_partition_metrics,
    run_stability_analysis,
    summarize_pairwise_stability,
    summarize_stability,
)


def _config():
    return {
        "mode": "development",
        "random_seed": 11,
        "seeds": [11, 12, 13],
        "k_values": [2, 3],
        "batch_sizes": [8, 16],
        "orderings": ["original", "reversed", "seeded_shuffle"],
        "representations": ["original_l2", "pca_64"],
        "primary_k": 2,
        "primary_batch_size": 8,
        "cluster_max_iter": 10,
        "n_init": 2,
        "target_rule": "largest_cluster",
    }


def test_stability_design_changes_only_the_predeclared_factor_within_each_analysis():
    design = build_stability_design(_config())

    stochastic = design[design["instability_type"].eq("stochastic")]
    assert set(stochastic["k"]) == {2, 3}
    assert set(stochastic["seed"]) == {11, 12, 13}
    assert stochastic["batch_size"].nunique() == 1
    assert stochastic["order"].eq("original").all()
    assert stochastic["representation"].eq("original_l2").all()

    order = design[design["instability_type"].eq("operational_order")]
    assert set(order["order"]) == {"original", "reversed", "seeded_shuffle"}
    assert order[["k", "seed", "batch_size", "representation"]].drop_duplicates().shape[0] == 1

    batch = design[design["instability_type"].eq("operational_batch")]
    assert set(batch["batch_size"]) == {8, 16}
    assert batch[["k", "seed", "order", "representation"]].drop_duplicates().shape[0] == 1

    representation = design[design["instability_type"].eq("representation")]
    assert set(representation["representation"]) == {"original_l2", "pca_64"}
    assert representation[["k", "seed", "batch_size", "order"]].drop_duplicates().shape[0] == 1


def test_final_design_contains_requested_grid_100_explicit_seeds_and_batch_sizes():
    config = _config()
    config.update(
        {
            "mode": "final",
            "seeds": list(range(1000, 1100)),
            "k_values": [32, 48, 64, 80, 96, 128],
            "batch_sizes": [256, 512, 1024, 2048, 4096],
            "primary_k": 64,
            "primary_batch_size": 1024,
        }
    )
    design = build_stability_design(config)
    stochastic = design[design["instability_type"].eq("stochastic")]
    assert len(stochastic) == 600
    assert stochastic["seed"].nunique() == 100
    assert set(stochastic["k"]) == {32, 48, 64, 80, 96, 128}
    assert set(design.loc[design["instability_type"].eq("operational_batch"), "batch_size"]) == {
        256,
        512,
        1024,
        2048,
        4096,
    }


def test_stability_summary_exports_all_requested_distribution_statistics():
    frame = pd.DataFrame(
        {
            "instability_type": ["stochastic"] * 4,
            "k": [2] * 4,
            "ari": [1.0, 2.0, 3.0, 4.0],
            "target_jaccard": [0.1, 0.2, 0.3, 0.4],
            "target_size": [10, 20, 30, 40],
            "target_prevalence": [0.1, 0.2, 0.3, 0.4],
        }
    )
    summary = summarize_stability(frame, ["instability_type", "k"])
    ari = summary.loc[summary["metric"].eq("ari")].iloc[0]
    assert ari["mean"] == 2.5
    assert np.isclose(ari["std"], np.std([1, 2, 3, 4], ddof=1))
    assert ari["median"] == 2.5
    assert ari["q1"] == 1.75
    assert ari["q3"] == 3.25
    assert ari["min"] == 1.0
    assert ari["max"] == 4.0
    assert np.isclose(ari["p05"], 1.15)
    assert np.isclose(ari["p95"], 3.85)


def test_pairwise_metrics_are_symmetric_have_unit_diagonal_and_no_reference_run():
    partitions = {
        "run_a": (np.array([0, 0, 1, 1]), np.array([True, True, False, False])),
        "run_b": (np.array([1, 1, 0, 0]), np.array([True, True, False, False])),
        "run_c": (np.array([0, 1, 0, 1]), np.array([True, False, True, False])),
    }
    pairwise = pairwise_partition_metrics(partitions)
    ari = pairwise.pivot(index="run_a", columns="run_b", values="ari")
    jaccard = pairwise.pivot(index="run_a", columns="run_b", values="target_jaccard")
    assert np.allclose(ari, ari.T)
    assert np.allclose(jaccard, jaccard.T)
    assert np.allclose(np.diag(ari), 1.0)
    assert np.allclose(np.diag(jaccard), 1.0)


def test_stability_summary_uses_unique_off_diagonal_pairs_not_a_canonical_run():
    runs = pd.DataFrame(
        {
            "run_id": ["run_a", "run_b", "run_c"],
            "comparison_group": ["group"] * 3,
            "instability_type": ["stochastic"] * 3,
            "k": [2] * 3,
            "target_size": [10, 20, 30],
            "target_prevalence": [0.1, 0.2, 0.3],
        }
    )
    partitions = {
        "run_a": (np.array([0, 0, 1, 1]), np.array([True, True, False, False])),
        "run_b": (np.array([1, 1, 0, 0]), np.array([True, True, False, False])),
        "run_c": (np.array([0, 1, 0, 1]), np.array([True, False, True, False])),
    }
    pairwise = pairwise_partition_metrics(partitions)
    pairwise.insert(0, "comparison_group", "group")

    summary = summarize_pairwise_stability(runs, pairwise)

    ari = summary.loc[summary["metric"].eq("ari")].iloc[0]
    jaccard = summary.loc[summary["metric"].eq("target_jaccard")].iloc[0]
    assert ari["n"] == 3
    assert jaccard["n"] == 3
    assert np.isclose(ari["mean"], pairwise.loc[pairwise["run_a"] < pairwise["run_b"], "ari"].mean())
    assert np.isclose(
        jaccard["mean"],
        pairwise.loc[pairwise["run_a"] < pairwise["run_b"], "target_jaccard"].mean(),
    )


def test_synthetic_stability_run_writes_run_summary_and_pairwise_outputs(tmp_path):
    rng = np.random.default_rng(4)
    vectors = np.vstack(
        [rng.normal(2, 0.2, size=(20, 6)), rng.normal(-2, 0.2, size=(20, 6))]
    ).astype("float32")
    config = _config()
    config.update(
        {
            "seeds": [11, 12],
            "k_values": [2],
            "representations": ["original_l2"],
            "orderings": ["original", "reversed"],
        }
    )

    runs, summary, pairwise = run_stability_analysis(vectors, config, tmp_path)

    assert {"target_size", "target_prevalence"}.issubset(runs.columns)
    assert {"canonical_run_id", "ari", "target_jaccard"}.isdisjoint(runs.columns)
    assert {"mean", "std", "median", "q1", "q3", "min", "max", "p05", "p95"}.issubset(summary.columns)
    assert {"run_a", "run_b", "ari", "target_jaccard"}.issubset(pairwise.columns)
    assert (tmp_path / "stability_runs.csv").exists()
    assert (tmp_path / "stability_summary.csv").exists()
    assert (tmp_path / "stability_pairwise.csv").exists()


def test_representation_stability_writes_complete_pca_specification(tmp_path):
    vectors = np.random.default_rng(71).normal(size=(90, 70)).astype("float32")
    config = _config()
    config.update(
        {
            "seeds": [11],
            "k_values": [2],
            "batch_sizes": [16],
            "orderings": ["original"],
            "representations": ["original_l2", "pca_64"],
            "primary_batch_size": 16,
            "pca_64": {
                "name": "pca_64",
                "n_components": 64,
                "svd_solver": "randomized",
                "whiten": False,
                "random_state": 11,
                "centering": True,
                "l2_before": True,
                "l2_after": True,
            },
        }
    )

    runs, _, _ = run_stability_analysis(vectors, config, tmp_path)

    assert "pca_64" in set(runs["representation"])
    assert (tmp_path / "pca_specification.json").exists()


def test_repository_configs_predeclare_fast_development_and_complete_final_designs():
    development = json.loads(Path("research_audit_v2/configs/development.yaml").read_text(encoding="utf-8"))
    final = json.loads(Path("research_audit_v2/configs/final.yaml").read_text(encoding="utf-8"))

    assert isinstance(development["seeds"], list)
    assert 1 <= len(development["seeds"]) <= 5
    assert development["max_records"] <= 1000
    assert final["seeds"] and len(final["seeds"]) == 100
    assert len(set(final["seeds"])) == 100
    assert final["k_values"] == [32, 48, 64, 80, 96, 128]
    assert final["batch_sizes"] == [256, 512, 1024, 2048, 4096]
    assert final["max_records"] is None
    assert final["pca_64"] == {
        "name": "pca_64",
        "n_components": 64,
        "svd_solver": "randomized",
        "whiten": False,
        "random_state": 20260727,
        "centering": True,
        "l2_before": True,
        "l2_after": True,
    }
    expected_evidence = {
        "11724": {"path": "analisis_report/preprocess_report.csv", "method": "csv_rows"},
        "9764": {"path": "analisis_report/processing_report.csv", "method": "csv_rows"},
        "9546": {
            "path": "analisis_report/processing_report.csv",
            "method": "csv_value_count",
            "column": "status",
            "value": "success",
        },
    }
    assert development["historical_evidence"] == expected_evidence
    assert final["historical_evidence"] == expected_evidence
    assert development["disputed_pair"] == [9546, 9584]
    assert final["disputed_pair"] == [9546, 9584]


def test_stability_resume_reuses_only_hash_compatible_private_checkpoints(tmp_path, monkeypatch):
    vectors = np.random.default_rng(90).normal(size=(40, 6)).astype("float32")
    config = _config()
    config.update(
        {
            "seeds": [11],
            "k_values": [2],
            "batch_sizes": [8],
            "orderings": ["original"],
            "representations": ["original_l2"],
        }
    )
    tables = tmp_path / "tables"
    checkpoints = tmp_path / "private-checkpoints"
    first, _, _ = run_stability_analysis(
        vectors, config, tables, checkpoint_root=checkpoints, resume=False
    )

    from research_audit_v2.second_phase.src import stability as module

    calls = 0
    original = module._fit_partition

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_partition", counted)
    resumed, _, _ = run_stability_analysis(
        vectors, config, tables, checkpoint_root=checkpoints, resume=True
    )
    assert calls == 0
    pd.testing.assert_frame_equal(first, resumed)

    changed = vectors.copy()
    changed[0, 0] += 1
    run_stability_analysis(changed, config, tables, checkpoint_root=checkpoints, resume=True)
    assert calls > 0
    assert not any(path.is_relative_to(tables) for path in checkpoints.rglob("*"))
