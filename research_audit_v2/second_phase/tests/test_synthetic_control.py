from research_audit_v2.second_phase.src.controls import synthetic_geometry_control


def test_synthetic_geometry_control_demonstrates_circular_separability_and_target_instability(tmp_path):
    output = tmp_path / "synthetic_geometry_control.csv"

    result = synthetic_geometry_control(seed=17, output_path=output)

    assert result["pass"].all()
    assert set(result["demonstration"]) == {
        "clustering_generates_synthetic_target",
        "same_geometry_score_has_high_separability",
        "target_changes_when_clustering_is_perturbed",
    }
    separability = result[result["demonstration"].eq("same_geometry_score_has_high_separability")].iloc[0]
    instability = result[result["demonstration"].eq("target_changes_when_clustering_is_perturbed")].iloc[0]
    assert separability["roc_auc"] > 0.95
    assert instability["target_jaccard"] < 0.8
    assert result["interpretation"].eq("methodological_demonstration_only").all()
    assert output.exists()
