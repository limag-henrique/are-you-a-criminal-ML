# Final reproduction report

## Execution status

- Configuration: `final`.
- Scope: complete configured scientific run.
- Run-manifest status at report generation: `complete`.
- Public-output privacy gate: required before the run manifest can be marked complete.

## Changes implemented

- Strict aligned manifest/embedding contracts, atomic writers, incremental hashed run manifest and safe resume checks.
- Grouped cross-fitting with train-only transformation, clustering, target selection, centroid, calibration and threshold.
- Separate stochastic, operational and representation stability analyses with pairwise ARI/Jaccard.
- Fully specified PCA-64 reconstruction, probable-duplicate group audit, evidence-classified count lineage and synthetic methodological control.
- Fail-closed public-output scanner and synthetic-only CI.

## Tests

Executed `python -m pytest -q research_audit_v2`: **78 passed in 18.56s**. Evidence: `reports/final_verification.json`.

## Experiments effectively executed

- Grouped cross-fitting: 5 folds; mean metrics `{"balanced_accuracy": 0.649563, "brier": 0.043299, "f1": 0.338413, "pr_auc": 0.348207, "pr_auc_baseline": 0.05389, "precision": 0.370329, "recall": 0.333153, "roc_auc": 0.89617}`. Evidence: `tables/cross_fitted_metrics.csv`.
- Stability: 611 runs across `operational_batch, operational_order, representation, stochastic`. Evidence: `tables/stability_runs.csv`, `tables/stability_summary.csv`, `tables/stability_pairwise.csv`.
- Synthetic geometry control: 3/3 demonstrations passed. Evidence: `tables/synthetic_geometry_control.csv`.
- Probable-duplicate grouping: 9422 groups, 59 non-singleton groups and grouped-record proportion 0.012550. Evidence: `tables/group_id_statistics.json`.

## Historical results preserved

- 11,724: documented_historical_artifact; evidence `tables/data_lineage.csv`.
- 9,764: documented_historical_artifact; evidence `tables/data_lineage.csv`.
- 9,546: documented_historical_artifact; evidence `tables/data_lineage.csv`.

## Results reproduced

- 9,584: verified_current_artifact; evidence `tables/data_lineage.csv`.
- 9,482: verified_current_artifact; evidence `tables/data_lineage.csv`.

## New methodological reconstructions

- `group_id` is a probable-duplicate grouping based on declared cosine similarity, never confirmed identity.
- Cluster target, grouped cross-fitting, PCA-64, thresholds/calibration and stability analyses are new reconstructions, not attributed to the historical pipeline.
- All predictive metrics are internal recovery of a synthetic target only.

## Historical information not recovered

- No unresolved count claim.
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
