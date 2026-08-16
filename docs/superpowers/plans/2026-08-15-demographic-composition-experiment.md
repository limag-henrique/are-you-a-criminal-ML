# Demographic Composition Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create commits; repository policy reserves commits for a human.

**Goal:** Build and run an isolated, reproducible FairFace experiment comparing four demographic compositions under an otherwise fixed clustering and scoring pipeline.

**Architecture:** A small package under `research_audit_v2/demographic_composition` owns deterministic cohort construction, a private resumable embedding cache, scenario analysis, and aggregate reporting. It imports the existing ArcFace extractor and audited largest-cluster rule without modifying main-pipeline code. Public artifacts contain only pseudonymous IDs and aggregates; sensitive paths and embeddings remain in ignored directories.

**Tech Stack:** Python 3.10-3.12, NumPy, pandas, scikit-learn, matplotlib, InsightFace/ONNX Runtime, pytest.

## Global Constraints

- Use only FairFace `source_race_label`; never infer labels.
- Perturb `Middle Eastern`; use exactly 36,456 distinct records per scenario without replacement.
- B has 5,208/category; C has 2,604 perturbed and 5,642/other; D has 10,416 perturbed and 4,340/other; A preserves source proportions by largest-remainder allocation.
- Fix model/preprocessing/clustering/scoring parameters across A-D.
- Do not modify or join main-pipeline records, images, manifests, or embeddings.
- Do not change branches or create commits.
- Interpret ROC-AUC/PR-AUC only as recovery of a synthetic clustering-derived target.

---

### Task 1: Deterministic scenario construction

**Files:**
- Create: `research_audit_v2/demographic_composition/__init__.py`
- Create: `research_audit_v2/demographic_composition/config.json`
- Create: `research_audit_v2/demographic_composition/cohorts.py`
- Create: `research_audit_v2/demographic_composition/tests/test_cohorts.py`

**Interfaces:**
- Produces: `largest_remainder_quotas(counts, total) -> dict[str, int]`, `scenario_quotas(catalog, config) -> dict[str, dict[str, int]]`, and `build_scenarios(catalog, config) -> tuple[pd.DataFrame, pd.DataFrame]` where selection columns are `scenario`, `record_id`, `source_race_label`, `relative_path`, `selection_rank` and reserves use the same schema without `scenario`.

- [ ] Write failing tests asserting exact B/C/D quotas, A total/proportional allocation, deterministic output, unique records within scenarios, and same-group reserve ordering.

```python
def test_declared_scenarios_have_exact_quotas_and_no_duplicates():
    selected, _ = build_scenarios(synthetic_catalog(), config())
    observed = selected.groupby(["scenario", "source_race_label"]).size().unstack(fill_value=0)
    assert observed.loc["B"].eq(5208).all()
    assert observed.loc["C", "Middle Eastern"] == 2604
    assert observed.loc["D", "Middle Eastern"] == 10416
    assert selected.groupby("scenario")["record_id"].nunique().eq(36456).all()
```

- [ ] Run `python -m pytest -q research_audit_v2/demographic_composition/tests/test_cohorts.py` and verify failure because the package does not exist.
- [ ] Implement stable SHA-256 pseudonyms, shared per-group seeded permutations, largest-remainder A quotas, fixed B/C/D quotas, and deterministic reserves.
- [ ] Re-run the focused test, then `python -m pytest -q research_audit_v2`; require both to pass.
- [ ] Checkpoint with `git diff --check` and `git status --short`; do not commit.

### Task 2: Resumable private embedding extraction

**Files:**
- Create: `research_audit_v2/demographic_composition/embeddings.py`
- Create: `research_audit_v2/demographic_composition/tests/test_embeddings.py`

**Interfaces:**
- Consumes: selection and reserve frames from Task 1; `face_profile_ml.extractor.ArcFaceEmbedder`.
- Produces: `FairFaceAlignedCropEmbedder`, `extract_union_embeddings(selected, reserves, image_root, private_root, config, embedder_factory=FairFaceAlignedCropEmbedder) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]`, and atomic checkpoint files keyed by catalog/config hashes. The official `margin025` crops are already dlib-aligned and are sent directly to ArcFace recognition in batches.

- [ ] Write failing tests with a deterministic fake embedder proving union-only extraction, resume without re-extraction, same-group failure replacement, preserved quotas, aligned rows/vectors, finite L2 vectors, and invalid checkpoint rejection.
- [ ] Run the focused test and verify the expected missing-interface failure.
- [ ] Implement dependency-injected extraction using `ArcFaceEmbedder.extract_path`, atomic NPZ/JSON checkpoints, per-record status, and fail-closed quota validation after reserve replacement.
- [ ] Run focused and repository tests; require clean passes.
- [ ] Checkpoint with diff/status only.

### Task 3: Fixed clustering, scoring, and stability analysis

**Files:**
- Create: `research_audit_v2/demographic_composition/analysis.py`
- Create: `research_audit_v2/demographic_composition/tests/test_analysis.py`

**Interfaces:**
- Consumes: aligned selection frame/vectors; audited `choose_target_cluster` and L2 normalization.
- Produces: `fit_scenario_run(records, vectors, scenario, seed, k, config) -> RunResult`, `pairwise_seed_stability(partitions) -> pd.DataFrame`, `compare_scenarios_on_intersection(partitions) -> pd.DataFrame`, and `cross_fitted_scores(records, vectors, scenario, seed, k, config) -> pd.DataFrame`.

- [ ] Write failing synthetic tests asserting largest-cluster selection, target size/prevalence, cosine-centroid ordering, cluster demographic counts summing to N, symmetric pairwise ARI/Jaccard, record-ID intersection alignment across scenarios, and NaN ROC/PR for a single target class.

```python
def test_scenario_comparison_aligns_only_shared_record_ids():
    compared = compare_scenarios_on_intersection({"A": partition_a, "C": partition_c})
    assert compared.loc[0, "intersection_n"] == 3
    assert 0 <= compared.loc[0, "ari"] <= 1
    assert 0 <= compared.loc[0, "target_jaccard"] <= 1
```

- [ ] Run the focused test and verify the missing-interface failure.
- [ ] Implement MiniBatchKMeans with shared `batch_size=1024`, `max_iter=100`, `n_init=3`, target centroid scoring, five-fold record-disjoint evaluation, run-level compositions, and pairwise metrics.
- [ ] Run focused and repository tests; require clean passes.
- [ ] Checkpoint with diff/status only.

### Task 4: Aggregate tables, figures, and objective conclusion

**Files:**
- Create: `research_audit_v2/demographic_composition/reporting.py`
- Create: `research_audit_v2/demographic_composition/tests/test_reporting.py`
- Create on execution: `DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md`

**Interfaces:**
- Consumes: run, cross-fit, stability, composition, failure, and provenance tables.
- Produces: `summarize_results(tables, config) -> dict[str, pd.DataFrame]`, `classify_relevance(summary, thresholds) -> dict[str, object]`, and `write_report(output_root, destination, config) -> Path` plus SVG figures.

- [ ] Write failing tests for relevance thresholds (`ARI<.90`, `Jaccard<.80`, prevalence delta `>=.02`, AUC delta `>=.03`), complete A-D tables, linked SVGs, explicit limitations, and absence of images, embeddings, absolute paths, or identity claims.
- [ ] Run the focused test and verify expected failure.
- [ ] Implement CSV summaries and SVGs for scenario composition, target size/prevalence, ARI/Jaccard, ROC/PR, `k`/seed sensitivity, and cluster demographic distributions; generate the root Markdown solely from structured artifacts.
- [ ] Run focused and repository tests; require clean passes.
- [ ] Checkpoint with diff/status only.

### Task 5: Orchestration, smoke run, full run, and verification

**Files:**
- Create: `research_audit_v2/demographic_composition/run_experiment.py`
- Create: `research_audit_v2/demographic_composition/tests/test_run_experiment.py`
- Create during execution under ignored paths: `research_audit_v2/.sensitive/demographic_composition/*`, `research_audit_v2/outputs/demographic_composition/*`

**Interfaces:**
- Produces CLI: `python -m research_audit_v2.demographic_composition.run_experiment --config ... [--smoke] [--resume]`.

- [ ] Write a failing synthetic integration test requiring config/input hashes, phase statuses, exact parameter equality across scenarios, atomic completion manifest, and refusal to report an incomplete run.
- [ ] Run the test and verify expected failure.
- [ ] Implement phased orchestration: catalog validation → cohort selection → embedding union/resume → analysis grid → cross-fitting → tables/figures → privacy scan → completion manifest → root report.
- [ ] Run the integration test and full repository suite.
- [ ] Verify runtime imports with `python -c "import insightface, onnxruntime; print('OK')"`; repair only the project environment if needed without changing dependency policy.
- [ ] Run `--smoke`, inspect all generated tables and SVGs, then run the full fixed grid `k=[32,48,64,80,96,128]`, ten explicit seeds derived from `20260815`, and `--resume`.
- [ ] Run `python -m pytest -q`, `git diff --check`, the public privacy scan, quota/hash validation, and compare main-pipeline file hashes to their pre-run values.
- [ ] Confirm `DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md` answers the requested question from completed artifacts and clearly distinguishes relevant changes from internal metric variation.
