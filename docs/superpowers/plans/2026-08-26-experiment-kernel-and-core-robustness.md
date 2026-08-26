# Experiment Kernel and Core Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Steps use checkbox syntax. Repository policy prohibits automatic commits.

**Goal:** Build a resumable experiment kernel and use it for target ablation, AUC reconciliation, repeated-refit uncertainty, and clustering robustness.

**Architecture:** Immutable specifications create stable hashes. Fold-level clustering fits are reused across target rules, while scoring and calibration remain rule-specific. Thin scripts write validated bundles with qualified metric names and explicit failure states.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, SciPy, PyArrow, pytest.

**Spec:** docs/superpowers/specs/2026-08-26-quality-maximization-experiments-design.md

## Global Constraints

- Preserve existing local changes and the current branch.
- Never create commits.
- Use .venv311\Scripts\python.exe for scientific commands.
- Never cite smoke artifacts as evidence.
- Keep AUC qualified as internal cluster-target recovery.
- Public outputs contain no images, embeddings, absolute paths, or identifiers.

---

### Task 1: Immutable experiment specifications

**Files:**
- Create: face_profile_ml/experiment_specs.py
- Create: tests/test_experiment_specs.py
- Modify: scripts/README.md

**Interfaces:**
- Produces FitSpec, AnalysisSpec, canonical_json(value) -> str, and stable_id(prefix, value) -> str.
- FitSpec fields: encoder, backend, n_init, k, seed, fold, grouping_threshold.
- AnalysisSpec fields: fit_id, target_rule, target_seed, protocol_id.

- [ ] **Step 1: Record the baseline**

Run:

~~~powershell
git status --short --branch
.\.venv311\Scripts\python.exe -m pytest tests research_audit_v2/tests research_audit_v2/second_phase/tests -q
~~~

Capture exit status, pass/fail count, and failing names in artifacts/logs/quality_maximization_baseline.txt through normal test-output capture. Do not fix unrelated baseline failures.

- [ ] **Step 2: Write the failing stability test**

~~~python
from face_profile_ml.experiment_specs import AnalysisSpec, FitSpec

def test_fit_id_is_stable_and_excludes_target_rule() -> None:
    fit = FitSpec("arcface", "minibatch", 10, 64, 7, 2, 0.999)
    assert fit.fit_id == FitSpec("arcface", "minibatch", 10, 64, 7, 2, 0.999).fit_id
    largest = AnalysisSpec(fit.fit_id, "largest", 101, "oof-v1")
    compact = AnalysisSpec(fit.fit_id, "compact", 101, "oof-v1")
    assert largest.fit_id == compact.fit_id
    assert largest.spec_id != compact.spec_id
~~~

- [ ] **Step 3: Verify red**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_experiment_specs.py -q

Expected: import failure for face_profile_ml.experiment_specs.

- [ ] **Step 4: Implement the minimal contract**

Use frozen dataclasses, sorted JSON keys, compact separators, UTF-8, and SHA-256 truncated to 16 hexadecimal characters. Reject k below 2, n_init below 1, negative folds, and empty categorical fields in __post_init__.

- [ ] **Step 5: Verify green**

Run:

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_experiment_specs.py -q
git diff --check
git status --short
~~~

### Task 2: Configurable clustering backends

**Files:**
- Modify: face_profile_ml/clustering_backends.py
- Modify: tests/test_clustering_backends.py

**Interfaces:**
- Produces build_backend(name, *, n_clusters, n_init, batch_size=1024, max_iter=100).
- Each backend implements fit(X, seed) -> FittedClustering.
- FittedClustering exposes labels, centers, inertia, n_iter, and predict(X).

- [ ] **Step 1: Write a failing parameter test**

~~~python
def test_kmeans_backends_honor_declared_n_init() -> None:
    mini = build_backend("minibatch", n_clusters=2, n_init=20)
    full = build_backend("kmeans", n_clusters=2, n_init=50)
    assert mini.n_init == 20
    assert full.n_init == 50
~~~

- [ ] **Step 2: Verify the signature failure**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_clustering_backends.py -q

- [ ] **Step 3: Implement the adapter**

Keep fit_predict(X, seed) as a compatibility wrapper. For GMM, expose means as centers and lower_bound_ as the objective; unavailable inertia/iteration values are documented NaN. Agglomerative remains legacy-only.

- [ ] **Step 4: Run backend and legacy CV tests**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_clustering_backends.py tests/test_cross_validation.py -q

### Task 3: Reusable fold fits and rule-specific analysis

**Files:**
- Create: face_profile_ml/experiment_runner.py
- Create: tests/test_experiment_runner.py
- Modify: face_profile_ml/cross_validation.py
- Modify: face_profile_ml/target_rules.py
- Modify: tests/test_target_rules.py

**Interfaces:**
- fit_grouped_folds(samples, embeddings, fit_specs) -> dict[str, FoldFit].
- analyze_fold_fit(fold_fit, rule, target_seed) -> pd.DataFrame.
- run_specifications(samples, embeddings, fit_specs, rules, protocol_id) -> ExperimentResult.
- ExperimentResult contains oof_predictions, fit_index, specification_metrics, and failures.

- [ ] **Step 1: Write a failing reuse test**

~~~python
def test_rules_reuse_one_clustering_fit_per_fold(counting_backend, samples, embeddings) -> None:
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]
    result = run_specifications(
        samples, embeddings, specs,
        ["largest", "compact", "central"], "oof-v1",
        backend_factory=counting_backend,
    )
    assert counting_backend.fit_count == 4
    assert result.specification_metrics["target_rule"].nunique() == 3
~~~

- [ ] **Step 2: Verify red**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_experiment_runner.py -q

- [ ] **Step 3: Make isolated distinct from outlier**

Implement isolated as the observed cluster whose centroid maximizes its minimum distance to every other observed centroid. Replace the distance-matrix diagonal with infinity and break ties by lowest cluster ID. Map legacy separated to isolated; retain outlier as distance from the global centroid.

- [ ] **Step 4: Add the geometry regression**

Create centroids where farthest-from-global differs from largest-nearest-neighbor-distance and assert outlier != isolated.

- [ ] **Step 5: Implement fit reuse and qualified metrics**

Pool OOF rows by spec_id. Emit oof_pooled_cluster_recovery_roc_auc, oof_pooled_cluster_recovery_pr_auc, oof_brier, prevalence, target_size, eligible, and status. Emit ineligible_single_class rather than a blank cell. Check raw/calibrated ROC-AUC rank equivalence within 1e-12.

- [ ] **Step 6: Preserve run_grouped_cluster_cv**

Route one legacy backend/rule through the runner and translate qualified metrics to the existing auc, pr_auc, brier, and balanced_accuracy keys.

- [ ] **Step 7: Verify the slice**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_target_rules.py tests/test_experiment_runner.py tests/test_cross_validation.py -q

### Task 4: Atomic artifact bundles

**Files:**
- Create: face_profile_ml/experiment_cache.py
- Create: tests/test_experiment_cache.py

**Interfaces:**
- ArtifactBundle(root, experiment_name, config, inputs).
- Methods: write_fit, read_fit, record_failure, write_tables, validate, complete.

- [ ] **Step 1: Write a failing corruption test**

~~~python
def test_corrupt_checkpoint_is_not_complete(tmp_path) -> None:
    bundle = ArtifactBundle(tmp_path, "ablation", config(), inputs())
    bundle.write_fit("fit-1", valid_fold_fit())
    bundle.fit_path("fit-1").write_bytes(b"corrupt")
    assert bundle.validate().status == "invalid"
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.complete(expected_fit_ids={"fit-1"})
~~~

- [ ] **Step 2: Verify red**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_experiment_cache.py -q

- [ ] **Step 3: Implement atomic writes**

Write beside the destination, flush/close, and use Path.replace. Store SHA-256 for every checkpoint. Use Parquet for tables and sorted UTF-8 JSON for manifests. Store relative paths only.

- [ ] **Step 4: Implement completion validation**

Require exact expected fit/spec ID sets. Allow explicitly ineligible cells, but reject missing, corrupt, or unclassified cells. Refuse resume when config/input hashes differ.

- [ ] **Step 5: Verify green**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_experiment_cache.py -q

### Task 5: Primary-corpus target ablation

**Files:**
- Modify: scripts/run_target_ablation.py
- Create: tests/test_run_target_ablation.py
- Create: configs/experiments/target_ablation_confirmatory.json

**Interfaces:**
- CLI consumes primary manifest/embeddings and JSON config.
- Produces common bundle, rule_compatibility.parquet, and ablation_summary.csv.

- [ ] **Step 1: Write a failing synthetic CLI test**

Use 60 rows, two seeds, two k values, six rules, and three folds. Assert fit count equals seeds × k × folds, not that count × rules. Assert random has substantive_rule=false. Assert compatibility is symmetric with diagonal 1.

- [ ] **Step 2: Verify the current script fails**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_run_target_ablation.py -q

- [ ] **Step 3: Add the confirmatory config**

Use seeds 1001 through 1020, k=[32,64,128], five folds, plausible rules largest/compact/central/outlier/isolated, null rule random, AUC threshold 0.85, and Jaccard threshold 0.10.

- [ ] **Step 4: Implement summaries**

Aggregate pooled OOF metrics per specification, medians/percentile intervals across seeds, the fraction of plausible rules above 0.85, and exact pairwise Jaccard distributions. Exclude random from substantive success counts.

- [ ] **Step 5: Run smoke and tests**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_run_target_ablation.py -q
.\.venv311\Scripts\python.exe scripts/run_target_ablation.py --config configs/experiments/target_ablation_confirmatory.json --seeds 1001,1002 --k-values 4,8 --folds 3 --max-samples 240 --out-dir artifacts/ablation_smoke_v2
~~~

### Task 6: AUC protocol reconciliation

**Files:**
- Modify: scripts/run_groupid_sensitivity.py
- Create: tests/test_groupid_protocol_reconciliation.py
- Create: configs/experiments/groupid_reconciliation_confirmatory.json

**Interfaces:**
- Produces protocol_matrix.csv, threshold OOF bundles, and reconciliation.json.

- [ ] **Step 1: Write a one-factor comparator test**

Create two protocols differing only in grouping threshold and assert all other fields are byte-identical. Add a changed score definition and assert it is reported as a second difference.

- [ ] **Step 2: Implement complete protocol records**

Record corpus/embedding/sample/group/fold hashes, backend, k, seed, n_init, target rule, raw score, calibration, aggregation, and metric version.

- [ ] **Step 3: Route all thresholds through the same runner**

Use 0.995, 0.997, 0.999, 0.9995; MiniBatchKMeans; k=64; n_init=3; seed 42; pooled OOF metrics. Compare Jaccard/ARI against 0.9995 on aligned IDs.

- [ ] **Step 4: Generate evidence-led reconciliation**

The JSON contains comparable, differing_fields, reproduced_range, and a fixed-template explanation. It cannot claim threshold causality when another field differs.

- [ ] **Step 5: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_groupid_protocol_reconciliation.py -q

### Task 7: Conditional bootstrap and repeated refits

**Files:**
- Modify: face_profile_ml/bootstrap_ci.py
- Create: face_profile_ml/repeated_grouped_cv.py
- Modify: tests/test_bootstrap_ci.py
- Create: tests/test_repeated_grouped_cv.py

**Interfaces:**
- BootstrapResult adds uncertainty_scope="conditional_on_fitted_folds".
- repeated_group_assignments(samples, repeats, folds, seed) -> pd.DataFrame.
- summarize_refit_dispersion(metrics) -> pd.DataFrame.

- [ ] **Step 1: Write failing scope/disjointness tests**

Assert every bootstrap result serializes the conditional scope. Across five repeats, every group appears in one fold per repeat, all folds are populated, and assignments are deterministic.

- [ ] **Step 2: Implement stable group bin packing**

Shuffle unique groups with seed + repeat, sort by size using a random tie key, and assign each whole group to the currently smallest fold.

- [ ] **Step 3: Implement separate dispersion summaries**

Report mean, standard deviation, minimum, maximum, and percentile interval across repeated pooled OOF fits. Do not merge this with the 2,000-resample conditional bootstrap.

- [ ] **Step 4: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_bootstrap_ci.py tests/test_repeated_grouped_cv.py -q

### Task 8: MiniBatch/KMeans and n_init robustness

**Files:**
- Modify: scripts/run_clustering_comparison.py
- Create: tests/test_run_clustering_comparison.py
- Create: configs/experiments/clustering_robustness_confirmatory.json

**Interfaces:**
- Produces common bundle, within_backend_stability.parquet, between_backend_compatibility.parquet, and clustering_summary.csv.

- [ ] **Step 1: Write a failing complete-grid test**

On a fixture, assert every backend={minibatch,kmeans} × n_init={3,10} × seed={1,2} × fold={0,1,2} cell exists once and no agglomerative cell exists.

- [ ] **Step 2: Implement the declared grid**

Use n_init=[3,10,20,50], seeds 2001 through 2020, k=64, five folds, and largest rule.

- [ ] **Step 3: Implement stability summaries**

Compute within-backend target Jaccard/partition ARI across seeds and between-backend compatibility at matched seed/n_init. Record runtime, inertia, convergence, prevalence, and qualified OOF recovery.

- [ ] **Step 4: Add reduced exploratory GMM**

Use seeds 2001,2002,2003 and mark every row analysis_tier=exploratory. Keep it out of confirmatory headline summaries.

- [ ] **Step 5: Verify plan-one deliverable**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests -q
git diff --check
git status --short
~~~

