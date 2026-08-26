# Encoder, FairFace, and Multiverse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Do not commit automatically.

**Goal:** Add an independent frozen encoder, run symmetric paired FairFace perturbations, and build a cached multiverse.

**Architecture:** A frozen encoder protocol writes hashed private caches. FairFace cohorts use nested category permutations for maximal pairing. The multiverse consumes the shared fit contract and calculates exact compatibility with packed boolean arrays.

**Tech Stack:** Python 3.11, OpenCV 4.10+, NumPy, pandas, scikit-learn, PyArrow, Matplotlib, pytest.

**Spec:** docs/superpowers/specs/2026-08-26-quality-maximization-experiments-design.md

## Global Constraints

- Complete the experiment-kernel plan first.
- Preserve local changes and branch; never commit.
- Keep model binaries and embeddings private.
- Never infer FairFace categories or transfer them to the primary corpus.

---

### Task 1: Frozen SFace encoder and provenance

**Files:**
- Create: face_profile_ml/encoders.py
- Create: tests/test_encoders.py
- Create: scripts/extract_sface_embeddings.py
- Create: tests/test_extract_sface_embeddings.py
- Create: configs/encoders/sface_2021dec.json

**Interfaces:**
- FrozenEncoder.encode_aligned_bgr(image) -> np.ndarray.
- SFaceEncoder(model_path).encode_aligned_bgr(image) -> np.ndarray.
- CLI produces a private NPZ and public sface_embedding_manifest.csv plus sface_model_provenance.json.

- [ ] **Step 1: Write a failing normalization test**

~~~python
def test_sface_encoder_returns_finite_l2_vector(fake_model) -> None:
    encoder = SFaceEncoder("model.onnx", recognizer_factory=lambda _: fake_model)
    vector = encoder.encode_aligned_bgr(np.zeros((112, 112, 3), dtype=np.uint8))
    assert vector.ndim == 1
    assert np.isfinite(vector).all()
    assert np.isclose(np.linalg.norm(vector), 1.0)
~~~

- [ ] **Step 2: Verify red**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_encoders.py -q

- [ ] **Step 3: Implement explicit preprocessing**

Read BGR, resize to 112×112 when needed, call cv2.FaceRecognizerSF.feature, flatten float32 output, reject non-finite/zero-norm vectors, and L2-normalize. Do not call ArcFace or PCA.

- [ ] **Step 4: Implement model verification**

The config stores official OpenCV Zoo URL, byte length, reviewed SHA-256, Apache-2.0 directory URL, and training_data_provenance_status="incomplete". Refuse hash mismatch.

- [ ] **Step 5: Implement resumable extraction**

Use the same aligned primary crops, pseudonymous IDs, and atomic batches of 128. Validate model/input/config hashes on resume. Emit success, missing_image, decode_error, or encoder_error.

- [ ] **Step 6: Verify and smoke**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_encoders.py tests/test_extract_sface_embeddings.py -q
.\.venv311\Scripts\python.exe scripts/extract_sface_embeddings.py --max-samples 20 --out-dir artifacts/sface_smoke
~~~

Expected: vectors stay under the ignored private cache and the public manifest has no image paths.

### Task 2: ArcFace/SFace common-universe comparison

**Files:**
- Modify: scripts/run_embedding_decoupling.py
- Create: tests/test_run_encoder_comparison.py
- Create: configs/experiments/encoder_comparison_confirmatory.json

**Interfaces:**
- Produces common_universe.csv, per-encoder bundles, cross_encoder_compatibility.parquet, and encoder_summary.csv.

- [ ] **Step 1: Write a failing ID-alignment test**

Given ArcFace IDs {a,b,c} and SFace IDs {b,c,d}, assert the common IDs are exactly {b,c}, stably ordered, with both matrices aligned to that order.

- [ ] **Step 2: Correct PCA terminology**

Rename the legacy PCA output arcface_pca64_sensitivity. Reject a purported independent encoder whose provenance parent hash equals the ArcFace embedding hash.

- [ ] **Step 3: Implement the confirmatory grid**

Use encoders arcface/sface; seeds 3001 through 3020; k=[32,64,128]; five grouped folds; MiniBatchKMeans n_init=10; largest rule.

- [ ] **Step 4: Implement compatibility**

Report per-encoder stability and matched seed/k cross-encoder ARI/Jaccard. Always join by ID and report common-universe coverage.

- [ ] **Step 5: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_run_encoder_comparison.py -q

### Task 3: Symmetric nested FairFace scenarios

**Files:**
- Create: face_profile_ml/fairface_scenarios.py
- Create: tests/test_fairface_scenarios.py
- Modify: scripts/run_fairface_replicated.py
- Create: configs/experiments/fairface_paired_confirmatory.json

**Interfaces:**
- fairface_quotas(categories) -> pd.DataFrame.
- build_paired_scenarios(catalog, replication, seed) -> pd.DataFrame.
- Output includes scenario, focal_category, direction, replication, sample_id, fairface_category, selection_rank.

- [ ] **Step 1: Write exact quota tests**

Assert baseline has 1,920 per category; each under scenario has 960 focal and 2,080 others; each over scenario has 3,840 focal and 1,600 others; all 15 scenarios total 13,440 with no duplicates.

- [ ] **Step 2: Write the nesting test**

For each replication/category, assert focal under IDs are a subset of baseline IDs, which are a subset of focal over IDs. Assert deterministic replay and different selections across replications.

- [ ] **Step 3: Verify current behavior fails**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_fairface_scenarios.py -q

- [ ] **Step 4: Implement prefix selection**

Derive per-category RNG seed from SHA-256 of base_seed, replication, and category. Validate pool capacity and name the insufficient category/count on failure.

- [ ] **Step 5: Cross factors**

Use identical clustering seeds [4001,4002] for every replication/scenario. Primary k=64. Sensitivity k=[32,64,128] for replications 0 through 9.

- [ ] **Step 6: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_fairface_scenarios.py -q

### Task 4: Paired FairFace contrasts and variability

**Files:**
- Create: face_profile_ml/fairface_analysis.py
- Create: tests/test_fairface_analysis.py
- Modify: face_profile_ml/variance_decomposition.py
- Modify: tests/test_variance_decomposition.py

**Interfaces:**
- paired_scenario_contrasts(predictions, baseline="balanced") -> pd.DataFrame.
- factor_variability(metrics, outcome, factors) -> pd.DataFrame.
- hierarchical_contrast_interval(contrasts, replication_col, seed_col, n_bootstrap, seed).

- [ ] **Step 1: Write a failing self-comparison test**

Assert baseline rows are absent from contrasts. Every perturbation reports intersection_n, intersection_fraction, delta_auc, target_jaccard, and partition_ari.

- [ ] **Step 2: Write a factor-identifiability test**

Inject independent scenario, replication, and seed effects into a balanced fixture; assert full-rank design and dominant scenario contribution. Supply nested seeds and assert rank-deficiency rejection.

- [ ] **Step 3: Implement ID-intersection contrasts**

Join on pseudonymous IDs, never positions. Record scenario, baseline, intersection sizes, and shared fraction.

- [ ] **Step 4: Implement variability and intervals**

Use categorical fixed effects with partial sums of squares on the balanced primary grid. Bootstrap replications, then seeds within sampled replications. Label outputs descriptive, not causal.

- [ ] **Step 5: Verify and smoke**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_fairface_scenarios.py tests/test_fairface_analysis.py tests/test_variance_decomposition.py -q
.\.venv311\Scripts\python.exe scripts/run_fairface_replicated.py --config configs/experiments/fairface_paired_confirmatory.json --n-replications 2 --seeds 4001,4002 --k-values 8 --folds 3 --max-samples-per-scenario 210 --out-dir artifacts/fairface_paired_smoke_v2
~~~

### Task 5: Packed compatibility and cached multiverse

**Files:**
- Create: face_profile_ml/compatibility.py
- Create: tests/test_compatibility.py
- Modify: scripts/run_multiverse.py
- Create: tests/test_run_multiverse.py
- Create: configs/experiments/multiverse_confirmatory.json

**Interfaces:**
- pack_targets(matrix: np.ndarray) -> PackedTargets.
- pairwise_jaccard(packed: PackedTargets) -> np.ndarray.
- Produces specification_curve.csv, canonical_compatibility.csv, factor-stratified compatibility, exact pairwise_jaccard.npz, and figures.

- [ ] **Step 1: Write exact packed-Jaccard tests**

Generate 17 non-byte-aligned boolean targets. Compare every packed result to direct NumPy intersection/union; assert symmetry, diagonal 1, and error below 1e-12.

- [ ] **Step 2: Write a fit-reuse grid test**

For two encoders, two backends, two k, two n_init, two seeds, three rules, and three folds, assert fit count is 96 and analytical specification-fold count is 288.

- [ ] **Step 3: Implement the grid**

Use encoders arcface/sface; backends minibatch/kmeans; k=[32,64,128]; n_init=[3,10,20]; seeds 5001 through 5010; five plausible rules; five folds.

- [ ] **Step 4: Reuse only compatible checkpoints**

Reuse by fit_id only when input, common-universe, fold, encoder-model, and implementation hashes match. Recompute missing/incompatible cells.

- [ ] **Step 5: Implement specification-curve reporting**

Sort by qualified pooled OOF AUC with stable spec_id tie-break. Plot AUC, prevalence, and canonical Jaccard. Annotate predeclared 0.85/0.10 thresholds without selecting a winning pipeline.

- [ ] **Step 6: Verify and smoke**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_compatibility.py tests/test_run_multiverse.py -q
.\.venv311\Scripts\python.exe scripts/run_multiverse.py --config configs/experiments/multiverse_confirmatory.json --seeds 5001,5002 --k-values 4,8 --n-init 3 --max-samples 240 --out-dir artifacts/multiverse_smoke_v2
~~~

