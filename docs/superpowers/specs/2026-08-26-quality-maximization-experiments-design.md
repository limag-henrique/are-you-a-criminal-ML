# Quality-Maximization Experiments and Manuscripts Design

**Date:** 2026-08-26  
**Status:** approved in chat; implementation not yet started  
**Primary manuscript:** `artigo/conteudo.tex`

## Objective

Strengthen the study's central empirical claim: many plausible, internally recoverable definitions of an endogenous facial-embedding target can identify mutually incompatible sets of records. The work must replace smoke-only evidence with traceable confirmatory runs, reconcile previously incomparable AUC values, broaden robustness across composition, clustering, and representation, audit image redistribution claims, and produce both an updated ABNT monograph and a separate anonymous ACM-style manuscript of at most fourteen content pages.

The project remains an audit of invalid construct inference. It does not build, validate, or endorse a criminality detector, infer demographic attributes for the wanted-person corpus, or treat embedding geometry as a social ground truth.

## Global constraints

- Preserve all pre-existing uncommitted work; never reset, discard, or overwrite unrelated changes.
- Work only on the current branch and never create a Git commit.
- Keep confirmatory, exploratory, smoke, and failed runs physically and semantically distinct.
- Never report smoke output as scientific evidence.
- Use deterministic configurations, atomic checkpoints, input hashes, model hashes, environment metadata, and resumable execution.
- Public outputs must exclude images, face embeddings, direct identifiers, absolute local paths, and inferred sensitive attributes.
- Treat ROC-AUC and PR-AUC as internal target-recovery metrics, never as construct, criterion, or external validity.
- Preserve the current caution that AUC 0.896 and Jaccard 0.004 come from different designs and are not a per-specification pair.
- Update Graphify with `graphify update .` after substantial exploration and implementation.

## Chosen architecture

The existing experiment scripts become thin entry points over a shared experiment kernel. The kernel owns configurations, deterministic folds, clustering fits, target selection, scoring, caching, provenance, aggregation, and artifact validation. A fit is keyed by the factors that can change the fitted partition; target rules reuse that fit rather than recomputing clustering.

The architecture is intentionally incremental. Existing public functions in `face_profile_ml` remain compatible where possible, while new experiment-specific contracts live in focused modules. A workflow-engine rewrite is out of scope.

### Module responsibilities

- `face_profile_ml/experiment_specs.py`: immutable, serializable specifications and canonical `spec_id`/`fit_id` hashes.
- `face_profile_ml/experiment_cache.py`: atomic fit/prediction checkpoints, completion states, resume validation, and cache invalidation by input/config hash.
- `face_profile_ml/experiment_runner.py`: grouped out-of-fold fitting, fit reuse across target rules, qualified metric names, and structured failure records.
- `face_profile_ml/experiment_reporting.py`: confirmatory summaries, uncertainty, compatibility matrices, and evidence tables.
- `face_profile_ml/encoders.py`: frozen encoder protocol and SFace implementation alongside the existing ArcFace artifacts.
- `face_profile_ml/licensing.py`: source-level redistribution evidence schema and conservative publication classification.
- `scripts/run_*.py`: argument parsing and calls into the shared kernel only.

### Public testing seams

Tests observe behavior at these public boundaries:

1. serialized experiment configuration to deterministic `fit_id` and `spec_id`;
2. input table plus embeddings to OOF predictions and structured run status;
3. completed artifacts to summary tables and figures;
4. source-license evidence to redistribution recommendation;
5. validated evidence tables to manuscript consistency checks.

Tests do not mock private clustering helpers or assert internal call order.

## Common artifact contract

Every confirmatory experiment directory contains:

- `config.json`: exact declared grid and confirmatory thresholds;
- `run_manifest.json`: timestamps, command, Python/package versions, CPU/RAM information, Git revision and dirty-state flag;
- `inputs.json`: relative input identifiers, SHA-256 hashes, shapes, dtypes, and model hashes;
- `fit_index.parquet`: one row per fit with `fit_id`, factors, fold, status, runtime, convergence metadata, and checkpoint path;
- `specification_metrics.parquet`: one row per analytical specification with qualified metrics and eligibility status;
- `oof_predictions.parquet`: pseudonymous sample/group/fold IDs, target, raw score, calibrated probability, and specification identifiers where storage is tractable;
- experiment-specific compatibility and contrast tables;
- `failures.csv`: explicit reason codes rather than silently missing values;
- `completion.json`: expected/completed/failed counts, validation status, and a final content hash.

An experiment is citable only when `completion.json` says `complete`, all expected cells are either successful or explicitly ineligible, hashes validate, and the public privacy scan passes.

## Metric vocabulary

Every metric name carries its evaluation protocol. The principal recovery metric is `oof_pooled_cluster_recovery_roc_auc`: pooled OOF ROC-AUC against fold-specific membership in the target cluster learned from the complementary training fold. Per-fold AUC is diagnostic only and is undefined when a test fold contains one class.

The calibrated probability is used for Brier score and calibration diagnostics. Because logistic calibration is monotone, ROC-AUC and PR-AUC are also computed from the raw margin and checked for rank equivalence. The manuscript explains that the target and score share geometry, making high recoverability expected and scientifically distinct from external validity.

Compatibility is reported with Jaccard on target membership and ARI on full partitions. Comparisons state their record universe. For partially overlapping FairFace scenarios, metrics are computed only on the intersection and always report `intersection_n` and `intersection_fraction`.

## Experiment 1: target-rule ablation

### Dataset and design

Use the primary wanted-person corpus with 9,482 valid embeddings and its audited `group_id`. The confirmatory grid is:

- `k = {32, 64, 128}`;
- 20 declared clustering seeds;
- five grouped folds;
- five plausible target rules: `largest`, `compact`, `central`, `outlier`, and `isolated`;
- `random` as a seeded null-control rule, excluded from substantive success counts.

`isolated` selects the cluster whose centroid maximizes its minimum distance to every other cluster centroid. This is distinct from `outlier`, which maximizes distance from the global sample centroid. The current `separated` alias is deprecated rather than counted as a separate rule.

Clustering is fit once per `(encoder, backend, n_init, k, seed, fold)` and reused across rules. The random-rule seed is a separate declared factor derived from, but not equal to, the clustering seed.

### Outcomes

For every `(rule, seed, k)` specification, pool OOF predictions and report recovery ROC-AUC, PR-AUC, Brier score, prevalence, eligibility, and target size. Compute pairwise Jaccard between rules on aligned records.

The preregistered descriptive thresholds are:

- high internal recoverability: pooled ROC-AUC at least 0.85;
- strong target incompatibility: pairwise Jaccard below 0.10.

The main result reports the fraction of plausible rules meeting the recovery threshold and the distribution of pairwise incompatibility. Thresholds organize interpretation; continuous estimates and uncertainty remain primary.

## Experiment 2: paired and replicated FairFace composition

### Symmetric scenarios

Use only the seven historical categories supplied by FairFace. No label is inferred or transferred. Use a fixed total of 13,440 records per scenario, which is supported without replacement by the smallest available category pool in the local embedding cache.

- Balanced baseline: 1,920 records per category.
- For each of seven categories, one underrepresentation scenario: 960 focal records and 2,080 from each other category.
- For each category, one overrepresentation scenario: 3,840 focal records and 1,600 from each other category.

This yields one baseline plus fourteen symmetric perturbations.

### Pairing and replication

Run 50 sampling replications. Within each replication, create one seeded permutation per category and select prefixes of those permutations for every scenario. This nested construction maximizes record pairing: the smaller quota is a subset of the larger quota for the same category. Use the same two clustering seeds in every replication so sampling replication and algorithmic seed are crossed rather than aliased.

The primary analysis fixes `k=64`. A predeclared sensitivity uses `k={32,64,128}` on replications 0--9. Five record-level folds are used, with the limitation that FairFace lacks verified identity groups stated explicitly.

### Estimands and summaries

For each perturbation, compute paired scenario-minus-baseline contrasts for recovery AUC, prevalence, target size, ARI, and target Jaccard on the shared-record intersection. Save intersection size, intersection fraction, and category composition of the target.

Composition, sampling replication, clustering seed, and `k` are not described as causal effects. Their attributable variability is summarized with a balanced fixed-factor model and partial sums of squares, while uncertainty for scenario contrasts comes from a hierarchical bootstrap over replication and seed. Scenario-baseline self-comparisons are excluded.

The current A--D FairFace analysis remains explicitly exploratory and moves to supplementary material.

## Experiment 3: clustering and `n_init` robustness

On the primary corpus, compare:

- `MiniBatchKMeans` and conventional `KMeans`;
- `n_init = {3, 10, 20, 50}`;
- 20 seeds;
- `k=64`;
- five grouped folds;
- the canonical `largest` rule.

Report within-backend stability across seeds, between-backend compatibility at matched seeds and `n_init`, target prevalence, fit runtime, inertia, convergence, and internal recovery metrics. The confirmatory interpretation centers on target stability, not the nearly tautological in-sample geometry. GMM is an exploratory appendix analysis at a reduced grid. Agglomerative clustering is excluded because it ignores seeds and has unsuitable quadratic scaling at this sample size.

## Experiment 4: independent encoder

Use OpenCV SFace as the second frozen encoder. It differs from the current ArcFace ResNet-50 representation in backbone and loss family, runs through OpenCV on CPU, and has a compact official ONNX artifact. Record the download URL, SHA-256, preprocessing, output dimension, OpenCV version, and the model-directory Apache-2.0 statement. Also disclose that the exact training-data provenance of the distributed ONNX weight is not fully documented.

Extract SFace vectors once from the same aligned primary face crops used for ArcFace, without reusing ArcFace embeddings or fitting PCA. Failed records receive explicit status; confirmatory cross-encoder comparisons use only the common successfully encoded record universe and preserve the audited grouping.

The grid is two encoders by 20 seeds by `k={32,64,128}` by five folds using the canonical rule. Report per-encoder recovery and stability, plus cross-encoder Jaccard/ARI on aligned records. PCA-64 remains a within-ArcFace sensitivity and is never called an independent encoder.

## Experiment 5: multiverse and specification curve

The multiverse reuses completed fits from the prior experiments. Its compact confirmatory grid is:

- encoders: ArcFace and SFace;
- clustering: MiniBatchKMeans and KMeans;
- `k = {32,64,128}`;
- `n_init = {3,10,20}`;
- 10 declared seeds;
- target rules: the five plausible rules above.

This defines 360 fit cells and 1,800 analytical specifications per fold-aware experiment. Rules share fits. The `n_init=50` clustering result remains in the dedicated robustness experiment but is excluded here to control cost.

The specification curve orders cells by pooled OOF recovery AUC and shows prevalence and compatibility without implying independent observations. Report compatibility against a canonical ArcFace/MiniBatch/k64/n_init10/largest specification, exact within-factor contrasts, and an exact vectorized pairwise Jaccard matrix stored as packed boolean arrays. Avoid millions of pandas joins.

The confirmatory statement is the joint distribution of high recovery and low compatibility, not a selected best pipeline. The AUC 0.85 and Jaccard 0.10 thresholds are declared before reading confirmatory results.

## Experiment 6: reconcile the AUC discrepancy

Re-run `group_id` thresholds `0.995`, `0.997`, `0.999`, and `0.9995` through the same shared OOF runner, using the same embeddings, canonical rule, `k`, seed, score, calibration, pooled aggregation, and metric implementation as the primary AUC 0.896 protocol. Save a protocol-comparison table that changes one factor at a time and identifies the first factor that reproduces the 0.979--0.984 range.

The manuscript must do one of two things based on evidence:

1. if threshold alone explains the change, report it as genuine grouping sensitivity under an otherwise identical protocol; or
2. if another factor explains it, rename the old metric and state explicitly that the values were not protocol-comparable.

No explanation is written before the diagnostic artifacts determine which case holds.

## Experiment 7: uncertainty

Retain the existing 2,000-resample grouped/fold-preserving bootstrap of stored OOF predictions, label it as conditional on the fitted fold solutions, and state that it excludes training/clustering variability.

Add a limited repeated grouped refit robustness on the primary canonical specification: five deterministic repeated group allocations times five folds. Report between-refit dispersion separately. Do not merge the conditional bootstrap and refit dispersion into a single interval without a justified hierarchical procedure.

## Literature positioning

Add and cite at least:

- Li et al. (2026), *The Proxy Presumption: From Semantic Embeddings to Valid Social Measures*, ACL 2026, DOI `10.18653/v1/2026.acl-long.1048`;
- Seppälä and Hirvonen (2026), *Facial Analysis AI as Social Pseudotechnology*, DOI `10.1007/s13194-026-00732-1`.

The positioning distinguishes construct-validity protocols for embedding-derived social measures from this project's contribution: endogenous labels, a constructive proof, empirical non-uniqueness, computational stability, sociotechnical selection, and an executable facial-analysis audit.

## Image licensing and redistribution audit

Create a source-level evidence table for FBI, Interpol, EU Most Wanted, DEA, Brazilian police sources, Kaggle packaging, FairFace, and every other image source present in the manifest. Each row records source URL, access date, asserted license, copyright owner if stated, redistribution permission, database terms, privacy/biometric concerns, evidence excerpt hash, and review status.

Classification is conservative:

- `redistributable`: explicit source-level authority covers the image and redistribution mode;
- `metadata_only`: collection/use may be documented, but relicensing or redistribution authority is absent or ambiguous;
- `restricted`: terms prohibit the intended redistribution;
- `unknown`: evidence is insufficient.

A Kaggle CC0 label never overrides upstream rights. Unless every redistributed photograph has adequate source-level authority, the recommended release contains code, hashes, pseudonymized manifests, aggregate statistics, and acquisition instructions, but no photographs. This task changes documentation and release manifests; it does not destructively delete local research data.

## Manuscript deliverables

### ABNT monograph

Update `artigo/conteudo.tex` and `artigo/referencias.bib` from validated artifacts only. Replace pending-language with results only for completed experiments. Correct stale artifact paths. Explain conditional uncertainty, the AUC reconciliation, target-rule non-uniqueness, paired FairFace findings, clustering/encoder dependence, multiverse results, and licensing conclusions.

The prose remains human, constructive, and methodologically positive. It emphasizes what the audit establishes and what future systems should validate, without sensationalism or anthropomorphic claims.

### Anonymous ACM-style article

Create an independent `artigo-acm/` document using `acmart` review/anonymous layout. It does not `\input` the ABNT monograph. It contains a concise, self-contained narrative, figures and tables generated from the same validated artifacts, anonymous data/code availability language, ethics, limitations, and an AI-use statement appropriate to the user's non-submission context.

The compiled main content, including figures and tables, must not exceed fourteen pages; references are outside that limit. Author-identifying metadata, university, DOI, ISBN, repository links, acknowledgements, CRediT, and identifying positionality are excluded from the anonymous build.

## Execution strategy

The laptop is CPU-only with approximately 8 GiB RAM. Runs execute sequentially with at most one memory-intensive fit worker. The order is:

1. establish passing baseline tests and environment;
2. implement the shared kernel and validate smoke fixtures;
3. run target ablation and AUC reconciliation;
4. run clustering robustness;
5. extract and validate SFace embeddings;
6. run independent-encoder comparisons;
7. run paired FairFace with resume checkpoints;
8. run the cached multiverse;
9. aggregate evidence and update manuscripts.

Every long run records progress and may be resumed after interruption. A runtime benchmark is recorded before each confirmatory grid. The grid is never reduced after inspecting scientific outcomes; any resource-driven deviation is decided from timing/memory evidence, documented, and labeled.

## Error handling and scientific safeguards

- Single-class folds/specifications become `ineligible_single_class`, not empty cells.
- Missing embeddings are reported and common-universe comparisons are recomputed explicitly.
- Non-convergence, memory errors, corrupt checkpoints, hash mismatches, and interrupted fits have distinct statuses.
- A partial run cannot generate a confirmatory summary.
- Figures and manuscript tables are generated from machine-readable summaries, not manually transcribed values.
- Assertions check totals, uniqueness, fold group disjointness, factor crossing, scenario quotas, pairing, metric bounds, symmetric compatibility, and absence of sensitive public fields.
- Confirmatory thresholds and grids live in configuration files committed only by a human.

## Verification

Verification includes focused red-green tests for each public seam, the full `tests/` suite, both `research_audit_v2` suites, smoke integration runs, artifact schema/hash checks, public-output privacy scans, manuscript value-to-artifact consistency checks, BibTeX resolution, LaTeX compilation, PDF page-count/visual inspection, `git diff --check`, and `graphify update .`.

No Git commit is created automatically.
