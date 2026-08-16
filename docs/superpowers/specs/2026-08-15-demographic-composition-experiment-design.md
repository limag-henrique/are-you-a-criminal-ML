# Demographic Composition Experiment Design

## Objective

Measure whether controlled changes in FairFace demographic composition alter clustering, the largest-cluster target, target prevalence, stability, or internal recovery metrics. The experiment remains isolated from the main pipeline and never links FairFace records to project records or infers demographic labels.

## Dataset and categories

- Dataset: local FairFace release documented in `docs/external_benchmark_inventory.md`.
- Demographic variable: the seven `source_race_label` categories supplied by FairFace.
- Perturbed group: `Middle Eastern`, selected because it has the lowest original prevalence.
- Every scenario contains 36,456 distinct images sampled without replacement.
- Scenario A preserves the original FairFace proportions through deterministic quota allocation.
- Scenario B contains 5,208 images per category.
- Scenario C contains 2,604 `Middle Eastern` images (7.14%) and 5,642 images from each other category.
- Scenario D contains 10,416 `Middle Eastern` images (28.57%) and 4,340 images from each other category.
- Deterministic same-group reserve lists replace images that fail embedding extraction while preserving quotas.

## Isolation and shared parameters

New code and outputs live under `research_audit_v2/demographic_composition/`. Existing main-pipeline modules are imported read-only; no main-pipeline source, manifest, record, image, or embedding is modified or joined to FairFace.

One extraction pass computes ArcFace embeddings for the union of selected images. The official `margin025` release is already cropped and aligned with `dlib.get_face_chip()`; those aligned crops are therefore passed directly to the ArcFace recognizer in deterministic batches. This avoids a second detector whose default threshold rejected the tightly framed crops. All scenarios use the same model, upstream alignment, L2 normalization, MiniBatchKMeans implementation, `n_init`, batch size, iteration limit, seed list, `k` list, target rule, scoring rule, and evaluation method. Only scenario membership and therefore demographic composition vary.

## Analysis flow

For each scenario and each declared `(seed, k)` run:

1. Load the selected FairFace embeddings and L2-normalize them.
2. Fit MiniBatchKMeans.
3. Select the largest cluster; ties use the lowest numeric label.
4. Score observations by cosine similarity to the target-cluster centroid.
5. Evaluate the score against cluster-derived target membership.

ROC-AUC and PR-AUC use grouped cross-fitting: representations, clustering, target selection, centroid, calibration, and thresholds are fit only on training observations. FairFace has no repeated identity field, so each source record is its own group and this limitation is stated. Metrics quantify only internal recovery of a synthetic clustering-derived target.

## Comparisons

- Target size and prevalence for every scenario, seed, and `k`.
- Pairwise ARI and target Jaccard across seeds within each scenario and `k`.
- Pairwise scenario comparisons on the intersection of record IDs only, preventing unequal record universes from invalidating ARI/Jaccard.
- Cross-fitted ROC-AUC and PR-AUC when both target classes occur.
- Counts and proportions of every FairFace category within every cluster, plus target-cluster composition.
- Sensitivity summaries over the identical declared seed and `k` grid.
- Absolute and relative changes from scenario A, with distributional summaries rather than cherry-picked runs.

A relevant change is declared before results as any of: median pairwise ARI below 0.90, median target Jaccard below 0.80, target-prevalence absolute change of at least 0.02, or ROC-AUC/PR-AUC absolute change of at least 0.03. Continuous estimates and dispersion are still reported so the thresholds do not replace judgment.

## Reproducibility and outputs

The experiment writes a machine-readable configuration, input hashes, quota/selection manifests, extraction failures, run-level metrics, stability tables, cluster-composition tables, SVG figures, and a completion manifest. Public reports contain no images, embeddings, local absolute paths, or inferred attributes.

The root deliverable `DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md` embeds comparative tables, links the generated figures, records limitations and deviations, and ends with an objective answer about material changes.

## Validation

Tests are written before implementation and verify exact quotas, determinism, no duplicated records, disjoint scenario calculations, reserve replacement, largest-cluster tie handling, intersection-based stability, undefined ROC/PR behavior for single-class folds, output privacy, and non-modification of main-pipeline files. A smoke run precedes the complete experiment; the final report is generated only from completed, validated artifacts.

## Ethical limits

FairFace labels are dataset-provided historical categories, not self-identification or biological ground truth. Results support no inference about identity, criminality, behavior, risk, or group superiority. No association is made between FairFace individuals and any person or record in the main dataset.
