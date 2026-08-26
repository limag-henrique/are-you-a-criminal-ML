# Confirmatory Execution and Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans because long runs share checkpoints and execute sequentially on this machine. Do not commit automatically.

**Goal:** Execute every declared experiment without post-result grid changes, freeze evidence, compile both manuscripts, inspect PDFs, and leave an auditable worktree.

**Architecture:** A preflight gate validates code/config/input hashes. Long CPU-only jobs run one at a time with resume checkpoints. A final evidence manifest binds manuscript values to complete experiment bundles.

**Tech Stack:** PowerShell, Python 3.11, pytest, LaTeX/Tectonic, PDF rendering, Graphify.

**Spec:** docs/superpowers/specs/2026-08-26-quality-maximization-experiments-design.md

## Global Constraints

- Complete the preceding three plans first.
- Never change a confirmatory grid after reading its scientific result.
- Resource deviations require a pre-result benchmark and new reviewed config hash.
- Run one memory-intensive experiment at a time on the 8 GiB CPU-only host.
- Never commit or switch branches.

---

### Task 1: Preflight and evidence-freeze gates

**Files:**
- Create: scripts/preflight_confirmatory.py
- Create: scripts/freeze_scientific_evidence.py
- Create: tests/test_confirmatory_gates.py

**Interfaces:**
- Preflight exits zero only when environment, configs, inputs, model hashes, disk, and baseline tests pass.
- Freeze writes artifacts/confirmatory_evidence_manifest.json only when all required bundles validate.

- [ ] **Step 1: Write failing missing-bundle/hash tests**

Assert preflight reports every missing requirement in one run. Assert freeze refuses partial bundles, smoke paths, mismatched hashes, dirty generated tables, and unknown analysis tiers.

- [ ] **Step 2: Implement preflight**

Require Python 3.11 or 3.12, imports for NumPy/pandas/sklearn/SciPy/PyArrow/OpenCV, at least 10 GiB free disk, valid model hashes, readable inputs, and exact configs. Report current RAM as warning.

- [ ] **Step 3: Implement freeze**

Bind experiment name, config/input/completion hashes, table/figure hashes, and timestamp. Manuscript generation consumes only this file.

- [ ] **Step 4: Verify**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_confirmatory_gates.py -q
.\.venv311\Scripts\python.exe scripts/preflight_confirmatory.py
~~~

### Task 2: Pre-result runtime benchmark

**Files:**
- Generate: artifacts/logs/confirmatory_runtime_benchmarks.json

- [ ] **Step 1: Benchmark representative work**

Run one fold for MiniBatch and KMeans at n_init=3,10,20,50 on the primary corpus; one MiniBatch FairFace fold at 13,440 rows; and SFace on 100 images. Suppress scientific result summaries.

- [ ] **Step 2: Record extrapolation**

Store seconds, peak working set, n, dimensions, backend, and full-grid wall-time estimate. Change only batch size/worker count for resource reasons; scientific factor changes require review before results are read.

### Task 3: Core confirmatory runs

**Files:**
- Generate: artifacts/ablation_confirmatory_v2/
- Generate: artifacts/groupid_reconciliation_confirmatory/
- Generate: artifacts/clustering_robustness_confirmatory/
- Generate: artifacts/repeated_refit_confirmatory/

- [ ] **Step 1: Run ablation**

~~~powershell
.\.venv311\Scripts\python.exe scripts/run_target_ablation.py --config configs/experiments/target_ablation_confirmatory.json --out-dir artifacts/ablation_confirmatory_v2 --resume
~~~

- [ ] **Step 2: Validate ablation**

Require exact fit/spec counts, no missing cells, privacy pass, and completion hash.

- [ ] **Step 3: Run AUC reconciliation**

~~~powershell
.\.venv311\Scripts\python.exe scripts/run_groupid_sensitivity.py --config configs/experiments/groupid_reconciliation_confirmatory.json --out-dir artifacts/groupid_reconciliation_confirmatory --resume
~~~

- [ ] **Step 4: Run clustering robustness**

~~~powershell
.\.venv311\Scripts\python.exe scripts/run_clustering_comparison.py --config configs/experiments/clustering_robustness_confirmatory.json --out-dir artifacts/clustering_robustness_confirmatory --resume
~~~

- [ ] **Step 5: Run five grouped refits**

Use the canonical primary specification and write separate refit-dispersion output. Do not combine it with conditional bootstrap intervals.

### Task 4: Encoder and FairFace runs

**Files:**
- Generate: private SFace cache
- Generate: artifacts/encoder_comparison_confirmatory/
- Generate: artifacts/fairface_paired_confirmatory_v2/

- [ ] **Step 1: Acquire/verify SFace**

Download only from the official config URL, hash it, and abort if it differs from the reviewed expected value.

- [ ] **Step 2: Extract SFace**

Run resumable extraction over 9,482 aligned primary records and validate common-universe coverage.

- [ ] **Step 3: Run encoder comparison**

~~~powershell
.\.venv311\Scripts\python.exe scripts/run_embedding_decoupling.py --config configs/experiments/encoder_comparison_confirmatory.json --out-dir artifacts/encoder_comparison_confirmatory --resume
~~~

- [ ] **Step 4: Run paired FairFace**

~~~powershell
.\.venv311\Scripts\python.exe scripts/run_fairface_replicated.py --config configs/experiments/fairface_paired_confirmatory.json --out-dir artifacts/fairface_paired_confirmatory_v2 --resume
~~~

- [ ] **Step 5: Validate design**

Assert 50 replications, 15 scenarios, identical seed pair, exact quotas, nested sets, no baseline self-comparisons, and complete primary/sensitivity cells.

### Task 5: Cached multiverse run

**Files:**
- Generate: artifacts/multiverse_confirmatory_v2/

- [ ] **Step 1: Dry-run cache plan**

List reused/missing fit IDs. Confirm 360 fit-factor cells and 1,800 analytical specifications before fold expansion.

- [ ] **Step 2: Execute with resume**

~~~powershell
.\.venv311\Scripts\python.exe scripts/run_multiverse.py --config configs/experiments/multiverse_confirmatory.json --out-dir artifacts/multiverse_confirmatory_v2 --resume
~~~

- [ ] **Step 3: Validate outputs**

Check exact matrix symmetry/diagonal, canonical comparisons, factor summaries, curve ordering, and figure hashes.

### Task 6: Freeze evidence and build manuscripts

**Files:**
- Generate: artifacts/confirmatory_evidence_manifest.json
- Generate: artigo/generated/*
- Generate: artigo/main.pdf
- Generate: artigo-acm/main.pdf

- [ ] **Step 1: Freeze evidence**

Run: .\.venv311\Scripts\python.exe scripts/freeze_scientific_evidence.py

- [ ] **Step 2: Generate LaTeX evidence**

Run: .\.venv311\Scripts\python.exe scripts/build_manuscript_evidence.py

- [ ] **Step 3: Compile ABNT**

Use the verified LaTeX engine for the full BibTeX cycle. Require no undefined citations/references or missing figures.

- [ ] **Step 4: Compile ACM style**

Require anonymous review mode and content-end page at or below 14.

- [ ] **Step 5: Inspect both PDFs visually**

Inspect title/anonymity, abstract, headline tables, specification curve, FairFace figure, ethics/licensing, bibliography, clipping, overflow, fonts, and blank pages. Record pages/findings in artifacts/logs/pdf_visual_verification.md.

### Task 7: Final repository verification

- [ ] **Step 1: Run all tests**

~~~powershell
.\.venv311\Scripts\python.exe -m pytest tests research_audit_v2/tests research_audit_v2/second_phase/tests -q
~~~

- [ ] **Step 2: Run integrity checks**

Validate evidence hashes, table provenance, manuscript macro equality, privacy, license schema, ACM anonymity, and page count.

- [ ] **Step 3: Update Graphify**

Run: graphify update .

- [ ] **Step 4: Inspect final diff**

~~~powershell
git diff --check
git status --short --branch
git diff --stat
~~~

- [ ] **Step 5: Prepare human handoff**

Report completed experiments, artifact paths, headline findings, test totals, PDF page counts, unresolved rights/provenance limits, runtime, and suggested human commit grouping. Do not commit.

