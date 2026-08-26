# Licensing, Literature, and Manuscripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Do not create commits.

**Goal:** Produce a source-level redistribution audit, integrate the nearest 2026 literature, update the ABNT monograph from validated evidence, and create an anonymous ACM-style article capped at fourteen content pages.

**Architecture:** Structured evidence drives conservative rights classifications and generated manuscript tables. LaTeX claims are checked against artifact values. The ACM-style document is independent from the ABNT source but shares validated tables and figures.

**Tech Stack:** Markdown/CSV/JSON, Python, BibTeX, abnTeX2, acmart, latexmk or Tectonic, PDF visual verification.

**Spec:** docs/superpowers/specs/2026-08-26-quality-maximization-experiments-design.md

## Global Constraints

- Use primary or official sources for legal terms, papers, datasets, and format.
- The rights audit is evidence gathering, not legal advice.
- Do not delete local images or publish sensitive data.
- Insert numbers only from complete hash-validated bundles.
- Keep prose human, constructive, and methodologically positive.
- Never commit automatically.

---

### Task 1: Licensing evidence contract and source audit

**Files:**
- Create: face_profile_ml/licensing.py
- Create: tests/test_licensing.py
- Create: docs/licensing/source_evidence.csv
- Create: docs/licensing/redistribution_audit.md

**Interfaces:**
- classify_redistribution(evidence: SourceEvidence) -> str returns redistributable, metadata_only, restricted, or unknown.
- Evidence columns: source_id, source_name, source_url, accessed_at, asserted_license, copyright_owner, redistribution_permission, database_terms, privacy_risk, evidence_sha256, status, notes.

- [ ] **Step 1: Write failing conservative tests**

~~~python
def test_kaggle_cc0_does_not_override_unknown_upstream_rights() -> None:
    evidence = SourceEvidence(asserted_license="CC0", upstream_permission=None)
    assert classify_redistribution(evidence) == "metadata_only"
~~~

Also assert explicit owner permission becomes redistributable, explicit prohibition becomes restricted, and absent evidence becomes unknown.

- [ ] **Step 2: Implement classifier/schema validation**

Reject unknown statuses and evidence rows without source URL, access date, or evidence hash. Treat aggregator licenses as insufficient when upstream authority is absent.

- [ ] **Step 3: Inventory sources from repository facts**

Use rg across manifests, scrape scripts, benchmark inventory, and dataset docs. Assign stable source IDs. Include FBI, Interpol, EU Most Wanted, DEA, Brazilian police sources, Kaggle package, FairFace, BFW, and every discovered source.

- [ ] **Step 4: Research official terms**

Record direct official pages and SHA-256 of captured textual evidence. Paraphrase terms and keep quotations minimal. Separate copyright, database rights, privacy, biometric risk, and research ethics.

- [ ] **Step 5: Generate the audit**

Recommend metadata-only release whenever upstream redistribution authority is ambiguous. State that a Kaggle CC0 label cannot create missing upstream rights. Do not delete local research files.

- [ ] **Step 6: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_licensing.py research_audit_v2/tests/test_privacy.py -q

### Task 2: Primary-source 2026 literature

**Files:**
- Modify: artigo/referencias.bib
- Create: docs/research/2026-embedding-construct-validity.md
- Create: tests/test_bibliography_entries.py

**Interfaces:**
- BibTeX keys li2026proxypresumption and seppala2026pseudotechnology.

- [ ] **Step 1: Write failing metadata tests**

Assert exact DOIs 10.18653/v1/2026.acl-long.1048 and 10.1007/s13194-026-00732-1, year 2026, exact titles, and author surnames.

- [ ] **Step 2: Verify red**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_bibliography_entries.py -q

- [ ] **Step 3: Add verified entries**

Use official ACL Anthology BibTeX and Springer metadata. Preserve accents with BibTeX-safe forms.

- [ ] **Step 4: Write the positioning note**

For each work record its question, method, relevance, overlap, difference, and supported manuscript sections. Distinguish construct-validity protocol from endogenous-label non-uniqueness and philosophical pseudotechnology analysis from the executable audit.

- [ ] **Step 5: Verify green**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_bibliography_entries.py -q

### Task 3: Generated evidence and claim consistency

**Files:**
- Create: scripts/build_manuscript_evidence.py
- Create: tests/test_manuscript_evidence.py
- Generate: artigo/generated/experiment_macros.tex
- Generate: artigo/generated/*.tex
- Generate: artigo/generated figures
- Generate: artigo/generated/evidence_manifest.json

**Interfaces:**
- CLI consumes only the frozen evidence manifest and complete bundles.
- Every macro maps to bundle, table, row key, column, and content hash.

- [ ] **Step 1: Write a failing incomplete-bundle test**

Given completion.status="partial", assert nonzero exit and no TeX output.

- [ ] **Step 2: Write exact formatting tests**

Given AUC 0.89567, require 0.896. Require fixed three-decimal intervals and a source spec_id.

- [ ] **Step 3: Implement atomic generation**

Generate macros for counts, qualified metrics, intervals, rule results, compatibility, FairFace contrasts, clustering, encoders, multiverse, and AUC reconciliation.

- [ ] **Step 4: Correct stale paths**

Replace vague artifact references with actual validated paths. Test that every local \path reference in artigo/conteudo.tex resolves from repository root.

- [ ] **Step 5: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_manuscript_evidence.py -q

### Task 4: Update the ABNT monograph

**Files:**
- Modify: artigo/conteudo.tex
- Modify: artigo/main.tex only for required generated inputs/packages
- Modify: artigo/README.md
- Create: tests/test_manuscript_claims.py

**Interfaces:**
- artigo/conteudo.tex imports artigo/generated/experiment_macros.tex and generated tables.

- [ ] **Step 1: Write failing manuscript assertions**

Require both 2026 citations, conditional OOF uncertainty, internal-recovery vocabulary, rights conclusion, and absence of pending language for completed experiments. Reject smoke directory names as evidence.

- [ ] **Step 2: Revise methods**

Document reusable fits, five distinct rules plus null control, symmetric nested FairFace, crossed factors, n_init, SFace, multiverse, qualified AUC, and uncertainty scopes.

- [ ] **Step 3: Revise results from macros only**

Report empirical non-uniqueness, paired FairFace contrasts, backend/initialization robustness, cross-encoder dependence, multiverse distribution, and AUC reconciliation. Use generated macros for every headline number.

- [ ] **Step 4: Revise discussion, ethics, and limitations**

Position against Li et al. and Seppälä/Hirvonen. State SFace provenance limits, FairFace category limits, conditional bootstrap scope, absence of causal demographic inference, and redistribution recommendation.

- [ ] **Step 5: Edit for constructive tone**

Frame results as safeguards for valid measurement: external construct justification, predeclared targets, stability, representation robustness, and rights-aware release. Remove repetition without weakening the conclusion that facial geometry cannot validate criminality.

- [ ] **Step 6: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_manuscript_claims.py tests/test_manuscript_evidence.py tests/test_bibliography_entries.py -q

### Task 5: Anonymous fourteen-page ACM-style article

**Files:**
- Create: artigo-acm/main.tex
- Create: artigo-acm/references.bib
- Create: artigo-acm/sections/01-introduction.tex
- Create: artigo-acm/sections/02-related-work.tex
- Create: artigo-acm/sections/03-method.tex
- Create: artigo-acm/sections/04-results.tex
- Create: artigo-acm/sections/05-discussion.tex
- Create: artigo-acm/sections/06-ethics-limitations.tex
- Create: artigo-acm/sections/07-conclusion.tex
- Create: artigo-acm/README.md
- Create: tests/test_acm_anonymity_and_length.py

**Interfaces:**
- Uses \documentclass[manuscript,screen,review,anonymous]{acmart}.
- Compiled content ends at label content-end on page 14 or earlier.

- [ ] **Step 1: Write failing structure/anonymity tests**

Reject author name, UFMG, ISBN, DOI, repository/Kaggle URLs, acknowledgements, CRediT, identifying local paths, and nonanonymous class options. Require ethics, limitations, data/code availability, and AI-use disclosure.

- [ ] **Step 2: Create an independent article**

Do not input artigo/conteudo.tex. Copy generated evidence through an explicit build step. Keep references outside the content count.

- [ ] **Step 3: Write the concise narrative**

Allocate about 1.25 pages introduction, 1.5 related work, 3 method/formalism, 3.5 results, 2 discussion, 1.5 ethics/limitations, and 0.5 conclusion, leaving space for tables/figures.

- [ ] **Step 4: Add the page marker**

Place \label{content-end} immediately before the bibliography. Parse the label page from the aux file and fail above 14.

- [ ] **Step 5: Verify**

Run: .\.venv311\Scripts\python.exe -m pytest tests/test_acm_anonymity_and_length.py -q

