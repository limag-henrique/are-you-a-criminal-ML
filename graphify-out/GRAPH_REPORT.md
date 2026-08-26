# Graph Report - ia-uspJailer  (2026-08-25)

## Corpus Check
- 169 files · ~854,187 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1268 nodes · 2664 edges · 81 communities (50 shown, 31 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a1ed5478`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cli.py
- GallerySimilarityScorer
- scan_public_tree
- run_experiment.py
- target_rules.py
- run_grouped_cluster_cv
- run_second_phase.py
- cross_fitting.py
- preprocess_faces.py
- standardize_faces.py
- extract_union_embeddings
- stability.py
- recover_rejected_passport.py
- choose_target_cluster
- run_all.py
- scrape_dea_fugitives.js
- Face Profile ML
- scrape_interpol.py
- write_csv
- source_enrichment
- scrape_fbi_criminals.py
- build_semantic_manifest.py
- Restrições globais
- Experimento de composição demográfica
- Face Profile ML - Relatorio final
- bootstrap_metric
- audit_data_lineage
- Final reproduction report
- fetch_research_benchmarks.py
- Padronizacao de faces
- Demographic Composition Experiment Design
- reconcile
- scrape_eu_most_wanted.py
- validate_research_benchmarks.py
- Implementação das melhorias para submissão FAccT
- synthetic_geometry_control
- Global Constraints
- Manuscript update
- scan_public_outputs
- build_research_benchmark_catalogs.py
- Protocolo de avaliação de equidade e validade
- Plano resumido da segunda fase
- predictive_models
- Monografia ABNT - projeto pronto para Overleaf
- Inventário reprodutível de benchmarks externos
- Plano de auditoria de pesquisa v2
- Research audit v2
- Second-phase final report
- sanitize_filename
- Documentação do Projeto
- extract_photos.py
- filter_dataset.py
- plot_performance_stability_paradox.py
- Benchmarks externos para auditoria acadêmica
- register_model_hash.py
- face_profile_ml/__init__.py
- demographic_composition/__init__.py
- research_audit_v2/__init__.py
- limitations_detected.md
- SECOND_PHASE_GAP_ANALYSIS.md
- second_phase/__init__.py
- deviation_log.md
- EDITORIAL_IMPACT_SUMMARY.md
- embedding_representation_report.md
- face_preprocessing_specification.md
- leakage_and_cross_fitting_report.md
- MANUSCRIPT_REQUIRED_CHANGES.md
- minibatch_sensitivity_report.md
- missingness_and_failure_report.md
- negative_controls_report.md
- numerical_reproducibility_report.md
- preprocessing_sensitivity_report.md
- privacy_scan_report.md
- program_corrections.md
- source_influence_report.md
- second_phase/README.md
- second_phase/src/__init__.py
- research_audit_v2/src/__init__.py
- fairness/README.md
- face-profile-ml

## God Nodes (most connected - your core abstractions)
1. `GallerySimilarityScorer` - 39 edges
2. `write_csv()` - 38 edges
3. `run_audit()` - 26 edges
4. `l2_normalize()` - 23 edges
5. `run_experiment()` - 22 edges
6. `atomic_write_json()` - 22 edges
7. `scan_public_tree()` - 22 edges
8. `main()` - 22 edges
9. `run_fold()` - 21 edges
10. `choose_target_cluster()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `AppHandler` --uses--> `ScoreCalibrator`  [INFERRED]
  scripts/serve_similarity_app.py → face_profile_ml/calibration.py
- `GallerySimilarityScorer` --uses--> `ScoreCalibrator`  [INFERRED]
  scripts/serve_similarity_app.py → face_profile_ml/calibration.py
- `SimilarityScorer` --uses--> `ScoreCalibrator`  [INFERRED]
  scripts/serve_similarity_app.py → face_profile_ml/calibration.py
- `SimilarityThresholds` --uses--> `ScoreCalibrator`  [INFERRED]
  scripts/serve_similarity_app.py → face_profile_ml/calibration.py
- `FairFaceAlignedCropEmbedder` --uses--> `ArcFaceEmbedder`  [INFERRED]
  research_audit_v2/demographic_composition/embeddings.py → face_profile_ml/extractor.py

## Import Cycles
- None detected.

## Communities (81 total, 31 thin omitted)

### Community 0 - "cli.py"
Cohesion: 0.05
Nodes (72): ArgumentParser, ndarray, Path, ScoreCalibrator, add_feature_args(), build_parser(), cmd_audit_fairness(), cmd_calibrate() (+64 more)

### Community 1 - "GallerySimilarityScorer"
Cohesion: 0.06
Nodes (46): BaseHTTPRequestHandler, select_available_providers(), l2_normalize(), ndarray, AppHandler, _clamp(), compose_white_background(), encode_preview_jpeg() (+38 more)

### Community 2 - "scan_public_tree"
Cohesion: 0.06
Nodes (62): MonkeyPatch, _assert_recorded_hashes(), finalize_completed_run(), main(), Any, Path, Post-run verification that records tests without inventing their outcome., Execute the declared suite and atomically persist its real exit status. (+54 more)

### Community 3 - "run_experiment.py"
Cohesion: 0.07
Nodes (64): Figure, _aligned(), compare_scenarios_on_intersection(), cross_fitted_scores(), fit_scenario_run(), _jaccard(), _model(), pairwise_seed_stability() (+56 more)

### Community 4 - "target_rules.py"
Cohesion: 0.05
Nodes (58): ArrayTransform, AgglomerativeBackend, build_backend(), ClusteringBackend, GMMBackend, KMeansBackend, MiniBatchKMeansBackend, ndarray (+50 more)

### Community 5 - "run_grouped_cluster_cv"
Cohesion: 0.05
Nodes (55): DataFrame, ndarray, Grouped cross-fitting for explicitly endogenous cluster targets., Fit clustering and calibration on training folds and emit one row per held-out…, run_grouped_cluster_cv(), _scores(), cosine_similarity_groups(), ndarray (+47 more)

### Community 6 - "run_second_phase.py"
Cohesion: 0.06
Nodes (58): AlignedInputs, ContractError, DataFrame, ndarray, Series, ValueError, Small dependency-free contracts for restricted audit inputs and public outputs., Raised before a non-conformant artifact can enter the analysis. (+50 more)

### Community 7 - "cross_fitting.py"
Cohesion: 0.09
Nodes (40): _fit_cluster(), FitAuditTrail, FoldResult, LeakageError, _metrics(), Any, DataFrame, MiniBatchKMeans (+32 more)

### Community 8 - "preprocess_faces.py"
Cohesion: 0.13
Nodes (39): aligned_square_crop(), ambiguous_embedding_clusters(), approve_candidate(), blockiness_score(), Candidate, cluster_by_embeddings(), cluster_by_identity_key(), color_bins() (+31 more)

### Community 9 - "standardize_faces.py"
Cohesion: 0.14
Nodes (31): align_face(), choose_main_face(), copy_rejected_original(), crop_center(), crop_face_square(), detect_faces(), elliptical_face_mask(), FaceDetection (+23 more)

### Community 10 - "extract_union_embeddings"
Cohesion: 0.11
Nodes (28): _bbox_area(), FaceEmbedding, ndarray, Path, Read an image from disk while preserving Windows Unicode paths and EXIF…, read_bgr_image(), extract_union_embeddings(), _extractor_key() (+20 more)

### Community 11 - "stability.py"
Cohesion: 0.16
Nodes (30): build_stability_design(), _checkpoint_key(), _explicit_seeds(), _fit_partition(), _load_checkpoint(), _order_indices(), pairwise_partition_metrics(), Any (+22 more)

### Community 12 - "recover_rejected_passport.py"
Cohesion: 0.17
Nodes (26): align_by_eyes(), clamp_box(), composite_with_white(), ellipse_mask(), FaceDetection, iou(), keep_center_component(), list_images() (+18 more)

### Community 13 - "choose_target_cluster"
Cohesion: 0.14
Nodes (24): Negative and ground-truth controls that never use restricted inputs., clustering_sensitivity(), MiniBatchKMeans, ndarray, Path, Predeclared one-factor sensitivity analyses for clustering implementation…, _run(), best_match_proportion() (+16 more)

### Community 14 - "run_all.py"
Cohesion: 0.14
Nodes (22): compare_artifacts(), Path, Controlled comparisons only where preserved inputs can be established., Any, Path, Shared deterministic and privacy-safe utilities., Read JSON-compatible YAML configuration and validate the minimum contract., read_config() (+14 more)

### Community 15 - "scrape_dea_fugitives.js"
Cohesion: 0.16
Nodes (22): { chromium }, countImageFiles(), crypto, csvEscape(), discoverPagination(), downloadImage(), ensureDir(), extensionFrom() (+14 more)

### Community 16 - "Face Profile ML"
Cohesion: 0.11
Nodes (18): 1. Primeira execucao completa, 2. Projeto ja instalado, Arquitetura de matching consentido, Artefatos gerados, Auditoria de equidade, Auditoria visual, Como o score da galeria e calculado, Corrigindo erro do InsightFace (+10 more)

### Community 17 - "scrape_interpol.py"
Cohesion: 0.20
Nodes (16): cffi_get_image(), cffi_get_json(), iter_notice_pages(), list_notice_images(), load_metadata(), main(), _paginate_query_pages(), process_notice() (+8 more)

### Community 18 - "write_csv"
Cohesion: 0.20
Nodes (14): determinism(), failure_summary(), final_reports(), _hash(), ndarray, Path, Create public, aggregate second-phase evidence reports., circularity() (+6 more)

### Community 19 - "source_enrichment"
Cohesion: 0.18
Nodes (13): DataFrame, ndarray, Path, Descriptive source composition with transparent limitations., source_enrichment(), benjamini_hochberg(), bootstrap_proportion(), ndarray (+5 more)

### Community 20 - "scrape_fbi_criminals.py"
Cohesion: 0.18
Nodes (15): clean_filename(), create_session(), download_image(), get_file_extension_from_url(), load_existing_metadata(), main(), process_criminal(), Processes a single criminal item: parses details and downloads all associated… (+7 more)

### Community 21 - "build_semantic_manifest.py"
Cohesion: 0.28
Nodes (14): build_report(), choose_target_cluster(), loose_identity(), main(), parse_args(), DataFrame, Namespace, ndarray (+6 more)

### Community 22 - "Restrições globais"
Cohesion: 0.14
Nodes (13): Auditoria científica reproduzível — plano de implementação, Auto-revisão do plano, Restrições globais, Tarefa 10: corrida final e relatórios finais, Tarefa 1: baseline e inventário da implementação existente, Tarefa 2: infraestrutura reprodutível e contratos, Tarefa 3: scanner de privacidade e CI sintética, Tarefa 4: cross-fitting agrupado sem vazamento (+5 more)

### Community 23 - "Experimento de composição demográfica"
Cohesion: 0.15
Nodes (12): Conclusão objetiva, Critérios pré-declarados, Desenho, Distribuição no cluster-alvo (`k=64`), Estabilidade por seed (`k=64`), Experimento de composição demográfica, Gráficos, Limitações (+4 more)

### Community 24 - "Face Profile ML - Relatorio final"
Cohesion: 0.15
Nodes (12): Arquitetura operacional atual, Auditoria visual, Baseline atual, Face Profile ML - Relatorio final, Limitacoes, Metodologia, Objetivo, Preprocessamento e qualidade (+4 more)

### Community 25 - "bootstrap_metric"
Cohesion: 0.29
Nodes (10): bootstrap_auc(), bootstrap_metric(), BootstrapResult, _jaccard(), _metric_function(), ndarray, Paired, reproducible bootstrap confidence intervals for OOF metrics., Bootstrap paired observations and return a percentile interval. (+2 more)

### Community 26 - "audit_data_lineage"
Cohesion: 0.27
Nodes (11): _artifact_supports_count(), audit_data_lineage(), _evaluate_historical_evidence(), Any, DataFrame, Path, Evidence-classified count lineage without assumed sequential transitions., Classify each claimed aggregate independently from all other claims. (+3 more)

### Community 27 - "Final reproduction report"
Cohesion: 0.17
Nodes (11): Changes implemented, Exact reproduction commands, Execution status, Experiments effectively executed, Final reproduction report, Historical information not recovered, Historical results preserved, New methodological reconstructions (+3 more)

### Community 28 - "fetch_research_benchmarks.py"
Cohesion: 0.40
Nodes (10): collect_files(), download(), extract_bfw_face_crops(), main(), Any, Path, Fetch documented face-analysis benchmarks into a git-ignored directory. The…, Verify and expand the nested canonical cropped-face release from BFW. (+2 more)

### Community 29 - "Padronizacao de faces"
Cohesion: 0.18
Nodes (10): Arquivos rejeitados, Comando para processar todo o dataset, Experimentos para a submissão FAccT, ia-usp-jailer, Instalar dependencias, Padronizacao de faces, Relatorio CSV, Rodar com fundo branco (+2 more)

### Community 30 - "Demographic Composition Experiment Design"
Cohesion: 0.20
Nodes (9): Analysis flow, Comparisons, Dataset and categories, Demographic Composition Experiment Design, Ethical limits, Isolation and shared parameters, Objective, Reproducibility and outputs (+1 more)

### Community 31 - "reconcile"
Cohesion: 0.29
Nodes (9): public_lineage(), DataFrame, load_audited_records(), Any, DataFrame, Path, Reconcile observable pipeline stages and produce pseudonymous lineage., Load successful embeddings and create non-reversible public row labels. (+1 more)

### Community 32 - "scrape_eu_most_wanted.py"
Cohesion: 0.29
Nodes (9): create_session(), download_image(), extract_field(), main(), Limpa o nome para uso seguro como nome de arquivo., Cria uma sessão requests com retries automáticos., Baixa a imagem se ela não existir., Extrai texto do field-content dado uma classe do container. (+1 more)

### Community 33 - "validate_research_benchmarks.py"
Cohesion: 0.42
Nodes (9): bfw_image_root(), _decode_status(), main(), DataFrame, Path, Validate local benchmark paths, deterministic image samples, and BFW pairs., _stratified_sample(), validate_bfw_pairs() (+1 more)

### Community 34 - "Implementação das melhorias para submissão FAccT"
Cohesion: 0.22
Nodes (8): Alterações em arquivos existentes, Artefatos gerados nesta implementação, Decisões de implementação, Executores criados, Execuções confirmatórias pendentes, Implementação das melhorias para submissão FAccT, Módulos criados, Verificação

### Community 35 - "synthetic_geometry_control"
Cohesion: 0.31
Nodes (7): negative_controls(), DataFrame, Path, Demonstrate circular geometry and target instability on generated data., synthetic_geometry_control(), test_negative_controls_pass_on_generated_data(), test_synthetic_geometry_control_demonstrates_circular_separability_and_target_instability()

### Community 36 - "Global Constraints"
Cohesion: 0.25
Nodes (7): Demographic Composition Experiment Implementation Plan, Global Constraints, Task 1: Deterministic scenario construction, Task 2: Resumable private embedding extraction, Task 3: Fixed clustering, scoring, and stability analysis, Task 4: Aggregate tables, figures, and objective conclusion, Task 5: Orchestration, smoke run, full run, and verification

### Community 37 - "Manuscript update"
Cohesion: 0.25
Nodes (7): Claims that can be strengthened, Claims that must be weakened, Gaps that remain non-identifiable, Manuscript update, New results, Numbers that remain supportable, Numbers to replace

### Community 38 - "scan_public_outputs"
Cohesion: 0.39
Nodes (6): Path, Public-output guardrail scanner., scan_public_outputs(), Path, test_privacy_scanner_accepts_pseudonymous_aggregate(), test_privacy_scanner_rejects_url()

### Community 39 - "build_research_benchmark_catalogs.py"
Cohesion: 0.57
Nodes (7): _bfw_image_root(), build_bfw(), build_fairface(), _existing(), main(), Path, Create local, pseudonymous catalogs for the downloaded research benchmarks. The…

### Community 40 - "Protocolo de avaliação de equidade e validade"
Cohesion: 0.29
Nodes (6): Ameaças à validade, Conjunto de avaliação, Escopo e alegação possível, Métricas e incerteza, Protocolo de avaliação de equidade e validade, Resultados responsáveis

### Community 41 - "Plano resumido da segunda fase"
Cohesion: 0.29
Nodes (6): Critério de parada, Estado de partida verificado, Plano de trabalho, Plano resumido da segunda fase, Tarefas condicionais ou impossíveis no estado atual, Tarefas executáveis com os artefatos locais

### Community 42 - "predictive_models"
Cohesion: 0.38
Nodes (6): _metrics(), predictive_models(), DataFrame, ndarray, Path, Technical-predictability analysis with grouped folds and no social claims.

### Community 43 - "Monografia ABNT - projeto pronto para Overleaf"
Cohesion: 0.33
Nodes (5): Arquivos principais, Como usar no Overleaf, Compilação local, Monografia ABNT - projeto pronto para Overleaf, Referências bibliográficas

### Community 44 - "Inventário reprodutível de benchmarks externos"
Cohesion: 0.33
Nodes (5): Aquisição validada em 2026-07-27, Delimitação, Inventário reprodutível de benchmarks externos, Reexecução, Termos, citação e limites

### Community 45 - "Plano de auditoria de pesquisa v2"
Cohesion: 0.33
Nodes (5): Achados da inspeção inicial, Compatibilidade com o repositório observado, Escopo e salvaguardas, Plano de auditoria de pesquisa v2, Plano de implementação

### Community 46 - "Research audit v2"
Cohesion: 0.33
Nodes (5): Comandos de reprodução, Escopo científico, Outputs, Reprodutibilidade e retomada, Research audit v2

### Community 47 - "Second-phase final report"
Cohesion: 0.40
Nodes (4): Executive result, Leakage, Limitations, Second-phase final report

### Community 48 - "sanitize_filename"
Cohesion: 0.67
Nodes (3): Sanitize the name to be used as a filename., sanitize_filename(), scrape()

## Knowledge Gaps
- **149 isolated node(s):** `face-profile-ml`, `fs`, `path`, `crypto`, `{ chromium }` (+144 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ArcFaceEmbedder` connect `cli.py` to `GallerySimilarityScorer`, `extract_union_embeddings`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `atomic_write_json()` connect `run_experiment.py` to `scan_public_tree`, `run_second_phase.py`, `cross_fitting.py`, `extract_union_embeddings`, `stability.py`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `run_grouped_cluster_cv()` connect `run_grouped_cluster_cv` to `cli.py`, `target_rules.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `GallerySimilarityScorer` (e.g. with `ScoreCalibrator` and `ArcFaceEmbedder`) actually correct?**
  _`GallerySimilarityScorer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `face-profile-ml`, `fs`, `path` to the rest of the system?**
  _149 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `cli.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05422838031533684 - nodes in this community are weakly interconnected._
- **Should `GallerySimilarityScorer` be split into smaller, more focused modules?**
  _Cohesion score 0.05543071161048689 - nodes in this community are weakly interconnected._