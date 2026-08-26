# Implementação das melhorias para submissão FAccT

Data da implementação: 25 de agosto de 2026.

Este documento registra as alterações derivadas de `tasks.txt` e `plano de implementacao.txt`, os contratos públicos adicionados, os artefatos gerados e a diferença entre execução confirmatória e smoke test. Nenhum resultado reduzido é evidência científica do artigo.

## Decisões de implementação

- O comando `fit` existente continua responsável pelo modelo de perfil facial. A avaliação endógena por clustering foi isolada no novo subcomando `cv-fit`, evitando alterar a semântica de um comando já publicado.
- Os seis seletores de alvo são funções puras e o seletor aleatório recebe uma semente explícita.
- OOF usa `GroupKFold`; clustering, regra do alvo e calibração são ajustados somente no treino de cada dobra.
- A decomposição de variância usa efeitos fixos e contribuição marginal de cada fator, calculada pela diferença de soma de quadrados entre modelo completo e modelo reduzido.
- A rotina FairFace pareada reserva um núcleo estratificado comum e sorteia complementos independentes por cenário.
- Resultados de grades extensas só entram no artigo depois de uma execução confirmatória completa. Smoke tests verificam contratos, formatos e integração.

## Módulos criados

| Arquivo | Responsabilidade |
|---|---|
| `face_profile_ml/endogenous_target.py` | Construção `Z=f(X), Y=h(Z), s=g(Z)`, métricas interna/externa e testemunha parametrizada da Proposição 1. |
| `face_profile_ml/target_rules.py` | Regras `largest`, `compact`, `separated`, `random`, `central` e `outlier`. |
| `face_profile_ml/variance_decomposition.py` | Decomposição de efeitos fixos para cenário, replicação, seed e k. |
| `face_profile_ml/bootstrap_ci.py` | IC bootstrap pareado para AUC, PR-AUC, Brier, Jaccard e balanced accuracy. |
| `face_profile_ml/clustering_backends.py` | Interface uniforme para MiniBatchKMeans, KMeans, GMM e Agglomerative. |
| `face_profile_ml/cross_validation.py` | Avaliação OOF agrupada do alvo endógeno e contrato de predições por observação. |
| `face_profile_ml/grouping.py` | Reconstrução de `group_id` por componentes conexos de similaridade cosseno. |

## Executores criados

| Arquivo | Saída padrão |
|---|---|
| `scripts/run_proposition_demo.py` | `artifacts/proposition/proposition_results.csv` |
| `scripts/run_fairface_replicated.py` | `artifacts/fairface_replicated/oof_predictions.parquet`, `variance_decomposition.csv`, `run_metrics.csv` e `run_manifest.json` |
| `scripts/run_target_ablation.py` | `artifacts/ablation/ablation_results.csv` e `incompatibility_matrix.csv` |
| `scripts/plot_performance_stability_paradox.py` | `artigo/figuras/paradox_performance_stability.png` |
| `scripts/run_groupid_sensitivity.py` | `artifacts/groupid_sensitivity/sensitivity_curve.csv` |
| `scripts/register_model_hash.py` | `artifacts/model_hashes.json` e, opcionalmente, atualização do manifesto FairFace |
| `scripts/run_clustering_comparison.py` | `artifacts/clustering_comparison/backend_metrics.csv` e `cross_backend_jaccard.csv` |
| `scripts/run_embedding_decoupling.py` | `artifacts/embedding_decoupling/decoupling_results.csv` |
| `scripts/run_multiverse.py` | `artifacts/multiverse/specification_curve.csv`, `pairwise_jaccard.csv` e `specification_curve.png` |

## Alterações em arquivos existentes

- `face_profile_ml/cli.py`: adiciona `cv-fit`, persistência OOF, ICs bootstrap e manifesto com SHA-256.
- `pyproject.toml`: declara Matplotlib, PyArrow e SciPy, usados por figuras, Parquet e componentes conexos.
- `artigo/conteudo.tex`: formaliza a Proposição 1, nomeia o Resultado Empírico 1, usa a figura AUC × Jaccard, registra os protocolos implementados e reorganiza ameaças à validade.
- `scripts/README.md`: aponta para este inventário e documenta a política de execução confirmatória.

## Artefatos gerados nesta implementação

Confirmatórios ou derivados de resultados preservados:

- `artifacts/proposition/proposition_results.csv`: 30 construções sintéticas, com d=512, n=500, k em 2/4/8 e seeds 0--9. AUC interna foi 1,0 em todas; AUC contra o controle independente teve média entre 0,495 e 0,511 conforme k.
- `artifacts/model_hashes.json`: SHA-256 pós-hoc de `w600k_r50.onnx`, `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
- `artigo/figuras/paradox_performance_stability.png`: figura derivada de `cross_fitted_metrics.csv` e `stability_summary.csv` preservados.
- `artigo/main.pdf`: artigo recompilado com Tectonic portátil e inspecionado visualmente nas páginas da proposição, ablação, resultado central e limitações.

Verificações reduzidas, sem valor confirmatório:

- `artifacts/proposition_smoke/`
- `artifacts/ablation_smoke/`
- `artifacts/fairface_replicated_smoke/`
- `artifacts/groupid_sensitivity_smoke/`
- `artifacts/clustering_comparison_smoke/`
- `artifacts/embedding_decoupling_smoke/`
- `artifacts/multiverse_smoke/`

## Execuções confirmatórias pendentes

As grades completas de FairFace replicado, ablação, sensibilidade de `group_id`, comparação de backends, desacoplamento e multiverse são computacionalmente extensas e não foram executadas como parte desta alteração. Em especial, o default FairFace representa 50 replicações × 5 seeds × 3 valores de k × 4 cenários × 5 dobras, ou 15.000 ajustes de clustering. O artigo declara essa pendência e não apresenta números dos smoke tests como achados.

## Verificação

```powershell
python -m pytest tests -q
python scripts/run_proposition_demo.py
python scripts/plot_performance_stability_paradox.py
cd artigo
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

O repositório mantém a política humana de commits: nenhuma alteração é commitada automaticamente.
