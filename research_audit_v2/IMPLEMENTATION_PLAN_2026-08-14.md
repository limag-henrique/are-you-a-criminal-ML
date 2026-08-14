# Auditoria científica reproduzível — plano de implementação

> **Para execução agentic:** usar `superpowers:executing-plans` e `superpowers:test-driven-development`; executar cada tarefa em ordem e marcar cada checkbox somente após a verificação correspondente.

**Objetivo:** transformar `research_audit_v2` em uma auditoria computacional testada, reprodutível e capaz de gerar por código os resultados científicos e os dois relatórios finais solicitados.

**Arquitetura:** manter o pacote isolado, separar contratos/execução/cross-fitting/estabilidade/linhagem/privacidade/relatórios em módulos pequenos e fazer todos os outputs passarem por escrita atômica. Configurações `development` e `final` dirigem o mesmo pipeline; fixtures sintéticas exercitam a CI, enquanto artefatos reais são usados somente nas execuções locais autorizadas.

**Stack:** Python 3.14 local, NumPy, pandas, SciPy, scikit-learn, matplotlib, pytest e GitHub Actions com fixtures sintéticas.

## Restrições globais

- Não modificar `face_profile_ml`, artefatos históricos ou alterações preexistentes fora de `research_audit_v2`.
- Não criar commits nem trocar de branch.
- `group_id` significa somente duplicidade provável.
- Não inferir fonte, identidade, raça/cor, data ou regra histórica ausente.
- Classificar fatos como histórico preservado, reproduzido, reconstrução nova ou não recuperado.
- Outputs públicos não podem conter imagens, nomes, URLs, caminhos pessoais ou embeddings individuais.
- Métricas são somente recuperação interna de alvo sintético, nunca criminalidade, culpa, identidade, raça/cor ou equidade biométrica.

---

### Tarefa 1: baseline e inventário da implementação existente

**Arquivos:** leitura de `research_audit_v2/**`; registrar achados neste plano e nos relatórios finais.

- [x] Ler integralmente `README.md`, `PLAN.md`, `SECOND_PHASE_GAP_ANALYSIS.md` e `SECOND_PHASE_PLAN.md`.
- [x] Verificar branch, worktree sujo e alterações preexistentes sem modificá-las.
- [x] Executar `python -m pytest -q research_audit_v2` e registrar o baseline (`12 passed`).
- [x] Comparar código, configuração, manifestos e outputs rastreados; confirmar que o estado “complete” atual não satisfaz o contrato solicitado.

### Tarefa 2: infraestrutura reprodutível e contratos

**Arquivos:** modificar `src/common.py`, `second_phase/src/data_contracts.py`; criar `second_phase/src/run_manifest.py`, `second_phase/src/io.py`; testar em `second_phase/tests/test_data_contracts.py`, `test_atomic_io.py`, `test_run_manifest.py`.

**Interfaces:** `validate_audit_inputs(manifest, embeddings) -> AlignedInputs`; `atomic_write_*`; `RunManifest.start(...)`, `record_output(...)`, `complete(...)`, `fail(...)`.

- [x] Escrever testes que falhem para NaN/Inf, matriz vazia, dimensão inválida, índice fracionário/duplicado/fora da faixa, ordem desalinhada e linhas bem-sucedidas ausentes.
- [x] Executar os testes e confirmar falhas pelas lacunas comportamentais.
- [x] Implementar contrato único que retorna manifesto e embeddings explicitamente alinhados por `embedding_index`.
- [x] Escrever testes de escrita atômica, limpeza do temporário, substituição e retomada incompatível por hash/configuração.
- [x] Implementar escrita atômica de CSV/JSON/texto e retomada versionada por hash.
- [x] Escrever testes do manifesto criado antes da carga pesada e atualizado em falha/sucesso, com commit, config, seeds, versões, sistema, parâmetros, hashes, horários, status e outputs.
- [x] Implementar o manifesto incremental e executar os testes do módulo e a suíte completa (`36 passed`).

### Tarefa 3: scanner de privacidade e CI sintética

**Arquivos:** modificar `second_phase/src/privacy_scan.py`; criar fixtures em `second_phase/tests/fixtures/`; criar `.github/workflows/research-audit-v2.yml`; testar em `test_public_output_privacy.py` e `test_synthetic_pipeline.py`.

- [x] Escrever testes que falhem para nomes de colunas sensíveis com caixa/espaços, caminhos Windows/Unix, URLs, e-mails, nomes de imagens, vetores serializados e binários proibidos.
- [x] Implementar scanner com relatório estruturado e allowlist SVG estrita; o bloqueio orquestrado será integrado na Tarefa 9.
- [x] Criar fixture sintética determinística contendo manifestos/grupos/embeddings sem dados pessoais.
- [x] Criar teste end-to-end rápido sobre fixture sintética; workflow CI já configurado para executar somente testes/controle sintético.
- [x] Executar testes direcionados e suíte completa (`36 passed`).

### Tarefa 4: cross-fitting agrupado sem vazamento

**Arquivos:** reescrever `second_phase/src/cross_fitting.py`; criar `second_phase/src/transforms.py`; ampliar `second_phase/tests/test_cross_fitting.py`.

**Interfaces:** `GroupedCrossFitter.run(records, embeddings, config) -> (fold_metrics, split_composition, audit_events)`; transformadores recebem somente índices de treino em `fit`.

- [x] Escrever teste RED que rejeite qualquer `group_id` compartilhado entre treino e teste.
- [x] Escrever testes RED instrumentados que detectem transformação/PCA, clustering, seleção de alvo, centroide, limiar ou calibração ajustados com índices de teste.
- [x] Escrever testes RED para composição por dobra: `n`, grupos, positivos, negativos, prevalência, ROC-AUC, PR-AUC e baseline PR-AUC.
- [x] Escrever testes RED para limiar escolhido somente no treino; F1/precision/recall/balanced accuracy somente quando válidos; Brier somente com calibração probabilística treinada sem teste.
- [x] Implementar pipeline por dobra com todas as operações ajustadas exclusivamente no treino e avaliação do teste somente após o congelamento da dobra.
- [x] Executar testes direcionados, mutações mentais de leakage e suíte completa.

### Tarefa 5: PCA-64 auditável e estabilidade de representação

**Arquivos:** criar `second_phase/src/representation.py`; modificar `second_phase/src/sensitivity.py`; testar em `second_phase/tests/test_representation.py`.

- [x] Escrever testes RED para `n_components=64`, solver explícito, `whiten`, `random_state`, centering, variância explicada/cumulativa e L2 pré/pós-PCA.
- [x] Escrever teste RED que prova que PCA de uma dobra não vê registros de teste.
- [x] Implementar especificação/transformação serializável e output `pca_specification.json`.
- [x] Implementar comparação entre embedding original e PCA-64 completamente especificada.
- [x] Executar testes direcionados e suíte completa.

### Tarefa 6: estabilidade estocástica, operacional e pairwise

**Arquivos:** criar `second_phase/src/stability.py`; modificar configs `development.yaml` e `final.yaml`; testar em `second_phase/tests/test_stability.py`.

- [x] Escrever testes RED separando seed, ordem, batch size e representação, mantendo os demais parâmetros fixos.
- [x] Escrever testes RED para grade final `k=[32,48,64,80,96,128]`, 100 seeds e batches `[256,512,1024,2048,4096]`.
- [x] Escrever testes RED para ARI, Jaccard do alvo, tamanho/prevalência e resumo com média, DP, mediana, Q1/Q3, mínimo/máximo, P5/P95.
- [x] Escrever testes RED para matrizes pairwise simétricas, diagonal 1 e referência não arbitrária.
- [x] Implementar execução retomável por célula, checkpoint por hash e outputs estruturados.
- [x] Tornar `development` curto e representativo; manter `final` integral e imutável.
- [x] Executar testes direcionados e suíte completa.

### Tarefa 7: validação de `group_id` e linhagem de dados

**Arquivos:** modificar `src/deduplication.py`, `src/provenance.py`; criar `second_phase/src/group_audit.py`, `data_lineage.py`; testar em `test_group_audit.py`, `test_data_lineage.py`.

- [x] Escrever testes RED para número de grupos, unitários/não unitários, registros/proporção agrupados, média/mediana/máximo e distribuição de tamanhos.
- [x] Escrever teste RED que exige métrica e limiar e proíbe linguagem de identidade confirmada.
- [x] Escrever teste RED para amostra próxima ao limiar contendo apenas IDs pseudônimos e similaridade agregada, sem embeddings/caminhos.
- [x] Implementar outputs `group_id_statistics.json/csv` e amostra segura quando houver pares elegíveis.
- [x] Escrever testes RED para as cinco contagens e quatro classificações de evidência; não modelar as contagens como sequência sem prova.
- [x] Implementar pesquisa documental por artefato/hash e marcar 9.546↔9.584 como lacuna histórica se não houver evidência.
- [x] Executar testes direcionados e suíte completa.

### Tarefa 8: controle sintético metodológico

**Arquivos:** ampliar `second_phase/src/controls.py`; testar em `second_phase/tests/test_synthetic_control.py`.

- [x] Escrever teste RED para geração de rótulo por clustering e score da mesma geometria com alta separabilidade.
- [x] Escrever teste RED mostrando mudança do próprio alvo sob perturbação predefinida do clustering.
- [x] Implementar experimento e output agregado identificado exclusivamente como demonstração metodológica.
- [x] Executar testes direcionados e suíte completa.

### Tarefa 9: orquestração, desenvolvimento e outputs científicos

**Arquivos:** reescrever `second_phase/src/run_second_phase.py`; atualizar `README.md`; centralizar outputs em `research_audit_v2/outputs/`.

- [x] Escrever teste end-to-end RED que inicia manifesto, executa etapas na ordem, registra outputs/hashes e falha fechado na privacidade.
- [x] Implementar CLI `--config`, `--resume` e modos development/final usando o mesmo pipeline.
- [x] Documentar comandos exatos para testes, desenvolvimento e final.
- [x] Executar todos os testes (`78 passed`).
- [x] Executar configuração de desenvolvimento e verificar esquema, hashes, completude, atomicidade e privacidade de todos os outputs.

### Tarefa 10: corrida final e relatórios finais

**Arquivos:** gerar por código `research_audit_v2/FINAL_REPRODUCTION_REPORT.md` e `research_audit_v2/MANUSCRIPT_UPDATE.md`; outputs em `research_audit_v2/outputs/`.

- [x] Estimar custo a partir da corrida de desenvolvimento e verificar recursos disponíveis (benchmark não científico: 4,95 s/célula representativa).
- [x] Executar configuração final integral se viável; corrida completa com 611/611 células e manifesto `complete`.
- [x] Executar scanner de privacidade final e confirmar ausência de imagens, nomes, URLs, caminhos pessoais e embeddings individuais (0 achados).
- [x] Gerar `FINAL_REPRODUCTION_REPORT.md` somente a partir de manifestos/tabelas produzidos.
- [x] Gerar `MANUSCRIPT_UPDATE.md` com cada afirmação ligada ao arquivo estruturado que a sustenta.
- [x] Conferir manualmente correspondência entre relatórios e resultados, executar a suíte final e registrar comandos/resultado (`78 passed in 18.56s`).

## Auto-revisão do plano

- Cobertura: todos os requisitos do texto anexado estão mapeados às Tarefas 2–10.
- Escopo: mudanças ficam em `research_audit_v2`, exceto o workflow sintético solicitado em `.github/workflows/`.
- Consistência: desenvolvimento e final usam o mesmo orquestrador; diferenças são apenas parâmetros declarados.
- Política Git: nenhum passo cria commit ou muda branch.
- Ausências históricas: são outputs de não identificabilidade, nunca preenchidas por suposição.
