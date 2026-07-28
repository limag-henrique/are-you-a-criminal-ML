# QA numérico

- Manifesto: 9.584 linhas; embeddings: 9.482 vetores de 512 dimensões.
- Cross-fitting agrupado em cinco dobras: `group_overlap = 0` em todas as dobras; médias ROC-AUC = 0,929, PR-AUC = 0,479, F1 = 0,477 e Brier = 0,185.
- Sensibilidades: ordem, batch_size e PCA-64 produziram ARI de aproximadamente 0,07-0,11 fora da referência; Jaccard do alvo chegou a 0.
- A corrida de desenvolvimento 20 sementes × 6 valores de k foi interrompida sem manifesto final e não foi usada.
- `pytest -q` foi reexecutado: falhou porque `pytest` não está no PATH. A execução anterior pelo Python do ambiente virtual excedeu 60 s; portanto, o baseline de testes permanece inconclusivo.
