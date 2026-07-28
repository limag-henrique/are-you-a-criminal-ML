# Registro de alterações

| change_id | problema | evidência | alteração aplicada | impacto | status |
|---|---|---|---|---|---|
| C01 | métricas internas interpretadas além do escopo | leakage_audit.csv | reclassificadas como recuperação de rótulo sintético | modera conclusão | concluído |
| C02 | ausência de validação fora da amostra | cross_fitted_metrics.csv | incluído cross-fitting agrupado e dispersão por dobra | fortalece desenho, não validade externa | concluído |
| C03 | alegações por fonte sem campo documentado | source_influence_report.md | removidas do resultado principal | evita inferência não identificável | concluído |
| C04 | estabilidade histórica sem estado controlável | article_result_reconciliation.csv | removida como resultado final | evita reprodução aparente | concluído |
| C05 | sensibilidade computacional omitida | tabelas de ordem/minibatch/PCA | incluída Tabela 5 | qualifica estabilidade | concluído |
