# Lacunas da segunda fase

| Requisito | Situação | Evidência / restrição |
|---|---|---|
| Auditoria de implementação anterior | implementado parcialmente | Há pacote inicial e testes, mas não há corrida v2 concluída. |
| Matriz de uso de dados e auditoria de vazamento | não implementado | A primeira fase só produz diagnóstico interno. |
| Cross-fitting por grupo | não implementado | Exige módulo novo e validação dos grupos. |
| Separação exploratória/confirmatória e configuração bloqueada | não implementado | Deve preceder resultados da segunda fase. |
| Especificação de pré-processamento | implementado parcialmente | README e scripts descrevem parte do fluxo; falta contrato auditável. |
| Sensibilidade de detector/alinhamento/recorte | impossível com os artefatos disponíveis | Ambiente atual Python 3.14 não é o suportado pelo README; pesos/configuração histórica completa ausentes. |
| Sensibilidade L2, dtype e ordem | não implementado | Executável sobre embeddings atuais. |
| Ordem e parâmetros do MiniBatchKMeans | implementado parcialmente | Há estabilidade por semente/k, não há desenho de ordem/batch. |
| Sensibilidade de representação | não implementado | Executável sem alterar o método principal. |
| Influência por fonte | impossível com os artefatos disponíveis | O embedding manifesto não possui fonte documentada. |
| Influência por grupo e subconjunto | não implementado | Executável após validação dos grupos. |
| Mecanismo de falhas técnicas | implementado parcialmente | Há indicadores técnicos no manifesto; faltam tabelas e modelo descritivo. |
| Controles negativos e sintéticos | implementado parcialmente | Há testes pequenos; faltam os controles e relatórios requeridos. |
| Testes de propriedades | implementado parcialmente | Métricas básicas cobertas; faltam splits, NaN, agregados e determinismo. |
| Contratos de dados | não implementado | Carregamento atual verifica apenas algumas colunas. |
| Determinismo entre ambientes | não implementado | Só o ambiente local é acessível; CPU/GPU e ambiente limpo são condicionais. |
| Regressão científica/golden outputs | não implementado | Deve usar somente fixtures sintéticos. |
| Falhas e retomada segura | implementado parcialmente | CSV usa escrita atômica; faltam testes de interrupção e cache. |
| Cache e incompatibilidade de artefato | não implementado | Manifesto de execução inicial é incompleto e só ocorre ao final. |
| Scanner de privacidade | implementado parcialmente | Há padrões básicos e testes; faltam padrões, relatório e hook. |
| Qualidade, tipos, cobertura e CI | não implementado | Tipos/docstrings parciais; ferramentas e workflow ausentes. |
| Medidas de desempenho | não implementado | Só tempos por clustering estão planejados. |
| Análise temporal | impossível com os artefatos disponíveis | Não foram localizadas datas confiáveis. |
| Reconciliação com resultados do artigo | implementado parcialmente | 9.584 e 9.482 verificáveis; demais valores/artefatos não reproduzíveis ainda. |
| Relatórios finais da segunda fase | não implementado | Dependem das validações acima. |
