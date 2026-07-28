# Plano resumido da segunda fase

## Estado de partida verificado

A primeira fase possui uma implementação inicial isolada, com configurações,
proveniência, grupos de similaridade extrema, estabilidade, circularidade,
enriquecimento, modelos técnicos, relatórios e oito testes sintéticos. Ela não
produziu uma corrida de desenvolvimento concluída: a configuração de 20 sementes
e seis valores de k com 30 iterações excedeu a janela operacional e foi
interrompida sem manifesto final. Não existem resultados anteriores em
`research_audit_v2/outputs/` que possam ser tratados como auditados.

O estado histórico preservado continua incompleto: não foi localizado o código
da heurística de cluster-alvo, a análise K-means histórica, a regressão
histórica nem uma atribuição de fonte documentada no manifesto de embeddings.

## Plano de trabalho

1. Corrigir a base da primeira fase antes de reutilizá-la: contratos de dados,
   erro de métricas, escrita atômica, manifestos completos, configuração de
   desenvolvimento viável, controle explícito de sementes e scanner de
   privacidade.
2. Implementar a auditoria de vazamento como desenho separado: diagnóstico
   interno, cross-fitting por `group_id` e repetições treino/teste. Todo fit de
   transformação, cluster, alvo, centroide e limiar ficará estritamente no
   treino; quedas de métricas serão relatadas, não otimizadas.
3. Criar sensibilidades de ordem/minibatch e representação de embedding usando
   desenho reduzido predefinido. Sensibilidade ao pré-processamento só será
   executada para opções efetivamente reproduzíveis com os pesos e ambiente
   locais; as demais serão especificadas como indisponíveis.
4. Adicionar controles negativos, dados sintéticos com verdade conhecida,
   testes de propriedades, contratos, testes de falha e golden outputs. Eles
   não usarão imagens pessoais nem embeddings individuais reais.
5. Executar determinismo, influência de fontes/grupos, falhas técnicas e
   reconciliação com o artigo apenas quando as variáveis necessárias forem
   documentadas. Ausências de fonte, datas ou estado histórico serão resultados
   de não identificabilidade, não preenchidas por inferência.
6. Adicionar CI apenas com dados sintéticos, versão de artefatos por hash e
   relatórios públicos sem dados pessoais. A fase final gerará a matriz de
   robustez e as mudanças necessárias no manuscrito.

## Tarefas executáveis com os artefatos locais

- Cross-fitting agrupado usando os embeddings atuais e grupos de similaridade
  extrema, com a ressalva de que o alvo será uma regra reconstruída.
- Testes de ordem/minibatch, representação, precisão numérica, determinismo,
  controles negativos, contratos, falhas e privacidade.
- Análise de falhas técnicas nas colunas já presentes no embedding manifesto.
- Reconciliação direta das contagens 9.584 e 9.482, e registro explícito das
  transições anteriores não resolvidas.

## Tarefas condicionais ou impossíveis no estado atual

- Comparação causal com pipeline/heurística histórica: impossível sem estado
  histórico preservado verificável.
- Leave-one-source-out e balanceamento por fonte: bloqueados enquanto a fonte
  não estiver documentada no manifesto, pois não será inferida de nomes/URLs.
- Sensibilidade de detector, alinhamento, recorte ou pesos: condicional à
  reprodução do ambiente InsightFace suportado e a configurações históricas
  completas; o Python atual é 3.14, fora do intervalo recomendado pelo README.
- Teste temporal: impossível até que datas de registro confiáveis sejam
  localizadas.

## Critério de parada

Nenhum resultado será classificado como externo, biométrico, social ou causal.
Resultados parciais terão `completion_status` explícito. A configuração
confirmatória será congelada antes de observar os resultados correspondentes,
e desvios serão registrados em vez de sobrescrever artefatos.
