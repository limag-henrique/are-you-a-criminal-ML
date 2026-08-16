# Experimento de composição demográfica

## Desenho

Foram usadas exclusivamente as sete categorias fornecidas pelo FairFace. `Middle Eastern` foi o grupo reduzido em C e aumentado em D. Cada cenário contém 36.456 amostras distintas, sem reposição. A liberação `margin025`, recortada e alinhada upstream com `dlib.get_face_chip()`, foi fornecida diretamente ao reconhecedor ArcFace em lotes. Modelo, normalização L2, MiniBatchKMeans, seeds, valores de `k`, escolha do maior cluster, score cosseno e avaliação foram mantidos iguais.

Esta é a execução integral pré-declarada.

Não houve associação com indivíduos ou registros do pipeline principal, nem inferência de raça/cor. ROC-AUC e PR-AUC medem apenas recuperação interna do alvo sintético derivado do clustering.

## Resultados principais (`k=64`)

| Cenário | Tamanho mediano do alvo | Prevalência mediana | ARI vs. A | Jaccard do alvo vs. A | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 894 | 0.025 | — | — | 0.908 | 0.448 |
| B | 936 | 0.026 | 0.182 | 0.001 | 0.903 | 0.440 |
| C | 978 | 0.027 | 0.199 | 0.001 | 0.893 | 0.400 |
| D | 908 | 0.025 | 0.189 | 0.001 | 0.910 | 0.466 |

## Estabilidade por seed (`k=64`)

| Cenário | ARI mediano entre seeds | Jaccard mediano do alvo entre seeds |
| --- | ---: | ---: |
| A | 0.186 | 0.000 |
| B | 0.188 | 0.015 |
| C | 0.190 | 0.001 |
| D | 0.198 | 0.000 |

A baixa sobreposição do maior cluster ocorre também dentro de cada cenário. Portanto, diferenças de identidade do alvo contra A devem ser interpretadas junto dessa instabilidade basal.

## Variações contra A (`k=64`)

| Cenário | Δ prevalência | Δ ROC-AUC | Δ PR-AUC |
| --- | ---: | ---: | ---: |
| B | +0.001 | -0.005 | -0.009 |
| C | +0.002 | -0.016 | -0.048 |
| D | +0.000 | +0.002 | +0.018 |

## Sensibilidade a k

Prevalência mediana do maior cluster:

| k | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.048 | 0.045 | 0.048 | 0.047 |
| 48 | 0.032 | 0.033 | 0.032 | 0.032 |
| 64 | 0.025 | 0.026 | 0.027 | 0.025 |
| 80 | 0.023 | 0.023 | 0.023 | 0.022 |
| 96 | 0.021 | 0.021 | 0.020 | 0.021 |
| 128 | 0.018 | 0.018 | 0.017 | 0.017 |

## Distribuição no cluster-alvo (`k=64`)

Média da proporção de cada categoria entre seeds (cada linha soma 1, salvo arredondamento):

| Cenário | Black | East Asian | Indian | Latino_Hispanic | Middle Eastern | Southeast Asian | White |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.084 | 0.172 | 0.120 | 0.155 | 0.144 | 0.094 | 0.231 |
| B | 0.101 | 0.101 | 0.105 | 0.131 | 0.247 | 0.059 | 0.255 |
| C | 0.075 | 0.097 | 0.136 | 0.198 | 0.126 | 0.082 | 0.286 |
| D | 0.071 | 0.129 | 0.097 | 0.119 | 0.318 | 0.092 | 0.175 |

## Gráficos

- [Composição dos cenários](research_audit_v2/outputs/demographic_composition/figures/scenario_composition.svg)
- [Tamanho e prevalência do alvo](research_audit_v2/outputs/demographic_composition/figures/target_prevalence.svg)
- [Estabilidade por seed](research_audit_v2/outputs/demographic_composition/figures/stability.svg)
- [ROC-AUC e PR-AUC](research_audit_v2/outputs/demographic_composition/figures/auc_metrics.svg)
- [Sensibilidade a k](research_audit_v2/outputs/demographic_composition/figures/k_sensitivity.svg)
- [Distribuição demográfica no cluster-alvo](research_audit_v2/outputs/demographic_composition/figures/target_demographic_distribution.svg)

As tabelas completas por seed, `k`, cluster e categoria estão em `research_audit_v2/outputs/demographic_composition/tables`.

## Critérios pré-declarados

Uma mudança é relevante quando ARI mediano < 0,90, Jaccard mediano < 0,80, variação absoluta de prevalência >= 0,02 ou variação absoluta de ROC-AUC/PR-AUC >= 0,03.

- B: ARI mediano abaixo de 0,90; Jaccard mediano do alvo abaixo de 0,80.
- C: ARI mediano abaixo de 0,90; Jaccard mediano do alvo abaixo de 0,80; mudança absoluta de PR-AUC de pelo menos 0,03.
- D: ARI mediano abaixo de 0,90; Jaccard mediano do alvo abaixo de 0,80.

## Reprodutibilidade

A configuração versionável está em `research_audit_v2/demographic_composition/config.json`; os resultados agregados, hashes da configuração, catálogo e vetores ficam no manifesto `research_audit_v2/outputs/demographic_composition/run_manifest.json`. Seeds: 20260815, 20260816, 20260817, 20260818, 20260819, 20260820, 20260821, 20260822, 20260823, 20260824. Grade de `k`: 32, 48, 64, 80, 96, 128.

```bash
python -m research_audit_v2.demographic_composition.run_experiment --config research_audit_v2/demographic_composition/config.json --resume
```

## Limitações

Os rótulos históricos do FairFace não são autoidentificação nem verdade biológica. O uso direto dos crops evita uma segunda detecção incompatível com o enquadramento apertado, mas herda o alinhamento dlib da base e não testa sensibilidade a outro alinhador. FairFace não fornece identidades repetidas adequadas a esta análise; cada registro foi tratado como grupo próprio no cross-fitting. Métricas internas não validam categorias sociais, identidade, criminalidade, comportamento, risco ou superioridade de grupo.

## Conclusão objetiva

A composição demográfica modificou de forma relevante pelo menos um resultado nos cenários B, C, D, segundo os limiares pré-declarados.

- **Clustering:** sim. B, C e D apresentaram ARI mediano contra A abaixo de 0,90, indicando partições materialmente diferentes na interseção de registros.
- **Cluster-alvo:** sim. O Jaccard mediano do maior cluster contra A ficou abaixo de 0,80 nos três cenários; a identidade do alvo mudou.
- **Prevalência:** não de forma relevante. A maior variação absoluta contra A foi 0.002 (limiar: 0,020); em `k=64`, as medianas ficaram entre 0.025 e 0.027.
- **Estabilidade:** todos os cenários foram muito sensíveis a `seed` (ARI mediano entre 0.186 e 0.198; Jaccard mediano entre 0.000 e 0.015). Não surgiu uma melhora ou piora direcional robusta atribuível à composição.
- **Métricas internas:** ROC-AUC não mudou de forma relevante (maior |delta|: 0.016). PR-AUC mudou de forma relevante apenas em C.
- **Sensibilidade a `k`:** relevante para a prevalência em todos os cenários, que variou de 0.017 a 0.048 na grade; as diferenças A–D permaneceram pequenas em cada `k`.
