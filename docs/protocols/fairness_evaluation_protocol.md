# Protocolo de avaliação de equidade e validade

## Escopo e alegação possível

O sistema mede similaridade de embeddings faciais entre imagens autorizadas.
Ele não prediz criminalidade, periculosidade, culpabilidade, propensão criminal
nem pertencimento racial/étnico. Uma lista de pessoas procuradas é um registro
administrativo de busca e publicação institucional, não uma amostra da
criminalidade nem da população. Portanto, frequências nessa lista não permitem
concluir que um grupo é mais ou menos criminoso.

O estudo pode investigar se a cobertura da fonte e os erros de verificação do
sistema variam entre coortes documentadas. Esse desenho avalia viés de dados e
de medição; não confirma diferenças biológicas ou morais entre grupos.

## Conjunto de avaliação

1. Use imagens e metadados com autorização, finalidade definida, retenção
   limitada e revisão ética/jurídica. Não use uma base de procurados como proxy
   de criminalidade ou para rotular raça/etnia.
2. Mantenha pessoas, e não imagens, separadas entre treinamento, calibração e
   teste. Para reconhecimento, cada pessoa deve ter imagens distintas de
   inscrição e consulta; não avalie uma imagem contra ela própria.
3. Registre a origem das coortes: autoidentificação ou fonte documental,
   idioma, data, cobertura e valores ausentes. Nunca infira atributo sensível
   pela face, nome ou nacionalidade.
4. Estratifique por protocolo de tom de pele quando houver anotação humana
   válida, autoidentificação étnico-racial quando autorizada, gênero/sexo
   autoidentificado, faixa etária, qualidade, condição de captura e fonte.
   Publique também as interseções predefinidas.
5. Pré-registre exclusões, limiar operacional, tamanho mínimo por grupo e o
   plano de análise antes de consultar os resultados.

## Métricas e incerteza

Reporte por grupo `n`, genuínos, impostores, AUC, EER, FMR, FNMR, TPR e TNR.
Use um único limiar global, escolhido na calibração e congelado antes do teste;
compare lacunas de FMR/FNMR com o agregado. Reporte intervalos bootstrap de
95%, grupos suprimidos e resultados indeterminados. Nunca reporte uma lacuna
como evidência sem o denominador, o intervalo de incerteza e os controles de
qualidade/origem.

Rode a auditoria em dados pseudonimizados:

```powershell
.\.venv\Scripts\face-profile.exe audit-fairness `
  --scores caminho\cohort_scores.csv `
  --group-columns skin_tone_protocol,ethnicity_self_described,sex_or_gender_self_described,age_band,quality,capture_condition,source_jurisdiction `
  --min-group-n 30 `
  --bootstrap-rounds 2000 `
  --threshold 0.73 `
  --out-dir artifacts\fairness
```

O comando produz `group_metrics.csv` e `fairness_summary.json`. Grupos menores
que `--min-group-n` são marcados como suprimidos, mas não devem ser usados em
comparações substantivas. Para publicação, `--threshold` deve receber o limiar
previamente congelado na calibração, antes de abrir o conjunto de teste. Sem
esse argumento, a ferramenta calcula o EER no próprio arquivo e marca o
resultado como exploratório em `threshold_source`.

## Ameaças à validade

- Viés de seleção/publicação: organizações, jurisdições e políticas diferentes
  determinam quem aparece na fonte e com que imagem.
- Viés de detecção e qualidade: iluminação, pose, câmera, compressão, idade da
  foto e múltiplos rostos afetam a extração antes do matching.
- Rótulo e causalidade: “procurado” não é condenação nem medida de incidência;
  não confunda observação administrativa com comportamento individual.
- Desbalanceamento e interseções raras: não agregue grupos heterogêneos para
  mascarar erro e não interprete amostras pequenas.

## Resultados responsáveis

Apresente resultados descritivos, o período e a cobertura das fontes, as taxas
de dados ausentes e uma seção explícita de limitações. Qualquer decisão humana
deve revisar o caso; o score não é identificação, prova ou base autônoma para
abordagem, vigilância, prisão ou sanção.
