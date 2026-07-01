# Face Profile ML - Relatorio final

## Objetivo

Este projeto implementa um pipeline de modelagem de perfil facial usando
embeddings pre-treinados. A rede neural e usada apenas para deteccao,
alinhamento e extracao de embeddings. Nao ha fine-tuning.

## Metodologia

1. `manifest.csv` lista imagens, identidade aproximada, qualidade, split e peso.
2. `face-profile extract` carrega imagens com correcao EXIF, detecta/alinha
   faces com InsightFace e salva embeddings L2.
3. `scripts/build_semantic_manifest.py` cria splits semanticos a partir da
   geometria dos embeddings.
4. `face-profile fit` modela o perfil com media ponderada, top-k cosine,
   Mahalanobis regularizada e One-Class SVM opcional.
5. `face-profile calibrate` ajusta score calibrado com regressao logistica.
6. `face-profile evaluate` calcula ROC, AUC, EER, FMR e FNMR, inclusive por qualidade.
7. `scripts/build_audit_report.py` gera HTML/CSV de auditoria visual dos splits.

## Arquitetura operacional atual

O modo recomendado para matching consentido e o aplicativo de galeria local,
nao o modelo agregado de perfil. Ele usa `buffalo_l` como extrator ArcFace
pre-treinado, sem fine-tuning. O upload e comparado apenas contra embeddings
locais autorizados.

A resposta separa:

- melhor candidato individual (`best_cosine`, `best_match_similarity_percent`);
- similaridade geral contra a galeria (`overall_gallery_similarity_percent`);
- densidade/top-k de referencias semelhantes;
- distinctividade ou unicidade relativa;
- percentil e FMR estimado contra distribuicao impostora local;
- avisos para baixa qualidade, multiplas faces e referencias quase duplicadas.

O score visual nao e probabilidade absoluta nem prova de identidade. Ele deve
ser usado como medida de similaridade visual contra referencias conhecidas.

## Preprocessamento e qualidade

O extrator atual:

- corrige orientacao EXIF antes da deteccao;
- usa o maior rosto detectado para o upload;
- rejeita referencias com deteccao abaixo de `--min-det-score`, rosto pequeno
  abaixo de `--min-face-area-ratio` ou multiplas faces, salvo quando
  `--allow-multiple-faces` for informado;
- registra `face_count`, `face_area_ratio`, dimensoes da imagem e parametros do
  modelo em `embedding_manifest.csv` e `extract_metadata.json`;
- salva crop alinhado de auditoria quando `--save-aligned` e usado.

## Similaridade e limiares

O score da galeria combina melhor cosine, top-k ponderado, densidade de
referencias acima de limiares e percentil impostor. Duplicatas quase identicas
sao agrupadas para nao inflar densidade, top-k agregado ou calibracao de FMR.

Limiares padrao:

- `low=0.30`
- `medium=0.40`
- `high=0.55`
- `very_high=0.70`
- `near_duplicate=0.985`

Esses limiares devem ser recalibrados se a galeria mudar de camera, dominio,
qualidade ou populacao.

## Privacidade e consentimento

Para uso real:

- manter apenas imagens autorizadas no manifesto;
- proteger `artifacts/embeddings.npy`, `embedding_manifest.csv` e imagens de
  referencia como dados biometricos sensiveis;
- nao interpretar o resultado como identificacao automatica;
- auditar inclusao, remocao e expiracao de consentimento;
- remover imagens e embeddings quando o consentimento for revogado.

## Baseline atual

Extracao regenerada com `buffalo_l` em CPU:

- imagens no manifesto: 9.584
- embeddings extraidos: 9.482
- sem face detectada: 98
- referencias rejeitadas por multiplas faces: 4
- dimensao: 512
- det-size: 320
- `min_det_score`: 0.50
- `min_face_area_ratio`: 0.01

Manifesto semantico atual:

- `profile`: 751
- `calib_pos`: 219
- `test_pos`: 258
- `calib_neg`: 1.931
- `test_neg`: 2.069
- `ignore`: 4.356

Metricas finais do modelo agregado legado apos recalculo:

- AUC geral: 0.9960
- EER geral: 0.0246
- AUC high: 0.9989
- AUC mid: 0.9966
- AUC low: 0.9940

O aplicativo de galeria continua sendo o caminho recomendado para comparacao
operacional, porque retorna candidatos concretos e separa melhor match,
similaridade geral, densidade, distinctividade, percentil e FMR.

## Auditoria visual

Foi gerado o relatorio:

- `artifacts/audit/index.html`
- `artifacts/audit/audit_samples.csv`
- `artifacts/audit/audit_summary.json`

Conclusao visual: o split `profile` e os positivos aleatorios formam um grupo
facial coerente, majoritariamente masculino, frontal e com tracos/enquadramento
semelhantes. Os exemplos dificeis mostram alguns positivos semanticamente mais
fracos e negativos visualmente parecidos com o grupo positivo.

Foi testado um candidato mais estrito:

- `positive_threshold`: 0.26
- positivos totais: 843
- `profile`: 514
- AUC geral: 0.9989
- EER geral: 0.0089

Esse candidato reduziu cobertura e nao melhorou as metricas gerais, entao o
baseline atual foi mantido.

## Uso operacional

Extrair embeddings em CPU:

```powershell
.\.venv\Scripts\face-profile.exe extract `
  --manifest manifest.csv `
  --out-dir artifacts `
  --save-aligned `
  --model-name buffalo_l `
  --det-size 320 `
  --min-det-score 0.50 `
  --min-face-area-ratio 0.01
```

Treinar, calibrar e avaliar:

```powershell
.\.venv\Scripts\face-profile.exe fit `
  --features artifacts/embedding_manifest.csv `
  --embeddings artifacts/embeddings.npy `
  --out-dir artifacts/model `
  --profile-splits profile `
  --use-ocsvm

.\.venv\Scripts\face-profile.exe calibrate `
  --model-dir artifacts/model `
  --features artifacts/embedding_manifest.csv `
  --embeddings artifacts/embeddings.npy `
  --positive-splits calib_pos `
  --negative-splits calib_neg

.\.venv\Scripts\face-profile.exe evaluate `
  --model-dir artifacts/model `
  --features artifacts/embedding_manifest.csv `
  --embeddings artifacts/embeddings.npy `
  --positive-splits test_pos `
  --negative-splits test_neg `
  --out-dir artifacts/eval
```

Demo OpenCV:

```powershell
.\.venv\Scripts\face-profile.exe demo `
  --model-dir artifacts/model `
  --camera 0 `
  --frame-window 9 `
  --model-name buffalo_l `
  --det-size 320
```

## Limitacoes

- Os rotulos positivos e negativos atuais foram derivados dos embeddings, nao
  de anotacao humana.
- As metricas atuais validam separacao geometrica do cluster escolhido, nao uma
  definicao juridica, investigativa ou operacional de perfil facial.
- O dataset tem muitas identidades com poucas imagens; isso limita validacao
  robusta de variacao intra-pessoa.
- Auditoria humana ainda e necessaria antes de uso real.

## Proximo passo recomendado

Definir o perfil facial alvo real com exemplos positivos revisados manualmente.
Depois disso, atualizar `manifest.csv`, retreinar, recalibrar e comparar as
metricas contra este baseline semantico.
