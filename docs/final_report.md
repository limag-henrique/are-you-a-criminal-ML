# Face Profile ML - Relatorio final

## Objetivo

Este projeto implementa um pipeline de modelagem de perfil facial usando
embeddings pre-treinados. A rede neural e usada apenas para deteccao,
alinhamento e extracao de embeddings. Nao ha fine-tuning.

## Metodologia

1. `manifest.csv` lista imagens, identidade aproximada, qualidade, split e peso.
2. `face-profile extract` detecta/alinha faces com InsightFace e salva embeddings L2.
3. `scripts/build_semantic_manifest.py` cria splits semanticos a partir da
   geometria dos embeddings.
4. `face-profile fit` modela o perfil com media ponderada, top-k cosine,
   Mahalanobis regularizada e One-Class SVM opcional.
5. `face-profile calibrate` ajusta score calibrado com regressao logistica.
6. `face-profile evaluate` calcula ROC, AUC, EER, FMR e FNMR, inclusive por qualidade.
7. `scripts/build_audit_report.py` gera HTML/CSV de auditoria visual dos splits.

## Baseline atual

Extracao feita com `buffalo_s` em CPU:

- imagens no manifesto: 9.584
- embeddings extraidos: 9.472
- sem face detectada: 112
- dimensao: 512

Manifesto semantico atual:

- `profile`: 751
- `calib_pos`: 219
- `test_pos`: 258
- `calib_neg`: 1.931
- `test_neg`: 2.069
- `ignore`: 4.356

Metricas finais do baseline:

- AUC geral: 0.9993
- EER geral: 0.0087
- AUC high: 1.0000
- AUC mid: 0.9983
- AUC low: 0.9999

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
  --model-name buffalo_s `
  --det-size 320
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
  --model-name buffalo_s `
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
