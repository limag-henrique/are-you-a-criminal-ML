# Face Profile ML

Projeto Python para modelagem de perfil facial usando embeddings pre-treinados.
Ele nao faz fine-tuning de rede neural: a rede e usada apenas como extrator de
embeddings, e o perfil e modelado com metodos classicos.

## O que esta implementado

- Leitura e validacao de `manifest.csv` com `path`, `subject_id`, `quality`,
  `split` e `weight`.
- Deteccao e alinhamento facial com InsightFace.
- Extracao de embeddings ArcFace pre-treinados (`buffalo_l` por padrao).
- Normalizacao L2 dos embeddings.
- Modelo de perfil com media ponderada, top-k cosine similarity, Mahalanobis
  regularizada e One-Class SVM opcional.
- `score_raw` combinado a partir dos metodos.
- Calibracao de score com exemplos positivos e negativos.
- ROC, AUC, EER, FMR e FNMR.
- Avaliacao separada por qualidade `high`, `mid` e `low`.
- Demo em tempo real com OpenCV usando mediana de multiplos frames.
- Persistencia em `.pkl`, `.npy` e `.json`.

## Instalacao

Use Python 3.10, 3.11 ou 3.12. O backend `insightface` ainda costuma ter
suporte irregular em Python 3.13+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[arcface]"
```

Para GPU, instale o extra e garanta que o runtime CUDA esteja correto:

```bash
pip install -e ".[gpu]"
```

Em Python 3.13+, o pacote principal instala, mas a extracao ArcFace dependera
de o ecossistema InsightFace/ONNX Runtime publicar wheels compativeis.

## Manifest

O `manifest.csv` deve ter as colunas:

```csv
path,subject_id,quality,split,weight
photos/Mid quality/example.jpg,person_001,mid,profile,1.0
photos/Low quality/query.jpg,person_002,low,calib_neg,1.0
```

Valores aceitos para `quality`: `high`, `mid`, `low`, alem de aliases como
`best quality`, `mid quality` e `low quality`.

Uma convencao simples de splits:

- `profile` ou `enroll`: imagens positivas usadas para construir o perfil.
- `calib_pos` e `calib_neg`: exemplos para calibrar o score.
- `test_pos` e `test_neg`: exemplos para avaliacao final.

Como o dataset tem uma imagem por pessoa, o projeto modela um perfil agregado
de um conjunto positivo, nao faz treino supervisionado de identidade.

## Fluxo principal

Extrair embeddings:

```bash
face-profile extract ^
  --manifest manifest.csv ^
  --out-dir artifacts ^
  --save-aligned
```

Treinar o perfil:

```bash
face-profile fit ^
  --features artifacts/embedding_manifest.csv ^
  --embeddings artifacts/embeddings.npy ^
  --out-dir artifacts/model ^
  --profile-splits profile,enroll ^
  --use-ocsvm
```

Calibrar:

```bash
face-profile calibrate ^
  --model-dir artifacts/model ^
  --features artifacts/embedding_manifest.csv ^
  --embeddings artifacts/embeddings.npy ^
  --positive-splits calib_pos ^
  --negative-splits calib_neg
```

Avaliar:

```bash
face-profile evaluate ^
  --model-dir artifacts/model ^
  --features artifacts/embedding_manifest.csv ^
  --embeddings artifacts/embeddings.npy ^
  --positive-splits test_pos ^
  --negative-splits test_neg ^
  --out-dir artifacts/eval
```

Demo em tempo real:

```bash
face-profile demo ^
  --model-dir artifacts/model ^
  --camera 0 ^
  --frame-window 9
```

## Artefatos gerados

Extracao:

- `artifacts/embeddings.npy`
- `artifacts/embedding_manifest.csv`
- `artifacts/aligned/*.jpg` quando `--save-aligned` e usado

Modelo:

- `artifacts/model/profile_model.pkl`
- `artifacts/model/profile_embeddings.npy`
- `artifacts/model/profile_mean.npy`
- `artifacts/model/profile_inv_cov.npy`
- `artifacts/model/model_metadata.json`
- `artifacts/model/calibrator.pkl` apos calibracao
- `artifacts/model/calibrator_metadata.json` apos calibracao

Metricas:

- `artifacts/eval/eval_scores.csv`
- `artifacts/eval/metrics.json`

## Observacoes

O modelo neural nao e treinado. Trocar ArcFace por AdaFace ou MagFace pode ser
feito criando outro extrator que respeite a interface de embeddings L2
normalizados. O restante do pipeline continua igual.
