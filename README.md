# Face Profile ML

Projeto Python para modelagem de perfil facial usando embeddings pre-treinados.
Ele nao faz fine-tuning de rede neural: a rede e usada apenas como extrator de
embeddings, e o perfil e modelado com metodos classicos.

## Objetivo operacional

O sistema recebe uma imagem de rosto enviada pelo usuario, detecta/alinha o
maior rosto, extrai um embedding facial normalizado e compara esse embedding
com o conjunto local de imagens de referencia. A resposta operacional e uma
pontuacao de similaridade visual, acompanhada dos candidatos mais proximos,
cosine bruto, percentil contra impostores e estimativa de taxa de falso match.

A pontuacao nao deve ser interpretada como prova de identidade nem como
probabilidade absoluta. Quando a imagem enviada ja existe nas referencias, o
resultado pode chegar a 100% por comparacao quase identica com a propria imagem.

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
- Aplicativo local de similaridade por galeria com top-k candidatos, imagem da
  referencia mais proxima, COSIM, FMR estimado, avisos de qualidade e marcacao
  de imagem de referencia quase identica.
- Persistencia em `.pkl`, `.npy` e `.json`.

## Quick setup

Use Python 3.10, 3.11 ou 3.12. O backend `insightface` ainda costuma ter
suporte irregular em Python 3.13+; portanto, nao use Python 3.13 para rodar a
aplicacao com deteccao facial e ArcFace.

Como este repositorio ja inclui os modelos treinados e configurados em
`artifacts/model`, nao e necessario rodar o pipeline de treinamento para usar a
aplicacao local.

### 1. Primeira execucao completa

Use estes passos se voce nunca rodou o projeto nesta maquina. Rode os comandos
abaixo no PowerShell, dentro da pasta do projeto.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[arcface]"
.\.venv\Scripts\python.exe -c "import insightface, onnxruntime; print('InsightFace e ONNX Runtime instalados com sucesso')"
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py --model-dir artifacts/model --model-name buffalo_s --det-size 320 --port 8766
```

O comando `python --version` deve mostrar Python 3.10, 3.11 ou 3.12. Se ele
mostrar Python 3.13 ou superior, instale uma versao suportada e crie o ambiente
virtual com ela, por exemplo:

```powershell
py -3.12 -m venv .venv
```

### 2. Projeto ja instalado

Use estes comandos se o ambiente virtual e as dependencias ja foram instalados.

```powershell
.\.venv\Scripts\python.exe -c "import insightface, onnxruntime; print('Dependencias OK')"
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py --model-dir artifacts/model --model-name buffalo_s --det-size 320 --port 8766
```

Apos iniciar, acesse `http://127.0.0.1:8766` no navegador e permita o uso da
camera.

### Corrigindo erro do InsightFace

Se aparecer a mensagem abaixo, o ambiente ativo nao tem o `insightface` e o
`onnxruntime` instalados:

```text
InsightFace is required for detection/alignment/ArcFace embeddings. Install with: pip install insightface onnxruntime
```

Corrija rodando:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[arcface]"
.\.venv\Scripts\python.exe -c "import insightface, onnxruntime; print('InsightFace e ONNX Runtime instalados com sucesso')"
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py --model-dir artifacts/model --model-name buffalo_s --det-size 320 --port 8766
```

Se o comando de instalacao falhar, confira primeiro a versao do Python com
`python --version`. Em Python 3.13+, o extra `arcface` nao instala
`insightface`/`onnxruntime` por causa das restricoes de compatibilidade do
projeto.

Para GPU, instale o extra manualmente e garanta que o runtime CUDA esteja
correto:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gpu]"
```

Em Python 3.13+, o pacote principal instala, mas a extracao ArcFace dependera
de o ecossistema InsightFace/ONNX Runtime publicar wheels compativeis.

---

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

## 🧠 Treinamento e Avaliação (Avançado)

Caso você queira rodar o treinamento desde o zero com o próprio conjunto de imagens (e extrair os embeddings), você precisará rodar o pipeline completo abaixo.

Extrair embeddings:

```bash
face-profile extract ^
  --manifest manifest.csv ^
  --out-dir artifacts ^
  --save-aligned ^
  --model-name buffalo_s ^
  --det-size 320
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

## Auditoria visual

Depois de extrair embeddings e treinar/calibrar um modelo, gere um relatorio
HTML com thumbnails por split:

```bash
python scripts/build_audit_report.py ^
  --features artifacts/embedding_manifest.csv ^
  --embeddings artifacts/embeddings.npy ^
  --model-dir artifacts/model ^
  --out-dir artifacts/audit ^
  --samples-per-split 64 ^
  --hard-examples-per-class 32
```

Abra `artifacts/audit/index.html`. O relatorio tambem grava:

- `artifacts/audit/audit_samples.csv`
- `artifacts/audit/audit_summary.json`

Use a auditoria para validar visualmente se `profile`, `calib_pos` e
`test_pos` representam o perfil facial pretendido. Se o perfil alvo real for
outro, ajuste `manifest.csv` com anotacao humana ou rode novamente
`scripts/build_semantic_manifest.py` com thresholds diferentes e retreine.

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

A rede neural base (extrator de features) não é treinada neste pipeline (são utilizados modelos pré-treinados, como o ArcFace). O treinamento realizado pelo projeto diz respeito apenas ao modelo de perfil facial (One-Class SVM, métricas de distância) e ao modelo calibrador, que aprendem a partir dos embeddings extraídos. Trocar ArcFace por AdaFace ou MagFace pode ser feito criando outro extrator que respeite a interface de embeddings L2 normalizados. O restante do pipeline continua igual.

Os splits atuais foram derivados da geometria dos embeddings, nao de anotacao
humana. Portanto, as metricas medem a separacao desse perfil semantico
automatico contra negativos distantes; elas nao provam que o perfil coincide
com uma definicao operacional externa.
