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
- Score visual de galeria que combina melhor vizinho, top-k ponderado,
  densidade de referencias semelhantes, percentil impostor e distinctividade.
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
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py --model-dir artifacts/model --model-name buffalo_l --det-size 320 --port 8766
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
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py --model-dir artifacts/model --model-name buffalo_l --det-size 320 --port 8766
```

Apos iniciar, acesse `http://127.0.0.1:8766` no navegador e permita o uso da
camera.

### Como o score da galeria e calculado

O app local usa o ArcFace/InsightFace somente para extrair embeddings L2
normalizados. A comparacao por galeria calcula cosine similarity com todas as
referencias e retorna campos separados:

- `best_cosine` e `best_match_similarity_percent`: similaridade da referencia
  mais proxima.
- `similarity_percent` / `overall_gallery_similarity_percent`: score visual
  final de 0 a 100, combinando melhor match, top-k ponderado, densidade de
  referencias acima dos limiares e percentil contra impostores.
- `raw_scores`: cosines brutos de melhor match, media top-k e top-k ponderado.
- `threshold_counts`: quantidade de referencias acima de `low`, `medium`,
  `high`, `very_high` e `near_duplicate`, contada por grupos de duplicatas
  proximas para nao inflar densidade com copias da mesma referencia.
- `percentile_rank` e `estimated_false_match_rate`: estimativas relativas a
  distribuicao impostora amostrada na propria galeria.
- `distinctiveness_percent` / `uniqueness_percent`: quao pouco denso e o
  entorno do rosto na galeria; rostos parecidos com muitas referencias tendem a
  ter menor distinctividade.

O score nao e prova de identidade nem probabilidade absoluta. Uma imagem quase
identica a uma referencia local pode chegar perto de 100%, mas o retorno marca
`reference_image_match=true` e inclui um aviso de possivel duplicata.

### Arquitetura de matching consentido

O fluxo operacional e:

1. manter apenas imagens autorizadas no `manifest.csv`;
2. carregar cada imagem com correcao de orientacao EXIF;
3. detectar faces com InsightFace, rejeitando referencias com deteccao fraca,
   rosto pequeno ou multiplas faces quando `--allow-multiple-faces` nao for usado;
4. alinhar pela geometria facial do proprio InsightFace e extrair embedding
   ArcFace L2-normalizado;
5. calcular cosine similarity entre o upload e todos os embeddings locais;
6. retornar somente similaridade contra referencias conhecidas, top-k
   candidatos, percentil, FMR estimado e avisos de qualidade.

O modelo padrao e `buffalo_l`, que usa detector SCRFD e reconhecimento ArcFace
512-D. Ele e mais pesado que `buffalo_s`, mas e a escolha padrao para a galeria
porque reduz variancia de deteccao/alinhamento e melhora robustez em imagens
com pose, iluminacao e qualidade diferentes. O modelo nao e fine-tuned neste
projeto.

### Formula do score visual

`similarity_percent` e uma escala interpretavel, nao uma probabilidade. Ela
combina:

- melhor cosine individual;
- cosine medio do top-k e top-k ponderado;
- densidade de referencias acima dos limiares configurados;
- percentil contra a distribuicao impostora local, excluindo duplicatas quase
  identicas quando possivel.

Campos separados deixam claro o significado operacional:

- `best_match_similarity_percent`: quao forte e o candidato mais proximo.
- `overall_gallery_similarity_percent`: quao parecido o upload e com a galeria
  como um todo.
- `distinctiveness_percent`: quao pouco denso e o entorno facial na galeria.
- `estimated_false_match_rate`: fracao de comparacoes impostoras de calibracao
  com score igual ou maior que o melhor cosine.

Os limiares de cosine podem ser ajustados no app:

```powershell
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py ^
  --model-dir artifacts/model ^
  --model-name buffalo_l ^
  --det-size 320 ^
  --similarity-thresholds low=0.30,medium=0.40,high=0.55,very_high=0.70,near_duplicate=0.985
```

### Limiares, privacidade e limitacoes

Os limiares padrao foram escolhidos para separar baixa, media, alta,
muito alta e quase duplicata em embeddings ArcFace normalizados. Ajuste esses
valores com validacao local se a galeria mudar de dominio, camera, qualidade ou
populacao. Lookalikes podem receber score moderado ou alto quando muitos
candidatos ficam acima dos limiares, mas isso nao deve ser tratado como prova
de identidade.

Salvaguardas esperadas para uso real:

- usar somente imagens coletadas com autorizacao e finalidade documentada;
- manter a galeria local protegida e auditar quem adiciona/remove referencias;
- nao enviar imagens ou embeddings para servicos externos durante o matching;
- expor resultado como similaridade visual entre referencias conhecidas, nunca
  como identificacao absoluta;
- remover imagens/embeddings quando o consentimento expirar ou for revogado.

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
.\.venv\Scripts\python.exe scripts\serve_similarity_app.py --model-dir artifacts/model --model-name buffalo_l --det-size 320 --port 8766
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
  --model-name buffalo_l ^
  --det-size 320 ^
  --min-det-score 0.50 ^
  --min-face-area-ratio 0.01
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
