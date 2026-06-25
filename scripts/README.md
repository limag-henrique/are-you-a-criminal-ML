# ia-usp-jailer

## Padronizacao de faces

O script `scripts/standardize_faces.py` prepara as imagens de `data_curated` para uso em modelos de similaridade facial. Ele cria novas imagens padronizadas e nunca altera nem sobrescreve os arquivos originais.

### Instalar dependencias

Recomenda-se usar um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

O script prefere MediaPipe para deteccao, landmarks e segmentacao de fundo. Se MediaPipe nao estiver disponivel no Python atual, ele usa fallback com OpenCV para deteccao/alinhamento e uma mascara eliptica para neutralizar o fundo.

### Rodar com fundo branco

Este e o modo padrao e salva JPG em RGB, sem canal alpha, com fundo branco puro.

```bash
python scripts/standardize_faces.py ^
  --input_root "data_curated" ^
  --output_root "data_processed/face_standardized" ^
  --size 512 ^
  --background white
```

Saida principal:

```text
data_processed/face_standardized/white_bg/Best quality
data_processed/face_standardized/white_bg/Mid quality
data_processed/face_standardized/white_bg/Low quality
```

### Rodar com fundo transparente

Este modo salva PNG com canal alpha. Ele e util quando voce quer preservar a separacao do rosto/pessoa para composicao posterior, mas para a maioria dos treinamentos de similaridade facial o fundo branco costuma ser mais simples e consistente.

```bash
python scripts/standardize_faces.py ^
  --input_root "data_curated" ^
  --output_root "data_processed/face_standardized" ^
  --size 512 ^
  --background transparent
```

Saida principal:

```text
data_processed/face_standardized/transparent_bg/Best quality
data_processed/face_standardized/transparent_bg/Mid quality
data_processed/face_standardized/transparent_bg/Low quality
```

### Testar com uma amostra

Para processar apenas 10 imagens de cada pasta de qualidade:

```bash
python scripts/standardize_faces.py ^
  --input_root "data_curated" ^
  --output_root "data_processed/face_standardized_test" ^
  --size 512 ^
  --background white ^
  --sample_per_folder 10
```

### Arquivos rejeitados

Imagens sem face detectada sao copiadas para:

```text
data_processed/face_standardized/rejected/no_face_detected
```

Imagens corrompidas ou ilegiveis sao copiadas para:

```text
data_processed/face_standardized/rejected/corrupted
```

Os nomes preservam o stem original e adicionam um hash curto derivado do caminho original, reduzindo risco de colisao e mantendo rastreabilidade.

### Relatorio CSV

O arquivo abaixo e gerado a cada execucao:

```text
data_processed/face_standardized/processing_report.csv
```

Colunas principais:

- `original_path`: caminho da imagem original.
- `output_path`: caminho da imagem padronizada ou da copia rejeitada.
- `quality_folder`: pasta de qualidade original.
- `status`: `success`, `rejected`, `corrupted` ou `error`.
- `reason`: motivo detalhado do status.
- `num_faces_detected`: quantidade de faces detectadas; quando ha mais de uma, a maior e escolhida.
- `face_confidence`: confianca da face principal.
- `original_width` e `original_height`: dimensoes apos correcao de orientacao EXIF.
- `output_width` e `output_height`: dimensoes finais da imagem salva.

### Comando para processar todo o dataset

```bash
python scripts/standardize_faces.py ^
  --input_root "data_curated" ^
  --output_root "data_processed/face_standardized" ^
  --size 512 ^
  --background white
```
