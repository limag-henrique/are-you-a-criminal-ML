# Benchmarks externos para auditoria acadêmica

O repositório não contém imagens dos benchmarks. Baixe as versões integrais,
documentadas e com metadados de proveniência usando:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_research_benchmarks.py
.\.venv\Scripts\python.exe scripts\build_research_benchmark_catalogs.py
.\.venv\Scripts\python.exe scripts\validate_research_benchmarks.py
```

O procedimento adiciona os dados em `datasets/external/`, diretório ignorado
pelo Git, e registra URL, tamanho e SHA-256 de cada arquivo em
`provenance.json`.

O segundo comando constrói `fairface_catalog.csv`, `bfw_catalog.csv` e os
`bfw_pairs.csv` oficiais no mesmo diretório ignorado. O BFW é expandido até as
faces recortadas canônicas somente depois de validar o MD5 publicado pelos
autores.

O terceiro comando decodifica cada imagem e confirma que todos os pares BFW
apontam para imagens existentes, têm rótulo binário e preservam os cinco folds.
O relatório local é salvo em `benchmark_validation.json`.

## Recursos incorporados

- **FairFace**: a liberação oficial declara licença CC BY 4.0 e fornece 108.501
  retratos alinhados, com rótulos de faixa etária, gênero e sete categorias de
  raça do benchmark. É apropriado para auditoria de cobertura de extração e
  análises de pares impostores; não possui identidade repetida suficiente para
  medir FNMR de verificação.
- **Balanced Faces in the Wild (BFW)**: a liberação dos autores descreve uma
  base de pesquisa para verificação, equilibrada em oito estratos definidos
  pelo benchmark e com pares genuínos/impostores. Os arquivos de imagem não
  devem ser redistribuídos. As anotações são categorias históricas do benchmark,
  não devem ser apresentadas como autoidentificação nem inferidas em novas
  pessoas.

Nenhum recurso deve ser usado para investigar indivíduos, predizer
criminalidade ou alimentar uma lista de procurados. Para a publicação, informe
as licenças/termos consultados, a data, os hashes registrados, a versão dos
pesos ArcFace e as limitações dos rótulos do benchmark.
