# Inventário reprodutível de benchmarks externos

## Delimitação

Os arquivos de imagem deste inventário ficam em `datasets/external/`, que é
ignorado pelo Git para não redistribuir dados biométricos. O script
`scripts/fetch_research_benchmarks.py` recupera as liberações integrais e salva
URLs, data de acesso, tamanho e SHA-256 em `provenance.json`; em seguida,
`scripts/build_research_benchmark_catalogs.py` gera os catálogos e pares
locais. A execução é idempotente quando os arquivos e hashes já existem.

Nenhuma categoria é inferida pela aplicação. `source_race_label`,
`source_gender_label`, `source_group` e atributos similares preservam apenas a
anotação fornecida pelo benchmark. Eles não são autoidentificação, diagnóstico,
medida biológica ou rótulo aplicável a uma pessoa nova.

## Aquisição validada em 2026-07-27

| Recurso | Imagens locais | Uso metodológico | Verificações realizadas |
| --- | ---: | --- | --- |
| FairFace | 97.698 | cobertura de extração e distribuição de similaridade entre impostores | SHA-256 dos três arquivos, CRC do ZIP integral, todos os caminhos verificados, 1.400 JPEGs decodificados em amostra estratificada determinística |
| BFW | 20.000 | verificação 1:1 em pares genuínos/impostores | SHA-256 do release, MD5 e CRC do ZIP canônico de faces, todos os caminhos verificados, 800 JPEGs decodificados em amostra estratificada determinística |

O BFW local contém 923.898 pares: 242.519 genuínos, 681.379 impostores e cinco
folds oficiais. Há 2.500 imagens em cada um dos oito grupos definidos pelo
benchmark (`AF`, `AM`, `BF`, `BM`, `IF`, `IM`, `WF`, `WM`). A validação confirmou
referência existente para cada uma das duas imagens de todos os pares e rótulos
binários válidos. O relatório local correspondente é
`datasets/manifests/benchmark_validation.json`.

O FairFace tem classes de idade, gênero e sete categorias históricas de raça do
benchmark; a distribuição local está em
`datasets/manifests/benchmark_catalog_summary.json`. Como essa liberação não
oferece identidades repetidas para os pares necessários, ela não deve ser usada
para estimar FNMR de verificação.

## Reexecução

```powershell
.\.venv\Scripts\python.exe scripts\fetch_research_benchmarks.py
.\.venv\Scripts\python.exe scripts\build_research_benchmark_catalogs.py
.\.venv\Scripts\python.exe scripts\validate_research_benchmarks.py `
  --decode-mode stratified `
  --sample-per-stratum 100
```

`--decode-mode all` permanece disponível para uma decodificação integral, mas
pode exceder o limite de execução de uma sessão curta. A checagem CRC integral
ocorre antes da extração de cada ZIP e a validação padrão verifica a existência
de todos os caminhos, todos os pares BFW e uma amostra estratificada de pixels.

## Termos, citação e limites

- FairFace: a página oficial declara CC BY 4.0. Cite Karkkainen & Joo (2021) e
  mantenha a atribuição exigida pela licença.
- BFW: use exclusivamente como benchmark de pesquisa sob os termos da
  liberação dos autores; não publique, faça upload ou redistribua as imagens.
  Cite Robinson et al. (2020).
- Não associe nenhuma das bases a crime, suspeita, culpa, risco ou lista de
  pessoas procuradas. Elas avaliam taxas de erro de comparação facial, não
  comportamento humano.
- O artigo deve informar que os rótulos representam o esquema das bases de
  origem e as limitações de seleção, rotulagem e representatividade. Não cabe
  concluir causalidade ou superioridade/inferioridade de qualquer grupo.
