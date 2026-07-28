# Plano de auditoria de pesquisa v2

## Escopo e salvaguardas

Esta auditoria avalia propriedades computacionais e de proveniência de um conjunto
institucional já presente no repositório. Ela não realiza identificação nominal,
classificação de raça/cor por imagem ou publicação de fotografias, nomes, URLs ou
outros dados pessoais. Identificadores usados nos resultados públicos serão
derivados por hash com sal local e os relatórios conterão somente agregados.

Resultados e artefatos históricos serão tratados como entradas somente de leitura.
Nenhum arquivo histórico será apagado ou sobrescrito. Resultados novos ficarão
exclusivamente em `research_audit_v2/outputs/`.

## Achados da inspeção inicial

- Commit atual: `8eefdacd064c3fb3c0284d87769ebf37b5cfd4be`.
- O commit histórico `99d6b800` existe no histórico local; sua árvore contém
  artefatos de processamento e imagens alinhadas.
- Há embeddings contemporâneos em `artifacts/embeddings.npy`, manifestos em
  `manifest.csv` e `artifacts/embedding_manifest.csv`, além de logs e relatórios
  de pré-processamento em `analisis_report/`.
- O repositório atual contém extração, manifesto e perfil/similaridade, mas a
  busca de código não identificou um pipeline de K-means, uma regra de seleção de
  cluster-alvo ou a regressão logística descritos no pedido. A auditoria deve
  portanto distinguir resultados reproduzidos de reconstruções analíticas novas.
- Há alterações não rastreadas e dois documentos rastreados removidos no diretório
  de trabalho antes desta auditoria; elas permanecem fora do escopo.
- `pytest -q` não pôde ser iniciado porque o executável `pytest` não está no PATH;
  a execução via interpretador será verificada posteriormente e o estado será
  registrado sem mascarar falhas.

## Plano de implementação

1. Congelar a inspeção: produzir inventário com hashes, contagens, esquema dos
   manifestos, referências Git e disponibilidade de artefatos, sem copiar dados
   pessoais para saídas públicas.
2. Criar módulos determinísticos para proveniência, pseudonimização, deduplicação,
   métricas de agrupamento, seleção explícita de alvo, incerteza, enriquecimento e
   modelos preditivos. Cada módulo deve falhar de forma informativa quando faltar
   entrada histórica.
3. Adotar configurações `development` e `final`: desenvolvimento com 20 sementes
   e permutações reduzidas; final com 100 sementes, k em 32/48/64/80/96/128 e
   parâmetros explicitamente registrados. A execução final poderá ser custosa.
4. Reconstruir apenas transições respaldadas por arquivos existentes; rotular cada
   elo como documentado, inferido, provável ou não resolvido. Não inventar a
   explicação das contagens históricas ausentes.
5. Produzir análises de duplicidade e estabilidade a partir de imagens/embeddings
   disponíveis. Similaridade extrema será descrita somente como grupo de
   duplicidade provável, nunca como identidade.
6. Executar modelos nulos, permutações, validação cruzada estratificada e por
   grupo, além de enriquecimento por fonte, reportando variabilidade interna
   condicional ao conjunto auditado, não inferência populacional.
7. Gerar tabelas, figuras vetoriais e relatórios públicos livres de dados pessoais,
   acompanhados de manifesto de execução e testes sintéticos independentes das
   imagens reais.
8. Executar testes e uma corrida de desenvolvimento antes da corrida final. A
   corrida final só será apresentada como concluída se todos os insumos necessários
   estiverem localmente disponíveis e o tempo/recursos permitirem.

## Compatibilidade com o repositório observado

A estrutura solicitada é compatível com o repositório: pode ser adicionada como
um pacote Python isolado, consumindo `artifacts/embeddings.npy`, os manifestos e
os relatórios históricos por caminhos configuráveis. Ela não deve substituir o
pacote existente `face_profile_ml` nem seus artefatos. A limitação crítica é a
ausência, até aqui, do código/estado preservado da análise de agrupamento e do
artefato explicitamente identificado como release 1.0.1; comparações históricas
só serão executadas se esses materiais puderem ser localizados de modo verificável.
