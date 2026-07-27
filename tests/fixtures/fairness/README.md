# Dados de coorte para auditoria

Este diretório contém somente um esquema sintético; ele não é uma base de
faces nem uma taxonomia para rotular pessoas por aparência. Para o estudo,
substitua o arquivo-modelo por comparações de verificação produzidas em um
conjunto de imagens autorizado, com rótulos de coorte voluntários ou extraídos
de fonte documental legítima e auditável.

Não deduza raça, etnia, nacionalidade ou cor da pele por fotografia, nome,
país de origem ou registro policial. `skin_tone_protocol`, quando aplicável,
deve resultar de protocolo humano documentado, com anotadores, iluminação,
escala, acordo entre avaliadores e opção de não resposta. Os valores são
exemplos e não categorias obrigatórias.

Colunas essenciais:

- `comparison_id`: identificador pseudonimizado da comparação;
- `label`: `1` para comparação genuína e `0` para impostora;
- `score`: escore contínuo já calculado pelo sistema;
- coortes: usar apenas atributos que o comitê de ética/consentimento autorizar;
- `quality`, `capture_condition` e `source_jurisdiction`: controles para não
  atribuir à demografia um erro causado por qualidade, origem ou contexto.

O artefato de auditoria não deve incluir fotos, nomes, identificadores públicos
ou embeddings. Mantenha o arquivo real fora do Git e aplique supressão para
grupos pequenos.
