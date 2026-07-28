# Limitações detectadas na inspeção inicial

1. A árvore de trabalho já contém remoções e arquivos não rastreados que não
   pertencem a esta auditoria. Eles não serão modificados.
2. O repositório observado contém imagens e manifestos com dados pessoais. Eles
   são insumos restritos; saídas públicas da auditoria usarão somente identificadores
   pseudonimizados e agregados.
3. O commit `99d6b800` existe, mas a presença do commit não demonstra, por si só,
   a disponibilidade de todos os pesos, ambiente e resultados que formaram a
   análise histórica. Esta disponibilidade será testada por hash e caminho.
4. Não foi localizada tag/release `1.0.1` no repositório local durante a inspeção
   inicial.
5. Não foi localizado código atual de K-means, seleção de cluster-alvo, OCSVM ou
   regressão logística. Logo, qualquer regra de alvo implementada na auditoria será
   uma reconstrução explicitamente nova, salvo descoberta posterior de fonte
   histórica verificável.
6. O comando literal `pytest -q` falhou antes da coleta por ausência do executável
   no PATH, embora o pacote esteja instalado. A auditoria verificará o comando
   equivalente `python -m pytest -q` e registrará ambos os estados.
7. As contagens históricas citadas no pedido (11.724, 9.764, 9.546, 9.584 e 9.482)
   exigem reconciliação contra os arquivos disponíveis; não serão presumidas corretas
   até que a evidência seja localizada.
8. Sem rótulos de identidade validados, similaridade visual ou de embedding não
   permite afirmar que dois registros retratam a mesma pessoa.
