Listas públicas de procurados não são medidas populacionais de criminalidade: auditoria sociotécnica de seleção e validade inferencial em um pipeline de similaridade facial

Public wanted-person lists are not population measures of criminality: a sociotechnical audit of selection and inferential validity in a face-similarity pipeline

HENRIQUE LIMA GUSMÃO¹

¹ Ciência da Computação, Universidade Federal de Minas Gerais, Belo Horizonte, MG, Brasil

Autor correspondente: henriquelimagusmao@ufmg.br  |  ORCID: https://orcid.org/0009-0003-5884-2489

# Resumo

Listas públicas de procurados resultam de decisões de investigação, priorização, cooperação e publicidade; não constituem amostras da criminalidade populacional. Este artigo audita os artefatos disponíveis em um estado preservado de um pipeline de similaridade facial. Foram reconciliados manifestos e embeddings, auditadas duplicidades, avaliadas sensibilidades de ordem, minibatch e representação, e aplicado cross-fitting agrupado em cinco dobras. O manifesto contém 9.584 linhas e há 9.482 vetores válidos de 512 dimensões. O cross-fitting não deixou grupos de duplicidade atravessarem treino e teste e recuperou o rótulo sintético com ROC-AUC média de 0,929, PR-AUC de 0,479, F1 de 0,477 e Brier de 0,185; a variação entre dobras é material. A partição foi sensível à ordem dos registros, ao batch_size e à redução PCA-64: vários ARIs contra o método de referência ficaram entre aproximadamente 0,07 e 0,11, e o Jaccard do cluster-alvo chegou a zero. Esses resultados quantificam dependência interna de um rótulo construído no espaço de embeddings, não reconhecimento facial, validade externa, criminalidade ou equidade biométrica. Sem estado histórico completo, variável de fonte documentada e datas confiáveis, não são identificáveis a reprodução causal da análise histórica, a influência por fonte ou uma análise temporal.

Palavras-chave: listas de procurados; viés de seleção; validade de construto; auditoria algorítmica; similaridade facial.

# Abstract

Public wanted-person lists are shaped by investigation, prioritization, cooperation, and disclosure decisions; they are not samples of population-level criminality. This article audits the artifacts available in a preserved state of a face-similarity pipeline. We reconciled manifests and embeddings, audited duplicates, assessed sensitivity to record order, minibatch size, and representation, and applied five-fold grouped cross-fitting. The manifest contains 9,584 rows and 9,482 valid 512-dimensional vectors. No duplicate group crossed training and test sets in cross-fitting; mean ROC-AUC was 0.929, PR-AUC 0.479, F1 0.477, and Brier score 0.185 for recovery of the synthetic label, with material between-fold variation. Record ordering, batch size, and PCA-64 substantially changed the partition: several adjusted Rand indices against the reference method were about 0.07-0.11, and target-cluster Jaccard reached zero. These results quantify internal dependence of a label constructed in embedding space, not face recognition, external validity, criminality, or biometric fairness. A complete historical state, a documented source variable, and reliable dates were unavailable; causal reproduction of the historical analysis, source-influence analysis, and temporal analysis are therefore not identifiable.

Keywords: wanted-person lists; selection bias; construct validity; algorithmic auditing; face similarity.

# 1 Introdução

Desigualdades raciais no sistema penal são questões empíricas e institucionais; não podem ser resolvidas contando aparências em fotografias de pessoas procuradas. O SISDEPEN compila dados administrativos informados semestralmente pelas unidades da federação, enquanto uma lista pública de procurados registra outro estágio do processo: pessoas que uma instituição decidiu localizar e divulgar. Confundir esses universos transforma seleção institucional em atributo de grupos sociais.

A própria Interpol define a Red Notice como pedido internacional de localização e prisão provisória, não como mandado internacional ou declaração de culpa, e apenas parte dos avisos é exibida publicamente. A presença numa página depende da jurisdição, do tipo de medida, da política de publicidade, da vigência da busca, da existência de fotografia e da capacidade técnica de coleta. Pessoas encontradas e registros retirados deixam de estar observáveis. Portanto, a base é condicionada a um evento de seleção.

Em sistemas de aprendizagem de máquina, o problema envolve validade de construto, amostragem, mensuração, pré-processamento e avaliação. Um modelo pode separar com grande precisão um rótulo que ele próprio ajudou a construir e, ainda assim, não medir o fenômeno social alegado. Aqui, os embeddings faciais foram agrupados; um cluster foi escolhido por heurística; e escores derivados da mesma geometria foram avaliados contra esse rótulo. A pergunta científica é, então, menos “quão alta é a AUC?” e mais “qual grandeza esse experimento identifica?”.

O objetivo é auditar a cadeia sociotécnica do repositório ia-uspJailer, medir propriedades computacionais que são diagnosticáveis nos artefatos disponíveis e demarcar inferências não identificáveis. As contribuições são: (i) reconciliação do fluxo técnico disponível; (ii) avaliação agrupada, sem sobreposição de grupos entre treino e teste, da recuperação de um rótulo sintético; e (iii) evidência de sensibilidade da partição à ordem, ao minibatch e à representação. O estudo distingue rastreabilidade, reexecução, reprodução numérica e replicação independente.

As perguntas de pesquisa são: RQ1. Quais estágios institucionais e técnicos condicionam a inclusão de registros no corpus auditado? RQ2. Em que medida as partições e o cluster-alvo permanecem estáveis sob alterações controladas de ordem, minibatch e representação? RQ3. Em que medida escores internos recuperam um rótulo endógeno, em vez de validar um construto externo? RQ4. Quais alegações são identificáveis, diagnosticáveis ou não identificáveis com as variáveis, unidades e denominadores disponíveis?

# 2 Materiais e métodos

## 2.1 Unidade de análise, fontes e snapshot

A unidade observada é um registro público coletado, não uma ocorrência criminal nem uma pessoa única. Foram inspecionados manifestos, imagens alinhadas, embeddings ArcFace de 512 dimensões, relatórios de pré-processamento, código e saídas da auditoria. O commit de referência para os resultados finais é 734f5979b0b659727c71b07c77ee7f0b9431d2b9. O depósito Zenodo possui DOI de conceito 10.5281/zenodo.21536403 e registro versionado 10.5281/zenodo.21632858 (versão 1.0.1). Esses identificadores permitem localizar o estado preservado, mas não recompõem por si só ambiente histórico, regras de seleção de alvo ou pesos do artefato histórico.

## 2.2 Matriz de auditoria e classes de identificabilidade

Cada alegação foi classificada antes da análise. “Identificável” designa uma grandeza com variável, unidade e denominador observáveis no snapshot. “Diagnosticável” designa propriedade interna que pode ser testada sem representar o construto social de interesse. “Não identificável” designa estimando para o qual faltam variáveis, denominadores ou contrafactuais essenciais. Essa classificação evita tratar ausência de evidência como evidência de ausência.

Tabela 1 — Matriz formal de auditoria e identificabilidade

Fonte: elaboração própria. A classe “diagnosticável” refere-se à operação interna do pipeline, não à validade externa.

## 2.3 Modelo de seleção

A inclusão final foi representada como condicionamento em S=1, evento dependente de contexto institucional, intensidade de busca, decisão de publicação e existência de fotografia utilizável. Raça/cor autodeclarada e fenótipo percebido são construtos distintos e não foram observados. O diagrama é conceitual: organiza mecanismos plausíveis, mas não estima efeitos causais.

## 2.4 Lacuna de rastreabilidade da fonte

A atribuição por fonte presente em versões anteriores foi construída por regra documental auxiliar, confrontando identificadores e nomes-base com diretórios brutos. Essa regra não está propagada como campo documentado no manifesto que sustenta a auditoria final. Por isso, ela não é tratada como variável analítica nem como validação independente da fonte; seu uso para enriquecimento, influência ou comparação temporal foi suspenso.

## 2.5 Comparação de partições e estabilidade multissemente

O método de referência utiliza embeddings L2-normalizados e MiniBatchKMeans. As análises de sensibilidade são classificadas como sensibilidade, não como escolha retrospectiva de configuração: ordem dos registros, batch_size e PCA-64 foram comparados com a configuração de referência. A corrida de desenvolvimento planejada com 20 sementes e seis valores de k não terminou na janela operacional e foi interrompida sem manifesto final; nenhum resultado parcial dessa corrida é tratado como válido.

## 2.6 Ablação e desenho do experimento interno

O diagnóstico original de todos os registros tem alto risco de circularidade, pois a construção do cluster-alvo e o escore de centroide reutilizam a mesma geometria. Para evitar essa reutilização direta, o desenho confirmatório adotou cinco dobras por group_id: em cada dobra, o agrupamento, a seleção do cluster-alvo e o centroide foram ajustados somente no treino, e o teste foi apenas pontuado. A normalização L2 é determinística por vetor; nenhuma transformação global ajustada foi aplicada aos dados de teste. O rótulo permanece algorítmico e sintético: cross-fitting elimina o vazamento direto, não cria rótulo externo, identidade ou validade biométrica.

## 2.7 Dependência de fonte, qualidade técnica e imagens de conveniência

A análise de influência por fonte, enriquecimento, balanceamento e leave-one-source-out não foi executada como resultado final porque a atribuição de fonte não está documentada como variável no manifesto de auditoria. Não se imputou nem se inferiu fonte para preencher essa lacuna. Assim, esses efeitos são não identificáveis nos artefatos atualmente localizados e alegações anteriores de enriquecimento por fonte foram retiradas do corpo principal.

## 2.8 Níveis de verificabilidade computacional

Tabela 2 — Escada de verificabilidade adotada

Rastreabilidade permite localizar código, artefatos e transformações; reexecução permite executar o procedimento registrado; reprodução numérica requer recuperar os mesmos resultados a partir de estado preservado; e replicação independente exige novo conjunto ou protocolo. O teste de determinismo disponível comparou duas execuções consecutivas em CPU sobre uma matriz sintética e foi bit a bit idêntico. Ele não cobre GPU, outro BLAS, outro CPU nem ambiente limpo.

# 3 Resultados

## 3.1 Fluxo de registros e cobertura técnica

O estado preservado contém 9.584 linhas no manifesto analítico e 9.482 embeddings válidos. A matriz de embeddings tem 512 dimensões. Há imagens alinhadas, manifestos, embeddings e relatórios de pré-processamento. Os números históricos de 11.724 registros brutos, 9.764 entradas na padronização e 9.546 processamentos bem-sucedidos são mantidos apenas como reconciliação documental de estágios, pois o artefato bruto integral não estava disponível para auditoria independente nesta revisão. Entre as 102 falhas finais documentadas, 98 foram registradas como ausência de face detectável e quatro como múltiplas faces. Cobertura técnica não equivale a robustez biométrica.

Tabela 3 — Quantidades por estágio e mudança relativa ao estágio anterior

Percentuais usam 11.724 como denominador. A passagem de 9.546 para 9.584 não é retenção: 38 linhas foram acrescentadas sem transição documental completa.

## 3.2 Lacuna de rastreabilidade da fonte

A concentração por fonte e a atribuição documental descritas no manuscrito anterior não foram usadas como resultados analíticos finais. Embora existam relatos e diretórios de origem, a variável de fonte não está documentada no manifesto de auditoria que sustenta o cross-fitting. Sem essa variável e sem denominadores institucionais, não é possível estimar enriquecimento, influência por fonte, efeito causal de fonte ou composição temporal. A ausência é informativa e é tratada como limitação de proveniência, não como convite à imputação.

Tabela 4 — Análises por fonte: disponibilidade e identificabilidade

Não é reportada AUC de previsibilidade por fonte ou variáveis técnicas como resultado final. Os valores anteriores dependiam de uma atribuição de fonte que não constitui campo documentado no manifesto auditado; mantê-los sugeriria uma precisão que os artefatos disponíveis não sustentam.

## 3.3 Estados históricos e reprodução não controlável

Métricas históricas de ARI, NMI, pareamento húngaro e Jaccard aparecem em versões anteriores do manuscrito, mas não puderam ser reproduzidas de modo controlado nesta revisão. Faltam o estado histórico de agrupamento, a regra documentada de seleção do alvo e informações suficientes sobre o ambiente e os pesos de extração. Portanto, esses números são removidos como resultados finais e registrados no inventário como não reproduzidos.

Tabela 5 — Sensibilidades da partição e do cluster-alvo

A comparação entre estados preservados permanece uma hipótese metodologicamente relevante, mas não é tratada como evidência final de divergência histórica enquanto não houver estado histórico completo e verificável. A revisão não seleciona retrospectivamente as métricas mais favoráveis disponíveis.

## 3.4 Sensibilidades computacionais da partição

A estabilidade foi diretamente testada por sensibilidades controladas. Ao alterar a ordem dos registros, ARIs contra o método de referência ficaram entre 0,092 e 0,113, e o Jaccard do alvo variou de 0,001 a 0,007. Ao alterar batch_size, os ARIs ficaram entre 0,072 e 0,092 fora do valor de referência, e o Jaccard atingiu 0,000 em batch_size=2048. A representação PCA-64 obteve ARI=0,109 e Jaccard do alvo=0,000 contra a referência L2-normalizada. Esses resultados qualificam fortemente qualquer alegação de estabilidade; são sensibilidades condicionais aos registros observados, não incerteza populacional.

## 3.5 Circularidade e avaliação cross-fitted

Tabela 6 — Métricas por dobra no cross-fitting agrupado

A recuperação de um rótulo sintético foi avaliada no desenho agrupado e cross-fitted. Em todas as cinco dobras, group_overlap foi zero: grupos de duplicidade provável não atravessaram treino e teste. A média entre dobras foi ROC-AUC=0,929, PR-AUC=0,479, acurácia balanceada=0,725, precisão=0,477, revocação=0,477, F1=0,477 e Brier=0,185. As métricas variaram materialmente entre dobras; elas não são reconhecimento facial, validação externa ou avaliação biométrica por grupo.

O diagnóstico de todos os registros é mantido apenas para explicar a circularidade avaliativa: um escore de centroide derivado da mesma geometria que constrói o alvo pode recuperar esse alvo artificialmente bem. O resultado confirmatório é o cross-fitting por grupos, que evita o ajuste do agrupamento, do alvo e do centroide sobre os registros de teste, sem converter o rótulo sintético em construto externo.

# 4 Discussão

## 4.1 O que foi medido

O resultado empírico central é delimitado. RQ1: a observação é condicionada por publicação, coleta, padronização, detecção de face e extração, e a linhagem entre esses estágios é incompleta. RQ2: a partição e o cluster-alvo não foram estáveis sob alterações de ordem, minibatch e PCA-64. RQ3: o cross-fitting mostra recuperação mensurável do rótulo sintético sem vazamento direto, mas não valida um construto externo. RQ4: fluxo técnico, grupos de duplicidade e propriedades computacionais são identificáveis ou diagnosticáveis; influência por fonte, análise temporal, seletividade racial e desempenho biométrico não são identificáveis com os artefatos localizados.

## 4.2 Endogeneidade do alvo, circularidade e vazamento

O alvo é endógeno porque nasce no mesmo espaço geométrico do qual se deriva o escore. É mais preciso chamar o problema de circularidade avaliativa do que de validação externa: no diagnóstico de todos os registros há reutilização da geometria; no cross-fitting, essa reutilização direta no teste é removida, mas o resultado continua sendo um rótulo algorítmico. ROC-AUC, PR-AUC, F1 e Brier, portanto, descrevem recuperação interna da partição sintética.

## 4.3 Lacuna de fonte e não identificabilidade

A revisão não atribui causalidade nem influência a fontes. Sem campo de fonte documentado no manifesto e sem denominadores institucionais, enriquecimento, balanceamento e leave-one-source-out não são análises identificáveis. Essa restrição não enfraquece a necessidade de auditoria de proveniência; ela impede que uma narrativa de fonte seja apoiada por associação reconstruída ad hoc.

## 4.4 Validade de construto, interna, externa e estatística

A validade de construto é limitada porque aparência facial, raça/cor, identidade e criminalidade são conceitos distintos. A estabilidade computacional também é limitada: mudanças de ordem, batch_size e PCA-64 produziram ARIs baixos contra o método de referência e, em algumas condições, clusters-alvo disjuntos. O teste de determinismo CPU sintético foi bit a bit idêntico, mas não garante equivalência entre ambientes. A validade externa não foi demonstrada para outras fontes, períodos, tarefas ou populações.

## 4.5 Seleção institucional e interpretação racial

O corpus não permite estimar diferenças de criminalidade segundo raça/cor. Ele também não permite avaliação biométrica: faltam pares genuínos e impostores por identidade, limiares operacionais e rótulos demográficos validados. Qualquer extensão sobre seletividade racial exigiria dados administrativos governados, pontos de decisão, raça/cor autodeclarada quando cabível e denominadores apropriados; categorias derivadas de imagens não substituem esses requisitos.

## 4.6 Resultado negativo como contribuição científica

Demonstrar não identificabilidade é um resultado metodológico. A revisão torna explícitos três limites: não há estado histórico suficiente para reprodução causal da análise histórica; não há variável de fonte documentada para análise de influência; e não há datas confiáveis para análise temporal. A contribuição científica está em impedir que métricas internas e artefatos incompletos sejam convertidos em alegações sociais, causais ou biométricas.

# 5 Limitações e agenda de pesquisa

As limitações são substantivas. O autor também é desenvolvedor do repositório auditado, não houve auditoria independente e a proveniência entre estágios é incompleta. Pessoas únicas não podem ser identificadas com certeza; group_id representa apenas duplicidade provável. Não há estado histórico completo, pesos e ambiente históricos, variável de fonte documentada, datas confiáveis, denominadores institucionais, desfechos jurídicos comparáveis, rótulos raciais validados, protocolo biométrico por identidade ou amostragem probabilística. A corrida de 20 sementes por seis valores de k não foi concluída; seu resultado parcial não é usado. O teste de determinismo foi restrito a CPU e dados sintéticos. O scanner de privacidade aprovado verifica as saídas públicas, não substitui avaliação ética ou jurídica.

Uma extensão adequada requer três protocolos distintos: para seleção institucional, painel temporal com entradas, remoções, motivo, jurisdição e denominadores; para seletividade racial, dados administrativos governados e raça/cor autodeclarada quando jurídica e eticamente cabível; para biometria, pares por identidade, condições de captura, limiares operacionais e incerteza por grupo. Esses protocolos não podem ser substituídos por agrupamento de embeddings ou por classificações de aparência.

# 6 Conclusão

A análise dos artefatos disponíveis no estado preservado auditado sustenta uma conclusão delimitada: listas públicas de procurados são registros institucionais selecionados, não medidas populacionais de criminalidade. O cross-fitting por grupos mostra recuperação mensurável de um rótulo sintético, enquanto as sensibilidades de ordem, minibatch e PCA-64 mostram forte dependência da partição da configuração computacional. Esses resultados não autorizam inferir diferenças de criminalidade segundo raça/cor, reconhecer identidades nem avaliar equidade biométrica. O valor científico do estudo é tornar auditáveis os limites entre dado público, seleção institucional, vetor de representação facial e alegação social.

# Declarações

Disponibilidade de código e artefatos. O repositório público está em https://github.com/limag-henrique/are-you-a-criminal-ML; o commit correspondente aos resultados finais é 734f5979b0b659727c71b07c77ee7f0b9431d2b9. O depósito Zenodo de conceito é https://doi.org/10.5281/zenodo.21536403, e o registro versionado 1.0.1 é https://doi.org/10.5281/zenodo.21632858. Esses recursos permitem localizar o código e os artefatos preservados, mas não equivalem a reprodução numérica do estado histórico ausente.

Financiamento. Não foi localizada declaração documental verificável de financiamento nos artefatos auditados; a informação deverá ser confirmada pelo autor antes de submissão.

Conflitos de interesse. O autor desenvolveu o repositório auditado; esse conflito intelectual é declarado. Não foi localizada declaração documental verificável sobre conflitos financeiros adicionais; a informação deverá ser confirmada pelo autor antes de submissão.

Contribuições do autor. H. L. Gusmão: concepção, software, curadoria, análise, visualização e redação.

Ética, proteção de dados e governança. Não foi localizada aprovação ética aplicável nos artefatos auditados, e esta revisão não presume sua existência. O estudo não classifica raça/cor a partir de imagens, não realiza inferência individual e não recomenda uso operacional. Qualquer pesquisa futura com dados pessoais vinculados requer avaliação ética e jurídica própria.

Uso de inteligência artificial na pesquisa e redação. A declaração específica deve ser confirmada pelo autor de acordo com a política do periódico; esta revisão editorial e metodológica não substitui essa declaração.

# Referências

BENDER, E. M.; FRIEDMAN, B. Data statements for natural language processing: toward mitigating system bias and enabling better science. Transactions of the Association for Computational Linguistics, v. 6, p. 587–604, 2018. DOI: 10.1162/tacl_a_00041. Disponível em: https://aclanthology.org/Q18-1041/. Acesso em: 27 jul. 2026.

BUOLAMWINI, J.; GEBRU, T. Gender Shades: intersectional accuracy disparities in commercial gender classification. Proceedings of Machine Learning Research, v. 81, p. 77–91, 2018. Disponível em: https://proceedings.mlr.press/v81/buolamwini18a.html. Acesso em: 27 jul. 2026.

CONSELHO NACIONAL DE JUSTIÇA; PROGRAMA DAS NAÇÕES UNIDAS PARA O DESENVOLVIMENTO. Caderno temático de relações raciais: diretrizes gerais para atuação dos serviços penais. Brasília, DF: CNJ, 2024. Disponível em: https://bibliotecadigital.cnj.jus.br/handle/123456789/939. Acesso em: 27 jul. 2026.

DENG, J. et al. ArcFace: additive angular margin loss for deep face recognition. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. p. 4690–4699. DOI: 10.1109/CVPR.2019.00482. Disponível em: https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html. Acesso em: 27 jul. 2026.

ELWERT, F.; WINSHIP, C. Endogenous selection bias: the problem of conditioning on a collider variable. Annual Review of Sociology, v. 40, p. 31–53, 2014. DOI: 10.1146/annurev-soc-071913-043455. Disponível em: https://doi.org/10.1146/annurev-soc-071913-043455. Acesso em: 27 jul. 2026.

ENSIGN, D. et al. Runaway feedback loops in predictive policing. Proceedings of Machine Learning Research, v. 81, p. 160–171, 2018. Disponível em: https://proceedings.mlr.press/v81/ensign18a.html. Acesso em: 27 jul. 2026.

FAWCETT, T. An introduction to ROC analysis. Pattern Recognition Letters, v. 27, n. 8, p. 861–874, 2006. DOI: 10.1016/j.patrec.2005.10.010. Disponível em: https://doi.org/10.1016/j.patrec.2005.10.010. Acesso em: 27 jul. 2026.

GEBRU, T. et al. Datasheets for datasets. Communications of the ACM, v. 64, n. 12, p. 86–92, 2021. DOI: 10.1145/3458723. Disponível em: https://doi.org/10.1145/3458723. Acesso em: 27 jul. 2026.

GROTHER, P.; NGAN, M.; HANAOKA, K. Face Recognition Vendor Test Part 3: demographic effects. NISTIR 8280. Gaithersburg: NIST, 2019. DOI: 10.6028/NIST.IR.8280. Disponível em: https://doi.org/10.6028/NIST.IR.8280. Acesso em: 27 jul. 2026.

HERNÁN, M. A.; HERNÁNDEZ-DÍAZ, S.; ROBINS, J. M. A structural approach to selection bias. Epidemiology, v. 15, n. 5, p. 615–625, 2004. DOI: 10.1097/01.ede.0000135174.63482.43. Disponível em: https://pubmed.ncbi.nlm.nih.gov/15308962/. Acesso em: 27 jul. 2026.

HUBERT, L.; ARABIE, P. Comparing partitions. Journal of Classification, v. 2, p. 193–218, 1985. DOI: 10.1007/BF01908075. Disponível em: https://doi.org/10.1007/BF01908075. Acesso em: 27 jul. 2026.

INTERPOL. About Red Notices. Lyon: Interpol, 2026. Disponível em: https://www.interpol.int/en/How-we-work/Notices/Red-Notices. Acesso em: 27 jul. 2026.

INTERPOL. View Red Notices. Lyon: Interpol, 2026. Disponível em: https://www.interpol.int/en/How-we-work/Notices/Red-Notices/View-Red-Notices. Acesso em: 27 jul. 2026.

ISO. ISO/IEC 19795-1:2021: information technology — biometric performance testing and reporting — Part 1: principles and framework. Geneva: ISO, 2021. Disponível em: https://www.iso.org/standard/73515.html. Acesso em: 27 jul. 2026.

ISO. ISO/IEC 19795-10:2024: information technology — biometric performance testing and reporting — Part 10: quantifying biometric system performance variation across demographic groups. Geneva: ISO, 2024. Disponível em: https://www.iso.org/standard/81223.html. Acesso em: 27 jul. 2026.

JACOBS, A. Z.; WALLACH, H. Measurement and fairness. In: ACM Conference on Fairness, Accountability, and Transparency, 2021. p. 375–385. DOI: 10.1145/3442188.3445901. Disponível em: https://doi.org/10.1145/3442188.3445901. Acesso em: 27 jul. 2026.

KÄRKKÄINEN, K.; JOO, J. FairFace: face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In: IEEE/CVF Winter Conference on Applications of Computer Vision, 2021. p. 1548–1558. DOI: 10.1109/WACV48630.2021.00159. Disponível em: https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html. Acesso em: 27 jul. 2026.

KAUFMAN, S. et al. Leakage in data mining: formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, v. 6, n. 4, art. 15, 2012. DOI: 10.1145/2382577.2382579. Disponível em: https://doi.org/10.1145/2382577.2382579. Acesso em: 27 jul. 2026.

KUHN, H. W. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, v. 2, n. 1–2, p. 83–97, 1955. DOI: 10.1002/nav.3800020109. Disponível em: https://doi.org/10.1002/nav.3800020109. Acesso em: 27 jul. 2026.

MITCHELL, M. et al. Model cards for model reporting. In: Conference on Fairness, Accountability, and Transparency, 2019. p. 220–229. DOI: 10.1145/3287560.3287596. Disponível em: https://doi.org/10.1145/3287560.3287596. Acesso em: 27 jul. 2026.

PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011. Disponível em: https://www.jmlr.org/papers/v12/pedregosa11a.html. Acesso em: 27 jul. 2026.

RAJI, I. D. et al. Closing the AI accountability gap: defining an end-to-end framework for internal algorithmic auditing. In: Conference on Fairness, Accountability, and Transparency, 2020. p. 33–44. DOI: 10.1145/3351095.3372873. Disponível em: https://doi.org/10.1145/3351095.3372873. Acesso em: 27 jul. 2026.

SANDVE, G. K. et al. Ten simple rules for reproducible computational research. PLOS Computational Biology, v. 9, n. 10, e1003285, 2013. DOI: 10.1371/journal.pcbi.1003285. Disponível em: https://doi.org/10.1371/journal.pcbi.1003285. Acesso em: 27 jul. 2026.

SCHÖLKOPF, B. et al. Estimating the support of a high-dimensional distribution. Neural Computation, v. 13, n. 7, p. 1443–1471, 2001. DOI: 10.1162/089976601750264965. Disponível em: https://doi.org/10.1162/089976601750264965. Acesso em: 27 jul. 2026.

SECRETARIA NACIONAL DE POLÍTICAS PENAIS. Relatório de Informações Penais: 14º ciclo SISDEPEN, 2º semestre de 2023. Brasília, DF: SENAPPEN, 2024. Disponível em: https://www.gov.br/senappen/pt-br/servicos/sisdepen/relatorios/relatorios-de-informacoes-penitenciarias/relotario-2o-semestre-de-2023.pdf. Acesso em: 27 jul. 2026.

SURESH, H.; GUTTAG, J. A framework for understanding sources of harm throughout the machine learning life cycle. In: Equity and Access in Algorithms, Mechanisms, and Optimization, 2021. p. 1–9. DOI: 10.1145/3465416.3483305. Disponível em: https://doi.org/10.1145/3465416.3483305. Acesso em: 27 jul. 2026.

VINH, N. X.; EPPS, J.; BAILEY, J. Information theoretic measures for clusterings comparison: variants, properties, normalization and correction for chance. Journal of Machine Learning Research, v. 11, p. 2837–2854, 2010. Disponível em: https://jmlr.csail.mit.edu/papers/v11/vinh10a.html. Acesso em: 27 jul. 2026.
