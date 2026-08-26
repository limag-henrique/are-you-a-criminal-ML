# Monografia ABNT - projeto pronto para Overleaf

Este projeto foi convertido do Word e ajustado para uso direto no **Overleaf**, preservando a estrutura da monografia, imagens, tabelas/quadros, legendas, listas, sumário e a diagramação ABNT do documento de origem.

## Como usar no Overleaf

1. No Overleaf, escolha **New Project > Upload Project**.
2. Envie o arquivo ZIP deste projeto inteiro, sem descompactar.
3. O arquivo principal é `main.tex` e fica na raiz do projeto.
4. Clique em **Recompile**.

O projeto usa **pdfLaTeX**, que é o compilador padrão do Overleaf. Não é necessário instalar Arial, Arimo ou qualquer outra fonte no computador: o pacote `arimo` usado aqui faz parte do TeX Live do Overleaf.

## Referências bibliográficas

As referências foram movidas para `referencias.bib` e são formatadas pelo `abntex2cite` no sistema autor-data (`alf`). A lista de referências é gerada automaticamente com BibTeX.

A linha `\nocite{*}` em `conteudo.tex` foi mantida para preservar todas as 40 referências existentes na versão Word, inclusive as referências de apoio que não aparecem em uma chamada bibliográfica explícita no texto.

Para novas citações, use por exemplo:

```tex
\cite{bowyer2020}
```

para uma citação entre parênteses, ou:

```tex
\citeonline{bowyer2020}
```

para integrar o autor à frase.

## Arquivos principais

- `main.tex` - preâmbulo, configuração ABNT e elementos pré-textuais;
- `conteudo.tex` - corpo da monografia e apêndice;
- `referencias.bib` - base bibliográfica BibTeX;
- `figuras/` - quatro imagens incorporadas no documento original;
- `compilar.bat` - opção de compilação local no Windows.

## Compilação local

Se desejar compilar fora do Overleaf:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

No Overleaf esse ciclo é executado automaticamente pelo `latexmk` ao clicar em **Recompile**.
