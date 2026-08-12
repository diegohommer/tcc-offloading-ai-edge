# TCC, LaTeX (rascunho parcial)

Baseado no template infufrgs (https://github.com/schnorr/infufrgs), opcoes
[cic,dipl] (Ciencia da Computacao, Trabalho de Conclusao).

## Conteudo deste pacote

- `tcc.tex` — documento principal. Capitulos 1 (Introducao) e 2 (Revisao
  Bibliografica) escritos. Metodologia, resultados e conclusao ainda faltam.
- `tcc.bib` — referencias usadas nos capitulos escritos, com metadados
  conferidos (titulo, autores, veiculo, DOI/arXiv) contra as paginas
  originais.
- `infufrgs.cls` — classe LaTeX do INF/UFRGS (Lucas Schnorr).
- `abntex2cite.sty`, `abntex2-alf.bst`, `abntex2-options.bib` — pacote de
  citacao ABNT (projeto abntex2), incluidos aqui porque o ambiente onde
  este pacote foi montado nao tinha o abntex2 completo disponivel via
  gerenciador de pacotes do sistema. Se sua instalacao TeX Live/MiKTeX ja
  tiver abntex2 instalado (a maioria tem, e o padrao do Overleaf tambem
  tem), pode ignorar esses tres arquivos e usar os do seu sistema.

## Como compilar

```sh
pdflatex tcc
bibtex   tcc
pdflatex tcc
pdflatex tcc
```

Testado nesta sessao com TeX Live 2023 (pdfTeX), compila sem erros e sem
citacoes ou referencias indefinidas. 16 paginas no estado atual (capa,
resumo, sumario, listas, introducao, revisao bibliografica, referencias).

## Pendencias antes de avancar

- Orientador (Prof. Nazar) ainda nao confirmado oficialmente; nome
  completo e titulacao a atualizar em `\advisor` no preambulo de `tcc.tex`.
- Metodologia, resultados e conclusao dependem do experimento descrito
  nas secoes 12 e 13 do documento mestre do projeto (ainda nao executado).
