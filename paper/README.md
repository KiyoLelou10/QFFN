# Manuscript source

This directory contains the manuscript and the figure and style files needed by its Springer Nature `sn-jnl` source.

With a sufficiently complete TeX Live installation, compile from this directory with:

```bash
pdflatex qfnn_paper.tex
pdflatex qfnn_paper.tex
pdflatex qfnn_paper.tex
```

The bibliography is self-contained in the `.tex` file, so BibTeX is not required. Three LaTeX passes resolve the table of contents, theorem references, citations, and page references.
