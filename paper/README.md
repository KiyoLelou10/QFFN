# Manuscript source

This directory contains Version 7 of the manuscript and the figure and style files needed by its Springer Nature `sn-jnl` source.

With a sufficiently complete TeX Live installation, compile from this directory with:

```bash
pdflatex qfnn_chebyshev_revised_v7.tex
pdflatex qfnn_chebyshev_revised_v7.tex
pdflatex qfnn_chebyshev_revised_v7.tex
```

The bibliography is self-contained in the `.tex` file, so BibTeX is not required. Three LaTeX passes resolve the table of contents, theorem references, citations, and page references.
