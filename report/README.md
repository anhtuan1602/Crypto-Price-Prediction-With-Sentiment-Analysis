# Technical report (LaTeX)

`main.tex` is the Group 08 technical report, `refs.bib` its bibliography and
`figures/` its images. `main.pdf` is a compiled copy of the current draft.

## Build

```bash
latexmk -pdf main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf, upload the whole `report/` folder and compile `main.tex` with pdfLaTeX.

## Before submitting

The results come from the first training run (2026-11-28, 282 samples).

* Numbers to refresh are highlighted in yellow: `\prelim{...}` in the source.
* Red boxes, `\UPDATE{...}`, say what to change after the final retrain and
  where to find the value (`artifacts/report.md`, `report.json` or MLflow).
* Blue boxes, `\TOFILL{...}`, are for the team: names, student IDs,
  university and course, and member contributions.

When everything is final, set `\showupdatesfalse` near the top of `main.tex`.
The highlights and boxes then disappear from the PDF.
