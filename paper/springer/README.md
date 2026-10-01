# Springer LNCS version

`accuracy_illusion_springer.tex` — Springer Lecture Notes in Computer Science
(`llncs`), single column, 12 pages including references.

Built with: `tectonic accuracy_illusion_springer.tex`

## Relationship to the other versions

Same results, same numbers, same harness as `paper/short6` (IEEE, 6 pages) and
`paper/accuracy_illusion.tex` (IEEE, 14 pages). Every figure in all three comes
from the committed JSON in `results/`.

Relative to the IEEE short version this one adds the methodological comparison
with prior work (Table 1) and the per-cost-regime break-even discussion, and
drops the autocorrelation figure, whose two numbers are stated in the text.

## Word version

`accuracy_illusion_springer.docx` is generated from the same source:

```
pandoc _docx_src.tex -f latex -t docx --number-sections -o accuracy_illusion_springer.docx
```

where `_docx_src.tex` is the LNCS source with class-specific markup replaced and
`\cite` keys resolved to the bracketed numbers of the bibliography. The PDF is
the authoritative artefact; the Word file is for venues that require it.
