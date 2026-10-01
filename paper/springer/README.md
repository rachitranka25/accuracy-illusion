# Springer LNCS version

`accuracy_illusion_springer.tex` — Springer Lecture Notes in Computer Science
(`llncs`), single column, 10 pages including references.

Built with: `tectonic accuracy_illusion_springer.tex`

## Relationship to the other versions

Same results, same numbers, same harness as `paper/short6` (IEEE, 6 pages) and
`paper/accuracy_illusion.tex` (IEEE, 14 pages). Every figure in all three comes
from the committed JSON in `results/`.

Relative to the IEEE short version this one adds a prose comparison against
prior work and the per-cost-regime break-even figures. To fit ten pages it
carries two tables rather than five and no figure; the numbers from the
remaining tables are stated in the text, and the reference list is 27 rather
than 33.

## Word version

`accuracy_illusion_springer.docx` is generated from the same source:

```
pandoc _docx_src.tex -f latex -t docx --number-sections -o accuracy_illusion_springer.docx
```

where `_docx_src.tex` is the LNCS source with class-specific markup replaced and
`\cite` keys resolved to the bracketed numbers of the bibliography. The PDF is
the authoritative artefact; the Word file is for venues that require it.
