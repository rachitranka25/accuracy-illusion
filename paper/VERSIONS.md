# Three versions of the paper

| File | Format | Pages | Intended for |
|---|---|---|---|
| `accuracy_illusion.pdf` | IEEE conference | 14 | Journals (JFDS, Quantitative Finance, Digital Finance); the complete record |
| `icaif/accuracy_illusion_icaif.pdf` | **ACM `sigconf`** | **6** | **ACM ICAIF** — limit is 8 pages *including references* |
| `short6/accuracy_illusion_6pg.pdf` | IEEE conference | 6 | IEEE venues with a 6-page limit |

## ICAIF specifics

ICAIF '26 requires the **ACM `acmart` template in `sigconf`** form, not IEEE,
and caps submissions at **eight pages including references**. Over-length papers
are rejected without review, and supplementary material is not accepted. Ours
sits at six, so there is room if a reviewer asks for more.

The template currently uses the `nonacm` option, which suppresses the ACM
reference block. For a real submission remove `nonacm` and add the conference
data ACM supplies on acceptance. **If the venue reviews double-blind, add the
`anonymous` option and strip the author block, the affiliation and the GitHub
URL** — the URL appears in the introduction and would deanonymise the
submission.

## What the short versions drop

Both six-page versions keep the full argument and every headline number. They
cut:

- the per-model breakdown table (the 48-test aggregate carries it),
- the live-session estimator comparison table (kept as prose),
- the option-chain feature table (kept as prose),
- the model-zoo forest plot and, in the IEEE short version, the zero-skill
  distribution figure,
- most of the discussion around each result, not the results.

Nothing in the short versions contradicts the long one. Every number in all
three traces to `results/*.json`.

## Rebuilding

```bash
cd paper            && tectonic accuracy_illusion.tex
cd paper/icaif      && tectonic accuracy_illusion_icaif.tex
cd paper/short6     && tectonic accuracy_illusion_6pg.tex
```

Figures are generated from committed results by `python3 -m research.figures`
and copied into each version's `figures/` directory.
