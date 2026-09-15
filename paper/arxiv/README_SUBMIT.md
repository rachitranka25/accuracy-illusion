# arXiv submission package

Everything here is ready to upload. I cannot submit on your behalf: arXiv
requires your own account, and a first submission to q-fin needs an endorsement
from an existing author in that archive. The licence must also be accepted by
you as the author.

## What to upload

Upload `arxiv.tar.gz` (or the loose files: `accuracy_illusion.tex` plus the
`figures/` directory). arXiv compiles LaTeX itself — do not upload the PDF
unless you choose the "PDF only" route, which forfeits the HTML rendering and
is not recommended.

The document uses only packages in arXiv's standard TeX Live: IEEEtran,
fontenc, lmodern, cite, amsmath, amssymb, amsfonts, graphicx, booktabs, url,
xcolor, hyperref, balance, stfloats.

## Form fields

**Primary category:** `q-fin.ST` (Statistical Finance)

**Cross-lists:** `q-fin.TR` (Trading and Market Microstructure), `stat.AP`
(Applications). Add `cs.LG` only if you want the ML audience; it draws
reviewers who will not know the econometrics.

**Title**

    The Accuracy Illusion: How Overlapping Samples Manufacture Confidence,
    Not Skill, in Intraday Index Forecasting

**Authors**

    Rachit Ranka, Aaditya Saini

**Affiliation**

    Department of Computer Science Engineering, Chandigarh University,
    Mohali, India

**Abstract:** paste from the compiled PDF, or from the `abstract` environment
in the .tex with LaTeX markup stripped.

**Comments field (optional but worth filling):**

    12 pages, 5 figures, 11 tables. Code, raw results and the data corpus:
    https://github.com/rachitranka25/accuracy-illusion

**MSC / ACM class:** leave blank.

**Licence:** CC BY 4.0 is the usual choice for a preprint you want cited and
reused. CC BY-NC-SA if you would rather restrict commercial reuse.

## Endorsement

If arXiv asks for an endorsement, that means neither author has published in
q-fin before. Any researcher with prior q-fin submissions can endorse. A
supervisor at Chandigarh University who has posted to q-fin, or a co-author on
a future version, is the normal route. The endorsement code arXiv shows you is
what they need.

## Before you click submit

- The repository must be public and the link live, since the paper cites it.
- Once posted, the paper is permanent. Versions can be added; nothing is ever
  removed.
- Posting to arXiv does not preclude submitting to ACM ICAIF, IEEE CIFEr or a
  journal. Check the venue's policy if it is double-blind: the author names,
  the affiliation and the GitHub link would all need removing from the
  submitted copy, though the arXiv version can stay up.
