# References

Verified against publisher records. Anything not confirmed to a publisher or
DOI is listed under *Unverified* and must not be cited until checked.

## Methods this work uses directly

**Bailey, D. H., & López de Prado, M. (2014).** The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
*Journal of Portfolio Management*, 40(5), 94–107.
SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
→ Used in `research/strategy_sweep.py`. Reported, and then **flagged as
inapplicable** to our strategy library because the trial Sharpe dispersion is
driven by turnover rather than noise; replaced with a block-bootstrap null.

**Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017).**
The Probability of Backtest Overfitting.
*Journal of Computational Finance*, 20(4), 39–70.
SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
→ CSCV implemented in `research/strategy_sweep.py`. Reported as **degenerate**
in our setting, since nothing is significantly profitable in-sample.

**Pesaran, M. H., & Timmermann, A. (1992).** A Simple Nonparametric Test of
Predictive Performance. *Journal of Business & Economic Statistics*, 10(4).
DOI: <https://doi.org/10.1080/07350015.1992.10509922>
→ Implemented in `research/common.py`. The correct null for directional
accuracy: it compares the hit-rate against what independence between forecast
and outcome would produce given their own marginals, which a binomial test
against 0.5 does not.

**Diebold, F. X., & Mariano, R. S. (1995).** Comparing Predictive Accuracy.
*Journal of Business & Economic Statistics*, 13(3), 253–263.
→ Implemented in `research/common.py` with a Newey-West variance, since
h-step-ahead forecast errors are serially correlated by construction.

**Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).** Temporal Fusion
Transformers for Interpretable Multi-horizon Time Series Forecasting.
*International Journal of Forecasting*, 37(4), 1748–1764.
DOI: <https://doi.org/10.1016/j.ijforecast.2021.03.012>
→ `research/model_zoo.py` implements a **compact variant**: variable selection
network, GRN gating, and interpretable multi-head attention over an LSTM
encoder. Static covariates and the quantile head are omitted because this task
has neither. Must be described as a compact variant, never as the published
model.

**Kolm, P. N., Turiel, J., & Westray, N. (2023).** Deep Order Flow Imbalance:
Extracting Alpha at Multiple Horizons from the Limit Order Book.
*Mathematical Finance*, 33(4), 1044–1081.
DOI: <https://doi.org/10.1111/mafi.12413>
→ The basis for the claim that documented short-horizon alpha lives in
order-flow rather than in bar data, and the motivation for
`oracle/data/order_flow.py`. Note what it actually requires: Nasdaq
message-level order book data across 115 stocks. Cite it as evidence of where
the edge is, and of how far that is from retail reach — not as a result we
replicate.

**López de Prado, M. (2018).** *Advances in Financial Machine Learning.*
Wiley.
→ Source of triple-barrier labelling, meta-labelling, and purging with embargo.
The purged walk-forward in `research/common.py` follows its logic. Our two
meta-labelling attempts both failed out-of-sample; report that as our result on
our data, not as a criticism of the method.

## To be added before submission

The following claims currently appear in `findings/` sourced to a literature
review rather than to specific papers. Each needs a verified citation or must
be softened to a statement about our own data:

- "Peer-reviewed work on liquid index intraday prediction reports 50–55% as
  normal and 55–58% as a realistic ceiling."
- "Review papers treat claimed accuracies of 80–90% as a diagnostic of
  lookahead bias."
- The widely circulated XGBoost result reporting ~71% directional accuracy —
  needs the actual paper, or the sentence must be removed.
- Standard references for volatility clustering (Engle/Bollerslev) and the
  intraday U-shape (Admati & Pfleiderer, or Harris) to support
  [finding 04](../findings/04-volatility-predictability.md).
- Standard references for the variance risk premium (Carr & Wu; Bakshi &
  Kapadia) to support [finding 05](../findings/05-variance-risk-premium.md).
- Overlapping-observations inference in finance — the econometrics of
  overlapping returns (Hansen & Hodrick; Richardson & Smith; Britten-Jones,
  Neuberger & Nolte) is the closest existing literature to this paper's central
  mechanism and **must** be engaged with. The contribution here is not that
  overlap inflates variance — that is known — but the measurement of what it
  does to *reported directional hit-rates* on a live system, and the
  multiplicity that follows.

## Positioning note

The honest framing of the contribution, to be defended rather than overstated:

1. The overlapping-sample problem is **known** in econometrics. What is not
   documented is its size and shape in the specific practice of every-minute
   multi-horizon directional dashboards, where it combines with prediction
   persistence and a multiplicity of index × horizon panels.
2. The negative result on direction is **consistent** with the literature, not
   novel on its own. Its value is the completeness of the control set
   (shuffled labels, positive controls, nine families, block-bootstrap null)
   and the fact that everything is reproducible.
3. The genuinely novel quantitative contribution is the **required-versus-achieved
   accuracy framing**: break-even accuracy computed from measured move
   distributions, showing that below a horizon threshold no accuracy suffices.
4. The dataset construction — a decade of free EOD option chains, two years of
   participant-wise OI, and the finding that the NSE archive is not geo-blocked —
   is a reusable artefact independent of the analysis.
