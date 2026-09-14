# 01 — The directional ceiling

**Claim tested.** That a sufficiently good model can predict the direction of
the next 5–60 minutes on BANKNIFTY / NIFTY 50 from freely available 5-minute
OHLC data.

**Verdict.** It cannot. The limit is not the model. The information is not in
the data.

---

## The ceiling test

The usual way to argue about model quality is to try more models. That argument
never ends, because there is always one more architecture. So the question was
asked in a form that does not depend on the model at all.

A deliberately over-capacity gradient-boosted model was fitted to the 5-minute
feature set with the direction label, and four numbers were read off:

| Measurement | Result | What it means |
|---|---|---|
| Training accuracy | **96%** | The model has more than enough capacity to memorise. |
| Test accuracy | **49%** | None of what it memorised generalises. |
| Test accuracy on **shuffled** labels | **50%** | Identical to the real labels. |
| Return autocorrelation | **≈ 0.00** | The target has no serial structure to exploit. |
| 1-nearest-neighbour "oracle" | **49%** | Even the closest historical analogue does not rhyme. |

The shuffled-label row is the decisive one. If the features carried signal,
destroying the link between features and labels would degrade test accuracy.
It did not change it. The pipeline performs the same on real data and on noise,
which means it was reading noise the whole time.

> Status: **RECORDED** (research session, 1 Aug 2026). The protocol is four
> lines of scikit-learn and reproduces trivially on any of the cached `_5m_long.csv`
> files.

## Every model family, same answer

Seven families were trained on the identical 30-minute direction task, with the
same walk-forward split:

| Model | Out-of-sample direction accuracy |
|---|---|
| Gradient Boosting | ~50% |
| LightGBM | 49–53% |
| XGBoost | ~50% |
| LSTM | 49% |
| Transformer | 51% |
| Temporal Fusion Transformer | 46% |
| k-Nearest Neighbours | 50% |

XGBoost is worth a note. It was tried because a widely circulated paper reports
~71% directional accuracy with it. Reproduced on NSE index data it returns ~50%.
The difference is not the algorithm — it is that the published result comes from
a less efficient market and a selectively sampled high-volatility subset.

> Status: **RECORDED** (1 Aug 2026).

## Why the features cannot help

The live engine's feature set — RSI, EMA, MACD histogram, Bollinger width and
position, ATR%, Parkinson volatility, VWAP distance, time-of-day, day-of-week,
regime volatility, Hurst exponent — is thirteen columns computed from **one**
underlying series. They are transforms of price. Reshaping price cannot reveal
something price did not already contain.

This is the reasoning that motivated the move to open interest and participant
positioning in [finding 06](06-open-interest.md): not a better model, a
genuinely different information source.

## The strategy sweep

Separately from the model work, 52 rule-based strategies were vectorised and
scored over 17 months of 5-minute bars — momentum, mean reversion, moving-average
crosses, MACD, RSI, stochastic, Bollinger, VWAP, breakout, gap, streak,
volatility-regime, time-of-day and day-of-week variants — each with a
first-half / second-half stability gate.

**Nothing cleared ~53–54% out-of-sample with stability.** Two candidates looked
better and are dissected in [finding 02](02-measurement-illusions.md).

> Status: **RECORDED** (29 Jul 2026).

## What the literature says

This matches the published consensus rather than contradicting it. Peer-reviewed
work on liquid index intraday prediction reports 50–55% as normal and 55–58% as
a realistic ceiling; results above 60% are rare and heavily caveated. In review
papers, claimed accuracies of 80–90% are treated as a **diagnostic of lookahead
bias**, not as an achievement.

The genuine short-horizon edge documented in the literature lives in order-flow
and limit-order-book microstructure — tick-level Level-2 data with colocated
execution. That is a data and latency gap, not an intelligence gap, and it is
not reachable from a retail account. Our own live order-flow collector
([finding 09](09-data-assets.md)) exists precisely because that is the only
direction left that is not already closed.

## Consequence

Everything downstream in this project follows from this finding:

1. Direction is not the product. It is the control condition.
2. The question changes from *which way* to *how far* → [finding 04](04-volatility-predictability.md).
3. Anything that reports high directional accuracy is assumed to be a
   measurement error until proven otherwise → [finding 02](02-measurement-illusions.md).
