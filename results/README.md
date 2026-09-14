# Results

Raw machine-readable output from the reproduction scripts in
[`../research/`](../research/). Every number cited as **REPRODUCIBLE** in
[`../findings/`](../findings/) is traceable to a file here.

| File | Produced by | Contains |
|---|---|---|
| `ceiling_test_*.json` | `research.ceiling_test` | Over-capacity train/test gap, shuffled-label control, positive controls, 1-NN oracle, majority baseline, overlapping vs non-overlapping autocorrelation. |
| `model_zoo_*.json` | `research.model_zoo` | Nine model families on identical purged walk-forward splits, each with a shuffled-label control, exact binomial CIs and Pesaran-Timmermann p-values. |
| `overlap_simulation_*.json` | `research.overlap_simulation` | Measured prediction persistence, the session hit-rate distribution of a zero-skill forecaster, CI understatement factors, multiplicity. |
| `strategy_sweep_*.json` | `research.strategy_sweep` | 86 strategies with one-sided profitability tests, Bonferroni and Benjamini-Hochberg corrections, Deflated Sharpe, block-bootstrap empirical null, PBO. |
| `cost_geometry.json` | `research.cost_geometry` | Move distributions, cost-to-move ratios, break-even accuracy per horizon and per cost regime. |

## Regenerating

From the repository root, with `data_cache/` populated (see
[`../findings/09-data-assets.md`](../findings/09-data-assets.md)):

```bash
python3 -m research.ceiling_test        --horizon 30m
python3 -m research.model_zoo           --horizon 30m
python3 -m research.overlap_simulation  --horizon 30m
python3 -m research.strategy_sweep      --horizon 30m
python3 -m research.cost_geometry
```

All scripts use a fixed seed (`research/common.py: SEED = 0`) and pin native
thread pools to one thread, so runs are reproducible on the same library
versions. `model_zoo` takes roughly 30 minutes per horizon on a laptop; the
others complete in seconds to a few minutes.
