# research/ — walk-forward experiment harness

Tooling behind the ranker v2 decision (see the main README, "Ranker v2").
Not part of the daily pipeline.

| File | What it does |
| --- | --- |
| `wf_lib.py` | Panel loading, annual expanding-window walk-forward with a 21-day purge, fast evaluator (21 rebalance offsets, top-N after the quality filter, 5 bps/side), IC stats, prediction cache |
| `extra_features.py` | First-pass builder for the 17 candidate features → `data/processed/extra_features.parquet`. Superseded by Bucket 9 in `scripts/features.py`; kept so the logged runs are reproducible |
| `run_wf.py` | Run one variant (`--model xgb/lgb`, `--label`, `--features`, `--rank-all`, `--drop`, `--seeds`, `--stride`) and cache its OOS predictions to `research/cache/` |
| `eval_wf.py` | Compare cached variants on DEV / HOLDOUT, sweep `--top-n` / `--hold`, `--yearly` excess table, `blend:a+b` rank averages |
| `confirm.py` | Re-score cached predictions with `scripts/backtest.py`'s daily engine |
| `factor_sorts.py` | Single-feature top-40 / bottom-40 sorts (no model) |
| `make_refs.py` | Reference series: committed static model, random scores, EW-universe and SPY stats |
| `static_seeds.py` | Refit the exact v1 recipe with other seeds to see where the committed model sits in its own seed distribution |
| `queue*.sh` | The experiment batches, in the order they were run; outputs in `logs/` |

Windows were fixed before any experiment ran and never moved:

- **DEV 2012-01-01 → 2022-12-31** — every choice (model, features, params) was made here.
- **HOLDOUT 2023-01-01 → latest** — looked at only to confirm.

Caveats that apply to every number here: yfinance has no prices for ~290
names that left the index (survivorship bias, flatters high-vol tilts), and
the static `xgb_v1.json` was selected after many sweeps scored on 2021+, so
its 2021+ numbers are optimistic.

```bash
uv run python research/extra_features.py               # only needed for the logged runs
uv run python research/run_wf.py --name lr_default --stride 3 --rank-all --features extra \
    --model lgb --label y_relevance --params '{"objective":"lambdarank","lambdarank_truncation_level":50}'
uv run python research/eval_wf.py static lr_default --top-n 20,40 --hold 21,63 --yearly
uv run python research/confirm.py static lr_default --start 2023-01-01
```

**Reading `logs/`:** every `*.log` / `queue*.out` was produced *before* the
index re-entry label fix (main README, "Ranker v2"), so their DEV numbers
from the fast evaluator are inflated by up to ~5 CAGR pts.
`logs/rescore_after_labelfix.txt` and `logs/rescore_final.txt` re-score the
same cached predictions with corrected forward returns — use those. The
daily-engine sweeps (`sweep_*.out`, `final_sweep_*.out`, `decomp_*.out`)
never had the bug.
