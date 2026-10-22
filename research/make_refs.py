#!/usr/bin/env python3
"""Reference score series: the committed static model, a random score, and
an equal-weight-universe benchmark printout."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_lib as W
import strategy
from features import ALL_FEATURES
from data import load_market

panel = W.load_research_panel(extra=False)
m = strategy.load_model(os.path.join(W._ROOT, "models", "xgb_v1.json"))
W.save_preds("static", pd.Series(m.predict(panel[ALL_FEATURES]), index=panel.index), panel)
rng = np.random.default_rng(0)
W.save_preds("random", pd.Series(rng.normal(size=len(panel)), index=panel.index), panel)

spy = load_market()["SPY"]["Close"]; spy.index = pd.to_datetime(spy.index)
for w, (s, e) in {"dev": (W.DEV_START, W.DEV_END), "hold": (W.HOLDOUT_START, None)}.items():
    uni = W.stats_from_periods(W.universe_periods(panel, s, e))
    last = panel.loc[panel["fwd_raw"].notna(), "date"].max()
    sp = spy[(spy.index >= s) & (spy.index <= (e or last))]
    r = sp.pct_change().dropna(); yrs = len(r) / 252
    cagr = (sp.iloc[-1] / sp.iloc[0]) ** (1 / yrs) - 1; vol = r.std() * np.sqrt(252)
    dd = (sp / sp.cummax() - 1).min()
    print(f"{w}: EW-universe CAGR {uni['cagr']:+.1%} Sharpe {uni['sharpe']:.2f} MaxDD {uni['maxdd']:+.1%} | "
          f"SPY CAGR {cagr:+.1%} Sharpe {cagr/vol:.2f} MaxDD {dd:+.1%}")
