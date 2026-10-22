#!/usr/bin/env python3
"""Re-score every cached variant (after the index re-entry label fix)."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_lib as W

names = sys.argv[1:]
p = W.load_research_panel(extra=False)
wins = {"dev": (W.DEV_START, W.DEV_END), "hold": (W.HOLDOUT_START, None), "2021+": ("2021-01-01", None)}
rows = []
for n in names:
    s = W.load_preds(n, p)
    if s is None:
        continue
    r = W.evaluate(p, s, label=n, windows=wins, with_ic=False)
    rows.append({"variant": n, **{f"{w}_{k}": r.loc[w, k] for w in wins for k in ("cagr", "sharpe")}})
df = pd.DataFrame(rows)
for c in df.columns[1:]:
    df[c] = df[c].map(lambda x: f"{x:+.1%}" if c.endswith("cagr") else f"{x:.2f}")
print(df.to_string(index=False))
