#!/usr/bin/env python3
"""Single-feature top-40 / bottom-40 sorts on DEV and HOLDOUT (no model)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_lib as W
from features import ALL_FEATURES
from extra_features import EXTRA_FEATURES

panel = W.load_research_panel(extra=True)
cols = [c for c in ALL_FEATURES + EXTRA_FEATURES if c != "gics_sector" and not c.endswith("_rank")]
cols = list(dict.fromkeys(cols))
rows = []
for c in cols:
    s = panel[c]
    if s.nunique() < 20 or s.groupby(panel["date"]).nunique().median() < 50:
        continue  # broadcast regime features can't sort a cross-section
    for sign, tag in ((1, "hi"), (-1, "lo")):
        r = W.evaluate(panel, sign * s, top_n=40, label=f"{c}:{tag}", with_ic=False)
        rows.append({"feat": f"{c}:{tag}",
                     "dev_cagr": r.loc["dev", "cagr"], "dev_sh": r.loc["dev", "sharpe"],
                     "hold_cagr": r.loc["hold", "cagr"], "hold_sh": r.loc["hold", "sharpe"],
                     "dev_dd": r.loc["dev", "maxdd"]})
df = pd.DataFrame(rows).sort_values("dev_sh", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
