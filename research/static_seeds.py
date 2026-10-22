#!/usr/bin/env python3
"""How lucky is the committed static model? Refit the exact v1 recipe
(train 2007-2017, early-stop on val 2018-2020 top-40 mean return, the
saved params from reports/train_metrics.json) with different seeds and
cache each as pred_static_s<seed>. Compare their 2021+ results with the
committed xgb_v1.json (seed 15, picked after many sweeps).
"""
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import wf_lib as W  # noqa: E402
import train as T  # noqa: E402
from dataset import split  # noqa: E402
from features import ALL_FEATURES  # noqa: E402

seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4,5").split(",")]
params = json.load(open(os.path.join(W._ROOT, "reports", "train_metrics.json")))["params"]
panel = W.load_research_panel(extra=False)
tr, va, _ = split(panel.rename(columns={"y_demeaned": "forward_21d_return_demeaned"}))
tr = tr.dropna(subset=["forward_21d_return_demeaned"]); va = va.dropna(subset=["forward_21d_return_demeaned"])
for s in seeds:
    m = T._make_model(params, va["date"], seed=s)
    m.fit(tr[ALL_FEATURES], tr["forward_21d_return_demeaned"],
          eval_set=[(va[ALL_FEATURES], va["forward_21d_return_demeaned"])], verbose=False)
    W.save_preds(f"static_s{s}", pd.Series(m.predict(panel[ALL_FEATURES]), index=panel.index), panel)
    print(f"seed {s}: best_iteration={m.best_iteration}", flush=True)
