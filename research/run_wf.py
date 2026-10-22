#!/usr/bin/env python3
"""Run one walk-forward variant and cache its OOS predictions.

    uv run python research/run_wf.py --name wf_base
    uv run python research/run_wf.py --name wf_extra --features extra
    uv run python research/run_wf.py --name lgb_rank --model lgb --params '{"objective":"lambdarank"}' --label y_relevance

Predictions land in research/cache/pred_<name>.parquet; compare with eval_wf.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import wf_lib as W  # noqa: E402
from extra_features import EXTRA_FEATURES  # noqa: E402
from features import ALL_FEATURES  # noqa: E402


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("date")["fwd_clip"]
    # Per-date percentile rank, centred: robust to fat tails, same scale every date.
    panel["y_rank"] = g.rank(pct=True) - 0.5
    # Integer relevance grades (0..9 deciles) for LambdaRank-style objectives.
    panel["y_relevance"] = np.floor(g.rank(pct=True, method="first") * 10).clip(0, 9)
    # Binary: in the top decile of forward return.
    panel["y_top_decile"] = (g.rank(pct=True) >= 0.9).astype(float)
    panel.loc[panel["fwd_clip"].isna(), ["y_rank", "y_relevance", "y_top_decile"]] = np.nan
    # Sector-neutral: demean within (date, sector) instead of date.
    panel["y_sec"] = panel["fwd_clip"] - panel.groupby(["date", "gics_sector"], observed=True)["fwd_clip"].transform("mean")
    # 63-trading-day horizon (the live book rebalances quarterly).
    f63 = panel["fwd_raw_63"].clip(-0.75, 0.75)
    panel["y63"] = f63 - f63.groupby(panel["date"]).transform("mean")
    panel["y63_relevance"] = np.floor(f63.groupby(panel["date"]).rank(pct=True, method="first") * 10).clip(0, 9)
    panel.loc[f63.isna(), "y63_relevance"] = np.nan
    return panel


BROADCAST = {"spy_ret_21d", "spy_rsi_14", "vix_level", "term_spread_10y3m", "ig_credit_spread", "inflation_5y5y"}


def rank_all(panel: pd.DataFrame, cols: list[str]) -> list[str]:
    """Replace every per-ticker numeric feature by its per-date percentile
    (drops duplicate raw/rank pairs); broadcast regime + sector kept as-is."""
    out = []
    g = panel.groupby("date", sort=False)
    for c in cols:
        if c in BROADCAST or c == "gics_sector" or c.endswith("_rank") or c.endswith("_secrank"):
            if c.endswith("_rank") and c[:-5] in cols:
                continue  # raw version is ranked below
            out.append(c)
            continue
        rc = f"{c}__r"
        panel[rc] = g[c].rank(pct=True)
        out.append(rc)
    return list(dict.fromkeys(out))


FEATURE_SETS = {
    "base": list(ALL_FEATURES),
}


def feature_set(name: str, panel: pd.DataFrame) -> list[str]:
    if name == "base":
        return list(ALL_FEATURES)
    if name == "extra":
        return list(ALL_FEATURES) + [c for c in EXTRA_FEATURES if c in panel.columns]
    if name.startswith("base+"):
        add = name[5:].split(",")
        return list(ALL_FEATURES) + add
    if name.startswith("cols:"):
        return name[5:].split(",")
    raise SystemExit(f"unknown feature set {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="base")
    ap.add_argument("--drop", default="", help="comma-separated features to remove")
    ap.add_argument("--model", default="xgb", choices=["xgb", "lgb"])
    ap.add_argument("--params", default="{}", help="JSON overrides")
    ap.add_argument("--rounds", type=int, default=300, help="lgb boosting rounds")
    ap.add_argument("--label", default="y_demeaned")
    ap.add_argument("--seeds", default="15")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rolling", type=int, default=None)
    ap.add_argument("--first-year", type=int, default=W.FIRST_FOLD_YEAR)
    ap.add_argument("--rank-all", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    panel = add_labels(W.load_research_panel(extra=True))
    cols = feature_set(args.features, panel)
    if args.drop:
        cols = [c for c in cols if c not in set(args.drop.split(","))]
    if args.rank_all:
        cols = rank_all(panel, cols)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    overrides = json.loads(args.params)
    if args.model == "xgb":
        objective = overrides.pop("objective", "reg:squarederror")
        params = dict(W.BASE_PARAMS)
        params.update(overrides)
        fp = W.xgb_fit_predict(params, seeds=seeds, objective=objective)
        params["objective"] = objective
    else:
        params = {"objective": "regression", "learning_rate": 0.02, "num_leaves": 31,
                  "min_data_in_leaf": 2000, "feature_fraction": 0.6, "bagging_fraction": 0.7,
                  "bagging_freq": 1, "lambda_l2": 1.0}
        params.update(overrides)
        fp = W.lgb_fit_predict(params, seeds=seeds, num_boost_round=args.rounds)
    print(f"[{args.name}] {len(cols)} features, label={args.label}, model={args.model} "
          f"params={params} seeds={seeds} stride={args.stride}", flush=True)
    preds = W.walk_forward(panel, cols, fp, label_col=args.label, date_stride=args.stride,
                           rolling_years=args.rolling, first_year=args.first_year,
                           purge=63 if args.label.startswith("y63") else W.HORIZON)
    W.save_preds(args.name, preds, panel)
    print(W.fmt(W.evaluate(panel, preds, label=args.name)))
    print(f"[{args.name}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
