#!/usr/bin/env python3
"""Re-score cached predictions with backtest.py's daily engine.

    uv run python research/confirm.py static wf_best --start 2023-01-01
    uv run python research/confirm.py wf_best --start 2021-01-01 --leverage 1.35 --hold 63 --sector-cap 0.4

Same code path as scripts/backtest.py (run_shifted_starts + compute_stats),
just with the model predictions swapped for a cached walk-forward series.
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import wf_lib as W  # noqa: E402
import backtest as B  # noqa: E402
import strategy  # noqa: E402
from data import load_market  # noqa: E402
from dataset import load_panel  # noqa: E402
from eval_wf import get_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--hold", type=int, default=21)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--vol-target", type=float, default=strategy.DEFAULT_VOL_TARGET)
    ap.add_argument("--sector-cap", type=float, default=None)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--no-quality", action="store_true")
    args = ap.parse_args()

    panel = load_panel()
    rp = W.load_research_panel(extra=False)
    md = load_market()
    market = strategy.prepare_market(md["SPY"], md["VIX"])
    rows = []
    for name in args.names:
        score = get_score(name, rp)
        sc = pd.DataFrame({"date": rp["date"], "ticker": rp["ticker"], "predicted_return": score})
        tp = panel.merge(sc, on=["date", "ticker"], how="inner")
        tp = tp[(tp["date"] >= args.start) & tp["predicted_return"].notna()]
        if args.end:
            tp = tp[tp["date"] <= args.end]
        common = dict(top_n=args.top_n, hold_days=args.hold, cost_per_side=strategy.COST_PER_SIDE,
                      vol_target=args.vol_target, weight_mode="equal",
                      quality_filter_on=not args.no_quality, leverage=args.leverage,
                      margin_rate=strategy.DEFAULT_MARGIN_RATE, lag=args.lag,
                      sector_cap=args.sector_cap)
        for variant, vt in (("raw", False), ("voltgt", True)):
            curves, _, _, _ = B.run_shifted_starts(tp, market, vol_target_on=vt, **common)
            s = B.compute_stats(curves.mean(axis=1))
            per = [B.compute_stats(curves[c])["cagr"] for c in curves.columns]
            rows.append({"variant": f"{name} {variant}", "cagr": s["cagr"], "sharpe": s["sharpe"],
                         "maxdd": s["max_dd"], "vol": s["ann_vol"], "nav": s["final_nav"],
                         "off_min": min(per), "off_max": max(per)})
        end = tp["date"].max()
    spy = B.compute_stats(B.spy_buy_and_hold(market, pd.Timestamp(tp["date"].min())).loc[:end])
    rows.append({"variant": "SPY", "cagr": spy["cagr"], "sharpe": spy["sharpe"], "maxdd": spy["max_dd"],
                 "vol": spy["ann_vol"], "nav": spy["final_nav"], "off_min": float("nan"), "off_max": float("nan")})
    df = pd.DataFrame(rows)
    print(f"\nwindow {args.start} → {end.date()}  top_n={args.top_n} hold={args.hold} lev={args.leverage} "
          f"cap={args.sector_cap} lag={args.lag}")
    for c in ["cagr", "maxdd", "vol", "off_min", "off_max"]:
        df[c] = df[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:.2f}")
    df["nav"] = df["nav"].map(lambda x: f"{x:.2f}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
