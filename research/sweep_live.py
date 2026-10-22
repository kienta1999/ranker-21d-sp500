#!/usr/bin/env python3
"""Live-config sweep on the daily engine (scripts/backtest.py).

Runs each (model, top_n, hold, leverage) once from --start, then slices the
mean-of-offsets equity curve into DEV / HOLDOUT / 2021+ windows.

    uv run python research/sweep_live.py --top-n 20,30,40 --hold 21,63 --leverage 1.0,1.35
"""
import argparse
import itertools
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

WINDOWS = {"dev": ("2012-01-01", "2022-12-31"), "hold": ("2023-01-01", None), "2021+": ("2021-01-01", None)}


def window_stats(curve: pd.Series, start, end) -> dict:
    c = curve.loc[start:end] if end else curve.loc[start:]
    return B.compute_stats(c / c.iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v2,static")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--top-n", default="40")
    ap.add_argument("--hold", default="21")
    ap.add_argument("--leverage", default="1.0")
    ap.add_argument("--sector-cap", type=float, default=strategy.LIVE_SECTOR_CAP)
    ap.add_argument("--vol-target", type=float, default=strategy.DEFAULT_VOL_TARGET)
    ap.add_argument("--variants", default="voltgt", help="raw,voltgt")
    ap.add_argument("--drift", default="0", help="0,1 — buy-and-hold weights between rebalances")
    args = ap.parse_args()
    if args.sector_cap is not None and args.sector_cap <= 0:
        args.sector_cap = None  # 0 → unconstrained (strategy.top_picks treats 0 as 1 name/sector)

    panel = load_panel()
    md = load_market()
    market = strategy.prepare_market(md["SPY"], md["VIX"])
    scored = {}
    for m in args.models.split(","):
        if m == "v2":
            scored[m] = B.predict_test(panel, B.MODEL_PATH, args.start)
        elif m == "static":
            scored[m] = B.predict_test(panel, B.LEGACY_MODEL_PATH, args.start)
        else:
            raise SystemExit(m)
    rows = []
    grid = itertools.product(scored, [int(x) for x in args.top_n.split(",")],
                             [int(x) for x in args.hold.split(",")],
                             [float(x) for x in args.leverage.split(",")],
                             args.variants.split(","),
                             [bool(int(x)) for x in args.drift.split(",")])
    for m, n, h, lev, var, drift in grid:
        tp = scored[m]
        curves, _, _, _ = B.run_shifted_starts(
            tp, market, vol_target_on=(var == "voltgt"), top_n=n, hold_days=h,
            cost_per_side=strategy.COST_PER_SIDE, vol_target=args.vol_target,
            weight_mode="equal", quality_filter_on=True, leverage=lev,
            margin_rate=strategy.DEFAULT_MARGIN_RATE, sector_cap=args.sector_cap, drift=drift)
        mean = curves.mean(axis=1)
        row = {"model": m, "n": n, "hold": h, "lev": lev, "var": var, "drift": int(drift)}
        for w, (s, e) in WINDOWS.items():
            st = window_stats(mean, s, e)
            row[f"{w}_cagr"] = st["cagr"]
            row[f"{w}_sh"] = st["sharpe"]
            row[f"{w}_dd"] = st["max_dd"]
        rows.append(row)
        print({k: (f"{v:+.3f}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    spy = B.spy_buy_and_hold(market, pd.Timestamp(args.start)).loc[:mean.index.max()]
    srow = {"model": "SPY", "n": 0, "hold": 0, "lev": 1.0, "var": "-", "drift": 0}
    for w, (s, e) in WINDOWS.items():
        st = window_stats(spy, s, e)
        srow[f"{w}_cagr"], srow[f"{w}_sh"], srow[f"{w}_dd"] = st["cagr"], st["sharpe"], st["max_dd"]
    rows.append(srow)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.endswith("_cagr") or c.endswith("_dd"):
            df[c] = df[c].map(lambda x: f"{x:+.1%}")
        elif c.endswith("_sh"):
            df[c] = df[c].map(lambda x: f"{x:.2f}")
    print(f"\nend {mean.index.max().date()}  sector_cap={args.sector_cap} vol_target={args.vol_target}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
