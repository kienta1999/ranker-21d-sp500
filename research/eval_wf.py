#!/usr/bin/env python3
"""Compare cached walk-forward variants on DEV and HOLDOUT.

    uv run python research/eval_wf.py wf_base wf_extra --top-n 40
    uv run python research/eval_wf.py wf_base --top-n 20,40,60 --hold 21,63
    uv run python research/eval_wf.py wf_base wf_extra --yearly
    uv run python research/eval_wf.py 'blend:wf_a+wf_b' ...   # per-date rank average
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import wf_lib as W  # noqa: E402


def get_score(name: str, panel: pd.DataFrame) -> pd.Series:
    if name.startswith("blend:"):
        parts = name[6:].split("+")
        ranks = []
        for p in parts:
            w = 1.0
            if "*" in p:
                w, p = p.split("*")
                w = float(w)
            s = W.load_preds(p, panel)
            ranks.append(w * s.groupby(panel["date"]).rank(pct=True))
        return sum(ranks)
    s = W.load_preds(name, panel)
    if s is None:
        raise SystemExit(f"no cached preds for {name}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--top-n", default="40")
    ap.add_argument("--hold", default="21")
    ap.add_argument("--sector-cap", type=float, default=None)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--yearly", action="store_true")
    ap.add_argument("--no-ic", action="store_true")
    ap.add_argument("--leverage", type=float, default=1.0)
    args = ap.parse_args()

    panel = W.load_research_panel(extra=False)
    frames = []
    yearly = {}
    for name in args.names:
        score = get_score(name, panel)
        for n in [int(x) for x in args.top_n.split(",")]:
            for h in [int(x) for x in args.hold.split(",")]:
                lab = f"{name} n={n} h={h}"
                frames.append(W.evaluate(panel, score, top_n=n, hold=h, label=lab,
                                         quality=not args.no_quality,
                                         sector_cap=args.sector_cap,
                                         with_ic=not args.no_ic,
                                         leverage=args.leverage))
        if args.yearly:
            yearly[name] = W.yearly_excess(panel, score, int(args.top_n.split(",")[0]),
                                           quality=not args.no_quality)
    res = pd.concat(frames)
    for w in ["dev", "hold"]:
        print(f"\n=== {w.upper()} ===")
        print(W.fmt(res.loc[[w]].reset_index(drop=True)))
    if args.yearly:
        y = pd.DataFrame(yearly)
        print("\n=== yearly annualised excess vs equal-weight universe (top-N, overlapping) ===")
        print(y.map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()
