#!/usr/bin/env python3
"""Ranker v2 — LightGBM LambdaRank on per-date percentile features.

What changed vs xgb_v1 (and why — full write-up in README "Ranker v2"):
  1. Features: every per-ticker numeric feature is replaced by its per-date
     cross-sectional percentile before training/scoring. Raw levels (vol,
     ATR, insider dollars…) drift across eras, so a split learned on 2009
     vol levels misfires in 2017; percentiles mean the same thing every
     day. Broadcast regime features (SPY/VIX/FRED) stay raw — they only
     matter through interactions — and the walk-forward ablation found
     they add nothing here, so v2 drops them (and with them its dependence
     on the FRED feed). Adds the 17 Bucket-9 features.
  2. Objective: LambdaRank with per-date query groups and decile relevance
     grades of the clipped forward 21d return, truncated at the top 50 —
     it optimises "get the top of the list right", which is what a top-40
     basket trades. Plain RMSE regression spends its capacity on the
     middle of the cross-section.
  3. Training data: ALL dates whose 21d label is realised (no frozen
     2007-2017 window, no val-set early stopping). The backtest uses an
     annual walk-forward (retrain every January on everything realised
     before it), so every backtest prediction is out-of-sample and the
     live model is simply the latest fold.
  4. Seed ensemble: 3 boosters, per-date rank-averaged (5 seeds and
     stride-1 training were both no better in the walk-forward).

CLI:
    uv run python scripts/ranker_v2.py train          # full-data live model → models/ranker_v2/
    uv run python scripts/ranker_v2.py walkforward    # OOS scores 2012→ → data/processed/ranker_v2_oos.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MACRO_REGIME_FEATURES,
    MARKET_REGIME_FEATURES,
    NULLABLE_FEATURES,
    V2_FEATURES,
    load_features,
)
from labels import panel_forward_return  # noqa: E402

_ROOT = os.path.dirname(_HERE)
MODEL_DIR = os.path.join(_ROOT, "models", "ranker_v2")
OOS_PATH = os.path.join(_ROOT, "data", "processed", "ranker_v2_oos.parquet")

HORIZON = 21
LABEL_CLIP = 0.5
FIRST_FOLD_YEAR = 2012
SEEDS = (1, 2, 3)
NUM_BOOST_ROUND = 300
DATE_STRIDE = 3  # train on every 3rd date — labels overlap 20/21 days, and stride 1 was no better in walk-forward

LGB_PARAMS: dict = {
    "objective": "lambdarank",
    "lambdarank_truncation_level": 50,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
}

# Broadcast SPY/VIX/FRED regime features are excluded: same value for every
# ticker on a date, and dropping them was neutral on the 2012-2022 dev
# walk-forward (dev CAGR 26.2% vs 26.6%, inside seed noise).
BROADCAST: set[str] = set(MARKET_REGIME_FEATURES) | set(MACRO_REGIME_FEATURES)
RAW_FEATURES: list[str] = [c for c in dict.fromkeys(ALL_FEATURES + V2_FEATURES) if c not in BROADCAST]


def _model_columns() -> tuple[list[str], list[str]]:
    """(columns to percentile-rank, columns passed through as-is)."""
    to_rank, passthrough = [], []
    for c in RAW_FEATURES:
        if c in CATEGORICAL_FEATURES or c.endswith("_secrank"):
            passthrough.append(c)
        elif c.endswith("_rank") and c[:-5] in RAW_FEATURES:
            continue  # v1 rank column; the raw value is ranked below
        else:
            to_rank.append(c)
    return to_rank, passthrough


TO_RANK, PASSTHROUGH = _model_columns()
MODEL_FEATURES: list[str] = [f"{c}__r" for c in TO_RANK] + PASSTHROUGH


def eligible(panel: pd.DataFrame) -> pd.DataFrame:
    """Same row gate as the v1 pipeline (strategy.filter_valid_features):
    every non-nullable v1 feature present. v2 additions are all nullable."""
    required = [c for c in ALL_FEATURES if c not in CATEGORICAL_FEATURES and c not in NULLABLE_FEATURES]
    return panel.dropna(subset=required)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix: per-date percentiles of TO_RANK + PASSTHROUGH columns
    (sector category, within-sector ranks, sector mean returns).

    Ranks are taken over whatever rows of each date `df` holds, so callers
    must pass the full eligible cross-section of a date (today.py does).
    """
    g = df.groupby("date", sort=False)
    X = pd.DataFrame(index=df.index)
    for c in TO_RANK:
        X[f"{c}__r"] = g[c].rank(pct=True)
    for c in PASSTHROUGH:
        X[c] = df[c]
    return X[MODEL_FEATURES]


def add_relevance(features: pd.DataFrame) -> pd.DataFrame:
    """Forward 21d return → per-date decile grade 0..9 over the eligible rows.

    The forward return is taken on the full member panel (same as
    labels.py) *before* the eligibility gate, so a ticker with a
    feature gap doesn't get a label that spans the wrong number of days.
    """
    panel = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["fwd_21d"] = panel_forward_return(panel, HORIZON).clip(-LABEL_CLIP, LABEL_CLIP)
    panel = eligible(panel)
    panel = panel[panel["date"] >= "2007-01-01"].copy()
    pct = panel["fwd_21d"].groupby(panel["date"]).rank(pct=True, method="first")
    panel["relevance"] = np.floor(pct * 10).clip(0, 9)
    return panel.drop(columns="fwd_21d")


class RankerV2:
    """Seed ensemble of LightGBM boosters; `score` returns the per-date
    average percentile (higher = better), used as `predicted_return`."""

    def __init__(self, boosters: list, meta: dict | None = None):
        self.boosters = boosters
        self.meta = meta or {}

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = transform(df)
        dates = df["date"].to_numpy()
        out = np.zeros(len(df))
        for b in self.boosters:
            p = pd.Series(b.predict(X), index=df.index)
            out += p.groupby(dates).rank(pct=True).to_numpy()
        return out / len(self.boosters)

    def save(self, path: str = MODEL_DIR) -> None:
        os.makedirs(path, exist_ok=True)
        for i, b in enumerate(self.boosters):
            b.save_model(os.path.join(path, f"booster_{i}.txt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = MODEL_DIR) -> "RankerV2":
        import lightgbm as lgb

        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        boosters = [lgb.Booster(model_file=os.path.join(path, f"booster_{i}.txt"))
                    for i in range(meta["n_boosters"])]
        return cls(boosters, meta)


def fit(train: pd.DataFrame, seeds=SEEDS, stride: int = DATE_STRIDE) -> RankerV2:
    import lightgbm as lgb

    tr = train[train["relevance"].notna()]
    if stride > 1:
        keep = np.sort(tr["date"].unique())[::stride]
        tr = tr[tr["date"].isin(keep)]
    tr = tr.sort_values("date")
    X = transform(tr)
    groups = tr.groupby("date", sort=True).size().to_numpy()
    boosters = []
    for s in seeds:
        params = dict(LGB_PARAMS, seed=s)
        ds = lgb.Dataset(X, tr["relevance"], group=groups, categorical_feature=CATEGORICAL_FEATURES)
        boosters.append(lgb.train(params, ds, num_boost_round=NUM_BOOST_ROUND))
    meta = {
        "n_boosters": len(boosters), "seeds": list(seeds), "params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND, "features": MODEL_FEATURES,
        "train_rows": len(tr), "train_start": str(tr["date"].min().date()),
        "train_end": str(tr["date"].max().date()), "stride": stride,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    return RankerV2(boosters, meta)


def load_training_panel() -> pd.DataFrame:
    panel = add_relevance(load_features())
    panel["gics_sector"] = panel["gics_sector"].astype("category")
    return panel.reset_index(drop=True)


def train_cutoff(dates: pd.DatetimeIndex, first_test_date: pd.Timestamp) -> pd.Timestamp:
    """Last training date whose HORIZON-day label is realised strictly
    before `first_test_date` (purge = HORIZON + 1 trading days)."""
    pos = dates.searchsorted(first_test_date)
    return dates[pos - HORIZON - 1]


def walkforward(panel: pd.DataFrame, first_year: int = FIRST_FOLD_YEAR,
                seeds=SEEDS, stride: int = DATE_STRIDE) -> pd.DataFrame:
    """Annual expanding-window OOS scores → DataFrame[date, ticker, predicted_return]."""
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    frames = []
    for year in range(first_year, dates[-1].year + 1):
        t0 = time.time()
        test_dates = dates[(dates >= f"{year}-01-01") & (dates <= f"{year}-12-31")]
        cutoff = train_cutoff(dates, test_dates[0])
        model = fit(panel[panel["date"] <= cutoff], seeds=seeds, stride=stride)
        te = panel[panel["date"].isin(test_dates)]
        frames.append(pd.DataFrame({"date": te["date"], "ticker": te["ticker"],
                                    "predicted_return": model.score(te)}))
        print(f"  fold {year}: train ≤{cutoff.date()} ({model.meta['train_rows']:,} rows), "
              f"test {len(te):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["train", "walkforward"])
    ap.add_argument("--stride", type=int, default=DATE_STRIDE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = load_training_panel()
    if args.cmd == "train":
        dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
        realised = panel.loc[panel["relevance"].notna(), "date"].max()
        print(f"Training ranker v2 on all realised labels ≤{realised.date()} "
              f"({len(SEEDS)} seeds, stride {args.stride})...", flush=True)
        model = fit(panel, stride=args.stride)
        model.meta["panel_end"] = str(dates[-1].date())
        model.save(args.out or MODEL_DIR)
        print(f"  -> {args.out or MODEL_DIR}  ({model.meta['train_rows']:,} rows, "
              f"{len(MODEL_FEATURES)} features)")
    else:
        print(f"Walk-forward ranker v2 ({len(SEEDS)} seeds, stride {args.stride})...", flush=True)
        oos = walkforward(panel, stride=args.stride)
        out = args.out or OOS_PATH
        oos.to_parquet(out, index=False)
        print(f"  -> {out}  ({len(oos):,} rows, {oos['date'].min().date()} → {oos['date'].max().date()})")


if __name__ == "__main__":
    main()
