"""Walk-forward research harness.

Why this exists: the production model is trained once on 2007-2017 and
early-stopped on 2018-2020, so (a) it never learns from 2018+ data and
(b) the 2021+ test window has been re-used for many design decisions.
This harness retrains every January on all data whose labels were fully
realised before that January (expanding window, 21-trading-day purge),
so every prediction is out-of-sample and the final fold is the model you
would trade live.

Evaluation windows (fixed up front, before any experiment ran):
    DEV      2012-01-01 → 2022-12-31   — choose variants here
    HOLDOUT  2023-01-01 → latest       — confirm only, never tune on it

The fast evaluator approximates backtest.py: for each of 21 rebalance
offsets, hold the equal-weighted top-N (after the cataclysmic quality
filter) for 21 trading days, buy-and-hold within the period, 5 bps per
side on turnover. Candidates that win here are re-checked with
backtest.py's daily engine (`daily_backtest`).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import strategy  # noqa: E402
from labels import panel_forward_return  # noqa: E402
from features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NULLABLE_FEATURES,
)

FEATURES_PATH = os.path.join(_ROOT, "data", "processed", "features.parquet")
EXTRA_PATH = os.path.join(_ROOT, "data", "processed", "extra_features.parquet")
CACHE_DIR = os.path.join(_ROOT, "research", "cache")

DEV_START, DEV_END = "2012-01-01", "2022-12-31"
HOLDOUT_START = "2023-01-01"
FIRST_FOLD_YEAR = 2012
HORIZON = 21
COST = strategy.COST_PER_SIDE

BASE_PARAMS = {  # the saved production model's params (reports/train_metrics.json)
    "max_depth": 5,
    "learning_rate": 0.0015032457864721574,
    "n_estimators": 41,  # its best_iteration
    "min_child_weight": 9,
    "subsample": 0.6723478460829574,
    "colsample_bytree": 0.5786453944890463,
    "reg_lambda": 0.004741934136013863,
}


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────


def load_research_panel(extra: bool = True) -> pd.DataFrame:
    """features.parquet + unclipped/clipped forward labels (+ extra features).

    Rows are restricted to the production model-ready set (non-nullable
    features present) so the baseline matches dataset.load_panel exactly.
    """
    panel = pd.read_parquet(FEATURES_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Span-checked: a ticker that left the index and re-joined years later
    # would otherwise get a multi-year "21-day" return (labels.py docstring).
    fwd = panel_forward_return(panel, HORIZON)
    panel["fwd_raw"] = fwd  # unclipped — used for P&L
    panel["fwd_raw_63"] = panel_forward_return(panel, 63)
    clipped = fwd.clip(-0.5, 0.5)
    panel["fwd_clip"] = clipped
    panel["y_demeaned"] = clipped - clipped.groupby(panel["date"]).transform("mean")

    if extra and os.path.exists(EXTRA_PATH):
        ex = pd.read_parquet(EXTRA_PATH)
        ex["date"] = pd.to_datetime(ex["date"])
        # features.parquet carries Bucket 9 itself now; only merge what's missing
        # (the research-only *_rank columns of the new features).
        ex = ex[["date", "ticker"] + [c for c in ex.columns if c not in panel.columns]]
        panel = panel.merge(ex, on=["date", "ticker"], how="left")

    required = [
        c for c in ALL_FEATURES
        if c not in CATEGORICAL_FEATURES and c not in NULLABLE_FEATURES
    ]
    panel = panel.dropna(subset=required).reset_index(drop=True)
    panel = panel[panel["date"] >= "2007-01-01"].reset_index(drop=True)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward training
# ─────────────────────────────────────────────────────────────────────────────


def fold_bounds(dates: pd.DatetimeIndex, year: int, purge: int = HORIZON) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(train_cutoff, test_start, test_end) for the fold predicting `year`.

    train_cutoff is HORIZON+1 trading days before test_start so every
    training label (which looks HORIZON days ahead) is realised before the
    first test date — no label overlap across the boundary.
    """
    test_dates = dates[(dates >= f"{year}-01-01") & (dates <= f"{year}-12-31")]
    test_start, test_end = test_dates[0], test_dates[-1]
    pos = dates.get_loc(test_start)
    train_cutoff = dates[pos - purge - 1]
    return train_cutoff, test_start, test_end


def walk_forward(
    panel: pd.DataFrame,
    feature_cols: list[str],
    fit_predict,
    label_col: str = "y_demeaned",
    first_year: int = FIRST_FOLD_YEAR,
    last_year: int | None = None,
    train_start: str = "2007-01-01",
    rolling_years: int | None = None,
    date_stride: int = 1,
    verbose: bool = True,
    purge: int = HORIZON,
) -> pd.Series:
    """Return out-of-sample predictions (index aligned to `panel`) for every
    row in years first_year..last_year.

    `fit_predict(train_df, test_df, feature_cols, label_col) -> np.ndarray`
    is the model under test. `date_stride` keeps every k-th training date
    (labels on adjacent dates overlap 20/21 days, so thinning is nearly free).
    """
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    if last_year is None:
        last_year = dates[-1].year
    preds = pd.Series(np.nan, index=panel.index)
    stride_dates = set(dates[::date_stride])
    for year in range(first_year, last_year + 1):
        t0 = time.time()
        cutoff, ts, te = fold_bounds(dates, year, purge)
        lo = pd.Timestamp(train_start)
        if rolling_years is not None:
            lo = max(lo, ts - pd.DateOffset(years=rolling_years))
        tr_mask = (panel["date"] >= lo) & (panel["date"] <= cutoff) & panel[label_col].notna()
        if date_stride > 1:
            tr_mask &= panel["date"].isin(stride_dates)
        te_mask = (panel["date"] >= ts) & (panel["date"] <= te)
        tr = panel.loc[tr_mask]
        te_df = panel.loc[te_mask]
        preds.loc[te_df.index] = fit_predict(tr, te_df, feature_cols, label_col)
        if verbose:
            print(f"    fold {year}: train {len(tr):,} rows ≤{cutoff.date()}  "
                  f"test {len(te_df):,}  ({time.time() - t0:.1f}s)", flush=True)
    return preds


def xgb_fit_predict(params: dict | None = None, seeds=(15,), objective="reg:squarederror"):
    import xgboost as xgb

    params = dict(BASE_PARAMS if params is None else params)

    def _fp(tr, te, cols, label):
        out = np.zeros(len(te))
        for s in seeds:
            common = dict(objective=objective, tree_method="hist", enable_categorical=True,
                          random_state=s, n_jobs=4, max_cat_to_onehot=1)
            if objective.startswith("rank:"):
                trs = tr.sort_values("date")
                qid = trs["date"].factorize(sort=True)[0]
                m = xgb.XGBRanker(**params, **common)
                m.fit(trs[cols], trs[label], qid=qid)
            else:
                m = xgb.XGBRegressor(**params, **common)
                m.fit(tr[cols], tr[label])
            p = m.predict(te[cols])
            # rank-average across seeds so each seed carries equal weight
            out += pd.Series(p, index=te.index).groupby(te["date"]).rank(pct=True).to_numpy()
        return out / len(seeds)

    return _fp


def lgb_fit_predict(params: dict, seeds=(15,), num_boost_round: int = 300):
    import lightgbm as lgb

    def _fp(tr, te, cols, label):
        out = np.zeros(len(te))
        for s in seeds:
            p = dict(params)
            p.update({"seed": s, "verbose": -1, "num_threads": 4})
            if p.get("objective") in ("lambdarank", "rank_xendcg"):
                trs = tr.sort_values("date")
                grp = trs.groupby("date", sort=True).size().to_numpy()
                ds = lgb.Dataset(trs[cols], trs[label], group=grp)
            else:
                ds = lgb.Dataset(tr[cols], tr[label])
            m = lgb.train(p, ds, num_boost_round=num_boost_round)
            pr = m.predict(te[cols])
            out += pd.Series(pr, index=te.index).groupby(te["date"]).rank(pct=True).to_numpy()
        return out / len(seeds)

    return _fp


# ─────────────────────────────────────────────────────────────────────────────
# Fast evaluation
# ─────────────────────────────────────────────────────────────────────────────

_QF_COLS = ("debt_to_equity", "current_ratio", "sales_growth_yoy", "insider_net_dollar_60d")


def quality_mask(df: pd.DataFrame) -> pd.Series:
    t = strategy.QUALITY_FILTER_DEFAULTS
    bad = (
        (df["debt_to_equity"] > t["max_debt_to_equity"])
        | (df["current_ratio"] < t["min_current_ratio"])
        | (df["sales_growth_yoy"] < t["min_sales_growth_yoy"])
        | (df["insider_net_dollar_60d"] < t["max_insider_net_sell_60d"])
    )
    return ~bad.fillna(False)


def select_top(df: pd.DataFrame, score: pd.Series, top_n: int,
               quality: bool = True, sector_cap: float | None = None) -> pd.Series:
    """Boolean mask: row is in its date's top-N by `score`."""
    s = score.copy()
    if quality:
        s = s.where(quality_mask(df))
    s = s.where(s.notna())
    if sector_cap is None:
        r = s.groupby(df["date"]).rank(ascending=False, method="first")
        return r <= top_n
    cap = max(1, int(sector_cap * top_n))
    tmp = pd.DataFrame({"date": df["date"], "sec": df["gics_sector"].astype(str), "s": s}).dropna()
    tmp = tmp.sort_values(["date", "s"], ascending=[True, False])
    tmp["sec_rank"] = tmp.groupby(["date", "sec"]).cumcount() + 1
    tmp = tmp[tmp["sec_rank"] <= cap]
    tmp["r"] = tmp.groupby("date").cumcount() + 1
    mask = pd.Series(False, index=df.index)
    mask.loc[tmp.index[tmp["r"] <= top_n]] = True
    return mask


def period_returns(df: pd.DataFrame, picks: pd.Series, start: str, end: str | None,
                   hold: int = HORIZON, cost: float = COST,
                   weights: pd.Series | None = None) -> dict[int, pd.Series]:
    """Per offset: Series of net period returns indexed by rebalance date."""
    fcol = "fwd_raw" if hold == HORIZON else f"fwd_raw_{hold}"
    d = df.loc[picks, ["date", "ticker", fcol]].rename(columns={fcol: "fwd_raw"}).copy()
    d = d[(d["date"] >= start) & ((d["date"] <= end) if end else True)]
    # Dates whose forward window isn't realised yet (the last HORIZON days of
    # the panel) are dropped; an isolated NaN on a realised date means the
    # name left the panel mid-hold → treat as flat.
    realised = df[fcol].notna().groupby(df["date"]).mean()
    d = d[d["date"].map(realised) > 0.5]
    d["fwd_raw"] = d["fwd_raw"].fillna(0.0)
    if weights is not None:
        d["w"] = weights.loc[d.index]
        d["w"] = d["w"] / d.groupby("date")["w"].transform("sum")
    else:
        d["w"] = 1.0 / d.groupby("date")["ticker"].transform("count")
    all_dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    all_dates = all_dates[(all_dates >= start) & ((all_dates <= end) if end else True)]
    gross = (d["w"] * d["fwd_raw"]).groupby(d["date"]).sum()
    wmap = {dt: dict(zip(g["ticker"], g["w"])) for dt, g in d.groupby("date")}
    out = {}
    for off in range(hold):
        rb = all_dates[off::hold]
        rb = rb[rb.isin(gross.index)]
        # drop a trailing period whose forward return isn't realised yet
        rets, prev = [], {}
        for dt in rb:
            w = wmap[dt]
            # drifted previous weights are approximated by the target weights
            turnover = sum(abs(w.get(t, 0) - prev.get(t, 0)) for t in set(w) | set(prev))
            rets.append(gross[dt] - turnover * cost)
            prev = w
        out[off] = pd.Series(rets, index=rb)
    return out


def stats_from_periods(per: dict[int, pd.Series], hold: int = HORIZON,
                       leverage: float = 1.0, margin: float = strategy.DEFAULT_MARGIN_RATE,
                       exposure: dict | None = None) -> dict:
    ppy = 252 / hold
    rows = []
    for off, r in per.items():
        r = r.dropna()
        if len(r) < 4:
            continue
        lev = leverage if exposure is None else exposure[off].reindex(r.index).fillna(leverage)
        rr = lev * r - np.maximum(lev - 1, 0) * margin / ppy
        eq = (1 + rr).cumprod()
        yrs = len(rr) / ppy
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
        vol = rr.std() * np.sqrt(ppy)
        dd = (eq / eq.cummax() - 1).min()
        rows.append((cagr, vol, cagr / vol if vol > 0 else 0, rr.mean() / rr.std() * np.sqrt(ppy), dd))
    a = np.array(rows)
    return {
        "cagr": a[:, 0].mean(), "vol": a[:, 1].mean(), "sharpe": a[:, 2].mean(),
        "sharpe_ari": a[:, 3].mean(), "maxdd": a[:, 4].mean(),
        "cagr_min": a[:, 0].min(), "cagr_max": a[:, 0].max(),
    }


def universe_periods(df: pd.DataFrame, start, end, hold=HORIZON):
    """Equal-weight whole-universe benchmark (no costs)."""
    allm = pd.Series(True, index=df.index)
    return period_returns(df, allm, start, end, hold, cost=0.0)


def ic_stats(df: pd.DataFrame, score: pd.Series, start, end) -> dict:
    m = (df["date"] >= start) & ((df["date"] <= end) if end else True) & score.notna() & df["fwd_raw"].notna()
    sub = df.loc[m, ["date"]].copy()
    sub["s"] = score[m].groupby(sub["date"]).rank()
    sub["y"] = df.loc[m, "fwd_raw"].groupby(sub["date"]).rank()
    ic = sub.groupby("date").apply(lambda g: np.corrcoef(g["s"], g["y"])[0, 1])
    monthly = ic.resample("ME").mean()
    return {"ic": ic.mean(), "ic_t": monthly.mean() / monthly.std() * np.sqrt(len(monthly))}


def evaluate(df: pd.DataFrame, score: pd.Series, top_n: int = 40, quality: bool = True,
             sector_cap: float | None = None, hold: int = HORIZON, label: str = "",
             windows=None, with_ic: bool = True, leverage: float = 1.0) -> pd.DataFrame:
    if windows is None:
        windows = {"dev": (DEV_START, DEV_END), "hold": (HOLDOUT_START, None)}
    picks = select_top(df, score, top_n, quality, sector_cap)
    rows = {}
    for w, (s, e) in windows.items():
        st = stats_from_periods(period_returns(df, picks, s, e, hold), hold, leverage=leverage)
        if with_ic:
            st.update(ic_stats(df, score, s, e))
        rows[w] = st
    out = pd.DataFrame(rows).T
    out.insert(0, "variant", label)
    return out


def yearly_excess(df: pd.DataFrame, score: pd.Series, top_n=40, quality=True, start="2012-01-01") -> pd.Series:
    """Mean 21d top-N return minus universe mean, averaged within each year (all dates, overlapping)."""
    picks = select_top(df, score, top_n, quality)
    m = df["date"] >= start
    top = df.loc[picks & m].groupby("date")["fwd_raw"].mean()
    uni = df.loc[m].groupby("date")["fwd_raw"].mean()
    ex = (top - uni).dropna()
    return ex.groupby(ex.index.year).mean() * (252 / HORIZON)


def fmt(df: pd.DataFrame) -> str:
    cols = [c for c in ["variant", "cagr", "sharpe", "maxdd", "vol", "cagr_min", "cagr_max", "ic", "ic_t"] if c in df.columns]
    d = df[cols].copy()
    for c in ["cagr", "maxdd", "vol", "cagr_min", "cagr_max"]:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:+.1%}")
    for c in ["sharpe", "ic_t"]:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:.2f}")
    if "ic" in d:
        d["ic"] = d["ic"].map(lambda x: f"{x:+.4f}")
    return d.to_string()


def save_preds(name: str, preds: pd.Series, panel: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = pd.DataFrame({"date": panel["date"], "ticker": panel["ticker"], "pred": preds})
    out.dropna().to_parquet(os.path.join(CACHE_DIR, f"pred_{name}.parquet"), index=False)


def load_preds(name: str, panel: pd.DataFrame) -> pd.Series | None:
    p = os.path.join(CACHE_DIR, f"pred_{name}.parquet")
    if not os.path.exists(p):
        return None
    pr = pd.read_parquet(p)
    m = panel[["date", "ticker"]].merge(pr, on=["date", "ticker"], how="left")
    return pd.Series(m["pred"].to_numpy(), index=panel.index)
