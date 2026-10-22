#!/usr/bin/env python3
"""Candidate new features → data/processed/extra_features.parquet.

Every value on row date=D uses only data with date <= D (rolling windows,
positive shifts, and an earnings-announcement return that only becomes
visible on the trading day after the announcement window closes).

Per-ticker, from OHLCV:
    mom_12_1          close[t-21] / close[t-252] - 1   (classic 12-1 momentum)
    ret_126d          6-month return
    idio_vol_60d      residual vol vs SPY over 60d (annualised)
    max_ret_21d       max daily return over 21d (Bali-Cakici-Whitelaw MAX / lottery)
    skew_60d          skewness of daily returns, 60d
    amihud_21d        log1p(1e9 * mean(|r| / dollar volume)) over 21d
    log_dollar_vol_63d  log mean dollar volume, 63d (size / liquidity proxy)
    dvol_trend        log(dollar vol 21d / dollar vol 252d)
    dist_52w_low      close / 252d low - 1
    info_discrete     sign(ret_252d) * (%neg days - %pos days) over 252d (frog-in-the-pan)
    seasonal_ret      mean of the same-calendar-month 21d return at annual lags 1..5
    ret_5d            short-term reversal input
    ear_3d            abnormal (vs SPY) return over [e-1, e+1] around the latest
                      earnings announcement, visible from e+2 on, NaN after 90 days
Panel-level:
    sector_ret_63d / sector_ret_21d   equal-weighted sector mean return
    mom_12_1_secrank, idio_vol_60d_secrank   within-(date, sector) percentile
    <x>_rank          cross-sectional percentile of each new numeric feature
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from data import load_market, load_prices  # noqa: E402
from earnings import load_earnings_dates  # noqa: E402

OUT = os.path.join(_ROOT, "data", "processed", "extra_features.parquet")
FEATURES_PATH = os.path.join(_ROOT, "data", "processed", "features.parquet")

NEW_RAW = [
    "mom_12_1", "ret_126d", "idio_vol_60d", "max_ret_21d", "skew_60d",
    "amihud_21d", "log_dollar_vol_63d", "dvol_trend", "dist_52w_low",
    "info_discrete", "seasonal_ret", "ret_5d", "ear_3d",
]
RANKED = [
    "mom_12_1", "ret_126d", "idio_vol_60d", "max_ret_21d", "skew_60d",
    "amihud_21d", "log_dollar_vol_63d", "dvol_trend", "info_discrete",
    "seasonal_ret", "ret_5d", "ear_3d",
]
SECTOR = ["sector_ret_63d", "sector_ret_21d", "mom_12_1_secrank", "idio_vol_60d_secrank"]
EXTRA_FEATURES = NEW_RAW + [f"{c}_rank" for c in RANKED] + SECTOR


def per_ticker(px: pd.DataFrame, spy_ret: pd.Series, edates: pd.DatetimeIndex) -> pd.DataFrame:
    c, v = px["Close"], px["Volume"]
    r = c.pct_change()
    out = pd.DataFrame(index=px.index)
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    out["ret_126d"] = c.pct_change(126)
    out["ret_5d"] = c.pct_change(5)

    sr = spy_ret.reindex(px.index)
    cov = r.rolling(60).cov(sr)
    var_s = sr.rolling(60).var()
    beta = cov / var_s
    var_r = r.rolling(60).var()
    out["idio_vol_60d"] = np.sqrt((var_r - beta**2 * var_s).clip(lower=0)) * np.sqrt(252)

    out["max_ret_21d"] = r.rolling(21).max()
    out["skew_60d"] = r.rolling(60).skew()

    dv = c * v
    illiq = (r.abs() / dv.replace(0, np.nan)).rolling(21, min_periods=15).mean()
    out["amihud_21d"] = np.log1p(1e9 * illiq)
    dv63 = dv.rolling(63, min_periods=40).mean()
    out["log_dollar_vol_63d"] = np.log(dv63.replace(0, np.nan))
    out["dvol_trend"] = np.log(dv.rolling(21).mean() / dv.rolling(252, min_periods=150).mean())

    out["dist_52w_low"] = c / c.rolling(252).min() - 1

    pos = (r > 0).astype(float).rolling(252).mean()
    neg = (r < 0).astype(float).rolling(252).mean()
    out["info_discrete"] = np.sign(c.pct_change(252)) * (neg - pos)

    # Same-calendar-month return at annual lags (Heston & Sadka 2008):
    # the 21d return realised over the window starting ~1y, 2y, ... ago at
    # this point of the year → predicts the coming 21d.
    r21_fwd_lagged = []
    for k in range(1, 6):
        # return over (t - 252k, t - 252k + 21] — fully in the past for k ≥ 1
        r21_fwd_lagged.append(c.shift(252 * k - 21) / c.shift(252 * k) - 1)
    out["seasonal_ret"] = pd.concat(r21_fwd_lagged, axis=1).mean(axis=1, skipna=True)
    out.loc[c.shift(252 * 2).isna(), "seasonal_ret"] = np.nan  # need ≥2 years

    # Earnings announcement abnormal return.
    out["ear_3d"] = np.nan
    if len(edates):
        idx = px.index
        logc = np.log(c)
        logs = np.log((1 + sr.fillna(0)).cumprod())
        vals = np.full(len(idx), np.nan)
        for e in edates:
            p = idx.searchsorted(e)  # first trading day ≥ e
            if p < 1 or p + 2 >= len(idx):
                continue
            lo, hi = p - 1, p + 1
            ab = (logc.iloc[hi] - logc.iloc[lo]) - (logs.iloc[hi] - logs.iloc[lo])
            vals[p + 2] = ab  # visible two trading days after the announcement day
        s = pd.Series(vals, index=idx).ffill(limit=63)
        out["ear_3d"] = s
    return out


def main() -> None:
    keys = pd.read_parquet(FEATURES_PATH, columns=["date", "ticker", "gics_sector", "ret_21d", "ret_63d"])
    keys["date"] = pd.to_datetime(keys["date"])
    prices = load_prices()
    spy = load_market()["SPY"]
    spy_ret = spy["Close"].pct_change()
    spy_ret.index = pd.to_datetime(spy_ret.index)
    earn = load_earnings_dates(include_upcoming=False)
    e_by = {t: pd.DatetimeIndex(g["report_date"]).sort_values() for t, g in earn.groupby("ticker")}

    tickers = set(keys["ticker"].unique())
    frames = []
    for t, px in tqdm(prices.items(), desc="extra"):
        if t not in tickers:
            continue
        px = px.copy()
        px.index = pd.to_datetime(px.index)
        f = per_ticker(px, spy_ret, e_by.get(t, pd.DatetimeIndex([])))
        f["ticker"] = t
        frames.append(f.reset_index().rename(columns={"Date": "date", "index": "date"}))
    ex = pd.concat(frames, ignore_index=True)
    ex["date"] = pd.to_datetime(ex["date"])
    panel = keys.merge(ex, on=["date", "ticker"], how="left")

    g = panel.groupby("date", sort=False)
    for col in RANKED:
        panel[f"{col}_rank"] = g[col].rank(pct=True)
    gs = panel.groupby(["date", "gics_sector"], observed=True, sort=False)
    panel["sector_ret_63d"] = gs["ret_63d"].transform("mean")
    panel["sector_ret_21d"] = gs["ret_21d"].transform("mean")
    panel["mom_12_1_secrank"] = gs["mom_12_1"].rank(pct=True)
    panel["idio_vol_60d_secrank"] = gs["idio_vol_60d"].rank(pct=True)
    unknown = panel["gics_sector"].astype(str) == "Unknown"
    panel.loc[unknown, SECTOR] = np.nan

    out = panel[["date", "ticker"] + EXTRA_FEATURES]
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows, {len(EXTRA_FEATURES)} features")
    print(out[EXTRA_FEATURES].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
