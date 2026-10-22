#!/usr/bin/env python3
"""Self-check for scripts/ranker_v2.py on a small synthetic panel.

    uv run python scripts/test_ranker_v2.py

Covers: per-date percentile transform, label grading, the walk-forward
purge (no training label may overlap the test window), that the model
learns a planted signal, save/load round-trip, and strategy dispatch.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranker_v2 as R  # noqa: E402
import strategy  # noqa: E402
from features import CATEGORICAL_FEATURES  # noqa: E402

SECTORS = ["Tech", "Health", "Fin", "Energy"]


def _synthetic(n_dates: int = 400, n_tickers: int = 120, seed: int = 0) -> pd.DataFrame:
    """Random-walk prices where `ret_63d` (and only it) predicts the next
    21 days: each ticker's drift is proportional to its trailing signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    rows = []
    quality = rng.normal(size=n_tickers)  # persistent per-ticker drift
    for i in range(n_tickers):
        r = 0.002 * quality[i] + rng.normal(0, 0.01, n_dates)
        close = 100 * np.exp(np.cumsum(r))
        df = pd.DataFrame({"date": dates, "ticker": f"T{i:03d}", "close": close})
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    panel["gics_sector"] = pd.Categorical(
        panel["ticker"].map(lambda t: SECTORS[int(t[1:]) % 4]))
    # Fill every model input with noise, then plant the signal.
    for c in R.RAW_FEATURES:
        if c in CATEGORICAL_FEATURES:
            continue
        panel[c] = rng.normal(size=len(panel))
    panel["ret_63d"] = panel["ticker"].map(dict(zip([f"T{i:03d}" for i in range(n_tickers)], quality)))
    panel["ret_63d"] += rng.normal(0, 0.3, len(panel))
    return panel


def main() -> None:
    panel = _synthetic()

    # 1. transform(): per-date percentiles in (0, 1], broadcast passthrough,
    #    v1 *_rank duplicates dropped, column order == MODEL_FEATURES.
    X = R.transform(panel)
    assert list(X.columns) == R.MODEL_FEATURES
    r = X["ret_63d__r"]
    assert r.between(0, 1, inclusive="right").all()
    per_date_max = r.groupby(panel["date"]).max()
    assert np.allclose(per_date_max, 1.0)
    assert "ret_63d_rank" not in R.MODEL_FEATURES  # raw ret_63d is ranked instead
    assert not any(c.startswith("vix_level") for c in R.MODEL_FEATURES)  # broadcast dropped
    assert "gics_sector" in R.MODEL_FEATURES and "sector_ret_63d__r" in R.MODEL_FEATURES
    # Ranks depend only on the date's own cross-section.
    one_day = panel[panel["date"] == panel["date"].iloc[0]]
    assert np.allclose(R.transform(one_day)["ret_63d__r"], X.loc[one_day.index, "ret_63d__r"])

    # 2. add_relevance(): grades 0..9, ~10% each per date, NaN where the
    #    21d forward price isn't known yet.
    graded = R.add_relevance(panel.assign(
        **{c: 0.0 for c in R.ALL_FEATURES if c not in CATEGORICAL_FEATURES and c not in panel}))
    last = graded["date"].max()
    assert graded.loc[graded["date"] == last, "relevance"].isna().all()
    rel = graded["relevance"].dropna()
    assert set(rel.unique()) == set(range(10))
    share = rel.value_counts(normalize=True)
    assert share.between(0.08, 0.12).all(), share

    # 3. train_cutoff(): the last training label (cutoff + 21 trading days)
    #    must land strictly before the first test date.
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    test_start = dates[300]
    cutoff = R.train_cutoff(dates, test_start)
    label_end = dates[dates.get_loc(cutoff) + R.HORIZON]
    assert label_end < test_start, (cutoff, label_end, test_start)
    assert dates.get_loc(test_start) - dates.get_loc(cutoff) == R.HORIZON + 1

    # 4. fit() learns the planted signal: out-of-sample top-decile names
    #    must have clearly higher planted quality than the bottom decile.
    old = (R.SEEDS, R.NUM_BOOST_ROUND, R.LGB_PARAMS["min_data_in_leaf"])
    R.NUM_BOOST_ROUND = 40
    R.LGB_PARAMS["min_data_in_leaf"] = 50
    R.LGB_PARAMS["num_threads"] = 2
    try:
        train = graded[graded["date"] <= cutoff]
        model = R.fit(train, seeds=(1, 2))
        test = graded[graded["date"] >= test_start]
        s = pd.Series(model.score(test), index=test.index)
        pct = s.groupby(test["date"]).rank(pct=True)
        top_q = test.loc[pct >= 0.9, "ret_63d"].mean()
        bot_q = test.loc[pct <= 0.1, "ret_63d"].mean()
        assert top_q - bot_q > 1.0, (top_q, bot_q)
        assert model.meta["train_end"] <= str(cutoff.date())

        # 5. save/load round-trip reproduces scores exactly, and
        #    strategy.load_model/predict dispatch to v2 for a directory.
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            again = strategy.load_model(tmp)
            assert isinstance(again, R.RankerV2)
            scored = strategy.predict(test, again)
            assert np.allclose(scored["predicted_return"].to_numpy(), s.to_numpy())
    finally:
        R.SEEDS, R.NUM_BOOST_ROUND, R.LGB_PARAMS["min_data_in_leaf"] = old
        R.LGB_PARAMS["num_threads"] = 4

    # 6. Edge: an empty frame transforms to an empty matrix, not an error.
    empty = R.transform(panel.iloc[:0])
    assert empty.shape == (0, len(R.MODEL_FEATURES))

    print("test_ranker_v2: all checks passed")


if __name__ == "__main__":
    main()
