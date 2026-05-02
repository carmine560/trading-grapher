import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

import indicators


def test_ema_masks_warmup_period():
    series = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)

    result = indicators.ema(series, span=3)
    expected = series.ewm(span=3).mean()
    expected.iloc[:2] = np.nan

    assert_series_equal(result, expected)


def test_vwap_resets_daily_and_skips_afternoon_only_day():
    index = pd.DatetimeIndex(
        [
            "2024-01-02 09:00:00",
            "2024-01-02 09:01:00",
            "2024-01-03 12:30:00",
            "2024-01-03 12:31:00",
        ]
    )
    high = pd.Series([10.0, 13.0, 20.0, 21.0], index=index)
    low = pd.Series([8.0, 11.0, 18.0, 19.0], index=index)
    close = pd.Series([9.0, 12.0, 19.0, 20.0], index=index)
    volume = pd.Series([100.0, 200.0, 150.0, 150.0], index=index)

    result = indicators.vwap(
        high,
        low,
        close,
        volume,
        morning_session_end="11:30:00",
    )

    assert result.iloc[0] == 9.0
    assert result.iloc[1] == 11.0
    assert result.iloc[2:].isna().all()


def test_stochastics_handles_flat_ranges_without_infinity():
    index = pd.date_range("2024-01-02 09:00:00", periods=4, freq="min")
    high = pd.Series([10.0, 10.0, 10.0, 10.0], index=index)
    low = pd.Series([10.0, 10.0, 10.0, 10.0], index=index)
    close = pd.Series([10.0, 10.0, 10.0, 10.0], index=index)

    result = indicators.stochastics(
        high,
        low,
        close,
        k=2,
        d=2,
        smooth_k=1,
    )

    assert np.isfinite(result.to_numpy()[~np.isnan(result.to_numpy())]).all()
    assert result.iloc[-1]["k"] == 0.0
    assert result.iloc[-1]["d"] == 0.0
