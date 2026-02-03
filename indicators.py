"""Calculate exponential moving averages and stochastic oscillator values."""

import sys

import numpy as np
import pandas as pd


def ema(series, span):
    """Calculate the exponential moving average of a series."""
    ema = series.ewm(span=span).mean()
    ema.iloc[: span - 1] = np.nan
    return ema


def tema(series, span):
    """Calculate the triple exponential moving average of a series."""
    ema_1 = ema(series, span)
    ema_2 = ema(ema_1, span)
    ema_3 = ema(ema_2, span)
    tema = 3 * (ema_1 - ema_2) + ema_3
    tema.iloc[: 3 * (span - 1)] = np.nan
    return tema


def vwap(high, low, close, volume, morning_session_end=None):
    """Calculate daily-reset VWAP with optional morning-session filtering."""
    typical_price = (high + low + close) / 3

    def calc_vwap(group):
        day_timestamps = group.index

        first_timestamp = day_timestamps.min()
        if pd.isna(first_timestamp):
            return pd.Series(np.nan, index=day_timestamps)

        if morning_session_end is None:
            has_morning_session = True
        else:
            has_morning_session = (
                first_timestamp
                < first_timestamp.normalize()
                + pd.Timedelta(morning_session_end)
            )
        if not has_morning_session:
            return pd.Series(np.nan, index=day_timestamps)

        day_typical_prices = typical_price.loc[day_timestamps]
        day_volumes = volume.loc[day_timestamps]

        if day_typical_prices.notna().any() and day_volumes.notna().any():
            return (
                day_typical_prices * day_volumes
            ).cumsum() / day_volumes.cumsum()

        return pd.Series(np.nan, index=day_timestamps)

    # Compute VWAP per calendar day and align the results with the original
    # index.
    return typical_price.groupby(pd.Grouper(freq="D")).transform(calc_vwap)


def stochastics(high, low, close, k, d, smooth_k):
    """Calculate the stochastic oscillator values for a given dataset."""
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()

    stochastics = 100 * (close - lowest_low)
    diff = highest_high - lowest_low
    if diff.eq(0).any().any():
        diff += sys.float_info.epsilon

    stochastics /= diff

    stochastics_k = stochastics.rolling(smooth_k).mean()
    stochastics_d = stochastics_k.rolling(d).mean()

    return pd.DataFrame({"k": stochastics_k, "d": stochastics_d})
