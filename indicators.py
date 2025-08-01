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
