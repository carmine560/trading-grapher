import sys
import types

import pandas as pd
import pytest


if "mplfinance" not in sys.modules:
    sys.modules["mplfinance"] = types.SimpleNamespace(
        make_addplot=lambda *args, **kwargs: None,
        plot=lambda *args, **kwargs: (None, []),
    )

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.SimpleNamespace(Ticker=None)

import trading_grapher as tg


def test_validate_interval_rejects_unknown_values():
    with pytest.raises(SystemExit) as excinfo:
        tg.validate_interval("10m")

    assert excinfo.value.code == 1


def test_resample_ohlcv_aggregates_and_drops_midday_break():
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    index = pd.date_range(
        "2024-01-02 09:00:00",
        "2024-01-02 12:34:00",
        freq="min",
    )
    values = pd.Series(range(len(index)), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            tg.OPEN: values,
            tg.HIGH: values + 1,
            tg.LOW: values - 1,
            tg.CLOSE: values + 0.5,
            tg.VOLUME: 1,
        },
        index=index,
    )

    result = tg.resample_ohlcv(config, frame, "5m")

    first_bar = result.loc[pd.Timestamp("2024-01-02 09:00:00")]
    assert first_bar[tg.OPEN] == 0.0
    assert first_bar[tg.HIGH] == 5.0
    assert first_bar[tg.LOW] == -1.0
    assert first_bar[tg.CLOSE] == 4.5
    assert first_bar[tg.VOLUME] == 5
    assert pd.Timestamp("2024-01-02 11:30:00") not in result.index
    assert pd.Timestamp("2024-01-02 12:25:00") not in result.index
    assert pd.Timestamp("2024-01-02 12:30:00") in result.index


@pytest.mark.parametrize(
    ("trade_data", "expected"),
    [
        (
            {
                "entry_price": 100.0,
                "exit_price": 110.0,
                "order_specification": "long",
            },
            10.0,
        ),
        (
            {
                "entry_price": 100.0,
                "exit_price": 90.0,
                "order_specification": "short",
            },
            10.0,
        ),
        (
            {
                "entry_price": float("nan"),
                "exit_price": 90.0,
                "order_specification": "long",
            },
            0,
        ),
    ],
)
def test_calculate_trade_result_handles_direction_and_missing_values(
    trade_data,
    expected,
):
    assert tg._calculate_trade_result(trade_data) == expected
