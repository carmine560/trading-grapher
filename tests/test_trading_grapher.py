import sys
import types
from types import SimpleNamespace

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


def test_main_fills_missing_trade_number_from_nan(monkeypatch):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    journal = pd.DataFrame(
        {
            "Number": [float("nan")],
            "Entry date": [pd.Timestamp("2024-01-02")],
            "Entry time": [pd.Timestamp("2024-01-02 09:00:00").time()],
            "Symbol": ["1234"],
            "Order specification": ["long"],
            "Entry price": [100.0],
            "Exit time": [pd.Timestamp("2024-01-02 13:00:00").time()],
            "Exit price": [101.0],
        }
    )
    captured_trade_data = {}

    monkeypatch.setattr(
        tg,
        "get_arguments",
        lambda: SimpleNamespace(
            f=None,
            d=None,
            i=None,
            dates=["2024-01-02"],
            G=False,
            J=False,
            I=False,
            S=False,
            C=False,
        ),
    )
    monkeypatch.setattr(
        tg.file_utilities,
        "get_config_path",
        lambda _: "/tmp/x",
    )
    monkeypatch.setattr(tg, "configure", lambda _: config)
    monkeypatch.setattr(
        tg.file_utilities,
        "create_launchers_exit",
        lambda args, script_path: None,
    )
    monkeypatch.setattr(
        tg,
        "configure_exit",
        lambda args, config_path, trading_path, trading_sheet: None,
    )
    monkeypatch.setattr(tg.pd, "read_excel", lambda *args, **kwargs: journal)
    monkeypatch.setattr(tg, "save_market_data", lambda *args, **kwargs: None)

    def fake_plot_charts(
        config,
        trade_data,
        market_data_path,
        charts_directory,
        interval,
        style,
    ):
        captured_trade_data.update(trade_data)

    monkeypatch.setattr(tg, "plot_charts", fake_plot_charts)
    real_import_module = tg.importlib.import_module

    def fake_import_module(name):
        if name == "styles.fluorite":
            return SimpleNamespace(style={"custom_style": {}, "rc": {}})
        return real_import_module(name)

    monkeypatch.setattr(tg.importlib, "import_module", fake_import_module)

    tg.main()

    assert captured_trade_data["optional_number"] == 1


def test_main_reports_chart_directory_discrepancies(monkeypatch, capsys):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    journal = pd.DataFrame(
        {
            "Entry date": [pd.Timestamp("2024-01-02")],
            "Entry time": [pd.Timestamp("2024-01-02 09:00:00").time()],
            "Symbol": ["1234"],
            "Order specification": ["long"],
            "Entry price": [100.0],
            "Exit time": [pd.Timestamp("2024-01-02 13:00:00").time()],
            "Exit price": [101.0],
            "Chart file": ["2024-01-02-01-1234.png"],
        }
    )
    discrepancies = {
        "unexpected_files": ["/charts/extra.png"],
        "missing_files": ["/charts/missing.png"],
    }

    monkeypatch.setattr(
        tg,
        "get_arguments",
        lambda: SimpleNamespace(
            f=None,
            d=None,
            i=None,
            dates=["2024-01-02"],
            G=False,
            J=False,
            I=False,
            S=False,
            C=False,
        ),
    )
    monkeypatch.setattr(
        tg.file_utilities,
        "get_config_path",
        lambda _: "/tmp/x",
    )
    monkeypatch.setattr(tg, "configure", lambda _: config)
    monkeypatch.setattr(
        tg.file_utilities,
        "create_launchers_exit",
        lambda args, script_path: None,
    )
    monkeypatch.setattr(
        tg,
        "configure_exit",
        lambda args, config_path, trading_path, trading_sheet: None,
    )
    monkeypatch.setattr(tg.pd, "read_excel", lambda *args, **kwargs: journal)
    monkeypatch.setattr(tg, "save_market_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(tg, "plot_charts", lambda *args, **kwargs: None)
    real_import_module = tg.importlib.import_module

    def fake_import_module(name):
        if name == "styles.fluorite":
            return SimpleNamespace(style={"custom_style": {}, "rc": {}})
        return real_import_module(name)

    monkeypatch.setattr(tg.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        tg.file_utilities,
        "compare_directory_list",
        lambda *args, **kwargs: discrepancies,
    )

    tg.main()

    captured = capsys.readouterr()
    assert "/charts/extra.png file is not in the list." in captured.out
    assert "/charts/missing.png file does not exist" in captured.out


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


def test_save_market_data_writes_session_filtered_csv(tmp_path, monkeypatch):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    config["Volume"]["quantile_threshold"] = "0.5"
    timezone = config["Market Data"]["timezone"]
    market_data_path = tmp_path / "2024-01-02-pm-1234.csv"
    index = pd.date_range(
        "2024-01-02 09:00:00",
        "2024-01-02 15:29:00",
        freq="min",
        tz=timezone,
    )
    symbol_data = pd.DataFrame(
        {
            tg.OPEN: 100.0,
            tg.HIGH: 101.0,
            tg.LOW: 99.0,
            tg.CLOSE: 100.5,
            tg.VOLUME: [10] * (len(index) - 1) + [10_000],
        },
        index=index,
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            assert interval == "1m"
            assert period == "5d"
            return symbol_data

    real_timestamp = pd.Timestamp

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    tg.save_market_data(config, trade_data, str(market_data_path))

    saved = pd.read_csv(market_data_path, index_col=0, parse_dates=True)

    assert saved.index[0] == real_timestamp("2024-01-02 09:00:00+09:00")
    assert saved.index[-1] == real_timestamp("2024-01-02 15:29:00+09:00")
    assert real_timestamp("2024-01-02 11:30:00+09:00") not in saved.index
    assert real_timestamp("2024-01-02 12:29:00+09:00") not in saved.index
    assert real_timestamp("2024-01-02 12:30:00+09:00") in saved.index
    assert saved[tg.VOLUME].max() == 10


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
