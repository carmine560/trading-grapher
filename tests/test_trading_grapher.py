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

from core_utilities.errors import MarketDataError
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


def test_main_exits_with_market_data_error_message(monkeypatch, capsys):
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
        }
    )

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
    monkeypatch.setattr(
        tg,
        "save_market_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MarketDataError("market data refresh failed")
        ),
    )
    monkeypatch.setattr(tg, "plot_charts", lambda *args, **kwargs: None)
    real_import_module = tg.importlib.import_module

    def fake_import_module(name):
        if name == "styles.fluorite":
            return SimpleNamespace(style={"custom_style": {}, "rc": {}})
        return real_import_module(name)

    monkeypatch.setattr(tg.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit) as excinfo:
        tg.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "market data refresh failed" in captured.out


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (pd.NaT, "Trade row 0 is missing entry_date."),
        ("abc", "Trade row 0 has invalid entry_date: abc"),
    ],
)
def test_validate_trade_data_rejects_bad_entry_date(value, message):
    trade_data = {
        "entry_date": value,
        "symbol": "1234",
        "order_specification": "long",
        "exit_time": pd.Timestamp("2024-01-02 13:00:00").time(),
    }

    with pytest.raises(MarketDataError, match=message):
        tg._validate_trade_data(trade_data, 0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Symbol", "", "Trade row 0 is missing symbol."),
        (
            "Order specification",
            float("nan"),
            "Trade row 0 has invalid order_specification.",
        ),
        (
            "Order specification",
            "   ",
            "Trade row 0 has empty order_specification.",
        ),
        (
            "Exit time",
            "abc",
            "Trade row 0 has invalid exit_time: abc",
        ),
    ],
)
def test_main_validates_trade_rows_before_processing(
    monkeypatch,
    capsys,
    field,
    value,
    message,
):
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
        }
    )
    journal.loc[0, field] = value

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

    with pytest.raises(SystemExit) as excinfo:
        tg.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert message in captured.out


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


def test_save_market_data_raises_market_data_error_on_fetch_failure(
    monkeypatch,
):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    timezone = config["Market Data"]["timezone"]
    real_timestamp = pd.Timestamp

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            raise RuntimeError("boom")

    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())
    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    with pytest.raises(MarketDataError, match="Unable to fetch market data"):
        tg.save_market_data(config, trade_data, "/tmp/not-used.csv")


def test_save_market_data_rejects_empty_symbol_data(tmp_path, monkeypatch):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    timezone = config["Market Data"]["timezone"]
    market_data_path = tmp_path / "empty.csv"
    real_timestamp = pd.Timestamp

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            return pd.DataFrame()

    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())
    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    with pytest.raises(MarketDataError, match="No market data returned"):
        tg.save_market_data(config, trade_data, str(market_data_path))


def test_save_market_data_rejects_missing_ohlcv_columns(
    tmp_path,
    monkeypatch,
):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    timezone = config["Market Data"]["timezone"]
    market_data_path = tmp_path / "missing-volume.csv"
    real_timestamp = pd.Timestamp
    index = pd.date_range(
        "2024-01-02 09:00:00",
        periods=2,
        freq="min",
        tz=timezone,
    )

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            return pd.DataFrame(
                {
                    tg.OPEN: [100.0, 101.0],
                    tg.HIGH: [101.0, 102.0],
                    tg.LOW: [99.0, 100.0],
                    tg.CLOSE: [100.5, 101.5],
                },
                index=index,
            )

    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())
    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    with pytest.raises(MarketDataError, match="missing columns: Volume"):
        tg.save_market_data(config, trade_data, str(market_data_path))


def test_save_market_data_allows_sparse_ohlcv_rows(tmp_path, monkeypatch):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    timezone = config["Market Data"]["timezone"]
    market_data_path = tmp_path / "sparse.csv"
    index = pd.date_range(
        "2024-01-02 09:00:00",
        periods=2,
        freq="min",
        tz=timezone,
    )
    symbol_data = pd.DataFrame(
        {
            tg.OPEN: [float("nan"), float("nan")],
            tg.HIGH: [float("nan"), float("nan")],
            tg.LOW: [float("nan"), float("nan")],
            tg.CLOSE: [float("nan"), float("nan")],
            tg.VOLUME: [float("nan"), float("nan")],
        },
        index=index,
    )
    real_timestamp = pd.Timestamp

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            return symbol_data

    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())
    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    tg.save_market_data(config, trade_data, str(market_data_path))

    saved = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    assert saved.isna().any().any()


def test_save_market_data_normalizes_object_volume_values(
    tmp_path,
    monkeypatch,
):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    config["Volume"]["quantile_threshold"] = "1.0"
    timezone = config["Market Data"]["timezone"]
    market_data_path = tmp_path / "object-volume.csv"
    index = pd.date_range(
        "2024-01-02 09:00:00",
        periods=2,
        freq="min",
        tz=timezone,
    )
    symbol_data = pd.DataFrame(
        {
            tg.OPEN: [100.0, 101.0],
            tg.HIGH: [101.0, 102.0],
            tg.LOW: [99.0, 100.0],
            tg.CLOSE: [100.5, 101.5],
            tg.VOLUME: ["10", ""],
        },
        index=index,
    )
    real_timestamp = pd.Timestamp

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            return symbol_data

    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())
    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    tg.save_market_data(config, trade_data, str(market_data_path))

    saved = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    assert saved.iloc[0][tg.VOLUME] == 10
    assert pd.isna(saved.iloc[1][tg.VOLUME])


def test_save_market_data_handles_non_integer_volume_threshold(
    tmp_path,
    monkeypatch,
):
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    config["Volume"]["quantile_threshold"] = "0.75"
    timezone = config["Market Data"]["timezone"]
    market_data_path = tmp_path / "quantile-volume.csv"
    index = pd.date_range(
        "2024-01-02 09:00:00",
        periods=2,
        freq="min",
        tz=timezone,
    )
    symbol_data = pd.DataFrame(
        {
            tg.OPEN: [100.0, 101.0],
            tg.HIGH: [101.0, 102.0],
            tg.LOW: [99.0, 100.0],
            tg.CLOSE: [100.5, 101.5],
            tg.VOLUME: [1, 2],
        },
        index=index,
    )
    real_timestamp = pd.Timestamp

    class FakeTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp("2024-01-04 09:00:00", tz=tz)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, period):
            return symbol_data

    monkeypatch.setattr(tg.pd, "Timestamp", FakeTimestamp())
    monkeypatch.setattr(tg.yfinance, "Ticker", FakeTicker)

    trade_data = {
        "entry_date": real_timestamp("2024-01-02 00:00:00", tz=timezone),
        "exit_time": "13:00:00",
        "symbol": "1234",
    }

    tg.save_market_data(config, trade_data, str(market_data_path))

    saved = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    assert saved[tg.VOLUME].dropna().max() == 1


def test_plot_charts_raises_market_data_error_on_csv_failure():
    config = tg.configure("/tmp/not-used.ini", can_override=False)
    trade_data = {
        "optional_percentage_change": float("nan"),
        "entry_price": 100.0,
    }

    with pytest.raises(MarketDataError, match="Unable to read market data"):
        tg.plot_charts(
            config,
            trade_data,
            "/tmp/does-not-exist.csv",
            "/tmp",
            "1m",
            {},
        )


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
