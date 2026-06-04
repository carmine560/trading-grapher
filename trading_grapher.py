#!/usr/bin/env python3

"""Visualize trade data using charts and technical indicators."""

import argparse
import configparser
import importlib
import os
import sys
import tempfile
from datetime import datetime
from enum import Enum

import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance

import indicators
from core_utilities import data_utilities, file_utilities
from core_utilities.config_common import ConfigError
from core_utilities.config_diff import check_config_changes
from core_utilities.config_io import read_config, write_file_atomically
from core_utilities.config_prompt import modify_section
from core_utilities.config_validation import evaluate_value
from core_utilities.errors import CoreUtilitiesError, MarketDataError

ISO_DATE_FORMAT = "%Y-%m-%d"
TRADING_JOURNAL_COLUMNS = [
    "optional_number",
    "entry_date",
    "entry_time",
    "symbol",
    "order_specification",
    "entry_price",
    "optional_tactic",
    "optional_entry_reason",
    "exit_time",
    "exit_price",
    "optional_exit_reason",
    "optional_percentage_change",
] + [f"optional_note_{index}" for index in range(1, 11)]

DATETIME = "Datetime"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
VOLUME = "Volume"

OHLCV_COLUMNS = (OPEN, HIGH, LOW, CLOSE, VOLUME)

DATE_FORMAT = "%b %-d"
TIME_FORMAT = "%-H:%M"

INTERVALS = {
    "1m": {"freq": "1min", "minutes": 1},
    "2m": {"freq": "2min", "minutes": 2},
    "3m": {"freq": "3min", "minutes": 3},
    "5m": {"freq": "5min", "minutes": 5},
}
MARKET_DATA_PERIOD_IN_DAYS = 7
HALF_BAR_WIDTH = 0.5


class RefreshDecision(Enum):
    """Describe why market data should or should not be fetched."""

    OUT_OF_RANGE = "out_of_range"
    COOLDOWN = "cooldown"
    CACHE_FRESH = "cache_fresh"
    NEED_REFRESH = "need_refresh"


# Entry Point


def main():
    """Parse trade data, save market data, plot charts, and check charts."""
    try:
        args = get_arguments()
        config_path = file_utilities.get_config_path(__file__)
        config = configure(config_path)
        trading_path = (
            args.f[0] if args.f else config["General"]["trading_path"]
        )
        trading_sheet = config["General"]["trading_sheet"]

        if file_utilities.create_launchers_exit(args, __file__):
            return
        configure_exit(args, config_path, trading_path, trading_sheet)

        try:
            dates = [
                pd.Timestamp(datetime.strptime(date, ISO_DATE_FORMAT))
                for date in args.dates
            ]
        except ValueError as e:
            raise MarketDataError(
                f"Invalid date. Expected format {ISO_DATE_FORMAT}: {e}"
            ) from e

        trading_journal = read_trading_journal(trading_path, trading_sheet)
        entry_date_column = config["Trading Journal"]["entry_date"]
        if entry_date_column not in trading_journal.columns:
            raise MarketDataError(
                f"Trading journal is missing entry_date column "
                f"'{entry_date_column}'."
            )
        raw_entry_dates = trading_journal[entry_date_column]
        normalized_entry_dates = pd.to_datetime(
            raw_entry_dates, errors="coerce"
        )
        invalid_entry_dates = (
            normalized_entry_dates.isna() & raw_entry_dates.notna()
        )
        if invalid_entry_dates.any():
            invalid_index = invalid_entry_dates[invalid_entry_dates].index[0]
            raise MarketDataError(
                f"Trade row {invalid_index} has invalid entry_date: "
                f"{raw_entry_dates.loc[invalid_index]}"
            )
        trading_journal = trading_journal.copy()
        trading_journal[entry_date_column] = (
            normalized_entry_dates.dt.normalize()
        )
        charts_directory = (
            args.d[0] if args.d else config["General"]["charts_directory"]
        )
        file_utilities.check_directory(charts_directory)
        interval = validate_interval(
            args.i[0] if args.i else config["Chart"]["interval"]
        )
        has_plotted = False

        # Run the directory discrepancy check later if any chart was plotted.
        for date in dates:
            has_plotted |= plot_trades_for_date(
                config,
                trading_journal,
                date,
                charts_directory,
                interval,
            )

        if has_plotted:
            report_chart_directory_discrepancies(
                config,
                trading_journal,
                charts_directory,
            )
    # MarketDataError already inherits from CoreUtilitiesError.
    except (CoreUtilitiesError, ConfigError) as e:
        print(e)
        sys.exit(1)


def plot_trades_for_date(
    config, trading_journal, date, charts_directory, interval
):
    """Plot all trades for a single journal date."""
    trades = trading_journal.loc[
        trading_journal[config["Trading Journal"]["entry_date"]] == date
    ]
    if trades.empty:
        return False

    first_index = next(trades.iterrows())[0]
    for index, trade in trades.iterrows():
        trade_data = {
            column: trade.get(config["Trading Journal"][column])
            for column in TRADING_JOURNAL_COLUMNS
        }
        _validate_trade_data(trade_data, index)

        if pd.isna(trade_data["optional_number"]):
            trade_data["optional_number"] = index - first_index + 1

        trade_data["entry_date"] = trade_data["entry_date"].tz_localize(
            config["Market Data"]["timezone"]
        )

        trade_data["order_specification"] = trade_data[
            "order_specification"
        ].lower()
        session = (
            "am"
            if pd.Timedelta(str(trade_data["exit_time"]))
            < pd.Timedelta(hours=12)
            else "pm"
        )
        market_data_path = os.path.join(
            charts_directory,
            f"{trade_data['entry_date'].strftime(ISO_DATE_FORMAT)}"
            f"-{session}-{trade_data['symbol']}.csv",
        )

        style_name = "fluorite"
        for option in config.options("Styles"):
            key, value = evaluate_value(config["Styles"][option])
            field_value = trade_data.get(key)
            if pd.isna(field_value):
                continue
            if str(value) in str(field_value):
                style_name = option
                break

        try:
            style = importlib.import_module(f"styles.{style_name}").style
        except ModuleNotFoundError as e:
            raise MarketDataError(
                f"Unable to load style '{style_name}': {e}"
            ) from e

        save_market_data(config, trade_data, market_data_path)
        plot_charts(
            config,
            trade_data,
            market_data_path,
            charts_directory,
            interval,
            style,
        )

    return True


# CLI and Configuration


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "dates",
        nargs="*",
        default=[pd.Timestamp.now().strftime(ISO_DATE_FORMAT)],
        help="specify dates in the format %%Y-%%m-%%d [default: today]",
    )
    parser.add_argument(
        "-f",
        nargs=1,
        help="specify the file path to the trading journal spreadsheet",
        metavar="FILE",
    )
    parser.add_argument(
        "-d",
        nargs=1,
        help="specify the directory path"
        " for storing historical data and charts",
        metavar="DIRECTORY",
    )
    parser.add_argument(
        "-i",
        nargs=1,
        help="specify the bar interval for chart rendering",
        metavar="INTERVAL",
    )

    file_utilities.add_launcher_options(group)
    group.add_argument(
        "-G", action="store_true", help="configure general options and exit"
    )
    group.add_argument(
        "-J",
        action="store_true",
        help="configure the columns of the trading journal and exit",
    )
    group.add_argument(
        "-I",
        action="store_true",
        help="configure the bar interval and exit",
    )
    group.add_argument(
        "-S",
        action="store_true",
        help="configure the styles based on the trade context and exit",
    )
    group.add_argument(
        "-C", action="store_true", help="check configuration changes and exit"
    )

    return parser.parse_args()


def configure(config_path, can_interpolate=True, can_override=True):
    """Get the configuration parser object with the set up configuration."""
    if can_interpolate:
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
    else:
        config = configparser.ConfigParser(interpolation=None)

    config["General"] = {
        "trading_path": os.path.join(
            os.path.expanduser("~"), "Documents/Trading", "Trading.ods"
        ),
        "trading_sheet": "Trading Journal",
        "charts_directory": os.path.join(
            os.path.expanduser("~"), "Documents/Trading"
        ),
    }
    config["Market Data"] = {
        "opening_time": "09:00:00",
        "has_midday_break": "True",
        "morning_session_end": "11:30:00",
        "afternoon_session_start": "12:30:00",
        "closing_time": "15:30:00",
        "delay": "20",
        "timezone": "Asia/Tokyo",
        "exchange_suffix": ".T",
    }

    config["Trading Journal"] = {
        "optional_number": "Number",
        "entry_date": "Entry date",
        "entry_time": "Entry time",
        "symbol": "Symbol",
        "order_specification": "Order specification",
        "entry_price": "Entry price",
        "optional_tactic": "Tactic",
        "optional_entry_reason": "Entry reason",
        "exit_time": "Exit time",
        "exit_price": "Exit price",
        "optional_exit_reason": "Exit reason",
        "optional_percentage_change": "Percentage change",
        "optional_chart_file": "Chart file",
    }
    for index in range(1, 11):
        config["Trading Journal"][f"optional_note_{index}"] = f"Note {index}"

    config["Chart"] = {
        "interval": "1m",
        "width": "1280",
        "height": "720",
        "scale_padding_top": "0.0",
        "scale_padding_right": "0.02",
        "scale_padding_bottom": "1.3",
        "scale_padding_left": "0.8",
    }
    config["Active Trading Hours"] = {
        "is_added": "True",
        "start_time": "${Market Data:opening_time}",
        "end_time": "${Market Data:closing_time}",
    }
    config["EMA"] = {
        "is_added": "True",
        "short_term_period": "5",
        "medium_term_period": "25",
        "long_term_period": "75",
    }
    config["VWAP"] = {
        "is_added": "True",
    }
    config["MACD"] = {
        "is_added": "True",
        "short_term_period": "12",
        "long_term_period": "26",
        "signal_period": "9",
    }
    config["Stochastics"] = {
        "is_added": "True",
        "k_period": "5",
        "d_period": "3",
        "smooth_k_period": "3",
    }
    config["Volume"] = {"is_added": "True", "quantile_threshold": "0.99"}
    config["Minor X-ticks"] = {"is_added": "True"}
    config["Tooltips"] = {"is_added": "True"}
    config["Text"] = {"is_added": "True", "default_y_offset_ratio": "-0.008"}
    config["Styles"] = {
        "amber": ("", ""),
        "ametrine": ("", ""),
        "fluorite": ("", ""),
        "opal": ("", ""),
    }

    if can_override:
        read_config(config, config_path)

    return config


def read_trading_journal(trading_path, trading_sheet):
    """Read the trading journal spreadsheet with input-specific context."""
    try:
        return pd.read_excel(trading_path, sheet_name=trading_sheet)
    except Exception as e:
        raise MarketDataError(
            f"Unable to read trading journal '{trading_path}' "
            f"sheet '{trading_sheet}': {e}"
        ) from e


def configure_exit(args, config_path, trading_path, trading_sheet):
    """Configure parameters based on command-line arguments and exit."""
    backup_parameters = {"number_of_backups": 8}
    if any((args.G, args.J, args.I, args.S)):
        config = configure(config_path, can_interpolate=False)
        for argument, (section, option, prompts, all_values) in {
            "G": ("General", None, None, None),
            "J": ("Trading Journal", None, {"value": "column"}, None),
            "I": (
                "Chart",
                "interval",
                {"value": "interval"},
                sorted(INTERVALS),
            ),
            "S": (
                "Styles",
                None,
                {"values": ("column", "value")},
                (["any"] + TRADING_JOURNAL_COLUMNS, None),
            ),
        }.items():
            if getattr(args, argument):
                modify_section(
                    config,
                    section,
                    config_path,
                    backup_parameters=backup_parameters,
                    option=option,
                    prompts=prompts,
                    all_values=(
                        tuple(
                            read_trading_journal(
                                trading_path, trading_sheet
                            ).columns
                        )
                        if argument == "J"
                        else all_values
                    ),
                )
                break

        sys.exit()
    if args.C:
        check_config_changes(
            configure(config_path, can_interpolate=False, can_override=False),
            config_path,
            excluded_sections=("Trading Journal",),
            backup_parameters=backup_parameters,
        )
        sys.exit()


def report_chart_directory_discrepancies(
    config, trading_journal, charts_directory
):
    """Report missing and unexpected chart files from the journal list."""
    chart_file_column = config["Trading Journal"]["optional_chart_file"]
    if chart_file_column not in trading_journal.columns:
        return

    chart_files = trading_journal[chart_file_column].dropna()
    chart_files = chart_files[
        chart_files.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
    ]
    discrepancies = file_utilities.compare_directory_list(
        charts_directory,
        r"\d{4}-\d{2}-\d{2}-\d{2}-\w+\.png",
        chart_files,
    )
    for path in discrepancies["unexpected_files"]:
        print(f"The {path} file is not in the list.")
    for path in discrepancies["missing_files"]:
        print(f"The {path} file does not exist in the directory.")


# Time and Interval Utilities


def _validate_trade_entry_date(trade_data, row_index):
    """Normalize and validate the trade entry date."""
    if pd.isna(trade_data["entry_date"]):
        raise MarketDataError(f"Trade row {row_index} is missing entry_date.")
    try:
        trade_data["entry_date"] = pd.Timestamp(trade_data["entry_date"])
    except (TypeError, ValueError) as e:
        raise MarketDataError(
            f"Trade row {row_index} has invalid entry_date: "
            f"{trade_data['entry_date']}"
        ) from e


def _validate_trade_time(trade_data, row_index, field_name):
    """Normalize and validate a trade time field."""
    if pd.isna(trade_data[field_name]):
        raise MarketDataError(
            f"Trade row {row_index} is missing {field_name}."
        )
    try:
        time_value = pd.Timedelta(str(trade_data[field_name]))
    except (TypeError, ValueError) as e:
        raise MarketDataError(
            f"Trade row {row_index} has invalid {field_name}: "
            f"{trade_data[field_name]}"
        ) from e
    trade_data[field_name] = (pd.Timestamp("1970-01-01") + time_value).time()


def _validate_trade_numeric_fields(trade_data, row_index):
    """Normalize and validate optional numeric trade fields."""
    for field in (
        "optional_number",
        "entry_price",
        "exit_price",
        "optional_percentage_change",
    ):
        value = trade_data.get(field)
        if pd.isna(value):
            continue
        try:
            trade_data[field] = pd.to_numeric(value)
        except (TypeError, ValueError) as e:
            raise MarketDataError(
                f"Trade row {row_index} has invalid {field}: {value}"
            ) from e


def _validate_trade_data(trade_data, row_index):
    """Validate required trade-row fields before transformation."""
    _validate_trade_entry_date(trade_data, row_index)

    if not trade_data["symbol"] or pd.isna(trade_data["symbol"]):
        raise MarketDataError(f"Trade row {row_index} is missing symbol.")

    if not isinstance(trade_data["order_specification"], str):
        raise MarketDataError(
            f"Trade row {row_index} has invalid order_specification."
        )

    if not trade_data["order_specification"].strip():
        raise MarketDataError(
            f"Trade row {row_index} has empty order_specification."
        )

    _validate_trade_numeric_fields(trade_data, row_index)
    _validate_trade_time(trade_data, row_index, "entry_time")
    _validate_trade_time(trade_data, row_index, "exit_time")


def validate_interval(interval):
    """Return 'interval' if it is supported, otherwise exit with an error."""
    if interval not in INTERVALS:
        print(
            f"Invalid interval '{interval}'."
            f" Supported values: {', '.join(sorted(INTERVALS))}"
        )
        sys.exit(1)
    return interval


def get_interval_minutes(interval):
    """Return the width of 'interval' in minutes (e.g., 5)."""
    return INTERVALS[interval]["minutes"]


def create_timestamp(date, time):
    """Create a pandas Timestamp by adding a time duration to a date."""
    if pd.isna(date):
        return pd.NaT
    else:
        return date + (
            pd.Timedelta(time)
            if isinstance(time, str)
            else pd.Timedelta(str(time))
        )


# Market Data Acquisition and Preparation


def _validate_symbol_data(symbol_data, trade_data):
    """Validate the raw market-data frame returned by the provider."""
    if symbol_data.empty:
        raise MarketDataError(
            f"No market data returned for {trade_data['symbol']}."
        )

    required_columns = set(OHLCV_COLUMNS)
    missing_columns = required_columns.difference(symbol_data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise MarketDataError(
            f"Market data for {trade_data['symbol']} is missing columns: "
            f"{missing_text}"
        )


def determine_market_data_refresh_decision(
    entry_date,
    last_bar_time,
    modified_time,
    now,
    delay_minutes,
    period_in_days,
):
    """Return the market-data refresh decision for the current cache state."""
    delta = now.normalize() - entry_date
    # Yahoo Finance does not support requests beyond the recent market-data
    # window for this trade date.
    if period_in_days <= delta.days:
        return RefreshDecision.OUT_OF_RANGE
    # Avoid requesting Yahoo Finance again within one minute of the last
    # cache update to reduce connection refusals from rapid retries.
    if now < modified_time + pd.Timedelta(minutes=1):
        return RefreshDecision.COOLDOWN
    # Do not refresh if the cached file already contains bars newer than the
    # provider delay window.
    if last_bar_time + pd.Timedelta(minutes=delay_minutes) < modified_time:
        return RefreshDecision.CACHE_FRESH
    return RefreshDecision.NEED_REFRESH


def _build_formalized_market_data(
    config, trade_data, symbol_data, interval, freq, bar_timedelta
):
    """Shape raw provider data into the cache layout for one trade."""
    _validate_symbol_data(symbol_data, trade_data)
    symbol_data[VOLUME] = pd.to_numeric(symbol_data[VOLUME], errors="coerce")

    volume_threshold = symbol_data[VOLUME].quantile(
        float(config["Volume"]["quantile_threshold"])
    )
    if pd.notna(volume_threshold):
        symbol_data[VOLUME] = symbol_data[VOLUME].clip(
            upper=int(volume_threshold)
        )
    symbol_data[VOLUME] = symbol_data[VOLUME].astype("Int64")

    previous = symbol_data[symbol_data.index < trade_data["entry_date"]]
    previous_with_data = previous.dropna()
    if not previous_with_data.empty:
        previous_date = pd.Timestamp.date(previous_with_data.tail(1).index[0])
        previous_date = pd.Timestamp(
            previous_date, tz=config["Market Data"]["timezone"]
        )

    morning = pd.Timedelta(str(trade_data["exit_time"])) < pd.Timedelta(
        hours=12
    )

    if morning and not previous_with_data.empty:
        start = create_timestamp(
            previous_date, config["Market Data"]["afternoon_session_start"]
        )
        end = create_timestamp(
            trade_data["entry_date"],
            config["Market Data"]["morning_session_end"],
        )
        end -= bar_timedelta
    else:
        start = create_timestamp(
            trade_data["entry_date"], config["Market Data"]["opening_time"]
        )
        end = create_timestamp(
            trade_data["entry_date"], config["Market Data"]["closing_time"]
        )
        end -= bar_timedelta

    formalized = pd.DataFrame(
        symbol_data,
        index=pd.date_range(start, end, freq=freq),
        columns=OHLCV_COLUMNS,
    )
    formalized.index.name = DATETIME
    formalized[VOLUME] = pd.to_numeric(
        formalized[VOLUME], errors="coerce"
    ).astype("Int64")

    if config["Market Data"].getboolean("has_midday_break"):
        if morning and not previous_with_data.empty:
            start = create_timestamp(
                previous_date, config["Market Data"]["closing_time"]
            )
            end = create_timestamp(
                trade_data["entry_date"],
                config["Market Data"]["opening_time"],
            )
            end -= bar_timedelta
            exclusion = pd.date_range(start=start, end=end, freq=freq)
            formalized = formalized.loc[~formalized.index.isin(exclusion)]
        else:
            start = create_timestamp(
                trade_data["entry_date"],
                config["Market Data"]["morning_session_end"],
            )
            end = create_timestamp(
                trade_data["entry_date"],
                config["Market Data"]["afternoon_session_start"],
            )
            end -= bar_timedelta
            exclusion = pd.date_range(start=start, end=end, freq=freq)
            formalized = formalized.loc[~formalized.index.isin(exclusion)]

    return formalized


def save_market_data(config, trade_data, market_data_path):
    """Save historical market data for a given symbol to a CSV file."""
    now = pd.Timestamp.now(tz=config["Market Data"]["timezone"])
    last_bar_time = modified_time = pd.Timestamp(
        0, tz=config["Market Data"]["timezone"]
    )
    if os.path.isfile(market_data_path):
        try:
            formalized = pd.read_csv(
                market_data_path, index_col=0, parse_dates=True
            )
            required_columns = set(OHLCV_COLUMNS)
            missing_columns = required_columns.difference(formalized.columns)
            if missing_columns:
                missing_text = ", ".join(sorted(missing_columns))
                raise MarketDataError(
                    f"Cached market data from {market_data_path} "
                    f"is missing columns: {missing_text}"
                )
            last_bar_time = formalized.tail(1).index[0]
        except MarketDataError:
            raise
        except Exception as e:
            raise MarketDataError(
                f"Unable to read cached market data from "
                f"{market_data_path}: {e}"
            ) from e
        modified_time = pd.Timestamp(
            os.path.getmtime(market_data_path),
            tz=config["Market Data"]["timezone"],
            unit="s",
        )

    refresh_decision = determine_market_data_refresh_decision(
        trade_data["entry_date"],
        last_bar_time,
        modified_time,
        now,
        int(config["Market Data"]["delay"]),
        MARKET_DATA_PERIOD_IN_DAYS,
    )
    if refresh_decision is not RefreshDecision.NEED_REFRESH:
        if (
            refresh_decision is RefreshDecision.OUT_OF_RANGE
            and not os.path.isfile(market_data_path)
        ):
            raise MarketDataError(
                f"No cached market data exists for "
                f"{trade_data['symbol']} on "
                f"{trade_data['entry_date'].strftime(ISO_DATE_FORMAT)} "
                f"at {market_data_path}. Yahoo Finance 1m data is only "
                f"requested within the last {MARKET_DATA_PERIOD_IN_DAYS} "
                "days."
            )
        return
    interval = "1m"
    freq = INTERVALS[interval]["freq"]
    bar_timedelta = pd.Timedelta(minutes=get_interval_minutes(interval))

    try:
        symbol_data = yfinance.Ticker(
            f"{trade_data['symbol']}"
            f"{config['Market Data']['exchange_suffix']}"
        ).history(
            interval=interval,
            period=f"{MARKET_DATA_PERIOD_IN_DAYS}d",
        )
    except Exception as e:
        raise MarketDataError(
            f"Unable to fetch market data for {trade_data['symbol']}: {e}"
        ) from e

    formalized = _build_formalized_market_data(
        config, trade_data, symbol_data, interval, freq, bar_timedelta
    )

    if not formalized[list(OHLCV_COLUMNS)].notna().all(axis=1).any():
        raise MarketDataError(
            f"Market data for {trade_data['symbol']} has no usable "
            "OHLC rows after session filtering."
        )

    try:
        write_file_atomically(
            market_data_path,
            "w",
            lambda f: formalized.to_csv(f),
            newline="",
        )
    except Exception as e:
        raise MarketDataError(
            f"Unable to write market data to {market_data_path}: {e}"
        ) from e


def resample_ohlcv(config, df, interval):
    """Resample 1-minute OHLCV to N-minute bars within exchange hours."""
    if get_interval_minutes(interval) == 1:
        return df

    resampled = df.resample(INTERVALS[interval]["freq"]).agg(
        {
            OPEN: "first",
            HIGH: "max",
            LOW: "min",
            CLOSE: "last",
            VOLUME: "sum",
        }
    )
    resampled = resampled.loc[
        resampled.index.normalize().isin(df.index.normalize().unique())
    ]

    index = resampled.index
    trading_mask = np.zeros(len(index), dtype=bool)
    if config["Market Data"].getboolean("has_midday_break"):
        trading_mask[
            index.indexer_between_time(
                config["Market Data"]["opening_time"],
                config["Market Data"]["morning_session_end"],
                include_end=False,
            )
        ] = True
        trading_mask[
            index.indexer_between_time(
                config["Market Data"]["afternoon_session_start"],
                config["Market Data"]["closing_time"],
                include_end=False,
            )
        ] = True
    else:
        trading_mask[
            index.indexer_between_time(
                config["Market Data"]["opening_time"],
                config["Market Data"]["closing_time"],
                include_end=False,
            )
        ] = True
    resampled = resampled.loc[trading_mask]

    return resampled


# Plot Chart Orchestration


def _calculate_trade_result(trade_data):
    """Return the profit or loss for a trade based on order type and prices."""
    if pd.isna(trade_data["entry_price"]) or pd.isna(trade_data["exit_price"]):
        return 0

    if "long" in trade_data["order_specification"]:
        return trade_data["exit_price"] - trade_data["entry_price"]
    if "short" in trade_data["order_specification"]:
        return trade_data["entry_price"] - trade_data["exit_price"]
    return 0


def _calculate_percentage_change(trade_data, result):
    """Return the percentage change for a trade."""
    if not pd.isna(trade_data["optional_percentage_change"]):
        return trade_data["optional_percentage_change"]
    if pd.isna(trade_data["entry_price"]) or trade_data["entry_price"] == 0:
        return 0.0
    return 100 * result / trade_data["entry_price"]


def _prepare_parameters(config, formalized, trade_data, result, style):
    """Prepare timestamps, prices, and colors for entry and exit points."""
    timestamps = {
        "start": create_timestamp(
            trade_data["entry_date"],
            config["Active Trading Hours"]["start_time"],
        ),
        "end": create_timestamp(
            trade_data["entry_date"],
            config["Active Trading Hours"]["end_time"],
        ),
        "entry": None,
        "exit": None,
    }
    if isinstance(timestamps["end"], pd.Timestamp):
        timestamps["end"] = min(formalized.tail(1).index[0], timestamps["end"])

    prices = {"closing": 0.0, "opening": 0.0, "entry": 0.0, "exit": 0.0}
    colors = {
        "closing": style["rc"]["axes.edgecolor"],
        "opening": style["rc"]["axes.edgecolor"],
        "entry": style["custom_style"]["neutral_color"],
        "exit": style["custom_style"]["neutral_color"],
    }

    previous = formalized[formalized.index < trade_data["entry_date"]]
    current = formalized[trade_data["entry_date"] <= formalized.index]
    previous = previous.dropna(subset=list(OHLCV_COLUMNS))
    current = current.dropna(subset=list(OHLCV_COLUMNS))
    if not previous.empty:
        prices["closing"] = previous.tail(1)[CLOSE].iloc[0]
    if not current.empty:
        prices["opening"] = current.head(1)[OPEN].iloc[0]

    # nan is not recognized as False in a boolean context.
    if not pd.isna(trade_data["entry_time"]) and not pd.isna(
        trade_data["entry_price"]
    ):
        timestamps["entry"] = create_timestamp(
            trade_data["entry_date"], trade_data["entry_time"]
        )
        prices["entry"] = trade_data["entry_price"]

    if not pd.isna(trade_data["exit_time"]) and not pd.isna(
        trade_data["exit_price"]
    ):
        timestamps["exit"] = create_timestamp(
            trade_data["entry_date"], trade_data["exit_time"]
        )
        prices["exit"] = trade_data["exit_price"]
        if result > 0:
            colors["exit"] = style["custom_style"]["profit_color"]
        elif result < 0:
            colors["exit"] = style["custom_style"]["loss_color"]

    return (timestamps, prices, colors)


def _add_indicators(config, formalized, addplot, style, panel):
    """Add enabled technical indicators to the plot and manage panels."""
    if config["EMA"].getboolean("is_added"):
        add_emas(config, formalized, addplot, style)

    if config["VWAP"].getboolean("is_added"):
        add_vwap(config, formalized, addplot, style)

    if config["MACD"].getboolean("is_added"):
        panel = add_macd(config, formalized, panel, addplot, style)

    stochastics_panel = None
    if config["Stochastics"].getboolean("is_added"):
        previous_panel = panel
        panel = add_stochastics(config, formalized, panel, addplot, style)
        if panel != previous_panel:
            stochastics_panel = panel

    if config["Volume"].getboolean("is_added"):
        panel += 1

    return panel, stochastics_panel


def _add_all_tooltips(
    config,
    axlist,
    formalized,
    trade_data,
    prices,
    timestamps,
    result,
    percentage_change,
    style,
    colors,
):
    """Add all enabled tooltips for prices, entry, and exit to the chart."""
    if not config["Tooltips"].getboolean("is_added"):
        return

    if (
        prices["closing"]
        and prices["opening"] != trade_data["entry_price"]
        and prices["opening"] != trade_data["exit_price"]
    ):
        delta = prices["opening"] - prices["closing"]
        add_tooltips(
            axlist,
            prices["opening"],
            f"{delta:.1f}, {100 * delta / prices['closing']:.2f}%",
            style["custom_style"]["tooltip_color"],
            colors["opening"],
            style["custom_style"]["tooltip_bbox_alpha"],
        )

    if not pd.isna(trade_data["entry_price"]):
        acronym = data_utilities.create_acronym(
            trade_data["optional_entry_reason"]
        )
        add_tooltips(
            axlist,
            trade_data["entry_price"],
            acronym or "",
            style["custom_style"]["tooltip_color"],
            colors["entry"],
            style["custom_style"]["tooltip_bbox_alpha"],
            formalized=formalized,
            timestamp=timestamps["entry"],
        )

    if not pd.isna(trade_data["exit_price"]):
        acronym = data_utilities.create_acronym(
            trade_data["optional_exit_reason"]
        )
        add_tooltips(
            axlist,
            trade_data["exit_price"],
            f"{f'{acronym}, ' if acronym else ''}"
            f"{result:.1f}, {percentage_change:.2f}%",
            style["custom_style"]["tooltip_color"],
            colors["exit"],
            style["custom_style"]["tooltip_bbox_alpha"],
            formalized=formalized,
            timestamp=timestamps["exit"],
        )


def plot_charts(
    config, trade_data, market_data_path, charts_directory, interval, style
):
    """Plot trading charts with entry and exit points, and indicators."""
    try:
        formalized = pd.read_csv(
            market_data_path, index_col=0, parse_dates=True
        )
    except Exception as e:
        raise MarketDataError(
            f"Unable to read market data from {market_data_path}: {e}"
        ) from e

    formalized = resample_ohlcv(config, formalized, interval)
    if not formalized[list(OHLCV_COLUMNS)].notna().all(axis=1).any():
        raise MarketDataError(
            f"Market data from {market_data_path} has no usable "
            "OHLC rows after resampling."
        )

    try:
        result = _calculate_trade_result(trade_data)
        percentage_change = _calculate_percentage_change(trade_data, result)

        timestamps, prices, colors = _prepare_parameters(
            config, formalized, trade_data, result, style
        )

        addplot = []
        panel = 0
        panel, stochastics_panel = _add_indicators(
            config, formalized, addplot, style, panel
        )

        fig, axlist = mpf.plot(
            formalized,
            addplot=addplot,
            closefig=True,
            datetime_format=f"{DATE_FORMAT}, {TIME_FORMAT}",
            figsize=(
                int(config["Chart"]["width"]) / 100,
                int(config["Chart"]["height"]) / 100,
            ),
            fill_between=dict(
                alpha=style["custom_style"]["filled_area_alpha"],
                color=colors["exit"],
                y1=trade_data["entry_price"],
                y2=trade_data["exit_price"],
                zorder=1,
            ),
            hlines=dict(
                alpha=style["custom_style"]["line_alpha"],
                colors=list(colors.values()),
                hlines=list(prices.values()),
                linestyle=[
                    style["custom_style"]["closing_line"],
                    style["custom_style"]["opening_line"],
                    style["custom_style"]["entry_line"],
                    style["custom_style"]["exit_line"],
                ],
                linewidths=1,
            ),
            returnfig=True,
            scale_padding={
                "top": float(config["Chart"]["scale_padding_top"]),
                "right": float(config["Chart"]["scale_padding_right"]),
                "bottom": float(config["Chart"]["scale_padding_bottom"]),
                "left": float(config["Chart"]["scale_padding_left"]),
            },
            scale_width_adjustment=dict(candle=1.5),
            style=style,
            tight_layout=True,
            type="candle",
            volume=config["Volume"].getboolean("is_added"),
            volume_panel=panel,
        )

        add_vertical_elements(
            formalized,
            timestamps,
            axlist,
            colors,
            style,
            config["Active Trading Hours"].getboolean("is_added"),
        )

        add_axis_ticks(config, axlist, style, interval, stochastics_panel)
        _add_all_tooltips(
            config,
            axlist,
            formalized,
            trade_data,
            prices,
            timestamps,
            result,
            percentage_change,
            style,
            colors,
        )
        _add_trade_text(config, axlist, trade_data, style, interval)
    except Exception as e:
        raise MarketDataError(
            f"Unable to render chart from {market_data_path}: {e}"
        ) from e

    chart_path = os.path.join(
        charts_directory,
        f"{trade_data['entry_date'].strftime(ISO_DATE_FORMAT)}"
        f"-{int(trade_data['optional_number']):02}"
        f"-{trade_data['symbol']}.png",
    )
    fd, temporary_chart_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(chart_path)}.",
        suffix=".tmp",
        dir=charts_directory,
    )
    os.close(fd)
    try:
        fig.savefig(temporary_chart_path, format="png")
        with open(temporary_chart_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary_chart_path, chart_path)
        try:
            directory_fd = os.open(
                charts_directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    except Exception as e:
        raise MarketDataError(
            f"Unable to write chart to {chart_path}: {e}"
        ) from e
    finally:
        if os.path.exists(temporary_chart_path):
            os.remove(temporary_chart_path)


# Low-Level Plotting Primitives


def get_x(index, timestamp, method="ffill"):
    """Map a real timestamp to a bar index for plotting."""
    if index is None or len(index) == 0:
        return None
    if timestamp is None or pd.isna(timestamp):
        return None
    position = index.get_indexer([timestamp], method=method)[0]
    return None if position == -1 else position


def add_emas(config, formalized, addplot, style):
    """Add exponential moving average plots to the existing plots."""
    periods_and_colors = [
        (config["EMA"]["short_term_period"], style["mavcolors"][0]),
        (config["EMA"]["medium_term_period"], style["mavcolors"][1]),
        (config["EMA"]["long_term_period"], style["mavcolors"][2]),
    ]
    for period, color in periods_and_colors:
        series = indicators.ema(formalized[CLOSE], int(period))
        if series.notna().any():
            addplot.append(mpf.make_addplot(series, color=color, width=0.8))


def add_vwap(config, formalized, addplot, style):
    """Add volume-weighted average price plot to the existing plots."""
    series = indicators.vwap(
        formalized[HIGH],
        formalized[LOW],
        formalized[CLOSE],
        formalized[VOLUME],
        morning_session_end=(
            config["Market Data"]["morning_session_end"]
            if config["Market Data"].getboolean("has_midday_break")
            else None
        ),
    )
    color = style["mavcolors"][3]

    if series.notna().any():
        addplot.append(mpf.make_addplot(series, color=color, width=0.8))


def add_macd(config, formalized, panel, addplot, style, ma="ema"):
    """Add moving average convergence divergence plots to the given panel."""
    if ma == "ema":
        macd = indicators.ema(
            formalized[CLOSE], int(config["MACD"]["short_term_period"])
        ) - indicators.ema(
            formalized[CLOSE], int(config["MACD"]["long_term_period"])
        )
        ylabel = "MACD"
    elif ma == "tema":
        macd = indicators.tema(
            formalized[CLOSE], int(config["MACD"]["short_term_period"])
        ) - indicators.tema(
            formalized[CLOSE], int(config["MACD"]["long_term_period"])
        )
        ylabel = "MACD TEMA"

    signal = macd.ewm(span=int(config["MACD"]["signal_period"])).mean()
    histogram = macd - signal

    # Skip if 'macd' (and related series) is all-NaN; mplfinance crashes
    # otherwise.
    if not macd.notna().any():
        return panel

    panel += 1

    addplot.extend(
        [
            mpf.make_addplot(
                macd,
                color=style["mavcolors"][0],
                panel=panel,
                width=0.8,
                ylabel=ylabel,
            ),
            mpf.make_addplot(
                signal,
                color=style["mavcolors"][1],
                panel=panel,
                secondary_y=False,
                width=0.8,
            ),
            mpf.make_addplot(
                histogram,
                color=[
                    (
                        style["mavcolors"][2]
                        if value >= 0
                        else style["mavcolors"][3]
                    )
                    for value in histogram
                ],
                panel=panel,
                secondary_y=False,
                type="bar",
                width=1.0,
            ),
        ]
    )

    return panel


def add_stochastics(config, formalized, panel, addplot, style):
    """Add stochastic oscillator plots to the given panel."""
    df = indicators.stochastics(
        formalized[HIGH],
        formalized[LOW],
        formalized[CLOSE],
        k=int(config["Stochastics"]["k_period"]),
        d=int(config["Stochastics"]["d_period"]),
        smooth_k=int(config["Stochastics"]["smooth_k_period"]),
    )

    # Initialize 'k' and 'd' with NaN, then fill where data exists.
    formalized["k"] = np.nan
    formalized["d"] = np.nan
    formalized.loc[df.index, "k"] = df["k"]
    formalized.loc[df.index, "d"] = df["d"]

    # Skip if both 'k' and 'd' are all-NaN; mplfinance crashes otherwise.
    k_has_data = formalized["k"].notna().any()
    d_has_data = formalized["d"].notna().any()
    if not k_has_data and not d_has_data:
        return panel

    if not k_has_data:
        formalized["k"] = 50.0
    if not d_has_data:
        formalized["d"] = 50.0

    panel += 1

    addplot.extend(
        [
            mpf.make_addplot(
                formalized["k"],
                color=style["mavcolors"][0],
                panel=panel,
                width=0.8,
                ylabel="Stochastics",
            ),
            mpf.make_addplot(
                formalized["d"],
                color=style["mavcolors"][1],
                panel=panel,
                secondary_y=False,
                width=0.8,
            ),
        ]
    )

    return panel


def add_minor_xticks(axlist, minor_grid_alpha, minor_tick_step):
    """Add minor x-ticks and their grid between panels."""
    axlist[0].set_xticks(
        np.arange(*axlist[0].get_xlim(), minor_tick_step), minor=True
    )
    for index, _ in enumerate(axlist):
        if (index % 2) == 0:
            axlist[index].grid(which="minor", alpha=minor_grid_alpha)


def add_axis_ticks(config, axlist, style, interval, stochastics_panel):
    """Add chart x-axis ticks and stochastic panel y-axis reference ticks."""
    minutes = get_interval_minutes(interval)
    major_tick_step = 30 / minutes
    minor_tick_step = 10 / minutes
    axlist[0].set_xticks(np.arange(*axlist[0].get_xlim(), major_tick_step))
    if config["Minor X-ticks"].getboolean("is_added"):
        add_minor_xticks(
            axlist,
            style["custom_style"]["minor_grid_alpha"],
            minor_tick_step,
        )

    if stochastics_panel is not None:
        axlist[2 * stochastics_panel].set_yticks([20.0, 50.0, 80.0])


def add_vertical_elements(
    formalized,
    timestamps,
    axlist,
    colors,
    style,
    is_active_trading_hours_added,
):
    """Add vertical elements between panels at the specified timestamps."""
    start_x = end_x = entry_x = exit_x = None
    if timestamps["start"] and timestamps["end"]:
        start_x = get_x(formalized.index, timestamps["start"])
        end_x = get_x(formalized.index, timestamps["end"])
    if timestamps["entry"]:
        entry_x = get_x(formalized.index, timestamps["entry"])
    if timestamps["exit"]:
        exit_x = get_x(formalized.index, timestamps["exit"])

    for index, _ in enumerate(axlist):
        if (index % 2) == 0:
            if (
                is_active_trading_hours_added
                and start_x is not None
                and end_x is not None
            ):
                # Force a redraw of the y-limits to ensure all plot
                # elements are taken into account.
                axlist[index].set_ylim(*axlist[index].get_ylim())
                axlist[index].fill_betweenx(
                    axlist[index].get_ylim(),
                    start_x - HALF_BAR_WIDTH,
                    end_x + HALF_BAR_WIDTH,
                    facecolor=style["custom_style"][
                        "active_trading_hours_color"
                    ],
                    zorder=0,
                )
            if entry_x is not None:
                axlist[index].axvline(
                    alpha=style["custom_style"]["line_alpha"],
                    color=colors["entry"],
                    linestyle=style["custom_style"]["entry_line"],
                    linewidth=1,
                    x=entry_x,
                )
            if exit_x is not None:
                axlist[index].axvline(
                    alpha=style["custom_style"]["line_alpha"],
                    color=colors["exit"],
                    linestyle=style["custom_style"]["exit_line"],
                    linewidth=1,
                    x=exit_x,
                )


def add_tooltips(
    axlist,
    price,
    string,
    color,
    bbox_color,
    bbox_alpha,
    formalized=None,
    timestamp=None,
):
    """Add tooltips to the specified axes."""
    axlist[0].text(
        0.0 - HALF_BAR_WIDTH,
        price,
        string,
        c=color,
        ha="right",
        size="small",
        va="center",
        bbox=dict(
            alpha=bbox_alpha,
            boxstyle="round, pad=0.2",
            ec="none",
            fc=bbox_color,
        ),
    )

    x = get_x(formalized.index if formalized is not None else None, timestamp)
    if x is not None:
        last_primary_axes = len(axlist) - 2
        bottom, _ = axlist[last_primary_axes].get_ylim()

        axlist[last_primary_axes].text(
            x,
            bottom,
            timestamp.strftime(TIME_FORMAT),
            c=color,
            ha="center",
            size="small",
            va="top",
            bbox=dict(
                alpha=bbox_alpha,
                boxstyle="round, pad=0.2",
                ec="none",
                fc=bbox_color,
            ),
        )


def add_text(
    axlist,
    default_y_offset_ratio,
    title,
    note_series,
    bbox_color,
    bbox_alpha,
    interval,
):
    """Add a title and notes to the last primary axes."""
    # Use the last panel to prevent other panels from overwriting the
    # text.
    last_primary_axes = len(axlist) - 2
    bottom, top = axlist[last_primary_axes].get_ylim()
    height = top - bottom

    x_offset = 1.0 / get_interval_minutes(interval) - HALF_BAR_WIDTH
    panel_offset_factors = {0: 0, 2: 2.5 * height, 4: height, 6: 2 * height}
    panel_offset_factor = panel_offset_factors.get(last_primary_axes)
    y_offset_ratios = {
        0: default_y_offset_ratio,
        2: 3.5 * default_y_offset_ratio,
        4: 4.5 * default_y_offset_ratio,
        6: 5.5 * default_y_offset_ratio,
    }
    y_offset_ratio = y_offset_ratios.get(last_primary_axes)
    y = top + panel_offset_factor + y_offset_ratio * height

    axlist[last_primary_axes].text(
        x_offset,
        y,
        title,
        va="top",
        weight="bold",
        bbox=dict(
            alpha=bbox_alpha,
            boxstyle="square, pad=0.0",
            ec="none",
            fc=bbox_color,
        ),
    )

    notes = (
        "\n\n"
        + "\n".join(
            f"{index + 1}. {note}" for index, note in enumerate(note_series)
        )
        if not note_series.empty
        else ""
    )

    if notes:
        axlist[last_primary_axes].text(
            x_offset,
            y,
            notes,
            va="top",
            zorder=1,
            bbox=dict(
                alpha=bbox_alpha,
                boxstyle="square, pad=0.0",
                ec="none",
                fc=bbox_color,
            ),
        )


def _add_trade_text(config, axlist, trade_data, style, interval):
    """Add the trade summary text block when enabled."""
    if not config["Text"].getboolean("is_added"):
        return

    tactic = trade_data["optional_tactic"]
    full_date_format = f"%a, {DATE_FORMAT}, ’%y,"
    notes = [
        trade_data[f"optional_note_{i}"]
        for i in range(1, 11)
        if trade_data[f"optional_note_{i}"]
    ]
    add_text(
        axlist,
        float(config["Text"]["default_y_offset_ratio"]),
        f"Trade {trade_data['optional_number']}"
        f" for {trade_data['symbol']}"
        f" using {trade_data['order_specification'].title()}"
        f"{f'—{tactic.title()}' if pd.notna(tactic) else ''}\n"
        f"on {trade_data['entry_date'].strftime(full_date_format)}"
        f" at {trade_data['entry_time'].strftime(TIME_FORMAT)}",
        pd.Series(notes).dropna(),
        style["facecolor"],
        style["custom_style"]["text_bbox_alpha"],
        interval,
    )


if __name__ == "__main__":
    main()
