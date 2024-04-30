#!/usr/bin/env python3

"""Visualize trading data using charts and technical indicators."""

import argparse
import configparser
import importlib
import os
import re
import sys

from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import mplfinance as mpf
import numpy as np
import pandas as pd

import configuration
import file_utilities


def main():
    """Parse trade dates, save market data, plot charts, and check charts."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-C', action='store_true',
        help='check configuration changes and exit')
    parser.add_argument('dates', nargs='*',
                        default=[pd.Timestamp.now().strftime('%Y-%m-%d')],
                        help='specify dates in the format %%Y-%%m-%%d')
    args = parser.parse_args()

    config_path = file_utilities.get_config_path(__file__)
    if args.C:
        default_config = configure(config_path, can_interpolate=False,
                                   can_override=False)
        configuration.check_config_changes(
            default_config, config_path,
            backup_parameters={'number_of_backups': 8})
        return
    else:
        config = configure(config_path)

    trading_journal = pd.read_excel(
        config['General']['trading_path'],
        sheet_name=config['General']['trading_sheet'])
    trading_journal_columns = ['number', 'symbol', 'trade_type', 'tactic',
                               'entry_date', 'entry_time', 'entry_price',
                               'entry_reason', 'exit_date', 'exit_time',
                               'exit_price', 'exit_reason', 'change']
    for i in range(1, 11):
        trading_journal_columns.append(f'optional_note_{i}')

    for date in pd.to_datetime(args.dates):
        trades = trading_journal.loc[
            trading_journal[config['Trading Journal']['entry_date']] == date]
        for _, trade in trades.iterrows():
            trade_data = {}
            for column in trading_journal_columns:
                trade_data[column] = trade.get(
                    config['Trading Journal'][column])

            save_market_data(config, trade_data)
            plot_chart(config, trade_data)

    # TODO: add if-statement
    check_charts(config, trading_journal[config['Trading Journal']['chart']])


def configure(config_path, can_interpolate=True, can_override=True):
    """Get the configuration parser object with the set up configuration."""
    if can_interpolate:
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    else:
        config = configparser.ConfigParser(interpolation=None)

    # TODO: add Indicators or MACD and Stochastics
    config['General'] = {
        # TODO: add option '2022-12-14'
        'trading_directory': os.path.join(os.path.expanduser('~'),
                                          'Documents/Trading'),
        'trading_path': os.path.join('${trading_directory}', 'Trading.ods'),
        'trading_sheet': 'Trading Journal',
        'style': 'fluorite'}    # TODO: add ametrine
    config['Market Data'] = {
        'time_zone': 'Asia/Tokyo'}
    config['Trading Journal'] = {
        'number': 'Number',
        'symbol': 'Symbol',
        'trade_type': 'Trade type',
        'tactic': 'Tactic',
        'entry_date': 'Entry date',
        'entry_time': 'Entry time',
        'entry_price': 'Entry price',
        'entry_reason': 'Entry reason',
        'exit_date': 'Exit date',
        'exit_time': 'Exit time',
        'exit_price': 'Exit price',
        'exit_reason': 'Exit reason',
        'change': 'Change',
        'chart': 'Chart'}
    for i in range(1, 11):
        config['Trading Journal'][f'optional_note_{i}'] = ''

    if can_override:
        configuration.read_config(config, config_path)
        configuration.write_config(config, config_path)  # TODO

    return config


def get_variables(config, symbol, entry_date, number):
    """Generate base string, market data path, and localize entry date."""
    base = f"{entry_date.strftime('%Y-%m-%d')}-{int(number):02}-{symbol}"
    market_data = os.path.join(
        config['General']['trading_directory'],
        f"{entry_date.strftime('%Y-%m-%d')}-00-{symbol}.csv")
    entry_date = entry_date.tz_localize(config['Market Data']['time_zone'])

    return base, market_data, entry_date


def save_market_data(config, trade_data):
    """Save historical market data for a given symbol to a CSV file."""
    _, market_data, entry_date = get_variables(config, trade_data['symbol'],
                                               trade_data['entry_date'],
                                               trade_data['number'])
    PERIOD_IN_DAYS = 7

    delta = (
        pd.Timestamp.now(tz=config['Market Data']['time_zone']).normalize()
        - entry_date)
    last = modified_time = pd.Timestamp(0,
                                        tz=config['Market Data']['time_zone'])

    if os.path.exists(market_data):
        formalized = pd.read_csv(market_data, index_col=0, parse_dates=True)
        last = formalized.tail(1).index[0]
        modified_time = pd.Timestamp(os.path.getmtime(market_data),
                                     tz=config['Market Data']['time_zone'],
                                     unit='s')

    if (PERIOD_IN_DAYS <= 1 + delta.days
        or last + pd.Timedelta(minutes=30) < modified_time
        or pd.Timestamp.now(tz=config['Market Data']['time_zone'])
        < modified_time + pd.Timedelta(minutes=1)):
        return
    else:
        my_share = share.Share(f"{trade_data['symbol']}.T")
        try:
            symbol_data = my_share.get_historical(
                share.PERIOD_TYPE_DAY, PERIOD_IN_DAYS,
                share.FREQUENCY_TYPE_MINUTE, 1)
        except YahooFinanceError as e:
            print(e.message)
            sys.exit(1)

        df = pd.DataFrame(symbol_data)
        df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert(
            config['Market Data']['time_zone'])
        # TODO: configure q
        q = df.volume.quantile(0.99)
        df['volume'] = df['volume'].mask(df['volume'] > q, q)

        previous = df[df.index < entry_date]
        if previous:
            previous_date = pd.Timestamp.date(
                previous.dropna().tail(1).index[0])
            previous_date = pd.Timestamp(previous_date,
                                         tz=config['Market Data']['time_zone'])

        morning = (pd.Timedelta(str(trade_data['exit_time']))
                   < pd.Timedelta(hours=12))
        if morning and previous:
            start = previous_date + pd.Timedelta(hours=12, minutes=30)
            end = entry_date + pd.Timedelta(hours=11, minutes=29)
        else:
            start = entry_date + pd.Timedelta(hours=9)
            end = entry_date + pd.Timedelta(hours=14, minutes=59)

        formalized = pd.DataFrame(columns=('open', 'high', 'low', 'close',
                                           'volume'),
                                  index=pd.date_range(start, end, freq='min'))
        formalized.index.name = 'timestamp'
        formalized = formalized.astype('float')
        formalized.update(df)

        if morning and previous:
            start = previous_date + pd.Timedelta(hours=15)
            end = entry_date + pd.Timedelta(hours=8, minutes=59)
            exclusion = pd.date_range(start=start, end=end, freq='min')
            formalized = formalized.loc[~formalized.index.isin(exclusion)]
        else:
            formalized = formalized.between_time('12:30:00', '11:29:00')

        if formalized.isna().values.all():
            print('values are missing')
            return

        formalized.to_csv(market_data)


def plot_chart(config, trade_data):
    """Plot a trading chart with entry and exit points, and indicators."""
    base, market_data, entry_date = get_variables(config, trade_data['symbol'],
                                                  trade_data['entry_date'],
                                                  trade_data['number'])

    if os.path.exists(market_data):
        formalized = pd.read_csv(market_data, index_col=0, parse_dates=True)
    else:
        print(market_data, 'does not exist')
        sys.exit(1)

    try:
        style = importlib.import_module(
            f"styles.{config['General']['style']}").style
    except ModuleNotFoundError as e:
        print(e)
        sys.exit(1)

    entry_timestamp = exit_timestamp = None
    entry_color = style['tg_neutral_color']
    addplot = hlines = colors = []

    previous = formalized[formalized.index < entry_date]
    current = formalized[entry_date <= formalized.index]
    previous_close = current_open = 0.0

    if previous.notnull().values.any():
        previous_close = previous.dropna().tail(1).close.iloc[0]
        current_open = current.dropna().head(1).open.iloc[0]
        hlines = [previous_close, current_open]
        colors = [style['rc']['axes.edgecolor'], style['rc']['axes.edgecolor']]

    if trade_data['trade_type'].lower() == 'long':
        marker = 'o'
    elif trade_data['trade_type'].lower() == 'short':
        marker = 'D'

    marker_alpha = 0.2

    if (not pd.isna(trade_data['entry_time'])
        and not pd.isna(trade_data['entry_price'])):
        formalized['entry_point'] = pd.Series(dtype='float')
        entry_timestamp = (entry_date
                           + pd.Timedelta(str(trade_data['entry_time'])))
        formalized.loc[entry_timestamp, 'entry_point'] = (
            trade_data['entry_price'])
        entry_apd = mpf.make_addplot(formalized.entry_point, type='scatter',
                                     markersize=100, marker=marker,
                                     color=entry_color, edgecolors='none',
                                     alpha=marker_alpha)
        addplot.append(entry_apd)
        hlines.append(trade_data['entry_price'])
        colors.append(entry_color)

    result = 0.0
    exit_color = entry_color
    if (not pd.isna(trade_data['exit_time'])
        and not pd.isna(trade_data['exit_price'])):
        if trade_data['trade_type'].lower() == 'long':
            result = trade_data['exit_price'] - trade_data['entry_price']
        elif trade_data['trade_type'].lower() == 'short':
            result = trade_data['entry_price'] - trade_data['exit_price']
        if result > 0:
            exit_color = style['tg_profit_color']
        elif result < 0:
            exit_color = style['tg_loss_color']

        formalized['exit_point'] = pd.Series(dtype='float')
        exit_date = trade_data['exit_date'].tz_localize(
            config['Market Data']['time_zone'])
        exit_timestamp = exit_date + pd.Timedelta(str(trade_data['exit_time']))
        formalized.loc[exit_timestamp, 'exit_point'] = trade_data['exit_price']
        exit_apd = mpf.make_addplot(formalized.exit_point, type='scatter',
                                    markersize=100, marker=marker,
                                    color=exit_color, edgecolors='none',
                                    alpha=marker_alpha)
        addplot.append(exit_apd)
        hlines.append(trade_data['exit_price'])
        colors.append(exit_color)

    marker_coordinate_alpha = 0.4

    if hlines and colors:
        hlines = dict(hlines=hlines, colors=colors, linestyle='dotted',
                      linewidths=1, alpha=marker_coordinate_alpha)

    add_ma(config, formalized, mpf, addplot, style)

    panel = 0
    panel = add_macd(config, formalized, panel, mpf, addplot, style)
    panel = stoch_panel = add_stochastics(config, formalized, panel, mpf,
                                          addplot, style)

    panel += 1
    # TODO: use fill
    fig, axlist = mpf.plot(formalized, type='candle', volume=True,
                           # TODO: modify width
                           tight_layout=True, figsize=(1152 / 100, 648 / 100),
                           style=style,
                           scale_padding={'top': 0, 'right': 0.05,
                                          'bottom': 1.5},
                           scale_width_adjustment=dict(candle=1.5),
                           hlines=hlines, addplot=addplot, returnfig=True,
                           closefig=True, volume_panel=panel)

    left, right = axlist[0].get_xlim()
    axlist[0].set_xticks(np.arange(left, right, 30))
    axlist[0].set_xticks(np.arange(left, right, 10), minor=True)
    axlist[2 * stoch_panel].set_yticks([20.0, 50.0, 80.0])

    for index, _ in enumerate(axlist):
        if (index % 2) == 0:
            axlist[index].grid(which='minor', alpha=0.2)
            if entry_timestamp:
                axlist[index].axvline(
                    x=formalized.index.get_loc(entry_timestamp),
                    color=entry_color, linestyle='dotted', linewidth=1,
                    alpha=marker_coordinate_alpha)
            if exit_timestamp:
                axlist[index].axvline(
                    x=formalized.index.get_loc(exit_timestamp),
                    color=exit_color, linestyle='dotted', linewidth=1,
                    alpha=marker_coordinate_alpha)

    x_offset = 1.2

    if previous_close:
        if (current_open != trade_data['entry_price']
            and current_open != trade_data['exit_price']):
            delta = current_open - previous_close
            string = f'{delta:.1f}, {delta / previous_close * 100:.2f}%'
            add_tooltips(config, axlist, x_offset, current_open, string,
                         style['tg_tooltip_color'],
                         style['rc']['axes.edgecolor'])

    last_primary_axis = len(axlist) - 2
    if not pd.isna(trade_data['entry_price']):
        acronym = create_acronym(trade_data['entry_reason'])
        if acronym:
            add_tooltips(config, axlist, x_offset, trade_data['entry_price'],
                         acronym, style['tg_tooltip_color'], entry_color,
                         last_primary_axis, formalized, entry_timestamp)
    if not pd.isna(trade_data['exit_price']):
        acronym = create_acronym(trade_data['exit_reason'])
        if acronym:
            string = f"{acronym}, {result:.1f}, {trade_data['change']:.2f}%"
        else:
            string = f"{result:.1f}, {trade_data['change']:.2f}%"

        add_tooltips(config, axlist, x_offset, trade_data['exit_price'],
                     string, style['tg_tooltip_color'], exit_color,
                     last_primary_axis, formalized, exit_timestamp)

    notes = []
    for index in range(1, 11):
        note_column = trade_data[f'optional_note_{index}']
        if note_column:
            notes.append(note_column)

    add_text(panel, axlist, x_offset, 0.07,
             (f"Trade {trade_data['number']} for {trade_data['symbol']}"
              f" using {trade_data['trade_type'].title()}"
              f" {create_acronym(trade_data['tactic'])}"
              f" at {entry_date.strftime('%Y-%m-%d')}"
              f" {trade_data['entry_time'].strftime('%H:%M')}"),
             pd.Series(notes).dropna(), style['facecolor'])

    fig.savefig(os.path.join(config['General']['trading_directory'],
                             base + '.png'))


def add_ma(config, formalized, mpf, addplot, style, ma='ema'):
    """Add Exponential Moving Average (EMA) plots to the existing plots."""
    if ma == 'ema':
        ma_1 = ema(formalized.close, 5)
        ma_2 = ema(formalized.close, 25)
        ma_3 = ema(formalized.close, 75)

    ma_apd = [
        mpf.make_addplot(ma_1, color=style['mavcolors'][0], width=0.8),
        mpf.make_addplot(ma_2, color=style['mavcolors'][1], width=0.8),
        mpf.make_addplot(ma_3, color=style['mavcolors'][2], width=0.8)]
    addplot.extend(ma_apd)


def add_macd(config, formalized, panel, mpf, addplot, style, ma='ema'):
    """Add Moving Average Convergence Divergence (MACD) plots to the panel."""
    if ma == 'ema':
        macd = ema(formalized.close, 12) - ema(formalized.close, 26)
        ylabel = 'MACD'
    elif ma == 'tema':
        macd = tema(formalized.close, 12) - tema(formalized.close, 26)
        ylabel = 'MACD TEMA'

    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    panel += 1
    macd_apd = [
        mpf.make_addplot(macd, panel=panel, color=style['mavcolors'][0],
                         width=0.8, ylabel=ylabel),
        mpf.make_addplot(signal, panel=panel, color=style['mavcolors'][1],
                         width=0.8, secondary_y=False),
        mpf.make_addplot(histogram, type='bar', width=1.0, panel=panel,
                         color=style['mavcolors'][2], secondary_y=False)]
    addplot.extend(macd_apd)

    return panel


def ema(series, span):
    """Calculate the Exponential Moving Average (EMA) of a series."""
    ema = series.ewm(span=span).mean()
    ema.iloc[:span - 1] = np.nan
    return ema


def tema(series, span):
    """Calculate the Triple Exponential Moving Average (TEMA) of a series."""
    ema_1 = ema(series, span)
    ema_2 = ema(ema_1, span)
    ema_3 = ema(ema_2, span)
    tema = 3 * (ema_1 - ema_2) + ema_3
    tema.iloc[:3 * (span - 1)] = np.nan
    return tema


def add_stochastics(config, formalized, panel, mpf, addplot, style):
    """Add stochastic oscillator plots to the given panel."""
    k = 5
    d = 3
    smooth_k = 3
    df = stochastics(formalized.high, formalized.low, formalized.close, k=k,
                     d=d, smooth_k=smooth_k)
    if df.k.dropna().empty:
        df.k.fillna(50.0, inplace=True)
    if df.d.dropna().empty:
        df.d.fillna(50.0, inplace=True)

    formalized['k'] = pd.Series(dtype='float')
    formalized['d'] = pd.Series(dtype='float')
    formalized.update(df)
    panel += 1
    stoch_apd = [mpf.make_addplot(formalized.k, panel=panel,
                                  color=style['mavcolors'][0], width=0.8,
                                  ylabel='Stochastics'),
                 mpf.make_addplot(formalized.d, panel=panel,
                                  color=style['mavcolors'][1], width=0.8,
                                  secondary_y=False)]
    addplot.extend(stoch_apd)

    return panel


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

    return pd.DataFrame({'k': stochastics_k, 'd': stochastics_d})


def create_acronym(phrase):
    """Generate an acronym from the given phrase."""
    if isinstance(phrase, str):
        acronym = ''
        for word in re.split(r'[\W_]+', phrase):
            acronym = acronym + word[0].upper()

        return acronym


def add_tooltips(config, axlist, x_offset, price, string, color, bbox_color,
                 last_primary_axis=None, formalized=None, timestamp=None):
    """Add tooltips to the specified axes list."""
    alpha = 0.8
    bbox_alpha = 0.6

    axlist[0].text(
        -x_offset, price, string, alpha=alpha, c=color, size='small',
        ha='right', va='center',
        bbox=dict(boxstyle='round, pad=0.2', alpha=bbox_alpha, ec='none',
                  fc=bbox_color))
    if timestamp:
        bottom, top = axlist[last_primary_axis].get_ylim()
        axlist[last_primary_axis].text(
            formalized.index.get_loc(timestamp), -0.03 * (top - bottom),
            timestamp.strftime('%H:%M'), alpha=alpha, c=color, size='small',
            ha='center', va='top',
            bbox=dict(boxstyle='round, pad=0.2', alpha=bbox_alpha, ec='none',
                      fc=bbox_color))


def add_text(panel, axlist, x_offset, y_offset_ratio, title, note_series,
             bbox_color):
    # TODO: fix docstring
    """Add a title and notes to the top of the specified panel."""
    # Use the last panel to prevent other panels from overwriting the
    # text.
    axis_index = 2 * panel
    bottom, top = axlist[axis_index].get_ylim()

    axlist[axis_index].text(x_offset,
                            panel * top - y_offset_ratio * (top - bottom),
                            title, weight='bold', va='top')

    errors = ''
    for note_index, value in note_series.items():
        if note_index == 0:
            errors = f'\n{note_index + 1}. {value}'
        else:
            errors = f'{errors}\n{note_index + 1}. {value}'

    if errors:
        axlist[axis_index].text(x_offset,
                                panel * top - y_offset_ratio * (top - bottom),
                                errors, va='top', zorder=0,
                                bbox=dict(alpha=0.5, ec='none', fc=bbox_color))


def check_charts(config, charts):
    """Validate charts in the trading directory and print invalid ones."""
    for f in os.listdir(config['General']['trading_directory']):
        if f.endswith('.png') and f not in charts.values:
            print(os.path.join(config['General']['trading_directory'], f))

    for value in charts:
        if isinstance(value, str) and not os.path.exists(
                os.path.join(config['General']['trading_directory'], value)):
            print(value)


if __name__ == '__main__':
    main()
