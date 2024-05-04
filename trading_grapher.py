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

DATE_FORMAT = '%b %-d'
TIME_FORMAT = '%-H:%M'


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
    trading_journal_columns = ['optional_number', 'symbol', 'trade_type',
                               'optional_tactic', 'entry_date', 'entry_time',
                               'entry_price', 'optional_entry_reason',
                               'exit_date', 'exit_time', 'exit_price',
                               'optional_exit_reason', 'change']
    for i in range(1, 11):
        trading_journal_columns.append(f'optional_note_{i}')

    try:
        style = importlib.import_module(
            f"styles.{config['General']['style']}").style
    except ModuleNotFoundError as e:
        print(e)
        sys.exit(1)

    has_plotted = False
    for date in pd.to_datetime(args.dates):
        trades = trading_journal.loc[
            trading_journal[config['Trading Journal']['entry_date']] == date]
        if not trades.empty:
            first_index = next(trades.iterrows())[0]
            for index, trade in trades.iterrows():
                trade_data = {}
                for column in trading_journal_columns:
                    trade_data[column] = trade.get(
                        config['Trading Journal'][column])

                if not trade_data['optional_number']:
                    trade_data['optional_number'] = index - first_index + 1

                trade_data['entry_date'] = (
                    trade_data['entry_date'].tz_localize(
                        config['Market Data']['time_zone']))
                trade_data['exit_date'] = (
                    trade_data['exit_date'].tz_localize(
                        config['Market Data']['time_zone']))
                market_data_path = os.path.join(
                    config['General']['trading_directory'],
                    (f"{trade_data['entry_date'].strftime('%Y-%m-%d')}-00"
                     f"-{trade_data['symbol']}.csv"))

                save_market_data(config, trade_data, market_data_path)
                plot_chart(config, trade_data, market_data_path, style)
                has_plotted = True

    if (has_plotted
        and config['Trading Journal']['optional_chart_file']
        in trading_journal.columns):
        file_utilities.compare_directory_list(
            config['General']['trading_directory'],
            r'\d{4}-\d{2}-\d{2}-\d{2}-\w+\.png',
            trading_journal[config['Trading Journal']['optional_chart_file']])


def configure(config_path, can_interpolate=True, can_override=True):
    """Get the configuration parser object with the set up configuration."""
    if can_interpolate:
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    else:
        config = configparser.ConfigParser(interpolation=None)

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
        'optional_number': 'Number',
        'symbol': 'Symbol',
        'trade_type': 'Trade type',
        'optional_tactic': 'Tactic',
        'entry_date': 'Entry date',
        'entry_time': 'Entry time',
        'entry_price': 'Entry price',
        'optional_entry_reason': 'Entry reason',
        'exit_date': 'Exit date',
        'exit_time': 'Exit time',
        'exit_price': 'Exit price',
        'optional_exit_reason': 'Exit reason',
        'change': 'Change',   # TODO: calculate change and add optional_ prefix
        'optional_chart_file': 'optional_chart_file'}
    for i in range(1, 11):
        config['Trading Journal'][f'optional_note_{i}'] = ''

    config['EMA'] = {
        'is_added': 'True'}
    config['MACD'] = {
        'is_added': 'True'}
    config['Stochastics'] = {
        'is_added': 'True'}
    config['Volume'] = {
        'is_added': 'True'}
    config['Tooltips'] = {
        'is_added': 'True'}
    config['Text'] = {
        'is_added': 'True'}

    if can_override:
        configuration.read_config(config, config_path)
        # configuration.write_config(config, config_path)  # TODO

    return config


def save_market_data(config, trade_data, market_data_path):
    """Save historical market data for a given symbol to a CSV file."""
    PERIOD_IN_DAYS = 7
    delta = (
        pd.Timestamp.now(tz=config['Market Data']['time_zone']).normalize()
        - trade_data['entry_date'])
    last = modified_time = pd.Timestamp(0,
                                        tz=config['Market Data']['time_zone'])
    if os.path.exists(market_data_path):
        formalized = pd.read_csv(market_data_path, index_col=0,
                                 parse_dates=True)
        last = formalized.tail(1).index[0]
        modified_time = pd.Timestamp(os.path.getmtime(market_data_path),
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
        # Yahoo Finance's historical data is in UTC, but lacks explicit
        # time zone information.
        df.index = df.index.tz_localize('UTC').tz_convert(
            config['Market Data']['time_zone'])
        # TODO: configure q
        q = df.volume.quantile(0.99)
        df['volume'] = df['volume'].mask(df['volume'] > q, q)

        previous = df[df.index < trade_data['entry_date']]
        if not previous.empty:
            previous_date = pd.Timestamp.date(
                previous.dropna().tail(1).index[0])
            previous_date = pd.Timestamp(previous_date,
                                         tz=config['Market Data']['time_zone'])

        morning = (pd.Timedelta(str(trade_data['exit_time']))
                   < pd.Timedelta(hours=12))
        if morning and not previous.empty:
            start = previous_date + pd.Timedelta(hours=12, minutes=30)
            end = trade_data['entry_date'] + pd.Timedelta(hours=11, minutes=29)
        else:
            start = trade_data['entry_date'] + pd.Timedelta(hours=9)
            end = trade_data['entry_date'] + pd.Timedelta(hours=14, minutes=59)

        formalized = pd.DataFrame(columns=('open', 'high', 'low', 'close',
                                           'volume'),
                                  index=pd.date_range(start, end, freq='min'))
        formalized.index.name = 'timestamp'
        formalized = formalized.astype('float')
        formalized.update(df)

        if morning and not previous.empty:
            start = previous_date + pd.Timedelta(hours=15)
            end = trade_data['entry_date'] + pd.Timedelta(hours=8, minutes=59)
            exclusion = pd.date_range(start=start, end=end, freq='min')
            formalized = formalized.loc[~formalized.index.isin(exclusion)]
        else:
            formalized = formalized.between_time('12:30:00', '11:29:00')

        if formalized.isna().values.all():
            print('Values are missing.')
            sys.exit(1)

        try:
            formalized.to_csv(market_data_path)
        except Exception as e:
            print(e)
            sys.exit(1)


def plot_chart(config, trade_data, market_data_path, style):
    """Plot a trading chart with entry and exit points, and indicators."""
    try:
        formalized = pd.read_csv(market_data_path, index_col=0,
                                 parse_dates=True)
    except Exception as e:
        print(e)
        sys.exit(1)

    entry_timestamp = exit_timestamp = None
    addplot = []
    close_open_entry_exit_hlines = [None, None, None, None]
    # TODO: rename tg
    close_open_entry_exit_colors = [None, None, style['tg']['neutral_color'],
                                    style['tg']['neutral_color']]

    previous = formalized[formalized.index < trade_data['entry_date']]
    current = formalized[trade_data['entry_date'] <= formalized.index]
    previous_close = current_open = 0.0

    if previous.notnull().values.any():
        previous_close = previous.dropna().tail(1).close.iloc[0]
        close_open_entry_exit_hlines[0] = previous_close
        close_open_entry_exit_colors[0] = style['rc']['axes.edgecolor']
        current_open = current.dropna().head(1).open.iloc[0]
        close_open_entry_exit_hlines[1] = current_open
        close_open_entry_exit_colors[1] = style['rc']['axes.edgecolor']

    if trade_data['trade_type'].lower() == 'long':
        marker = 'o'
    elif trade_data['trade_type'].lower() == 'short':
        marker = 'D'

    # TODO: move to style
    marker_alpha = 0.2

    # TODO: create add_marker()
    # nan is not recognized as False in a boolean context.
    if (not pd.isna(trade_data['entry_time'])
        and not pd.isna(trade_data['entry_price'])):
        formalized['entry_point'] = pd.Series(dtype='float')
        entry_timestamp = (trade_data['entry_date']
                           + pd.Timedelta(str(trade_data['entry_time'])))
        formalized.loc[entry_timestamp, 'entry_point'] = (
            trade_data['entry_price'])
        # TODO: rename to addplot
        entry_apd = mpf.make_addplot(formalized.entry_point, type='scatter',
                                     markersize=100, marker=marker,
                                     color=close_open_entry_exit_colors[2],
                                     edgecolors='none', alpha=marker_alpha)
        addplot.append(entry_apd)
        close_open_entry_exit_hlines[2] = trade_data['entry_price']
        close_open_entry_exit_colors[2] = close_open_entry_exit_colors[2]

    result = 0.0
    if (not pd.isna(trade_data['exit_time'])
        and not pd.isna(trade_data['exit_price'])):
        if trade_data['trade_type'].lower() == 'long':
            result = trade_data['exit_price'] - trade_data['entry_price']
        elif trade_data['trade_type'].lower() == 'short':
            result = trade_data['entry_price'] - trade_data['exit_price']
        if result > 0:
            close_open_entry_exit_colors[3] = style['tg']['profit_color']
        elif result < 0:
            close_open_entry_exit_colors[3] = style['tg']['loss_color']

        formalized['exit_point'] = pd.Series(dtype='float')
        exit_timestamp = (trade_data['exit_date']
                          + pd.Timedelta(str(trade_data['exit_time'])))
        formalized.loc[exit_timestamp, 'exit_point'] = trade_data['exit_price']
        exit_apd = mpf.make_addplot(formalized.exit_point, type='scatter',
                                    markersize=100, marker=marker,
                                    color=close_open_entry_exit_colors[3],
                                    edgecolors='none', alpha=marker_alpha)
        addplot.append(exit_apd)
        close_open_entry_exit_hlines[3] = trade_data['exit_price']

    # TODO: move to style
    marker_coordinate_alpha = 0.4

    panel = 0
    if config['EMA'].getboolean('is_added'):
        add_emas(config, formalized, mpf, addplot, style)
    if config['MACD'].getboolean('is_added'):
        panel = add_macd(config, formalized, panel, mpf, addplot, style)
    if config['Stochastics'].getboolean('is_added'):
        panel = stoch_panel = add_stochastics(config, formalized, panel, mpf,
                                              addplot, style)
    if config['Volume'].getboolean('is_added'):
        panel += 1

    fig, axlist = mpf.plot(
        formalized, addplot=addplot, closefig=True,
        datetime_format=f'{DATE_FORMAT}, {TIME_FORMAT}',
        figsize=(1152 / 100, 648 / 100),
        fill_between=dict(y1=trade_data['entry_price'],
                          y2=trade_data['exit_price'], alpha=0.04,
                          color=close_open_entry_exit_colors[3], zorder=0),
        hlines=dict(hlines=close_open_entry_exit_hlines,
                    colors=close_open_entry_exit_colors, linestyle='dotted',
                    linewidths=1, alpha=marker_coordinate_alpha),
        returnfig=True,
        scale_padding={'top': 0, 'right': 0.05, 'bottom': 1.4},
        scale_width_adjustment=dict(candle=1.5), style=style,
        tight_layout=True, type='candle',
        volume=config['Volume'].getboolean('is_added'), volume_panel=panel)

    left, right = axlist[0].get_xlim()
    axlist[0].set_xticks(np.arange(left, right, 30))
    axlist[0].set_xticks(np.arange(left, right, 10), minor=True)

    if config['Stochastics'].getboolean('is_added'):
        axlist[2 * stoch_panel].set_yticks([20.0, 50.0, 80.0])

    add_entry_exit_vlines(axlist, formalized, entry_timestamp,
                          close_open_entry_exit_colors[2], exit_timestamp,
                          close_open_entry_exit_colors[3],
                          marker_coordinate_alpha)

    if (previous_close and current_open != trade_data['entry_price']
        and current_open != trade_data['exit_price']):
        delta = current_open - previous_close
        string = f'{delta:.1f}, {delta / previous_close * 100:.2f}%'
        add_tooltips(config, axlist, current_open, string,
                     style['tg']['tooltip_color'],
                     close_open_entry_exit_colors[1])

    if not pd.isna(trade_data['entry_price']):
        acronym = create_acronym(trade_data['optional_entry_reason'])
        if acronym:
            add_tooltips(config, axlist, trade_data['entry_price'], acronym,
                         style['tg']['tooltip_color'],
                         close_open_entry_exit_colors[2],
                         formalized=formalized, timestamp=entry_timestamp)
    if not pd.isna(trade_data['exit_price']):
        acronym = create_acronym(trade_data['optional_exit_reason'])
        if acronym:
            string = f"{acronym}, {result:.1f}, {trade_data['change']:.2f}%"
        else:
            string = f"{result:.1f}, {trade_data['change']:.2f}%"

        add_tooltips(config, axlist, trade_data['exit_price'], string,
                     style['tg']['tooltip_color'],
                     close_open_entry_exit_colors[3], formalized=formalized,
                     timestamp=exit_timestamp)

    notes = []
    for index in range(1, 11):
        note_column = trade_data[f'optional_note_{index}']
        if note_column:
            notes.append(note_column)

    tactic = create_acronym(trade_data['optional_tactic'])
    full_date_format = f'%a, {DATE_FORMAT}, {chr(39)}%y,'
    add_text(axlist,
             (f"Trade {trade_data['optional_number']}"
              f" for {trade_data['symbol']}"
              f" using {trade_data['trade_type'].title()}"
              f"{f' {tactic}' if tactic else ''}"
              f" on {trade_data['entry_date'].strftime(full_date_format)}"
              f" at {trade_data['entry_time'].strftime(TIME_FORMAT)}"),
             pd.Series(notes).dropna(), style['facecolor'])

    fig.savefig(os.path.join(
        config['General']['trading_directory'],
        (f"{trade_data['entry_date'].strftime('%Y-%m-%d')}"
         f"-{int(trade_data['optional_number']):02}"
         f"-{trade_data['symbol']}.png")))


def append_marker_parameters(time_zone,
                             trade_data,
                             formalized,
                             marker,
                             entry_color,
                             marker_alpha,
                             style,
                             addplot,
                             hlines,
                             colors,
                             ):
    """Add markers."""


def add_emas(config, formalized, mpf, addplot, style):
    """Add exponential moving average plots to the existing plots."""
    ma_1 = ema(formalized.close, 5)
    ma_2 = ema(formalized.close, 25)
    ma_3 = ema(formalized.close, 75)

    ma_apd = [
        mpf.make_addplot(ma_1, color=style['mavcolors'][0], width=0.8),
        mpf.make_addplot(ma_2, color=style['mavcolors'][1], width=0.8),
        mpf.make_addplot(ma_3, color=style['mavcolors'][2], width=0.8)]
    addplot.extend(ma_apd)


def add_macd(config, formalized, panel, mpf, addplot, style, ma='ema'):
    """Add moving average convergence divergence plots to the given panel."""
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
                         color=[style['mavcolors'][2] if value >= 0
                                else style['mavcolors'][3]
                                for value in histogram],
                         secondary_y=False)]
    addplot.extend(macd_apd)

    return panel


def ema(series, span):
    """Calculate the exponential moving average of a series."""
    ema = series.ewm(span=span).mean()
    ema.iloc[:span - 1] = np.nan
    return ema


def tema(series, span):
    """Calculate the triple exponential moving average of a series."""
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


# TODO: rename to vline
def add_entry_exit_vlines(axlist, formalized, entry_timestamp, entry_color,
                          exit_timestamp, exit_color, marker_coordinate_alpha):
    """Add vertical lines between panels at entry and exit points."""
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


def create_acronym(phrase):
    """Generate an acronym from the given phrase."""
    acronym = ''
    if isinstance(phrase, str):
        for word in re.split(r'[\W_]+', phrase):
            acronym = acronym + word[0].upper()
    return acronym


def add_tooltips(config, axlist, price, string, color, bbox_color,
                 formalized=None, timestamp=None):
    """Add tooltips to the specified axes."""
    # Calculate x_offset and y_offset_ratios using points_to_pixels()
    # and transform(). The values are currently obtained heuristically.
    x_offset = -1.2
    alpha = 0.8
    bbox_alpha = 0.6

    axlist[0].text(x_offset, price, string, alpha=alpha, c=color, size='small',
                   ha='right', va='center',
                   bbox=dict(boxstyle='round, pad=0.2', alpha=bbox_alpha,
                             ec='none', fc=bbox_color))

    if timestamp:
        last_primary_axes = len(axlist) - 2
        bottom, top = axlist[last_primary_axes].get_ylim()
        y_offset_ratios = {0: -0.006, 2: -0.02, 4: -0.025, 6: -0.03}
        y_offset_ratio = y_offset_ratios.get(last_primary_axes)

        axlist[last_primary_axes].text(
            formalized.index.get_loc(timestamp),
            bottom + y_offset_ratio * (top - bottom),
            timestamp.strftime(TIME_FORMAT), alpha=alpha, c=color,
            size='small', ha='center', va='top',
            bbox=dict(boxstyle='round, pad=0.2', alpha=bbox_alpha, ec='none',
                      fc=bbox_color))


def add_text(axlist, title, note_series, bbox_color):
    """Add a title and notes to the last primary axes."""
    # Use the last panel to prevent other panels from overwriting the
    # text.
    last_primary_axes = len(axlist) - 2
    bottom, top = axlist[last_primary_axes].get_ylim()
    height = top - bottom

    # Calculate x_offset and y_offset_ratios using points_to_pixels()
    # and transform(). The values are currently obtained heuristically.
    # Additionally, modify panel_offset_factors if panel_ratios is
    # specified.
    x_offset = 1.2
    default_panel_offset_factor = (last_primary_axes / 2 - 1) * height
    panel_offset_factors = {0: 0, 2: 2.5 * height,
                            4: default_panel_offset_factor,
                            6: default_panel_offset_factor}
    panel_offset_factor = panel_offset_factors.get(last_primary_axes)
    y_offset_ratios = {0: -0.012, 2: -0.04, 4: -0.06, 6: -0.07}
    y_offset_ratio = y_offset_ratios.get(last_primary_axes)
    y = top + panel_offset_factor + y_offset_ratio * height

    axlist[last_primary_axes].text(x_offset, y, title, weight='bold', va='top')

    notes = ''
    for note_index, value in note_series.items():
        if note_index == 0:
            notes = f'\n{note_index + 1}. {value}'
        else:
            notes = f'{notes}\n{note_index + 1}. {value}'

    if notes:
        axlist[last_primary_axes].text(
            x_offset, y, notes, va='top', zorder=1,
            bbox=dict(alpha=0.5, ec='none', fc=bbox_color))


if __name__ == '__main__':
    main()
