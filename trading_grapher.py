#!/usr/bin/env python3

"""Visualize trade data using charts and technical indicators."""

import argparse
import configparser
import importlib
import os
import sys

from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import mplfinance as mpf
import numpy as np
import pandas as pd

import configuration
import file_utilities

ISO_DATE_FORMAT = '%Y-%m-%d'
DATE_FORMAT = '%b %-d'
TIME_FORMAT = '%-H:%M'


def main():
    """Parse trade data, save market data, plot charts, and check charts."""
    args = get_arguments()
    config_path = file_utilities.get_config_path(__file__)
    backup_parameters = {'number_of_backups': 8}

    if any((args.G, args.J)):
        config = configure(config_path, can_interpolate=False)
        if args.G and configuration.modify_section(
                config, 'General', config_path,
                backup_parameters=backup_parameters):
            return
        if args.J and configuration.modify_section(
                config, 'Trading Journal', config_path,
                backup_parameters=backup_parameters):
            return
    elif args.C:
        default_config = configure(config_path, can_interpolate=False,
                                   can_override=False)
        configuration.check_config_changes(
            default_config, config_path, backup_parameters=backup_parameters)
        return
    else:
        config = configure(config_path)

    if args.B:
        file_utilities.create_bash_wrapper(__file__, args.B)
        return

    trading_path = args.f[0] if args.f else config['General']['trading_path']
    charts_directory = (args.d[0] if args.d
                        else config['General']['charts_directory'])

    trading_journal = pd.read_excel(
        trading_path, sheet_name=config['General']['trading_sheet'])
    trading_journal_columns = (
        ['optional_number', 'symbol', 'trade_type', 'optional_tactic',
         'entry_date', 'entry_time', 'entry_price', 'optional_entry_reason',
         'exit_date', 'exit_time', 'exit_price', 'optional_exit_reason',
         'optional_percentage_change']
        + [f'optional_note_{index}' for index in range(1, 11)])

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
                trade_data = {
                    column: trade.get(config['Trading Journal'][column])
                    for column in trading_journal_columns}

                if not trade_data['optional_number']:
                    trade_data['optional_number'] = index - first_index + 1

                for d in ['entry_date', 'exit_date']:
                    trade_data[d] = trade_data[d].tz_localize(
                        config['Market Data']['timezone'])

                market_data_path = os.path.join(
                    charts_directory,
                    f"{trade_data['entry_date'].strftime(ISO_DATE_FORMAT)}-00"
                    f"-{trade_data['symbol']}.csv")

                save_market_data(config, trade_data, market_data_path)
                plot_charts(config, trade_data, market_data_path, style,
                            charts_directory)
                has_plotted = True

    if (has_plotted
        and config['Trading Journal']['optional_chart_file']
        in trading_journal.columns):
        file_utilities.compare_directory_list(
            charts_directory, r'\d{4}-\d{2}-\d{2}-\d{2}-\w+\.png',
            trading_journal[config['Trading Journal']['optional_chart_file']])


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        'dates', nargs='*',
        default=[pd.Timestamp.now().strftime(ISO_DATE_FORMAT)],
        help='specify dates in the format %%Y-%%m-%%d')
    parser.add_argument(
        '-f', nargs=1,
        help='specify the file path to the trading journal spreadsheet',
        metavar='FILE')
    parser.add_argument(
        '-d', nargs=1,
        help='specify the directory path'
        ' for storing historical data and charts',
        metavar='DIRECTORY')
    group.add_argument(
        '-B', nargs='?', const='.',
        help=f'generate a Bash script for running {os.path.basename(__file__)}'
        ' with an optional output directory',
        metavar='DIRECTORY')
    group.add_argument(
        '-G', action='store_true',
        help='configure general options and exit')
    group.add_argument(
        '-J', action='store_true',
        help='configure the columns of the trading journal and exit')
    group.add_argument(
        '-C', action='store_true',
        help='check configuration changes and exit')

    return parser.parse_args()


def configure(config_path, can_interpolate=True, can_override=True):
    """Get the configuration parser object with the set up configuration."""
    if can_interpolate:
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    else:
        config = configparser.ConfigParser(interpolation=None)

    config['General'] = {
        'trading_path': os.path.join(os.path.expanduser('~'),
                                     'Documents/Trading', 'Trading.ods'),
        'trading_sheet': 'Trading Journal',
        # TODO: add completion
        'style': 'fluorite',    # TODO: add ametrine and amber
        'charts_directory': os.path.join(os.path.expanduser('~'),
                                         'Documents/Trading')}
    config['Market Data'] = {
        'opening_time': '09:00:00',
        'closing_time': '15:30:00',
        'timezone': 'Asia/Tokyo'}

    config['Trading Journal'] = { # TODO: add completion
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
        'optional_percentage_change': 'Percentage Change',
        'optional_chart_file': 'optional_chart_file'}
    for index in range(1, 11):
        config['Trading Journal'][f'optional_note_{index}'] = f'Note {index}'

    config['Active Trading Hours'] = {
        'is_added': 'True',
        # TODO: add completion
        'start_time': '${Market Data:opening_time}',
        'end_time': '${Market Data:closing_time}'}
    config['EMA'] = {
        'is_added': 'True',
        'short_term_period': '5',
        'medium_term_period': '25',
        'long_term_period': '75'}
    config['MACD'] = {
        'is_added': 'True',
        'short_term_period': '12',
        'long_term_period': '26',
        'signal_period': '9'}
    config['Stochastics'] = {
        'is_added': 'True',
        'k_period': '5',
        'd_period': '3',
        'smooth_k_period': '3'}
    config['Volume'] = {
        'is_added': 'True',
        'quantile_threshold': '0.99'}
    config['Minor X-ticks'] = {
        'is_added': 'True'}
    config['Tooltips'] = {
        'is_added': 'True'}
    config['Text'] = {
        'is_added': 'True'}

    if can_override:
        configuration.read_config(config, config_path)

    return config


def save_market_data(config, trade_data, market_data_path):
    """Save historical market data for a given symbol to a CSV file."""
    PERIOD_IN_DAYS = 7
    delta = (
        pd.Timestamp.now(tz=config['Market Data']['timezone']).normalize()
        - trade_data['entry_date'])
    last = modified_time = pd.Timestamp(0,
                                        tz=config['Market Data']['timezone'])
    if os.path.exists(market_data_path):
        formalized = pd.read_csv(market_data_path, index_col=0,
                                 parse_dates=True)
        last = formalized.tail(1).index[0]
        modified_time = pd.Timestamp(os.path.getmtime(market_data_path),
                                     tz=config['Market Data']['timezone'],
                                     unit='s')

    if (PERIOD_IN_DAYS <= 1 + delta.days
        or last + pd.Timedelta(minutes=30) < modified_time
        or pd.Timestamp.now(tz=config['Market Data']['timezone'])
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
            config['Market Data']['timezone'])
        q = df.volume.quantile(float(config['Volume']['quantile_threshold']))
        df['volume'] = df['volume'].mask(df['volume'] > q, q)

        previous = df[df.index < trade_data['entry_date']]
        if not previous.empty:
            previous_date = pd.Timestamp.date(
                previous.dropna().tail(1).index[0])
            previous_date = pd.Timestamp(previous_date,
                                         tz=config['Market Data']['timezone'])

        morning = (pd.Timedelta(str(trade_data['exit_time']))
                   < pd.Timedelta(hours=12))
        # TODO: extend trading hours
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


def plot_charts(config, trade_data, market_data_path, style, charts_directory):
    """Plot trading charts with entry and exit points, and indicators."""
    try:
        formalized = pd.read_csv(market_data_path, index_col=0,
                                 parse_dates=True)
    except Exception as e:
        print(e)
        sys.exit(1)

    result = 0
    if (not pd.isna(trade_data['entry_price'])
        and not pd.isna(trade_data['exit_price'])):
        result = (trade_data['exit_price'] - trade_data['entry_price']
                  if trade_data['trade_type'].lower() == 'long'
                  else trade_data['entry_price'] - trade_data['exit_price']
                  if trade_data['trade_type'].lower() == 'short'
                  else 0)

    percentage_change = (100 * result / trade_data['entry_price']
                         if pd.isna(trade_data['optional_percentage_change'])
                         else trade_data['optional_percentage_change'])

    timestamps = {
        'start': create_timestamp(
            trade_data['entry_date'],
            config['Active Trading Hours']['start_time']),
        'end': create_timestamp(
            trade_data['entry_date'],
            config['Active Trading Hours']['end_time']),
        'entry': None, 'exit': None}
    if isinstance(timestamps['end'], pd.Timestamp):
        timestamps['end'] = min(formalized.tail(1).index[0], timestamps['end'])

    addplot = []
    closing_opening_entry_exit_prices = [None, None, None, None]
    closing_opening_entry_exit_colors = [
        style['rc']['axes.edgecolor'], style['rc']['axes.edgecolor'],
        style['custom_style']['neutral_color'],
        style['custom_style']['neutral_color']]

    prepare_marker_parameters(formalized, trade_data, result, style,
                              timestamps, addplot,
                              closing_opening_entry_exit_prices,
                              closing_opening_entry_exit_colors)
    if None in closing_opening_entry_exit_prices:
        print('Key trading prices are incomplete.')
        return

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
        fill_between=dict(alpha=style['custom_style']['filled_area_alpha'],
                          color=closing_opening_entry_exit_colors[3],
                          y1=trade_data['entry_price'],
                          y2=trade_data['exit_price'], zorder=1),
        hlines=dict(alpha=style['custom_style']['line_alpha'],
                    colors=closing_opening_entry_exit_colors,
                    hlines=closing_opening_entry_exit_prices,
                    linestyle=[style['custom_style']['closing_line'],
                               style['custom_style']['opening_line'],
                               style['custom_style']['entry_line'],
                               style['custom_style']['exit_line']],
                    linewidths=1),
        returnfig=True, scale_padding={'top': 0, 'right': 0.05, 'bottom': 1.4},
        scale_width_adjustment=dict(candle=1.5), style=style,
        tight_layout=True, type='candle',
        volume=config['Volume'].getboolean('is_added'), volume_panel=panel)

    add_vertical_elements(
        formalized, timestamps, axlist, closing_opening_entry_exit_colors,
        style, config['Active Trading Hours'].getboolean('is_added'))

    axlist[0].set_xticks(np.arange(*axlist[0].get_xlim(), 30))
    if config['Minor X-ticks'].getboolean('is_added'):
        add_minor_xticks(axlist, style['custom_style']['minor_grid_alpha'])

    if config['Stochastics'].getboolean('is_added'):
        axlist[2 * stoch_panel].set_yticks([20.0, 50.0, 80.0])

    if (config['Tooltips'].getboolean('is_added')
        and closing_opening_entry_exit_prices[0]
        and closing_opening_entry_exit_prices[1] != trade_data['entry_price']
        and closing_opening_entry_exit_prices[1] != trade_data['exit_price']):
        delta = (closing_opening_entry_exit_prices[1]
                 - closing_opening_entry_exit_prices[0])
        add_tooltips(
            config, axlist, closing_opening_entry_exit_prices[1],
            f'{delta:.1f},'
            f' {100 * delta / closing_opening_entry_exit_prices[0]:.2f}%',
            style['custom_style']['tooltip_color'],
            closing_opening_entry_exit_colors[1],
            style['custom_style']['tooltip_bbox_alpha'])

    if (config['Tooltips'].getboolean('is_added')
        and not pd.isna(trade_data['entry_price'])):
        acronym = file_utilities.create_acronym(
            trade_data['optional_entry_reason'])
        add_tooltips(config, axlist, trade_data['entry_price'],
                     f'{acronym}' if acronym else '',
                     style['custom_style']['tooltip_color'],
                     closing_opening_entry_exit_colors[2],
                     style['custom_style']['tooltip_bbox_alpha'],
                     formalized=formalized, timestamp=timestamps['entry'])

    if (config['Tooltips'].getboolean('is_added')
        and not pd.isna(trade_data['exit_price'])):
        acronym = file_utilities.create_acronym(
            trade_data['optional_exit_reason'])
        add_tooltips(config, axlist, trade_data['exit_price'],
                     f"{f'{acronym}, ' if acronym else ''}"
                     f"{result:.1f}, {percentage_change:.2f}%",
                     style['custom_style']['tooltip_color'],
                     closing_opening_entry_exit_colors[3],
                     style['custom_style']['tooltip_bbox_alpha'],
                     formalized=formalized, timestamp=timestamps['exit'])

    if config['Text'].getboolean('is_added'):
        tactic = file_utilities.create_acronym(trade_data['optional_tactic'])
        full_date_format = f'%a, {DATE_FORMAT}, {chr(39)}%y,'
        notes = [trade_data[f'optional_note_{i}'] for i in range(1, 11)
                 if trade_data[f'optional_note_{i}']]
        add_text(axlist,
                 f"Trade {trade_data['optional_number']}"
                 f" for {trade_data['symbol']}"
                 f" using {trade_data['trade_type'].title()}"
                 f"{f' {tactic}' if tactic else ''}"
                 f" on {trade_data['entry_date'].strftime(full_date_format)}"
                 f" at {trade_data['entry_time'].strftime(TIME_FORMAT)}",
                 pd.Series(notes).dropna(), style['facecolor'],
                 style['custom_style']['text_bbox_alpha'])

    fig.savefig(os.path.join(
        charts_directory,
        f"{trade_data['entry_date'].strftime(ISO_DATE_FORMAT)}"
        f"-{int(trade_data['optional_number']):02}"
        f"-{trade_data['symbol']}.png"))


def create_timestamp(date, time):
    """Create a pandas Timestamp by adding a time duration to a date."""
    if pd.isna(date):
        return pd.NaT
    else:
        return date + (pd.Timedelta(time)
                       if isinstance(time, str) else pd.Timedelta(str(time)))


def prepare_marker_parameters(formalized, trade_data, result, style,
                              timestamps, addplot,
                              close_open_entry_exit_prices,
                              close_open_entry_exit_colors):
    """Prepare entry and exit marker parameters for plotting charts."""
    previous = formalized[formalized.index < trade_data['entry_date']]
    current = formalized[trade_data['entry_date'] <= formalized.index]

    if previous.notnull().values.any():
        close_open_entry_exit_prices[0] = (
            previous.dropna().tail(1).close.iloc[0])
        close_open_entry_exit_prices[1] = current.dropna().head(1).open.iloc[0]

    # nan is not recognized as False in a boolean context.
    if (not pd.isna(trade_data['entry_time'])
        and not pd.isna(trade_data['entry_price'])):
        formalized['entry_point'] = pd.Series(dtype='float')
        timestamps['entry'] = create_timestamp(trade_data['entry_date'],
                                               trade_data['entry_time'])
        formalized.loc[timestamps['entry'], 'entry_point'] = (
            trade_data['entry_price'])
        addplot.append(mpf.make_addplot(
            formalized.entry_point,
            alpha=style['custom_style']['marker_alpha'],
            color=close_open_entry_exit_colors[2], edgecolors='none',
            marker=style['custom_style'].get(
                f"{trade_data['trade_type'].lower()}_marker"),
            markersize=100, type='scatter'))
        close_open_entry_exit_prices[2] = trade_data['entry_price']

    if (not pd.isna(trade_data['exit_time'])
        and not pd.isna(trade_data['exit_price'])):
        formalized['exit_point'] = pd.Series(dtype='float')
        timestamps['exit'] = create_timestamp(trade_data['exit_date'],
                                              trade_data['exit_time'])
        formalized.loc[timestamps['exit'], 'exit_point'] = (
            trade_data['exit_price'])
        addplot.append(mpf.make_addplot(
            formalized.exit_point, alpha=style['custom_style']['marker_alpha'],
            color=close_open_entry_exit_colors[3], edgecolors='none',
            marker=style['custom_style'].get(
                f"{trade_data['trade_type'].lower()}_marker"),
            markersize=100, type='scatter'))
        close_open_entry_exit_prices[3] = trade_data['exit_price']
        if result > 0:
            close_open_entry_exit_colors[3] = (
                style['custom_style']['profit_color'])
        elif result < 0:
            close_open_entry_exit_colors[3] = (
                style['custom_style']['loss_color'])


def add_emas(config, formalized, mpf, addplot, style):
    """Add exponential moving average plots to the existing plots."""
    ma_1 = ema(formalized.close, int(config['EMA']['short_term_period']))
    ma_2 = ema(formalized.close, int(config['EMA']['medium_term_period']))
    ma_3 = ema(formalized.close, int(config['EMA']['long_term_period']))

    ma_addplot = [
        mpf.make_addplot(ma_1, color=style['mavcolors'][0], width=0.8),
        mpf.make_addplot(ma_2, color=style['mavcolors'][1], width=0.8),
        mpf.make_addplot(ma_3, color=style['mavcolors'][2], width=0.8)]
    addplot.extend(ma_addplot)


def add_macd(config, formalized, panel, mpf, addplot, style, ma='ema'):
    """Add moving average convergence divergence plots to the given panel."""
    if ma == 'ema':
        macd = (
            ema(formalized.close, int(config['MACD']['short_term_period']))
            - ema(formalized.close, int(config['MACD']['long_term_period'])))
        ylabel = 'MACD'
    elif ma == 'tema':
        macd = (
            tema(formalized.close, int(config['MACD']['short_term_period']))
            - tema(formalized.close, int(config['MACD']['long_term_period'])))
        ylabel = 'MACD TEMA'

    signal = macd.ewm(span=int(config['MACD']['signal_period'])).mean()
    histogram = macd - signal
    panel += 1
    macd_addplot = [
        mpf.make_addplot(macd, color=style['mavcolors'][0], panel=panel,
                         width=0.8, ylabel=ylabel),
        mpf.make_addplot(signal, color=style['mavcolors'][1], panel=panel,
                         secondary_y=False, width=0.8),
        mpf.make_addplot(histogram,
                         color=[style['mavcolors'][2] if value >= 0
                                else style['mavcolors'][3]
                                for value in histogram],
                         panel=panel, secondary_y=False, type='bar',
                         width=1.0)]
    addplot.extend(macd_addplot)

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
    df = stochastics(formalized.high, formalized.low, formalized.close,
                     k=int(config['Stochastics']['k_period']),
                     d=int(config['Stochastics']['d_period']),
                     smooth_k=int(config['Stochastics']['smooth_k_period']))
    if df.k.dropna().empty:
        df.k.fillna(50.0, inplace=True)
    if df.d.dropna().empty:
        df.d.fillna(50.0, inplace=True)

    formalized['k'] = pd.Series(dtype='float')
    formalized['d'] = pd.Series(dtype='float')
    formalized.update(df)
    panel += 1
    stoch_addplot = [mpf.make_addplot(formalized.k,
                                      color=style['mavcolors'][0], panel=panel,
                                      width=0.8, ylabel='Stochastics'),
                     mpf.make_addplot(formalized.d,
                                      color=style['mavcolors'][1], panel=panel,
                                      secondary_y=False, width=0.8)]
    addplot.extend(stoch_addplot)

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


def add_minor_xticks(axlist, minor_grid_alpha):
    """Add minor x-ticks and their grid between panels."""
    axlist[0].set_xticks(np.arange(*axlist[0].get_xlim(), 10), minor=True)
    for index, _ in enumerate(axlist):
        if (index % 2) == 0:
            axlist[index].grid(which='minor', alpha=minor_grid_alpha)


def add_vertical_elements(formalized, timestamps, axlist,
                          close_open_entry_exit_colors, style,
                          is_active_trading_hours_added):
    """Add vertical elements between panels at the specified timestamps."""
    for index, _ in enumerate(axlist):
        if (index % 2) == 0:
            if (is_active_trading_hours_added
                and timestamps['start'] and timestamps['end']):
                # Force a redraw of the y-limits to ensure all plot
                # elements are taken into account.
                axlist[index].set_ylim(*axlist[index].get_ylim())
                axlist[index].fill_betweenx(
                    axlist[index].get_ylim(),
                    formalized.index.get_loc(timestamps['start']),
                    formalized.index.get_loc(timestamps['end']),
                    facecolor=style['custom_style'][
                        'active_trading_hours_color'],
                    zorder=0)
            if timestamps['entry']:
                axlist[index].axvline(
                    alpha=style['custom_style']['line_alpha'],
                    color=close_open_entry_exit_colors[2],
                    linestyle=style['custom_style']['entry_line'],
                    linewidth=1,
                    x=formalized.index.get_loc(timestamps['entry']))
            if timestamps['exit']:
                axlist[index].axvline(
                    alpha=style['custom_style']['line_alpha'],
                    color=close_open_entry_exit_colors[3],
                    linestyle=style['custom_style']['exit_line'],
                    linewidth=1,
                    x=formalized.index.get_loc(timestamps['exit']))


def add_tooltips(config, axlist, price, string, color, bbox_color, bbox_alpha,
                 formalized=None, timestamp=None):
    """Add tooltips to the specified axes."""
    # Calculate x_offset and y_offset_ratios using points_to_pixels()
    # and transform(). The values are currently obtained heuristically.
    x_offset = -1.2
    axlist[0].text(x_offset, price, string, c=color, ha='right', size='small',
                   va='center',
                   bbox=dict(alpha=bbox_alpha, boxstyle='round, pad=0.2',
                             ec='none', fc=bbox_color))

    if timestamp:
        last_primary_axes = len(axlist) - 2
        bottom, top = axlist[last_primary_axes].get_ylim()
        y_offset_ratios = {0: -0.006, 2: -0.02, 4: -0.025, 6: -0.03}
        y_offset_ratio = y_offset_ratios.get(last_primary_axes)

        axlist[last_primary_axes].text(
            formalized.index.get_loc(timestamp),
            bottom + y_offset_ratio * (top - bottom),
            timestamp.strftime(TIME_FORMAT), c=color, ha='center',
            size='small', va='top',
            bbox=dict(alpha=bbox_alpha, boxstyle='round, pad=0.2', ec='none',
                      fc=bbox_color))


def add_text(axlist, title, note_series, bbox_color, bbox_alpha):
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

    axlist[last_primary_axes].text(
        x_offset, y, title, va='top', weight='bold',
        bbox=dict(alpha=bbox_alpha, boxstyle='square, pad=0.1', ec='none',
                  fc=bbox_color))

    notes = ''
    for note_index, value in note_series.items():
        notes = (f'\n{note_index + 1}. {value}' if note_index == 0
                 else f'{notes}\n{note_index + 1}. {value}')

    if notes:
        axlist[last_primary_axes].text(
            x_offset, y, notes, va='top', zorder=1,
            bbox=dict(alpha=bbox_alpha, boxstyle='square, pad=0.1', ec='none',
                      fc=bbox_color))


if __name__ == '__main__':
    main()
