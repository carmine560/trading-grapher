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
import indicators

TRADING_JOURNAL_COLUMNS = (
    ['optional_number', 'symbol', 'trade_type', 'optional_tactic',
     'entry_date', 'entry_time', 'entry_price', 'optional_entry_reason',
     'exit_date', 'exit_time', 'exit_price', 'optional_exit_reason',
     'optional_percentage_change']
    + [f'optional_note_{index}' for index in range(1, 11)])
ISO_DATE_FORMAT = '%Y-%m-%d'
DATE_FORMAT = '%b %-d'
TIME_FORMAT = '%-H:%M'


def main():
    """Parse trade data, save market data, plot charts, and check charts."""
    args = get_arguments()
    config_path = file_utilities.get_config_path(__file__)
    config = configure(config_path)
    trading_path = args.f[0] if args.f else config['General']['trading_path']
    trading_sheet = config['General']['trading_sheet']

    file_utilities.create_launchers_exit(args, __file__)
    configure_exit(args, config_path, trading_path, trading_sheet)

    trading_journal = pd.read_excel(trading_path, sheet_name=trading_sheet)
    charts_directory = (args.d[0] if args.d
                        else config['General']['charts_directory'])
    has_plotted = False

    for date in pd.to_datetime(args.dates):
        trades = trading_journal.loc[
            trading_journal[config['Trading Journal']['entry_date']] == date]
        if not trades.empty:
            first_index = next(trades.iterrows())[0]
            for index, trade in trades.iterrows():
                trade_data = {
                    column: trade.get(config['Trading Journal'][column])
                    for column in TRADING_JOURNAL_COLUMNS}

                if not trade_data['optional_number']:
                    trade_data['optional_number'] = index - first_index + 1
                for d in ['entry_date', 'exit_date']:
                    trade_data[d] = trade_data[d].tz_localize(
                        config['Market Data']['timezone'])

                trade_data['trade_type'] = trade_data['trade_type'].lower()
                market_data_path = os.path.join(
                    charts_directory,
                    f"{trade_data['entry_date'].strftime(ISO_DATE_FORMAT)}-00"
                    f"-{trade_data['symbol']}.csv")

                style_name = 'fluorite'
                for s in config.options('Styles'):
                    c, v = configuration.evaluate_value(config['Styles'][s])
                    if c == 'any' or trade_data.get(c) == v:
                        style_name = s
                        break

                try:
                    style = importlib.import_module(
                        f"styles.{style_name}").style
                except ModuleNotFoundError as e:
                    print(e)
                    sys.exit(1)

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
        help='specify dates in the format %%Y-%%m-%%d [default: today]')
    parser.add_argument(
        '-f', nargs=1,
        help='specify the file path to the trading journal spreadsheet',
        metavar='FILE')
    parser.add_argument(
        '-d', nargs=1,
        help='specify the directory path'
        ' for storing historical data and charts',
        metavar='DIRECTORY')

    file_utilities.add_launcher_options(group)

    group.add_argument(
        '-G', action='store_true',
        help='configure general options and exit')
    group.add_argument(
        '-J', action='store_true',
        help='configure the columns of the trading journal and exit')
    group.add_argument(
        '-S', action='store_true',
        help='configure the styles based on the trade context and exit')
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
        'charts_directory': os.path.join(os.path.expanduser('~'),
                                         'Documents/Trading')}
    config['Market Data'] = {
        'opening_time': '09:00:00',
        'morning_session_end': '11:30:00',
        'afternoon_session_start': '12:30:00',
        'closing_time': '15:30:00',
        'delay': '20',
        'timezone': 'Asia/Tokyo',
        'exchange_suffix': '.T'}

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
        'optional_percentage_change': 'Percentage Change',
        'optional_chart_file': 'optional_chart_file'}
    for index in range(1, 11):
        config['Trading Journal'][f'optional_note_{index}'] = f'Note {index}'

    config['Chart'] = {
        'width': '1280',
        'height': '720',
        'scale_padding_top': '0.0',
        'scale_padding_right': '0.02',
        'scale_padding_bottom': '1.3',
        'scale_padding_left': '0.8'}
    config['Active Trading Hours'] = {
        'is_added': 'True',
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
        'is_added': 'True',
        'default_y_offset_ratio': '-0.008'}
    config['Styles'] = {
        'amber': ('', ''),
        'ametrine': ('', ''),
        'fluorite': ('', ''),
        'opal': ('', '')}

    if can_override:
        configuration.read_config(config, config_path)

    return config


def configure_exit(args, config_path, trading_path, trading_sheet):
    """Configure parameters based on command-line arguments and exit."""
    backup_parameters = {'number_of_backups': 8}
    if any((args.G, args.J, args.S)):
        config = configure(config_path, can_interpolate=False)
        for argument, (
                section, option, prompts, all_values
        ) in {
            'G': ('General', None, None, None),
            'J': ('Trading Journal', None, {'value': 'column'}, None),
            'S': ('Styles', None, {'values': ('column', 'value')},
                  (['any'] + TRADING_JOURNAL_COLUMNS, None))}.items():
            if getattr(args, argument):
                configuration.modify_section(
                    config, section, config_path,
                    backup_parameters=backup_parameters, option=option,
                    prompts=prompts,
                    all_values=(
                        tuple(pd.read_excel(trading_path,
                                            sheet_name=trading_sheet).columns)
                        if argument == 'J' else all_values))
                break

        sys.exit()
    if args.C:
        configuration.check_config_changes(
            configure(config_path, can_interpolate=False, can_override=False),
            config_path, backup_parameters=backup_parameters)
        sys.exit()


def save_market_data(config, trade_data, market_data_path):
    """Save historical market data for a given symbol to a CSV file."""
    PERIOD_IN_DAYS = 7
    delta = (
        pd.Timestamp.now(tz=config['Market Data']['timezone']).normalize()
        - trade_data['entry_date'])
    last = modified_time = pd.Timestamp(0,
                                        tz=config['Market Data']['timezone'])
    if os.path.isfile(market_data_path):
        formalized = pd.read_csv(market_data_path, index_col=0,
                                 parse_dates=True)
        last = formalized.tail(1).index[0]
        modified_time = pd.Timestamp(os.path.getmtime(market_data_path),
                                     tz=config['Market Data']['timezone'],
                                     unit='s')

    if (PERIOD_IN_DAYS <= 1 + delta.days
        or last + pd.Timedelta(minutes=int(config['Market Data']['delay']))
        < modified_time
        or pd.Timestamp.now(tz=config['Market Data']['timezone'])
        < modified_time + pd.Timedelta(minutes=1)):
        return
    else:
        my_share = share.Share(f"{trade_data['symbol']}"
                               f"{config['Market Data']['exchange_suffix']}")
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

        if morning and not previous.empty:
            start = create_timestamp(
                previous_date,
                config['Market Data']['afternoon_session_start'])
            end = create_timestamp(
                trade_data['entry_date'],
                config['Market Data']['morning_session_end'])
            end -= pd.Timedelta(minutes=1)
        else:
            start = create_timestamp(trade_data['entry_date'],
                                     config['Market Data']['opening_time'])
            end = create_timestamp(trade_data['entry_date'],
                                   config['Market Data']['closing_time'])
            end -= pd.Timedelta(minutes=1)

        formalized = pd.DataFrame(columns=('open', 'high', 'low', 'close',
                                           'volume'),
                                  index=pd.date_range(start, end, freq='min'))
        formalized.index.name = 'timestamp'
        formalized = formalized.astype('float')
        formalized.update(df)

        if morning and not previous.empty:
            start = create_timestamp(previous_date,
                                     config['Market Data']['closing_time'])
            end = create_timestamp(trade_data['entry_date'],
                                   config['Market Data']['opening_time'])
            end -= pd.Timedelta(minutes=1)
            exclusion = pd.date_range(start=start, end=end, freq='min')
            formalized = formalized.loc[~formalized.index.isin(exclusion)]
        else:
            start = create_timestamp(
                trade_data['entry_date'],
                config['Market Data']['morning_session_end'])
            end = create_timestamp(
                trade_data['entry_date'],
                config['Market Data']['afternoon_session_start'])
            end -= pd.Timedelta(minutes=1)
            exclusion = pd.date_range(start=start, end=end, freq='min')
            formalized = formalized.loc[~formalized.index.isin(exclusion)]

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
                  if trade_data['trade_type'] == 'long'
                  else trade_data['entry_price'] - trade_data['exit_price']
                  if trade_data['trade_type'] == 'short'
                  else 0)

    addplot = []
    panel = 0
    timestamps, prices, colors = prepare_parameters(config, formalized,
                                                    trade_data, result, style)
    percentage_change = (100 * result / trade_data['entry_price']
                         if pd.isna(trade_data['optional_percentage_change'])
                         else trade_data['optional_percentage_change'])

    if config['EMA'].getboolean('is_added'):
        add_emas(config, formalized, addplot, style)
    if config['MACD'].getboolean('is_added'):
        panel = add_macd(config, formalized, panel, addplot, style)
    if config['Stochastics'].getboolean('is_added'):
        panel = stochastics_panel = add_stochastics(config, formalized, panel,
                                                    addplot, style)
    if config['Volume'].getboolean('is_added'):
        panel += 1

    fig, axlist = mpf.plot(
        formalized, addplot=addplot, closefig=True,
        datetime_format=f'{DATE_FORMAT}, {TIME_FORMAT}',
        figsize=(int(config['Chart']['width']) / 100,
                 int(config['Chart']['height']) / 100),
        fill_between=dict(alpha=style['custom_style']['filled_area_alpha'],
                          color=colors['exit'], y1=trade_data['entry_price'],
                          y2=trade_data['exit_price'], zorder=1),
        hlines=dict(alpha=style['custom_style']['line_alpha'],
                    colors=list(colors.values()), hlines=list(prices.values()),
                    linestyle=[style['custom_style']['closing_line'],
                               style['custom_style']['opening_line'],
                               style['custom_style']['entry_line'],
                               style['custom_style']['exit_line']],
                    linewidths=1),
        returnfig=True,
        scale_padding={
            'top': float(config['Chart']['scale_padding_top']),
            'right': float(config['Chart']['scale_padding_right']),
            'bottom': float(config['Chart']['scale_padding_bottom']),
            'left': float(config['Chart']['scale_padding_left'])},
        scale_width_adjustment=dict(candle=1.5), style=style,
        tight_layout=True, type='candle',
        volume=config['Volume'].getboolean('is_added'), volume_panel=panel)

    add_vertical_elements(
        formalized, timestamps, axlist, colors, style,
        config['Active Trading Hours'].getboolean('is_added'))

    axlist[0].set_xticks(np.arange(*axlist[0].get_xlim(), 30))
    if config['Minor X-ticks'].getboolean('is_added'):
        add_minor_xticks(axlist, style['custom_style']['minor_grid_alpha'])

    if config['Stochastics'].getboolean('is_added'):
        axlist[2 * stochastics_panel].set_yticks([20.0, 50.0, 80.0])

    if (config['Tooltips'].getboolean('is_added') and prices['closing']
        and prices['opening'] != trade_data['entry_price']
        and prices['opening'] != trade_data['exit_price']):
        delta = prices['opening'] - prices['closing']
        add_tooltips(axlist, prices['opening'],
                     f"{delta:.1f}, {100 * delta / prices['closing']:.2f}%",
                     style['custom_style']['tooltip_color'], colors['opening'],
                     style['custom_style']['tooltip_bbox_alpha'])

    if (config['Tooltips'].getboolean('is_added')
        and not pd.isna(trade_data['entry_price'])):
        acronym = file_utilities.create_acronym(
            trade_data['optional_entry_reason'])
        add_tooltips(axlist, trade_data['entry_price'],
                     f'{acronym}' if acronym else '',
                     style['custom_style']['tooltip_color'], colors['entry'],
                     style['custom_style']['tooltip_bbox_alpha'],
                     formalized=formalized, timestamp=timestamps['entry'])

    if (config['Tooltips'].getboolean('is_added')
        and not pd.isna(trade_data['exit_price'])):
        acronym = file_utilities.create_acronym(
            trade_data['optional_exit_reason'])
        add_tooltips(axlist, trade_data['exit_price'],
                     f"{f'{acronym}, ' if acronym else ''}"
                     f"{result:.1f}, {percentage_change:.2f}%",
                     style['custom_style']['tooltip_color'], colors['exit'],
                     style['custom_style']['tooltip_bbox_alpha'],
                     formalized=formalized, timestamp=timestamps['exit'])

    if config['Text'].getboolean('is_added'):
        tactic = file_utilities.create_acronym(trade_data['optional_tactic'])
        full_date_format = f'%a, {DATE_FORMAT}, {chr(39)}%y,'
        notes = [trade_data[f'optional_note_{i}'] for i in range(1, 11)
                 if trade_data[f'optional_note_{i}']]
        add_text(axlist, float(config['Text']['default_y_offset_ratio']),
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


def prepare_parameters(config, formalized, trade_data, result, style):
    """Prepare timestamps, prices, and colors for entry and exit points."""
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

    prices = {'closing': 0.0, 'opening': 0.0, 'entry': 0.0, 'exit': 0.0}
    colors = {'closing': style['rc']['axes.edgecolor'],
              'opening': style['rc']['axes.edgecolor'],
              'entry': style['custom_style']['neutral_color'],
              'exit': style['custom_style']['neutral_color']}

    previous = formalized[formalized.index < trade_data['entry_date']]
    current = formalized[trade_data['entry_date'] <= formalized.index]

    if previous.notnull().values.any():
        prices['closing'] = previous.dropna().tail(1).close.iloc[0]
        prices['opening'] = current.dropna().head(1).open.iloc[0]

    # nan is not recognized as False in a boolean context.
    if (not pd.isna(trade_data['entry_time'])
        and not pd.isna(trade_data['entry_price'])):
        timestamps['entry'] = create_timestamp(trade_data['entry_date'],
                                               trade_data['entry_time'])
        prices['entry'] = trade_data['entry_price']

    if (not pd.isna(trade_data['exit_time'])
        and not pd.isna(trade_data['exit_price'])):
        timestamps['exit'] = create_timestamp(trade_data['exit_date'],
                                              trade_data['exit_time'])
        prices['exit'] = trade_data['exit_price']
        if result > 0:
            colors['exit'] = style['custom_style']['profit_color']
        elif result < 0:
            colors['exit'] = style['custom_style']['loss_color']

    return (timestamps, prices, colors)


def create_timestamp(date, time):
    """Create a pandas Timestamp by adding a time duration to a date."""
    if pd.isna(date):
        return pd.NaT
    else:
        return date + (pd.Timedelta(time)
                       if isinstance(time, str) else pd.Timedelta(str(time)))


def add_emas(config, formalized, addplot, style):
    """Add exponential moving average plots to the existing plots."""
    addplot.extend([
        mpf.make_addplot(
            indicators.ema(formalized.close,
                           int(config['EMA']['short_term_period'])),
            color=style['mavcolors'][0], width=0.8),
        mpf.make_addplot(
            indicators.ema(formalized.close,
                           int(config['EMA']['medium_term_period'])),
            color=style['mavcolors'][1], width=0.8),
        mpf.make_addplot(
            indicators.ema(formalized.close,
                           int(config['EMA']['long_term_period'])),
            color=style['mavcolors'][2], width=0.8)])


def add_macd(config, formalized, panel, addplot, style, ma='ema'):
    """Add moving average convergence divergence plots to the given panel."""
    if ma == 'ema':
        macd = (
            indicators.ema(formalized.close,
                           int(config['MACD']['short_term_period']))
            - indicators.ema(formalized.close,
                             int(config['MACD']['long_term_period'])))
        ylabel = 'MACD'
    elif ma == 'tema':
        macd = (
            indicators.tema(formalized.close,
                            int(config['MACD']['short_term_period']))
            - indicators.tema(formalized.close,
                              int(config['MACD']['long_term_period'])))
        ylabel = 'MACD TEMA'

    signal = macd.ewm(span=int(config['MACD']['signal_period'])).mean()
    histogram = macd - signal
    panel += 1

    addplot.extend([
        mpf.make_addplot(macd, color=style['mavcolors'][0], panel=panel,
                         width=0.8, ylabel=ylabel),
        mpf.make_addplot(signal, color=style['mavcolors'][1], panel=panel,
                         secondary_y=False, width=0.8),
        mpf.make_addplot(histogram,
                         color=[style['mavcolors'][2] if value >= 0
                                else style['mavcolors'][3]
                                for value in histogram],
                         panel=panel, secondary_y=False, type='bar',
                         width=1.0)])

    return panel


def add_stochastics(config, formalized, panel, addplot, style):
    """Add stochastic oscillator plots to the given panel."""
    df = indicators.stochastics(
        formalized.high, formalized.low, formalized.close,
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

    addplot.extend([
        mpf.make_addplot(formalized.k, color=style['mavcolors'][0],
                         panel=panel, width=0.8, ylabel='Stochastics'),
        mpf.make_addplot(formalized.d, color=style['mavcolors'][1],
                         panel=panel, secondary_y=False, width=0.8)])

    return panel


def add_minor_xticks(axlist, minor_grid_alpha):
    """Add minor x-ticks and their grid between panels."""
    axlist[0].set_xticks(np.arange(*axlist[0].get_xlim(), 10), minor=True)
    for index, _ in enumerate(axlist):
        if (index % 2) == 0:
            axlist[index].grid(which='minor', alpha=minor_grid_alpha)


def add_vertical_elements(formalized, timestamps, axlist, colors, style,
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
                        'active_trading_hours_color'], zorder=0)
            if timestamps['entry']:
                axlist[index].axvline(
                    alpha=style['custom_style']['line_alpha'],
                    color=colors['entry'],
                    linestyle=style['custom_style']['entry_line'], linewidth=1,
                    x=formalized.index.get_loc(timestamps['entry']))
            if timestamps['exit']:
                axlist[index].axvline(
                    alpha=style['custom_style']['line_alpha'],
                    color=colors['exit'],
                    linestyle=style['custom_style']['exit_line'], linewidth=1,
                    x=formalized.index.get_loc(timestamps['exit']))


def add_tooltips(axlist, price, string, color, bbox_color, bbox_alpha,
                 formalized=None, timestamp=None):
    """Add tooltips to the specified axes."""
    axlist[0].text(0, price, string, c=color, ha='right', size='small',
                   va='center',
                   bbox=dict(alpha=bbox_alpha, boxstyle='round, pad=0.2',
                             ec='none', fc=bbox_color))

    if timestamp:
        last_primary_axes = len(axlist) - 2
        bottom, _ = axlist[last_primary_axes].get_ylim()

        axlist[last_primary_axes].text(
            formalized.index.get_loc(timestamp), bottom,
            timestamp.strftime(TIME_FORMAT), c=color, ha='center',
            size='small', va='top',
            bbox=dict(alpha=bbox_alpha, boxstyle='round, pad=0.2', ec='none',
                      fc=bbox_color))


def add_text(axlist, default_y_offset_ratio, title, note_series, bbox_color,
             bbox_alpha):
    """Add a title and notes to the last primary axes."""
    # Use the last panel to prevent other panels from overwriting the
    # text.
    last_primary_axes = len(axlist) - 2
    bottom, top = axlist[last_primary_axes].get_ylim()
    height = top - bottom

    x_offset = 1
    panel_offset_factors = {0: 0, 2: 2.5 * height, 4: height, 6: 2 * height}
    panel_offset_factor = panel_offset_factors.get(last_primary_axes)
    y_offset_ratios = {0: default_y_offset_ratio,
                       2: 3.5 * default_y_offset_ratio,
                       4: 4.5 * default_y_offset_ratio,
                       6: 5.5 * default_y_offset_ratio}
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
