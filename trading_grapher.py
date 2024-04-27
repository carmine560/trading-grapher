#!/usr/bin/env python3

"""Visualize trading data using charts and technical indicators."""

import argparse
import configparser
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
    """
    Parse trade dates, save market data, plot charts, and check charts.

    This function parses the command-line arguments for trade dates. For
    each date, it retrieves the corresponding trades from the journal,
    saves the market data, and plots the trading chart. After processing
    all dates, it checks the charts for any discrepancies.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-C', action='store_true',
        help='check configuration changes and exit')
    parser.add_argument('dates', nargs='*',
                        default=[pd.Timestamp.now().strftime('%Y-%m-%d')],
                        help='specify dates')
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
    for date in pd.to_datetime(args.dates):
        trades = trading_journal.loc[
            trading_journal[config['Trading Journal']['entry_date']] == date]
        for _, trade in trades.iterrows():
            save_market_data(config, trade)
            plot_chart(config, trade)

    check_charts(config, trading_journal[config['Trading Journal']['chart']])


def configure(config_path, can_interpolate=True, can_override=True):
    """
    Get the configuration parser object with the set up configuration.

    This function initializes a configuration parser object based on the
    provided configuration file path. It allows optional interpolation
    and overriding of existing configuration settings.

    Args:
        config_path (str): The path to the configuration file.
        can_interpolate (bool, optional): If True, enable interpolation.
            Defaults to True.
        can_override (bool, optional): If True, allow overriding of
            existing configuration settings. Defaults to True.

    Returns:
        ConfigParser: The configuration parser object.
    """
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
        'theme': 'Dark'}
    config['Market Data'] = {
        'time_zone': 'Asia/Tokyo'}
    config['Trading Journal'] = {
        'entry_date': 'Entry date',
        'number': 'Number',
        'entry_time': 'Entry time',
        'symbol': 'Symbol',
        'trade_type': 'Trade type',
        'entry_price': 'Entry price',
        'tactic': 'Tactic',
        'entry_reason': 'Entry reason',
        'exit_date': 'Exit date',
        'exit_time': 'Exit time',
        'exit_price': 'Exit price',
        'exit_reason': 'Exit reason',
        'change': 'Change',
        'error_1': 'Error 1',
        'error_2': 'Error 2',
        'error_3': 'Error 3',
        'error_4': 'Error 4',
        'error_5': 'Error 5',
        'error_6': 'Error 6',
        'error_7': 'Error 7',
        'error_8': 'Error 8',
        'error_9': 'Error 9',
        'error_10': 'Error 10',
        'chart': 'Chart'}
    config['Dark'] = {          # Fluorite, Ametrine
        'face_color': '#242424',
        'figure_color': '#242424',
        'grid_color': '#3d3d3d',
        'edge_color': '#999999',
        'tick_color': '#999999',
        'label_color': '#999999',
        'up_color': 'mediumspringgreen',
        'down_color': 'hotpink',
        'primary_color': 'darksalmon',
        'secondary_color': 'cornflowerblue',
        'tertiary_color': 'rebeccapurple',
        'tooltip_color': 'black',
        'neutral_color': 'lightgray',
        'profit_color': '${up_color}',
        'loss_color': '${down_color}',
        'text_color': '#f6f3e8'}
    config['Light'] = {
        'face_color': '#fafafa',
        'figure_color': 'white',
        'grid_color': '#d0d0d0',
        'edge_color': '#f0f0f0',
        'tick_color': '#101010',
        'label_color': '#101010',
        'up_color': '#00b060',
        'down_color': '#fe3032',
        'primary_color': '#ff7f0e',
        'secondary_color': '#1f77b4',
        'tertiary_color': '#e377c2',
        'tooltip_color': 'white',
        'neutral_color': 'black',
        'profit_color': '${up_color}',
        'loss_color': '${down_color}',
        'text_color': 'black'}

    if can_override:
        configuration.read_config(config, config_path)
        configuration.write_config(config, config_path)  # TODO

    return config


def get_variables(config, symbol, entry_date, number):
    """
    Generate base string, market data path, and localize entry date.

    This function generates a base string using the entry date, trade
    number, and symbol. It also constructs the path to the market data
    file and localizes the entry date to the specified timezone.

    Args:
        config (ConfigParser): The configuration parser object.
        symbol (str): The trading symbol.
        entry_date (datetime): The date of the trade entry.
        number (int): The number of the trade.

    Returns:
        tuple: A tuple containing the base string, the path to the
            market data, and the localized entry date.
    """
    base = f"{entry_date.strftime('%Y-%m-%d')}-{int(number):02}-{symbol}"
    market_data = os.path.join(
        config['General']['trading_directory'],
        f"{entry_date.strftime('%Y-%m-%d')}-00-{symbol}.csv")
    entry_date = entry_date.tz_localize(config['Market Data']['time_zone'])

    return base, market_data, entry_date


def save_market_data(config, trade):
    """
    Save historical market data for a given symbol to a CSV file.

    Retrieves historical market data for a given trading symbol from
    Yahoo Finance, processes it, and saves it to a CSV file. The data
    includes open, high, low, close, and volume information. The
    function checks if the data is up-to-date and only retrieves new
    data if necessary.

    Args:
        config (ConfigParser): A ConfigParser instance containing the
            configuration settings.
        trade (Series): A pandas Series representing a trade entry with
            the following keys:
            - 'entry_date' (datetime): The date of the trade entry.
            - 'number' (int): The number of the trade.
            - 'symbol' (str): The trading symbol.
            - 'exit_time' (time): The time of the trade exit.
    """
    entry_date = trade[config['Trading Journal']['entry_date']]
    number = trade[config['Trading Journal']['number']]
    symbol = trade[config['Trading Journal']['symbol']]
    exit_time = trade[config['Trading Journal']['exit_time']]

    PERIOD_IN_DAYS = 7
    _, market_data, entry_date = get_variables(config, symbol, entry_date,
                                               number)
    delta = pd.Timestamp.now(
        tz=config['Market Data']['time_zone']).normalize() - entry_date
    last = modified_time = pd.Timestamp(
        0, tz=config['Market Data']['time_zone'])

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
        my_share = share.Share(f'{symbol}.T')
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
        q = df.volume.quantile(0.99)
        df['volume'] = df['volume'].mask(df['volume'] > q, q)

        previous = df[df.index < entry_date]
        if len(previous):
            previous_date = pd.Timestamp.date(
                previous.dropna().tail(1).index[0])
            previous_date = pd.Timestamp(previous_date,
                                         tz=config['Market Data']['time_zone'])

        morning = pd.Timedelta(str(exit_time)) < pd.Timedelta(hours=12)
        if morning and len(previous):
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

        if morning and len(previous):
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


def plot_chart(config, trade):
    """
    Plot a trading chart with entry and exit points, and indicators.

    Generates a trading chart for a given symbol, with markers for entry
    and exit points, and plots for various technical indicators. It also
    adds tooltips for the entry and exit points, and any errors.

    Args:
        config (ConfigParser): A ConfigParser instance containing the
            configuration settings.
        trade (Series): A pandas Series representing a trade entry with
            the following keys:
            - 'entry_date' (datetime): The date of the trade entry.
            - 'number' (int): The number of the trade.
            - 'entry_time' (str): The time of the trade entry.
            - 'symbol' (str): The trading symbol.
            - 'trade_type' (str): The type of the trade ('long' or
              'short').
            - 'entry_price' (float): The price at the trade entry.
            - 'tactic' (str): The trading tactic used.
            - 'entry_reason' (str): The reason for the trade entry.
            - 'exit_date' (datetime): The date of the trade exit.
            - 'exit_time' (str): The time of the trade exit.
            - 'exit_price' (float): The price at the trade exit.
            - 'exit_reason' (str): The reason for the trade exit.
            - 'change' (float): The price change from the trade entry to
              the exit.
            - 'error_1' to 'error_10' (str): Potential errors in the
              trade.
    """
    entry_date = trade[config['Trading Journal']['entry_date']]
    number = trade[config['Trading Journal']['number']]
    entry_time = trade[config['Trading Journal']['entry_time']]
    symbol = trade[config['Trading Journal']['symbol']]
    trade_type = trade[config['Trading Journal']['trade_type']]
    entry_price = trade[config['Trading Journal']['entry_price']]
    tactic = trade[config['Trading Journal']['tactic']]
    entry_reason = trade[config['Trading Journal']['entry_reason']]
    exit_date = trade[config['Trading Journal']['exit_date']]
    exit_time = trade[config['Trading Journal']['exit_time']]
    exit_price = trade[config['Trading Journal']['exit_price']]
    exit_reason = trade[config['Trading Journal']['exit_reason']]
    change = trade[config['Trading Journal']['change']]
    error_1 = trade[config['Trading Journal']['error_1']]
    error_2 = trade[config['Trading Journal']['error_2']]
    error_3 = trade[config['Trading Journal']['error_3']]
    error_4 = trade[config['Trading Journal']['error_4']]
    error_5 = trade[config['Trading Journal']['error_5']]
    error_6 = trade[config['Trading Journal']['error_6']]
    error_7 = trade[config['Trading Journal']['error_7']]
    error_8 = trade[config['Trading Journal']['error_8']]
    error_9 = trade[config['Trading Journal']['error_9']]
    error_10 = trade[config['Trading Journal']['error_10']]

    base, market_data, entry_date = get_variables(config, symbol, entry_date,
                                                  number)
    theme = config[config['General']['theme']]

    if os.path.exists(market_data):
        formalized = pd.read_csv(market_data, index_col=0, parse_dates=True)
    else:
        print(market_data, 'does not exist')
        sys.exit(1)

    entry_timestamp = exit_timestamp = None
    entry_color = theme['neutral_color']
    addplot = []
    hlines = []
    colors = []

    previous = formalized[formalized.index < entry_date]
    previous_close = 0.0
    current = formalized[formalized.index >= entry_date]
    current_open = 0.0

    if len(previous.dropna()):
        previous_close = previous.dropna().tail(1).close.iloc[0]
        current_open = current.dropna().head(1).open.iloc[0]
        hlines = [previous_close, current_open]
        # TODO: theme
        colors = ['gray', 'gray']

    if trade_type == 'long':
        marker = 'o'
    elif trade_type == 'short':
        marker = 'D'

    marker_alpha = 0.2

    if not pd.isna(entry_time) and not pd.isna(entry_price):
        formalized['entry_point'] = pd.Series(dtype='float')
        entry_timestamp = entry_date + pd.Timedelta(str(entry_time))
        formalized.loc[entry_timestamp, 'entry_point'] = entry_price
        entry_apd = mpf.make_addplot(formalized.entry_point, type='scatter',
                                     markersize=100, marker=marker,
                                     color=entry_color, edgecolors='none',
                                     alpha=marker_alpha)
        addplot.append(entry_apd)
        hlines.append(entry_price)
        colors.append(entry_color)

    result = 0.0
    exit_color = entry_color
    if not pd.isna(exit_time) and not pd.isna(exit_price):
        if trade_type == 'long':
            result = exit_price - entry_price
        elif trade_type == 'short':
            result = entry_price - exit_price
        if result > 0:
            exit_color = theme['profit_color']
        elif result < 0:
            exit_color = theme['loss_color']

        formalized['exit_point'] = pd.Series(dtype='float')
        exit_date = exit_date.tz_localize(config['Market Data']['time_zone'])
        exit_timestamp = exit_date + pd.Timedelta(str(exit_time))
        formalized.loc[exit_timestamp, 'exit_point'] = exit_price
        exit_apd = mpf.make_addplot(formalized.exit_point, type='scatter',
                                    markersize=100, marker=marker,
                                    color=exit_color, edgecolors='none',
                                    alpha=marker_alpha)
        addplot.append(exit_apd)
        hlines.append(exit_price)
        colors.append(exit_color)

    marker_coordinate_alpha = 0.4

    if len(hlines) and len(colors):
        hlines = dict(hlines=hlines, colors=colors, linestyle='dotted',
                      linewidths=1, alpha=marker_coordinate_alpha)

    add_ma(config, formalized, mpf, addplot)

    panel = 0
    panel = add_macd(config, formalized, panel, mpf, addplot)
    panel = stoch_panel = add_stoch(config, formalized, panel, mpf, addplot)

    # TODO: base_mpl_style
    style = {'base_mpl_style': 'dark_background',
             # TODO: candle_up_color, etc.
             'marketcolors': {'candle': {'up': theme['up_color'],
                                         'down': theme['down_color']},
                              'edge': {'up': theme['up_color'],
                                       'down': theme['down_color']},
                              'wick': {'up': theme['up_color'],
                                       'down': theme['down_color']},
                              'ohlc': {'up': theme['up_color'],
                                       'down': theme['down_color']},
                              'volume': {'up': theme['up_color'],
                                         'down': theme['down_color']},
                              'vcedge': {'up': theme['up_color'],
                                         'down': theme['down_color']},
                              'vcdopcod': None,
                              'alpha': None},
             'mavcolors': None,
             'facecolor': theme['face_color'],
             'figcolor': theme['figure_color'],
             'gridcolor': theme['grid_color'],
             'gridstyle': '-',
             'y_on_right': None,
             # TODO: minor
             'rc': {'axes.edgecolor': theme['edge_color'],
                    'axes.labelcolor': theme['label_color'],
                    'figure.titlesize': 'x-large',
                    'figure.titleweight': 'semibold',
                    'text.color': theme['text_color'],
                    'xtick.color': theme['tick_color'],
                    'ytick.color': theme['tick_color']}}

    panel += 1
    fig, axlist = mpf.plot(formalized, type='candle', volume=True,
                           tight_layout=True, figsize=(1152 / 100, 648 / 100),
                           style=style,
                           scale_padding={'top': 0, 'right': 0, 'bottom': 1.5},
                           scale_width_adjustment=dict(candle=1.5),
                           hlines=hlines, addplot=addplot, returnfig=True,
                           closefig=True, volume_panel=panel)

    left, right = axlist[0].get_xlim()
    axlist[0].set_xticks(np.arange(left, right, 30))
    axlist[0].set_xticks(np.arange(left, right, 10), minor=True)
    axlist[2 * stoch_panel].set_yticks([20.0, 50.0, 80.0])
    for i in range(len(axlist)):
        if (i % 2) == 0:
            axlist[i].grid(which='minor', alpha=0.2)
            if entry_timestamp:
                axlist[i].axvline(x=formalized.index.get_loc(entry_timestamp),
                                  color=entry_color, linestyle='dotted',
                                  linewidth=1, alpha=marker_coordinate_alpha)
            if exit_timestamp:
                axlist[i].axvline(x=formalized.index.get_loc(exit_timestamp),
                                  color=exit_color, linestyle='dotted',
                                  linewidth=1, alpha=marker_coordinate_alpha)

    if previous_close:
        if current_open != entry_price and current_open != exit_price:
            delta = current_open - previous_close
            style = f'{delta:.1f}, {delta / previous_close * 100:.2f}%'
            add_tooltips(config, axlist, current_open, style, 'gray')

    last_primary_axis = len(axlist) - 2
    if not pd.isna(entry_price):
        acronym = create_acronym(entry_reason)
        if acronym:
            add_tooltips(config, axlist, entry_price, acronym, entry_color,
                         last_primary_axis, formalized, entry_timestamp)
    if not pd.isna(exit_price):
        acronym = create_acronym(exit_reason)
        if acronym:
            style = f'{acronym}, {result:.1f}, {change:.2f}%'
        else:
            style = f'{result:.1f}, {change:.2f}%'

        add_tooltips(config, axlist, exit_price, style, exit_color,
                     last_primary_axis, formalized, exit_timestamp)

    error_series = pd.Series(
        [error_1, error_2, error_3, error_4, error_5, error_6, error_7,
         error_8, error_9, error_10]).dropna()
    add_errors(error_series, axlist)

    acronym = create_acronym(tactic)
    if acronym:
        title = base + ', ' + acronym
    else:
        title = base

    fig.suptitle(title, size='medium', alpha=0.4)
    fig.savefig(os.path.join(config['General']['trading_directory'],
                             base + '.png'))


def add_ma(config, formalized, mpf, addplot, ma='ema'):
    """
    Add Exponential Moving Average (EMA) plots to the existing plots.

    This function calculates the EMA for the given data with different
    spans, and creates plots for these values which are added to the
    existing plots.

    Args:
        config (ConfigParser): The configuration parser object.
        formalized (DataFrame): The DataFrame containing the closing
            prices.
        mpf (module): The mplfinance module used for creating the plots.
        addplot (list): The list of existing plots to which the new
            plots will be added.
        ma (str, optional): The type of moving average to use. Defaults
            to 'ema'.
    """
    if ma == 'ema':
        ma_1 = ema(formalized.close, 5)
        ma_2 = ema(formalized.close, 25)
        ma_3 = ema(formalized.close, 75)

    theme = config[config['General']['theme']]
    ma_apd = [
        mpf.make_addplot(ma_1, color=theme['primary_color'], width=0.8),
        mpf.make_addplot(ma_2, color=theme['secondary_color'], width=0.8),
        mpf.make_addplot(ma_3, color=theme['tertiary_color'], width=0.8)]
    addplot.extend(ma_apd)


def add_macd(config, formalized, panel, mpf, addplot, ma='ema'):
    """
    Add Moving Average Convergence Divergence (MACD) plots to the panel.

    This function calculates the MACD values for the given data, adds
    them to the formalized DataFrame, and creates plots for these values
    which are added to the given panel. The type of moving average used
    ('ema' or 'tema') can be specified.

    Args:
        config (ConfigParser): The configuration parser object.
        formalized (DataFrame): The DataFrame containing the closing
            prices.
        panel (int): The panel number to which the plots will be added.
        mpf (module): The mplfinance module used for creating the plots.
        addplot (list): The list of existing plots to which the new
            plots will be added.
        ma (str, optional): The type of moving average to use ('ema' or
            'tema'). Defaults to 'ema'.

    Returns:
        int: The updated panel number.
    """
    if ma == 'ema':
        macd = ema(formalized.close, 12) - ema(formalized.close, 26)
        ylabel = 'MACD'
    elif ma == 'tema':
        macd = tema(formalized.close, 12) - tema(formalized.close, 26)
        ylabel = 'MACD TEMA'

    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    panel += 1
    theme = config[config['General']['theme']]
    macd_apd = [
        mpf.make_addplot(macd, panel=panel, color=theme['primary_color'],
                         width=0.8, ylabel=ylabel),
        mpf.make_addplot(signal, panel=panel, color=theme['secondary_color'],
                         width=0.8, secondary_y=False),
        mpf.make_addplot(histogram, type='bar', width=1.0, panel=panel,
                         color=theme['tertiary_color'], secondary_y=False)]
    addplot.extend(macd_apd)

    return panel


def ema(series, span):
    """
    Calculate the Exponential Moving Average (EMA) of a series.

    This function computes the EMA of a given series. The EMA is a type
    of moving average that gives more weight to recent prices, which can
    make it more responsive to new information.

    Args:
        series (Series): The time series to compute the EMA of.
        span (int): The span of periods to consider for the moving
            average.

    Returns:
        Series: The EMA of the input series.
    """
    ema = series.ewm(span=span).mean()
    ema.iloc[:span - 1] = np.nan
    return ema


def tema(series, span):
    """
    Calculate the Triple Exponential Moving Average (TEMA) of a series.

    This function computes the TEMA of a given series. The TEMA is a
    technical indicator used to smooth price fluctuations and filter out
    volatility.

    Args:
        series (Series): The time series to compute the TEMA of.
        span (int): The span of periods to consider for the moving
            average.

    Returns:
        Series: The TEMA of the input series.
    """
    ema_1 = ema(series, span)
    ema_2 = ema(ema_1, span)
    ema_3 = ema(ema_2, span)
    tema = 3 * (ema_1 - ema_2) + ema_3
    tema.iloc[:3 * (span - 1)] = np.nan
    return tema


def add_stoch(config, formalized, panel, mpf, addplot):
    """
    Add stochastic oscillator plots to the given panel.

    This function calculates the stochastic oscillator values for the
    given data, adds them to the formalized DataFrame, and creates plots
    for these values which are added to the given panel.

    Args:
        config (ConfigParser): The configuration parser object.
        formalized (DataFrame): The DataFrame containing the high, low,
            and close prices.
        panel (int): The panel number to which the plots will be added.
        mpf (module): The mplfinance module used for creating the plots.
        addplot (list): The list of existing plots to which the new
            plots will be added.

    Returns:
        int: The updated panel number.
    """
    k = 5
    d = 3
    smooth_k = 3
    df = stoch(formalized.high, formalized.low, formalized.close, k=k, d=d,
               smooth_k=smooth_k)
    if df.k.dropna().empty:
        df.k.fillna(50.0, inplace=True)
    if df.d.dropna().empty:
        df.d.fillna(50.0, inplace=True)

    formalized['k'] = pd.Series(dtype='float')
    formalized['d'] = pd.Series(dtype='float')
    formalized.update(df)
    panel += 1
    theme = config[config['General']['theme']]
    stoch_apd = [mpf.make_addplot(formalized.k, panel=panel,
                                  color=theme['primary_color'], width=0.8,
                                  ylabel='Stochastics'),
                 mpf.make_addplot(formalized.d, panel=panel,
                                  color=theme['secondary_color'], width=0.8,
                                  secondary_y=False)]
    addplot.extend(stoch_apd)

    return panel


def stoch(high, low, close, k, d, smooth_k):
    """
    Calculate the stochastic oscillator values for a given dataset.

    This function computes the stochastic oscillator values (K and D)
    for a given dataset. The stochastic oscillator is a momentum
    indicator comparing a particular closing price of a security to a
    range of its prices over a certain period of time.

    Args:
        high (Series): The high prices for each period.
        low (Series): The low prices for each period.
        close (Series): The closing prices for each period.
        k (int): The lookback period for the %K line.
        d (int): The smoothing period for the %D line.
        smooth_k (int): The smoothing period for the %K line.

    Returns:
        DataFrame: A DataFrame with the stochastic oscillator values (%K
            and %D).
    """
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()

    stoch = 100 * (close - lowest_low)
    diff = highest_high - lowest_low
    if diff.eq(0).any().any():
        diff += sys.float_info.epsilon

    stoch /= diff

    stoch_k = stoch.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(d).mean()

    return pd.DataFrame({'k': stoch_k, 'd': stoch_d})


def create_acronym(phrase):
    """
    Generate an acronym from the given phrase.

    This function takes a string phrase, splits it into words, and
    constructs an acronym by taking the first letter of each word and
    capitalizing it.

    Args:
        phrase (str): The phrase to convert into an acronym.

    Returns:
        str: The acronym created from the initial letters of the words
            in the phrase.
    """
    if isinstance(phrase, str):
        acronym = ''
        for word in re.split(r'[\W_]+', phrase):
            acronym = acronym + word[0].upper()

        return acronym


def add_tooltips(config, axlist, price, s, color,
                 last_primary_axis=None, formalized=None, timestamp=None):
    """
    Add tooltips to the specified axes list.

    This function adds tooltips to the first plot in the axlist and, if
    a timestamp is provided, to the last primary axis. The tooltips are
    formatted with the provided text, color, and alpha values.

    Args:
        config (ConfigParser): The configuration parser object.
        axlist (list): A list of axes objects to which the tooltips will
            be added.
        price (float): The price at which the tooltip will be placed.
        s (str): The text to be displayed in the tooltip.
        color (str): The color of the tooltip.
        last_primary_axis (int, optional): The index of the last primary
            axis. Defaults to None.
        formalized (pd.DataFrame, optional): A pandas DataFrame with a
            timestamp index. Defaults to None.
        timestamp (datetime, optional): The timestamp at which the
            tooltip will be placed. Defaults to None.
    """
    tooltip_foreground_alpha = 0.8
    tooltip_background_alpha = 0.6
    theme = config[config['General']['theme']]

    axlist[0].text(
        -1.2, price, s, alpha=tooltip_foreground_alpha,
        c=theme['tooltip_color'], size='small', ha='right', va='center',
        bbox=dict(boxstyle='round, pad=0.2',
                  alpha=tooltip_background_alpha, ec='none',
                  fc=color))
    if timestamp:
        bottom, top = axlist[last_primary_axis].get_ylim()
        axlist[last_primary_axis].text(
            formalized.index.get_loc(timestamp), -0.03 * (top - bottom),
            timestamp.strftime('%H:%M'), alpha=tooltip_foreground_alpha,
            c=theme['tooltip_color'], size='small', ha='center', va='top',
            bbox=dict(boxstyle='round, pad=0.2',
                      alpha=tooltip_background_alpha, ec='none', fc=color))


def add_errors(error_series, axlist):
    """
    Add error messages to the top of the first plot in axlist.

    This function iterates over the items in error_series, formats them
    into a string, and if any errors exist, adds them to the top of the
    first plot in axlist.

    Args:
        error_series (Series): A pandas Series containing error
            messages.
        axlist (list): A list of axes objects to which the error
            messages will be added.
    """
    errors = ''
    for index, value in error_series.items():
        if index == 0:
            errors = f'Errors:\n{index + 1}. {value}'
        else:
            errors = errors + f'\n{index + 1}. {value}'

    if errors:
        bottom, top = axlist[0].get_ylim()
        axlist[0].text(0, top, errors, alpha=0.8, va='top')


def check_charts(config, charts):
    """
    Validate charts in the trading directory and print invalid ones.

    This function checks for '.png' files in the trading directory that
    are not referenced in the journal's chart values. It also verifies
    that referenced charts in the journal exist in the trading
    directory. Any discrepancies found are printed to the console.

    Args:
        config (ConfigParser): A ConfigParser instance containing the
            configuration settings.
        charts (Series): A pandas Series containing chart references.
    """
    for f in os.listdir(config['General']['trading_directory']):
        if f.endswith('.png') and f not in charts.values:
            print(os.path.join(config['General']['trading_directory'], f))

    for value in charts:
        if isinstance(value, str) and not os.path.exists(
                os.path.join(config['General']['trading_directory'], value)):
            print(value)


if __name__ == '__main__':
    main()
