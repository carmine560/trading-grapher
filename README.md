# trading-grapher #

<!-- Python script that visualizes the results of stock day trading in a
spreadsheet using mplfinance based on historical data from Yahoo Finance -->

The `trading_grapher.py` Python script visualizes the results of stock day
trading in a spreadsheet, as well as the MACD and the stochastics, using the
`mplfinance` package based on historical data from [Yahoo
Finance](https://finance.yahoo.com/).

## Prerequisites ##

`trading_grapher.py` has been tested on Debian Testing on WSL 2 and requires
the following packages:

  * [`mplfinance`](https://github.com/matplotlib/mplfinance) to plot trade data
    and the indicators based on historical data
  * [`odfpy`](https://github.com/eea/odfpy) to read the trading journal
    recorded in an OpenDocument Spreadsheet file
  * [`prompt_toolkit`](https://github.com/prompt-toolkit/python-prompt-toolkit)
    to complete possible values or a previous value in configuring
  * [`yfinance`](https://github.com/ranaroussi/yfinance) to retrieve historical
    data from Yahoo Finance

Install each package as needed. For example:

``` shell
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -U
```

## Usage ##

First, configure the path to your trading journal spreadsheet, the sheet name,
and the directory path for storing historical data and charts:

``` shell
./trading_grapher.py -G
```

Next, configure the columns of the trading journal:

``` shell
./trading_grapher.py -J
```

The `~/.config/trading-grapher/trading_grapher.ini` configuration file stores
the configurations above. Then, execute:

``` shell
./trading_grapher.py [%Y-%m-%d ...]
```

### Options ###

  * `-f FILE`: specify the file path to the trading journal spreadsheet
  * `-d DIRECTORY`: specify the directory path for storing historical data and
    charts
  * `-BS`: save a Bash script to `$HOME/Downloads` to launch this script and
    exit
  * `-G`: configure general options and exit
  * `-J`: configure the columns of the trading journal and exit
  * `-S`: configure the styles based on the trade context and exit
  * `-C`: check configuration changes and exit

## Styles ##

`trading_grapher.py` provides the following style modules, which you can
specify using the `-S` option.

### `amber` ###

![A “amber” style chart showing the result of a specific stock day
trade](examples/amber.png)

### `ametrine` ###

![A “ametrine” style chart showing the result of a specific stock day
trade](examples/ametrine.png)

### `fluorite` (Default) ###

![A “fluorite” style chart showing the result of a specific stock day
trade](examples/fluorite.png)

### `opal` ###

![A “opal” style chart showing the result of a specific stock day
trade](examples/opal.png)

## To Do

Visit the “[To Do](https://github.com/carmine560/trading-grapher/wiki#to-do)”
section in the wiki.

## License ##

This project is licensed under the [MIT License](LICENSE.md). The `.gitignore`
file is sourced from [`gitignore`](https://github.com/github/gitignore), which
is licensed under the CC0-1.0 license.
