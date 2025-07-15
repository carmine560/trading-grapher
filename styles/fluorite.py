"""Configure the style settings for mplfinance plots."""

style = {
    "style_name": "fluorite",
    "base_mpl_style": "default",
    "marketcolors": {
        "candle": {"up": "mediumspringgreen", "down": "hotpink"},
        "edge": {"up": "mediumspringgreen", "down": "hotpink"},
        "wick": {"up": "white", "down": "white"},
        "ohlc": {"up": "mediumspringgreen", "down": "hotpink"},
        "volume": {"up": "mediumspringgreen", "down": "hotpink"},
        "vcedge": {"up": "mediumspringgreen", "down": "hotpink"},
        "vcdopcod": False,
        "alpha": 1.0,
    },
    "mavcolors": [
        "darksalmon",
        "cornflowerblue",
        "mediumpurple",
        "darkmagenta",
    ],
    "y_on_right": False,
    "gridcolor": "#3d3d3d",
    "gridstyle": "-",
    "facecolor": "#232323",
    "figcolor": "#2b2b2b",
    "rc": {
        "axes.edgecolor": "darkgray",
        "axes.labelcolor": "darkgray",
        "text.color": "gainsboro",
        "xtick.color": "darkgray",
        "ytick.color": "darkgray",
    },
    # The mplfinance/_styles.py module does not use the following
    # custom keys.
    "custom_style": {
        "neutral_color": "gainsboro",
        "profit_color": "mediumspringgreen",
        "loss_color": "hotpink",
        "closing_line": "--",
        "opening_line": "--",
        "entry_line": ":",
        "exit_line": ":",
        "line_alpha": 0.4,
        "filled_area_alpha": 0.04,
        "active_trading_hours_color": "#1b1b1b",
        "minor_grid_alpha": 0.2,
        "tooltip_color": "black",
        "tooltip_bbox_alpha": 0.6,
        "text_bbox_alpha": 0.5,
    },
}
