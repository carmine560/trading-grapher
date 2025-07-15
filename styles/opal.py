"""Configure the style settings for mplfinance plots."""

style = {
    "style_name": "opal",
    "base_mpl_style": "default",
    "marketcolors": {
        "candle": {"up": "lightskyblue", "down": "salmon"},
        "edge": {"up": "lightskyblue", "down": "salmon"},
        "wick": {"up": "white", "down": "white"},
        "ohlc": {"up": "lightskyblue", "down": "salmon"},
        "volume": {"up": "lightskyblue", "down": "salmon"},
        "vcedge": {"up": "lightskyblue", "down": "salmon"},
        "vcdopcod": False,
        "alpha": 1.0,
    },
    "mavcolors": ["burlywood", "skyblue", "royalblue", "darkslateblue"],
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
        "profit_color": "lightskyblue",
        "loss_color": "salmon",
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
