"""Configure the style settings for mplfinance plots."""

style = {'base_mpl_style': 'default',
         'marketcolors': {'candle': {'up': 'mediumspringgreen',
                                     'down': 'hotpink'},
                          'edge': {'up': 'mediumspringgreen',
                                   'down': 'hotpink'},
                          'wick': {'up': 'mediumspringgreen',
                                   'down': 'hotpink'},
                          'ohlc': {'up': 'mediumspringgreen',
                                   'down': 'hotpink'},
                          'volume': {'up': 'mediumspringgreen',
                                     'down': 'hotpink'},
                          'vcedge': {'up': 'mediumspringgreen',
                                     'down': 'hotpink'},
                          'vcdopcod': None,
                          'alpha': None},
         'mavcolors': ['darksalmon',
                       'cornflowerblue',
                       'mediumpurple',
                       'rebeccapurple'],
         'facecolor': '#242424',
         'figcolor': '#242424',
         'gridcolor': '#3d3d3d',
         'gridstyle': '-',
         'y_on_right': None,
         'rc': {'axes.edgecolor': '#999999',
                'axes.labelcolor': '#999999',
                'figure.titlesize': 'x-large',
                'figure.titleweight': 'semibold',
                'text.color': '#f6f3e8',
                'xtick.color': '#999999',
                'ytick.color': '#999999'},
         # The mplfinance._styledata module does not use the following
         # custom keys.
         'custom_style': {'tooltip_color': 'black',
                          'neutral_color': 'lightgray',
                          'profit_color': 'mediumspringgreen',
                          'loss_color': 'hotpink',
                          'marker_alpha': 0.2,
                          'marker_coordinate_alpha': 0.4}}
