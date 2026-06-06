# Project Policy

  * This project is for Japanese stock day trading. Treat `symbol` as the
    existing Japanese stock identifier, not as a general cross-market ticker
    abstraction.
  * Do not propose symbol normalization, broader ticker support,
    ticker-format validation, alternate symbol taxonomies, or cross-market
    symbol handling unless explicitly requested or already implemented.
  * Do not recommend market-data refresh or cache-freshness changes that use
    missing OHLCV rows as a freshness signal unless the repository already
    does so. Sparse OHLCV data can be a normal market condition here, and
    related suggestions need direct evidence from the current code and tests.
