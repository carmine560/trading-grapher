# Repository Guidance

  * Treat `trade_data["symbol"]` as an opaque journal identifier.
  * Do not propose normalization or format validation for symbols unless the
    repository already enforces that rule or the user explicitly asks for it.
  * Only recommend symbol-related changes when there is concrete repository
    evidence and a clear practical benefit.
  * Do not recommend market-data refresh or cache-freshness changes that use
    missing OHLCV rows as a freshness signal unless the repository already
    does so. Sparse OHLCV data can be a normal market condition here, and
    related suggestions need direct evidence from the current code and tests.
